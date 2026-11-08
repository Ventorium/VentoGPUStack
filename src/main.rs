//! VentoGPUStack — 基于 Pingora + russh 的 GPU 机房管理网关
//!
//! 子命令：
//!   run          启动（默认），管理端 HTTP 服务 + 机群周期采集
//!   init         生成 admin.yaml（随机密码 + JWT 密钥）
//!   set-password 修改管理员密码
//!   import       从 CSV 导入服务器清单（第一轮盘点产出的 gpu_servers.csv）

mod admin;
mod ai;
mod collect;
mod config;
mod deploy;
mod discover;
mod scheduler;
mod ssh;

use std::path::Path;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use config::{AdminConf, AppState, MainConf, ServersFile, ServicesFile};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "ventogpustack", version, about = "VentoGPUStack — GPU 机房管理网关")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// 启动服务（默认子命令）
    Run {
        /// 主配置文件
        #[arg(short, long, default_value = "config/ventogpustack.yaml")]
        config: String,
    },
    /// 生成 admin.yaml（随机管理员密码 + JWT 密钥）
    Init {
        /// 输出文件
        #[arg(short, long, default_value = "config/admin.yaml")]
        file: String,
        /// 管理员用户名
        #[arg(long, default_value = "admin")]
        user: String,
        /// 指定密码（默认随机生成并打印）
        #[arg(long)]
        password: Option<String>,
    },
    /// 修改管理员密码
    SetPassword {
        /// admin.yaml 路径
        #[arg(short, long, default_value = "config/admin.yaml")]
        file: String,
        /// 新密码（缺省时交互式输入）
        #[arg(long)]
        password: Option<String>,
        /// 管理员用户名
        #[arg(short, long, default_value = "admin")]
        user: String,
    },
    /// 从 CSV 导入服务器清单 → servers.yaml
    Import {
        /// 输入 CSV（第一轮盘点产出）
        #[arg(short, long, default_value = "gpu_servers.csv")]
        csv: String,
        /// 输出 servers.yaml 路径
        #[arg(short, long, default_value = "config/servers.yaml")]
        out: String,
        /// SSH 端口
        #[arg(long, default_value_t = 22)]
        port: u16,
        /// SSH 用户名
        #[arg(long)]
        user: String,
        /// SSH 密码
        #[arg(long, default_value = "")]
        password: String,
        /// 导入时默认禁用（连通性差的机器）
        #[arg(long)]
        default_disabled: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        None => run("config/ventogpustack.yaml"),
        Some(Cmd::Run { config }) => run(&config),
        Some(Cmd::Init { file, user, password }) => cmd_init(&file, &user, password),
        Some(Cmd::SetPassword { file, password, user }) => cmd_set_password(&file, password, &user),
        Some(Cmd::Import { csv, out, port, user, password, default_disabled }) => {
            cmd_import(&csv, &out, port, &user, &password, default_disabled)
        }
    }
}

// ---------------------------------------------------------------------------
// init / set-password / import
// ---------------------------------------------------------------------------

fn gen_password(len: usize) -> String {
    use rand::Rng;
    const CHARS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}

fn cmd_init(file: &str, user: &str, password: Option<String>) {
    if Path::new(file).exists() {
        eprintln!("{file} 已存在，拒绝覆盖（如需重置请手动删除后重新 init）");
        std::process::exit(1);
    }
    let password = password.unwrap_or_else(|| gen_password(24));
    let hash = AdminConf::generate_password_hash(&password)
        .unwrap_or_else(|e| panic!("生成密码哈希失败: {e}"));
    let admin = AdminConf {
        username: user.to_string(),
        password_hash: hash,
        jwt_secret: AdminConf::generate_secret(32),
        jwt_ttl_hours: 72,
    };
    admin.save(Path::new(file)).unwrap_or_else(|e| panic!("写入 {file} 失败: {e}"));
    if password == gen_password(0) {
        // 不会走到这里，仅占位
    }
    println!("已生成 {file}");
    println!("  管理员用户: {user}");
    println!("  管理员密码: {password}   （请立即妥善保管，可用 set-password 修改）");
}

fn cmd_set_password(file: &str, password: Option<String>, user: &str) {
    let mut admin = AdminConf::load(Path::new(file))
        .unwrap_or_else(|e| panic!("加载 {file} 失败: {e}"));
    let password = match password {
        Some(p) if !p.is_empty() => p,
        _ => {
            eprint!("请输入新密码: ");
            let mut buf = String::new();
            std::io::stdin().read_line(&mut buf).expect("读取密码失败");
            buf.trim().to_string()
        }
    };
    admin.username = user.to_string();
    admin.password_hash =
        AdminConf::generate_password_hash(&password).unwrap_or_else(|e| panic!("{e}"));
    admin
        .save(Path::new(file))
        .unwrap_or_else(|e| panic!("写入 {file} 失败: {e}"));
    println!("密码已更新（{file}）");
}

/// 从第一轮盘点的 CSV 生成 servers.yaml。
fn cmd_import(
    csv: &str,
    out: &str,
    port: u16,
    user: &str,
    password: &str,
    default_disabled: bool,
) {
    let text = std::fs::read_to_string(csv).unwrap_or_else(|e| panic!("读取 {csv} 失败: {e}"));
    let mut servers: Vec<config::ServerConf> = Vec::new();
    let mut id: u64 = 1;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        // 期望列: ip, name, host, port, user, password, notes（宽松解析：第 1 列必须是 IP）
        if !cols.first().is_some_and(|c| c.chars().all(|c| c.is_ascii_digit() || c == '.')) {
            continue; // 表头或脏行
        }
        let n = cols.len();
        let (name, notes) = if n >= 7 {
            (cols[1].to_string(), cols[6..].join(","))
        } else if n >= 6 {
            (cols[1].to_string(), cols[5].to_string())
        } else if n >= 2 {
            (cols[1].to_string(), String::new())
        } else {
            (format!("gpu-{id}"), String::new())
        };
        servers.push(config::ServerConf {
            id,
            name,
            host: cols[0].to_string(),
            port,
            user: user.to_string(),
            password: password.to_string(),
            notes,
            enabled: !default_disabled,
        });
        id += 1;
    }
    if servers.is_empty() {
        panic!("CSV 中没有解析到任何服务器行");
    }
    let file = ServersFile { servers };
    file.save(Path::new(out)).unwrap_or_else(|e| panic!("写入 {out} 失败: {e}"));
    println!("已导入 {} 台服务器 → {out}", file.servers.len());
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn run(config_path: &str) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let main_conf = MainConf::load(Path::new(config_path))
        .unwrap_or_else(|e| panic!("加载主配置失败: {e}"));

    // admin.yaml 缺失时自动生成（密码打印到日志，可用 set-password 修改）
    let admin_path = main_conf.admin_path();
    if let Some(parent) = admin_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let admin = if admin_path.exists() {
        AdminConf::load(&admin_path).unwrap_or_else(|e| panic!("加载 admin.yaml 失败: {e}"))
    } else {
        let pw = gen_password(24);
        let hash = AdminConf::generate_password_hash(&pw)
            .unwrap_or_else(|e| panic!("生成密码哈希失败: {e}"));
        let a = AdminConf {
            username: "admin".into(),
            password_hash: hash,
            jwt_secret: AdminConf::generate_secret(32),
            jwt_ttl_hours: 72,
        };
        a.save(&admin_path).unwrap_or_else(|e| panic!("写入 admin.yaml 失败: {e}"));
        log::info!(
            "已自动生成 {} —— 管理员用户: admin, 密码: {pw}（请尽快用 set-password 修改）",
            admin_path.display()
        );
        a
    };

    let servers = ServersFile::load(&main_conf.servers_path())
        .unwrap_or_else(|e| panic!("加载 servers.yaml 失败: {e}"));
    let services = ServicesFile::load(&main_conf.services_path())
        .unwrap_or_else(|e| panic!("加载 services.yaml 失败: {e}"));
    let _ = std::fs::create_dir_all(main_conf.log_dir());
    log::info!(
        "配置加载完成: {} 台服务器, {} 个模型服务",
        servers.servers.len(),
        services.services.len()
    );

    let state = AppState::new(
        main_conf.clone(),
        PathBuf::from(config_path),
        admin,
        servers,
        services,
        main_conf.audit_path(),
    );

    // 写 pid 文件
    if let Some(parent) = Path::new(&main_conf.pid_file).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&main_conf.pid_file, std::process::id().to_string());

    // 程序化构造 ServerConf
    let mut conf = pingora::server::configuration::ServerConf::default();
    conf.daemon = false;
    conf.threads = main_conf.threads as usize;
    conf.grace_period_seconds = Some(main_conf.grace_period_seconds);
    conf.upgrade_sock = main_conf.upgrade_sock.clone();
    conf.pid_file = main_conf.pid_file.clone();
    if let Some(e) = &main_conf.error_log {
        conf.error_log = Some(e.clone());
    }

    let opt = pingora::prelude::Opt {
        upgrade: false,
        daemon: false,
        nocapture: false,
        test: false,
        conf: None,
    };
    let mut server = pingora::prelude::Server::new_with_opt_and_conf(Some(opt), conf);
    server.bootstrap();

    // 管理端（独立服务，永不走上游）
    let mut admin_svc = pingora::proxy::http_proxy_service_with_name(
        &server.configuration,
        admin::AdminGw { state: state.clone() },
        "ventogpustack-admin",
    );
    admin_svc.add_tcp(&main_conf.admin_listen);
    server.add_service(admin_svc);
    log::info!("管理端监听 {}", main_conf.admin_listen);

    // 后台：机群周期采集
    server.add_service(pingora::prelude::background_service(
        "vgs-refresh",
        RefreshLoop { state: state.clone() },
    ));
    // 后台：配置热重载（main / servers / services yaml）
    server.add_service(pingora::prelude::background_service(
        "vgs-config-watch",
        ConfigWatcher {
            state: state.clone(),
            main_path: PathBuf::from(config_path),
        },
    ));

    log::info!("VentoGPUStack 启动完成");
    server.run_forever();
}

use std::path::PathBuf;

/// 机群周期采集：首次立即执行，之后每 refresh_interval_secs 一轮，
/// 上一轮未结束时跳过（refresh_inflight 互斥）。
struct RefreshLoop {
    state: Arc<AppState>,
}

#[async_trait::async_trait]
impl pingora::services::background::BackgroundService for RefreshLoop {
    async fn start(&self, mut shutdown: pingora::server::ShutdownWatch) {
        loop {
            if self.state.refresh_inflight.swap(true, std::sync::atomic::Ordering::SeqCst) {
                // 上一轮还没跑完，稍等
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                }
                continue;
            }
            let interval = self.state.main_conf().refresh_interval_secs.max(5);
            let state = self.state.clone();
            let mut handle = tokio::spawn(async move {
                let r = admin::api::refresh_all(&state).await;
                state.refresh_inflight.store(false, std::sync::atomic::Ordering::SeqCst);
                r
            });
            tokio::select! {
                r = &mut handle => match r {
                    Ok((ok_n, failed)) => {
                        log::info!("机群采集完成: 在线 {ok_n} 台, 异常 {} 台", failed.len());
                    }
                    Err(e) => log::error!("机群采集任务异常: {e}"),
                },
                _ = shutdown.changed() => {
                    let _ = handle.await;
                    break;
                }
            }
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs(interval)) => {}
            }
        }
        self.state
            .refresh_inflight
            .store(false, std::sync::atomic::Ordering::SeqCst);
        log::info!("机群采集循环退出");
    }
}

/// 文件监听：主配置 / servers.yaml / services.yaml mtime 变化时热重载；解析失败保留旧值。
struct ConfigWatcher {
    state: Arc<AppState>,
    main_path: PathBuf,
}

#[async_trait::async_trait]
impl pingora::services::background::BackgroundService for ConfigWatcher {
    async fn start(&self, mut shutdown: pingora::server::ShutdownWatch) {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
        let mut last_main = std::fs::metadata(&self.main_path)
            .and_then(|m| m.modified())
            .ok();
        let main_conf_path = || self.state.main_conf().clone();
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = tick.tick() => {}
            }
            let mc = main_conf_path();
            // 1) 主配置
            let mtime = std::fs::metadata(&self.main_path).and_then(|m| m.modified()).ok();
            if mtime != last_main {
                last_main = mtime;
                match MainConf::load(&self.main_path) {
                    Ok(c) => {
                        log::info!("主配置已热重载");
                        self.state.set_main(c);
                    }
                    Err(e) => log::error!("重载主配置失败，保留旧配置: {e}"),
                }
            }
            // 2) servers.yaml
            let sp = mc.servers_path();
            if let Ok(text) = std::fs::read_to_string(&sp) {
                match serde_yaml::from_str::<ServersFile>(&text) {
                    Ok(f) if f.servers.len() != self.state.servers.load().servers.len()
                        || serde_json::to_value(&f).unwrap_or(Value::Null)
                            != serde_json::to_value(self.state.servers.load().as_ref()).unwrap_or(Value::Null) =>
                    {
                        log::info!("servers.yaml 变更，热重载: {} 台", f.servers.len());
                        self.state.servers.store(Arc::new(f));
                    }
                    Ok(_) => {}
                    Err(e) => log::error!("重载 servers.yaml 失败，保留旧配置: {e}"),
                }
            }
            // 3) services.yaml
            let vp = mc.services_path();
            if let Ok(text) = std::fs::read_to_string(&vp) {
                match serde_yaml::from_str::<ServicesFile>(&text) {
                    Ok(f) => {
                        let cur = self.state.services.load();
                        let changed =
                            serde_json::to_value(&f).ok() != serde_json::to_value(cur.as_ref()).ok();
                        if changed {
                            log::info!("services.yaml 变更，热重载: {} 个服务", f.services.len());
                            self.state.services.store(Arc::new(f));
                        }
                    }
                    Err(e) => log::error!("重载 services.yaml 失败，保留旧配置: {e}"),
                }
            }
        }
        log::info!("配置监听服务退出");
    }
}
