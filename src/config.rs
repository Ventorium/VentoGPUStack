//! VentoGPUStack 配置模型：
//! - `MainConf`：服务端自身配置（ventogpustack.yaml）
//! - `ServersFile`：GPU 服务器清单（servers.yaml，管理端每次变更后原子落盘 + ArcSwap 热切换）
//! - `ServicesFile`：模型服务注册表（services.yaml，同上）
//! - `AdminConf`：管理端认证配置（admin.yaml）
//! - `ServerStatus`：单台机器的实时采集结果（仅内存，周期性刷新）
//! - `AppState`：各服务共享的运行时状态

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 主配置（ventogpustack.yaml）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConf {
    /// 是否启用 LLM 辅助（驱动升级研判 / 调度方案）
    pub enabled: bool,
    /// LLM 命令行，例如 `dsh`
    pub command: String,
    /// 使用 headless 的 profile，例如 `headless`
    pub profile: String,
    /// 单次调用超时（秒）
    pub timeout_secs: u64,
}

impl Default for LlmConf {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "dsh".into(),
            profile: "headless".into(),
            timeout_secs: 180,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MainConf {
    pub version: u32,
    pub admin_listen: String,
    pub threads: u32,
    pub grace_period_seconds: u64,
    pub upgrade_sock: String,
    pub pid_file: String,
    pub admin_file: String,
    pub servers_file: String,
    pub services_file: String,
    pub log_dir: String,
    /// 机群状态周期刷新间隔（秒）
    pub refresh_interval_secs: u64,
    /// 单台 SSH 执行超时（秒）
    pub ssh_timeout_secs: u64,
    /// 并发采集台数
    pub ssh_concurrency: u32,
    /// 可选：错误日志文件；留空则输出到 stdout
    #[serde(default)]
    pub error_log: Option<String>,
    pub llm: LlmConf,
}

impl Default for MainConf {
    fn default() -> Self {
        Self {
            version: 1,
            admin_listen: "0.0.0.0:8900".into(),
            threads: 4,
            grace_period_seconds: 30,
            upgrade_sock: "/opt/ventogpustack/upgrade.sock".into(),
            pid_file: "/opt/ventogpustack/ventogpustack.pid".into(),
            admin_file: "/opt/ventogpustack/admin.yaml".into(),
            servers_file: "/opt/ventogpustack/servers.yaml".into(),
            services_file: "/opt/ventogpustack/services.yaml".into(),
            log_dir: "/opt/ventogpustack/logs".into(),
            refresh_interval_secs: 30,
            ssh_timeout_secs: 25,
            ssh_concurrency: 8,
            error_log: None,
            llm: LlmConf::default(),
        }
    }
}

impl MainConf {
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("主配置文件不存在: {}", path.display()));
        }
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let conf: MainConf = serde_yaml::from_str(&text).map_err(|e| e.to_string())?;
        Ok(conf)
    }

    pub fn servers_path(&self) -> PathBuf {
        PathBuf::from(&self.servers_file)
    }

    pub fn services_path(&self) -> PathBuf {
        PathBuf::from(&self.services_file)
    }

    pub fn admin_path(&self) -> PathBuf {
        PathBuf::from(&self.admin_file)
    }

    pub fn log_dir(&self) -> PathBuf {
        PathBuf::from(&self.log_dir)
    }

    pub fn audit_path(&self) -> PathBuf {
        self.log_dir().join("audit.jsonl")
    }
}

// ---------------------------------------------------------------------------
// 服务器清单（servers.yaml）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConf {
    pub id: u64,
    pub name: String,
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    pub user: String,
    pub password: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_port() -> u16 {
    22
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServersFile {
    pub servers: Vec<ServerConf>,
}

impl ServersFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_yaml::from_str(&text).map_err(|e| format!("解析 servers.yaml 失败: {e}"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text = serde_yaml::to_string(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("yaml.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// 模型服务注册表（services.yaml）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeployType {
    Vllm,
    Sglang,
    Docker,
    Python,
}

impl DeployType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeployType::Vllm => "vllm",
            DeployType::Sglang => "sglang",
            DeployType::Docker => "docker",
            DeployType::Python => "python",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Stopped,
    Running,
    Migrating,
    Error,
}

impl Default for ServiceStatus {
    fn default() -> Self {
        ServiceStatus::Stopped
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConf {
    pub id: u64,
    pub name: String,
    /// 模型文件在机群内的绝对路径（各机器通常同构）
    pub model_path: String,
    /// 当前 / 最近一次部署所在服务器 id
    pub server_id: u64,
    pub deploy_type: DeployType,
    #[serde(default = "default_service_port")]
    pub port: u16,
    /// 张量并行卡数（vllm / sglang 使用）
    #[serde(default = "default_tp")]
    pub tp: u32,
    /// vllm / sglang 附加参数
    #[serde(default)]
    pub extra_args: String,
    #[serde(default)]
    pub docker_image: String,
    #[serde(default)]
    pub docker_args: String,
    /// python 自定义命令模板，占位符 {model} {port} {gpus}
    #[serde(default)]
    pub python_cmd: String,
    /// 停止时 pkill 的匹配模式；留空则用 model_path
    #[serde(default)]
    pub stop_pattern: String,
    #[serde(default)]
    pub status: ServiceStatus,
    #[serde(default)]
    pub pid: Option<u32>,
    #[serde(default)]
    pub notes: String,
}

fn default_service_port() -> u16 {
    8000
}

fn default_tp() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServicesFile {
    pub services: Vec<ServiceConf>,
}

impl ServicesFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_yaml::from_str(&text).map_err(|e| format!("解析 services.yaml 失败: {e}"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text = serde_yaml::to_string(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("yaml.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// 管理端认证配置（admin.yaml）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdminConf {
    pub username: String,
    /// argon2id 格式密码哈希
    pub password_hash: String,
    pub jwt_secret: String,
    pub jwt_ttl_hours: u64,
}

impl Default for AdminConf {
    fn default() -> Self {
        Self {
            username: "admin".into(),
            password_hash: String::new(),
            jwt_secret: String::new(),
            jwt_ttl_hours: 72,
        }
    }
}

impl AdminConf {
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("管理端配置文件不存在: {}", path.display()));
        }
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let conf: AdminConf = serde_yaml::from_str(&text).map_err(|e| e.to_string())?;
        if conf.jwt_secret.is_empty() {
            return Err("admin.yaml: jwt_secret 为空".into());
        }
        Ok(conf)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text = serde_yaml::to_string(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("yaml.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// argon2id 密码哈希（供 init / set-password 使用）
    pub fn generate_password_hash(password: &str) -> Result<String, String> {
        use argon2::password_hash::{
            rand_core::OsRng, SaltString,
        };
        use argon2::{Argon2, PasswordHasher};
        let salt = SaltString::generate(&mut OsRng);
        Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map(|h| h.to_string())
            .map_err(|e| e.to_string())
    }

    /// 生成 n 字节随机 JWT 密钥（hex 编码）
    pub fn generate_secret(n: usize) -> String {
        let bytes: Vec<u8> = (0..n).map(|_| rand::random::<u8>()).collect();
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// ---------------------------------------------------------------------------
// 实时采集结果（仅内存）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuStat {
    pub idx: u32,
    pub name: String,
    pub total_mib: u32,
    pub used_mib: u32,
    pub util_pct: u32,
    pub temp_c: u32,
    /// 功耗（瓦）。nvidia-smi power.draw 输出形如 `19.80`,必须用浮点。
    pub power_w: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcStat {
    pub pid: u32,
    pub name: String,
    pub gpu_idx: Vec<u32>,
    pub mem_mib: u32,
    #[serde(default)]
    pub cmdline: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    pub path: String,
    pub size_gb: f64,
    #[serde(default)]
    pub kind: String,
    /// 正在使用该模型的进程名
    #[serde(default)]
    pub in_use_by: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerStatus {
    pub online: bool,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub hostname: String,
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub ram_total_gb: u32,
    pub ram_used_gb: u32,
    pub disk_total_gb: u32,
    pub disk_used_gb: u32,
    #[serde(default)]
    pub driver_version: String,
    #[serde(default)]
    pub cuda_version: String,
    #[serde(default)]
    pub gpus: Vec<GpuStat>,
    #[serde(default)]
    pub procs: Vec<ProcStat>,
    #[serde(default)]
    pub models: Vec<ModelInfo>,
    /// 远程可用工具探测，如 `vllm=/usr/local/bin/vllm`
    #[serde(default)]
    pub tools: Vec<String>,
    pub updated_at: i64,
}

impl ServerStatus {
    pub fn offline(id: u64, error: String) -> Self {
        let _ = id;
        Self {
            online: false,
            error,
            updated_at: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// 驱动升级任务（内存态；目标机 /tmp/vgs_drvup_{id}.status 为执行进度真相源）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct DrvTask {
    pub server_id: u64,
    pub name: String,
    pub host: String,
    /// 当前驱动版本（任务创建时采集）
    #[serde(default)]
    pub current_driver: String,
    pub target_driver: String,
    #[serde(default)]
    pub target_cuda: String,
    /// 阶段：preparing/stopping/download/transferring/installing/reboot-wait/accepting/restarting/done/fail
    pub phase: String,
    pub started_at: i64,
    #[serde(default)]
    pub auto_reboot: bool,
    /// 升级前平台停止、成功后需要恢复的服务 id
    #[serde(default)]
    pub stopped_services: Vec<u64>,
    /// 网关侧流水线日志（保留最近 400 行）
    #[serde(default)]
    pub log: Vec<String>,
    /// 目标机安装脚本日志尾部
    #[serde(default)]
    pub remote_log: String,
    #[serde(default)]
    pub error: Option<String>,
    /// 安装后 AI 验收结论 {ok, driver, note}
    #[serde(default)]
    pub ai_verdict: Option<serde_json::Value>,
    /// 升级后服务恢复结果 [{name, ok, error}]
    #[serde(default)]
    pub restarted: Vec<serde_json::Value>,
    #[serde(default)]
    pub final_driver: String,
}

impl DrvTask {
    /// 任务是否仍在执行（未结束）
    pub fn active(&self) -> bool {
        !matches!(self.phase.as_str(), "done" | "fail")
    }
}

// ---------------------------------------------------------------------------
// 共享运行时状态
// ---------------------------------------------------------------------------

pub struct AppState {
    /// 主配置（ArcSwap 支持运行时热更新）
    pub main: ArcSwap<MainConf>,
    /// 主配置文件路径（settings 落盘用）
    pub main_path: PathBuf,
    pub admin: AdminConf,
    pub servers: ArcSwap<ServersFile>,
    pub services: ArcSwap<ServicesFile>,
    /// server.id -> 最新一次采集结果
    pub statuses: ArcSwap<HashMap<u64, ServerStatus>>,
    /// 是否有全量刷新在进行中
    pub refresh_inflight: std::sync::atomic::AtomicBool,
    pub audit: AuditLog,
    /// server.id -> 进行中的驱动升级任务（内存态，进程重启即清空）
    pub drv_tasks: std::sync::Arc<std::sync::Mutex<HashMap<u64, DrvTask>>>,
}

impl AppState {
    pub fn new(
        main: MainConf,
        main_path: PathBuf,
        admin: AdminConf,
        servers: ServersFile,
        services: ServicesFile,
        audit_path: PathBuf,
    ) -> Arc<Self> {
        let statuses: HashMap<u64, ServerStatus> = servers
            .servers
            .iter()
            .map(|s| (s.id, ServerStatus::offline(s.id, "等待首次采集".into())))
            .collect();
        Arc::new(Self {
            main: ArcSwap::from_pointee(main),
            main_path,
            admin,
            servers: ArcSwap::from_pointee(servers),
            services: ArcSwap::from_pointee(services),
            statuses: ArcSwap::from_pointee(statuses),
            refresh_inflight: std::sync::atomic::AtomicBool::new(false),
            audit: AuditLog::new(audit_path),
            drv_tasks: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
        })
    }

    pub fn main_conf(&self) -> std::sync::Arc<MainConf> {
        self.main.load_full()
    }

    pub fn set_main(&self, conf: MainConf) {
        self.main.store(Arc::new(conf));
    }

    pub fn find_server(&self, id: u64) -> Option<ServerConf> {
        self.servers.load().servers.iter().find(|s| s.id == id).cloned()
    }

    pub fn find_service(&self, id: u64) -> Option<ServiceConf> {
        self.services.load().services.iter().find(|s| s.id == id).cloned()
    }

    pub fn status_of(&self, id: u64) -> Option<ServerStatus> {
        self.statuses.load().get(&id).cloned()
    }
}

// ---------------------------------------------------------------------------
// 审计日志（JSONL）
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AuditEntry {
    time: String,
    user: String,
    action: String,
    target: String,
    detail: String,
    ip: String,
}

pub struct AuditLog {
    path: PathBuf,
    guard: Mutex<()>,
}

impl AuditLog {
    pub fn new(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self {
            path,
            guard: Mutex::new(()),
        }
    }

    pub fn record(&self, user: &str, action: &str, target: &str, detail: &str, ip: &str) {
        let entry = serde_json::json!({
            "time": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            "user": user,
            "action": action,
            "target": target,
            "detail": detail,
            "ip": ip,
        });
        let _guard = self.guard.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        else {
            return;
        };
        let _ = writeln!(f, "{}", entry);
    }

    pub fn tail(&self, limit: usize) -> Vec<serde_json::Value> {
        let Ok(content) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        content
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .take(limit)
            .collect()
    }
}
