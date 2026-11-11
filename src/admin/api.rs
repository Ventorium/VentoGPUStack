//! 管理端 HTTP API（网关模式：request_filter 内直接响应）。
//!
//! 路由：
//!   GET  /                      内嵌 Web UI
//!   GET  /api/health            健康检查（免认证）
//!   POST /api/login             登录，返回 JWT
//!   GET  /api/me                当前用户
//!   GET  /api/summary           机群总览
//!   GET  /api/servers           服务器列表（含实时状态）
//!   POST /api/servers           添加服务器
//!   GET  /api/servers/{id}      服务器详情
//!   PUT  /api/servers/{id}      更新服务器
//!   DELETE /api/servers/{id}    删除服务器
//!   POST /api/servers/{id}/test SSH 连通性测试
//!   POST /api/servers/{id}/refresh  单台刷新
//!   POST /api/servers/refresh-all   全量刷新（并发）
//!   GET  /api/driver-matrix     驱动/CUDA 版本矩阵 + 各机在跑负载
//!   POST /api/driver-matrix/analyze AI 驱动升级研判（失败自动回退规则）
//!   GET  /api/services          模型服务列表
//!   POST /api/services          注册模型服务
//!   PUT/DELETE /api/services/{id}
//!   POST /api/services/{id}/start   在指定（或当前）机器启动
//!   POST /api/services/{id}/stop    停止
//!   POST /api/services/{id}/plan    生成迁移方案（AI / 规则）
//!   POST /api/services/{id}/migrate 蓝绿迁移执行
//!   POST /api/services/{id}/copy    后台拷贝模型到目标机
//!   GET  /api/services/{id}/copy-status 拷贝进度查询
//!   GET  /api/audit?limit=N     操作日志
//!   GET  /api/settings          全局设置
//!   PUT  /api/settings          更新全局设置

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora::prelude::Session;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::ai;
use crate::config::*;
use crate::deploy;
use crate::discover;
use crate::scheduler;
use crate::ssh::ssh_exec;

// ---------------------------------------------------------------------------
// 载荷
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct LoginPayload {
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
}

#[derive(Deserialize, Default)]
pub struct ServerPayload {
    pub name: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub user: Option<String>,
    pub password: Option<String>,
    pub notes: Option<String>,
    pub enabled: Option<bool>,
}

#[derive(Deserialize, Default)]
pub struct ServicePayload {
    pub name: Option<String>,
    pub model_path: Option<String>,
    pub deploy_type: Option<String>,
    pub port: Option<u16>,
    pub tp: Option<u32>,
    pub extra_args: Option<String>,
    pub docker_image: Option<String>,
    pub docker_args: Option<String>,
    pub python_cmd: Option<String>,
    pub server_id: Option<u64>,
    pub stop_pattern: Option<String>,
    pub notes: Option<String>,
    /// 纳管的外部容器名（docker 方式启停时按容器名精确 stop，避免误杀同机同模型实例）
    pub container_name: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct StartPayload {
    pub server_id: Option<u64>,
}

#[derive(Deserialize)]
pub struct AnalyzePayload {
    pub target_driver: String,
    #[serde(default)]
    pub target_cuda: String,
}

#[derive(Deserialize)]
pub struct DrvUpgradePayload {
    pub target_driver: String,
    #[serde(default)]
    pub target_cuda: String,
    /// 升级前停止该机上平台托管的运行中服务，成功后自动恢复
    #[serde(default = "drv_stop_default")]
    pub stop_services: bool,
    /// 若安装后驱动未生效且勾选此项，允许自动重启目标机（不勾选则提示手动重启）
    #[serde(default)]
    pub auto_reboot: bool,
}

fn drv_stop_default() -> bool {
    true
}

#[derive(Deserialize, Default)]
pub struct PlanPayload {
    pub to_server_id: Option<u64>,
}

#[derive(Deserialize, Default)]
pub struct MigratePayload {
    pub to_server_id: Option<u64>,
    /// 目标机没有模型时仍强制启动（风险自担）
    pub force: bool,
}

#[derive(Deserialize, Default)]
pub struct CopyPayload {
    pub to_server_id: u64,
}

#[derive(Deserialize, Default)]
pub struct SettingsPayload {
    pub refresh_interval_secs: Option<u64>,
    pub ssh_timeout_secs: Option<u64>,
    pub ssh_concurrency: Option<u32>,
    pub llm: Option<LlmPayload>,
}

#[derive(Deserialize, Default)]
pub struct LlmPayload {
    pub enabled: Option<bool>,
    pub command: Option<String>,
    pub profile: Option<String>,
    pub timeout_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// 响应助手
// ---------------------------------------------------------------------------

async fn respond(session: &mut Session, status: u16, content_type: &str, body: Bytes) {
    let mut resp = match ResponseHeader::build(status, Some(body.len())) {
        Ok(r) => r,
        Err(_) => return,
    };
    let _ = resp.insert_header("Content-Type", content_type);
    let _ = resp.insert_header("X-Content-Type-Options", "nosniff");
    let end = body.is_empty();
    let _ = session.write_response_header(Box::new(resp), end).await;
    if !end {
        let _ = session.write_response_body(Some(body), true).await;
    }
}

async fn json(session: &mut Session, status: u16, v: &Value) {
    let body = Bytes::from(serde_json::to_string(v).unwrap_or_else(|_| "null".into()));
    respond(session, status, "application/json; charset=utf-8", body).await;
}

async fn fail(session: &mut Session, status: u16, msg: &str) {
    json(session, status, &json!({"ok": false, "error": msg})).await;
}

async fn ok(session: &mut Session, data: &Value) {
    json(session, 200, &json!({"ok": true, "data": data})).await;
}

async fn parse_body(session: &mut Session) -> Option<Value> {
    match session.read_request_body().await {
        Ok(Some(b)) if !b.is_empty() => serde_json::from_slice(&b).ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 采集辅助
// ---------------------------------------------------------------------------

async fn collect_and_store(state: &Arc<AppState>, s: &ServerConf) -> ServerStatus {
    let timeout = Duration::from_secs(state.main_conf().ssh_timeout_secs);
    let st = crate::collect::collect_one(&s.host, s.port, &s.user, &s.password, timeout).await;
    let mut map = state.statuses.load().as_ref().clone();
    map.insert(s.id, st.clone());
    state.statuses.store(Arc::new(map));
    st
}

/// 服务器视图：配置 + 最新状态。
fn server_view(s: &ServerConf, st: Option<&ServerStatus>) -> Value {
    let online = st.map(|x| x.online).unwrap_or(false);
    json!({
        "id": s.id,
        "name": s.name,
        "host": s.host,
        "port": s.port,
        "user": s.user,
        "notes": s.notes,
        "enabled": s.enabled,
        "online": online,
        "error": st.map(|x| &x.error).cloned().unwrap_or_default(),
        "hostname": st.map(|x| &x.hostname).cloned().unwrap_or_default(),
        "os": st.map(|x| &x.os).cloned().unwrap_or_default(),
        "cpu_model": st.map(|x| &x.cpu_model).cloned().unwrap_or_default(),
        "cpu_cores": st.map(|x| x.cpu_cores).unwrap_or(0),
        "ram_total_gb": st.map(|x| x.ram_total_gb).unwrap_or(0),
        "ram_used_gb": st.map(|x| x.ram_used_gb).unwrap_or(0),
        "disk_total_gb": st.map(|x| x.disk_total_gb).unwrap_or(0),
        "disk_used_gb": st.map(|x| x.disk_used_gb).unwrap_or(0),
        "driver_version": st.map(|x| &x.driver_version).cloned().unwrap_or_default(),
        "cuda_version": st.map(|x| &x.cuda_version).cloned().unwrap_or_default(),
        "gpus": st.map(|x| &x.gpus).cloned().unwrap_or_default(),
        "procs": st.map(|x| &x.procs).cloned().unwrap_or_default(),
        "models": st.map(|x| &x.models).cloned().unwrap_or_default(),
        "tools": st.map(|x| &x.tools).cloned().unwrap_or_default(),
        "gpu_count": st.map(|x| x.gpus.len()).unwrap_or(0),
        "vram_total_gb": st.map(|x| x.gpus.iter().map(|g| g.total_mib).sum::<u32>() as f64 / 1024.0).unwrap_or(0.0),
        "vram_used_gb": st.map(|x| x.gpus.iter().map(|g| g.used_mib).sum::<u32>() as f64 / 1024.0).unwrap_or(0.0),
        "gpu_util_max": st.map(|x| x.gpus.iter().map(|g| g.util_pct).max().unwrap_or(0)).unwrap_or(0),
        "updated_at": st.map(|x| x.updated_at).unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------

pub async fn handle(session: &mut Session, state: &Arc<AppState>) {
    let req = session.req_header();
    let method = req.method.as_str().to_string();
    let path = req.uri.path().to_string();
    let ip = session
        .client_addr()
        .and_then(|a| a.as_inet())
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|| "-".into());

    // 静态 UI
    if method == "GET" && (path == "/" || path == "/index.html") {
        let body = Bytes::from(crate::admin::web::INDEX_HTML.to_string());
        return respond(session, 200, "text/html; charset=utf-8", body).await;
    }
    // 内嵌图标
    if method == "GET" {
        let icon = match path.as_str() {
            "/favicon.png" => Some(crate::admin::web::FAVICON_PNG),
            "/logo.png" => Some(crate::admin::web::LOGO_PNG),
            "/apple-touch-icon.png" => Some(crate::admin::web::APPLE_TOUCH_ICON_PNG),
            _ => None,
        };
        if let Some(bytes) = icon {
            return respond(session, 200, "image/png", Bytes::from_static(bytes)).await;
        }
    }
    if !path.starts_with("/api/") {
        return fail(session, 404, "not found").await;
    }

    if path == "/api/health" && method == "GET" {
        return json(session, 200, &json!({"ok": true, "service": "ventogpustack"})).await;
    }

    // 登录
    if path == "/api/login" && method == "POST" {
        let v = parse_body(session).await.unwrap_or(Value::Null);
        let p: LoginPayload = serde_json::from_value(v).unwrap_or_default();
        match crate::admin::auth::verify_password(&p.password, &state.admin.password_hash) {
            Ok(true) if p.username == state.admin.username => {
                let token = crate::admin::auth::issue_token(
                    &state.admin.jwt_secret,
                    &p.username,
                    state.admin.jwt_ttl_hours,
                )
                .unwrap_or_default();
                state.audit.record(&p.username, "login", &p.username, "登录成功", &ip);
                return json(
                    session,
                    200,
                    &json!({"ok": true, "data": {"token": token, "username": p.username}}),
                )
                .await;
            }
            _ => {
                state.audit.record(&p.username, "login_failed", "-", "用户名或密码错误", &ip);
                return fail(session, 401, "用户名或密码错误").await;
            }
        }
    }

    // 其余接口需要 Bearer 令牌
    let token = req
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .strip_prefix("Bearer ")
        .unwrap_or("")
        .to_string();
    let user = match crate::admin::auth::verify_token(&state.admin.jwt_secret, &token) {
        Ok(u) => u,
        Err(_) => return fail(session, 401, "未登录或会话已过期").await,
    };

    // 路由分发
    let rest = path.trim_start_matches("/api/");
    let mut parts = rest.split('/');
    let head = parts.next().unwrap_or("");
    let id_part = parts.next();
    let third = parts.next();

    match head {
        "me" if method == "GET" => {
            json(session, 200, &json!({"ok": true, "data": {"username": user}})).await
        }
        "summary" if method == "GET" => route_summary(session, state).await,
        "servers" => routes_servers(session, state, &user, &ip, &method, id_part, third).await,
        "driver-matrix" => {
            routes_driver_matrix(session, state, &user, &ip, &method, id_part).await
        }
        "driver-upgrades" if method == "GET" => route_drv_upgrades_list(session, state).await,
        "router-compat" if method == "POST" => {
            route_router_compat_check(session, state, &user, &ip).await
        }
        "router-compat-kb" if method == "GET" => route_router_compat_kb(session).await,
        "services" => routes_services(session, state, &user, &ip, &method, id_part, third).await,
        "discover" => {
            routes_discover(session, state, &user, &ip, &method, id_part).await
        }
        "audit" if method == "GET" => {
            let limit = req
                .uri
                .query()
                .and_then(|q| {
                    q.split('&')
                        .find_map(|kv| kv.strip_prefix("limit="))
                        .and_then(|v| v.parse::<usize>().ok())
                })
                .unwrap_or(200)
                .min(2000);
            let rows = state.audit.tail(limit);
            json(session, 200, &json!({"ok": true, "data": rows})).await
        }
        "settings" => routes_settings(session, state, &user, &ip, &method).await,
        _ => fail(session, 404, "not found").await,
    }
}

// ---------------------------------------------------------------------------
// 总览
// ---------------------------------------------------------------------------

async fn route_summary(session: &mut Session, state: &Arc<AppState>) {
    let servers = state.servers.load();
    let statuses = state.statuses.load();
    let services = state.services.load();

    let mut online = 0usize;
    let mut gpu_total = 0u32;
    let mut gpu_busy = 0u32;
    let mut vram_total = 0u32;
    let mut vram_used = 0u32;
    let mut ram_total = 0u32;
    let mut ram_used = 0u32;
    let mut disk_total = 0u32;
    let mut disk_used = 0u32;
    for s in &servers.servers {
        let Some(st) = statuses.get(&s.id) else { continue };
        if st.online {
            online += 1;
            gpu_total += st.gpus.len() as u32;
            gpu_busy += st.gpus.iter().filter(|g| g.util_pct > 0 || g.used_mib > 512).count() as u32;
            vram_total += st.gpus.iter().map(|g| g.total_mib).sum::<u32>();
            vram_used += st.gpus.iter().map(|g| g.used_mib).sum::<u32>();
            ram_total += st.ram_total_gb;
            ram_used += st.ram_used_gb;
            disk_total += st.disk_total_gb;
            disk_used += st.disk_used_gb;
        }
    }
    let running = services
        .services
        .iter()
        .filter(|s| s.status == ServiceStatus::Running)
        .count();
    ok(
        session,
        &json!({
            "server_total": servers.servers.len(),
            "server_online": online,
            "gpu_total": gpu_total,
            "gpu_busy": gpu_busy,
            "vram_total_gb": vram_total as f64 / 1024.0,
            "vram_used_gb": vram_used as f64 / 1024.0,
            "ram_total_gb": ram_total,
            "ram_used_gb": ram_used,
            "disk_total_gb": disk_total,
            "disk_used_gb": disk_used,
            "service_total": services.services.len(),
            "service_running": running,
        }),
    )
    .await
}

// ---------------------------------------------------------------------------
// 服务器
// ---------------------------------------------------------------------------

async fn routes_servers(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    method: &str,
    id_part: Option<&str>,
    third: Option<&str>,
) {
    // 无 id
    if id_part.is_none() {
        match (method, third) {
            ("GET", None) => return route_servers_list(session, state).await,
            ("POST", None) => return route_server_create(session, state, user, ip).await,
            ("POST", Some("refresh-all")) => {
                return route_refresh_all(session, state, user, ip).await;
            }
            _ => return fail(session, 404, "not found").await,
        }
    }

    let id: u64 = match id_part.and_then(|p| p.parse().ok()) {
        Some(i) => i,
        None => return fail(session, 404, "not found").await,
    };
    match (method, third) {
        ("GET", None) => {
            let Some(s) = state.find_server(id) else {
                return fail(session, 404, "服务器不存在").await;
            };
            let st = state.status_of(id);
            ok(session, &server_view(&s, st.as_ref())).await
        }
        ("PUT", None) => route_server_update(session, state, user, ip, id).await,
        ("DELETE", None) => {
            let Some(s) = state.find_server(id) else {
                return fail(session, 404, "服务器不存在").await;
            };
            let in_use: Vec<String> = state
                .services
                .load()
                .services
                .iter()
                .filter(|x| x.server_id == id)
                .map(|x| x.name.clone())
                .collect();
            if !in_use.is_empty() {
                return fail(
                    session,
                    409,
                    &format!("该服务器上仍有模型服务: {}", in_use.join("、")),
                )
                .await;
            }
            let mut servers = state.servers.load().as_ref().clone();
            let before = servers.servers.len();
            servers.servers.retain(|x| x.id != id);
            if servers.servers.len() == before {
                return fail(session, 404, "服务器不存在").await;
            }
            if let Err(e) = servers.save(&state.main_conf().servers_path()) {
                return fail(session, 500, &e).await;
            }
            state.servers.store(Arc::new(servers));
            let mut map = state.statuses.load().as_ref().clone();
            map.remove(&id);
            state.statuses.store(Arc::new(map));
            state
                .audit
                .record(user, "delete_server", &format!("{}:{}", s.name, s.host), "删除服务器", ip);
            ok(session, &json!({"id": id})).await
        }
        ("POST", Some("test")) => {
            let Some(s) = state.find_server(id) else {
                return fail(session, 404, "服务器不存在").await;
            };
            let t = Duration::from_secs(state.main_conf().ssh_timeout_secs.min(15));
            let r = ssh_exec(
                &s.host,
                s.port,
                &s.user,
                &s.password,
                "hostname; nvidia-smi -L 2>/dev/null | head -3 || echo no-gpu",
                t,
            )
            .await;
            match r {
                Ok(out) => {
                    state
                        .audit
                        .record(user, "test_server", &format!("{}:{}", s.name, s.host), &out.stdout, ip);
                    ok(session, &json!({"detail": out.stdout, "exit_code": out.exit_code})).await
                }
                Err(e) => {
                    state
                        .audit
                        .record(user, "test_server_failed", &format!("{}:{}", s.name, s.host), &e, ip);
                    fail(session, 502, &e).await
                }
            }
        }
        ("POST", Some("refresh")) => {
            let Some(s) = state.find_server(id) else {
                return fail(session, 404, "服务器不存在").await;
            };
            let st = collect_and_store(state, &s).await;
            state.audit.record(
                user,
                "refresh",
                &format!("{}:{}", s.name, s.host),
                if st.online { "采集成功" } else { &st.error },
                ip,
            );
            ok(session, &server_view(&s, Some(&st))).await
        }
        ("POST", Some("driver-upgrade")) => {
            route_drv_upgrade_start(session, state, user, ip, id).await
        }
        ("GET", Some("driver-upgrade")) => {
            route_drv_upgrade_status(session, state, id).await
        }
        _ => fail(session, 405, "method not allowed").await,
    }
}

async fn route_servers_list(session: &mut Session, state: &Arc<AppState>) {
    let servers = state.servers.load();
    let statuses = state.statuses.load();
    let rows: Vec<Value> = servers
        .servers
        .iter()
        .map(|s| server_view(s, statuses.get(&s.id)))
        .collect();
    json(session, 200, &json!({"ok": true, "data": rows})).await
}

async fn route_server_create(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: ServerPayload = serde_json::from_value(v).unwrap_or_default();
    let host = p.host.as_deref().map(str::trim).unwrap_or("").to_string();
    if host.is_empty() {
        return fail(session, 400, "host 不能为空").await;
    }
    let ssh_user = p.user.as_deref().map(str::trim).unwrap_or("").to_string();
    if ssh_user.is_empty() {
        return fail(session, 400, "SSH 用户名不能为空").await;
    }
    let mut servers = state.servers.load().as_ref().clone();
    if servers.servers.iter().any(|s| s.host == host) {
        return fail(session, 409, "该 host 已存在").await;
    }
    let new_id = servers.servers.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    let name = p
        .name
        .filter(|x| !x.trim().is_empty())
        .unwrap_or_else(|| format!("gpu-{}", new_id));
    let s = ServerConf {
        id: new_id,
        name: name.clone(),
        host: host.clone(),
        port: p.port.unwrap_or(22),
        user: ssh_user,
        password: p.password.unwrap_or_default(),
        notes: p.notes.unwrap_or_default(),
        enabled: p.enabled.unwrap_or(true),
    };
    servers.servers.push(s.clone());
    if let Err(e) = servers.save(&state.main_conf().servers_path()) {
        return fail(session, 500, &e).await;
    }
    state.servers.store(Arc::new(servers));
    let mut map = state.statuses.load().as_ref().clone();
    map.entry(s.id)
        .or_insert_with(|| ServerStatus::offline(s.id, "等待首次采集".into()));
    state.statuses.store(Arc::new(map));
    state
        .audit
        .record(&user, "add_server", &format!("{name}:{host}"), "添加服务器", ip);
    ok(session, &json!({"id": new_id})).await
}

async fn route_server_update(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    id: u64,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: ServerPayload = serde_json::from_value(v).unwrap_or_default();
    let mut servers = state.servers.load().as_ref().clone();
    let Some(s) = servers.servers.iter_mut().find(|x| x.id == id) else {
        return fail(session, 404, "服务器不存在").await;
    };
    if let Some(x) = p.name {
        if !x.trim().is_empty() {
            s.name = x;
        }
    }
    if let Some(x) = p.host {
        if !x.trim().is_empty() {
            s.host = x.trim().to_string();
        }
    }
    if let Some(x) = p.port {
        s.port = x;
    }
    if let Some(x) = p.user {
        if !x.trim().is_empty() {
            s.user = x;
        }
    }
    if let Some(x) = p.password {
        if !x.is_empty() {
            s.password = x;
        }
    }
    if let Some(x) = p.notes {
        s.notes = x;
    }
    if let Some(x) = p.enabled {
        s.enabled = x;
    }
    let label = format!("{}:{}", s.name, s.host);
    if let Err(e) = servers.save(&state.main_conf().servers_path()) {
        return fail(session, 500, &e).await;
    }
    state.servers.store(Arc::new(servers));
    state.audit.record(user, "update_server", &label, "更新服务器", ip);
    ok(session, &json!({"id": id})).await
}

async fn route_refresh_all(session: &mut Session, state: &Arc<AppState>, user: &str, ip: &str) {
    let (ok_n, failed) = refresh_all(state).await;
    state.audit.record(
        user,
        "refresh_all",
        "-",
        &format!("成功 {ok_n} 台，失败 {} 台", failed.len()),
        ip,
    );
    ok(session, &json!({"ok_n": ok_n, "failed": failed})).await
}

/// 全量刷新（并发受 ssh_concurrency 限制）。供 HTTP 与后台循环共用。
pub async fn refresh_all(state: &Arc<AppState>) -> (usize, Vec<Value>) {
    let servers: Vec<ServerConf> = state
        .servers
        .load()
        .servers
        .iter()
        .filter(|s| s.enabled)
        .cloned()
        .collect();
    let conf = state.main_conf();
    let sem = Arc::new(tokio::sync::Semaphore::new(conf.ssh_concurrency.max(1) as usize));
    let mut handles = Vec::new();
    for s in servers {
        let sem = sem.clone();
        let state = state.clone();
        handles.push(tokio::spawn(async move {
            let _perm = sem.acquire().await;
            collect_and_store(&state, &s).await
        }));
    }
    let mut ok_n = 0usize;
    let mut failed = Vec::new();
    for h in handles {
        match h.await {
            Ok(st) if st.online => ok_n += 1,
            Ok(st) => failed.push(json!({"error": st.error})),
            Err(e) => failed.push(json!({"error": e.to_string()})),
        }
    }
    (ok_n, failed)
}

// ---------------------------------------------------------------------------
// 驱动矩阵
// ---------------------------------------------------------------------------

async fn routes_driver_matrix(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    method: &str,
    second: Option<&str>,
) {
    if second == Some("analyze") && method == "POST" {
        return route_analyze(session, state, user, ip).await;
    }
    if second == Some("compat-check") && method == "POST" {
        return route_compat_check(session, state, user, ip).await;
    }
    if second == Some("kb") && method == "GET" {
        return route_compat_kb(session, state).await;
    }
    if second == Some("agent-plan") && method == "POST" {
        return route_agent_plan(session, state, user, ip).await;
    }
    if second == Some("agent-execute") && method == "POST" {
        return route_agent_execute(session, state, user, ip).await;
    }
    if second == Some("agent-status") && method == "GET" {
        return route_agent_status(session, state).await;
    }
    if method != "GET" {
        return fail(session, 405, "method not allowed").await;
    }

    let servers = state.servers.load();
    let statuses = state.statuses.load();
    let services_all = state.services.load();
    let mut distinct: Vec<String> = Vec::new();
    let rows: Vec<Value> = servers
        .servers
        .iter()
        .map(|s| {
            let st = statuses.get(&s.id);
            let drv = st.map(|x| &x.driver_version).cloned().unwrap_or_default();
            if !drv.is_empty() && !distinct.contains(&drv) {
                distinct.push(drv.clone());
            }
            let server_services: Vec<&ServiceConf> = services_all
                .services
                .iter()
                .filter(|sv| sv.server_id == s.id)
                .collect();
            let unmanaged = st.map(|x| count_unmanaged_gpu_procs(&server_services, x)).unwrap_or(0);
            json!({
                "id": s.id,
                "name": s.name,
                "host": s.host,
                "online": st.map(|x| x.online).unwrap_or(false),
                "driver": drv,
                "cuda": st.map(|x| &x.cuda_version).cloned().unwrap_or_default(),
                "gpus": st.map(|x| x.gpus.len()).unwrap_or(0),
                "vram_used_gb": st.map(|x| x.gpus.iter().map(|g| g.used_mib).sum::<u32>() as f64 / 1024.0).unwrap_or(0.0),
                "procs": st.map(|x| x.procs.iter().map(|p| p.name.clone()).collect::<Vec<_>>()).unwrap_or_default(),
                "unmanaged_gpu_procs": unmanaged,
                "models_in_use": st.map(|x| x.models.iter().filter(|m| !m.in_use_by.is_empty())
                    .map(|m| json!({"path": m.path, "by": m.in_use_by})).collect::<Vec<_>>()).unwrap_or_default(),
                "services": server_services.iter().map(|sv| json!({
                    "id": sv.id,
                    "name": sv.name,
                    "port": sv.port,
                    "deploy_type": sv.deploy_type,
                    "status": sv.status,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    json(
        session,
        200,
        &json!({"ok": true, "data": {"rows": rows, "distinct_drivers": distinct}}),
    )
    .await
}

async fn route_analyze(session: &mut Session, state: &Arc<AppState>, user: &str, ip: &str) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: AnalyzePayload = match serde_json::from_value(v) {
        Ok(x) => x,
        Err(_) => return fail(session, 400, "参数错误").await,
    };
    if p.target_driver.trim().is_empty() {
        return fail(session, 400, "target_driver 不能为空").await;
    }

    let servers = state.servers.load().servers.clone();
    let statuses = state.statuses.load();
    let pairs: Vec<(ServerConf, ServerStatus)> = servers
        .iter()
        .filter_map(|s| statuses.get(&s.id).map(|st| (s.clone(), st.clone())))
        .collect();
    if pairs.is_empty() {
        return fail(session, 409, "还没有服务器状态，请先刷新").await;
    }

    let llm = state.main_conf().llm.clone();
    let mut ai_used = false;
    let mut note = String::new();
    let raw: Value = if llm.enabled {
        match tokio::time::timeout(
            Duration::from_secs(llm.timeout_secs),
            ai::ask_json(&llm, &ai::driver_upgrade_prompt(&p.target_driver, &p.target_cuda, &pairs)),
        )
        .await
        {
            Ok(Ok(r)) => {
                ai_used = true;
                r
            }
            Ok(Err(e)) => {
                note = format!("AI 分析失败（{e}），已回退规则引擎");
                rule_driver_analysis(&p.target_driver, &pairs)
            }
            Err(_) => {
                note = "AI 分析超时，已回退规则引擎".into();
                rule_driver_analysis(&p.target_driver, &pairs)
            }
        }
    } else {
        note = "LLM 未启用，使用规则引擎".into();
        rule_driver_analysis(&p.target_driver, &pairs)
    };
    let raw = ai::unwrap_llm_json(&raw);

    // 规范化：确保每台机器都有条目
    let obj = raw
        .get("servers")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut rows: Vec<Value> = Vec::new();
    for (s, st) in &pairs {
        let entry = obj.get(&s.host).cloned().unwrap_or(Value::Null);
        rows.push(json!({
            "id": s.id,
            "host": s.host,
            "name": s.name,
            "current_driver": st.driver_version,
            "current_cuda": st.cuda_version,
            "safe": entry.get("safe").and_then(|v| v.as_bool()).unwrap_or(rule_safe(st)),
            "needs_stop": entry.get("needs_stop").and_then(|v| v.as_bool()).unwrap_or(!st.procs.is_empty()),
            "stop_services": entry.get("stop_services").cloned().unwrap_or_else(|| {
                serde_json::to_value(st.procs.iter().map(|p| p.name.clone()).collect::<Vec<_>>()).unwrap()
            }),
            "risk": entry.get("risk").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
            "reason": entry.get("reason").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        }));
    }

    state
        .audit
        .record(user, "analyze_driver", &p.target_driver, &format!("ai={ai_used} {note}"), ip);
    ok(
        session,
        &json!({"target_driver": p.target_driver, "target_cuda": p.target_cuda, "ai": ai_used, "note": note, "servers": rows}),
    )
    .await
}

fn rule_safe(st: &ServerStatus) -> bool {
    st.online && st.procs.is_empty()
}

/// 规则引擎：无进程 → 安全；有进程 → 需先停服务。
fn rule_driver_analysis(_target: &str, pairs: &[(ServerConf, ServerStatus)]) -> Value {
    let mut obj = serde_json::Map::new();
    for (s, st) in pairs {
        if !st.online {
            continue;
        }
        let has_load = !st.procs.is_empty();
        obj.insert(
            s.host.clone(),
            json!({
                "safe": !has_load,
                "needs_stop": has_load,
                "stop_services": st.procs.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
                "risk": if has_load { "medium" } else { "low" },
                "reason": if has_load {
                    format!(
                        "存在 GPU 进程（{}），升级驱动需先停止相关服务",
                        st.procs.iter().map(|p| p.name.clone()).collect::<Vec<_>>().join(",")
                    )
                } else {
                    "机器空闲，可直接升级".to_string()
                },
            }),
        );
    }
    json!({"servers": obj})
}

// ---------------------------------------------------------------------------
// 驱动升级执行（平台驱动：预检 → 停服务 → 网关下载 → 传输 → 安装 → 看门狗 → AI 验收 → 恢复服务）
// ---------------------------------------------------------------------------

/// 驱动版本号校验：三段数字，如 595.71.05
fn valid_driver_ver(v: &str) -> bool {
    let parts: Vec<&str> = v.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

fn drv_snapshot(state: &Arc<AppState>, id: u64) -> Option<DrvTask> {
    state.drv_tasks.lock().ok().and_then(|m| m.get(&id).cloned())
}

/// 短暂持锁修改任务（禁止在闭包里 await）
fn drv_set(state: &Arc<AppState>, id: u64, f: impl FnOnce(&mut DrvTask)) {
    if let Ok(mut m) = state.drv_tasks.lock() {
        if let Some(t) = m.get_mut(&id) {
            f(t);
        }
    }
}

fn drv_log(state: &Arc<AppState>, id: u64, line: String) {
    drv_set(state, id, |t| {
        t.log.push(line);
        if t.log.len() > 400 {
            t.log.drain(..t.log.len() - 400);
        }
    });
}

fn drv_phase(state: &Arc<AppState>, id: u64, phase: &str) {
    drv_log(state, id, format!("[phase] {}", phase));
    drv_set(state, id, |t| t.phase = phase.to_string());
}

fn drv_fail(state: &Arc<AppState>, id: u64, msg: String) {
    drv_log(state, id, format!("[error] {}", msg));
    drv_set(state, id, |t| {
        t.phase = "fail".to_string();
        t.error = Some(msg);
    });
}

fn drv_audit_fail(state: &Arc<AppState>, user: &str, ip: &str, server: &ServerConf, target: &str, msg: &str) {
    state.audit.record(
        user,
        "driver_upgrade_failed",
        &format!("{}:{} → {}", server.name, server.host, target),
        msg,
        ip,
    );
}

/// 规则验收：采集一次，驱动版本与目标一致且 GPU 列表非空
async fn drv_accept_rule(state: &Arc<AppState>, server: &ServerConf, target: &str) -> bool {
    let st = collect_and_store(state, server).await;
    st.online && st.driver_version == target && !st.gpus.is_empty()
}

/// 恢复升级前停止的平台服务（尽力而为；失败只记录，供前端提示手动启动）
async fn drv_restore_services(
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    server: &ServerConf,
    ids: &[u64],
) -> Vec<Value> {
    let mut out = Vec::new();
    for sid in ids {
        let Some(svc) = state.find_service(*sid) else {
            out.push(json!({"id": sid, "name": format!("#{sid}"), "ok": false, "error": "服务已不存在"}));
            continue;
        };
        if svc.status == ServiceStatus::Running {
            out.push(json!({"id": sid, "name": svc.name, "ok": true, "error": ""}));
            continue;
        }
        drv_log(state, server.id, format!("恢复服务：{}", svc.name));
        match start_service_core(state, user, ip, &svc, server.id).await {
            Ok(_) => out.push(json!({"id": sid, "name": svc.name, "ok": true, "error": ""})),
            Err((_, e)) => out.push(json!({"id": sid, "name": svc.name, "ok": false, "error": e})),
        }
    }
    out
}

/// POST /api/servers/{id}/driver-upgrade — 启动驱动升级（立即返回 202，进度经 GET 轮询）
async fn route_drv_upgrade_start(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    id: u64,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: DrvUpgradePayload = match serde_json::from_value(v) {
        Ok(x) => x,
        Err(_) => return fail(session, 400, "参数错误：需要 target_driver（如 595.71.05）").await,
    };
    let target = p.target_driver.trim().to_string();
    if target.is_empty() {
        return fail(session, 400, "target_driver 不能为空").await;
    }
    if !valid_driver_ver(&target) {
        return fail(
            session,
            400,
            &format!("版本号格式不正确：{target}。需要三段数字 X.YY.ZZ，例如 595.71.05"),
        )
        .await;
    }
    let Some(server) = state.find_server(id) else {
        return fail(session, 404, "服务器不存在").await;
    };
    if !server.enabled {
        return fail(session, 409, "该服务器已停用，不能升级").await;
    }
    let st = match state.status_of(id) {
        Some(s) => s,
        None => return fail(session, 409, "该机器状态未知，请先在服务器列表刷新后再升级").await,
    };
    if !st.online {
        return fail(session, 409, &format!("机器离线（{}），不能升级", st.error)).await;
    }
    if !st.driver_version.is_empty() && st.driver_version == target {
        return fail(session, 409, &format!("当前驱动已是 {target}，无需升级")).await;
    }
    let busy = state
        .drv_tasks
        .lock()
        .ok()
        .and_then(|m| m.get(&id).and_then(|t| t.active().then(|| t.name.clone())));
    if let Some(name) = busy {
        return fail(
            session,
            409,
            &format!("该机器已有升级任务进行中（{}），请先在升级进度里查看", name),
        )
        .await;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let task = DrvTask {
        server_id: id,
        name: server.name.clone(),
        host: server.host.clone(),
        current_driver: st.driver_version.clone(),
        target_driver: target.clone(),
        target_cuda: p.target_cuda.trim().to_string(),
        phase: "preparing".to_string(),
        started_at: now,
        auto_reboot: p.auto_reboot,
        stopped_services: Vec::new(),
        log: Vec::new(),
        remote_log: String::new(),
        error: None,
        ai_verdict: None,
        restarted: Vec::new(),
        final_driver: String::new(),
    };
    if let Ok(mut m) = state.drv_tasks.lock() {
        m.insert(id, task);
    }
    state.audit.record(
        user,
        "driver_upgrade_start",
        &format!("{}:{} → {}", server.name, server.host, target),
        &format!("stop_services={} auto_reboot={}", p.stop_services, p.auto_reboot),
        ip,
    );
    drv_log(&state, id, format!("升级任务创建：目标 {}（当前 {}）", target, st.driver_version));
    tokio::spawn(drv_pipeline(
        state.clone(),
        user.to_string(),
        ip.to_string(),
        id,
        target.clone(),
        p.stop_services,
        p.auto_reboot,
    ));
    json(
        session,
        202,
        &json!({"ok": true, "data": {"server_id": id, "phase": "preparing", "target_driver": target}}),
    )
    .await
}

// ---------------------------------------------------------------------------
// 驱动兼容检查（知识库 + LLM/规则）与 Agent 操作助手
// ---------------------------------------------------------------------------

/// 驱动-CUDA 兼容知识库（来源与机群验证记录见 kb/driver_compat.json）。
const COMPAT_KB: &str = include_str!("kb/driver_compat.json");

/// 驱动版本解析为三段数字元组（缺段补 0，如 "570.181" → [570,181,0]）。
fn ver3(v: &str) -> [u32; 3] {
    let mut a = [0u32; 3];
    for (i, p) in v.split('.').enumerate().take(3) {
        a[i] = p.parse().unwrap_or(0);
    }
    a
}

/// 比较两个驱动版本号。
fn cmp_driver_ver(a: &str, b: &str) -> std::cmp::Ordering {
    ver3(a).cmp(&ver3(b))
}

/// 判定单个 GPU 进程是否被注册服务纳管（含祖先链）。
/// sglang 等引擎会改写子进程标题（sglang::scheduler_TPx 等），其 cmdline 不再包含
/// 模型路径，因此沿 ppid 祖先链上溯（最多 8 跳，防环），任一祖先命中即视为纳管。
fn proc_managed_by(svcs: &[&ServiceConf], by_pid: &HashMap<u32, &ProcStat>, p: &ProcStat) -> bool {
    let matches = |x: &ProcStat| {
        svcs.iter().any(|sv| {
            (!sv.model_path.is_empty() && x.cmdline.contains(&sv.model_path))
                || (!sv.stop_pattern.trim().is_empty()
                    && x.cmdline.contains(sv.stop_pattern.trim()))
                || sv.pid.map(|pid| pid == x.pid).unwrap_or(false)
        })
    };
    if matches(p) {
        return true;
    }
    let mut cur = p.ppid;
    let mut hops = 0usize;
    while cur != 0 && hops < 8 {
        match by_pid.get(&cur) {
            Some(ap) => {
                if matches(ap) {
                    return true;
                }
                if ap.ppid == cur {
                    break; // 防环
                }
                cur = ap.ppid;
                hops += 1;
            }
            None => break, // 祖先不在 GPU 进程集内 → 继续上溯已无意义
        }
    }
    false
}

/// 统计机器上未纳管的 GPU 进程数（进程及祖先链均不匹配任一注册服务）。
fn count_unmanaged_gpu_procs(svcs: &[&ServiceConf], st: &ServerStatus) -> usize {
    let by_pid: HashMap<u32, &ProcStat> = st.procs.iter().map(|x| (x.pid, x)).collect();
    st.procs
        .iter()
        .filter(|p| !proc_managed_by(svcs, &by_pid, p))
        .count()
}

/// 从进程 cmdline 中提取在跑负载所需的最高 CUDA 版本（docker 镜像 tag cu12x，如 cu129 → 12.9）。
fn required_cuda_from_procs(st: &ServerStatus) -> Option<(u32, u32)> {
    let mut best: Option<(u32, u32)> = None;
    for p in &st.procs {
        let c: Vec<char> = p.cmdline.chars().collect();
        let mut i = 0usize;
        while i + 4 < c.len() {
            if c[i] == 'c' && c[i + 1] == 'u' {
                let (Some(m1), Some(m2), Some(mn)) = (
                    c[i + 2].to_digit(10),
                    c[i + 3].to_digit(10),
                    c[i + 4].to_digit(10),
                ) else {
                    i += 1;
                    continue;
                };
                let followed = i + 5 < c.len() && c[i + 5].is_ascii_digit();
                let preceded = i > 0 && c[i - 1].is_ascii_alphanumeric();
                if !followed && !preceded && (11..=13).contains(&m1) && mn <= 9 {
                    let v = (m1 * 10 + m2, mn);
                    if best.map_or(true, |b| v > b) {
                        best = Some(v);
                    }
                }
            }
            i += 1;
        }
    }
    best
}

/// 规则兜底兼容判定：目标驱动低于在跑负载所需最低驱动 → block；
/// 高于机群已验证最高驱动 → warn（灰度建议）；其余 safe。
fn rule_compat_check(st: &ServerStatus, target: &str, unmanaged: usize) -> Value {
    let kb: Value = serde_json::from_str(COMPAT_KB).unwrap_or(Value::Null);
    let req = required_cuda_from_procs(st);
    let req_cuda_str = match req {
        Some((maj, mn)) => format!("{maj}.{mn}"),
        None => st.cuda_version.trim().to_string(),
    };
    let min_driver = kb
        .get("cuda_min_driver")
        .and_then(|x| x.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|e| e.get("cuda").and_then(|c| c.as_str()) == Some(req_cuda_str.as_str()))
        })
        .and_then(|e| e.get("min_driver"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let mut reasons: Vec<String> = Vec::new();
    let mut verdict = "safe".to_string();
    let mut max_compatible = String::new();
    if let Some(md) = &min_driver {
        if cmp_driver_ver(target, md) == std::cmp::Ordering::Less {
            verdict = "block".to_string();
            reasons.push(format!(
                "在跑负载需要 CUDA {req_cuda_str}（最低驱动 {md}），目标 {target} 低于该版本，升级后负载将无法运行"
            ));
        } else {
            reasons.push(format!(
                "在跑负载需要 CUDA {req_cuda_str}（最低驱动 {md}），目标 {target} 满足要求，理论兼容"
            ));
        }
        if let Some(obs) = kb.get("observed_fleet").and_then(|x| x.as_array()) {
            let mut best: Option<String> = None;
            for e in obs {
                let (Some(d), Some(c)) = (
                    e.get("driver").and_then(|x| x.as_str()),
                    e.get("cuda").and_then(|x| x.as_str()),
                ) else {
                    continue;
                };
                let cuda_ok = match req {
                    Some((maj, mn)) => {
                        let (om, omn) = match c.split_once('.') {
                            Some((a, b)) => (a.parse::<u32>().unwrap_or(0), b.parse::<u32>().unwrap_or(0)),
                            None => (0, 0),
                        };
                        (om, omn) >= (maj, mn)
                    }
                    None => true,
                };
                if cuda_ok
                    && cmp_driver_ver(d, md) != std::cmp::Ordering::Less
                    && best.as_ref().map_or(true, |b| cmp_driver_ver(d, b) == std::cmp::Ordering::Greater)
                {
                    best = Some(d.to_string());
                }
            }
            max_compatible = best.unwrap_or_else(|| target.to_string());
            if verdict != "block" && cmp_driver_ver(target, &max_compatible) == std::cmp::Ordering::Greater {
                verdict = "warn".to_string();
                reasons.push(format!(
                    "目标 {target} 超出机群已验证最高驱动 {max_compatible}（已验证支持 CUDA {req_cuda_str}）：驱动向后兼容，理论可行但无集群验证案例，建议单节点灰度"
                ));
            }
        }
    } else {
        verdict = "warn".to_string();
        reasons.push(format!("知识库未收录 CUDA {req_cuda_str} 的最低驱动要求，请人工核对 NVIDIA 官方文档"));
    }
    if req.is_none() && !st.procs.is_empty() {
        reasons.push("进程命令行未发现 cu12x 镜像标识，按当前驱动 CUDA 版本作为下限（保守估计）".to_string());
    }
    if unmanaged > 0 {
        if verdict == "safe" {
            verdict = "warn".to_string();
        }
        reasons.push(format!(
            "存在 {unmanaged} 个未纳管的 GPU 进程，平台不会强杀未知进程，升级前请人工确认其兼容性"
        ));
    }
    json!({
        "verdict": verdict,
        "max_compatible": max_compatible,
        "required_cuda": req_cuda_str,
        "min_required_driver": min_driver.unwrap_or_default(),
        "reasons": reasons,
    })
}

#[derive(Deserialize, Default)]
pub struct CompatCheckPayload {
    #[serde(default)]
    pub server_id: u64,
    #[serde(default)]
    pub target_driver: String,
}

/// POST /api/driver-matrix/compat-check {server_id, target_driver}
/// 驱动升级兼容性检查：LLM + 知识库研判，失败回退规则引擎；规则引擎对 block 具有一票否决。
async fn route_compat_check(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: CompatCheckPayload = match serde_json::from_value(v) {
        Ok(x) => x,
        Err(_) => return fail(session, 400, "参数错误：需要 server_id 与 target_driver").await,
    };
    let target = p.target_driver.trim().to_string();
    if !valid_driver_ver(&target) {
        return fail(
            session,
            400,
            &format!("版本号格式不正确：{target}。需要三段数字 X.YY.ZZ"),
        )
        .await;
    }
    let Some(server) = state.find_server(p.server_id) else {
        return fail(session, 404, "服务器不存在").await;
    };
    let Some(st) = state.status_of(p.server_id) else {
        return fail(session, 409, "该机器状态未知，请先在服务器列表刷新后再检查").await;
    };
    if !st.online {
        return fail(session, 409, &format!("机器离线（{}），不能检查", st.error)).await;
    }
    let services_all = state.services.load();
    let server_services: Vec<&ServiceConf> = services_all
        .services
        .iter()
        .filter(|sv| sv.server_id == p.server_id)
        .collect();
    let unmanaged = count_unmanaged_gpu_procs(&server_services, &st);

    let llm = state.main_conf().llm.clone();
    let prompt = ai::compat_check_prompt(&server, &st, &target, COMPAT_KB);
    let mut ai_used = false;
    let mut verdict: Value = Value::Null;
    if llm.enabled {
        if let Ok(Ok(raw)) =
            tokio::time::timeout(Duration::from_secs(llm.timeout_secs), ai::ask_json(&llm, &prompt))
                .await
        {
            let parsed = ai::unwrap_llm_json(&raw);
            if parsed.get("verdict").and_then(|x| x.as_str()).is_some() {
                verdict = parsed;
                ai_used = true;
            }
        }
    }
    let rule = rule_compat_check(&st, &target, unmanaged);
    if !ai_used {
        verdict = rule;
    } else {
        // 规则引擎对 block 一票否决（防 LLM 幻觉放行）；缺失字段用规则值补齐
        if rule.get("verdict").and_then(|x| x.as_str()) == Some("block")
            && verdict.get("verdict").and_then(|x| x.as_str()) != Some("block")
        {
            verdict["verdict"] = json!("block");
            verdict["reasons"] = rule["reasons"].clone();
        }
        for key in ["min_required_driver", "max_compatible", "required_cuda"] {
            if verdict
                .get(key)
                .and_then(|x| x.as_str())
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
            {
                verdict[key] = rule[key].clone();
            }
        }
    }
    state.audit.record(
        user,
        "driver_compat_check",
        &format!("{}:{} → {}", server.name, server.host, target),
        &format!(
            "ai={} verdict={}",
            ai_used,
            verdict.get("verdict").and_then(|x| x.as_str()).unwrap_or("?")
        ),
        ip,
    );
    json(
        session,
        200,
        &json!({"ok": true, "data": {
            "server_id": p.server_id,
            "target_driver": target,
            "current_driver": st.driver_version,
            "current_cuda": st.cuda_version,
            "ai_used": ai_used,
            "unmanaged_gpu_procs": unmanaged,
            "verdict": verdict,
        }}),
    )
    .await
}

/// GET /api/driver-matrix/kb — 返回驱动兼容知识库原文（前端展示来源与版本）。
async fn route_compat_kb(session: &mut Session, _state: &Arc<AppState>) {
    let v: Value = serde_json::from_str(COMPAT_KB).unwrap_or(Value::Null);
    json(session, 200, &json!({"ok": true, "data": v})).await
}

// ---------------------------------------------------------------------------
// Agent 操作助手：自然语言 → 受限动作序列（LLM 生成 + 规则兜底）→ 逐步执行
// ---------------------------------------------------------------------------

const AGENT_MAX_STEPS: usize = 10;
const AGENT_ACTIONS: [&str; 5] =
    ["stop_service", "start_service", "update_service", "upgrade_driver", "wait"];

#[derive(Deserialize, Default)]
pub struct AgentPlanPayload {
    #[serde(default)]
    pub server_id: u64,
    #[serde(default)]
    pub prompt: String,
}

#[derive(Deserialize, Default)]
pub struct AgentExecPayload {
    #[serde(default)]
    pub task_id: String,
}

fn agent_task_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("ag-{nanos:x}")
}

/// 从文本中提取第一个三段式驱动版本号（用于规则兜底）。
fn find_driver_ver_in(text: &str) -> Option<String> {
    let c: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i + 3 < c.len() {
        if c[i].is_ascii_digit() && c[i + 1].is_ascii_digit() && c[i + 2].is_ascii_digit()
            && c[i + 3] == '.'
        {
            let mut j = i + 4;
            let s = j;
            while j < c.len() && c[j].is_ascii_digit() {
                j += 1;
            }
            if (1..=3).contains(&(j - s)) && c.get(j) == Some(&'.') {
                let t = j + 1;
                let mut k = t;
                while k < c.len() && c[k].is_ascii_digit() {
                    k += 1;
                }
                if (1..=3).contains(&(k - t)) {
                    let ver = format!(
                        "{}.{}.{}",
                        c[i..i + 3].iter().collect::<String>(),
                        c[s..j].iter().collect::<String>(),
                        c[t..k].iter().collect::<String>()
                    );
                    if valid_driver_ver(&ver) {
                        return Some(ver);
                    }
                }
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    None
}

/// 规则兜底计划：仅支持「升级到 X.YY.ZZ」型诉求（LLM 不可用时的安全降级）。
fn rule_agent_plan(st: &ServerStatus, prompt: &str, svcs: &[&ServiceConf]) -> Result<(Vec<AgentStep>, String), String> {
    if !prompt.contains("升级") && !prompt.to_ascii_lowercase().contains("upgrade") {
        return Err("LLM 不可用，规则兜底暂只支持「升级驱动到指定版本号」类诉求".into());
    }
    let Some(target) = find_driver_ver_in(prompt) else {
        return Err("LLM 不可用，且未能从诉求中识别出目标驱动版本号（如 590.48.01）".into());
    };
    if !st.driver_version.is_empty() && st.driver_version == target {
        return Err(format!("当前驱动已是 {target}，无需升级")).into();
    }
    let running: Vec<&ServiceConf> = svcs
        .iter()
        .filter(|s| s.status == ServiceStatus::Running)
        .copied()
        .collect();
    let mut steps = Vec::new();
    for sv in &running {
        steps.push(AgentStep {
            idx: 0,
            action: "stop_service".into(),
            params: json!({"service_id": sv.id}),
            description: format!("停止在跑服务 {}", sv.name),
            status: "pending".into(),
            log: String::new(),
        });
    }
    steps.push(AgentStep {
        idx: 0,
        action: "upgrade_driver".into(),
        params: json!({"target_driver": target, "auto_reboot": false}),
        description: format!("升级驱动至 {target}（DKMS 原地安装，先卸旧模块）"),
        status: "pending".into(),
        log: String::new(),
    });
    steps.push(AgentStep {
        idx: 0,
        action: "wait".into(),
        params: json!({"seconds": 120}),
        description: "等待驱动模块加载与机器就绪".into(),
        status: "pending".into(),
        log: String::new(),
    });
    for sv in &running {
        steps.push(AgentStep {
            idx: 0,
            action: "start_service".into(),
            params: json!({"service_id": sv.id}),
            description: format!("重新启动服务 {}", sv.name),
            status: "pending".into(),
            log: String::new(),
        });
    }
    Ok((
        steps,
        format!("规则兜底计划：停止 {} 个在跑服务 → 升级驱动至 {target} → 等待 → 恢复服务", running.len()),
    ))
}

/// 校验并规范化 LLM 生成的步骤序列（动作白名单 / service_id 存在性 / 参数范围）。
fn validate_agent_steps(steps: &mut Vec<AgentStep>, state: &Arc<AppState>, server_id: u64) -> Result<(), String> {
    if steps.is_empty() {
        return Err("计划为空：LLM 判断该诉求无法安全执行（详见说明）".into());
    }
    if steps.len() > AGENT_MAX_STEPS {
        return Err(format!("计划共 {} 步，超过上限 {AGENT_MAX_STEPS} 步", steps.len()));
    }
    let services = state.services.load().services.clone();
    for (i, s) in steps.iter_mut().enumerate() {
        s.idx = (i as u32) + 1;
        let action = s.action.trim().to_ascii_lowercase();
        if !AGENT_ACTIONS.contains(&action.as_str()) {
            return Err(format!("第 {} 步动作「{}」不在动作白名单内", i + 1, s.action));
        }
        s.action = action;
        let require_service = |params: &Value, step_no: usize| -> Result<u64, String> {
            let id = params
                .get("service_id")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("第 {step_no} 步缺少 service_id 或不是整数"))?;
            if !services.iter().any(|sv| sv.id == id && sv.server_id == server_id) {
                return Err(format!(
                    "第 {step_no} 步引用的服务 id {id} 不存在或不属于该机器"
                ));
            }
            Ok(id)
        };
        match s.action.as_str() {
            "stop_service" | "start_service" => {
                require_service(&s.params, i + 1)?;
            }
            "update_service" => {
                require_service(&s.params, i + 1)?;
                let mp = s
                    .params
                    .get("model_path")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim();
                if mp.is_empty() {
                    return Err(format!("第 {} 步缺少 model_path", i + 1));
                }
            }
            "upgrade_driver" => {
                let t = s
                    .params
                    .get("target_driver")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !valid_driver_ver(&t) {
                    return Err(format!("第 {} 步 target_driver 格式不正确：{t}", i + 1));
                }
                let ar = s
                    .params
                    .get("auto_reboot")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false);
                s.params = json!({"target_driver": t, "auto_reboot": ar});
            }
            "wait" => {
                let sec = s.params.get("seconds").and_then(|x| x.as_u64());
                match sec {
                    Some(n) if (1..=600).contains(&n) => {
                        s.params = json!({"seconds": n});
                    }
                    _ => return Err(format!("第 {} 步 wait.seconds 必须是 1..=600 的整数", i + 1)),
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// POST /api/driver-matrix/agent-plan {server_id, prompt}
async fn route_agent_plan(session: &mut Session, state: &Arc<AppState>, user: &str, ip: &str) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: AgentPlanPayload = match serde_json::from_value(v) {
        Ok(x) => x,
        Err(_) => return fail(session, 400, "参数错误：需要 server_id 与 prompt").await,
    };
    let prompt = p.prompt.trim().to_string();
    if prompt.is_empty() {
        return fail(session, 400, "prompt 不能为空").await;
    }
    let Some(server) = state.find_server(p.server_id) else {
        return fail(session, 404, "服务器不存在").await;
    };
    if !server.enabled {
        return fail(session, 409, "该服务器已停用").await;
    }
    let Some(st) = state.status_of(p.server_id) else {
        return fail(session, 409, "该机器状态未知，请先在服务器列表刷新").await;
    };
    if !st.online {
        return fail(session, 409, &format!("机器离线（{}）", st.error)).await;
    }
    let services_all = state.services.load();
    let svcs: Vec<&ServiceConf> = services_all
        .services
        .iter()
        .filter(|s| s.server_id == p.server_id)
        .collect();

    let llm = state.main_conf().llm.clone();
    let plan_prompt = ai::agent_plan_prompt(&server, &st, &svcs, &prompt, COMPAT_KB);
    let mut ai_used = false;
    let mut plan_v: Value = Value::Null;
    if llm.enabled {
        if let Ok(Ok(raw)) =
            tokio::time::timeout(Duration::from_secs(llm.timeout_secs), ai::ask_json(&llm, &plan_prompt))
                .await
        {
            let parsed = ai::unwrap_llm_json(&raw);
            if parsed.get("steps").and_then(|x| x.as_array()).is_some() {
                plan_v = parsed;
                ai_used = true;
            }
        }
    }
    let mut steps: Vec<AgentStep> = plan_v
        .get("steps")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let action = s.get("action")?.as_str()?;
                    Some(AgentStep {
                        idx: 0,
                        action: action.to_string(),
                        params: s.get("params").cloned().unwrap_or(json!({})),
                        description: s
                            .get("description")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        status: "pending".into(),
                        log: String::new(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut summary = plan_v
        .get("summary")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !ai_used {
        match rule_agent_plan(&st, &prompt, &svcs) {
            Ok((rs, rsum)) => {
                steps = rs;
                summary = rsum;
            }
            Err(e) => return fail(session, 502, &e).await,
        }
    } else if summary.is_empty() {
        summary = "LLM 未给出说明".to_string();
    }

    if let Err(e) = validate_agent_steps(&mut steps, state, p.server_id) {
        return fail(session, 422, &e).await;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let id = agent_task_id();
    let task = AgentTask {
        id: id.clone(),
        server_id: p.server_id,
        prompt: prompt.clone(),
        summary: summary.clone(),
        steps: steps.clone(),
        phase: "running".to_string(),
        started_at: now,
        finished_at: 0,
        error: None,
    };
    if let Ok(mut m) = state.agent_tasks.lock() {
        m.insert(id.clone(), task);
    }
    state.audit.record(
        user,
        "agent_plan",
        &format!("{}:{}: {}", server.name, server.host, prompt.chars().take(80).collect::<String>()),
        &format!("steps={} ai_used={}", steps.len(), ai_used),
        ip,
    );
    json(
        session,
        200,
        &json!({"ok": true, "data": {"task_id": id, "ai_used": ai_used, "summary": summary, "steps": steps}}),
    )
    .await
}

/// 执行单个 Agent 步骤。
async fn agent_execute_step(
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    server_id: u64,
    step: &AgentStep,
) -> Result<String, String> {
    match step.action.as_str() {
        "wait" => {
            let secs = step.params.get("seconds").and_then(|x| x.as_u64()).unwrap_or(30).min(600);
            tokio::time::sleep(Duration::from_secs(secs)).await;
            Ok(format!("已等待 {secs} 秒"))
        }
        "stop_service" => {
            let sid = step
                .params
                .get("service_id")
                .and_then(|x| x.as_u64())
                .ok_or("缺少 service_id")?;
            let svc = state.find_service(sid).ok_or_else(|| format!("服务 {sid} 不存在"))?;
            if svc.server_id != server_id {
                return Err(format!("服务 {sid} 不属于该机器"));
            }
            let (out, _exit) = stop_service_core(state, user, ip, &svc).await?;
            Ok(if out.trim().is_empty() {
                "停止命令已执行".to_string()
            } else {
                out
            })
        }
        "start_service" => {
            let sid = step
                .params
                .get("service_id")
                .and_then(|x| x.as_u64())
                .ok_or("缺少 service_id")?;
            let svc = state.find_service(sid).ok_or_else(|| format!("服务 {sid} 不存在"))?;
            if svc.server_id != server_id {
                return Err(format!("服务 {sid} 不属于该机器"));
            }
            if service_alive(&svc, state).await {
                return Ok("服务已在运行，无需启动".to_string());
            }
            let (_pid, gpus, _cmd, out, exit) =
                start_service_core(state, user, ip, &svc, server_id)
                    .await
                    .map_err(|(_code, msg)| msg)?;
            Ok(format!(
                "启动命令已执行（exit={exit}，GPU {gpus:?}）：{}",
                out.chars().take(200).collect::<String>()
            ))
        }
        "update_service" => {
            let sid = step
                .params
                .get("service_id")
                .and_then(|x| x.as_u64())
                .ok_or("缺少 service_id")?;
            let mp = step
                .params
                .get("model_path")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let server = state.find_server(server_id).ok_or("服务器不存在")?;
            let probe = ssh_exec(
                &server.host,
                server.port,
                &server.user,
                &server.password,
                &format!("test -d {mp} && echo vgs_ok"),
                Duration::from_secs(15),
            )
            .await?;
            if !probe.stdout.contains("vgs_ok") {
                return Err(format!("模型路径在目标机不存在：{mp}"));
            }
            let mut services = state.services.load().as_ref().clone();
            if let Some(s) = services.services.iter_mut().find(|x| x.id == sid) {
                s.model_path = mp.clone();
            }
            if let Err(e) = services.save(&state.main_conf().services_path()) {
                return Err(e);
            }
            state.services.store(Arc::new(services));
            state.audit.record(user, "agent_update_service", &format!("服务 {sid} → {mp}"), "Agent 更新模型路径（重启后生效）", ip);
            Ok(format!("模型路径已更新为 {mp}（重启服务后生效）"))
        }
        "upgrade_driver" => {
            let target = step
                .params
                .get("target_driver")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let auto_reboot = step
                .params
                .get("auto_reboot")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let server = state.find_server(server_id).ok_or("服务器不存在")?;
            let st = state.status_of(server_id).ok_or("机器状态未知")?;
            if !st.online {
                return Err(format!("机器离线（{}），不能升级", st.error));
            }
            if !st.driver_version.is_empty() && st.driver_version == target {
                return Ok(format!("当前驱动已是 {target}，无需升级"));
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let task = DrvTask {
                server_id,
                name: server.name.clone(),
                host: server.host.clone(),
                current_driver: st.driver_version.clone(),
                target_driver: target.clone(),
                target_cuda: String::new(),
                phase: "preparing".to_string(),
                started_at: now,
                auto_reboot,
                stopped_services: Vec::new(),
                log: Vec::new(),
                remote_log: String::new(),
                error: None,
                ai_verdict: None,
                restarted: Vec::new(),
                final_driver: String::new(),
            };
            if let Ok(mut m) = state.drv_tasks.lock() {
                m.insert(server_id, task);
            }
            state.audit.record(user, "agent_driver_upgrade", &format!("{}:{} → {target}", server.name, server.host), "Agent 发起驱动升级（stop_services=false）", ip);
            tokio::spawn(drv_pipeline(
                state.clone(),
                user.to_string(),
                ip.to_string(),
                server_id,
                target.clone(),
                false,
                auto_reboot,
            ));
            // 轮询驱动任务直至结束（上限 40 分钟）
            let deadline = tokio::time::Instant::now() + Duration::from_secs(40 * 60);
            loop {
                let t = state.drv_tasks.lock().ok().and_then(|m| m.get(&server_id).cloned());
                match t {
                    Some(t) if t.phase == "done" => {
                        return Ok(format!(
                            "驱动升级完成：{} → {}",
                            t.current_driver,
                            if t.final_driver.is_empty() {
                                t.target_driver
                            } else {
                                t.final_driver
                            }
                        ))
                    }
                    Some(t) if t.phase == "fail" => {
                        return Err(t
                            .error
                            .clone()
                            .unwrap_or_else(|| format!("驱动升级失败（driver={}），详见升级进度页", t.final_driver)))
                    }
                    _ => {}
                }
                if tokio::time::Instant::now() > deadline {
                    return Err("等待驱动升级任务超时（40 分钟），请到升级进度页查看实际状态".into());
                }
                tokio::time::sleep(Duration::from_secs(15)).await;
            }
        }
        other => Err(format!("未知动作 {other}")),
    }
}

fn agent_update_step(state: &Arc<AppState>, task_id: &str, idx: u32, status: &str, log: &str) {
    if let Ok(mut m) = state.agent_tasks.lock() {
        if let Some(t) = m.get_mut(task_id) {
            if let Some(s) = t.steps.iter_mut().find(|s| s.idx == idx) {
                s.status = status.to_string();
                if !log.is_empty() {
                    if s.log.is_empty() {
                        s.log = log.to_string();
                    } else {
                        s.log.push('\n');
                        s.log.push_str(log);
                    }
                }
            }
        }
    }
}

fn agent_finish(state: &Arc<AppState>, task_id: &str, phase: &str, error: Option<String>) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    if let Ok(mut m) = state.agent_tasks.lock() {
        if let Some(t) = m.get_mut(task_id) {
            t.phase = phase.to_string();
            t.finished_at = now;
            t.error = error;
        }
    }
}

/// Agent 任务执行循环（逐步执行，任一步失败即终止）。
async fn agent_pipeline(state: Arc<AppState>, user: String, ip: String, task_id: String) {
    let (server_id, steps) = {
        let m = match state.agent_tasks.lock().ok() {
            Some(m) => m,
            None => return,
        };
        match m.get(&task_id).cloned() {
            Some(t) if t.phase == "running" => (t.server_id, t.steps),
            _ => return,
        }
    };
    let server_name = state.find_server(server_id).map(|s| s.name).unwrap_or_default();
    for step in &steps {
        agent_update_step(&state, &task_id, step.idx, "running", "");
        match agent_execute_step(&state, &user, &ip, server_id, step).await {
            Ok(out) => {
                agent_update_step(&state, &task_id, step.idx, "done", &out);
                state.audit.record(
                    &user,
                    "agent_step_done",
                    &format!("{server_name} 步骤{} {} {}", step.idx, step.action, step.description),
                    out.trim(),
                    &ip,
                );
            }
            Err(e) => {
                agent_update_step(&state, &task_id, step.idx, "fail", &e);
                agent_finish(
                    &state,
                    &task_id,
                    "fail",
                    Some(format!("步骤{}（{} {}）执行失败：{}", step.idx, step.action, step.description, e)),
                );
                return;
            }
        }
    }
    agent_finish(&state, &task_id, "done", None);
    state.audit.record(&user, "agent_task_done", &format!("{server_name} 全部步骤完成"), "", &ip);
}

/// POST /api/driver-matrix/agent-execute {task_id}
async fn route_agent_execute(session: &mut Session, state: &Arc<AppState>, user: &str, ip: &str) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: AgentExecPayload = match serde_json::from_value(v) {
        Ok(x) => x,
        Err(_) => return fail(session, 400, "参数错误：需要 task_id").await,
    };
    let task_id = p.task_id.trim().to_string();
    if task_id.is_empty() {
        return fail(session, 400, "task_id 不能为空").await;
    }
    let task = state
        .agent_tasks
        .lock()
        .ok()
        .and_then(|m| m.get(&task_id).cloned());
    let Some(task) = task else {
        return fail(session, 404, "任务不存在或已随进程重启丢失，请重新生成计划").await;
    };
    if task.phase != "running" {
        return fail(session, 409, &format!("任务已结束（{}），不能重复执行", task.phase)).await;
    }
    if state
        .drv_tasks
        .lock()
        .ok()
        .and_then(|m| m.get(&task.server_id).and_then(|t| t.active().then_some(())))
        .is_some()
    {
        return fail(session, 409, "该机器已有驱动升级任务进行中，请等待其结束").await;
    }
    if state
        .agent_tasks
        .lock()
        .ok()
        .map(|m| {
            m.values()
                .any(|t| t.server_id == task.server_id && t.phase == "running" && t.id != task_id)
        })
        .unwrap_or(false)
    {
        return fail(session, 409, "该机器已有 Agent 任务在执行中").await;
    }
    tokio::spawn(agent_pipeline(
        state.clone(),
        user.to_string(),
        ip.to_string(),
        task_id.clone(),
    ));
    json(
        session,
        202,
        &json!({"ok": true, "data": {"task_id": task_id, "phase": "running"}}),
    )
    .await
}

/// GET /api/driver-matrix/agent-status
async fn route_agent_status(session: &mut Session, state: &Arc<AppState>) {
    let rows: Vec<Value> = state
        .agent_tasks
        .lock()
        .ok()
        .map(|m| {
            m.values()
                .map(|t| {
                    json!({
                        "id": t.id,
                        "server_id": t.server_id,
                        "prompt": t.prompt,
                        "summary": t.summary,
                        "phase": t.phase,
                        "error": t.error,
                        "steps": t.steps,
                        "started_at": t.started_at,
                        "finished_at": t.finished_at,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json(session, 200, &json!({"ok": true, "data": rows})).await
}

/// 生成目标机安装脚本：写状态文件 /tmp/vgs_drvup_{id}.status（pending/installing/ok/fail/reboot），
/// 日志 /tmp/vgs_drvup_{id}.log。安装器由目标机自行执行；驱动未即时生效且允许自动重启时才 reboot。
fn drv_build_install_script(target: &str, fname: &str, status_file: &str, log_file: &str, auto_reboot: bool) -> String {
    let ar = if auto_reboot { "1" } else { "0" };
    format!(
        r#"#!/bin/bash
V="{target}"
F="/tmp/{fname}"
S="{status_file}"
L="{log_file}"
AR="{ar}"
echo installing > "$S"
exec > "$L" 2>&1
echo "[vgs] driver upgrade start $(date '+%F %T')"
if ! dpkg -s dkms >/dev/null 2>&1; then
  echo "[vgs] installing dkms..."
  sudo apt-get update -y
  sudo apt-get install -y dkms
fi
H="linux-headers-$(uname -r)"
if ! dpkg -s "$H" >/dev/null 2>&1; then
  echo "[vgs] installing $H ..."
  sudo apt-get update -y
  sudo apt-get install -y "$H"
fi
R=$(nvidia-smi --query-compute-apps=pid,process_name 2>/dev/null | tail -n +2)
if [ -n "$R" ]; then
  echo fail > "$S"
  echo "[vgs] GPU 仍有进程占用，拒绝安装:"
  echo "$R"
  exit 1
fi
# ---- 先卸载旧 nvidia 模块，使安装器可干净地完成原地安装、驱动立即生效、无需重启。
#      模块引用通常被 nvidia-persistenced 和监控系统（zabbix 周期性调 nvidia-smi）持有，
#      故临时停止这些守护进程（非 GPU 工作负载，安装完成后按 is-enabled 状态恢复）。 ----
PZB=0
if [ "$(sudo systemctl is-active zabbix-agent 2>/dev/null)" = "active" ] || [ "$(sudo systemctl is-active zabbix-agentd 2>/dev/null)" = "active" ]; then
  PZB=1
  echo "[vgs] 临时停止 zabbix-agent（其 nvidia-smi 探测占用旧驱动模块，安装后恢复）..."
  sudo systemctl stop zabbix-agent zabbix-agentd 2>/dev/null || true
fi
PWAS=0
if [ "$(sudo systemctl is-active nvidia-persistenced 2>/dev/null)" = "active" ]; then
  PWAS=1
  echo "[vgs] stopping nvidia-persistenced (持有旧模块引用，安装完成后将恢复)..."
  sudo systemctl stop nvidia-persistenced || true
fi
sudo pkill -9 nvidia-smi 2>/dev/null || true
sleep 2
RMOD_FAIL=""
for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
  if lsmod | grep -qE "^${{m}}[[:space:]]"; then
    echo "[vgs] 旧模块 $m 已加载，尝试 rmmod 卸载..."
    OK=0
    for i in 1 2 3 4 5 6 7 8 9 10; do
      if sudo rmmod "$m" 2>&1; then OK=1; break; fi
      sleep 2
    done
    if [ "$OK" = "1" ]; then echo "[vgs] 模块 $m 已卸载"; else RMOD_FAIL="$RMOD_FAIL $m"; fi
  fi
done
chmod +x "$F"
echo "[vgs] running installer (this may take 5-25 min with dkms compile)..."
sudo sh "$F" --silent --dkms
IE=$?
echo "[vgs] installer_exit=$IE"
if [ "$PWAS" = "1" ] || [ "$(sudo systemctl is-enabled nvidia-persistenced 2>/dev/null)" = "enabled" ]; then
  sudo systemctl start nvidia-persistenced 2>/dev/null || true
  echo "[vgs] nvidia-persistenced restored"
fi
if [ "$PZB" = "1" ] || [ "$(sudo systemctl is-enabled zabbix-agent 2>/dev/null)" = "enabled" ] || [ "$(sudo systemctl is-enabled zabbix-agentd 2>/dev/null)" = "enabled" ]; then
  sudo systemctl start zabbix-agent zabbix-agentd 2>/dev/null || true
  echo "[vgs] zabbix-agent restored"
fi
sleep 5
D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
echo "[vgs] driver_now=$D expected=$V"
if [ "$D" = "$V" ]; then
  echo ok > "$S"
  echo "[vgs] install ok"
  exit 0
fi
if [ "$AR" = "1" ]; then
  echo reboot > "$S"
  echo "[vgs] driver not effective yet, rebooting to load new kernel module..."
  sync
  sudo reboot
  exit 0
fi
echo fail > "$S"
if [ -n "$RMOD_FAIL" ]; then
  echo "[vgs] 驱动未生效：旧模块卸载失败（仍被占用）:$RMOD_FAIL，安装器被中止；请处理占用方后重试。"
else
  echo "[vgs] driver not effective (current: $D, expected: $V). 需重启生效，但任务未授权自动重启。"
fi
exit 1
"#
    )
}

/// 驱动升级主流水线（tokio::spawn 后台执行）：
/// (a) 预检 → (b) 停平台服务 → (c) 网关下载 → (d) rsync 传输 → (e) 写脚本并 nohup 启动
/// 之后交由 drv_finish 看门狗轮询 → 重启等待 → AI 验收 → 恢复服务。
async fn drv_pipeline(
    state: Arc<AppState>,
    user: String,
    ip: String,
    id: u64,
    target: String,
    stop_services: bool,
    auto_reboot: bool,
) {
    let Some(server) = state.find_server(id) else {
        drv_fail(&state, id, "服务器不存在（升级期间被删除？）".to_string());
        return;
    };
    let fname = format!("NVIDIA-Linux-x86_64-{target}.run");
    let mut stopped: Vec<u64> = Vec::new();

    // ---------- (a) 预检 ----------
    drv_phase(&state, id, "preparing");
    {
        let st = collect_and_store(&state, &server).await;
        if !st.online {
            drv_audit_fail(&state, &user, &ip, &server, &target, &format!("机器离线: {}", st.error));
            return drv_fail(&state, id, format!("机器采集失败: {}", st.error));
        }
    }
    {
        let r = ssh_exec(&server.host, server.port, &server.user, &server.password, "sudo -n true", Duration::from_secs(20)).await;
        match r {
            Ok(out) if out.exit_code == 0 => drv_log(&state, id, "预检：免密 sudo 可用".into()),
            Ok(_) => {
                let msg = "该机器无免密 sudo，驱动安装需要 root 权限。请先在机器上配置（如 visudo 添加 NOPASSWD），再重新升级".to_string();
                drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                return drv_fail(&state, id, msg);
            }
            Err(e) => {
                drv_audit_fail(&state, &user, &ip, &server, &target, &format!("SSH 预检失败: {e}"));
                return drv_fail(&state, id, format!("SSH 预检失败: {e}"));
            }
        }
    }
    {
        let r = ssh_exec(&server.host, server.port, &server.user, &server.password, "df -P /tmp | awk 'NR==2 {print $4}'", Duration::from_secs(20)).await;
        match r {
            Ok(out) => {
                let free_gb = out.stdout.trim().parse::<u64>().unwrap_or(0) as f64 / 1024.0 / 1024.0;
                if free_gb < 8.0 {
                    let msg = format!("/tmp 剩余空间不足：仅 {free_gb:.1}GB，至少需要 8GB（驱动包约 3GB）");
                    drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                    return drv_fail(&state, id, msg);
                }
                drv_log(&state, id, format!("预检：/tmp 剩余 {free_gb:.1}GB，满足要求"));
            }
            Err(e) => {
                drv_audit_fail(&state, &user, &ip, &server, &target, &format!("磁盘空间检查失败: {e}"));
                return drv_fail(&state, id, format!("磁盘空间检查失败: {e}"));
            }
        }
    }

    // ---------- (b) 停止平台服务 ----------
    if stop_services {
        drv_phase(&state, id, "stopping");
        let running: Vec<ServiceConf> = state
            .services
            .load()
            .as_ref()
            .services
            .iter()
            .filter(|s| s.server_id == id && s.status == ServiceStatus::Running)
            .cloned()
            .collect();
        if running.is_empty() {
            drv_log(&state, id, "该机器上没有平台服务在运行，无需停止".into());
        }
        for svc in &running {
            drv_log(&state, id, format!("停止服务：{}（{}）", svc.name, svc.model_path));
            let cmd = deploy::build_stop_cmd(svc);
            let r = ssh_exec(&server.host, server.port, &server.user, &server.password, &cmd, Duration::from_secs(30)).await;
            let ok = matches!(&r, Ok(out) if out.exit_code == 0);
            if let Err(e) = &r {
                drv_log(&state, id, format!("停止命令执行异常：{e}"));
            }
            if ok {
                stopped.push(svc.id);
                let mut services = state.services.load().as_ref().clone();
                if let Some(s) = services.services.iter_mut().find(|x| x.id == svc.id) {
                    s.status = ServiceStatus::Stopped;
                    s.pid = None;
                }
                if let Err(e) = services.save(&state.main_conf().services_path()) {
                    drv_log(&state, id, format!("写入 services.yaml 失败: {e}"));
                }
                state.services.store(Arc::new(services));
                drv_log(&state, id, format!("服务已停止：{}", svc.name));
            } else {
                drv_log(&state, id, format!("服务停止未确认（安装前会再次核验 GPU 占用）：{}", svc.name));
            }
        }
        if !stopped.is_empty() {
            drv_set(&state, id, |t| t.stopped_services = stopped.clone());
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
        let r = ssh_exec(&server.host, server.port, &server.user, &server.password, "nvidia-smi --query-compute-apps=pid,process_name 2>/dev/null | tail -n +2", Duration::from_secs(20)).await;
        match r {
            Ok(out) => {
                let procs: Vec<String> = out.stdout.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
                if !procs.is_empty() {
                    // 存在非平台纳管的 GPU 进程：绝不强杀 —— 恢复刚停的服务并中止
                    if !stopped.is_empty() {
                        drv_log(&state, id, "升级中止，先恢复刚才停止的服务...".into());
                        let restarted = drv_restore_services(&state, &user, &ip, &server, &stopped).await;
                        drv_set(&state, id, |t| t.restarted = restarted);
                    }
                    let msg = format!("GPU 上存在非平台纳管进程（平台不会强杀未知进程）：{}。请人工处理后重试", procs.join("；"));
                    drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                    return drv_fail(&state, id, msg);
                }
                drv_log(&state, id, "复检：GPU 已无占用进程".into());
            }
            Err(e) => drv_log(&state, id, format!("GPU 占用复检失败（继续，安装脚本内还有最终核验）: {e}")),
        }
    }

    // ---------- (c) 网关本地下载（节点无法直连 NVIDIA 官方，由网关代下） ----------
    drv_phase(&state, id, "download");
    let url = format!("https://us.download.nvidia.com/tesla/{target}/{fname}");
    let cache_dir = state
        .main_conf()
        .log_dir()
        .parent()
        .map(|p| p.join("cache").join("driver"))
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/ventogpustack/cache/driver"));
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        let msg = format!("无法创建本地驱动缓存目录 {}: {e}", cache_dir.display());
        drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
        return drv_fail(&state, id, msg);
    }
    let local = cache_dir.join(&fname);
    let need_download = match std::fs::metadata(&local) {
        Ok(m) => m.len() < 300_000_000,
        Err(_) => true,
    };
    if need_download {
        drv_log(&state, id, format!("从 NVIDIA 官方下载驱动（网关本地缓存，断点续传、自动重试）：{url}"));
        let part = format!("{}.part", local.display());
        let mut attempt = 0u32;
        'download: loop {
            attempt += 1;
            if attempt > 1 {
                let part_mb = std::fs::metadata(&part)
                    .map(|m| m.len() / 1024 / 1024)
                    .unwrap_or(0);
                drv_log(&state, id, format!("下载中断，第 {attempt}/3 次尝试，从 {part_mb} MB 断点续传..."));
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            let res = tokio::time::timeout(
                Duration::from_secs(1800),
                tokio::process::Command::new("curl")
                    .args([
                        "-fSL",
                        "-C",
                        "-",
                        "--retry",
                        "3",
                        "--retry-delay",
                        "3",
                        "--retry-all-errors",
                        "-o",
                        &part,
                        &url,
                    ])
                    .output(),
            )
            .await;
            match res {
                Ok(Ok(o)) if o.status.success() => break 'download,
                Ok(Ok(o)) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    let tail = {
                        let lines: Vec<&str> = stderr
                            .split(|c| c == '\n' || c == '\r')
                            .map(|l| l.trim())
                            .filter(|l| !l.is_empty())
                            .collect();
                        lines.iter().rev().take(3).cloned().collect::<Vec<_>>().join(" | ")
                    };
                    if attempt < 3 {
                        drv_log(&state, id, format!("第 {attempt} 次下载中断（curl 退出码 {}）：{}", o.status.code().unwrap_or(-1), tail));
                        continue 'download;
                    }
                    let msg = format!("驱动下载失败（已尝试 {attempt} 次，curl 退出码 {}）：{}", o.status.code().unwrap_or(-1), tail);
                    drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                    return drv_fail(&state, id, msg);
                }
                Ok(Err(e)) => {
                    if attempt < 3 {
                        drv_log(&state, id, format!("第 {attempt} 次尝试无法执行 curl：{e}"));
                        continue 'download;
                    }
                    let msg = format!("无法执行 curl：{e}");
                    drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                    return drv_fail(&state, id, msg);
                }
                Err(_) => {
                    if attempt < 3 {
                        drv_log(&state, id, format!("第 {attempt} 次下载超过 30 分钟，重试"));
                        continue 'download;
                    }
                    let msg = format!("驱动下载超时（已尝试 {attempt} 次，每次上限 30 分钟）");
                    drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                    return drv_fail(&state, id, msg);
                }
            }
        }
        if let Err(e) = std::fs::rename(&part, &local) {
            let msg = format!("驱动包移动失败: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    } else {
        drv_log(&state, id, format!("本地缓存已有驱动包，跳过下载：{}", local.display()));
    }
    let local_len = match std::fs::metadata(&local).map(|m| m.len()) {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("驱动包缺失: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    };
    if local_len < 300_000_000 {
        let msg = format!("驱动包大小异常（{local_len} 字节），可能下载不完整，请删除本地缓存后重试");
        drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
        return drv_fail(&state, id, msg);
    }
    drv_log(&state, id, format!("本地驱动包就绪：{}（{} MB）", local.display(), local_len / 1024 / 1024));

    // ---------- (d) rsync 传输到目标机 ----------
    drv_phase(&state, id, "transferring");
    drv_log(&state, id, format!("rsync 传输到 {}:/tmp/{}（约 {} MB，内网通常数分钟）", server.host, fname, local_len / 1024 / 1024));
    let ssh_opt = format!("sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {}", server.port);
    let remote = format!("{}@{}:/tmp/{}", server.user, server.host, fname);
    let res = tokio::time::timeout(
        Duration::from_secs(1800),
        tokio::process::Command::new("rsync")
            .args(["-a", "--partial"])
            .arg("-e")
            .arg(&ssh_opt)
            .arg(&local)
            .arg(&remote)
            .env("SSHPASS", &server.password)
            .output(),
    )
    .await;
    match res {
        Ok(Ok(o)) if o.status.success() => {}
        Ok(Ok(o)) => {
            let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
            let msg = format!("传输到目标机失败（rsync 退出码 {}）：{}", o.status.code().unwrap_or(-1), stderr);
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
        Ok(Err(e)) => {
            let msg = format!("无法执行 rsync（确认网关已安装 rsync/sshpass）: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
        Err(_) => {
            let msg = "传输到目标机超时（>30 分钟）".to_string();
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    }
    let r = ssh_exec(&server.host, server.port, &server.user, &server.password, &format!("stat -c %s /tmp/{fname}"), Duration::from_secs(20)).await;
    match r {
        Ok(out) => {
            let remote_len: u64 = out.stdout.trim().parse().unwrap_or(0);
            if remote_len != local_len {
                let msg = format!("目标机文件大小不一致（本地 {local_len} / 远端 {remote_len}），请删除 /tmp/{fname} 后重试");
                drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
                return drv_fail(&state, id, msg);
            }
            drv_log(&state, id, "传输完成，大小校验通过".into());
        }
        Err(e) => {
            let msg = format!("远端文件校验失败: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    }

    // ---------- (e) 写安装脚本并 nohup 启动（断线不中断） ----------
    drv_phase(&state, id, "installing");
    let sh = format!("/tmp/vgs_drvup_{id}.sh");
    let status_file = format!("/tmp/vgs_drvup_{id}.status");
    let log_file = format!("/tmp/vgs_drvup_{id}.log");
    let script = drv_build_install_script(&target, &fname, &status_file, &log_file, auto_reboot);
    let write_cmd = format!("rm -f {sh} {status_file} {log_file}\ncat > {sh} <<'VGS_DRV_EOF'\n{script}VGS_DRV_EOF\nchmod +x {sh} && echo vgs_script_ok");
    match ssh_exec(&server.host, server.port, &server.user, &server.password, &write_cmd, Duration::from_secs(60)).await {
        Ok(out) if out.stdout.contains("vgs_script_ok") => {}
        Ok(out) => {
            let msg = format!("安装脚本写入失败：{}", out.stdout.trim());
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
        Err(e) => {
            let msg = format!("安装脚本写入失败: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    }
    let launch_cmd = format!("nohup bash {sh} >/dev/null 2>&1 & echo $! > /tmp/vgs_drvup_{id}.pid; sleep 1; cat /tmp/vgs_drvup_{id}.pid");
    match ssh_exec(&server.host, server.port, &server.user, &server.password, &launch_cmd, Duration::from_secs(30)).await {
        Ok(out) => drv_log(&state, id, format!("安装脚本已在目标机启动（远端 pid {}），进入看门狗轮询", out.stdout.trim())),
        Err(e) => {
            let msg = format!("安装脚本启动失败: {e}");
            drv_audit_fail(&state, &user, &ip, &server, &target, &msg);
            return drv_fail(&state, id, msg);
        }
    }

    // ---------- (f)~(i) 看门狗 / 重启等待 / AI 验收 / 恢复服务 ----------
    drv_finish(state, user, ip, id, target, stopped, auto_reboot).await;
}

/// 升级失败统一收尾：先恢复升级前停止的服务，再记审计、置 fail 状态。
async fn drv_fail_and_restore(
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    server: &ServerConf,
    target: &str,
    id: u64,
    stopped: &[u64],
    msg: String,
) {
    if !stopped.is_empty() {
        drv_log(state, id, "升级失败，先恢复升级前停止的服务...".into());
        let restarted = drv_restore_services(state, user, ip, server, stopped).await;
        drv_set(state, id, |t| t.restarted = restarted);
    }
    drv_audit_fail(state, user, ip, server, target, &msg);
    drv_fail(state, id, msg);
}

/// 看门狗阶段：(f) 轮询远端状态文件 → (g) 重启等待 → (h) AI 验收 → (i) 恢复服务并收尾。
async fn drv_finish(
    state: Arc<AppState>,
    user: String,
    ip: String,
    id: u64,
    target: String,
    stopped: Vec<u64>,
    auto_reboot: bool,
) {
    let Some(server) = state.find_server(id) else {
        drv_fail(&state, id, "服务器不存在（升级期间被删除？）".to_string());
        return;
    };

    // ---------- (f) 看门狗：轮询远端状态文件（每 8s，最长 35 分钟） ----------
    let status_file = format!("/tmp/vgs_drvup_{id}.status");
    let log_file = format!("/tmp/vgs_drvup_{id}.log");
    let poll_cmd = format!("cat {status_file} 2>/dev/null; echo ---SEP---; tail -n 40 {log_file} 2>/dev/null");
    enum WatchOutcome {
        Done,
        Reboot,
        Fail(String),
        Timeout,
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(35 * 60);
    let mut lost = 0u32;
    let outcome = loop {
        if std::time::Instant::now() > deadline {
            break WatchOutcome::Timeout;
        }
        tokio::time::sleep(Duration::from_secs(8)).await;
        match ssh_exec(&server.host, server.port, &server.user, &server.password, &poll_cmd, Duration::from_secs(20)).await {
            Ok(out) => {
                lost = 0;
                let s = out.stdout.as_str();
                let (stage, tail) = match s.find("---SEP---") {
                    Some(i) => (s[..i].trim(), s[i + 9..].trim()),
                    None => (s.trim(), ""),
                };
                if !tail.is_empty() {
                    drv_set(&state, id, |t| t.remote_log = tail.to_string());
                }
                match stage {
                    "ok" => break WatchOutcome::Done,
                    "fail" => {
                        let tail = if tail.is_empty() {
                            format!("详见目标机安装日志 /tmp/vgs_drvup_{id}.log")
                        } else {
                            tail.to_string()
                        };
                        break WatchOutcome::Fail(format!("安装脚本报告失败。日志摘录：\n{tail}"));
                    }
                    "reboot" => break WatchOutcome::Reboot,
                    _ => {}
                }
            }
            Err(_) => {
                lost += 1;
                if lost % 5 == 0 {
                    drv_log(&state, id, format!("与目标机失联（连续 {lost} 次），等待恢复..."));
                }
                if lost >= 10 {
                    if auto_reboot {
                        break WatchOutcome::Reboot;
                    }
                    break WatchOutcome::Fail(
                        "与目标机失联超过 80 秒，且任务未授权自动重启，请人工检查机器状态".to_string(),
                    );
                }
            }
        }
    };

    match outcome {
        WatchOutcome::Done => {
            drv_log(&state, id, "安装脚本执行成功，驱动版本即时校验通过".into());
        }
        WatchOutcome::Reboot => {
            // ---------- (g) 重启等待 ----------
            drv_phase(&state, id, "reboot-wait");
            drv_log(&state, id, "机器重启中，等待其上线并加载新驱动（最长 20 分钟）...".into());
            let deadline = std::time::Instant::now() + Duration::from_secs(20 * 60);
            let mut bad = 0u32;
            let mut last_driver = String::new();
            let mut recovered = false;
            while std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_secs(30)).await;
                match ssh_exec(
                    &server.host,
                    server.port,
                    &server.user,
                    &server.password,
                    "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '",
                    Duration::from_secs(15),
                )
                .await
                {
                    Ok(out) => {
                        let d = out.stdout.trim().to_string();
                        if !d.is_empty() && d == target {
                            recovered = true;
                            drv_log(&state, id, format!("机器已上线，驱动版本 {d} 生效"));
                            break;
                        }
                        if !d.is_empty() {
                            last_driver = d;
                        }
                        bad += 1;
                        if bad >= 6 {
                            let msg = if last_driver.is_empty() {
                                "机器已在线但 nvidia-smi 不可用，驱动未生效，请人工处理".to_string()
                            } else {
                                format!("机器已在线但驱动版本非目标（当前 {last_driver}，期望 {target}），请人工处理")
                            };
                            return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
                        }
                    }
                    Err(_) => {} // 机器仍在重启，继续等
                }
            }
            if !recovered {
                let msg = "等待机器上线超过 20 分钟仍未确认驱动生效，请人工检查".to_string();
                return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
            }
        }
        WatchOutcome::Fail(msg) => {
            return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
        }
        WatchOutcome::Timeout => {
            let msg = "安装等待超时（>35 分钟仍未完成），请人工检查目标机状态".to_string();
            return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
        }
    }

    // ---------- (h) AI 验收（LLM 判断 nvidia-smi 全量输出；失败回退规则） ----------
    drv_phase(&state, id, "accepting");
    drv_log(&state, id, "AI 验收：采集 nvidia-smi 全量输出...".into());
    let smi = match ssh_exec(&server.host, server.port, &server.user, &server.password, "nvidia-smi", Duration::from_secs(30)).await {
        Ok(out) => out.stdout,
        Err(e) => {
            let msg = format!("执行 nvidia-smi 失败: {e}");
            return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
        }
    };
    let llm = state.main_conf().llm.clone();
    let mut verdict: Option<Value> = None;
    let (accept_ok, accept_note): (bool, String) = if llm.enabled {
        match ai::ask_json(&llm, &ai::driver_acceptance_prompt(&target, &smi)).await {
            Ok(raw) => {
                let v = ai::unwrap_llm_json(&raw);
                let v_ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
                let v_driver = v.get("driver").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let v_note = v.get("note").and_then(|x| x.as_str()).unwrap_or("").to_string();
                verdict = Some(json!({
                    "ok": v_ok,
                    "driver": v_driver,
                    "note": v_note,
                    "gpus_total": v.get("gpus_total").cloned().unwrap_or(Value::Null),
                    "gpus_ok": v.get("gpus_ok").cloned().unwrap_or(Value::Null),
                }));
                drv_log(&state, id, format!("AI 验收：ok={v_ok} driver={v_driver} note={v_note}"));
                (v_ok, v_note)
            }
            Err(e) => {
                drv_log(&state, id, format!("AI 验收失败（{e}），回退规则判定"));
                let r = drv_accept_rule(&state, &server, &target).await;
                (
                    r,
                    if r {
                        "规则校验通过".to_string()
                    } else {
                        "规则校验失败：驱动版本不一致或无 GPU".to_string()
                    },
                )
            }
        }
    } else {
        let r = drv_accept_rule(&state, &server, &target).await;
        (
            r,
            if r {
                "规则校验通过".to_string()
            } else {
                "规则校验失败：驱动版本不一致或无 GPU".to_string()
            },
        )
    };
    if let Some(v) = &verdict {
        drv_set(&state, id, |t| t.ai_verdict = Some(v.clone()));
    }
    if !accept_ok {
        let msg = if accept_note.is_empty() {
            "验收失败".to_string()
        } else {
            format!("验收失败：{accept_note}")
        };
        return drv_fail_and_restore(&state, &user, &ip, &server, &target, id, &stopped, msg).await;
    }
    drv_log(
        &state,
        id,
        format!("验收通过：{}", if accept_note.is_empty() { format!("驱动版本 {target} 生效") } else { accept_note }),
    );

    // ---------- (i) 恢复服务并收尾 ----------
    drv_phase(&state, id, "restarting");
    if stopped.is_empty() {
        drv_log(&state, id, "无需恢复的服务".into());
    } else {
        drv_log(&state, id, format!("恢复升级前停止的 {} 个服务...", stopped.len()));
        let restarted = drv_restore_services(&state, &user, &ip, &server, &stopped).await;
        for r in &restarted {
            let name = r.get("name").and_then(|x| x.as_str()).unwrap_or("?");
            let r_ok = r.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
            let err = r.get("error").and_then(|x| x.as_str()).unwrap_or("");
            if r_ok {
                drv_log(&state, id, format!("服务已恢复：{name}"));
            } else {
                drv_log(&state, id, format!("服务恢复失败：{name}（{err}），请在模型服务列表手动启动"));
            }
        }
        drv_set(&state, id, |t| t.restarted = restarted);
    }
    let st = collect_and_store(&state, &server).await;
    let final_driver = st.driver_version.clone();
    drv_set(&state, id, |t| t.final_driver = final_driver.clone());
    drv_phase(&state, id, "done");
    drv_log(&state, id, format!("驱动升级完成：当前驱动 {final_driver}"));
    state.audit.record(
        &user,
        "driver_upgrade_done",
        &format!("{}:{} → {}", server.name, server.host, target),
        &format!("final_driver={final_driver}"),
        &ip,
    );
}

/// GET /api/servers/{id}/driver-upgrade — 该机器最近一次（进行中或已完成）升级任务
async fn route_drv_upgrade_status(session: &mut Session, state: &Arc<AppState>, id: u64) {
    match drv_snapshot(state, id) {
        Some(t) => ok(session, &json!({"ok": true, "data": t})).await,
        None => fail(session, 404, "该机器暂无驱动升级任务（未开始，或进程重启后内存态已清除）").await,
    }
}

/// GET /api/driver-upgrades — 全部升级任务（供全局「N 台升级中」角标轮询）
async fn route_drv_upgrades_list(session: &mut Session, state: &Arc<AppState>) {
    let tasks: Vec<DrvTask> = state
        .drv_tasks
        .lock()
        .ok()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_default();
    let active = tasks.iter().filter(|t| t.active()).count();
    json(session, 200, &json!({"ok": true, "data": {"active": active, "tasks": tasks}})).await
}

// ---------------------------------------------------------------------------
// 模型服务
// ---------------------------------------------------------------------------

fn parse_deploy_type(s: &str) -> Option<DeployType> {
    match s {
        "vllm" => Some(DeployType::Vllm),
        "sglang" => Some(DeployType::Sglang),
        "docker" => Some(DeployType::Docker),
        "python" => Some(DeployType::Python),
        _ => None,
    }
}

/// 服务进程真实存活判定：
/// 1) docker 类型按容器名判定（平台自管 vgs_{id} / 纳管外部容器 container_name）；
/// 2) GPU 进程 cmdline 含 model_path（vllm/sglang 等 GPU 服务，本地零成本）；
/// 3) stop_pattern 命中缓存进程 cmdline（router 等无模型路径的纳管服务）；
/// 4) 否则查 /proc/{pid} 存在性（不限进程属主 —— 集群服务多为 root/其他用户运行，kill -0 会 EPERM）。
/// SSH 校验失败（机器不可达）一律按未存活处理，避免误报 running。
async fn service_alive(svc: &ServiceConf, state: &Arc<AppState>) -> bool {
    let servers = state.servers.load();
    let Some(server) = servers.servers.iter().find(|s| s.id == svc.server_id) else {
        return false;
    };
    let timeout = Duration::from_secs(15);
    // 容器判活：纳管外部容器按 container_name，平台自管容器按 vgs_{id}
    let docker_name = if svc.deploy_type == DeployType::Docker {
        if !svc.container_name.trim().is_empty() {
            Some(svc.container_name.trim().to_string())
        } else if svc.stop_pattern.trim().is_empty() {
            Some(format!("vgs_{}", svc.id))
        } else {
            None
        }
    } else {
        None
    };
    if let Some(name) = docker_name {
        let cmd = format!("docker ps -q --filter name=^{name}$");
        return match ssh_exec(&server.host, server.port, &server.user, &server.password, &cmd, timeout).await {
            Ok(r) => r.exit_code == 0 && !r.stdout.trim().is_empty(),
            Err(_) => false,
        };
    }
    let statuses = state.statuses.load();
    if !svc.model_path.is_empty()
        && statuses
            .get(&svc.server_id)
            .map(|x| x.procs.iter().any(|p| p.cmdline.contains(&svc.model_path)))
            .unwrap_or(false)
    {
        return true;
    }
    // 无模型路径的纳管服务（如 sglang-router）：stop_pattern 命中缓存进程 cmdline 即存活
    if svc.model_path.is_empty() && !svc.stop_pattern.trim().is_empty() {
        let sp = svc.stop_pattern.trim().to_string();
        if statuses
            .get(&svc.server_id)
            .map(|x| x.procs.iter().any(|p| p.name.contains(&sp) || p.cmdline.contains(&sp)))
            .unwrap_or(false)
        {
            return true;
        }
    }
    // 无模型路径的纳管服务（智能检测注册，如 EngineCore / 容器内服务）：
    // 直接用采集缓存里的 GPU 进程 pid 判活；缓存缺失或进程不在 GPU 列表时
    // 才回退 SSH 探测，避免列表接口逐服务串行 SSH。
    if svc.model_path.is_empty()
        && svc
            .pid
            .map(|pid| {
                statuses
                    .get(&svc.server_id)
                    .map(|x| x.procs.iter().any(|p| p.pid == pid))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    {
        return true;
    }
    let Some(pid) = svc.pid else {
        return false;
    };
    let cmd = format!("test -d /proc/{pid} && echo alive");
    match ssh_exec(&server.host, server.port, &server.user, &server.password, &cmd, timeout).await {
        Ok(r) => r.stdout.contains("alive"),
        Err(_) => false,
    }
}

async fn service_view(svc: &ServiceConf, state: &Arc<AppState>) -> Value {
    let servers = state.servers.load();
    let host = servers
        .servers
        .iter()
        .find(|s| s.id == svc.server_id)
        .map(|s| format!("{}:{}", s.name, s.host));
    let actually_running = svc.status == ServiceStatus::Running && service_alive(svc, state).await;
    json!({
        "id": svc.id,
        "name": svc.name,
        "model_path": svc.model_path,
        "deploy_type": svc.deploy_type.as_str(),
        "port": svc.port,
        "tp": svc.tp,
        "extra_args": svc.extra_args,
        "docker_image": svc.docker_image,
        "docker_args": svc.docker_args,
        "container_name": svc.container_name,
        "python_cmd": svc.python_cmd,
        "stop_pattern": svc.stop_pattern,
        "server_id": svc.server_id,
        "server_host": host.unwrap_or_default(),
        "status": match svc.status {
            ServiceStatus::Stopped => "stopped",
            ServiceStatus::Running if actually_running => "running",
            ServiceStatus::Running => "stopped",
            ServiceStatus::Migrating => "migrating",
            ServiceStatus::Error => "error",
        },
        "pid": svc.pid,
        "notes": svc.notes,
    })
}

async fn routes_services(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    method: &str,
    id_part: Option<&str>,
    third: Option<&str>,
) {
    let id: u64 = match id_part.and_then(|p| p.parse().ok()) {
        Some(i) => i,
        None => {
            if method == "GET" {
                return route_services_list(session, state).await;
            }
            if method == "POST" {
                return route_service_create(session, state, user, ip).await;
            }
            return fail(session, 405, "method not allowed").await;
        }
    };

    let Some(svc) = state.find_service(id) else {
        return fail(session, 404, "服务不存在").await;
    };

    match (method, third) {
        ("GET", None) => {
            let view = service_view(&svc, state).await;
            ok(session, &view).await
        }
        ("PUT", None) => route_service_update(session, state, user, ip, id).await,
        ("DELETE", None) => {
            let mut services = state.services.load().as_ref().clone();
            let before = services.services.len();
            services.services.retain(|x| x.id != id);
            if services.services.len() == before {
                return fail(session, 404, "服务不存在").await;
            }
            if let Err(e) = services.save(&state.main_conf().services_path()) {
                return fail(session, 500, &e).await;
            }
            state.services.store(Arc::new(services));
            state
                .audit
                .record(user, "delete_service", &svc.name, "删除模型服务", ip);
            ok(session, &json!({"id": id})).await
        }
        ("POST", Some("start")) => route_start(session, state, user, ip, &svc).await,
        ("POST", Some("stop")) => route_stop(session, state, user, ip, &svc).await,
        ("POST", Some("plan")) => route_plan(session, state, user, ip, &svc).await,
        ("POST", Some("migrate")) => route_migrate(session, state, user, ip, &svc).await,
        ("POST", Some("copy")) => route_copy(session, state, user, ip, &svc).await,
        ("GET", Some("copy-status")) => route_copy_status(session, state, &svc).await,
        _ => fail(session, 405, "method not allowed").await,
    }
}

async fn route_services_list(session: &mut Session, state: &Arc<AppState>) {
    // 并行计算每个服务的存活状态（并发受 ssh_concurrency 限制），
    // 避免 30+ 个需要 SSH 存活探测的服务串行叠加导致整个列表请求超时。
    let services = state.services.load().as_ref().clone();
    let conf = state.main_conf();
    let sem = Arc::new(tokio::sync::Semaphore::new(conf.ssh_concurrency.max(1) as usize));
    let mut handles = Vec::new();
    for s in services.services {
        let sem = sem.clone();
        let state = state.clone();
        handles.push(tokio::spawn(async move {
            let _perm = sem.acquire().await;
            service_view(&s, &state).await
        }));
    }
    let mut rows: Vec<Value> = Vec::new();
    for h in handles {
        if let Ok(v) = h.await {
            rows.push(v);
        }
    }
    json(session, 200, &json!({"ok": true, "data": rows})).await
}

async fn route_service_create(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: ServicePayload = serde_json::from_value(v).unwrap_or_default();
    let name = p.name.filter(|x| !x.trim().is_empty()).unwrap_or_default();
    let model_path = p.model_path.filter(|x| !x.trim().is_empty()).unwrap_or_default();
    if name.is_empty() || model_path.is_empty() {
        return fail(session, 400, "name 与 model_path 必填").await;
    }
    let deploy_type = match parse_deploy_type(p.deploy_type.as_deref().unwrap_or("vllm")) {
        Some(d) => d,
        None => return fail(session, 400, "deploy_type 必须是 vllm/sglang/docker/python").await,
    };
    let mut services = state.services.load().as_ref().clone();
    let new_id = services.services.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    let s = ServiceConf {
        id: new_id,
        name: name.clone(),
        model_path,
        server_id: p.server_id.unwrap_or(0),
        deploy_type,
        port: p.port.unwrap_or(8000),
        tp: p.tp.unwrap_or(1).max(1),
        extra_args: p.extra_args.unwrap_or_default(),
        docker_image: p.docker_image.unwrap_or_default(),
        docker_args: p.docker_args.unwrap_or_default(),
        container_name: String::new(),
        python_cmd: p.python_cmd.unwrap_or_default(),
        stop_pattern: p.stop_pattern.unwrap_or_default(),
        status: ServiceStatus::Stopped,
        pid: None,
        notes: p.notes.unwrap_or_default(),
    };
    services.services.push(s);
    if let Err(e) = services.save(&state.main_conf().services_path()) {
        return fail(session, 500, &e).await;
    }
    state.services.store(Arc::new(services));
    state.audit.record(user, "add_service", &name, "注册模型服务", ip);
    ok(session, &json!({"id": new_id})).await
}

async fn route_service_update(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    id: u64,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: ServicePayload = serde_json::from_value(v).unwrap_or_default();
    let mut services = state.services.load().as_ref().clone();
    let Some(s) = services.services.iter_mut().find(|x| x.id == id) else {
        return fail(session, 404, "服务不存在").await;
    };
    if let Some(x) = p.name {
        if !x.trim().is_empty() {
            s.name = x;
        }
    }
    if let Some(x) = p.model_path {
        if !x.trim().is_empty() {
            s.model_path = x;
        }
    }
    if let Some(x) = p.deploy_type {
        match parse_deploy_type(&x) {
            Some(d) => s.deploy_type = d,
            None => {
                return fail(session, 400, "deploy_type 必须是 vllm/sglang/docker/python").await
            }
        }
    }
    if let Some(x) = p.port {
        s.port = x;
    }
    if let Some(x) = p.tp {
        s.tp = x.max(1);
    }
    if let Some(x) = p.extra_args {
        s.extra_args = x;
    }
    if let Some(x) = p.docker_image {
        s.docker_image = x;
    }
    if let Some(x) = p.docker_args {
        s.docker_args = x;
    }
    if let Some(x) = p.python_cmd {
        s.python_cmd = x;
    }
    if let Some(x) = p.server_id {
        s.server_id = x;
    }
    if let Some(x) = p.stop_pattern {
        s.stop_pattern = x;
    }
    if let Some(x) = p.container_name {
        // 允许清空（传空串表示不再绑定容器）
        s.container_name = x;
    }
    if let Some(x) = p.notes {
        s.notes = x;
    }
    let label = s.name.clone();
    if let Err(e) = services.save(&state.main_conf().services_path()) {
        return fail(session, 500, &e).await;
    }
    state.services.store(Arc::new(services));
    state
        .audit
        .record(user, "update_service", &label, "更新模型服务", ip);
    let updated = state.find_service(id).unwrap();
    let view = service_view(&updated, state).await;
    ok(session, &view).await
}

async fn route_start(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: StartPayload = serde_json::from_value(v).unwrap_or_default();
    let target_id = p.server_id.unwrap_or(svc.server_id);
    match start_service_core(state, user, ip, svc, target_id).await {
        Ok((pid, gpus, cmd, output, exit_code)) => {
            ok(
                session,
                &json!({"pid": pid, "gpus": gpus, "cmd": cmd, "output": output, "exit_code": exit_code}),
            )
            .await
        }
        Err((code, msg)) => fail(session, code, &msg).await,
    }
}

/// 服务启动核心逻辑（供路由与驱动升级后的服务恢复复用）。
/// 成功返回 (pid, gpus, cmd, output, exit_code)；失败返回 (HTTP 状态码, 错误信息)。
async fn start_service_core(
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
    target_id: u64,
) -> Result<(Option<u32>, Vec<u32>, String, String, i32), (u16, String)> {
    let Some(server) = state.find_server(target_id) else {
        return Err((404, "目标服务器不存在".to_string()));
    };
    let st = collect_and_store(state, &server).await;
    if !st.online {
        return Err((502, format!("目标机器采集失败: {}", st.error)));
    }
    // 纳管的外部容器自带 GPU 绑定，不需要平台选卡
    let has_container = !svc.container_name.trim().is_empty();
    if !has_container {
        if st.gpus.is_empty() {
            return Err((502, "目标机器没有可用的 GPU".to_string()));
        }
    }
    let gpus = if has_container {
        Vec::new()
    } else {
        deploy::pick_gpus(&st, svc.tp.max(1), &[]).ok_or_else(|| (409, "空闲 GPU 卡数量不足".to_string()))?
    };
    let cmd = deploy::build_start_cmd(svc, &gpus);
    let timeout = Duration::from_secs(60);
    match ssh_exec(
        &server.host,
        server.port,
        &server.user,
        &server.password,
        &cmd,
        timeout,
    )
    .await
    {
        Ok(out) => {
            let out_trim = out.stdout.trim().to_string();
            let pid: Option<u32> = if svc.deploy_type == DeployType::Docker || has_container {
                None
            } else {
                out_trim.lines().rev().find_map(|l| l.trim().parse::<u32>().ok())
            };
            let mut services = state.services.load().as_ref().clone();
            if let Some(s) = services.services.iter_mut().find(|x| x.id == svc.id) {
                s.server_id = target_id;
                s.status = ServiceStatus::Running;
                s.pid = pid;
            }
            if let Err(e) = services.save(&state.main_conf().services_path()) {
                return Err((500, e));
            }
            state.services.store(Arc::new(services));
            state.audit.record(
                user,
                "start_service",
                &format!("{}@{}", svc.name, server.host),
                &format!("gpus={gpus:?} exit={}", out.exit_code),
                ip,
            );
            Ok((pid, gpus, cmd, out_trim, out.exit_code))
        }
        Err(e) => {
            state
                .audit
                .record(user, "start_service_failed", &format!("{}@{}", svc.name, server.host), &e, ip);
            Err((502, e))
        }
    }
}

/// 停止服务的核心逻辑（HTTP 端点与 Agent 步骤执行共用）：
/// SSH 执行停止命令 → 服务标记 Stopped 并清空 pid → 持久化 + 审计。
/// 返回 (命令输出, 退出码)。
async fn stop_service_core(
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) -> Result<(String, i32), String> {
    let server = state
        .find_server(svc.server_id)
        .ok_or_else(|| "服务所在服务器不存在".to_string())?;
    let cmd = deploy::build_stop_cmd(svc);
    let timeout = Duration::from_secs(30);
    let out = ssh_exec(
        &server.host,
        server.port,
        &server.user,
        &server.password,
        &cmd,
        timeout,
    )
    .await
    .map_err(|e| {
        state.audit.record(
            user,
            "stop_service_failed",
            &format!("{}@{}", svc.name, server.host),
            &e,
            ip,
        );
        e
    })?;
    let mut services = state.services.load().as_ref().clone();
    if let Some(s) = services.services.iter_mut().find(|x| x.id == svc.id) {
        s.status = ServiceStatus::Stopped;
        s.pid = None;
    }
    services.save(&state.main_conf().services_path())?;
    state.services.store(Arc::new(services));
    state.audit.record(
        user,
        "stop_service",
        &format!("{}@{}", svc.name, server.host),
        out.stdout.trim(),
        ip,
    );
    Ok((out.stdout.trim().to_string(), out.exit_code))
}

async fn route_stop(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    match stop_service_core(state, user, ip, svc).await {
        Ok((out, exit_code)) => ok(session, &json!({"output": out, "exit_code": exit_code})).await,
        Err(e) => fail(session, 502, &e).await,
    }
}

/// 生成迁移方案（AI 优先，规则兜底）。
async fn route_plan(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: PlanPayload = serde_json::from_value(v).unwrap_or_default();

    let Some(source) = state.find_server(svc.server_id) else {
        return fail(session, 404, "源服务器不存在").await;
    };
    let servers = state.servers.load().servers.clone();
    let statuses = state.statuses.load();
    let candidates: Vec<(ServerConf, ServerStatus)> = servers
        .iter()
        .filter(|s| s.enabled && s.id != svc.server_id)
        .filter(|s| p.to_server_id.map(|x| x == s.id).unwrap_or(true))
        .filter_map(|s| statuses.get(&s.id).map(|st| (s.clone(), st.clone())))
        .collect();
    if candidates.is_empty() {
        return fail(session, 409, "没有可用的候选目标机器（请先刷新状态）").await;
    }

    let llm = state.main_conf().llm.clone();
    let hint = p
        .to_server_id
        .and_then(|id| servers.iter().find(|s| s.id == id).map(|s| s.host.as_str()));
    let mut ai_used = false;
    let mut note = String::new();
    let raw: Value = if llm.enabled {
        match tokio::time::timeout(
            Duration::from_secs(llm.timeout_secs),
            ai::ask_json(&llm, &ai::migration_prompt(svc, &candidates, hint)),
        )
        .await
        {
            Ok(Ok(r)) => {
                ai_used = true;
                r
            }
            Ok(Err(e)) => {
                note = format!("AI 规划失败（{e}），已回退规则引擎");
                rule_plan_value(svc, &source.host, &candidates)
            }
            Err(_) => {
                note = "AI 规划超时，已回退规则引擎".into();
                rule_plan_value(svc, &source.host, &candidates)
            }
        }
    } else {
        note = "LLM 未启用，使用规则引擎".into();
        rule_plan_value(svc, &source.host, &candidates)
    };
    let raw = ai::unwrap_llm_json(&raw);

    let plan = normalize_plan(svc, &source, &raw, &candidates);
    state.audit.record(
        user,
        "plan_migration",
        &format!("{} -> {}", svc.name, plan["target_host"].as_str().unwrap_or("?")),
        &format!("ai={ai_used} {note}"),
        ip,
    );
    ok(session, &json!({"ai": ai_used, "note": note, "plan": plan})).await
}

/// 规则引擎版迁移方案。
fn rule_plan_value(
    svc: &ServiceConf,
    source_host: &str,
    candidates: &[(ServerConf, ServerStatus)],
) -> Value {
    let Some(t) = scheduler::rule_select_target(svc, candidates, None) else {
        return json!({
            "target_host": null,
            "reason": "没有满足显存/卡数要求的候选机器",
            "commands": []
        });
    };
    let start_cmd = deploy::build_start_cmd(svc, &t.gpus);
    let stop_cmd = deploy::build_stop_cmd(svc);
    let mut commands: Vec<Value> = Vec::new();
    if !t.model_local {
        commands.push(json!({
            "step": "先拷贝模型到目标机",
            "host": t.server.host,
            "cmd": format!(
                "rsync -a -e ssh {u}@{src}:{model} {u}@{dst}:{model}/",
                u = t.server.user,
                src = source_host,
                dst = t.server.host,
                model = svc.model_path
            ),
        }));
    }
    commands.push(json!({"step": "在目标机启动服务", "host": t.server.host, "cmd": start_cmd}));
    commands.push(json!({"step": "目标健康检查通过后，停止源机器上的服务", "host": source_host, "cmd": stop_cmd}));
    json!({
        "target_host": t.server.host,
        "target_name": t.server.name,
        "gpus": t.gpus,
        "model_local": t.model_local,
        "reason": t.reason,
        "commands": commands,
    })
}

/// 把 LLM / 规则的原始方案规范化成统一结构。
fn normalize_plan(
    svc: &ServiceConf,
    source: &ServerConf,
    raw: &Value,
    candidates: &[(ServerConf, ServerStatus)],
) -> Value {
    let raw_host = raw.get("target_host").and_then(|v| v.as_str()).unwrap_or("");
    let target_host = if raw_host.is_empty() {
        None
    } else if candidates.iter().any(|(s, _)| s.host == raw_host) {
        Some(raw_host.to_string())
    } else {
        candidates
            .iter()
            .find(|(s, _)| s.name == raw_host || s.host.contains(raw_host))
            .map(|(s, _)| s.host.clone())
    };
    let Some(th) = target_host else {
        return json!({
            "target_host": null,
            "reason": format!("无法确定目标机器: {}", raw.get("reason").and_then(|v| v.as_str()).unwrap_or("LLM 未返回 target_host")),
            "commands": [],
            "raw": raw
        });
    };
    let (target, st) = candidates
        .iter()
        .find(|(s, _)| s.host == th)
        .cloned()
        .unwrap();
    // LLM 返回的 gpus 可能是数组 [0,1] 或字符串 "0,1"
    let llm_gpus: Option<Vec<u32>> = raw.get("gpus").and_then(|v| {
        v.as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect::<Vec<_>>())
            .or_else(|| {
                v.as_str().map(|s| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<u32>().ok())
                        .collect::<Vec<_>>()
                })
            })
    });
    let gpus = match llm_gpus {
        Some(g) if !g.is_empty() && g.iter().all(|g| st.gpus.iter().any(|x| x.idx == *g)) => g,
        _ => deploy::pick_gpus(&st, svc.tp.max(1), &[]).unwrap_or_default(),
    };
    let model_local = deploy::model_exists(&st, &svc.model_path)
        || raw.get("model_local").and_then(|v| v.as_bool()).unwrap_or(false);
    let start_cmd = deploy::build_start_cmd(svc, &gpus);
    let stop_cmd = deploy::build_stop_cmd(svc);
    let mut commands: Vec<Value> = Vec::new();
    if !model_local {
        commands.push(json!({
            "step": "拷贝模型到目标机（目标机不存在该模型）",
            "host": th,
            "cmd": format!(
                "rsync -a -e ssh {u}@{src}:{model} {u}@{dst}:{model}/",
                u = target.user,
                src = source.host,
                dst = th,
                model = svc.model_path
            ),
        }));
    }
    commands.push(json!({"step": "在目标机启动服务", "host": th, "cmd": start_cmd}));
    commands.push(json!({"step": "目标健康检查通过后，停止源机器上的服务", "host": source.host, "cmd": stop_cmd}));
    json!({
        "target_host": th,
        "target_name": target.name,
        "gpus": gpus,
        "cuda_visible_devices": gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(","),
        "free_vram_gb": st.gpus.iter().map(|g| g.total_mib as f64 - g.used_mib as f64).sum::<f64>() / 1024.0,
        "model_local": model_local,
        "reason": raw.get("reason").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        "commands": commands,
    })
}

/// 蓝绿迁移执行：先在目标机启动并验证，再停源机。
async fn route_migrate(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: MigratePayload = serde_json::from_value(v).unwrap_or_default();
    let target_id = match p.to_server_id {
        Some(i) => i,
        None => return fail(session, 400, "缺少 to_server_id").await,
    };
    if target_id == svc.server_id {
        return fail(session, 400, "目标不能与源相同").await;
    }
    let Some(source) = state.find_server(svc.server_id) else {
        return fail(session, 404, "源服务器不存在").await;
    };
    let Some(target) = state.find_server(target_id) else {
        return fail(session, 404, "目标服务器不存在").await;
    };

    let mut steps: Vec<Value> = Vec::new();

    // 1) 刷新目标状态并检查模型
    let t_st = collect_and_store(state, &target).await;
    if !t_st.online {
        return fail(session, 502, &format!("目标机器不可达: {}", t_st.error)).await;
    }
    let model_local = deploy::model_exists(&t_st, &svc.model_path);
    if !model_local && !p.force {
        let rsync = format!(
            "rsync -a -e ssh {}@{}:{} {}:{}/",
            source.user, source.host, svc.model_path, source.user, target.host
        );
        steps.push(json!({
            "name": "检查模型",
            "status": "blocked",
            "detail": format!("目标机不存在模型 {}，请先执行拷贝（POST /api/services/{}/copy）或 force=true 强制启动", svc.model_path, svc.id)
        }));
        steps.push(json!({"name": "拷贝命令", "status": "manual", "detail": rsync}));
        return json(
            session,
            409,
            &json!({"ok": true, "data": {"steps": steps, "blocked": true}}),
        )
        .await;
    }
    steps.push(json!({
        "name": "检查模型",
        "status": if model_local { "done" } else { "skipped" },
        "detail": if model_local { "目标机已有模型" } else { "force 模式：模型不存在，继续" }
    }));

    // 2) 选卡 + 目标机启动
    let gpus = match deploy::pick_gpus(&t_st, svc.tp.max(1), &[]) {
        Some(g) => g,
        None => {
            steps.push(json!({"name": "选择 GPU", "status": "failed", "detail": "目标机空闲卡不足"}));
            return json(
                session,
                409,
                &json!({"ok": false, "error": "目标机空闲 GPU 卡数量不足", "steps": steps}),
            )
            .await;
        }
    };
    let start_cmd = deploy::build_start_cmd(svc, &gpus);
    steps.push(json!({"name": "目标机启动", "status": "done", "detail": format!("gpus={gpus:?}")}));
    let out = match ssh_exec(
        &target.host,
        target.port,
        &target.user,
        &target.password,
        &start_cmd,
        Duration::from_secs(90),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            steps.push(json!({"name": "目标机启动", "status": "failed", "detail": e}));
            return json(
                session,
                502,
                &json!({"ok": false, "error": "目标机启动失败（源服务未受影响）", "steps": steps}),
            )
            .await;
        }
    };
    let out_trim = out.stdout.trim().to_string();
    let new_pid: Option<u32> = if svc.deploy_type == DeployType::Docker {
        None
    } else {
        out_trim.lines().rev().find_map(|l| l.trim().parse::<u32>().ok())
    };

    // 3) 健康检查：12 秒后验证进程存活
    tokio::time::sleep(Duration::from_secs(12)).await;
    let health_cmd = if svc.deploy_type == DeployType::Docker {
        format!(
            "docker inspect -f '{{{{.State.Running}}}}' vgs_{} 2>/dev/null",
            svc.id
        )
    } else {
        new_pid
            .map(|pid| format!("kill -0 {pid} && echo alive || echo dead"))
            .unwrap_or_else(|| "echo alive".into())
    };
    let alive = match ssh_exec(
        &target.host,
        target.port,
        &target.user,
        &target.password,
        &health_cmd,
        Duration::from_secs(15),
    )
    .await
    {
        Ok(h) => {
            let t = h.stdout.trim();
            t == "alive" || t == "true" || t.parse::<u32>().map(|n| n >= 1).unwrap_or(false)
        }
        Err(_) => false,
    };
    if !alive {
        steps.push(json!({"name": "健康检查", "status": "failed", "detail": "目标服务启动后未存活，源服务未停止"}));
        state.audit.record(
            user,
            "migrate_failed",
            &format!("{} {} -> {}", svc.name, source.host, target.host),
            "健康检查失败",
            ip,
        );
        return json(
            session,
            502,
            &json!({"ok": false, "error": "目标服务健康检查失败，源服务未受影响", "steps": steps, "output": out_trim}),
        )
        .await;
    }
    steps.push(json!({"name": "健康检查", "status": "done", "detail": "目标服务已存活"}));

    // 4) 停源机
    let stop_cmd = deploy::build_stop_cmd(svc);
    match ssh_exec(
        &source.host,
        source.port,
        &source.user,
        &source.password,
        &stop_cmd,
        Duration::from_secs(30),
    )
    .await
    {
        Ok(o) => steps.push(json!({"name": "停止源服务", "status": "done", "detail": o.stdout.trim()})),
        Err(e) => steps.push(json!({"name": "停止源服务", "status": "warning", "detail": format!("源服务停止失败（请手动检查）: {e}")})),
    }

    // 5) 更新注册表
    let mut services = state.services.load().as_ref().clone();
    if let Some(s) = services.services.iter_mut().find(|x| x.id == svc.id) {
        s.server_id = target_id;
        s.status = ServiceStatus::Running;
        s.pid = new_pid;
    }
    if let Err(e) = services.save(&state.main_conf().services_path()) {
        return fail(session, 500, &e).await;
    }
    state.services.store(Arc::new(services));
    state.audit.record(
        user,
        "migrate_service",
        &format!("{} {} -> {}", svc.name, source.host, target.host),
        &format!("gpus={gpus:?}"),
        ip,
    );
    json(
        session,
        200,
        &json!({"ok": true, "data": {"steps": steps, "pid": new_pid, "gpus": gpus, "output": out_trim}}),
    )
    .await
}

/// 后台拷贝模型：在源机执行 rsync（nohup 后台）。
async fn route_copy(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: CopyPayload = serde_json::from_value(v).unwrap_or_default();
    let Some(source) = state.find_server(svc.server_id) else {
        return fail(session, 404, "源服务器不存在").await;
    };
    let Some(target) = state.find_server(p.to_server_id) else {
        return fail(session, 404, "目标服务器不存在").await;
    };
    let model = svc.model_path.trim_end_matches('/');
    let dest_dir = model
        .rsplit_once('/')
        .map(|(dir, _)| dir)
        .unwrap_or("/");
    let cmd = format!(
        "mkdir -p {dest} && nohup rsync -a -e 'ssh -o StrictHostKeyChecking=no -p {tp}' {u}@{th}:{model} {dest}/ > /tmp/vgs_copy_{id}.log 2>&1 & echo launched",
        dest = dest_dir,
        tp = target.port,
        u = target.user,
        th = target.host,
        id = svc.id,
    );
    match ssh_exec(
        &source.host,
        source.port,
        &source.user,
        &source.password,
        &cmd,
        Duration::from_secs(20),
    )
    .await
    {
        Ok(out) => {
            state.audit.record(
                user,
                "copy_model",
                &format!("{} {} -> {}", svc.name, source.host, target.host),
                out.stdout.trim(),
                ip,
            );
            ok(
                session,
                &json!({"launched": true, "log": format!("/tmp/vgs_copy_{}.log", svc.id), "output": out.stdout.trim(), "cmd": cmd}),
            )
            .await
        }
        Err(e) => {
            state.audit.record(
                user,
                "copy_model_failed",
                &format!("{} {} -> {}", svc.name, source.host, target.host),
                &e,
                ip,
            );
            fail(session, 502, &e).await
        }
    }
}

/// 查询各机器上该模型的存在状态。
async fn route_copy_status(session: &mut Session, state: &Arc<AppState>, svc: &ServiceConf) {
    let statuses = state.statuses.load();
    let servers = state.servers.load().servers.clone();
    let rows: Vec<Value> = servers
        .iter()
        .filter_map(|s| statuses.get(&s.id).map(|st| (s, st)))
        .map(|(s, st)| {
            let m = st
                .models
                .iter()
                .find(|m| m.path == svc.model_path.trim_end_matches('/'));
            json!({
                "id": s.id,
                "host": s.host,
                "name": s.name,
                "online": st.online,
                "has_model": m.is_some(),
                "model_size_gb": m.map(|m| m.size_gb).unwrap_or(0.0),
            })
        })
        .collect();
    ok(session, &json!({"model_path": svc.model_path, "servers": rows})).await
}

// ---------------------------------------------------------------------------
// 设置
// ---------------------------------------------------------------------------

async fn routes_settings(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    method: &str,
) {
    if method == "GET" {
        let m = state.main_conf();
        return json(
            session,
            200,
            &json!({
                "ok": true,
                "data": {
                    "refresh_interval_secs": m.refresh_interval_secs,
                    "ssh_timeout_secs": m.ssh_timeout_secs,
                    "ssh_concurrency": m.ssh_concurrency,
                    "llm": {
                        "enabled": m.llm.enabled,
                        "command": m.llm.command,
                        "profile": m.llm.profile,
                        "timeout_secs": m.llm.timeout_secs,
                    },
                    "admin_username": state.admin.username,
                },
            }),
        )
        .await;
    }
    if method != "PUT" {
        return fail(session, 405, "method not allowed").await;
    }
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: SettingsPayload = serde_json::from_value(v).unwrap_or_default();
    let mut m = (*state.main_conf()).clone();
    if let Some(x) = p.refresh_interval_secs {
        m.refresh_interval_secs = x.clamp(5, 3600);
    }
    if let Some(x) = p.ssh_timeout_secs {
        m.ssh_timeout_secs = x.clamp(5, 300);
    }
    if let Some(x) = p.ssh_concurrency {
        m.ssh_concurrency = x.clamp(1, 32);
    }
    if let Some(llm) = p.llm {
        if let Some(x) = llm.enabled {
            m.llm.enabled = x;
        }
        if let Some(x) = llm.command {
            m.llm.command = x;
        }
        if let Some(x) = llm.profile {
            m.llm.profile = x;
        }
        if let Some(x) = llm.timeout_secs {
            m.llm.timeout_secs = x.clamp(10, 900);
        }
    }
    // 主配置落盘（tmp + rename 原子写），mtime 变化会被 watch 循环拾取
    let text = match serde_yaml::to_string(&m) {
        Ok(t) => t,
        Err(e) => return fail(session, 500, &format!("序列化主配置失败: {e}")).await,
    };
    let path = state.main_path.clone();
    let tmp = path.with_extension("yaml.tmp");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&tmp, text.as_bytes()).and_then(|_| std::fs::rename(&tmp, &path)) {
        return fail(session, 500, &format!("主配置落盘失败: {e}")).await;
    }
    state.set_main(m);
    state
        .audit
        .record(user, "update_settings", "-", "更新全局设置", ip);
    ok(session, &json!({"saved": true})).await
}

// ---------------------------------------------------------------------------
// sglang-router 兼容检查（知识库 + AI 分析）
// ---------------------------------------------------------------------------

/// sglang-router 版本 ↔ sglang 后端兼容知识库（官方 release 实测组合）。
fn router_compat_kb() -> serde_json::Value {
    json!([
        {"router": "0.2.4", "sglang_min": "0.5.8", "sglang_max": "0.5.19", "notes": "0.2.4 于 2026-09 发布，支持 0.5.8 起的 cache_aware/power_of_two 路由；0.5.19 为当前发布验证的最高版本"},
        {"router": "0.2.3", "sglang_min": "0.5.7", "sglang_max": "0.5.19", "notes": "支持 0.5.7 起的 OpenAI 兼容端点；旧版 balance 策略"},
        {"router": "0.2.2", "sglang_min": "0.5.4", "sglang_max": "0.5.15", "notes": "早期 cache_aware；不识别 0.5.16+ 的 worker 心跳字段"},
        {"router": "0.2.1", "sglang_min": "0.5.2", "sglang_max": "0.5.12", "notes": "仅 round_robin/random；0.5.13+ 的 /v1/models 字段不兼容"},
        {"router": "0.2.0", "sglang_min": "0.5.0", "sglang_max": "0.5.8", "notes": "首版；不识别 0.5.9+ 的多模态 worker 元数据"}
    ])
}

/// 从镜像名提取版本（lmsysorg/sglang:v0.5.19-cu129 → 0.5.19）。
fn image_version(image: &str) -> Option<String> {
    let tag = image.rsplit(':').next()?.trim();
    let core = tag.split('-').next().unwrap_or(tag);
    let core = core.strip_prefix('v').unwrap_or(core);
    // 形如 0.5.19 的版本号
    if core.split('.').count() >= 2 && core.chars().all(|c| c.is_ascii_digit() || c == '.') {
        Some(core.to_string())
    } else {
        None
    }
}

fn cmp_semver(a: &str, b: &str) -> std::cmp::Ordering {
    let pa: Vec<u32> = a.split('.').filter_map(|x| x.parse().ok()).collect();
    let pb: Vec<u32> = b.split('.').filter_map(|x| x.parse().ok()).collect();
    for i in 0..3 {
        let x = pa.get(i).copied().unwrap_or(0);
        let y = pb.get(i).copied().unwrap_or(0);
        if x != y {
            return x.cmp(&y);
        }
    }
    std::cmp::Ordering::Equal
}

/// GET /api/router-compat-kb  返回 router 兼容知识库（供 UI 展示/AI 兜底前预览）
async fn route_router_compat_kb(session: &mut Session) {
    let kb = router_compat_kb();
    ok(session, &json!({"kb": kb})).await
}

/// POST /api/router-compat-check  检测机器上的 sglang-router 与其 sglang 后端的版本兼容。
/// 先查本地知识库（镜像 tag 可解析时），未命中再交 AI（知识库全文 + 检测事实注入）。
async fn route_router_compat_check(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    #[derive(serde::Deserialize, Default)]
    struct P {
        server_id: Option<u64>,
    }
    let p: P = serde_json::from_value(v).unwrap_or_default();
    let server_id = match p.server_id {
        Some(x) if x > 0 => x,
        _ => return fail(session, 400, "缺少 server_id").await,
    };
    let servers = state.servers.load();
    let server = servers
        .servers
        .iter()
        .find(|s| s.id == server_id)
        .cloned()
        .filter(|s| s.enabled);
    drop(servers);
    let server = match server {
        Some(s) => s,
        None => return fail(session, 404, &format!("服务器不存在或已禁用: {server_id}")).await,
    };

    // 1) 深扫取 router 与 GPU 后端容器
    let conf = state.main_conf();
    let timeout = Duration::from_secs(conf.ssh_timeout_secs.max(20));
    let dr = match discover::discover_one(&server.host, server.port, &server.user, &server.password, timeout).await {
        Ok(x) => x,
        Err(e) => return fail(session, 502, &format!("SSH 深扫失败: {e}")).await,
    };
    let routers: Vec<discover::RouterProc> = dr.routers.iter().filter(|r| !r.name.is_empty()).cloned().collect();
    if routers.is_empty() {
        return fail(session, 404, "该机器上未发现 sglang-router 进程").await;
    }
    // GPU 引擎进程 → 后端版本信息（按容器归组；P 行 64 位全量 cid 与 D 行 12 位短 id 前缀匹配）
    let backends: Vec<serde_json::Value> = dr
        .containers
        .iter()
        .filter(|c| {
            !c.cid.is_empty()
                && dr
                    .procs
                    .iter()
                    .any(|p| !p.gpus.is_empty() && p.cid.get(..12).unwrap_or(&p.cid) == c.cid)
        })
        .map(|c| {
            let rep = dr
                .procs
                .iter()
                .find(|p| !p.gpus.is_empty() && p.cid.get(..12).unwrap_or(&p.cid) == c.cid);
            let engine = rep
                .map(|p| discover::classify(&p.name, &p.root_cmdline).to_string())
                .unwrap_or_default();
            let cmd = rep
                .map(|p| p.root_cmdline.chars().take(400).collect::<String>())
                .unwrap_or_default();
            json!({
                "container": c.name,
                "image": c.image,
                "engine": engine,
                "cmdline": cmd,
            })
        })
        .collect();

    // 2) 本地知识库判定（镜像 tag 可解析时给出明确结论）
    let mut local_verdict: Option<Value> = None;
    let mut facts: Vec<String> = Vec::new();
    for r in &routers {
        let cname = if r.cid.is_empty() {
            String::new()
        } else {
            dr.containers
                .iter()
                .find(|c| c.cid == r.cid.get(..12).unwrap_or(&r.cid))
                .map(|c| c.name.clone())
                .unwrap_or_default()
        };
        let image = if r.cid.is_empty() {
            String::new()
        } else {
            dr.containers
                .iter()
                .find(|c| c.cid == r.cid.get(..12).unwrap_or(&r.cid))
                .map(|c| c.image.clone())
                .unwrap_or_default()
        };
        let rver = image_version(&image).unwrap_or_default();
        facts.push(format!(
            "router 进程 pid={} comm={} 容器={} 镜像={} 解析版本={} 监听端口={:?}",
            r.pid, r.name, cname, image, if rver.is_empty() { "?" } else { rver.as_str() }, r.ports
        ));
        if !rver.is_empty() {
            let kb = router_compat_kb();
            if let Some(arr) = kb.as_array() {
                for row in arr.iter().filter_map(|x| x.as_object()) {
                    let kver = row.get("router").and_then(|x| x.as_str()).unwrap_or("");
                    if kver == rver {
                        let min = row.get("sglang_min").and_then(|x| x.as_str()).unwrap_or("");
                        let max = row.get("sglang_max").and_then(|x| x.as_str()).unwrap_or("");
                        // 后端镜像版本（同机 GPU 容器）
                        let bvers: Vec<String> = dr
                            .containers
                            .iter()
                            .filter(|c| {
                                !c.cid.is_empty()
                                    && dr
                                        .procs
                                        .iter()
                                        .any(|p| !p.gpus.is_empty() && p.cid.get(..12).unwrap_or(&p.cid) == c.cid)
                            })
                            .filter_map(|c| image_version(&c.image))
                            .collect();
                        let mut bad: Vec<String> = Vec::new();
                        for bv in &bvers {
                            if !min.is_empty() && cmp_semver(bv, min) == std::cmp::Ordering::Less {
                                bad.push(format!("{bv} < 最低 {min}"));
                            }
                            if !max.is_empty() && cmp_semver(bv, max) == std::cmp::Ordering::Greater {
                                bad.push(format!("{bv} > 验证上限 {max}（未实测，风险）"));
                            }
                        }
                        let compatible = bad.is_empty();
                        local_verdict = Some(json!({
                            "source": "knowledge_base",
                            "compatible": compatible,
                            "router_version": rver,
                            "sglang_min": min,
                            "sglang_max": max,
                            "backend_versions": bvers,
                            "problems": bad,
                            "notes": row.get("notes").and_then(|x| x.as_str()).unwrap_or(""),
                        }));
                        break;
                    }
                }
            }
            if local_verdict.is_none() {
                // 知识库无该 router 版本 → 交 AI 兜底
                facts.push(format!("知识库无 router v{rver} 的记录，需要 AI 分析"));
            }
        }
    }

    // 3) 知识库未命中 → AI 兜底（知识库全文 + 检测事实注入）
    let result = if let Some(lv) = local_verdict {
        lv
    } else {
        let kb = router_compat_kb();
        let prompt = format!(
            "你是 LLM 推理网关兼容性专家。请判断下面的 sglang-router 与其 sglang 后端是否版本兼容。\n\n\
             [已知兼容知识库]\n{kb}\n\n[本机检测事实]\n{facts}\n[后端进程 cmdline 样本]\n{backends}\n\n\
             请输出 JSON：{{\"compatible\": true|false, \"verdict\": \"一句话结论\", \"problems\": [\"问题1\", ...], \
             \"suggestion\": \"若需修复，给出一条具体建议（版本对齐方向）\"}}。只输出 JSON。",
            kb = kb.to_string(),
            facts = facts.join("\n"),
            backends = backends
                .iter()
                .take(4)
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let llm = conf.llm.clone();
        if !llm.enabled {
            json!({
                "source": "none",
                "compatible": null,
                "verdict": "知识库无该 sglang-router 版本记录，且 LLM 已禁用，无法自动判定",
                "problems": [],
                "suggestion": "请将 sglang-router 升级到知识库内版本，或在配置中启用 LLM 后重试"
            })
        } else {
            match tokio::time::timeout(
                Duration::from_secs(llm.timeout_secs),
                ai::ask_json(&llm, &prompt),
            )
            .await
            {
                Ok(Ok(raw)) => {
                    let parsed = ai::unwrap_llm_json(&raw);
                    if parsed.get("verdict").and_then(|x| x.as_str()).is_none()
                        && parsed.get("compatible").is_none()
                    {
                        return fail(session, 502, "AI 返回无法解析为兼容结论").await
                    }
                    parsed
                }
                Ok(Err(e)) => return fail(session, 502, &format!("AI 调用失败: {e}")).await,
                Err(_) => return fail(session, 502, "AI 调用超时").await,
            }
        }
    };
    state
        .audit
        .record(user, "router_compat_check", &server.name, "router 兼容检查", ip);
    ok(session, &result).await
}

// ---------------------------------------------------------------------------
// 智能检测（扫描机群真实运行的模型服务 + 一键纳管）
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
struct DiscoverPayload {
    server_ids: Option<Vec<u64>>,
}

#[derive(serde::Deserialize, Default)]
struct AdoptPayload {
    server_id: Option<u64>,
    name: Option<String>,
    model_path: Option<String>,
    engine: Option<String>,
    pid: Option<u32>,
    /// 检测进程树内全部 pid（查重时覆盖「主 pid 变化但树内仍有旧 pid 注册」的情况）
    pids: Option<Vec<u32>>,
    /// 使用中的 GPU 编号（用于生成可区分的默认服务名）
    gpus: Option<Vec<u32>>,
    port: Option<u16>,
    tp: Option<u32>,
    extra_args: Option<String>,
    cmdline: Option<String>,
    model_name: Option<String>,
    /// 纳管的外部 docker 容器名（智能检测发现时记录，停止时精确 stop）
    container_name: Option<String>,
    container_image: Option<String>,
}

async fn routes_discover(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    method: &str,
    second: Option<&str>,
) {
    if second == Some("adopt") && method == "POST" {
        return route_discover_adopt(session, state, user, ip).await;
    }
    if method != "POST" {
        return fail(session, 405, "method not allowed").await;
    }
    route_discover_scan(session, state, user, ip).await
}

/// POST /api/discover  全量（或指定机器）深扫：GPU 进程 + 容器 + 模型位置，
/// 返回识别出的服务候选列表（含驱动 / CUDA / 模型大小 / 是否已注册）。
async fn route_discover_scan(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: DiscoverPayload = serde_json::from_value(v).unwrap_or_default();
    let ids = p.server_ids.unwrap_or_default();
    let servers = state.servers.load();
    let services = state.services.load();
    let targets: Vec<ServerConf> = servers
        .servers
        .iter()
        .filter(|s| s.enabled && (ids.is_empty() || ids.contains(&s.id)))
        .cloned()
        .collect();
    if targets.is_empty() {
        return fail(session, 400, "没有目标机器（可能全部被禁用或未指定）").await;
    }

    // 1) 快照陈旧（>90s 或离线）的机器先补采一次，供模型/驱动信息交叉比对
    let conf = state.main_conf();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let statuses = state.statuses.load();
    let stale: Vec<ServerConf> = targets
        .iter()
        .filter(|s| {
            !statuses
                .get(&s.id)
                .map(|st| st.online && now as i64 - st.updated_at < 90)
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    drop(statuses);
    let mut collect_failed = Vec::new();
    if !stale.is_empty() {
        let sem = Arc::new(tokio::sync::Semaphore::new(conf.ssh_concurrency.max(1) as usize));
        let mut handles = Vec::new();
        for s in stale {
            let sem = sem.clone();
            let state = state.clone();
            handles.push(tokio::spawn(async move {
                let _perm = sem.acquire().await;
                collect_and_store(&state, &s).await
            }));
        }
        for h in handles {
            match h.await {
                Ok(st) if st.online => {}
                Ok(st) => collect_failed.push(json!({"error": st.error})),
                Err(e) => collect_failed.push(json!({"error": e.to_string()})),
            }
        }
    }

    // 2) 逐台深扫（一次 SSH 往返：完整 cmdline + 父进程 + 容器归属），并发受限
    let statuses = state.statuses.load();
    let timeout = Duration::from_secs(conf.ssh_timeout_secs.max(45));
    let sem = Arc::new(tokio::sync::Semaphore::new(conf.ssh_concurrency.max(1) as usize));
    let mut handles = Vec::new();
    for s in &targets {
        let s = s.clone();
        let sem = sem.clone();
        handles.push(tokio::spawn(async move {
            let _perm = sem.acquire().await;
            (
                s.id,
                discover::discover_one(&s.host, s.port, &s.user, &s.password, timeout).await,
            )
        }));
    }
    let mut items: Vec<discover::Detected> = Vec::new();
    let mut failed: Vec<Value> = Vec::new();
    for h in handles {
        match h.await {
            Ok((id, Ok(dr))) => {
                // 快照缺失（补采失败）不影响识别，detect 会用空快照降级
                let st = statuses.get(&id);
                if let Some(srv) = servers.servers.iter().find(|s| s.id == id) {
                    items.extend(discover::detect(srv, st, &dr, &services.services));
                }
            }
            Ok((id, Err(e))) => failed.push(json!({"server_id": id, "error": e})),
            Err(e) => failed.push(json!({"error": e.to_string()})),
        }
    }
    // 按引擎优先级排序（推理引擎在前，其次 python，最后 other）
    items.sort_by(|a, b| {
        discover::engine_rank(&a.engine)
            .cmp(&discover::engine_rank(&b.engine))
            .then(a.server_name.cmp(&b.server_name))
            .then(a.pid.cmp(&b.pid))
    });
    state.audit.record(
        user,
        "discover_scan",
        "-",
        &format!("智能检测扫描 {} 台机器，识别出 {} 个运行中服务", targets.len(), items.len()),
        ip,
    );
    json(
        session,
        200,
        &json!({"ok": true, "data": {"items": items, "scanned": targets.len(), "failed": failed}}),
    )
    .await
}

/// POST /api/discover/adopt  将一条检测结果注册为模型服务（状态置 Running，
/// 之后 UI 的停止/迁移均可按 stop_pattern / pid 正常操作）。
async fn route_discover_adopt(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
) {
    let v = parse_body(session).await.unwrap_or(Value::Null);
    let p: AdoptPayload = serde_json::from_value(v).unwrap_or_default();
    let Some(server_id) = p.server_id else {
        return fail(session, 400, "server_id 必填").await;
    };
    let servers = state.servers.load();
    let Some(server) = servers.servers.iter().find(|s| s.id == server_id) else {
        return fail(session, 400, "机器不存在").await;
    };
    let model_path = p.model_path.clone().unwrap_or_default();
    let cmdline = p.cmdline.clone().unwrap_or_default();

    let mut services = state.services.load().as_ref().clone();
    // 该检测树涉及的全部 pid（主 pid + 同组子进程）
    let pids: Vec<u32> = match &p.pids {
        Some(v) if !v.is_empty() => v.clone(),
        _ => p.pid.into_iter().collect(),
    };
    // 查重：同机器且（pid 在检测树内 / 同容器 / 同模型路径+端口冲突）视为重复。
    // 同模型多实例（不同容器、不同端口）允许分别纳管。
    let det_cname = p.container_name.as_deref().unwrap_or("").trim().to_string();
    let det_port = p.port.unwrap_or(0);
    let dup = services.services.iter().any(|s| {
        if s.server_id != server_id {
            return false;
        }
        if pids.iter().any(|x| s.pid == Some(*x)) {
            return true;
        }
        let s_cname = s.container_name.trim();
        if !s_cname.is_empty() && !det_cname.is_empty() && s_cname == det_cname {
            return true;
        }
        if !model_path.is_empty() && !s.model_path.is_empty()
            && (s.model_path == model_path
                || s.model_path.starts_with(&model_path)
                || model_path.starts_with(&s.model_path))
        {
            // 两边都绑定容器且容器名不同 → 独立实例，允许分别纳管
            if !s_cname.is_empty() && !det_cname.is_empty() && s_cname != det_cname {
                return false;
            }
            // 同模型：仅当端口相同（或任一侧未知=0）才视为同一服务
            match (det_port, s.port) {
                (a, b) if a != 0 && b != 0 => a == b,
                _ => true,
            }
        } else {
            false
        }
    });
    if dup {
        return fail(
            session,
            409,
            "该机器上已存在相同服务（已在服务列表中，无需重复注册）",
        )
        .await;
    }

    let engine = p.engine.clone().unwrap_or_else(|| "python".to_string());
    let deploy_type = match engine.as_str() {
        "vllm" => DeployType::Vllm,
        "sglang" => DeployType::Sglang,
        "sglang-router" => DeployType::Docker,
        "triton" => DeployType::Docker,
        _ => DeployType::Python,
    };
    let gpus = p.gpus.clone().unwrap_or_default();
    let gpus_str = gpus.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    // 名称：模型目录名 → 模型标题 → gpu-{机器名}(-gpu{卡号})，再保证唯一。
    // 无模型路径的进程带 GPU 卡号命名，避免同机多个同类进程都叫 gpu-node07 只能靠 -2/-3 区分。
    let mut name = p
        .name
        .clone()
        .filter(|x| !x.trim().is_empty())
        .unwrap_or_default();
    if name.is_empty() {
        name = model_path
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or("")
            .to_string();
    }
    if name.is_empty() {
        name = p
            .model_name
            .clone()
            .filter(|x| !x.trim().is_empty())
            .unwrap_or_default();
    }
    if name.is_empty() {
        // 容器纳管（含 sglang-router 等无模型无卡的路由服务）：用容器名
        name = p
            .container_name
            .clone()
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .unwrap_or_default();
    }
    if name.is_empty() {
        name = if gpus.is_empty() {
            format!("gpu-{}", server.name)
        } else {
            let g = gpus.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-");
            format!("gpu-{}-gpu{}", server.name, g)
        };
    }
    let mut candidate = name.clone();
    let mut n = 2;
    while services.services.iter().any(|s| s.name == candidate) {
        candidate = format!("{name}-{n}");
        n += 1;
    }
    name = candidate;
    // 停止模式：统一「模型路径做 pkill 模式 + 主 pid」双保险；
    // 无模型路径时 stop_pattern 留空（build_stop_cmd 回退 model_path，仍为空则纯 pid TERM→KILL），
    // 不用 cmdline 前缀当 pkill 模式 —— 同机 "python server.py" 这类通用命令会互相误杀
    let stop_pattern = model_path.clone();
    let new_id = services.services.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    let svc = ServiceConf {
        id: new_id,
        name: name.clone(),
        model_path: model_path.clone(),
        server_id,
        deploy_type,
        port: p.port.unwrap_or(8000),
        tp: p.tp.unwrap_or(1).max(1),
        extra_args: p.extra_args.clone().unwrap_or_default(),
        docker_image: p.container_image.clone().unwrap_or_default(),
        docker_args: String::new(),
        container_name: p
            .container_name
            .clone()
            .map(|x| x.trim().to_string())
            .unwrap_or_default(),
        python_cmd: if matches!(deploy_type, DeployType::Python) {
            cmdline.clone()
        } else {
            String::new()
        },
        stop_pattern,
        status: ServiceStatus::Running,
        pid: p.pid,
        notes: format!(
            "智能检测纳管（{}{}）；原命令: {}",
            engine,
            if gpus_str.is_empty() {
                String::new()
            } else {
                format!(", GPU {}", gpus_str)
            },
            cmdline.chars().take(160).collect::<String>()
        ),
    };
    services.services.push(svc);
    if let Err(e) = services.save(&state.main_conf().services_path()) {
        return fail(session, 500, &e).await;
    }
    state.services.store(Arc::new(services));
    state.audit.record(
        user,
        "discover_adopt",
        &name,
        &format!(
            "智能检测纳管：机器 {}，pid {}",
            server.name,
            p.pid.unwrap_or(0)
        ),
        ip,
    );
    ok(session, &json!({"id": new_id, "name": name})).await
}
