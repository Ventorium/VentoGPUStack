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
    if method != "GET" {
        return fail(session, 405, "method not allowed").await;
    }

    let servers = state.servers.load();
    let statuses = state.statuses.load();
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
                "models_in_use": st.map(|x| x.models.iter().filter(|m| !m.in_use_by.is_empty())
                    .map(|m| json!({"path": m.path, "by": m.in_use_by})).collect::<Vec<_>>()).unwrap_or_default(),
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
/// 1) GPU 进程 cmdline 含 model_path（vllm/sglang 等 GPU 服务，本地零成本）；
/// 2) 否则查 /proc/{pid} 存在性（不限进程属主 —— 集群服务多为 root/其他用户运行，kill -0 会 EPERM）；
/// 3) docker 类型检查容器是否存在。
/// SSH 校验失败（机器不可达）一律按未存活处理，避免误报 running。
async fn service_alive(svc: &ServiceConf, state: &Arc<AppState>) -> bool {
    let servers = state.servers.load();
    let Some(server) = servers.servers.iter().find(|s| s.id == svc.server_id) else {
        return false;
    };
    let timeout = Duration::from_secs(15);
    // 自管容器（平台 docker run 启动，容器名 vgs_{id}）按容器名判定；
    // 智能检测纳管的外部容器 stop_pattern 非空，走下面的 cmdline/pid 判定。
    if svc.deploy_type == DeployType::Docker && svc.stop_pattern.trim().is_empty() {
        let cmd = format!("docker ps -q --filter name=^vgs_{}$", svc.id);
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
    let Some(server) = state.find_server(target_id) else {
        return fail(session, 404, "目标服务器不存在").await;
    };
    let st = collect_and_store(state, &server).await;
    if !st.online {
        return fail(session, 502, &format!("目标机器采集失败: {}", st.error)).await;
    }
    if st.gpus.is_empty() {
        return fail(session, 502, "目标机器没有可用的 GPU").await;
    }
    let gpus = match deploy::pick_gpus(&st, svc.tp.max(1), &[]) {
        Some(g) => g,
        None => return fail(session, 409, "空闲 GPU 卡数量不足").await,
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
            let pid: Option<u32> = if svc.deploy_type == DeployType::Docker {
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
                return fail(session, 500, &e).await;
            }
            state.services.store(Arc::new(services));
            state.audit.record(
                user,
                "start_service",
                &format!("{}@{}", svc.name, server.host),
                &format!("gpus={gpus:?} exit={}", out.exit_code),
                ip,
            );
            ok(
                session,
                &json!({"pid": pid, "gpus": gpus, "cmd": cmd, "output": out_trim, "exit_code": out.exit_code}),
            )
            .await
        }
        Err(e) => {
            state
                .audit
                .record(user, "start_service_failed", &format!("{}@{}", svc.name, server.host), &e, ip);
            fail(session, 502, &e).await
        }
    }
}

async fn route_stop(
    session: &mut Session,
    state: &Arc<AppState>,
    user: &str,
    ip: &str,
    svc: &ServiceConf,
) {
    let Some(server) = state.find_server(svc.server_id) else {
        return fail(session, 404, "服务所在服务器不存在").await;
    };
    let cmd = deploy::build_stop_cmd(svc);
    let timeout = Duration::from_secs(30);
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
            let mut services = state.services.load().as_ref().clone();
            if let Some(s) = services.services.iter_mut().find(|x| x.id == svc.id) {
                s.status = ServiceStatus::Stopped;
                s.pid = None;
            }
            if let Err(e) = services.save(&state.main_conf().services_path()) {
                return fail(session, 500, &e).await;
            }
            state.services.store(Arc::new(services));
            state
                .audit
                .record(user, "stop_service", &format!("{}@{}", svc.name, server.host), out.stdout.trim(), ip);
            ok(session, &json!({"output": out.stdout.trim(), "exit_code": out.exit_code})).await
        }
        Err(e) => {
            state
                .audit
                .record(user, "stop_service_failed", &format!("{}@{}", svc.name, server.host), &e, ip);
            fail(session, 502, &e).await
        }
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
    let rank = |e: &str| -> u8 {
        match e {
            "vllm" | "sglang" | "triton" | "xinference" | "paddlex" | "funasr" | "ollama" => 0,
            "python" => 1,
            _ => 2,
        }
    };
    items.sort_by(|a, b| {
        rank(&a.engine)
            .cmp(&rank(&b.engine))
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
    // 查重：同机器同模型路径（互为前缀视为同一模型），或同机器且 pid 在检测树内
    // （即使已注册服务的 pid 是该进程组的子/父进程，也视为重复，避免重复纳管）
    let dup = services
        .services
        .iter()
        .any(|s| {
            s.server_id == server_id
                && (if !model_path.is_empty() && !s.model_path.is_empty() {
                    s.model_path == model_path
                        || s.model_path.starts_with(&model_path)
                        || model_path.starts_with(&s.model_path)
                } else {
                    pids.iter().any(|x| s.pid == Some(*x))
                })
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
        docker_image: String::new(),
        docker_args: String::new(),
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
