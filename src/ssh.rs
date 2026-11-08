//! SSH 客户端封装（纯 Rust，基于 russh）。
//! 对目标机器执行单条命令并回收 stdout 与退出码。

use std::sync::Arc;
use std::time::Duration;

use russh::client;
use russh::{ChannelMsg, Disconnect};

#[derive(Debug, Clone)]
pub struct SshExecResult {
    pub stdout: String,
    pub exit_code: i32,
}

struct Handler;

impl client::Handler for Handler {
    type Error = russh::Error;

    fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::PublicKeyOrCertificate,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        async { Ok(true) }
    }
}

/// 建立 SSH 连接并执行一条命令。
#[allow(clippy::too_many_arguments)]
pub async fn ssh_exec(
    host: &str,
    port: u16,
    user: &str,
    password: &str,
    command: &str,
    timeout: Duration,
) -> Result<SshExecResult, String> {
    let config = client::Config {
        inactivity_timeout: Some(timeout),
        ..Default::default()
    };
    let config = Arc::new(config);

    let handle = client::connect(config, (host, port), Handler)
        .await
        .map_err(|e| format!("连接失败: {e}"))?;
    let mut handle = handle;

    let auth = handle
        .authenticate_password(user, password)
        .await
        .map_err(|e| format!("认证失败: {e}"))?;
    if !auth.success() {
        return Err("认证失败: 用户名或密码错误".into());
    }

    let res = tokio::time::timeout(timeout, run_command(&mut handle, command))
        .await
        .map_err(|_| format!("执行超时（{}s）", timeout.as_secs()))?
        .map_err(|e| format!("执行失败: {e}"))?;

    // 优雅断开；失败不影响结果
    let _ = handle
        .disconnect(Disconnect::ByApplication, "ventogpustack", "en")
        .await;

    Ok(res)
}

async fn run_command(
    handle: &mut client::Handle<Handler>,
    command: &str,
) -> Result<SshExecResult, String> {
    let mut channel = handle
        .channel_open_session()
        .await
        .map_err(|e| format!("打开会话失败: {e}"))?;
    channel
        .exec(true, command)
        .await
        .map_err(|e| format!("下发命令失败: {e}"))?;

    let mut stdout = String::new();
    let mut exit_code: i32 = -1;
    loop {
        let Some(msg) = channel.wait().await else {
            break;
        };
        match msg {
            ChannelMsg::Data { data } => stdout.push_str(&String::from_utf8_lossy(&data)),
            ChannelMsg::ExitStatus { exit_status } => exit_code = exit_status as i32,
            ChannelMsg::Close => break,
            _ => {}
        }
    }
    Ok(SshExecResult { stdout, exit_code })
}
