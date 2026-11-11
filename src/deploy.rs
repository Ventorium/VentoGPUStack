//! 模型服务部署命令构造与远程执行。
//! 异构部署：vllm / sglang / docker / python，由 ServiceConf.deploy_type 决定。

use crate::config::{DeployType, ServerConf, ServerStatus, ServiceConf};
use crate::ssh::ssh_exec;
use std::time::Duration;

/// 选择 GPU 卡：优先选空闲卡（util=0 且显存占用小），返回卡号列表。
pub fn pick_gpus(status: &ServerStatus, count: u32, exclude: &[u32]) -> Option<Vec<u32>> {
    let mut cards: Vec<(u32, u32, u32)> = status
        .gpus
        .iter()
        .filter(|g| !exclude.contains(&g.idx))
        .map(|g| (g.idx, g.util_pct, g.used_mib))
        .collect();
    cards.sort_by_key(|c| (c.1, c.2, c.0));
    let chosen: Vec<u32> = cards.iter().take(count as usize).map(|c| c.0).collect();
    if chosen.len() == count as usize {
        Some(chosen)
    } else {
        None
    }
}

/// 目标机器上是否已有该模型目录。
pub fn model_exists(status: &ServerStatus, model_path: &str) -> bool {
    status.models.iter().any(|m| m.path == model_path)
}

/// 生成启动命令。gpus 为 CUDA_VISIBLE_DEVICES 的卡号列表。
pub fn build_start_cmd(service: &ServiceConf, gpus: &[u32]) -> String {
    // 纳管的外部容器：按容器名幂等启动（已运行则无操作），不涉及 GPU 选择
    if !service.container_name.trim().is_empty() {
        let cname = service.container_name.trim();
        return format!(
            "docker start {cname} 2>/dev/null && echo vgs_started; docker inspect -f '{{{{.State.Status}}}}' {cname} 2>/dev/null || echo unknown"
        );
    }
    let cvd: Vec<String> = gpus.iter().map(|g| g.to_string()).collect();
    let cvd = cvd.join(",");
    let log = format!("/tmp/vgs_{}.log", service.id);
    let pid_file = format!("/tmp/vgs_{}.pid", service.id);
    match service.deploy_type {
        DeployType::Vllm => {
            let extra = if service.extra_args.trim().is_empty() {
                String::new()
            } else {
                format!(" {}", service.extra_args.trim())
            };
            format!(
                "CUDA_VISIBLE_DEVICES={cvd} nohup vllm serve {model} \
                 --served-model-name {name} --port {port} --tensor-parallel-size {tp}{extra} \
                 > {log} 2>&1 & echo $! > {pid_file} && cat {pid_file}",
                model = service.model_path,
                name = service.name,
                port = service.port,
                tp = service.tp,
            )
        }
        DeployType::Sglang => {
            let extra = if service.extra_args.trim().is_empty() {
                String::new()
            } else {
                format!(" {}", service.extra_args.trim())
            };
            format!(
                "CUDA_VISIBLE_DEVICES={cvd} nohup python3 -m sglang.launch_server \
                 --model-path {model} --port {port} --tp {tp}{extra} \
                 > {log} 2>&1 & echo $! > {pid_file} && cat {pid_file}",
                model = service.model_path,
                port = service.port,
                tp = service.tp,
            )
        }
        DeployType::Docker => {
            let devs: Vec<String> = gpus.iter().map(|g| g.to_string()).collect();
            let extra = if service.docker_args.trim().is_empty() {
                String::new()
            } else {
                format!(" {}", service.docker_args.trim())
            };
            format!(
                "docker rm -f vgs_{id} 2>/dev/null; docker run -d --name vgs_{id} \
                 --gpus \"device={devs}\" --ipc=host \
                 -v {model}:{model} -p {port}:{port} \
                 {image}{extra} && docker logs vgs_{id} 2>&1 | tail -5",
                id = service.id,
                devs = devs.join(","),
                model = service.model_path,
                port = service.port,
                image = service.docker_image,
            )
        }
        DeployType::Python => {
            let cmd = service.python_cmd.replace("{model}", &service.model_path)
                .replace("{port}", &service.port.to_string())
                .replace("{gpus}", &cvd);
            format!(
                "CUDA_VISIBLE_DEVICES={cvd} nohup bash -c '{cmd}' \
                 > {log} 2>&1 & echo $! > {pid_file} && cat {pid_file}"
            )
        }
    }
}

/// pkill 自匹配防护：远端命令由 `sh -c` 执行，其 cmdline 含完整命令文本（即 pattern 本身），
/// 直接 `pkill -f {pattern}` 会连自己的会话 shell 一起杀掉。
/// 将 pattern 首字符包进字符类（`/models` → `[/]models`）后，shell 自身不再匹配，目标进程照常匹配。
fn pkill_safe_pattern(pattern: &str) -> String {
    let mut chars = pattern.chars();
    match chars.next() {
        Some(c) => format!("[{c}]{rest}", rest = chars.collect::<String>()),
        None => String::new(),
    }
}

/// 生成停止命令：优先 pkill 进程模式；docker 则 rm -f 容器。
/// 集群服务多为 root/其他用户运行：先 `sudo -n` 信号（可杀任意属主），
/// 无免密 sudo 的机器上自动回退为本用户信号（平台自管的进程仍可停止）。
pub fn build_stop_cmd(service: &ServiceConf) -> String {
    // 纳管的外部容器（任何 deploy_type）：优先按容器名精确停止，
    // 避免 pkill 误伤同机同模型路径的其它实例
    if !service.container_name.trim().is_empty() {
        let cname = service.container_name.trim().to_string();
        return format!(
            "{}; echo stopped",
            sig(&format!("docker stop -t 60 {cname} 2>/dev/null"))
        );
    }
    let pattern = if service.stop_pattern.trim().is_empty() {
        service.model_path.clone()
    } else {
        service.stop_pattern.trim().to_string()
    };
    /// "sudo 优先，失败回退本用户"
    fn sig(cmd: &str) -> String {
        format!("sudo -n {cmd} 2>/dev/null || {cmd} 2>/dev/null")
    }
    match service.deploy_type {
        DeployType::Docker => {
            if pattern.is_empty() {
                // 平台自管容器（容器名 vgs_{id}）
                format!("docker rm -f vgs_{} 2>/dev/null; echo done", service.id)
            } else {
                // 智能检测纳管的外部容器：按进程模式停止（容器主进程退出后容器即停止）
                let pk = pkill_safe_pattern(&pattern);
                format!(
                    "{}; sleep 3; {}; echo stopped",
                    sig(&format!("pkill -f {pk}")),
                    sig(&format!("pkill -9 -f {pk}"))
                )
            }
        }
        _ => {
            let pid = service.pid.map(|p| p.to_string()).unwrap_or_default();
            let pidfile = format!("/tmp/vgs_{}.pid", service.id);
            let mut cmds = Vec::new();
            if !pid.is_empty() {
                cmds.push(sig(&format!("kill {pid}")));
            }
            // 空 pattern 时禁止 pkill -f（空模式会匹配所有进程）
            if !pattern.is_empty() {
                let pk = pkill_safe_pattern(&pattern);
                cmds.push(sig(&format!("pkill -f {pk}")));
                cmds.push(format!("sleep 3; {}", sig(&format!("pkill -9 -f {pk}"))));
            } else if !pid.is_empty() {
                // 只有 pid（如 spawn 子进程组）：TERM 后升级 KILL
                cmds.push(format!("sleep 3; {}", sig(&format!("kill -9 {pid}"))));
            }
            cmds.push(format!("rm -f {pidfile} 2>/dev/null; echo stopped"));
            cmds.join(" ; ")
        }
    }
}

/// 停止后校验：相关进程是否已退出、显存是否释放。
pub async fn verify_stopped(server: &ServerConf, timeout: Duration) -> (bool, String) {
    let cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\\n' ' '";
    match ssh_exec(&server.host, server.port, &server.user, &server.password, cmd, timeout).await {
        Ok(r) => (true, format!("各卡剩余显存(MiB): {}", r.stdout.trim())),
        Err(e) => (false, format!("校验失败: {e}")),
    }
}
