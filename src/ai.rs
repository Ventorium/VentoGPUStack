//! LLM 集成：调用 `dsh headless` 执行研判 / 调度任务，并稳健提取末尾的 JSON 对象。
//! 所有调用均为异步（单次 10~40s），超时后回退到规则引擎（scheduler.rs）。

use std::time::Duration;

use serde_json::Value;

use crate::config::{LlmConf, ServerConf, ServerStatus, ServiceConf};

/// 调用 LLM 并把结果解析为 JSON；失败时返回 Err（由调用方决定回退策略）。
pub async fn ask_json(conf: &LlmConf, prompt: &str) -> Result<serde_json::Value, String> {
    if !conf.enabled {
        return Err("LLM 已禁用".into());
    }
    let timeout = Duration::from_secs(conf.timeout_secs.max(10));
    let output = tokio::time::timeout(
        timeout,
        tokio::process::Command::new(&conf.command)
            .arg("headless")
            .arg(&conf.profile)
            .arg(prompt)
            .output(),
    )
    .await
    .map_err(|_| format!("LLM 调用超时（{}s）", timeout.as_secs()))?
    .map_err(|e| format!("启动 LLM 进程失败: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("LLM 进程退出码 {:?}: {}", output.status.code(), stderr.trim()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    extract_last_json(&stdout)
        .ok_or_else(|| format!("LLM 输出中未找到可解析的 JSON 对象（输出长度 {}）", stdout.len()))
}

/// 从输出中提取最后一个能解析成功的顶层 JSON 对象。
/// LLM 会在 JSON 前后附带动机说明，这里做平衡括号扫描兜底。
pub fn extract_last_json(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 0;
    let mut start: Option<usize> = None;
    let mut last_good: Option<(usize, usize)> = None;
    let mut in_str = false;
    let mut esc = false;
    for (i, b) in bytes.iter().enumerate() {
        let c = *b as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        if serde_json::from_str::<Value>(&text[s..=i]).is_ok() {
                            last_good = Some((s, i));
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    last_good.and_then(|(s, e)| serde_json::from_str(&text[s..=e]).ok())
}

// ---------------------------------------------------------------------------
// Prompt 构造
// ---------------------------------------------------------------------------

fn json_hints() -> &'static str {
    "要求：\n1. 只基于输入数据做判断，不要编造不存在的机器或服务。\n2. 最后只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字）。\n3. 严格遵循输入中 output_format 的键名：每台机器必须包含 safe、risk、needs_stop、stop_services、reason 五个字段，reason 用中文一句话说明。\n"
}

/// 驱动 / CUDA 升级研判 prompt。
pub fn driver_upgrade_prompt(
    target_driver: &str,
    target_cuda: &str,
    servers: &[(ServerConf, ServerStatus)],
) -> String {
    let mut fleet = serde_json::json!({});
    for (s, st) in servers {
        if !st.online {
            continue;
        }
        fleet[s.host.clone()] = serde_json::json!({
            "current_driver": st.driver_version,
            "current_cuda": st.cuda_version,
            "gpus": st.gpus.len(),
            "gpu_memory_used_mib": st.gpus.iter().map(|g| g.used_mib).sum::<u32>(),
            "gpu_util_pct_max": st.gpus.iter().map(|g| g.util_pct).max().unwrap_or(0),
            "ram_used_gb": st.ram_used_gb,
            "running_processes": st.procs.iter().map(|p| serde_json::json!({
                "name": p.name,
                "gpu_mem_mib": p.mem_mib,
                "cmdline": p.cmdline.chars().take(120).collect::<String>(),
            })).collect::<Vec<_>>(),
        });
    }
    let data = serde_json::json!({
        "task": "判断对下列每台机器升级 NVIDIA 驱动与 CUDA 是否安全",
        "target_driver": target_driver,
        "target_cuda": target_cuda,
        "servers": fleet,
        "output_format": {
            "servers": {
                "<机器IP>": {
                    "safe": "true 或 false",
                    "risk": "low / medium / high",
                    "needs_stop": "true 或 false",
                    "stop_services": ["需要停止的进程或服务的简短名称"],
                    "reason": "一句中文理由，结合当前驱动版本、在跑负载与显存占用说明风险"
                }
            }
        }
    });
    format!(
        "{}{}",
        serde_json::to_string_pretty(&data).unwrap_or_default(),
        json_hints()
    )
}

/// 迁移调度 prompt：选择目标机器与卡，给出完整命令序列。
pub fn migration_prompt(
    service: &ServiceConf,
    servers: &[(ServerConf, ServerStatus)],
    hint_target_host: Option<&str>,
) -> String {
    let mut fleet: Vec<serde_json::Value> = Vec::new();
    for (s, st) in servers {
        if !st.online || !s.enabled {
            continue;
        }
        fleet.push(serde_json::json!({
            "host": s.host,
            "name": s.name,
            "gpus": st.gpus.iter().map(|g| serde_json::json!({
                "idx": g.idx,
                "total_mib": g.total_mib,
                "used_mib": g.used_mib,
                "util_pct": g.util_pct,
            })).collect::<Vec<_>>(),
            "ram_used_gb": st.ram_used_gb,
            "ram_total_gb": st.ram_total_gb,
            "model_dirs": st.models.iter().map(|m| &m.path).collect::<Vec<_>>(),
            "running_processes": st.procs.iter().map(|p| &p.name).collect::<Vec<_>>(),
        }));
    }
    let data = serde_json::json!({
        "task": "为模型服务选择最佳迁移目标机器与 GPU，并给出完整迁移命令序列",
        "service": {
            "name": service.name,
            "model_path": service.model_path,
            "deploy_type": service.deploy_type.as_str(),
            "port": service.port,
            "tp": service.tp,
            "extra_args": service.extra_args,
            "docker_image": service.docker_image,
            "python_cmd": service.python_cmd,
        },
        "preferred_target_host": hint_target_host,
        "candidates": fleet,
    });
    format!(
        "{}{}",
        serde_json::to_string_pretty(&data).unwrap_or_default(),
        json_hints()
    )
}

/// 兼容 LLM 可能把 JSON 放在 answer 字段里的情况。
pub fn unwrap_llm_json(v: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.get("answer").and_then(|a| a.as_str()) {
        if let Some(inner) = extract_last_json(obj) {
            return inner;
        }
    }
    v.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_with_chatter() {
        let text = "dsh: reasoning:\n先分析各机器空闲情况……\n\n{\"target_host\": \"192.0.2.10\", \"gpus\": [0], \"safe\": true}\n以上是我给出的建议。";
        let v = extract_last_json(text).unwrap();
        assert_eq!(v["target_host"], "192.0.2.10");
        assert_eq!(v["safe"], true);
    }

    #[test]
    fn extract_nested_json() {
        let text = "结果：\n{\"result\": {\"a\": 1}, \"note\": \"含 { 花括号\"}\n";
        let v = extract_last_json(text).unwrap();
        assert_eq!(v["result"]["a"], 1);
    }

    #[test]
    fn extract_no_json() {
        assert!(extract_last_json("no json here").is_none());
    }
}
