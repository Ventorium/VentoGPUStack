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

/// 驱动升级安装后验收 prompt：对照预期驱动版本，检查 nvidia-smi 全量输出。
pub fn driver_acceptance_prompt(expected_driver: &str, smi_lines: &str) -> String {
    let data = serde_json::json!({
        "task": "验收一次刚完成的 NVIDIA 驱动升级是否成功",
        "expected_driver": expected_driver,
        "nvidia_smi_output": smi_lines,
        "check_points": [
            "nvidia-smi 顶部显示的 Driver Version 是否与 expected_driver 一致（或属于同一主版本升级后允许的新版本）",
            "GPU 数量是否齐全（每张卡都应出现在列表中，而非缺失或 N/A）",
            "是否出现 Xid 错误、ECC 异常、温度/功耗 N/A、显存不可读等故障信号",
        ],
        "output_format": {
            "ok": "true 或 false（仅当驱动版本匹配且所有 GPU 健康时为 true）",
            "driver": "nvidia-smi 中实际显示的 Driver Version 字符串",
            "gpus_total": "nvidia-smi 中列出的 GPU 总数（整数）",
            "gpus_ok": "其中状态正常的 GPU 数（整数）",
            "note": "一句中文验收结论，若失败请说明具体原因"
        }
    });
    format!(
        "{}\n要求：只基于上面的 nvidia-smi 输出判断，不要编造。最后只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字），严格使用 output_format 的键名。",
        serde_json::to_string_pretty(&data).unwrap_or_default()
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

// ---------------------------------------------------------------------------
// 驱动兼容检查 / Agent 操作计划 prompt
// ---------------------------------------------------------------------------

/// 驱动升级兼容性检查 prompt：结合知识库与机器在跑负载，判定目标驱动是否安全。
pub fn compat_check_prompt(
    server: &ServerConf,
    st: &ServerStatus,
    target_driver: &str,
    kb: &str,
) -> String {
    let data = serde_json::json!({
        "task": "判断把该机的 NVIDIA 驱动升级到 target_driver 是否安全（核心依据：在跑负载所需的最低驱动版本 vs 目标驱动）",
        "server": {
            "name": server.name,
            "host": server.host,
            "current_driver": st.driver_version,
            "current_cuda": st.cuda_version,
        },
        "registered_services": st.models.iter().map(|m| serde_json::json!({
            "path": m.path,
            "in_use_by": m.in_use_by,
        })).collect::<Vec<_>>(),
        "running_gpu_processes": st.procs.iter().map(|p| serde_json::json!({
            "name": p.name,
            "cmdline": p.cmdline.chars().take(200).collect::<String>(),
        })).collect::<Vec<_>>(),
        "target_driver": target_driver,
        "knowledge_base": kb,
        "output_format": {
            "verdict": "safe / warn / block 三选一：block=目标驱动低于某在跑负载所需最低驱动（升级后该负载必挂）；safe=满足所有最低驱动要求；warn=理论兼容但超出知识库已验证范围或存在未纳管 GPU 进程需人工确认",
            "max_compatible": "知识库中该机在跑负载所支持的『已验证最高驱动版本』（取 observed_fleet 中 CUDA 满足要求且分支最高的驱动；无依据则填知识库 branch_cuda 中最高驱动段）",
            "required_cuda": "在跑负载中最高的 CUDA 版本要求（如 12.9；docker 镜像按 tag cu12x 判断；无法判断时按 current_cuda 保守处理）",
            "min_required_driver": "required_cuda 对应的最低驱动版本（来自知识库 cuda_min_driver）",
            "reasons": ["中文要点数组：说明判定依据，如某负载需要 CUDA x.y 最低驱动 z.z"]
        }
    });
    format!(
        "{}{}",
        serde_json::to_string_pretty(&data).unwrap_or_default(),
        "要求：1. 只依据输入的知识库与负载信息判断，不要编造版本。2. 目标驱动低于 min_required_driver 必须给 block；高于知识库已收录的最高分支给 warn；其余给 safe。3. 最后只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字），严格使用 output_format 的键名。\n"
    )
}

/// Agent 操作计划 prompt：把用户的自然语言运维诉求转成受限动作序列。
pub fn agent_plan_prompt(
    server: &ServerConf,
    st: &ServerStatus,
    services: &[&ServiceConf],
    prompt: &str,
    kb: &str,
) -> String {
    let svcs: Vec<serde_json::Value> = services
        .iter()
        .map(|sv| {
            serde_json::json!({
                "id": sv.id,
                "name": sv.name,
                "model_path": sv.model_path,
                "port": sv.port,
                "deploy_type": sv.deploy_type.as_str(),
                "docker_image": sv.docker_image,
                "docker_args": sv.docker_args,
                "status": sv.status,
                "pid": sv.pid,
                "container_name": sv.container_name,
                "stop_pattern": sv.stop_pattern,
            })
        })
        .collect();
    let data = serde_json::json!({
        "task": "为用户的运维诉求生成在该机器上执行的操作计划（步骤化、可逐步执行）",
        "server": {
            "id": server.id,
            "name": server.name,
            "host": server.host,
            "current_driver": st.driver_version,
            "current_cuda": st.cuda_version,
            "gpus": st.gpus.len(),
        },
        "registered_services": svcs,
        "running_gpu_processes": st.procs.iter().map(|p| serde_json::json!({
            "pid": p.pid,
            "name": p.name,
            "gpus": p.gpu_idx,
            "cmdline": p.cmdline.chars().take(160).collect::<String>(),
        })).collect::<Vec<_>>(),
        "driver_kb_summary": kb,
        "user_request": prompt,
        "actions_vocabulary": {
            "stop_service": {"service_id": "整数，必须是 registered_services 中的 id", "description": "中文说明"},
            "start_service": {"service_id": "整数，必须是 registered_services 中的 id", "description": "中文说明"},
            "update_service": {"service_id": "整数", "model_path": "要替换成的新模型路径", "description": "中文说明（热切换模型后由后续 start 生效）"},
            "upgrade_driver": {"target_driver": "三段式版本号 如 590.48.01", "auto_reboot": "true/false", "description": "中文说明"},
            "wait": {"seconds": "整数，1..=600", "description": "中文说明"}
        },
        "output_format": {
            "summary": "一句话中文说明整体思路",
            "steps": [ {"action": "动作名", "params": "对应动作的参数对象", "description": "中文说明"} ]
        }
    });
    format!(
        "{}{}",
        serde_json::to_string_pretty(&data).unwrap_or_default(),
        "要求：1. steps 不超过 10 步；每步必须是 actions_vocabulary 中的动作之一，service_id 必须真实存在，不得编造。2. 涉及驱动升级时必须先 stop_service 停掉该机的 GPU 服务，升级完成后再 start_service 恢复；无法安全完成的诉求，只输出无法执行的说明，不要硬造步骤。3. 最后只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字），严格使用 output_format 的键名。\n"
    )
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
