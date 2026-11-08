//! 规则引擎：LLM 不可用 / 失败时的确定性迁移目标选择。
//! 打分 = 空闲显存 + 空闲卡数权重 + 空闲内存，惩罚源机器与端口占用。

use crate::config::{ServerConf, ServerStatus, ServiceConf};
use crate::deploy::{model_exists, pick_gpus};

#[derive(Debug, Clone, serde::Serialize)]
pub struct TargetPlan {
    pub server: ServerConf,
    pub gpus: Vec<u32>,
    pub score: f64,
    pub model_local: bool,
    pub free_vram_gb: f64,
    pub free_ram_gb: u32,
    pub reason: String,
}

fn free_vram_gb(status: &ServerStatus) -> f64 {
    status
        .gpus
        .iter()
        .map(|g| (g.total_mib.saturating_sub(g.used_mib)) as f64 / 1024.0)
        .sum()
}

/// 在候选（在线且启用）服务器中为服务挑选最佳目标。
pub fn rule_select_target(
    service: &ServiceConf,
    candidates: &[(ServerConf, ServerStatus)],
    prefer_host: Option<&str>,
) -> Option<TargetPlan> {
    let mut scored: Vec<TargetPlan> = Vec::new();
    for (s, st) in candidates {
        if !st.online || !s.enabled {
            continue;
        }
        // 卡数与显存基本满足
        let gpus = pick_gpus(st, service.tp.max(1), &[])?;
        let local = model_exists(st, &service.model_path);
        let fv = free_vram_gb(st);
        let need_gb = service.tp.max(1) as f64 * 4.0; // 粗略：每卡至少留 4GB
        if fv < need_gb {
            continue;
        }
        let idle_cards = st.gpus.iter().filter(|g| g.util_pct == 0 && g.used_mib < 512).count();
        let mut score = fv
            + idle_cards as f64 * 6.0
            + st.ram_total_gb.saturating_sub(st.ram_used_gb) as f64 * 0.05;
        if s.id == service.server_id {
            score -= 500.0; // 不迁回源机器（除非别无选择）
        }
        if !local {
            score -= 1.0; // 模型不在本地，需要拷贝
        }
        if let Some(h) = prefer_host {
            if s.host == h {
                score += 10_000.0; // 用户指定目标优先
            }
        }
        let mut reason = format!(
            "空闲显存 {fv:.0}GB / 空闲卡 {idle_cards} 张 / 空闲内存 {}GB",
            st.ram_total_gb.saturating_sub(st.ram_used_gb)
        );
        if !local {
            reason.push_str("；模型需先拷贝");
        }
        scored.push(TargetPlan {
            server: s.clone(),
            gpus,
            score,
            model_local: local,
            free_vram_gb: fv,
            free_ram_gb: st.ram_total_gb.saturating_sub(st.ram_used_gb),
            reason,
        });
    }
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().next()
}
