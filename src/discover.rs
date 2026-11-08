//! 智能检测：SSH 深扫机器上正在运行的 GPU 计算进程（完整 cmdline、父进程 cmdline、容器归属），
//! 结合周期采集快照（驱动 / CUDA / 模型目录扫描）识别服务引擎类型
//! （vllm / sglang / triton / xinference / paddlex / funasr / ollama / python / other）、
//! 模型路径、端口、TP，并把多进程（TP worker / spawn 子进程 / 改名子进程）归并为同一服务，
//! 供「注册」纳管为可启停 / 迁移的模型服务。

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde::Serialize;

use crate::config::{ServerConf, ServerStatus};
use crate::ssh::ssh_exec;

/// 深扫脚本：GPU 进程完整 cmdline + 祖先链业务命令 / 容器归属 + docker 容器列表（一次 SSH 往返）。
/// P 行格式：`P|pid|name|gi|mem|cid|ppid|cmd|pcmd|rootcmd|rootpid`
/// （cmd / pcmd / rootcmd 内的 '|' 已在 shell 侧替换为 '/'；rootcmd = 祖先链上最近的真实业务命令，rootpid 为其 pid）。
const DISCOVER_SCRIPT: &str = r#"
echo "@@PROCDEEP@@"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpm=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F',' '{u=$2;gsub(/^[ \t]+|[ \t]+$/,"",u);print toupper(u)" "$1}')
  nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v m="$gpm" '
    BEGIN{n=split(m,lines,"\n");for(i=1;i<=n;i++){split(lines[i],kv," ");if(kv[1]!="")idx[kv[1]]=kv[2]}}
    {
      p=$1;gsub(/^[ \t]+|[ \t]+$/,"",p);
      nm=$2;gsub(/[][|]/,"_",nm);
      mem=$3;gsub(/^[ \t]+|[ \t]+$/,"",mem);
      u=$4;gsub(/^[ \t]+|[ \t]+$/,"",u);
      i=(toupper(u) in idx)?idx[toupper(u)]:"-";
      printf "%s|%s|%s|%s\n",p,nm,i,mem
    }' | while IFS='|' read -r pid name gi mem; do
    [ -n "$pid" ] || continue
    case "$pid" in *[!0-9]*) continue;; esac
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed 's/|/\//g' | cut -c1-1024)
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' \t\n')
    cid=""; pcmd=""; rootcmd=""; rootpid=""
    cur="$ppid"; depth=0
    # 向上遍历祖先链（最多 6 层）：找容器 shim 与最近的业务命令
    # （vLLM Worker_TP0 -> EngineCore -> vllm serve ...；容器内进程 -> containerd-shim）
    while [ -n "$cur" ] && [ "$cur" != "1" ] && [ "$depth" -lt 6 ]; do
      acmd=$(tr '\0' ' ' < "/proc/$cur/cmdline" 2>/dev/null | sed 's/|/\//g' | cut -c1-512)
      [ -n "$pcmd" ] || pcmd="$acmd"
      if [ -z "$cid" ] && [ -n "$acmd" ]; then
        case "$acmd" in
          *containerd-shim*)
            cid=$(printf '%s\n' "$acmd" | sed -n 's/.*-id \([0-9a-f]\{12,\}\).*/\1/p' | head -1)
            ;;
        esac
      fi
      if [ -z "$rootcmd" ] && [ -n "$acmd" ] && [ "${#acmd}" -ge 15 ]; then
        case "$acmd" in
          *spawn_main*|*forkserver*|*"::"*|*containerd*|*sshd*|*systemd*|-bash*|-sh*|bash*|sh*|tmux*|screen*|cron*|sleep*|"docker exec "*|"docker attach "*|nerdctl*|podman*) : ;;
          *--*|*/*) rootcmd="$acmd"; rootpid="$cur" ;;
        esac
      fi
      cur=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' \t\n')
      depth=$((depth + 1))
    done
    printf 'P|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "$pid" "$name" "$gi" "$mem" "$cid" "$ppid" "$cmd" "$pcmd" "$rootcmd" "$rootpid"
  done
fi
echo "@@DOCKER@@"
if command -v docker >/dev/null 2>&1; then
  # 一次 docker ps 同时取 短id/容器名/镜像，避免逐容器 inspect 慢调用拖超时
  docker ps --format '{{.ID}} {{.Names}} {{.Image}}' 2>/dev/null | head -40 | while read -r cid nm image; do
    [ -n "$cid" ] || continue
    printf 'D|%s|%s|%s\n' "$cid" "$nm" "$image"
  done
fi
echo "@@END@@"
"#;

/// GPU 计算进程（深扫原始行，多卡行在 detect 中按 pid 合并）。
#[derive(Debug, Clone, Default, Serialize)]
pub struct DeepProc {
    pub pid: u32,
    pub name: String,
    pub gpus: Vec<u32>,
    pub mem_mib: u32,
    /// 归属容器 id（空 = 宿主机进程）
    pub cid: String,
    pub ppid: u32,
    pub cmdline: String,
    /// 父进程 cmdline（截断 512 字符）
    pub parent_cmdline: String,
    /// 祖先链上最近的「真实业务命令」cmdline（setproctitle 改写场景用，如 vLLM Worker → vllm serve）
    pub root_cmdline: String,
    /// root_cmdline 对应的祖先 pid（0 = 未找到）
    pub root_pid: u32,
}

/// 运行中的 docker 容器。
#[derive(Debug, Clone, Default, Serialize)]
pub struct Container {
    pub cid: String,
    pub name: String,
    pub image: String,
}

/// 一台机器的深扫结果。
#[derive(Debug, Clone, Default)]
pub struct DiscoverResult {
    pub procs: Vec<DeepProc>,
    pub containers: Vec<Container>,
}

/// 对一台机器执行一次深扫（一次 SSH 往返）。
pub async fn discover_one(
    host: &str,
    port: u16,
    user: &str,
    password: &str,
    timeout: Duration,
) -> Result<DiscoverResult, String> {
    let r = ssh_exec(host, port, user, password, DISCOVER_SCRIPT, timeout)
        .await
        .map_err(|e| e.to_string())?;
    if !r.stdout.contains("@@PROCDEEP@@") {
        return Err(format!("SSH 退出码 {} 且无可用输出", r.exit_code));
    }
    let mut out = DiscoverResult::default();
    let mut cur = "";
    for line in r.stdout.lines() {
        let l = line.trim_end();
        if l == "@@PROCDEEP@@" {
            cur = "proc";
            continue;
        }
        if l == "@@DOCKER@@" {
            cur = "docker";
            continue;
        }
        if l == "@@END@@" {
            cur = "";
            continue;
        }
        match cur {
            "proc" => {
                let Some(rest) = l.strip_prefix("P|") else {
                    continue;
                };
                // pid|name|gi|mem|cid|ppid|cmd|pcmd|rootcmd|rootpid（cmd 字段内 '|' 已在 shell 侧替换为 '/'）
                let parts: Vec<&str> = rest.splitn(10, '|').collect();
                if parts.len() < 8 {
                    continue;
                }
                let pid: u32 = parts[0].parse().unwrap_or(0);
                if pid == 0 {
                    continue;
                }
                let gi: u32 = parts[2].parse().unwrap_or(u32::MAX);
                let mem: u32 = parts[3].parse().unwrap_or(0);
                let ppid: u32 = parts[5].parse().unwrap_or(0);
                let root_cmdline = parts.get(8).copied().unwrap_or("").to_string();
                let root_pid: u32 = parts.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);
                out.procs.push(DeepProc {
                    pid,
                    name: parts[1].to_string(),
                    gpus: if gi == u32::MAX {
                        Vec::new()
                    } else {
                        vec![gi]
                    },
                    mem_mib: mem,
                    cid: parts[4].to_string(),
                    ppid,
                    cmdline: parts[6].to_string(),
                    parent_cmdline: parts[7].to_string(),
                    root_cmdline,
                    root_pid: root_pid,
                });
            }
            "docker" => {
                let Some(rest) = l.strip_prefix("D|") else {
                    continue;
                };
                let parts: Vec<&str> = rest.splitn(3, '|').collect();
                if parts.len() < 3 {
                    continue;
                }
                out.containers.push(Container {
                    cid: parts[0].to_string(),
                    name: parts[1].to_string(),
                    image: parts[2].to_string(),
                });
            }
            _ => {}
        }
    }
    Ok(out)
}

/// 检测出的模型服务候选（多进程已归并为同一服务）。
#[derive(Debug, Clone, Serialize)]
pub struct Detected {
    pub server_id: u64,
    pub server_name: String,
    pub host: String,
    pub driver_version: String,
    pub cuda_version: String,
    /// vllm | sglang | triton | xinference | paddlex | funasr | ollama | python | other
    pub engine: String,
    /// 服务主 pid（worker 组取主进程 / 父进程 pid）
    pub pid: u32,
    pub proc_name: String,
    pub gpus: Vec<u32>,
    pub mem_mib: u32,
    /// 代表启动命令（worker 组取父进程命令）
    pub cmdline: String,
    pub model_path: String,
    /// 模型名（进程标题解析 / 路径尾段），路径未知时用于展示
    pub model_name: String,
    /// 模型是否命中本机模型目录扫描（可给出体积 / 类型）
    pub model_known: bool,
    pub model_size_gb: f64,
    pub model_kind: String,
    /// 0 = 未识别
    pub port: u16,
    /// 0 = 未识别
    pub tp: u32,
    /// 启动命令中除标准参数外的剩余参数（vllm/sglang 重启用）
    pub extra_args: String,
    pub container_name: String,
    pub container_image: String,
    /// 是否已存在于注册表（同机同模型 或 同 pid）
    pub registered: bool,
}

fn tokens_of(cmdline: &str) -> Vec<String> {
    cmdline.split_whitespace().map(|s| s.to_string()).collect()
}

fn base_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// 取 `--flag value` 或 `--flag=value` 形式的参数值。
fn flag_value(tokens: &[String], flags: &[&str]) -> Option<String> {
    for (i, t) in tokens.iter().enumerate() {
        for f in flags {
            if t == f {
                if let Some(v) = tokens.get(i + 1) {
                    if !v.starts_with("--") {
                        return Some(v.clone());
                    }
                }
            }
            let pre = format!("{f}=");
            if let Some(v) = t.strip_prefix(&pre) {
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

fn is_known_model(v: &str, known: &[String]) -> bool {
    known
        .iter()
        .any(|k| k == v || k.starts_with(&format!("{v}/")) || v.starts_with(&format!("{k}/")))
}

/// 是否为 multiprocessing spawn/forkserver 子进程（cmdline 只有样板、无业务信息）。
pub fn is_worker_cmdline(s: &str) -> bool {
    s.contains("spawn_main") || s.contains("forkserver")
}

/// 进程是否属于「worker」：spawn/forkserver 子进程，或被改名成 `XXX::yyy` 的引擎子进程。
fn is_worker_proc(p: &DeepProc) -> bool {
    is_worker_cmdline(&p.cmdline) || p.name.contains("::")
}

/// 进程的有效命令：worker / 被改名标题进程优先用祖先链找到的业务命令，其次父进程 cmdline。
fn eff_cmd(p: &DeepProc) -> &str {
    if !p.cmdline.is_empty()
        && !p.cmdline.contains("::")
        && !is_worker_cmdline(&p.cmdline)
    {
        return &p.cmdline;
    }
    if !p.root_cmdline.is_empty() {
        return &p.root_cmdline;
    }
    if !p.parent_cmdline.is_empty() {
        return &p.parent_cmdline;
    }
    &p.cmdline
}

/// 从 cmdline 提取模型路径：
/// 1) 常见 --model* 旗标；2) `serve <path>` 位置参数；
/// 3) 与本机模型目录扫描结果交叉比对（cmdline 中出现已知路径）；
/// 4) 启发式：路径段含 "model" 的绝对路径。
pub fn extract_model_path(cmdline: &str, known: &[String]) -> (String, bool) {
    let tokens = tokens_of(cmdline);
    const FLAGS: [&str; 9] = [
        "--model-path",
        "--model_path",
        "--model-repository",
        "--model-name-or-path",
        "--model_name_or_path",
        "--pretrained_model_name_or_path",
        "--checkpoint-path",
        "--pack",
        "--model",
    ];
    if let Some(v) = flag_value(&tokens, &FLAGS) {
        if v.starts_with('/') && v.len() > 2 {
            let hit = is_known_model(&v, known);
            return (v, hit);
        }
    }
    if let Some(i) = tokens.iter().position(|t| t == "serve") {
        if let Some(v) = tokens.get(i + 1) {
            if v.starts_with('/') && v.len() > 2 {
                let hit = is_known_model(v, known);
                return (v.clone(), hit);
            }
        }
    }
    let mut best = String::new();
    for k in known {
        if cmdline.contains(k.as_str()) && k.len() > best.len() {
            best = k.clone();
        }
    }
    if !best.is_empty() {
        return (best, true);
    }
    for t in &tokens {
        if t.starts_with('/') && t.len() > 3 {
            let low = t.to_lowercase();
            if low.contains("model")
                && !low.starts_with("/proc")
                && !low.starts_with("/dev")
                && !low.contains(".json")
            {
                let hit = is_known_model(t, known);
                return (t.clone(), hit);
            }
        }
    }
    (String::new(), false)
}

/// 引擎分类（按进程名 + 有效命令）。
pub fn classify(name: &str, cmdline: &str) -> &'static str {
    let c = cmdline.to_lowercase();
    let n = name.to_lowercase();
    if n.starts_with("vllm")
        || c.contains("vllm.entrypoints")
        || c.contains("vllm serve")
        || n.contains("vllm::")
    {
        "vllm"
    } else if c.contains("sglang") || n.contains("sglang::") || c.contains("sgl_diffusion") {
        "sglang"
    } else if n.contains("tritonserver") || c.contains("tritonserver") || c.contains("triton_server") {
        "triton"
    } else if c.contains("xinference") {
        "xinference"
    } else if c.contains("paddlex") {
        "paddlex"
    } else if c.contains("funasr") {
        "funasr"
    } else if c.contains("ollama") {
        "ollama"
    } else if n.starts_with("python") || n == "swift" || c.contains("ms-swift") {
        "python"
    } else {
        "other"
    }
}

/// 引擎排序权重：推理引擎优先，其次 python，最后 other。
fn engine_rank(e: &str) -> u8 {
    match e {
        "vllm" | "sglang" | "triton" | "xinference" | "paddlex" | "funasr" | "ollama" => 0,
        "python" => 1,
        _ => 2,
    }
}

/// vllm：剔除二进制/serve/模型/标准旗标后的剩余参数。
fn vllm_extra(tokens: &[String], model: &str) -> String {
    const DROP: [&str; 5] = [
        "--served-model-name",
        "--port",
        "--tensor-parallel-size",
        "--tp",
        "--model",
    ];
    let mut out = Vec::new();
    let mut i = 1;
    while i < tokens.len() {
        let t = &tokens[i];
        if t == "serve" || (t == model && !model.is_empty()) {
            i += 1;
            continue;
        }
        if DROP.contains(&t.as_str()) {
            i += 2;
            continue;
        }
        out.push(t.clone());
        i += 1;
    }
    out.join(" ")
}

/// sglang：剔除解释器/-m 模块/标准旗标后的剩余参数。
fn sglang_extra(tokens: &[String], model: &str) -> String {
    const DROP: [&str; 5] = ["--model-path", "--model_path", "--port", "--tp", "--model"];
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if i == 0 {
            i += 1;
            continue;
        }
        if DROP.contains(&t.as_str()) {
            i += 2;
            continue;
        }
        if t == "-m" {
            i += 2;
            continue;
        }
        if t == model && !model.is_empty() {
            i += 1;
            continue;
        }
        out.push(t.clone());
        i += 1;
    }
    out.join(" ")
}

/// 进程标题形如 `Model: <name>-<idx>`（setproctitle 改名）时取出模型名。
fn parse_model_title(name: &str) -> Option<String> {
    let rest = name.strip_prefix("Model:")?.trim();
    let rest = rest.split(" -").next().unwrap_or(rest);
    let base = rest
        .rsplit_once('-')
        .filter(|(_, d)| d.chars().all(|c| c.is_ascii_digit()))
        .map(|(b, _)| b)
        .unwrap_or(rest)
        .trim();
    if base.is_empty() {
        None
    } else {
        Some(base.to_string())
    }
}

// ---------------------------------------------------------------------------
// 并查集（字符串键，组规模很小）
// ---------------------------------------------------------------------------

struct UF {
    parent: HashMap<String, String>,
}

impl UF {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }
    fn find(&mut self, start: &str) -> String {
        let mut cur = start.to_string();
        loop {
            match self.parent.get(&cur).cloned() {
                Some(p) if p != cur => cur = p,
                _ => return cur,
            }
        }
    }
    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(rb, ra);
        }
    }
}

/// 结合深扫结果与采集快照，产出一台机器的服务候选列表（多进程已归并为同一服务）。
/// `st` 可为 None（快照缺失时仅按深扫结果识别，模型 / 驱动信息为空）。
pub fn detect(
    server: &ServerConf,
    st: Option<&ServerStatus>,
    dr: &DiscoverResult,
    existing: &[crate::config::ServiceConf],
) -> Vec<Detected> {
    let known: Vec<String> = st
        .map(|s| s.models.iter().map(|m| m.path.clone()).collect())
        .unwrap_or_default();
    let driver_version = st.map(|s| s.driver_version.clone()).unwrap_or_default();
    let cuda_version = st.map(|s| s.cuda_version.clone()).unwrap_or_default();
    let cont: HashMap<&str, &Container> = dr.containers.iter().map(|c| (c.cid.as_str(), c)).collect();

    // 同一 pid 的多卡行合并
    let mut procs: Vec<DeepProc> = Vec::new();
    for p in &dr.procs {
        match procs.iter_mut().find(|x| x.pid == p.pid) {
            Some(e) => {
                for g in &p.gpus {
                    if !e.gpus.contains(g) {
                        e.gpus.push(*g);
                    }
                }
                e.mem_mib += p.mem_mib;
            }
            None => procs.push(p.clone()),
        }
    }
    if procs.is_empty() {
        return Vec::new();
    }
    for p in &mut procs {
        p.gpus.sort();
        p.gpus.dedup();
    }
    procs.sort_by_key(|p| p.pid);
    let pid_set: HashSet<u32> = procs.iter().map(|p| p.pid).collect();
    let key = |pid: u32| format!("p{pid}");

    // 1) worker 归并到父进程（父不在 GPU 进程集时按 ppid 值归并，同一父 = 同一服务）
    let mut uf = UF::new();
    for p in &procs {
        if is_worker_proc(p) && p.ppid != 0 && p.ppid != 1 {
            if pid_set.contains(&p.ppid) {
                uf.union(&key(p.pid), &key(p.ppid));
            } else {
                uf.union(&key(p.pid), &format!("off{}", p.ppid));
            }
        }
    }
    // 2) 同机 cmdline 完全相同的非 worker 进程 → 同一服务的多 worker（如 paddlex 双卡）
    for i in 0..procs.len() {
        for j in (i + 1)..procs.len() {
            let (a, b) = (&procs[i], &procs[j]);
            if !is_worker_proc(a) && !is_worker_proc(b) && !a.cmdline.is_empty() && a.cmdline == b.cmdline {
                uf.union(&key(a.pid), &key(b.pid));
            }
        }
    }
    // 3) 按归并后组提取模型，同模型的组再合并（如 launcher + 改名 scheduler）
    let build_groups = |uf: &mut UF| -> HashMap<String, Vec<usize>> {
        let mut g: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, p) in procs.iter().enumerate() {
            let root = uf.find(&key(p.pid));
            g.entry(root).or_default().push(i);
        }
        g
    };
    let mut changed = true;
    while changed {
        changed = false;
        let groups = build_groups(&mut uf);
        let mut model_of: HashMap<String, String> = HashMap::new();
        for (root, idxs) in &groups {
            let mut best = String::new();
            for &i in idxs {
                let (mdl, _) = extract_model_path(eff_cmd(&procs[i]), &known);
                if mdl.len() > best.len() {
                    best = mdl;
                }
            }
            model_of.insert(root.clone(), best);
        }
        let roots: Vec<String> = groups.keys().cloned().collect();
        for a in 0..roots.len() {
            for b in (a + 1)..roots.len() {
                let (ma, mb) = (&model_of[&roots[a]], &model_of[&roots[b]]);
                if !ma.is_empty() && ma == mb {
                    uf.union(&roots[a], &roots[b]);
                    changed = true;
                }
            }
        }
    }
    let groups = build_groups(&mut uf);

    // 4) 逐组产出 Detected
    let mut out: Vec<Detected> = Vec::new();
    for (_root, idxs) in &groups {
        let members: Vec<&DeepProc> = idxs.iter().map(|&i| &procs[i]).collect();

        // 代表命令：优先非 worker 自身 cmdline（取最长）；全 worker 组则取父进程 cmdline
        let mut rep_cmd: Option<String> = None;
        let mut worker_cmd: Option<String> = None;
        for m in &members {
            let c = eff_cmd(m);
            if c.is_empty() {
                continue;
            }
            let slot = if is_worker_proc(m) { &mut worker_cmd } else { &mut rep_cmd };
            match slot {
                Some(w) if w.len() >= c.len() => {}
                _ => *slot = Some(c.to_string()),
            }
        }
        let mut rep_cmd = rep_cmd.or(worker_cmd).unwrap_or_default();
        if rep_cmd.is_empty() {
            for m in &members {
                if !m.cmdline.is_empty() {
                    rep_cmd = m.cmdline.clone();
                    break;
                }
            }
        }
        let toks = tokens_of(&rep_cmd);
        let mut proc_name = toks
            .first()
            .map(|t| base_name(t).to_string())
            .filter(|s| !s.is_empty() && s != "Model:")
            .unwrap_or_default();
        if proc_name.is_empty() {
            proc_name = members[0].name.clone();
        }
        if let Some(m) = members.iter().find(|m| m.name.starts_with("Model:")) {
            proc_name = m.name.clone();
        }
        let engine = classify(&proc_name, &rep_cmd).to_string();
        let (model, model_known) = extract_model_path(&rep_cmd, &known);
        let port = flag_value(&toks, &["--port", "--http-port"])
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(0);
        let tp = flag_value(&toks, &["--tensor-parallel-size", "--tensor-parallel", "--tp", "--tp-size", "-tp"])
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let gpus: Vec<u32> = members
            .iter()
            .flat_map(|m| m.gpus.iter().copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mem: u32 = members.iter().map(|m| m.mem_mib).sum();
        let pids_in: HashSet<u32> = members.iter().map(|m| m.pid).collect();

        // 服务主 pid：组内「是其他成员之父」的根进程；否则全体 worker 的共同父 pid；
        // 否则最小 pid 的根；否则代表进程。
        let roots_in: Vec<&DeepProc> = members
            .iter()
            .copied()
            .filter(|m| m.ppid == 0 || !pids_in.contains(&m.ppid))
            .collect();
        let all_worker_same_parent = members.iter().all(|m| {
            is_worker_proc(m) && m.ppid != 0 && m.ppid != 1
        }) && members.iter().map(|m| m.ppid).collect::<HashSet<_>>().len() == 1;
        // 全 worker 组共享同一业务祖先进程（祖先链上溯找到，如 vllm 主进程）→ 服务真实主 pid
        let all_worker = members.iter().all(|m| is_worker_proc(m));
        let shared_root = if all_worker {
            let rps: HashSet<u32> = members
                .iter()
                .map(|m| m.root_pid)
                .filter(|p| *p != 0)
                .collect();
            if rps.len() == 1 {
                *rps.iter().next().unwrap()
            } else {
                0
            }
        } else {
            0
        };
        let pid = if let Some(r) = roots_in
            .iter()
            .find(|r| members.iter().any(|m| m.pid != r.pid && m.ppid == r.pid))
        {
            r.pid
        } else if shared_root != 0 {
            shared_root
        } else if all_worker_same_parent {
            members[0].ppid
        } else if let Some(r) = roots_in.first() {
            r.pid
        } else {
            members
                .iter()
                .max_by_key(|m| m.cmdline.len())
                .map(|m| m.pid)
                .unwrap_or(0)
        };

        let (cname, cimage) = members
            .iter()
            .find(|m| !m.cid.is_empty())
            .and_then(|m| {
                // P 行是 64 位全量 cid，D 行是 12 位短 id：取前缀匹配
                let short = m.cid.get(..12).unwrap_or(&m.cid);
                cont.get(short)
            })
            .map(|c| (c.name.clone(), c.image.clone()))
            .unwrap_or_default();

        let mut model_name = String::new();
        for m in &members {
            if let Some(nm) = parse_model_title(&m.name) {
                model_name = nm;
                break;
            }
        }
        if model_name.is_empty() && !model.is_empty() {
            model_name = base_name(&model).to_string();
        }

        let (size, kind) = st
            .map(|s| {
                s.models
                    .iter()
                    .find(|m| {
                        !model.is_empty()
                            && (m.path == model
                                || model.starts_with(&m.path)
                                || m.path.starts_with(&model))
                    })
                    .map(|m| (m.size_gb, m.kind.clone()))
            })
            .flatten()
            .unwrap_or((0.0, String::new()));

        let registered = existing.iter().any(|s| {
            s.server_id == server.id
                && ((!model.is_empty()
                    && !s.model_path.is_empty()
                    && (s.model_path == model
                        || s.model_path.starts_with(&model)
                        || model.starts_with(&s.model_path)))
                    || (s.pid.is_some() && (pids_in.contains(&s.pid.unwrap()) || s.pid == Some(pid))))
        });

        let extra_args = match engine.as_str() {
            "vllm" => vllm_extra(&toks, &model),
            "sglang" => sglang_extra(&toks, &model),
            _ => String::new(),
        };

        out.push(Detected {
            server_id: server.id,
            server_name: server.name.clone(),
            host: server.host.clone(),
            driver_version: driver_version.clone(),
            cuda_version: cuda_version.clone(),
            engine,
            pid,
            proc_name,
            gpus,
            mem_mib: mem,
            cmdline: rep_cmd,
            model_path: model,
            model_name,
            model_known,
            model_size_gb: size,
            model_kind: kind,
            port,
            tp,
            extra_args,
            container_name: cname,
            container_image: cimage,
            registered,
        });
    }
    out.sort_by(|a, b| {
        engine_rank(&a.engine)
            .cmp(&engine_rank(&b.engine))
            .then(a.server_name.cmp(&b.server_name))
            .then(a.pid.cmp(&b.pid))
    });
    out
}

/// 正则转义（供 pkill -f 使用）。
pub fn regex_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if matches!(
            c,
            '.' | '*' | '[' | ']' | '(' | ')' | '{' | '}' | '+' | '?' | '^' | '$' | '|' | '\\'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}
