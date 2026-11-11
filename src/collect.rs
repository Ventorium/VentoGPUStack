//! 远程机群信息采集：远程 bash 脚本 + 分节解析（逻辑移植自 collect.js）。
//! 脚本按 `@@SECTION@@` 分节输出，解析器把每节映射为 `ServerStatus` 字段。

use std::time::Duration;

use crate::config::{GpuStat, ModelInfo, ProcStat, ServerStatus};
use crate::ssh::ssh_exec;

/// 单台服务器采集入口。失败时返回 online=false 的占位状态（保留错误原因）。
pub async fn collect_one(
    host: &str,
    port: u16,
    user: &str,
    password: &str,
    timeout: Duration,
) -> ServerStatus {
    let cmd = format!("bash -s <<'VGS_SCRIPT'\n{}\nVGS_SCRIPT", COLLECT_SCRIPT);
    match ssh_exec(host, port, user, password, &cmd, timeout).await {
        Ok(r) if r.exit_code == 0 => parse_sections(&r.stdout),
        Ok(r) => ServerStatus::offline(
            0,
            format!("采集脚本退出码 {}", r.exit_code),
        ),
        Err(e) => ServerStatus::offline(0, e),
    }
}

pub(crate) const COLLECT_SCRIPT: &str = r#"
echo "@@HOST@@"
hostname 2>/dev/null
echo "@@CPU@@"
grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//'
echo "@@CORES@@"
nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo
echo "@@RAM@@"
awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.0f %.0f\n",(t-a)/1048576,t/1048576}' /proc/meminfo
echo "@@OS@@"
. /etc/os-release 2>/dev/null && printf '%s\n' "$PRETTY_NAME"
echo "@@DISK@@"
df -B1 --output=size,used / 2>/dev/null | tail -1 | awk '{printf "%.0f %.0f\n", $1/1073741824, $2/1073741824}'
echo "@@GPUDEV@@"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || true
elif command -v lspci >/dev/null 2>&1; then
  lspci 2>/dev/null | grep -i 'vga compatible controller\|3d controller' | grep -ci nvidia || true
else
  echo 0
fi
echo "@@NVSMI@@"
if command -v nvidia-smi >/dev/null 2>&1; then
  smi_line=$(nvidia-smi 2>/dev/null | awk '/Driver Version/{print; exit}')
  drv=$(printf '%s\n' "$smi_line" | awk '{for(i=1;i<=NF;i++) if($i=="Version:"){print $(i+1); exit}}')
  cuda=$(printf '%s\n' "$smi_line" | awk '{for(i=1;i<=NF;i++) if($i=="CUDA"){print $(i+2); exit}}')
  printf '%s|%s\n' "$drv" "$cuda"
else
  printf '|\n'
fi
echo "@@GPUDATA@@"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{
    gsub(/^[ \t]+|[ \t]+$/,"",$1); gsub(/^[ \t]+|[ \t]+$/,"",$2);
    gsub(/^[ \t]+|[ \t]+$/,"",$3); gsub(/^[ \t]+|[ \t]+$/,"",$4);
    gsub(/^[ \t]+|[ \t]+$/,"",$5); gsub(/^[ \t]+|[ \t]+$/,"",$6);
    gsub(/^[ \t]+|[ \t]+$/,"",$7);
    printf "%s|%s|%s|%s|%s|%s|%s\n",$1,$2,$3,$4,$5,$6,$7
  }'
fi
echo "@@APPS@@"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpm=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F',' '{u=$2;gsub(/^[ \t]+|[ \t]+$/,"",u);print toupper(u)" "$1}')
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v m="$gpm" '
    BEGIN{n=split(m,lines,"\n");for(i=1;i<=n;i++){split(lines[i],kv," ");if(kv[1]!="")idx[kv[1]]=kv[2]}}
    {
      p=$1;gsub(/^[ \t]+|[ \t]+$/,"",p);
      name=$2;
      mem=$3;gsub(/^[ \t]+|[ \t]+$/,"",mem);
      u=$4;gsub(/^[ \t]+|[ \t]+$/,"",u);
      i=(toupper(u) in idx)?idx[toupper(u)]:"-";
      printf "%s|%s|%s|%s\n",p,name,i,mem
    }')
  printf '%s\n' "$apps" | while IFS='|' read -r pid name gi mem; do
    [ -n "$pid" ] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-256 | tr '|' '/')
    pp=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' \t\n')
    printf '%s|%s|%s|%s|%s|%s\n' "$pid" "$name" "$gi" "$mem" "$cmd" "$pp"
  done
fi
echo "@@MODELS@@"
timeout 15 bash -c '
  roots="/workspace/models /workspace /data/models /data /models /root/models /root/.cache/huggingface /opt/models /srv/models /mnt/*/models /home/*/.cache/huggingface /home/*/models"
  for root in $roots; do
    [ -d "$root" ] || continue
    find "$root" -maxdepth 3 -type f \( -name "config.json" -o -name "*.safetensors" -o -name "*.gguf" -o -name "*.onnx" \) 2>/dev/null | head -60 | while IFS= read -r f; do
      dirname "$f"
    done
  done
' | sort -u | while IFS= read -r d; do
  [ -n "$d" ] || continue
  size=$(du -sb "$d" 2>/dev/null | cut -f1)
  kind="dir"
  [ -f "$d/config.json" ] && kind="hf"
  ls "$d"/*.safetensors >/dev/null 2>&1 && kind="safetensors"
  ls "$d"/*.gguf >/dev/null 2>&1 && kind="gguf"
  ls "$d"/*.onnx >/dev/null 2>&1 && kind="onnx"
  printf '%s|%s|%s\n' "$d" "$(awk -v s="${size:-0}" 'BEGIN{printf "%.1f", s/1073741824}')" "$kind"
done
echo "@@TOOLS@@"
bash -lc 'for t in vllm sglang docker python3 rsync curl; do p=$(command -v $t 2>/dev/null) && echo "$t=$p"; done' 2>/dev/null
bash -lc 'ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1' 2>/dev/null | sed 's|^|cuda_home=|'
bash -lc 'ls -d /opt/conda/envs/* /usr/local/conda/envs/* /home/*/miniconda3/envs/* /home/*/anaconda3/envs/* /home/*/conda/envs/* 2>/dev/null' 2>/dev/null | head -20 | sed 's|^|conda_env=|'
"#;

/// 解析 `@@SECTION@@` 分节输出为 `ServerStatus`。
pub fn parse_sections(out: &str) -> ServerStatus {
    let mut st = ServerStatus {
        online: true,
        updated_at: chrono::Utc::now().timestamp(),
        ..Default::default()
    };

    let sections = split_sections(out);
    for (name, body) in sections {
        let lines: Vec<&str> = body
            .lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty())
            .collect();
        let first_line = lines.first().copied().unwrap_or("");
        match name.as_str() {
            "HOST" => st.hostname = first_line.to_string(),
            "CPU" => st.cpu_model = first_line.to_string(),
            "CORES" => st.cpu_cores = parse_u32(first_line),
            "RAM" => {
                if let Some(l) = lines.first().copied() {
                    let mut it = l.split_whitespace();
                    st.ram_used_gb = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                    st.ram_total_gb = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                }
            }
            "OS" => st.os = first_line.to_string(),
            "DISK" => {
                if let Some(l) = lines.first().copied() {
                    let mut it = l.split_whitespace();
                    st.disk_total_gb = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                    st.disk_used_gb = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                }
            }
            "NVSMI" => {
                if let Some(l) = lines.first().copied() {
                    let mut it = l.splitn(2, '|');
                    st.driver_version = it.next().unwrap_or("").to_string();
                    st.cuda_version = it.next().unwrap_or("").to_string();
                }
            }
            "GPUDATA" => {
                st.gpus = lines.iter().copied().filter_map(parse_gpu_line).collect();
            }
            "APPS" => {
                st.procs = lines.iter().copied().filter_map(parse_app_line).collect();
            }
            "MODELS" => {
                st.models = lines
                    .iter()
                    .filter_map(|l| {
                        let mut it = l.splitn(3, '|');
                        let path = it.next()?.trim().to_string();
                        let size_gb = it.next()?.trim().parse().ok()?;
                        let kind = it.next().unwrap_or("dir").trim().to_string();
                        Some(ModelInfo { path, size_gb, kind, in_use_by: Vec::new() })
                    })
                    .collect();
            }
            "TOOLS" => {
                st.tools = lines.iter().map(|l| l.trim().to_string()).collect();
            }
            _ => {}
        }
    }

    // 模型目录去重：嵌套在另一个已发现模型目录内的丢弃
    st.models.sort_by(|a, b| a.path.cmp(&b.path));
    let paths: Vec<String> = st.models.iter().map(|m| m.path.clone()).collect();
    st.models.retain(|m| {
        !paths.iter().any(|p| {
            p != &m.path && m.path.starts_with(p.as_str()) && m.path.len() > p.len()
        })
    });

    // 依据进程命令行标注模型占用
    for p in &st.procs {
        if p.cmdline.is_empty() {
            continue;
        }
        for m in &mut st.models {
            if p.cmdline.contains(&m.path) && !m.in_use_by.contains(&p.name) {
                m.in_use_by.push(p.name.clone());
            }
        }
    }
    st
}

fn split_sections(out: &str) -> Vec<(String, String)> {
    // 第一遍：定位各节标记的行号
    let mut marks: Vec<(usize, String)> = Vec::new();
    for (i, line) in out.lines().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("@@").and_then(|s| s.strip_suffix("@@")) {
            marks.push((i, name.to_string()));
        }
    }
    // 第二遍：按标记切出每节正文
    let mut sections = Vec::new();
    for (k, &(start, ref name)) in marks.iter().enumerate() {
        let end = marks.get(k + 1).map(|m| m.0).unwrap_or(out.lines().count());
        let body = out
            .lines()
            .take(end)
            .skip(start + 1)
            .collect::<Vec<_>>()
            .join("\n");
        sections.push((name.clone(), body));
    }
    sections
}

fn parse_gpu_line(line: &str) -> Option<GpuStat> {
    let mut it = line.splitn(7, '|');
    Some(GpuStat {
        idx: it.next()?.trim().parse().ok()?,
        name: it.next()?.trim().to_string(),
        total_mib: it.next()?.trim().parse().ok()?,
        used_mib: it.next()?.trim().parse().ok()?,
        util_pct: it.next()?.trim().parse().ok()?,
        temp_c: it.next()?.trim().parse().ok()?,
        power_w: it.next()?.trim().parse().ok()?,
    })
}

fn parse_app_line(line: &str) -> Option<ProcStat> {
    let mut it = line.splitn(6, '|');
    let pid = it.next()?.trim().parse().ok()?;
    let name = it.next()?.trim().to_string();
    let gpus = it
        .next()?
        .trim()
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let mem_mib = it.next()?.trim().parse().ok()?;
    let cmdline = it.next().unwrap_or("").to_string();
    let ppid = it.next().map(|s| s.trim().parse().unwrap_or(0)).unwrap_or(0);
    Some(ProcStat { pid, name, gpu_idx: gpus, mem_mib, cmdline, ppid })
}

fn parse_u32(s: &str) -> u32 {
    s.trim().parse().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::parse_sections;

    /// 回归：power.draw 是小数（如 19.80），字段必须是浮点，否则整行被丢弃 → gpus=[]
    #[test]
    fn gpudata_fractional_power_parses() {
        let out = "\
@@HOST@@
node04
@@NVSMI@@
550.144.03|12.4
@@DISK@@
733 155
@@GPUDEV@@
8
@@GPUDATA@@
0|NVIDIA GeForce RTX 3090|24576|2|0|27|19.80
1|NVIDIA GeForce RTX 3090|24576|2|0|27|19.85
@@APPS@@
@@TOOLS@@
docker=/usr/bin/docker
";
        let st = parse_sections(out);
        assert_eq!(st.hostname, "node04");
        assert_eq!(st.driver_version, "550.144.03");
        assert_eq!(st.cuda_version, "12.4");
        assert_eq!(st.disk_used_gb, 155);
        assert_eq!(st.gpus.len(), 2, "GPUDATA 行必须解析成功");
        assert_eq!(st.gpus[0].idx, 0);
        assert_eq!(st.gpus[0].power_w, 19.8);
        assert!(st.procs.is_empty());
        assert_eq!(st.tools, vec!["docker=/usr/bin/docker".to_string()]);
    }
}
