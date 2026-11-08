# VentoGPUStack

GPU 机房管理网关 —— 单二进制 Rust 服务（Pingora 0.9 + russh 0.65），面向 28 台
RTX 3090 服务器机房的集中管理。

## 功能

| 模块 | 说明 |
|---|---|
| 服务器管理 | UI 增删改、SSH 连通性测试、单台/全量刷新 |
| 机群概览 | GPU 利用率 / 显存 / 内存 / 磁盘 / 驱动分布，30s 周期采集（可配） |
| 驱动升级研判 | 全机群驱动/CUDA 矩阵；LLM 结合在跑负载判断逐台升级安全性（可直升 / 需先停哪些服务），失败自动回退规则引擎 |
| 模型服务 | 注册 vllm / sglang / docker / python 四类服务；启动时自动挑空闲卡并设置 `CUDA_VISIBLE_DEVICES`（TP/DP） |
| 调度迁移 | AI 生成迁移方案（打分：空闲显存 + 空载卡数 + 本地模型），蓝绿执行：目标机先起 → 健康检查 → 停源机 |
| 模型拷贝 | 后台 rsync 源机 → 目标机，不阻塞 UI |
| 审计日志 | 所有写操作 JSONL 落盘，UI 可查看 |
| 热重载 | 主配置 / servers.yaml / services.yaml 修改即生效（mtime 监听） |

LLM 后端：`dsh headless headless "<prompt>"`（DeepSeek Harness headless profile），
可在设置页改命令 / profile / 超时 / 开关。

## 构建

```bash
# 最新 stable 工具链（rust-toolchain.toml 已固定）
source ~/.cargo/env
cargo build --release          # 产物 target/release/ventogpustack
```

## 子命令

```bash
ventogpustack init [--file config/admin.yaml] [--user admin] [--password ...]
ventogpustack run --config config/ventogpustack.yaml
ventogpustack set-password [--file config/admin.yaml] [--password ...]
ventogpustack import --csv gpu_servers.csv --out config/servers.yaml --user <ssh用户> --password '<密码>'
```

- 不带子命令默认 `run`。
- `admin.yaml` 缺失时 `run` 会自动生成并**把随机密码打印到日志**。

## 部署（/opt/ventogpustack）

```bash
sudo mkdir -p /opt/ventogpustack/{bin,logs}
sudo cp target/release/ventogpustack /opt/ventogpustack/bin/
# 复制 config/*.yaml 到 /opt/ventogpustack/ 并按需修改（主配置内路径均指向 /opt/ventogpustack/）
sudo cp deploy/ventogpustack.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ventogpustack
```

浏览器访问 `http://<host>:8900`，首次用日志中打印的临时密码登录，随后
`./target/release/ventogpustack set-password` 修改。

## 配置

| 文件 | 内容 |
|---|---|
| `config/ventogpustack.yaml` | 监听端口、线程、采集间隔/超时/并发、LLM 后端、各文件路径 |
| `config/servers.yaml` | 服务器清单（ssh 凭据），`enabled: false` 不参与采集 |
| `config/admin.yaml` | 管理员（argon2id 哈希 + JWT 密钥），由 init 生成，勿提交 git |
| `config/services.yaml` | 模型服务注册表，运行中由 UI 写入 |

## API（Bearer JWT）

```
POST /api/login                  登录 → {token, username}
GET  /api/me                     当前用户
GET  /api/summary                机群汇总（无认证）
GET  /api/servers                列表（含实时状态）
POST /api/servers                新增      PUT/DELETE /api/servers/{id}
POST /api/servers/{id}/test      SSH 连通性
POST /api/servers/{id}/refresh   单台采集
POST /api/servers/refresh-all    全量采集（并发受 ssh_concurrency 限制）
GET  /api/driver-matrix          驱动矩阵 + 版本分布
POST /api/driver-matrix/analyze  LLM 驱动升级研判 {target_driver, target_cuda}
GET  /api/services               服务列表
POST /api/services               注册      PUT/DELETE /api/services/{id}
POST /api/services/{id}/start    启动（自动选卡）
POST /api/services/{id}/stop     停止
POST /api/services/{id}/plan     AI 迁移方案
POST /api/services/{id}/migrate  蓝绿迁移（目标机无模型 → 409 + rsync 提示）
POST /api/services/{id}/copy     后台 rsync 模型
GET  /api/audit?limit=300        审计日志
GET/PUT /api/settings            全局设置（PUT 原子写回主配置）
```

## 安全说明

- 登录：argon2id 密码哈希 + JWT HS256（默认 72h 有效期）。
- 采集/操作全部通过 russh 直连 SSH（密码认证），不经任何中间代理。
- `servers.yaml` 含明文 SSH 凭据，生产部署请收紧文件权限（chmod 600）。
