# 在 macOS 上从零运行 FrontierAgent

本文面向第一次接触 FrontierAgent 的同事。完成 `git clone` 后，可以用
一条命令安装依赖、配置模型端点并启动终端界面。

> macOS 负责运行 FrontierAgent，而不是运行仓库里的 NVIDIA SGLang 镜像。
> Apple Silicon 和 Intel Mac 都不是 NVIDIA CUDA 主机。模型应来自托管的
> OpenAI-compatible API，或者运行在远程 Linux NVIDIA 服务器上的 SGLang。

## 1. 运行方式概览

```text
MacBook
├── FrontierAgent TUI
├── ReAct 或 Agent Team workflow
├── 当前项目目录（Agent 可以读取、分析和按审批修改）
└── HTTPS / SSH tunnel
    └── 托管模型 API，或远程 Linux 上的 SGLang
```

最少需要：

- 一台可以访问模型端点的 MacBook；
- Git。新 Mac 第一次运行 `git` 时，系统可能提示安装 Command Line Tools；
- 一个 OpenAI-compatible 模型端点的 URL、模型名和 API key；
- 访问 GitHub 和 Python/uv 软件源的网络。

Python 3.12 和项目依赖由 `uv` 管理。一键脚本发现 `uv` 不存在时，会从
[Astral 官方安装地址](https://docs.astral.sh/uv/getting-started/installation/)
下载 installer；建议对供应链有严格要求的团队先审阅脚本或统一预装 `uv`。

## 2. 最快启动

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
./scripts/run-macos.sh
```

首次运行会依次：

1. 确认当前系统是 macOS；
2. 查找 `uv`，缺失时使用官方 installer 安装；
3. 询问 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`；
4. 创建权限为 `600` 的 `.env`，且不会回显 API key；
5. 安装 Python 3.12 和 FrontierAgent 依赖；
6. 以 ReAct 模式启动 TUI。

`.env` 已被 Git 忽略，不应提交。脚本发现已有 `.env` 时不会覆盖它；如果
必填项为空，会提示手动补全后退出。

常见启动方式：

```bash
# 单 Agent，适合代码阅读、文件处理和顺序执行任务
./scripts/run-macos.sh --mode react

# Coordinator + 多个子 Agent，适合可拆分的调研任务
./scripts/run-macos.sh --mode agent_team

# 让 Agent 操作另一个项目，而不是 FrontierAgent 仓库自身
./scripts/run-macos.sh --mode react --cwd /Users/your-name/work/my-project

# 只安装和配置，不启动 TUI
./scripts/run-macos.sh --setup-only
```

查看全部选项：

```bash
./scripts/run-macos.sh --help
```

## 3. 模型端点怎么填

FrontierAgent 使用 OpenAI-compatible Chat Completions API。最基本的配置是：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://模型服务地址/v1
OPENAI_MODEL=服务端暴露的模型名称
```

注意：

- `OPENAI_BASE_URL` 通常需要包含 `/v1`，不要填写到
  `/v1/chat/completions`；
- `OPENAI_MODEL` 必须与服务端接受的 model ID 一致；
- 代理、公司 VPN 或证书拦截可能导致连接失败；
- 不要在聊天、截图、Issue 或 PR 中粘贴 `.env` 和 API key。

如果使用远程 Linux 机器上的 SGLang，推荐让 SGLang 只监听远程回环地址，
再从 Mac 建立 SSH tunnel：

```bash
ssh -N -L 30000:127.0.0.1:30000 user@gpu-server
```

保持该终端运行，在另一个终端配置：

```dotenv
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://127.0.0.1:30000/v1
OPENAI_MODEL=local-model
```

远程服务器的 SGLang 部署和显存配置请参考
[SGLang 配置说明](../../config/sglang/README.md)。不要把无认证的 SGLang
端口直接暴露到公网。

## 4. ReAct 和 Agent Team 如何选择

| 模式 | 适用场景 | 特点 |
|---|---|---|
| `react` | 修改代码、阅读文件、生成单个交付物、顺序调研 | 一个有状态 Agent，执行路径更直接、模型调用更少 |
| `agent_team` | 多来源调研、可以并行拆分的问题、需要交叉核对 | 主 Agent 分配子任务、收集报告并综合，调用量和成本更高 |

拿不准时先用 `react`。只有任务确实可以分解，或者需要多个独立证据来源时再用
`agent_team`。

## 5. Native 与 Docker Desktop

不安装 Docker Desktop 也可以运行。脚本默认让 FrontierAgent 自动选择运行时：

- Docker daemon 可用时，可使用 Linux agent 容器作为命令执行边界；
- 否则使用 workspace-local native runtime。

显式选择：

```bash
# 强制使用本机运行时
./scripts/run-macos.sh --native

# 强制使用 Docker Desktop；Docker 未启动时会给出错误
./scripts/run-macos.sh --docker
```

Native 模式下，命令使用当前 macOS 用户权限执行。修改操作仍经过 FrontierAgent
审批，但这不是操作系统级沙箱。不要把不可信仓库、模型或任务交给 Native 模式。
Docker 能提供更强的文件和进程边界，但不能让 Mac 获得 NVIDIA CUDA 能力。

## 6. 手动安装

不希望脚本自动安装软件时，可手动执行：

```bash
# 二选一
brew install uv
# 或使用 uv 官方 standalone installer
# curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
uv sync --python 3.12
cp .env.example .env
chmod 600 .env
$EDITOR .env

uv run frontier-agent --mode react --cwd "$PWD"
```

如果公司要求固定依赖来源或禁用自动下载，应由管理员预装 `uv` 和 Python 3.12，
然后再执行 `uv sync`。

## 7. 日常使用

启动交互式 TUI：

```bash
./scripts/run-macos.sh --mode react --cwd /path/to/project
```

直接执行一次任务并在终端打印结果：

```bash
uv run frontier-agent \
  --mode react \
  --cwd /path/to/project \
  --print \
  "解释这个项目的模块边界，不要修改文件"
```

将文件或目录作为只读输入：

```bash
uv run frontier-agent \
  --mode react \
  --cwd /path/to/project \
  --input /path/to/reference.pdf
```

恢复会话：

```bash
uv run frontier-agent --resume
uv run frontier-agent --resume SESSION_ID
```

建议第一次在新项目中使用时明确说明“只分析，不修改”。需要修改时先查看审批
界面中的命令和 diff，不要为了省事默认添加 `--yes`。

## 8. 可选的联网与长上下文配置

网页搜索和抓取不是启动 TUI 的必需条件。需要时编辑 `.env`：

```dotenv
SERPER_API_KEY=
SERPER_BASE_URL=https://google.serper.dev
JINA_API_KEY=
JINA_BASE_URL=https://r.jina.ai
```

模型上下文与单次输出上限也可以覆盖：

```dotenv
OPENAI_CONTEXT_WINDOW=131072
OPENAI_MAX_INPUT_TOKENS=110000
OPENAI_MAX_TOKENS=8192
```

这些值必须符合真实端点能力。输入上限应小于 context window，并为输出和工具调用
留出空间。工作流 profile 的完整注释位于：

- [`workflows/stateful_react_agent/profiles/`](../../workflows/stateful_react_agent/profiles/)
- [`workflows/agent_team/profiles/`](../../workflows/agent_team/profiles/)

## 9. 常见问题

### `git` 或 Command Line Tools 不存在

运行：

```bash
xcode-select --install
```

安装完成后重新打开终端，再执行 clone。

### `uv` 安装后仍提示找不到

重新打开终端，或确认以下位置是否存在：

```bash
ls -l "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" 2>/dev/null
```

一键脚本会主动检查这两个路径，通常不需要修改 shell profile。

### 401/403

检查 API key、模型服务权限以及 `.env` 中是否包含多余引号或空格。使用私有模型
时还要确认账号拥有该模型权限。

### 404 或 model not found

通常是 `OPENAI_BASE_URL` 缺少 `/v1`，或者 `OPENAI_MODEL` 与服务端暴露的名称
不一致。SGLang 可通过 `GET /v1/models` 查看模型名。

### 连接超时

检查 VPN、代理、DNS 和公司证书策略。远程 SGLang 场景还要确认 SSH tunnel
终端仍在运行，并在 Mac 上测试：

```bash
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30000/v1/models
```

### Docker 模式失败

先启动 Docker Desktop，再运行：

```bash
docker info
./scripts/run-macos.sh --docker
```

如果只想立即使用，可改为 `--native`，但要理解 Native 模式的权限边界。

### 首次安装很慢

`uv` 可能需要下载 Python 3.12 和依赖。公司代理、软件源速度和首次 Docker
镜像构建都会影响耗时；后续运行会复用缓存。

## 10. 更新与清理

更新代码和依赖：

```bash
cd FrontierAgent
git pull --ff-only
./scripts/run-macos.sh --setup-only
```

运行状态、缓存和输出主要位于仓库内的 `.apodex/`。删除仓库前先备份需要保留的
交付物和会话。`.env` 含密钥，应安全删除而不是复制到公共位置。

英文简版见 [macOS installation](macos.md)，其他操作系统见
[installation chooser](README.md)。
