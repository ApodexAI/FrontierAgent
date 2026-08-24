# FrontierAgent TUI 开箱即用教程（macOS / Linux）

[English](tui-endpoint-quickstart.md) · [文档索引](../README.md)

本教程面向希望直接使用 FrontierAgent 的开发者：在 macOS 或 Linux
终端运行 TUI，并通过 `.env` 连接现成的 OpenAI-compatible LLM endpoint。
全程不部署模型、不需要 NVIDIA GPU，也不要求 Docker。

## 完成后你会得到什么

```text
你的 macOS / Linux 终端
├── FrontierAgent 全屏 TUI
├── 目标代码仓库（由 --cwd 指定）
└── HTTPS → 已有的 OpenAI-compatible LLM endpoint
```

FrontierAgent 和模型服务相互独立。本机只运行 Agent 与工具；推理由
`.env` 中配置的远程 endpoint 完成。

## 1. 准备条件

你只需要：

- macOS 或 Linux 终端；
- Git；
- 能访问 GitHub、Python 软件源和 LLM endpoint 的网络；
- endpoint 的 API key、base URL 和 model ID。

项目使用 `uv` 管理 Python 3.12 和依赖。下面的一键脚本会在缺少 `uv`
时从 Astral 官方地址安装它。团队环境若禁止自动下载，请先由管理员安装
[`uv`](https://docs.astral.sh/uv/getting-started/installation/)。

## 2. 克隆项目并配置 `.env`

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
chmod 600 .env
```

用编辑器打开 `.env`，至少填写下面三项：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=your-model-id
```

配置规则：

- endpoint 必须兼容 OpenAI Chat Completions API；
- `OPENAI_BASE_URL` 通常以 `/v1` 结尾，不要写成
  `/v1/chat/completions`；
- `OPENAI_MODEL` 必须是 endpoint 接受的准确 model ID；
- `.env` 已被 Git 忽略。不要提交、截图或分享它；
- 环境变量优先于 `.env`，因此 shell 中已有的同名变量会覆盖文件值。

`SERPER_API_KEY`、`JINA_API_KEY` 等搜索和文档能力都是可选项，不影响
首次启动 TUI。

## 3. 启动 TUI

以下命令必须在克隆得到的 `FrontierAgent` 目录中运行。把
`/absolute/path/to/your-project` 替换为希望 Agent 操作的代码仓库绝对路径。

### macOS

```bash
./scripts/run-macos.sh \
  --mode react \
  --cwd /absolute/path/to/your-project
```

### Linux

```bash
./scripts/run-linux.sh \
  --mode react \
  --cwd /absolute/path/to/your-project
```

脚本会安装所需的 Python 3.12 与依赖、校验 `.env`，随后打开全屏 TUI；
它不会安装或启动本地 LLM。首次安装需要下载依赖，之后会复用缓存。

默认 runtime 选择与平台有关：

- Linux 默认使用 workspace-local Native runtime；
- macOS 默认自动选择：Docker daemon 可用时使用 Docker，否则回退 Native；
- 传入 `--native` 可明确要求 Native；
- Linux 安装且支持 bubblewrap 时，可传入 `--bwrap` 使用轻量文件系统隔离；
- Docker daemon 可用时，可传入 `--docker` 使用容器隔离。

Linux 的选择可以概括为：

| 启动方式 | 行为 |
|---|---|
| 不传 runtime 参数 | 默认 Native，不探测或自动切换到 Docker/bubblewrap |
| `--native` | 明确要求 Native |
| `--bwrap` | 明确要求 bubblewrap；不可用时启动失败 |
| `--docker` | 明确要求 Docker；不可用时启动失败 |

显式选择隔离模式后采用 fail-closed 语义：如果 bubblewrap 或 Docker 不可用，
FrontierAgent 会报错退出，不会静默回退到 Native。这样可以避免开发者误以为任务
仍运行在隔离环境中。

在 Debian/Ubuntu 上安装 bubblewrap 并启动：

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap
./scripts/run-linux.sh --bwrap --mode react --cwd /absolute/path/to/your-project
```

只有安装二进制还不一定够；宿主机或云平台还必须允许 Linux user namespaces。
启动脚本会检查实际可用性。

例如，希望在任一平台明确使用 Docker 时运行：

```bash
# 需要已安装并启动 Docker
./scripts/run-macos.sh --docker --mode react --cwd /absolute/path/to/your-project
./scripts/run-linux.sh --docker --mode react --cwd /absolute/path/to/your-project
```

Docker 只作为 Agent 命令执行的隔离边界，仍然直接调用 `.env` 中的远程 LLM
endpoint，不会在本机部署模型。

如果只想先完成安装和配置校验，在上述命令中加入 `--setup-only`。

## 4. 运行第一个任务

TUI 打开后，在底部输入框粘贴以下只读任务并按 Enter：

```text
请阅读这个仓库的 README 和项目配置，概括它的用途、入口和本地测试命令。只分析，不修改文件。
```

推荐先使用 `react` 模式。它由一个有状态 Agent 顺序完成代码阅读、命令执行
和文件工作。任务确实可以并行拆分时，可退出并改用：

```bash
# macOS
./scripts/run-macos.sh --mode agent_team --cwd /absolute/path/to/your-project

# Linux
./scripts/run-linux.sh --mode agent_team --cwd /absolute/path/to/your-project
```

`agent_team` 会由 coordinator 分派多个子任务，通常消耗更多 endpoint token。

## 5. TUI 中最常用的操作

| 操作 | 用途 |
|---|---|
| `/help` 或 `F1` | 查看完整帮助 |
| `/config` | 查看脱敏后的 provider、model 和 endpoint 配置 |
| `/mode react` | 切换到单 Agent workflow |
| `/mode agent_team` | 切换到 Agent Team workflow |
| `/plan` | 先调查和给出计划，批准计划前禁止修改 |
| `/attach <path>` | 添加只读文件或目录输入 |
| `/log` | 显示当前运行 trace 文件的位置 |
| `/revert` | 撤销本会话由文件编辑工具记录的改动 |
| `/resume` | 在 TUI 中选择并恢复已保存会话 |
| `Ctrl-C` | 中断当前任务 |
| `/exit` | 退出 TUI |

Agent 执行写入操作时默认显示 diff 并请求批准。初次使用不要添加 `--yes`。
Native 模式不是操作系统沙箱，获批的命令拥有当前 macOS/Linux 用户的权限；
需要容器隔离边界时可显式选择 `--docker`。

## 6. 文件与会话在哪里

对 `--cwd /absolute/path/to/your-project` 启动的 Native 运行，状态默认位于：

```text
/absolute/path/to/your-project/.apodex/
├── runs/<session-id>/       # trace、engine log、checkpoint 和 outputs
└── runtime/native/          # workspace-local 缓存、临时文件和依赖
```

查看可恢复会话：

```bash
uv run frontier-agent --cwd /absolute/path/to/your-project --resume
```

使用指定 session ID 恢复：

```bash
uv run frontier-agent \
  --cwd /absolute/path/to/your-project \
  --resume SESSION_ID
```

## 7. 常见问题

### 启动后不是全屏 TUI

全屏 TUI 要求 stdin 和 stdout 都连接到正常 TTY。不要把启动命令放在管道、
重定向或 `TERM=dumb` 环境中，也不要添加 `--print`、`--no-tui`、
`--no-color` 或 `--theme mono`。

### `.env` 缺少必填项

确认 `FrontierAgent/.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和
`OPENAI_MODEL` 都不是空值，然后从 FrontierAgent 仓库根目录重新运行脚本。

### 401 / 403

API key 无效、过期或没有目标模型权限。修正 `.env` 后重新启动。

### 404 / model not found

确认 base URL 通常包含 `/v1`，但不包含 `/chat/completions`；同时核对
endpoint 实际暴露的 model ID。

### 超时或无法连接

检查 endpoint 是否能从当前机器访问，以及 VPN、代理、DNS、TLS 证书和防火墙
策略。FrontierAgent 的启动预检只验证本地配置格式，不会联网测试 key。

### `uv` 安装后仍找不到

重新打开终端，或先按 `uv` 官方安装文档把其 bin 目录加入 `PATH`。项目脚本也会
主动检查 `$HOME/.local/bin/uv` 和 `$HOME/.cargo/bin/uv`。

## 8. 日常启动的最短命令

完成首次安装后，仍从 FrontierAgent 仓库根目录运行：

```bash
# macOS
./scripts/run-macos.sh --cwd /absolute/path/to/your-project

# Linux
./scripts/run-linux.sh --cwd /absolute/path/to/your-project
```

下一步请阅读[中文 TUI 使用教程](../tui-user-guide.zh-CN.md)，其中包含右侧三个
Tab、空格预览、审批以及 Agent Team 异步干预的完整操作流程。

更底层的 CLI/TUI、审批、附件、主题和会话参考见
[`apodex/README.md`](../../apodex/README.md)。平台细节见
[macOS 安装指南](macos.md)和 [Linux 安装指南](linux.md)。
