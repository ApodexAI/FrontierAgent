# FrontierAgent HF Space Demo — 部署指引

> 这份文档面向「拿到 HF org 权限后要把 Demo 发布上去」的人，只讲两件事：
> **我们做了什么** 和 **你在 Hugging Face 侧怎么把它启动起来**。
>
> 完整技术手册见同目录 [`README.md`](README.md)（英文，**权威版本**，含全部变量
> 说明、安全边界、回滚方式）。本文是它的精简中文入口：改动请先改英文版，两者
> 冲突时一律以 `README.md` 为准。

---

## 一、我们做了什么（实验逻辑）

### 1. 目标不是部署模型，而是把已有的 Agent 跑到网页上

```
浏览器
  ↓
Gradio Web UI            ← Space 负责
  ↓
FrontierAgent react runtime   ← Space 负责（真实的 stateful-react-agent pipeline）
  ↓
OpenAI 兼容 API endpoint      ← 外部，Space 只调用
```

Space 里**不下载、不运行任何模型权重**。换模型 = 改一个环境变量。

### 2. 关键设计：没有重写 Agent，而是接到运行时已有的两个接口

这是整个方案能成立的原因。`react` 之前只能从终端启动（`TerminalSession` +
Textual TUI），但运行时本身早就留好了非终端调用的口子：

| 接口 | 用途 |
| --- | --- |
| `metadata['sdk_extra_observers']` | 塞进一个普通 observer，就能拿到流式 token、tool 开始/结束、最终答案。**不需要 stdin、不需要 Textual、不解析 ANSI**。 |
| `metadata['pause_check']` | Stop 按钮走协作式停止：agent 在下一个 turn 边界收尾，仍然给出已有的答案，而不是被硬杀。 |
| `metadata['profile_overrides']` | 模型 / endpoint / 密钥 / 各种限额从这里注入，所以换模型不用改代码、也不用改 profile YAML。 |

因此 `frontier_agent/` 核心代码**零改动**，所有 Space 专属逻辑都在
`deploy/huggingface/`。CLI / TUI 行为不受影响。

踩过两个坑（都在代码注释里写明了原因，别改掉）：
- observer 必须声明 `wants_llm_delta`，否则运行时判定不需要流式，答案会一次性弹出；
- observer 必须 `critical = True`，否则 hook 被当后台任务派发、顺序不保证，流式文字会乱序。

### 3. 公开 Demo 的安全边界

- **没有 shell**：`bash`、`run_python_code`、`download_file`、文件编辑器、子 agent 工具全部禁用，`DEMO_ALLOWED_TOOLS` 只能进一步收窄、不能放宽。
- **文件只能待在自己 session 里**：每个访客一个随机 24 字节 ID 的目录；工具调用前会拦一道，路径越界直接拒绝。
- **下载只给自己的 `outputs/`**：目录穿越、绝对路径、软链外逃、`.env`/`secret`/`token` 之类文件名全部拒绝。
- **密钥不外泄**：事件、答案、日志里都会被掩码，包括上游 endpoint 回显出来的。镜像里没有 `.env`、没有 key。
- **一次只跑一个任务**：运行时有进程级全局状态，所以 `DEMO_MAX_CONCURRENCY` 会被强制压到 1。要扩容用多副本，别改这个值。

### 4. 已经验证过的

| 项目 | 结果 |
| --- | --- |
| 自动化测试 | `tests/test_hf_space_{config,leaks,runtime,ui}.py` —— 配置/密钥泄漏/真实 runtime 端到端/Gradio 四层，全部不需要真实 token（UI 那层需要 `hf-space` extra，否则跳过） |
| Docker | build / run 成功，UID 1000 非 root，容器内监听 `0.0.0.0:7860`，SIGTERM 干净退出 |
| 真实端点（`apodex-1.1-397b-fp8`） | tool calling ✅ 流式 ✅ 真实联网搜索 ✅ |
| 端到端实跑 | MCP 调研任务 19s / 6 turns / 7 tool calls；ReAct 论文调研 42s，均产出真实可下载文件 |
| Space 目录树 | `publish.sh` 组装出的树**已验证能 build 并正常服务**（HF 执行的就是这个构建） |

> 这一栏不写具体的通过数量：测试会持续增加，写死的数字过两周就是错的。要当前
> 结果就跑 `uv run pytest tests/test_hf_space_*.py`。

---

## 二、在 Hugging Face 侧启动 Demo

前置条件：HF org 的 Space 创建权限；一个可用的 OpenAI 兼容 endpoint（**必须支持
tool calling**）。

### 步骤 1：建 Space

Hugging Face → **New Space**：

| 选项 | 值 |
| --- | --- |
| SDK | **Docker** |
| Template | **Blank**（不要选别的模板） |
| Hardware | **CPU basic** 够了（模型在外部，Space 不跑推理） |
| Visibility | 建议先 **Private**，验证完再公开 |

可选：开 **Persistent storage**，session 产物能跨重启保留（程序会自动用 `/data`）。

### 步骤 2：准备要推上去的代码

HF 只认 Space 仓库**根目录**的 `Dockerfile`，而本仓库根目录的 `Dockerfile` 是 CLI
镜像。所以别直接推整个 repo，用脚本组装：

```bash
./deploy/huggingface/publish.sh /tmp/space-tree
```

它会把运行时需要的包拷过去、把 Space 的 `Dockerfile` 放到根目录、把
`README.space.md` 装成 Space 的 `README.md`（它的 YAML front matter 才是告诉 HF
「用 Docker、端口 7860」的东西）。脚本**不会自己 push**，也不会拷任何 `.env`。

### 步骤 3：推代码

```bash
cd /tmp/space-tree
git init && git add -A && git commit -m "FrontierAgent react demo"
git remote add space https://huggingface.co/spaces/<org>/<space-name>
git push --force space HEAD:main
```

> 推送需要 HF 的 write token（`huggingface-cli login`，或在 URL 里带
> `https://<user>:<hf_token>@huggingface.co/...`）。

### 步骤 4：填 Variables / Secrets

Space → **Settings** → *Variables and secrets*。

**Variables**（明文，必填）：

| 名称 | 说明 |
| --- | --- |
| `OPENAI_BASE_URL` | API **base** 地址，以 `/v1` 结尾。见下方「最容易踩的坑」 |
| `OPENAI_MODEL` | endpoint 实际提供的模型名（会放进请求体） |
| `SUMMARY_LLM_BASE_URL` | **完整的** chat-completions URL，注意和上面不一样 |
| `SUMMARY_LLM_MODEL_NAME` | 网页正文抽取用的模型，用小而快的最好 |

**Variables**（可选，调限额；不填就用默认值）：

| 名称 | 默认 | 说明 |
| --- | --- | --- |
| `HF_MODEL_ID` | `apodex/Apodex-1.1-mini` | 页面上显示的模型名（纯展示） |
| `DEMO_MAX_TURNS` | `24` | 单任务 turn 上限 |
| `DEMO_TASK_TIMEOUT_SECONDS` | `600` | 单任务墙钟上限；大 reasoning 模型建议 `900` |
| `DEMO_MAX_OUTPUT_TOKENS` | `4096` | 大 reasoning 模型建议 `16384` |
| `DEMO_QUEUE_SIZE` | `4` | 排队上限，超了直接拒绝 |
| `DEMO_SESSION_TTL_SECONDS` | `3600` | session 目录保留时长 |
| `SERPER_BASE_URL` / `JINA_BASE_URL` | 厂商公网地址 | 走内部代理时才需要填 |

**Secrets**（隐藏，不要放进 Variables）：

| 名称 | 必要性 | 不填的后果 |
| --- | --- | --- |
| `OPENAI_API_KEY` | **必填** | 启动即拒绝服务 |
| `SERPER_API_KEY` | 强烈建议 | `web_search` 返回**零结果**，Demo 看起来像坏了 |
| `SUMMARY_LLM_API_KEY` | 强烈建议 | `web_fetch` 每个网页都抽取失败，Agent 只能靠搜索摘要 |
| `JINA_API_KEY` | 可选 | 网页抓取质量下降，会退回直连 HTTP |

> 本服务器上这些搜索类 key 存在 Vault 的 `secret/agentos`
> （`SERPER_API_KEY` / `JINA_API_KEY` 及对应的 `*_BASE_URL`，走内部代理地址）。

改 Variables / Secrets 只会重启，不需要重新 build。

### 步骤 5：等 build，然后验收

Space 页面状态走 **Building → Running**。打开 Logs，应当看到：

```
FrontierAgent Demo: workflow=react model=… served_model=… endpoint=… runtime_root=…
```

如果它打印的是配置错误列表并拒绝启动——这是**故意的**，照着提示改对应变量即可。
预检会替你挡掉最常见的几种配错。

打开 Space URL，提交一个需要用工具的任务，例如：

```
Search the web for what the Model Context Protocol is, then write a
4-sentence explanation to outputs/mcp.md and tell me what you wrote.
```

应当依次看到：`Queued/Running` → 右侧逐条 tool activity → 答案逐字流出 →
`● Completed`（显示 turns / tool calls）→ 右下出现可下载文件。

再顺手确认边界生效：让它 `Read /etc/passwd`，应当被拒绝（`refused`）。

---

## 三、最容易踩的坑

**1. `OPENAI_BASE_URL` 填错（最高频）**

```
✅  https://your-endpoint.example.com/v1
✅  https://router.huggingface.co/v1
❌  https://huggingface.co/apodex/Apodex-1.1-mini        ← 这是模型仓库网页，不是 API
❌  https://your-endpoint.example.com/v1/chat/completions ← 太具体，程序自己会拼路由
```

前两种错法预检会在启动时直接点名拒绝，不会等到第一次提问才报一个莫名的 404。

**2. `SUMMARY_LLM_BASE_URL` 和 `OPENAI_BASE_URL` 格式不一样**

前者要**完整**的 chat-completions URL，后者只要 base。同一个 endpoint 是这样：

```
OPENAI_BASE_URL      = https://example.com/api/v1
SUMMARY_LLM_BASE_URL = https://example.com/api/v1/chat/completions
```

**3. endpoint 不支持 tool calling**

那就不是配置问题。`react` 是 agent loop 而不是 chat completion，不支持工具的
endpoint 只会给出「像聊天一样」的回答、没有任何 tool activity。这种情况请按
需求文档 §18-E **上报 endpoint 能力不足，不要把 Demo 降级成聊天机器人**。

自检方式：

```bash
curl -s -X POST "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"'"$OPENAI_MODEL"'","max_tokens":2048,
       "messages":[{"role":"user","content":"Call the write_file tool now."}],
       "tools":[{"type":"function","function":{"name":"write_file","parameters":
         {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
          "required":["path","content"]}}}]}'
```

返回里出现 `"finish_reason":"tool_calls"` 和 `tool_calls` 数组即为通过。

**4. 大 reasoning 模型下的限额**

reasoning 模型在吐出第一个可见 token 之前可能思考几分钟。建议
`DEMO_TASK_TIMEOUT_SECONDS=900`、`DEMO_MAX_OUTPUT_TOKENS=16384`，否则容易看到
被截断的回答。

**5. Office 文档交付物用不了**

Demo 能产出 `.md` / `.csv` / `.txt` / `.json`，但产不出 `.docx` / `.xlsx` /
`.pptx`。这**是**一个取舍——曾经不是。以前 `create_file` 把约 131 KB 的 base64
writer 内联进单个 `sh -c` 参数，超过 Linux 单参数 131,072 字节上限，任何调用都
`E2BIG`；这个缺陷已在上游修掉（writer 改走 stdin），`openpyxl` / `python-docx` /
`python-pptx` 也已经是包的硬依赖，镜像里本来就有。现在不开它的原因有两条：它是
整个 demo 里唯一还会起子进程的工具（用本进程的解释器执行模型写出的 JSON
program），而 demo 的containment 只检查顶层路径参数，不检查 `ops` 里嵌套的次级
写入目标。要开的话这两条都得先处理，详见 `README.md`。

---

## 四、发布前不用重新开发就能先本地验一遍

完全不需要真实 token 和 GPU：

```bash
# 终端 1：假的 OpenAI 兼容 endpoint（会脚本化一次 tool 调用）
uv run python -m deploy.huggingface.mock_llm --port 8018 --tool-demo

# 终端 2：无头跑通 runtime → tool → artifact → final answer
SANDBOX_BACKEND=native uv run python -m deploy.huggingface.poc
```

有真实 endpoint 时：

```bash
export OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_API_KEY=...
SANDBOX_BACKEND=native uv run python -m deploy.huggingface.poc --real
```

浏览器验：按 `README.md` §2 起 Docker，端口只绑到 `127.0.0.1:7860`，再用
`ssh -N -L 7860:127.0.0.1:7860 USER@SERVER` 从本地访问 `http://localhost:7860`，
开发服务不会暴露到公网。

---

## 五、回滚

- **配置改错** → 把 Variable / Secret 改回去，Space 自动重启，不用 rebuild。
- **代码有问题** → `git push --force space <上一个 sha>:main`，或 Space
  Settings → **Factory rebuild**。
