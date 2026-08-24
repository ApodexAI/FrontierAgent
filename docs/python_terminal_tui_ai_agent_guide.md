# Python Terminal TUI 组件选型（设计决策记录）—— 为 AI Agent Framework 打造终端前端

> 本文是 Textual + Rich 选型的设计决策记录，不是安装或使用教程；文件名中的
> `guide` 属于历史遗留。寻找安装、工作流、SGLang 或开发文档请从
> [文档索引](../docs/README.md) 开始。

> 目标：在终端中运行、非网页、轻量但美观的可渲染前端，用于 AI Agent Framework。

> **项目决策（已冻结，2026-08-04）**：交互式 TUI 唯一使用
> **Textual**，富文本渲染与非 TTY/单次执行降级使用 **Rich**。
> 不再引入 Urwid、PyTermTk、PyTermGUI 或 Panel。后续工作是优化
> `apodex/tui/` 的交互、性能和可访问性，而不是继续更换或叠加 UI 框架。

## 零、当前项目基线

FrontierAgent 已经实现了上述组合，不是待选型状态：

- `apodex/tui/`：Textual 全屏 TUI，包含对话、计划、工具活动、状态栏、审批、模型选择、会话恢复，以及多格式 Deliverables 文件系统预览（支持代码、Markdown、PDF、Office 三件套 `.docx`/`.xlsx`/`.pptx`、Jupyter Notebook `.ipynb`、3D/生物分子模型 `.pdb`/`.stl`/`.obj`/`.gltf`、图片 ANSI 字符画 preview 与压缩包目录树）。
- `apodex/render.py`：Rich 渲染器，同时服务于 TUI 内的 renderable 和 `--no-tui` 行模式。
- Agent 引擎与 UI 通过 render sink/observer 解耦；不为任何新 UI 引入第二套事件循环。

当前流式策略在 sink 层合并高频 token，流式期用 Text 低成本刷新，
并对 UI 刷新做时间/字符节流，结束时一次性转为 Rich Markdown。长会话只在
DOM 中保留最近 300 个 transcript block，恢复会话也只渲染最近窗口；完整
对话仍保存在 session/trace 中。超大 Markdown 的前台渲染设有 200,000 字符
安全阈值，diff 和工具输出继续沿用各自的行数/字符阈值。

Textual `MarkdownStream` 仍是后续可评估的
实现细节，但它不是更换技术栈，且必须先通过应用退出和线程执行器
兼容性测试。

重新评估技术栈只在以下条件之一成立时触发：Textual 不再维护、
出现无法规避的跨平台终端兼容问题，或已确认的产品需求无法由
Textual/ Rich 实现。

### 发布兼容性基线

第四批优化把“能否进入全屏 TUI”收敛为单一、可测试的能力判定，
而不是散落在 UI 各处的特例：

| 环境 | 预期模式 | 当前验证方式 |
|---|---|---|
| Linux/macOS/Windows Terminal，stdin/stdout 都是 TTY | Textual 全屏 | 自动能力矩阵；各平台发布前真机验收 |
| SSH、tmux，正常 `TERM` 且双 TTY | Textual 全屏 | `xterm-256color` / `screen-256color` 自动矩阵；真机验收 |
| 窄终端（低至 40×10） | Textual，自动隐藏侧栏 | Textual headless smoke test |
| stdin 或 stdout 被重定向、`-p`、`--no-tui` | Rich 行模式 | 自动矩阵与 CLI smoke |
| `TERM=dumb`、`NO_COLOR`、`--no-color`、mono 主题 | mono 行模式 | 自动矩阵 |

自动测试只证明能力选择和无头渲染，不冒充操作系统真机验证。
每次候选发布仍应在 Linux、macOS、Windows Terminal 以及至少一个
SSH/tmux 会话中人工完成：启动、输入、流式回答、审批、窗口缩放、
中断、恢复和退出。行模式输入不再交给后台 executor 线程，避免
终端断开或测试捕获 stdin 时，残留的非守护线程阻止进程退出。

### 交互体验基线

第五批不增加 Agent 能力，只减少用户记忆和误操作成本：输入框支持
上下键历史并在回到末尾时恢复未提交草稿；斜杠命令提供实时提示、
Tab 补全和 `Ctrl-P` 命令面板；`F1` 展示上下文帮助；`Ctrl-B` 手动
控制侧栏，并与窄终端自动隐藏共同生效。审批弹窗默认选中 **No**，
显示 diff 的行数及增删规模，长预览可用 `Ctrl-U` / `Ctrl-D` 翻页。
这些交互均使用键盘完成，并有 Textual headless 行为测试覆盖。

### 开源 demo 与配置边界

TUI 是 FrontierAgent harness 的本地 demo 前端，采用 BYOK
（Bring Your Own Key）：用户在 shell 环境或本地 `.env` 中配置
provider API key、base URL 和 model。产品不包含账号、login/logout、
OAuth、订阅额度、云端 workspace 或密钥托管；TUI 也不输入、
展示或持久化 key。启动预检只做本地结构校验，不为验证
key 而发送网络请求；`/config` 只显示脱敏的运行诊断。

---

## 一、结论先行：应该怎么选？

| 你的场景 | 推荐方案 | 核心理由 |
|---|---|---|
| **完整交互界面**（聊天式对话、工具侧栏、多面板、日志区） | **Textual** | 异步原生、CSS 驱动布局、1670 万色、流式 Markdown 支持（Textual 4+）、生态最活跃、支持部署到浏览器 |
| **轻量流式输出**（Agent CLI 只需漂亮地打印/打印进度条/代码高亮） | **Rich** | 几乎零学习成本、依赖极小、完美配合 asyncio 流式写入 token |
| 追求最小依赖 + 自定义程度高 | **Urwid** / **PyTermTk** | 经典 curses 风格或 PyQt 式 API，但视觉现代化程度不如前两者 |

> ⚠️ 关于 Panel：**Panel 不是终端 TUI**，它是 Holoviz 的 Web 面板库（基于 Bokeh），只适合浏览器 Dashboard。如果你需要"在终端跑"，请排除 Panel。

---

## 二、主流框架横向对比

| 维度 | **Textual** | **Rich** | **Urwid** | **PyTermTk** | **PyTermGUI** |
|---|---|---|---|---|---|
| GitHub Stars | ~12K+（年增迅猛） | ~33K+ | ~8K+ | ~5K+ | ~3K+ |
| 定位 | 全功能 TUI **框架** | 终端富文本 **渲染库** | curses 风格 **组件库** | PyQt 式 TUI **框架** | 模块化 **TUI 框架** |
| 底层 | 自研异步渲染引擎（基于 Rich） | 自研渲染器 | ncurses / Unicurses | Tk | 自研渲染器 |
| 异步支持 | ✅ asyncio 原生 | ✅ 可配合 asyncio | ⚠️ 需事件循环集成 | ⚠️ 同步为主 | ⚠️ 部分支持 |
| 布局系统 | CSS 驱动（flex/grid）+ `compose()` 声明式 | 无布局（纯顺序输出 + Live 刷新） | Widget/Signal 信号模型 | 类似 Qt 的层级树 | 模块化组件 API |
| 美观度 | ⭐⭐⭐⭐⭐ 现代感强，内置主题 + 1670 万色 | ⭐⭐⭐⭐ markdown/语法高亮/进度条 | ⭐⭐ 经典 ASCII 风格 | ⭐⭐⭐ 类 Qt 风格 | ⭐⭐⭐ 中等 |
| 鼠标/动画 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 流式 Markdown | ✅ Textual 4.0+ 原生支持 | ✅ `Markdown` + `Live` | ⚠️ 需自行拼接 | ⚠️ 需自行处理 | ⚠️ 需自行处理 |
| 依赖体积 | ~8–12 MB（含 rich） | ~1–2 MB | ~500 KB | ~3–5 MB | ~2–4 MB |
| 上手难度 | 中等 | ⭐ 极低 | 高 | 中高 | 中 |
| 跨平台 | Win/macOS/Linux | 全平台 | 全平台 | Windows/macOS/Linux | 全平台 |
| **AI Agent 适用性** | 🥇 首选 | 🥈 轻量首选 | 可用 | 可用 | 小众 |

---

## 三、核心推荐详解

### 🥇 Textual —— 完整交互 Agent UI

**为什么最适合 AI Agent：**
1. **异步原生**：所有 I/O 都是 `await`，天然适合 LLM 流式调用（不阻塞 UI）。
2. **流式 Markdown**（Textual 4.0+）：LLM 返回的 markdown 可边写边高亮显示。
3. **声明式布局**：`compose()` + CSS，快速搭建聊天窗口 + 工具侧栏 + 日志面板的多屏结构。
4. **1670 万色 + 主题系统**：深色主题开箱即用，符合开发者审美。
5. **一套代码双端运行**：可通过 `textual serve` 把同一套 App 部署到浏览器（Textual Web）。
6. **生态参考丰富**：Textualize 官方有 ChatGPT TUI 示例；社区有 `aider`、`Instrukt` 等终端 Agent 项目参考。

**安装与运行示例：**

```bash
pip install textual
cd /workspace/examples
python3 textual_agent.py   # 快捷键：Ctrl+C 退出 · Ctrl+D 暂停/恢复 Agent
```

**示例文件已验证可运行**（文本位于 `/workspace/examples/textual_agent.py`），包含：
- 顶部状态栏（带时间）
- 对话区：模拟 LLM token 逐字流式填充
- 输入行：Input 框 + 发送/深度思考按钮
- 工具栏侧面板：工具列表展示
- 运行日志区：Agent 内部调用过程实时滚动

核心代码片段：

```python
async def send_message(self):
    msg = self.query_one(Input).value.strip()
    live_logs = self.query_one("#logs-text")
    live_logs.update("[yellow]正在生成响应...[/yellow]")
    buffer = ""
    for i in range(0, len(full_text), 8):
        await asyncio.sleep(0.02)      # 模拟网络延迟
        chunk = full_text[i:i+8]
        buffer += chunk
        self.query_one("#chat-area").update(buffer)   # 增量刷新 Widget 内容
        if "结论" in chunk:
            live_logs.update("[green]响应完成[/green]")
```

---

### 🥈 Rich —— 轻量级流式渲染器

**适用场景**：Agent CLI 不需要复杂布局，只需要：
- 漂亮的错误 traceback
- Markdown 代码块高亮显示
- 进度条 / Spinner / 控制台面板
- Live Display 实现流式刷新的表格/状态

**关键点**：Rich 是**渲染库**，不是布局框架 —— 没有 widget、没有事件循环。但可以用 asyncio + `Live` 做流式输出，完全不阻塞事件循环。

**安装与运行示例：**

```bash
pip install rich
cd /workspace/examples
python3 rich_agent_output.py
```

**示例文件已验证可运行**（文本位于 `/workspace/examples/rich_agent_output.py`），演示了：
- 用 `Live` + `Markdown` 实时流式更新 Agent 思考过程
- `Panel` 包裹工具输出，Syntax 做代码高亮
- 状态 Spinner + Markdown 内容组合渲染

关键代码片段：

```python
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax

with Live(console=console) as live:
    live.update(Markdown("# 等待输入...\n"))
    for frag in llm_stream("分析用户代码"):
        buffer += frag
        live.update(Markdown(buffer), refresh=True)  # 流式刷新
```

---

### 🥉 Urwid / PyTermTk —— 备选方案

| | Urwid | PyTermTk |
|---|---|---|
| 特点 | Python 最老牌的 TUI 库，成熟稳定，信号/监听器模式 | 自包含库，提供类 PyQt 的 API（Button、Tree、TextEdit 等） |
| 适合 | 对 curses 熟悉、需要极细粒度控制的工具 | 熟悉 Qt 的开发者，想快速构建表单式终端界面 |
| 局限 | 视觉偏复古，文档和社区不如 Rich/Textual 活跃 | 相对小众，社区资源少 |  |

---

## 四、针对 AI Agent Framework 的架构建议

### 混合架构（推荐）

很多真实 Agent 项目采用 **"Rich 做流式输出 + 可选 Textual 做交互外壳"** 的组合，核心逻辑与 UI 完全解耦：

```
agent-core/           # 纯 Python 业务逻辑（无 UI 耦合）
agent-cli/            # 命令行入口：Rich Console + Live + Progress
agent-tui/            # 可选：Textual 外壳，复用 agent-core 的逻辑
agent-web/            # 可选：Gradio/Streamlit（如需 Web）
```

这样想换前端或加 Web 版都毫不费力。

### Agent 典型 TUI 布局模式

```
┌─────────────────────────────────────────────────────┐
│ [Header] AgentName v1.0    🕒 14:30                 │
├─────────────────────────────────────────────────────┤
│ 状态：● 空闲 / ● 运行中                              │
├────────────────────────────┬────────────────────────┤
│ 【对话区】                 │ 【工具栏】               │
│ User → 帮我写个函数        │ • 文件搜索             │
│ Agent → def f():          │ • 代码生成             │
│     ...                    │ • 日志分析             │
│     return analyze(...)    │ • 模型切换             │
├────────────────────────────┼────────────────────────┤
│ 【指令输入】输入→[发送]    │ 【运行日志】            │
│                          │ 调用 model... ✓         │
│                          │ 读取 file... ✓           │
└────────────────────────────┴────────────────────────┘
```

Textual 的 `Container` + `Horizontal` + CSS grid 布局完美匹配这种结构。

### 流式输出的最佳实践

无论是 Rich 还是 Textual，LLM 流式输出的共同模式是：**"缓冲区 + 增量更新"**：

```python
buffer = ""
for token in llm_response.stream():      # SseStream / openai 流式迭代器
    buffer += token
    console_or_widget.update(buffer)     # 只刷新变化部分
```

⚠️ 避免每次全量重建 Renderable，会导致终端闪烁、性能差。

---

## 五、参考资源与真实项目

- **Textual 官方文档**：框架介绍、组件库、教程 —— https://textual.textualize.io/
- **Rich 官方文档**：Console/Panel/Live/Markdown —— https://rich.readthedocs.io/
- **Anatomy of a Textual UI**：官方用 Textual 聊 AI agent 的实战文章 —— https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/
- **Aider**：终端 AI 编程助手（用 Rich），47K+ stars —— https://github.com/Aider-AI/aider
- **Instrukt**：终端 AI 助手 TUI —— https://github.com/dosisod/instrukt
- **Awesome-TUIs**：最全 TUI 库清单 —— https://github.com/rothgar/awesome-tuis

---

## 六、总结

| 需求 | 选择 |
|---|---|
| 完整交互式的漂亮 Agent 控制台 | **Textual**（现代、异步、流式 Markdown，首选） |
| 极简的漂亮命令行输出 | **Rich**（几行代码搞定，零负担） |
| 核心逻辑与 UI 解耦、多端复用 | 两者配合 + 架构分层 |

### 示例代码已就绪

所有示例均可直接运行，已放置在 `/workspace/examples/`：

| 文件 | 说明 | 运行方式 |
|---|---|---|
| `rich_agent_output.py` | 轻量流式渲染示例 | `python3 rich_agent_output.py` |
| `textual_agent.py` | 交互式 Agent TUI 示例 | `python3 textual_agent.py` |

两个示例均已在当前环境安装依赖（Rich 14.3.3 / Textual 8.2.7）并验证可运行，Textual 版本还通过 `textual.testing.run_test` 完成了流式渲染验证。
