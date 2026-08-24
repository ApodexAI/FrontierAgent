# FrontierAgent TUI 使用教程

[English](tui-user-guide.md) · [安装与 endpoint 配置](install/tui-endpoint-quickstart.zh-CN.md) · [文档索引](README.md)

本教程从 TUI 已经打开的状态开始，介绍界面布局、右侧四个 Tab、文件预览、
审批，以及 `agent_team` 运行时的异步干预。还没有启动 TUI 时，请先完成
[macOS / Linux 开箱即用教程](install/tui-endpoint-quickstart.zh-CN.md)。

## 1. 先认识界面

```text
┌──────────────────────────────────────────────────────────────────────┐
│ FrontierAgent · workflow · session · workspace                 F2   │
├──────────────────────────────────────────────┬───────────────────────┤
│                                              │ Plan Activity Files Diff│
│ 对话与执行记录                               │                       │
│                                              │ 当前 Tab 的内容       │
│ 用户任务、思考、工具调用、审批结果和最终报告 │                       │
├──────────────────────────────────────────────┴───────────────────────┤
│ 状态 · 耗时 · workflow · model · context · tools · queued           │
│ 附件（如果有）                                                       │
│ 输入任务或运行中的补充指令                                           │
└──────────────────────────────────────────────────────────────────────┘
```

顶部显示当前 workflow、会话名和工作目录。左侧是完整 transcript；右侧是工作
状态；底部状态栏显示当前阶段、耗时、模型、上下文余量、工具数量、等待处理的
干预数量，以及本次会话的文件变更统计。终端宽度小于 100 列时，右侧栏会自动隐藏；加宽终端或按 `Ctrl-B` 恢复。

> 截图位 1：完整 TUI 首页。建议同时标出顶部信息、左侧 transcript、右侧
> Tab、状态栏和输入框。

## 2. 右侧各个 Tab 分别做什么

用 `Ctrl-Tab` 按 `Plan → Activity → Files` 循环切换；`Ctrl-Shift-Tab`
反向切换。当本次会话产生文件变更后，可点击的 `Diff` Tab 才会出现，并加入
`Files` 之后的切换循环。

### Plan：计划与任务看板

`Plan` 回答“现在准备做什么、做到哪一步了”。

- `react` 模式中，它显示 Agent 通过 todo 工具维护的步骤；
- `agent_team` 模式中，它显示 coordinator 创建和更新的任务；
- 图标区分待处理、进行中、完成和取消状态；
- Tab 标题中的 `已完成数/总数` 可以在内容滚动后继续提供整体进度；
- 新任务尚未创建计划时会显示 `no plan yet`，这不代表程序异常。

Plan 是只读状态面板，不需要把键盘焦点移进去。提交新任务时，右侧会自动回到
Plan，方便观察任务拆解。

> 截图位 2：`agent_team` 已创建任务后的 Plan Tab，最好包含不同任务状态和标题
> 中的完成进度。

### Activity：工具与子 Agent 的实时活动

`Activity` 回答“现在谁在做什么、工具是否成功、花了多久”。每行会显示状态图标、
名称、持续时间和摘要；成功、失败、跳过、中断和仍在运行都有不同状态。

进入 Activity 后：

1. 用 `↑` / `↓` 选择一行；
2. 对普通工具调用按 `Space`，查看 state、duration、call ID 和 details；
3. 再按 `Space` 或 `Esc` 关闭详情；
4. 按 `Esc` 从列表返回底部输入框。

在 `agent_team` 模式中，Activity 还会分为两组：

- `SUB-AGENTS`：每个后台 worker 的 queued、running、ready、failed 等状态；
- `COORDINATOR`：主协调器自己的工具调用。

选中一个 sub-agent 行并按 `Space`，会在原位展开或收起它的 thinking、message、
tool call、tool result 和 error 事件；选中展开后的事件再按 `Space` 可打开详情。
组标题会显示 worker 总数，以及 live / failed 数量。每个 worker 还有由名称推断的
专业标记和稳定颜色，便于并行运行时快速区分。

> 截图位 3：Agent Team 的 Activity Tab。建议让 `SUB-AGENTS` 和
> `COORDINATOR` 同时可见，并展开一个 sub-agent。

### Files：本次会话的交付物

`Files` 回答“Agent 最终产出了哪些文件”。按 `Ctrl-O` 可直接在 Files 和 Plan
之间切换；任务完成并产生输出时，TUI 也会展示 Files。面板顶部显示宿主机交付物
路径；Docker 模式下还会同时显示 Agent 看到的挂载路径。单独的 `Work:` 行指向
本次 run 的中间工作目录；其中内容不会混入正式交付物列表。

在 Files 中：

1. 用 `↑` / `↓` 选择文件；
2. 按 `Space` 打开只读预览；
3. 用 `Ctrl-U` / `Ctrl-D` 上下翻页；
4. 按 `Space` 或 `Esc` 关闭预览；
5. 在文件列表按 `Esc` 返回 Plan。

TUI 可预览常见源码和纯文本、Markdown、CSV、PDF、Word、Excel、PowerPoint、
Jupyter Notebook、图片、压缩包以及部分 3D / 生物结构格式。某些 Office、PDF
或图片格式需要可选 reader 依赖；不支持的类型仍会保留在 Files 中，但不能在
TUI 内打开。预览有大小和行数上限，完整内容以磁盘文件为准。

最终回答会自动保存为 `final-report.md`。Native 模式下，默认输出目录是：

```text
<你的 --cwd>/.apodex/runs/<session-id>/outputs/
```

> 截图位 4：Files 列表与一个 `Space` 打开的预览。代码、Markdown 或图片最能
> 展现预览效果。

### Diff：本次会话的文件变更

`Diff` 用 unified diff 展示本次会话在磁盘上产生的变更，并按主题颜色区分新增与
删除行；没有变更时该 Tab 会隐藏。任务完成后，有变更会自动打开 `Diff`，否则打开
`Files`。对同一文件进行多轮修改时，比较基线始终是本次会话第一次触碰该文件前的
内容；因此仓库中原本存在、与 Agent 无关的脏改动不会被算到 Agent 名下。新建和
删除文件会使用常见的 `/dev/null` header。

变更有两个来源。文件工具（`write_file`、`file_editor` 系列、`delete_file`）会
显式给出目标路径，直接快照即可。而非只读的 `bash` 命令不声明写入目标，因此在
工具阶段前后各扫描一次工作目录和本次会话的 outputs 目录，只保留真正发生变化的
文件——shell 脚本、`sed -i`、脚本生成的文件、`rm` 删除都会出现在这里。二进制和
超大文件因为留不下可比较的基线，会被跳过而不是给出错误的结果。

**`/revert` 只撤销第一类。** 扫描只能看出"树变了"，看不出"是谁改的"：同一时间
窗口里其他东西的写入——你自己的编辑器、watcher、dev server——和 shell 命令无法
区分，把它们一起还原会毁掉本次会话根本没做过的工作。因此扫描发现的文件在这里
展示，但会原样留在磁盘上，`/revert` 会单独列出它们交给你自己处理。

Tab 标题显示变更文件数；pane 顶部和底部状态栏用绿色 `+新增行数`、红色
`-删除行数` 展示总计。内容在后台刷新，方向键、`PgUp`/`PgDn`、`Home`/`End` 可以
滚动，按 `Esc` 返回 Plan。

## 3. 一次任务的标准操作流程

在底部输入任务并按 Enter：

```text
阅读当前仓库，找出测试入口和最重要的模块，生成一份 Markdown 架构说明。修改前先说明计划。
```

运行时建议按这个顺序观察：

1. 在 Plan 查看拆解和进度；
2. 在 Activity 查看工具调用；Agent Team 模式下可展开 worker；
3. 出现审批窗口时检查命令或 diff，再决定是否执行；
4. 如需调整方向，直接在底部输入普通文本并按 Enter；
5. 完成后按 `Ctrl-G` 跳到最终报告，或按 `Ctrl-O` 查看文件；
6. 在同一输入框继续提问，会保留当前会话上下文和已经修改的 workspace。

## 4. 运行中异步干预（steer / intervene）

当状态栏显示 Agent 正在工作时，仍可在输入框键入普通文本。例如：

```text
先不要改代码；把重点改为分析测试覆盖缺口，并在报告中给出证据。
```

按 Enter 后，TUI 会显示 queued 提示，状态栏中的 `queued` 或 `q` 数量增加。
这条消息会在下一个安全的 agent turn 边界作为新用户指令注入。

需要理解它的协作语义：

- 它是异步排队，不会终止正在进行的 LLM 请求或工具调用；
- 在 `react` 中，指令交给当前 Agent；
- 在 `agent_team` 中，指令交给 coordinator；已经派出的 sub-agents 不会被直接
  打断，coordinator 可在下一轮调整后续委派、校验重点或最终综合；
- 如果消息到达得太晚、当前任务已经结束，它不会丢失，而会作为紧接着的 follow-up
  任务运行；
- 多条干预会排队，因此指令要短、明确，并在后续消息中说明是否取代前一条；
- 运行中输入 slash command 不会排队执行。TUI 会提示先按 `Ctrl-C` 中断。

如果目的是立即停止当前任务，请按 `Ctrl-C`，不要发送“停止”。TUI 会保存到最近
一个已完成 turn 的状态；空闲时再按 `Ctrl-C` 会退出应用。

> 截图位 5：Agent Team 正在运行、输入一条干预后，transcript 显示 queued 提示，
> 状态栏显示 `queued 1`，Activity 中仍有 worker 运行。

## 5. 审批窗口怎么选

写文件或执行需要确认的命令时，TUI 会显示目标、原因，以及命令或 diff 预览。
默认选中 `No`，避免误按 Enter 执行。

| 按键 / 选项 | 含义 |
|---|---|
| `y` | 仅批准这一次 |
| `n` 或 `Esc` | 拒绝 |
| `m` | 本会话中自动批准 bash；只用于 Docker 或可信环境 |
| `a` | 本会话允许所有普通审批 |
| `A` | 持久记住并允许这一类命令 |
| `e` | 拒绝当前操作，并输入替代指令 |
| `Ctrl-U` / `Ctrl-D` | 翻阅较长的命令或 diff |

高风险操作不会接受单键 `y`，必须完整输入 `yes`。首次使用建议逐次审批；Native
runtime 不是操作系统沙箱，获批命令拥有当前用户权限。

## 6. 阅读和整理长 transcript

| 操作 | 用途 |
|---|---|
| `Alt-J` / `Alt-K` | 在可见 transcript block 间移动 |
| `Alt-Enter` | 展开或收起当前 thinking / process block |
| `Ctrl-G` 或 `/report` | 跳到最新最终报告 |
| `Ctrl-Y` 或 `/copy` | 复制最新最终报告 |
| `/filter thinking` | 只看 thinking |
| `/filter tools` | 只看工具调用与结果 |
| `/filter errors` | 只看错误 |
| `/filter report` | 只看最终报告 |
| `/filter all` | 恢复全部内容 |
| `/find <文字>` | 搜索 transcript |

很长的会话会隐藏较旧的渲染 block 以保持 TUI 流畅，但完整历史仍保存在会话中。
上下文接近上限时可运行 `/compact` 压缩较早内容。

## 7. 附件、会话和 workflow

常用命令：

```text
/attach <path>       把文件或目录复制为本会话的只读输入
/attachments         列出附件
/detach <name>       删除附件副本，不影响源文件
/workflow react      切换为单 Agent 顺序执行
/workflow agent_team 切换为 coordinator + 并行 sub-agents
/new                 保存当前会话并开始空白会话
/fork                从当前上下文分叉一个新会话
/rename <name>       给会话命名
/resume              选择历史会话继续
/context             查看上下文与 token 使用量
/config              查看脱敏后的 endpoint / model 配置
/log                 查看 JSONL trace 路径
/revert              撤销本会话经文件编辑工具记录的改动
```

输入 `@` 加文件名的一部分，可以搜索显式附件和当前 `--cwd` 下的文件，再按
`Tab` 补全引用。工作区文件已经挂载，不会重复复制到附件区。macOS 可用
`Ctrl-V` 或 `/paste` 读取 Finder 文件和系统剪贴板图片；Linux 请使用
`/attach <path>`。`--input` 和 `/attach` 的相对路径都从当前 `--cwd` 开始解析，
可以引用其任意层级子目录中的文件或目录；绝对路径仍然支持。切换 workflow
会重置对话上下文，因此应在新任务开始前切换。

从终端直接粘贴多行或超长文本时，输入框会显示紧凑的 `[Pasted text …]` 标记；
按 Enter 后会展开并把包含原始换行的完整文本一次性发送给 agent。

## 8. 推荐的模式选择

- 代码定位、单仓库修改、连续追问：先用 `react`；
- 可拆成多个独立调查、需要交叉验证或综合多份报告：用 `agent_team`；
- Agent Team 会并行调用 endpoint，通常更快地产生多角度证据，但 token 和并发
  消耗也更高；
- 是否使用 Native、bubblewrap 或 Docker 是 runtime 选择，与 `react` /
  `agent_team` workflow 选择相互独立。

随时按 `F1` 查看内置快捷键，按 `F2` 打开主题、workflow、行为、权限和会话设置。

需要直接用于截图和演示的英文任务，可使用
[TUI Demo Queries](tui-demo-queries.md)，其中第三组包含 mock attachments 和
Agent Team 异步干预步骤。
