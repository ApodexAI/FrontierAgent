# 持久多任务基础

`frontier-agent tasks` 提供本机常驻任务服务和独立的任务看板。任务、执行尝试、用户输入、操作回执和恢复上下文保存到 SQLite；任务执行在各自的 Docker 容器中进行。关闭 CLI 或看板不会取消服务中的任务。

这是 [Super Agent 升级方案](super-agent-product-design.zh-CN.md) 的 P0 基础实现。服务复用 ReAct 与 Agent Team，当前由用户明确提交任务。跨入口 Router、GitHub/IM 连接和主动规则属于后续阶段。

## 启动与提交

需要 Python 项目环境、运行中的 Docker daemon，以及包含本次代码的镜像。已有的旧镜像需要重新构建。服务所在主机必须在线。

在仓库根目录构建一次：

```bash
docker build -t apodex:local .
```

模型配置沿用项目 `.env`、用户环境配置和导出的环境变量。服务默认传入 OpenAI、Anthropic、Jina、Serper、Tavily 相关变量及模型和搜索 provider 选择；其他 provider 的变量可以逐项使用 `--forward-env NAME` 传入。密钥值不写入任务数据库或 Docker 命令参数。

建议为每个工作区固定一个绝对状态目录，并在所有命令中使用它。下面在当前项目中操作，状态目录默认为 `.apodex/tasks`：

```bash
# 启动独立后台服务；任务 Worker 最多同时运行两个
uv run frontier-agent tasks start --image apodex:local --concurrency 2

# 提交 ReAct 任务，返回完整任务 ID
uv run frontier-agent tasks submit "分析这份材料并生成有依据的报告" \
  --input ./docs/framework.md --request-key research-001

# 使用 Agent Team 执行另一项任务
uv run frontier-agent tasks submit "比较两个设计并给出实现建议" \
  --workflow agent_team --input ./docs --request-key design-001

# 查看任务；打开独立 TUI 看板
uv run frontier-agent tasks list
uv run frontier-agent tasks ui
```

也可以使用 `tasks serve` 在前台运行，供终端调试或进程管理器托管。`start` 的启动日志位于状态目录中的 `service.log`；启动失败时直接报告该路径。部署时可用操作系统服务管理器在主机重启后自动运行 `serve`。

`--input` 将明确选择的文件或目录复制为只读快照，总量上限为 256 MiB，同名输入和符号链接会被拒绝。目录快照跳过 `.git`、`.venv`、`.apodex`、`node_modules` 和 Python/pytest 缓存。快照成功后任务才进入可执行队列。用户在提交后修改原文件，不影响已有任务。

Worker 通过 `/inputs` 读取快照，在 `/workspace` 工作，在 `/outputs` 保存成果。需要修改附带的项目时，应先复制到工作目录；不会直接修改用户原项目。成果位于状态目录的 `tasks/<ticket-id>/outputs/`。

## 任务操作

将以下命令的 `TASK_ID` 替换为提交时返回的完整 ID：

```bash
uv run frontier-agent tasks show TASK_ID
uv run frontier-agent tasks events TASK_ID
uv run frontier-agent tasks input TASK_ID "补充约束：兼容现有调用方式" --request-key feedback-001
uv run frontier-agent tasks pause TASK_ID
uv run frontier-agent tasks resume TASK_ID
uv run frontier-agent tasks cancel TASK_ID
uv run frontier-agent tasks accept TASK_ID

# 停止服务并保留所有任务；下次启动时恢复可安全继续的工作
uv run frontier-agent tasks stop
```

同一请求键与相同参数重复提交时返回原任务；参数改变则拒绝复用。同一输入键也不会重复添加消息。任务取消或完成后，追加消息不能将其自动唤醒。

| 状态 | 含义与下一步 |
| --- | --- |
| queued | 等待执行容量 |
| running | Worker 正在运行；可追加输入 |
| pausing / cancelling | 正在停止容器；未确认停止前不会重新分配 |
| paused | 已暂停或预算耗尽；可在调整预算后恢复 |
| blocked | 存在结果不明确的操作；需要核对后恢复 |
| failed | 本次执行失败；查看日志后明确重试 |
| ready | Worker 报告成果就绪，等待人验收；追加要求会再次入队 |
| completed | 用户已验收 |
| cancelled | 用户已取消；需要继续时新建任务 |

后台命令通过 `bash` 返回 Job ID，随后用 `job_status` 或 `job_cancel` 查询、取消；完成事件会送回执行循环。普通用户无需手动管理这些 Job。长命令存在时，任务不会仅因主循环输出一段文字就被标记为可验收。

## 预算与恢复

默认每项任务累计运行 3600 秒、200 次 agent loop 模型尝试、最多 3 次执行分配，内部最多并行 4 个子 Agent。服务的 `--concurrency` 控制跨任务并行数，因此需要同时考虑两个并发上限。

```bash
uv run frontier-agent tasks submit "处理任务" \
  --seconds 1800 --model-calls 100 --attempts 3 --parallel-agents 2

# 调整的是整个任务的累计上限，已经消耗的预算不会归零
uv run frontier-agent tasks resume TASK_ID --seconds 7200 --model-calls 400 --attempts 5
```

模型次数计数覆盖接入执行观察器的主循环和 Agent Team 子循环，包括它们的物理请求重试。辅助摘要、语言识别等循环外调用不在该次数指标内；它不是整个平台的精确调用或费用上限。token 仅记录可观测到的响应使用量，不作为硬额度。整个 Worker 的运行时间和容器 CPU、内存及进程限制提供外层约束。

异常退出时若没有已确认的结束时间，时长按分配开始至恢复核对的时间保守计入，最多计到该任务的剩余时长上限；长时间停机后可能需要手动增加额度。这一计数用于停止执行，不是费用账单。

恢复时，服务先停止上次分配对应的具名容器，确认停止后才允许新分配。执行者发送的状态必须匹配当前分配、服务身份、有效租约和任务状态。旧 Worker 的迟到消息不能推进任务。

每次工具调用先持久化意图，控制服务确认后才执行；执行结果再持久化确认。如果在这两次确认之间中断，系统无法断言外部动作是否成功，任务会进入 `blocked`。查询 `show` 的 `operations` 和已有成果，核对后明确记录结论：

```bash
uv run frontier-agent tasks resume TASK_ID \
  --reconcile "已核对目标系统，该操作成功；后续只验证现有结果，不再次提交"
```

恢复采用“重新进入 workflow，携带目标、全部用户输入、近期回合上下文、操作结果和已有文件”的方式。它不会还原所有 Python 协程或逐个复活原有子 Agent。已确认的最终报告也会被恢复，不会仅因为服务在更新最终任务状态前崩溃而再次执行。

取消会立即拒绝新的 Worker 操作请求，随后停止执行环境。正在进行的命令可能已有部分副作用，取消不代表回滚。服务失联时 Worker 的控制心跳会终止执行；控制通道没有确认时，操作不能继续放行。

## 实现边界

- 状态目录属于一个本机 OS 账户的信任范围，CLI 和 TUI 直接通过事务访问 SQLite。当前没有远程 HTTP API、多用户认证或多副本调度。
- 服务使用文件锁确保一个状态目录只有一个调度者；数据库事务、唯一索引和执行租约提供额外约束。
- 每个 Worker 的容器只挂载本任务的工作目录、成果目录和只读输入。不会挂载控制数据库、Docker socket 或其他任务的目录；没有 native 执行回退。
- 网络仍可访问所配置的模型与工具服务；当前没有按任务划分的网络域名策略或外部发布审批系统。
- 队列中的任务即持久化的启动意图，Assignment 是执行记录；这里尚未实现跨平台消息发送 outbox。
- `frontier_agent.runtime.WorkflowSession` 是独立于评测目录的产品运行入口，旧 `BenchmarkSession` 保留为兼容导入。旧 CLI 运行方式和 benchmark 的内存状态语义保留。
- 历史和文件目前不自动清理；数据库备份与成果保留策略由部署方管理。不要在服务运行时直接删除状态目录。

## 验证

```bash
uv run pytest tests/test_superagent_store.py tests/test_superagent_service.py tests/test_superagent_worker.py -q
uv run python tools/import_smoke.py
uv run pyright
```

测试覆盖数据库重开、并发去重与领取、租约过期、旧结果拒绝、取消、恢复预算、未确认操作、报告与新输入的竞争、后台 Job、主循环和子循环观察器，以及 CLI/TUI。子进程协议测试使用确定性 Worker，workflow 测试使用模型替身；运行真实 Docker 容器与真实模型仍需要对应部署环境。
