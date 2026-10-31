# ADR-001：Agent 编排层的框架选型

- **状态**：已接受（Accepted）—— 采纳 A（手写补齐），C 已评估后否决
- **日期**：2026-10-30
- **决策范围**：在线编排层 —— chat 的 ReAct 循环、deliverable 的规划-执行流程、工具与状态管理
- **不在范围**：检索链、入库管道、LLM 客户端、评测口径（这三块已在 `ARCHITECTURE.md` 与 `EVALUATION.md` 定型，见附录 B）
- **可复核性**：本 ADR 不引用任何未落盘的数字；文中代码事实均可在当前 HEAD 上复核（`git log`）

---

## 1. 背景

JD 侧的事实（30 秒版）：

- 2026 年国内 AI 应用开发岗的 JD 里，LangChain / LangGraph / LlamaIndex 出现频率高，措辞以「至少一种」「优先」「加分项」为主；
- **RAG 已经从"能力项"降为"基础设施项"** —— 被当作前提，而不是卖点；
- 唯一把 RAG 写成硬门槛的 JD，写的是「有主导建设完整 RAG 系统的实践经验，**而非仅限于组件调用**」。

项目侧的事实：

| 事实 | 证据 |
|---|---|
| 两条在线路径形态完全不同：chat 是单轮 ReAct（库内应答 **6~8s**，SSE 流式），deliverable 是规划 + 3–6 个子任务（**40–60s，同步阻塞**） | `ARCHITECTURE.md` 第三节、`app/main.py` 的 `/api/deliverable` 注释 |
| chat 有 SSE 事件契约、五条终止路径的 FakeLLM 测试与故障注入，另有 848 项测试整体兜底 | `backend/tests/`、`EVALUATION.md` |
| deliverable 只有纯函数层测试（36 条），**编排层零测试** | `test_deliverable_pure.py` |
| deliverable 是串行执行：子任务数由规划器决定（`PLAN_PROMPT` 要求 3–6 个），每个子任务 = 2–4 次检索 + 1 次抽取 | `deliverable.py` 的 `run()` / `PLAN_PROMPT` |
| 生产路径零框架依赖：`/api/chat`、`/api/deliverable` 的 import 图里没有 langchain / llama-index；`langgraph` 仅被实验 G 的对照臂引用 | `backend/requirements.txt`、`react_langgraph.py` |
| 检索、入库、LLM 客户端都是针对性实现，不是通用胶水 | `hybrid.py`、`ingest/`、`llm/client.py` |

---

## 2. 问题拆解

真正要回答的不是「用不用框架」，而是：

> **哪一层需要框架的哪种能力，代价是多少，不做的损失是什么？**

拆成四个可独立判定的子问题：

- **Q1**：chat 的 ReAct 循环要不要换成 LangGraph？
- **Q2**：deliverable 的编排要不要换成 LangGraph？
- **Q3**：检索 / 入库要不要换成 LlamaIndex 或 LangChain 的组件？
- **Q4**：要不要为了 JD 关键词迁一层？

---

## 3. 判据

| 编号 | 判据 | 类型 | 说明 |
|---|---|---|---|
| **H1** | chat 路径的可控性不下降 | 硬 | 四道安全阀、SearchGuard 熔断、SSE 事件契约、P50 延迟 |
| **H2** | A–F 六组实验的可比性不被破坏 | 硬 | 同一份语料、同一套评测口径；`_meta` 追溯不能断 |
| **H3** | 迁移必须买到「手写拿不到，或总成本（含长期维护）明显更高」的东西 | 硬 | 判据是**总成本**，用 §11 的定价表衡量，不预设天数 —— 并行、恢复、审批手写都做得到 |
| **S1** | 迁移成本 ≤ 2 周 | 软 | 单人、投递期，不能改到一半不能演示 |
| **S2** | 面试可讲且可验证 | 软 | 有代码、有数字、能解释取舍 |
| **S3** | 依赖增量可控 | 软 | 可 pin、可卸载，不污染现有核心依赖（fastapi / openai / lancedb / SQLAlchemy） |

---

## 4. 选项与评估

| 选项 | H1 | H2 | H3 | S1 | 结论 |
|---|---|---|---|---|---|
| **A. 全手写补齐** | ✅ | ✅ | 并行已做（0.5 天）；恢复 / 审批 / 可观测已于 2026-10-31 补齐（实际合计 ≈3.5–4 天） | 原估 4–5.5 天，实际 ≈3.5–4 天 | **采纳（已全部落地）** |
| **B. chat 迁 LangGraph** | ⚠️ SSE 事件需要重放、消息链要逐字对齐 | ⚠️ 生成侧 C/E 需重跑才能说"没变" | ❌ chat 不需要 checkpoint，收益接近 0 | ~2 周 | 拒绝 |
| **C. deliverable 迁 LangGraph，chat 冻结** | ✅ 不涉及 | ✅ 不涉及 | 只买标准状态语义 + 图级 trace（并行已由手写拿到，见 §13） | 8–10 天 | **已评估后否决** |
| **D. 全量迁 LangChain + LlamaIndex** | ❌ | ❌ 入库 / 检索一变，A–F 全部失效 | — | > 4 周 | 拒绝 |

### 4.1 为什么 chat 不迁（Q1）

chat 的需求是：**低延迟、精确停机、事件按固定顺序外发**。这三件事手写内核已经做到，而 LangGraph 的卖点（持久化、恢复、人机协作、多步编排）在这里一个都用不上。迁过去要付出的代价是具体的：

- SSE 事件流要重新映射（`token` / `tool_start` / `tool_end` / `done` 的成对与顺序约束）；
- 消息链格式要逐字对齐，否则「两版输出逐字一致」这类守卫会成片变红 —— 而它们正是证明「只有循环实现不同」的唯一手段；
- 生成侧实验 C/E 需要重跑，否则"没变"这句话没有证据；
- LangGraph 1.2 起 prebuilt 的 `create_react_agent` 已迁出 LangGraph（到 `langchain.agents`），"开箱即用的 ReAct"这条路本身已经不存在 —— 迁过去同样是手搭节点，收益进一步被压缩。

**收益为 0、风险为实**，因此冻结。

### 4.2 为什么 deliverable 迁（Q2）

`DeliverableAgent` 现在是一个**手写的 Plan-and-Execute**：规划 → 逐个 `_run_subtask` → 合并 → 冲突检测 → 渲染。它的问题不是"写得不好"，而是三件事它当时没有：并行、跨进程的恢复、执行前的审批。

> **状态说明（2026-10-31）**：下表是决策当时（2026-10-30）的缺口快照；
> 四个缺口现已全部由手写补齐 —— 并行、文件快照恢复、审批、SSE 事件流，
> 见 §12 的状态列。本节保留当时的判断，不复写历史。

| 缺口 | 现状 | 迁后的能力 |
|---|---|---|
| **并行** | 子任务串行，40–60s | `Send` fan-out，受**新增的**子任务级闸门约束；收益待实测（40–60s 里远程调用占多少未知，并行还可能降低 KV cache 命中） |
| **恢复** | 进程一挂，已完成子任务的成本全丢，且不知道跑到哪了 | `thread_id` + checkpoint，从失败节点续跑 |
| **审批** | 一次请求直接花掉全部子任务的预算 | `interrupt` 停在计划后，人改完再执行 |
| **可观测** | 只有最终 `elapsed_ms` | 每个 `subtask_start/end` 都能外发，前端可做进度 |

这三项手写也做得到（并行几十行；恢复与审批约 2.5–3 天，探针估算待核实）—— 所以迁移的正当性**不是"别人做不到"**，而是"用框架换一套标准状态语义、更少的自有并发/恢复代码、以及图级可观测"，即 H3 重述后的**总成本**判据。且不触碰 chat，满足 H1/H2。

**框架省不掉的部分**：审批卡片、进度条、审批态交互都在 `DeliverablePanel.tsx`（§7.1），手写与框架**一模一样**。迁移真正省的只有并发调度与 checkpoint 样板（各约 0.5 天）。净收益必须扣掉这块，再与"标准语义 + 图级可观测 + 生态证据"相加 —— 这是成本结构论证，比"能不能"更难被反驳。

### 4.3 为什么检索 / 入库不迁（Q3）

见附录 B。一句话：**LlamaIndex 能替的只有"切块 → embed → upsert"约 40 行，而模板展平、按小节切分、两级指纹、三级增量，它一样都没有；换掉等于把 A–F 的可比性前提一起换掉。**

---

## 5. 决策

**采纳 A（手写补齐），选项 C 降级为「已评估后否决」。** 依据是 §12 的定价表与 §13 的证据。

1. **chat 路径冻结**：`app/agent/react.py` 只允许注释级修正，不允许行为变更；生产入口 `/api/chat`、`/api/chat/stream` 继续挂 `ReActAgent`。
2. **deliverable 不迁移**：在现有 `DeliverableAgent` 上补齐四个缺口 —— 并行（已完成，`ParallelDeliverableAgent`）、恢复、审批、可观测；纯函数与数据结构不动。
3. **不引入 LlamaIndex 重写入库**；如需要，另开一个"检索对照臂"实验（附录 B.3），不以 ADR 形式绑定。
4. **不替换 LLM 客户端**：`llm/client.py` 的降级链、`<think>` 剥除、内联工具调用兜底、embedding/rerank 缓存全部保留；不使用 `langchain-openai`。
5. **依赖边界**：生产路径不引入 `langgraph`；`react_langgraph.py` 只作为"评估过框架边界"的对照臂保留，`langgraph` 在 `requirements.txt` 里是**实验依赖**（对照臂守卫需要）。

> 此前那版"采纳 C、以 P1 为准"的决策已被 P1 实测推翻：并行收益手写就拿到了（−47%），四个缺口全部手写完成后框架版仍要多付 4.5–6 天。决策过程保留在 §11 的决策门与 §13；§7 的迁移方案作为"以后重开时可以直接用"的记录保留，不执行。

---

## 6. 后果（若迁移；本轮已否决）

> 本轮决策为**不迁移**，以下保留为"若迁移"的后果分析，供以后重开时参考。

**正向**

- deliverable 从"一次调用跑几十秒的同步接口"变成"可暂停、可恢复、可并行的工作流"，这是能现场演示的东西；
- 面试可讲的不再是"我评估过框架"，而是"我在需要状态与恢复的那一层用了框架，其余层没用，理由是 X"；
- 新增 HTTP 端到端测试（`TestClient`），补上现有 848 项测试里"没有一条真发 HTTP"的空白。

**负向**

- 失去"全仓零框架依赖"这句干净的话（但 chat 冻结 + 本 ADR 把理由写死，叙事仍然成立）；
- 新增一个依赖面：LangGraph 的版本升级、checkpoint 语义、`Send` 与 reducer 的行为都要跟踪；
- 多一层间接：图节点出错时，栈会比直接调用深。

**中性**

- `/api/deliverable` 的对外语义保持不变，前端可以渐进式接入新端点。

---

## 7. 实施边界

### 7.1 动与不动

| 模块 | 动作 | 理由 |
|---|---|---|
| `app/agent/react.py` | **冻结** | 生产 chat 路径 + 848 测试的重心；任何行为变更都会污染 A–F |
| `app/agent/deliverable.py` | 保留全部纯函数与 dataclass；`run()` 拆成节点可调用的方法 | 纯函数是这次迁移唯一稳定的地基 |
| `app/agent/workflow.py` | **新增**：状态图、节点、Send、checkpoint、事件映射 | 不侵入现有模块 |
| `app/agent/workflow.py` 的 fan-out | 新增**子任务级**并发闸门 + `max_concurrency` 配置 | 现有 `_semaphore` 只挡入口请求，不限制图内并行度 |
| `app/main.py` | 新增 3 个端点；`/api/deliverable` 保持同步语义 | 向后兼容 |
| `app/store/persist.py` | 新增 `deliverable_runs` 表（`run_id` / `thread_id` / `phase` / `state_json` / `updated_at`） | 断点续跑与可观测的最小实现 |
| `backend/requirements.txt` | 新增 `langgraph`（固定版本） | 显式声明，不靠环境残留 |
| `frontend/src/components/DeliverablePanel.tsx` | 计划审批卡片 + 子任务进度 | 产品化演示 |
| `app/retrieval/`、`app/ingest/` | **不动** | 与编排无关 |

### 7.2 checkpoint 方案（先自建，后评估）

| 阶段 | 方案 | 理由 |
|---|---|---|
| P1 | `MemorySaver` + 每步结束后把 `state` 快照写 `deliverable_runs.state_json` | 不引入新驱动；恢复靠快照重放；进程内语义完整 |
| P2 | 评估社区包 [`langgraph-checkpoint-mysql`](https://github.com/tjni/langgraph-checkpoint-mysql) | 只有 P1 被证明不够（例如需要跨进程 `get_state`）才引入 |

**不在 P1 引入社区包的理由**：它引入自己的 SQLAlchemy / MySQL 驱动栈，与现有 `aiomysql` 异步栈并存会产生两套连接管理；而"断点续跑"这个能力用一张快照表就能演示，先证明能力、再换基础设施。

### 7.3 目标结构

```
START → route ─(越界)→ refuse → END
             │
             └→ plan → [interrupt：计划审批] → dispatch
                                                ├─Send→ subtask 1 ─┐
                                                ├─Send→ subtask 2 ─┤
                                                └─Send→ subtask N ─┴→ merge → conflicts → render → END
```

状态（`TypedDict`，`results` 用 reducer 合并，天然支持并行）：

```python
class WorkflowState(TypedDict, total=False):
    question: str
    game: str
    topic: str
    subject: str
    plan: list[dict]                                   # [{name, goal, queries}]
    results: Annotated[list[SubtaskResult], operator.add]
    conflicts: list[Conflict]
    markdown: str
    payload: dict
    phase: str
    errors: list[str]
```

**约束**：图里只放"会失败、需要重试、需要暂停"的步骤。纯函数（`render_markdown` / `detect_conflicts` / `dedupe_same_page_facts` / `subject_matches`）继续留在 `deliverable.py`，不做成节点 —— 它们没有状态，做成节点只会让测试和调试变难。

### 7.4 API 契约

| 端点 | 语义 | 返回 |
|---|---|---|
| `POST /api/deliverable/plan` | 只做规划，停在审批点 | `{run_id, thread_id, plan, subject, phase}` |
| `POST /api/deliverable/{run_id}/resume` | 带（可修改的）计划继续执行 | SSE 流或 `202 + stream_url` |
| `GET /api/deliverable/{run_id}/stream` | 进度事件流 | SSE |
| `POST /api/deliverable` | **保留**：内部走 plan → 自动审批 → 执行 → 返回最终产物 | 现有响应结构不变 |

事件词汇沿用现有 SSE 惯例（`event: <name>` + JSON data）：

| 图内事件 | SSE event | 前端行为 |
|---|---|---|
| plan 节点产出 | `plan_ready` | 渲染计划，显示「批准 / 修改」 |
| 每个 Send 开始 | `subtask_start` | 进度 +1，显示子任务名 |
| 每个 subtask 返回 | `subtask_end` | 打勾或标红，带 `status` / `facts` 数 |
| merge 完成 | `merge_done` | 显示事实总数、冲突数 |
| 图结束 | `done` | 渲染 Markdown / JSON，带 `elapsed_ms` |

### 7.5 失败与恢复语义

- **单子任务失败**：该子任务返回 `status=error` 的结果，不阻塞其他子任务（现有 `_run_subtask` 已是这个语义，保持不变）；
- **进程崩溃 / 重启**：从 `<DATA_ROOT>/runs/deliverable/<run_id>.json` 的快照续跑；已完成的子任务按 **子任务名** 匹配直接复用，不重跑（v1 用文件，不依赖 MySQL）；
- **快照粒度**：串行版每完成一个子任务落一次；并行版每批 `gather` 返回后落一次（崩溃时最多丢当前这一批）；
- **API**：`POST /api/deliverable/{run_id}/resume` 续跑、`GET /api/deliverable/{run_id}` 查进度；续跑返回结构与 `/deliverable` 一致；
- **审批（HITL）**：`POST /api/deliverable/plan` 只规划（`phase=awaiting_approval`，不检索不抽取）；`POST /api/deliverable/{run_id}/approve` 按原计划或**改过的计划**执行；已 `done` 的 run 再审批返回 409，避免"审批"变成第二次重跑；
- **可观测**：`POST /api/deliverable/stream` 与 `/{run_id}/approve/stream` 边跑边发 `plan → subtask_start → subtask_end → result`；事件回调失败只记日志，不打断执行；SSE 断线后任务照常跑完并落快照，可用 resume/status 接回；
- **快照缺失 / 损坏**：存储层按"没有快照"处理（重跑），不把 500 抛给调用方。

---

## 8. 验收标准

验收必须分两层，**不能混用**（这条规则由 §11 P1b 的 A/A 实测逼出来）：

- **控制流层（FakeLLM，离线）**：同一批脚本化输入，两版实现可以要求**逐字相同** —— 这是唯一能这么要求的层；
- **行为层（真实 API）**：同一题跑两遍本身就会不同。A/A 实测 15 题里「逐字相同」**0/15**，归一化后仍 **0/15**；可用的只有比率：步数相同 **12/15**、工具序列相同 **10/15**。

| 项 | 标准 | 测法 |
|---|---|---|
| 并行收益 | **逐题配对比值的中位数 ≤ 0.60**（同一题内串行 vs 并行，消掉题目难度差异）；轮级"中位数之比"只作参考，不作验收 | 6 题 × 3 轮 ABBA（n=18），产物 `exp_deliverable_parallel{,_r2,_r3}.json` |
| 控制流等价 | 同一批 FakeLLM 输入，两版 markdown / json **逐字相同** | P0 基线（17 条，§11） |
| 行为等价（真实 API） | 归一化后**不劣于同实现基线**（逐字等价在真实 API 下不可达 —— A/A 同实现重跑也是 0/15，故只能比「比率」与「差值」）；步数 / 工具序列相同率不低于 A/A 基线（12/15、10/15）；**judge ｜Δ｜ 低于噪声底 → 判「无可测差异」**；若声称有差异，须**超过**噪声底才可归因 | **已结账（2026-10-31，实验 G，sr 15 条，配对 n=13）**：步数相同率 **12/15**（= 基线，达标）；**judge ｜Δ｜=0.077 < 噪声底 0.357** → 无可测差异；**工具序列一项未评** —— 脚本未记录该字段，事后补不出来；逐条见 `docs/framework_comparison.md`「验收逐条结账」；噪声底见下 |
| 成本可解释 | token 变化必须实测：n=18 实测 **+0.5%**（**prompt+completion 合计** 330,389 → 332,119；拆开看 prompt **+0.86%**、completion **−2.39%**；单轮在 +4.6% ~ −4.7% 之间波动，属噪声）。⚠️ 合成一个「token」会盖住机制信号：涨的是 **prompt** token（并发下 KV cache 命中下降），这正是本行要盯的那一项；并发度上调后再看 KV cache 是否退化 | 同批任务三轮合计 usage |
| 断点续跑 | 杀掉进程后 resume 能完成，且已完成的子任务调用次数为 0 | 计数断言（FakeLLM / 打点） |
| chat 未受影响 | `/api/chat` 与 `/api/chat/stream` 的事件结构与答案不变；`main.py` 不 import 新模块 | 源码断言 + 现有 848 项测试 |
| 测试 | 新增 ≥ 25 项：图结构、并行、interrupt/resume、快照序列化、HTTP 端到端 | pytest |
| 静态检查 | `ruff check .` 全绿 | CI |

**明确不做**：不为了让数字好看而重跑 A–F；不把 deliverable 接进 chat 的评测口径。

> **噪声底（已补齐，不再是区间）**（`data/eval/exp_G_aa_noise.json`，sr 15 条，2026-10-30）：
> 失败留痕字段已补上 —— `JudgeScore.as_dict()` 的键是 `reason`（**不是** `judge_reason`；
> 读错键名会**静默**拿到 None，失败在产物里零留痕）。现在每侧都记着 `reason` 与 `failed`，
> `judge_error:` 前缀与 `exp_common.is_kept` 同源。
>
> 本轮 15 对里 **8 对 Judge 调用超时**（sr-002 / sr-008 / sr-023 / sr-025 / sr-028 /
> sr-009 / sr-031 / sr-030，其中 sr-008 两遍都超时），**剔除后只剩 7 对有效**：
> 平均 |Δ| = **0.357**、中位 0.000、P90 1.5、最大 **1.5**；4 对同分、3 对异分。
>
> ⚠️ **这个 n 偏薄**（Judge 侧超时率 53%，是当晚 SiliconFlow 过载所致），
> 引用时必须连 n 一起写。它支撑的是「**0.1 分量级的差异读不出来**」这个结论，
> **不是**「抖动幅度恰好等于 0.357」。
> 纯 Judge 抖动（同答案判两遍）的独立测量脚本在 `backend/scripts/probe_judge_noise.py`
> —— 当晚 SiliconFlow 过载，三对全超时，未能跑出可用样本，故此处只引用 A/A 的数。

---

## 9. 风险与回退

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| LangGraph 版本升级破坏图语义 | 中 | 高 | 固定版本；升级前先跑图结构测试 |
| 并行触发上游 429 | 中 | 中 | **新增子任务级闸门**（现有 `_semaphore` 只挡入口请求）；`max_concurrency` 可配 |
| checkpoint 快照与图状态不一致 | 中 | 高 | 快照带 `phase` + `state_version`；resume 前校验 |
| 迁移期间无法演示 | 中 | 高 | `/api/deliverable` 保持旧实现可切换（`DELIVERABLE_ENGINE=legacy\|graph`） |
| 叙事反噬："为了 JD 改的" | 低 | 中 | 本 ADR + chat 冻结就是回答 |

**回退开关**：`DELIVERABLE_ENGINE=legacy` 一行环境变量回退到现有 `DeliverableAgent`，保留两条路径直到验收达标。

---

## 10. 复查条件

出现以下任一情况时重开此 ADR：

1. LangGraph 引入的额外开销超过单次交付物总耗时的 15%；
2. 社区 checkpoint 包或 LangGraph 主线出现破坏性变更，且升级成本 > 3 天；
3. 生产环境要求 Postgres / 多实例，快照表方案不够用；
4. 面试反馈显示"手写内核"这条叙事已成为负分项（目前判断不是）；
5. §11 的 P1 定价表原估手写剩余 4–5.5 天；实际 2026-10-31 四个缺口全部完成（≈3.5–4 天）→ C 已降级为「已评估后否决」；若将来需要跨实例恢复或图级 trace，按 §13 的"什么情况下会改用"重开。

---

## 11. 实施顺序（含决策门）

| 阶段 | 内容 | 成本 | 决策门 |
|---|---|---|---|
| **P0** | 补 deliverable 编排层行为基线：FakeLLM 驱动，钉住「规划 → 子任务 → 合并 → 冲突 → 渲染」的现有产出 | 0.5–1 天 | 无 —— 迁不迁都要做；没有它，"恢复"只是重跑、"审批"没有可断言的状态、"可观测"没有事件顺序 |
| **P1** ✅ | 并行实测（6 题 × 3 轮 ABBA，n=18）+ 三个缺口的代码级成本探针 | 已完成（约 1 天） | 逐题配对比值中位数 0.532（−46.8%）、token +0.5%、36 次 0 error；定价表见 §12 |
| **P1b** ✅ | A/A 同实现重跑（sr 15 条 × 2 遍），量出评测链路噪声底 | 已完成（0.5 天） | 检索侧 0（同 query 两次结果逐字相同）；逐字 0/15、归一化后仍 0/15；步数 **12/15**、工具序列 **10/15**；真实分叉 model **27** / tool **12**（另有 **36** 个纯计时站点被归一化吃掉，不算行为分叉）；Judge 噪声 **0.357**（判定成功的 7 对；8 对因 Judge 超时不可比，已剔除并单列）——见 §8 的口径警告 |
| **G** ✅ | 正式对照：手写 `while` 循环 vs LangGraph `StateGraph`（sr 15 条、ABBA 平衡顺序、预热 1 条、强制串行），按 §8 行为层口径**逐条结账** | 已完成（2026-10-31） | judge Δ=**+0.077**（p=0.317、d=0.28）、步数 / 循环耗时 / token 的 p 全 > 0.3 → **两版均无可测差异**；步数相同率 **12/15**（= A/A 基线，达标）；⚠️ **工具序列一项未评**（脚本未记录该字段，事后补不出来）；产物 `data/eval/exp_G_sr15.json`（git `68512ef`），逐条见 `docs/framework_comparison.md`「验收逐条结账」 |
| **P2** | 只有定价表显示"手写总成本 ≥ 5 天、且需要图级可观测或社区维护"时才执行；范围按探针裁剪 | 8–10 天 | 验收见 §8 |

**P1 测的是"并行值不值"，不是"框架值不值"。** 如果并行拿不到收益，瓶颈就在远程调用本身，框架同样拿不到 —— 那时迁移只能由"恢复 / 审批 / 可观测"来付账，与速度无关。

**决策门（按定价表，不按单点）**：

| 手写剩余成本 | 结论 |
|---|---|
| ≤3 天 | 不迁（**当前成立**：四个缺口已全部完成，实际 ≈3.5–4 天） |
| **原估 4–5.5 天，实际 ≈3.5–4 天** | **不迁** —— 本轮结论；C 降级为「已评估后否决」，采纳 A |
| 需要图级 trace / 标准状态语义 / 多实例恢复 | 迁 P2，范围按 §12 探针裁剪 |

定价表必须扣除两边共有的前端成本（审批卡片 / 进度 UI），否则会把"框架省不掉的部分"算成框架收益。

**P0 状态（2026-10-30 已完成）**：`backend/tests/test_deliverable_orchestration.py` **17 条**全部离线，覆盖输出可复现 / 计划顺序 / 检索失败降级 / 抽取异常 / 截断 / 编号越界 / 主题闸门 / 前缀去重 / 冲突标注 / 引用去重 / 规划失败 / 6 条上限 / 串并行等价 / 并行保序。已做双向验证：`subs[:6]`→`subs[:5]`、计划顺序反转、并行结果反转，三处变异都会让对应用例变红。

**P0 发现并已修复的一处矛盾**：`truncated` 分支原先在 JSON 仍可解析时继续采纳事实，而 `notes` 文案写的是"事实未能采纳"。现在截断时**一条事实都不采纳**（`deliverable.py`），并有用例 `test_截断信号误报时也不得采纳事实` 守着；把这一行改回去，该用例会红（已双向验证）。

## 12. P1 定价表（代码级探针，2026-10-30）

> 口径：只估"改哪些文件、多少行、要不要重建表/快照"，**不写实现**。
> 除"并行"一行是已完成的实测外，其余是探针估算，供 §11 的决策门使用。

| 缺口 | 手写成本 | 关键改动 | 为什么不是更少 |
|---|---|---|---|
| **并行** | ✅ 已完成并实测（≈0.5 天） | `deliverable.py` 的 `_execute_plan` 接缝 + `ParallelDeliverableAgent`（约 55 行）+ 2 条用例 | n=18（6 题 × 3 轮，ABBA）：逐题配对比值中位数 **0.532（墙钟 −46.8%）**，18/18 次更快（符号检验 p≈7.6e-6）；token +0.5%、facts +0.6%、36 次运行 0 error。产物 `data/eval/exp_deliverable_parallel{,_r2,_r3}.json` |
| **恢复** | ✅ v1 已完成（≈1 天） | `deliverable_store.py`（快照 + Protocol 接缝）、`run(resume=True)`、2 个端点、6 条用例 | 原计划建 `deliverable_runs` 表；实际用文件快照（MySQL 曾停服，恢复能力不该绑在它上面）。多实例部署时此实现不成立，换 MySQL/Redis 只需实现 `SnapshotStore` 两个方法 |
| **审批** | ✅ v1 已完成（≈1 天） | `plan_draft` / `approve`、2 个端点、`DeliverablePanel.tsx` 计划卡片（勾选即改计划）、6 条用例 | 原估 1.5–2 天；实际做的是"选择保留哪些子任务"而不是自由编辑 goal/queries，省掉了表单与校验交互，服务端仍接受完整 plan 替换 |
| **可观测** | ✅ v1 已完成（≈0.5 天） | `deliverable.py` 的 `on_event` 钩子、`main.py` 2 个 SSE 端点、前端进度、3 条用例 | 事件只做观测：回调失败不打断执行；SSE 断线后任务照常跑完，可 resume/status 接回 |
| **合计** | **✅ 全部完成（实际 ≈3.5–4 天）** | | 并行 0.5 + 恢复 1 + 审批 1 + 可观测 0.5–1；原估 4–5.5 天，实际略快 |

**对照框架版**：P2 8–10 天（含 checkpoint、interrupt、前端、测试、文档）。手写实际 **≈3.5–4 天**，框架多付 **4.5–6 天**；四个能力都已由手写拿到，框架剩下的可买项只有**标准状态语义 + 图级 trace**。

> ⚠️ **并行那一行的两个口径要说清**（都是实测，不是估算）：
> - **墙钟比值 0.532** = **逐题配对比值的中位数**（同一题内 并行/串行，消掉题目难度差异），这是 §8 的验收口径；
>   轮级聚合口径另算，三轮分别是 **0.62 / 0.53 / 0.60**（各自取中位数再比，混入题目难度，只作参考）。
> - **token +0.5%** = **prompt + completion 合计**（330,389 → 332,119）。拆开看：
>   prompt **+0.86%**、completion **−2.39%** —— 涨的其实是 prompt（并发下 KV cache 命中下降），
>   这正是本节「成本可解释」要盯的那一项，合成一个数会把它盖住。
> - ⚠️ **规划器是非确定性的**：18 对里有 5 对两臂的子任务数不同（同一道题两次规划出的子任务数不一样），
>   严格说「同一道题」在两臂里工作量不等。但**工作量相同的 13 对中位 0.531、工作量不同的 5 对中位 0.534**，
>   两组几乎一致 → **并行收益不是「恰好少干了活」造成的**，结论不受影响。

**这张表怎么读**：

- 四个缺口已全部手写完成（实际 ≈3.5–4 天）→ 框架溢价 4.5–6 天，**不迁**；
- 只做 1–2 个缺口时溢价更高（框架 8–10 天不因少做而线性下降）—— 上一轮的判断在"全做"这个最不利假设下仍然成立；
- **前端审批 UI 两边都要写**（约 2 天），已从框架收益里扣除 —— 这是上一轮评审里最容易被漏掉的一项。

**结论（2026-10-30）**：按此表，C 相对 A 的差额是 4.5–6 天，且并行、断点续跑、审批、可观测都已由手写拿到 —— **C 否决，采纳 A**（见 §5、§13）。

---

## 13. 为什么不用 LangGraph（结论与证据）

一句话：**不是框架不好，是它在本项目的两条路径上都买不到足以支付 4.5–6 天差额的东西。**

| 问题 | 证据 | 结论 |
|---|---|---|
| chat 要不要迁？ | 需要精确控制四道安全阀、熔断、SSE 事件契约，且不能动 A–F 的可比性；LangGraph 1.2 起连 prebuilt 的 `create_react_agent` 都迁走了（要自己搭节点） | 迁过去也是手搭节点，收益 0、风险实 → **不迁** |
| deliverable 的并行要不要靠框架？ | 手写 `asyncio.gather` 已实现并实测：n=18，逐题配对比值中位数 0.532（墙钟 **−46.8%**），18/18 次更快（符号检验 p≈7.6e-6），token **+0.5%**、facts +0.6%、36 次 0 error | 手写 0.5 天拿到全部并行收益 → **不需要框架** |
| 四个缺口都做完了吗？ | 手写实际 **≈3.5–4 天** vs 框架 8–10 天（框架版要重做全部四项）；审批 UI 两边都要写 | 框架贵 4.5–6 天，只多买"标准状态语义 + 图级 trace" |
| 框架能让答案更好吗？ | **直接实测**（实验 G，2026-10-31，sr 15 条、配对 n=13）：HAND vs LangGraph `StateGraph` 的 judge Δ=**+0.077**、p=0.317、d=0.28；步数 / 循环耗时 / token 的 p 全 > 0.3 —— **均无可测差异**。A/A 噪声底同时给出判据：同实现重跑 judge 平均 \|Δ\| **0.357**（仅 7 对有效，8 对 Judge 超时被剔除）、最大 1.5 | 实测两版**没有**可测的质量差异；即便有 <1.5 分的"提升"，本项目**读不出来** |

**什么情况下会改用 LangGraph**：需要跨进程 / 多实例的 durable execution、人工审核成为产品主流程、或团队要共用一套图与 trace。那是"运维与协作"需求，不是"能力"需求 —— 届时 §7 的实施边界可以直接拿来用。

**对简历 / 面试的说法**：不要讲"我没用框架"，要讲"我量化了框架的边界" —— 并行收益（n=18，−47%）、框架对照（实验 G，judge Δ=+0.077，无可测差异）、噪声底（0.357）、定价表（多付 4.5–6 天）四个数字就是证据（§12 与 `data/eval/exp_deliverable_parallel*.json`、`data/eval/exp_G_sr15.json`）。

---

## 附录 A：可辩论的分歧点（给交叉验证用）

这几处没有唯一正确答案，我把自己的立场和反方论据都写出来，方便逐条对照另一份方案：

| # | 分歧点 | 我的立场 | 反方可能的论据 |
|---|---|---|---|
| 1 | chat 要不要一起迁 | 不迁，冻结 | "全仓 LangGraph"故事更完整，面试时不用解释两套实现 |
| 2 | checkpoint 用社区包还是自建 | P1 自建快照，P2 再评估 | 社区包一行 `checkpointer=` 就能用，自建是重复造轮子 |
| 3 | HITL 放在哪 | plan 之后、执行之前 | 每个子任务前也停一次，能省更多钱（但会让 6 次点击变成负担） |
| 4 | 只上 LangGraph 还是连 LangChain 一起 | 只用 `langgraph`（隐含 `langchain-core`），不引 `langchain` 全家桶 | JD 写的是 LangChain，只写 LangGraph 可能被追问 |
| 5 | LlamaIndex 做不做对照臂 | 可做，但不进本次 ADR | 一条命令换框架、用现成评测集，3–5 天能拿一个数字，性价比高 |
| 6 | 保留 `/api/deliverable` 同步接口 | 保留（兼容 + 回退） | 两套入口会让"到底哪条是生产路径"变模糊 |

---

## 附录 B：被否决方案的具体证据

### B.1 为什么不用 `langchain-openai` 替换 `llm/client.py`

`llm/client.py` 里有一部分是框架**不提供**、且被测试锁死的行为：

| 能力 | 位置 | 框架对应物 |
|---|---|---|
| 降级链 + 故障分类（429/5xx/超时降级，400/401 不降级） | `fallback.py` | 无（`with_fallbacks` 不分类） |
| MiniMax `<think>` 剥除 | `strip_think()` | 无 |
| 内联工具调用兜底（正文 JSON → tool_calls） | `parse_inline_tool_calls()` | 无 |
| rerank 走裸 HTTP `{base}/rerank` | `rerank()` | **不在 OpenAI 协议内** |
| embedding / rerank 的 LRU + 批量分流 | `_embed_cache` / `_rerank_cache` | 需自建 |

替换的净效果是：多一个依赖，少四条行为。

### B.2 为什么不用 LlamaIndex 重写 ingestion

| 能力 | 现状 | LlamaIndex 对应物 |
|---|---|---|
| MediaWiki 抓取（WAF / 567 挑战页 / revision / recentchanges） | `wiki_client.py` | 无（`SimpleDirectoryReader` 只读本地） |
| wikitext 模板展平（不展平 = 969 字洗成 20 字） | `template_parser.py` | 无 |
| 按小节 + 句界的分块与确定性 `chunk_id` | `chunker.py` | `SentenceSplitter`（会丢掉 section 与幂等 id） |
| 两级内容指纹 + 三级增量（15 次请求 vs 全量 71 次） | `fingerprint.py` / `delta_sync.py` | 只能对已加载文档做哈希 |

真正可替的只有 `切块 → embed → 分批 upsert` 约 40 行。

### B.3 如果要做 LlamaIndex，正确的位置是"对照臂"

现有 `Retriever` Protocol（`retrieve(query, config) -> list[RetrievalHit]`）已经是一个接缝：把 LlamaIndex 的 `QueryFusionRetriever + MetadataFilters + rerank` 包一层 adapter，用**现有 48 题**跑一遍，得到一个可比的 recall@3 / MRR。这件事的价值是"我知道框架到什么水平"，而不是"我把系统换了"。

---

## 附录 C：此 ADR 不依赖已删除的对照实验

回退时改动被存进了 `stash@{0}`（含未跟踪文件树：`react_langgraph.py`、`exp_g_agent_impl.py`、`test_react_langgraph.py`、`framework_comparison.md`），随时可取回（`git checkout 'stash@{0}^3' -- <路径>`）。本文早期版本曾误写为"已删除"，此处更正。本 ADR **不以该实验为前提**：

- 它回答的是「chat 换不换框架」，而 Q1 的结论由架构决定（不需要持久化 / HITL，且要保 SSE 契约），不依赖 p 值；
- 本 ADR 需要的是「deliverable 缺什么能力」——这是需求判断，不是效果对比；
- 若之后仍想补对照数字，正确做法是等 `workflow.py` 落地后，**直接在图这一层量框架开销**（同一任务，图 vs 直接函数调用），这比 15 条题的 judge 差值更贴近工程问题。

**恢复对照臂的前置条件**：已于 2026-10-30 完成 A/A 同实现重跑（sr 15 条，`data/eval/exp_G_aa_noise.json`）。结论：**步数分叉是随机发生的，不是某一版循环的属性** —— 预跑观察到 sr-007 5 vs 4、sr-008 1 vs 3，而在**同实现**重跑的 A/A 里这两题分别变成 4 vs 4、3 vs 3（**根本没分叉**），分叉换到了另外三题（sr-013 4 vs 3、sr-009 6 vs 7、sr-032 2 vs 1，合计 12/15 步数相同）。分叉来自模型措辞（`temperature=0` 下仍非确定）与实时抓取的 query 措辞差异，**不是两版循环不等价**。因此该实验只能按「行为层」口径引用（见 §8），且任何数字都必须带上噪声底。
