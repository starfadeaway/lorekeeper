# 框架选型对照：手写 ReAct 内核 vs LangGraph

> 一句话：**不是"框架好不好"，是"在我这个场景里差多少、差在哪"。**
>
> **本轮结论（ADR-001，2026-10-30）**：**不迁移**。并行收益已由手写实现拿到
> （n=18，墙钟 −46%、token +0.5%）；四个缺口（并行 / 恢复 / 审批 / 可观测）
> 已于 2026-10-31 全部手写落地（实际 ≈3.5–4 天）vs 框架 8–10 天。
> 本文件保留对照臂与实验设计，作为"框架边界"的证据；正式 HAND vs LG 对照**已于
> 2026-10-31 跑完**（n=15，见下方实测结果）。
>
> 结论在下方「实测结果」一节；本节先说为什么做这个对照、以及怎么做才算数。

## 为什么要做

2026 年国内 AI 应用开发岗位的 JD 里，LangChain / LangGraph / LlamaIndex
出现频率很高。抓到的 8 份真实 JD 里，措辞以「**至少一种**」「**优先**」
「**加分项**」为主 —— 唯一把 RAG 写成硬门槛的那份写的是
「有主导建设完整 RAG 系统的实践经验，**而非仅限于组件调用**」。

本项目是手写 ReAct 内核（`backend/app/agent/react.py`），
简历上明写「未依赖 LangChain / LlamaIndex」。

「手写」可以是能力，也可以是偏见 —— 区别只在于有没有数据。
所以这个对照要产出的不是"我用过了"，而是：

1. **差多少**：同一批题、同一模型、同一语料、同一工具集下，两套实现的差；
2. **差在哪**：差的是质量、还是开销、还是能力边界；
3. **边界**：什么场景下我应该改用框架 —— 这才是面试里能顶住追问的部分。

## 两个框架各自是什么（2026-10 实测核对）

| | 定位（官方说法） | 强项 | 与本项目的关系 |
|---|---|---|---|
| **LangGraph** | "low-level orchestration framework"，提供 durable execution / streaming / human-in-the-loop / persistence / memory | 有状态、多步、可恢复的 agent 编排 | **本项目是 agent-first**，LangGraph 是对应生态；但 durable execution / checkpoint / HITL **恰好都不是本项目要的**（不做持久化、不做多 Agent、不做公网部署） |
| **LlamaIndex** | retrieval-first：ingestion / indexing / query engine / hybrid search / reranking 开箱即用 | RAG 数据管道与检索质量 | 本项目的 ingestion 是**针对中文游戏 wiki 特写**的模板展平，通用抽象套不上；而它的 hybrid search + reranking（BM25 ∥ 向量 → RRF → Cross-Encoder）**本项目已经自己实现了** |

> 依据：LangGraph 官方 README（1.2.18，2026-10-26）自述
> "We recommend you use LangGraph when you have advanced needs that require
> a combination of deterministic and agentic workflows, **heavy customization**,
> and **carefully controlled latency**"；
> 而 "durable execution, persistence, human-in-the-loop" 是它提供的基础设施。
>
> LlamaIndex 与 LangChain 的官方对比口径为
> **agent-first vs retrieval-first**：LangChain/LangGraph 赢在多步 agent 编排，
> LlamaIndex 赢在 RAG 管道与检索质量。

### 一个必须说清的版本事实

**LangGraph 1.2 起，prebuilt 的 `create_react_agent` 已迁出 LangGraph**
（`langgraph-prebuilt` 1.1.0 起标 `@deprecated`，新位置是
`langchain.agents.create_agent`）。

也就是说：**2026 年想用 LangGraph 写 ReAct，两边都得自己搭节点。**
这让「手写 while 循环 vs LangGraph StateGraph」变成一个**对等对照**，
而不是「手搓 vs 开箱即用」—— 反而更公平，也更有说服力。

本项目对照臂正是用 `StateGraph` 手搭 `agent → tools → agent` 的节点图。

## 实验设计：怎么保证「单变量」

实验代号 **G**，脚本 `backend/scripts/exp_g_agent_impl.py`，
产物 `data/eval/exp_G_sr15.json`。

### 变量只有一个

| 保持不变 | 唯一变化 |
|---|---|
| 同一批题（sr 15 条代表性子集）、同一模型 `deepseek-chat`、temperature=0 | **循环怎么实现** |
| 同一套提示词（`_prepare` → `build_system_prompt`） | 手写 `while True` ↔ LangGraph `StateGraph` |
| 同一套工具（4 个，与实验 C 的 C2 档一致）、同一套 schema | |
| 同一套安全阀（步数 / token / 墙钟 / 单工具）、同一个检索熔断 | |
| 同一套注入包裹、同一套引用收集与生成后引用校验 | |
| 同一个 Judge（SiliconFlow `Qwen/Qwen3-8B`）；喂给 Judge 的证据是同一份（Agent 循环内仍各自实时检索，见下） | |

实现上不是"照着抄一份"：`LangGraphReActAgent` **继承** `ReActAgent`，
只替换 `_run_once` 一个方法，其余全部是同一份代码。
抄一份的话两版会各自演化，之后谁改了哪边都看不出来，
实验数字的分叉就再也归因不到"循环实现"这一个变量上。

守卫 `backend/tests/test_react_langgraph.py` 用**脚本化 FakeLLM** 逐字比对两版产出的
消息链与每一步模型看到的上下文 —— 一旦消息格式有差异，这条会红。

注意它守的是**控制流层**（同一份脚本输入），不是真实 API 的逐字一致：
A/A 实测同一实现跑两遍，「逐字相同」只有 **0/15**，归一化后仍是 **0/15**。

（归一化会剥掉工具耗时与服务端随机生成的 tool_call id。实测 15 题里
**36 个差异站点**属于纯计时噪声 —— 被归一化吃掉，**不算行为分叉**；
剩下的真实分叉是模型侧 **27** 个、工具侧 **12** 个。
把计时噪声算进「工具侧分叉」的话，会与「检索侧噪声底 = 0」的结论自相矛盾。）

真实 API 下能要求的只有「归一化后**不劣于同实现基线** + 步数/工具序列比率 + 答案差值过噪声底」，
见 ADR-001 §8。

### 三个会把延迟对比带进沟里的坑（都是本项目踩过的）

1. **位置效应**：同一批题上，先跑的那一组会承担冷启动
   （建连接、JIT、embedding/rerank 缓存未命中）。
   本项目实测过这个坑：只预热第一条 query、剩余 14 条冷启动全落在先跑的那组时，
   **同一份代码隔一分钟重跑，中位数能从 +18.4% 翻到 −31.2%**（`docs/experiments.md`）。
   → 对策：**ABBA 平衡顺序**（奇数题 HAND→LG、偶数题 LG→HAND），
   两个 arm 各占"先跑""后跑"位置各一半。
2. **预热**：整个流程的第一次调用一定最慢。
   → 对策：正式统计前先跑 1 条题（两个 arm 都跑，结果**丢弃**）。
3. **并发排队**：并发下 `latency_ms` 含排队时间。
   → 对策：**强制串行**（本脚本不提供并发参数）。

另外，每条题的检索**只做一次**，但要注意这份预取证据**只喂给 Judge**；
两个 arm 在 ReAct 循环里仍会各自实时调用 `search_kb` / `get_page`。
所以「Judge 的输入完全一致」成立，「差异只剩答案本身」**不成立** ——
query 措辞与实时抓取结果都会引入分叉（2026-10-30 的 A/A 实测：同实现重跑，
步数只有 12/15 相同、工具序列 10/15 相同）。这是一个**已知未消除的隐藏变量**；
要消掉它，必须把工具输出冻结（stub `get_page` 或缓存工具返回），那属于下一轮改造。

### 一个必须说清的限制

对照臂**只实现非流式路径**（`_run_once`）。流式（`_run_stream`）显式抛
`NotImplementedError`：留一个"能跑但没人验证过"的分支比明确不支持更危险 ——
它会在有人真去用它的时候静默给出不一样的行为。

生产 API（`/api/chat`、`/api/chat/stream`）**仍然只挂手写内核**，
守卫 `test_生产入口没有引用对照臂` 盯着这一点。

## 实测结果

### 运行条件

- sr 15 条代表性子集（single_hop 5 / numerical 2 / semantic 3 / multi_hop 2 / unanswerable 3）；
- `deepseek-chat`、temperature=0、4 个工具、ABBA 平衡顺序、预热 1 条（丢弃）、强制串行；
- 产物 `data/eval/exp_G_sr15.json`（run 时 HEAD `8cacffb`，产物入库于 `68512ef`；3932 chunk / 706 页，langgraph 1.2.18）；
- ⚠️ 已知未消除变量：预取证据只喂 Judge，两个 arm 在循环里**各自实时检索** ——
  query 措辞与抓取结果会引入分叉（见上文）。

### 汇总（n=15）

| 指标 | HAND（手写 while） | LG（LangGraph StateGraph） |
|---|---|---|
| judge 均分 | 4.67 | 4.81 |
| correctness | 4.93 | 5.00 |
| faithfulness | 4.40 | 4.62 |
| 平均步数 | 2.2 | 1.9 |
| prompt token | 6,974 | 7,403 |
| 工具失败率 | 0.091 | 0.000 |
| 平均端到端延迟 | 4,284ms | 4,095ms |

> ⚠️ **「工具失败率 0.091 vs 0.000」这一格不能读成实现差异**：它**全部来自 sr-009 一个 case**
> （HAND 7 步里失败 3 次，LG 2 步 0 次）。而 A/A 实测显示，**同一实现**重跑 sr-009 时
> 本来就是 6 步 vs 7 步、2 次 vs 3 次工具失败 —— HAND 在本实验里的 7 步 / 3 次失败
> 落在**同实现自己的波动区间内**。
> 所以这格差异**在噪声内**，读不出「手写内核的工具调用更不稳」。
> 同理，HAND 的 7 步 vs LG 的 2 步（sr-009）是全表最大的一处步数差，
> 但它也落在 A/A 同实现重跑的 6–7 步区间里。

### 配对检验（n=13；sr-009 / sr-011 的 LG 侧 Judge 超时，按 `judge_error` 剔除）

| 指标 | Δ（LG − HAND） | p | Cohen's d（配对口径） |
|---|---|---|---|
| judge_score | **+0.077** | 0.317 | 0.28 |
| loop_ms | +222ms | 0.399 | 0.23 |
| steps | +0.077 | 0.665 | 0.12 |
| prompt_tokens | +842 | 0.304 | 0.29 |

### 验收逐条结账（对照 ADR-001 §8 的行为层）

| ADR §8 要求 | 实测 | 结论 |
|---|---|---|
| 归一化后等价（逐字） | 剥耗时与随机 tool_call id 后，两版仍 **0/15** 逐字相同 | ⚠️ **逐字等价在真实 API 下不可达** —— 同实现重跑也是 0/15（A/A）。所以这一项**只能按「不劣于同实现基线」解读，不能字面判通过**。实测未见劣于基线：步数 12/15 = 基线、judge Δ 远低于噪声底 |
| 步数相同率 ≥ A/A 基线（12/15） | **12/15**（分叉在 sr-007 / sr-013 / sr-009） | ✅ **正好达标**，不低于基线 |
| 工具序列相同率 ≥ A/A 基线（10/15） | **本轮无法评** —— `exp_g_agent_impl.py` 没记录工具序列，产物 row 里没有该字段 | ⚠️ **没测**，不是「测了没过」 |
| 答案 judge 差值 > 噪声底（0.357） | Δ = **+0.077** | ✅ 远低于噪声底 |

> ⚠️ **工具序列那一项是「没测」，不是「测了没过」** —— 脚本压根没记这个字段，
> 事后也补不出来（产物里没有）。能算的最近代理指标是 `guard.unique_queries`
> （去重后的 query 条数）：两版相同 **12/15**，与步数相同率一致；
> 但它**不是序列本身**，不能拿它当该项验收通过。
> 要真评这一项，得在脚本里补记工具序列后重跑 —— 本轮没做，如实记为未评。

### 结论

1. **两版在质量、步数、循环耗时、token 上均无可测差异**：p 全部 > 0.3，
   效应量都在 0.12–0.29 的小量级。
2. **judge 差值 0.077 远小于 A/A 噪声底 0.357**（同实现重跑的平均 |Δ|，n=7 有效）——
   方向本身都读不出来，不能写成"LG 更好"。
3. 因此"迁不迁"**不能**由"框架更好 / 更快"来支持；它只能由成本与生态决定
   （ADR-001 §12 定价表、§13）。
4. 适用边界：n=13 有效配对、sr 单库、live 检索未冻结；不外推到三库，
   也不能反过来写成"框架一定更差"。
5. **工具失败率的表面差异不构成实现优劣的证据**（全部来自 sr-009 一个 case，
   且落在同实现重跑的波动区间内）；**工具序列相同率这一项本轮未评**，
   原因是脚本没记录该字段 —— 见上方「验收逐条结账」。

### 验收口径：控制流层与行为层

| 层 | 能要求什么 | 依据 |
|---|---|---|
| 控制流（FakeLLM） | 两版消息链**逐字相同** | `test_react_langgraph.py`，可达 |
| 行为（真实 API） | 归一化后**不劣于同实现基线**；步数与工具序列比率不低于 A/A 基线（**12/15、10/15**）；答案 judge 差值超过噪声底 | `data/eval/exp_G_aa_noise.json`（A/A 有效对 **7**，平均 \|Δ\|=**0.357**、最大 **1.5**）；实验 G 的逐条结账见上方「验收逐条结账」 |

所以正式对照（HAND vs LG）在真实 API 下**不能**用"逐字相同"当通过条件。
本次对照已按该口径跑完（n=15，见上）：差异全部落在 A/A 噪声内。

## 复现方式

```powershell
# 跑对照（sr 15 条代表性子集；预热 1 条 + ABBA 平衡 + 强制串行）
.venv\Scripts\python.exe backend\scripts\exp_g_agent_impl.py --limit 0 --game sr --warmup 1 --out exp_G_sr15.json

# 控制流守卫（FakeLLM 下两版消息链必须逐字相同）
.venv\Scripts\python.exe -m pytest backend/tests/test_react_langgraph.py -v
```
