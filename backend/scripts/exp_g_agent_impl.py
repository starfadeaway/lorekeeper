"""实验 G：Agent 循环实现的对照实验（手写 while vs LangGraph StateGraph）。

## 它回答什么

「手写 ReAct 内核」和「用 LangGraph 搭同一个循环」，
在**同一批题、同一模型、同一语料、同一工具集、同一 Judge**下差多少？

这是本项目里唯一一个**变量是"实现方式"而不是"策略"**的实验：
实验 A/B/C/D/E/F 改的全是检索策略、工具能力、embedding 模型、分块粒度；
只有实验 G 改的是「这段控制流由谁写」。

## 为什么值得跑而不是嘴上说说

JD 里 LangChain / LangGraph / LlamaIndex 出现频率很高，而本项目刻意手写了内核。
「手写」可以是能力，也可以是偏见 —— 区别只在于有没有数据。
跑完这组，面试时能说的是「我评估过，并且知道差在哪」，
而不是「我听说过」。

## 单变量怎么保证

两个 arm **共用同一份代码**：`LangGraphReActAgent` 继承 `ReActAgent`，
只替换 `_run_once` 这一个方法。安全阀、检索熔断、提示词、工具执行、
注入包裹、引用收集、生成后引用校验全部是同一份实现，不是「照着抄一份」。

消息链也必须逐字一致 —— 两个 arm 发给模型的 `messages` 长得不一样的话，
差异里就混进了「消息格式」这个隐藏变量。所以对照臂不用 LangGraph 的
`ToolNode`（那会把消息转成 LangChain 类型再序列化）。

**检索证据的边界（已知、尚未消除的隐藏变量）**：每条题预取一次检索，
喂给 Judge 的那份两个 arm 共用；但 Agent 循环里两个 arm **各自实时检索**
（`get_page` 更是实时抓 wiki），所以「两个 arm 看到的检索结果相同」
**并不成立**。这个差异要么用「冻结工具输出」消掉，
要么先用 **A/A 同实现重跑**量出评测链路自己的噪声底，
再判断观测到的差异里有多少真属于「循环实现」这一个变量。

## 两条执行纪律

**一、ABBA 平衡顺序 + 预热。** 同一批题上比两套实现，延迟是最容易被污染的指标：

- *位置效应*：每条题里先跑的那个 arm 会承担**冷启动**（建连接、JIT、
  embedding 缓存未命中、模型端首次请求），后跑的那个占便宜。
  固定 HAND→LG 的话，这个偏差会**系统性地**压低 LangGraph 的延迟数字 ——
  而延迟正是本实验最想量的东西。
  所以顺序按题号交替：奇数题 HAND→LG、偶数题 LG→HAND，
  两个 arm 各占「先跑」和「后跑」的位置各一半。
- *预热效应*：整个流程的第一次调用一定最慢。所以正式统计前先跑
  `--warmup` 条题（两个 arm 都跑，结果**丢弃**），把连接与缓存都热起来。

同一批题先跑完 HAND 再跑完 LG 的做法，会把「模型端在某个时段负载偏高」
混进两档的差异里 —— 而那正是延迟差最容易被放大的时候。

**二、强制串行（concurrency=1）。** 实验 C 的注释里写过：
并发下 `latency_ms` 含排队时间，跨并发度不可比。
本实验要比的恰恰是 LangGraph 的调度开销 vs 手写循环的开销，
一旦并发，两者都会被排队时间淹没。所以这里**不提供并发参数**。

## 用法

    .venv\\Scripts\\python.exe scripts\\exp_g_agent_impl.py
    .venv\\Scripts\\python.exe scripts\\exp_g_agent_impl.py --limit 15 --game sr
    .venv\\Scripts\\python.exe scripts\\exp_g_agent_impl.py --game all --out exp_G_all.json

产物：`data/eval/exp_G_agent_impl.json`
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.react import ReActAgent  # noqa: E402
from app.agent.react_langgraph import (  # noqa: E402
    _LANGGRAPH_IMPORT_ERROR,
    LangGraphReActAgent,
)
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import (  # noqa: E402
    aggregate_rows,
    build_meta,
    case_row,
    fmt_p,
    load_cases,
    paired_scores,
    parse_games,
    print_summary,
    save,
    subset_composition,
)

#: 两个 arm 的顺序 = 产物里的先后顺序。
#: `paired_scores` 取**首尾**做对比，Δ = 末档 − 首档 = LG − HAND，
#: 正数 = LangGraph 更好。这个方向约定与 exp_common 的既有用法一致。
ARMS = ("HAND", "LG")

#: 工具集固定为**全 4 工具**，与实验 C 的 C2 档一致。
#: 这样 HAND 档的 judge 分数可以直接跟 exp_C.json 的 C2 对照 ——
#: 同一批题、同一模型、同一工具，能跑出接近的数就说明本实验的口径没跑偏。
TOOLS = ["search_kb", "get_page", "calculator", "list_games"]


async def run_case(case, *, label: str, judge: Judge, llm, schemas, ctx) -> dict:
    """跑单条 case 的一个 arm。

    `contexts`（检索证据）由调用方算一次后传进来，**只喂给 Judge**：
    两个 arm 的 Judge 输入完全一致，judge 分数的差异只剩「答案本身」。

    ⚠️ **这里的边界必须说清楚**：这份预取证据**不进入 Agent 循环**。
    两个 arm 在循环里各自实时调用 `search_kb` / `get_page`，
    也就是说「两个 arm 看到的检索证据相同」**是不成立的**，
    成立的只有「两个 Judge 看到的证据相同」。
    这是本实验目前**已知但未消除**的隐藏变量，来源有三：
    远程 rerank / 模型在 temperature=0 下的非确定性、
    `get_page` 实时抓 wiki 时两次内容可能不同、
    以及 A2 档 `get_page` 会不会触发不同的 WAF 挑战路径。
    消掉它只有两条路：在实验里**冻结工具输出**（把 `get_page` stub 掉），
    或先量出**同实现重跑的分叉率**（A/A 噪声底），
    再判断观测到的差异里有多少真属于「循环实现」这一个变量。
    """
    s = get_settings()
    agent_cls = ReActAgent if label == "HAND" else LangGraphReActAgent
    agent = agent_cls(settings=s, llm=llm, tools=schemas)

    contexts = ctx["contexts"]
    t0 = time.perf_counter()
    try:
        st = await agent.run(case.question, ctx=ctx["tool_ctx"])
        answer = st.answer
        usage = {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens}
        steps, failures, refs = len(st.steps), st.tool_failures, st.refs
        guard, stop_reason = st.guard_stats(), st.stop_reason
        loop_ms = st.elapsed_ms
        err = None
    except Exception as exc:  # noqa: BLE001
        answer, usage, steps, failures, refs = "", {}, 0, 0, []
        guard, stop_reason = {}, None
        loop_ms = 0
        err = f"{type(exc).__name__}: {exc}"
    dt = int((time.perf_counter() - t0) * 1000)

    js = await judge.score(
        question=case.question, gold=case.gold_answer, answer=answer, contexts=contexts,
    )
    d = js.as_dict()
    row = case_row(
        case, d,
        usage=usage, steps=steps, failures=failures, refs=refs,
        latency_ms=dt, err=err, answer=answer,
        stop_reason=stop_reason, guard=guard,
    )
    # 额外两个字段（不破坏 exp_common 的 16 键契约 —— case_row 的键名是
    # 跨脚本契约，但**追加**字段不会让 aggregate_rows 读错任何东西）：
    #   loop_ms —— 纯 Agent 循环耗时（state.elapsed_ms），不含检索与 Judge；
    #   impl    —— 这一行是哪个实现跑出来的，混排时能一眼分开。
    # 延迟为什么要两个：latency_ms 是端到端口径（与实验 C 一致），
    # loop_ms 才是「循环本身花了多久」—— 后者才是本实验要比的东西。
    row["loop_ms"] = loop_ms
    row["impl"] = label
    return row


def loop_aggregate(rows) -> dict:
    """`loop_ms` 的聚合。

    **不进 `aggregate_rows`** —— 那份是跨实验共享口径，改它会牵动
    实验 A~F 的历史产物。这里的聚合只在本脚本内部用，
    且明确按 `is_kept` 剔除失败行，与 aggregate_rows 保持同一样本集。
    """
    from exp_common import is_kept

    kept = [r for r in rows if is_kept(r)]
    ok = kept or list(rows)
    n = len(ok) or 1
    return {
        "avg_loop_ms": int(sum(r.get("loop_ms", 0) for r in ok) / n),
        "median_loop_ms": int(sorted(r.get("loop_ms", 0) for r in ok)[n // 2]) if ok else 0,
        "avg_steps": round(sum(r.get("steps", 0) for r in ok) / n, 2),
        "avg_prompt_tokens": int(sum(r.get("prompt_tokens", 0) for r in ok) / n),
        "n_ok": len(ok),
        "n": len(rows),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = 用代表性子集")
    ap.add_argument("--game", default="sr", help="目标库：sr / ys / zzz / all")
    ap.add_argument("--warmup", type=int, default=1,
                    help="正式统计前先跑几条题把连接与缓存热起来（结果丢弃）。"
                         "0 = 不预热；调大不会让结论更可信，只会更慢。")
    ap.add_argument("--out", default="exp_G_agent_impl.json")
    args = ap.parse_args()

    # LangGraph 没装就**直接失败**，不产出半个产物。
    # 只跑 HAND 的产物会被误读成「对照做完了」，那比不做更糟。
    if _LANGGRAPH_IMPORT_ERROR is not None:
        print("✗ 本实验需要 langgraph，当前环境没装：", _LANGGRAPH_IMPORT_ERROR)
        print("  装法：pip install 'langgraph==1.2.18'")
        sys.exit(2)

    case_list = load_cases(args.limit, game=args.game)
    games = parse_games(args.game)
    print(f"评测子集：{len(case_list)} 条，库 {'/'.join(games) or args.game}")
    print(f"arm 顺序 {ARMS}（ABBA 平衡：奇数题 HAND→LG、偶数题 LG→HAND，串行）")
    print(f"题型构成：{subset_composition(case_list)}")
    print(f"工具集（全 4 工具，与实验 C 的 C2 档一致）：{TOOLS}")

    llm = get_llm()
    judge = Judge(get_llm())
    schemas = [sc for sc in get_schemas() if sc["function"]["name"] in set(TOOLS)]

    # 先串行把 ToolContext 建好：第一次建 BM25 索引要几秒，
    # 放在并发里会让一批题同时抢着建。exp_c 里也是这么做的。
    for case in case_list:
        ToolContext.build(case.game)

    rows_by_arm: dict[str, list[dict]] = {a: [] for a in ARMS}

    async def one_case(case, order, *, collect: bool) -> None:
        """跑一条题的两个 arm。

        `collect=False` 时结果丢弃 —— 预热轮用。
        `order` 决定两个 arm 谁先跑：奇数题 HAND→LG、偶数题 LG→HAND。
        """
        # 预取一次检索，**只喂给 Judge**（两个 arm 共用同一份 judge 证据）。
        # ⚠️ 但 Agent 循环里两个 arm **各自实时检索** ——
        # `agent.run(...)` 拿到的是 tool_ctx，循环内的 search_kb / get_page
        # 会各自再查一遍。所以「两个 arm 看到的检索证据相同」是**不成立**的，
        # 成立的只有「两个 Judge 看到的证据相同」。
        # 这是本实验的已知隐藏变量，处置见 meta.note 与文档。
        tool_ctx = ToolContext.build(case.game)
        cfg = RetrievalConfig(game=case.game, final_top_k=5)
        try:
            ev = await tool_ctx.retriever.retrieve(case.question, cfg)
            contexts = contexts_from_hits(ev)
        except Exception:  # noqa: BLE001
            contexts = ""

        for label in order:
            row = await run_case(
                case, label=label, judge=judge, llm=llm, schemas=schemas,
                ctx={"contexts": contexts, "tool_ctx": tool_ctx},
            )
            if collect:
                rows_by_arm[label].append(row)
            tag = "" if collect else " [预热，结果丢弃]"
            print(f"  [{label}] {row['case_id']:<10} score={row['judge_score']:<6} "
                  f"steps={row['steps']} tok={row['prompt_tokens']:>6} "
                  f"loop={row['loop_ms']:>6}ms{tag}", flush=True)

    # ---- 预热轮：把连接、JIT、embedding 缓存、模型端首包都热起来 ----
    # 丢弃结果，只为让正式轮的两个 arm 都不再承担冷启动。
    # 预热用的题仍在正式集里（评测集固定 56 条，没有"额外的题"），
    # 正式轮会再跑一遍 —— 这没问题：我们比的是**两个 arm 的差值**，
    # 不是绝对延迟，两者都热了差值才可信。
    warm = min(max(0, args.warmup), len(case_list))
    for w in range(warm):
        order = ARMS if w % 2 == 0 else tuple(reversed(ARMS))
        print(f"--- 预热 {w + 1}/{warm}：{case_list[w].case_id}（{'→'.join(order)}）---", flush=True)
        await one_case(case_list[w], order, collect=False)

    # ---- 正式轮：ABBA 平衡顺序 ----
    for idx, case in enumerate(case_list, start=1):
        # 奇数题 HAND 先跑、偶数题 LG 先跑 ——
        # 两个 arm 各占「先跑」和「后跑」的位置各一半，
        # 位置效应（谁承担冷启动、谁占模型端缓存便宜）被抵消掉。
        order = ARMS if idx % 2 == 1 else tuple(reversed(ARMS))
        await one_case(case, order, collect=True)

    results = []
    for label in ARMS:
        rows = rows_by_arm[label]
        results.append({
            "label": label,
            "tools": TOOLS,
            "concurrency": 1,
            "aggregate": aggregate_rows(rows),
            "loop_aggregate": loop_aggregate(rows),
            "rows": rows,
        })

    meta = build_meta(args)
    meta.update({
        "script": "exp_g_agent_impl.py",
        "experiment": "G",
        "arms": list(ARMS),
        # build_meta 用 getattr(args, "labels", None) 取标签 —— 本脚本的标签是
        # 两个**实现**而不是策略档位，用 ARMS 填上，免得产物里留个 null。
        "labels": list(ARMS),
        "variable": "agent 循环实现（手写 while vs LangGraph StateGraph）",
        "tool_set": TOOLS,
        "concurrency": 1,
        "order": "ABBA（奇数题 HAND→LG、偶数题 LG→HAND），抵消位置效应",
        "warmup_cases": warm,
        "note": (
            "ABBA 平衡顺序 + 预热轮（结果丢弃）+ 强制串行："
            "并发下 latency_ms 含排队时间，会把 LangGraph 的调度开销差淹没；"
            "固定顺序则会让先跑的那个 arm 系统性承担冷启动。"
            "每条题预取一次检索证据、两个 arm 共用（仅喂 Judge）；"
            "Agent 循环内两个 arm 各自实时检索 —— 这是已知未消除的隐藏变量。"
        ),
        "langgraph_version": _langgraph_version(),
    })
    results[0]["_meta"] = meta
    save(results, args.out)
    print_summary(results, f"实验 G：Agent 循环实现对照（HAND vs LG，n={len(case_list)}）")

    print("\n--- 纯循环耗时（不含检索与 Judge）---")
    for r in results:
        la = r["loop_aggregate"]
        print(f"  {r['label']:<6} avg={la['avg_loop_ms']:>6}ms  median={la['median_loop_ms']:>6}ms  "
              f"steps={la['avg_steps']:.2f}  tok={la['avg_prompt_tokens']:>6}")

    for metric in ("judge_score", "loop_ms", "latency_ms", "steps", "prompt_tokens"):
        t = paired_scores(results, metric)
        if not t:
            continue
        print(f"\n--- 配对检验 HAND → LG（metric={metric}，n={t['n']}）---")
        print(f"  Δ={t['mean_diff']:+.4f}  t={t['t']}  p={fmt_p(t['p'])}  Cohen's d={t['cohens_d']}")


def _langgraph_version() -> str | None:
    """记下实际用的 langgraph 版本。

    没有它，产物只说「用 LangGraph 跑的」，但半年后没人说得清
    当时是 1.2.11 还是 1.2.18 —— 而这两个版本之间 prebuilt agent
    整个迁了包（`langgraph-prebuilt` → `langchain.agents`）。
    """
    try:
        from importlib.metadata import version

        return version("langgraph")
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    asyncio.run(main())
