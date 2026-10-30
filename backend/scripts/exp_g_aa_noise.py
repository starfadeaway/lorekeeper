"""实验 G 附设：A/A 同实现重跑的噪声底。

## 它回答什么

「同一个实现、同一个题、同一份提示词，**再跑一遍**会不会不一样？」

这是所有单变量对照的地基。观测到的差异里，有多少属于「循环实现」，
有多少属于**评测链路自己**。

## 实测踩到的三个坑（都写进代码了）

**一、工具输出里带耗时。** 检索结果头部有 `... 37ms 来源=bm25+vector+rerank`，
两遍的耗时必然不同 —— 于是「消息链逐字比对」会永远失败，
而**检索结果本身完全相同**（`top_score` / `top_page` / 来源都没变）。
所以比对前必须把耗时归一化，否则会把「计时噪声」读成「检索噪声」。

**二、tool_call 的 id 是服务端随机生成的。** 形如
`call_00_h7k1tyE3oew6h3ertaMg0328`，每遍都不同。
id 只是锚点，**不是行为差异** —— 但它会让逐字比对必然失败。
所以比对前必须把 id 归一化。

**三、模型的措辞每次都变。** 这不是"噪声"可以忽略的东西，
它就是真实行为：同一题两次问，模型给的 `content` 措辞不同、
最终答案文本不同。所以**"两版逐字相同"在真实 API 下不是可达目标**，
可达的目标是「归一化之后结构等价 + 工具调用序列等价 + 步数相同 +
答案要点等价」。FakeLLM 守卫能证明逐字相同，但那只证明**控制流**，
不证明真实环境下的**行为**。

## 归因：扫全链，列出所有分叉点

| 分叉点的 role | 归因 | 含义 |
|---|---|---|
| `assistant` | **模型侧** | 模型这次的说法不一样（措辞 / 工具参数 / 答案文本） |
| `tool` | **工具侧** | 工具这次返回的内容不一样（**归一化耗时后**才算数） |

同一题可能**同时**有模型侧与工具侧分叉 —— 只报第一个会漏掉。

## 三个口径各报各的

| 口径 | 比什么 | 回答什么 |
|---|---|---|
| raw | 完全逐字 | 「一个字都不差」——真实环境下几乎必然 0 |
| normalized | 剥耗时 + 归一化 tool_call id | **结构是否等价** |
| tool_seq | 工具名 + 参数序列 | **Agent 的动作序列是否等价** |

再加两个行为指标：**步数**、**答案要点**（由 Judge 打分衡量）。

## 与实验 G 的条件对齐

- 每题内**立刻**跑 pass1 → pass2（不是先跑完 pass1 再跑 pass2）：
  这正是实验 G 里「第 2 个 arm 面对第 1 个 arm 跑热之后的缓存」的条件
- 两遍**共用同一个 `ToolContext`**，**不清缓存**

## 用法

    .venv\\Scripts\\python.exe backend\\scripts\\exp_g_aa_noise.py --cases sr-007,sr-008,sr-001,sr-002
    .venv\\Scripts\\python.exe backend\\scripts\\exp_g_aa_noise.py --judge
    .venv\\Scripts\\python.exe backend\\scripts\\exp_g_aa_noise.py            # 全部 15 条

产物：`data/eval/exp_G_aa_noise.json`
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.react import ReActAgent  # noqa: E402
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import (  # noqa: E402
    build_meta,
    load_cases,
    save,
)

#: 与实验 G 保持一致：全 4 个工具。
TOOLS = ["search_kb", "get_page", "calculator", "list_games"]

#: 工具耗时字段：`... 37ms 来源=bm25+vector+rerank` / `（542ms，共 1667 字）`
_MS_RE = re.compile(r"\d+(?:\.\d+)?\s*ms")
#: 服务端随机生成的 tool_call id
_CALL_ID_RE = re.compile(r"call_[A-Za-z0-9_]+")


def normalize_text(s: str) -> str:
    """剥掉计时噪声与随机 id。

    这两样都是**协议/运行时产物**，不是行为：
    耗时每次必然不同；tool_call id 由服务端每次随机生成。
    不剥掉它们，任何逐字比对都必然失败，且失败原因与实现无关。
    """
    s = _MS_RE.sub("<MS>", s or "")
    s = _CALL_ID_RE.sub("call_<ID>", s)
    return s


def norm_message(m: dict) -> dict:
    """归一化一条消息（content + tool_calls）。"""
    c = m.get("content")
    c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, sort_keys=True)
    calls = m.get("tool_calls") or []
    return {
        "role": m.get("role"),
        "content": normalize_text(c),
        "tool_calls": json.dumps(
            [{"n": (x.get("function") or {}).get("name") or x.get("name"),
              "a": (x.get("function") or {}).get("arguments", x.get("args", {}))}
             for x in calls],
            ensure_ascii=False, sort_keys=True,
        ),
    }


def tool_signature(messages: list[dict]) -> list[str]:
    """Agent 的动作序列：每次工具调用记一条「工具名(参数)」。"""
    out: list[str] = []
    for m in messages or []:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            name = fn.get("name") or c.get("name") or "?"
            args = fn.get("arguments", c.get("args", {}))
            if isinstance(args, str):
                with contextlib.suppress(Exception):
                    args = json.loads(args)
            out.append(f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})")
    return out


def digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def diff_sample(a: str, b: str, width: int = 60) -> dict:
    """两段文本的差异片段 —— 光说"不同"没用，要知道差在哪。"""
    import difflib

    sm = difflib.SequenceMatcher(None, a or "", b or "", autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return {"identical": True}
    first = ops[0]
    lo = max(0, first[1] - width // 2)
    hi = min(len(a or ""), first[2] + width // 2)
    lo2 = max(0, first[3] - width // 2)
    hi2 = min(len(b or ""), first[4] + width // 2)
    return {
        "identical": False,
        "n_diff_regions": len(ops),
        "n_diff_chars": sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal"),
        "p1_ctx": (a or "")[lo:hi],
        "p2_ctx": (b or "")[lo2:hi2],
    }


def all_diffs(m1: list[dict], m2: list[dict]) -> list[dict]:
    """扫全链，列出**所有**不同的位置（归一化之后）。"""
    out: list[dict] = []
    for i in range(max(len(m1 or []), len(m2 or []))):
        a = m1[i] if i < len(m1 or []) else None
        b = m2[i] if i < len(m2 or []) else None
        if a == b:
            continue
        role = (a or b or {}).get("role") or "?"
        item = {"index": i, "role": role}
        if a is None or b is None:
            item["kind"] = "missing"
        else:
            na, nb = norm_message(a), norm_message(b)
            item["kind"] = "tool_calls" if na["tool_calls"] != nb["tool_calls"] else "content"
            if na["content"] != nb["content"]:
                item["content_diff"] = diff_sample(na["content"], nb["content"])
            if na["tool_calls"] != nb["tool_calls"]:
                item["tool_calls_diff"] = diff_sample(na["tool_calls"], nb["tool_calls"])
        out.append(item)
    return out


async def run_once(agent: ReActAgent, case, tool_ctx) -> dict:
    """跑一遍，收集可比对的全部字段。异常不吞 —— 异常本身就是一种分叉。"""
    t0 = time.perf_counter()
    try:
        st = await agent.run(case.question, ctx=tool_ctx)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "messages": [],
        }
    return {
        "error": None,
        "steps": len(st.steps),
        "prompt_tokens": st.prompt_tokens,
        "completion_tokens": st.completion_tokens,
        "stop_reason": st.stop_reason,
        "answer": st.answer,
        "answer_sha": digest(st.answer),
        "tool_failures": st.tool_failures,
        "elapsed_ms": st.elapsed_ms,
        "messages": st.messages or [],
    }


def _dump_partial(rows: list[dict], out: str) -> None:
    """把已完成的结果增量写进 `.partial.json`。

    正式产物 `out` 仍要等全部跑完才写（避免半截产物被当成完整产物读），
    但 partial 让「跑了一半被杀」不再是零产出。
    """
    from exp_common import EVAL_DIR

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EVAL_DIR / f"{Path(out).stem}.partial.json"
    tmp.write_text(
        json.dumps({"rows": rows, "n_done": len(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="",
                    help="逗号分隔的 case_id；留空 = SUBSET_IDS 全部 15 条")
    ap.add_argument("--game", default="sr")
    ap.add_argument("--warmup", type=int, default=1,
                    help="正式统计前先跑几条把连接与缓存热起来（结果丢弃）")
    ap.add_argument("--judge", action="store_true",
                    help="额外给两遍的答案各打一次分 —— 量 Judge 侧噪声，"
                         "这是直接决定简历上 judge 类数字可信度的那一层")
    ap.add_argument("--out", default="exp_G_aa_noise.json")
    args = ap.parse_args()

    s = get_settings()
    llm = get_llm()
    judge = Judge(get_llm()) if args.judge else None
    schemas = [sc for sc in get_schemas() if sc["function"]["name"] in set(TOOLS)]

    cases = load_cases(0, game=args.game)
    if args.cases:
        want = [c.strip() for c in args.cases.split(",") if c.strip()]
        by_id = {c.case_id: c for c in cases}
        picked = [by_id[i] for i in want if i in by_id]
        missing = [i for i in want if i not in by_id]
        if missing:
            print(f"⚠ 这些 case_id 不在评测集里，已跳过：{missing}")
        cases = picked
    if not cases:
        print("✗ 没有可跑的 case")
        sys.exit(2)

    print(f"A/A 同实现重跑：{len(cases)} 条题，每题两遍，同实现（手写 ReAct 内核）")
    print(f"工具集：{TOOLS}（与实验 G 一致）")
    if args.judge:
        print("Judge 侧噪声：开（每遍的答案各判一次）")

    for w in range(min(max(0, args.warmup), len(cases))):
        c = cases[w]
        ctx = ToolContext.build(c.game)
        ag = ReActAgent(settings=s, llm=llm, tools=schemas)
        r = await run_once(ag, c, ctx)
        print(f"--- 预热 {w + 1}：{c.case_id}（结果丢弃）steps={r.get('steps')} ---", flush=True)

    rows: list[dict] = []
    for case in cases:
        tool_ctx = ToolContext.build(case.game)
        ag1 = ReActAgent(settings=s, llm=llm, tools=schemas)
        ag2 = ReActAgent(settings=s, llm=llm, tools=schemas)

        p1 = await run_once(ag1, case, tool_ctx)
        p2 = await run_once(ag2, case, tool_ctx)

        m1 = p1.get("messages") or []
        m2 = p2.get("messages") or []
        raw_same = m1 == m2
        diffs = all_diffs(m1, m2)
        norm_same = not diffs
        ts1, ts2 = tool_signature(m1), tool_signature(m2)

        buckets_here: dict[str, int] = {}
        for d in diffs:
            key = "model" if d["role"] == "assistant" else ("tool" if d["role"] == "tool" else f"other:{d['role']}")
            buckets_here[key] = buckets_here.get(key, 0) + 1

        row = {
            "case_id": case.case_id,
            "raw_identical": raw_same,
            "normalized_identical": norm_same,
            "tool_seq_identical": ts1 == ts2,
            "steps_identical": p1.get("steps") == p2.get("steps"),
            "answer_identical": p1.get("answer_sha") == p2.get("answer_sha"),
            "buckets": buckets_here,
            "n_diff_sites": len(diffs),
            "diffs": diffs,
            "tool_seq_p1": ts1,
            "tool_seq_p2": ts2,
        }

        if judge is not None:
            # 预取一份证据喂 Judge —— 两遍的 Judge 输入必须完全一致，
            # 这样分数的差异只来自「答案不同」与「Judge 自己抖」。
            try:
                ev = await tool_ctx.retriever.retrieve(case.question, RetrievalConfig(game=case.game, final_top_k=5))
                contexts = contexts_from_hits(ev)
            except Exception:  # noqa: BLE001
                contexts = ""
            j1 = await judge.score(question=case.question, gold=case.gold_answer,
                                   answer=p1.get("answer") or "", contexts=contexts)
            j2 = await judge.score(question=case.question, gold=case.gold_answer,
                                   answer=p2.get("answer") or "", contexts=contexts)
            d1, d2 = j1.as_dict(), j2.as_dict()
            row["judge"] = {
                "p1_score": d1.get("judge_score"),
                "p2_score": d2.get("judge_score"),
                "delta": (d1.get("judge_score") or 0) - (d2.get("judge_score") or 0),
                "p1_reason": d1.get("judge_reason"),
                "p2_reason": d2.get("judge_reason"),
            }

        row["p1"] = {k: v for k, v in p1.items() if k != "messages"}
        row["p2"] = {k: v for k, v in p2.items() if k != "messages"}
        rows.append(row)

        # **增量落盘** —— 每跑完一题就把当前全部结果写一次。
        #
        # 这个项目的教训是实测出来的：实验脚本静默挂死 65 分钟、磁盘上零产出，
        # 三处根因之一就是「结果只在全部跑完才落盘一次」。
        # 本脚本每题要打 2 次 Judge，而 Judge 偶发 120s 超时（实测 15 题里出现 4 次），
        # 一次超时就可能把整轮拖到超时边界上 —— 那样跑完的部分全丢。
        #
        # 中间产物写 `.partial.json`，跑完再落正式文件并删掉 partial。
        # 这样即使被中途杀掉，也留得下已完成的部分。
        _dump_partial(rows, args.out)

        jtxt = ""
        if judge is not None:
            jtxt = f"  judge {row['judge']['p1_score']}/{row['judge']['p2_score']}"
        print(
            f"  {case.case_id:<9} norm {'同' if norm_same else '异'}  "
            f"toolseq {'同' if ts1 == ts2 else '异'}  "
            f"steps {p1.get('steps')}/{p2.get('steps')}  "
            f"answer {'同' if row['answer_identical'] else '异'}  "
            f"分叉 {buckets_here}{jtxt}",
            flush=True,
        )

    n = len(rows) or 1
    agg = {
        "n": len(rows),
        "raw_identical": sum(1 for r in rows if r["raw_identical"]),
        "normalized_identical": sum(1 for r in rows if r["normalized_identical"]),
        "tool_seq_identical": sum(1 for r in rows if r["tool_seq_identical"]),
        "steps_identical": sum(1 for r in rows if r["steps_identical"]),
        "answer_identical": sum(1 for r in rows if r["answer_identical"]),
    }
    buckets: dict[str, int] = {}
    for r in rows:
        for k, v in r["buckets"].items():
            buckets[k] = buckets.get(k, 0) + v

    print("\n--- A/A 噪声底 ---")
    print(f"  n = {len(rows)}")
    print(f"  原始逐字相同    ：{agg['raw_identical']}/{len(rows)}  ({agg['raw_identical'] / n:.0%})")
    print(f"  归一化后相同    ：{agg['normalized_identical']}/{len(rows)}  ({agg['normalized_identical'] / n:.0%})")
    print(f"  工具序列相同    ：{agg['tool_seq_identical']}/{len(rows)}  ({agg['tool_seq_identical'] / n:.0%})")
    print(f"  步数相同        ：{agg['steps_identical']}/{len(rows)}  ({agg['steps_identical'] / n:.0%})")
    print(f"  答案逐字相同    ：{agg['answer_identical']}/{len(rows)}  ({agg['answer_identical'] / n:.0%})")
    print("  归一化后的分叉点归因（同一题可能同时有模型侧与工具侧）：")
    for b, c in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {b:<14} {c} 个分叉点  涉及 {sum(1 for r in rows if b in r['buckets'])}/{len(rows)} 题")
    if judge is not None:
        deltas = [abs(r["judge"]["delta"]) for r in rows if "judge" in r]
        pairs = [(r["judge"]["p1_score"], r["judge"]["p2_score"]) for r in rows if "judge" in r]
        same = sum(1 for a, b in pairs if a == b)
        mean_abs = sum(deltas) / len(deltas) if deltas else 0
        print("  Judge 侧噪声：")
        print(f"    两遍分数相同：{same}/{len(rows)}")
        print(f"    平均 |Δ| = {mean_abs:.3f}  最大 |Δ| = {max(deltas) if deltas else 0:.3f}")

    results = [{
        "label": "AA",
        "tool_set": TOOLS,
        "arms": ["pass1", "pass2"],
        "aggregate": agg | {"buckets": buckets},
        "rows": rows,
    }]

    meta = build_meta(args)
    meta.update({
        "script": "exp_g_aa_noise.py",
        "experiment": "G-AA",
        "arms": ["pass1", "pass2"],
        "labels": ["pass1", "pass2"],
        "variable": "无（同一实现重复两遍，量评测链路自身的噪声）",
        "implementation": "手写 ReAct 内核（ReActAgent）",
        "tool_set": TOOLS,
        "warmup_cases": min(max(0, args.warmup), len(cases)),
        "judge": "开" if args.judge else "不开（本实验只测 Agent 侧可复现性）",
        "normalization": "剥工具耗时（<MS>）+ 归一化 tool_call id（call_<ID>）",
        "note": (
            "同实现、同题、同 prompt、共用同一 ToolContext、不清缓存，"
            "每题内立刻跑 pass1→pass2 —— 与实验 G 的条件对齐。"
            "分叉点按 role 归因：assistant=模型侧、tool=工具侧。"
            "耗时与 tool_call id 属协议/计时产物，比对前已归一化。"
        ),
    })
    results[0]["_meta"] = meta
    save(results, args.out)
    print(f"\n产物：data/eval/{args.out}")


if __name__ == "__main__":
    asyncio.run(main())
