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

**四、Judge 超时会让「兜底 0 分」混进噪声底。**
`JudgeScore.as_dict()` 的键是 **`reason`**，里面写着 `judge_error:APITimeoutError`，
分数被兜成 0。第一版本脚本写的是 `d1.get("judge_reason")` ——
`as_dict()` 里根本没这个键，于是取到的**永远是 None**，而且**不报错**：
产物里 15 行的 reason 全是 null，那 5 个 0 分完全分不清是「判了 0」还是「超时」。
更要命的是它们各自的 Δ 是 5.0 / 0 / −4.0 / 5.0 / 5.0 —— 全是「真分 vs 兜底 0」，
混进均值会把噪声底从 0.75 撑到 1.77。
所以：**失败对必须剔除且单列**，Δ 必须是 `None` 而不是「真分 − 0」。
判据复用 `exp_common.JUDGE_ERROR_PREFIX`，不另起一套。

## 归因：扫全链，列出所有分叉点

| 分叉点的 role | 归因 | 含义 |
|---|---|---|
| `assistant` | **模型侧** | 模型这次的说法不一样（措辞 / 工具参数 / 答案文本） |
| `tool` | **工具侧** | 工具这次返回的内容不一样（**归一化耗时后**才算数） |

同一题可能**同时**有模型侧与工具侧分叉 —— 只报第一个会漏掉。

每个站点还带一个 `kind`，标明**归一化之后还差吗**：

- `raw_only` —— 归一化后完全一致，差异只在耗时 / 随机 id 上。
  这是**计时噪声**，不是行为分叉，**单独计数**，不混进上面的归因表。
  （混进去的话，「工具侧有 37 个分叉点」会与「检索侧噪声底 = 0」自相矛盾。）
- `content` / `tool_calls` / `missing` —— 归一化后仍不同，是**真实分叉**。

`normalized_identical` 只看真实分叉站点：耗时差异被归一化吃掉之后，
如果两版仍然一致，那才是真的「归一化后结构等价」。

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
    JUDGE_ERROR_PREFIX,
    build_meta,
    load_cases,
    save,
)


def judge_failed(reason: str | None) -> bool:
    """这一条 Judge 判定是不是**失败**（而不是真判了 0 分）。

    `JudgeScore.as_dict()` 的键是 **`reason`** —— 不是 `judge_reason`。
    `judge_error:` 前缀由 `judge.py` 在捕获异常时写进 reason
    （`judge_error:APITimeoutError`），`exp_common.case_row` 再把它映射成
    row 里的 `judge_reason` 字段。

    ⚠️ **踩过的坑**：本脚本第一版写的是 `d1.get("judge_reason")` ——
    `as_dict()` 里根本没有这个键，于是取到的永远是 None，
    **Judge 超时在产物里完全没留痕**。后果是噪声底只能给一个区间
    （排除失败对 0.75 / 把兜底 0 当真分数 1.77），
    而且半年后看这份 JSON 没人能分辨那 5 个 0 是「判了 0 分」还是「超时」。
    判据要与 `exp_common.judge_failed` 完全一致，所以复用同一个前缀常量。
    """
    return str(reason or "").startswith(JUDGE_ERROR_PREFIX)


def judge_side(d: dict) -> dict:
    """把 `JudgeScore.as_dict()` 摊成**可留痕**的一侧记录。

    ⚠️ 键名坑（就是这个 bug 的根）：`as_dict()` 的键是 `reason`，
    **没有** `judge_reason`。`judge_reason` 是 `exp_common.case_row`
    把 `judged["reason"]` 摊进 row 之后才有的名字 —— 拿它去读 as_dict
    取到的**永远是 None**，而且**不报错**。

    静默的后果：Judge 超时（`judge.py` 捕获异常后写
    `judge_error:APITimeoutError` 并把分数兜成 0）在产物里零留痕，
    15 题里那 5 个 0 分半年后分不清是「判了 0」还是「链路超时」。
    """
    reason = d.get("reason")
    return {
        "score": d.get("judge_score"),
        "correctness": d.get("correctness"),
        "faithfulness": d.get("faithfulness"),
        "reason": reason,
        "failed": judge_failed(reason),
    }


def judge_pair(d1: dict, d2: dict, *, agent_failed: bool = False) -> dict:
    """一题的 Judge 一对记录。

    **Δ 只有两遍都真判成功时才有意义** —— 一边超时兜成 0 分时，
    「真分 − 0」不是抖动，是链路故障，混进去会把噪声底算大。

    实测踩到的量级：上一轮产物里 5 个含 0 分的行，Δ 分别是
    5.0 / 0 / −4.0 / 5.0 / 5.0 —— 全是「真分 vs 兜底 0」，
    一个都不是 Judge 的抖动。

    `agent_failed` 覆盖另一种「该对不该比」：Agent 自己抛异常时
    答案是空串，Judge 会回「空回答」（不是 judge_error），
    两边都空 → Δ=0，看上去像「两遍一模一样」，
    实际是**两遍都没跑成**。那不是一致性，是没数据。
    """
    s1, s2 = judge_side(d1), judge_side(d2)
    usable = all(isinstance(s["score"], int | float) for s in (s1, s2))
    valid = usable and not agent_failed and not (s1["failed"] or s2["failed"])
    delta = (float(s1["score"]) - float(s2["score"])) if valid else None
    return {
        "p1": s1,
        "p2": s2,
        "agent_failed": agent_failed,
        "valid": valid,
        "delta": delta,
        "abs_delta": abs(delta) if delta is not None else None,
    }


def judge_stats(rows: list[dict]) -> dict:
    """从 rows 里算 Judge 侧噪声底。

    **失败对必须剔除，且要单独计数。**
    剔除是为了均值不被链路故障污染；单独计数是因为「链路健康度」
    本身也是一个数字 —— 不能拿 0.75 去讲「抖动幅度」，
    同时把「15 次里有 5 次超时」当成噪音藏起来。
    """
    judged = [r for r in rows if isinstance(r.get("judge"), dict)]
    valid_ids: list[str] = []
    failed_ids: list[str] = []
    p1_fail = p2_fail = 0
    agent_fail = 0
    for r in judged:
        j = r["judge"]
        f1, f2 = bool(j["p1"]["failed"]), bool(j["p2"]["failed"])
        p1_fail += int(f1)
        p2_fail += int(f2)
        if j.get("agent_failed"):
            agent_fail += 1
        (valid_ids if j.get("valid") else failed_ids).append(r["case_id"])
    deltas = sorted(
        float(r["judge"]["abs_delta"]) for r in judged if r["judge"].get("valid")
    )
    n = len(deltas)
    return {
        "n_judged": len(judged),
        "n_valid": n,
        "n_failed_pairs": len(failed_ids),
        "judge_failed_pairs": len(failed_ids) - agent_fail,
        "agent_failed_pairs": agent_fail,
        "failed_case_ids": failed_ids,
        "p1_failures": p1_fail,
        "p2_failures": p2_fail,
        "identical_pairs": sum(1 for d in deltas if d < 1e-9),
        "differing_pairs": n - sum(1 for d in deltas if d < 1e-9),
        "mean_abs_delta": round(sum(deltas) / n, 4) if n else None,
        "max_abs_delta": round(max(deltas), 4) if n else None,
        "median_abs_delta": round(deltas[n // 2], 4) if n else None,
        "p90_abs_delta": round(deltas[min(n - 1, int(n * 0.9))], 4) if n else None,
    }


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
    """扫全链，列出**所有**不同的位置。

    每一站都必须标明一件事：**归一化之后还差吗**。

    | kind | 含义 |
    |---|---|
    | `raw_only` | 归一化后**完全一致** —— 差异只在耗时 / 随机 id 上，**不是行为分叉** |
    | `content` | 归一化后 `content` 仍不同 → 真实分叉 |
    | `tool_calls` | 归一化后 `tool_calls` 仍不同 → 真实分叉（模型换了工具或参数） |
    | `missing` | 一边多一条消息 → 结构就不一样，最硬的差异 |

    ⚠️ **第一版在这里埋了个坑**：`normalized_identical` 被写成 `not diffs`，
    而 `diffs` 收录的是**原始**差异 —— 于是「归一化后是否一致」其实等于
    「原始是否一致」，归一化根本没参与判定。实测数据上两者碰巧都是 0/15，
    数字没错；但口径是错的，换一批数据就会给出错误结论。
    同理，`buckets` 原来把 `raw_only` 站点也计成「工具侧分叉」——
    而耗时差异恰恰是**最该被归一化吃掉**的那部分。
    """
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
            content_diff = na["content"] != nb["content"]
            calls_diff = na["tool_calls"] != nb["tool_calls"]
            if calls_diff:
                item["kind"] = "tool_calls"
            elif content_diff:
                item["kind"] = "content"
            else:
                item["kind"] = "raw_only"
            if content_diff:
                item["content_diff"] = diff_sample(na["content"], nb["content"])
            if calls_diff:
                item["tool_calls_diff"] = diff_sample(na["tool_calls"], nb["tool_calls"])
        out.append(item)
    return out


def normalized_identical(diffs: list[dict]) -> bool:
    """归一化之后两版消息链是否一致。

    只看**真实差异站点**（`kind != "raw_only"`）——
    `raw_only` 的差异已经被耗时与随机 id 的归一化吃掉了，
    拿它算「不一致」等于把计时噪声当成行为差异。
    """
    return not any(d.get("kind") != "raw_only" for d in diffs)


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
    跑完之后由 `_clear_partial` 把它删掉 —— 它没有 `_meta`、没有 `aggregate`，
    留着会被误当成正式产物。
    """
    from exp_common import EVAL_DIR

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EVAL_DIR / f"{Path(out).stem}.partial.json"
    tmp.write_text(
        json.dumps({"rows": rows, "n_done": len(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_partial(out: str) -> None:
    """跑完之后把 `.partial.json` 删掉。

    为什么必须删：partial 是**中间产物** —— 它没有 `_meta`、没有 `aggregate`，
    只有 `rows` 和 `n_done`。留在 `data/eval/` 里有两个害处：
      1. 它和正式产物长得像（同样的文件名主体），半年后有人读错文件；
      2. 它是完整数据的重复副本，体积翻倍。
    正式产物写成功之后，partial 的使命（跑一半被杀也不丢结果）就结束了。
    """
    from exp_common import EVAL_DIR

    tmp = EVAL_DIR / f"{Path(out).stem}.partial.json"
    if tmp.exists():
        tmp.unlink()


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
        norm_same = normalized_identical(diffs)
        ts1, ts2 = tool_signature(m1), tool_signature(m2)

        # 只把**真实差异**（归一化后仍不同）算进分叉归因。
        # 纯耗时 / 随机 id 差异单列 —— 那是计时噪声，恰恰是归一化要吃掉的东西，
        # 把它算成「工具侧分叉」会与「检索侧噪声底 = 0」的结论自相矛盾。
        buckets_here: dict[str, int] = {}
        timing_only = 0
        for d in diffs:
            if d["kind"] == "raw_only":
                timing_only += 1
                continue
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
            "n_real_diff_sites": len(diffs) - timing_only,
            "n_timing_only_sites": timing_only,
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
            # 键名必须是 `reason` —— `as_dict()` 里没有 `judge_reason`，
            # 读错键名会静默拿到 None，Judge 超时就在产物里零留痕。
            row["judge"] = judge_pair(
                j1.as_dict(), j2.as_dict(),
                agent_failed=bool(p1.get("error") or p2.get("error")),
            )

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
            j = row["judge"]
            if j["valid"]:
                jtxt = f"  judge {j['p1']['score']}/{j['p2']['score']} Δ={j['delta']:+.3f}"
            else:
                why = "agent 失败" if j.get("agent_failed") else "judge 链路失败"
                jtxt = f"  judge 不可比（{why}）"
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
    timing_total = sum(r.get("n_timing_only_sites", 0) for r in rows)
    real_total = sum(r.get("n_real_diff_sites", 0) for r in rows)

    print("\n--- A/A 噪声底 ---")
    print(f"  n = {len(rows)}")
    print(f"  原始逐字相同    ：{agg['raw_identical']}/{len(rows)}  ({agg['raw_identical'] / n:.0%})")
    print(f"  归一化后相同    ：{agg['normalized_identical']}/{len(rows)}  ({agg['normalized_identical'] / n:.0%})")
    print(f"  工具序列相同    ：{agg['tool_seq_identical']}/{len(rows)}  ({agg['tool_seq_identical'] / n:.0%})")
    print(f"  步数相同        ：{agg['steps_identical']}/{len(rows)}  ({agg['steps_identical'] / n:.0%})")
    print(f"  答案逐字相同    ：{agg['answer_identical']}/{len(rows)}  ({agg['answer_identical'] / n:.0%})")
    print("  归一化后的分叉点归因（只算真实差异；纯耗时差异另列）：")
    for b, c in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {b:<14} {c} 个分叉点  涉及 {sum(1 for r in rows if b in r['buckets'])}/{len(rows)} 题")
    print(f"    其中「归一化后已消失」的纯耗时/随机 id 差异：{timing_total} 个站点"
          f"（这才是归一化要吃掉的部分，不算行为分叉）")
    print(f"    真实分叉点合计：{real_total} 个")
    if judge is not None:
        jst = judge_stats(rows)
        agg["judge"] = jst
        print("  Judge 侧噪声（失败对已剔除，单列）：")
        print(f"    判定成功 {jst['n_valid']} 对 / 不可比 {jst['n_failed_pairs']} 对"
              f"（judge 链路失败 {jst['judge_failed_pairs']}、agent 失败 {jst['agent_failed_pairs']}）")
        if jst["failed_case_ids"]:
            print(f"    不可比题号：{', '.join(jst['failed_case_ids'])}")
        print(f"    p1 失败 {jst['p1_failures']} 次 / p2 失败 {jst['p2_failures']} 次")
        if jst["n_valid"]:
            print(f"    两遍分数相同：{jst['identical_pairs']}/{jst['n_valid']}"
                  f"  不同：{jst['differing_pairs']}/{jst['n_valid']}")
            print(f"    平均 |Δ| = {jst['mean_abs_delta']:.3f}"
                  f"  中位数 {jst['median_abs_delta']:.3f}"
                  f"  P90 {jst['p90_abs_delta']:.3f}"
                  f"  最大 |Δ| = {jst['max_abs_delta']:.3f}")
        else:
            print("    没有可比的对 —— 本轮 Judge 侧全失败，噪声底测不出来")

    results = [{
        "label": "AA",
        "tool_set": TOOLS,
        "arms": ["pass1", "pass2"],
        "aggregate": agg | {
            "buckets": buckets,
            "n_timing_only_sites": timing_total,
            "n_real_diff_sites": real_total,
        },
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
        "normalization": (
            "剥工具耗时（<MS>）+ 归一化 tool_call id（call_<ID>）。"
            "每个差异站点按 kind 分类：raw_only=归一化后已一致（计时噪声，不算行为分叉）/"
            "content / tool_calls / missing=真实分叉；"
            "normalized_identical 只看真实分叉站点。"
        ),
        "note": (
            "同实现、同题、同 prompt、共用同一 ToolContext、不清缓存，"
            "每题内立刻跑 pass1→pass2 —— 与实验 G 的条件对齐。"
            "分叉点按 role 归因：assistant=模型侧、tool=工具侧。"
            "耗时与 tool_call id 属协议/计时产物，比对前已归一化。"
            "Judge 侧：失败对（judge_error:* 超时/限流，或 Agent 侧异常）"
            "一律剔除并单列计数，Δ 记 None —— 「真分 − 兜底 0」不是抖动。"
        ),
    })
    results[0]["_meta"] = meta
    save(results, args.out)
    _clear_partial(args.out)
    print(f"\n产物：data/eval/{args.out}")


if __name__ == "__main__":
    asyncio.run(main())
