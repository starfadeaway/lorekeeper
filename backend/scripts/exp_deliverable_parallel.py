"""实验 P1：交付物任务「串行 for」vs「并行 gather」的实测对照。

## 它回答什么

ADR-001 §11 的 P1 只需要回答一个问题：

> 把 `DeliverableAgent._execute_plan` 的串行 for 换成 `asyncio.gather`，
> **墙钟能省多少、token 会不会涨**？

产物是 P1 定价表的第一行。至于"答案质量有没有变"，这个脚本**不测**：
交付物没有 gold，真实 API 下 judge 差值还要先过 A/A 噪声底（ADR-001 §8）。
质量在控制流层由 `test_deliverable_orchestration.py` 的并行等价用例守着。

## 为什么 ABBA 平衡顺序

同一批题上，先跑的那一组会承担冷启动（连接、BM25 建索引、embedding 缓存）。
`docs/experiments.md` 记过一次实测：同一份代码隔一分钟重跑，中位数能从
+18.4% 翻到 −31.2%。所以每道题的两组**交替先后**，位置效应在汇总时对消。

## 为什么强制串行跑用例

并发跑多道题会让 `wall_ms` 混入排队时间，而本项目要测的正是"同一道题内部
并行 vs 串行"的差。`--parallel-limit` 只控制**一道题内部**的子任务并发度，
两者不是一回事。

## 口径

- `wall_ms`：从 `agent.run()` 进到返回的墙钟，含规划 + 子任务 + 合并渲染；
- `prompt_tokens` / `completion_tokens`：把 LLM 包一层计数器累加，不含 Judge；
- `facts_total` / `subtasks_ok`：产物完整性的旁证（并行若把事实弄丢，这里会变）；
- `subtask_ms_sum`：各子任务 `latency_ms` 之和。串行时它≈wall，并行时它≈N×wall
  的一部分 —— 这两个数的比例就是"有多少时间真的重叠了"。

## 用法

```powershell
.venv\\Scripts\\python.exe backend\\scripts\\exp_deliverable_parallel.py --game sr --limit 0 --parallel-limit 4 --out exp_deliverable_parallel.json
```
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.agent.deliverable import (  # noqa: E402
    DeliverableAgent,
    ParallelDeliverableAgent,
)
from app.agent.tools import ToolContext  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.llm.client import get_llm  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"

#: 题库：交付物任务没有 gold，所以这里只固定**问题**，不固定答案。
#: 题面刻意写成"整理…"这类会触发 3–6 个子任务规划的形式。
CASES: list[dict] = [
    {"case_id": "dlv-sr-01", "game": "sr", "question": "整理三月七的养成要点：属性、遗器、配队三方面都要"},
    {"case_id": "dlv-sr-02", "game": "sr", "question": "整理银狼的养成要点：行迹、光锥、遗器"},
    {"case_id": "dlv-sr-03", "game": "sr", "question": "整理黄泉的养成要点：属性阈值、光锥、配队"},
    {"case_id": "dlv-sr-04", "game": "sr", "question": "整理流萤的养成要点：属性、遗器、光锥"},
    {"case_id": "dlv-sr-05", "game": "sr", "question": "对比三月七与银狼的定位差异，分别列出关键属性"},
    {"case_id": "dlv-sr-06", "game": "sr", "question": "整理符玄的养成要点：属性、遗器、配队"},
]


class CountingLLM:
    """把 LLM 包一层，累加 usage —— 交付物结果对象里本来没有 token 字段。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    async def chat(self, messages, **kw):  # noqa: D102 - 与 LLMClient 同形状
        out = await self.inner.chat(messages, **kw)
        usage = out.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.calls += 1
        return out


def build_agent(label: str, settings, llm, game: str, parallel_limit: int):
    kwargs = dict(settings=settings, llm=llm, ctx=ToolContext.build(game), top_k=5)
    if label == "serial":
        return DeliverableAgent(**kwargs)
    return ParallelDeliverableAgent(**kwargs, parallel_limit=parallel_limit)


async def run_one(label: str, case: dict, settings, inner_llm, parallel_limit: int,
                  position: int, timeout: float) -> dict:
    """跑一条用例的一个 arm；返回一行产物（与实验 A–F 的 row 形状不同，但可读）。

    **每条用例独立超时**：2026-10-30 的 A/A 重跑就在这个项目上挂死过一次
    （3 条 Established TLS 连接、CPU 不再增长、partial 35 分钟不写），
    而实验脚本通常无人值守 —— 没有外层超时，一次挂死会带走整轮。
    超时的那一行记为 error 并继续，不让它把后面的用例一起拖没。
    """
    llm = CountingLLM(inner_llm)
    agent = build_agent(label, settings, llm, case["game"], parallel_limit)
    t0 = time.perf_counter()
    err = None
    try:
        res = await asyncio.wait_for(
            agent.run(case["question"], game=case["game"]), timeout=timeout
        )
        wall_ms = int((time.perf_counter() - t0) * 1000)
        statuses = [s.status for s in res.subtasks]
        row = {
            "case_id": case["case_id"],
            "impl": label,
            "position": position,          # 0=先跑，1=后跑；ABBA 汇总时用来核对平衡
            "wall_ms": wall_ms,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "llm_calls": llm.calls,
            "plan_size": len(res.plan),
            "subtasks_total": len(res.subtasks),
            "subtasks_ok": sum(1 for s in statuses if s == "ok"),
            "subtasks_empty": sum(1 for s in statuses if s == "empty"),
            "subtasks_error": sum(1 for s in statuses if s == "error"),
            "subtasks_truncated": sum(1 for s in statuses if s == "truncated"),
            "facts_total": sum(len(s.facts) for s in res.subtasks),
            "subtask_ms_sum": sum(s.latency_ms for s in res.subtasks),
            "statuses": statuses,
        }
    except Exception as exc:  # noqa: BLE001 - 实验脚本：失败要留行，不能丢样本
        wall_ms = int((time.perf_counter() - t0) * 1000)
        err = f"{type(exc).__name__}: {exc}"
        row = {
            "case_id": case["case_id"], "impl": label, "position": position,
            "wall_ms": wall_ms, "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens, "llm_calls": llm.calls,
            "plan_size": 0, "subtasks_total": 0, "subtasks_ok": 0,
            "subtasks_empty": 0, "subtasks_error": 0, "subtasks_truncated": 0,
            "facts_total": 0, "subtask_ms_sum": 0, "statuses": [],
        }
    if err:
        row["error"] = err
    return row


def aggregate(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if r["impl"] == label and not r.get("error")]
    if not ok:
        return {"impl": label, "n": 0}
    walls = sorted(r["wall_ms"] for r in ok)
    return {
        "impl": label,
        "n": len(ok),
        "median_wall_ms": int(statistics.median(walls)),
        "mean_wall_ms": int(sum(walls) / len(walls)),
        "median_prompt_tokens": int(statistics.median(r["prompt_tokens"] for r in ok)),
        "median_completion_tokens": int(statistics.median(r["completion_tokens"] for r in ok)),
        "facts_total": sum(r["facts_total"] for r in ok),
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - 无 git 环境时不该让实验挂掉
        return ""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--parallel-limit", type=int, default=4, help="单题内部子任务并发上限")
    ap.add_argument("--warmup", type=int, default=1, help="预热题数（两组都跑，结果丢弃）")
    ap.add_argument("--timeout", type=float, default=300.0, help="单条用例超时（秒）")
    ap.add_argument("--out", default="exp_deliverable_parallel.json")
    args = ap.parse_args()

    settings = get_settings()
    llm = get_llm(settings)
    cases = [c for c in CASES if c["game"] == args.game]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"没有 {args.game} 的用例")

    print(f"cases={len(cases)} parallel_limit={args.parallel_limit} "
          f"ABBA=奇数题 serial→parallel、偶数题反向", flush=True)

    # ---- 预热：建 BM25、热连接，结果丢弃 ----
    for i in range(min(args.warmup, len(cases))):
        await run_one("serial", cases[i], settings, llm, args.parallel_limit, 0, args.timeout)
        await run_one("parallel", cases[i], settings, llm, args.parallel_limit, 1, args.timeout)
    print(f"预热完成 {args.warmup} 条（结果丢弃）", flush=True)

    rows: list[dict] = []
    out_path = EVAL_DIR / args.out
    for idx, case in enumerate(cases, start=1):
        order = ("serial", "parallel") if idx % 2 == 1 else ("parallel", "serial")
        for position, label in enumerate(order):
            row = await run_one(label, case, settings, llm, args.parallel_limit, position, args.timeout)
            rows.append(row)
            print(f"  [{label:<8}] {case['case_id']} wall={row['wall_ms']:>6}ms "
                  f"tok={row['prompt_tokens'] + row['completion_tokens']:>6} "
                  f"facts={row['facts_total']:>2} ok={row['subtasks_ok']}/{row['subtasks_total']}"
                  f"{' ERR=' + row['error'] if row.get('error') else ''}", flush=True)
        # 增量落盘：任一用例崩掉也不带走已完成的部分
        payload = [{
            "label": "P1-parallel",
            "rows": rows,
            "aggregate": {
                "serial": aggregate(rows, "serial"),
                "parallel": aggregate(rows, "parallel"),
            },
            "_meta": {
                "script": "exp_deliverable_parallel.py",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "game": args.game,
                "parallel_limit": args.parallel_limit,
                "order": "ABBA（奇数题 serial→parallel、偶数题反向）",
                "measured": ["wall_ms", "prompt_tokens", "completion_tokens", "facts_total"],
                "not_measured": "答案质量（交付物无 gold；真实 API 需先过 A/A 噪声底）",
            },
        }]
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    serial = aggregate(rows, "serial")
    parallel = aggregate(rows, "parallel")
    print("\n=== 汇总 ===", flush=True)
    for agg in (serial, parallel):
        print(f"  {agg['impl']:<8} n={agg['n']} median_wall={agg['median_wall_ms']}ms "
              f"facts={agg['facts_total']}", flush=True)
    if serial["median_wall_ms"] and parallel["median_wall_ms"]:
        ratio = parallel["median_wall_ms"] / serial["median_wall_ms"]
        print(f"  墙钟比 parallel/serial = {ratio:.2f}（并行收益 {1 - ratio:.0%}）", flush=True)
    print(f"产物：{out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
