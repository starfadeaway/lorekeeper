"""探针：Judge 自身的抖动（同一答案、同一证据，判两遍）。

## 它回答什么

「同一个答案让 Judge 打两次分，会不会不一样？」

这和 `exp_g_aa_noise.py` 测的不是同一件事：

| 脚本 | 变化的东西 | 测出的噪声 |
|---|---|---|
| `exp_g_aa_noise.py`（A/A） | 重跑整个 Agent（答案也变） | **端到端**噪声 = 答案差异 + Judge 差异 |
| 本脚本 | 只重跑 Judge（答案固定） | **纯 Judge** 噪声 |

两个数一减，就能说清「某档的 judge 均分差 0.13」里有多少是
**答案本身不稳**、有多少是 **Judge 自己会抖**。

## 为什么必须测

简历上「全工具档 judge 反低 0.13」这类量级的结论，
在没有噪声底之前是无法辩护的：
分不清「这一档真的更差」和「Judge 这次手抖」，
数字就只是读数，不是结论。

而项目自己的 `exp_common.judge_failed()` 已经在处理更极端的一种抖动 ——
**Judge 调用直接超时**（记 `judge_error:APITimeoutError`，分数兜成 0）。
实测 15 题里出现 4 次（见 `exp_G_aa_noise.json` 的 meta）。

## 用法

    .venv\\Scripts\\python.exe backend\\scripts\\probe_judge_noise.py --label C0 --limit 15

产物：`data/eval/probe_judge_noise.json`
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.tools import ToolContext  # noqa: E402
from app.eval.cases import ALL_CASES  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import EVAL_DIR, build_meta, is_kept, judge_failed, save  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="exp_C_sr.json",
                    help="取答案的产物文件（data/eval 下）")
    ap.add_argument("--label", default="C0",
                    help="取哪一档的答案。默认 C0（无工具档）—— "
                         "它的分数分布 1.0~5.0 有区分度，能测出真实抖动；"
                         "C1/C2 档大多是 5.0 满分，天生就不会抖，测了等于没测")
    ap.add_argument("--limit", type=int, default=15, help="最多取几条")
    ap.add_argument("--out", default="probe_judge_noise.json")
    args = ap.parse_args()

    judge = Judge(get_llm())

    prod = json.loads((EVAL_DIR / args.product).read_text(encoding="utf-8"))
    arm = next((a for a in prod if a.get("label") == args.label), None)
    if arm is None:
        print(f"✗ {args.product} 里没有档位 {args.label}")
        sys.exit(2)

    # 只取「判定成功」的答案 —— judge 失败行的 answer 是空串，
    # 拿它去测 Judge 等于测"Judge 面对空回答打几分"，不是这里要的东西。
    rows = [r for r in (arm.get("rows") or []) if is_kept(r) and not judge_failed(r)]
    rows = rows[: args.limit]

    by_id = {}
    for cases in ALL_CASES.values():
        for c in cases:
            by_id[c.case_id] = c

    print(f"Judge 抖动探针：{args.product} 的 {args.label} 档，取 {len(rows)} 条，各判两遍")
    out_rows: list[dict] = []
    partial = EVAL_DIR / "probe_judge_noise.partial.json"

    for r in rows:
        case = by_id.get(r["case_id"])
        if case is None:
            print(f"  ⚠ {r['case_id']} 不在评测集里，跳过")
            continue
        # 检索是确定性的（实测见 exp_G_aa_noise.json），所以重跑一次拿证据即可，
        # 且两遍 Judge 拿到的是**完全相同**的证据 —— 变量只剩 Judge 本身。
        try:
            ctx = ToolContext.build(case.game)
            ev = await ctx.retriever.retrieve(case.question, RetrievalConfig(game=case.game, final_top_k=5))
            contexts = contexts_from_hits(ev)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ {case.case_id} 检索失败：{type(exc).__name__}: {exc}")
            contexts = ""

        t0 = time.perf_counter()
        s1 = await judge.score(question=case.question, gold=case.gold_answer,
                               answer=r.get("answer") or "", contexts=contexts)
        s2 = await judge.score(question=case.question, gold=case.gold_answer,
                               answer=r.get("answer") or "", contexts=contexts)
        d1, d2 = s1.as_dict(), s2.as_dict()
        row = {
            "case_id": case.case_id,
            "orig_judge_score": r.get("judge_score"),
            "p1": d1,
            "p2": d2,
            "delta_score": (d1.get("judge_score") or 0) - (d2.get("judge_score") or 0),
            "delta_correctness": (d1.get("correctness") or 0) - (d2.get("correctness") or 0),
            "delta_faithfulness": (d1.get("faithfulness") or 0) - (d2.get("faithfulness") or 0),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
        out_rows.append(row)
        partial.write_text(json.dumps({"rows": out_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {case.case_id:<9} {d1.get('judge_score')} / {d2.get('judge_score')}  "
            f"Δ={row['delta_score']:+.3f}  (correctness {d1.get('correctness')}/{d2.get('correctness')}, "
            f"faithfulness {d1.get('faithfulness')}/{d2.get('faithfulness')})",
            flush=True,
        )

    n = len(out_rows) or 1
    same = sum(1 for r in out_rows if abs(r["delta_score"]) < 1e-9)
    absd = [abs(r["delta_score"]) for r in out_rows]
    print("\n--- Judge 自身抖动 ---")
    print(f"  n = {len(out_rows)}")
    print(f"  两遍分数完全相同：{same}/{len(out_rows)}  ({same / n:.0%})")
    print(f"  平均 |Δ| = {sum(absd) / n:.3f}   最大 |Δ| = {max(absd) if absd else 0:.3f}")
    if absd:
        srt = sorted(absd)
        print(f"  |Δ| 中位数 = {srt[len(srt) // 2]:.3f}   P90 = {srt[int(len(srt) * 0.9)]:.3f}")

    results = [{
        "label": "JUDGE-NOISE",
        "product": args.product,
        "source_label": args.label,
        "aggregate": {
            "n": len(out_rows),
            "identical": same,
            "mean_abs_delta": round(sum(absd) / n, 4),
            "max_abs_delta": round(max(absd), 4) if absd else 0,
        },
        "rows": out_rows,
    }]
    meta = build_meta(args)
    meta.update({
        "script": "probe_judge_noise.py",
        "experiment": "G-JUDGE-NOISE",
        "variable": "无（同一答案、同一证据，Judge 判两遍）",
        "source_product": args.product,
        "source_label": args.label,
        "note": (
            "同答案同证据判两遍，量的是 Judge 自身抖动。"
            "与 exp_g_aa_noise.py 的端到端噪声区别：那边答案也变，这边答案固定。"
        ),
    })
    results[0]["_meta"] = meta
    save(results, args.out)
    print(f"\n产物：data/eval/{args.out}")


if __name__ == "__main__":
    asyncio.run(main())
