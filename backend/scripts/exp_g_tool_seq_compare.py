"""实验 G 的「工具序列相同率」离线对比（不调 API）。

为什么单独一个脚本：对比逻辑要能被单测、能离线重跑，不该绑在一次要花钱的
API 实验上。实验跑到一半挂了，指标计算这部分也应当是验证过的。

口径与 A/A 基线完全一致：**逐题配对计数** —— 一题一票，两臂 `tool_seq`
逐字相同才算这一题相同。不是先各自聚合再比。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval"


def load_rows(path: Path) -> list[dict]:
    """读产物：exp_g_agent_impl 写的是 [HAND 块, LG 块]，两块 rows 拼起来。"""
    blocks = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for block in blocks:
        rows.extend(block.get("rows") or [])
    return rows


def load_baseline(path: Path = EVAL_DIR / "exp_G_aa_noise.json") -> dict:
    """A/A 基线**从产物读**，不手抄 —— 手抄的数字和产物分叉时没人会发现。"""
    block = json.loads(Path(path).read_text(encoding="utf-8"))[0]
    agg = block["aggregate"]
    return {
        "n": agg["n"],
        "tool_seq_identical": agg["tool_seq_identical"],
        "steps_identical": agg["steps_identical"],
    }


def compare(rows: list[dict], *, baseline: dict | None = None) -> dict:
    """逐题配对，算工具序列相同率；缺一边按「不一致」处理并标注。"""
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row.get("impl"))] = row

    per_case: list[dict] = []
    for case_id in sorted(by_case):
        hand = by_case[case_id].get("HAND")
        lg = by_case[case_id].get("LG")
        hand_seq = list((hand or {}).get("tool_seq") or [])
        lg_seq = list((lg or {}).get("tool_seq") or [])
        missing = hand is None or lg is None
        identical = (not missing) and hand_seq == lg_seq
        per_case.append({
            "case_id": case_id,
            "identical": identical,
            "missing_side": missing,
            "hand_calls": len(hand_seq),
            "lg_calls": len(lg_seq),
            "hand": hand_seq,
            "lg": lg_seq,
        })

    n = len(per_case)
    identical_n = sum(1 for p in per_case if p["identical"])
    return {
        "n": n,
        "identical": identical_n,
        "different_cases": [p["case_id"] for p in per_case if not p["identical"]],
        "per_case": per_case,
        "baseline": baseline or {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G 工具序列相同率（离线）")
    ap.add_argument("--in", dest="inp", default="exp_G_sr15_v2.json", help="G 产物文件名")
    ap.add_argument("--aa", default="exp_G_aa_noise.json", help="A/A 基线产物")
    ap.add_argument("--out", default="exp_G_tool_seq.json", help="对比结果文件名")
    args = ap.parse_args()

    rows = load_rows(EVAL_DIR / args.inp)
    result = compare(rows, baseline=load_baseline(EVAL_DIR / args.aa))
    (EVAL_DIR / args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    base = result["baseline"]
    print(f"n={result['n']}  identical={result['identical']}  "
          f"baseline={base.get('tool_seq_identical')}/{base.get('n')}")
    print("different:", ", ".join(result["different_cases"]) or "（无）")
    for p in result["per_case"]:
        mark = "OK" if p["identical"] else "XX"
        print(f"  {mark} {p['case_id']:<10} HAND={p['hand_calls']} LG={p['lg_calls']}")


if __name__ == "__main__":
    main()
