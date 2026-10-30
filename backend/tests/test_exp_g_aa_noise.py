"""守卫：A/A 噪声底脚本的 Judge 失败**必须留痕**，且不得污染噪声底。

缺陷（2026-10-30 实测踩到，DeepSeek 评审时指出）：
    `exp_g_aa_noise.py` 第一版把 `JudgeScore.as_dict()` 当成有 `judge_reason` 键，
    写成 `d1.get("judge_reason")` —— 但 `as_dict()` 的键是 **`reason`**，
    没有 `judge_reason`。所以取到的**永远是 None**。

    为什么这个 bug 比「报个错」严重得多 —— 它是**静默**的：
      1. Judge 超时（`judge.py` 捕获异常后写 `judge_error:APITimeoutError`
         并把分数兜成 0）在产物里**零留痕**。15 题里那 5 个 0 分，
         半年后没人分得清是「真判了 0」还是「链路超时」；
      2. 噪声底只能给一个**区间**：剔除失败对是 0.75，把兜底 0 当真分算是 1.77。
         产物无法自证是哪一个 —— 一个数字有两种读法，就等于没有数字。

    这正是本项目的老坑：**评测链路的故障被读成了被测对象的能力**。
    `exp_common.judge_failed()` 早就为这件事而生（实验 E 的 sr-008），
    本脚本第一版没接上它。

本组测试锁住六条不变量：
    1. `as_dict()` 的键是 `reason`、**没有** `judge_reason`（锁住这个坑本身，
       防止有人「顺手」把 `judge_reason` 加回 as_dict 造成两边口径打架）；
    2. Judge 失败必须被标出来（`failed=True` + 原始 reason 留着）；
    3. **失败对不参与**噪声底的均值 / 最大值 —— 一边超时兜成 0 时，
       「真分 − 0」不是抖动，是链路故障；
    4. 失败对的 Δ 必须是 `None`，不能是「0 减法」出来的假值；
    5. **归一化判定必须比归一化之后的差异** —— 只差耗时的站点是 `raw_only`，
       归一化之后两版仍然一致；
    6. 只有真实分叉站点才进分叉归因表，纯计时噪声单列。

第 5、6 条锁的是第二个缺陷（同样 2026-10-30 实测发现）：
第一版把 `normalized_identical` 写成 `not diffs`，而 `diffs` 收录的是**原始**差异 ——
归一化压根没参与判定，只是碰巧与真值同号。
同理，`buckets` 原来把纯耗时站点也计成「工具侧分叉」，
会让「工具侧 37 个分叉点」与「检索侧噪声底 = 0」自相矛盾。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp_g_aa_noise as aa  # noqa: E402
from app.eval.judge import JudgeScore  # noqa: E402
from exp_common import JUDGE_ERROR_PREFIX  # noqa: E402

# --------------------------------------------------------------------------- #
# 构造：模拟一次 Judge 打分的结果
# --------------------------------------------------------------------------- #

def _ok(score: float, reason: str = "答得不错") -> dict:
    """一次**正常**的 Judge 打分（as_dict 的形状）。"""
    return JudgeScore(
        correctness=score, faithfulness=score, completeness=score,
        reason=reason, raw="",
    ).as_dict()


def _failed(exc: str = "APITimeoutError") -> dict:
    """一次**失败**的 Judge 打分 —— judge.py 捕获异常后的兜底形状。"""
    return JudgeScore(
        correctness=0.0, faithfulness=0.0, completeness=0.0,
        reason=f"{JUDGE_ERROR_PREFIX}:{exc}", raw="",
    ).as_dict()


def _row(case_id: str, d1: dict, d2: dict, *, agent_failed: bool = False) -> dict:
    return {"case_id": case_id, "judge": aa.judge_pair(d1, d2, agent_failed=agent_failed)}


# --------------------------------------------------------------------------- #
# 1. 键名本身：锁住这个坑
# --------------------------------------------------------------------------- #

def test_as_dict_key_is_reason_not_judge_reason():
    """`as_dict()` 的键是 `reason`。

    这条看起来像废话，但它锁的是本 bug 的**根**：
    只要有人给 `as_dict()` 加一个 `judge_reason` 键（或把脚本改成读它），
    两套字段名就会同时存在、指向同一件事，且永远不会报错。
    """
    d = _ok(4.0)
    assert "reason" in d
    assert "judge_reason" not in d, (
        "as_dict() 不该有 judge_reason —— 本项目统一用 reason，"
        "judge_reason 是 exp_common.case_row 摊进 row 之后的字段名"
    )


def test_judge_side_reads_reason_not_none():
    """`judge_side` 必须真的取到 reason，而不是 None。"""
    side = aa.judge_side(_ok(4.0, reason="命中关键事实"))
    assert side["reason"] == "命中关键事实"
    assert side["failed"] is False
    assert side["score"] == 4.0


# --------------------------------------------------------------------------- #
# 2. 失败必须留痕
# --------------------------------------------------------------------------- #

def test_judge_side_marks_failure_and_keeps_reason():
    side = aa.judge_side(_failed())
    assert side["failed"] is True
    assert side["reason"].startswith(JUDGE_ERROR_PREFIX), (
        "失败原因必须原样留着 —— 半年后要靠它分清「判了 0」还是「超时」"
    )
    assert side["score"] == 0.0


def test_judge_failed_matches_exp_common_judgement():
    """本脚本的判据必须与 `exp_common.judge_failed` 同口径。

    两处各判各的 = 同一个前缀在两个文件里含义可能漂移。
    """
    import exp_common

    for reason in (None, "", "正常理由", f"{JUDGE_ERROR_PREFIX}:APITimeoutError",
                   f"{JUDGE_ERROR_PREFIX}:RateLimitError"):
        assert aa.judge_failed(reason) == exp_common.judge_failed(
            {"judge_reason": reason}
        ), f"口径不一致：reason={reason!r}"


def test_empty_answer_reason_is_not_a_failure():
    """回归保护：空回答走的是「空回答」，前缀不同，不能被当成链路失败。

    这与 `exp_common` 的既有约定一致 —— 空回答是**模型**没答出来，
    judge_error 是**链路**挂了，两者方向相反，混起来会再次误判均值。
    """
    assert aa.judge_failed("空回答") is False


# --------------------------------------------------------------------------- #
# 3. 失败对不污染噪声底
# --------------------------------------------------------------------------- #

def test_failed_pair_has_no_delta():
    """一边超时 → Δ 必须是 None，不能是「真分 − 0」。"""
    pair = aa.judge_pair(_ok(4.0), _failed())
    assert pair["valid"] is False
    assert pair["delta"] is None
    assert pair["abs_delta"] is None


def test_valid_pair_has_delta():
    pair = aa.judge_pair(_ok(4.0), _ok(2.0))
    assert pair["valid"] is True
    assert pair["delta"] == pytest.approx(2.0)
    assert pair["abs_delta"] == pytest.approx(2.0)


def test_judge_stats_excludes_failed_pairs_from_mean():
    """核心：失败对不进均值。

    构造一个「失败对会把均值显著拉大」的场景：
      有效对 Δ 分别是 0 和 0（均值 0）
      再加一个失败对（真分 4.0 vs 兜底 0 → 减出来是 4.0）
    如果失败对没被剔除，均值会变成 4/3 ≈ 1.33 —— 一眼就看得出来。
    """
    rows = [
        _row("a", _ok(4.0), _ok(4.0)),
        _row("b", _ok(2.0), _ok(2.0)),
        _row("c", _ok(4.0), _failed()),
    ]
    st = aa.judge_stats(rows)
    assert st["n_valid"] == 2
    assert st["n_failed_pairs"] == 1
    assert st["mean_abs_delta"] == pytest.approx(0.0)
    assert st["max_abs_delta"] == pytest.approx(0.0)
    assert st["failed_case_ids"] == ["c"]


def test_judge_stats_counts_failures_per_side():
    """失败要分侧计数 —— 「两遍都超时」和「一遍超时」不是一回事。"""
    rows = [
        _row("a", _failed(), _ok(4.0)),      # p1 失败
        _row("b", _ok(4.0), _failed()),      # p2 失败
        _row("c", _failed(), _failed()),     # 两遍都失败
    ]
    st = aa.judge_stats(rows)
    assert st["n_judged"] == 3
    assert st["n_valid"] == 0
    assert st["n_failed_pairs"] == 3
    assert st["p1_failures"] == 2
    assert st["p2_failures"] == 2


def test_judge_stats_reports_identical_and_differing_pairs():
    rows = [
        _row("a", _ok(4.0), _ok(4.0)),   # 同分
        _row("b", _ok(4.0), _ok(2.0)),   # 差 2.0
        _row("c", _ok(3.0), _ok(2.5)),   # 差 0.5
    ]
    st = aa.judge_stats(rows)
    assert st["identical_pairs"] == 1
    assert st["n_valid"] == 3
    assert st["mean_abs_delta"] == pytest.approx((0.0 + 2.0 + 0.5) / 3, abs=1e-4)
    assert st["max_abs_delta"] == pytest.approx(2.0)


def test_judge_stats_on_empty_rows_is_safe():
    """没跑 judge 时不能炸 —— aggregate 里那一节要能安全缺席。"""
    st = aa.judge_stats([])
    assert st["n_judged"] == 0
    assert st["n_valid"] == 0
    assert st["mean_abs_delta"] is None
    assert st["max_abs_delta"] is None


def test_judge_stats_ignores_rows_without_judge_key():
    """没开 --judge 时 row 里没有 judge 键，这些行不算「失败」。"""
    rows = [{"case_id": "a"}, {"case_id": "b", "judge": aa.judge_pair(_ok(4.0), _ok(4.0))}]
    st = aa.judge_stats(rows)
    assert st["n_judged"] == 1
    assert st["n_valid"] == 1
    assert st["n_failed_pairs"] == 0


# --------------------------------------------------------------------------- #
# 4. Agent 侧失败也要排除
# --------------------------------------------------------------------------- #

def test_agent_failed_pair_is_not_counted_as_identical():
    """Agent 抛异常 → 答案是空串 → Judge 回「空回答」（**不是** judge_error）。

    两遍都空时 Δ=0，看上去像「两遍一模一样」，
    实际是**两遍都没跑成** —— 那不是一致性，是没数据。
    """
    pair = aa.judge_pair(
        _ok(0.0, reason="空回答"), _ok(0.0, reason="空回答"), agent_failed=True
    )
    assert pair["valid"] is False
    assert pair["delta"] is None


def test_judge_stats_separates_agent_failures_from_judge_failures():
    """两种不可比要分开计数 —— 性质不同：一个是 Agent 挂了，一个是 Judge 挂了。"""
    rows = [
        _row("a", _ok(4.0), _ok(4.0)),
        _row("b", _ok(0.0, reason="空回答"), _ok(0.0, reason="空回答"), agent_failed=True),
        _row("c", _ok(4.0), _failed()),
    ]
    st = aa.judge_stats(rows)
    assert st["n_judged"] == 3
    assert st["n_valid"] == 1
    assert st["n_failed_pairs"] == 2
    assert st["agent_failed_pairs"] == 1
    assert st["judge_failed_pairs"] == 1
    assert st["failed_case_ids"] == ["b", "c"]


# --------------------------------------------------------------------------- #
# 5. 归一化判定：必须比的是**归一化之后**的差异，不是原始差异
# --------------------------------------------------------------------------- #

def _msg(role: str, content: str, tool_calls=None) -> dict:
    return {"role": role, "content": content, "tool_calls": tool_calls or []}


def test_timing_only_difference_is_raw_only():
    """只有耗时不同 → 归一化后一致 → kind=raw_only，**不算行为分叉**。

    这是第二版修掉的缺陷：第一版把 `normalized_identical` 写成 `not diffs`，
    而 `diffs` 收的是**原始**差异 —— 归一化压根没参与判定。
    换成一批「只差耗时」的数据时，第一版会给出「归一化后不一致」的假结论。
    """
    m1 = [_msg("tool", "检索到 3 条（37ms 来源=bm25+vector+rerank）")]
    m2 = [_msg("tool", "检索到 3 条（40ms 来源=bm25+vector+rerank）")]
    diffs = aa.all_diffs(m1, m2)
    assert len(diffs) == 1
    assert diffs[0]["kind"] == "raw_only"
    assert aa.normalized_identical(diffs) is True


def test_tool_call_id_difference_is_raw_only():
    """tool_call id 由服务端随机生成 → 归一化后一致。"""
    m1 = [_msg("tool", json.dumps({"tool_call_id": "call_00_abc", "text": "x"}))]
    m2 = [_msg("tool", json.dumps({"tool_call_id": "call_00_xyz", "text": "x"}))]
    diffs = aa.all_diffs(m1, m2)
    assert diffs[0]["kind"] == "raw_only"
    assert aa.normalized_identical(diffs) is True


def test_real_content_difference_is_not_normalized_identical():
    """措辞不同是真的分叉 —— 模型这次的说法变了。"""
    diffs = aa.all_diffs([_msg("assistant", "有对话、无商店")],
                         [_msg("assistant", "无商店，有对话")])
    assert diffs[0]["kind"] == "content"
    assert "content_diff" in diffs[0]
    assert aa.normalized_identical(diffs) is False


def test_tool_call_arguments_difference_is_tool_calls_kind():
    """工具参数不同 → 模型换了动作，属真实分叉。"""
    m1 = [_msg("assistant", "", [{"function": {"name": "search_kb", "arguments": {"query": "a b"}}}])]
    m2 = [_msg("assistant", "", [{"function": {"name": "search_kb", "arguments": {"query": "b a"}}}])]
    diffs = aa.all_diffs(m1, m2)
    assert diffs[0]["kind"] == "tool_calls"
    assert aa.normalized_identical(diffs) is False


def test_missing_message_is_hard_difference():
    """一边多一条消息 → 结构就不一样，是最硬的差异。"""
    diffs = aa.all_diffs([_msg("assistant", "a")],
                         [_msg("assistant", "a"), _msg("assistant", "b")])
    assert diffs[0]["kind"] == "missing"
    assert aa.normalized_identical(diffs) is False


def test_identical_messages_produce_no_diff():
    m = [_msg("assistant", "same")]
    assert aa.all_diffs(m, [_msg("assistant", "same")]) == []
    assert aa.normalized_identical([]) is True


def test_mixed_sites_only_real_ones_count():
    """混合站点：只要有一个真实分叉，整条就不算「归一化后一致」。"""
    m1 = [_msg("tool", "命中 3 条（37ms）"), _msg("assistant", "答案甲")]
    m2 = [_msg("tool", "命中 3 条（40ms）"), _msg("assistant", "答案乙")]
    diffs = aa.all_diffs(m1, m2)
    kinds = sorted(d["kind"] for d in diffs)
    assert kinds == ["content", "raw_only"]
    assert aa.normalized_identical(diffs) is False


# --------------------------------------------------------------------------- #
# 6. 跑完必须清掉 partial（它没有 _meta / aggregate，留着会被误读）
# --------------------------------------------------------------------------- #

def test_clear_partial_removes_the_intermediate_file(tmp_path, monkeypatch):
    """partial 是中间产物：跑完就要删。

    它和正式产物同名主体，只缺 `_meta` 与 `aggregate` ——
    留在 `data/eval/` 里，半年后有人读错文件都不知道读到了哪一份。
    """
    import exp_common

    monkeypatch.setattr(exp_common, "EVAL_DIR", tmp_path)
    aa._dump_partial([{"case_id": "a"}], "exp_G_aa_noise.json")
    p = tmp_path / "exp_G_aa_noise.partial.json"
    assert p.exists()
    aa._clear_partial("exp_G_aa_noise.json")
    assert not p.exists(), "跑完没清 partial，中间产物会跟正式产物混淆"


def test_clear_partial_is_safe_when_absent(tmp_path, monkeypatch):
    """没有 partial 时调用它不该抛 —— 正常路径也可能先删过。"""
    import exp_common

    monkeypatch.setattr(exp_common, "EVAL_DIR", tmp_path)
    aa._clear_partial("exp_G_aa_noise.json")
