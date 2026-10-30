"""守卫：Judge 抖动探针的失败对不能混进抖动统计。

缺陷（2026-10-30，与 `exp_g_aa_noise.py` 同源）：
    两份脚本第一版都把 `JudgeScore.as_dict()` 当成有 `judge_reason` 键 ——
    而 `as_dict()` 的键是 **`reason`**。读错键名**静默**拿到 None：
    Judge 超时在产物里零留痕，而汇总统计把「真分 − 兜底 0」当成了抖动，
    把「Judge 抖动幅度」这个数算大。

    本探针测的是「同一答案判两遍」，所以**每一条有效对都是纯 Judge 抖动**；
    一旦混进失败对，测出来的就不是 Judge 的抖，而是链路的故障。

本组测试锁住：
    1. 失败按 `reason` 的 `judge_error:` 前缀识别；
    2. 失败对不进统计口径；
    3. 判据与 `exp_common.judge_failed` 同源（不另起一套）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp_common  # noqa: E402
import probe_judge_noise as probe  # noqa: E402
from app.eval.judge import JudgeScore  # noqa: E402
from exp_common import JUDGE_ERROR_PREFIX  # noqa: E402


def _ok(score: float) -> dict:
    return JudgeScore(
        correctness=score, faithfulness=score, completeness=score,
        reason="答得不错", raw="",
    ).as_dict()


def _failed(exc: str = "APITimeoutError") -> dict:
    return JudgeScore(
        correctness=0.0, faithfulness=0.0, completeness=0.0,
        reason=f"{JUDGE_ERROR_PREFIX}:{exc}", raw="",
    ).as_dict()


def test_as_dict_has_reason_not_judge_reason():
    """锁住键名 —— 根因就在这里。"""
    d = _ok(4.0)
    assert "reason" in d
    assert "judge_reason" not in d


def test_judge_failed_reason_detects_prefix():
    assert probe.judge_failed_reason(f"{JUDGE_ERROR_PREFIX}:APITimeoutError") is True
    assert probe.judge_failed_reason(f"{JUDGE_ERROR_PREFIX}:RateLimitError") is True


def test_judge_failed_reason_ignores_normal_reasons():
    """回归保护：正常理由与「空回答」都不是链路失败。

    「空回答」是**模型**没答出来，`judge_error:` 是**链路**挂了，
    两者方向相反 —— 混起来会把「模型答不出」读成「评测链路抖」。
    """
    assert probe.judge_failed_reason("答得不错") is False
    assert probe.judge_failed_reason("空回答") is False
    assert probe.judge_failed_reason(None) is False
    assert probe.judge_failed_reason("") is False


def test_judge_failed_reason_matches_exp_common():
    """判据必须与 `exp_common.judge_failed` 同口径。"""
    for reason in (None, "", "正常理由", "空回答",
                   f"{JUDGE_ERROR_PREFIX}:APITimeoutError"):
        assert probe.judge_failed_reason(reason) == exp_common.judge_failed(
            {"judge_reason": reason}
        ), f"口径不一致：reason={reason!r}"
