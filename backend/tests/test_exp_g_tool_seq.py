"""实验 G 工具序列补测的守卫：签名口径、行 schema、对比口径、反向验证。

关键约束：`tool_signature` **必须复用 A/A 基线那一份**（`exp_g_aa_noise.py`），
否则 10/15 的基线与新结果不可比 —— 这比数字错更难发现。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import exp_g_agent_impl as g  # noqa: E402
from exp_g_aa_noise import tool_signature  # noqa: E402
from exp_g_tool_seq_compare import compare  # noqa: E402


def call(name: str, args, cid: str = "call-1") -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def test_签名纯函数可重复(monkeypatch):
    msgs = [call("search_kb", {"query": "三月七"})]
    assert tool_signature(msgs) == tool_signature(msgs)


def test_tool_call_id_不影响签名():
    a = [call("search_kb", {"query": "q"}, cid="aaa")]
    b = [call("search_kb", {"query": "q"}, cid="bbb")]
    assert tool_signature(a) == tool_signature(b)


def test_参数字典序不影响签名():
    a = [call("search_kb", {"query": "q", "game": "sr"})]
    b = [call("search_kb", '{"game": "sr", "query": "q"}')]
    assert tool_signature(a) == tool_signature(b)


class _FakeJudge:
    async def score(self, **kw):
        return _FakeScore()


class _FakeScore:
    def as_dict(self) -> dict:
        return {"correctness": 5.0, "faithfulness": 5.0, "completeness": 5.0,
                "judge_score": 5.0, "reason": "ok", "failed": False}


def _case() -> SimpleNamespace:
    return SimpleNamespace(case_id="sr-001", category="single_hop",
                           question="三月七生命值", gold_answer="1200")


def _state(messages):
    return SimpleNamespace(
        answer="1200", prompt_tokens=10, completion_tokens=5, steps=[1],
        tool_failures=0, refs=[], guard_stats=lambda: {},
        stop_reason="answer", elapsed_ms=1, messages=messages,
    )


async def test_run_case_成功路径带tool_seq(monkeypatch):
    class FakeAgent:
        def __init__(self, **kw):
            pass

        async def run(self, question, ctx=None):
            return _state([call("search_kb", {"query": "q"})])

    monkeypatch.setattr(g, "ReActAgent", FakeAgent)
    row = await g.run_case(_case(), label="HAND", judge=_FakeJudge(),
                           llm=None, schemas=[], ctx={"contexts": "", "tool_ctx": None})
    assert row["tool_seq"] == ['search_kb({"query": "q"})']


async def test_run_case_异常路径也有tool_seq(monkeypatch):
    class BoomAgent:
        def __init__(self, **kw):
            pass

        async def run(self, question, ctx=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(g, "ReActAgent", BoomAgent)
    row = await g.run_case(_case(), label="HAND", judge=_FakeJudge(),
                           llm=None, schemas=[], ctx={"contexts": "", "tool_ctx": None})
    assert row["tool_seq"] == []
    assert "boom" in (row["error"] or "")


def _rows(hand_seq, lg_seq, case_id="c1"):
    return [
        {"case_id": case_id, "impl": "HAND", "tool_seq": hand_seq},
        {"case_id": case_id, "impl": "LG", "tool_seq": lg_seq},
    ]


def test_对比_一致与不一致():
    same = compare(_rows(["a(1)"], ["a(1)"]))
    assert same["n"] == 1 and same["identical"] == 1
    diff = compare(_rows(["a(1)"], ["a(2)"]))
    assert diff["identical"] == 0 and diff["different_cases"] == ["c1"]


def test_对比_缺一边算不一致且不崩():
    rows = [{"case_id": "c1", "impl": "HAND", "tool_seq": ["a(1)"]}]
    out = compare(rows)
    assert out["n"] == 1 and out["identical"] == 0
    assert out["per_case"][0]["missing_side"] is True


def test_反向验证_换签名口径结果会变():
    """只取工具名时两题相同；带参数时不同 —— 证明口径真的参与了判定。"""
    m1 = [call("search_kb", {"query": "a"})]
    m2 = [call("search_kb", {"query": "b"})]

    def name_only(messages):
        return [c["function"]["name"] for m in messages or []
                for c in (m.get("tool_calls") or [])]

    with_args = compare(_rows(tool_signature(m1), tool_signature(m2)))
    by_name = compare(_rows(name_only(m1), name_only(m2)))

    assert with_args["identical"] == 0
    assert by_name["identical"] == 1
