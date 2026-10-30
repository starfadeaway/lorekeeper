"""守卫：工具失败必须「看得见」——日志里有一行，SSE 事件里带原因。

起因：2026-10-31 一次 zzz 检索失败，前端只显示「0ms · 80 字符」，
MySQL 恰好停着（`save_run` 静默返回 False），失败文本只喂给了模型。
事后既查不到 DB、也看不到 UI，等于没有任何证据。

所以钉两条不变量：
1. `_exec_call` 对非熔断失败打 WARNING（含工具名 + 错误首行）；
2. 流式 `tool_end` 事件必须带 `error`（成功时为 None）。
"""
from __future__ import annotations

import logging

from app.agent.react import ReActAgent
from app.agent.tools import ToolContext
from tests.fakes import FakeLLM

TOOL_SCHEMA = [{"type": "function", "function": {"name": "search_kb"}}]


def make_ctx(settings) -> ToolContext:
    return ToolContext(
        settings=settings, retriever=None, store=None,
        game="zzz", disclose_revision=True,
    )


async def test_工具失败写日志且tool_end带原因(settings, caplog, monkeypatch):
    async def fake_call_tool(name, args, ctx, timeout=None):  # noqa: ANN001 - 与真实签名同形
        return "[检索失败] RuntimeError: embedding down", False

    monkeypatch.setattr("app.agent.react.call_tool", fake_call_tool)

    agent = ReActAgent(
        settings=settings,
        llm=FakeLLM([
            {"tool": ("search_kb", {"query": "邦布升级"})},
            {"answer": "检索暂时不可用，稍后再试。"},
        ]),
        tools=TOOL_SCHEMA,
    )

    events = []
    with caplog.at_level(logging.WARNING, logger="app.agent.react"):
        async for ev in await agent.run("绝区零的邦布等级怎么提升？", ctx=make_ctx(settings), stream=True):
            events.append(ev)

    tool_ends = [e for e in events if e["event"] == "tool_end"]
    assert tool_ends, "没有收到 tool_end 事件"
    data = tool_ends[0]["data"]
    assert data["ok"] is False
    assert "embedding down" in data["error"]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("工具失败" in r.getMessage() and "search_kb" in r.getMessage() for r in warnings), (
        "失败没有留日志：MySQL 不可用时将无任何证据"
    )


async def test_成功的tool_end不带error(settings, monkeypatch):
    async def fake_call_tool(name, args, ctx, timeout=None):  # noqa: ANN001
        return "[search_kb] status=ok hits=1 top_score=0.9 12ms", True

    monkeypatch.setattr("app.agent.react.call_tool", fake_call_tool)
    agent = ReActAgent(
        settings=settings,
        llm=FakeLLM([
            {"tool": ("search_kb", {"query": "邦布升级"})},
            {"answer": "邦布等级靠邦布经验材料提升。"},
        ]),
        tools=TOOL_SCHEMA,
    )

    events = []
    async for ev in await agent.run("绝区零的邦布等级怎么提升？", ctx=make_ctx(settings), stream=True):
        events.append(ev)

    data = next(e["data"] for e in events if e["event"] == "tool_end")
    assert data["ok"] is True
    assert data["error"] is None
