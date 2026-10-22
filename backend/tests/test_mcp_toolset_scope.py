"""守卫：MCP 侧与 Agent 侧的工具集**本来就该不同**。

这不是 bug，是两个消费者：

  * MCP 面向外部 host（Claude Desktop / Cursor），要全量 4 个工具才能做问答；
  * 本项目 Agent 走进程内直调，取 `AGENT_TOOLS`（生产默认只有 search_kb）。

此前 `mcp_server.build_server` 的注释写成「必须与 Agent 启用的是同一个」，
而 `mcp_stdio_entry` 又没传 `only` —— 代码违背了自己的注释，且没有测试
指出这层矛盾（`test_mcp_parity` 反倒把「全量」当成了预期）。

本守卫锁住三件事：
  1. 生产 Agent 默认只有 search_kb；
  2. MCP 暴露的是全量 MCP_TOOLS；
  3. **两者不相等** —— 如果哪天有人把 MCP 改成跟随 AGENT_TOOLS，
     这条会红，提醒他先想清楚外部 host 会不会因此少工具。
"""
from __future__ import annotations

import pytest
from app.agent.mcp_server import MCP_TOOLS, build_server
from tests.fakes import FakeCtx


def test_生产Agent默认只有search_kb(settings):
    assert settings.agent_tool_names == ["search_kb"], (
        f"生产默认工具集变了：{settings.agent_tool_names}"
    )


def test_MCP暴露的是全量工具():
    assert set(MCP_TOOLS) == {"search_kb", "get_page", "calculator", "list_games"}, (
        f"MCP_TOOLS 变了：{set(MCP_TOOLS)}"
    )


@pytest.mark.asyncio
async def test_两个口径不该相等_差异是刻意的(settings):
    """如果有人把 MCP 改成跟随 AGENT_TOOLS，这条会红。"""
    from mcp import Client

    server = build_server(FakeCtx())  # 不传 only = 全量
    async with Client(server) as client:
        res = await client.list_tools()
        mcp_names = {t.name for t in res.tools}

    agent_names = set(settings.agent_tool_names or [])
    assert mcp_names == set(MCP_TOOLS), f"MCP 侧不是全量：{mcp_names}"
    assert mcp_names != agent_names, (
        "MCP 暴露的工具集与生产 Agent 相同了 —— 这会让外部 host 只拿到 "
        f"{agent_names}。若是有意为之，请先确认外部接入方真的不需要 "
        "calculator / get_page。"
    )
