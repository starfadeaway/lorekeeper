"""守卫：LangGraph 对照臂与手写内核**行为必须逐字一致**。

实验 G 要回答「手写 ReAct 循环 vs LangGraph StateGraph 差多少」。
这个问题只有一个前提：**两版除了循环实现以外没有任何差别**。

差别的来源有三个，每一个都会污染结论：

1. **消息链不同**。如果两个 arm 发给模型的 `messages` 长得不一样
   （字段顺序、空 content、tool_call_id 的写法、工具结果的包裹方式），
   差异里就混进了「消息格式」这个隐藏变量 ——
   而这恰恰是最容易被读成「框架影响了回答质量」的那一类。

2. **安全阀 / 熔断的行为不同**。手写版在循环开头查 `max_steps`，
   对照臂在图节点里查；两处若不同步，一个跑满 8 步另一个跑 4 步，
   步数差就变成了「实现差异」，而它本该是「策略没变」的证据。

3. **异常路径不同**。模型调用失败时两边都得记成 `stop_reason="error"`，
   而不是一边记错一边把异常抛出去 —— 抛出去的那条 case 会缺行，
   配对检验的分母就悄悄变了，而均值看起来仍然「正常」。

本守卫用**同一个脚本化的 FakeLLM** 分别喂两个实现，然后逐字比对：
最终消息链、模型每一步看到的上下文、答案、终止原因、步数、token 计数。

**未装 langgraph 时全部 skip。** 理由：CI 不必为对照臂多装一个包
（对照是实验行为，不是运行必需），但本地跑对照实验之前必须先过这道守卫。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from app.agent.react import ReActAgent
from app.agent.react_langgraph import (
    _LANGGRAPH_IMPORT_ERROR,
    LangGraphReActAgent,
)
from app.agent.tools import ToolContext
from tests.fakes import FakeLLM

BACKEND = Path(__file__).resolve().parents[1]

#: 没装 langgraph 就跳过「需要对照臂」的用例，但**不**跳过
#: 「生产入口不该引用对照臂」那条 —— 那条与装没装无关。
needs_langgraph = pytest.mark.skipif(
    _LANGGRAPH_IMPORT_ERROR is not None,
    reason="未装 langgraph，对照臂无法运行",
)

TOOL_SCHEMA = [{"type": "function", "function": {"name": "search_kb"}}]


def make_ctx(settings) -> ToolContext:
    """轻量 ToolContext：retriever/store 都不需要。

    刻意**不**用 `ToolContext.build()` —— 那个会真的建 BM25 索引。
    这里要的是「工具层输出完全可控」，用真检索反而让断言依赖语料状态。
    """
    return ToolContext(
        settings=settings, retriever=None, store=None, game="sr", disclose_revision=True
    )


@pytest.fixture()
def stub_tool(monkeypatch):
    """把工具执行换成确定性的假实现。

    两个实现**共用这一个 stub** —— 工具输出逐字相同，
    消息链才有资格逐字比对。
    """

    async def fake_call_tool(name, args, ctx, timeout=None):
        return "检索到：太空封印站\n出处：https://wiki.example/space-station", True

    monkeypatch.setattr("app.agent.react.call_tool", fake_call_tool)


async def run_one(agent_cls, script, settings, ctx):
    """跑一个实现，返回 (AgentState, FakeLLM)。"""
    llm = FakeLLM(script)
    agent = agent_cls(settings=settings, llm=llm, tools=TOOL_SCHEMA)
    state = await agent.run("太空封印站两件套效果是什么？", ctx=ctx)
    return state, llm


# --------------------------------------------------------------------------- #
# 1. 逐字一致：这是本守卫存在的全部理由
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@needs_langgraph
async def test_两版产出的消息链逐字相同(settings, stub_tool):
    """最终 `messages` 必须逐字相同 —— 不只是「都有引用」，而是完全一致。"""
    script = [
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "太空封印站是空间站的封印装置。"},
    ]
    ctx = make_ctx(settings)
    hand_state, _ = await run_one(ReActAgent, script, settings, ctx)
    lg_state, _ = await run_one(LangGraphReActAgent, script, settings, ctx)

    assert hand_state.messages == lg_state.messages, (
        "两个实现发给模型的最终消息链不同 —— 实验 G 的结论会被"
        "「消息格式」这个隐藏变量污染"
    )


@pytest.mark.asyncio
@needs_langgraph
async def test_两版每一步的上下文逐字相同(settings, stub_tool):
    """比最终消息链更强的一层：**每一步**模型看到的 messages 都要一样。

    只比最终链不够：中间某一步的上下文若不同（哪怕最终被追平），
    那一步的模型决策就不可比，而步数与 token 的对比也就失去意义。
    """
    script = [
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"tool": ("search_kb", {"query": "太空封印站 效果"})},
        {"answer": "答案是效果为 A。"},
    ]
    ctx = make_ctx(settings)
    _, hand_llm = await run_one(ReActAgent, script, settings, ctx)
    _, lg_llm = await run_one(LangGraphReActAgent, script, settings, ctx)

    assert hand_llm.seen_messages == lg_llm.seen_messages, (
        "两个实现在某一步给模型看到的上下文不同："
        f"手写 {len(hand_llm.seen_messages)} 步 / 对照 {len(lg_llm.seen_messages)} 步"
    )


@pytest.mark.asyncio
@needs_langgraph
async def test_两版的答案与终止原因一致(settings, stub_tool):
    script = [
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "太空封印站是空间站的封印装置。"},
    ]
    ctx = make_ctx(settings)
    hand_state, _ = await run_one(ReActAgent, script, settings, ctx)
    lg_state, _ = await run_one(LangGraphReActAgent, script, settings, ctx)

    assert hand_state.answer == lg_state.answer
    assert hand_state.stop_reason == lg_state.stop_reason == "answer"
    assert len(hand_state.steps) == len(lg_state.steps)
    assert hand_state.prompt_tokens == lg_state.prompt_tokens
    assert hand_state.completion_tokens == lg_state.completion_tokens


# --------------------------------------------------------------------------- #
# 2. 安全阀：两边都必须在同一步数上停
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@needs_langgraph
async def test_两版都在同一个步数上限停机(settings, stub_tool):
    """max_steps 触发时，两版必须停在同一处、给出同一个终止原因。

    这里刻意让脚本比上限长：FakeLLM 脚本用完再调会 AssertionError，
    所以「多调一次」会变成硬失败而不是悄悄多跑几步。
    """
    max_steps = settings.agent_max_steps
    script = [{"tool": ("search_kb", {"query": f"q{i}"})} for i in range(max_steps)]
    script.append({"answer": "已经到上限了，给出收尾回答。"})
    ctx = make_ctx(settings)
    hand_state, _ = await run_one(ReActAgent, script, settings, ctx)
    lg_state, _ = await run_one(LangGraphReActAgent, script, settings, ctx)

    assert hand_state.stop_reason == "max_steps"
    assert lg_state.stop_reason == "max_steps", (
        f"对照臂的终止原因是 {lg_state.stop_reason!r}，不是 max_steps —— "
        "说明安全阀在两版里不同步"
    )
    assert len(hand_state.steps) == len(lg_state.steps) == max_steps


# --------------------------------------------------------------------------- #
# 3. 异常路径：两边都得记成 error，不能一边抛出去
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@needs_langgraph
async def test_模型失败时两版都记成error而不是抛异常(settings, stub_tool):
    """抛异常的那一版会缺一行产物，配对检验的分母就悄悄变了。"""
    script = [{"error": "boom"}]
    ctx = make_ctx(settings)
    hand_state, _ = await run_one(ReActAgent, script, settings, ctx)
    lg_state, _ = await run_one(LangGraphReActAgent, script, settings, ctx)

    assert hand_state.stop_reason == "error"
    assert lg_state.stop_reason == "error", (
        f"对照臂的终止原因是 {lg_state.stop_reason!r} —— "
        "异常必须被兜住并记成 error，否则那条 case 会缺行"
    )
    assert "boom" in lg_state.answer


# --------------------------------------------------------------------------- #
# 4. 边界：对照臂不得渗进生产路径
# --------------------------------------------------------------------------- #
def test_生产入口没有引用对照臂():
    """对照臂只服务于实验 G；生产 API 必须仍然只挂手写内核。

    对照臂是**实验代码**，不是产品特性。它一旦被接进 `/api/chat`，
    线上跑的就是实验用的那条路径 —— 而那条路径的流式分支是明确未实现的。
    """
    src = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
    assert "LangGraphReActAgent" not in src, "main.py 引入了对照臂"
    assert "react_langgraph" not in src, "main.py 引入了 react_langgraph 模块"


def test_对照臂不改变默认工具集(settings):
    """对照臂继承的是默认工具集，不能偷偷多挂工具。

    实验 G 的单变量是「循环实现」；工具集变了就变成了两个变量。
    """
    agent = LangGraphReActAgent(settings=settings, llm=FakeLLM([]), tools=None)
    names = [t["function"]["name"] for t in agent.tools]
    assert names == settings.agent_tool_names, (
        f"对照臂的默认工具集变成了 {names}，生产默认是 {settings.agent_tool_names}"
    )


@pytest.mark.skipif(_LANGGRAPH_IMPORT_ERROR is None, reason="本环境装了 langgraph")
def test_未装langgraph时报明确错误():
    """没装时要给出**带安装指引**的错误，而不是 ImportError 裸崩。"""
    with pytest.raises(RuntimeError, match="langgraph"):
        LangGraphReActAgent()


@pytest.mark.asyncio
@needs_langgraph
async def test_流式路径显式未实现(settings):
    """留一个「能跑但没人验证过」的流式分支，比明确不支持更危险。

    它会在有人真去用它的时候静默给出不一样的行为 ——
    而 `LangGraphReActAgent` 继承的 `_run_stream` 是手写版的，
    跑起来不会报错，只是行为和类名承诺的不一样。

    异常必须在 `await run(..., stream=True)` 时就抛出来，
    而不是等到第一次迭代 —— 后者会让人以为「拿到了一个能用的流」。
    """
    agent = LangGraphReActAgent(settings=settings, llm=FakeLLM([]), tools=TOOL_SCHEMA)
    with pytest.raises(NotImplementedError):
        await agent.run("随便问点什么", ctx=make_ctx(settings), stream=True)
