"""LangGraph 版 ReAct 循环 —— 实验 G 的对照臂（不改手写内核）。

## 为什么要做这个文件

2026 年国内 AI 应用开发岗位的 JD 里，LangChain / LangGraph / LlamaIndex 出现频率很高。
本项目是手写 ReAct 内核（`react.py`），简历上明写「未依赖 LangChain / LlamaIndex」。
这条差异化不该为了迎合关键词丢掉，但「没用过」和「评估过并给出结论」是两件事：
前者要挨反问，后者要拿数据说话。

所以这里做的事是**加一个对照臂**，不是换掉主线：用 LangGraph 的 `StateGraph`
复现 `react.py::_run_once` 的同一条 while 循环，让「循环实现」成为实验 G 唯一的变量。

## 复现得有多严格

刻意**继承** `ReActAgent` 并只替换 `_run_once` 一个方法。安全阀
（`_budget_exceeded`）、检索熔断（`SearchGuard`）、提示词（`_prepare`）、
工具执行（`_exec_call`）、注入包裹（`_tool_result_message`）、
引用收集与生成后校验（`_collect_refs` / `_audit_citations`）
全部是**同一份代码**，不是「照着抄一份」。

抄一份的代价：两版会各自演化，之后谁改了哪边都看不出来，
实验数字的分叉就再也归因不到「循环实现」这一个变量上 ——
而单变量是这个项目全部结论的立足点。

## 消息链必须逐字一致

`messages` 全程是 **OpenAI 原生 dict**（`{"role", "content", "tool_calls"}`），
不经过 LangChain 的 message 类型。一旦转成 `AIMessage` / `ToolMessage`，
重序列化时字段顺序、空 content、tool_call_id 的写法都可能变，
测出来的差异里就混进了「消息格式不同」这个隐藏变量。

工具执行节点因此**不**用 LangGraph 的 `ToolNode`：那个节点要 LangChain 的
工具对象与消息类型。我们自己拼 tool 消息，格式与手写版逐字相同。

## 未装 langgraph 时

本模块**不硬依赖** langgraph：import 失败时模块照常可用，
真正用到的 `_build_graph` 才抛带安装指引的异常。
CI 与生产环境因此不必为此多装一个包 —— 对照是实验行为，不是运行必需。
"""
from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from app.agent.react import (
    AgentState,
    ReActAgent,
    SearchGuard,
)
from app.agent.tools import ToolContext

logger = logging.getLogger(__name__)

#: langgraph 是**可选依赖**。import 失败不抛 —— 让模块照常可 import，
#: 这样「没装 langgraph」这件事由真正用到它的那一行报出来，
#: 而不是让整个 `app.agent` 包的 import 链断掉。
_LANGGRAPH_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - 取决于运行环境是否装了 langgraph
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    _LANGGRAPH_IMPORT_ERROR = exc
    END = START = StateGraph = None  # type: ignore[assignment]


class _LGState(TypedDict, total=False):
    """LangGraph 的图状态。

    **刻意不放 messages 的 reducer。** LangChain 默认用 `add_messages`，
    那会把 dict 转成 `AIMessage` / `ToolMessage` 再重新序列化 ——
    字段顺序、空 content、tool_call_id 的写法都可能变，
    「消息格式不同」会变成一个混进结论里的隐藏变量。

    这里所有可变对象（messages / AgentState / ToolContext / SearchGuard）
    都是**引用共享**：节点直接 mutate 它们，图状态里只带一个路由标记。
    不带 checkpointer 时 LangGraph 不会拷贝 state，所以引用保持不变。

    `pending_calls` 存的是 `LLMClient.chat` 返回的**原始** tool_call 格式
    （`{"id", "name", "args"}`），不是 OpenAI 原生格式 ——
    `_exec_call` 要的是前者，而 messages 里存的是后者，两者不能混用。
    """

    messages: list[dict]
    agent_state: AgentState
    context: ToolContext
    guard: SearchGuard
    deadline: float
    t0: float
    route: str
    pending_calls: list[dict]


class LangGraphReActAgent(ReActAgent):
    """用 LangGraph `StateGraph` 跑同一条 ReAct 循环的对照实现。

    与 `ReActAgent` 的差异**只有循环怎么跑**：
    手写版是一个显式 `while True`；这里是一个 `agent → tools → agent` 的图。

    只覆盖**非流式**路径（`_run_once`）。流式路径（`_run_stream`）显式报错：
    评测走的是非流式，留一个能跑但没人验证过的流式分支，
    比明确不支持更危险 —— 它会在有人真去用它的时候静默给出不一样的行为。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 图在**构造时**编译一次，不在每次 run 时编译：
        # 编译是一次性开销，放在构造里才不会被算进每题延迟。
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ #
    def _build_graph(self):
        """编译 StateGraph：START → agent →（条件）→ tools → agent → END。"""
        if _LANGGRAPH_IMPORT_ERROR is not None:
            raise RuntimeError(
                "LangGraphReActAgent 需要 langgraph，但当前环境没装。"
                "装法：pip install 'langgraph==1.2.18'（Python >= 3.10）。"
                "手写版 ReActAgent 不依赖它 —— 这个类只是实验 G 的对照臂。"
            ) from _LANGGRAPH_IMPORT_ERROR

        graph = StateGraph(_LGState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        # 条件边只认一个字符串路由标记。不用 `Command(goto=...)`：
        # 那个写法要在节点里返回 Command 对象，而本图的节点还要 mutate
        # 共享对象，返回值越复杂越容易在「返回值覆盖 state」这件事上出错。
        graph.add_conditional_edges(
            "agent",
            lambda lg: lg.get("route", "end"),
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ------------------------------------------------------------------ #
    async def _agent_node(self, lg: _LGState) -> dict:
        """决策节点：查安全阀 → 问模型 → 有 tool_calls 就交给 tools 节点。

        与手写版 `_run_once` 循环体第 1~3 步逐句对应。
        """
        messages = lg["messages"]
        state: AgentState = lg["agent_state"]
        deadline = lg["deadline"]

        # 1) 安全阀检查放在每轮开头：工具执行完、下一轮决策前。
        reason = self._budget_exceeded(state, deadline)
        if reason:
            await self._finalize(messages, state, reason)
            return {"route": "end"}

        # 2) 决策：把完整 messages 交给模型，带工具 schema。
        try:
            out = await self.chat_fn(messages, tools=self._tool_schemas, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            # 模型调用失败不重试：重试只会放大成本，且同样的输入大概率同样失败。
            logger.exception("模型调用失败")
            state.stop_reason = "error"
            state.answer = f"[系统] 模型调用失败：{type(exc).__name__}: {exc}"
            return {"route": "end"}

        state.prompt_tokens += out["usage"]["prompt_tokens"]
        state.completion_tokens += out["usage"]["completion_tokens"]

        # 3) 没有 tool_calls = 模型认为信息够了，这是终止信号。
        if not out["tool_calls"]:
            messages.append({"role": "assistant", "content": out["content"] or ""})
            state.answer = out["content"]
            state.stop_reason = "answer"
            self._collect_refs(state)
            self._audit_citations(state)
            self.last_state = state
            return {"route": "end"}

        # 工具调用：assistant 消息必须带 tool_calls，后面的 tool 消息才有锚点。
        messages.append(self.llm.assistant_tool_message(out["content"], out["tool_calls"]))
        # 原始 tool_call 走 state 传给 tools 节点 —— 不能从 messages 里反解，
        # 那里存的是 OpenAI 原生格式（name/arguments 嵌在 function 里），
        # 而 `_exec_call` 要的是 `{"id", "name", "args"}`。
        return {"route": "tools", "pending_calls": out["tool_calls"]}

    # ------------------------------------------------------------------ #
    async def _tools_node(self, lg: _LGState) -> dict:
        """工具执行节点：执行本轮全部 tool_calls，逐个回填 tool 消息。

        与手写版 `_run_once` 循环体第 4 步逐句对应。
        """
        messages = lg["messages"]
        state: AgentState = lg["agent_state"]
        context = lg["context"]
        guard = lg["guard"]

        # 一个 assistant 消息的每个 tool_call 后面都紧跟一条 tool 消息，
        # 所以这个 for 里的两行必须成对出现，中途 continue/break 会让消息链断掉。
        for call in lg.get("pending_calls") or []:
            text, _ = await self._exec_call(call, context, guard, state)
            messages.append(self._tool_result_message(call, text, state))

        return {"route": "agent"}

    # ------------------------------------------------------------------ #
    async def _run_once(
        self,
        question: str,
        session_key: str | None,
        history: list[dict] | None,
        ctx: ToolContext | None,
    ) -> AgentState:
        """非流式主循环：走 LangGraph 图，其余与手写版共用。"""
        context, messages, state = self._prepare(question, session_key, history, ctx)
        # 越界问题：不进 ReAct 循环，直接拒答（与手写版同一分支）
        if state.uncovered_game:
            await self._refuse_uncovered(state)
            return state

        guard = SearchGuard()
        state.guard = guard
        t0 = time.perf_counter()
        deadline = t0 + self.s.agent_wall_timeout

        lg: _LGState = {
            "messages": messages,
            "agent_state": state,
            "context": context,
            "guard": guard,
            "deadline": deadline,
            "t0": t0,
            "route": "agent",
        }

        # recursion_limit 是 LangGraph 的硬上限，超了抛 GraphRecursionError。
        # 安全阀在 agent 节点开头就查步数，正常走不到这里；
        # 这个上限是**兜底**，给成「够跑完 max_steps 轮」的量再留点余量。
        # 每轮消耗 2 个 super-step（agent + tools）。
        limit = (self.s.agent_max_steps + 2) * 2 + 10
        try:
            await self._graph.ainvoke(lg, config={"recursion_limit": limit})
        except Exception as exc:  # noqa: BLE001
            # 图跑挂了就如实记下来，绝不假装跑完了：
            # 半截结果混进均值，看起来像「这一档更差」，实际是「它没跑完」。
            logger.exception("LangGraph 图执行失败")
            state.stop_reason = "error"
            state.answer = f"[系统] LangGraph 图执行失败：{type(exc).__name__}: {exc}"

        # elapsed_ms 在**图外**统一设一次：LangGraph 的收尾调度开销
        # 发生在最后一个节点返回之后，只在节点里测会把这部分漏掉。
        # 口径必须是「端到端」—— 两个 arm 用同一种测法，差值才有意义。
        state.messages = messages
        state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self.last_state = state
        return state

    # ------------------------------------------------------------------ #
    def _run_stream(self, question, session_key, history, ctx):  # noqa: ANN001
        """流式路径**未实现** —— 显式报错，不留一个静默走错实现的分支。

        刻意做成**同步**方法（父类是 async generator）：
        同步方法在 `run()` 的协程体里被调用，异常会**立刻**从 `await run(...)`
        抛出来；而 async generator 要等到第一次迭代才炸 ——
        那种「拿到一个流、开始消费才发现不能用」的失败更难排查。

        为什么要显式未实现：父类的 `_run_stream` 是手写版的，
        继承下来能跑、也不报错，只是行为与类名承诺的不一样。
        """
        raise NotImplementedError(
            "LangGraphReActAgent 只实现非流式路径（实验 G 的评测走的就是它）。"
            "要流式请用 ReActAgent —— 两者共用同一份安全阀/熔断/校验代码，"
            "但循环实现不同，流式语义需要单独对齐后再开。"
        )
