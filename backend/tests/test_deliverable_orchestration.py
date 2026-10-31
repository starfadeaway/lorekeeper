"""交付物编排层的**行为基线**（ADR-001 §11 的 P0）。

## 这个文件为什么存在

`DeliverableAgent` 的编排层（规划 → 逐子任务 → 合并 → 冲突 → 渲染）此前
**一条测试都没有**：`test_deliverable_pure.py` 覆盖的是 36 条纯函数，
而"这几个纯函数是怎么被串起来的"没有任何断言。

后果不是"少几条测试"，而是**迁移前后无法证明"没变"**：
不管是换成 LangGraph 还是手写并行，只要没有基线，
"输出一样"就只能靠肉眼看，而肉眼看不出的恰恰是事实顺序、状态字段、
失败降级这些地方。

所以本文件钉三件事：

1. **输出**：同一批 FakeLLM 输入，`render_markdown` 与 `build_json` 要可复现
   （除时间戳 / run_id 外逐字相同）；
2. **顺序**：子任务在产物里的顺序 = 计划顺序（**不是完成顺序**）——
   这是将来并行化最容易破坏、又最不容易被发现的一条；
3. **失败语义**：检索失败、抽取异常、输出截断、证据编号越界、主题闸门，
   各自落到哪个 `status`、哪些计数要留痕。

## 它刻意不做的事

- **不测事件顺序**：`/api/deliverable` 现在是一次性返回，根本没有进度事件。
  事件契约是 P2 的产物，用现在的代码去钉它只会钉出一个假契约。
- **不测耗时**：`latency_ms` 在断言语义里没有意义，确定性测试里一律归零。
- **不测真实检索/真实模型**：那是 `smoke_refactor.py` 的活。这里全部是替身，
  CI 不打网络。

## 这一层能保证什么、不能保证什么

真实 API 下同一题跑两遍本身就不同：A/A 实测「逐字相同」**0/15**、归一化后仍 **0/15**，
只有步数 11/15、工具序列 10/15 这类比率可用（ADR-001 §8）。
所以本文件是**控制流层**的基线；行为层的等价性只能按「归一化 + 比率 + 噪声底」判，
那不是这一层能提供的保证 —— 把它写进这里是防止后来的人拿这个文件去反驳
「真实 API 下为什么两次不一样」。

## 编排层"完成"的判据（给后面的 P1/P2 用）

跑完本文件应能回答三个问题：

- **恢复**：一个子任务失败后，其余子任务的事实还在不在？（`test_*不阻塞*`）
- **审批**：停在 plan 之后时，`plan` / `subject` 是不是可断言的对象？——现在是
  `DeliverableResult.plan` 与 `.subject`，将来换成图状态时这两个字段必须仍然可读；
- **可观测**：每个子任务的 `status` / `facts` 数 / `notes` 是否都在产物里？
  （事件只是这些字段的投影）
"""
from __future__ import annotations

import asyncio
import json
import re

from app.agent.deliverable import (
    DeliverableAgent,
    ParallelDeliverableAgent,
    build_json,
    render_markdown,
)
from app.agent.deliverable_store import InMemorySnapshotStore
from app.agent.tools import ToolContext
from app.retrieval.hybrid import RetrievalHit
from app.store.types import RetrievedChunk
from tests.fakes import FakeLLM

REV = "2025-03-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# 替身：检索器与数据构造
# --------------------------------------------------------------------------- #
def hit(page: str, text: str, *, rev: str = REV, url: str | None = None,
        chunk_id: str | None = None) -> RetrievalHit:
    """构造一条检索结果，metadata 的键集与生产一致（见 conftest.chunk_meta）。"""
    return RetrievalHit(
        chunk=RetrievedChunk(
            chunk_id=chunk_id or f"sr-1-{page}",
            text=text,
            score=0.9,
            metadata={
                "game": "sr",
                "page_id": 1,
                "page_title": page,
                "section": "",
                "chunk_index": 0,
                "last_revision": 1,
                "last_revision_ts": rev,
                "source_url": url or f"https://wiki.invalid/{page}",
                "truncated": 0,
            },
        ),
        score=0.9,
        rank_source="bm25",
    )


class FakeRetriever:
    """按 query 返回预设结果；可指定某些 query 抛异常。

    只记录**被查过哪些 query**，不记录顺序 —— 顺序是并行化以后必然变化的，
    拿它当不变量会把"合理的变化"判成回归。
    """

    def __init__(self, hits_by_query: dict[str, list[RetrievalHit]] | None = None,
                 *, error_for: set[str] | None = None,
                 delay_for: dict[str, float] | None = None) -> None:
        self.hits_by_query = hits_by_query or {}
        self.error_for = error_for or set()
        self.delay_for = delay_for or {}
        self.queries: list[str] = []

    async def retrieve(self, query, config):  # noqa: ANN001 - 与 HybridRetriever 同形状
        self.queries.append(query)
        if query in self.error_for:
            raise RuntimeError("retriever down")
        if query in self.delay_for:
            await asyncio.sleep(self.delay_for[query])
        return list(self.hits_by_query.get(query, []))


class PromptAwareLLM:
    """按 **prompt 内容**回答，而不是按调用顺序。

    `FakeLLM` 是脚本式的（第 N 次调用返回第 N 条）—— 串行下没问题，
    并行下调用顺序会变，抽取结果就会被发给错误的子任务。
    这个替身按 prompt 里的「目标：…」分派，因此串行与并行拿到同一组成分，
    产物才有资格逐字比对。
    """

    def __init__(self, plan_payload: str, facts_by_goal: dict[str, list[dict]]) -> None:
        self.plan_payload = plan_payload
        self.facts_by_goal = facts_by_goal
        self.calls: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001 - 满足 LLMClient 形状
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        if "拆成 3-6" in prompt:
            return {"content": self.plan_payload, "tool_calls": [], "usage": usage}
        for goal, facts in self.facts_by_goal.items():
            if f"目标：{goal}" in prompt:
                return {"content": facts_json(*facts), "tool_calls": [], "usage": usage}
        raise AssertionError(f"PromptAwareLLM 收到未知 prompt：{prompt[:80]!r}")


def plan_json(subtasks: list[dict], *, topic: str = "三月七养成指南",
              subject: str = "三月七") -> str:
    return json.dumps(
        {"topic": topic, "subject": subject, "subtasks": subtasks},
        ensure_ascii=False,
    )


def facts_json(*facts: dict) -> str:
    return json.dumps({"facts": list(facts)}, ensure_ascii=False)


def make_agent(settings, script: list[dict], retriever: FakeRetriever, *, top_k: int = 5):
    ctx = ToolContext(
        settings=settings, retriever=retriever, store=None,
        game="sr", disclose_revision=True,
    )
    return DeliverableAgent(
        settings=settings, llm=FakeLLM(script), ctx=ctx, top_k=top_k,
        # 断点快照注入内存替身：不给的话每个用例都会往 data/runs/ 写文件。
        store=InMemorySnapshotStore(),
    )


def canonical(res) -> tuple[str, dict]:
    """去掉时间戳 / run_id / 延迟这些"每次不同但不该影响语义"的字段。"""
    md = render_markdown(res)
    md = re.sub(r"^> 生成于 .*$", "> 生成于 <ts>", md, flags=re.M)
    md = re.sub(r"耗时 \d+ms", "耗时 <ms>", md)
    data = build_json(res)
    for key in ("run_id", "generated_at", "elapsed_ms"):
        data.pop(key, None)
    for section in data["sections"]:
        section["latency_ms"] = 0
    return md, data


# --------------------------------------------------------------------------- #
# 1. 正常路径：输出应当完整且可复现
# --------------------------------------------------------------------------- #
async def test_两个子任务产出完整交付物(settings):
    retriever = FakeRetriever({
        "三月七 生命值": [hit("三月七", "80级生命值 1200 点")],
        "三月七 遗器": [hit("三月七", "推荐 4 件套")],
    })
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "拿到 80 级生命值", "queries": ["三月七 生命值"]},
            {"name": "遗器", "goal": "推荐遗器", "queries": ["三月七 遗器"]},
        ])},
        {"answer": facts_json(
            {"i": 1, "key": "hp_80", "label": "生命值", "value": "1200", "unit": "点"},
        )},
        {"answer": facts_json(
            {"i": 1, "key": "relic_set", "label": "遗器推荐", "value": "4件套"},
        )},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.meta["facts_total"] == 2
    assert res.meta["subtasks_ok"] == 2
    assert [s.name for s in res.subtasks] == ["属性", "遗器"]

    md, data = canonical(res)
    assert md.startswith("# 三月七养成指南")
    assert "| 生命值 | 1200 | 点 | 三月七 | 2025-03-01 |" in md
    assert data["sections"][0]["facts"][0]["source_url"] == "https://wiki.invalid/三月七"
    # 同一页面被两个子任务各引用一次，但 refs 按 url 去重
    assert len(res.refs) == 1


async def test_同一输入两次运行产出逐字相同(settings):
    """除了时间戳，两次运行必须给出同一个字符串。

    这条是"迁移前后可以对比"的前提：如果同一份输入都不能复现，
    迁移后输出的任何差异都无法归因。
    """
    def build():
        retriever = FakeRetriever({"三月七 生命值": [hit("三月七", "生命值 1200")]})
        script = [
            {"answer": plan_json([
                {"name": "属性", "goal": "拿到生命值", "queries": ["三月七 生命值"]},
            ])},
            {"answer": facts_json(
                {"i": 1, "key": "hp_80", "label": "生命值", "value": "1200"},
            )},
        ]
        return make_agent(settings, script, retriever)

    first = await build().run("整理三月七的养成指南")
    second = await build().run("整理三月七的养成指南")
    assert canonical(first) == canonical(second)


async def test_子任务顺序等于计划顺序(settings):
    """产物顺序必须跟计划走，而不是跟完成时间走。

    并行化之后，完成顺序天然会乱；如果合并时按完成顺序 append，
    产物里的段落顺序就会在两次运行之间漂移 —— 而这件事**不会报错**。
    """
    retriever = FakeRetriever({
        "q1": [hit("三月七", "一")],
        "q2": [hit("三月七", "二")],
        "q3": [hit("三月七", "三")],
    })
    script = [
        {"answer": plan_json([
            {"name": "甲", "goal": "g1", "queries": ["q1"]},
            {"name": "乙", "goal": "g2", "queries": ["q2"]},
            {"name": "丙", "goal": "g3", "queries": ["q3"]},
        ])},
        {"answer": facts_json({"i": 1, "key": "k1", "label": "l1", "value": "v1"})},
        {"answer": facts_json({"i": 1, "key": "k2", "label": "l2", "value": "v2"})},
        {"answer": facts_json({"i": 1, "key": "k3", "label": "l3", "value": "v3"})},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")
    assert [s.name for s in res.subtasks] == ["甲", "乙", "丙"]


# --------------------------------------------------------------------------- #
# 2. 失败语义：每种失败落到哪个 status
# --------------------------------------------------------------------------- #
async def test_检索失败降级为空且不阻塞其他子任务(settings):
    """检索抛异常不是"整个交付物失败"，而是"这个子任务没证据"。

    这是现在的行为，也必须是迁移后的行为：一个子任务的检索挂掉，
    不该把另外五个子任务已经花掉的成本一起丢掉。
    """
    retriever = FakeRetriever(
        {"三月七 技能": [hit("三月七", "战技 造成 120% 伤害")]},
        error_for={"三月七 属性"},
    )
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
            {"name": "技能", "goal": "g2", "queries": ["三月七 技能"]},
        ])},
        # 属性没有召回，不会走到抽取；这里只准备技能那一次
        {"answer": facts_json(
            {"i": 1, "key": "skill_mult", "label": "战技倍率", "value": "120%"},
        )},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert [s.status for s in res.subtasks] == ["empty", "ok"]
    assert res.subtasks[0].notes.startswith("本子任务的关键词没有召回到任何 chunk")
    assert res.meta["facts_total"] == 1
    assert res.meta["subtasks_ok"] == 1


async def test_抽取异常标记error且不影响其他子任务(settings):
    retriever = FakeRetriever({
        "三月七 属性": [hit("三月七", "生命值 1200")],
        "三月七 技能": [hit("三月七", "战技 120%")],
    })
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
            {"name": "技能", "goal": "g2", "queries": ["三月七 技能"]},
        ])},
        {"error": "boom"},  # 第一个子任务的抽取失败
        {"answer": facts_json({"i": 1, "key": "skill_mult", "label": "倍率", "value": "120%"})},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.subtasks[0].status == "error"
    assert "事实抽取失败" in res.subtasks[0].notes
    assert res.subtasks[1].status == "ok"
    assert res.meta["subtasks_error"] == 1


async def test_输出被截断时标记truncated且不采纳事实(settings):
    """`completion_tokens` 顶到上限 == JSON 被截断 == 事实不可信。

    这条曾经把 5/5 子任务静默变成"没查到"（见 deliverable.py 的注释），
    所以截断必须是**独立 status**，不能混进 empty。
    """
    retriever = FakeRetriever({"三月七 属性": [hit("三月七", "生命值 1200")]})
    # 真实截断最常见的形态：JSON 没有闭合括号，_parse_json 只能返回 None。
    # 「完整 JSON + 顶满 usage」的误报场景见下一个用例。
    cut_off = '{"facts": [{"i": 1, "key": "hp", "label": "生命值", "value": "120'
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
        ])},
        {"answer": cut_off, "usage": {"prompt_tokens": 100, "completion_tokens": 2048}},
        # 加预算重试一次，仍然顶到新的上限
        {"answer": cut_off, "usage": {"prompt_tokens": 100, "completion_tokens": 4096}},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.subtasks[0].status == "truncated"
    assert res.subtasks[0].facts == []
    assert res.meta["subtasks_truncated"] == 1


async def test_截断信号误报时也不得采纳事实(settings):
    """`completion_tokens` 顶到上限、但 JSON 恰好完整 —— 误报场景。

    这种组合在真实链路里很少见，但它是**唯一**能让「状态=truncated」
    与「事实表里有值」同时成立的路径。口径只能有一个：
    既然 notes 写"未能采纳"，事实表里就不能出现它，
    否则同一份交付物会同时进「结构化事实」与「未能完成的子任务」两节，
    读者无法判断该信哪一节 —— 而且没有任何报错。
    """
    retriever = FakeRetriever({"三月七 属性": [hit("三月七", "生命值 1200")]})
    payload = facts_json({"i": 1, "key": "hp", "label": "生命值", "value": "1200"})
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
        ])},
        {"answer": payload, "usage": {"prompt_tokens": 100, "completion_tokens": 2048}},
        {"answer": payload, "usage": {"prompt_tokens": 100, "completion_tokens": 4096}},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.subtasks[0].status == "truncated"
    assert res.subtasks[0].facts == []
    md = render_markdown(res)
    assert "未能完成的子任务" in md
    # 不能同时出现在事实表里 —— 这一行是这条用例的全部意义
    assert "| 生命值 |" not in md


async def test_证据编号越界的事实被整条丢弃(settings):
    retriever = FakeRetriever({"三月七 属性": [hit("三月七", "生命值 1200")]})
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
        ])},
        {"answer": facts_json(
            {"i": 99, "key": "hp", "label": "生命值", "value": "1200"},
        )},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.subtasks[0].facts == []
    # 有召回、但事实全被丢掉 → empty，而不是 ok
    assert res.subtasks[0].status == "empty"
    assert "没有抽到" in res.subtasks[0].notes


# --------------------------------------------------------------------------- #
# 3. 质量闸门：主题、去重、冲突
# --------------------------------------------------------------------------- #
async def test_主题闸门拦下无关页面并留痕(settings):
    """被共享属性词召回的其他角色页，事实必须拦下且计数。"""
    retriever = FakeRetriever({"三月七 属性": [hit("长夜月", "生命值 9999")]})
    script = [
        {"answer": plan_json(
            [{"name": "属性", "goal": "g1", "queries": ["三月七 属性"]}],
            subject="三月七",
        )},
        {"answer": facts_json(
            {"i": 1, "key": "hp", "label": "生命值", "value": "9999"},
        )},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert res.meta["topic_gate_enabled"] is True
    assert res.meta["facts_offtopic_dropped"] == 1
    assert res.subtasks[0].offtopic_pages == ["长夜月"]
    assert res.subtasks[0].facts == []


async def test_同页前缀回声合并保留更长值(settings):
    """同一页同 key 同 label、一个值是另一个的前缀 → 合并，留完整的。"""
    long_value = "抗性穿透提高20%。忆灵施放后获得30%忆质"
    retriever = FakeRetriever({"三月七 星魂": [hit("三月七", "星魂6描述")]})
    script = [
        {"answer": plan_json([
            {"name": "星魂", "goal": "g1", "queries": ["三月七 星魂"]},
        ])},
        {"answer": facts_json(
            {"i": 1, "key": "star6_desc", "label": "星魂6描述", "value": "抗性穿透提高20%。"},
            {"i": 1, "key": "star6_desc", "label": "星魂6描述", "value": long_value},
        )},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert len(res.subtasks[0].facts) == 1
    assert res.subtasks[0].facts[0].value == long_value
    assert res.subtasks[0].duplicate_dropped == 1


async def test_跨页面冲突只标记不判对错(settings):
    """同一主语、同一 key、不同页面的不同取值 → 冲突；verdict 必须是"无法判断"。"""
    retriever = FakeRetriever({
        "三月七 属性": [hit("三月七/属性", "生命值 1200")],
        "三月七 技能": [hit("三月七/技能", "生命值 1350")],
    })
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
            {"name": "技能", "goal": "g2", "queries": ["三月七 技能"]},
        ])},
        {"answer": facts_json({"i": 1, "key": "hp_80", "label": "生命值", "value": "1200"})},
        {"answer": facts_json({"i": 1, "key": "hp_80", "label": "生命值", "value": "1350"})},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")

    assert len(res.conflicts) == 1
    assert "无法判断" in res.conflicts[0].verdict
    assert res.meta["conflicts_total"] == 1
    md, _ = canonical(res)
    assert "## ⚠️ 检测到数据冲突" in md


# --------------------------------------------------------------------------- #
# 4. 调用面：谁被调用、调用几次
# --------------------------------------------------------------------------- #
async def test_无召回的子任务不调用抽取(settings):
    """没有证据就不该问模型 —— 那一次调用既不省钱也不产生事实。

    FakeLLM 脚本用完再调用会硬失败，所以"多调了一次"在这里是可测的。
    """
    retriever = FakeRetriever({"三月七 技能": [hit("三月七", "战技 120%")]})
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
            {"name": "技能", "goal": "g2", "queries": ["三月七 技能"]},
        ])},
        {"answer": facts_json({"i": 1, "key": "skill", "label": "倍率", "value": "120%"})},
    ]
    await make_agent(settings, script, retriever).run("整理三月七的养成指南")
    # 只断言"各查了一次"，不断言顺序 —— 并行化后顺序会变，那不是回归。
    assert sorted(retriever.queries) == sorted(["三月七 属性", "三月七 技能"])


async def test_引用按url去重(settings):
    same_url = "https://wiki.invalid/三月七"
    retriever = FakeRetriever({
        "三月七 属性": [hit("三月七", "生命值 1200", url=same_url)],
        "三月七 技能": [hit("三月七", "战技 120%", url=same_url)],
    })
    script = [
        {"answer": plan_json([
            {"name": "属性", "goal": "g1", "queries": ["三月七 属性"]},
            {"name": "技能", "goal": "g2", "queries": ["三月七 技能"]},
        ])},
        {"answer": facts_json({"i": 1, "key": "hp", "label": "生命值", "value": "1200"})},
        {"answer": facts_json({"i": 1, "key": "skill", "label": "倍率", "value": "120%"})},
    ]
    res = await make_agent(settings, script, retriever).run("整理三月七的养成指南")
    assert len(res.refs) == 1
    assert res.refs[0]["url"] == same_url


# --------------------------------------------------------------------------- #
# 5. 规划边界
# --------------------------------------------------------------------------- #
async def test_计划失败时返回空计划而不是硬编码模板(settings):
    """规划失败必须显式空着 —— 退模板会分不清"模型没规划"和"真的只有这几项"。"""
    retriever = FakeRetriever({})
    res = await make_agent(settings, [{"error": "plan down"}], retriever).run("随便问问")

    assert res.plan == []
    assert res.subtasks == []
    assert res.meta["subtasks_total"] == 0
    assert "没有抽到任何事实" in render_markdown(res)


async def test_计划最多六个子任务(settings):
    """规划器最多取 6 个子任务 —— 再多的成本与时长都不可接受。"""
    seven = [{"name": f"任务{i}", "goal": "g", "queries": [f"q{i}"]} for i in range(7)]
    retriever = FakeRetriever({})
    res = await make_agent(
        settings, [{"answer": plan_json(seven)}], retriever,
    ).run("整理三月七的养成指南")

    assert res.meta["subtasks_total"] == 6
    assert [s.name for s in res.subtasks] == [f"任务{i}" for i in range(6)]


# --------------------------------------------------------------------------- #
# 6. 并行对照（P1）：并行不改变产物
# --------------------------------------------------------------------------- #
def _parallel_fixture(settings, *, delay_for: dict[str, float] | None = None):
    """三个子任务、三份事实；并行版最多同时跑三个。"""
    plan_payload = plan_json([
        {"name": "甲", "goal": "g1", "queries": ["q1"]},
        {"name": "乙", "goal": "g2", "queries": ["q2"]},
        {"name": "丙", "goal": "g3", "queries": ["q3"]},
    ])
    facts_by_goal = {
        "g1": [{"i": 1, "key": "k1", "label": "l1", "value": "v1"}],
        "g2": [{"i": 1, "key": "k2", "label": "l2", "value": "v2"}],
        "g3": [{"i": 1, "key": "k3", "label": "l3", "value": "v3"}],
    }
    hits = {
        "q1": [hit("三月七", "一")],
        "q2": [hit("三月七", "二")],
        "q3": [hit("三月七", "三")],
    }

    def build(agent_cls, **kwargs):
        retriever = FakeRetriever(hits, delay_for=delay_for)
        llm = PromptAwareLLM(plan_payload, facts_by_goal)
        ctx = ToolContext(
            settings=settings, retriever=retriever, store=None,
            game="sr", disclose_revision=True,
        )
        return agent_cls(
            settings=settings, llm=llm, ctx=ctx, top_k=5,
            store=InMemorySnapshotStore(), **kwargs,
        ), retriever

    return build


async def test_并行与串行产出逐字相同(settings):
    """P1 的第一个问题：把串行 for 换成 gather，产物会不会变？

    这是"并行不改变语义"在控制流层的证明；真实 API 下的行为层
    只能按"归一化 + 比率 + 噪声底"判（见 ADR-001 §8）。
    """
    build = _parallel_fixture(settings)
    serial, serial_retriever = build(DeliverableAgent)
    parallel, parallel_retriever = build(ParallelDeliverableAgent, parallel_limit=3)

    a = await serial.run("整理三月七的养成指南")
    b = await parallel.run("整理三月七的养成指南")

    assert canonical(a) == canonical(b)
    assert sorted(serial_retriever.queries) == sorted(parallel_retriever.queries)


async def test_并行产物顺序仍按计划(settings):
    """让第一个子任务最慢 —— 产物顺序必须是计划顺序，不是完成顺序。

    并行最常见的回归不是"结果错了"，而是"段落顺序在两次运行之间漂移"，
    而它不会报错。
    """
    build = _parallel_fixture(settings, delay_for={"q1": 0.05})
    parallel, _ = build(ParallelDeliverableAgent, parallel_limit=3)
    res = await parallel.run("整理三月七的养成指南")

    assert [s.name for s in res.subtasks] == ["甲", "乙", "丙"]
    assert [f.value for s in res.subtasks for f in s.facts] == ["v1", "v2", "v3"]
    md, _ = canonical(res)
    assert md.index("### 甲") < md.index("### 乙") < md.index("### 丙")
