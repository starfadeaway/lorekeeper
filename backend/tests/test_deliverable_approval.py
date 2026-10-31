"""审批（HITL）的守卫：只出计划不执行、批准原计划、改计划、非法计划被拒。

全部离线：FakeLLM + FakeRetriever + InMemorySnapshotStore。
"""
from __future__ import annotations

import json

import pytest
from app.agent.deliverable import DeliverableAgent, Fact, SubtaskResult
from app.agent.deliverable_store import InMemorySnapshotStore
from app.agent.tools import ToolContext
from app.retrieval.hybrid import RetrievalHit
from app.store.types import RetrievedChunk
from tests.fakes import FakeLLM

REV = "2025-03-01T00:00:00Z"


def hit(page: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk=RetrievedChunk(
            chunk_id=f"sr-1-{page}", text=text, score=0.9,
            metadata={"game": "sr", "page_id": 1, "page_title": page,
                      "section": "", "chunk_index": 0, "last_revision": 1,
                      "last_revision_ts": REV,
                      "source_url": f"https://wiki.invalid/{page}", "truncated": 0},
        ),
        score=0.9, rank_source="bm25",
    )


class FakeRetriever:
    def __init__(self, hits: dict[str, list[RetrievalHit]] | None = None) -> None:
        self.hits = hits or {}
        self.queries: list[str] = []

    async def retrieve(self, query, config):  # noqa: ANN001
        self.queries.append(query)
        return list(self.hits.get(query, []))


def make_agent(settings, script, retriever, store):
    ctx = ToolContext(settings=settings, retriever=retriever, store=None,
                      game="sr", disclose_revision=True)
    return DeliverableAgent(settings=settings, llm=FakeLLM(script), ctx=ctx,
                            top_k=5, store=store)


def plan_json(names: list[str], subject: str = "三月七") -> str:
    return json.dumps(
        {"topic": "三月七养成", "subject": subject,
         "subtasks": [{"name": n, "goal": f"g-{n}", "queries": [f"q-{n}"]} for n in names]},
        ensure_ascii=False,
    )


def facts_json(key: str, value: str) -> str:
    return json.dumps(
        {"facts": [{"i": 1, "key": key, "label": key, "value": value}]},
        ensure_ascii=False,
    )


def done_subtask(name: str, key: str, value: str) -> SubtaskResult:
    return SubtaskResult(
        name=name, goal=f"g-{name}", status="ok", queries=[f"q-{name}"],
        facts=[Fact(key=key, label=key, value=value,
                    source_url="https://wiki.invalid/三月七", page_title="三月七",
                    revision=REV, subtask=name)],
    )


async def draft(settings, names: list[str], extract_names: list[str] | None = None):
    """建草稿；`extract_names` 控制脚本里准备哪几个子任务的抽取结果。

    改计划/崩溃续跑的用例里，只保留真正会执行的那些，避免"脚本错位"。
    """
    extract_names = extract_names if extract_names is not None else names
    store = InMemorySnapshotStore()
    retriever = FakeRetriever({f"q-{n}": [hit("三月七", f"{n} 数值 1")] for n in names})
    agent = make_agent(settings, [{"answer": plan_json(names)}] + [
        {"answer": facts_json(f"k{n}", "1")} for n in extract_names
    ], retriever, store)
    res = await agent.plan_draft("整理三月七的养成指南")
    return res, agent, store, retriever


# --------------------------------------------------------------------------- #
async def test_只出计划不执行(settings):
    res, agent, store, retriever = await draft(settings, ["甲", "乙"])

    assert res.plan and [p["name"] for p in res.plan] == ["甲", "乙"]
    assert agent.llm.calls == 1, "规划之外不应有抽取调用"
    assert retriever.queries == [], "不执行就不应检索"
    ckpt = store.saved[res.run_id]
    assert ckpt.phase == "awaiting_approval"
    assert ckpt.subtasks == []


async def test_批准原计划后执行全部子任务(settings):
    draft_res, agent, store, retriever = await draft(settings, ["甲", "乙"])

    res = await agent.approve(draft_res.run_id)

    assert [s.name for s in res.subtasks] == ["甲", "乙"]
    assert len(res.subtasks[0].facts) == 1
    assert agent.llm.calls == 3  # 1 次规划 + 2 次抽取
    assert store.saved[draft_res.run_id].phase == "done"


async def test_改计划后只执行保留下来的子任务(settings):
    draft_res, agent, store, retriever = await draft(
        settings, ["甲", "乙"], extract_names=["乙"],
    )
    edited = [{"name": "乙", "goal": "g-乙", "queries": ["q-乙"]}]

    res = await agent.approve(draft_res.run_id, plan=edited)

    assert [s.name for s in res.subtasks] == ["乙"]
    assert retriever.queries == ["q-乙"]
    assert agent.llm.calls == 2  # 规划 + 一次抽取
    assert store.saved[draft_res.run_id].plan[0]["name"] == "乙"


async def test_非法计划被拒且不触发执行(settings):
    draft_res, agent, store, retriever = await draft(settings, ["甲", "乙"])
    calls_before = agent.llm.calls

    bad_plans = [
        [],
        [{"name": "甲", "goal": "g", "queries": []}],
        [{"name": "甲", "goal": "g", "queries": ["q"]},
         {"name": "甲", "goal": "g", "queries": ["q"]}],
        [{"name": f"n{i}", "goal": "g", "queries": ["q"]} for i in range(7)],
        [{"name": "甲", "goal": "", "queries": ["q"]}],
    ]
    for bad in bad_plans:
        with pytest.raises(ValueError):
            await agent.approve(draft_res.run_id, plan=bad)

    assert agent.llm.calls == calls_before, "非法计划不应产生任何模型调用"
    assert retriever.queries == []


async def test_未知run_id与重复审批(settings):
    agent = make_agent(settings, [], FakeRetriever(), InMemorySnapshotStore())
    with pytest.raises(KeyError):
        await agent.approve("run00000000000009")

    draft_res, agent, store, _ = await draft(settings, ["甲"])
    await agent.approve(draft_res.run_id)
    with pytest.raises(RuntimeError):
        await agent.approve(draft_res.run_id)


async def test_崩溃后重新审批只跑未完成(settings):
    draft_res, agent, store, retriever = await draft(
        settings, ["甲", "乙"], extract_names=["乙"],
    )
    # 模拟第一次审批跑到一半崩了：快照里只剩"甲"完成、phase=running
    ckpt = store.saved[draft_res.run_id]
    ckpt.phase = "running"
    ckpt.subtasks = [asdict_subtask(done_subtask("甲", "k甲", "1"))]
    agent.llm.calls = 0  # 重置计数，只看续跑部分

    res = await agent.approve(draft_res.run_id)

    assert [s.name for s in res.subtasks] == ["甲", "乙"]
    assert retriever.queries == ["q-乙"], "已完成的甲不该再检索"
    assert agent.llm.calls == 1, "只该抽取乙"


def asdict_subtask(st: SubtaskResult) -> dict:
    from dataclasses import asdict
    return asdict(st)


async def test_状态接口返回完整计划而不是名字(settings, monkeypatch):
    """审批要基于状态里的计划改，所以 `plan` 必须是完整对象。

    2026-10-31 冒烟发现的契约 bug：状态接口只回名字，调用方拼不出 approve 入参。
    """
    from app.agent.deliverable_store import Checkpoint
    from app.main import deliverable_status

    store = InMemorySnapshotStore()
    await store.save(Checkpoint(
        run_id="run00000000000011", question="整理三月七的养成指南", game="sr",
        topic="三月七养成", subject="三月七",
        plan=[{"name": "甲", "goal": "g甲", "queries": ["q甲"]}],
        subtasks=[], phase="awaiting_approval",
    ))
    monkeypatch.setattr("app.main.FileSnapshotStore", lambda _root: store)

    out = await deliverable_status("run00000000000011")

    assert out["plan"][0]["name"] == "甲"
    assert out["plan"][0]["queries"] == ["q甲"]
    assert out["plan_names"] == ["甲"]
