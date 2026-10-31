"""可观测事件的顺序契约：plan → subtask_start → subtask_end → …，且失败不打断执行。"""
from __future__ import annotations

import json

from app.agent.deliverable import DeliverableAgent
from app.agent.deliverable_store import InMemorySnapshotStore
from app.agent.tools import ToolContext
from app.retrieval.hybrid import RetrievalHit
from app.store.types import RetrievedChunk
from tests.fakes import FakeLLM

REV = "2025-03-01T00:00:00Z"


def hit(text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk=RetrievedChunk(
            chunk_id="sr-1-x", text=text, score=0.9,
            metadata={"game": "sr", "page_id": 1, "page_title": "三月七",
                      "section": "", "chunk_index": 0, "last_revision": 1,
                      "last_revision_ts": REV,
                      "source_url": "https://wiki.invalid/三月七", "truncated": 0},
        ),
        score=0.9, rank_source="bm25",
    )


class FakeRetriever:
    async def retrieve(self, query, config):  # noqa: ANN001
        return [hit(f"{query} 1200")]


def make_agent(settings, script):
    ctx = ToolContext(settings=settings, retriever=FakeRetriever(), store=None,
                      game="sr", disclose_revision=True)
    return DeliverableAgent(settings=settings, llm=FakeLLM(script), ctx=ctx,
                            top_k=5, store=InMemorySnapshotStore())


def plan_json(names: list[str]) -> str:
    return json.dumps(
        {"topic": "三月七养成", "subject": "三月七",
         "subtasks": [{"name": n, "goal": f"g-{n}", "queries": [f"q-{n}"]} for n in names]},
        ensure_ascii=False,
    )


def facts_json() -> str:
    return json.dumps({"facts": [{"i": 1, "key": "hp", "label": "生命", "value": "1200"}]},
                      ensure_ascii=False)


async def test_run_事件顺序与字段(settings):
    events = []

    async def on_event(ev):
        events.append(ev)

    agent = make_agent(settings, [{"answer": plan_json(["甲", "乙"])},
                                  {"answer": facts_json()}, {"answer": facts_json()}])
    await agent.run("整理三月七的养成指南", on_event=on_event)

    assert [e["event"] for e in events] == [
        "plan", "subtask_start", "subtask_end", "subtask_start", "subtask_end",
    ]
    assert [p["name"] for p in events[0]["data"]["plan"]] == ["甲", "乙"]
    assert events[1]["data"] == {"name": "甲", "index": 1, "total": 2}
    assert events[2]["data"]["name"] == "甲"
    assert events[2]["data"]["status"] == "ok"
    assert events[2]["data"]["facts"] == 1
    assert events[4]["data"]["name"] == "乙"


async def test_approve_也发计划事件(settings):
    events = []

    async def on_event(ev):
        events.append(ev)

    agent = make_agent(settings, [{"answer": plan_json(["甲"])}, {"answer": facts_json()}])
    draft = await agent.plan_draft("整理三月七的养成指南")
    events.clear()
    await agent.approve(draft.run_id, on_event=on_event)

    assert [e["event"] for e in events] == ["plan", "subtask_start", "subtask_end"]
    assert events[0]["data"]["reused"] == []


async def test_回调抛错不影响执行(settings):
    calls = []

    async def broken(ev):
        calls.append(ev["event"])
        raise RuntimeError("SSE 客户端断线")

    agent = make_agent(settings, [{"answer": plan_json(["甲"])}, {"answer": facts_json()}])
    res = await agent.run("整理三月七的养成指南", on_event=broken)

    assert res.subtasks and res.subtasks[0].status == "ok"
    assert calls == ["plan", "subtask_start", "subtask_end"]
