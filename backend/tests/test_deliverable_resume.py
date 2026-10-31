"""断点续跑（v1）的守卫：快照可逆、已完成不重跑、未知 run_id 报错。

这些用例全部离线：FakeLLM + FakeRetriever + InMemorySnapshotStore。
真实文件存储的原子写另有一条用例（tmp_path），不碰仓库的 data/。
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from app.agent.deliverable import DeliverableAgent, Fact, SubtaskResult
from app.agent.deliverable_store import (
    Checkpoint,
    FileSnapshotStore,
    InMemorySnapshotStore,
    safe_run_id,
)
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


def plan(*names: str) -> list[dict]:
    return [{"name": n, "goal": f"g-{n}", "queries": [f"q-{n}"]} for n in names]


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


def checkpoint(run_id: str, names: list[str], completed: list[SubtaskResult], phase: str) -> Checkpoint:
    return Checkpoint(
        run_id=run_id, question="整理三月七的养成指南", game="sr",
        topic="三月七养成", subject="三月七", plan=plan(*names),
        subtasks=[asdict(st) for st in completed], phase=phase,
    )


# --------------------------------------------------------------------------- #
async def test_一次运行会按子任务落快照(settings):
    store = InMemorySnapshotStore()
    retriever = FakeRetriever({"q-甲": [hit("三月七", "生命值 1200")]})
    agent = make_agent(
        settings,
        [{"answer": plan_json(["甲"])},
         {"answer": facts_json("hp", "1200")}],
        retriever, store,
    )

    res = await agent.run("整理三月七的养成指南")

    ckpt = store.saved[res.run_id]
    assert ckpt.phase == "done"
    assert [st["name"] for st in ckpt.subtasks] == ["甲"]
    assert ckpt.subtasks[0]["facts"][0]["source_url"] == "https://wiki.invalid/三月七"
    # planned + running + done 三次写入
    assert store.save_calls >= 3


async def test_续跑只跑未完成的子任务(settings):
    store = InMemorySnapshotStore()
    store.saved["run00000000000001"] = checkpoint(
        "run00000000000001", ["甲", "乙"],
        [done_subtask("甲", "hp", "1200")], phase="running",
    )
    # 续跑时**不该**再调用规划；脚本里只有"乙"的抽取
    retriever = FakeRetriever({"q-乙": [hit("三月七", "攻击力 500")]})
    agent = make_agent(
        settings, [{"answer": facts_json("atk", "500")}], retriever, store,
    )

    res = await agent.run("", run_id="run00000000000001", resume=True)

    assert [st.name for st in res.subtasks] == ["甲", "乙"]
    assert [len(st.facts) for st in res.subtasks] == [1, 1]
    assert retriever.queries == ["q-乙"], "已完成的子任务不该再检索"
    assert agent.llm.calls == 1, "续跑只该花一次抽取调用（不重新规划）"
    assert store.saved["run00000000000001"].phase == "done"


async def test_已完成运行续跑不调用模型(settings):
    store = InMemorySnapshotStore()
    store.saved["run00000000000002"] = checkpoint(
        "run00000000000002", ["甲"],
        [done_subtask("甲", "hp", "1200")], phase="done",
    )

    class _Boom:
        calls = 0

        async def retrieve(self, *a, **kw):
            self.calls += 1
            raise AssertionError("已完成的任务不该再检索")

    retriever = _Boom()
    agent = make_agent(settings, [{"error": "不该调用模型"}], retriever, store)

    res = await agent.run("", run_id="run00000000000002", resume=True)

    assert len(res.subtasks[0].facts) == 1
    assert agent.llm.calls == 0
    assert retriever.calls == 0


async def test_未知run_id抛KeyError(settings):
    agent = make_agent(settings, [], FakeRetriever(), InMemorySnapshotStore())
    with pytest.raises(KeyError):
        await agent.run("", run_id="run00000000000003", resume=True)


async def test_文件快照可写可读(tmp_path, settings):
    store = FileSnapshotStore(tmp_path / "deliverable")
    ckpt = checkpoint("run00000000000004", ["甲"], [done_subtask("甲", "hp", "1200")], "running")
    await store.save(ckpt)

    back = await store.load("run00000000000004")
    assert back is not None
    assert back.phase == "running"
    assert back.subtasks[0]["facts"][0]["value"] == "1200"
    assert back.updated_at, "落盘时应带上时间戳"


def test_run_id非法字符被拒(tmp_path):
    store = FileSnapshotStore(tmp_path / "d")
    with pytest.raises(ValueError):
        store.path_of("../../etc/passwd")
    with pytest.raises(ValueError):
        safe_run_id("a" * 5)
