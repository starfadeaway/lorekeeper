"""交付物端点的 HTTP 层验收：真实 ASGI 栈 + 打桩 Agent/存储，不调 API。

为什么用 `httpx.ASGITransport` 而不是 `TestClient`：
`TestClient(app)` 会跑 lifespan，而 lifespan 会预热真实检索（LanceDB + BM25）并连
MySQL —— 单测里既慢又依赖本机数据。ASGITransport 同样走完整 ASGI 路由/参数解析/
序列化/SSE，只是不跑 lifespan，正合本文件要验的那一层。
"""
from __future__ import annotations

import json

import app.main as main
import httpx
import pytest
from app.agent.deliverable import DeliverableResult, Fact, SubtaskResult
from app.agent.deliverable_store import Checkpoint

RUN = "run00000000000021"
PLAN = [
    {"name": "甲", "goal": "g甲", "queries": ["q甲"]},
    {"name": "乙", "goal": "g乙", "queries": ["q乙"]},
]


def _result(*, plan=None, subtasks=None) -> DeliverableResult:
    st = SubtaskResult(
        name="甲", goal="g甲", status="ok", queries=["q甲"], latency_ms=10,
        facts=[Fact(key="hp", label="生命值", value="1200",
                    source_url="https://wiki.invalid/三月七", page_title="三月七",
                    revision="2025-03-01", subtask="甲")],
    )
    return DeliverableResult(
        run_id=RUN, topic="三月七养成", question="整理三月七养成", game="sr",
        plan=plan if plan is not None else PLAN,
        subtasks=[st] if subtasks is None else subtasks,
        generated_at="2026-10-31 22:00:00", subject="三月七", elapsed_ms=12,
        meta={"subtasks_total": 1, "subtasks_ok": 1, "known_limits": "x"},
    )


class FakeAgent:
    def __init__(self, **kw):
        pass

    async def plan_draft(self, question, game=None):
        return _result(plan=PLAN, subtasks=[])

    async def approve(self, run_id, plan=None, *, on_event=None):
        if run_id == "missing01":
            raise KeyError(run_id)
        await _events(on_event)
        return _result(subtasks=[SubtaskResult(name="甲", goal="g甲")])

    async def run(self, question, game=None, run_id=None, resume=False, on_event=None):
        if run_id == "missing01":
            raise KeyError(run_id)
        await _events(on_event)
        return _result()


async def _events(on_event):
    if on_event is None:
        return
    await on_event({"event": "plan", "data": {"topic": "t", "subject": "s",
                                              "plan": PLAN, "reused": []}})
    await on_event({"event": "subtask_start", "data": {"name": "甲", "index": 1, "total": 1}})
    await on_event({"event": "subtask_end", "data": {"name": "甲", "status": "ok",
                                                     "facts": 1, "ms": 10}})


class StubStore:
    def __init__(self, ckpt):
        self.ckpt = ckpt

    async def load(self, run_id):
        return self.ckpt


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                             base_url="http://test")


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(main, "DeliverableAgent", FakeAgent)


def _checkpoint() -> Checkpoint:
    return Checkpoint(
        run_id=RUN, question="整理三月七养成", game="sr",
        topic="三月七养成", subject="三月七", plan=PLAN, subtasks=[],
        phase="awaiting_approval",
    )


def _frames(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        lines = frame.split("\n")
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if ev and data:
            out.append((ev, json.loads(data)))
    return out


async def test_plan_端点只规划不执行(fake_agent):
    async with _client() as c:
        r = await c.post("/api/deliverable/plan", json={"question": "整理三月七养成", "game": "sr"})
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == RUN and len(body["plan"]) == 2
    assert "result" not in body  # 没有执行结果


async def test_approve_端点返回完整结果(fake_agent):
    async with _client() as c:
        r = await c.post(f"/api/deliverable/{RUN}/approve", json={"plan": PLAN})
    assert r.status_code == 200
    body = r.json()
    assert body["json"]["run_id"] == RUN
    assert body["status_counts"]["subtasks_ok"] == 1
    assert "markdown" in body and "result" not in body


async def test_resume_端点转发run_id(fake_agent):
    async with _client() as c:
        r = await c.post(f"/api/deliverable/{RUN}/resume")
    assert r.status_code == 200
    assert r.json()["json"]["run_id"] == RUN


async def test_status_端点返回完整计划(monkeypatch):
    monkeypatch.setattr(main, "FileSnapshotStore", lambda _root: StubStore(_checkpoint()))
    async with _client() as c:
        r = await c.get(f"/api/deliverable/{RUN}")
    assert r.status_code == 200
    body = r.json()
    assert body["phase"] == "awaiting_approval"
    assert body["plan"][0]["queries"] == ["q甲"]


async def test_stream_端点事件顺序正确(fake_agent):
    async with _client() as c:
        async with c.stream("POST", "/api/deliverable/stream",
                            json={"question": "整理三月七养成", "game": "sr"}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            text = "".join([chunk async for chunk in r.aiter_text()])
    frames = _frames(text)
    assert [e for e, _ in frames] == ["plan", "subtask_start", "subtask_end", "result"]
    assert frames[-1][1]["json"]["run_id"] == RUN


async def test_approve_stream_端点事件顺序正确(fake_agent):
    async with _client() as c:
        async with c.stream("POST", f"/api/deliverable/{RUN}/approve/stream",
                            json={"plan": PLAN}) as r:
            text = "".join([chunk async for chunk in r.aiter_text()])
    frames = _frames(text)
    assert [e for e, _ in frames] == ["plan", "subtask_start", "subtask_end", "result"]


async def test_未知run返回404(fake_agent):
    async with _client() as c:
        r = await c.post("/api/deliverable/missing01/approve", json={})
    assert r.status_code == 404


async def test_非法run_id返回400(monkeypatch):
    async with _client() as c:
        r = await c.get("/api/deliverable/bad")
    assert r.status_code == 400
