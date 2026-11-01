"""HTTP 端到端：真实 ASGI 路由层 + 打桩的 Agent / 检索 / 向量库，不调 API。

为什么需要这一层：`/chat` 与 `/chat/stream` 的路径、请求体校验、响应序列化、SSE
帧格式与顺序，都是函数级测试覆盖不到的；2026-10-31 的 `/deliverable` 冒烟已经
证明过这一点（`/{run_id}/status` 不存在、status 的 plan 只回名字）。

用 `httpx.ASGITransport` 而不是 `TestClient`：不跑 lifespan，避免预热真实
LanceDB/BM25 和连 MySQL；路由、中间件、序列化仍然是真的。
"""
from __future__ import annotations

import json

import app.main as main
import httpx
from app.retrieval.hybrid import RetrievalHit
from app.store.types import RetrievedChunk


class _FakeState:
    run_id = "run-http-1"
    game = "sr"
    answer = "三月七的突破材料是……"
    steps: list = []
    tool_failures = 0
    stop_reason = "answer"
    elapsed_ms = 12
    prompt_tokens = 100
    completion_tokens = 20
    refs: list = [{"url": "https://wiki.invalid/三月七", "title": "三月七"}]
    citation_audit = None

    def guard_stats(self) -> dict:
        return {}

    def injection_summary(self) -> dict:
        return {}


class _FakeAgent:
    backend = "deepseek"
    last_state = _FakeState()

    async def run(self, question, **kw):  # noqa: ANN001
        if kw.get("stream"):
            async def _gen():
                yield {"event": "token", "data": {"text": "三月七"}}
                yield {"event": "token", "data": {"text": "的突破材料是……"}}
                yield {"event": "done", "data": {"answer": _FakeState.answer,
                                                 "refs": _FakeState.refs}}

            return _gen()
        return _FakeState()


class _FakeRetriever:
    async def retrieve(self, query, cfg):  # noqa: ANN001
        return [
            RetrievalHit(
                chunk=RetrievedChunk(
                    chunk_id="chunk-1", text="三月七 突破材料 1200", score=0.9,
                    metadata={"section": "突破", "last_revision_ts": "2025-08-09",
                              "page_title": "三月七",
                              "source_url": "https://wiki.invalid/三月七"},
                ),
                score=0.9, rank_source="bm25",
            )
        ]


class _FakeLLM:
    def available_backends(self):
        return [{"id": "deepseek", "model": "deepseek-chat", "configured": True}]

    def fallback_stats(self):
        return {"chain": [], "total": 0, "by_pair": [], "recent": []}

    def cache_stats(self):
        return {"embed": {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0},
                "rerank": {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}}


class _FakeVectorStore:
    def list_collections(self):
        return [{"name": "kb_sr_baai-bge-m3_base", "count": 1}]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                             base_url="http://test")


def _frames(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        lines = frame.split("\n")
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if ev and data:
            out.append((ev, json.loads(data)))
    return out


async def test_health_返回前端依赖的快照(monkeypatch):
    monkeypatch.setattr(main, "get_vector_store", lambda _s: _FakeVectorStore())
    monkeypatch.setattr(main, "get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(main, "db_status", lambda: {"state": "ready", "detail": ""})

    async with _client() as c:
        r = await c.get("/api/health")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["ready"] is True
    assert set(body["games"]) == {"sr", "ys", "zzz"}
    assert body["tools"] and "search_kb" in body["tools"]
    assert body["collections"][0]["name"].startswith("kb_sr")


async def test_retrieve_返回带来源的命中(monkeypatch):
    monkeypatch.setattr(main, "get_retriever", lambda _s: _FakeRetriever())

    async with _client() as c:
        r = await c.post("/api/retrieve", json={"query": "三月七 突破材料", "game": "sr"})

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "三月七 突破材料"
    hit = body["hits"][0]
    assert hit["rank"] == 1
    assert hit["source"] == "bm25"
    assert hit["page_title"] and hit["url"]


async def test_chat_非流式返回结构化字段(monkeypatch):
    monkeypatch.setattr(main, "_build_agent", lambda req: (_FakeAgent(), None))
    monkeypatch.setattr("app.store.persist.save_run", _async_true)

    async with _client() as c:
        r = await c.post("/api/chat", json={"question": "三月七突破材料", "game": "sr"})

    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "run-http-1"
    assert body["answer"]
    assert body["trace"] == []
    assert body["usage"] == {"prompt": 100, "completion": 20}


async def test_chat_stream_帧格式与顺序(monkeypatch):
    monkeypatch.setattr(main, "_build_agent", lambda req: (_FakeAgent(), None))
    monkeypatch.setattr("app.store.persist.save_run", _async_true)

    async with _client() as c:
        async with c.stream("POST", "/api/chat/stream",
                            json={"question": "三月七突破材料", "game": "sr"}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            text = "".join([chunk async for chunk in r.aiter_text()])

    frames = _frames(text)
    assert [e for e, _ in frames] == ["token", "token", "done"]
    assert frames[0][1]["text"] == "三月七"
    assert frames[-1][1]["persisted"] is True
    assert frames[-1][1]["backend"] == "deepseek"


async def test_请求体缺字段返回422():
    async with _client() as c:
        r = await c.post("/api/chat", json={"game": "sr"})
    assert r.status_code == 422


async def _async_true(*a, **k):  # noqa: ANN001
    return True
