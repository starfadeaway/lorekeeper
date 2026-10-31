"""交付物任务的断点快照存储（v1：文件，不依赖 MySQL）。

## 为什么先落成文件而不是建表

`docs/adr_001_orchestration.md` §12 的定价表当时写的是"models.py 新表 + persist.py 存取"。
真做的时候选了文件，理由有两条：

1. **MySQL 已经不是可靠的强依赖**：2026-10-30 排查一次工具失败时，`mysql80` 服务
   停着，`save_run` 静默返回 False。把"断点续跑"建在同一条链路上，等于把恢复能力
   也绑在它上面；
2. **单机项目**：快照只有本机一个读者/写者，文件 + 原子替换足够，不需要事务与索引。

代价写在明处：**多实例/多机部署时这个实现不成立**。那时候要换 MySQL/Redis 实现，
`SnapshotStore` 这个 Protocol 就是给那次替换留的接缝。

## 一致性

写入走"同目录临时文件 + `os.replace`"，读者要么看到旧版本、要么看到新版本，
不会读到半个 JSON。文件名来自 `safe_run_id` 校验过的 run_id，避免路径穿越。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

#: run_id 来自 HTTP 路径，必须限制字符集 —— 否则 `../../x` 能把快照写出目录。
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def safe_run_id(run_id: str) -> str:
    """校验 run_id 可作为文件名；不合法直接抛，而不是悄悄清洗。"""
    if not _SAFE_RUN_ID.match(str(run_id or "")):
        raise ValueError("run_id 只允许 8–64 位字母 / 数字 / 下划线 / 连字符")
    return run_id


@dataclass
class Checkpoint:
    """一次交付物运行的断点。

    `subtasks` 存的是**已完成子任务的原始 dict**（`SubtaskResult` 的 asdict），
    不在这里 import `deliverable` 的类型 —— 那会形成循环依赖，也会让存储层
    被迫跟着业务 dataclass 一起改字段。
    """

    run_id: str
    question: str
    game: str
    topic: str
    subject: str
    plan: list[dict]
    subtasks: list[dict]
    phase: str                      # planned / running / done
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        return cls(
            run_id=str(data["run_id"]),
            question=str(data.get("question", "")),
            game=str(data.get("game", "")),
            topic=str(data.get("topic", "")),
            subject=str(data.get("subject", "")),
            plan=list(data.get("plan") or []),
            subtasks=list(data.get("subtasks") or []),
            phase=str(data.get("phase", "running")),
            updated_at=str(data.get("updated_at", "")),
        )

    def completed_names(self) -> set[str]:
        return {str(st.get("name", "")) for st in self.subtasks}


class SnapshotStore(Protocol):
    """断点存储的接缝。换 MySQL / Redis 时只实现这两个方法。"""

    async def save(self, ckpt: Checkpoint) -> None: ...

    async def load(self, run_id: str) -> Checkpoint | None: ...


class FileSnapshotStore:
    """`<data_root>/runs/deliverable/<run_id>.json`，原子写入。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_of(self, run_id: str) -> Path:
        return self.root / f"{safe_run_id(run_id)}.json"

    async def save(self, ckpt: Checkpoint) -> None:
        ckpt = Checkpoint(
            **{**ckpt.to_dict(), "updated_at": datetime.now().isoformat(timespec="seconds")}
        )

        def _write() -> None:
            self.root.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(ckpt.to_dict(), f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path_of(ckpt.run_id))
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

        await asyncio.to_thread(_write)

    async def load(self, run_id: str) -> Checkpoint | None:
        p = self.path_of(run_id)
        if not p.exists():
            return None

        def _read() -> dict:
            return json.loads(p.read_text(encoding="utf-8"))

        try:
            data = await asyncio.to_thread(_read)
        except (OSError, json.JSONDecodeError) as exc:
            # 快照损坏不能把主流程拖死：退化成"没有断点"，本次重跑。
            logger.warning("交付物快照损坏，忽略：%s (%s)", p, exc)
            return None
        return Checkpoint.from_dict(data)


class InMemorySnapshotStore:
    """测试替身；也用于不想落盘的脚本。"""

    def __init__(self) -> None:
        self.saved: dict[str, Checkpoint] = {}
        self.save_calls = 0

    async def save(self, ckpt: Checkpoint) -> None:
        self.save_calls += 1
        self.saved[ckpt.run_id] = Checkpoint.from_dict(ckpt.to_dict())

    async def load(self, run_id: str) -> Checkpoint | None:
        return self.saved.get(run_id)
