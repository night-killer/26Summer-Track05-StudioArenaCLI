"""Runtime configuration for Studio Arena automation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(project_root: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return project_root / path


@dataclass(slots=True)
class ArenaRuntimeConfig:
    """Runtime configuration shared across CLI, services, and Synergy bridge."""

    project_root: Path
    work_dir: Path
    state_dir: Path
    db_path: Path
    initial_wallet: int = 20_000
    free_token_budget: int = 1_000_000
    overage_cost_per_million: int = 2
    operating_cost_per_day: int = 500
    wallet_floor_hard: int = 1_500
    wallet_floor_soft: int = 4_000
    token_tight_threshold: int = 400_000
    token_emergency_threshold: int = 150_000

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "ArenaRuntimeConfig":
        resolved_root = Path(project_root or Path(__file__).resolve().parents[2])
        work_dir = _resolve_path(
            resolved_root,
            os.environ.get("STUDIO_ARENA_WORK_DIR", "arena"),
        )
        state_dir = _resolve_path(
            resolved_root,
            os.environ.get("STUDIO_ARENA_STATE_DIR", ".studio_arena"),
        )
        db_path_raw = os.environ.get("STUDIO_ARENA_DB_PATH", "").strip()
        db_path = (
            _resolve_path(resolved_root, db_path_raw)
            if db_path_raw
            else state_dir / "state.db"
        )
        return cls(
            project_root=resolved_root,
            work_dir=work_dir,
            state_dir=state_dir,
            db_path=db_path,
        )

    def ensure_directories(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
