"""Factory helpers for CLI and tests."""

from __future__ import annotations

import os
from pathlib import Path

from .client import ArenaParticipantClient
from .config import ArenaRuntimeConfig
from .runtime import ArenaAutomationRuntime
from .store import ArenaStateStore


def create_runtime_config(project_root: Path | None = None) -> ArenaRuntimeConfig:
    config = ArenaRuntimeConfig.from_env(project_root=project_root)
    config.ensure_directories()
    return config


def create_client_from_env() -> ArenaParticipantClient:
    competition_id = os.environ.get("ARENA_COMPETITION_ID", "")
    agent_secret = os.environ.get("ARENA_AGENT_SECRET", "")
    if not competition_id:
        raise ValueError("ARENA_COMPETITION_ID is required")
    if not agent_secret:
        raise ValueError("ARENA_AGENT_SECRET is required")
    return ArenaParticipantClient(
        arena_base_url=os.environ.get("ARENA_BASE_URL", "https://api.holosai.io"),
        competition_id=competition_id,
        agent_secret=agent_secret,
        agora_base_url=os.environ.get("AGORA_BASE_URL", "https://agora.holosai.io"),
    )


def create_runtime(project_root: Path | None = None) -> ArenaAutomationRuntime:
    config = create_runtime_config(project_root=project_root)
    return ArenaAutomationRuntime(
        client=create_client_from_env(),
        config=config,
        store=ArenaStateStore(config.db_path),
    )
