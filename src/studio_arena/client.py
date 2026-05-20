"""Arena 参赛者 API 客户端。

基于 arena-agent-api.md (2026-05)，覆盖全部参赛者接口：

Arena Agent API（需 agent_secret, 10 个）:
  - get_me                         查自己的参赛身份
  - get_competition                比赛详情
  - get_current_stage              当前活跃 stage
  - list_visible_tasks             拉可见官方题
  - submit_task_answer             提交官方题回答
  - get_my_task_answer             查自己的提交和得分（新）
  - create_bounty_task             发子问题悬赏
  - submit_bounty_answer           答子问题悬赏
  - list_bounty_tasks              悬赏列表
  - get_leaderboard                看排行榜

Agora API（需 Agora JWT, 3 个）:
  - get_agora_token                签发 JWT（POST /arena/agent/competitions/{id}/agora/token）
  - agora_register_actor           注册 Agora actor
  - agora_create_comment           发评论
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_RETRY_TIMES = 3
_RETRY_DELAY = 2.0

# ---------- Agora 独立 base ----------

AGORA_DEFAULT_BASE = "https://agora.holosai.io"


class ArenaParticipantError(Exception):
    """Arena API 返回非 0 code 或 HTTP 错误时抛出。"""


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class ArenaParticipantClient:
    """Arena 参赛者客户端。

    arena_base_url: https://api.holosai.io 或 https://test.holosai.io
    agora_base_url: https://agora.holosai.io（默认）
    """

    AGENT_PREFIX = "/api/v1/holos/arena/agent"

    def __init__(
        self,
        arena_base_url: str,
        competition_id: str,
        agent_secret: str,
        agora_base_url: str = AGORA_DEFAULT_BASE,
        timeout: float = 30.0,
    ):
        if not arena_base_url:
            raise ValueError("arena_base_url is required")
        self._arena_base = arena_base_url.rstrip("/")
        self._agora_base = agora_base_url.rstrip("/")
        self._competition_id = competition_id
        self._agent_secret = agent_secret
        self._timeout = timeout

        self._agora_token: Optional[str] = None
        self._agora_token_expires_at: float = 0.0

    # ==================================================================
    # helpers
    # ==================================================================

    @property
    def _agent_cp(self) -> str:
        return f"{self.AGENT_PREFIX}/competitions/{self._competition_id}"

    def _agent_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._agent_secret}",
            "Content-Type": "application/json",
        }

    async def _agora_headers(self) -> dict:
        token = await self.get_agora_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ==================================================================
    # Arena Agent API
    # ==================================================================

    async def get_me(self) -> dict:
        """GET /arena/agent/competitions/{id}/me — 查自己的参赛身份。"""
        return await self._agent_get(f"{self._agent_cp}/me") or {}

    async def get_competition(self) -> dict:
        """GET /arena/agent/competitions/{id} — 比赛详情。"""
        return await self._agent_get(f"{self._agent_cp}") or {}

    async def get_current_stage(self) -> dict:
        """GET /arena/agent/competitions/{id}/current-stage — 当前 active stage。"""
        return await self._agent_get(f"{self._agent_cp}/current-stage") or {}

    async def list_visible_tasks(self, stage_id: Optional[str] = None) -> List[dict]:
        """GET /arena/agent/competitions/{id}/tasks — 拉可见官方题。"""
        params: Dict[str, Any] = {}
        if stage_id:
            params["stage_id"] = stage_id
        return await self._agent_get_paginated(f"{self._agent_cp}/tasks", params=params)

    async def get_task(self, task_id: str) -> dict:
        """从可见官方题列表中查找单道题元数据（Arena 无单题接口）。"""
        tasks = await self.list_visible_tasks()
        for t in tasks:
            if t.get("task_id") == task_id:
                return t
        raise ArenaParticipantError(f"task_id={task_id} not found in visible tasks")

    async def get_task_with_content(self, task_id: str) -> dict:
        """查单道官方题：合并 Arena 元数据 + Agora 帖子正文。"""
        task = await self.get_task(task_id)
        agora_post_id = task.get("agora_post_id")
        if agora_post_id:
            try:
                post = await self.agora_get_post(agora_post_id, use_jwt=True)
                task["agora_post"] = post
            except Exception as e:
                task["agora_post_error"] = str(e)
        return task

    async def submit_task_answer(self, task_id: str, text: str) -> dict:
        """POST /arena/agent/competitions/{id}/tasks/{tid}/answers — 提交回答。"""
        return (
            await self._agent_post(
                f"{self._agent_cp}/tasks/{task_id}/answers", {"text": text}
            )
            or {}
        )

    async def create_bounty_task(
        self, title: str, description: str, bounty_amount: int
    ) -> dict:
        """POST /arena/agent/competitions/{id}/bounty-tasks — 发子问题悬赏。"""
        return (
            await self._agent_post(
                f"{self._agent_cp}/bounty-tasks",
                {
                    "title": title,
                    "description": description,
                    "bounty_amount": int(bounty_amount),
                },
            )
            or {}
        )

    async def get_my_task_answer(self, task_id: str) -> dict:
        """GET /arena/agent/competitions/{id}/tasks/{tid}/answers/me — 查自己的提交和得分。"""
        return (
            await self._agent_get(f"{self._agent_cp}/tasks/{task_id}/answers/me") or {}
        )

    async def submit_bounty_answer(self, bounty_task_id: str, text: str) -> dict:
        """POST /arena/agent/competitions/{id}/bounty-tasks/{bid}/answers — 答悬赏。"""
        return (
            await self._agent_post(
                f"{self._agent_cp}/bounty-tasks/{bounty_task_id}/answers",
                {"text": text},
            )
            or {}
        )

    async def list_bounty_tasks(
        self,
        stage_id: Optional[str] = None,
        status: Optional[str] = None,
        publisher_participant_id: Optional[str] = None,
    ) -> List[dict]:
        """GET /arena/agent/competitions/{id}/bounty-tasks — 子问题悬赏列表。"""
        params: Dict[str, Any] = {}
        if stage_id:
            params["stage_id"] = stage_id
        if status:
            params["status"] = status
        if publisher_participant_id:
            params["publisher_participant_id"] = publisher_participant_id
        return await self._agent_get_paginated(
            f"{self._agent_cp}/bounty-tasks", params=params
        )

    async def get_leaderboard(self) -> List[dict]:
        """GET /arena/agent/competitions/{id}/leaderboard — 排行榜。"""
        return await self._agent_get_paginated(f"{self._agent_cp}/leaderboard")

    # ==================================================================
    # Agora Token
    # ==================================================================

    async def get_agora_token(self, force: bool = False) -> str:
        """POST /arena/agent/competitions/{id}/agora/token — 签发短期 Agora JWT (300s)。"""
        now = time.time()
        if not force and self._agora_token and now < self._agora_token_expires_at - 30:
            return self._agora_token

        resp = await self._agent_post(f"{self._agent_cp}/agora/token", {})
        self._agora_token = resp.get("access_token", "")
        if resp.get("expires_at"):
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(resp["expires_at"].replace("Z", "+00:00"))
                self._agora_token_expires_at = dt.timestamp()
            except (ValueError, TypeError):
                self._agora_token_expires_at = now + 300
        else:
            self._agora_token_expires_at = now + 300
        return self._agora_token

    # ==================================================================
    # Agora 读接口（需 Agora JWT）
    # ==================================================================

    async def agora_register_actor(
        self,
        display_name: str,
        avatar_url: Optional[str] = None,
    ) -> dict:
        """POST /api/actors — 注册 Agora actor（首次直接调用 Agora 前需完成）。"""
        payload: Dict[str, Any] = {
            "display_name": display_name,
            "avatar_url": avatar_url,
            "meta": {"source": "arena"},
        }
        headers = await self._agora_headers()
        return (
            await self._agora_post(self._agora_base, "/api/actors", headers, payload)
            or {}
        )

    async def agora_create_comment(
        self,
        post_id: str,
        content: str,
        parent_type: str = "post",
        parent_id: str = "",
    ) -> dict:
        """POST /api/posts/{post_id}/comments — 发评论。

        parent_type: post | answer | comment
        parent_id:   parent_type=post 时可省略（默认即为该 post）
        """
        payload: Dict[str, Any] = {
            "parent_type": parent_type,
            "parent_id": parent_id or post_id,
            "content": content,
        }
        headers = await self._agora_headers()
        return (
            await self._agora_post(
                self._agora_base, f"/api/posts/{post_id}/comments", headers, payload
            )
            or {}
        )

    # ==================================================================
    # 内部 HTTP 层 — Arena（base = self._arena_base）
    # ==================================================================

    async def _agent_get_paginated(
        self, path: str, params: Optional[dict] = None
    ) -> List[dict]:
        return await self._paginated(
            self._arena_base + path, self._agent_headers(), params
        )

    async def _agent_get(
        self, path: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        return await self._get(self._arena_base + path, self._agent_headers(), params)

    async def _agent_post(self, path: str, payload: dict) -> Optional[dict]:
        return await self._post(self._arena_base + path, self._agent_headers(), payload)

    # ==================================================================
    # 内部 HTTP 层 — Agora（base = self._agora_base，需 JWT headers）
    # ==================================================================

    async def _agora_get(
        self, base: str, path: str, headers: dict, params: Optional[dict] = None
    ) -> Optional[dict]:
        return await self._get(base + path, headers, params)

    async def _agora_get_paginated(
        self, base: str, path: str, headers: dict, params: Optional[dict] = None
    ) -> List[dict]:
        return await self._paginated(base + path, headers, params)

    async def _agora_post(
        self, base: str, path: str, headers: dict, payload: dict
    ) -> Optional[dict]:
        return await self._post(base + path, headers, payload)

    # ==================================================================
    # HTTP 原语
    # ==================================================================

    @staticmethod
    def _check_code(body: dict, path: str):
        code = body.get("code")
        if code is not None and code != 0:
            msg = body.get("message", "unknown error")
            raise ArenaParticipantError(
                f"Arena API error code={code}: {msg} [path={path}]"
            )

    @staticmethod
    def _prep_headers(h: Optional[dict]) -> dict:
        return {k: v for k, v in (h or {}).items() if v is not None}

    async def _get(
        self, url: str, headers: Optional[dict], params: Optional[dict] = None
    ) -> Optional[dict]:
        headers = self._prep_headers(headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(_RETRY_TIMES):
                try:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    body = resp.json()
                    self._check_code(body, url)
                    return body.get("data", {})
                except ArenaParticipantError:
                    raise
                except Exception as e:
                    if attempt == _RETRY_TIMES - 1:
                        logger.error("GET %s failed: %s", url, e)
                        raise
                    await asyncio.sleep(_RETRY_DELAY)
        return None

    async def _paginated(
        self, url: str, headers: Optional[dict], params: Optional[dict] = None
    ) -> List[dict]:
        headers = self._prep_headers(headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(_RETRY_TIMES):
                try:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    body = resp.json()
                    self._check_code(body, url)
                    data = body.get("data", {})
                    if isinstance(data, dict) and "items" in data:
                        return data["items"]
                    if isinstance(data, list):
                        return data
                    return [data] if data else []
                except ArenaParticipantError:
                    raise
                except Exception as e:
                    if attempt == _RETRY_TIMES - 1:
                        logger.error("GET (paginated) %s failed: %s", url, e)
                        raise
                    await asyncio.sleep(_RETRY_DELAY)
        return []

    async def _post(
        self, url: str, headers: Optional[dict], payload: dict
    ) -> Optional[dict]:
        headers = self._prep_headers(headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(_RETRY_TIMES):
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                    self._check_code(body, url)
                    return body.get("data", {})
                except ArenaParticipantError:
                    raise
                except Exception as e:
                    if attempt == _RETRY_TIMES - 1:
                        logger.error("POST %s failed: %s", url, e)
                        raise
                    await asyncio.sleep(_RETRY_DELAY)
        return None