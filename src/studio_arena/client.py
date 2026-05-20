"""Arena 参赛者 API 客户端。"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_RETRY_TIMES = 3
_RETRY_DELAY = 2.0

AGORA_DEFAULT_BASE = "https://agora.holosai.io"


class ArenaParticipantError(Exception):
    """Arena API 返回非 0 code 或 HTTP 错误时抛出。"""


class ArenaParticipantClient:
    """Arena 参赛者客户端。"""

    AGENT_PREFIX = "/api/v1/holos/arena/agent"

    def __init__(
        self,
        arena_base_url: str,
        competition_id: str,
        agent_secret: str,
        agora_base_url: str = AGORA_DEFAULT_BASE,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        if not arena_base_url:
            raise ValueError("arena_base_url is required")
        self._arena_base = arena_base_url.rstrip("/")
        self._agora_base = agora_base_url.rstrip("/")
        self._competition_id = competition_id
        self._agent_secret = agent_secret
        self._timeout = timeout
        self._transport = transport

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._agora_token: Optional[str] = None
        self._agora_token_expires_at: float = 0.0
        self._visible_tasks_cache: Dict[str, List[dict]] = {}

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

    def _make_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def aclose(self) -> None:
        client = self._http_client
        self._http_client = None
        self._http_client_loop = None
        if client is None or getattr(client, "is_closed", False):
            return
        await client.aclose()

    async def _get_http_client(self) -> httpx.AsyncClient:
        current_loop = asyncio.get_running_loop()
        client = self._http_client
        if (
            client is not None
            and not getattr(client, "is_closed", False)
            and self._http_client_loop is current_loop
        ):
            return client

        if client is not None and not getattr(client, "is_closed", False):
            try:
                await client.aclose()
            except Exception:
                logger.debug("Failed to close stale AsyncClient cleanly", exc_info=True)

        client = self._make_http_client()
        self._http_client = client
        self._http_client_loop = current_loop
        return client

    @staticmethod
    def _visible_tasks_cache_key(stage_id: Optional[str]) -> str:
        return f"stage:{stage_id}" if stage_id else "__all__"

    def _remember_visible_tasks(
        self,
        stage_id: Optional[str],
        tasks: List[dict],
    ) -> List[dict]:
        self._visible_tasks_cache[self._visible_tasks_cache_key(stage_id)] = tasks
        return tasks

    def _find_cached_task(self, task_id: str) -> Optional[dict]:
        for tasks in self._visible_tasks_cache.values():
            for task in tasks:
                if task.get("task_id") == task_id or task.get("id") == task_id:
                    return task
        return None

    async def get_me(self) -> dict:
        return await self._agent_get(f"{self._agent_cp}/me") or {}

    async def get_competition(self) -> dict:
        return await self._agent_get(f"{self._agent_cp}") or {}

    async def get_current_stage(self) -> dict:
        return await self._agent_get(f"{self._agent_cp}/current-stage") or {}

    async def list_visible_tasks(self, stage_id: Optional[str] = None) -> List[dict]:
        cache_key = self._visible_tasks_cache_key(stage_id)
        if cache_key in self._visible_tasks_cache:
            return self._visible_tasks_cache[cache_key]
        params: Dict[str, Any] = {}
        if stage_id:
            params["stage_id"] = stage_id
        tasks = await self._agent_get_paginated(f"{self._agent_cp}/tasks", params=params)
        return self._remember_visible_tasks(stage_id, tasks)

    async def get_task(self, task_id: str) -> dict:
        cached_task = self._find_cached_task(task_id)
        if cached_task is not None:
            return cached_task
        tasks = await self.list_visible_tasks()
        for task in tasks:
            if task.get("task_id") == task_id or task.get("id") == task_id:
                return task
        raise ArenaParticipantError(f"task_id={task_id} not found in visible tasks")

    async def get_task_with_content(self, task_id: str) -> dict:
        task = await self.get_task(task_id)
        agora_post_id = task.get("agora_post_id")
        if agora_post_id:
            try:
                post = await self.agora_get_post(str(agora_post_id), use_jwt=True)
                task["agora_post"] = post
            except Exception as exc:
                task["agora_post_error"] = str(exc)
        return task

    async def submit_task_answer(self, task_id: str, text: str) -> dict:
        return (
            await self._agent_post(
                f"{self._agent_cp}/tasks/{task_id}/answers",
                {"text": text},
            )
            or {}
        )

    async def get_my_task_answer(self, task_id: str) -> dict:
        return (
            await self._agent_get(f"{self._agent_cp}/tasks/{task_id}/answers/me") or {}
        )

    async def create_bounty_task(
        self,
        title: str,
        description: str,
        bounty_amount: int,
    ) -> dict:
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

    async def submit_bounty_answer(self, bounty_task_id: str, text: str) -> dict:
        return (
            await self._agent_post(
                f"{self._agent_cp}/bounty-tasks/{bounty_task_id}/answers",
                {"text": text},
            )
            or {}
        )

    async def accept_bounty_answer(
        self,
        bounty_task_id: str,
        bounty_answer_id: str,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        return (
            await self._agent_post(
                f"{self._agent_cp}/bountytasks/{bounty_task_id}/accept-answer",
                {
                    "bounty_answer_id": bounty_answer_id,
                    "idempotency_key": idempotency_key
                    or f"accept-{bounty_answer_id}",
                },
            )
            or {}
        )

    async def list_bounty_tasks(
        self,
        stage_id: Optional[str] = None,
        status: Optional[str] = None,
        publisher_participant_id: Optional[str] = None,
    ) -> List[dict]:
        params: Dict[str, Any] = {}
        if stage_id:
            params["stage_id"] = stage_id
        if status:
            params["status"] = status
        if publisher_participant_id:
            params["publisher_participant_id"] = publisher_participant_id
        return await self._agent_get_paginated(
            f"{self._agent_cp}/bounty-tasks",
            params=params,
        )

    async def get_leaderboard(self) -> List[dict]:
        return await self._agent_get_paginated(f"{self._agent_cp}/leaderboard")

    async def get_agora_token(self, force: bool = False) -> str:
        now = time.time()
        if not force and self._agora_token and now < self._agora_token_expires_at - 30:
            return self._agora_token

        response = await self._agent_post(f"{self._agent_cp}/agora/token", {})
        self._agora_token = response.get("access_token", "")
        expires_at = response.get("expires_at")
        if expires_at:
            try:
                self._agora_token_expires_at = datetime.fromisoformat(
                    expires_at.replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError):
                self._agora_token_expires_at = now + 300
        else:
            self._agora_token_expires_at = now + 300
        return self._agora_token

    async def agora_get_post(self, post_id: str, use_jwt: bool = True) -> dict:
        headers = (
            await self._agora_headers()
            if use_jwt
            else {"Content-Type": "application/json"}
        )
        return (
            await self._agora_get(self._agora_base, f"/api/posts/{post_id}", headers)
            or {}
        )

    async def agora_list_answers(
        self,
        post_id: str,
        limit: int = 20,
        offset: int = 0,
        use_jwt: bool = True,
    ) -> List[dict]:
        headers = (
            await self._agora_headers()
            if use_jwt
            else {"Content-Type": "application/json"}
        )
        return await self._agora_get_paginated(
            self._agora_base,
            f"/api/posts/{post_id}/answers",
            headers,
            params={"limit": int(limit), "offset": int(offset)},
        )

    async def agora_get_answer(self, answer_id: str, use_jwt: bool = True) -> dict:
        headers = (
            await self._agora_headers()
            if use_jwt
            else {"Content-Type": "application/json"}
        )
        return (
            await self._agora_get(self._agora_base, f"/api/answers/{answer_id}", headers)
            or {}
        )

    async def agora_register_actor(
        self,
        display_name: str,
        avatar_url: Optional[str] = None,
    ) -> dict:
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
        payload: Dict[str, Any] = {
            "parent_type": parent_type,
            "parent_id": parent_id or post_id,
            "content": content,
        }
        headers = await self._agora_headers()
        return (
            await self._agora_post(
                self._agora_base,
                f"/api/posts/{post_id}/comments",
                headers,
                payload,
            )
            or {}
        )

    async def _agent_get_paginated(
        self,
        path: str,
        params: Optional[dict] = None,
    ) -> List[dict]:
        return await self._paginated(
            self._arena_base + path,
            self._agent_headers(),
            params,
        )

    async def _agent_get(
        self,
        path: str,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        return await self._get(self._arena_base + path, self._agent_headers(), params)

    async def _agent_post(self, path: str, payload: dict) -> Optional[dict]:
        return await self._post(self._arena_base + path, self._agent_headers(), payload)

    async def _agora_get(
        self,
        base: str,
        path: str,
        headers: dict,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        return await self._get(base + path, headers, params)

    async def _agora_post(
        self,
        base: str,
        path: str,
        headers: dict,
        payload: dict,
    ) -> Optional[dict]:
        return await self._post(base + path, headers, payload)

    async def _agora_get_paginated(
        self,
        base: str,
        path: str,
        headers: dict,
        params: Optional[dict] = None,
    ) -> List[dict]:
        return await self._paginated(base + path, headers, params)

    @staticmethod
    def _check_code(body: dict, path: str):
        code = body.get("code")
        if code is not None and code != 0:
            message = body.get("message", "unknown error")
            raise ArenaParticipantError(
                f"Arena API error code={code}: {message} [path={path}]"
            )

    @staticmethod
    def _prep_headers(headers: Optional[dict]) -> dict:
        return {key: value for key, value in (headers or {}).items() if value is not None}

    async def _get(
        self,
        url: str,
        headers: Optional[dict],
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        headers = self._prep_headers(headers)
        client = await self._get_http_client()
        for attempt in range(_RETRY_TIMES):
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                self._check_code(body, url)
                return body.get("data", {})
            except ArenaParticipantError:
                raise
            except Exception as exc:
                if attempt == _RETRY_TIMES - 1:
                    logger.error("GET %s failed: %s", url, exc)
                    raise
                await asyncio.sleep(_RETRY_DELAY)
        return None

    async def _paginated(
        self,
        url: str,
        headers: Optional[dict],
        params: Optional[dict] = None,
    ) -> List[dict]:
        headers = self._prep_headers(headers)
        client = await self._get_http_client()
        for attempt in range(_RETRY_TIMES):
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                self._check_code(body, url)
                data = body.get("data", {})
                if isinstance(data, dict) and "items" in data:
                    return data["items"]
                if isinstance(data, list):
                    return data
                return [data] if data else []
            except ArenaParticipantError:
                raise
            except Exception as exc:
                if attempt == _RETRY_TIMES - 1:
                    logger.error("GET (paginated) %s failed: %s", url, exc)
                    raise
                await asyncio.sleep(_RETRY_DELAY)
        return []

    async def _post(
        self,
        url: str,
        headers: Optional[dict],
        payload: dict,
    ) -> Optional[dict]:
        headers = self._prep_headers(headers)
        client = await self._get_http_client()
        for attempt in range(_RETRY_TIMES):
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                self._check_code(body, url)
                return body.get("data", {})
            except ArenaParticipantError:
                raise
            except Exception as exc:
                if attempt == _RETRY_TIMES - 1:
                    logger.error("POST %s failed: %s", url, exc)
                    raise
                await asyncio.sleep(_RETRY_DELAY)
        return None
