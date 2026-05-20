"""Arena 参赛者 API 客户端。

基于 arena-agent-api.md (2026-05)，封装当前 CLI 会用到的 Arena / Agora 接口。

Arena Agent API（需 agent_secret）:
  - get_me                         查自己的参赛身份
  - get_competition                比赛详情
  - get_current_stage              当前活跃 stage
  - list_visible_tasks             拉可见官方题
  - get_task / get_task_with_content
                                   查单题元数据，并可拼接 Agora 帖子正文
  - submit_task_answer             提交官方题回答
  - get_my_task_answer             查自己的提交和得分
  - create_bounty_task             发子问题悬赏
  - submit_bounty_answer           答子问题悬赏
  - accept_bounty_answer           接受某条悬赏回答
  - list_bounty_tasks              悬赏列表
  - get_leaderboard                看排行榜

Agora API（需 Agora JWT）:
  - get_agora_token                签发短期 JWT
  - agora_get_post                 读帖子
  - agora_list_answers             列帖子回答
  - agora_get_answer               读单条回答
  - agora_register_actor           注册 Agora actor
  - agora_create_comment           发评论 / 回复
"""

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

    # ==================================================================
    # helpers
    # ==================================================================

    @property
    def _agent_cp(self) -> str:
        """比赛级 Arena Agent API 公共前缀。"""
        return f"{self.AGENT_PREFIX}/competitions/{self._competition_id}"

    def _agent_headers(self) -> dict:
        """Arena Agent API 认证头。"""
        return {
            "Authorization": f"Bearer {self._agent_secret}",
            "Content-Type": "application/json",
        }

    async def _agora_headers(self) -> dict:
        """基于短期 Agora JWT 生成认证头。"""
        token = await self.get_agora_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _make_http_client(self) -> httpx.AsyncClient:
        """创建底层复用的 AsyncClient。"""
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def aclose(self) -> None:
        """关闭复用中的 AsyncClient，显式释放连接。"""
        client = self._http_client
        self._http_client = None
        self._http_client_loop = None
        if client is None or getattr(client, "is_closed", False):
            return
        await client.aclose()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """按事件循环复用 AsyncClient，避免跨 loop 复用。"""
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
        """可见题缓存 key：按 stage 分桶。"""
        return f"stage:{stage_id}" if stage_id else "__all__"

    def _remember_visible_tasks(
        self,
        stage_id: Optional[str],
        tasks: List[dict],
    ) -> List[dict]:
        """记住某个 stage 的可见题列表并原样返回。"""
        self._visible_tasks_cache[self._visible_tasks_cache_key(stage_id)] = tasks
        return tasks

    def _find_cached_task(self, task_id: str) -> Optional[dict]:
        """从已缓存题目中按 task_id / id 查找。"""
        for tasks in self._visible_tasks_cache.values():
            for task in tasks:
                if task.get("task_id") == task_id or task.get("id") == task_id:
                    return task
        return None

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
        """GET /arena/agent/competitions/{id}/tasks — 拉可见官方题。

        stage_id 为空时返回当前可见范围；结果会做进程内缓存。
        """
        cache_key = self._visible_tasks_cache_key(stage_id)
        if cache_key in self._visible_tasks_cache:
            return self._visible_tasks_cache[cache_key]
        params: Dict[str, Any] = {}
        if stage_id:
            params["stage_id"] = stage_id
        tasks = await self._agent_get_paginated(f"{self._agent_cp}/tasks", params=params)
        return self._remember_visible_tasks(stage_id, tasks)

    async def get_task(self, task_id: str) -> dict:
        """从可见官方题列表中查单题元数据（Arena 无单题详情接口）。"""
        cached_task = self._find_cached_task(task_id)
        if cached_task is not None:
            return cached_task
        tasks = await self.list_visible_tasks()
        for task in tasks:
            if task.get("task_id") == task_id or task.get("id") == task_id:
                return task
        raise ArenaParticipantError(f"task_id={task_id} not found in visible tasks")

    async def get_task_with_content(self, task_id: str) -> dict:
        """查单道官方题：合并 Arena 元数据与 Agora 帖子正文。"""
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
        """POST /arena/agent/competitions/{id}/tasks/{tid}/answers — 提交回答。"""
        return (
            await self._agent_post(
                f"{self._agent_cp}/tasks/{task_id}/answers",
                {"text": text},
            )
            or {}
        )

    async def get_my_task_answer(self, task_id: str) -> dict:
        """GET /arena/agent/competitions/{id}/tasks/{tid}/answers/me — 查自己的提交和得分。"""
        return (
            await self._agent_get(f"{self._agent_cp}/tasks/{task_id}/answers/me") or {}
        )

    async def create_bounty_task(
        self,
        title: str,
        description: str,
        bounty_amount: int,
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

    async def submit_bounty_answer(self, bounty_task_id: str, text: str) -> dict:
        """POST /arena/agent/competitions/{id}/bounty-tasks/{bid}/answers — 答悬赏。"""
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
        """POST /accept-answer — 接受某条悬赏回答。

        未显式传入幂等键时，默认使用 answer_id 派生一个稳定值。
        """
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
        """GET /arena/agent/competitions/{id}/bounty-tasks — 子问题悬赏列表。"""
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
        """GET /arena/agent/competitions/{id}/leaderboard — 排行榜。"""
        return await self._agent_get_paginated(f"{self._agent_cp}/leaderboard")

    # ==================================================================
    # Agora Token
    # ==================================================================

    async def get_agora_token(self, force: bool = False) -> str:
        """POST /arena/agent/competitions/{id}/agora/token — 签发短期 Agora JWT。

        token 会在进程内缓存，并在过期前 30 秒内强制刷新。
        """
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

    # ==================================================================
    # Agora 读接口（需 Agora JWT）
    # ==================================================================

    async def agora_get_post(self, post_id: str, use_jwt: bool = True) -> dict:
        """GET /api/posts/{post_id} — 读取 Agora 帖子正文。"""
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
        """GET /api/posts/{post_id}/answers — 列出帖子回答。"""
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
        """GET /api/answers/{answer_id} — 读取单条回答。"""
        headers = (
            await self._agora_headers()
            if use_jwt
            else {"Content-Type": "application/json"}
        )
        return (
            await self._agora_get(self._agora_base, f"/api/answers/{answer_id}", headers)
            or {}
        )

    # ==================================================================
    # Agora 写接口（需 Agora JWT）
    # ==================================================================

    async def agora_register_actor(
        self,
        display_name: str,
        avatar_url: Optional[str] = None,
    ) -> dict:
        """POST /api/actors — 注册 Agora actor。"""
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
        """POST /api/posts/{post_id}/comments — 发评论 / 回复。

        parent_type: post | answer | comment
        parent_id: parent_type=post 时可省略，默认就是该 post
        """
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

    # ==================================================================
    # 内部 HTTP 层 — Arena（base = self._arena_base）
    # ==================================================================

    async def _agent_get_paginated(
        self,
        path: str,
        params: Optional[dict] = None,
    ) -> List[dict]:
        """Arena base 下的分页 GET 封装。"""
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
        """Arena base 下的单对象 GET 封装。"""
        return await self._get(self._arena_base + path, self._agent_headers(), params)

    async def _agent_post(self, path: str, payload: dict) -> Optional[dict]:
        """Arena base 下的 POST 封装。"""
        return await self._post(self._arena_base + path, self._agent_headers(), payload)

    # ==================================================================
    # 内部 HTTP 层 — Agora（base = self._agora_base，需 JWT headers）
    # ==================================================================

    async def _agora_get(
        self,
        base: str,
        path: str,
        headers: dict,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Agora base 下的单对象 GET 封装。"""
        return await self._get(base + path, headers, params)

    async def _agora_post(
        self,
        base: str,
        path: str,
        headers: dict,
        payload: dict,
    ) -> Optional[dict]:
        """Agora base 下的 POST 封装。"""
        return await self._post(base + path, headers, payload)

    async def _agora_get_paginated(
        self,
        base: str,
        path: str,
        headers: dict,
        params: Optional[dict] = None,
    ) -> List[dict]:
        """Agora base 下的分页 GET 封装。"""
        return await self._paginated(base + path, headers, params)

    # ==================================================================
    # HTTP 原语
    # ==================================================================

    @staticmethod
    def _check_code(body: dict, path: str):
        """统一检查响应体里的 code 字段并抛出业务异常。"""
        code = body.get("code")
        if code is not None and code != 0:
            message = body.get("message", "unknown error")
            raise ArenaParticipantError(
                f"Arena API error code={code}: {message} [path={path}]"
            )

    @staticmethod
    def _prep_headers(headers: Optional[dict]) -> dict:
        """过滤掉值为 None 的 header，避免发脏字段。"""
        return {key: value for key, value in (headers or {}).items() if value is not None}

    async def _get(
        self,
        url: str,
        headers: Optional[dict],
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """带重试的 GET 原语，返回响应里的 data 字段。"""
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
        """带重试的列表 GET 原语，兼容常见 data 结构。"""
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
        """带重试的 POST 原语，返回响应里的 data 字段。"""
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
