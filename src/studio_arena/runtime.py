"""Automation runtime for Studio Arena."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import ArenaRuntimeConfig
from .store import ArenaStateStore

_DEFAULT_GATE_MODE = "normal_gate"
_COVERAGE_GATE_MODE = "coverage_gate"
_SYNC_STATE_ANSWER_CONCURRENCY = 5
_PROBE_BOUNTY_AMOUNT = 1
_DEFAULT_PROBE_REPLY_TARGET = 4
_DEFAULT_PROBE_POLL_ATTEMPTS = 3
_DEFAULT_PROBE_POLL_INTERVAL_SECONDS = 0.0
_HIGH_BOUNTY_THRESHOLD = 100
_FOLLOWUP_FINAL_REWRITE_MODE = "final_rewrite"
_FOLLOWUP_REFINE_TRIGGER = "followup_refine"
_POST_COVERAGE_TRIGGER = "post_coverage"
_FOLLOWUP_LOW_SCORE_THRESHOLD = 0.5
_FOLLOWUP_REPLY_LIMIT = 1
_HIGH_BOUNTY_SKIP_REASONS = {"skipped_low_confidence", "skipped_unanswerable"}
_STABILITY_BLOCKING_ALERTS = {
    "coverage_stall",
    "repeated_submit_failure",
    "state_missing_for_active_run",
    "old_stage_context_detected",
    "frozen_task_context_detected",
}
_RAW_CITATION_PATTERNS = [
    re.compile(r"cite.*?"),
    re.compile(r"cite[^\n]+"),
]
_JWT_LIKE_PATTERN = re.compile(
    r"\b[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"
)
_BEARER_LIKE_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._=-]{16,}")
_SECRET_KEYWORDS = ("agent_secret", "ARENA_AGENT_SECRET", "apiKey", "access_token")
_FACT_MARKERS = ("事实", "已知", "fact", "facts", "known facts", "根据")
_INFERENCE_MARKERS = ("推断", "推论", "inference", "infer")
_ASSUMPTION_MARKERS = ("假设", "assumption", "assume")
_UNCERTAINTY_MARKERS = ("不确定", "限制", "uncertainty", "unknown", "risk")
_COMMENT_TEXT_KEYS = ("content", "text", "body", "message")
_COMMENT_LIST_KEYS = (
    "comments",
    "comment_list",
    "latest_comments",
    "followups",
    "replies",
)
_COMMENT_ID_KEYS = ("comment_id", "commentId", "id")
_COMMENT_PARENT_KEYS = (
    "parent_id",
    "parentId",
    "parent_type",
    "parentType",
    "post_id",
    "postId",
    "agora_post_id",
    "root_comment_id",
    "rootCommentId",
)
_AUTHOR_ID_KEYS = (
    "participant_id",
    "author_participant_id",
    "creator_participant_id",
    "user_id",
    "author_id",
    "actor_id",
    "agent_id",
    "creator_id",
    "owner_id",
)
_AUTHOR_CONTAINER_TOKENS = ("author", "actor", "agent", "user", "participant", "creator")
_AUTHOR_SELF_FLAGS = ("authored_by_me", "is_mine", "mine", "by_me", "self_authored")
_DOMAIN_AGENT_BINDINGS = {
    "medical": ("medical-agent", "arena-medical/SKILL.md"),
    "legal": ("legal-agent", "arena-legal/SKILL.md"),
    "business": ("business-agent", "arena-business/SKILL.md"),
    "industrial": ("industrial-agent", "arena-industrial/SKILL.md"),
    "science": ("science-agent", "arena-science/SKILL.md"),
    "general": ("general-agent", "arena-general/SKILL.md"),
}
_KEYWORD_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "into",
    "your",
    "need",
    "must",
    "should",
    "please",
    "using",
    "about",
    "what",
    "which",
    "when",
    "where",
    "have",
    "been",
    "will",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _deep_iter_dicts(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _deep_iter_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _deep_iter_dicts(item)


def _preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _normalize_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_scope_token(value: Any, fallback: str = "unknown") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "")).strip("-")
    return normalized or fallback


def _safe_file_token(value: Any, fallback: str = "task") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    return normalized or fallback


def _count_words(text: str) -> int:
    latin_tokens = re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*", text or "")
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text or "")
    return len(latin_tokens) + len(cjk_chars)


def _pick_first(source: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            normalized = str(item).strip()
            if normalized:
                items.append(normalized)
        return items
    return []


def _collect_rule_terms(
    sources: Sequence[Any],
    keys: Sequence[str],
) -> List[str]:
    collected: List[str] = []
    seen: set[str] = set()
    for source in sources:
        if isinstance(source, dict):
            candidates = _deep_iter_dicts(source)
        else:
            candidates = []
        for candidate in candidates:
            for key in keys:
                for item in _as_string_list(candidate.get(key)):
                    lowered = item.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    collected.append(item)
    return collected


def _keywords_from_text(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", (text or "").lower())
    keywords: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _KEYWORD_STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _extract_bounty_task_id(payload: Dict[str, Any]) -> Optional[str]:
    value = _pick_first(
        payload,
        ("bounty_task_id", "task_id", "id", "bountyTaskId"),
    )
    return str(value) if value not in (None, "") else None


def _extract_bounty_post_id(payload: Dict[str, Any]) -> Optional[str]:
    value = _pick_first(
        payload,
        ("agora_post_id", "post_id", "postId", "agoraPostId"),
    )
    return str(value) if value not in (None, "") else None


def _extract_bounty_status(payload: Dict[str, Any]) -> str:
    value = _pick_first(payload, ("status", "bounty_status", "state"))
    return str(value or "").strip().lower()


def _extract_bounty_publisher_id(payload: Dict[str, Any]) -> Optional[str]:
    value = _pick_first(
        payload,
        (
            "publisher_participant_id",
            "publisherParticipantId",
            "participant_id",
            "owner_id",
            "ownerId",
            "creator_id",
            "creatorId",
        ),
    )
    return str(value) if value not in (None, "") else None


def _extract_bounty_amount(payload: Dict[str, Any]) -> Optional[int]:
    value = _pick_first(
        payload,
        ("bounty_amount", "amount", "reward", "price", "bountyAmount"),
    )
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_bounty_created_at(payload: Dict[str, Any]) -> Optional[str]:
    value = _pick_first(
        payload,
        ("created_at", "createdAt", "published_at", "publishedAt", "updated_at"),
    )
    return str(value) if value not in (None, "") else None


def _bounty_text_blob(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("title", "description", "content", "body", "text", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def stable_evidence_blob(*payloads: Any) -> str:
    chunks: List[str] = []
    for payload in payloads:
        if payload in (None, "", [], {}):
            continue
        if isinstance(payload, str):
            chunks.append(payload)
        else:
            try:
                chunks.append(str(payload))
            except Exception:
                continue
    return "\n".join(chunks).lower()


def _walk_dict_paths(
    node: Any,
    path: Sequence[str] = (),
) -> Iterable[tuple[tuple[str, ...], Dict[str, Any]]]:
    if isinstance(node, dict):
        yield tuple(path), node
        for key, value in node.items():
            yield from _walk_dict_paths(value, (*path, str(key).lower()))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_dict_paths(item, (*path, f"[{index}]"))


def _path_mentions_comment(path: Sequence[str]) -> bool:
    for part in path:
        lowered = str(part).lower()
        if lowered.startswith("["):
            continue
        if "comment" in lowered or lowered in _COMMENT_LIST_KEYS:
            return True
        if lowered in {"followup", "reply"}:
            return True
    return False


def _extract_author_ids(item: Dict[str, Any]) -> set[str]:
    author_ids: set[str] = set()
    for key in _AUTHOR_ID_KEYS:
        value = item.get(key)
        if value not in (None, ""):
            author_ids.add(str(value))

    for key, value in item.items():
        lowered = str(key).lower()
        if not any(token in lowered for token in _AUTHOR_CONTAINER_TOKENS):
            continue
        for candidate in _deep_iter_dicts(value):
            for author_key in (*_AUTHOR_ID_KEYS, "id"):
                author_value = candidate.get(author_key)
                if author_value not in (None, ""):
                    author_ids.add(str(author_value))
    return author_ids


def _authored_by_me(item: Dict[str, Any], participant_id: Optional[str]) -> bool:
    if any(bool(item.get(flag)) for flag in _AUTHOR_SELF_FLAGS):
        return True
    if not participant_id:
        return False
    return str(participant_id) in _extract_author_ids(item)


def _looks_like_comment_candidate(item: Dict[str, Any], path: Sequence[str]) -> bool:
    content = _pick_first(item, _COMMENT_TEXT_KEYS)
    if not isinstance(content, str) or not content.strip():
        return False
    if _path_mentions_comment(path):
        return True

    keys = {str(key).lower() for key in item}
    has_explicit_comment_id = any(key in item for key in ("comment_id", "commentId"))
    has_commentish_key = any("comment" in key for key in keys)
    has_parent_meta = any(
        _pick_first(item, (key,)) not in (None, "") for key in _COMMENT_PARENT_KEYS
    )
    has_author_meta = bool(_extract_author_ids(item))
    has_parent_type = bool(_pick_first(item, ("parent_type", "parentType")))
    has_object_id = _pick_first(item, _COMMENT_ID_KEYS) not in (None, "")

    if has_explicit_comment_id and (
        has_parent_meta or has_author_meta or has_commentish_key
    ):
        return True
    if has_object_id and has_parent_meta and has_author_meta:
        return True
    if has_parent_type and (
        has_author_meta or has_explicit_comment_id or has_commentish_key
    ):
        return True
    return False


def _comment_matches_latest_reply(
    comment: Dict[str, Any],
    previous_state: Dict[str, Any],
) -> bool:
    last_reply_hash = str(previous_state.get("last_followup_reply_hash") or "")
    last_reply_comment_id = str(previous_state.get("last_followup_reply_comment_id") or "")
    last_reply_parent_id = str(previous_state.get("last_followup_reply_parent_id") or "")
    replied_comment_id = str(previous_state.get("last_replied_comment_id") or "")

    comment_id = str(comment.get("comment_id") or "")
    parent_id = str(comment.get("parent_id") or "")
    content_hash = str(comment.get("content_hash") or "")

    if last_reply_comment_id and comment_id and comment_id == last_reply_comment_id:
        return True
    if last_reply_hash and content_hash and content_hash == last_reply_hash:
        return True
    if (
        last_reply_parent_id
        and comment_id
        and comment_id == last_reply_parent_id
        and parent_id
        and parent_id == replied_comment_id
    ):
        return True
    if (
        last_reply_hash
        and replied_comment_id
        and parent_id == replied_comment_id
        and content_hash == last_reply_hash
    ):
        return True
    return False


def _first_numeric(node: Any, keys: Sequence[str]) -> Optional[float]:
    for candidate in _deep_iter_dicts(node):
        value = _pick_first(candidate, keys)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _first_string(node: Any, keys: Sequence[str]) -> Optional[str]:
    for candidate in _deep_iter_dicts(node):
        value = _pick_first(candidate, keys)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_comments(payload: Dict[str, Any], participant_id: Optional[str]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path, item in _walk_dict_paths(payload):
        if _looks_like_comment_candidate(item, path):
            candidates.append(item)

    comments: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for order, item in enumerate(candidates):
        comment_id = str(_pick_first(item, _COMMENT_ID_KEYS) or "").strip()
        content = _pick_first(item, _COMMENT_TEXT_KEYS)
        if not isinstance(content, str) or not content.strip():
            continue

        created_at = _pick_first(
            item,
            ("created_at", "createdAt", "commented_at", "updated_at"),
        )
        author_ids = sorted(_extract_author_ids(item))
        author_id = author_ids[0] if author_ids else None
        authored_by_me = _authored_by_me(item, participant_id)
        signature = _normalize_signature(content)
        content_hash = _hash_text(_normalize_text(content.strip()))
        dedupe_key = comment_id or signature
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        comments.append(
            {
                "comment_id": comment_id or signature[:12],
                "content": content.strip(),
                "created_at": str(created_at or ""),
                "post_id": _pick_first(item, ("post_id", "postId", "agora_post_id")),
                "parent_id": _pick_first(item, ("parent_id", "parentId")),
                "parent_type": _pick_first(item, ("parent_type", "parentType"))
                or "comment",
                "author_id": author_id,
                "author_ids": author_ids,
                "authored_by_me": authored_by_me,
                "signature": signature,
                "content_hash": content_hash,
                "_order": order,
            }
        )

        comments.sort(
        key=lambda item: (
            _parse_datetime(item.get("created_at")) is None,
            _parse_datetime(item.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
            int(item.get("_order", 0)),
        )
    )
    return comments


def _extract_answer_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _first_string(
        payload,
        ("text", "answer_text", "content", "body"),
    )
    answer_id = _first_string(payload, ("answer_id", "id"))
    submitted_at = _first_string(
        payload,
        ("submitted_at", "created_at", "answered_at", "updated_at"),
    )
    latest_score = _first_numeric(payload, ("score", "current_score", "final_score"))
    submitted = bool(text or answer_id or submitted_at or latest_score is not None)
    return {
        "submitted": submitted,
        "answer_id": answer_id,
        "submitted_at": submitted_at,
        "text": text or "",
        "excerpt": _preview(text or ""),
        "score": latest_score,
    }


def _classify_domain(task: Dict[str, Any]) -> str:
    searchable_bits: List[str] = []
    for key in ("title", "name", "summary", "description", "text", "content"):
        value = task.get(key)
        if isinstance(value, str):
            searchable_bits.append(value.lower())

    agora_post = task.get("agora_post") or {}
    if isinstance(agora_post, dict):
        for key in ("title", "content", "text", "summary"):
            value = agora_post.get(key)
            if isinstance(value, str):
                searchable_bits.append(value.lower())

    labels = task.get("labels") or task.get("tags") or []
    if isinstance(labels, list):
        searchable_bits.extend(str(label).lower() for label in labels)

    haystack = "\n".join(searchable_bits)
    scorecard = {
        "medical": (
            "clinical",
            "patient",
            "symptom",
            "treatment",
            "drug",
            "medical",
            "诊断",
            "治疗",
            "药物",
            "临床",
        ),
        "legal": (
            "law",
            "regulation",
            "compliance",
            "liability",
            "contract",
            "statute",
            "legal",
            "法律",
            "法规",
            "合规",
            "侵权",
        ),
        "business": (
            "market",
            "pricing",
            "roi",
            "revenue",
            "profit",
            "business",
            "finance",
            "strategy",
            "定价",
            "市场",
            "商业",
            "财务",
        ),
        "industrial": (
            "algorithm",
            "backend",
            "engineering",
            "system",
            "manufacturing",
            "code",
            "python",
            "sql",
            "程序",
            "算法",
            "工程",
            "后端",
            "渲染",
        ),
        "science": (
            "chemistry",
            "physics",
            "biology",
            "mechanism",
            "experiment",
            "molecule",
            "科学",
            "化学",
            "生物",
            "实验",
            "机理",
        ),
    }
    matched: List[str] = []
    for domain, keywords in scorecard.items():
        if any(keyword in haystack for keyword in keywords):
            matched.append(domain)
    if len(matched) == 1:
        return matched[0]
    if not matched:
        return "general"
    if "industrial" in matched and any(word in haystack for word in ("algorithm", "backend", "code", "程序", "算法")):
        return "industrial"
    return "general"


def _deadline_mode(stage: Dict[str, Any]) -> str:
    for key in ("deadline_at", "end_at", "ends_at", "closing_at"):
        deadline = _parse_datetime(stage.get(key))
        if not deadline:
            continue
        remaining = deadline - _utc_now()
        if remaining <= timedelta(hours=6):
            return "urgent"
        if remaining <= timedelta(hours=24):
            return "watch"
        return "normal"
    return "normal"


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for pattern in _RAW_CITATION_PATTERNS:
        normalized = pattern.sub("", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _is_low_score_followup_candidate(
    *,
    score: Any,
    followup_pending: bool,
    followup_reply_used: bool,
) -> bool:
    if not followup_pending or followup_reply_used:
        return False
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return False
    return numeric_score <= _FOLLOWUP_LOW_SCORE_THRESHOLD


@dataclass(slots=True)
class ArenaAutomationRuntime:
    """Programmatic orchestration runtime used by CLI and Synergy bridge."""

    client: Any
    config: ArenaRuntimeConfig
    store: ArenaStateStore

    def __post_init__(self) -> None:
        self.config.ensure_directories()

    def _followup_reply_used(self, task_state: Dict[str, Any]) -> bool:
        return bool(task_state.get("followup_reply_used"))

    def _followup_reply_budget_remaining(self, task_state: Dict[str, Any]) -> int:
        if self._followup_reply_used(task_state):
            return 0
        return _FOLLOWUP_REPLY_LIMIT

    def _followup_response_mode(
        self,
        *,
        effective_action: str,
        task_state: Dict[str, Any],
    ) -> Optional[str]:
        if effective_action != "reply_followup":
            return None
        return str(
            task_state.get("followup_response_mode") or _FOLLOWUP_FINAL_REWRITE_MODE
        )

    def _is_final_followup_rewrite(
        self,
        *,
        effective_action: str,
        task_state: Dict[str, Any],
    ) -> bool:
        return self._followup_response_mode(
            effective_action=effective_action,
            task_state=task_state,
        ) == _FOLLOWUP_FINAL_REWRITE_MODE

    async def aclose(self) -> None:
        aclose = getattr(self.client, "aclose", None)
        if aclose is None:
            return
        result = aclose()
        if asyncio.iscoroutine(result):
            await result

    async def _fetch_task_answers(
        self,
        task_ids: Sequence[str],
    ) -> List[tuple[Dict[str, Any], Optional[str]]]:
        if not task_ids:
            return []

        semaphore = asyncio.Semaphore(_SYNC_STATE_ANSWER_CONCURRENCY)

        async def fetch(task_id: str) -> tuple[Dict[str, Any], Optional[str]]:
            if not task_id:
                return {}, None
            async with semaphore:
                try:
                    return await self.client.get_my_task_answer(task_id), None
                except Exception as exc:
                    return {}, str(exc)

        return await asyncio.gather(*(fetch(task_id) for task_id in task_ids))

    async def _resolve_answer_context(
        self,
        task_id: str,
        participant_id: str,
        task_state: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        cached_summary = task_state.get("cached_my_answer_summary")
        cached_comments = task_state.get("cached_comments")
        if isinstance(cached_summary, dict) and isinstance(cached_comments, list):
            return cached_summary, [
                comment for comment in cached_comments if isinstance(comment, dict)
            ]

        payload: Dict[str, Any] = {}
        try:
            payload = await self.client.get_my_task_answer(task_id)
        except Exception as exc:
            payload = {"error": str(exc)}

        return _extract_answer_summary(payload), _extract_comments(
            payload,
            participant_id=participant_id,
        )

    def _task_status_from_context(
        self,
        *,
        task_state: Dict[str, Any],
        answer_summary: Dict[str, Any],
        followup_pending: bool,
        followup_reply_used: bool,
    ) -> str:
        if followup_pending and not followup_reply_used:
            return "followup_pending"
        if answer_summary.get("submitted"):
            score = answer_summary.get("score")
            if score is not None:
                try:
                    if float(score) <= 0:
                        return "improvable"
                except (TypeError, ValueError):
                    pass
            if task_state.get("pending_bounty_solution_packet"):
                return "improvable"
            return "submitted_once"
        return str(task_state.get("status") or "seen_unsubmitted")

    def _task_action_from_status(self, status: str) -> str:
        if status == "followup_pending":
            return "reply_followup"
        if status == "seen_unsubmitted":
            return "submit_official"
        if status == "improvable":
            return "improve_official"
        return "idle"

    def _task_dispatch_priority(self, action_kind: str) -> int:
        return {
            "submit_official": 1,
            "answer_bounty": 2,
            "reply_followup": 3,
            "improve_official": 4,
            "idle": 99,
        }.get(action_kind, 99)

    def _resolve_downstream_agent(
        self,
        *,
        domain: Optional[str],
        action_kind: str,
    ) -> tuple[str, str]:
        if action_kind == "answer_bounty":
            return "bounty-agent", "arena-bounty/SKILL.md"
        if action_kind == "reply_followup":
            return "followup-agent", "arena-followup/SKILL.md"
        resolved_domain = str(domain or "general")
        return _DOMAIN_AGENT_BINDINGS.get(
            resolved_domain,
            _DOMAIN_AGENT_BINDINGS["general"],
        )

    def _task_output_file_path(self, task_id: str, action_kind: str) -> str:
        file_stem = _safe_file_token(task_id, fallback="task")
        suffix = {
            "reply_followup": "followup_reply.md",
            "improve_official": "improve.md",
            "answer_bounty": "bounty_answer.md",
        }.get(action_kind, "draft.md")
        return str(self.config.work_dir / f"{file_stem}_{suffix}")

    def _bounty_output_file_path(self, bounty_task_id: str) -> str:
        file_stem = _safe_file_token(bounty_task_id, fallback="bounty")
        return str(self.config.work_dir / f"{file_stem}_bounty_answer.md")

    def _high_bounty_alerts(self) -> set[str]:
        alerts: set[str] = set()
        for kind in ("urgent_alerts", "monitor_report"):
            snapshot = self.store.get_latest_snapshot(kind)
            if isinstance(snapshot, list):
                alerts.update(str(item).strip() for item in snapshot if str(item).strip())
            elif isinstance(snapshot, dict):
                for key in ("urgent_alerts", "alerts", "items"):
                    value = snapshot.get(key)
                    if isinstance(value, list):
                        alerts.update(
                            str(item).strip() for item in value if str(item).strip()
                        )
            elif snapshot:
                alerts.add(str(snapshot).strip())
        return {item for item in alerts if item}

    def _official_coverage_stable(self, coverage: Dict[str, Any]) -> bool:
        if int(coverage.get("unsubmitted_count", 0) or 0) > 0:
            return False
        alerts = self._high_bounty_alerts()
        return not bool(alerts & _STABILITY_BLOCKING_ALERTS)

    def _remember_visible_bounties(
        self,
        *,
        bounty_tasks: Sequence[Dict[str, Any]],
        synced_at: str,
    ) -> None:
        for bounty in bounty_tasks:
            bounty_task_id = _extract_bounty_task_id(bounty)
            if not bounty_task_id:
                continue
            previous = self.store.get_bounty_state(bounty_task_id) or {}
            payload = dict(previous)
            payload.update(
                {
                    "bounty_task_id": bounty_task_id,
                    "stage_id": bounty.get("stage_id"),
                    "title": bounty.get("title"),
                    "bounty_amount": _extract_bounty_amount(bounty),
                    "status": _extract_bounty_status(bounty) or previous.get("status") or "open",
                    "publisher_participant_id": _extract_bounty_publisher_id(bounty),
                    "agora_post_id": _extract_bounty_post_id(bounty),
                    "created_at": _extract_bounty_created_at(bounty),
                    "updated_at": synced_at,
                }
            )
            self.store.upsert_bounty_state(payload)

    async def _list_open_bounty_tasks(self, *, stage_id: Optional[str]) -> List[Dict[str, Any]]:
        list_bounty_tasks = getattr(self.client, "list_bounty_tasks", None)
        if not callable(list_bounty_tasks):
            return []
        payload = await list_bounty_tasks(stage_id=stage_id, status="open")
        return list(payload or [])

    def _list_visible_bounty_summaries(self, *, stage_id: Optional[str]) -> List[Dict[str, Any]]:
        visible: List[Dict[str, Any]] = []
        for bounty_state in self.store.list_bounty_states():
            bounty_stage_id = str(bounty_state.get("stage_id") or "")
            if stage_id and bounty_stage_id not in ("", str(stage_id)):
                continue
            visible.append(
                {
                    "bounty_task_id": bounty_state.get("bounty_task_id"),
                    "id": bounty_state.get("bounty_task_id"),
                    "stage_id": bounty_state.get("stage_id"),
                    "title": bounty_state.get("title"),
                    "description": bounty_state.get("description"),
                    "bounty_amount": bounty_state.get("bounty_amount"),
                    "status": bounty_state.get("status") or "open",
                    "publisher_participant_id": bounty_state.get("publisher_participant_id"),
                    "agora_post_id": bounty_state.get("agora_post_id"),
                    "created_at": bounty_state.get("created_at"),
                }
            )
        return visible

    def _eligible_high_bounties(
        self,
        *,
        bounty_tasks: Sequence[Dict[str, Any]],
        stage_id: Optional[str],
        participant_id: str,
    ) -> List[Dict[str, Any]]:
        eligible: List[Dict[str, Any]] = []
        for bounty in bounty_tasks:
            bounty_task_id = _extract_bounty_task_id(bounty)
            if not bounty_task_id:
                continue
            if stage_id and str(bounty.get("stage_id") or "") not in ("", str(stage_id)):
                continue
            if _extract_bounty_status(bounty) != "open":
                continue
            publisher_participant_id = _extract_bounty_publisher_id(bounty)
            if publisher_participant_id and publisher_participant_id == participant_id:
                continue
            bounty_amount = _extract_bounty_amount(bounty)
            if bounty_amount is None or bounty_amount < _HIGH_BOUNTY_THRESHOLD:
                continue
            persisted = self.store.get_bounty_state(bounty_task_id) or {}
            if persisted.get("last_answer_hash") and persisted.get("last_answered_at"):
                continue
            if persisted.get("last_skip_reason") in _HIGH_BOUNTY_SKIP_REASONS:
                continue
            eligible.append(
                {
                    "bounty_task_id": bounty_task_id,
                    "title": bounty.get("title"),
                    "description": bounty.get("description"),
                    "status": _extract_bounty_status(bounty) or "open",
                    "stage_id": bounty.get("stage_id") or stage_id,
                    "bounty_amount": bounty_amount,
                    "publisher_participant_id": publisher_participant_id,
                    "agora_post_id": _extract_bounty_post_id(bounty),
                    "created_at": _extract_bounty_created_at(bounty),
                    "raw_bounty": bounty,
                    "last_skip_reason": persisted.get("last_skip_reason"),
                    "last_skip_at": persisted.get("last_skip_at"),
                }
            )
        eligible.sort(
            key=lambda item: (
                -int(item.get("bounty_amount") or 0),
                -int(
                    (
                        _parse_datetime(item.get("created_at")) or datetime.min.replace(
                            tzinfo=timezone.utc
                        )
                    ).timestamp()
                ),
                str(item.get("bounty_task_id") or ""),
            )
        )
        return eligible

    def _high_bounty_phase_ready(
        self,
        *,
        coverage: Dict[str, Any],
        eligible_high_bounties: Sequence[Dict[str, Any]],
    ) -> bool:
        return self._official_coverage_stable(coverage) and bool(eligible_high_bounties)

    def _build_bounty_summary(
        self,
        *,
        bounty_tasks: Sequence[Dict[str, Any]],
        eligible_high_bounties: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        visible_open = 0
        visible_high = 0
        for bounty in bounty_tasks:
            if _extract_bounty_status(bounty) == "open":
                visible_open += 1
            amount = _extract_bounty_amount(bounty)
            if _extract_bounty_status(bounty) == "open" and amount is not None and amount >= _HIGH_BOUNTY_THRESHOLD:
                visible_high += 1
        top_high_bounty = eligible_high_bounties[0] if eligible_high_bounties else None
        return {
            "high_bounty_threshold": _HIGH_BOUNTY_THRESHOLD,
            "visible_bounty_count": len(list(bounty_tasks)),
            "visible_open_bounty_count": visible_open,
            "visible_high_bounty_count": visible_high,
            "eligible_high_bounty_count": len(list(eligible_high_bounties)),
            "top_high_bounty": top_high_bounty,
            "high_bounty_answer_mode": (
                "prioritized_before_followup" if eligible_high_bounties else "no_eligible_high_bounty"
            ),
        }

    def _build_bounty_dispatch_entry(
        self,
        *,
        bounty_summary: Dict[str, Any],
        execution_phase: Optional[str],
    ) -> Dict[str, Any]:
        bounty_task_id = str(bounty_summary.get("bounty_task_id") or "")
        should_dispatch = execution_phase == "high_bounty_sweep"
        dispatch_block_reason = None
        if not should_dispatch:
            if execution_phase == "first_pass_official":
                dispatch_block_reason = "waiting_for_stage_first_pass_complete"
            elif execution_phase == "followup_sweep":
                dispatch_block_reason = "waiting_for_high_bounty_sweep_complete"
            elif execution_phase == "post_coverage":
                dispatch_block_reason = "waiting_for_post_coverage_phase"
        return {
            "bounty_task_id": bounty_task_id,
            "stage_id": bounty_summary.get("stage_id"),
            "title": bounty_summary.get("title"),
            "bounty_amount": bounty_summary.get("bounty_amount"),
            "publisher_participant_id": bounty_summary.get("publisher_participant_id"),
            "agora_post_id": bounty_summary.get("agora_post_id"),
            "created_at": bounty_summary.get("created_at"),
            "task_agent_id": f"bounty-agent:{_safe_scope_token(bounty_task_id, fallback='bounty')}",
            "task_session_key": f"arena-bounty:{_safe_scope_token(bounty_task_id, fallback='bounty')}",
            "task_scope_locked": True,
            "isolated_task_context": True,
            "task_action_kind": "answer_bounty",
            "task_dispatch_priority": self._task_dispatch_priority("answer_bounty"),
            "downstream_agent": "bounty-agent",
            "selected_skill": "arena-bounty/SKILL.md",
            "task_action_gate_mode": _DEFAULT_GATE_MODE,
            "execution_phase": execution_phase,
            "task_output_file_path": self._bounty_output_file_path(bounty_task_id),
            "prepare_context_args": ["prepare-bounty-context", bounty_task_id, "--no-refresh"],
            "quality_check_action_kind": None,
            "finalize_command": "answer-bounty" if should_dispatch else None,
            "finalize_args_template": (
                ["answer-bounty", bounty_task_id, self._bounty_output_file_path(bounty_task_id)]
                if should_dispatch
                else []
            ),
            "dispatch_block_reason": dispatch_block_reason,
            "reason": "highest eligible high bounty should be handled before follow-up sweep",
            "requires_web_search": False,
            "should_dispatch": should_dispatch,
        }

    def _task_action_gate_mode(
        self,
        *,
        task_state: Dict[str, Any],
        sync_state: Dict[str, Any],
        followup_response_mode: Optional[str] = None,
    ) -> str:
        if followup_response_mode == _FOLLOWUP_FINAL_REWRITE_MODE:
            return _COVERAGE_GATE_MODE
        return self._suggest_gate_mode(task_state, sync_state)

    def _execution_phase(
        self,
        coverage: Dict[str, Any],
        *,
        eligible_high_bounties: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        if int(coverage.get("unsubmitted_count", 0) or 0) > 0:
            return "first_pass_official"
        if self._high_bounty_phase_ready(
            coverage=coverage,
            eligible_high_bounties=eligible_high_bounties or (),
        ):
            return "high_bounty_sweep"
        if int(coverage.get("followup_pending_count", 0) or 0) > 0:
            return "followup_sweep"
        return "post_coverage"

    def _phase_snapshot(
        self,
        coverage: Dict[str, Any],
        *,
        eligible_high_bounties: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        execution_phase = self._execution_phase(
            coverage,
            eligible_high_bounties=eligible_high_bounties,
        )
        followup_pending_count = int(coverage.get("followup_pending_count", 0) or 0)
        return {
            "execution_phase": execution_phase,
            "followup_sweep_ready": execution_phase == "followup_sweep",
            "deferred_followup_count": (
                followup_pending_count
                if execution_phase in {"first_pass_official", "high_bounty_sweep"}
                else 0
            ),
        }

    def _phase_allows_dispatch(
        self,
        *,
        action_kind: str,
        execution_phase: Optional[str],
    ) -> bool:
        if action_kind == "idle":
            return False
        if execution_phase is None:
            return True
        if execution_phase == "first_pass_official":
            return action_kind == "submit_official"
        if execution_phase == "high_bounty_sweep":
            return action_kind == "answer_bounty"
        if execution_phase == "followup_sweep":
            return action_kind == "reply_followup"
        if execution_phase == "post_coverage":
            return action_kind == "improve_official"
        return True

    def _dispatch_block_reason(
        self,
        *,
        action_kind: str,
        execution_phase: Optional[str],
    ) -> Optional[str]:
        if action_kind == "idle" or execution_phase is None:
            return None
        if execution_phase == "first_pass_official" and action_kind != "submit_official":
            return "waiting_for_stage_first_pass_complete"
        if execution_phase == "high_bounty_sweep" and action_kind != "answer_bounty":
            return "waiting_for_high_bounty_sweep_complete"
        if execution_phase == "followup_sweep" and action_kind != "reply_followup":
            return "waiting_for_followup_sweep_complete"
        if execution_phase == "post_coverage" and action_kind != "improve_official":
            return "waiting_for_post_coverage_phase"
        return None

    def _build_task_scope(
        self,
        *,
        task_id: str,
        stage_id: Optional[str],
        title: Optional[str],
        status: str,
        domain: Optional[str],
        gate_mode: str,
        latest_actionable_comment: Optional[Dict[str, Any]] = None,
        followup_response_mode: Optional[str] = None,
        followup_reply_budget_remaining: int = 0,
        followup_low_score_refine: bool = False,
        execution_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_action_kind = self._task_action_from_status(status)
        downstream_agent, selected_skill = self._resolve_downstream_agent(
            domain=domain,
            action_kind=task_action_kind,
        )
        task_scope_stage = _safe_scope_token(stage_id, fallback="unknown-stage")
        task_scope_task = _safe_scope_token(task_id, fallback="task")
        task_agent_id = f"task-agent:{task_scope_task}"
        task_session_key = f"arena:{task_scope_stage}:{task_scope_task}"
        output_file_path = self._task_output_file_path(task_id, task_action_kind)
        should_dispatch = self._phase_allows_dispatch(
            action_kind=task_action_kind,
            execution_phase=execution_phase,
        )
        dispatch_block_reason = None
        if task_action_kind != "idle" and not should_dispatch:
            dispatch_block_reason = self._dispatch_block_reason(
                action_kind=task_action_kind,
                execution_phase=execution_phase,
            )
        finalize_command = (
            "reply-followup" if task_action_kind == "reply_followup" else "submit-file"
        )
        if not should_dispatch:
            finalize_command = None
        comment_id = None
        post_id = None
        latest_comment_excerpt = None
        if isinstance(latest_actionable_comment, dict):
            comment_id = latest_actionable_comment.get("comment_id")
            post_id = latest_actionable_comment.get("post_id")
            latest_comment_excerpt = latest_actionable_comment.get("excerpt")

        return {
            "task_id": task_id,
            "stage_id": stage_id,
            "title": title,
            "task_agent_id": task_agent_id,
            "task_session_key": task_session_key,
            "task_scope_locked": True,
            "isolated_task_context": True,
            "task_action_kind": task_action_kind,
            "task_dispatch_priority": self._task_dispatch_priority(task_action_kind),
            "downstream_agent": downstream_agent,
            "selected_skill": selected_skill,
            "task_action_gate_mode": gate_mode,
            "execution_phase": execution_phase,
            "task_output_file_path": output_file_path,
            "prepare_context_args": ["prepare-context", task_id, "--no-refresh"],
            "quality_check_action_kind": task_action_kind if should_dispatch else None,
            "finalize_command": finalize_command,
            "finalize_args_template": (
                [finalize_command, task_id, output_file_path] if finalize_command else []
            ),
            "comment_id": comment_id,
            "post_id": post_id,
            "latest_comment_excerpt": latest_comment_excerpt,
            "followup_response_mode": followup_response_mode,
            "followup_reply_budget_remaining": int(followup_reply_budget_remaining or 0),
            "followup_low_score_refine": bool(followup_low_score_refine),
            "dispatch_block_reason": dispatch_block_reason,
            "should_dispatch": should_dispatch,
        }

    def _build_task_dispatch_entry(
        self,
        *,
        task_summary: Dict[str, Any],
        sync_state: Dict[str, Any],
        active_stage_id: Optional[str],
        execution_phase: Optional[str],
    ) -> Dict[str, Any]:
        task_id = str(task_summary.get("task_id") or "")
        stage_id = task_summary.get("stage_id") or active_stage_id
        status = str(task_summary.get("status") or "seen_unsubmitted")
        followup_response_mode = task_summary.get("followup_response_mode")
        task_scope = self._build_task_scope(
            task_id=task_id,
            stage_id=stage_id,
            title=task_summary.get("title"),
            status=status,
            domain=task_summary.get("domain"),
            gate_mode=self._task_action_gate_mode(
                task_state=task_summary,
                sync_state=sync_state,
                followup_response_mode=followup_response_mode,
            ),
            latest_actionable_comment=task_summary.get("latest_actionable_comment"),
            followup_response_mode=followup_response_mode,
            followup_reply_budget_remaining=int(
                task_summary.get("followup_reply_budget_remaining", 0)
            ),
            followup_low_score_refine=bool(task_summary.get("followup_low_score_refine")),
            execution_phase=execution_phase,
        )
        task_action_kind = task_scope["task_action_kind"]
        action_reason = {
            "reply_followup": "latest actionable follow-up should be handled inside an isolated task agent",
            "submit_official": "visible official task still needs a first submission",
            "improve_official": "submitted task is worth a bounded improvement pass",
            "idle": "task is already covered and does not need a dedicated agent right now",
        }.get(task_action_kind, "task state does not require dispatch")

        return {
            "task_id": task_id,
            "stage_id": stage_id,
            "title": task_summary.get("title"),
            "status": status,
            "domain": task_summary.get("domain"),
            "submitted": bool(task_summary.get("submitted")),
            "last_score": task_summary.get("last_score"),
            "latest_comment_id": task_summary.get("latest_comment_id"),
            "latest_comment_created_at": task_summary.get("latest_comment_created_at"),
            "latest_comment_excerpt": task_scope.get("latest_comment_excerpt"),
            "requires_web_search": bool(task_summary.get("requires_web_search", True)),
            "budget_mode": (sync_state.get("resource_snapshot") or {}).get("budget_mode"),
            "bounty_locked": (
                (sync_state.get("resource_snapshot") or {}).get("bounty_mode") == "locked"
            )
            and not bool(task_summary.get("followup_low_score_refine")),
            "reason": action_reason,
            **task_scope,
        }

    async def sync_state(self, include_leaderboard: bool = True) -> Dict[str, Any]:
        synced_at = _iso_now()
        me = await self.client.get_me()
        competition = await self.client.get_competition()
        current_stage = await self.client.get_current_stage()
        stage_id = current_stage.get("id") or current_stage.get("stage_id")
        tasks = await self.client.list_visible_tasks(stage_id=stage_id)
        bounty_tasks = await self._list_open_bounty_tasks(stage_id=stage_id)
        leaderboard = await self.client.get_leaderboard() if include_leaderboard else []

        participant_id = str(me.get("participant_id") or "")
        task_ids = [str(task.get("task_id") or task.get("id") or "") for task in tasks]
        task_answers = await self._fetch_task_answers(task_ids)
        task_summaries: List[Dict[str, Any]] = []
        followups: List[Dict[str, Any]] = []

        for task, (my_answer, answer_error) in zip(tasks, task_answers):
            task_id = str(task.get("task_id") or task.get("id") or "")
            answer_summary = _extract_answer_summary(my_answer)
            comments = _extract_comments(my_answer, participant_id=participant_id)

            task_summary = self._build_task_summary(
                task=task,
                my_answer=my_answer,
                participant_id=participant_id,
                answer_error=answer_error,
                synced_at=synced_at,
                answer_summary=answer_summary,
                comments=comments,
            )
            task_summaries.append(task_summary)
            latest_comment = task_summary.get("latest_actionable_comment")
            if latest_comment:
                followups.append(
                    {
                        "task_id": task_id,
                        "title": task_summary.get("title"),
                        "comment_id": latest_comment.get("comment_id"),
                        "post_id": latest_comment.get("post_id") or task.get("agora_post_id"),
                        "created_at": latest_comment.get("created_at"),
                        "excerpt": latest_comment.get("excerpt"),
                    }
                )
            persisted_task_state = dict(task_summary)
            persisted_task_state["cached_my_answer_summary"] = answer_summary
            persisted_task_state["cached_comments"] = comments
            self.store.upsert_task_state(persisted_task_state)

        coverage = self._build_coverage(task_summaries, current_stage)
        self._remember_visible_bounties(
            bounty_tasks=bounty_tasks,
            synced_at=synced_at,
        )
        eligible_high_bounties = self._eligible_high_bounties(
            bounty_tasks=bounty_tasks,
            stage_id=stage_id,
            participant_id=participant_id,
        )
        visible_bounty_summary = self._build_bounty_summary(
            bounty_tasks=bounty_tasks,
            eligible_high_bounties=eligible_high_bounties,
        )
        phase_snapshot = self._phase_snapshot(
            coverage,
            eligible_high_bounties=eligible_high_bounties,
        )
        resource_snapshot = self._build_resource_snapshot(
            me=me,
            leaderboard=leaderboard,
            coverage=coverage,
            synced_at=synced_at,
            eligible_high_bounties=eligible_high_bounties,
            visible_bounty_summary=visible_bounty_summary,
        )
        next_action = self._choose_next_action(
            task_summaries,
            coverage,
            resource_snapshot,
            eligible_high_bounties=eligible_high_bounties,
        )

        payload = {
            "synced_at": synced_at,
            "competition": competition,
            "current_stage": current_stage,
            "current_stage_task_ids": [
                str(task.get("task_id") or "")
                for task in task_summaries
                if str(task.get("task_id") or "")
            ],
            "resource_snapshot": resource_snapshot,
            "official_task_coverage": coverage,
            **phase_snapshot,
            "tasks": task_summaries,
            "followups": followups,
            "visible_bounty_summary": visible_bounty_summary,
            "eligible_high_bounties": eligible_high_bounties,
            "next_action": next_action,
        }
        self.store.save_snapshot("sync_state", payload, synced_at)
        self.store.save_snapshot("resource_snapshot", resource_snapshot, synced_at)
        self.store.save_snapshot("official_task_coverage", coverage, synced_at)
        self.store.append_run_log(
            "sync_state",
            {
                "task_count": len(task_summaries),
                "followup_count": coverage["followup_pending_count"],
                "unsubmitted_count": coverage["unsubmitted_count"],
                "next_action_kind": next_action["kind"],
                "execution_phase": payload["execution_phase"],
            },
            created_at=synced_at,
            status="success",
        )
        return payload

    async def next_action(self, refresh: bool = True) -> Dict[str, Any]:
        if refresh:
            return (await self.sync_state()).get("next_action", self._idle_action())
        snapshot = self.store.get_latest_snapshot("sync_state")
        if snapshot and snapshot.get("next_action"):
            return snapshot["next_action"]
        return (await self.sync_state()).get("next_action", self._idle_action())

    async def prepare_context(self, task_id: str, refresh: bool = True) -> Dict[str, Any]:
        if refresh:
            sync_state = await self.sync_state()
        else:
            sync_state = self.store.get_latest_snapshot("sync_state") or await self.sync_state()

        task = await self.client.get_task_with_content(task_id)
        task_state = self.store.get_task_state(task_id) or {}
        participant_id = str(sync_state["resource_snapshot"].get("participant_id") or "")
        answer_summary, comments = await self._resolve_answer_context(
            task_id=task_id,
            participant_id=participant_id,
            task_state=task_state,
        )
        latest_actionable = self._latest_actionable_comment(comments, task_state)
        suggested_domain = _classify_domain(task)
        suggested_gate_mode = self._suggest_gate_mode(task_state, sync_state)
        next_action = sync_state.get("next_action", self._idle_action())
        task_content = self._task_content(task)
        followup_reply_used = self._followup_reply_used(task_state)
        followup_reply_budget_remaining = self._followup_reply_budget_remaining(task_state)
        followup_low_score_refine = _is_low_score_followup_candidate(
            score=answer_summary.get("score"),
            followup_pending=bool(task_state.get("followup_pending")),
            followup_reply_used=followup_reply_used,
        )
        followup_response_mode = (
            _FOLLOWUP_FINAL_REWRITE_MODE
            if bool(task_state.get("followup_pending")) or followup_reply_used
            else None
        )
        task_status = self._task_status_from_context(
            task_state=task_state,
            answer_summary=answer_summary,
            followup_pending=bool(task_state.get("followup_pending")),
            followup_reply_used=followup_reply_used,
        )
        task_scope = self._build_task_scope(
            task_id=task_id,
            stage_id=task.get("stage_id") or sync_state.get("current_stage", {}).get("id"),
            title=task.get("title") or task.get("name") or task_state.get("title"),
            status=task_status,
            domain=suggested_domain,
            gate_mode=self._task_action_gate_mode(
                task_state={**task_state, "status": task_status},
                sync_state=sync_state,
                followup_response_mode=followup_response_mode,
            ),
            latest_actionable_comment=latest_actionable,
            followup_response_mode=followup_response_mode,
            followup_reply_budget_remaining=followup_reply_budget_remaining,
            followup_low_score_refine=followup_low_score_refine,
            execution_phase=sync_state.get("execution_phase"),
        )

        context = {
            "task_id": task_id,
            "title": task.get("title") or task.get("name") or task_state.get("title"),
            "task_metadata": {
                "stage_id": task.get("stage_id"),
                "agora_post_id": task.get("agora_post_id"),
                "status": task_state.get("status"),
            },
            "task_content": task_content,
            "my_recent_answer": answer_summary,
            "recent_comments_summary": [
                {
                    "comment_id": comment["comment_id"],
                    "created_at": comment["created_at"],
                    "excerpt": _preview(comment["content"], 200),
                    "authored_by_me": comment["authored_by_me"],
                    "actionable": not comment["authored_by_me"],
                }
                for comment in comments[-3:]
            ],
            "latest_actionable_comment": latest_actionable,
            "suggested_domain": suggested_domain,
            "mandatory_web_search": True,
            "web_search_budget": (
                "minimal"
                if suggested_domain == "industrial"
                and any(token in task_content.lower() for token in ("algorithm", "code", "python", "程序"))
                else "normal"
            ),
            "suggested_gate_mode": suggested_gate_mode,
            "budget_mode": sync_state["resource_snapshot"]["budget_mode"],
            "bounty_locked": sync_state["resource_snapshot"]["bounty_mode"] == "locked",
            "followup_pending": bool(task_state.get("followup_pending")),
            "followup_reply_used": followup_reply_used,
            "followup_reply_budget_remaining": followup_reply_budget_remaining,
            "followup_response_mode": followup_response_mode,
            "followup_low_score_refine": followup_low_score_refine,
            "latest_probe_bounty": task_state.get("latest_probe_bounty"),
            "last_probe_blocker_hash": task_state.get("last_probe_blocker_hash"),
            "last_bounty_verdict": task_state.get("last_bounty_verdict"),
            "pending_bounty_solution_packet": task_state.get(
                "pending_bounty_solution_packet"
            ),
            "last_bounty_solution_packet": task_state.get("last_bounty_solution_packet"),
            "repeat_issue_count": int(task_state.get("repeat_issue_count", 0)),
            "quality_expectations": self._quality_expectations(
                suggested_domain,
                suggested_gate_mode,
            ),
            "next_action_for_scope": next_action if next_action.get("task_id") == task_id else None,
            "task_status": task_status,
            "task_agent_id": task_scope["task_agent_id"],
            "task_session_key": task_scope["task_session_key"],
            "task_action_kind": task_scope["task_action_kind"],
            "task_dispatch_priority": task_scope["task_dispatch_priority"],
            "task_scope_locked": task_scope["task_scope_locked"],
            "isolated_task_context": task_scope["isolated_task_context"],
            "downstream_agent": task_scope["downstream_agent"],
            "selected_skill": task_scope["selected_skill"],
            "task_action_gate_mode": task_scope["task_action_gate_mode"],
            "task_output_file_path": task_scope["task_output_file_path"],
            "finalize_command": task_scope["finalize_command"],
            "execution_phase": sync_state.get("execution_phase"),
            "followup_sweep_ready": bool(sync_state.get("followup_sweep_ready")),
            "deferred_followup_count": int(sync_state.get("deferred_followup_count", 0) or 0),
            "dispatch_block_reason": task_scope["dispatch_block_reason"],
            "task_scope": task_scope,
        }
        self.store.save_snapshot("prepare_context", context, _iso_now())
        self.store.save_snapshot(f"prepare_context:{task_id}", context, _iso_now())
        return context

    async def prepare_bounty_context(
        self,
        bounty_task_id: str,
        refresh: bool = True,
    ) -> Dict[str, Any]:
        if refresh:
            sync_state = await self.sync_state()
        else:
            sync_state = self.store.get_latest_snapshot("sync_state") or await self.sync_state()

        visible_bounties = list(sync_state.get("eligible_high_bounties") or [])
        selected = next(
            (
                item
                for item in visible_bounties
                if str(item.get("bounty_task_id") or "") == str(bounty_task_id)
            ),
            None,
        )
        if selected is None:
            stage_id = (sync_state.get("current_stage") or {}).get("id") or (
                sync_state.get("current_stage") or {}
            ).get("stage_id")
            participant_id = str(
                (sync_state.get("resource_snapshot") or {}).get("participant_id") or ""
            )
            bounty_tasks = await self._list_open_bounty_tasks(stage_id=stage_id)
            selected = next(
                (
                    item
                    for item in self._eligible_high_bounties(
                        bounty_tasks=bounty_tasks,
                        stage_id=stage_id,
                        participant_id=participant_id,
                    )
                    if str(item.get("bounty_task_id") or "") == str(bounty_task_id)
                ),
                None,
            )
        if selected is None:
            raise ValueError(f"eligible high bounty not found: {bounty_task_id}")

        execution_phase = str(sync_state.get("execution_phase") or "")
        followup_sweep_ready = bool(sync_state.get("followup_sweep_ready"))
        deferred_followup_count = int(sync_state.get("deferred_followup_count", 0) or 0)
        if not refresh:
            local_stage_id = selected.get("stage_id") or (
                (sync_state.get("current_stage") or {}).get("id")
                or (sync_state.get("current_stage") or {}).get("stage_id")
            )
            participant_id = str(
                (sync_state.get("resource_snapshot") or {}).get("participant_id") or ""
            )
            local_tasks = [
                task_state
                for task_state in self.store.list_task_states()
                if str(task_state.get("stage_id") or "") == str(local_stage_id or "")
            ]
            local_coverage = self._build_coverage(
                local_tasks,
                sync_state.get("current_stage") or {},
            )
            local_high_bounties = self._eligible_high_bounties(
                bounty_tasks=self._list_visible_bounty_summaries(stage_id=local_stage_id),
                stage_id=local_stage_id,
                participant_id=participant_id,
            )
            execution_phase = self._execution_phase(
                local_coverage,
                eligible_high_bounties=local_high_bounties,
            )
            followup_sweep_ready = execution_phase == "followup_sweep"
            deferred_followup_count = (
                int(local_coverage.get("followup_pending_count", 0) or 0)
                if execution_phase in {"first_pass_official", "high_bounty_sweep"}
                else 0
            )

        raw_bounty = selected.get("raw_bounty") or {}
        if selected.get("agora_post_id") and "agora_post" not in raw_bounty:
            try:
                raw_bounty = dict(raw_bounty)
                raw_bounty["agora_post"] = await self.client.agora_get_post(
                    str(selected.get("agora_post_id")),
                    use_jwt=True,
                )
            except Exception as exc:
                raw_bounty = dict(raw_bounty)
                raw_bounty["agora_post_error"] = str(exc)

        prepared = {
            "bounty_task_id": str(selected.get("bounty_task_id") or ""),
            "title": selected.get("title"),
            "bounty_amount": selected.get("bounty_amount"),
            "publisher_participant_id": selected.get("publisher_participant_id"),
            "agora_post_id": selected.get("agora_post_id"),
            "created_at": selected.get("created_at"),
            "bounty_status": selected.get("status"),
            "bounty_description": selected.get("description") or raw_bounty.get("description"),
            "bounty_content": _bounty_text_blob(raw_bounty or selected),
            "bounty_metadata": {
                "stage_id": selected.get("stage_id"),
                "status": selected.get("status"),
                "agora_post_id": selected.get("agora_post_id"),
            },
            "raw_bounty": raw_bounty or selected.get("raw_bounty") or {},
            "web_search_budget": "normal",
            "mandatory_web_search": False,
            "suggested_gate_mode": _DEFAULT_GATE_MODE,
            "budget_mode": (sync_state.get("resource_snapshot") or {}).get("budget_mode"),
            "bounty_locked": False,
            "selected_skill": "arena-bounty/SKILL.md",
            "downstream_agent": "bounty-agent",
            "task_agent_id": f"bounty-agent:{_safe_scope_token(bounty_task_id, fallback='bounty')}",
            "task_session_key": f"arena-bounty:{_safe_scope_token(bounty_task_id, fallback='bounty')}",
            "task_action_kind": "answer_bounty",
            "task_dispatch_priority": self._task_dispatch_priority("answer_bounty"),
            "task_scope_locked": True,
            "isolated_task_context": True,
            "task_action_gate_mode": _DEFAULT_GATE_MODE,
            "task_output_file_path": self._bounty_output_file_path(bounty_task_id),
            "finalize_command": "answer-bounty",
            "execution_phase": execution_phase,
            "followup_sweep_ready": followup_sweep_ready,
            "deferred_followup_count": deferred_followup_count,
            "dispatch_block_reason": None
            if execution_phase == "high_bounty_sweep"
            else "waiting_for_high_bounty_sweep_complete",
            "next_action_for_scope": sync_state.get("next_action"),
        }
        created_at = _iso_now()
        self.store.save_snapshot("prepare_bounty_context", prepared, created_at)
        self.store.save_snapshot(f"prepare_bounty_context:{bounty_task_id}", prepared, created_at)
        return prepared

    async def dispatch_plan(
        self,
        refresh: bool = True,
        actionable_only: bool = False,
    ) -> Dict[str, Any]:
        if refresh:
            sync_state = await self.sync_state()
        else:
            sync_state = self.store.get_latest_snapshot("sync_state") or await self.sync_state()

        current_stage = sync_state.get("current_stage") or {}
        active_stage_id = current_stage.get("id") or current_stage.get("stage_id")
        task_summaries = list(sync_state.get("tasks") or [])
        eligible_high_bounties = list(sync_state.get("eligible_high_bounties") or [])
        execution_phase = str(
            sync_state.get("execution_phase")
            or self._execution_phase(
                sync_state.get("official_task_coverage") or {},
                eligible_high_bounties=eligible_high_bounties,
            )
        )
        dispatches = [
            self._build_task_dispatch_entry(
                task_summary=task,
                sync_state=sync_state,
                active_stage_id=active_stage_id,
                execution_phase=execution_phase,
            )
            for task in task_summaries
        ]
        bounty_dispatches = []
        if eligible_high_bounties:
            bounty_dispatches = [
                self._build_bounty_dispatch_entry(
                    bounty_summary=eligible_high_bounties[0],
                    execution_phase=execution_phase,
                )
            ]
        if actionable_only:
            dispatches = [item for item in dispatches if item.get("should_dispatch")]
            bounty_dispatches = [
                item for item in bounty_dispatches if item.get("should_dispatch")
            ]
        dispatches.sort(
            key=lambda item: (
                0 if item.get("should_dispatch") else 1,
                int(item.get("task_dispatch_priority", 99)),
                str(item.get("task_id") or ""),
            )
        )
        bounty_dispatches.sort(
            key=lambda item: (
                0 if item.get("should_dispatch") else 1,
                int(item.get("task_dispatch_priority", 99)),
                str(item.get("bounty_task_id") or ""),
            )
        )

        actionable_task_count = sum(1 for item in dispatches if item.get("should_dispatch"))
        actionable_bounty_count = sum(1 for item in bounty_dispatches if item.get("should_dispatch"))
        actionable_count = actionable_task_count + actionable_bounty_count
        payload = {
            "synced_at": sync_state.get("synced_at") or _iso_now(),
            "dispatch_mode": "per_task_isolated_agents",
            "active_stage_id": active_stage_id,
            "current_stage_task_ids": sync_state.get("current_stage_task_ids")
            or [
                str(task.get("task_id") or "")
                for task in task_summaries
                if str(task.get("task_id") or "")
            ],
            "next_action": sync_state.get("next_action") or self._idle_action(),
            "resource_snapshot": sync_state.get("resource_snapshot") or {},
            "official_task_coverage": sync_state.get("official_task_coverage") or {},
            "execution_phase": execution_phase,
            "followup_sweep_ready": bool(sync_state.get("followup_sweep_ready")),
            "deferred_followup_count": int(sync_state.get("deferred_followup_count", 0) or 0),
            "actionable_task_count": actionable_task_count,
            "actionable_bounty_count": actionable_bounty_count,
            "recommended_parallelism": actionable_count,
            "task_dispatches": dispatches,
            "bounty_dispatches": bounty_dispatches,
        }
        created_at = _iso_now()
        self.store.save_snapshot("dispatch_plan", payload, created_at)
        self.store.append_run_log(
            "dispatch_plan",
            {
                "dispatch_mode": payload["dispatch_mode"],
                "active_stage_id": active_stage_id,
                "visible_task_count": len(task_summaries),
                "visible_high_bounty_count": len(eligible_high_bounties),
                "actionable_task_count": actionable_task_count,
                "actionable_bounty_count": actionable_bounty_count,
                "recommended_parallelism": actionable_count,
                "execution_phase": execution_phase,
            },
            created_at=created_at,
            status="success",
        )
        return payload

    def quality_check(
        self,
        task_id: str,
        draft_text: str,
        gate_mode: str = _DEFAULT_GATE_MODE,
        action_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_text = _normalize_text(draft_text)
        task_state = self.store.get_task_state(task_id) or {}
        effective_action = action_kind or (
            "reply_followup" if task_state.get("followup_pending") else "submit_official"
        )
        effective_gate = gate_mode or _DEFAULT_GATE_MODE
        followup_response_mode = self._followup_response_mode(
            effective_action=effective_action,
            task_state=task_state,
        )
        final_followup_rewrite = self._is_final_followup_rewrite(
            effective_action=effective_action,
            task_state=task_state,
        )

        issues: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        metrics = {
            "char_count": len(normalized_text),
            "word_count": _count_words(normalized_text),
            "paragraph_count": len([part for part in normalized_text.split("\n\n") if part.strip()]),
            "contains_citation_marker": any(
                pattern.search(draft_text or "") for pattern in _RAW_CITATION_PATTERNS
            ),
        }

        if not normalized_text:
            issues.append(
                {
                    "code": "empty_draft",
                    "message": "Draft text is empty after normalization.",
                    "severity": "error",
                    "force_bypassable": False,
                }
            )

        minimum_words = 35 if effective_action == "reply_followup" else 90
        if effective_gate == _COVERAGE_GATE_MODE:
            minimum_words = 60 if effective_action == "reply_followup" else 120
        if final_followup_rewrite:
            minimum_words = max(minimum_words, 120)
        if metrics["word_count"] < minimum_words and normalized_text:
            issues.append(
                {
                    "code": "answer_too_short",
                    "message": f"Draft is shorter than the recommended floor ({minimum_words} words).",
                    "severity": "error",
                    "force_bypassable": True,
                }
            )

        if metrics["contains_citation_marker"]:
            warnings.append(
                {
                    "code": "citation_syntax_removed",
                    "message": "Raw citation syntax was removed during normalization.",
                }
            )

        lowered_text = normalized_text.lower()
        structure_required = effective_gate == _COVERAGE_GATE_MODE or final_followup_rewrite
        if _BEARER_LIKE_PATTERN.search(normalized_text) or _JWT_LIKE_PATTERN.search(
            normalized_text
        ) or any(keyword.lower() in lowered_text for keyword in _SECRET_KEYWORDS):
            issues.append(
                {
                    "code": "sensitive_token_like_content",
                    "message": "Draft appears to contain a token, secret, or authorization string.",
                    "severity": "error",
                    "force_bypassable": False,
                }
            )

        if structure_required:
            missing_sections = []
            if not any(marker in lowered_text for marker in _FACT_MARKERS):
                missing_sections.append("facts")
            if not any(marker in lowered_text for marker in _INFERENCE_MARKERS):
                missing_sections.append("inference")
            if not any(marker in lowered_text for marker in _ASSUMPTION_MARKERS):
                missing_sections.append("assumption")
            if not any(marker in lowered_text for marker in _UNCERTAINTY_MARKERS):
                missing_sections.append("uncertainty")
            if missing_sections:
                structure_label = (
                    "Final rewrite"
                    if final_followup_rewrite and effective_gate != _COVERAGE_GATE_MODE
                    else "Coverage gate"
                )
                issues.append(
                    {
                        "code": "coverage_gate_structure_missing",
                        "message": (
                            f"{structure_label} requires explicit fact / inference / assumption / "
                            f"uncertainty handling; missing: {', '.join(missing_sections)}."
                        ),
                        "severity": "error",
                        "force_bypassable": True,
                    }
                )

        latest_hash = task_state.get(
            "last_followup_reply_hash"
            if effective_action == "reply_followup"
            else "last_submission_hash"
        )
        draft_hash = _hash_text(normalized_text) if normalized_text else ""
        if latest_hash and draft_hash and latest_hash == draft_hash:
            issues.append(
                {
                    "code": "duplicate_draft",
                    "message": "Draft matches the latest persisted submission/reply for this task.",
                    "severity": "error",
                    "force_bypassable": True,
                }
            )

        decision = "pass" if not issues else "fail"
        return {
            "task_id": task_id,
            "gate_mode": effective_gate,
            "action_kind": effective_action,
            "decision": decision,
            "ok": decision == "pass",
            "issues": issues,
            "warnings": warnings,
            "normalized_text": normalized_text,
            "draft_hash": draft_hash,
            "metrics": metrics,
            "followup_response_mode": followup_response_mode,
        }

    async def submit_file(
        self,
        task_id: str,
        file_path: str,
        gate_mode: Optional[str] = None,
        force: bool = False,
        refresh_after: bool = False,
    ) -> Dict[str, Any]:
        draft_text = Path(file_path).read_text(encoding="utf-8")
        effective_gate = gate_mode or self._suggest_gate_mode(
            self.store.get_task_state(task_id) or {},
            self.store.get_latest_snapshot("sync_state") or {},
        )
        quality = self.quality_check(
            task_id=task_id,
            draft_text=draft_text,
            gate_mode=effective_gate,
            action_kind="submit_official",
        )

        hard_fail = [
            issue for issue in quality["issues"] if not issue.get("force_bypassable", True)
        ]
        soft_fail = [
            issue for issue in quality["issues"] if issue.get("force_bypassable", True)
        ]
        if hard_fail or (soft_fail and not force):
            status = "blocked"
            result = {
                "submitted": False,
                "task_id": task_id,
                "gate_mode": effective_gate,
                "quality": quality,
                "reason": "quality_gate_failed",
            }
            self.store.append_run_log(
                "submit_file",
                result,
                created_at=_iso_now(),
                task_id=task_id,
                status=status,
            )
            return result

        response = await self.client.submit_task_answer(task_id, quality["normalized_text"])
        timestamp = _iso_now()
        self.store.mark_submission(
            task_id=task_id,
            submission_hash=quality["draft_hash"],
            preview=_preview(quality["normalized_text"]),
            submitted_at=timestamp,
        )
        refreshed = await self.sync_state() if refresh_after else None
        result = {
            "submitted": True,
            "task_id": task_id,
            "gate_mode": effective_gate,
            "quality": quality,
            "response": response,
            "state_refresh_deferred": not refresh_after,
            "next_action": refreshed.get("next_action") if refreshed else None,
        }
        self.store.append_run_log(
            "submit_file",
            result,
            created_at=timestamp,
            task_id=task_id,
            status="success",
        )
        return result

    async def reply_followup(
        self,
        task_id: str,
        file_path: str,
        comment_id: Optional[str] = None,
        post_id: Optional[str] = None,
        force: bool = False,
        refresh_after: bool = False,
    ) -> Dict[str, Any]:
        draft_text = Path(file_path).read_text(encoding="utf-8")
        context = await self.prepare_context(task_id, refresh=True)
        task_state = self.store.get_task_state(task_id) or {}
        if self._followup_reply_used(task_state):
            result = {
                "replied": False,
                "task_id": task_id,
                "reason": "followup_reply_already_used",
            }
            self.store.append_run_log(
                "reply_followup",
                result,
                created_at=_iso_now(),
                task_id=task_id,
                status="blocked",
            )
            return result
        latest_comment = context.get("latest_actionable_comment") or {}
        target_comment_id = comment_id or latest_comment.get("comment_id")
        target_post_id = (
            post_id
            or latest_comment.get("post_id")
            or context["task_metadata"].get("agora_post_id")
        )
        if not target_comment_id:
            result = {
                "replied": False,
                "task_id": task_id,
                "post_id": target_post_id,
                "reason": "missing_comment_id",
            }
            self.store.append_run_log(
                "reply_followup",
                result,
                created_at=_iso_now(),
                task_id=task_id,
                status="blocked",
            )
            return result
        if not target_post_id:
            result = {
                "replied": False,
                "task_id": task_id,
                "comment_id": target_comment_id,
                "reason": "missing_post_id",
            }
            self.store.append_run_log(
                "reply_followup",
                result,
                created_at=_iso_now(),
                task_id=task_id,
                status="blocked",
            )
            return result

        effective_gate = (
            _COVERAGE_GATE_MODE
            if context.get("followup_response_mode") == _FOLLOWUP_FINAL_REWRITE_MODE
            else _DEFAULT_GATE_MODE
        )
        quality = self.quality_check(
            task_id=task_id,
            draft_text=draft_text,
            gate_mode=effective_gate,
            action_kind="reply_followup",
        )
        hard_fail = [
            issue for issue in quality["issues"] if not issue.get("force_bypassable", True)
        ]
        soft_fail = [
            issue for issue in quality["issues"] if issue.get("force_bypassable", True)
        ]
        if hard_fail or (soft_fail and not force):
            result = {
                "replied": False,
                "task_id": task_id,
                "quality": quality,
                "reason": "quality_gate_failed",
            }
            self.store.append_run_log(
                "reply_followup",
                result,
                created_at=_iso_now(),
                task_id=task_id,
                status="blocked",
            )
            return result

        response = await self.client.agora_create_comment(
            str(target_post_id),
            quality["normalized_text"],
            parent_type="comment",
            parent_id=str(target_comment_id),
        )
        timestamp = _iso_now()
        applied_solution_packet = None
        pending_solution_packet = context.get("pending_bounty_solution_packet")
        if isinstance(pending_solution_packet, dict) and pending_solution_packet.get(
            "integration_target"
        ) == "reply_followup":
            applied_solution_packet = pending_solution_packet
        self.store.mark_followup_replied(
            task_id=task_id,
            comment_id=str(target_comment_id),
            reply_hash=quality["draft_hash"],
            updated_at=timestamp,
            reply_comment_id=_pick_first(
                response or {},
                ("comment_id", "id", "commentId"),
            ),
            reply_parent_id=_pick_first(response or {}, ("parent_id", "parentId")),
            response_mode=context.get("followup_response_mode")
            or _FOLLOWUP_FINAL_REWRITE_MODE,
            applied_solution_packet=applied_solution_packet,
        )
        refreshed = await self.sync_state() if refresh_after else None
        result = {
            "replied": True,
            "task_id": task_id,
            "post_id": target_post_id,
            "comment_id": target_comment_id,
            "gate_mode": effective_gate,
            "quality": quality,
            "response": response,
            "followup_response_mode": context.get("followup_response_mode")
            or _FOLLOWUP_FINAL_REWRITE_MODE,
            "followup_low_score_refine": bool(context.get("followup_low_score_refine")),
            "applied_solution_packet": applied_solution_packet,
            "state_refresh_deferred": not refresh_after,
            "next_action": refreshed.get("next_action") if refreshed else None,
        }
        self.store.append_run_log(
            "reply_followup",
            result,
            created_at=timestamp,
            task_id=task_id,
            status="success",
        )
        return result

    async def publish_probe_bounty(
        self,
        task_id: str,
        subproblem: str,
        research_brief: Optional[Dict[str, Any]] = None,
        evidence_packet: Optional[Dict[str, Any]] = None,
        *,
        refresh: bool = True,
        target_reply_count: int = _DEFAULT_PROBE_REPLY_TARGET,
        poll_attempts: int = _DEFAULT_PROBE_POLL_ATTEMPTS,
        poll_interval_seconds: float = _DEFAULT_PROBE_POLL_INTERVAL_SECONDS,
        trigger_context: str = _POST_COVERAGE_TRIGGER,
    ) -> Dict[str, Any]:
        sync_state = await self.sync_state() if refresh else (
            self.store.get_latest_snapshot("sync_state") or await self.sync_state()
        )
        context = await self.prepare_context(task_id, refresh=False)
        timestamp = _iso_now()
        task_state = self.store.get_task_state(task_id) or {}
        resource_snapshot = sync_state.get("resource_snapshot") or {}
        coverage = sync_state.get("official_task_coverage") or {}

        blocker_hash = self._build_blocker_hash(task_id, subproblem)
        result_base = {
            "task_id": task_id,
            "bounty_amount": _PROBE_BOUNTY_AMOUNT,
            "bounty_subproblem": subproblem.strip(),
            "blocker_hash": blocker_hash,
            "trigger_context": trigger_context,
        }

        guard = self._check_probe_bounty_guardrails(
            task_id=task_id,
            sync_state=sync_state,
            context=context,
            task_state=task_state,
            trigger_context=trigger_context,
        )
        if not guard["ok"]:
            result = {
                **result_base,
                "published": False,
                "reused_existing_bounty": False,
                "bounty_verdict": "guardrail_blocked",
                "guardrail_failures": guard["failures"],
            }
            self.store.append_run_log(
                "publish_probe_bounty",
                result,
                created_at=timestamp,
                task_id=task_id,
                status="blocked",
            )
            return result

        existing_bounty = await self._find_matching_open_probe_bounty(
            task_id=task_id,
            blocker_hash=blocker_hash,
            participant_id=str(resource_snapshot.get("participant_id") or ""),
        )

        if existing_bounty:
            bounty_record = existing_bounty
            reused_existing_bounty = True
            publish_response = None
        else:
            title, description = self._build_probe_bounty_payload(
                task_id=task_id,
                task_title=context.get("title") or "",
                subproblem=subproblem,
                blocker_hash=blocker_hash,
            )
            publish_response = await self.client.create_bounty_task(
                title=title,
                description=description,
                bounty_amount=_PROBE_BOUNTY_AMOUNT,
            )
            bounty_record = dict(publish_response or {})
            reused_existing_bounty = False

        bounty_task_id = _extract_bounty_task_id(bounty_record) or ""
        post_id = _extract_bounty_post_id(bounty_record) or bounty_task_id
        bounty_status = _extract_bounty_status(bounty_record) or "open"

        self.store.record_probe_bounty(
            task_id=task_id,
            bounty_task_id=bounty_task_id,
            blocker_hash=blocker_hash,
            subproblem=subproblem.strip(),
            updated_at=timestamp,
            post_id=post_id,
            status=bounty_status,
            published_amount=_PROBE_BOUNTY_AMOUNT,
        )

        polled_answers, poll_status = await self._poll_probe_answers(
            post_id=post_id,
            target_reply_count=target_reply_count,
            poll_attempts=poll_attempts,
            poll_interval_seconds=poll_interval_seconds,
        )
        answer_details = await self._load_answer_details(polled_answers)
        candidate_answer_matrix, wrong_answer_tags, accepted_candidate = self._judge_probe_answers(
            task_id=task_id,
            subproblem=subproblem,
            research_brief=research_brief or {},
            evidence_packet=evidence_packet or {},
            context=context,
            answers=answer_details,
        )

        visibility_status = self._compute_visibility_status(
            visible_answer_count=len(answer_details),
            target_reply_count=target_reply_count,
        )
        if visibility_status != "full":
            accepted_candidate = None

        accepted_answer_id: Optional[str] = None
        accept_response: Optional[Dict[str, Any]] = None
        bounty_solution_packet: Optional[Dict[str, Any]] = None
        if accepted_candidate is not None:
            accepted_answer_id = str(accepted_candidate.get("answer_id") or "")
            accept_response = await self.client.accept_bounty_answer(
                bounty_task_id=bounty_task_id,
                bounty_answer_id=accepted_answer_id,
            )
            bounty_solution_packet = self._build_bounty_solution_packet(
                task_id=task_id,
                subproblem=subproblem,
                blocker_hash=blocker_hash,
                accepted_candidate=accepted_candidate,
                context=context,
                research_brief=research_brief or {},
                evidence_packet=evidence_packet or {},
                trigger_context=trigger_context,
            )
            verdict = "accepted_unique_winner"
        elif poll_status == "pending_answers":
            verdict = "pending_answers"
        else:
            verdict = "no_accept_high_uncertainty"

        self.store.record_bounty_verdict(
            task_id=task_id,
            verdict=verdict,
            updated_at=timestamp,
            visibility_status=visibility_status,
            accepted_answer_id=accepted_answer_id,
            wrong_answer_tags=wrong_answer_tags,
            solution_packet=bounty_solution_packet,
        )

        result = {
            **result_base,
            "published": not reused_existing_bounty,
            "reused_existing_bounty": reused_existing_bounty,
            "publish_response": publish_response,
            "bounty_task_id": bounty_task_id,
            "post_id": post_id,
            "visibility_status": visibility_status,
            "candidate_answer_matrix": candidate_answer_matrix,
            "wrong_answer_tags": wrong_answer_tags,
            "accepted_candidate": accepted_candidate,
            "bounty_verdict": verdict,
            "bounty_solution_packet": bounty_solution_packet,
            "accept_response": accept_response,
            "poll_status": poll_status,
            "official_task_coverage": {
                "unsubmitted_count": coverage.get("unsubmitted_count"),
                "followup_pending_count": coverage.get("followup_pending_count"),
            },
        }
        self.store.append_run_log(
            "publish_probe_bounty",
            result,
            created_at=timestamp,
            task_id=task_id,
            status="success" if verdict == "accepted_unique_winner" else "info",
        )
        return result

    async def answer_bounty(
        self,
        bounty_task_id: str,
        answer_text: str,
        *,
        refresh: bool = True,
    ) -> Dict[str, Any]:
        sync_state = await self.sync_state() if refresh else (
            self.store.get_latest_snapshot("sync_state") or await self.sync_state()
        )
        coverage = sync_state.get("official_task_coverage") or {}
        eligible_high_bounties = list(sync_state.get("eligible_high_bounties") or [])
        timestamp = _iso_now()
        selected = next(
            (
                item
                for item in eligible_high_bounties
                if str(item.get("bounty_task_id") or "") == str(bounty_task_id)
            ),
            None,
        )
        if coverage.get("unsubmitted_count", 0) > 0:
            result = {
                "submitted": False,
                "bounty_task_id": bounty_task_id,
                "reason": "official_coverage_not_stable",
            }
            self.store.append_run_log(
                "answer_bounty",
                result,
                created_at=timestamp,
                status="blocked",
            )
            return result
        if coverage.get("followup_pending_count", 0) > 0 and selected is None:
            persisted = self.store.get_bounty_state(bounty_task_id) or {}
            if persisted.get("last_skip_reason") not in _HIGH_BOUNTY_SKIP_REASONS:
                self.store.mark_bounty_skipped(
                    bounty_task_id=bounty_task_id,
                    updated_at=timestamp,
                    reason="skipped_unanswerable",
                    stage_id=(sync_state.get("current_stage") or {}).get("id")
                    or (sync_state.get("current_stage") or {}).get("stage_id"),
                )
            result = {
                "submitted": False,
                "bounty_task_id": bounty_task_id,
                "reason": "followup_pending_not_high_bounty",
            }
            self.store.append_run_log(
                "answer_bounty",
                result,
                created_at=timestamp,
                status="blocked",
            )
            return result

        normalized_text = _normalize_text(answer_text)
        response = await self.client.submit_bounty_answer(
            bounty_task_id=bounty_task_id,
            text=normalized_text,
        )
        answer_hash = _hash_text(normalized_text)
        self.store.mark_bounty_answered(
            bounty_task_id=bounty_task_id,
            updated_at=timestamp,
            answer_hash=answer_hash,
            stage_id=(selected or {}).get("stage_id"),
            title=(selected or {}).get("title"),
            bounty_amount=(selected or {}).get("bounty_amount"),
            status="answered_success",
            publisher_participant_id=(selected or {}).get("publisher_participant_id"),
        )
        result = {
            "submitted": True,
            "bounty_task_id": bounty_task_id,
            "bounty_amount": (selected or {}).get("bounty_amount"),
            "agora_post_id": (selected or {}).get("agora_post_id"),
            "publisher_participant_id": (selected or {}).get("publisher_participant_id"),
            "execution_phase": sync_state.get("execution_phase"),
            "response": response,
        }
        self.store.append_run_log(
            "answer_bounty",
            result,
            created_at=timestamp,
            status="success",
        )
        return result

    def _build_task_summary(
        self,
        task: Dict[str, Any],
        my_answer: Dict[str, Any],
        participant_id: str,
        answer_error: Optional[str],
        synced_at: str,
        answer_summary: Optional[Dict[str, Any]] = None,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        task_id = str(task.get("task_id") or task.get("id") or "")
        previous = self.store.get_task_state(task_id) or {}
        answer_summary = answer_summary or _extract_answer_summary(my_answer)
        comments = comments if comments is not None else _extract_comments(
            my_answer,
            participant_id=participant_id,
        )
        followup_reply_used = self._followup_reply_used(previous)
        latest_actionable = self._latest_actionable_comment(comments, previous)
        local_submission_known = bool(previous.get("submitted")) and bool(
            previous.get("last_submission_hash")
        )
        effective_submitted = bool(answer_summary["submitted"]) or local_submission_known
        effective_answer_excerpt = (
            answer_summary["excerpt"]
            or str(previous.get("last_submission_preview") or "")
        )
        effective_answer_submitted_at = (
            answer_summary.get("submitted_at")
            or previous.get("first_submitted_at")
        )

        repeat_issue_count = int(previous.get("repeat_issue_count", 0))
        if latest_actionable:
            previous_signature = previous.get("latest_comment_signature")
            previous_excerpt = previous.get("latest_comment_excerpt")
            if latest_actionable["signature"] == previous_signature:
                repeat_issue_count = int(previous.get("repeat_issue_count", 0))
            elif latest_actionable["excerpt"] == previous_excerpt:
                repeat_issue_count = int(previous.get("repeat_issue_count", 0)) + 1
            else:
                repeat_issue_count = 0

        followup_pending = bool(latest_actionable) and not followup_reply_used
        latest_score = answer_summary["score"]
        followup_low_score_refine = _is_low_score_followup_candidate(
            score=latest_score,
            followup_pending=followup_pending,
            followup_reply_used=followup_reply_used,
        )
        followup_response_mode = (
            str(previous.get("followup_response_mode") or _FOLLOWUP_FINAL_REWRITE_MODE)
            if followup_pending or followup_reply_used
            else None
        )

        status = "seen_unsubmitted"
        if effective_submitted:
            status = "submitted_once"
        if followup_pending:
            status = "followup_pending"
        elif (
            effective_submitted
            and answer_summary["score"] is not None
            and answer_summary["score"] <= 0
        ):
            status = "improvable"
        elif previous.get("pending_bounty_solution_packet"):
            status = "improvable"

        task_with_context = dict(task)
        task_with_context["agora_post"] = task_with_context.get("agora_post") or {}
        domain = _classify_domain(task_with_context)
        action_kind = self._task_action_from_status(status)
        downstream_agent, selected_skill = self._resolve_downstream_agent(
            domain=domain,
            action_kind=action_kind,
        )
        task_scope = self._build_task_scope(
            task_id=task_id,
            stage_id=task.get("stage_id"),
            title=task.get("title") or task.get("name") or f"task:{task_id}",
            status=status,
            domain=domain,
            gate_mode=(
                _COVERAGE_GATE_MODE
                if status == "seen_unsubmitted"
                or (
                    status == "followup_pending"
                    and followup_response_mode == _FOLLOWUP_FINAL_REWRITE_MODE
                )
                else _DEFAULT_GATE_MODE
            ),
            latest_actionable_comment=latest_actionable,
            followup_response_mode=followup_response_mode,
            followup_reply_budget_remaining=self._followup_reply_budget_remaining(previous),
            followup_low_score_refine=followup_low_score_refine,
        )

        summary = {
            "task_id": task_id,
            "stage_id": task.get("stage_id"),
            "title": task.get("title") or task.get("name") or f"task:{task_id}",
            "status": status,
            "domain": domain,
            "submitted": effective_submitted,
            "first_submitted_at": previous.get("first_submitted_at")
            or effective_answer_submitted_at,
            "last_submission_hash": previous.get("last_submission_hash"),
            "last_submission_preview": previous.get("last_submission_preview"),
            "last_score": latest_score,
            "latest_comment_id": latest_actionable.get("comment_id") if latest_actionable else None,
            "latest_comment_created_at": latest_actionable.get("created_at") if latest_actionable else None,
            "latest_comment_signature": latest_actionable.get("signature") if latest_actionable else None,
            "latest_comment_excerpt": latest_actionable.get("excerpt")
            if latest_actionable
            else None,
            "last_replied_comment_id": previous.get("last_replied_comment_id"),
            "last_followup_reply_hash": previous.get("last_followup_reply_hash"),
            "last_followup_reply_comment_id": previous.get(
                "last_followup_reply_comment_id"
            ),
            "last_followup_reply_parent_id": previous.get(
                "last_followup_reply_parent_id"
            ),
            "followup_reply_used": followup_reply_used,
            "followup_reply_budget_remaining": self._followup_reply_budget_remaining(previous),
            "followup_reply_used_at": previous.get("followup_reply_used_at"),
            "followup_response_mode": followup_response_mode,
            "followup_low_score_refine": followup_low_score_refine,
            "latest_probe_bounty": previous.get("latest_probe_bounty"),
            "last_probe_bounty_task_id": previous.get("last_probe_bounty_task_id"),
            "last_probe_blocker_hash": previous.get("last_probe_blocker_hash"),
            "last_probe_subproblem": previous.get("last_probe_subproblem"),
            "last_bounty_verdict": previous.get("last_bounty_verdict"),
            "last_bounty_visibility_status": previous.get(
                "last_bounty_visibility_status"
            ),
            "last_accepted_bounty_answer_id": previous.get(
                "last_accepted_bounty_answer_id"
            ),
            "last_wrong_answer_tags": previous.get("last_wrong_answer_tags"),
            "pending_bounty_solution_packet": previous.get(
                "pending_bounty_solution_packet"
            ),
            "last_bounty_solution_packet": previous.get("last_bounty_solution_packet"),
            "last_applied_bounty_solution_packet": previous.get(
                "last_applied_bounty_solution_packet"
            ),
            "last_applied_bounty_solution_at": previous.get(
                "last_applied_bounty_solution_at"
            ),
            "repeat_issue_count": repeat_issue_count,
            "followup_pending": followup_pending,
            "updated_at": synced_at,
            "my_answer_excerpt": effective_answer_excerpt,
            "answer_id": answer_summary.get("answer_id"),
            "my_answer_error": answer_error,
            "latest_actionable_comment": latest_actionable,
            "requires_web_search": True,
            "recommended_gate_mode": _COVERAGE_GATE_MODE
            if status == "seen_unsubmitted"
            else _DEFAULT_GATE_MODE,
            "task_action_kind": action_kind,
            "task_dispatch_priority": self._task_dispatch_priority(action_kind),
            "task_agent_id": task_scope["task_agent_id"],
            "task_session_key": task_scope["task_session_key"],
            "downstream_agent": downstream_agent,
            "selected_skill": selected_skill,
            "task_output_file_path": task_scope["task_output_file_path"],
        }
        return summary

    def _build_blocker_hash(self, task_id: str, subproblem: str) -> str:
        normalized = f"{task_id}::{_normalize_text(subproblem)}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _check_probe_bounty_guardrails(
        self,
        *,
        task_id: str,
        sync_state: Dict[str, Any],
        context: Dict[str, Any],
        task_state: Dict[str, Any],
        trigger_context: str = _POST_COVERAGE_TRIGGER,
    ) -> Dict[str, Any]:
        resource_snapshot = sync_state.get("resource_snapshot") or {}
        coverage = sync_state.get("official_task_coverage") or {}
        failures: List[str] = []
        is_followup_refine = trigger_context == _FOLLOWUP_REFINE_TRIGGER

        if coverage.get("unsubmitted_count", 0) > 0 and not is_followup_refine:
            failures.append("official_coverage_incomplete")
        if coverage.get("followup_pending_count", 0) > 0 and not is_followup_refine:
            failures.append("followup_pending")
        if resource_snapshot.get("bounty_mode") != "open" and not is_followup_refine:
            failures.append("bounty_mode_not_open")
        if int(resource_snapshot.get("wallet_balance", 0) or 0) < _PROBE_BOUNTY_AMOUNT:
            failures.append("wallet_cannot_cover_probe")
        if int(resource_snapshot.get("max_bounty_amount", 0) or 0) < _PROBE_BOUNTY_AMOUNT:
            failures.append("runtime_bounty_cap_below_probe")
        if (task_state.get("followup_pending") or context.get("followup_pending")) and not is_followup_refine:
            failures.append("task_followup_pending")
        if not task_state.get("submitted"):
            failures.append("task_not_submitted_once")
        if is_followup_refine:
            if self._followup_reply_used(task_state):
                failures.append("followup_reply_already_used")
            if not context.get("followup_low_score_refine"):
                failures.append("followup_low_score_refine_not_enabled")
            if context.get("followup_response_mode") != _FOLLOWUP_FINAL_REWRITE_MODE:
                failures.append("followup_not_in_final_rewrite_mode")

        return {"ok": not failures, "failures": failures}

    def _build_probe_bounty_payload(
        self,
        *,
        task_id: str,
        task_title: str,
        subproblem: str,
        blocker_hash: str,
    ) -> tuple[str, str]:
        narrowed = _normalize_text(subproblem)
        task_label = task_title.strip() or task_id
        title = f"[probe][official_task_id={task_id}][blocker_hash={blocker_hash}] {task_label}"
        description = "\n".join(
            [
                f"official_task_id={task_id}",
                f"blocker_hash={blocker_hash}",
                "mode=publish_probe_bounty",
                "publish_amount=1",
                "",
                narrowed,
            ]
        )
        return title, description

    async def _find_matching_open_probe_bounty(
        self,
        *,
        task_id: str,
        blocker_hash: str,
        participant_id: str,
    ) -> Optional[Dict[str, Any]]:
        list_bounty_tasks = getattr(self.client, "list_bounty_tasks", None)
        if not callable(list_bounty_tasks):
            return None
        visible = await list_bounty_tasks(
            status="open",
            publisher_participant_id=participant_id or None,
        )
        for bounty in visible:
            status = _extract_bounty_status(bounty)
            if status and status != "open":
                continue
            blob = _bounty_text_blob(bounty)
            if f"official_task_id={task_id}" not in blob:
                continue
            if f"blocker_hash={blocker_hash}" not in blob:
                continue
            return bounty
        return None

    async def _poll_probe_answers(
        self,
        *,
        post_id: str,
        target_reply_count: int,
        poll_attempts: int,
        poll_interval_seconds: float,
    ) -> tuple[List[Dict[str, Any]], str]:
        latest_answers: List[Dict[str, Any]] = []
        for attempt in range(max(1, int(poll_attempts))):
            latest_answers = await self.client.agora_list_answers(
                post_id,
                limit=max(target_reply_count, 20),
                offset=0,
            )
            if len(latest_answers) >= target_reply_count:
                return latest_answers, "target_reached"
            if attempt + 1 < max(1, int(poll_attempts)) and poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)
        return latest_answers, "pending_answers"

    async def _load_answer_details(
        self,
        answers: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for answer in answers:
            answer_id = str(_pick_first(answer, ("id", "answer_id", "answerId")) or "")
            if not answer_id:
                continue
            payload = await self.client.agora_get_answer(answer_id)
            detail = payload if isinstance(payload, dict) else {}
            detail.setdefault("id", answer_id)
            details.append(detail)
        return details

    def _judge_probe_answers(
        self,
        *,
        task_id: str,
        subproblem: str,
        research_brief: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        context: Dict[str, Any],
        answers: Sequence[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[str]], Optional[Dict[str, Any]]]:
        candidate_answer_matrix: List[Dict[str, Any]] = []
        wrong_answer_tags: Dict[str, List[str]] = {}
        passing_candidates: List[Dict[str, Any]] = []

        task_rules = _collect_rule_terms(
            [research_brief, evidence_packet, context],
            (
                "constraints",
                "requirements",
                "must_include",
                "must_not",
                "checklist",
                "critical_points",
            ),
        )
        expected_keywords = _keywords_from_text(subproblem) or _keywords_from_text(
            context.get("task_content") or ""
        )

        for answer in answers:
            answer_id = str(_pick_first(answer, ("id", "answer_id", "answerId")) or "")
            text = _pick_first(answer, ("content", "text", "body", "answer_text"))
            normalized_text = _normalize_text(str(text or ""))
            lowered_text = normalized_text.lower()
            tags: List[str] = []

            if not normalized_text:
                tags.append("non_actionable")
            if expected_keywords and not any(keyword in lowered_text for keyword in expected_keywords):
                tags.append("irrelevant")
            for rule in task_rules:
                rule_text = str(rule or "").strip()
                if not rule_text:
                    continue
                rule_keywords = _keywords_from_text(rule_text, limit=4)
                if rule_keywords and not all(keyword in lowered_text for keyword in rule_keywords):
                    tags.append("constraint_miss")
                    break
            if "according to" in lowered_text or "权威" in normalized_text or "官方" in normalized_text:
                evidence_hits = 0
                evidence_blob = stable_evidence_blob(research_brief, evidence_packet)
                for keyword in _keywords_from_text(normalized_text, limit=10):
                    if keyword and keyword in evidence_blob:
                        evidence_hits += 1
                if evidence_hits == 0:
                    tags.append("hallucinated_authority")
            contradiction_terms = _collect_rule_terms(
                [research_brief, evidence_packet],
                ("contradictions", "invalid_claims", "disallowed_claims"),
            )
            for term in contradiction_terms:
                term_text = str(term or "").strip()
                if term_text and term_text.lower() in lowered_text:
                    tags.append("contradiction")
                    break
            if normalized_text and len(normalized_text) < 20:
                tags.append("partial_only")
            if normalized_text and not any(
                token in lowered_text
                for token in (
                    "because",
                    "therefore",
                    "thus",
                    "why",
                    "thereby",
                    "因此",
                    "所以",
                    "因为",
                    "步骤",
                    "step",
                    "update",
                    "counting",
                    "correctness",
                )
            ):
                tags.append("unsupported_claim")

            unique_tags: List[str] = []
            seen: set[str] = set()
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)

            quality_score = max(0, 100 - 20 * len(unique_tags))
            candidate = {
                "answer_id": answer_id,
                "quality_score": quality_score,
                "wrong_answer_tags": unique_tags,
                "is_high_confidence": not unique_tags,
                "excerpt": _preview(normalized_text, 240),
                "text": normalized_text,
                "task_id": task_id,
            }
            candidate_answer_matrix.append(candidate)
            if unique_tags:
                wrong_answer_tags[answer_id] = unique_tags
            else:
                passing_candidates.append(candidate)

        accepted_candidate: Optional[Dict[str, Any]] = None
        if len(passing_candidates) == 1:
            accepted_candidate = passing_candidates[0]
        return candidate_answer_matrix, wrong_answer_tags, accepted_candidate

    def _compute_visibility_status(
        self,
        *,
        visible_answer_count: int,
        target_reply_count: int,
    ) -> str:
        if visible_answer_count >= max(1, int(target_reply_count)):
            return "full"
        if visible_answer_count > 0:
            return "partial"
        return "insufficient"

    def _build_bounty_solution_packet(
        self,
        *,
        task_id: str,
        subproblem: str,
        blocker_hash: str,
        accepted_candidate: Dict[str, Any],
        context: Dict[str, Any],
        research_brief: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        trigger_context: str = _POST_COVERAGE_TRIGGER,
    ) -> Dict[str, Any]:
        integration_target = (
            "reply_followup"
            if trigger_context == _FOLLOWUP_REFINE_TRIGGER
            else "improve_official"
        )
        return {
            "task_id": task_id,
            "blocker_hash": blocker_hash,
            "subproblem": _normalize_text(subproblem),
            "accepted_answer_id": accepted_candidate.get("answer_id"),
            "accepted_answer_excerpt": accepted_candidate.get("excerpt"),
            "accepted_answer_text": accepted_candidate.get("text"),
            "integration_target": integration_target,
            "previous_submission_preview": (self.store.get_task_state(task_id) or {}).get(
                "last_submission_preview"
            ),
            "task_title": context.get("title"),
            "research_brief": research_brief,
            "evidence_packet": evidence_packet,
            "trigger_context": trigger_context,
        }

    def _build_coverage(
        self,
        task_summaries: List[Dict[str, Any]],
        stage: Dict[str, Any],
    ) -> Dict[str, Any]:
        counts = {
            "unseen": 0,
            "seen_unsubmitted": 0,
            "submitted_once": 0,
            "followup_pending": 0,
            "improvable": 0,
            "closed": 0,
        }
        for task in task_summaries:
            status = task["status"]
            counts[status] = counts.get(status, 0) + 1

        return {
            "total_visible_tasks": len(task_summaries),
            "unsubmitted_count": counts.get("seen_unsubmitted", 0),
            "submitted_once_count": counts.get("submitted_once", 0),
            "followup_pending_count": counts.get("followup_pending", 0),
            "coverage_deadline_mode": _deadline_mode(stage),
            "states": counts,
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "title": task["title"],
                    "status": task["status"],
                    "domain": task.get("domain"),
                }
                for task in task_summaries
            ],
        }

    def _build_resource_snapshot(
        self,
        me: Dict[str, Any],
        leaderboard: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        synced_at: str,
        *,
        eligible_high_bounties: Sequence[Dict[str, Any]],
        visible_bounty_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        wallet_balance = int(me.get("wallet_balance", 0) or 0)
        token_used = int(me.get("token_used", 0) or 0)
        token_remaining = max(0, self.config.free_token_budget - token_used)
        if token_remaining > self.config.token_tight_threshold:
            budget_mode = "normal"
        elif token_remaining > self.config.token_emergency_threshold:
            budget_mode = "tight"
        else:
            budget_mode = "emergency"

        if wallet_balance <= self.config.wallet_floor_hard:
            wallet_mode = "hard_stop"
        elif wallet_balance <= self.config.wallet_floor_soft:
            wallet_mode = "soft_stop"
        else:
            wallet_mode = "open"

        spendable_wallet = max(0, wallet_balance - self.config.wallet_floor_hard)
        max_bounty_amount = min(1200, int(spendable_wallet * 0.12)) if spendable_wallet else 0

        previous_snapshot = self.store.get_latest_snapshot("resource_snapshot") or {}
        previous_synced_at = _parse_datetime(previous_snapshot.get("synced_at"))
        current_synced_at = _parse_datetime(synced_at)
        estimated_token_burn_today = token_used
        if previous_synced_at and current_synced_at and previous_synced_at.date() == current_synced_at.date():
            estimated_token_burn_today = max(
                0,
                token_used - int(previous_snapshot.get("token_used", 0) or 0),
            )

        participant_id = str(me.get("participant_id") or "")
        leaderboard_rank = None
        for index, row in enumerate(leaderboard, start=1):
            if str(row.get("participant_id") or "") == participant_id:
                leaderboard_rank = index
                break

        bounty_mode = (
            "locked"
            if coverage["unsubmitted_count"] > 0 or coverage["followup_pending_count"] > 0
            else "open"
        )
        return {
            "synced_at": synced_at,
            "participant_id": participant_id,
            "wallet_balance": wallet_balance,
            "wallet_mode": wallet_mode,
            "spendable_wallet": spendable_wallet,
            "max_bounty_amount": max_bounty_amount,
            "token_used": token_used,
            "token_remaining": token_remaining,
            "estimated_token_burn_today": estimated_token_burn_today,
            "budget_mode": budget_mode,
            "stage_progress": {
                "total_visible_tasks": coverage["total_visible_tasks"],
                "unsubmitted_count": coverage["unsubmitted_count"],
                "submitted_once_count": coverage["submitted_once_count"],
                "followup_pending_count": coverage["followup_pending_count"],
            },
            "open_followups": coverage["followup_pending_count"],
            "coverage_deadline_mode": coverage["coverage_deadline_mode"],
            "leaderboard_rank": leaderboard_rank,
            "leaderboard_size": len(leaderboard),
            "bounty_mode": bounty_mode,
            "high_bounty_threshold": _HIGH_BOUNTY_THRESHOLD,
            "eligible_high_bounty_count": len(eligible_high_bounties),
            "top_high_bounty": eligible_high_bounties[0] if eligible_high_bounties else None,
            "high_bounty_answer_mode": visible_bounty_summary.get("high_bounty_answer_mode")
            or "no_eligible_high_bounty",
        }

    def _choose_next_action(
        self,
        task_summaries: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        resource_snapshot: Dict[str, Any],
        *,
        eligible_high_bounties: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        phase_snapshot = self._phase_snapshot(
            coverage,
            eligible_high_bounties=eligible_high_bounties,
        )
        execution_phase = phase_snapshot["execution_phase"]
        unsubmitted_tasks = [
            task for task in task_summaries if task["status"] == "seen_unsubmitted"
        ]
        followup_tasks = [
            task for task in task_summaries if task["status"] == "followup_pending"
        ]
        if execution_phase == "first_pass_official" and unsubmitted_tasks:
            unsubmitted_tasks.sort(key=lambda item: item["task_id"])
            task = unsubmitted_tasks[0]
            return {
                "kind": "submit_official",
                "task_id": task["task_id"],
                "title": task["title"],
                "priority": 1,
                "reason": "visible official task still needs a first submission before follow-up sweep begins",
                "domain": task.get("domain"),
                "gate_mode": _COVERAGE_GATE_MODE,
                "requires_web_search": True,
                "comment_id": None,
                "post_id": None,
                "bounty_locked": True,
                "budget_mode": resource_snapshot["budget_mode"],
                **phase_snapshot,
            }

        if execution_phase == "high_bounty_sweep" and eligible_high_bounties:
            bounty = eligible_high_bounties[0]
            return {
                "kind": "answer_bounty",
                "task_id": None,
                "bounty_task_id": bounty.get("bounty_task_id"),
                "title": bounty.get("title"),
                "priority": 2,
                "reason": "highest eligible high bounty should be answered before follow-up sweep",
                "domain": None,
                "gate_mode": _DEFAULT_GATE_MODE,
                "requires_web_search": False,
                "comment_id": None,
                "post_id": bounty.get("agora_post_id"),
                "bounty_locked": False,
                "budget_mode": resource_snapshot["budget_mode"],
                "bounty_amount": bounty.get("bounty_amount"),
                "agora_post_id": bounty.get("agora_post_id"),
                "publisher_participant_id": bounty.get("publisher_participant_id"),
                "created_at": bounty.get("created_at"),
                **phase_snapshot,
            }

        if execution_phase == "followup_sweep" and followup_tasks:
            followup_tasks.sort(
                key=lambda item: _parse_datetime(
                    item.get("latest_actionable_comment", {}).get("created_at")
                )
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            task = followup_tasks[0]
            latest_comment = task.get("latest_actionable_comment") or {}
            followup_low_score_refine = bool(task.get("followup_low_score_refine"))
            return {
                "kind": "reply_followup",
                "task_id": task["task_id"],
                "title": task["title"],
                "priority": 1,
                "reason": "latest actionable comment has not been replied to yet",
                "domain": task.get("domain"),
                "gate_mode": _DEFAULT_GATE_MODE,
                "requires_web_search": False,
                "comment_id": latest_comment.get("comment_id"),
                "post_id": latest_comment.get("post_id"),
                "bounty_locked": not followup_low_score_refine,
                "budget_mode": resource_snapshot["budget_mode"],
                "followup_reply_used": bool(task.get("followup_reply_used")),
                "followup_reply_budget_remaining": int(
                    task.get("followup_reply_budget_remaining", 0)
                ),
                "followup_response_mode": task.get("followup_response_mode")
                or _FOLLOWUP_FINAL_REWRITE_MODE,
                "followup_low_score_refine": followup_low_score_refine,
                **phase_snapshot,
            }

        improvable_tasks = [
            task for task in task_summaries if task["status"] == "improvable"
        ]
        if execution_phase == "post_coverage" and improvable_tasks:
            task = improvable_tasks[0]
            return {
                "kind": "improve_official",
                "task_id": task["task_id"],
                "title": task["title"],
                "priority": 3,
                "reason": "submitted task has low score or repeated issues worth revisiting",
                "domain": task.get("domain"),
                "gate_mode": _DEFAULT_GATE_MODE,
                "requires_web_search": True,
                "comment_id": None,
                "post_id": None,
                "bounty_locked": resource_snapshot["bounty_mode"] == "locked",
                "budget_mode": resource_snapshot["budget_mode"],
                **phase_snapshot,
            }

        if coverage["total_visible_tasks"] == 0:
            return {
                **self._idle_action(**phase_snapshot),
                "reason": "no visible official tasks were returned by the current stage",
            }

        return self._idle_action(**phase_snapshot)

    def _task_content(self, task: Dict[str, Any]) -> str:
        for key in ("content", "text", "description", "body", "summary"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        agora_post = task.get("agora_post") or {}
        if isinstance(agora_post, dict):
            for key in ("content", "text", "body", "summary"):
                value = agora_post.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _latest_actionable_comment(
        self,
        comments: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        latest: Optional[Dict[str, Any]] = None
        for comment in reversed(comments):
            if comment["authored_by_me"] or _comment_matches_latest_reply(
                comment, previous_state
            ):
                continue
            latest = comment
            break
        if not latest:
            return None
        if self._followup_reply_used(previous_state):
            return None
        if latest["comment_id"] == previous_state.get("last_replied_comment_id"):
            return None
        return {
            "comment_id": latest["comment_id"],
            "post_id": latest.get("post_id"),
            "parent_id": latest.get("parent_id"),
            "parent_type": latest.get("parent_type"),
            "created_at": latest.get("created_at"),
            "excerpt": _preview(latest["content"], 240),
            "content": latest["content"],
            "signature": latest["signature"],
            "content_hash": latest.get("content_hash"),
            "already_replied": False,
        }

    def _suggest_gate_mode(
        self,
        task_state: Dict[str, Any],
        sync_state: Dict[str, Any],
    ) -> str:
        if task_state.get("status") == "seen_unsubmitted":
            return _COVERAGE_GATE_MODE
        coverage = sync_state.get("official_task_coverage") or {}
        if coverage.get("unsubmitted_count", 0) > 0:
            return _COVERAGE_GATE_MODE
        if coverage.get("coverage_deadline_mode") == "urgent":
            return _COVERAGE_GATE_MODE
        return _DEFAULT_GATE_MODE

    def _quality_expectations(self, domain: str, gate_mode: str) -> List[str]:
        expectations = [
            "Answer the prompt directly.",
            "Do not include raw web citation syntax.",
            "Keep claims and uncertainty explicit.",
        ]
        if gate_mode == _COVERAGE_GATE_MODE:
            expectations.append(
                "Separate fact, inference, assumption, and uncertainty explicitly."
            )
        if domain in {"legal", "medical"}:
            expectations.append("Prefer evidence-backed reasoning over stylistic polish.")
        if domain == "industrial":
            expectations.append("Validate technical steps and avoid unnecessary exposition.")
        return expectations

    def _idle_action(
        self,
        *,
        execution_phase: str = "post_coverage",
        followup_sweep_ready: bool = False,
        deferred_followup_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "kind": "idle",
            "task_id": None,
            "bounty_task_id": None,
            "title": None,
            "priority": 99,
            "reason": "official coverage is stable and no high bounty or follow-up is currently pending",
            "domain": None,
            "gate_mode": _DEFAULT_GATE_MODE,
            "requires_web_search": False,
            "comment_id": None,
            "post_id": None,
            "bounty_locked": False,
            "budget_mode": "normal",
            "execution_phase": execution_phase,
            "followup_sweep_ready": bool(followup_sweep_ready),
            "deferred_followup_count": int(deferred_followup_count or 0),
        }
