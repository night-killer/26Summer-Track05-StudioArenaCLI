"""SQLite persistence for Studio Arena automation state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArenaStateStore:
    """Persist sync snapshots, task state, and run logs in SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_kind_created
                ON snapshots(kind, created_at DESC);

                CREATE TABLE IF NOT EXISTS task_state (
                    task_id TEXT PRIMARY KEY,
                    stage_id TEXT,
                    title TEXT,
                    status TEXT NOT NULL,
                    domain TEXT,
                    submitted INTEGER NOT NULL DEFAULT 0,
                    first_submitted_at TEXT,
                    last_submission_hash TEXT,
                    last_submission_preview TEXT,
                    last_score REAL,
                    latest_comment_id TEXT,
                    latest_comment_created_at TEXT,
                    latest_comment_signature TEXT,
                    latest_comment_excerpt TEXT,
                    last_replied_comment_id TEXT,
                    last_followup_reply_hash TEXT,
                    repeat_issue_count INTEGER NOT NULL DEFAULT 0,
                    followup_pending INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_id TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bounty_state (
                    bounty_task_id TEXT PRIMARY KEY,
                    stage_id TEXT,
                    title TEXT,
                    bounty_amount INTEGER,
                    status TEXT NOT NULL DEFAULT 'open',
                    publisher_participant_id TEXT,
                    last_answer_hash TEXT,
                    last_answered_at TEXT,
                    last_skip_reason TEXT,
                    last_skip_at TEXT,
                    updated_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_log_created
                ON run_log(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_bounty_state_updated
                ON bounty_state(updated_at DESC, bounty_task_id ASC);
                """
            )

    def save_snapshot(self, kind: str, payload: Dict[str, Any], created_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO snapshots (kind, created_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (kind, created_at, json.dumps(payload, ensure_ascii=False)),
            )

    def get_latest_snapshot(self, kind: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM snapshots
                WHERE kind = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    def upsert_task_state(self, task_state: Dict[str, Any]) -> None:
        payload = dict(task_state)
        previous = self.get_task_state(payload["task_id"]) or {}
        for key in (
            "last_followup_reply_comment_id",
            "last_followup_reply_parent_id",
            "followup_reply_used",
            "followup_reply_budget_remaining",
            "followup_reply_used_at",
            "followup_response_mode",
            "followup_low_score_refine",
            "latest_probe_bounty",
            "last_probe_bounty_task_id",
            "last_probe_blocker_hash",
            "last_probe_subproblem",
            "last_bounty_verdict",
            "last_bounty_visibility_status",
            "last_accepted_bounty_answer_id",
            "last_wrong_answer_tags",
            "pending_bounty_solution_packet",
            "last_bounty_solution_packet",
            "last_applied_bounty_solution_packet",
            "last_applied_bounty_solution_at",
        ):
            if key not in payload and key in previous:
                payload[key] = previous.get(key)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_state (
                    task_id,
                    stage_id,
                    title,
                    status,
                    domain,
                    submitted,
                    first_submitted_at,
                    last_submission_hash,
                    last_submission_preview,
                    last_score,
                    latest_comment_id,
                    latest_comment_created_at,
                    latest_comment_signature,
                    latest_comment_excerpt,
                    last_replied_comment_id,
                    last_followup_reply_hash,
                    repeat_issue_count,
                    followup_pending,
                    updated_at,
                    raw_json
                ) VALUES (
                    :task_id,
                    :stage_id,
                    :title,
                    :status,
                    :domain,
                    :submitted,
                    :first_submitted_at,
                    :last_submission_hash,
                    :last_submission_preview,
                    :last_score,
                    :latest_comment_id,
                    :latest_comment_created_at,
                    :latest_comment_signature,
                    :latest_comment_excerpt,
                    :last_replied_comment_id,
                    :last_followup_reply_hash,
                    :repeat_issue_count,
                    :followup_pending,
                    :updated_at,
                    :raw_json
                )
                ON CONFLICT(task_id) DO UPDATE SET
                    stage_id = excluded.stage_id,
                    title = excluded.title,
                    status = excluded.status,
                    domain = excluded.domain,
                    submitted = excluded.submitted,
                    first_submitted_at = COALESCE(task_state.first_submitted_at, excluded.first_submitted_at),
                    last_submission_hash = COALESCE(excluded.last_submission_hash, task_state.last_submission_hash),
                    last_submission_preview = COALESCE(excluded.last_submission_preview, task_state.last_submission_preview),
                    last_score = excluded.last_score,
                    latest_comment_id = excluded.latest_comment_id,
                    latest_comment_created_at = excluded.latest_comment_created_at,
                    latest_comment_signature = excluded.latest_comment_signature,
                    latest_comment_excerpt = excluded.latest_comment_excerpt,
                    last_replied_comment_id = COALESCE(excluded.last_replied_comment_id, task_state.last_replied_comment_id),
                    last_followup_reply_hash = COALESCE(excluded.last_followup_reply_hash, task_state.last_followup_reply_hash),
                    repeat_issue_count = excluded.repeat_issue_count,
                    followup_pending = excluded.followup_pending,
                    updated_at = excluded.updated_at,
                    raw_json = excluded.raw_json
                """,
                {
                    "task_id": payload["task_id"],
                    "stage_id": payload.get("stage_id"),
                    "title": payload.get("title"),
                    "status": payload["status"],
                    "domain": payload.get("domain"),
                    "submitted": int(bool(payload.get("submitted"))),
                    "first_submitted_at": payload.get("first_submitted_at"),
                    "last_submission_hash": payload.get("last_submission_hash"),
                    "last_submission_preview": payload.get("last_submission_preview"),
                    "last_score": payload.get("last_score"),
                    "latest_comment_id": payload.get("latest_comment_id"),
                    "latest_comment_created_at": payload.get("latest_comment_created_at"),
                    "latest_comment_signature": payload.get("latest_comment_signature"),
                    "latest_comment_excerpt": payload.get("latest_comment_excerpt"),
                    "last_replied_comment_id": payload.get("last_replied_comment_id"),
                    "last_followup_reply_hash": payload.get("last_followup_reply_hash"),
                    "repeat_issue_count": int(payload.get("repeat_issue_count", 0)),
                    "followup_pending": int(bool(payload.get("followup_pending"))),
                    "updated_at": payload["updated_at"],
                    "raw_json": json.dumps(payload, ensure_ascii=False),
                },
            )

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT raw_json
                FROM task_state
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["raw_json"])

    def list_task_states(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT raw_json
                FROM task_state
                ORDER BY updated_at DESC, task_id ASC
                """
            ).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def upsert_bounty_state(self, bounty_state: Dict[str, Any]) -> None:
        payload = dict(bounty_state)
        previous = self.get_bounty_state(payload["bounty_task_id"]) or {}
        for key in (
            "last_answer_hash",
            "last_answered_at",
            "last_skip_reason",
            "last_skip_at",
        ):
            if key not in payload and key in previous:
                payload[key] = previous.get(key)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bounty_state (
                    bounty_task_id,
                    stage_id,
                    title,
                    bounty_amount,
                    status,
                    publisher_participant_id,
                    last_answer_hash,
                    last_answered_at,
                    last_skip_reason,
                    last_skip_at,
                    updated_at,
                    raw_json
                ) VALUES (
                    :bounty_task_id,
                    :stage_id,
                    :title,
                    :bounty_amount,
                    :status,
                    :publisher_participant_id,
                    :last_answer_hash,
                    :last_answered_at,
                    :last_skip_reason,
                    :last_skip_at,
                    :updated_at,
                    :raw_json
                )
                ON CONFLICT(bounty_task_id) DO UPDATE SET
                    stage_id = excluded.stage_id,
                    title = excluded.title,
                    bounty_amount = excluded.bounty_amount,
                    status = excluded.status,
                    publisher_participant_id = excluded.publisher_participant_id,
                    last_answer_hash = COALESCE(excluded.last_answer_hash, bounty_state.last_answer_hash),
                    last_answered_at = COALESCE(excluded.last_answered_at, bounty_state.last_answered_at),
                    last_skip_reason = COALESCE(excluded.last_skip_reason, bounty_state.last_skip_reason),
                    last_skip_at = COALESCE(excluded.last_skip_at, bounty_state.last_skip_at),
                    updated_at = excluded.updated_at,
                    raw_json = excluded.raw_json
                """,
                {
                    "bounty_task_id": payload["bounty_task_id"],
                    "stage_id": payload.get("stage_id"),
                    "title": payload.get("title"),
                    "bounty_amount": payload.get("bounty_amount"),
                    "status": payload.get("status") or "open",
                    "publisher_participant_id": payload.get("publisher_participant_id"),
                    "last_answer_hash": payload.get("last_answer_hash"),
                    "last_answered_at": payload.get("last_answered_at"),
                    "last_skip_reason": payload.get("last_skip_reason"),
                    "last_skip_at": payload.get("last_skip_at"),
                    "updated_at": payload["updated_at"],
                    "raw_json": json.dumps(payload, ensure_ascii=False),
                },
            )

    def get_bounty_state(self, bounty_task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT raw_json
                FROM bounty_state
                WHERE bounty_task_id = ?
                """,
                (bounty_task_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["raw_json"])

    def list_bounty_states(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT raw_json
                FROM bounty_state
                ORDER BY updated_at DESC, bounty_task_id ASC
                """
            ).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def mark_bounty_answered(
        self,
        *,
        bounty_task_id: str,
        updated_at: str,
        answer_hash: str,
        stage_id: Optional[str] = None,
        title: Optional[str] = None,
        bounty_amount: Optional[int] = None,
        status: str = "answered_success",
        publisher_participant_id: Optional[str] = None,
    ) -> None:
        current = self.get_bounty_state(bounty_task_id) or {
            "bounty_task_id": bounty_task_id,
            "status": status,
            "updated_at": updated_at,
        }
        current["stage_id"] = stage_id or current.get("stage_id")
        current["title"] = title or current.get("title")
        current["bounty_amount"] = (
            int(bounty_amount) if bounty_amount is not None else current.get("bounty_amount")
        )
        current["status"] = status
        current["publisher_participant_id"] = (
            publisher_participant_id or current.get("publisher_participant_id")
        )
        current["last_answer_hash"] = answer_hash
        current["last_answered_at"] = updated_at
        current["updated_at"] = updated_at
        self.upsert_bounty_state(current)

    def mark_bounty_skipped(
        self,
        *,
        bounty_task_id: str,
        updated_at: str,
        reason: str,
        stage_id: Optional[str] = None,
        title: Optional[str] = None,
        bounty_amount: Optional[int] = None,
        publisher_participant_id: Optional[str] = None,
    ) -> None:
        current = self.get_bounty_state(bounty_task_id) or {
            "bounty_task_id": bounty_task_id,
            "status": "open",
            "updated_at": updated_at,
        }
        current["stage_id"] = stage_id or current.get("stage_id")
        current["title"] = title or current.get("title")
        current["bounty_amount"] = (
            int(bounty_amount) if bounty_amount is not None else current.get("bounty_amount")
        )
        current["publisher_participant_id"] = (
            publisher_participant_id or current.get("publisher_participant_id")
        )
        current["last_skip_reason"] = reason
        current["last_skip_at"] = updated_at
        current["updated_at"] = updated_at
        self.upsert_bounty_state(current)

    def mark_submission(
        self,
        task_id: str,
        submission_hash: str,
        preview: str,
        submitted_at: str,
    ) -> None:
        current = self.get_task_state(task_id) or {
            "task_id": task_id,
            "status": "submitted_once",
            "submitted": True,
            "updated_at": submitted_at,
        }
        current["submitted"] = True
        current["status"] = "submitted_once"
        current["first_submitted_at"] = current.get("first_submitted_at") or submitted_at
        current["last_submission_hash"] = submission_hash
        current["last_submission_preview"] = preview
        if current.get("pending_bounty_solution_packet"):
            current["last_applied_bounty_solution_packet"] = current.get(
                "pending_bounty_solution_packet"
            )
            current["last_applied_bounty_solution_at"] = submitted_at
        current["pending_bounty_solution_packet"] = None
        current["updated_at"] = submitted_at
        self.upsert_task_state(current)

    def mark_followup_replied(
        self,
        task_id: str,
        comment_id: str,
        reply_hash: str,
        updated_at: str,
        reply_comment_id: Optional[str] = None,
        reply_parent_id: Optional[str] = None,
        response_mode: str = "final_rewrite",
        applied_solution_packet: Optional[Dict[str, Any]] = None,
    ) -> None:
        current = self.get_task_state(task_id) or {
            "task_id": task_id,
            "status": "submitted_once",
            "submitted": True,
            "updated_at": updated_at,
        }
        current["last_replied_comment_id"] = comment_id
        current["last_followup_reply_hash"] = reply_hash
        current["last_followup_reply_comment_id"] = reply_comment_id
        current["last_followup_reply_parent_id"] = reply_parent_id
        current["followup_reply_used"] = True
        current["followup_reply_budget_remaining"] = 0
        current["followup_reply_used_at"] = updated_at
        current["followup_response_mode"] = response_mode
        current["followup_low_score_refine"] = False
        current["followup_pending"] = False
        current["status"] = "submitted_once"
        if applied_solution_packet:
            current["last_applied_bounty_solution_packet"] = applied_solution_packet
            current["last_applied_bounty_solution_at"] = updated_at
            current["pending_bounty_solution_packet"] = None
        current["updated_at"] = updated_at
        self.upsert_task_state(current)

    def record_probe_bounty(
        self,
        task_id: str,
        bounty_task_id: str,
        blocker_hash: str,
        subproblem: str,
        updated_at: str,
        post_id: Optional[str] = None,
        status: str = "open",
        published_amount: int = 1,
    ) -> None:
        current = self.get_task_state(task_id) or {
            "task_id": task_id,
            "status": "submitted_once",
            "submitted": True,
            "updated_at": updated_at,
        }
        current["latest_probe_bounty"] = {
            "bounty_task_id": bounty_task_id,
            "post_id": post_id,
            "blocker_hash": blocker_hash,
            "subproblem": subproblem,
            "status": status,
            "published_amount": int(published_amount),
            "updated_at": updated_at,
        }
        current["last_probe_bounty_task_id"] = bounty_task_id
        current["last_probe_blocker_hash"] = blocker_hash
        current["last_probe_subproblem"] = subproblem
        current["updated_at"] = updated_at
        self.upsert_task_state(current)

    def record_bounty_verdict(
        self,
        task_id: str,
        verdict: str,
        updated_at: str,
        *,
        visibility_status: Optional[str] = None,
        accepted_answer_id: Optional[str] = None,
        wrong_answer_tags: Optional[Dict[str, Any]] = None,
        solution_packet: Optional[Dict[str, Any]] = None,
    ) -> None:
        current = self.get_task_state(task_id) or {
            "task_id": task_id,
            "status": "submitted_once",
            "submitted": True,
            "updated_at": updated_at,
        }
        current["last_bounty_verdict"] = verdict
        current["last_bounty_visibility_status"] = visibility_status
        current["last_accepted_bounty_answer_id"] = accepted_answer_id
        current["last_wrong_answer_tags"] = wrong_answer_tags or {}
        current["pending_bounty_solution_packet"] = solution_packet
        if solution_packet:
            current["last_bounty_solution_packet"] = solution_packet
            if current.get("submitted") and not current.get("followup_pending"):
                current["status"] = "improvable"
        current["updated_at"] = updated_at
        self.upsert_task_state(current)

    def append_run_log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        created_at: str,
        task_id: str | None = None,
        status: str = "info",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_log (created_at, event_type, status, task_id, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    event_type,
                    status,
                    task_id,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def recent_run_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, event_type, status, task_id, payload_json
                FROM run_log
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            results.append(
                {
                    "created_at": row["created_at"],
                    "event_type": row["event_type"],
                    "status": row["status"],
                    "task_id": row["task_id"],
                    "payload": payload,
                }
            )
        return results
