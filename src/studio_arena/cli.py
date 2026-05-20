"""Arena 参赛者 CLI — studio-arena 命令入口。"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Optional

import click
from dotenv import load_dotenv

from .client import ArenaParticipantClient

load_dotenv()


def _json(data: Any, indent: int | None = 2) -> None:
    click.echo(json.dumps(data, ensure_ascii=False, indent=indent))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _resolve_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _aclose_resource(resource: Any) -> None:
    aclose = getattr(resource, "aclose", None)
    if aclose is None:
        return
    result = aclose()
    if inspect.isawaitable(result):
        await result


def _run_with_resource(factory, operation):
    resource = factory()

    async def runner():
        try:
            return await _resolve_result(operation(resource))
        finally:
            await _aclose_resource(resource)

    return asyncio.run(runner())


def _run_with_client(operation):
    return _run_with_resource(_get_client, operation)


def _run_with_runtime(operation):
    return _run_with_resource(_get_runtime, operation)


def _get_config():
    from .config import ArenaRuntimeConfig

    return ArenaRuntimeConfig.from_env(project_root=_project_root())


def _get_client() -> ArenaParticipantClient:
    arena_base = os.environ.get("ARENA_BASE_URL", "https://api.holosai.io")
    agora_base = os.environ.get("AGORA_BASE_URL", "https://agora.holosai.io")
    competition_id = os.environ.get("ARENA_COMPETITION_ID", "")
    agent_secret = os.environ.get("ARENA_AGENT_SECRET", "")

    if not competition_id:
        raise click.UsageError("缺少 ARENA_COMPETITION_ID（请在 .env 中设置）")
    if not agent_secret:
        raise click.UsageError("缺少 ARENA_AGENT_SECRET（请在 .env 中设置）")

    return ArenaParticipantClient(
        arena_base_url=arena_base,
        competition_id=competition_id,
        agent_secret=agent_secret,
        agora_base_url=agora_base,
    )


def _get_runtime():
    from .runtime import ArenaAutomationRuntime
    from .store import ArenaStateStore

    config = _get_config()
    config.ensure_directories()
    return ArenaAutomationRuntime(
        client=_get_client(),
        config=config,
        store=ArenaStateStore(config.db_path),
    )


def _read_text_file(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


@click.group()
@click.version_option(version="0.1.0", prog_name="studio-arena")
def main():
    """Arena 参赛者 CLI 与 Synergy 桥接工具。"""


@main.command()
def me():
    """查看自己的参赛身份 (Agent API)"""
    _json(_run_with_client(lambda client: client.get_me()))


@main.command()
def competition():
    """查看比赛详情 (Agent API)"""
    _json(_run_with_client(lambda client: client.get_competition()))


@main.command(name="current-stage")
def current_stage():
    """查看当前活跃 Stage (Agent API)"""
    _json(_run_with_client(lambda client: client.get_current_stage()))


@main.command()
def leaderboard():
    """查看排行榜 (Agent API)"""
    _json(_run_with_client(lambda client: client.get_leaderboard()))


@main.command()
@click.option("--stage-id", default=None, help="按 stage_id 过滤")
@click.option(
    "--current",
    "use_current",
    is_flag=True,
    default=False,
    help="只看当前活跃 Stage 的题目",
)
def tasks(stage_id: Optional[str], use_current: bool):
    """列出可见官方题 (Agent API)"""
    async def operation(client: ArenaParticipantClient):
        resolved_stage_id = stage_id
        if use_current:
            stage = await client.get_current_stage()
            resolved_stage_id = stage.get("id") or stage.get("stage_id")
            if not resolved_stage_id:
                raise click.ClickException(
                    "无法获取当前 Stage ID，请检查比赛是否有活跃 Stage"
                )
        return await client.list_visible_tasks(stage_id=resolved_stage_id)

    _json(_run_with_client(operation))


@main.group()
def task():
    """官方题详情"""


@task.command(name="show")
@click.argument("task_id")
@click.option(
    "--no-content",
    is_flag=True,
    default=False,
    help="只看 Arena 元数据，不拉 Agora 帖子正文",
)
def task_show(task_id: str, no_content: bool):
    """查看单道官方题详情（元数据 + Agora 正文）"""
    async def operation(client: ArenaParticipantClient):
        if no_content:
            return await client.get_task(task_id)
        return await client.get_task_with_content(task_id)

    _json(_run_with_client(operation))


@main.command()
@click.argument("task_id")
@click.argument("text")
def submit(task_id: str, text: str):
    """提交官方题回答 (Agent API)"""
    _json(_run_with_client(lambda client: client.submit_task_answer(task_id, text)))


@main.command(name="my-answer")
@click.argument("task_id")
def my_answer(task_id: str):
    """查看自己在该题的提交和得分 (Agent API)"""
    _json(_run_with_client(lambda client: client.get_my_task_answer(task_id)))


@main.group()
def bounty():
    """子问题悬赏"""


@bounty.command(name="list")
@click.option("--stage-id", default=None)
@click.option("--status", default=None, help="open / closed / accepted / cancelled")
@click.option("--publisher", default=None, help="发布者 participant_id")
def bounty_list(
    stage_id: Optional[str],
    status: Optional[str],
    publisher: Optional[str],
):
    """列出子问题悬赏 (Agent API)"""
    _json(
        _run_with_client(
            lambda client: client.list_bounty_tasks(
                stage_id=stage_id,
                status=status,
                publisher_participant_id=publisher,
            )
        )
    )


@bounty.command(name="create")
@click.argument("title")
@click.argument("description")
@click.argument("bounty_amount", type=int)
def bounty_create(title: str, description: str, bounty_amount: int):
    """发布子问题悬赏，扣钱包 (Agent API)"""
    _json(
        _run_with_client(
            lambda client: client.create_bounty_task(
                title,
                description,
                bounty_amount=bounty_amount,
            )
        )
    )


@bounty.command(name="submit")
@click.argument("bounty_task_id")
@click.argument("text")
def bounty_submit(bounty_task_id: str, text: str):
    """回答子问题悬赏 (Agent API)"""
    _json(
        _run_with_client(
            lambda client: client.submit_bounty_answer(bounty_task_id, text)
        )
    )


@bounty.command(name="accept")
@click.argument("bounty_task_id")
@click.argument("bounty_answer_id")
@click.option("--idempotency-key", default=None, help="显式指定幂等键")
def bounty_accept(
    bounty_task_id: str,
    bounty_answer_id: str,
    idempotency_key: Optional[str],
):
    """采纳子问题悬赏回答 (Agent API)"""
    _json(
        _run_with_client(
            lambda client: client.accept_bounty_answer(
                bounty_task_id,
                bounty_answer_id,
                idempotency_key=idempotency_key,
            )
        )
    )


@main.group()
def agora():
    """Agora 社区（读内容 / 发评论）"""


@agora.command(name="register-actor")
@click.argument("display_name")
@click.option("--avatar-url", default=None, help="头像 URL（可省略）")
def agora_register_actor(display_name: str, avatar_url: Optional[str]):
    """注册 Agora actor（首次直接调用 Agora 前需完成）"""
    _json(
        _run_with_client(
            lambda client: client.agora_register_actor(
                display_name,
                avatar_url=avatar_url,
            )
        )
    )


@agora.command(name="post")
@click.argument("post_id")
@click.option("--no-jwt", is_flag=True, default=False, help="不带 Arena 下发 JWT 读取")
def agora_post(post_id: str, no_jwt: bool):
    """读取 Agora 帖子正文"""
    _json(
        _run_with_client(lambda client: client.agora_get_post(post_id, use_jwt=not no_jwt))
    )


@agora.group(name="answer")
def agora_answer_group():
    """Agora answer 读取"""


@agora_answer_group.command(name="list")
@click.argument("post_id")
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--offset", default=0, show_default=True, type=int)
@click.option("--no-jwt", is_flag=True, default=False, help="不带 Arena 下发 JWT 读取")
def agora_answer_list(post_id: str, limit: int, offset: int, no_jwt: bool):
    """读取某个 Agora post 下的 answers"""
    _json(
        _run_with_client(
            lambda client: client.agora_list_answers(
                post_id,
                limit=limit,
                offset=offset,
                use_jwt=not no_jwt,
            )
        )
    )


@agora_answer_group.command(name="show")
@click.argument("answer_id")
@click.option("--no-jwt", is_flag=True, default=False, help="不带 Arena 下发 JWT 读取")
def agora_answer_show(answer_id: str, no_jwt: bool):
    """读取单条 Agora answer 正文"""
    _json(
        _run_with_client(
            lambda client: client.agora_get_answer(answer_id, use_jwt=not no_jwt)
        )
    )


@agora.group(name="comment")
def agora_comment_group():
    """评论管理"""


@agora_comment_group.command(name="create")
@click.argument("post_id")
@click.argument("content")
@click.option("--parent-type", default="post", help="post / answer / comment")
@click.option("--parent-id", default="", help="parent_type=post 时可省略")
def agora_comment_create(post_id: str, content: str, parent_type: str, parent_id: str):
    """发评论（需 JWT）"""
    _json(
        _run_with_client(
            lambda client: client.agora_create_comment(
                post_id,
                content,
                parent_type=parent_type,
                parent_id=parent_id,
            )
        )
    )


@main.command(name="sync-state")
@click.option(
    "--skip-leaderboard",
    is_flag=True,
    default=False,
    help="跳过 leaderboard 拉取，减少一次远端请求",
)
def sync_state(skip_leaderboard: bool):
    """同步 Arena 状态并持久化为结构化 JSON"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.sync_state(
                include_leaderboard=not skip_leaderboard
            )
        )
    )


@main.command(name="next-action")
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
def next_action(no_refresh: bool):
    """输出程序化 next_action JSON"""
    _json(_run_with_runtime(lambda runtime: runtime.next_action(refresh=not no_refresh)))


@main.command(name="prepare-context")
@click.argument("task_id")
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
def prepare_context(task_id: str, no_refresh: bool):
    """构造任务上下文 JSON，供 Synergy 或自动流程消费"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.prepare_context(task_id, refresh=not no_refresh)
        )
    )


@main.command(name="prepare-bounty-context")
@click.argument("bounty_task_id")
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
def prepare_bounty_context(bounty_task_id: str, no_refresh: bool):
    """构造高额悬赏上下文 JSON，供 Synergy 或自动流程消费"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.prepare_bounty_context(
                bounty_task_id,
                refresh=not no_refresh,
            )
        )
    )


@main.command(name="dispatch-plan")
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
@click.option(
    "--actionable-only",
    is_flag=True,
    default=False,
    help="只输出当前建议分发独立 agent 的题目",
)
def dispatch_plan(no_refresh: bool, actionable_only: bool):
    """输出当前 stage 的按题独立 agent 分发计划 JSON"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.dispatch_plan(
                refresh=not no_refresh,
                actionable_only=actionable_only,
            )
        )
    )


@main.command(name="quality-check")
@click.argument("task_id")
@click.option("--file", "file_path", default=None, help="从文件读取 draft")
@click.option("--text", default=None, help="直接传入 draft 文本")
@click.option(
    "--gate-mode",
    type=click.Choice(["normal_gate", "coverage_gate"]),
    default="normal_gate",
    show_default=True,
)
@click.option(
    "--action-kind",
    type=click.Choice(["reply_followup", "submit_official", "improve_official", "answer_bounty"]),
    default=None,
)
def quality_check(
    task_id: str,
    file_path: Optional[str],
    text: Optional[str],
    gate_mode: str,
    action_kind: Optional[str],
):
    """执行结构化质量校验"""
    if bool(file_path) == bool(text):
        raise click.UsageError("请在 --file 和 --text 中二选一")
    draft_text = _read_text_file(file_path) if file_path else str(text)
    _json(
        _run_with_runtime(
            lambda runtime: runtime.quality_check(
                task_id=task_id,
                draft_text=draft_text,
                gate_mode=gate_mode,
                action_kind=action_kind,
            )
        )
    )


@main.command(name="submit-file")
@click.argument("task_id")
@click.argument("file_path")
@click.option(
    "--gate-mode",
    type=click.Choice(["normal_gate", "coverage_gate"]),
    default=None,
)
@click.option("--force", is_flag=True, default=False, help="允许跳过软性质量阻塞")
@click.option(
    "--refresh-after",
    is_flag=True,
    default=False,
    help="提交后立即执行一次全量 sync-state 刷新",
)
def submit_file(
    task_id: str,
    file_path: str,
    gate_mode: Optional[str],
    force: bool,
    refresh_after: bool,
):
    """从文件读取内容并执行质量检查后提交官方题"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.submit_file(
                task_id=task_id,
                file_path=file_path,
                gate_mode=gate_mode,
                force=force,
                refresh_after=refresh_after,
            )
        )
    )


@main.command(name="reply-followup")
@click.argument("task_id")
@click.argument("file_path")
@click.option("--comment-id", default=None, help="显式指定要回复的 comment_id")
@click.option("--post-id", default=None, help="显式指定 Agora post_id")
@click.option("--force", is_flag=True, default=False, help="允许跳过软性质量阻塞")
@click.option(
    "--refresh-after",
    is_flag=True,
    default=False,
    help="回复后立即执行一次全量 sync-state 刷新",
)
def reply_followup(
    task_id: str,
    file_path: str,
    comment_id: Optional[str],
    post_id: Optional[str],
    force: bool,
    refresh_after: bool,
):
    """从文件读取内容并回复 follow-up 评论"""
    _json(
        _run_with_runtime(
            lambda runtime: runtime.reply_followup(
                task_id=task_id,
                file_path=file_path,
                comment_id=comment_id,
                post_id=post_id,
                force=force,
                refresh_after=refresh_after,
            )
        )
    )


@main.command(name="publish-probe-bounty")
@click.argument("task_id")
@click.argument("subproblem")
@click.option(
    "--research-brief-json",
    default=None,
    help="研究摘要 JSON 字符串，供判别悬赏回复时使用",
)
@click.option(
    "--evidence-packet-json",
    default=None,
    help="证据包 JSON 字符串，供判别悬赏回复时使用",
)
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
@click.option("--target-reply-count", default=4, show_default=True, type=int)
@click.option("--poll-attempts", default=3, show_default=True, type=int)
@click.option("--poll-interval", default=0.0, show_default=True, type=float)
def publish_probe_bounty(
    task_id: str,
    subproblem: str,
    research_brief_json: Optional[str],
    evidence_packet_json: Optional[str],
    no_refresh: bool,
    target_reply_count: int,
    poll_attempts: int,
    poll_interval: float,
):
    """发布或复用 1 元探针悬赏，并对 replies 做判别/采纳"""
    research_brief = json.loads(research_brief_json) if research_brief_json else {}
    evidence_packet = json.loads(evidence_packet_json) if evidence_packet_json else {}
    _json(
        _run_with_runtime(
            lambda runtime: runtime.publish_probe_bounty(
                task_id=task_id,
                subproblem=subproblem,
                research_brief=research_brief,
                evidence_packet=evidence_packet,
                refresh=not no_refresh,
                target_reply_count=target_reply_count,
                poll_attempts=poll_attempts,
                poll_interval_seconds=poll_interval,
            )
        )
    )


@main.command(name="answer-bounty")
@click.argument("bounty_task_id")
@click.argument("file_path")
@click.option("--no-refresh", is_flag=True, default=False, help="使用最近一次同步快照")
def answer_bounty(bounty_task_id: str, file_path: str, no_refresh: bool):
    """从文件读取内容并回答他人悬赏"""
    answer_text = _read_text_file(file_path)
    _json(
        _run_with_runtime(
            lambda runtime: runtime.answer_bounty(
                bounty_task_id=bounty_task_id,
                answer_text=answer_text,
                refresh=not no_refresh,
            )
        )
    )


if __name__ == "__main__":
    main()
