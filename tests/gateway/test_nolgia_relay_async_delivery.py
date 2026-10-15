"""Nolgia relay runs: a process's exit reaches the customer (Nolgia fork).

A relay run (``POST /v1/runs`` from nolgia-api) is a stateless HTTP session, so the API server
binds ``async_delivery=False`` and ``terminal(notify=...)`` is refused: on 2026-10-15 a staff
proof turn's agent had to answer "this session can't send completion notices". When the platform
says it files this run's finished background work in its chat (the run body's
``background_results``, nolgia-api AGENT_BACKGROUND_RESULTS) and the run names its platform
session (``workspace_id``), the bind turns async delivery on, so the exit is handed to the
platform (gateway.platforms.nolgia_background_results). Everything else that async delivery
would open (watch patterns, heartbeats, detached and scheduled cron) stays off: the session is
completion-only.
"""

import json

import pytest

from gateway.platforms import api_server as api_server_mod
from gateway.platforms import nolgia_background_results as results
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import async_delivery_supported, clear_session_vars, set_session_vars

SESSION = "0f0e7a3c-5a74-4c86-9b1d-2f6f3c9d8e11"


@pytest.fixture
def platform_pod(monkeypatch):
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.ai")
    monkeypatch.setenv("NOLGIA_TOKEN", "nolp_pod")
    monkeypatch.delenv("NOLGIA_BACKGROUND_RESULTS", raising=False)


def _bind(**kw):
    return APIServerAdapter._bind_api_server_session(chat_id="run_1", session_key="run_1", session_id="run_1", **kw)


def test_request_flag_is_a_literal_true_only():
    assert api_server_mod._request_background_results({"background_results": True}) is True
    for body in ({}, {"background_results": "true"}, {"background_results": 1}, {"background_results": False}, None, []):
        assert api_server_mod._request_background_results(body) is False


def test_the_flag_never_breaks_an_idempotent_replay():
    assert "background_results" in api_server_mod._RUNS_IDEMPOTENCY_FINGERPRINT_EXCLUDED_KEYS


@pytest.mark.parametrize("kw, pod, expected", [
    ({"background_results": True, "workspace_id": SESSION}, True, True),
    ({"background_results": False, "workspace_id": SESSION}, True, False),
    ({"background_results": True, "workspace_id": "session-local"}, True, False),
    ({"background_results": True}, True, False),
    ({"background_results": True, "workspace_id": SESSION}, False, False),
])
def test_bind_turns_async_delivery_on_only_for_a_platform_session_the_platform_takes(monkeypatch, kw, pod, expected):
    if pod:
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.ai")
        monkeypatch.setenv("NOLGIA_TOKEN", "nolp_pod")
    else:
        monkeypatch.delenv("NOLGIA_API_URL", raising=False)
        monkeypatch.delenv("NOLGIA_TOKEN", raising=False)
    monkeypatch.delenv("NOLGIA_BACKGROUND_RESULTS", raising=False)
    tokens = _bind(**kw)
    try:
        assert async_delivery_supported() is expected
        assert results.completion_only_session() is expected
    finally:
        clear_session_vars(tokens)


def test_escape_hatch_keeps_relay_runs_stateless(platform_pod, monkeypatch):
    monkeypatch.setenv("NOLGIA_BACKGROUND_RESULTS", "0")
    tokens = _bind(background_results=True, workspace_id=SESSION)
    try:
        assert async_delivery_supported() is False
    finally:
        clear_session_vars(tokens)


class TestTerminalOnACompletionOnlySession:
    @pytest.fixture(autouse=True)
    def _clean_watchers(self):
        from tools.process_registry import process_registry

        process_registry.pending_watchers = []
        yield
        process_registry.pending_watchers = []

    def _run_bg(self, **kw):
        from tools.terminal_tool import terminal_tool

        return json.loads(terminal_tool(command="sleep 30 && echo DONE", background=True, **kw))

    def test_notify_is_kept_and_routed_with_a_note(self, platform_pod):
        tokens = _bind(background_results=True, workspace_id=SESSION)
        try:
            result = self._run_bg(notify_on_complete=True)
        finally:
            clear_session_vars(tokens)
        assert result.get("notify_on_complete") is True
        assert not result.get("notify_unsupported")
        assert "NOLGIA files its result in this chat" in result.get("notify_note", "")

    def test_watch_patterns_are_dropped(self, platform_pod):
        tokens = _bind(background_results=True, workspace_id=SESSION)
        try:
            result = self._run_bg(watch_patterns=["ERROR"])
        finally:
            clear_session_vars(tokens)
        assert not result.get("watch_patterns")
        assert "only the exit is delivered" in result.get("watch_patterns_ignored", "")

    def test_heartbeats_are_not_armed(self, platform_pod):
        from tools.terminal_tool import terminal_tool

        tokens = _bind(background_results=True, workspace_id=SESSION)
        try:
            result = json.loads(terminal_tool(command="sleep 30 && echo DONE", background=True, notify_on_complete=True, heartbeat=120))
        finally:
            clear_session_vars(tokens)
        assert result.get("notify_on_complete") is True
        assert "heartbeat_seconds" not in result
        assert "only the exit is delivered" in result.get("heartbeat_ignored", "")

    def test_without_the_platform_flag_nothing_changes(self, platform_pod):
        tokens = _bind(workspace_id=SESSION)
        try:
            result = self._run_bg(notify_on_complete=True)
        finally:
            clear_session_vars(tokens)
        assert result.get("notify_on_complete") is False
        assert result.get("notify_unsupported")


def test_cron_never_targets_a_completion_only_session(platform_pod):
    from tools.cronjob_job_args import _origin_from_env

    tokens = _bind(background_results=True, workspace_id=SESSION)
    try:
        assert async_delivery_supported() is True
        assert _origin_from_env() is None, "a scheduled fire must not report to a relay session"
    finally:
        clear_session_vars(tokens)

    # Control: a push-capable session keeps its origin.
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _origin_from_env()["platform"] == "telegram"
    finally:
        clear_session_vars(tokens)


def test_completion_only_is_false_on_push_sessions(platform_pod):
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert results.completion_only_session() is False
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag, expected", [(True, True), (None, False)])
async def test_a_relay_run_carries_the_platform_flag_into_its_session(platform_pod, tmp_path, monkeypatch, flag, expected):
    """End to end through POST /v1/runs: the body's background_results reaches the bind, so the
    run's tools see async delivery exactly when the platform said it takes the results."""
    pytest.importorskip("aiohttp")
    from unittest.mock import MagicMock, patch

    from aiohttp.test_utils import TestClient, TestServer

    from tests.gateway.test_session_workspace_isolation import _create_runs_app, _make_adapter, _wait_completed

    monkeypatch.setenv("HERMES_SESSION_SCRATCH_BASE", str(tmp_path / "scratch"))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    captured = {}

    def _run(user_message=None, conversation_history=None, task_id=None):
        captured["async"] = async_delivery_supported()
        captured["completion_only"] = results.completion_only_session()
        return {"final_response": "done"}

    body = {"input": "hi", "workspace_id": SESSION}
    if flag is not None:
        body["background_results"] = flag
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            agent = MagicMock()
            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
            mock_create.return_value = agent
            resp = await cli.post("/v1/runs", json=body)
            assert resp.status == 202
            await _wait_completed(cli, (await resp.json())["run_id"])

    assert captured == {"async": expected, "completion_only": expected}
