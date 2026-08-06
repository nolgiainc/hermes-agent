"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- POST /v1/runs — Idempotency-Key replay returns the original run_id
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- POST /v1/runs/{run_id}/steer — inject a mid-run user message
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    _IdempotencyCache,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# POST /v1/runs — Idempotency-Key replay (NOL-397)
# ---------------------------------------------------------------------------


def _make_completed_agent(final_response: str = "done") -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": final_response}
    mock_agent.session_prompt_tokens = 10
    mock_agent.session_completion_tokens = 5
    mock_agent.session_total_tokens = 15
    return mock_agent


async def _wait_terminal(cli: TestClient, run_id: str) -> dict:
    """Poll GET /v1/runs/{run_id} until the run reaches a terminal status."""
    status: dict = {}
    for _ in range(60):
        status_resp = await cli.get(f"/v1/runs/{run_id}")
        status = await status_resp.json()
        if status.get("status") in {"completed", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.05)
    return status


class TestRunsIdempotency:
    """A replayed Idempotency-Key must return the ORIGINAL run_id and the
    run must execute (and bill) exactly once — the contract that lets a
    supervisor retry a submit whose 202 was lost to an ambiguous 502/504
    or transport failure."""

    @pytest.mark.asyncio
    async def test_replayed_key_returns_original_run_id_and_executes_once(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server._idem_cache", _IdempotencyCache()):
                with patch.object(adapter, "_create_agent") as mock_create:
                    mock_create.return_value = _make_completed_agent()

                    body = {"input": "hello", "session_id": "idem-sid"}
                    headers = {"Idempotency-Key": "idem-nol397-replay"}

                    first = await cli.post("/v1/runs", json=body, headers=headers)
                    assert first.status == 202
                    run_id = (await first.json())["run_id"]

                    status = await _wait_terminal(cli, run_id)
                    assert status["status"] == "completed"

                    # Replay after the run settled (lost-202 salvage): the
                    # original run_id comes back and nothing re-executes.
                    replay = await cli.post("/v1/runs", json=body, headers=headers)
                    assert replay.status == 202
                    replay_data = await replay.json()
                    assert replay_data["run_id"] == run_id

                    assert mock_create.call_count == 1
                    assert mock_create.return_value.run_conversation.call_count == 1

                    # The replayed run_id still resolves to the settled result.
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    assert status["status"] == "completed"
                    assert status["output"] == "done"

    @pytest.mark.asyncio
    async def test_replay_of_running_submission_bypasses_concurrency_cap(self, adapter):
        """The original run may hold the last concurrency slot; its own
        replay must short-circuit to the cached run_id instead of bouncing
        with 429 until the run finishes (which could outlive the cache TTL
        and re-execute the work the retry was trying not to duplicate)."""
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server._idem_cache", _IdempotencyCache()):
                with patch.object(adapter, "_create_agent") as mock_create:
                    mock_agent, agent_ready, interrupted = _make_slow_agent()
                    mock_create.return_value = mock_agent

                    headers = {"Idempotency-Key": "idem-nol397-capped"}
                    try:
                        first = await cli.post(
                            "/v1/runs", json={"input": "slow"}, headers=headers
                        )
                        assert first.status == 202
                        run_id = (await first.json())["run_id"]
                        assert agent_ready.wait(timeout=3.0)

                        # The running agent holds the only slot: a fresh
                        # submission is capped...
                        fresh = await cli.post("/v1/runs", json={"input": "slow"})
                        assert fresh.status == 429

                        # ...but replaying the in-flight submission's key is
                        # admitted and returns the original run_id.
                        replay = await cli.post(
                            "/v1/runs", json={"input": "slow"}, headers=headers
                        )
                        assert replay.status == 202
                        assert (await replay.json())["run_id"] == run_id
                        assert mock_create.call_count == 1
                    finally:
                        interrupted.set()
                    status = await _wait_terminal(cli, run_id)
                    assert status["status"] in {"completed", "cancelled"}
                    assert mock_agent.run_conversation.call_count == 1

    @pytest.mark.asyncio
    async def test_distinct_or_absent_keys_start_distinct_runs(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server._idem_cache", _IdempotencyCache()):
                with patch.object(adapter, "_create_agent") as mock_create:
                    mock_create.side_effect = lambda **kwargs: _make_completed_agent()

                    body = {"input": "hello"}
                    run_ids = []
                    for headers in (
                        {"Idempotency-Key": "idem-nol397-a"},
                        {"Idempotency-Key": "idem-nol397-b"},
                        {},
                    ):
                        resp = await cli.post("/v1/runs", json=body, headers=headers)
                        assert resp.status == 202
                        run_id = (await resp.json())["run_id"]
                        run_ids.append(run_id)
                        status = await _wait_terminal(cli, run_id)
                        assert status["status"] == "completed"

        assert len(set(run_ids)) == 3
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_same_key_different_body_is_not_replayed(self, adapter):
        """Key reuse with a different submission body is a different request
        (fingerprint mismatch): it starts a new run, mirroring the
        /v1/chat/completions and /v1/responses fingerprint semantics."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server._idem_cache", _IdempotencyCache()):
                with patch.object(adapter, "_create_agent") as mock_create:
                    mock_create.side_effect = lambda **kwargs: _make_completed_agent()

                    headers = {"Idempotency-Key": "idem-nol397-fp"}

                    first = await cli.post(
                        "/v1/runs", json={"input": "hello"}, headers=headers
                    )
                    assert first.status == 202
                    first_id = (await first.json())["run_id"]
                    status = await _wait_terminal(cli, first_id)
                    assert status["status"] == "completed"

                    second = await cli.post(
                        "/v1/runs", json={"input": "different"}, headers=headers
                    )
                    assert second.status == 202
                    second_id = (await second.json())["run_id"]
                    status = await _wait_terminal(cli, second_id)
                    assert status["status"] == "completed"

        assert second_id != first_id
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_rotated_nolgia_token_still_replays(self, adapter):
        """nolgia_token is a short-lived per-attempt credential a supervisor
        may re-mint on retry; it must not participate in the fingerprint, or
        a rotated token would miss the cache and re-execute the run."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.platforms.api_server._idem_cache", _IdempotencyCache()):
                with patch.object(adapter, "_create_agent") as mock_create:
                    mock_create.return_value = _make_completed_agent()

                    headers = {"Idempotency-Key": "idem-nol397-token"}

                    first = await cli.post(
                        "/v1/runs",
                        json={"input": "hello", "nolgia_token": "tok-attempt-1"},
                        headers=headers,
                    )
                    assert first.status == 202
                    run_id = (await first.json())["run_id"]
                    status = await _wait_terminal(cli, run_id)
                    assert status["status"] == "completed"

                    replay = await cli.post(
                        "/v1/runs",
                        json={"input": "hello", "nolgia_token": "tok-attempt-2"},
                        headers=headers,
                    )
                    assert replay.status == 202
                    assert (await replay.json())["run_id"] == run_id
                    assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_capabilities_advertises_run_idempotency(self, adapter):
        """Supervisors gate widening their submit-retry classes on this
        flag, so pods that replay Idempotency-Key must advertise it."""
        app = _create_runs_app(adapter)
        app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["run_idempotency"] is True


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — inject a mid-run user message
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_active_run_calls_agent_steer(self, adapter):
        """Steer should pass the raw text to AIAgent.steer and return 202.

        The endpoint does NOT add its own attribution prefix — the agent
        loop wraps the text in the out-of-band user-message marker at
        splice time (agent.prompt_builder.format_steer_marker).
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_agent.steer = MagicMock(return_value=True)
                mock_create.return_value = mock_agent
                try:
                    resp = await cli.post("/v1/runs", json={"input": "hello"})
                    assert resp.status == 202
                    data = await resp.json()
                    run_id = data["run_id"]

                    agent_ready.wait(timeout=3.0)
                    await asyncio.sleep(0.1)
                    assert run_id in adapter._active_run_agents

                    steer_resp = await cli.post(
                        f"/v1/runs/{run_id}/steer",
                        json={"text": "focus on the tests"},
                    )
                    assert steer_resp.status == 202
                    steer_data = await steer_resp.json()
                    assert steer_data["run_id"] == run_id
                    assert steer_data["status"] == "steering"

                    mock_agent.steer.assert_called_once_with("focus on the tests")
                    # Steering must not interrupt the run.
                    mock_agent.interrupt.assert_not_called()
                finally:
                    # Unblock the slow agent thread so teardown doesn't wait.
                    interrupted.set()

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_nonexistent/steer", json={"text": "hi"}
            )
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_steer_completed_run_returns_404(self, adapter):
        """Steering a finished run should 404 (caller falls back to a new turn)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)
                assert run_id not in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer", json={"text": "too late"}
                )
                assert steer_resp.status == 404
                mock_agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_empty_text_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_agent.steer = MagicMock(return_value=True)
                mock_create.return_value = mock_agent
                try:
                    resp = await cli.post("/v1/runs", json={"input": "hello"})
                    assert resp.status == 202
                    run_id = (await resp.json())["run_id"]

                    agent_ready.wait(timeout=3.0)
                    await asyncio.sleep(0.1)

                    for bad_body in ({}, {"text": ""}, {"text": "   "}, {"text": 42}):
                        steer_resp = await cli.post(
                            f"/v1/runs/{run_id}/steer", json=bad_body
                        )
                        assert steer_resp.status == 400

                    mock_agent.steer.assert_not_called()
                finally:
                    interrupted.set()

    @pytest.mark.asyncio
    async def test_steer_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent
                try:
                    resp = await cli.post("/v1/runs", json={"input": "hello"})
                    assert resp.status == 202
                    run_id = (await resp.json())["run_id"]

                    agent_ready.wait(timeout=3.0)
                    await asyncio.sleep(0.1)

                    steer_resp = await cli.post(
                        f"/v1/runs/{run_id}/steer",
                        data=b"not json",
                        headers={"Content-Type": "application/json"},
                    )
                    assert steer_resp.status == 400
                finally:
                    interrupted.set()

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"text": "hi"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_steer_rejected_by_agent_returns_400(self, adapter):
        """If AIAgent.steer returns False (empty after trim), respond 400."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_agent.steer = MagicMock(return_value=False)
                mock_create.return_value = mock_agent
                try:
                    resp = await cli.post("/v1/runs", json={"input": "hello"})
                    assert resp.status == 202
                    run_id = (await resp.json())["run_id"]

                    agent_ready.wait(timeout=3.0)
                    await asyncio.sleep(0.1)

                    steer_resp = await cli.post(
                        f"/v1/runs/{run_id}/steer", json={"text": "hi"}
                    )
                    assert steer_resp.status == 400
                finally:
                    interrupted.set()

    @pytest.mark.asyncio
    async def test_unanswered_steer_surfaced_as_pending_steer(self, adapter):
        """A steer the model never saw is reported on the completed status.

        run_conversation returns ``pending_steer`` when a steer landed after
        the final tool batch (see agent/turn_finalizer.py); the API server
        must surface it so callers can re-send the text as a follow-up turn.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "leftover guidance",
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                await asyncio.sleep(0.3)

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] == "completed"
                assert status_data["pending_steer"] == "leftover guidance"


# ---------------------------------------------------------------------------
# Durable terminal run statuses (NOL-93)
# ---------------------------------------------------------------------------


class TestDurableRunStatus:
    """A /v1/runs turn can outlive its supervisor: the platform relay
    budget-fails the turn while the pod keeps executing and completes later.
    Terminal statuses must therefore survive both the in-memory TTL and a
    gateway restart, so a late GET /v1/runs/{run_id} salvages the outcome
    instead of forcing a full (billed) re-run. In-flight statuses must NOT
    persist — after a restart that work is genuinely gone and a durable
    "running" row would lie forever."""

    @pytest.mark.asyncio
    async def test_terminal_status_survives_adapter_restart(self):
        first = _make_adapter()
        first._set_run_status(
            "run_persist",
            "completed",
            output="the 30s spot is delivered",
            usage={"total_tokens": 6},
            last_event="run.completed",
        )

        # A fresh adapter simulates a gateway restart: empty _run_statuses,
        # same on-disk response_store.db (per-test HERMES_HOME).
        reborn = _make_adapter()
        assert reborn._run_statuses == {}
        app = _create_runs_app(reborn)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_persist")
            assert resp.status == 200
            status = await resp.json()
        assert status["status"] == "completed"
        assert status["output"] == "the 30s spot is delivered"
        assert status["usage"]["total_tokens"] == 6

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", ["failed", "cancelled"])
    async def test_failed_and_cancelled_statuses_persist(self, terminal):
        first = _make_adapter()
        first._set_run_status("run_terminal", terminal, error="boom")

        reborn = _make_adapter()
        app = _create_runs_app(reborn)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_terminal")
            assert resp.status == 200
            status = await resp.json()
        assert status["status"] == terminal

    @pytest.mark.asyncio
    async def test_in_flight_status_is_not_persisted(self):
        first = _make_adapter()
        first._set_run_status("run_live", "running")

        reborn = _make_adapter()
        app = _create_runs_app(reborn)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_live")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_memory_ttl_expiry_still_serves_terminal_status(self, adapter):
        adapter._set_run_status(
            "run_aged",
            "completed",
            output="done",
            last_event="run.completed",
        )
        adapter._sweep_orphaned_runs_once(time.time() + adapter._RUN_STATUS_TTL + 1)
        assert "run_aged" not in adapter._run_statuses

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_aged")
            assert resp.status == 200
            status = await resp.json()
        assert status["status"] == "completed"
        assert status["output"] == "done"

    def test_retention_prunes_expired_rows(self, adapter):
        store = adapter._response_store
        store.put_run_status("run_old", {"status": "completed", "output": "old"})
        expired = time.time() - store.RUN_STATUS_RETENTION_SECONDS - 60
        with store._run_status_lock:
            store._conn.execute(
                "UPDATE run_statuses SET updated_at = ? WHERE run_id = ?",
                (expired, "run_old"),
            )
            store._conn.commit()

        # Any later persist prunes rows past retention.
        store.put_run_status("run_new", {"status": "completed", "output": "new"})
        assert store.get_run_status("run_old") is None
        assert store.get_run_status("run_new")["output"] == "new"

    def test_capacity_prunes_oldest_rows(self, adapter):
        store = adapter._response_store
        keep = store.MAX_STORED_RUN_STATUSES
        for i in range(keep + 5):
            store.put_run_status(f"run_{i:04d}", {"status": "completed", "output": str(i)})
            # Distinct updated_at ordering without sleeping: backdate each row
            # progressively so eviction order is deterministic.
            with store._run_status_lock:
                store._conn.execute(
                    "UPDATE run_statuses SET updated_at = ? WHERE run_id = ?",
                    (time.time() - (keep + 5 - i), f"run_{i:04d}"),
                )
                store._conn.commit()
        store.put_run_status("run_final", {"status": "completed", "output": "final"})
        assert store.get_run_status("run_0000") is None
        assert store.get_run_status("run_final")["output"] == "final"

    def test_expired_row_is_not_served_on_read(self, adapter):
        """put_run_status only prunes when a newer write lands; the read path
        must enforce retention too or a quiet gateway serves stale results
        forever."""
        store = adapter._response_store
        store.put_run_status("run_stale", {"status": "completed", "output": "old"})
        expired = time.time() - store.RUN_STATUS_RETENTION_SECONDS - 60
        with store._run_status_lock:
            store._conn.execute(
                "UPDATE run_statuses SET updated_at = ? WHERE run_id = ?",
                (expired, "run_stale"),
            )
            store._conn.commit()
        assert store.get_run_status("run_stale") is None
        # And the row was evicted, not just filtered.
        with store._run_status_lock:
            row = store._conn.execute(
                "SELECT 1 FROM run_statuses WHERE run_id = ?", ("run_stale",)
            ).fetchone()
        assert row is None

    def test_non_durable_terminal_status_stays_in_memory_only(self):
        first = _make_adapter()
        first._set_run_status("run_teardown", "cancelled", durable=False)
        # Late pollers on the SAME process still see it...
        assert first._run_statuses["run_teardown"]["status"] == "cancelled"
        # ...but it is not persisted: a restarted gateway must 404 so the
        # caller resubmits the lost work instead of treating it as settled.
        reborn = _make_adapter()
        assert reborn._response_store.get_run_status("run_teardown") is None

    @pytest.mark.asyncio
    async def test_teardown_cancellation_is_not_persisted(self):
        """cancel_background_tasks() (gateway restart) cancelling a live
        _run_and_close task must not leave a durable 'cancelled' behind."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.05)

                task = adapter._active_run_tasks[run_id]
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                # In-memory: cancelled (same-process pollers stay informed).
                assert adapter._run_statuses[run_id]["status"] == "cancelled"
                # Durable: no settled terminal status — the run's journal
                # row stays unsettled so the NEXT boot reports the honest
                # restart failure (see TestRunJournal) instead of a
                # synthetic settled "cancelled".
                assert adapter._response_store.get_run_status(run_id) is None
                assert (
                    adapter._response_store.get_journal_row(run_id)["settled"]
                    is False
                )
                interrupted.set()  # release the executor thread

    @pytest.mark.asyncio
    async def test_user_stop_cancellation_is_persisted(self):
        """A /stop-initiated cancellation is a genuinely settled outcome and
        must survive a restart (the caller should NOT resubmit)."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.05)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                for _ in range(40):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "cancelled":
                        break
                    await asyncio.sleep(0.05)
                assert adapter._run_statuses[run_id]["status"] == "cancelled"
                assert adapter._response_store.get_run_status(run_id)["status"] == "cancelled"

    def test_corrupted_row_is_evicted(self, adapter):
        store = adapter._response_store
        with store._run_status_lock:
            store._conn.execute(
                "INSERT OR REPLACE INTO run_statuses (run_id, data, updated_at) VALUES (?, ?, ?)",
                ("run_bad", "{not json", time.time()),
            )
            store._conn.commit()
        assert store.get_run_status("run_bad") is None
        assert store.get_run_status("run_bad") is None  # evicted, stays gone


# ---------------------------------------------------------------------------
# Restart-durable run journal (NOL-423)
# ---------------------------------------------------------------------------


class TestRunJournal:
    """Runs are journaled durably at submission and reconciled on the next
    boot: a run a dead process left in flight becomes a durable ``failed``
    status with error_code="gateway_restart" + provenance, so a replacement
    pod answers status polls honestly (no 404) and the platform's
    capability-gated auto-resubmit can act safely. Autopilot caps
    termination grace at 600s on-demand / 25s on spot (NOL-410), so this
    journal — not graceful drain — is what makes restarts honest."""

    @pytest.fixture(autouse=True)
    def _isolate_journal_store_ownership(self, tmp_path, monkeypatch):
        """Keep the store-ownership lease inside the test's temp dir.

        Reconciliation is gated on owning the journal's store, which is
        proven with a machine-local scoped lock; point it at tmp_path so
        tests never touch the developer's real lock dir, and clear the
        in-process owner registry between tests.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
        from gateway.platforms import api_server as api_server_mod

        api_server_mod._JOURNAL_STORE_OWNERS.clear()
        yield
        api_server_mod._JOURNAL_STORE_OWNERS.clear()

    @pytest.mark.asyncio
    async def test_submission_journals_identity_not_payload(self):
        """The journal row lands before the run executes and carries hashes,
        sizes, and a bounded redacted tail — never the full input and never
        the run credential."""
        import hashlib

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        secret_token = "nolgia-tok-SECRET-4242"
        long_input = ("finish the quarterly report " * 40).strip()  # > tail bound
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": long_input,
                        "session_id": "journal-sid",
                        "nolgia_token": secret_token,
                    },
                    headers={"Idempotency-Key": "msg-journal-1"},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)

                row = adapter._response_store.get_journal_row(run_id)
                assert row is not None
                assert row["settled"] is False
                assert row["boot_uuid"] == adapter._boot_uuid
                data = row["data"]
                assert data["session_id"] == "journal-sid"
                assert data["idempotency_key"] == "msg-journal-1"
                assert data["pod"] == adapter._pod_identity
                assert data["input_sha256"] == hashlib.sha256(
                    long_input.encode("utf-8")
                ).hexdigest()
                assert data["input_bytes"] == len(long_input.encode("utf-8"))
                # Bounded tail, not the full payload.
                assert len(data["input_tail"]) <= 200
                assert long_input not in json.dumps(data)
                # Never a credential.
                assert secret_token not in json.dumps(data)

                interrupted.set()
                await _wait_terminal(cli, run_id)

    @pytest.mark.asyncio
    async def test_terminal_settle_is_transactional_and_idempotent(self):
        """A completed run's journal row settles with the terminal write;
        replaying the settle and reconciling a later boot never overwrite
        the real outcome (completed-during-shutdown wins)."""
        adapter = _make_adapter()
        store = adapter._response_store
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _make_completed_agent("all done")
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                status = await _wait_terminal(cli, run_id)
                assert status["status"] == "completed"

        row = store.get_journal_row(run_id)
        assert row["settled"] is True
        assert row["status"] == "completed"

        # Idempotent: replaying the settle rewrites the same rows.
        store.settle_run_status(run_id, store.get_run_status(run_id))
        assert store.get_run_status(run_id)["status"] == "completed"

        # A later boot must not double-mark the settled run.
        reconciled = store.reconcile_prior_boot_runs("some-other-boot")
        assert reconciled == []
        assert store.get_run_status(run_id)["status"] == "completed"
        assert store.get_run_status(run_id)["output"] == "all done"

    @pytest.mark.asyncio
    async def test_prior_boot_inflight_run_reconciled_as_gateway_restart(self):
        """The core NOL-423 contract: a run in flight when the process died
        answers the next boot's status poll as durably failed with
        error_code="gateway_restart" and provenance — never a 404."""
        first = _make_adapter()
        app = _create_runs_app(first)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(first, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs", json={"input": "hello", "session_id": "restart-sid"}
                )
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.05)

                # A fresh adapter simulates the replacement pod booting
                # against the same durable store. Reconciliation happens
                # once it owns the listener + the journal store (connect()),
                # never at construction — see
                # test_construction_alone_never_reconciles_live_runs.
                reborn = _make_adapter()
                assert reborn._reconcile_prior_boot_runs() is True
                reborn_app = _create_runs_app(reborn)
                async with TestClient(TestServer(reborn_app)) as reborn_cli:
                    status_resp = await reborn_cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()

                assert status["status"] == "failed"
                assert status["error_code"] == "gateway_restart"
                assert status["last_status"] == "running"
                assert status["session_id"] == "restart-sid"
                provenance = status["restart_provenance"]
                assert provenance["boot_uuid"] == first._boot_uuid
                assert provenance["last_status"] == "running"
                assert provenance["reconciled_by_pod"] == reborn._pod_identity

                # The journal row settled, so a THIRD boot re-marks nothing.
                assert first._response_store.get_journal_row(run_id) is not None
                third = _make_adapter()
                assert third._reconcile_prior_boot_runs() is True
                assert (
                    third._response_store.get_run_status(run_id)["error_code"]
                    == "gateway_restart"
                )
                interrupted.set()

    @pytest.mark.asyncio
    async def test_teardown_cancelled_run_reconciles_on_next_boot(self):
        """The durable=False teardown cancellation intentionally leaves the
        journal unsettled; the next boot turns it into the honest restart
        failure instead of the pre-journal 404."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.05)

                task = adapter._active_run_tasks[run_id]
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                interrupted.set()

        reborn = _make_adapter()
        assert reborn._reconcile_prior_boot_runs() is True
        reborn_app = _create_runs_app(reborn)
        async with TestClient(TestServer(reborn_app)) as cli:
            status_resp = await cli.get(f"/v1/runs/{run_id}")
            assert status_resp.status == 200
            status = await status_resp.json()
        assert status["status"] == "failed"
        assert status["error_code"] == "gateway_restart"

    def test_same_boot_rows_are_never_reconciled(self, adapter):
        """Liveness is boot identity, not heartbeat staleness: a slow or
        paused process is not dead, so same-boot rows are untouched no
        matter how stale their heartbeat is."""
        store = adapter._response_store
        store.journal_run_submitted(
            "run_live_here",
            adapter._boot_uuid,
            {"status": "running", "created_at": time.time() - 3600},
        )
        assert store.reconcile_prior_boot_runs(adapter._boot_uuid) == []
        assert store.get_journal_row("run_live_here")["settled"] is False
        assert store.get_run_status("run_live_here") is None

    def test_waiting_for_approval_orphan_reports_lost_approval(self, adapter):
        """Approval queues are process memory: an orphaned
        waiting_for_approval run reports the restart failure with its last
        status so the platform can tell the user the approval was lost."""
        store = adapter._response_store
        store.journal_run_submitted(
            "run_appr", "dead-boot", {"status": "queued", "created_at": time.time()}
        )
        store.journal_run_update(
            "run_appr", status="waiting_for_approval", last_event="approval.request"
        )
        reconciled = store.reconcile_prior_boot_runs("new-boot")
        assert [r["run_id"] for r in reconciled] == ["run_appr"]
        status = store.get_run_status("run_appr")
        assert status["status"] == "failed"
        assert status["error_code"] == "gateway_restart"
        assert status["last_status"] == "waiting_for_approval"
        assert status["last_event"] == "approval.request"

    def test_existing_terminal_status_wins_over_unsettled_journal(self, adapter):
        """Defense in depth for rows an older image wrote terminally without
        the transactional settle: the real outcome is kept, the row is only
        settled."""
        store = adapter._response_store
        store.journal_run_submitted(
            "run_done_old", "dead-boot", {"status": "running", "created_at": time.time()}
        )
        store.put_run_status(
            "run_done_old", {"run_id": "run_done_old", "status": "completed", "output": "kept"}
        )
        reconciled = store.reconcile_prior_boot_runs("new-boot")
        assert reconciled == []
        assert store.get_run_status("run_done_old")["status"] == "completed"
        assert store.get_journal_row("run_done_old")["settled"] is True

    def test_status_transitions_journal_immediately(self, adapter):
        store = adapter._response_store
        rid = "run_transitions"
        store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        adapter._set_run_status(rid, "running")
        assert store.get_journal_row(rid)["status"] == "running"
        # A transition inside the heartbeat window still writes immediately.
        adapter._set_run_status(
            rid, "waiting_for_approval", last_event="approval.request"
        )
        row = store.get_journal_row(rid)
        assert row["status"] == "waiting_for_approval"
        assert row["data"]["last_event"] == "approval.request"

    def test_heartbeats_are_throttled(self, adapter):
        """Repeat-status journal writes (tool lifecycle events) land at most
        once per heartbeat interval; the floor keeps journal I/O off the
        per-event hot path."""
        store = adapter._response_store
        rid = "run_heartbeat"
        store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        adapter._set_run_status(rid, "running", last_event="run.started")
        assert store.get_journal_row(rid)["data"]["last_event"] == "run.started"

        # Same status within the window: memory updates, journal does not.
        for i in range(5):
            adapter._set_run_status(rid, "running", last_event=f"tool.started:{i}")
        assert adapter._run_statuses[rid]["last_event"] == "tool.started:4"
        assert store.get_journal_row(rid)["data"]["last_event"] == "run.started"

        # Age the throttle window (no sleeping): the next event heartbeats.
        adapter._run_journal_heartbeat_at[rid] -= (
            adapter._RUN_JOURNAL_HEARTBEAT_SECONDS + 1
        )
        adapter._set_run_status(rid, "running", last_event="tool.completed")
        row = store.get_journal_row(rid)
        assert row["data"]["last_event"] == "tool.completed"
        assert row["data"]["heartbeat_at"] > 0

    def test_output_tail_is_bounded_and_flushed_on_heartbeat(self, adapter):
        store = adapter._response_store
        rid = "run_tail"
        store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        adapter._append_run_output_tail(rid, "x" * 12000)
        adapter._append_run_output_tail(rid, " the latest progress")
        assert (
            len(adapter._run_output_tails[rid])
            <= adapter._RUN_JOURNAL_OUTPUT_RAW_CHARS
        )
        adapter._set_run_status(rid, "running")  # transition flushes the tail
        tail = store.get_journal_row(rid)["data"]["output_tail"]
        assert tail.endswith("the latest progress")
        assert len(tail) <= adapter._RUN_JOURNAL_OUTPUT_TAIL_CHARS

    def test_journal_settlement_clears_throttle_state(self, adapter):
        rid = "run_cleanup"
        adapter._response_store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        adapter._set_run_status(rid, "running")
        adapter._append_run_output_tail(rid, "partial")
        adapter._set_run_status(rid, "completed", output="done")
        assert rid not in adapter._run_journal_heartbeat_at
        assert rid not in adapter._run_output_tails

    def test_late_event_cannot_reopen_settled_row(self, adapter):
        """A tool event landing after the terminal settle must not resurrect
        the journal row into a shape the next boot would re-fail."""
        store = adapter._response_store
        rid = "run_late_event"
        store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        adapter._set_run_status(rid, "completed", output="done")
        store.journal_run_update(rid, status="running", last_event="tool.started")
        row = store.get_journal_row(rid)
        assert row["settled"] is True
        assert row["status"] == "completed"

    def test_journal_capacity_evicts_settled_rows_first(self, adapter):
        """An unsettled row is the only record the next boot has of an
        in-flight run, so capacity pressure evicts settled rows first."""
        store = adapter._response_store
        store.MAX_STORED_RUN_STATUSES = 5
        now = time.time()
        for i in range(4):
            rid = f"run_settled_{i}"
            store.journal_run_submitted(rid, "boot-a", {"created_at": now})
            store.settle_run_status(rid, {"run_id": rid, "status": "completed"})
        for i in range(3):
            store.journal_run_submitted(
                f"run_open_{i}", "boot-a", {"status": "running", "created_at": now}
            )
        for i in range(3):
            assert store.get_journal_row(f"run_open_{i}") is not None
        remaining_settled = sum(
            1 for i in range(4) if store.get_journal_row(f"run_settled_{i}") is not None
        )
        assert remaining_settled == 2

    def test_unsettled_rows_survive_age_and_capacity_pruning(self, adapter):
        """A live run can go quiet longer than retention (an approval waiting
        on a human, a long external operation). Its unsettled row is the only
        record the next boot has of it, so neither age nor capacity may
        delete it — otherwise the gateway 404s a run it advertises as
        journaled."""
        store = adapter._response_store
        store.MAX_STORED_RUN_STATUSES = 2
        old = time.time() - store.RUN_STATUS_RETENTION_SECONDS - 3600
        store.journal_run_submitted(
            "run_quiet", "boot-a", {"status": "waiting_for_approval", "created_at": old}
        )
        with store._run_status_lock:
            store._conn.execute(
                "UPDATE run_journal SET updated_at = ? WHERE run_id = ?",
                (old, "run_quiet"),
            )
            store._conn.commit()
        store.journal_run_submitted(
            "run_settled_old", "boot-a", {"status": "queued", "created_at": old}
        )
        store.settle_run_status(
            "run_settled_old", {"run_id": "run_settled_old", "status": "completed"}
        )
        with store._run_status_lock:
            store._conn.execute(
                "UPDATE run_journal SET updated_at = ? WHERE run_id = ?",
                (old, "run_settled_old"),
            )
            store._conn.commit()

        # Aged out: only the settled row.
        store.journal_run_submitted(
            "run_new_1", "boot-a", {"status": "queued", "created_at": time.time()}
        )
        assert store.get_journal_row("run_settled_old") is None
        assert store.get_journal_row("run_quiet") is not None

        # Capacity is a bound on settled history: with nothing settled left
        # to evict it is exceeded rather than erasing live runs.
        for i in range(3):
            store.journal_run_submitted(
                f"run_new_{i + 2}",
                "boot-a",
                {"status": "running", "created_at": time.time()},
            )
        assert store.get_journal_row("run_quiet") is not None
        assert store.reconcile_prior_boot_runs("new-boot")

    def test_construction_alone_never_reconciles_live_runs(self, adapter):
        """A contender gateway constructs an adapter (and may then lose the
        port with EADDRINUSE) while the original keeps serving. Construction
        must not mark the live gateway's in-flight runs gateway_restart —
        that error is auto-resubmit-eligible, so it would duplicate tool side
        effects and credit spend on a run that is still executing."""
        store = adapter._response_store
        store.journal_run_submitted(
            "run_live_elsewhere",
            "live-boot",
            {"status": "running", "created_at": time.time()},
        )
        contender = _make_adapter()
        assert contender._response_store.get_run_status("run_live_elsewhere") is None
        assert store.get_journal_row("run_live_elsewhere")["settled"] is False
        assert contender._run_journal_ready is False

    def test_reconciliation_skipped_while_another_adapter_owns_the_store(self):
        """Two adapters, one shared response_store.db (multiplexed startup,
        reconnect churn): only the owner reconciles. The scoped lock is
        re-entrant per PID, so the in-process claim is what separates them."""
        first = _make_adapter()
        first._response_store.journal_run_submitted(
            "run_owned", "dead-boot", {"status": "running", "created_at": time.time()}
        )
        assert first._claim_run_journal_ownership() is True
        first._running = True  # connected: it is serving this store

        second = _make_adapter()
        assert second._reconcile_prior_boot_runs() is False
        assert second._run_journal_lock_identity is None
        assert first._response_store.get_journal_row("run_owned")["settled"] is False
        assert first._response_store.get_run_status("run_owned") is None

        # Ownership passes on once the owner disconnects.
        first._running = False
        first._release_run_journal_ownership()
        assert second._reconcile_prior_boot_runs() is True
        assert (
            second._response_store.get_run_status("run_owned")["error_code"]
            == "gateway_restart"
        )

    def test_memory_only_store_never_claims_the_journal_contract(self, adapter):
        """response_store.db can fail to open, and ResponseStore then falls
        back to :memory: — where every journal row dies with the process.
        The capability must report that honestly instead of telling the
        platform to stop treating a post-restart 404 as the lost-run
        signal."""
        from gateway.platforms.api_server import ResponseStore

        adapter._response_store = ResponseStore(db_path=":memory:")
        assert adapter._response_store.db_path is None
        assert adapter._reconcile_prior_boot_runs() is False
        adapter._run_journal_ready = adapter._reconcile_prior_boot_runs()
        assert adapter._run_journal_ready is False

    def test_failed_reconciliation_does_not_claim_the_journal_contract(self, adapter):
        """Reconciliation raising is logged, not fatal — but the boot has not
        reconciled, so it must not advertise the contract either."""
        with patch.object(
            adapter._response_store,
            "reconcile_prior_boot_runs",
            side_effect=RuntimeError("disk gone"),
        ):
            assert adapter._reconcile_prior_boot_runs() is False

    @pytest.mark.asyncio
    async def test_capabilities_reports_effective_journal_state(self, adapter):
        """The platform keys "gateway_restart failures are auto-resubmit-
        eligible, and 404-after-restart is no longer the signal" on this
        flag during the fleet roll, so it reports what this boot can
        actually deliver — false until reconciliation ran against a durable
        store."""
        app = _create_runs_app(adapter)
        app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            data = await resp.json()
            assert data["features"]["run_restart_journal"] is False

            adapter._run_journal_ready = adapter._reconcile_prior_boot_runs()
            assert adapter._run_journal_ready is True
            resp = await cli.get("/v1/capabilities")
            data = await resp.json()
            assert data["features"]["run_restart_journal"] is True

    def test_output_tail_redacts_credentials_straddling_the_boundary(self, adapter):
        """The journal's no-credentials contract must hold for a secret whose
        recognizable prefix sits ahead of the retained boundary: bounding raw
        text first would strip "-----BEGIN … KEY-----" and persist the
        remaining key bytes verbatim."""
        store = adapter._response_store
        rid = "run_secret_tail"
        store.journal_run_submitted(
            rid, adapter._boot_uuid, {"status": "queued", "created_at": time.time()}
        )
        key_body = "A" * 1800
        private_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + key_body
            + "\n-----END RSA PRIVATE KEY-----"
        )
        adapter._append_run_output_tail(rid, "streaming along " * 500)
        adapter._append_run_output_tail(rid, private_key)
        assert key_body not in adapter._run_output_tails[rid]

        adapter._set_run_status(rid, "running")
        tail = store.get_journal_row(rid)["data"]["output_tail"]
        assert key_body not in tail
        assert "PRIVATE KEY" in tail  # redacted marker, not the key material
        assert len(tail) <= adapter._RUN_JOURNAL_OUTPUT_TAIL_CHARS

    @pytest.mark.asyncio
    async def test_input_tail_redacts_credentials_straddling_the_boundary(self):
        """Same bug class on the sibling submission path: the input tail is
        redacted before it is bounded."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        key_body = "B" * 1800
        payload = (
            "deploy this key: -----BEGIN RSA PRIVATE KEY-----\n"
            + key_body
            + "\n-----END RSA PRIVATE KEY-----"
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent
                resp = await cli.post("/v1/runs", json={"input": payload})
                run_id = (await resp.json())["run_id"]
                agent_ready.wait(timeout=3.0)

                data = adapter._response_store.get_journal_row(run_id)["data"]
                assert key_body not in data["input_tail"]
                assert key_body not in json.dumps(data)
                assert len(data["input_tail"]) <= 200

                interrupted.set()
                await _wait_terminal(cli, run_id)

    @pytest.mark.asyncio
    async def test_streaming_deltas_heartbeat_the_output_tail(self):
        """A text-only response emits no status event after the `running`
        transition, so the deltas themselves must drive the throttled
        heartbeat — otherwise a preemption mid-stream leaves the journal with
        none of the partial output this feature promises survives."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        streaming = threading.Event()
        release = threading.Event()

        def _streaming_agent(**kwargs):
            delta_cb = kwargs.get("stream_delta_callback")
            mock_agent = MagicMock()

            def _run(user_message=None, conversation_history=None, task_id=None):
                delta_cb("partial progress ")
                streaming.set()
                release.wait(timeout=10)
                delta_cb("and more")
                return {"final_response": "partial progress and more"}

            mock_agent.run_conversation.side_effect = _run
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_streaming_agent):
                resp = await cli.post("/v1/runs", json={"input": "write me an essay"})
                run_id = (await resp.json())["run_id"]
                assert streaming.wait(timeout=3.0)
                # The `running` transition just wrote, so the first delta is
                # inside the throttle window. Age it the way a long stream
                # does (no sleeping) and let the next delta flush.
                adapter._run_journal_heartbeat_at[run_id] -= (
                    adapter._RUN_JOURNAL_HEARTBEAT_SECONDS + 1
                )
                release.set()
                status = await _wait_terminal(cli, run_id)
                assert status["status"] == "completed"

        row = adapter._response_store.get_journal_row(run_id)
        assert "partial progress" in row["data"]["output_tail"]
