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
                # Durable: nothing — a restarted gateway must 404 this run.
                assert adapter._response_store.get_run_status(run_id) is None
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
