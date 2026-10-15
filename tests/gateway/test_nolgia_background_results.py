"""Nolgia background results (Nolgia fork): a finished background job reaches the customer's chat.

On an ``api_server`` session the gateway's own handling of a background result (a delegation
delivery row in this pod's store, or a self-posted wake turn) never reaches a Nolgia customer:
the platform relay shows only the reply of the run the customer sent. On a platform pod the
result is handed to ``POST /v1/agent/background-results`` instead, which files it in the chat
and runs the turn that reports it. These tests pin who is filed, with which credential and
session, what retries, and that everything else keeps the gateway's own handling.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms import nolgia_background_results as results
from gateway.platforms.api_server import ResponseStore
from gateway.run import GatewayRunner

SESSION = "0f0e7a3c-5a74-4c86-9b1d-2f6f3c9d8e11"


class _Platform:
    """A fake nolgia-api: records each POST and answers the next queued status (default 202)."""

    def __init__(self):
        self.requests = []
        self.statuses = []
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                platform.requests.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization", ""),
                    "body": json.loads(self.rfile.read(length) or b"{}"),
                })
                status = platform.statuses.pop(0) if platform.statuses else 202
                payload = json.dumps({"user_message_id": "u", "delivery": "turn", "replayed": False}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def platform(monkeypatch):
    fake = _Platform()
    monkeypatch.setenv("NOLGIA_API_URL", fake.url)
    monkeypatch.setenv("NOLGIA_TOKEN", "nolp_pod")
    monkeypatch.delenv("NOLGIA_BACKGROUND_RESULTS", raising=False)
    results._failures.clear()
    yield fake
    results._failures.clear()
    fake.close()


def _adapter_with_run(run_id="run_abc", workspace_id=SESSION, session_id=None):
    store = ResponseStore(max_size=10, db_path=":memory:")
    seed = {"status": "completed", "created_at": 1.0, "session_id": session_id or run_id}
    if workspace_id:
        seed["workspace_id"] = workspace_id
    store.journal_run_submitted(run_id, "boot", seed)
    return SimpleNamespace(_response_store=store)


def _process_exit(**extra):
    return {"type": "completion", "session_id": "proc_render", "started_at": 1760425100.5, **extra}


def _delegation(**extra):
    return {"type": "async_delegation", "delegation_id": "deleg_42", "status": "completed", **extra}


def test_process_exit_is_filed_with_the_pod_credential_and_journaled_session(platform):
    outcome = results.file_result(_adapter_with_run(), "Render finished: 8 clips.", "run_abc", _process_exit())

    assert outcome == results.FILED
    assert len(platform.requests) == 1
    request = platform.requests[0]
    assert request["path"] == "/v1/agent/background-results"
    assert request["authorization"] == "Bearer nolp_pod"
    assert request["body"] == {
        "kind": "process", "text": "Render finished: 8 clips.",
        "source_id": "proc_render@1760425100.5", "session_id": SESSION,
    }


def test_delegation_uses_the_originating_run_credential_first(platform):
    evt = _delegation(origin_nolgia_token="nolt_turn")
    outcome = results.file_result(_adapter_with_run(), "Eight step times.", "run_abc", evt)

    assert outcome == results.FILED
    assert [r["authorization"] for r in platform.requests] == ["Bearer nolt_turn"]
    assert platform.requests[0]["body"]["kind"] == "delegation"
    assert platform.requests[0]["body"]["source_id"] == "deleg_42"


def test_expired_run_credential_falls_back_to_the_pod_credential(platform):
    platform.statuses = [401, 202]
    evt = _delegation(origin_nolgia_token="nolt_expired")
    outcome = results.file_result(_adapter_with_run(), "Eight step times.", "run_abc", evt)

    assert outcome == results.FILED
    assert [r["authorization"] for r in platform.requests] == ["Bearer nolt_expired", "Bearer nolp_pod"]
    assert platform.requests[1]["body"]["session_id"] == SESSION


def test_run_credential_alone_needs_no_journaled_session(platform):
    evt = _delegation(origin_nolgia_token="nolt_turn")
    outcome = results.file_result(_adapter_with_run(workspace_id=""), "Done.", "run_abc", evt)

    assert outcome == results.FILED
    assert "session_id" not in platform.requests[0]["body"]


def test_session_found_through_the_journaled_session_id(platform):
    adapter = _adapter_with_run(run_id="run_xyz", session_id="hq-session")
    outcome = results.file_result(adapter, "Done.", "hq-session", _process_exit())

    assert outcome == results.FILED
    assert platform.requests[0]["body"]["session_id"] == SESSION


@pytest.mark.parametrize("status", [404, 400, 403])
def test_a_refusal_keeps_the_gateways_own_handling(platform, status):
    platform.statuses = [status]
    outcome = results.file_result(_adapter_with_run(), "Done.", "run_abc", _process_exit())

    assert outcome == results.NOT_FILED
    assert len(platform.requests) == 1


@pytest.mark.parametrize("status", [500, 503, 429])
def test_a_transient_failure_retries_with_backoff_then_gives_up(platform, monkeypatch, status):
    clock = [1000.0]
    monkeypatch.setattr(results.time, "monotonic", lambda: clock[0])
    platform.statuses = [status] * 10
    adapter = _adapter_with_run()

    assert results.file_result(adapter, "Done.", "run_abc", _process_exit()) == results.RETRY
    assert len(platform.requests) == 1
    # Inside the backoff window nothing is sent; the caller just keeps the event.
    clock[0] += 1
    assert results.file_result(adapter, "Done.", "run_abc", _process_exit()) == results.RETRY
    assert len(platform.requests) == 1
    clock[0] += results._BACKOFF_START_SECONDS
    assert results.file_result(adapter, "Done.", "run_abc", _process_exit()) == results.RETRY
    assert len(platform.requests) == 2
    # Past the give-up bound the result falls back to the gateway's own delivery.
    clock[0] += results._GIVE_UP_AFTER_SECONDS
    assert results.file_result(adapter, "Done.", "run_abc", _process_exit()) == results.NOT_FILED
    assert len(platform.requests) == 3


def test_platform_unreachable_is_transient(monkeypatch):
    monkeypatch.setenv("NOLGIA_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("NOLGIA_TOKEN", "nolp_pod")
    results._failures.clear()
    try:
        assert results.file_result(_adapter_with_run(), "Done.", "run_abc", _process_exit()) == results.RETRY
    finally:
        results._failures.clear()


@pytest.mark.parametrize("evt", [
    {"type": "watch_match", "session_id": "proc_render"},
    {"type": "watch_disabled", "session_id": "proc_render"},
    _delegation(task_failure_notice=True),
    {"type": "async_delegation", "status": "completed"},  # no delegation id
    {"type": "completion"},  # no process id
])
def test_only_final_results_are_filed(platform, evt):
    assert results.file_result(_adapter_with_run(), "Matched a line.", "run_abc", evt) == results.NOT_FILED
    assert platform.requests == []


def test_no_session_and_no_run_credential_is_not_filed(platform):
    outcome = results.file_result(_adapter_with_run(workspace_id=""), "Done.", "run_abc", _process_exit())
    assert outcome == results.NOT_FILED
    assert platform.requests == []


def test_a_workspace_that_is_not_a_platform_session_is_ignored(platform):
    outcome = results.file_result(_adapter_with_run(workspace_id="session-local"), "Done.", "run_abc", _process_exit())
    assert outcome == results.NOT_FILED
    assert platform.requests == []


def test_enabled_needs_platform_credentials_and_honors_the_escape_hatch(monkeypatch):
    monkeypatch.delenv("NOLGIA_API_URL", raising=False)
    monkeypatch.setenv("NOLGIA_TOKEN", "nolp_pod")
    assert not results.enabled()
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.ai")
    assert results.enabled()
    monkeypatch.setenv("NOLGIA_BACKGROUND_RESULTS", "0")
    assert not results.enabled()


def test_long_results_keep_their_start_and_end():
    text = "START " + "x" * 50000 + " END"
    bounded = results.bounded_text(text)
    assert len(bounded) <= results._MAX_TEXT_CHARS
    assert bounded.startswith("START ")
    assert bounded.endswith(" END")
    assert "characters cut" in bounded
    assert results.bounded_text("  short  ") == "short"


def test_journal_workspace_id_reads_settled_runs():
    store = ResponseStore(max_size=10, db_path=":memory:")
    store.journal_run_submitted("run_1", "boot", {"status": "queued", "created_at": 1.0, "session_id": "run_1", "workspace_id": SESSION})
    store.settle_run_status("run_1", {"status": "completed"})
    assert store.journal_workspace_id("run_1") == SESSION
    assert store.journal_workspace_id("run_missing") == ""
    assert store.journal_workspace_id("") == ""


# ---------------------------------------------------------------------------
# Through the gateway's notification path
# ---------------------------------------------------------------------------

class _AdmittingHandler(AsyncMock):
    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


def _runner_with_api_server(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: all\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = _adapter_with_run(run_id="raw-run-sid")
    adapter.supports_async_delivery = False
    adapter.handle_message = _AdmittingHandler()
    adapter._host, adapter._port, adapter._api_key, adapter._model_name = "127.0.0.1", 8642, "k", "m"
    runner.adapters[Platform.API_SERVER] = adapter
    return runner, adapter


def _recording_wake(monkeypatch):
    import gateway.wake as wake_mod
    posts, persisted = [], []

    async def fake_self_post(adapter, *, text, session_id, **_kw):
        posts.append(session_id)

    async def fake_persist(adapter, *, text, session_id, evt=None, profile=""):
        persisted.append(session_id)

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    monkeypatch.setattr(wake_mod, "persist_delegation_delivery", fake_persist)
    return posts, persisted


@pytest.mark.asyncio
async def test_platform_pod_files_the_result_instead_of_waking_the_session(platform, monkeypatch, tmp_path):
    runner, adapter = _runner_with_api_server(monkeypatch, tmp_path)
    posts, persisted = _recording_wake(monkeypatch)

    evt = {**_process_exit(), "session_key": "raw-run-sid"}
    assert await runner._inject_watch_notification("[SYSTEM: render finished]", evt) is True

    assert posts == [] and persisted == []
    adapter.handle_message.assert_not_awaited()
    assert platform.requests[0]["body"]["session_id"] == SESSION
    assert platform.requests[0]["body"]["text"] == "[SYSTEM: render finished]"


@pytest.mark.asyncio
async def test_route_off_keeps_the_delegation_delivery_row(platform, monkeypatch, tmp_path):
    runner, _adapter = _runner_with_api_server(monkeypatch, tmp_path)
    posts, persisted = _recording_wake(monkeypatch)
    platform.statuses = [404]

    evt = {**_delegation(), "session_key": "raw-run-sid", "origin_session_id": "raw-run-sid"}
    assert await runner._inject_watch_notification("[ASYNC DELEGATION COMPLETE — deleg_42]", evt) is True

    assert len(platform.requests) == 1
    assert persisted == ["raw-run-sid"] and posts == []


@pytest.mark.asyncio
async def test_transient_platform_failure_is_retryable_not_a_wake(platform, monkeypatch, tmp_path):
    runner, _adapter = _runner_with_api_server(monkeypatch, tmp_path)
    posts, persisted = _recording_wake(monkeypatch)
    platform.statuses = [503]

    evt = {**_process_exit(), "session_key": "raw-run-sid"}
    assert await runner._inject_watch_notification("[SYSTEM: render finished]", evt) is False
    assert posts == [] and persisted == []


@pytest.mark.asyncio
async def test_without_platform_credentials_nothing_changes(monkeypatch, tmp_path):
    monkeypatch.delenv("NOLGIA_API_URL", raising=False)
    monkeypatch.delenv("NOLGIA_TOKEN", raising=False)
    runner, _adapter = _runner_with_api_server(monkeypatch, tmp_path)
    posts, persisted = _recording_wake(monkeypatch)

    evt = {**_process_exit(), "session_key": "raw-run-sid"}
    assert await runner._inject_watch_notification("[SYSTEM: render finished]", evt) is True
    assert posts == ["raw-run-sid"] and persisted == []
