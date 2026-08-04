"""Run-scoped Nolgia platform token (turn-scoped attribution, Nolgia fork).

``POST /v1/runs`` may carry a ``nolgia_token``: the run binds it as the
``HERMES_SESSION_NOLGIA_TOKEN`` session context var, so the run's platform
egress (asset uploads in ``gateway.platforms.nolgia_assets``) and its tool
subprocesses (the ``nolgia`` CLI, via the session-context env bridge)
authenticate as the exact turn that caused them. That is what lets the
Nolgia platform execute one user's turns CONCURRENTLY without
cross-attributing their output. An absent/empty token keeps the pod-wide
``NOLGIA_TOKEN`` behavior byte-for-byte.

The leak-safety properties under test mirror the existing
``HERMES_SESSION_*`` machinery deliberately:
- task-scoped ContextVar, cleared on run exit (no reused-executor-thread
  bleed);
- ``_UNSET``-strip in the subprocess bridge (no sibling-run inheritance);
- excluded from the shared bash snapshot (all concurrent runs share ONE
  shell environment, and ``export -p`` would otherwise hand one run's
  credential to every other run's later commands).
"""

import re
import threading
import time
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway import session_context  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms import nolgia_assets  # noqa: E402
from gateway.platforms.api_server import (  # noqa: E402
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.session_context import (  # noqa: E402
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    set_session_vars,
)
from tools.environments import base as env_base  # noqa: E402
from tools.environments.local import _inject_session_context_env  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_session_context():
    """Every test starts and ends with unbound session vars."""
    reset_session_vars()
    yield
    reset_session_vars()


@pytest.fixture(autouse=True)
def _clear_asset_cache():
    with nolgia_assets._asset_cache_lock:
        nolgia_assets._asset_cache.clear()
    yield
    with nolgia_assets._asset_cache_lock:
        nolgia_assets._asset_cache.clear()


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


async def _wait_completed(cli, run_id, tries=100, headers=None):
    for _ in range(tries):
        resp = await cli.get(f"/v1/runs/{run_id}", headers=headers or {})
        data = await resp.json()
        if data["status"] in {"completed", "failed", "cancelled"}:
            return data
        import asyncio

        await asyncio.sleep(0.02)
    raise AssertionError("run never settled")


class TestSessionContextVar:
    def test_set_and_clear_roundtrip(self):
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_abc")
        assert get_session_env("HERMES_SESSION_NOLGIA_TOKEN") == "nolt_abc"
        clear_session_vars(tokens)
        assert get_session_env("HERMES_SESSION_NOLGIA_TOKEN") == ""

    def test_default_binding_is_empty(self):
        tokens = set_session_vars(platform="api_server")
        assert get_session_env("HERMES_SESSION_NOLGIA_TOKEN") == ""
        clear_session_vars(tokens)


class TestPlatformTokenResolution:
    def test_prefers_run_scoped_token(self, monkeypatch):
        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert nolgia_assets._platform_token() == "nolt_run"
        finally:
            clear_session_vars(tokens)

    def test_falls_back_to_pod_env(self, monkeypatch):
        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        tokens = set_session_vars(platform="api_server")  # bound, but no run token
        try:
            assert nolgia_assets._platform_token() == "pod-wide"
        finally:
            clear_session_vars(tokens)

    def test_http_json_authorization_carries_scoped_token(self, monkeypatch):
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        captured = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        def _fake_urlopen(request, timeout=None):
            captured["auth"] = request.headers.get("Authorization")
            return _FakeResponse()

        monkeypatch.setattr(nolgia_assets.urllib.request, "urlopen", _fake_urlopen)
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            nolgia_assets._http_json("POST", "/assets/uploads", {"x": 1}, 5.0)
        finally:
            clear_session_vars(tokens)
        assert captured["auth"] == "Bearer nolt_run"

    def test_asset_cache_is_keyed_by_token(self, monkeypatch, tmp_path):
        """Run B must never reuse an asset uploaded under run A's credential:
        the asset carries A's turn attribution."""
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        calls = []

        def _fake_upload(path, content_type, size_bytes, deadline):
            calls.append(str(path))
            return f"asset-{len(calls)}"

        monkeypatch.setattr(nolgia_assets, "_upload_asset", _fake_upload)
        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)

        budget = nolgia_assets._MessageBudget()
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_a")
        try:
            first = nolgia_assets._resolve_tag(str(png), budget)
            again = nolgia_assets._resolve_tag(str(png), budget)
        finally:
            clear_session_vars(tokens)
        assert first == "asset:asset-1"
        assert again == "asset:asset-1", "same run reuses its own upload"

        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_b")
        try:
            other = nolgia_assets._resolve_tag(str(png), nolgia_assets._MessageBudget())
        finally:
            clear_session_vars(tokens)
        assert other == "asset:asset-2", "a different run's token re-uploads"
        assert len(calls) == 2


class TestSubprocessBridge:
    def test_bound_token_overrides_child_nolgia_token(self):
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            env = {"NOLGIA_TOKEN": "pod-wide"}
            _inject_session_context_env(env)
        finally:
            clear_session_vars(tokens)
        assert env["HERMES_SESSION_NOLGIA_TOKEN"] == "nolt_run"
        assert env["NOLGIA_TOKEN"] == "nolt_run"

    def test_unbound_run_keeps_pod_wide_token(self):
        tokens = set_session_vars(platform="api_server")  # engaged, no run token
        try:
            env = {"NOLGIA_TOKEN": "pod-wide"}
            _inject_session_context_env(env)
        finally:
            clear_session_vars(tokens)
        assert env["NOLGIA_TOKEN"] == "pod-wide"

    def test_sibling_task_strips_scoped_var_and_keeps_pod_token(self):
        # A task that never bound (engaged process, _UNSET context) must not
        # inherit another run's scoped var — and must keep the pod bearer.
        set_session_vars(platform="api_server", nolgia_token="nolt_run")
        reset_session_vars()  # simulate the sibling task's pre-bind reset
        env = {
            "NOLGIA_TOKEN": "pod-wide",
            "HERMES_SESSION_NOLGIA_TOKEN": "leaked-from-globals",
        }
        _inject_session_context_env(env)
        assert "HERMES_SESSION_NOLGIA_TOKEN" not in env
        assert env["NOLGIA_TOKEN"] == "pod-wide"

    def test_snapshot_excludes_nolgia_token(self):
        pattern = re.compile(env_base._SNAPSHOT_EXCLUDED_ENV_REGEX)
        assert pattern.match('declare -x NOLGIA_TOKEN="nolt_run"')
        assert pattern.match('declare -x HERMES_SESSION_NOLGIA_TOKEN="nolt_run"')
        # Unrelated exports (including the pod URL) still snapshot normally.
        assert not pattern.match('declare -x NOLGIA_API_URL="https://api"')
        assert not pattern.match('declare -x PATH="/usr/bin"')


class TestRunsEndpoint:
    @pytest.mark.asyncio
    async def test_malformed_token_is_rejected(self, monkeypatch):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for bad in [123, "tok\nen", "x" * 600]:
                resp = await cli.post("/v1/runs", json={"input": "hi", "nolgia_token": bad})
                assert resp.status == 400, f"expected 400 for {bad!r}"

    @pytest.mark.asyncio
    async def test_run_binds_token_for_agent_and_clears_after(self):
        """The agent's conversation executes with the run's token bound in its
        context (executor thread), and a later token-less run on the same
        process observes NOTHING — the reused-thread leak guard."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        seen = []

        def _observing_run(user_message=None, conversation_history=None, task_id=None):
            seen.append(get_session_env("HERMES_SESSION_NOLGIA_TOKEN"))
            return {"final_response": "done"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.side_effect = _observing_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs", json={"input": "hello", "nolgia_token": "nolt_run_1"}
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_completed(cli, run_id)

                resp = await cli.post("/v1/runs", json={"input": "hello again"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_completed(cli, run_id)

        assert seen == ["nolt_run_1", ""]

    @pytest.mark.asyncio
    async def test_capabilities_advertises_nolgia_run_token(self):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["nolgia_run_token"] is True


# ---------------------------------------------------------------------------
# NOL-413 regressions: every in-run CLI invocation carries the turn credential
# ---------------------------------------------------------------------------


def _simulated_cli_spawn_env() -> dict:
    """Build a tool-subprocess env exactly the way the terminal bridge does.

    ``{"NOLGIA_TOKEN": "pod-wide"}`` stands in for the pod deployment env; the
    real ``_inject_session_context_env`` then applies the task-local session
    context, so the returned ``NOLGIA_TOKEN`` is what a spawned ``nolgia
    assets upload`` would authenticate with.
    """
    env = {"NOLGIA_TOKEN": "pod-wide"}
    _inject_session_context_env(env)
    return env


class TestConcurrentRunsSequentialUploads:
    @pytest.mark.asyncio
    async def test_every_upload_carries_its_own_runs_credential(self):
        """Two OVERLAPPING /v1/runs, each performing three sequential CLI
        'uploads': every spawn env must carry its own run's token — never the
        sibling's, never the pod-wide bearer (NOL-413)."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        # Lockstep barrier: each upload round happens only when BOTH runs are
        # mid-turn, so a per-run binding that leaked through any process-global
        # slot would be caught (last-writer-wins would poison the sibling).
        barrier = threading.Barrier(2)
        recorded: dict = defaultdict(list)
        rec_lock = threading.Lock()

        def _observing_run(user_message=None, conversation_history=None, task_id=None):
            sid = get_session_env("HERMES_SESSION_ID")
            for _ in range(3):
                barrier.wait(timeout=15)
                env = _simulated_cli_spawn_env()
                with rec_lock:
                    recorded[sid].append(env.get("NOLGIA_TOKEN"))
            return {"final_response": "done"}

        def _fake_create_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _observing_run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                resp_a = await cli.post(
                    "/v1/runs",
                    json={"input": "a", "session_id": "sess-a", "nolgia_token": "nolt_run_a"},
                )
                resp_b = await cli.post(
                    "/v1/runs",
                    json={"input": "b", "session_id": "sess-b", "nolgia_token": "nolt_run_b"},
                )
                assert resp_a.status == 202 and resp_b.status == 202
                run_a = (await resp_a.json())["run_id"]
                run_b = (await resp_b.json())["run_id"]
                await _wait_completed(cli, run_a, tries=500)
                await _wait_completed(cli, run_b, tries=500)

        assert recorded["sess-a"] == ["nolt_run_a"] * 3
        assert recorded["sess-b"] == ["nolt_run_b"] * 3


class TestWakeContinuationRetainedToken:
    """Pod-internal continuation turns (gateway.wake's self-POST to
    /v1/chat/completions after a background delegation completes) carry no
    request credential. The adapter must re-bind the session's RETAINED run
    token so the continuation's uploads still attribute to the causing turn —
    the NOL-413 'sibling upload lost the credential' hole."""

    @pytest.mark.asyncio
    async def test_wake_selfpost_rebinds_the_sessions_run_token(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_runs_app(adapter)
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        seen: dict = {}
        auth = {"Authorization": "Bearer sk-secret"}

        def _observing_run(user_message=None, conversation_history=None, task_id=None, **_kw):
            seen["bound_token"] = get_session_env("HERMES_SESSION_NOLGIA_TOKEN")
            seen["child_env_token"] = _simulated_cli_spawn_env().get("NOLGIA_TOKEN")
            return {"final_response": "done"}

        def _fake_create_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _observing_run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            agent.session_id = "sess-wake"
            agent._last_compaction_in_place = False
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                # 1. Platform-submitted run binds AND retains the turn token.
                resp = await cli.post(
                    "/v1/runs",
                    headers=auth,
                    json={"input": "go", "session_id": "sess-wake", "nolgia_token": "nolt_wake"},
                )
                assert resp.status == 202
                await _wait_completed(cli, (await resp.json())["run_id"], headers=auth)
                assert seen == {
                    "bound_token": "nolt_wake",
                    "child_env_token": "nolt_wake",
                }
                seen.clear()

                # 2. Wake continuation: same session id, NO token in the
                #    request — exactly gateway.wake's self-post shape.
                adapter._session_db = MagicMock(
                    get_messages_as_conversation=MagicMock(return_value=[])
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "sess-wake", **auth},
                    json={"messages": [{"role": "user", "content": "delegation finished"}]},
                )
                assert resp.status == 200

        assert seen == {
            "bound_token": "nolt_wake",
            "child_env_token": "nolt_wake",
        }

    @pytest.mark.asyncio
    async def test_foreign_session_continuation_stays_pod_scoped(self):
        """A continuation for a session that never rode /v1/runs must keep
        today's behavior byte-for-byte: nothing bound, pod bearer for children."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_runs_app(adapter)
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        seen: dict = {}

        def _observing_run(user_message=None, conversation_history=None, task_id=None, **_kw):
            seen["bound_token"] = get_session_env("HERMES_SESSION_NOLGIA_TOKEN")
            seen["child_env_token"] = _simulated_cli_spawn_env().get("NOLGIA_TOKEN")
            return {"final_response": "done"}

        def _fake_create_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _observing_run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            agent.session_id = "sess-other"
            agent._last_compaction_in_place = False
            return agent

        adapter._session_db = MagicMock(
            get_messages_as_conversation=MagicMock(return_value=[])
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Hermes-Session-Id": "sess-other",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status == 200

        assert seen["bound_token"] == ""
        assert seen["child_env_token"] == "pod-wide"

    @pytest.mark.asyncio
    async def test_wake_selfpost_prefers_the_originating_runs_credential(self):
        """A wake carries the credential of the run whose detached delegation
        it reports. A NEWER turn on the same session must not re-attribute the
        continuation's uploads to itself, so the request credential wins over
        the session's retained one."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_runs_app(adapter)
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        seen: dict = {}

        def _observing_run(user_message=None, conversation_history=None, task_id=None, **_kw):
            seen["bound_token"] = get_session_env("HERMES_SESSION_NOLGIA_TOKEN")
            seen["child_env_token"] = _simulated_cli_spawn_env().get("NOLGIA_TOKEN")
            return {"final_response": "done"}

        def _fake_create_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _observing_run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            agent.session_id = "sess-wake"
            agent._last_compaction_in_place = False
            return agent

        adapter._session_db = MagicMock(
            get_messages_as_conversation=MagicMock(return_value=[])
        )
        # A later turn on this session has already overwritten the retained
        # token; the wake must NOT pick that one up.
        adapter._retain_session_run_token("sess-wake", "nolt_newer_turn")
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Hermes-Session-Id": "sess-wake",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={
                        "messages": [{"role": "user", "content": "delegation finished"}],
                        "nolgia_token": "nolt_origin_run",
                    },
                )
                assert resp.status == 200

        assert seen == {
            "bound_token": "nolt_origin_run",
            "child_env_token": "nolt_origin_run",
        }

    @pytest.mark.asyncio
    async def test_continuation_rejects_a_malformed_credential(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_runs_app(adapter)
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        async with TestClient(TestServer(app)) as cli:
            for bad in [123, "tok\nen", "x" * 600]:
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "nolgia_token": bad,
                    },
                )
                assert resp.status == 400, f"expected 400 for {bad!r}"


class TestRetentionLifecycle:
    """Retention is bounded by the originating run's lifecycle (holders) plus a
    settle grace, and scoped by profile — never by a fixed window from request
    receipt, and never by session id alone."""

    @staticmethod
    def _holders(adapter, key) -> dict:
        return {token: holders for token, _deadline, holders in adapter._session_run_tokens[key]}

    @staticmethod
    def _expire(adapter, key, *tokens) -> None:
        """Age out the named entries' graces (all of them when none given)."""
        adapter._session_run_tokens[key] = [
            (token, time.monotonic() - 1 if not tokens or token in tokens else deadline, holders)
            for token, deadline, holders in adapter._session_run_tokens[key]
        ]

    def test_grace_and_newest_submission_wins(self):
        adapter = _make_adapter()
        key = adapter._session_run_token_key("sess-1")
        adapter._retain_session_run_token("sess-1", "nolt_first")
        assert adapter._retained_session_run_token("sess-1") == "nolt_first"
        # A newer run for the same session takes over the fallback — a
        # continuation that carries no credential of its own follows the most
        # recent submission.
        adapter._retain_session_run_token("sess-1", "nolt_second")
        assert adapter._retained_session_run_token("sess-1") == "nolt_second"
        # Both runs settle; the entries then age out on the grace and do not
        # survive the next write's prune.
        adapter._release_session_run_token("sess-1", "nolt_first")
        adapter._release_session_run_token("sess-1", "nolt_second")
        assert self._holders(adapter, key) == {"nolt_first": 0, "nolt_second": 0}
        self._expire(adapter, key)
        assert adapter._retained_session_run_token("sess-1") == ""
        adapter._retain_session_run_token("sess-2", "nolt_other")
        assert key not in adapter._session_run_tokens
        # Empty ids/tokens are never retained.
        adapter._retain_session_run_token("", "nolt_x")
        adapter._retain_session_run_token("sess-3", "")
        assert adapter._session_run_token_key("") not in adapter._session_run_tokens
        assert adapter._session_run_token_key("sess-3") not in adapter._session_run_tokens

    def test_live_run_keeps_its_credential_past_the_grace(self):
        """A foreground /v1/runs turn can legitimately execute for longer than
        any fixed window. While the run still HOLDS the entry, an elapsed
        deadline must not expire it."""
        adapter = _make_adapter()
        key = adapter._session_run_token_key("sess-long")
        adapter._retain_session_run_token("sess-long", "nolt_long")
        # Simulate a run executing far longer than the settle grace.
        self._expire(adapter, key)
        assert self._holders(adapter, key) == {"nolt_long": 1}
        assert adapter._retained_session_run_token("sess-long") == "nolt_long"
        # A sibling session's write must not prune a held entry either.
        adapter._retain_session_run_token("sess-other", "nolt_other")
        assert key in adapter._session_run_tokens
        # Releasing the holder restarts the grace, then it ages out.
        adapter._release_session_run_token("sess-long", "nolt_long")
        assert adapter._retained_session_run_token("sess-long") == "nolt_long"
        assert self._holders(adapter, key) == {"nolt_long": 0}
        self._expire(adapter, key)
        assert adapter._retained_session_run_token("sess-long") == ""

    def test_release_settles_only_the_owning_runs_entry(self):
        """Holder counts belong to the run that retained the credential. A
        short run settling first must not decrement (or extend) a concurrent
        sibling run's entry — releasing by key alone would hand its hold to
        whichever token the key happened to point at."""
        adapter = _make_adapter()
        key = adapter._session_run_token_key("sess-two")
        adapter._retain_session_run_token("sess-two", "nolt_short")
        adapter._retain_session_run_token("sess-two", "nolt_long")
        adapter._release_session_run_token("sess-two", "nolt_short")
        assert self._holders(adapter, key) == {"nolt_short": 0, "nolt_long": 1}
        # Once the settled run's grace elapses, the still-executing sibling's
        # credential is the live answer — not the aged-out newer/older entry.
        self._expire(adapter, key, "nolt_short")
        assert adapter._retained_session_run_token("sess-two") == "nolt_long"
        # Releasing the sibling settles it too; the already-aged-out entry is
        # pruned by that write rather than lingering in the map.
        adapter._release_session_run_token("sess-two", "nolt_long")
        assert self._holders(adapter, key) == {"nolt_long": 0}

    def test_same_credential_resubmitted_shares_one_entry(self):
        """A retry/duplicate run of ONE platform turn holds the same entry, so
        the first release cannot settle it out from under the second run."""
        adapter = _make_adapter()
        key = adapter._session_run_token_key("sess-retry")
        adapter._retain_session_run_token("sess-retry", "nolt_same")
        adapter._retain_session_run_token("sess-retry", "nolt_same")
        assert self._holders(adapter, key) == {"nolt_same": 2}
        adapter._release_session_run_token("sess-retry", "nolt_same")
        assert self._holders(adapter, key) == {"nolt_same": 1}
        self._expire(adapter, key)
        assert adapter._retained_session_run_token("sess-retry") == "nolt_same"
        adapter._release_session_run_token("sess-retry", "nolt_same")
        self._expire(adapter, key)
        assert adapter._retained_session_run_token("sess-retry") == ""

    def test_default_profile_spellings_are_one_scope(self):
        """The default profile has two wire spellings — unprefixed ``/v1/...``
        resolves ``""`` while ``/p/default/v1/...`` resolves ``"default"`` —
        addressing ONE runtime. A run submitted through either alias must be
        recoverable through the other (the internal wake self-post falls back
        to the unprefixed route), so both canonicalize to one key."""
        adapter = _make_adapter()
        assert (
            adapter._session_run_token_key("sess-d", "default")
            == adapter._session_run_token_key("sess-d", "")
        )
        adapter._retain_session_run_token("sess-d", "nolt_default", profile="default")
        assert adapter._retained_session_run_token("sess-d") == "nolt_default"
        assert adapter._retained_session_run_token("sess-d", profile="default") == "nolt_default"
        # Still not a catch-all: a named profile keeps its own scope.
        assert adapter._retained_session_run_token("sess-d", profile="alpha") == ""
        # Release through the other spelling reaches the same entry.
        adapter._release_session_run_token("sess-d", "nolt_default", profile="")
        key = adapter._session_run_token_key("sess-d")
        assert self._holders(adapter, key) == {"nolt_default": 0}

    def test_retention_is_scoped_by_profile(self):
        """In multiplex mode every /p/<profile>/... route shares one adapter
        while session ids are caller-controlled: a request in profile B must
        never resolve the credential profile A retained under the same id."""
        adapter = _make_adapter()
        adapter._retain_session_run_token("sess-x", "nolt_profile_a", profile="alpha")
        assert adapter._retained_session_run_token("sess-x", profile="alpha") == "nolt_profile_a"
        assert adapter._retained_session_run_token("sess-x", profile="beta") == ""
        # The default profile is its own scope, not a catch-all.
        assert adapter._retained_session_run_token("sess-x") == ""
        adapter._retain_session_run_token("sess-x", "nolt_profile_b", profile="beta")
        assert adapter._retained_session_run_token("sess-x", profile="alpha") == "nolt_profile_a"
        assert adapter._retained_session_run_token("sess-x", profile="beta") == "nolt_profile_b"
        # Releasing one profile's holder leaves the other's entry untouched.
        adapter._release_session_run_token("sess-x", "nolt_profile_b", profile="beta")
        assert adapter._retained_session_run_token("sess-x", profile="alpha") == "nolt_profile_a"


class TestDelegationOriginCredential:
    """The dispatch-time credential rides the IN-MEMORY completion event (never
    the persisted event_json) so the wake it triggers attributes to the run that
    spawned the delegation."""

    def test_dispatch_reads_the_bound_run_credential(self):
        from tools.async_delegation import _current_origin_nolgia_token

        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_origin_run")
        try:
            assert _current_origin_nolgia_token() == "nolt_origin_run"
        finally:
            clear_session_vars(tokens)
        assert _current_origin_nolgia_token() == ""

    def test_non_api_server_platform_has_no_run_credential(self):
        from tools.async_delegation import _current_origin_nolgia_token

        tokens = set_session_vars(platform="api_server")
        try:
            assert _current_origin_nolgia_token() == ""
        finally:
            clear_session_vars(tokens)

    def test_completion_event_carries_the_dispatch_credential(self):
        from tools.async_delegation import _attach_origin_nolgia_token

        evt: dict = {"type": "async_delegation", "delegation_id": "d1"}
        _attach_origin_nolgia_token(evt, {"origin_nolgia_token": "nolt_origin_run"})
        assert evt["origin_nolgia_token"] == "nolt_origin_run"

        # A durably-replayed completion has no in-memory record token: no
        # field, so the wake falls back to the retained token / pod bearer.
        replayed: dict = {"type": "async_delegation", "delegation_id": "d1"}
        _attach_origin_nolgia_token(replayed, {})
        assert "origin_nolgia_token" not in replayed

    def test_stale_dispatch_credential_is_dropped(self):
        """A detached delegation can finish arbitrarily long after its
        originating turn settled. Past the credential's validity bound the wake
        must fall back to the retained token / pod bearer instead of presenting
        an expired token (which fails the upload outright)."""
        from tools.async_delegation import (
            _ORIGIN_NOLGIA_TOKEN_MAX_AGE_S,
            _attach_origin_nolgia_token,
        )

        fresh: dict = {"type": "async_delegation", "delegation_id": "d-fresh"}
        _attach_origin_nolgia_token(fresh, {
            "origin_nolgia_token": "nolt_origin_run",
            "origin_nolgia_token_at": time.monotonic() - 5,
        })
        assert fresh["origin_nolgia_token"] == "nolt_origin_run"

        stale: dict = {"type": "async_delegation", "delegation_id": "d-stale"}
        _attach_origin_nolgia_token(stale, {
            "origin_nolgia_token": "nolt_origin_run",
            "origin_nolgia_token_at": (
                time.monotonic() - _ORIGIN_NOLGIA_TOKEN_MAX_AGE_S - 1
            ),
        })
        assert "origin_nolgia_token" not in stale

    def test_dispatch_reads_the_origin_profile(self):
        """Routing metadata, not a secret: the wake self-post has to re-enter
        through the profile that served the originating turn."""
        from tools.async_delegation import _current_origin_profile

        tokens = set_session_vars(platform="api_server", profile="alpha")
        try:
            assert _current_origin_profile() == "alpha"
        finally:
            clear_session_vars(tokens)
        assert _current_origin_profile() == ""


class TestExecuteCodeSandboxCredential:
    """The execute_code sandbox builds its child env through its own allowlist
    scrub (no session-context bridge). When the pod GRANTS ``NOLGIA_TOKEN``
    via env passthrough, a bound run token must replace the pod-wide value —
    sandbox scripts spawn the nolgia CLI directly (the film pipeline's --json
    subprocess pattern), and those uploads must attribute to the causing turn
    (NOL-413). The substitution must never introduce a credential the scrub
    withheld."""

    def test_granted_pod_token_substituted_with_run_token(self):
        from tools.code_execution_tool import _scrub_child_env
        from tools.environments.local import apply_run_scoped_nolgia_token

        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            env = _scrub_child_env(
                {"NOLGIA_TOKEN": "pod-wide"},
                is_passthrough=lambda k: k == "NOLGIA_TOKEN",
            )
            assert env.get("NOLGIA_TOKEN") == "pod-wide", "scrub grants the pod value"
            apply_run_scoped_nolgia_token(env)
        finally:
            clear_session_vars(tokens)
        assert env["NOLGIA_TOKEN"] == "nolt_run"

    def test_withheld_token_never_introduced(self):
        from tools.code_execution_tool import _scrub_child_env
        from tools.environments.local import apply_run_scoped_nolgia_token

        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            env = _scrub_child_env({"NOLGIA_TOKEN": "pod-wide"})  # no passthrough
            assert "NOLGIA_TOKEN" not in env, "secret-substring scrub drops it"
            apply_run_scoped_nolgia_token(env)
        finally:
            clear_session_vars(tokens)
        assert "NOLGIA_TOKEN" not in env

    def test_unbound_context_keeps_pod_value(self):
        from tools.environments.local import apply_run_scoped_nolgia_token

        tokens = set_session_vars(platform="api_server")  # engaged, no run token
        try:
            env = {"NOLGIA_TOKEN": "pod-wide"}
            apply_run_scoped_nolgia_token(env)
        finally:
            clear_session_vars(tokens)
        assert env["NOLGIA_TOKEN"] == "pod-wide"


class TestExecuteCodeRemoteSandboxCredential:
    """Remote backends (docker/ssh/modal/...) never build a scrubbed child env —
    the sandbox script inherits the backend runtime's POD-wide bearer, so the
    local substitution cannot reach them. ``_remote_scoped_nolgia_token`` decides
    the per-command override, with the same grant gate: substitution only, never
    introduction."""

    @pytest.fixture
    def granted(self, monkeypatch):
        import tools.env_passthrough as ep

        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        monkeypatch.setattr(ep, "is_env_passthrough", lambda name: name == "NOLGIA_TOKEN")
        monkeypatch.setattr(ep, "resolve_passthrough_value", lambda _n, fallback: fallback)

    def test_granted_and_bound_yields_the_run_token(self, granted):
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert _remote_scoped_nolgia_token() == "nolt_run"
        finally:
            clear_session_vars(tokens)

    def test_granted_but_unbound_leaves_the_pod_value_alone(self, granted):
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        tokens = set_session_vars(platform="api_server")  # engaged, no run token
        try:
            assert _remote_scoped_nolgia_token() == ""
        finally:
            clear_session_vars(tokens)

    def test_withheld_credential_is_never_introduced(self, monkeypatch):
        import tools.env_passthrough as ep
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        monkeypatch.setattr(ep, "is_env_passthrough", lambda _name: False)
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert _remote_scoped_nolgia_token() == ""
        finally:
            clear_session_vars(tokens)

    def test_no_pod_credential_means_no_override(self, monkeypatch):
        import tools.env_passthrough as ep
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        monkeypatch.delenv("NOLGIA_TOKEN", raising=False)
        monkeypatch.setattr(ep, "is_env_passthrough", lambda name: name == "NOLGIA_TOKEN")
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert _remote_scoped_nolgia_token() == ""
        finally:
            clear_session_vars(tokens)

    def test_profile_scoped_pod_credential_is_resolved_not_read_from_environ(
        self, monkeypatch,
    ):
        """Under multiplexing the pod bearer lives in the ACTIVE PROFILE's
        secret scope, not in os.environ. Gating on the process-global value
        would decide 'this sandbox has no credential' for exactly the scoped
        deployments the substitution exists for, so the grant check resolves
        through the same passthrough path the child env uses."""
        import tools.env_passthrough as ep
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        monkeypatch.delenv("NOLGIA_TOKEN", raising=False)
        monkeypatch.setattr(ep, "is_env_passthrough", lambda name: name == "NOLGIA_TOKEN")
        monkeypatch.setattr(
            ep, "resolve_passthrough_value", lambda _n, _fallback: "profile-scoped-pod",
        )
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert _remote_scoped_nolgia_token() == "nolt_run"
        finally:
            clear_session_vars(tokens)

    def test_unscoped_multiplex_read_fails_closed(self, monkeypatch):
        """An unscoped read while multiplexing raises; no override is produced."""
        import tools.env_passthrough as ep
        from tools.code_execution_tool import _remote_scoped_nolgia_token

        def _boom(_name, _fallback):
            raise RuntimeError("unscoped secret read")

        monkeypatch.setenv("NOLGIA_TOKEN", "pod-wide")
        monkeypatch.setattr(ep, "is_env_passthrough", lambda name: name == "NOLGIA_TOKEN")
        monkeypatch.setattr(ep, "resolve_passthrough_value", _boom)
        tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run")
        try:
            assert _remote_scoped_nolgia_token() == ""
        finally:
            clear_session_vars(tokens)


class _FakeRemoteEnv:
    """Minimal remote-backend stand-in recording every (command, stdin) pair."""

    _stdin_mode = "pipe"

    def __init__(self, stdin_mode: str = "pipe"):
        self._stdin_mode = stdin_mode
        self.calls: list = []

    def get_temp_dir(self) -> str:
        return "/tmp"

    def execute(self, command, cwd="", *, timeout=None, stdin_data=None, **_kw):
        self.calls.append((command, stdin_data))
        output = "OK" if "command -v python3" in command else ""
        return {"output": output, "returncode": 0}


class TestRemoteSandboxCredentialDelivery:
    """How the granted credential reaches a remote sandbox script. A
    ``VAR=value cmd`` prefix lands in the wrapping shell's argv, which any
    co-tenant process in the sandbox can read out of /proc (and which backends
    are free to log), so the value travels on stdin instead."""

    def _script_call(self, env) -> tuple:
        script_calls = [c for c in env.calls if "python3 script.py" in c[0]]
        assert len(script_calls) == 1, script_calls
        return script_calls[0]

    def _run(self, monkeypatch, env, token="nolt_run"):
        import tools.code_execution_tool as cet

        monkeypatch.setattr(cet, "_get_or_create_env", lambda _tid: (env, "docker"))
        monkeypatch.setattr(cet, "_remote_scoped_nolgia_token", lambda: token)
        monkeypatch.setattr(cet, "_rpc_poll_loop", lambda *_a, **_kw: None)
        cet._execute_remote("print(1)", "task-remote", ["read_file"])

    def test_credential_travels_on_stdin_not_in_the_command(self, monkeypatch):
        env = _FakeRemoteEnv()
        self._run(monkeypatch, env)
        command, stdin_data = self._script_call(env)
        assert stdin_data == "nolt_run\n"
        assert "NOLGIA_TOKEN" in command, "the script still gets the override"
        # Not in this command, and not in any other command of the execution.
        assert all("nolt_run" not in cmd for cmd, _stdin in env.calls)

    def test_unavailable_credential_leaves_the_command_untouched(self, monkeypatch):
        env = _FakeRemoteEnv()
        self._run(monkeypatch, env, token="")
        command, stdin_data = self._script_call(env)
        assert stdin_data is None
        assert "NOLGIA_TOKEN" not in command

    def test_heredoc_backend_keeps_its_pod_credential(self, monkeypatch):
        """Heredoc backends splice stdin back INTO the command text, which would
        reintroduce the same argv exposure — they keep the pod-wide bearer
        rather than gain a better-attributed but leaked credential."""
        env = _FakeRemoteEnv(stdin_mode="heredoc")
        self._run(monkeypatch, env)
        command, stdin_data = self._script_call(env)
        assert stdin_data is None
        assert "NOLGIA_TOKEN" not in command
        assert all("nolt_run" not in cmd for cmd, _stdin in env.calls)
