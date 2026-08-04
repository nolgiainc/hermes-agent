"""Per-session workspace isolation on the API server (NOL-414, Nolgia fork).

Concurrent turns from different sessions used to execute against one shared
scratch surface (one default cwd, one TMPDIR — ``/opt/data/tmp`` on the
Nolgia pod), so two runs could overwrite each other's fixed-path artifacts
or pick up each other's outputs (NOL-402/NOL-408 residue). The gateway now
binds every API-server turn to a session-scoped workspace
(``<base>/session-<id8>/``, gateway.session_workspace) enforced as:

- the terminal/file-tool default cwd (per-session cwd record seed);
- the agent's logical cwd (``set_session_vars(cwd=...)``);
- the subprocess ``TMPDIR`` (session-context env bridge), excluded from the
  shared bash snapshot so one session's override can never persist into a
  sibling session's commands.

The ``session-<id8>`` derivation matches nolgia-api#249's relayed
attachment-staging convention (``agentAttachmentStagingDir``) byte-for-byte
so both sides name the same directory when keyed off the same session
identity; POST /v1/runs accepts an optional ``workspace_id`` for exactly
that alignment.

The centerpiece regression (ticket item 2): two CONCURRENT runs in
different sessions each write a fixed-name artifact; assert zero
cross-visibility and no shared-path collisions.
"""

import os
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.api_server import (  # noqa: E402
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.session_context import (  # noqa: E402
    clear_session_vars,
    get_session_env,
    reset_session_vars,
)
from gateway.session_workspace import (  # noqa: E402
    ensure_session_workspace,
    resolve_session_workspace,
    session_workspaces_enabled,
    workspace_dirname,
)
from tools import terminal_tool  # noqa: E402
from tools.environments import base as env_base  # noqa: E402
from tools.environments.local import _inject_session_context_env  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_session_context():
    reset_session_vars()
    yield
    reset_session_vars()


@pytest.fixture(autouse=True)
def _isolated_cwd_records():
    """Session cwd records are process-global; keep tests hermetic."""
    with terminal_tool._session_cwd_lock:
        saved = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    yield
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(saved)


@pytest.fixture
def scratch_base(tmp_path, monkeypatch):
    """Route workspaces under a test-owned base with auto-mode engaged."""
    base = tmp_path / "scratch"
    monkeypatch.setenv("HERMES_SESSION_SCRATCH_BASE", str(base))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
    return base


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


async def _wait_completed(cli, run_id, tries=400):
    import asyncio

    for _ in range(tries):
        resp = await cli.get(f"/v1/runs/{run_id}")
        data = await resp.json()
        if data["status"] in {"completed", "failed", "cancelled"}:
            return data
        await asyncio.sleep(0.02)
    raise AssertionError("run never settled")


# ---------------------------------------------------------------------------
# Derivation unit tests
# ---------------------------------------------------------------------------


class TestWorkspaceDerivation:
    def test_matches_nolgia_api_staging_convention(self, monkeypatch):
        """Byte-for-byte parity with nolgia-api's agentAttachmentStagingDir:
        "/opt/data/tmp/session-" + sessionID.String()[:8] + "/" (#249)."""
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("HERMES_SESSION_SCRATCH_BASE", "/opt/data/tmp")
        sid = "1d284a0a-7639-44ec-9019-7a79278a1db3"
        go_convention = "/opt/data/tmp/session-" + sid[:8] + "/"
        assert resolve_session_workspace(sid) + "/" == go_convention

    def test_run_prefix_is_stripped(self):
        assert workspace_dirname("run_74753356deadbeef") == "session-74753356"

    def test_uuid_id8(self):
        assert (
            workspace_dirname("a94ca390-1111-2222-3333-444444444444")
            == "session-a94ca390"
        )

    @pytest.mark.parametrize(
        "hostile",
        ["../../etc/passwd", "..", "a/../../b", "~", "./.", "\\..\\x"],
    )
    def test_hostile_ids_cannot_traverse(self, hostile):
        name = workspace_dirname(hostile)
        assert name is not None
        assert name.startswith("session-")
        assert "/" not in name and "\\" not in name
        assert ".." not in name and "~" not in name

    def test_empty_id_yields_none(self):
        assert workspace_dirname("") is None
        assert workspace_dirname(None) is None

    def test_base_falls_back_to_hermes_home_tmp(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_SESSION_SCRATCH_BASE", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert resolve_session_workspace("abcd1234-x") == str(
            tmp_path / "tmp" / "session-abcd1234"
        )


class TestEnablementGate:
    def test_auto_mode_disabled_by_pinned_terminal_cwd(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", "/some/workspace")
        assert session_workspaces_enabled() is False
        assert resolve_session_workspace("abcd1234") is None

    def test_auto_mode_enabled_without_terminal_cwd(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert session_workspaces_enabled() is True

    def test_auto_mode_enabled_when_bridge_set_home_fallback(self, monkeypatch):
        """The gateway startup bridge materializes TERMINAL_CWD=$HOME when
        terminal.cwd is unset (the pod: /opt/data). That is the shared
        default this feature exists to fix — NOT an operator pin."""
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", str(Path.home()))
        assert session_workspaces_enabled() is True

    @pytest.mark.parametrize("placeholder", [".", "auto", "cwd"])
    def test_auto_mode_enabled_on_placeholder_cwd(self, monkeypatch, placeholder):
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", placeholder)
        assert session_workspaces_enabled() is True

    def test_force_on_overrides_pinned_cwd(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "/some/workspace")
        monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "1")
        assert session_workspaces_enabled() is True

    def test_force_off(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "0")
        assert session_workspaces_enabled() is False


# ---------------------------------------------------------------------------
# Bind-time enforcement units
# ---------------------------------------------------------------------------


class TestBindSeedsWorkspace:
    def test_bind_creates_dir_and_binds_cwd_record_and_scratch_var(self, scratch_base):
        sid = "cafe0001-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            expected = str(scratch_base / "session-cafe0001")
            assert os.path.isdir(expected)
            assert get_session_env("HERMES_SESSION_SCRATCH_DIR") == expected
            assert terminal_tool.get_session_cwd(sid) == expected
            # The terminal's own resolver must pick the session dir over the
            # shared default for a command with no explicit workdir.
            resolved = terminal_tool._resolve_command_cwd(
                workdir=None, default_cwd="/shared/default", session_key=sid
            )
            assert resolved == expected
        finally:
            clear_session_vars(tokens)

    def test_bind_does_not_clobber_existing_session_cwd(self, scratch_base):
        """A continuing session's `cd` state survives the seed (seed-only)."""
        sid = "cafe0002-aaaa-bbbb-cccc-ddddeeeeffff"
        terminal_tool.record_session_cwd(sid, "/existing/worktree")
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            assert terminal_tool.get_session_cwd(sid) == "/existing/worktree"
            # TMPDIR scoping still applies even when the cwd stays put.
            assert get_session_env("HERMES_SESSION_SCRATCH_DIR") == str(
                scratch_base / "session-cafe0002"
            )
        finally:
            clear_session_vars(tokens)

    def test_disabled_bind_is_byte_identical_to_before(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "/pinned/workspace")
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        sid = "cafe0003-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            assert get_session_env("HERMES_SESSION_SCRATCH_DIR") == ""
            assert terminal_tool.get_session_cwd(sid) is None
        finally:
            clear_session_vars(tokens)


class TestSubprocessBridge:
    def test_bound_scratch_dir_redirects_child_tmpdir(self, scratch_base):
        sid = "cafe0004-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            env = {"TMPDIR": "/tmp"}
            _inject_session_context_env(env)
        finally:
            clear_session_vars(tokens)
        expected = str(scratch_base / "session-cafe0004")
        assert env["HERMES_SESSION_SCRATCH_DIR"] == expected
        assert env["TMPDIR"] == expected

    def test_unbound_task_keeps_inherited_tmpdir(self):
        from gateway.session_context import set_session_vars

        tokens = set_session_vars(platform="api_server")  # engaged, no scratch
        try:
            env = {"TMPDIR": "/tmp"}
            _inject_session_context_env(env)
        finally:
            clear_session_vars(tokens)
        assert env["TMPDIR"] == "/tmp"

    def test_snapshot_excludes_tmpdir_and_scratch_var(self):
        """One session's TMPDIR override must never persist into the shared
        bash snapshot a sibling session's later commands source."""
        pattern = re.compile(env_base._SNAPSHOT_EXCLUDED_ENV_REGEX)
        assert pattern.match('declare -x TMPDIR="/opt/data/tmp/session-aaaa1111"')
        assert pattern.match(
            'declare -x HERMES_SESSION_SCRATCH_DIR="/opt/data/tmp/session-aaaa1111"'
        )
        # The exclusion is exact-name anchored: TMPDIR-prefixed user vars and
        # unrelated exports still snapshot normally.
        assert not pattern.match('declare -x TMPDIRS="/x"')
        assert not pattern.match('declare -x PATH="/usr/bin"')
        assert not pattern.match('declare -x TMP="/x"')


# ---------------------------------------------------------------------------
# The NOL-414 item-2 regression: two concurrent runs, zero cross-visibility
# ---------------------------------------------------------------------------


def _observing_run_conversation(barrier, captured):
    """A fake agent turn that behaves like a real tool-using turn:

    resolves its default write directory exactly the way the terminal tool
    does (approval/session key → cwd record → shared default), resolves its
    subprocess TMPDIR through the real env bridge, then drops a FIXED-NAME
    artifact — the collision-prone pattern from NOL-402/NOL-408. Captures
    are keyed by the turn's own ``task_id`` (the session id), so the
    assertions cannot be confused by agent-construction ordering.
    """

    def _run(user_message=None, conversation_history=None, task_id=None):
        from tools.approval import get_current_session_key

        # Hold until both runs are inside their turn — proves true overlap.
        barrier.wait(timeout=20)

        session_key = get_current_session_key(default="") or (task_id or "")
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None,
            default_cwd="/shared/default",
            session_key=session_key,
        )
        env = {}
        _inject_session_context_env(env)

        artifact = os.path.join(cwd, "render.mp4")
        with open(artifact, "w", encoding="utf-8") as fh:
            fh.write(f"artifact-from-{task_id}")

        captured[task_id] = {
            "task_id": task_id,
            "cwd": cwd,
            "tmpdir": env.get("TMPDIR", ""),
            "scratch": env.get("HERMES_SESSION_SCRATCH_DIR", ""),
            "artifact": artifact,
        }
        # Second rendezvous: neither run finishes before the other wrote.
        barrier.wait(timeout=20)
        return {"final_response": "done"}

    return _run


class TestConcurrentSessionIsolation:
    @pytest.mark.asyncio
    async def test_two_concurrent_runs_zero_cross_visibility(self, scratch_base):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        barrier = threading.Barrier(2)
        captured = {}

        session_a = "aaaa1111-0000-4000-8000-000000000001"
        session_b = "bbbb2222-0000-4000-8000-000000000002"

        run_fn = _observing_run_conversation(barrier, captured)

        def _make_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = run_fn
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_make_agent):
                resp_a = await cli.post(
                    "/v1/runs", json={"input": "make video", "session_id": session_a}
                )
                resp_b = await cli.post(
                    "/v1/runs", json={"input": "make video", "session_id": session_b}
                )
                assert resp_a.status == 202 and resp_b.status == 202
                run_a = (await resp_a.json())["run_id"]
                run_b = (await resp_b.json())["run_id"]
                status_a = await _wait_completed(cli, run_a)
                status_b = await _wait_completed(cli, run_b)

        assert status_a["status"] == "completed", status_a
        assert status_b["status"] == "completed", status_b
        a, b = captured[session_a], captured[session_b]

        # No shared-path collision: the same fixed artifact name landed in two
        # different directories, and neither fell back to the shared default.
        assert a["artifact"] != b["artifact"]
        assert a["cwd"] != b["cwd"]
        assert a["cwd"] != "/shared/default" and b["cwd"] != "/shared/default"

        # Each turn's whole scratch surface is one session-scoped dir:
        # terminal default cwd == subprocess TMPDIR == bound workspace.
        for cap, sid in ((a, session_a), (b, session_b)):
            expected = str(scratch_base / f"session-{sid[:8]}")
            assert cap["cwd"] == expected
            assert cap["tmpdir"] == expected
            assert cap["scratch"] == expected

        # Zero cross-visibility: each session dir holds exactly its own
        # artifact with its own content — nothing overwritten, nothing leaked.
        content_a = Path(a["artifact"]).read_text(encoding="utf-8")
        content_b = Path(b["artifact"]).read_text(encoding="utf-8")
        assert content_a == f"artifact-from-{session_a}"
        assert content_b == f"artifact-from-{session_b}"
        assert os.listdir(a["cwd"]) == ["render.mp4"]
        assert os.listdir(b["cwd"]) == ["render.mp4"]
        assert Path(b["artifact"]).parent != Path(a["artifact"]).parent
        # And nothing landed at the shared base root itself.
        assert sorted(p.name for p in scratch_base.iterdir()) == [
            f"session-{session_a[:8]}",
            f"session-{session_b[:8]}",
        ]

    @pytest.mark.asyncio
    async def test_explicit_workspace_id_aligns_with_platform_staging_dir(
        self, scratch_base
    ):
        """A run naming its platform session via ``workspace_id`` executes in
        the exact directory nolgia-api's attachment-fetch instruction stages
        downloads into (session-<platform id8>), even though the pod session
        stays keyed by the run id (NOL-129)."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        platform_session = "5eca4c05-8374-413d-bc1d-6c05552daf0c"
        captured = {}

        def _run(user_message=None, conversation_history=None, task_id=None):
            captured["scratch"] = get_session_env("HERMES_SESSION_SCRATCH_DIR")
            captured["task_id"] = task_id
            return {"final_response": "done"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                agent = MagicMock()
                agent.run_conversation.side_effect = _run
                agent.session_prompt_tokens = 0
                agent.session_completion_tokens = 0
                agent.session_total_tokens = 0
                mock_create.return_value = agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hi", "workspace_id": platform_session},
                )
                assert resp.status == 202
                await _wait_completed(cli, (await resp.json())["run_id"])

        assert captured["scratch"] == str(
            scratch_base / f"session-{platform_session[:8]}"
        )
        # The pod session identity is untouched by workspace_id: with no
        # session_id supplied the run still keys its own state by run id.
        assert captured["task_id"].startswith("run_")

    @pytest.mark.asyncio
    async def test_malformed_workspace_id_is_rejected(self, scratch_base):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for bad in [123, "ws\nid", "x" * 200]:
                resp = await cli.post(
                    "/v1/runs", json={"input": "hi", "workspace_id": bad}
                )
                assert resp.status == 400, f"expected 400 for {bad!r}"

    @pytest.mark.asyncio
    async def test_idempotency_fingerprint_covers_workspace_id(self, scratch_base):
        """Same Idempotency-Key + different workspace_id is a DIFFERENT
        submission: it must start a new run, not replay one that executed
        against another session's scratch dir."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)

        def _run(user_message=None, conversation_history=None, task_id=None):
            return {"final_response": "done"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                agent = MagicMock()
                agent.run_conversation.side_effect = _run
                agent.session_prompt_tokens = 0
                agent.session_completion_tokens = 0
                agent.session_total_tokens = 0
                mock_create.return_value = agent

                first = await cli.post(
                    "/v1/runs",
                    json={"input": "hi", "workspace_id": "aaaa1111"},
                    headers={"Idempotency-Key": "key-1"},
                )
                assert first.status == 202
                first_id = (await first.json())["run_id"]
                await _wait_completed(cli, first_id)

                replay = await cli.post(
                    "/v1/runs",
                    json={"input": "hi", "workspace_id": "aaaa1111"},
                    headers={"Idempotency-Key": "key-1"},
                )
                assert (await replay.json())["run_id"] == first_id

                other = await cli.post(
                    "/v1/runs",
                    json={"input": "hi", "workspace_id": "bbbb2222"},
                    headers={"Idempotency-Key": "key-1"},
                )
                assert other.status == 202
                assert (await other.json())["run_id"] != first_id
