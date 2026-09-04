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
import time
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
    prune_session_workspaces,
    resolve_session_workspace,
    session_workspaces_active,
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
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    return base


def _visible_entries(path: str) -> list:
    """Directory contents minus Hermes' internal dot-files."""
    return sorted(name for name in os.listdir(path) if not name.startswith("."))


def _write_config(mapping: dict) -> Path:
    """Write config.yaml under the test-isolated HERMES_HOME."""
    import yaml

    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def _snapshot_env():
    """Minimal concrete BaseEnvironment for inspecting the wrapped script."""

    class _WrapOnlyEnv(env_base.BaseEnvironment):
        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            raise NotImplementedError

        def cleanup(self):
            pass

    return _WrapOnlyEnv(cwd="/tmp", timeout=10)


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
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert resolve_session_workspace("abcd1234-0000-4000-8000-000000000001") == str(
            tmp_path / "tmp" / "session-abcd1234"
        )


class TestNameCollisionResistance:
    """Distinct identities must never resolve to one scratch directory."""

    def test_prefix_sharing_non_uuid_ids_get_distinct_dirs(self):
        one = workspace_dirname("aaaaaaaa-one")
        two = workspace_dirname("aaaaaaaa-two")
        assert one != two
        assert one.startswith("session-aaaaaaaa-")
        assert two.startswith("session-aaaaaaaa-")

    def test_derivation_is_deterministic(self):
        assert workspace_dirname("aaaaaaaa-one") == workspace_dirname("aaaaaaaa-one")

    def test_convention_shaped_ids_keep_the_staging_name(self):
        """The suffix must not break nolgia-api#249 alignment for the ids the
        staging convention actually covers (platform UUIDs, run hex)."""
        assert (
            workspace_dirname("5eca4c05-8374-413d-bc1d-6c05552daf0c")
            == "session-5eca4c05"
        )
        assert workspace_dirname("run_74753356deadbeef") == "session-74753356"

    def test_uuid_prefix_collision_falls_back_to_a_distinct_dir(self, scratch_base):
        """Two DIFFERENT session UUIDs sharing the first 8 hex chars both
        derive session-aaaaaaaa; the second must not inherit the first's
        scratch dir."""
        first = ensure_session_workspace("aaaaaaaa-1111-4000-8000-000000000001")
        second = ensure_session_workspace("aaaaaaaa-2222-4000-8000-000000000002")
        assert first == str(scratch_base / "session-aaaaaaaa")
        assert second and second != first
        assert Path(second).name.startswith("session-aaaaaaaa-")
        # Re-binding the original identity still lands in the claimed dir.
        assert ensure_session_workspace("aaaaaaaa-1111-4000-8000-000000000001") == first


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

    @pytest.mark.parametrize(
        "backend", ["docker", "ssh", "modal", "daytona", "singularity", "vercel_sandbox"]
    )
    def test_non_local_backend_never_engages(self, monkeypatch, backend):
        """A gateway-host scratch dir does not exist inside a remote backend,
        and the TMPDIR bridge is LocalEnvironment-only — binding one would
        just make every command `cd` into a missing path (exit 126). Even a
        forced-on config must stay off there."""
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "1")
        assert session_workspaces_enabled() is False
        assert resolve_session_workspace("abcd1234") is None
        assert ensure_session_workspace("abcd1234") is None

    def test_local_backend_engages(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert session_workspaces_enabled() is True


class TestConfigYamlSettings:
    """The operator-facing surface is config.yaml; HERMES_* is an internal bridge."""

    def test_mode_off_from_config_disables(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        _write_config({"gateway": {"session_workspaces": {"mode": "off"}}})
        assert session_workspaces_enabled() is False

    def test_scratch_base_from_config(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_SESSION_SCRATCH_BASE", raising=False)
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        base = tmp_path / "cfg-scratch"
        _write_config(
            {"gateway": {"session_workspaces": {"scratch_base": str(base)}}}
        )
        assert resolve_session_workspace("abcd1234-0000-4000-8000-000000000001") == str(
            base / "session-abcd1234"
        )

    def test_env_bridge_overrides_config(self, monkeypatch, tmp_path):
        """Child processes only see env; the bridge must keep winning."""
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        _write_config({"gateway": {"session_workspaces": {"mode": "off"}}})
        monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "1")
        assert session_workspaces_enabled() is True


class TestCapabilityReflectsEffectiveConfig:
    """The platform raises per-user concurrency on this flag — it must not be
    advertised by a deployment that still shares one cwd/TMPDIR surface."""

    def test_active_when_base_is_writable(self, scratch_base):
        assert session_workspaces_active() is True

    def test_inactive_when_disabled(self, monkeypatch, scratch_base):
        monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "0")
        assert session_workspaces_active() is False

    def test_inactive_when_base_cannot_be_provisioned(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("HERMES_SESSION_SCRATCH_BASE", str(blocker / "scratch"))
        assert session_workspaces_active() is False

    @pytest.mark.asyncio
    async def test_capabilities_endpoint_tracks_effective_state(
        self, monkeypatch, scratch_base
    ):
        adapter = _make_adapter()
        app = web.Application()
        app["api_server_adapter"] = adapter
        app.router.add_get("/v1/capabilities", adapter._handle_capabilities)

        async with TestClient(TestServer(app)) as cli:
            data = await (await cli.get("/v1/capabilities")).json()
            assert data["features"]["session_workspaces"] is True

            monkeypatch.setenv("HERMES_SESSION_WORKSPACES", "0")
            data = await (await cli.get("/v1/capabilities")).json()
            assert data["features"]["session_workspaces"] is False


class TestRetention:
    """Ephemeral workspaces (stateless calls, session-less runs) are bounded."""

    def test_idle_workspaces_are_pruned_and_fresh_ones_kept(self, tmp_path):
        base = tmp_path / "scratch"
        base.mkdir()
        stale = base / "session-deadbeef"
        stale.mkdir()
        (stale / "huge.bin").write_text("x", encoding="utf-8")
        fresh = base / "session-cafe0001"
        fresh.mkdir()
        unrelated = base / "keep-me"
        unrelated.mkdir()

        old = time.time() - 72 * 3600
        os.utime(stale / "huge.bin", (old, old))
        os.utime(stale, (old, old))
        os.utime(unrelated, (old, old))

        removed = prune_session_workspaces(base=base, retention_hours=48)
        assert removed == 1
        assert not stale.exists()
        assert fresh.is_dir()
        # Only session-* directories are ever touched.
        assert unrelated.is_dir()

    def test_retention_zero_disables_the_sweep(self, tmp_path):
        base = tmp_path / "scratch"
        base.mkdir()
        stale = base / "session-deadbeef"
        stale.mkdir()
        old = time.time() - 999 * 3600
        os.utime(stale, (old, old))
        assert prune_session_workspaces(base=base, retention_hours=0) == 0
        assert stale.is_dir()

    def test_active_workspace_is_refreshed_on_bind(self, scratch_base):
        """A long-lived session that writes nothing this turn must not look
        idle to the sweep."""
        sid = "cafe0009-0000-4000-8000-000000000009"
        path = Path(ensure_session_workspace(sid))
        old = time.time() - 999 * 3600
        os.utime(path, (old, old))
        assert ensure_session_workspace(sid) == str(path)
        assert path.stat().st_mtime > old
        assert prune_session_workspaces(base=scratch_base, retention_hours=1) == 0
        assert path.is_dir()

    def test_workspace_reclaimed_after_the_scan_is_not_deleted(self, scratch_base, monkeypatch):
        """A stale session resumed WHILE the sweep runs keeps its workspace.

        The sweeper must not pass its age check, let `_provision` hand the
        same directory to the resuming turn, and then rmtree it anyway: that
        run would be left with a nonexistent cwd/TMPDIR and lost contents.
        """
        import gateway.session_workspace as sw

        sid = "cafe0010-0000-4000-8000-000000000010"
        path = Path(ensure_session_workspace(sid))
        (path / "artifact.txt").write_text("keep", encoding="utf-8")
        old = time.time() - 999 * 3600
        # Age every entry (the owner marker included) so the dir reads as idle.
        for target in (*path.iterdir(), path):
            os.utime(target, (old, old))

        real_newest = sw._newest_mtime
        calls = {"n": 0}

        def rebinding_newest(target):
            age = real_newest(target)
            calls["n"] += 1
            if calls["n"] == 1 and target == str(path):
                # The turn resumes right after the sweeper reads the age:
                # provisioning claims and refreshes the same directory.
                assert ensure_session_workspace(sid) == str(path)
            return age

        monkeypatch.setattr(sw, "_newest_mtime", rebinding_newest)
        assert prune_session_workspaces(base=scratch_base, retention_hours=1) == 0
        assert (path / "artifact.txt").read_text(encoding="utf-8") == "keep"

    def test_prune_throttle_is_per_scratch_base(self, tmp_path, monkeypatch):
        """Each profile's scratch base gets its own sweep budget: one global
        timestamp let a busy profile starve every other profile's base."""
        import gateway.session_workspace as sw

        swept = []
        monkeypatch.setattr(sw, "_last_prune_at", {})
        monkeypatch.setattr(
            sw,
            "prune_session_workspaces",
            lambda base=None, keep=(): swept.append(str(base)),
        )

        base_a = tmp_path / "profile-a" / "tmp"
        base_b = tmp_path / "profile-b" / "tmp"
        sw._maybe_prune(base_a)
        sw._maybe_prune(base_b)
        sw._maybe_prune(base_a)  # throttled within the interval

        assert swept == [str(base_a), str(base_b)]


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

    def test_snapshot_excludes_tmpdir_for_scoped_turns(self):
        """One session's Hermes-owned TMPDIR must never persist into the
        shared bash snapshot a sibling session's later commands source."""
        pattern = re.compile(env_base.snapshot_excluded_env_regex(scoped_tmpdir=True))
        assert pattern.match('declare -x TMPDIR="/opt/data/tmp/session-aaaa1111"')
        assert pattern.match(
            'declare -x HERMES_SESSION_SCRATCH_DIR="/opt/data/tmp/session-aaaa1111"'
        )
        # The exclusion is exact-name anchored: TMPDIR-prefixed user vars and
        # unrelated exports still snapshot normally.
        assert not pattern.match('declare -x TMPDIRS="/x"')
        assert not pattern.match('declare -x PATH="/usr/bin"')
        assert not pattern.match('declare -x TMP="/x"')

    def test_user_tmpdir_export_persists_outside_scoped_turns(self):
        """TMPDIR is a normal user-controlled shell variable: with no session
        workspace bound (CLI, workspaces disabled) an `export TMPDIR=/custom`
        must keep surviving into the next command."""
        pattern = re.compile(env_base.snapshot_excluded_env_regex(scoped_tmpdir=False))
        assert not pattern.match('declare -x TMPDIR="/custom"')
        assert pattern.match('declare -x HERMES_SESSION_ID="abc"')

        unscoped = env_base._export_dump_excluding_session_vars(
            '"$snap"', scoped_tmpdir=False
        )
        assert "TMPDIR" not in unscoped
        scoped = env_base._export_dump_excluding_session_vars(
            '"$snap"', scoped_tmpdir=True
        )
        assert "TMPDIR" in scoped

    def test_wrap_command_restores_scoped_tmpdir_after_sourcing(self, scratch_base):
        """Sourcing a snapshot written by an EARLIER unscoped turn must not
        overwrite this turn's bound TMPDIR: the wrapper saves it before the
        source and restores it immediately afterwards, exactly as it does for
        profile-scoped passthrough names."""
        env = _snapshot_env()
        env._snapshot_ready = True

        unscoped = env._wrap_command("echo hi", "/tmp")
        assert "_HERMES_RUNTIME_PASSTHROUGH_TMPDIR_VALUE" not in unscoped

        sid = "cafe0006-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            wrapped = env._wrap_command("echo hi", "/tmp")
        finally:
            clear_session_vars(tokens)

        save = wrapped.index("_HERMES_RUNTIME_PASSTHROUGH_TMPDIR_VALUE=${TMPDIR-}")
        source = wrapped.index("source ")
        restore = wrapped.index(
            'export TMPDIR="$_HERMES_RUNTIME_PASSTHROUGH_TMPDIR_VALUE"'
        )
        run = wrapped.index("eval 'echo hi'")
        # Saved before the source, restored after it, before the command runs.
        assert save < source < restore < run
        # The value travels in environment memory, never in the command string.
        assert str(scratch_base) not in wrapped

    @pytest.mark.skipif(os.name == "nt", reason="POSIX bash snapshot path")
    def test_stale_snapshot_tmpdir_does_not_win_over_the_bound_workspace(
        self, scratch_base, tmp_path
    ):
        """E2E through the real source-and-execute path: a snapshot polluted
        by an unscoped `export TMPDIR=/custom` must not redirect a later
        session-scoped run's temp writes out of its own workspace."""
        from tools.environments.local import LocalEnvironment

        custom = tmp_path / "user-tmp"
        custom.mkdir()
        env = LocalEnvironment(cwd=str(tmp_path), timeout=60)
        env.init_session()
        try:
            # An unscoped turn (CLI-style) persists its own TMPDIR.
            env.execute(f"export TMPDIR={custom}")
            assert custom.name in env.execute('printf "%s" "$TMPDIR"').get("output", "")

            sid = "cafe0007-aaaa-bbbb-cccc-ddddeeeeffff"
            expected = str(scratch_base / "session-cafe0007")
            result = {}

            def scoped_turn():
                tokens = APIServerAdapter._bind_api_server_session(
                    chat_id=sid, session_key=sid, session_id=sid
                )
                try:
                    result["tmpdir"] = env.execute(
                        'printf "%s" "$TMPDIR"'
                    ).get("output", "")
                finally:
                    clear_session_vars(tokens)

            thread = threading.Thread(target=scoped_turn)
            thread.start()
            thread.join(timeout=90)

            assert result.get("tmpdir", "").strip() == expected
        finally:
            env.cleanup()

    def test_scoped_tmpdir_detection_follows_the_bound_session(self, scratch_base):
        """The default (no explicit flag) tracks whether THIS turn owns its
        TMPDIR, so the suppression is scoped instead of global."""
        assert env_base._session_scoped_tmpdir_active() is False
        assert "TMPDIR" not in env_base._export_dump_excluding_session_vars('"$snap"')

        sid = "cafe0005-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            assert env_base._session_scoped_tmpdir_active() is True
            assert "TMPDIR" in env_base._export_dump_excluding_session_vars('"$snap"')
        finally:
            clear_session_vars(tokens)


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
        # (Hermes' own dot-prefixed workspace-ownership marker aside.)
        assert _visible_entries(a["cwd"]) == ["render.mp4"]
        assert _visible_entries(b["cwd"]) == ["render.mp4"]
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
        submission: it must never replay a run that executed against another
        session's scratch dir. The durable reservation store rejects the
        mismatch as 409 idempotency_key_conflict (upstream's contract)."""
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
                assert other.status == 409
                assert (await other.json())["error"]["code"] == "idempotency_key_conflict"


# ---------------------------------------------------------------------------
# The NOL-414 LIVE regression: the turn's ADVERTISED working directory
# ---------------------------------------------------------------------------
#
# The 2026-08-04 live smoke on the pod image containing the original NOL-414
# change: two concurrent platform-relay runs each created+seeded their session
# workspace correctly, yet BOTH wrote `marker.txt` at the shared home
# (`/opt/data`) and collided. Cause: the bind never set the runtime cwd, so
# the system prompt's "Current working directory" line (resolve_agent_cwd)
# fell back to TERMINAL_CWD — the shared home — and the model, told to work
# "in its current working directory", emitted ABSOLUTE shared-home paths that
# bypass the (relative-path) cwd record entirely. These tests pin the fix:
# the advertised cwd IS the session workspace on the live relay shape, while
# context-file discovery stays home-anchored and `cd` state / operator pins
# keep winning.


@pytest.fixture
def pod_home(tmp_path, monkeypatch):
    """The pod's exact shape: HOME == HERMES_HOME == TERMINAL_CWD.

    The startup bridge materializes TERMINAL_CWD=$HOME when terminal.cwd is
    unset (on the pod: /opt/data) — the home-fallback case that counts as
    NOT operator-pinned, so session workspaces stay auto-enabled.
    """
    home = tmp_path / "opt-data"
    (home / "tmp").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(home))
    monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SCRATCH_BASE", raising=False)
    return home


class TestAdvertisedRuntimeCwd:
    @pytest.mark.asyncio
    async def test_relay_run_advertises_session_workspace(self, pod_home):
        """The live-divergence shape: relay-style POST /v1/runs (no
        session_id — NOL-129), pod env. Inside the turn the ADVERTISED cwd
        (what the system prompt reports) must be the session workspace, and a
        real relative write_file through the full dispatcher must land there
        — never at the shared home."""
        import json as _json

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        captured = {}

        def _run(user_message=None, conversation_history=None, task_id=None):
            from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
            from model_tools import handle_function_call

            captured["task_id"] = task_id
            captured["advertised"] = str(resolve_agent_cwd())
            captured["context"] = resolve_context_cwd()
            result = handle_function_call(
                "write_file",
                {"path": "marker.txt", "content": "GAMMA-4472"},
                task_id=task_id,
            )
            captured["write"] = _json.loads(result)
            return {"final_response": "done"}

        def _make_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_make_agent):
                resp = await cli.post("/v1/runs", json={"input": "smoke"})
                assert resp.status == 202, await resp.text()
                run_id = (await resp.json())["run_id"]
                status = await _wait_completed(cli, run_id)

        assert status["status"] == "completed", status
        expected = str(pod_home / "tmp" / f"session-{captured['task_id'][4:12]}")
        # The prompt's "Current working directory" line names the session
        # workspace — NOT the shared home (the live bug).
        assert captured["advertised"] == expected
        assert captured["advertised"] != str(pod_home)
        # Context-file discovery keeps its pre-bind anchor (the home
        # workspace doctrine), never the empty scratch dir.
        assert str(captured["context"]) == str(pod_home)
        # And the real write landed inside the workspace.
        assert captured["write"]["resolved_path"] == os.path.join(
            expected, "marker.txt"
        )
        assert (Path(expected) / "marker.txt").read_text(
            encoding="utf-8"
        ) == "GAMMA-4472"
        assert not (pod_home / "marker.txt").exists()

    @pytest.mark.asyncio
    async def test_concurrent_relay_runs_advertise_distinct_workspaces(
        self, pod_home
    ):
        """Two overlapping relay runs: each turn is TOLD a different cwd."""
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        barrier = threading.Barrier(2)
        captured = {}

        def _run(user_message=None, conversation_history=None, task_id=None):
            from agent.runtime_cwd import resolve_agent_cwd

            barrier.wait(timeout=20)
            captured[task_id] = str(resolve_agent_cwd())
            barrier.wait(timeout=20)
            return {"final_response": "done"}

        def _make_agent(**kwargs):
            agent = MagicMock()
            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_make_agent):
                resp_a = await cli.post("/v1/runs", json={"input": "a"})
                resp_b = await cli.post("/v1/runs", json={"input": "b"})
                assert resp_a.status == 202 and resp_b.status == 202
                run_a = (await resp_a.json())["run_id"]
                run_b = (await resp_b.json())["run_id"]
                assert (await _wait_completed(cli, run_a))["status"] == "completed"
                assert (await _wait_completed(cli, run_b))["status"] == "completed"

        cwds = list(captured.values())
        assert len(cwds) == 2
        assert cwds[0] != cwds[1]
        assert str(pod_home) not in cwds
        for task_id, cwd in captured.items():
            assert cwd == str(pod_home / "tmp" / f"session-{task_id[4:12]}")

    def test_bind_advertises_surface_pin_over_scratch(self, pod_home, tmp_path):
        """A surface-registered workspace (ACP/TUI project root) is the
        session's durable workspace and is what gets advertised."""
        from agent.runtime_cwd import resolve_agent_cwd

        project = tmp_path / "project"
        project.mkdir()
        sid = "cafe0005-aaaa-bbbb-cccc-ddddeeeeffff"
        terminal_tool.register_task_env_overrides(sid, {"cwd": str(project)})
        try:
            tokens = APIServerAdapter._bind_api_server_session(
                chat_id=sid, session_key=sid, session_id=sid
            )
            try:
                assert str(resolve_agent_cwd()) == str(project)
            finally:
                clear_session_vars(tokens)
        finally:
            terminal_tool.clear_task_env_overrides(sid)

    def test_runs_shape_prefers_durable_session_over_approval_key(
        self, pod_home, tmp_path
    ):
        """The real /v1/runs caller shape: ``session_key`` is the run's
        approval key (a fresh run id per request), the durable conversation
        id arrives separately as ``session_id``/``chat_id``. The advertised
        cwd must come from the DURABLE identity — reading the per-run
        approval key first would shadow a continuing session's pinned
        workspace with this run's scratch root."""
        from agent.runtime_cwd import resolve_agent_cwd

        project = tmp_path / "project"
        project.mkdir()
        sid = "cafe0007-aaaa-bbbb-cccc-ddddeeeeffff"
        run_key = "run_0123456789abcdef0123456789abcdef"
        terminal_tool.register_task_env_overrides(sid, {"cwd": str(project)})
        try:
            tokens = APIServerAdapter._bind_api_server_session(
                chat_id=sid, session_key=run_key, session_id=sid
            )
            try:
                assert str(resolve_agent_cwd()) == str(project)
            finally:
                clear_session_vars(tokens)
        finally:
            terminal_tool.clear_task_env_overrides(sid)

    def test_cd_state_does_not_move_the_advertised_cwd(self, pod_home, tmp_path):
        """Prompt-cache safety: the advertised cwd is STABLE for the life of
        the conversation. A `cd` mid-session changes the mutable cwd record
        (which is what commands and relative paths keep resolving against)
        but must NOT change what the system prompt was told, or the stored
        prompt's cwd line goes stale and the whole prompt gets rebuilt."""
        from agent.runtime_cwd import resolve_agent_cwd

        sid = "cafe0008-aaaa-bbbb-cccc-ddddeeeeffff"
        workspace = pod_home / "tmp" / f"session-{sid[:8]}"
        cd_target = tmp_path / "elsewhere"
        cd_target.mkdir()

        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            assert str(resolve_agent_cwd()) == str(workspace)
        finally:
            clear_session_vars(tokens)

        # Turn 2, after the model `cd`'d somewhere during turn 1.
        terminal_tool.record_session_cwd(sid, str(cd_target))
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            assert str(resolve_agent_cwd()) == str(workspace)
            # The mutable record is untouched — `cd` still governs commands.
            assert terminal_tool.get_session_cwd(sid) == str(cd_target)
        finally:
            clear_session_vars(tokens)

    def test_pinned_workspace_bind_leaves_runtime_cwd_unbound(self, monkeypatch):
        """Operator-pinned TERMINAL_CWD disables the feature entirely: the
        bind must not set a runtime cwd, so the pin keeps governing the
        advertised directory exactly as before."""
        from agent import runtime_cwd

        monkeypatch.setenv("TERMINAL_CWD", "/pinned/workspace")
        monkeypatch.delenv("HERMES_SESSION_WORKSPACES", raising=False)
        sid = "cafe0006-aaaa-bbbb-cccc-ddddeeeeffff"
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            # Upstream folded the fork's ``_session_cwd_override()`` reader into
            # ``_resolve_configured_cwd``; the bound session cwd is the module
            # ContextVar itself, so read it directly: unset or cleared, never a path.
            assert runtime_cwd._SESSION_CWD.get() in (runtime_cwd._UNSET, "")
        finally:
            clear_session_vars(tokens)

    def _stub_agent(self, stored_prompt):
        """Minimal agent for the real ``_restore_or_build_system_prompt``.

        Only the collaborators that function touches are stubbed (the
        session-DB row and the prompt builder); the restore/rebuild DECISION
        — the thing under test — runs for real.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        agent = MagicMock()
        agent.session_id = "sess-1"
        agent.model = "hermes-4"
        agent.provider = "nolgia"
        agent.platform = "api_server"
        agent._use_prompt_caching = False
        agent._cached_system_prompt = None
        agent._session_db.get_session.return_value = {
            "system_prompt": stored_prompt
        }
        agent._build_system_prompt.side_effect = lambda *_a, **_k: (
            f"User home directory: {os.path.expanduser('~')}\n"
            f"Current working directory: {resolve_agent_cwd()}\n"
            "Model: hermes-4\nProvider: nolgia\nPlatform: api_server"
        )
        return agent

    @staticmethod
    def _persisted_prompt(cwd) -> str:
        return (
            f"User home directory: {os.path.expanduser('~')}\n"
            f"Current working directory: {cwd}\n"
            "Model: hermes-4\nProvider: nolgia\nPlatform: api_server"
        )

    def test_legacy_shared_home_prompt_is_rebuilt_for_the_workspace(
        self, pod_home
    ):
        """Upgrade path: a session persisted BEFORE this fix carries a
        ``Current working directory: <shared home>`` line. Reusing it verbatim
        would keep telling the model to write at the shared home even though
        the resolver now returns the isolated workspace, so the real
        restoration path must rebuild it exactly once — and the rebuild must
        advertise the session workspace."""
        from agent.conversation_loop import _restore_or_build_system_prompt

        sid = "cafe0009-aaaa-bbbb-cccc-ddddeeeeffff"
        workspace = pod_home / "tmp" / f"session-{sid[:8]}"
        agent = self._stub_agent(self._persisted_prompt(pod_home))

        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
        finally:
            clear_session_vars(tokens)

        assert agent._build_system_prompt.called
        assert (
            f"Current working directory: {workspace}"
            in agent._cached_system_prompt
        )
        assert f"Current working directory: {pod_home}\n" not in (
            agent._cached_system_prompt
        )
        # The rebuild is persisted, so the next turn restores it verbatim.
        agent._session_db.update_system_prompt.assert_called_once_with(
            "sess-1", agent._cached_system_prompt
        )

    def test_workspace_prompt_survives_a_cd_without_rebuild(self, pod_home, tmp_path):
        """Prompt caching is sacred: once a session's prompt advertises its
        workspace, a mid-conversation `cd` must not invalidate it. The stored
        prompt is reused verbatim on the next turn."""
        from agent.conversation_loop import _restore_or_build_system_prompt

        sid = "cafe000a-aaaa-bbbb-cccc-ddddeeeeffff"
        workspace = pod_home / "tmp" / f"session-{sid[:8]}"
        stored = self._persisted_prompt(workspace)
        agent = self._stub_agent(stored)

        cd_target = tmp_path / "elsewhere"
        cd_target.mkdir()
        terminal_tool.record_session_cwd(sid, str(cd_target))

        tokens = APIServerAdapter._bind_api_server_session(
            chat_id=sid, session_key=sid, session_id=sid
        )
        try:
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
        finally:
            clear_session_vars(tokens)

        assert not agent._build_system_prompt.called
        assert agent._cached_system_prompt == stored

    def test_context_cwd_ignores_only_the_scratch_workspace(self, pod_home):
        """resolve_context_cwd skips a session cwd equal to the bound scratch
        dir (doctrine must not re-anchor into an empty workspace) but keeps
        honoring a REAL bound workspace (ACP/TUI) exactly as before."""
        from agent.runtime_cwd import resolve_context_cwd
        from gateway.session_context import set_session_vars

        real_ws = pod_home / "checkout"
        real_ws.mkdir()
        tokens = set_session_vars(
            platform="acp", cwd=str(real_ws), scratch_dir=""
        )
        try:
            assert resolve_context_cwd() == real_ws
        finally:
            clear_session_vars(tokens)
