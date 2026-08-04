"""Tests for gateway/ability_freshness.py (NOL-416).

Covers:
- AbilityInstaller: versioned-dir + symlink-flip installs, legacy real-dir
  migration, version pruning, unsafe tar rejection, marker reads
- _SSEFrameParser: frame assembly, retry hints, comments, multi-line data
- AbilityFreshnessManager: turn-boundary check (short-circuit / reconcile /
  fail-open), between-turns deferral, event handling, heartbeat reporting
- APIServerAdapter wiring: gated on platform credentials

All offline: HTTP is faked at the client seam; no sockets, no timers.
"""

import asyncio
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from gateway.ability_freshness import (
    MARKER,
    AbilityFreshnessManager,
    AbilityInstaller,
    _SSEFrameParser,
    ability_freshness_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tar_gz(files: dict) -> bytes:
    """Build an in-memory tar.gz of {path: content}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeClient:
    """In-memory AbilityFreshnessClient double."""

    def __init__(self):
        self.manifest = []  # [{"slug": ..., "version": ...}]
        self.contents = {}  # slug -> {"version": ..., "content_base64": ...}
        self.reported = []  # list of dicts passed to report_installed
        self.manifest_calls = 0

    def set_content(self, slug, version, files):
        import base64

        self.contents[slug] = {
            "version": version,
            "content_base64": base64.b64encode(_tar_gz(files)).decode(),
        }

    def manifest_abilities(self):
        self.manifest_calls += 1
        return list(self.manifest)

    def ability_content(self, slug):
        return self.contents.get(slug)

    def report_installed(self, versions):
        self.reported.append(dict(versions))


# ---------------------------------------------------------------------------
# ability_freshness_enabled
# ---------------------------------------------------------------------------


def test_enabled_requires_both_credentials():
    assert not ability_freshness_enabled({})
    assert not ability_freshness_enabled({"NOLGIA_API_URL": "https://api"})
    assert not ability_freshness_enabled({"NOLGIA_TOKEN": "nol_x"})
    assert ability_freshness_enabled(
        {"NOLGIA_API_URL": "https://api", "NOLGIA_TOKEN": "nol_x"}
    )


def test_enabled_kill_switch():
    assert not ability_freshness_enabled(
        {
            "NOLGIA_API_URL": "https://api",
            "NOLGIA_TOKEN": "nol_x",
            "NOLGIA_ABILITY_EVENTS": "0",
        }
    )


# ---------------------------------------------------------------------------
# AbilityInstaller
# ---------------------------------------------------------------------------


def test_install_creates_versioned_dir_and_symlink(tmp_path):
    installer = AbilityInstaller(home=tmp_path)
    installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))

    link = tmp_path / "skills" / "short-film"
    assert link.is_symlink()
    assert (link / "SKILL.md").read_text() == "# v1"
    assert os.path.realpath(link) == str(
        tmp_path / "ability-versions" / "short-film" / "1.0.0"
    )
    # Marker is readable both through the link and in the version store —
    # the chart sidecar's up-to-date check keeps working.
    marker = json.loads((link / MARKER).read_text())
    assert marker == {"slug": "short-film", "version": "1.0.0"}
    assert installer.installed_version("short-film") == "1.0.0"


def test_install_accepts_wrapped_package_layout(tmp_path):
    """A tar whose files sit under a single top-level dir installs the same."""
    installer = AbilityInstaller(home=tmp_path)
    installer.install(
        "short-film", "1.0.0", _tar_gz({"short-film/SKILL.md": "# wrapped"})
    )
    assert (tmp_path / "skills" / "short-film" / "SKILL.md").read_text() == "# wrapped"


def test_install_flip_keeps_previous_version(tmp_path):
    installer = AbilityInstaller(home=tmp_path)
    installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))
    installer.install("short-film", "1.1.0", _tar_gz({"SKILL.md": "# v2"}))

    link = tmp_path / "skills" / "short-film"
    assert (link / "SKILL.md").read_text() == "# v2"
    # The previous version's tree is retained for in-flight turns.
    old = tmp_path / "ability-versions" / "short-film" / "1.0.0"
    assert (old / "SKILL.md").read_text() == "# v1"

    # A third install prunes the oldest (keep current + previous).
    installer.install("short-film", "1.2.0", _tar_gz({"SKILL.md": "# v3"}))
    assert not old.exists()
    assert (tmp_path / "ability-versions" / "short-film" / "1.1.0").exists()
    assert installer.installed_version("short-film") == "1.2.0"


def test_install_retention_ignores_directory_mtimes(tmp_path):
    """Retention keeps the version the flip replaced, whatever the mtimes say.

    Version dir timestamps are not a reliable install order: three installs can
    land inside one filesystem timestamp tick, and a filesystem that only keeps
    second-resolution timestamps ties them all. Here 1.0.0 is stamped NEWER
    than 1.1.0 to stand in for that lost ordering — retention must still drop
    1.0.0 and keep 1.1.0, the tree an in-flight turn may be reading.
    """
    installer = AbilityInstaller(home=tmp_path)
    installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))
    installer.install("short-film", "1.1.0", _tar_gz({"SKILL.md": "# v2"}))

    store = tmp_path / "ability-versions" / "short-film"
    os.utime(store / "1.1.0", (1_000_000, 1_000_000))
    os.utime(store / "1.0.0", (2_000_000, 2_000_000))

    installer.install("short-film", "1.2.0", _tar_gz({"SKILL.md": "# v3"}))

    assert not (store / "1.0.0").exists()
    assert (store / "1.1.0" / "SKILL.md").read_text() == "# v2"
    assert (store / "1.2.0" / "SKILL.md").read_text() == "# v3"


def test_install_migrates_legacy_real_dir(tmp_path):
    """The pre-NOL-416 layout (real dir from rmtree+rename) migrates to a
    symlink on the first push install."""
    legacy = tmp_path / "skills" / "short-film"
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text("# legacy")
    (legacy / MARKER).write_text(json.dumps({"slug": "short-film", "version": "0.9.0"}))

    installer = AbilityInstaller(home=tmp_path)
    assert installer.installed_version("short-film") == "0.9.0"
    installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))

    link = tmp_path / "skills" / "short-film"
    assert link.is_symlink()
    assert (link / "SKILL.md").read_text() == "# v1"
    # No stray legacy droppings.
    assert not (tmp_path / "skills" / ".short-film.legacy.tmp").exists()
    assert not (tmp_path / "skills" / ".short-film.link.tmp").exists()


def test_install_rejects_traversal_and_bad_names(tmp_path):
    installer = AbilityInstaller(home=tmp_path)
    with pytest.raises(ValueError):
        installer.install("short-film", "1.0.0", _tar_gz({"../escape": "x"}))
    with pytest.raises(ValueError):
        installer.install("../evil", "1.0.0", _tar_gz({"SKILL.md": "x"}))
    with pytest.raises(ValueError):
        installer.install("short-film", "../1.0", _tar_gz({"SKILL.md": "x"}))
    assert not (tmp_path / "skills" / "short-film").exists()


def test_installed_versions_lists_marker_bearing_entries(tmp_path):
    installer = AbilityInstaller(home=tmp_path)
    installer.install("a-skill", "1.0.0", _tar_gz({"SKILL.md": "a"}))
    installer.install("b-skill", "2.0.0", _tar_gz({"SKILL.md": "b"}))
    # A marker-less dir (e.g. a bundled skill) is not a marketplace install.
    (tmp_path / "skills" / "bundled").mkdir()
    assert installer.installed_versions() == {"a-skill": "1.0.0", "b-skill": "2.0.0"}


# ---------------------------------------------------------------------------
# _SSEFrameParser
# ---------------------------------------------------------------------------


def test_sse_parser_assembles_frames():
    parser = _SSEFrameParser()
    assert parser.feed_line("retry: 3000\n") is None
    assert parser.feed_line("\n") is None  # retry-only block: no frame
    assert parser.feed_line(": heartbeat\n") is None
    assert parser.feed_line("id: 1754300000000000000\n") is None
    assert parser.feed_line("event: ability_published\n") is None
    assert parser.feed_line('data: {"slug":"short-film",\n') is None
    frame = None
    assert parser.feed_line('data:  "version":"1.2.0"}\n') is None
    frame = parser.feed_line("\n")
    assert frame == {
        "event": "ability_published",
        "id": "1754300000000000000",
        "data": '{"slug":"short-film",\n "version":"1.2.0"}',
    }
    assert parser.retry_millis == 3000
    # Parser state resets between frames.
    assert parser.feed_line("event: ready\n") is None
    assert parser.feed_line("data: {}\n") is None
    frame = parser.feed_line("\n")
    assert frame["event"] == "ready"
    assert frame["id"] is None


# ---------------------------------------------------------------------------
# AbilityFreshnessManager
# ---------------------------------------------------------------------------


def _manager(tmp_path, client=None, idle=0):
    return AbilityFreshnessManager(
        installer=AbilityInstaller(home=tmp_path),
        client=client or FakeClient(),
        idle_check=lambda: idle,
        home=tmp_path,
    )


@pytest.mark.asyncio
async def test_ensure_fresh_short_circuits_when_subscriber_live(tmp_path):
    client = FakeClient()
    manager = _manager(tmp_path, client)
    manager.subscriber_live = True

    await manager.ensure_fresh_before_turn()

    assert client.manifest_calls == 0, "a live, caught-up subscription needs no GET"


@pytest.mark.asyncio
async def test_ensure_fresh_reconciles_and_installs_when_behind(tmp_path):
    client = FakeClient()
    client.manifest = [{"slug": "short-film", "version": "1.1.0"}]
    client.set_content("short-film", "1.1.0", {"SKILL.md": "# v2"})
    manager = _manager(tmp_path, client)
    manager.installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))

    await manager.ensure_fresh_before_turn()

    assert manager.installer.installed_version("short-film") == "1.1.0"
    # Visibility: the heartbeat reported the fresh state.
    assert client.reported and client.reported[-1] == {"short-film": "1.1.0"}


@pytest.mark.asyncio
async def test_ensure_fresh_installs_even_while_other_turns_run(tmp_path):
    """The turn-boundary path force-installs: the run about to start must
    never execute stale, even when a concurrent run is mid-turn (the flip is
    atomic and the previous version dir is retained for that run)."""
    client = FakeClient()
    client.manifest = [{"slug": "short-film", "version": "2.0.0"}]
    client.set_content("short-film", "2.0.0", {"SKILL.md": "# v2"})
    manager = _manager(tmp_path, client, idle=1)

    await manager.ensure_fresh_before_turn()

    assert manager.installer.installed_version("short-film") == "2.0.0"


@pytest.mark.asyncio
async def test_ensure_fresh_fails_open_on_platform_outage(tmp_path):
    class ExplodingClient(FakeClient):
        def manifest_abilities(self):
            raise OSError("api unreachable")

    manager = _manager(tmp_path, ExplodingClient())
    await manager.ensure_fresh_before_turn()  # must not raise


@pytest.mark.asyncio
async def test_published_event_defers_install_until_between_turns(tmp_path):
    client = FakeClient()
    client.set_content("short-film", "1.1.0", {"SKILL.md": "# v2"})
    busy = {"count": 1}
    manager = AbilityFreshnessManager(
        installer=AbilityInstaller(home=tmp_path),
        client=client,
        idle_check=lambda: busy["count"],
        home=tmp_path,
    )
    manager.installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))

    # Event arrives mid-turn: recorded, not installed.
    manager.note_published("short-film", "1.1.0")
    await manager._drain_pending()
    assert manager.installer.installed_version("short-film") == "1.0.0"

    # Turn ends -> idle -> the deferred install lands (and is reported).
    busy["count"] = 0
    manager.notify_turn_finished()
    for _ in range(500):
        await asyncio.sleep(0.01)
        if client.reported:
            break
    assert manager.installer.installed_version("short-film") == "1.1.0"
    assert client.reported[-1] == {"short-film": "1.1.0"}


@pytest.mark.asyncio
async def test_up_to_date_event_is_ignored(tmp_path):
    manager = _manager(tmp_path)
    manager.installer.install("short-film", "1.0.0", _tar_gz({"SKILL.md": "# v1"}))
    manager.note_published("short-film", "1.0.0")
    assert not manager._pending


@pytest.mark.asyncio
async def test_ready_frame_with_replay_none_triggers_reconcile(tmp_path):
    client = FakeClient()
    client.manifest = [{"slug": "short-film", "version": "1.0.0"}]
    client.set_content("short-film", "1.0.0", {"SKILL.md": "# v1"})
    manager = _manager(tmp_path, client)

    await manager._handle_frame(
        {"event": "ready", "id": "123", "data": '{"replay":"none"}'}
    )

    assert manager.subscriber_live is True
    assert client.manifest_calls == 1
    assert manager.installer.installed_version("short-film") == "1.0.0"
    assert manager.last_event_id() == "123"


@pytest.mark.asyncio
async def test_ready_frame_with_replay_complete_skips_reconcile(tmp_path):
    client = FakeClient()
    manager = _manager(tmp_path, client)

    await manager._handle_frame(
        {"event": "ready", "id": "456", "data": '{"replay":"complete"}'}
    )

    assert manager.subscriber_live is True
    assert client.manifest_calls == 0
    assert manager.last_event_id() == "456"


@pytest.mark.asyncio
async def test_published_frame_installs_when_idle_and_stores_id(tmp_path):
    client = FakeClient()
    client.set_content("short-film", "1.1.0", {"SKILL.md": "# v2"})
    manager = _manager(tmp_path, client)

    await manager._handle_frame(
        {
            "event": "ability_published",
            "id": "789",
            "data": '{"slug":"short-film","version":"1.1.0"}',
        }
    )

    assert manager.installer.installed_version("short-film") == "1.1.0"
    assert manager.last_event_id() == "789"


@pytest.mark.asyncio
async def test_unknown_event_types_are_ignored(tmp_path):
    manager = _manager(tmp_path)
    await manager._handle_frame({"event": "future_thing", "id": "1", "data": "{}"})
    assert not manager._pending


@pytest.mark.asyncio
async def test_not_entitled_content_is_skipped(tmp_path):
    client = FakeClient()  # no content registered -> ability_content None
    manager = _manager(tmp_path, client)
    manager.note_published("short-film", "1.0.0")
    await manager._drain_pending()
    assert manager.installer.installed_version("short-film") == ""


def test_prompt_cache_invalidated_after_install(tmp_path, monkeypatch):
    calls = []
    import agent.prompt_builder as prompt_builder

    monkeypatch.setattr(
        prompt_builder,
        "clear_skills_system_prompt_cache",
        lambda **kw: calls.append(kw),
    )
    client = FakeClient()
    client.set_content("short-film", "1.0.0", {"SKILL.md": "# v1"})
    manager = _manager(tmp_path, client)

    asyncio.run(manager._install_latest("short-film", "1.0.0"))

    assert calls == [{"clear_snapshot": True}]


# ---------------------------------------------------------------------------
# APIServerAdapter wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_freshness_disabled_without_credentials():
    pytest.importorskip("aiohttp")
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._start_ability_freshness()
    assert adapter._ability_freshness is None
    assert adapter._ability_events_task is None


@pytest.mark.asyncio
async def test_adapter_freshness_starts_with_credentials(monkeypatch, tmp_path):
    pytest.importorskip("aiohttp")
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.setenv("NOLGIA_API_URL", "https://api.invalid")
    monkeypatch.setenv("NOLGIA_TOKEN", "nol_test")

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._start_ability_freshness()
    try:
        assert adapter._ability_freshness is not None
        assert adapter._ability_events_task is not None
        # Turn hooks are safe to call with the manager in place.
        adapter._notify_ability_turn_finished()
        manager = adapter._ability_freshness
        manager.subscriber_live = True
        await adapter._ensure_abilities_fresh()
    finally:
        adapter._ability_freshness.stop()
        adapter._ability_events_task.cancel()
        try:
            await adapter._ability_events_task
        except (asyncio.CancelledError, Exception):
            pass
