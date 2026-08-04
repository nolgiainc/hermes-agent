"""Push-based marketplace ability freshness (NOL-416).

The Nolgia platform publishes new versions of marketplace abilities and pods
must run the latest content without a pod roll and without polling. This
module is the pod half of that design:

1. **Push channel** — a long-lived SSE subscription to the platform's
   ``GET /agent/abilities/events`` stream (``AbilityFreshnessManager
   .run_subscriber``). An ``ability_published`` frame marks the slug pending;
   installation happens only when the gateway is idle (between turns), never
   mid-run. The stream carries a monotonic ``id:`` per frame and honors
   ``Last-Event-ID`` replay, so a pod that was disconnected while a version
   published catches up on reconnect instead of silently staying stale.
2. **Turn-boundary guarantee** — before each agent run executes, the adapter
   calls :meth:`AbilityFreshnessManager.ensure_fresh_before_turn`. When the
   subscription is live and nothing is pending this is a no-network no-op;
   otherwise it reconciles once against ``GET /agent/manifest`` (a single
   GET). This is the hard correctness bar: a turn never executes on a stale
   ability even if every push failed. There is NO periodic polling anywhere
   in this module — the only loop is the subscriber's reconnect backoff,
   which runs only while the connection is down.
3. **Atomic install** — versioned directory plus symlink flip
   (:class:`AbilityInstaller`): content is unpacked into
   ``$HERMES_HOME/ability-versions/<slug>/<version>/`` and
   ``$HERMES_HOME/skills/<slug>`` is flipped to it with ``os.replace`` on a
   pre-created symlink, so a reader never observes a half-written skill dir
   (the legacy sidecar's rmtree+rename had a window with no dir at all). The
   previous version's directory is retained so an in-flight turn that
   resolved paths through the old link keeps working.
4. **Visibility** — after installs (and on connect) the pod reports its
   installed versions upstream via ``POST /agent/heartbeat`` so drift is
   observable in the admin surface rather than assumed. Reports are
   event-driven, never scheduled.

Stdlib-only for everything blocking (mirrors gateway/platforms/nolgia_assets:
must import, and fail soft, in any gateway process); aiohttp is imported
lazily inside the subscriber and its absence just disables the push half —
the turn-boundary check still guarantees correctness.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Marker file the platform sync tooling writes into an installed ability dir;
# {"slug": ..., "version": ...}. Shared contract with the chart's
# sync-skills.py sidecar — both writers converge on the same layout.
MARKER = ".nolgia-skill.json"

# Directory (under HERMES_HOME, deliberately OUTSIDE skills/) holding the
# versioned content store. Keeping it out of skills/ means the skill indexers
# (os.walk(followlinks=True)) never double-walk old versions.
VERSIONS_DIRNAME = "ability-versions"

# How many versions to retain per slug (current + previous): the previous
# version stays on disk so an in-flight turn that resolved real paths through
# the pre-flip link keeps a working tree.
KEEP_VERSIONS = 2

_STATE_FILENAME = "ability-events-last-id"


def ability_freshness_enabled(environ: Optional[Dict[str, str]] = None) -> bool:
    """True when the pod holds platform credentials and the feature is not
    explicitly disabled (NOLGIA_ABILITY_EVENTS=0)."""
    env = os.environ if environ is None else environ
    if str(env.get("NOLGIA_ABILITY_EVENTS", "")).strip() == "0":
        return False
    return bool(env.get("NOLGIA_API_URL")) and bool(env.get("NOLGIA_TOKEN"))


def _api_base() -> str:
    base = (os.environ.get("NOLGIA_API_URL") or "").rstrip("/")
    if base and not base.endswith("/v1"):
        base += "/v1"
    return base


def _safe_extract(tar_bytes: bytes, dest: str) -> None:
    """Extract a tar.gz, rejecting traversal and link members (same guard as
    the chart's sync-skills.py)."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(dest, member.name))
            if not target.startswith(os.path.realpath(dest) + os.sep):
                raise ValueError(f"unsafe tar member: {member.name}")
            if member.islnk() or member.issym():
                raise ValueError(f"link tar member rejected: {member.name}")
        tar.extractall(dest)


class AbilityInstaller:
    """Atomic versioned-dir + symlink-flip installer for marketplace abilities.

    Layout::

        $HERMES_HOME/ability-versions/<slug>/<version>/   # content + marker
        $HERMES_HOME/skills/<slug> -> ../ability-versions/<slug>/<version>

    A flip is one ``os.replace`` of a pre-created symlink: readers see the
    old complete tree or the new complete tree, never anything between. A
    pre-existing REAL directory (the legacy rmtree+rename layout) is migrated
    on first install: renamed aside, link flipped in, aside tree removed.
    """

    def __init__(self, home: Optional[Path] = None) -> None:
        self._home = Path(home) if home is not None else None

    @property
    def home(self) -> Path:
        return self._home if self._home is not None else get_hermes_home()

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def versions_dir(self) -> Path:
        return self.home / VERSIONS_DIRNAME

    def installed_version(self, slug: str) -> str:
        """The version recorded in the slug's marker ('' when absent/unreadable)."""
        marker = self.skills_dir / slug / MARKER
        try:
            with open(marker, encoding="utf-8") as fh:
                return str(json.load(fh).get("version") or "")
        except (OSError, ValueError):
            return ""

    def installed_versions(self) -> Dict[str, str]:
        """slug -> version for every marker-bearing entry under skills/."""
        versions: Dict[str, str] = {}
        try:
            entries = sorted(os.listdir(self.skills_dir))
        except OSError:
            return versions
        for entry in entries:
            version = self.installed_version(entry)
            if version:
                versions[entry] = version
        return versions

    def install(self, slug: str, version: str, tar_bytes: bytes) -> None:
        """Materialize one version and atomically flip skills/<slug> to it."""
        if not slug or "/" in slug or slug.startswith("."):
            raise ValueError(f"invalid ability slug: {slug!r}")
        if not version or "/" in version or version.startswith("."):
            raise ValueError(f"invalid ability version: {version!r}")

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        slug_store = self.versions_dir / slug
        slug_store.mkdir(parents=True, exist_ok=True)

        # 1. Stage the content next to its final home (same filesystem).
        staging = tempfile.mkdtemp(prefix=f".{version}-", dir=slug_store)
        try:
            _safe_extract(tar_bytes, staging)
            # Accept both package layouts: files at the tar root, or a single
            # top-level directory wrapping them (same as the chart sidecar).
            entries = os.listdir(staging)
            root = staging
            if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
                root = os.path.join(staging, entries[0])
            with open(os.path.join(root, MARKER), "w", encoding="utf-8") as fh:
                json.dump({"slug": slug, "version": version}, fh)

            version_dir = slug_store / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            os.rename(root, version_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        # 2. Flip the link. The symlink target is RELATIVE so the layout
        # survives the volume being mounted at a different absolute path.
        link_path = self.skills_dir / slug
        target = os.path.join("..", VERSIONS_DIRNAME, slug, version)
        tmp_link = self.skills_dir / f".{slug}.link.tmp"
        try:
            os.unlink(tmp_link)
        except OSError:
            pass
        os.symlink(target, tmp_link)

        legacy_aside: Optional[Path] = None
        if link_path.is_dir() and not link_path.is_symlink():
            # Legacy flat layout: move the real dir aside first (os.replace
            # cannot atomically replace a directory with a symlink). The
            # aside window is not atomic, but installs only run between
            # turns, and the flip itself still is.
            legacy_aside = self.skills_dir / f".{slug}.legacy.tmp"
            if legacy_aside.exists() or legacy_aside.is_symlink():
                shutil.rmtree(legacy_aside, ignore_errors=True)
            os.rename(link_path, legacy_aside)
        try:
            os.replace(tmp_link, link_path)
        except OSError:
            # Roll the legacy dir back so the slug never dangles.
            if legacy_aside is not None and not link_path.exists():
                os.rename(legacy_aside, link_path)
                legacy_aside = None
            try:
                os.unlink(tmp_link)
            except OSError:
                pass
            raise
        if legacy_aside is not None:
            shutil.rmtree(legacy_aside, ignore_errors=True)

        self._prune_old_versions(slug, keep=version)
        logger.info("ability %s: installed %s (symlink flip)", slug, version)

    def _prune_old_versions(self, slug: str, keep: str) -> None:
        """Retain the active version plus the newest KEEP_VERSIONS-1 others."""
        slug_store = self.versions_dir / slug
        try:
            candidates = [
                entry
                for entry in os.listdir(slug_store)
                if not entry.startswith(".") and entry != keep
            ]
        except OSError:
            return
        candidates.sort(
            key=lambda entry: (slug_store / entry).stat().st_mtime, reverse=True
        )
        for stale in candidates[KEEP_VERSIONS - 1 :]:
            shutil.rmtree(slug_store / stale, ignore_errors=True)


class AbilityFreshnessClient:
    """Blocking Nolgia API client for the freshness flows (stdlib urllib —
    call from a thread via asyncio.to_thread). Auth is the pod-wide
    NOLGIA_TOKEN: freshness is pod plumbing, never turn-scoped work."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        body = None
        headers = {
            "Authorization": "Bearer " + (os.environ.get("NOLGIA_TOKEN") or ""),
            "Accept": "application/json",
            "X-Nolgia-Surface": "hermes",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            _api_base() + path, data=body, headers=headers, method=method
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            raw = response.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def manifest_abilities(self) -> List[Dict[str, str]]:
        """GET /agent/manifest -> [{'slug': ..., 'version': ...}]."""
        manifest = self._request("GET", "/agent/manifest") or {}
        abilities = manifest.get("abilities") or []
        return [
            {"slug": str(a.get("slug") or ""), "version": str(a.get("version") or "")}
            for a in abilities
            if a.get("slug")
        ]

    def ability_content(self, slug: str) -> Optional[Dict[str, Any]]:
        """GET /abilities/{slug}/content -> {'version', 'content_base64'};
        None when not entitled (402)."""
        try:
            return self._request("GET", f"/abilities/{slug}/content")
        except urllib.error.HTTPError as err:
            if err.code == 402:
                logger.info("ability %s: not entitled (402) — skipping", slug)
                return None
            raise

    def report_installed(self, versions: Dict[str, str]) -> None:
        """POST /agent/heartbeat with the installed versions (best-effort;
        callers swallow errors — visibility must never break the pod)."""
        abilities = [
            {"slug": slug, "version": version}
            for slug, version in sorted(versions.items())
        ]
        self._request("POST", "/agent/heartbeat", {"abilities": abilities})


class _SSEFrameParser:
    """Incremental SSE parser: feed decoded lines, get complete frames."""

    def __init__(self) -> None:
        self._event = ""
        self._data: List[str] = []
        self._id: Optional[str] = None
        self.retry_millis: Optional[int] = None

    def feed_line(self, line: str) -> Optional[Dict[str, Any]]:
        line = line.rstrip("\r\n")
        if line == "":
            if not self._event and not self._data and self._id is None:
                return None
            frame = {
                "event": self._event or "message",
                "data": "\n".join(self._data),
                "id": self._id,
            }
            self._event = ""
            self._data = []
            self._id = None
            return frame
        if line.startswith(":"):
            return None  # comment / heartbeat
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            self._id = value
        elif field == "retry":
            try:
                self.retry_millis = int(value)
            except ValueError:
                pass
        return None


class AbilityFreshnessManager:
    """Coordinates the subscriber, the between-turns installer, and the
    turn-boundary check. One instance per gateway process."""

    def __init__(
        self,
        *,
        installer: Optional[AbilityInstaller] = None,
        client: Optional[AbilityFreshnessClient] = None,
        idle_check: Optional[Callable[[], int]] = None,
        home: Optional[Path] = None,
    ) -> None:
        self.installer = installer or AbilityInstaller(home=home)
        self.client = client or AbilityFreshnessClient()
        # idle_check returns the number of in-flight/pending agent turns
        # (APIServerAdapter.active_agent_work_count); 0 == between turns.
        self._idle_check = idle_check or (lambda: 0)
        self._pending: Dict[str, str] = {}  # slug -> latest known version
        self._install_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        # subscriber_live: the SSE stream is connected and caught up, so the
        # turn-boundary check can skip its network round-trip.
        self.subscriber_live = False
        self._stopped = False

    # -- Last-Event-ID persistence -------------------------------------

    def _state_path(self) -> Path:
        return self.installer.home / "state" / _STATE_FILENAME

    def last_event_id(self) -> str:
        try:
            return self._state_path().read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _store_event_id(self, event_id: str) -> None:
        if not event_id:
            return
        with self._state_lock:
            try:
                path = self._state_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(event_id, encoding="utf-8")
                os.replace(tmp, path)
            except OSError as exc:
                logger.debug("could not persist ability event id: %s", exc)

    # -- Turn-boundary check -------------------------------------------

    async def ensure_fresh_before_turn(self, timeout: float = 20.0) -> None:
        """Guarantee the turn about to execute runs the latest ability
        versions. No-network no-op when the subscription is live and nothing
        is pending; otherwise one manifest GET + installs. Fail-open on
        platform unavailability (a freshness outage must not take the agent
        down), loudly logged."""
        if self._stopped:
            return
        if self.subscriber_live and not self._pending:
            return
        try:
            await asyncio.wait_for(self._reconcile_and_drain(), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            logger.warning(
                "ability freshness turn-boundary check failed (running on "
                "currently installed versions): %s",
                exc,
            )

    async def _reconcile_and_drain(self) -> None:
        """One manifest read -> mark stale slugs pending -> install them."""
        manifest = await asyncio.to_thread(self.client.manifest_abilities)
        installed = await asyncio.to_thread(self.installer.installed_versions)
        for entry in manifest:
            slug, version = entry["slug"], entry["version"]
            if version and installed.get(slug) != version:
                self._pending[slug] = version
        await self._drain_pending(force=True)

    # -- Event handling -------------------------------------------------

    def note_published(self, slug: str, version: str) -> None:
        """Record one push event; install happens between turns."""
        if not slug or not version:
            return
        if self.installer.installed_version(slug) == version:
            return
        self._pending[slug] = version

    def notify_turn_finished(self) -> None:
        """Called by the adapter at every turn end: if publishes arrived while
        the turn ran, install them now that the gateway is (possibly) idle.
        Schedules onto the running loop; never blocks the caller."""
        if not self._pending or self._stopped:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._drain_pending())

    async def _drain_pending(self, force: bool = False) -> None:
        """Install pending versions. Unless force (the turn-boundary path,
        where the about-to-run turn has by definition not started), only runs
        while the gateway is idle — never mid-turn."""
        async with self._install_lock:
            while self._pending:
                if not force and self._idle_check() > 0:
                    return  # a turn started; its end (or its own turn-boundary
                    # check) resumes the drain
                slug, version = next(iter(self._pending.items()))
                self._pending.pop(slug, None)
                try:
                    await self._install_latest(slug, version)
                except Exception as exc:  # noqa: BLE001 - keep draining others
                    logger.warning("ability %s: install failed: %s", slug, exc)
            await self._report_installed()

    async def _install_latest(self, slug: str, version: str) -> None:
        if self.installer.installed_version(slug) == version:
            return
        payload = await asyncio.to_thread(self.client.ability_content, slug)
        if not payload:
            return
        content = base64.b64decode(payload.get("content_base64") or "")
        served_version = str(payload.get("version") or version)
        await asyncio.to_thread(self.installer.install, slug, served_version, content)
        self._invalidate_prompt_cache()

    async def _report_installed(self) -> None:
        try:
            versions = await asyncio.to_thread(self.installer.installed_versions)
            await asyncio.to_thread(self.client.report_installed, versions)
        except Exception as exc:  # noqa: BLE001 - visibility is best-effort
            logger.debug("ability heartbeat failed: %s", exc)

    def _invalidate_prompt_cache(self) -> None:
        """The in-process skills prompt cache is keyed by directory, not
        mtime — a symlink flip is invisible to it. Drop it (and the disk
        snapshot) so the NEXT turn's system prompt indexes the new content."""
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache

            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not clear skills prompt cache: %s", exc)

    # -- Subscriber ------------------------------------------------------

    def stop(self) -> None:
        self._stopped = True

    async def run_subscriber(self) -> None:
        """Long-lived SSE subscription with reconnect backoff. This loop only
        spins on CONNECTION FAILURE (protocol-level SSE reconnect, seeded by
        the server's retry hint) — while the stream is healthy it sits in a
        read, event-driven end to end."""
        try:
            import aiohttp  # noqa: F401  (lazy: absence disables push only)
        except Exception:  # noqa: BLE001
            logger.info(
                "aiohttp unavailable — ability push channel disabled; the "
                "turn-boundary check still guarantees freshness"
            )
            return

        backoff = 3.0
        while not self._stopped:
            try:
                retry_hint = await self._subscribe_once()
                backoff = max(retry_hint or 3.0, 1.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.info("ability events stream disconnected: %s", exc)
            self.subscriber_live = False
            if self._stopped:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def _subscribe_once(self) -> Optional[float]:
        """One SSE connection lifetime. Returns the server retry hint (s)."""
        import aiohttp

        headers = {
            "Authorization": "Bearer " + (os.environ.get("NOLGIA_TOKEN") or ""),
            "Accept": "text/event-stream",
            "X-Nolgia-Surface": "hermes",
        }
        last_id = self.last_event_id()
        if last_id:
            headers["Last-Event-ID"] = last_id

        parser = _SSEFrameParser()
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                _api_base() + "/agent/abilities/events", headers=headers
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"ability events stream refused: HTTP {response.status}"
                    )
                logger.info("ability events stream connected")
                async for raw_line in response.content:
                    if self._stopped:
                        return parser.retry_millis and parser.retry_millis / 1000.0
                    frame = parser.feed_line(raw_line.decode("utf-8", "replace"))
                    if frame is not None:
                        await self._handle_frame(frame)
        return parser.retry_millis and parser.retry_millis / 1000.0

    async def _handle_frame(self, frame: Dict[str, Any]) -> None:
        event = frame.get("event") or ""
        if event == "ready":
            self.subscriber_live = True
            try:
                data = json.loads(frame.get("data") or "{}")
            except ValueError:
                data = {}
            if frame.get("id"):
                self._store_event_id(str(frame["id"]))
            # none/truncated replay: events may have been missed entirely —
            # reconcile once from the manifest (event-driven, not periodic).
            if data.get("replay") in ("none", "truncated"):
                try:
                    await self._reconcile_and_drain()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ability reconcile after connect failed: %s", exc)
            return
        if event == "ability_published":
            try:
                data = json.loads(frame.get("data") or "{}")
            except ValueError:
                return
            self.note_published(
                str(data.get("slug") or ""), str(data.get("version") or "")
            )
            if frame.get("id"):
                self._store_event_id(str(frame["id"]))
            if self._pending and self._idle_check() == 0:
                await self._drain_pending()
            return
        # Unknown event types are ignored by contract (forward-compatible).
