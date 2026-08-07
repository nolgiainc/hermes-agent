"""Nolgia platform media GC: delete local media once it is CONFIRMED in the library.

Every generation on the Nolgia platform is persisted server-side first — the
generation job writes the asset to the platform's GCS bucket and it appears in
the user's library; the pod-local copy (``nolgia gen --out``, a chat-delivered
composite, a screenshot the agent re-hosted) is redundant the moment the
library holds the same bytes. Historically those local copies were never
reclaimed, which is what repeatedly filled agent PVCs and forced manual
cleanup jobs. This module makes the reclaim automatic — and *strictly
confirmed-only*: a local file is deleted ONLY when the platform library is
proven to hold it. When in doubt, keep. This is customer data.

Two mechanisms, both logged per deletion (path, size, confirmation basis):

1. **Post-upload hook** (:func:`on_confirmed_upload`) — called by
   ``nolgia_assets`` right after the three-step signed upload flow returns an
   asset id (``POST /assets/uploads/{id}/complete`` verifies the object
   server-side, so a returned id IS the library-persistence confirmation).
   The just-uploaded file is deleted immediately when it is still byte-for-byte
   the file that was uploaded (same size + mtime_ns as the pre-upload stat) and
   lives under ``HERMES_HOME``. Every confirmed upload is also recorded in a
   ledger so a later ``MEDIA:`` reference to the deleted path can resolve
   straight to ``asset:<uuid>`` instead of degrading to a dead filename.

2. **Safety-net sweeper** (:func:`sweep_once`, run at gateway startup and
   periodically) — walks ``HERMES_HOME`` for media files older than a
   conservative age threshold and deletes each ONLY after confirming the
   library holds the same content:

   - *ledger + API*: a ledger entry matches the file's (path, size, mtime_ns)
     → ``GET /assets/{id}`` must return the asset as ``ready``; or
   - *content match*: the file's MD5 equals the GCS object hash of a
     same-size library asset. The asset listing's ``signed_url`` points at
     GCS, and a 1-byte ranged GET returns ``x-goog-hash: md5=<base64>`` plus
     the full object size in ``content-range`` — an exact bytes-level match,
     no full download needed. This is what reclaims ``nolgia gen --out``
     downloads (byte-identical to the generated asset by construction).

   Anything unconfirmed — no ledger entry, no size match, hash mismatch, API
   error, file modified since confirmation — is kept, always.

Config (env, deployment-owned):

- ``NOLGIA_MEDIA_GC=0``            — master off switch (default ON in
  platform mode, i.e. when ``NOLGIA_API_URL`` + ``NOLGIA_TOKEN`` are present).
- ``NOLGIA_MEDIA_GC_ON_UPLOAD=0``  — disable only the immediate post-upload
  deletion (ledger recording continues; the sweeper still runs).
- ``NOLGIA_MEDIA_GC_MIN_AGE_HOURS``       — sweeper age threshold (default 6).
- ``NOLGIA_MEDIA_GC_INTERVAL_SECONDS``    — sweep period (default 3600).

Stdlib-only and fail-soft everywhere (mirrors ``nolgia_assets``): GC plumbing
must never take an upload, a turn, or the gateway down.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import stat as stat_module
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LEDGER_DIRNAME = "media-gc"
_LEDGER_FILENAME = "ledger.jsonl"

# Rewrite the ledger keeping only the newest entries once it grows past this
# many lines (bounds unbounded append growth on long-lived pods).
_LEDGER_COMPACT_LINES = 20000
_LEDGER_KEEP_LINES = 10000

_DEFAULT_MIN_AGE_HOURS = 6.0
_DEFAULT_INTERVAL_SECONDS = 3600.0
# Give the pod time to settle (skills sync, ability installs) before the
# startup sweep touches the network.
STARTUP_SWEEP_DELAY_SECONDS = 120.0

# Sweep work bounds: at most this many hash confirmations (each = one local
# MD5 pass + one 1-byte ranged GET) per sweep; the rest wait for the next one.
_MAX_CONFIRMATIONS_PER_SWEEP = 500
# Asset listing pages fetched to build the size index (newest first,
# limit=100 each). 5000 newest assets is far beyond what a pod's local
# accumulation window can outrun at one sweep per hour.
_MAX_LIST_PAGES = 50

_LIST_TIMEOUT_SECONDS = 30.0
_GET_ASSET_TIMEOUT_SECONDS = 30.0
_HASH_PROBE_TIMEOUT_SECONDS = 30.0

# Directory names never descended into (relative to any level): dot-dirs are
# skipped wholesale (covers .git — never touch a repo working tree — plus
# .ssh/.cache and friends), and these Hermes/platform-managed trees hold
# nothing that is ever "redundant local media".
_PROTECTED_DIRNAMES = frozenset(
    {
        "logs",
        "sessions",
        "memories",
        "skills",
        "plugins",
        "ability-versions",
        _LEDGER_DIRNAME,
        "disk-cleanup",
        "node_modules",
    }
)

_MD5_HASH_RE = re.compile(r"md5=([A-Za-z0-9+/=]+)")


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------


def media_gc_enabled() -> bool:
    """Master gate: platform mode present and not opted out."""
    if os.environ.get("NOLGIA_MEDIA_GC", "").strip() == "0":
        return False
    return bool(os.environ.get("NOLGIA_API_URL", "").strip()) and bool(
        os.environ.get("NOLGIA_TOKEN", "").strip()
    )


def _on_upload_delete_enabled() -> bool:
    if not media_gc_enabled():
        return False
    return os.environ.get("NOLGIA_MEDIA_GC_ON_UPLOAD", "").strip() != "0"


def _min_age_seconds() -> float:
    raw = os.environ.get("NOLGIA_MEDIA_GC_MIN_AGE_HOURS", "").strip()
    try:
        hours = float(raw) if raw else _DEFAULT_MIN_AGE_HOURS
    except ValueError:
        hours = _DEFAULT_MIN_AGE_HOURS
    return max(hours, 0.0) * 3600.0


def sweep_interval_seconds() -> float:
    raw = os.environ.get("NOLGIA_MEDIA_GC_INTERVAL_SECONDS", "").strip()
    try:
        interval = float(raw) if raw else _DEFAULT_INTERVAL_SECONDS
    except ValueError:
        interval = _DEFAULT_INTERVAL_SECONDS
    return max(interval, 60.0)


def _media_extensions() -> frozenset:
    """The platform-uploadable extension set (single source: nolgia_assets)."""
    from gateway.platforms import nolgia_assets

    return frozenset(nolgia_assets._EXT_CONTENT_TYPES.keys())


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class _Ledger:
    """Append-only JSONL record of confirmed uploads and GC deletions.

    ``uploaded`` events map a local file identity (path, size, mtime_ns) to
    the library asset id the platform confirmed for those exact bytes.
    ``deleted`` events remember which asset a *removed* path corresponded to,
    so egress can rewrite a stale ``MEDIA:`` reference to ``asset:<uuid>``
    instead of a dead filename.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or (get_hermes_home() / _LEDGER_DIRNAME / _LEDGER_FILENAME)
        self._lock = threading.Lock()
        self._loaded = False
        # (path, size, mtime_ns) -> asset_id for confirmed uploads
        self._uploaded: Dict[Tuple[str, int, int], str] = {}
        # path -> asset_id for files this module deleted
        self._deleted: Dict[str, str] = {}

    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw_lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in raw_lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            self._apply_locked(entry)

    def _apply_locked(self, entry: dict) -> None:
        event = entry.get("event")
        path = entry.get("path")
        asset_id = entry.get("asset_id")
        if not isinstance(path, str) or not isinstance(asset_id, str):
            return
        if event == "uploaded":
            size = entry.get("size")
            mtime_ns = entry.get("mtime_ns")
            if isinstance(size, int) and isinstance(mtime_ns, int):
                self._uploaded[(path, size, mtime_ns)] = asset_id
        elif event == "deleted":
            self._deleted[path] = asset_id

    def _append_locked(self, entry: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._maybe_compact_locked()
        except OSError:
            logger.debug("[nolgia_media_gc] ledger append failed", exc_info=True)

    def _maybe_compact_locked(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) <= _LEDGER_COMPACT_LINES:
                return
            keep = lines[-_LEDGER_KEEP_LINES:]
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.writelines(keep)
            os.replace(tmp, self._path)
        except OSError:
            logger.debug("[nolgia_media_gc] ledger compaction failed", exc_info=True)

    def record_uploaded(self, path: str, size: int, mtime_ns: int, asset_id: str) -> None:
        with self._lock:
            self._load_locked()
            entry = {
                "event": "uploaded",
                "path": path,
                "size": size,
                "mtime_ns": mtime_ns,
                "asset_id": asset_id,
                "ts": time.time(),
            }
            self._apply_locked(entry)
            self._append_locked(entry)

    def record_deleted(self, path: str, asset_id: str) -> None:
        with self._lock:
            self._load_locked()
            entry = {
                "event": "deleted",
                "path": path,
                "asset_id": asset_id,
                "ts": time.time(),
            }
            self._apply_locked(entry)
            self._append_locked(entry)

    def uploaded_asset_for(self, path: str, size: int, mtime_ns: int) -> Optional[str]:
        with self._lock:
            self._load_locked()
            return self._uploaded.get((path, size, mtime_ns))

    def deleted_asset_for(self, path: str) -> Optional[str]:
        with self._lock:
            self._load_locked()
            return self._deleted.get(path)


_ledger: Optional[_Ledger] = None
_ledger_lock = threading.Lock()


def _get_ledger() -> _Ledger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = _Ledger()
        return _ledger


def _reset_ledger_for_tests() -> None:
    global _ledger
    with _ledger_lock:
        _ledger = None


# ---------------------------------------------------------------------------
# Shared deletion primitive
# ---------------------------------------------------------------------------


def _is_under_hermes_home(path: Path) -> bool:
    try:
        home = get_hermes_home().resolve()
        path.resolve().relative_to(home)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _delete_confirmed(
    path: Path, size: int, mtime_ns: int, asset_id: str, basis: str
) -> bool:
    """Unlink ``path`` iff it is still exactly the confirmed bytes. Never raises.

    Re-stats immediately before the unlink: the file must still be a regular
    non-symlink file with the confirmed size and mtime_ns (i.e. untouched
    since the library was proven to hold this content). Any drift or error
    keeps the file.
    """
    try:
        stat = path.lstat()
        if not stat_module.S_ISREG(stat.st_mode):
            return False
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            logger.debug(
                "[nolgia_media_gc] %s changed since confirmation; keeping", path
            )
            return False
        path.unlink()
    except OSError:
        logger.debug("[nolgia_media_gc] could not delete %s", path, exc_info=True)
        return False
    _get_ledger().record_deleted(str(path), asset_id)
    logger.info(
        "[nolgia_media_gc] deleted %s (%d bytes; confirmed in library: %s, asset:%s)",
        path,
        size,
        basis,
        asset_id,
    )
    return True


# ---------------------------------------------------------------------------
# Post-upload hook (called from nolgia_assets on egress upload success)
# ---------------------------------------------------------------------------


def on_confirmed_upload(path: Path, size: int, mtime_ns: int, asset_id: str) -> None:
    """Record + (optionally) delete a file the platform just confirmed.

    ``size``/``mtime_ns`` are the stat values of the exact bytes that were
    uploaded (taken before the PUT); ``asset_id`` came back from
    ``POST /assets/uploads/{id}/complete``, which verifies the stored object
    server-side — the strongest confirmed-in-library signal that exists.
    Fail-soft: never raises into the egress path.
    """
    try:
        if not media_gc_enabled():
            return
        resolved = str(path)
        _get_ledger().record_uploaded(resolved, size, mtime_ns, asset_id)
        if not _on_upload_delete_enabled():
            return
        if path.suffix.lower() not in _media_extensions():
            return
        if not _is_under_hermes_home(path):
            logger.debug(
                "[nolgia_media_gc] %s outside HERMES_HOME; keeping", path
            )
            return
        _delete_confirmed(
            path, size, mtime_ns, asset_id, basis="upload-complete API confirmation"
        )
    except Exception:
        logger.debug("[nolgia_media_gc] on_confirmed_upload failed", exc_info=True)


def deleted_asset_reference(path_text: str) -> Optional[str]:
    """Asset id previously confirmed for a path this module has deleted.

    Lets egress rewrite a ``MEDIA:`` tag naming a GC-deleted file to the
    library asset that holds those bytes instead of a dead filename.
    """
    try:
        if not media_gc_enabled():
            return None
        candidate = os.path.normpath(os.path.expanduser(path_text.strip()))
        if not os.path.isabs(candidate):
            return None
        ledger = _get_ledger()
        found = ledger.deleted_asset_for(candidate)
        if found:
            return found
        # The ledger stores fully-resolved paths (validate_media_delivery_path
        # resolves symlinks); realpath keeps the missing leaf while resolving
        # any still-existing ancestor symlinks the raw tag went through.
        return ledger.deleted_asset_for(os.path.realpath(candidate))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Platform API helpers (sweeper side)
# ---------------------------------------------------------------------------


def _api_get(path: str, timeout: float) -> dict:
    from gateway.platforms import nolgia_assets

    return nolgia_assets._http_json("GET", path, None, timeout)


def _asset_ready_via_api(asset_id: str, size: int) -> bool:
    """True iff ``GET /assets/{id}`` shows a ready asset of the expected size."""
    try:
        asset = _api_get(f"/assets/{asset_id}", _GET_ASSET_TIMEOUT_SECONDS)
    except Exception:
        logger.debug(
            "[nolgia_media_gc] GET /assets/%s failed; keeping file", asset_id
        )
        return False
    if not isinstance(asset, dict) or str(asset.get("id", "")) != asset_id:
        return False
    status = str(asset.get("status") or "ready")
    if status != "ready":
        return False
    size_bytes = asset.get("size_bytes")
    if isinstance(size_bytes, int) and size_bytes != size:
        return False
    return True


def _gcs_object_md5(signed_url: str) -> Optional[Tuple[str, int]]:
    """(base64 MD5, total size) of the GCS object behind a signed URL.

    A 1-byte ranged GET: GCS returns object metadata headers
    (``x-goog-hash: md5=...``) and the full size in ``content-range`` without
    transferring the object. Returns None when the hash is unavailable
    (composite objects, non-GCS URL, any error).
    """
    try:
        request = urllib.request.Request(
            signed_url, headers={"Range": "bytes=0-0"}, method="GET"
        )
        with urllib.request.urlopen(
            request, timeout=_HASH_PROBE_TIMEOUT_SECONDS
        ) as response:
            response.read()
            hashes = response.headers.get_all("x-goog-hash") or []
            content_range = response.headers.get("Content-Range", "")
    except Exception:
        logger.debug("[nolgia_media_gc] hash probe failed", exc_info=True)
        return None
    md5_b64 = None
    for value in hashes:
        match = _MD5_HASH_RE.search(value)
        if match:
            md5_b64 = match.group(1)
            break
    if not md5_b64:
        return None
    total = None
    if "/" in content_range:
        try:
            total = int(content_range.rsplit("/", 1)[1])
        except ValueError:
            total = None
    if total is None:
        return None
    return md5_b64, total


def _local_md5_b64(path: Path) -> Optional[str]:
    try:
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return base64.b64encode(digest.digest()).decode("ascii")
    except OSError:
        return None


def _build_library_size_index() -> Dict[int, List[dict]]:
    """size_bytes → [asset, ...] for the user's newest library assets."""
    index: Dict[int, List[dict]] = {}
    cursor = None
    for _ in range(_MAX_LIST_PAGES):
        path = "/assets?limit=100"
        if cursor:
            path += "&cursor=" + urllib.parse.quote(str(cursor))
        try:
            page = _api_get(path, _LIST_TIMEOUT_SECONDS)
        except Exception:
            logger.debug("[nolgia_media_gc] asset listing failed", exc_info=True)
            break
        items = page.get("items") if isinstance(page, dict) else None
        if not isinstance(items, list) or not items:
            break
        for asset in items:
            if not isinstance(asset, dict):
                continue
            size_bytes = asset.get("size_bytes")
            if isinstance(size_bytes, int) and size_bytes > 0:
                index.setdefault(size_bytes, []).append(asset)
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return index


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------


def _iter_candidate_files(root: Path, min_age_seconds: float):
    """Yield (path, stat) for old-enough media files under ``root``.

    Skips dot-directories (which covers ``.git`` working trees), the
    protected Hermes-managed trees, and every symlink.
    """
    extensions = _media_extensions()
    now = time.time()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".")
            and name not in _PROTECTED_DIRNAMES
            and not os.path.islink(os.path.join(dirpath, name))
        ]
        for filename in filenames:
            if filename.startswith("."):
                continue
            if os.path.splitext(filename)[1].lower() not in extensions:
                continue
            path = Path(dirpath) / filename
            try:
                stat = path.lstat()
            except OSError:
                continue
            if not stat_module.S_ISREG(stat.st_mode):
                continue
            if stat.st_size <= 0:
                continue
            if now - stat.st_mtime < min_age_seconds:
                continue
            yield path, stat


def sweep_once(root: Optional[Path] = None) -> Tuple[int, int]:
    """One conservative confirm-then-delete pass. Returns (deleted, freed_bytes).

    Blocking (disk + network) — call off the event loop. Never raises.
    """
    if not media_gc_enabled():
        return (0, 0)
    try:
        base = (root or get_hermes_home()).resolve()
    except OSError:
        return (0, 0)
    ledger = _get_ledger()
    deleted = 0
    freed = 0
    confirmations = 0
    size_index: Optional[Dict[int, List[dict]]] = None
    try:
        for path, stat in _iter_candidate_files(base, _min_age_seconds()):
            if confirmations >= _MAX_CONFIRMATIONS_PER_SWEEP:
                logger.info(
                    "[nolgia_media_gc] sweep confirmation budget reached; "
                    "remaining candidates wait for the next sweep"
                )
                break

            # Basis 1: our own confirmed-upload ledger, re-verified via API.
            asset_id = ledger.uploaded_asset_for(
                str(path), stat.st_size, stat.st_mtime_ns
            )
            if asset_id:
                confirmations += 1
                if _asset_ready_via_api(asset_id, stat.st_size) and _delete_confirmed(
                    path,
                    stat.st_size,
                    stat.st_mtime_ns,
                    asset_id,
                    basis="upload ledger + GET /assets API re-check",
                ):
                    deleted += 1
                    freed += stat.st_size
                continue

            # Basis 2: exact content match against the library (size + MD5
            # against the GCS object hash). Build the size index lazily —
            # only when a candidate actually needs it.
            if size_index is None:
                size_index = _build_library_size_index()
            candidates = size_index.get(stat.st_size)
            if not candidates:
                continue
            confirmations += 1
            local_md5 = _local_md5_b64(path)
            if not local_md5:
                continue
            for asset in candidates:
                signed_url = asset.get("signed_url")
                remote_id = str(asset.get("id") or "")
                if not signed_url or not remote_id:
                    continue
                probe = _gcs_object_md5(str(signed_url))
                if probe is None:
                    continue
                remote_md5, remote_size = probe
                if remote_size != stat.st_size or remote_md5 != local_md5:
                    continue
                if _delete_confirmed(
                    path,
                    stat.st_size,
                    stat.st_mtime_ns,
                    remote_id,
                    basis="size + GCS MD5 content match",
                ):
                    deleted += 1
                    freed += stat.st_size
                break
    except Exception:
        logger.warning("[nolgia_media_gc] sweep aborted", exc_info=True)
    if deleted:
        logger.info(
            "[nolgia_media_gc] sweep reclaimed %d bytes across %d confirmed-in-library files",
            freed,
            deleted,
        )
    return (deleted, freed)


async def run_sweeper() -> None:
    """Startup + periodic sweep loop (started by the API server in platform mode)."""
    import asyncio

    await asyncio.sleep(STARTUP_SWEEP_DELAY_SECONDS)
    while True:
        if media_gc_enabled():
            try:
                await asyncio.to_thread(sweep_once)
            except Exception:
                logger.warning("[nolgia_media_gc] sweep failed", exc_info=True)
        await asyncio.sleep(sweep_interval_seconds())
