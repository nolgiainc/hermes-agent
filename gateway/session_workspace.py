"""Per-session scratch workspaces for concurrent API-server runs (NOL-414).

On a multi-session host every run used to execute against one shared
filesystem surface: the terminal's default cwd and the scratch convention
(``/opt/data/tmp`` on the Nolgia pod) were the same for every concurrent
turn, so two runs from different sessions could overwrite each other's
files or pick up each other's artifacts by fixed path (NOL-402/NOL-408).

This module derives a session-scoped scratch directory the gateway binds as
the run's default working/scratch surface:

    <base>/session-<id8>/

``<base>`` is ``gateway.session_workspaces.scratch_base`` from
``config.yaml`` when set, else ``$HERMES_HOME/tmp`` (``/opt/data/tmp`` on
the pod).  ``<id8>`` is the first 8 characters of the sanitized session
identity — for a platform session UUID this matches nolgia-api's
relayed-attachment staging convention byte-for-byte
(``agentAttachmentStagingDir``: ``"/opt/data/tmp/session-" +
sessionID.String()[:8] + "/"``, nolgia-api#249), so the directory the
platform's fetch instruction names and the scratch dir the gateway enforces
are the SAME directory whenever both sides key off the same session identity.
Run-keyed sessions (``run_<hex>`` — the /v1/runs default when the caller
supplies no session id) drop the ``run_`` prefix first, yielding the same
8-hex-char shape.

Names stay collision-resistant outside that convention: ``id8`` alone is
only used for hex-shaped identities (the platform-UUID / run-hex domain the
staging convention covers).  Any other identity carries a short digest of
its FULL id (``session-<id8>-<h6>``), so ``aaaaaaaa-one`` and
``aaaaaaaa-two`` get different directories instead of silently sharing one.
Provisioning additionally claims a directory with an owner marker: if a
different identity already owns the convention-aligned name, the loser falls
back to its digest-suffixed directory rather than sharing a scratch surface.

Scope (deliberate): this isolates the DEFAULT scratch/output surface of a
turn — the terminal/file-tool cwd seed and the subprocess ``TMPDIR``.  It
does not fence absolute paths: genuinely shared resources (git checkouts,
config, the skills catalog) stay reachable exactly as before, and a turn
that explicitly writes an absolute path outside its workspace is not
blocked.  Whole-home isolation is out of scope.

Configuration (``config.yaml``, ``gateway.session_workspaces``):

- ``mode`` — ``auto`` (default), ``on``, or ``off``.  ``auto`` engages
  unless the operator has pinned a REAL project workspace via
  ``terminal.cwd``; a pinned workspace is an explicit "all my turns work
  here" configuration that must keep winning.  The gateway's startup bridge
  sets ``TERMINAL_CWD`` to the HOME fallback whenever ``terminal.cwd`` is
  unset or a placeholder (gateway/run.py + gateway/cwd_placeholder.py — on
  the pod that is ``/opt/data``), so "set to home" and placeholders count as
  NOT pinned: that is the shared-surface default this module exists to fix,
  not an operator decision.
- ``scratch_base`` — base directory for the per-session dirs.
- ``retention_hours`` — bounded retention for abandoned workspaces
  (stateless Responses calls and session-less ``/v1/runs`` each mint a
  fresh one).  A throttled sweep removes ``session-*`` dirs whose newest
  content is older than this; ``0`` disables the sweep.

``HERMES_SESSION_WORKSPACES`` / ``HERMES_SESSION_SCRATCH_BASE`` /
``HERMES_SESSION_WORKSPACE_RETENTION_HOURS`` remain as INTERNAL env bridges
(the gateway runs agent turns in child processes that read env, and tests
pin them); config.yaml is the user/operator-facing surface.

Execution-backend scope: workspaces are provisioned on the gateway HOST and
bridged into child processes by ``LocalEnvironment``.  A remote/containerized
terminal backend (docker, ssh, modal, daytona, singularity, vercel_sandbox)
executes commands somewhere that host path does not exist, so a host-only
workspace would just make every command ``cd`` into a missing directory.
Session workspaces therefore engage only for the ``local`` backend,
including under ``mode: on``.
"""

import hashlib
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Matches nolgia-api's agentAttachmentStagingDir prefix ("session-").
SESSION_DIR_PREFIX = "session-"

# Chars allowed into the directory name. Session/workspace ids can arrive
# from request bodies, so the derivation must be traversal-safe: anything
# outside this set (path separators, dots, "~", control chars) is dropped
# BEFORE truncation.
_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")

# First-8 truncation matches nolgia-api's sessionID.String()[:8] — a UUID's
# first 8 chars are pure hex (no hyphen), so sanitize-then-truncate yields
# the identical id8 for a platform session UUID.
_ID8_LEN = 8

# Identities the staging convention actually covers: a platform session UUID
# or a run hex id (``run_`` already stripped). Only these may use the bare
# ``session-<id8>`` name; everything else gets a digest suffix so two ids
# sharing a first-8 prefix can never share a scratch dir.
_CONVENTION_ID_RE = re.compile(r"^[0-9a-fA-F]{8}[0-9a-fA-F-]*$")

# Length of the full-identity digest appended to non-convention names (and
# used as the collision escape hatch for convention names).
_ID_DIGEST_LEN = 6

# Written inside a workspace to record which identity owns it. Lets a second
# identity that derives the same convention-aligned name detect the clash and
# fall back to its own digest-suffixed directory.
_OWNER_MARKER = ".hermes-session-owner"

_FORCE_ON = frozenset({"1", "true", "on", "yes"})
_FORCE_OFF = frozenset({"0", "false", "off", "no"})

# Backends whose commands run on the gateway host filesystem.
_LOCAL_BACKENDS = frozenset({"", "local"})

_DEFAULT_RETENTION_HOURS = 48.0

# At most one retention sweep per scratch base per process per interval:
# provisioning happens on every turn and the sweep stats a directory tree.
_PRUNE_INTERVAL_SECONDS = 900.0

# Serializes provisioning's create/claim/refresh against the sweep's final
# age check + delete. Without one shared lock a sweeper could pass the age
# check, _provision could then hand the same (freshly refreshed) directory to
# a resuming turn, and the sweeper would still rmtree it, leaving that live
# run with a nonexistent cwd/TMPDIR and its workspace contents gone.
# Reentrant: ensure_session_workspace provisions and then sweeps on one thread.
_workspace_lock = threading.RLock()

# Last sweep time keyed by RESOLVED BASE. A multiplexed gateway serves several
# profiles and each normally resolves its own $HERMES_HOME/tmp; one global
# timestamp let the busiest profile win every 15-minute window and starve the
# others' bases, so their "bounded" scratch data would grow without limit.
_last_prune_at: dict = {}

_backend_warning_emitted = False


def _workspace_config() -> dict:
    """``gateway.session_workspaces`` from config.yaml (never raises)."""
    try:
        from hermes_cli.config import load_config_readonly

        gateway_cfg = load_config_readonly().get("gateway")
        if not isinstance(gateway_cfg, dict):
            return {}
        section = gateway_cfg.get("session_workspaces")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.debug("session workspace config read failed", exc_info=True)
        return {}


def _config_str(key: str, env_var: str) -> str:
    """Config value for *key*, with *env_var* as the internal bridge override.

    The env var exists because agent turns and tools run in child processes
    that only see env (and because tests pin it); config.yaml is the
    user-facing setting.
    """
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return raw
    value = _workspace_config().get(key)
    return "" if value is None else str(value).strip()


def _terminal_cwd_is_operator_pinned() -> bool:
    """True when TERMINAL_CWD names a real, deliberate project workspace.

    The gateway startup bridge always materializes TERMINAL_CWD from
    ``terminal.cwd``: when that setting is unset or a placeholder it falls
    back to HOME (gateway/run.py → gateway/cwd_placeholder.py), so a bare
    "is it set" check would read the pod's shared-home default
    (``/opt/data``) as an operator decision and permanently disable session
    workspaces on the exact deployment they exist for. Home and placeholders
    therefore count as NOT pinned.
    """
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if not raw:
        return False
    try:
        from gateway.cwd_placeholder import CWD_PLACEHOLDERS

        if raw in CWD_PLACEHOLDERS:
            return False
    except Exception:
        pass
    try:
        if Path(raw).expanduser().resolve() == Path.home().resolve():
            return False
    except OSError:
        pass
    return True


def _execution_backend() -> str:
    """Effective terminal backend (``TERMINAL_ENV`` bridge, else config)."""
    raw = os.environ.get("TERMINAL_ENV", "").strip().lower()
    if raw:
        return raw
    try:
        from hermes_cli.config import load_config_readonly

        terminal_cfg = load_config_readonly().get("terminal")
        if isinstance(terminal_cfg, dict):
            backend = str(terminal_cfg.get("backend") or "").strip().lower()
            if backend:
                return backend
    except Exception:
        logger.debug("terminal backend read failed", exc_info=True)
    return "local"


def _execution_is_local() -> bool:
    """Whether commands execute on the gateway host's filesystem.

    Docker/SSH/Modal/Daytona/Singularity/Vercel backends run commands in an
    environment where a gateway-host workspace path does not exist and where
    the ``TMPDIR`` bridge (LocalEnvironment) does not apply, so binding one
    would only break the turn (``builtin cd`` → exit 126).
    """
    return _execution_backend() in _LOCAL_BACKENDS


def session_workspaces_enabled() -> bool:
    """Whether the gateway should bind per-session scratch workspaces."""
    global _backend_warning_emitted

    mode = _config_str("mode", "HERMES_SESSION_WORKSPACES").lower()
    if mode in _FORCE_OFF:
        return False
    forced_on = mode in _FORCE_ON
    if not _execution_is_local():
        if forced_on and not _backend_warning_emitted:
            _backend_warning_emitted = True
            logger.warning(
                "session workspaces are enabled in config but the terminal "
                "backend is %r — a gateway-host scratch dir is not visible "
                "to that backend, so per-session workspaces stay off",
                _execution_backend(),
            )
        return False
    if forced_on:
        return True
    # auto: engage unless an operator pinned a real project workspace the
    # session seed must not shadow.
    return not _terminal_cwd_is_operator_pinned()


def scratch_base_dir() -> Path:
    """Base directory session workspaces live under.

    ``gateway.session_workspaces.scratch_base`` (or its
    ``HERMES_SESSION_SCRATCH_BASE`` bridge) first, else ``$HERMES_HOME/tmp``
    — ``/opt/data/tmp`` on the Nolgia pod, i.e. the exact base
    nolgia-api#249's staging convention uses.
    """
    raw = _config_str("scratch_base", "HERMES_SESSION_SCRATCH_BASE")
    if raw:
        return Path(raw).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "tmp"


def workspace_retention_hours() -> float:
    """Hours an idle session workspace survives before the sweep removes it."""
    raw = _config_str("retention_hours", "HERMES_SESSION_WORKSPACE_RETENTION_HOURS")
    if not raw:
        return _DEFAULT_RETENTION_HOURS
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_RETENTION_HOURS


def session_workspaces_active() -> bool:
    """Whether a workspace can ACTUALLY be provisioned right now.

    ``/v1/capabilities`` advertises the feature from this, not from the
    enablement flag alone: the platform raises per-user concurrency on that
    advertisement, so a deployment whose workspaces are off (mode ``off``,
    an operator-pinned cwd, a non-local backend) or whose scratch base is
    not writable must not claim isolation it does not have.
    """
    if not session_workspaces_enabled():
        return False
    try:
        base = scratch_base_dir()
        os.makedirs(base, mode=0o700, exist_ok=True)
        return os.access(base, os.W_OK | os.X_OK)
    except Exception:
        logger.debug("session workspace base is not provisionable", exc_info=True)
        return False


def _identity(workspace_id: Optional[str]) -> Optional[tuple]:
    """Return ``(raw_id, sanitized_candidate)`` or None when unusable."""
    raw = str(workspace_id or "").strip()
    if not raw:
        return None
    candidate = raw[4:] if raw.startswith("run_") else raw
    return raw, _ID_SANITIZE_RE.sub("", candidate)


def _id_digest(raw: str, length: int = _ID_DIGEST_LEN) -> str:
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:length]


def workspace_dirname(workspace_id: Optional[str]) -> Optional[str]:
    """Return the workspace directory name for *workspace_id*, or None.

    ``run_``-prefixed ids (the /v1/runs fallback session identity) drop the
    prefix so the derived name keeps the platform convention's 8-hex shape.
    Hex-shaped identities — platform session UUIDs and run ids, the domain
    nolgia-api#249's staging convention covers — get the bare
    ``session-<id8>`` name so both sides address the same directory.  Every
    other identity gets ``session-<id8>-<digest of the FULL id>``, so ids
    that merely share a prefix (``aaaaaaaa-one`` / ``aaaaaaaa-two``) never
    collapse onto one scratch dir.  Ids that sanitize away entirely (hostile
    or degenerate input) are named purely from the digest — still
    deterministic, never a traversal.
    """
    parts = _identity(workspace_id)
    if not parts:
        return None
    raw, cleaned = parts
    if not cleaned.strip("_-"):
        return SESSION_DIR_PREFIX + _id_digest(raw, _ID8_LEN)
    id8 = cleaned[:_ID8_LEN]
    if len(cleaned) >= _ID8_LEN and _CONVENTION_ID_RE.match(cleaned):
        return SESSION_DIR_PREFIX + id8
    return f"{SESSION_DIR_PREFIX}{id8}-{_id_digest(raw)}"


def _distinct_dirname(workspace_id: Optional[str]) -> Optional[str]:
    """Digest-suffixed name used when the primary name is owned elsewhere."""
    parts = _identity(workspace_id)
    if not parts:
        return None
    raw, cleaned = parts
    id8 = cleaned[:_ID8_LEN] if cleaned.strip("_-") else _id_digest(raw, _ID8_LEN)
    return f"{SESSION_DIR_PREFIX}{id8}-{_id_digest(raw)}"


def resolve_session_workspace(workspace_id: Optional[str]) -> Optional[str]:
    """Absolute session workspace path, or None when disabled/underivable."""
    if not session_workspaces_enabled():
        return None
    name = workspace_dirname(workspace_id)
    if not name:
        return None
    try:
        return str(scratch_base_dir() / name)
    except Exception:
        # get_hermes_home()/env resolution should never take a turn down.
        logger.debug("session workspace base resolution failed", exc_info=True)
        return None


def _claim_workspace(path: str, raw_id: str) -> bool:
    """Claim *path* for *raw_id*; False when another identity owns it.

    The marker is created with ``O_EXCL`` so two concurrent provisioners
    race for it atomically instead of both believing they own the directory.
    An unreadable/unwritable marker is treated as "ours" — losing isolation
    is bad, failing the turn over a marker file is worse.
    """
    marker = os.path.join(path, _OWNER_MARKER)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            with open(marker, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(4096).strip() == raw_id
        except OSError:
            logger.debug("session workspace owner marker unreadable: %s", marker)
            return True
    except OSError:
        logger.debug("session workspace owner marker unwritable: %s", marker)
        return True
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(raw_id)
    except OSError:
        logger.debug("session workspace owner marker write failed: %s", marker)
    return True


def _provision(base: Path, name: str, raw_id: str) -> Optional[str]:
    """Create + claim ``base/name``, returning its path or None.

    Held under ``_workspace_lock`` so the create/claim/refresh sequence cannot
    interleave with a concurrent sweep's age-check → delete: a workspace this
    call hands back is never one the sweeper is midway through reclaiming.
    """
    path = str(base / name)
    with _workspace_lock:
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
        except OSError as exc:
            logger.warning("could not create session workspace %s: %s", path, exc)
            return None
        if not _claim_workspace(path, raw_id):
            return None
        try:
            # Keep an active session's workspace young so the retention sweep
            # never reclaims a live turn's scratch dir just because the turn
            # wrote nothing this time.
            os.utime(path, None)
        except OSError:
            pass
    return path


def ensure_session_workspace(workspace_id: Optional[str]) -> Optional[str]:
    """Create (if needed) and return the session workspace path.

    Never raises: a filesystem failure logs and returns None, leaving the
    turn on the shared-default behavior instead of failing the run.
    """
    if not session_workspaces_enabled():
        return None
    parts = _identity(workspace_id)
    if not parts:
        return None
    raw_id, _ = parts
    try:
        base = scratch_base_dir()
    except Exception:
        logger.debug("session workspace base resolution failed", exc_info=True)
        return None

    names = []
    for candidate in (workspace_dirname(workspace_id), _distinct_dirname(workspace_id)):
        if candidate and candidate not in names:
            names.append(candidate)
    for name in names:
        path = _provision(base, name, raw_id)
        if path:
            _maybe_prune(base, keep=set(names))
            return path
    logger.warning(
        "could not provision a session workspace under %s for this turn", base
    )
    return None


# ---------------------------------------------------------------------------
# Bounded retention
# ---------------------------------------------------------------------------


def _newest_mtime(path: str) -> float:
    """Newest mtime of *path* or its immediate children.

    Shallow on purpose: a directory's own mtime misses writes into existing
    subdirectories, and walking a whole tree on every sweep is not worth it
    for an age heuristic.
    """
    newest = 0.0
    try:
        newest = os.stat(path).st_mtime
    except OSError:
        return 0.0
    try:
        for entry in os.scandir(path):
            try:
                newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def prune_session_workspaces(
    base: Optional[Path] = None,
    retention_hours: Optional[float] = None,
    keep: Iterable[str] = (),
) -> int:
    """Remove abandoned ``session-*`` workspaces under *base*.

    Stateless Responses calls and ``/v1/runs`` submissions without a session
    each mint their own workspace, and ``$HERMES_HOME/tmp`` usually has no
    external tmp cleaner, so without this normal API traffic accumulates
    directories (and arbitrarily large tool output) until the disk fills.
    Age is measured from the newest content in the directory, and an active
    turn refreshes its own workspace on every bind, so a long-lived session
    is never swept out from under itself.

    Returns the number of directories removed. ``retention_hours <= 0``
    disables the sweep.
    """
    hours = workspace_retention_hours() if retention_hours is None else retention_hours
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return 0
    if hours <= 0:
        return 0
    try:
        root = Path(base) if base is not None else scratch_base_dir()
    except Exception:
        return 0

    protected = {name for name in keep if name}
    cutoff = time.time() - hours * 3600.0
    removed = 0
    try:
        entries = list(os.scandir(root))
    except OSError:
        return 0
    for entry in entries:
        if entry.name in protected or not entry.name.startswith(SESSION_DIR_PREFIX):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        # Cheap pre-filter: skip obviously live workspaces without taking the
        # lock (a busy base holds mostly-fresh dirs).
        if _newest_mtime(entry.path) > cutoff:
            continue
        with _workspace_lock:
            # Final age check under the SAME lock _provision holds, so a turn
            # that reclaimed and refreshed this workspace since the scan can
            # no longer have its live scratch dir deleted underneath it.
            if _newest_mtime(entry.path) > cutoff:
                continue
            try:
                shutil.rmtree(entry.path)
            except OSError as exc:
                logger.debug(
                    "could not prune session workspace %s: %s", entry.path, exc
                )
                continue
        removed += 1
        logger.info(
            "pruned abandoned session workspace %s (idle > %sh)", entry.path, hours
        )
    return removed


def _maybe_prune(base: Path, keep: Iterable[str] = ()) -> None:
    """Sweep *base* at most once per ``_PRUNE_INTERVAL_SECONDS``.

    Throttled PER BASE: each profile in a multiplexed gateway resolves its own
    scratch base, and they must not share one budget (the busiest profile would
    otherwise be the only one ever swept).
    """
    try:
        key = os.path.abspath(os.path.expanduser(str(base)))
    except Exception:
        key = str(base)
    now = time.monotonic()
    with _workspace_lock:
        last = _last_prune_at.get(key)
        if last is not None and now - last < _PRUNE_INTERVAL_SECONDS:
            return
        _last_prune_at[key] = now
    try:
        prune_session_workspaces(base=base, keep=keep)
    except Exception:
        logger.debug("session workspace retention sweep failed", exc_info=True)
