"""Per-session scratch workspaces for concurrent API-server runs (NOL-414).

On a multi-session host every run used to execute against one shared
filesystem surface: the terminal's default cwd and the scratch convention
(``/opt/data/tmp`` on the Nolgia pod) were the same for every concurrent
turn, so two runs from different sessions could overwrite each other's
files or pick up each other's artifacts by fixed path (NOL-402/NOL-408).

This module derives a session-scoped scratch directory the gateway binds as
the run's default working/scratch surface:

    <base>/session-<id8>/

``<base>`` is ``$HERMES_SESSION_SCRATCH_BASE`` when set, else
``$HERMES_HOME/tmp`` (``/opt/data/tmp`` on the pod).  ``<id8>`` is the first
8 characters of the sanitized session identity — for a platform session UUID
this matches nolgia-api's relayed-attachment staging convention byte-for-byte
(``agentAttachmentStagingDir``: ``"/opt/data/tmp/session-" +
sessionID.String()[:8] + "/"``, nolgia-api#249), so the directory the
platform's fetch instruction names and the scratch dir the gateway enforces
are the SAME directory whenever both sides key off the same session identity.
Run-keyed sessions (``run_<hex>`` — the /v1/runs default when the caller
supplies no session id) drop the ``run_`` prefix first, yielding the same
8-hex-char shape.

Scope (deliberate): this isolates the DEFAULT scratch/output surface of a
turn — the terminal/file-tool cwd seed and the subprocess ``TMPDIR``.  It
does not fence absolute paths: genuinely shared resources (git checkouts,
config, the skills catalog) stay reachable exactly as before, and a turn
that explicitly writes an absolute path outside its workspace is not
blocked.  Whole-home isolation is out of scope.

Enablement: ``HERMES_SESSION_WORKSPACES`` — ``1/true/on`` forces on,
``0/false/off`` forces off.  Unset/``auto`` engages unless the operator has
pinned a REAL project workspace via ``TERMINAL_CWD``/``terminal.cwd``; a
pinned workspace is an explicit "all my turns work here" configuration that
must keep winning.  The gateway's startup bridge sets ``TERMINAL_CWD`` to
the HOME fallback whenever ``terminal.cwd`` is unset or a placeholder
(gateway/run.py + gateway/cwd_placeholder.py — on the pod that is
``/opt/data``), so "set to home" and placeholders count as NOT pinned:
that is the shared-surface default this module exists to fix, not an
operator decision.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

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

_FORCE_ON = frozenset({"1", "true", "on", "yes"})
_FORCE_OFF = frozenset({"0", "false", "off", "no"})


def _terminal_cwd_is_operator_pinned() -> bool:
    """True when TERMINAL_CWD names a real, deliberate project workspace.

    The gateway startup bridge always materializes TERMINAL_CWD: when
    ``terminal.cwd`` is unset or a placeholder it falls back to HOME
    (gateway/run.py → gateway/cwd_placeholder.py), so a bare "is it set"
    check would read the pod's shared-home default (``/opt/data``) as an
    operator decision and permanently disable session workspaces on the
    exact deployment they exist for. Home and placeholders therefore count
    as NOT pinned.
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


def session_workspaces_enabled() -> bool:
    """Whether the gateway should bind per-session scratch workspaces."""
    mode = os.environ.get("HERMES_SESSION_WORKSPACES", "").strip().lower()
    if mode in _FORCE_OFF:
        return False
    if mode in _FORCE_ON:
        return True
    # auto: engage unless an operator pinned a real project workspace the
    # session seed must not shadow.
    return not _terminal_cwd_is_operator_pinned()


def scratch_base_dir() -> Path:
    """Base directory session workspaces live under.

    ``$HERMES_SESSION_SCRATCH_BASE`` override first, else
    ``$HERMES_HOME/tmp`` — ``/opt/data/tmp`` on the Nolgia pod, i.e. the
    exact base nolgia-api#249's staging convention uses.
    """
    raw = os.environ.get("HERMES_SESSION_SCRATCH_BASE", "").strip()
    if raw:
        return Path(raw).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "tmp"


def workspace_dirname(workspace_id: Optional[str]) -> Optional[str]:
    """Return ``session-<id8>`` for *workspace_id*, or None when unusable.

    ``run_``-prefixed ids (the /v1/runs fallback session identity) drop the
    prefix so the derived name keeps the platform convention's 8-hex shape.
    Ids that sanitize away entirely (hostile or degenerate input) fall back
    to a content hash — still deterministic, never a traversal.
    """
    raw = str(workspace_id or "").strip()
    if not raw:
        return None
    candidate = raw[4:] if raw.startswith("run_") else raw
    cleaned = _ID_SANITIZE_RE.sub("", candidate)
    if cleaned.strip("_-"):
        id8 = cleaned[:_ID8_LEN]
    else:
        id8 = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:_ID8_LEN]
    return SESSION_DIR_PREFIX + id8


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


def ensure_session_workspace(workspace_id: Optional[str]) -> Optional[str]:
    """Create (if needed) and return the session workspace path.

    Never raises: a filesystem failure logs and returns None, leaving the
    turn on the shared-default behavior instead of failing the run.
    """
    path = resolve_session_workspace(workspace_id)
    if not path:
        return None
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError as exc:
        logger.warning("could not create session workspace %s: %s", path, exc)
        return None
    return path
