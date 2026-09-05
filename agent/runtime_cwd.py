"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory (`terminal.cwd`
is bridged to it once at gateway/cron startup; the local CLI leaves it unset and relies on
the launch dir). Reading it in one place keeps the system prompt, tool surfaces, and
context-file discovery agreeing on where the agent lives. Multi-session gateways can pin a
logical cwd via `_SESSION_CWD`.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)

# The package/source root (<root>/agent/runtime_cwd.py). A backend launched from or
# self-spawned into this tree (desktop default) must never let an os.getcwd() fallback
# inject this repo's contributor AGENTS.md as project context.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    """True only when ``p`` IS the package root or sits inside it — ancestors
    (a home dir containing the checkout) are legitimate workspaces."""
    try:
        p = p.resolve()
    except Exception:
        return False
    return p == _PACKAGE_ROOT or _PACKAGE_ROOT in p.parents


def set_session_cwd(cwd: str | None) -> Token:
    """Pin the logical cwd for the current context."""
    return _SESSION_CWD.set((cwd or "").strip())


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")


def scope_terminal_cwd() -> str:
    """Scope-aware TERMINAL_CWD value (may be empty) — every cwd consumer reads through this.

    Under gateway multiplexing the per-turn terminal scope carries the active profile's cwd;
    the process-global env var may hold another profile's. Only an ImportError falls back: an
    active refusal scope must raise, not silently resolve the launch profile's cwd.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def _existing_dir(raw: str, label: str) -> Path | None:
    p = Path(raw).expanduser()
    if p.is_dir():
        return p
    logger.warning("%s does not exist: %s", label, raw)
    return None


def _resolve_configured_cwd(*, override_is_final: bool, ignore_scratch_override: bool = False) -> Path | None:
    """Session override, then TERMINAL_CWD; each validated as a real directory.

    ``override_is_final``: a set-but-missing session override yields None
    instead of falling through to TERMINAL_CWD.

    ``ignore_scratch_override`` (NOL-414): a session override equal to the turn's bound
    scratch workspace (``_session_scratch_dir``) is treated as unset, so resolution falls
    through to TERMINAL_CWD exactly as it did before the bind.
    """
    override = _SESSION_CWD.get()
    override = "" if override is _UNSET else str(override).strip()
    if override and ignore_scratch_override and override == _session_scratch_dir():
        override = ""
    if override:
        p = _existing_dir(override, "configured working directory")
        if p is not None or override_is_final:
            return p
    raw = scope_terminal_cwd().strip()
    return _existing_dir(raw, "TERMINAL_CWD") if raw else None


def _session_scratch_dir() -> str:
    """The turn's bound session scratch workspace, if any (NOL-414).

    Read from the ``HERMES_SESSION_SCRATCH_DIR`` session contextvar the API
    server binds alongside ``cwd`` (gateway/session_context.py). Used by
    ``resolve_context_cwd`` to tell "the session cwd IS the per-session
    scratch workspace" apart from a real configured workspace: the former
    must not re-anchor context-file discovery (an empty scratch dir carries
    no AGENTS.md/.hermes.md, and following it would silently drop the
    home-workspace doctrine from the system prompt).
    """
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env("HERMES_SESSION_SCRATCH_DIR", "") or "").strip()
    except Exception:
        return ""


def resolve_agent_cwd() -> Path:
    """Configured cwd, else the launch dir (os.getcwd()'s OSError on a deleted cwd deliberately propagates)."""
    return _resolve_configured_cwd(override_is_final=False) or Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    """Configured cwd for context-file discovery, or None (build_context_files_prompt then falls back to the
    launch dir). An existing configured path is honored verbatim — including the Hermes source tree, a
    legitimate workspace when developing Hermes; fallback-directory policy lives in the caller.

    Exception (NOL-414): a session cwd equal to the turn's bound scratch workspace is a
    per-session ISOLATION surface, not a configured project workspace. The agent works there
    (resolve_agent_cwd, the prompt's "Current working directory" line, tool cwd records all
    agree), but doctrine discovery must keep anchoring where it did before the bind — the
    TERMINAL_CWD/home fallback — or every API-server turn on a session-workspace deployment
    silently loses its workspace AGENTS.md."""
    return _resolve_configured_cwd(override_is_final=True, ignore_scratch_override=True)
