"""Session-scoped context variables for the Hermes gateway.

Replaces the old ``os.environ``-based ``HERMES_SESSION_*`` state with task-local ``ContextVar``s
(inherited by ``run_in_executor`` threads), so concurrently handled messages no longer clobber each
other's routing ids.  ``get_session_env`` is a drop-in for ``os.getenv``.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

# "Never set here" (falls back to os.environ for CLI/cron) vs "" = explicitly cleared (no fallback).
_UNSET: Any = object()

# Process-level latch: has set_session_vars() ever bound a session?  When engaged, the subprocess
# env bridge treats ContextVars as authoritative and an _UNSET var as "no session in THIS task".
_session_context_engaged: bool = False


def session_context_engaged() -> bool:
    """True if any session has been bound via set_session_vars in this process."""
    return _session_context_engaged


# --- Per-task session variables: bound by set_session_vars / cleared to "" by clear_session_vars;
# tuple ORDER is the positional order of ``values`` in set_session_vars (zipped).
# * SCOPE_ID: platform-neutral scope (guild / workspace / Matrix server) so async producers can
#   persist a completion's full routing origin (relay egress guards need it).
# * UI_SESSION_ID: in-process UI tab id, separate from the durable SESSION_ID, so a stale/rotated
#   durable key is not consumed by the wrong poller.
# * MESSAGE_ID: reply anchor keeping notifications inside the originating Telegram topic.
# * CRON_SESSION: tri-state — _UNSET = legacy env fallback; "1" = cron; "" = non-cron, masks env.
# * NOLGIA_TOKEN: run-scoped Nolgia platform token. On the Nolgia platform relay each submitted
#   run may carry its own short-lived credential so the platform can attribute every outbound
#   call (asset uploads, generation) to the exact run that caused it — the pod-wide NOLGIA_TOKEN
#   cannot distinguish concurrent runs. Read by gateway.platforms.nolgia_assets (session var
#   first, pod env fallback) and bridged into tool subprocesses as NOLGIA_TOKEN by the
#   session-context env bridge. The HERMES_SESSION_ prefix is load-bearing: task-scoped (no
#   sibling-run leak: _UNSET strips) and excluded from the shared shell snapshot.
# * SCRATCH_DIR: session-scoped scratch workspace (NOL-414), bound by the API server to the
#   per-session directory (gateway.session_workspace, ``<base>/session-<id8>/``) that is this
#   turn's default write surface. The subprocess env bridge (tools/environments/local.py)
#   redirects the child's TMPDIR to it so concurrent runs from different sessions stop sharing
#   one scratch dir. Same HERMES_SESSION_ prefix contract as NOLGIA_TOKEN.
_SESSION_VARS = (
    _SESSION_PLATFORM, _SESSION_SOURCE, _SESSION_CHAT_ID, _SESSION_CHAT_TYPE,
    _SESSION_CHAT_NAME, _SESSION_THREAD_ID, _SESSION_USER_ID, _SESSION_USER_ID_ALT,
    _SESSION_USER_NAME, _SESSION_SCOPE_ID, _SESSION_KEY, _SESSION_ID,
    _SESSION_UI_SESSION_ID, _SESSION_MESSAGE_ID, _SESSION_PROFILE,
    _BROWSER_CONTROL_PRINCIPAL, _BROWSER_CONTROL_TRANSPORT_FAMILY,
    _SESSION_NOLGIA_TOKEN, _SESSION_SCRATCH_DIR, _CRON_SESSION,
) = tuple(ContextVar(name, default=_UNSET) for name in (
    "HERMES_SESSION_PLATFORM", "HERMES_SESSION_SOURCE", "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_TYPE", "HERMES_SESSION_CHAT_NAME", "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_USER_ID", "HERMES_SESSION_USER_ID_ALT", "HERMES_SESSION_USER_NAME",
    "HERMES_SESSION_SCOPE_ID", "HERMES_SESSION_KEY", "HERMES_SESSION_ID",
    "HERMES_UI_SESSION_ID", "HERMES_SESSION_MESSAGE_ID", "HERMES_SESSION_PROFILE",
    "HERMES_BROWSER_CONTROL_PRINCIPAL", "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY",
    "HERMES_SESSION_NOLGIA_TOKEN", "HERMES_SESSION_SCRATCH_DIR",
    "HERMES_CRON_SESSION",
))

# Whether this channel can route an ASYNC completion back AFTER the turn ends (see
# ``async_delivery_supported()``).  _UNSET => supported (CLI, contextvar-unaware paths); stateless
# adapters (API server, Kanban workers) opt OUT via ``supports_async_delivery = False`` at bind.
_SESSION_ASYNC_DELIVERY = ContextVar("HERMES_SESSION_ASYNC_DELIVERY", default=_UNSET)

# Cron auto-delivery vars, set per-job in run_job() so concurrent jobs don't clobber.
_CRON_AUTO_DELIVER_PLATFORM = ContextVar("HERMES_CRON_AUTO_DELIVER_PLATFORM", default=_UNSET)
_CRON_AUTO_DELIVER_CHAT_ID = ContextVar("HERMES_CRON_AUTO_DELIVER_CHAT_ID", default=_UNSET)
_CRON_AUTO_DELIVER_THREAD_ID = ContextVar("HERMES_CRON_AUTO_DELIVER_THREAD_ID", default=_UNSET)

# Legacy env-var name -> ContextVar for get_session_env (_SESSION_ASYNC_DELIVERY deliberately
# absent: it is a bool capability, read via async_delivery_supported).
_VAR_MAP = {var.name: var for var in (
    *_SESSION_VARS, _CRON_AUTO_DELIVER_PLATFORM, _CRON_AUTO_DELIVER_CHAT_ID,
    _CRON_AUTO_DELIVER_THREAD_ID,
)}


def _runtime_cwd(func: str, *args: Any) -> None:
    """Best-effort call of ``agent.runtime_cwd.<func>``; import/runtime failures are ignored."""
    try:
        from agent import runtime_cwd
        getattr(runtime_cwd, func)(*args)
    except Exception:
        pass


def set_current_session_id(session_id: str) -> None:
    """Synchronize ``HERMES_SESSION_ID`` across ContextVar and ``os.environ`` (tools read it
    with an os.environ fallback).  Delegated subagent children (built in the parent process)
    get ONLY the task-local write, or they would clobber the parent's id."""
    _SESSION_ID.set(session_id)
    try:
        from agent.delegation_context import is_delegated_child_context
        if is_delegated_child_context():
            return
    except Exception:
        pass
    os.environ["HERMES_SESSION_ID"] = session_id


@contextmanager
def scoped_current_session_id(session_id: str | None = None) -> Iterator[None]:
    """Bind a task-local session id and restore the prior value on exit; never touches
    ``os.environ``.  ``session_id=None`` is a pure save/restore boundary."""
    previous = _SESSION_ID.get()
    if session_id is not None:
        _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.set(previous)


def set_session_vars(
    platform: str = "", source: str = "", chat_id: str = "", chat_type: str = "",
    chat_name: str = "", thread_id: str = "", user_id: str = "", user_id_alt: str = "",
    user_name: str = "", scope_id: str = "", session_key: str = "", session_id: str = "",
    message_id: str = "", profile: str = "", browser_control_principal: str = "",
    browser_control_transport_family: str = "", cwd: str = "", async_delivery: bool = True,
    ui_session_id: str = "", nolgia_token: str = "", scratch_dir: str = "",
    cron_session: Any = _UNSET,
) -> list:
    """Set all session context variables and return reset tokens.  Call
    ``clear_session_vars(tokens)`` in a ``finally``; not nestable, clearing resets every var
    to ``""`` rather than restoring prior values (tokens are accepted only for API compat).

    ``nolgia_token`` is the run-scoped Nolgia platform credential (``HERMES_SESSION_NOLGIA_TOKEN``);
    empty means "no run scoping" (pod-wide bearer). ``scratch_dir`` binds the session-scoped
    scratch workspace (``HERMES_SESSION_SCRATCH_DIR``, see gateway.session_workspace): the
    subprocess env bridge points child TMPDIRs at it so concurrent turns from different sessions
    don't share one scratch surface."""
    global _session_context_engaged
    _session_context_engaged = True
    values = (
        platform, source, chat_id, chat_type, chat_name, thread_id, user_id, user_id_alt,
        user_name, scope_id, session_key, session_id, ui_session_id, message_id, profile,
        browser_control_principal, browser_control_transport_family, nolgia_token, scratch_dir,
        cron_session,
    )
    tokens = [var.set(value) for var, value in zip(_SESSION_VARS, values)]
    tokens.append(_SESSION_ASYNC_DELIVERY.set(bool(async_delivery)))
    _runtime_cwd("set_session_cwd", cwd)
    return tokens


def clear_session_vars(tokens: list) -> None:
    """Mark session context variables as explicitly cleared (``""``, not ``_UNSET``), so
    ``get_session_env`` returns empty instead of stale ``os.environ`` values.  Async-delivery
    goes back to ``_UNSET``: a cleared context is default-supported, not opted-out."""
    for var in _SESSION_VARS:
        var.set("")
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    _runtime_cwd("clear_session_cwd")


def reset_session_vars() -> None:
    """Reset every session var to ``_UNSET`` ("never bound here") for THIS context.  Call at
    the top of a fresh task *before* it binds: ``create_task`` snapshots the context, so B's
    task inherits A's already-set vars and a subprocess spawned before B binds would read A's
    identity.  ``_SESSION_ASYNC_DELIVERY`` (outside ``_VAR_MAP``) is reset explicitly too."""
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    _runtime_cwd("clear_session_cwd")


def set_session_nolgia_token(token: str):
    """Bind the run-scoped Nolgia platform token in THIS context.

    For code paths that need the token OUTSIDE a full ``set_session_vars``
    binding — e.g. the API server's post-run egress hop, which runs on the
    request task after the executor thread's session binding was already
    cleared. Returns the ContextVar reset token; pass it to
    :func:`reset_session_nolgia_token` in a ``finally`` block.
    """
    return _SESSION_NOLGIA_TOKEN.set(token)


def reset_session_nolgia_token(reset_token) -> None:
    """Undo :func:`set_session_nolgia_token`; never raises."""
    try:
        _SESSION_NOLGIA_TOKEN.reset(reset_token)
    except Exception:
        pass


def get_session_env(name: str, default: str = "") -> str:
    """Read a session var by legacy ``HERMES_SESSION_*`` name; drop-in for os.getenv.  The
    ContextVar wins if ever set here (even to ``""``); else ``os.environ``; else *default*."""
    var = _VAR_MAP.get(name)
    if var is not None and (value := var.get()) is not _UNSET:
        return value
    return os.getenv(name, default)


# Surfaces that are not a human chat channel (gateway binds HERMES_SESSION_PLATFORM, CLI/TUI/
# desktop bind HERMES_SESSION_SOURCE, so both are consulted).  Default-deny: an unrecognized
# identity counts as messaging.  Mirrors LOCAL_SESSION_SOURCE_IDS in apps/desktop session-source.ts.
NON_MESSAGING_SESSION_SURFACES = frozenset({
    "", "api_server", "cli", "codex", "desktop", "gateway", "kanban", "local",
    "msgraph_webhook", "tool", "tui", "webhook",
})


def session_is_messaging_surface() -> bool:
    """Whether this turn is delivered over a human messaging channel (checks
    ``HERMES_PLATFORM``, then the session platform, then the session source)."""
    platform = os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM", "")
    idents = (platform, get_session_env("HERMES_SESSION_SOURCE", ""))
    idents = (str(v or "").strip().lower() for v in idents)
    return any(ident and ident not in NON_MESSAGING_SESSION_SURFACES for ident in idents)


def declare_stateless_channel() -> None:
    """Declare that this session cannot receive an async background completion.  Unlike
    ``set_session_vars(async_delivery=False)`` this does NOT latch ``_session_context_engaged``
    (flipping the subprocess env bridge), which a one-shot CLI must not do as a side effect.

    See NousResearch/hermes-agent#53027 and #63142.
    """
    _SESSION_ASYNC_DELIVERY.set(False)


def async_delivery_supported() -> bool:
    """Whether the current session can deliver a background completion later.  False for
    stateless channels (:func:`declare_stateless_channel`) and Kanban workers
    (``HERMES_KANBAN_TASK``: one-shot subprocesses whose parent disappears after the turn)."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    value = _SESSION_ASYNC_DELIVERY.get()
    return True if value is _UNSET else bool(value)
