"""Nolgia platform background results: file a finished background job in the customer's chat.

The Nolgia platform relay runs each chat turn as one ``POST /v1/runs`` and shows the customer
that run's reply, nothing else. Work the agent sent to the background — an async delegation, or
a terminal process started with ``notify`` — finishes after the run has settled, and on an
``api_server`` session this gateway's own handling of the result never reaches the customer: a
delegation becomes a delivery row in this pod's session store (no client reads it), and a
process completion wakes a self-posted turn whose reply goes nowhere. The agent had meanwhile
told the customer "I'll send you the results as soon as it finishes" (prod, 2026-10-14).

This module hands such a result to the platform instead: ``POST /v1/agent/background-results``
(nolgia-api) files it in the customer's chat session and queues a normal agent turn that
reports it, so the reply, its files and the "Ready" alert reach the customer like any other
turn. Only final results are filed (a delegation's completion, a process's exit); interim
diagnostics and watch-pattern matches keep this gateway's own handling, since each filing is
a billed turn.

Credentials, in order:

1. The originating run's credential (``origin_nolgia_token``, a delegation completion only).
   The platform takes the chat session from the credential itself.
2. The pod-wide ``NOLGIA_TOKEN`` with the chat session named explicitly: the run journal keeps
   each relay run's ``workspace_id``, which the platform sets to its chat session id (NOL-414).

Outcomes: ``FILED`` (the platform has it; the caller acknowledges the event), ``RETRY`` (a
transient failure; the caller keeps the event and tries again, with backoff here), and
``NOT_FILED`` (not a platform result, the route is off for this account, or the platform
refused it; the caller falls back to its own handling, exactly as before this module). A
result that keeps failing transiently falls back after ``_GIVE_UP_AFTER_SECONDS``.

Enabled when the pod carries platform credentials (``NOLGIA_API_URL`` + ``NOLGIA_TOKEN``) and
``NOLGIA_BACKGROUND_RESULTS`` is not ``"0"`` (the escape hatch). The platform gates the route
per account (``AGENT_BACKGROUND_RESULTS``) and answers 404 while it is off, which reads as
``NOT_FILED``. Stdlib-only, like ``nolgia_assets``.
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

FILED = "filed"
RETRY = "retry"
NOT_FILED = "not_filed"

_ROUTE = "/agent/background-results"
_TIMEOUT_SECONDS = 20.0

# The platform's request limits (AgentBackgroundResultRequest in nolgia-api's openapi.yaml).
_MAX_TEXT_CHARS = 20000
_MAX_SOURCE_CHARS = 200

# Transient failures (network, 5xx, a full turn queue) are retried with exponential backoff,
# then given up on so the result still gets this gateway's own handling.
_BACKOFF_START_SECONDS = 4.0
_BACKOFF_MAX_SECONDS = 120.0
_GIVE_UP_AFTER_SECONDS = 30 * 60.0

# source key -> (first failure, next attempt not before, current delay), monotonic seconds.
_failures: Dict[str, Tuple[float, float, float]] = {}
_failures_lock = threading.Lock()


def enabled() -> bool:
    """True on a platform pod (credentials present) unless the escape hatch is set."""
    if os.environ.get("NOLGIA_BACKGROUND_RESULTS", "").strip() == "0":
        return False
    return bool(os.environ.get("NOLGIA_API_URL", "").strip()) and bool(
        os.environ.get("NOLGIA_TOKEN", "").strip()
    )


def relay_async_delivery(workspace_id: str) -> bool:
    """Whether a relay run may bind async delivery: on a platform pod (``enabled``) and for a run
    that names its platform chat session (the relay's ``workspace_id``, a UUID, NOL-414). The
    caller has already checked that the platform said it takes this run's background results
    (the run body's ``background_results``), so a process that exits after the turn reaches the
    customer through ``file_result`` rather than a self-posted wake nobody reads."""
    if not enabled():
        return False
    try:
        uuid.UUID(str(workspace_id or "").strip())
    except ValueError:
        return False
    return True


def completion_only_session() -> bool:
    """True on a relay run that has async delivery only for the platform hand-off: the API
    server binds async delivery for nothing else (``_bind_api_server_session``). Such a session
    takes a process's exit and nothing more: watch-pattern matches, heartbeats and detached cron
    runs would each become a billed platform turn, or a silent wake when the platform declines
    them, so they stay off exactly as on any other stateless HTTP session."""
    try:
        from gateway.session_context import async_delivery_supported, get_session_env
    except Exception:
        return False
    return get_session_env("HERMES_SESSION_PLATFORM", "") == "api_server" and async_delivery_supported()


COMPLETION_NOTE = (
    "When this process exits, NOLGIA files its result in this chat and runs a new turn in which you "
    "report it to the creator, so you may end this reply now and say the result will follow."
)


def _api_base() -> str:
    """Platform API base with the ``/v1`` suffix (as ``nolgia_assets`` builds it)."""
    url = os.environ.get("NOLGIA_API_URL", "").strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def result_kind(evt: Dict[str, Any]) -> Optional[str]:
    """``delegation`` or ``process`` for a final background result, else ``None``."""
    evt_type = evt.get("type")
    if evt_type == "async_delegation" and not evt.get("task_failure_notice"):
        return "delegation"
    if evt_type == "completion":
        return "process"
    return None


def source_id(evt: Dict[str, Any], kind: str) -> str:
    """The producer's own id for this result, stable across redeliveries of the same event.

    A process id is reused across incarnations, so its spawn time is part of the id (as in
    ``_completion_delivery_identity``)."""
    if kind == "delegation":
        source = str(evt.get("delegation_id") or "").strip()
    else:
        source = str(evt.get("session_id") or "").strip()
        started_at = evt.get("started_at")
        if source and started_at is not None:
            source = f"{source}@{started_at}"
    return source[:_MAX_SOURCE_CHARS]


def bounded_text(text: str) -> str:
    """Fit the platform's text limit, keeping the start and the end of a long result."""
    text = text.strip()
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    marker = "\n\n[... {} characters cut ...]\n\n"
    keep = (_MAX_TEXT_CHARS - len(marker.format(len(text)))) // 2
    return text[:keep] + marker.format(len(text) - 2 * keep) + text[-keep:]


def workspace_session(adapter: Any, raw_sid: str) -> str:
    """The platform chat session of the relay run behind *raw_sid*, or ``""``.

    The relay names its chat session id as the run's ``workspace_id`` (NOL-414) and the run
    journal keeps it for the journal's retention; only a UUID counts as a platform session."""
    store = getattr(adapter, "_response_store", None)
    lookup = getattr(store, "journal_workspace_id", None)
    if lookup is None or not raw_sid:
        return ""
    try:
        workspace = str(lookup(raw_sid) or "").strip()
        return str(uuid.UUID(workspace)) if workspace else ""
    except Exception:
        return ""


def _post(token: str, body: Dict[str, Any]) -> int:
    """POST the result; the HTTP status. Raises on a transport failure."""
    request = urllib.request.Request(
        _api_base() + _ROUTE,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as err:
        return int(err.code)


def _backing_off(key: str, now: float) -> bool:
    with _failures_lock:
        entry = _failures.get(key)
    return entry is not None and now < entry[1]


def _record_transient(key: str, now: float) -> bool:
    """Record one transient failure; False once the result has failed for too long."""
    with _failures_lock:
        first, _next_at, delay = _failures.get(key, (now, now, 0.0))
        if now - first >= _GIVE_UP_AFTER_SECONDS:
            _failures.pop(key, None)
            return False
        delay = min(max(_BACKOFF_START_SECONDS, delay * 2), _BACKOFF_MAX_SECONDS)
        _failures[key] = (first, now + delay, delay)
        return True


def _forget(key: str) -> None:
    with _failures_lock:
        _failures.pop(key, None)


def file_result(adapter: Any, text: str, raw_sid: str, evt: Dict[str, Any]) -> str:
    """File one background result with the platform: ``FILED``, ``RETRY`` or ``NOT_FILED``.

    Blocking (HTTP and the journal read): call it through ``asyncio.to_thread``."""
    kind = result_kind(evt)
    if kind is None or not text.strip():
        return NOT_FILED
    source = source_id(evt, kind)
    if not source:
        return NOT_FILED
    key = f"{raw_sid}|{kind}:{source}"
    now = time.monotonic()
    if _backing_off(key, now):
        return RETRY

    session = workspace_session(adapter, raw_sid)
    run_token = str(evt.get("origin_nolgia_token") or "").strip()
    pod_token = os.environ.get("NOLGIA_TOKEN", "").strip()
    attempts = []
    if run_token:
        attempts.append(("run", run_token))
    if session and pod_token:
        attempts.append(("pod", pod_token))
    if not attempts:
        return NOT_FILED

    body: Dict[str, Any] = {"kind": kind, "text": bounded_text(text), "source_id": source}
    if session:
        body["session_id"] = session
    for credential, token in attempts:
        try:
            status = _post(token, body)
        except Exception as err:
            logger.warning("Nolgia background result %s for session %s: transport failure: %s", source, raw_sid, err)
            status = 0
        if 200 <= status < 300:
            _forget(key)
            logger.info(
                "Nolgia background result %s (%s) filed in platform session %s (%s credential)",
                source, kind, session or "<from credential>", credential)
            return FILED
        if status in (401, 403) and credential == "run":
            # The originating run's credential has expired or names another session: the pod
            # credential with the journaled session is the second chance.
            continue
        if status == 0 or status == 429 or status >= 500:
            if _record_transient(key, now):
                return RETRY
            logger.warning(
                "Nolgia background result %s for session %s: giving up after %.0f minutes of failures; "
                "using this gateway's own delivery", source, raw_sid, _GIVE_UP_AFTER_SECONDS / 60)
            return NOT_FILED
        # 404 (route off for this account, or an unknown session), 400, or a refused credential.
        logger.info(
            "Nolgia background result %s for session %s not filed (HTTP %s); using this gateway's own delivery",
            source, raw_sid, status)
        _forget(key)
        return NOT_FILED
    _forget(key)
    return NOT_FILED
