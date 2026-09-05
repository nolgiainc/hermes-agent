"""Wake an existing agent session from a background completion event. Push-capable adapters
(``supports_async_delivery``) get a synthetic ``MessageEvent(internal=True)`` via handle_message;
stateless adapters (API server) would run that under a ``build_session_key()`` key that never
matches the raw ``X-Hermes-Session-Id`` real turns use (invisible parallel session), so we self-POST
``/v1/chat/completions`` with the raw id header to resume the REAL session. Exception:
async-delegation completions: the CLIENT owns the next turn, so they are never self-POSTed as a
new ``role=user`` prompt (could cross a pending human-confirmation gate); instead
``persist_delegation_delivery`` writes a durable DELIVERY row (``display_kind=
"async_delegation_complete"``, read by TUI/desktop pollers). Failures RAISE (after bounded retries
on transient errors) so callers can rewind cursors / retry instead of silently losing the event."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Wire spelling of the default profile: ``/p/default/...`` and the unprefixed routes are the same
# runtime, so it needs no prefix (see gateway.platforms.api_server._DEFAULT_PROFILE_NAME).
_DEFAULT_PROFILE_NAME = "default"

# A wake self-post runs the whole agent turn synchronously (stream=false); generous ceiling so long
# tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff between retries on transient failures. The API server has no per-session lock (concurrent
# turns are last-writer-wins) but DOES enforce a global max_concurrent_runs cap via HTTP 429, which
# is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)


def _profile_scoped_api_key(profile: str) -> str:
    """Resolve *profile*'s own ``API_SERVER_KEY``, or ``""``.

    Mirrors ``ApiServerAdapter._expected_api_key``: a ``/p/<profile>/`` request
    is authenticated against the key in THAT profile's secret scope, not the
    listener owner's, so a self-post through the mirror must present it. Never
    logs the key or the underlying error text.
    """
    try:
        from agent.secret_scope import get_secret
        from gateway.run import _profile_runtime_scope
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.profiles import get_profile_dir

        with _profile_runtime_scope(get_profile_dir(profile)):
            key = get_secret("API_SERVER_KEY", "") or ""
        return key if has_usable_secret(key, min_length=16) else ""
    except Exception as exc:
        logger.warning(
            "Could not resolve a profile-scoped API_SERVER_KEY for %r: %s",
            profile,
            type(exc).__name__,
        )
        return ""


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends. Reads
    ``supports_async_delivery`` off the adapter class rather than the request-scoped contextvar
    (background watchers run outside any bound session context). Adapters that don't declare
    the flag are push-capable."""
    return bool(getattr(adapter, "supports_async_delivery", True))


async def deliver_wake(
    adapter: Any, *, text: str, session_id: str = "", source: Any = None,
    nolgia_token: str = "", profile: str = "",
) -> None:
    """Deliver a wake turn to the session behind ``adapter``. ``session_id`` is the RAW session id
    (``X-Hermes-Session-Id`` / state.db key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` for the synthetic event — required for push-capable adapters. Raises on
    failure so the caller can rewind/retry.

    ``profile`` is the multiplex profile of the session being woken (the profile that served the
    originating turn). A multiplexed session lives in ITS profile's runtime — own HERMES_HOME/state,
    own secret scope, own retained-credential scope — so the self-post must re-enter through that
    profile's ``/p/<profile>/...`` route; the unprefixed route would run the continuation under the
    DEFAULT profile, where the session's history and its retained turn credential do not exist
    (NOL-413). ``""`` (or ``"default"``) keeps the unprefixed route. Push-capable adapters ignore
    it: their wake re-enters through the normal message pipeline, already bound to the right profile.

    ``nolgia_token`` is the run-scoped Nolgia credential of the ORIGINATING turn (the run whose
    detached work this wake reports). Carried on the self-post so the continuation's CLI work
    attributes to that run rather than to whichever turn the session ran most recently —
    same-session concurrent runs are supported and a detached delegation has no wall-clock bound,
    so the two can differ (NOL-413). Empty means "no run scoping"; the API server then falls back
    to the session's retained token. Push-capable adapters ignore it."""
    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError("deliver_wake: push-capable adapter requires a SessionSource")
        from gateway.platforms.base import MessageEvent, MessageType
        synth_event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=True)
        await adapter.handle_message(synth_event)
        return
    if not session_id:
        raise ValueError("deliver_wake: non-push adapter (supports_async_delivery=False) "
                         "requires the raw session id to self-post the wake turn")
    # Forward the NOL-413 attribution/profile fields only when set: the historical two-argument
    # self-post shape stays byte-for-byte the same for wakes that carry neither.
    extra = {k: v for k, v in (("nolgia_token", nolgia_token), ("profile", profile)) if v}
    await _self_post_chat_completion(adapter, text=text, session_id=session_id, **extra)


def _delegation_display_metadata(evt: dict) -> dict:
    """Display-only metadata for a persisted delegation delivery row. Mirrors
    ``tui_gateway.server._async_delegation_display_metadata`` (same ``display_kind`` consumer
    contract) without importing the TUI stack."""
    raw_results = evt.get("results")
    results = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(1 for r in results if r.get("status") in {"completed", "success"})
    failed_count = sum(1 for r in results if r.get("status") in {"failed", "error"})
    metadata = {"delegation_id": str(evt.get("delegation_id") or ""), "task_count": task_count,
                "completed_count": completed_count or task_count - failed_count,
                "failed_count": failed_count}
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


async def persist_delegation_delivery(
    adapter: Any, *, text: str, session_id: str, evt: Optional[dict] = None, profile: str = "",
) -> None:
    """Persist an async-delegation completion as a durable DELIVERY row (see module docstring)
    WITHOUT running any agent turn. Raises on failure so the caller can release the durable claim
    and retry.

    85957: on stateless api_server sessions the client owns the turn after ``event.complete`` — a completion
    must never become a new ``role=user`` prompt via the self-post (that starts an unauthorized agent turn
    and can cross a pending human-confirmation gate). Instead, append the completion to the session
    transcript as a timeline bookkeeping row (``display_kind="async_delegation_complete"`` + display
    metadata — the exact shape the TUI/desktop delivery path persists), WITHOUT running any agent turn.
    Clients polling ``GET /api/sessions/{id}/messages`` see it immediately; the pre-request repair belt
    folds it into the next real client turn as context. See #85957.
    """
    if not session_id:
        raise ValueError("persist_delegation_delivery: raw session id required to persist "
                         "the completion on the api_server session transcript")
    ensure = getattr(adapter, "_ensure_session_db", None)
    target_profile = str(profile or "").strip()
    if target_profile:
        from hermes_cli.profiles import get_profile_dir, profile_exists

        if not profile_exists(target_profile):
            raise RuntimeError(
                f"persist_delegation_delivery: profile {target_profile!r} is unavailable"
            )
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home_token = set_hermes_home_override(str(get_profile_dir(target_profile)))
        try:
            db: Any = await asyncio.to_thread(ensure) if callable(ensure) else None
        finally:
            reset_hermes_home_override(home_token)
    else:
        db = await asyncio.to_thread(ensure) if callable(ensure) else None
    if db is None:
        raise RuntimeError("persist_delegation_delivery: api_server SessionDB unavailable — "
                           f"cannot persist completion for session {session_id}")
    await asyncio.to_thread(
        db.append_message, session_id, "user", content=text,
        display_kind="async_delegation_complete", display_metadata=_delegation_display_metadata(evt or {}),
    )
    logger.info(
        "async delegation completion persisted as delivery row for api_server session %s (no wake turn)", session_id
    )


async def _self_post_chat_completion(
    adapter: Any, *, text: str, session_id: str, nolgia_token: str = "", profile: str = "",
) -> None:
    """POST the wake text to the in-pod API server as a normal session turn, using the adapter's
    own bind host/port/key. Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY``, so a missing key is a hard error rather than a wake in a fresh session
    nobody watches.

    A named ``profile`` posts through that profile's ``/p/<profile>/`` mirror (see ``deliver_wake``),
    authenticated with the profile's OWN ``API_SERVER_KEY`` — the multiplex auth gate resolves the
    expected key through the profile secret scope, so the listener owner's key is rejected there. A
    profile whose key cannot be resolved keeps the historical unprefixed post rather than turning
    every wake into a 401."""
    import aiohttp
    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"  # wildcard bind — connect over loopback
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError("wake self-post requires API_SERVER_KEY: session continuation via "
                           "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
                           "server, so the wake cannot reach the target session")
    path_prefix = ""
    target_profile = str(profile or "").strip()
    if target_profile and target_profile != _DEFAULT_PROFILE_NAME:
        profile_key = _profile_scoped_api_key(target_profile)
        if profile_key:
            path_prefix = "/p/" + quote(target_profile, safe="")
            api_key = profile_key
        else:
            logger.warning(
                "wake self-post for session %s cannot resolve a usable API_SERVER_KEY for profile "
                "%r; falling back to the default-profile route", session_id, target_profile)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}{path_prefix}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "X-Hermes-Session-Id": session_id}
    payload = {"model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
               "messages": [{"role": "user", "content": text}], "stream": False}
    if nolgia_token:
        # Attribution for the continuation turn: the ORIGINATING run's credential, so its tool
        # subprocesses (nolgia CLI uploads) name the turn whose work this wake finishes. Loopback
        # POST on the in-pod API server, gated by the same bearer as every other request; never logged.
        payload["nolgia_token"] = nolgia_token
    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:  # concurrency cap — transient; back off and retry
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) for session {session_id}"
                        )
                        logger.warning("%s; attempt %d/%d", last_err, attempt + 1, attempts)
                        continue
                    if resp.status >= 400:  # non-transient (auth/validation): fail immediately
                        body = (await resp.text())[:300]
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    logger.info("wake self-post delivered for session %s (attempt %d)", session_id, attempt + 1)
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning("wake self-post transient failure for session %s (attempt %d/%d): %s",
                           session_id, attempt + 1, attempts, exc)
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after {attempts} attempts: {last_err}"
    ) from last_err
