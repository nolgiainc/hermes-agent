"""Wake an existing agent session from a background completion event.

Two delivery strategies, selected by the target adapter's
``supports_async_delivery`` capability flag:

* Push-capable adapters (telegram, discord, plugin platforms, ...): inject a
  synthetic ``MessageEvent(internal=True)`` through ``adapter.handle_message``
  — the pre-existing wake path, preserved exactly.

* Stateless request/response adapters (the API server,
  ``supports_async_delivery = False``): ``handle_message`` would run the wake
  turn under a ``build_session_key()``-derived key
  (``agent:main:api_server:group:<sid>``) that NEVER matches the raw
  ``X-Hermes-Session-Id`` key real gateway/HQ turns run under
  (``_bind_api_server_session``), so the wake lands in a parallel, invisible
  session. Instead we self-POST ``/v1/chat/completions`` on the in-pod API
  server with the raw session id in the ``X-Hermes-Session-Id`` header — the
  exact entry point real turns use — so the wake turn resumes the REAL
  session, with full history, and its result is visible the next time the
  client polls/reopens the conversation.

Failures RAISE (after bounded retries on transient errors) so callers can
rewind cursors / retry instead of silently losing the event.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Wire spelling of the default profile: ``/p/default/...`` and the unprefixed
# routes are the same runtime, so it needs no prefix (see
# gateway.platforms.api_server._DEFAULT_PROFILE_NAME).
_DEFAULT_PROFILE_NAME = "default"

# A wake self-post runs the entire agent turn synchronously (stream=false);
# generous ceiling so long tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff delays between retries on transient failures (429 concurrency cap,
# connection errors). The API server has no per-session lock — concurrent
# turns on one session are last-writer-wins — but it DOES enforce a global
# max_concurrent_runs cap via HTTP 429, which is worth waiting out.
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
    """Whether this adapter can push a message to the user after a turn ends.

    Mirrors ``gateway.session_context.async_delivery_supported`` but reads the
    capability off the adapter class (``supports_async_delivery``) instead of
    the request-scoped contextvar — background watchers run outside any bound
    session context. Adapters that don't declare the flag are push-capable.
    """
    return bool(getattr(adapter, "supports_async_delivery", True))


async def deliver_wake(
    adapter: Any,
    *,
    text: str,
    session_id: str = "",
    source: Any = None,
    nolgia_token: str = "",
    profile: str = "",
) -> None:
    """Deliver a wake turn to the session behind ``adapter``.

    ``session_id`` is the RAW session id (the ``X-Hermes-Session-Id`` value /
    ``state.db`` key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` used to build the synthetic event — required for
    push-capable adapters.

    ``profile`` is the multiplex profile of the session being woken (the profile
    that served the originating turn). A multiplexed session lives in ITS
    profile's runtime — own ``HERMES_HOME``/state, own secret scope, own
    retained-credential scope — so the self-post must re-enter through that
    profile's ``/p/<profile>/...`` route; posting to the unprefixed route would
    run the continuation under the DEFAULT profile, where the session's history
    and its retained turn credential do not exist (NOL-413). ``""`` (or
    ``"default"``) is the default profile and keeps the unprefixed route.
    Push-capable adapters ignore it: their wake re-enters through the normal
    message pipeline, which is already bound to the right profile.

    ``nolgia_token`` is the run-scoped Nolgia credential of the ORIGINATING
    turn (the run whose detached work this wake reports). Carried on the
    self-post so the continuation's CLI work attributes to that run rather than
    to whichever turn the session ran most recently — same-session concurrent
    runs are supported, and a detached delegation has no wall-clock bound, so
    the two can differ (NOL-413). Empty means "no run scoping"; the API server
    then falls back to the session's retained token. Push-capable adapters
    ignore it: their wake re-enters through the normal message pipeline, which
    binds its own session context.

    Raises on failure (bad arguments, exhausted retries, HTTP error) so the
    caller can rewind/retry instead of treating the wake as delivered.
    """
    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError(
                "deliver_wake: push-capable adapter requires a SessionSource"
            )
        from gateway.platforms.base import MessageEvent, MessageType

        synth_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
        )
        await adapter.handle_message(synth_event)
        return

    if not session_id:
        raise ValueError(
            "deliver_wake: non-push adapter (supports_async_delivery=False) "
            "requires the raw session id to self-post the wake turn"
        )
    await _self_post_chat_completion(
        adapter,
        text=text,
        session_id=session_id,
        nolgia_token=nolgia_token,
        profile=profile,
    )


async def _self_post_chat_completion(
    adapter: Any,
    *,
    text: str,
    session_id: str,
    nolgia_token: str = "",
    profile: str = "",
) -> None:
    """POST the wake text to the in-pod API server as a normal session turn.

    Uses the adapter's own bind host/port/key (``ApiServerAdapter.__init__``).
    Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY`` being configured, so a missing key is a hard error —
    raise loudly rather than run the wake in a fresh fingerprint-derived
    session nobody is looking at.

    A named ``profile`` posts through that profile's ``/p/<profile>/`` mirror
    (see ``deliver_wake``), authenticated with the profile's OWN
    ``API_SERVER_KEY`` — the multiplex auth gate resolves the expected key
    through the profile secret scope, so the listener owner's key is rejected
    there. A profile whose key cannot be resolved keeps the historical
    unprefixed post rather than turning every wake into a 401.
    """
    import aiohttp

    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        # Wildcard bind address — connect over loopback.
        host = "127.0.0.1"
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError(
            "wake self-post requires API_SERVER_KEY: session continuation via "
            "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
            "server, so the wake cannot reach the target session"
        )

    path_prefix = ""
    target_profile = str(profile or "").strip()
    if target_profile and target_profile != _DEFAULT_PROFILE_NAME:
        profile_key = _profile_scoped_api_key(target_profile)
        if profile_key:
            path_prefix = "/p/" + quote(target_profile, safe="")
            api_key = profile_key
        else:
            logger.warning(
                "wake self-post for session %s cannot resolve a usable "
                "API_SERVER_KEY for profile %r; falling back to the "
                "default-profile route",
                session_id,
                target_profile,
            )

    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}{path_prefix}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Hermes-Session-Id": session_id,
    }
    payload = {
        "model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }
    if nolgia_token:
        # Attribution for the continuation turn: the ORIGINATING run's
        # credential, so its tool subprocesses (nolgia CLI uploads) name the
        # turn whose work this wake finishes. Loopback POST on the in-pod API
        # server, gated by the same bearer as every other request; never logged.
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
                    if resp.status == 429:
                        # Global concurrency cap (max_concurrent_runs) —
                        # transient; back off and retry.
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) "
                            f"for session {session_id}"
                        )
                        logger.warning(
                            "%s; attempt %d/%d", last_err, attempt + 1, attempts
                        )
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        # Non-transient (auth/validation) — fail immediately.
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: "
                            f"HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    logger.info(
                        "wake self-post delivered for session %s (attempt %d)",
                        session_id,
                        attempt + 1,
                    )
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning(
                "wake self-post transient failure for session %s "
                "(attempt %d/%d): %s",
                session_id,
                attempt + 1,
                attempts,
                exc,
            )
            continue
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after "
        f"{attempts} attempts: {last_err}"
    ) from last_err
