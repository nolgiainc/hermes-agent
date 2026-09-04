#!/usr/bin/env python3
"""
Vision Tools Module

This module provides vision analysis tools that work with image URLs.
Uses the centralized auxiliary vision router, which can select OpenRouter,
Nous, Codex, native Anthropic, or a custom OpenAI-compatible endpoint.

Available tools:
- vision_analyze_tool: Analyze images from URLs with custom prompts

Features:
- Downloads images from URLs and converts to base64 for API compatibility
- Comprehensive image description
- Context-aware analysis based on user queries
- Automatic temporary file cleanup
- Proper error handling and validation
- Debug logging support

Usage:
    from vision_tools import vision_analyze_tool
    import asyncio
    
    # Analyze an image
    result = await vision_analyze_tool(
        image_url="https://example.com/image.jpg",
        user_prompt="What architectural style is this building?"
    )
"""

import base64
import contextlib
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional
from urllib.parse import urlparse
import httpx

# ``agent.auxiliary_client`` pulls credential_pool → hermes_cli.auth → httpx
# → rich (~50 ms cold); only vision handlers need it. Loaded lazily; both
# names stay module attributes so tests can keep patching
# ``tools.vision_tools.async_call_llm``. Truthy-skip: injected mocks win.
async_call_llm: Any = None
extract_content_or_reasoning: Any = None


def _load_auxiliary_client() -> None:
    global async_call_llm, extract_content_or_reasoning
    if async_call_llm is None or extract_content_or_reasoning is None:
        from agent.auxiliary_client import (
            async_call_llm as _acl,
            extract_content_or_reasoning as _ecr,
        )
        if async_call_llm is None:
            async_call_llm = _acl
        if extract_content_or_reasoning is None:
            extract_content_or_reasoning = _ecr


from hermes_constants import get_hermes_dir
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access
import sys

logger = logging.getLogger(__name__)

_debug = DebugSession("vision_tools", env_var="VISION_TOOLS_DEBUG")

# Configurable HTTP download timeout for _download_image().
# Separate from auxiliary.vision.timeout which governs the LLM API call.
# Resolution: config.yaml auxiliary.vision.download_timeout → env var → 30s default.
def _resolve_download_timeout() -> float:
    env_val = os.getenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    try:
        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "download_timeout")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 30.0


_VISION_DOWNLOAD_TIMEOUT = _resolve_download_timeout()

# Per-attempt timeout for the vision/video LLM call itself.  Separate from
# _VISION_DOWNLOAD_TIMEOUT above, which governs only the HTTP image fetch.
# Resolution: env HERMES_VISION_TIMEOUT → config.yaml auxiliary.vision.timeout
# → the caller's default, mirroring _resolve_download_timeout().
#
# 60s, not the old 120s: every other auxiliary task defaults to 30s
# (_DEFAULT_AUX_TIMEOUT), and a healthy provider returns this tool's capped
# max_tokens=2000 analysis in well under 20s.  A vision call still running at
# 60s is stalled, not slow, and a stall here is pure dead wall-clock inside a
# user-visible agent run.  Local VLMs (llama.cpp, ollama) that legitimately
# need longer raise it with one line of config or the env var.
_VISION_DEFAULT_TIMEOUT = 60.0


def _resolve_vision_timeout(
    default: float = _VISION_DEFAULT_TIMEOUT, *, floor: float = 0.0
) -> float:
    """Resolve the per-attempt vision LLM timeout, never below *floor*.

    ``floor`` exists for ``video_analyze_tool``, whose payloads are an order of
    magnitude larger than a single image: it keeps a low image-tuned config
    value from starving video analysis.  Best-effort throughout — any config
    read or parse failure falls back to *default*.
    """
    try:
        env_val = os.getenv("HERMES_VISION_TIMEOUT", "").strip()
        if env_val:
            try:
                resolved = float(env_val)
            except ValueError:
                pass
            else:
                return max(resolved, floor)

        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "timeout")
        resolved = float(val) if val is not None else default
    except Exception:
        resolved = default
    return max(resolved, floor)


# Hard cap on downloaded image file size (50 MB). Prevents OOM from
# attacker-hosted multi-gigabyte files or decompression bombs.
_VISION_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


import threading

# ---------------------------------------------------------------------------
# Consecutive-timeout loop guard (NOL-197)
# ---------------------------------------------------------------------------
# Measured on a live production pod (NOL-151 run): 6 of 15 vision_analyze
# calls timed out at ~68s each, and the agent's response to each timeout was
# to downsize the same contact sheet and resubmit — full res, 1400px, 1024px
# all burned the entire per-attempt budget, ~7 minutes of a 16.6-minute tail,
# before it finally switched to single frames (which mostly succeed). The
# structured error alone did not tell the model that a THIRD resolution of an
# image the route just proved it cannot analyze within budget is not a new
# experiment. This guard does: after _VISION_TIMEOUT_SWITCH_AT consecutive
# timeouts the error's analysis text orders the strategy switch outright.
#
# Process-wide by design (a threading.Lock, not per-session): concurrent
# vision calls run on per-thread event loops (see the CPU-cap note below),
# and every session in the process shares one auxiliary vision route — two
# consecutive full-budget burns are evidence about the ROUTE, whoever made
# them. Any success resets the streak.
_VISION_TIMEOUT_SWITCH_AT = 2
_vision_timeout_streak_lock = threading.Lock()
_vision_timeout_streak = 0


def _is_vision_timeout(exc: Exception) -> bool:
    """A full-budget request timeout, as the auxiliary router classifies it
    (same predicate family as auxiliary_client._is_timeout_error)."""
    try:
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    except ImportError:
        pass
    if "Timeout" in type(exc).__name__:
        return True
    return "timed out" in str(exc).lower()


def _record_vision_timeout() -> int:
    """Advance the consecutive-timeout streak; returns the new count."""
    global _vision_timeout_streak
    with _vision_timeout_streak_lock:
        _vision_timeout_streak += 1
        return _vision_timeout_streak


def _reset_vision_timeout_streak() -> None:
    global _vision_timeout_streak
    with _vision_timeout_streak_lock:
        _vision_timeout_streak = 0


def _vision_timeout_analysis(exc: Exception, streak: int,
                             timeout: float, *,
                             probe_active: bool = False) -> str:
    """The analysis text for a timed-out vision call — streak-aware so the
    second consecutive burn says 'switch strategy', not just 'it failed'."""
    if streak >= _VISION_TIMEOUT_SWITCH_AT:
        text = (
            f"Vision request timed out after its {timeout:.0f}s per-attempt "
            f"budget — the {_ordinal(streak)} consecutive vision timeout. "
            "STOP: do not resize this image and retry — an image class that "
            "has timed out twice will keep timing out at any resolution "
            "(dense composites like contact sheets, grids and collages are "
            "the classic case). Switch strategy NOW: analyze single small "
            "frames one at a time (those fit the budget), or record this "
            "check as 'QC unavailable — vision timeout' and continue the "
            f"task. Error: {exc}"
        )
        if probe_active:
            text += (
                " (Degraded mode is active: vision calls are running with a "
                f"reduced {timeout:.0f}s probe budget so repeated failures "
                "cannot burn run budget; the full budget restores "
                "automatically after the next successful vision call.)"
            )
        return text
    return (
        f"Vision request timed out after its {timeout:.0f}s per-attempt "
        "budget. One more attempt on this image is reasonable (a smaller "
        "or simpler version helps); if that also times out, stop retrying "
        "this image class and switch strategy — single small frames, or "
        f"record the check as unavailable and continue. Error: {exc}"
    )


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _current_vision_timeout_streak() -> int:
    """Read the consecutive-timeout streak without mutating it."""
    with _vision_timeout_streak_lock:
        return _vision_timeout_streak


# ---------------------------------------------------------------------------
# Degraded-mode probe budget (NOL-253)
# ---------------------------------------------------------------------------
# The NOL-197 guard above rewrites the ERROR TEXT once the streak reaches
# _VISION_TIMEOUT_SWITCH_AT, but the text is advisory: the NOL-151 live run
# (session 8050af4b) drove the streak to 7, every member burning the full
# per-attempt budget — ~7 minutes of a 16.6-minute post-generation tail on
# calls that returned nothing.  This is the mechanical half of the guard:
# once the streak is at or past the switch threshold, vision calls run with
# a reduced probe budget instead of the full one.  A healthy route answers
# a QC-sized request well inside the probe window (healthy analyses finish
# in well under 20s — see the 60s-budget rationale above), so recovery is
# still discovered, and any success resets the streak and restores the full
# budget.  A route that stays degraded now costs probe-seconds per call,
# not full-budget-seconds.
#
# Resolution: env HERMES_VISION_PROBE_TIMEOUT → config
# auxiliary.vision.probe_timeout → 15.0.  Values <= 0 disable the cap
# (degraded-mode calls keep the full budget).
_VISION_DEFAULT_PROBE_TIMEOUT = 15.0


def _resolve_vision_probe_timeout(
    default: float = _VISION_DEFAULT_PROBE_TIMEOUT,
) -> float:
    """Resolve the degraded-mode per-attempt budget.  Best-effort: any
    config read or parse failure falls back to *default*."""
    try:
        env_val = os.getenv("HERMES_VISION_PROBE_TIMEOUT", "").strip()
        if env_val:
            try:
                return float(env_val)
            except ValueError:
                pass

        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "probe_timeout")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# CPU-burst concurrency cap (vision encode/resize)
# ---------------------------------------------------------------------------
# A single agent turn can fan out N vision_analyze calls at once (the classic
# trigger is "analyze every frame of this video" — ffmpeg explodes a clip into
# dozens of frames, the model then calls vision_analyze on each). Each call does
# a CPU-heavy base64-encode + (sometimes) Pillow resize. The tool executor runs
# concurrent tool calls on a ThreadPoolExecutor (agent.tool_executor =
# 8 workers) PER SESSION, and several agent sessions share one process (the
# dashboard runs the agent in-process). Unbounded, a video-frame fan-out across
# one or more sessions runs *every* encode at once, saturates all cores, and
# leaves no CPU to service the shared asyncio event loop that serves the
# dashboard's /api/status liveness probe — so the instance flaps to UNHEALTHY
# even though nothing has crashed (observed in prod, June 2026).
#
# The fix is NOT to cap how many vision analyses run — multi-image workflows
# ("compare these 6 screenshots", "read this 10-page scan") legitimately want
# high concurrency, and the slow part (the LLM stream) is network-bound and
# harmless to the loop. We cap ONLY the CPU burst: the encode/resize is offloaded
# to a dedicated, bounded executor sized to the host's usable core count. That
# is the resource the incident actually exhausted (cores), so bounding it to
# cores is *correct*, not an arbitrary number — excess encodes queue on the
# executor instead of all running at once, the LLM calls stay fully concurrent,
# and the loop always keeps a core. No fixed ceiling: the limit tracks the host.
#
# A threading primitive (NOT asyncio) is required: each vision call is dispatched
# through model_tools._run_async on a PER-THREAD event loop, so an asyncio
# executor/semaphore bound to one loop cannot coordinate across them. A
# ThreadPoolExecutor is loop- and thread-agnostic.
import threading  # noqa: F401  (kept for downstream importers / patch targets)


def _detect_host_cpus() -> int:
    """Best-effort host CPU count, honoring cgroup/affinity limits when set.

    Prefers ``os.sched_getaffinity`` (the CPUs this process may actually run
    on — respects container/cpuset pinning) and falls back to
    ``os.cpu_count()``. Returns at least 1.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_vision_cpu_workers() -> int:
    """Resolve how many vision encode/resize bursts may run concurrently.

    Defaults to the host's usable core count (``_detect_host_cpus``) — no fixed
    ceiling, because the cap tracks the actual exhausted resource (CPU cores),
    not a magic number. The LLM call is NOT covered by this limit, so legitimate
    multi-image fan-out keeps full request concurrency; only the simultaneous
    CPU bursts are bounded so the event loop always keeps a core.

    Resolution order: HERMES_VISION_MAX_CONCURRENCY env →
    config.yaml auxiliary.vision.max_concurrency → host core count. Any value
    that parses to < 1 is ignored in favor of the next source so the cap can
    never be disabled into an unbounded encode storm.
    """
    env_val = os.getenv("HERMES_VISION_MAX_CONCURRENCY", "").strip()
    if env_val:
        try:
            parsed = int(env_val)
            if parsed >= 1:
                return parsed
        except ValueError:
            pass
    try:
        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "max_concurrency")
        if val is not None:
            parsed = int(val)
            if parsed >= 1:
                return parsed
    except Exception:
        pass
    return _detect_host_cpus()


_VISION_CPU_WORKERS = _resolve_vision_cpu_workers()

# Dedicated, bounded executor for the CPU-bound encode/resize burst ONLY. We do
# NOT use the default executor (run_in_executor(None, ...)) — that pool is shared
# with the gateway and web server, so a fan-out would park encode work there and
# starve those callers. Sizing it to the usable core count means at most
# _VISION_CPU_WORKERS encodes run at once; further encodes queue on this
# executor's work queue, leaving cores free for the event loop. The LLM call is
# deliberately left OUTSIDE this executor so multi-image workflows keep full
# request concurrency.
_vision_cpu_executor = ThreadPoolExecutor(
    max_workers=_VISION_CPU_WORKERS,
    thread_name_prefix="vision-encode",
)


async def _run_encode_on_cpu_executor(fn, *args, **kwargs):
    """Run a sync encode/resize callable on the bounded vision CPU executor.

    Offloads CPU-bound image work to :data:`_vision_cpu_executor` so it (a)
    never runs on the caller's event-loop thread and (b) is bounded to the
    host's usable core count process-wide. Excess encodes queue on the
    executor instead of all running at once, leaving cores free for the loop.
    The LLM call must NOT be routed through here — only the encode/resize.
    """
    import functools
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _vision_cpu_executor, functools.partial(fn, *args, **kwargs)
    )


def _image_url_shape_ok(url: str) -> bool:
    """HTTP(S) shape check only (scheme, netloc). No DNS."""
    if not url or not isinstance(url, str):
        return False
    # Basic HTTP/HTTPS URL check
    if not url.startswith(("http://", "https://")):
        return False
    # Parse to ensure we at least have a network location; still allow URLs
    # without file extensions (e.g. CDN endpoints that redirect to images).
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    return True


def _validate_image_url(url: str) -> bool:
    """Validate image URL for sync callers and tests (SSRF via sync DNS check)."""
    if not _image_url_shape_ok(url):
        return False
    # Block private/internal addresses to prevent SSRF
    from tools.url_safety import is_safe_url
    return is_safe_url(url)


async def _validate_image_url_async(url: str) -> bool:
    """Validate remote image URL without blocking the event loop on DNS."""
    if not _image_url_shape_ok(url):
        return False
    from tools.url_safety import async_is_safe_url
    return await async_is_safe_url(url)


def _detect_image_mime_type_from_bytes(data: bytes) -> Optional[str]:
    """Magic-byte MIME sniff on raw bytes (authoritative; no extension trust).

    Returns ``None`` for anything without a recognized image header — including
    SVG, which has no magic bytes. The resolver special-cases SVG (sniffs
    ``<svg``) and passes it through for rasterization at the call sites.
    """
    header = data[:64]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


# Media types the major vision providers (Anthropic in particular) accept for
# inline base64 images.  Anything outside this set — SVG, BMP, TIFF, etc. — is
# rejected with a non-retryable 400.  Because a vision tool-result is baked into
# immutable conversation history and re-sent every turn, embedding an
# unsupported media_type permanently wedges the session (retries re-send the
# same bad bytes).  We MUST normalize to one of these before embedding.
_ANTHROPIC_SUPPORTED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


def _rasterize_svg_to_png(svg_path: Path, out_path: Path) -> bool:
    """Best-effort SVG → PNG rasterization. Returns True on success.

    Tries, in order: cairosvg, svglib+reportlab, then system rasterizers
    (rsvg-convert, inkscape).  All are soft dependencies; if none is available
    we return False and the caller rejects the image with an actionable error
    rather than embedding an unsupported media_type that would wedge the
    session.
    """
    # 1) cairosvg (pure-python-ish, most common)
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(url=str(svg_path), write_to=str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        pass
    # 2) svglib + reportlab
    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPM  # type: ignore
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            renderPM.drawToFile(drawing, str(out_path), fmt="PNG")
            return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        pass
    # 3) system rasterizers
    import shutil as _shutil
    import subprocess as _subprocess
    for cmd in (
        ["rsvg-convert", "-o", str(out_path), str(svg_path)],
        ["inkscape", str(svg_path), "--export-type=png",
         f"--export-filename={out_path}"],
    ):
        if _shutil.which(cmd[0]):
            try:
                _subprocess.run(
                    cmd, check=True, capture_output=True, timeout=30,
                    stdin=_subprocess.DEVNULL,
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    return True
            except Exception:
                continue
    return False


def _normalize_to_supported_image(
    image_path: Path, detected_mime: str
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Ensure an image is in a vision-provider-supported format.

    Returns a 3-tuple ``(path, mime, error)``:
      - If ``detected_mime`` is already supported: ``(image_path, detected_mime, None)``.
      - If conversion succeeds: ``(new_png_path, "image/png", None)`` — the new
        path is a temp file the CALLER must clean up.
      - If conversion is impossible: ``(None, None, <error message>)``.

    SVG is rasterized to PNG (best-effort, soft deps).  Other raster formats
    Pillow can read (BMP, TIFF, etc.) are re-encoded to PNG.  This runs BEFORE
    the image is base64-embedded into conversation history, so an unsupported
    media_type can never reach the provider and wedge the session.
    """
    if detected_mime in _ANTHROPIC_SUPPORTED_MEDIA_TYPES:
        return image_path, detected_mime, None

    out_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"converted_{uuid.uuid4()}.png"

    # SVG: needs a rasterizer (Pillow cannot render SVG).
    if detected_mime == "image/svg+xml":
        if _rasterize_svg_to_png(image_path, out_path):
            return out_path, "image/png", None
        return (
            None,
            None,
            "This is an SVG, which vision models cannot read directly, and no "
            "SVG rasterizer is installed (tried cairosvg, svglib, rsvg-convert, "
            "inkscape). Convert the SVG to PNG first — e.g. open it in a browser "
            "and screenshot it, or install a rasterizer "
            "(`pip install cairosvg`) — then re-run vision_analyze on the PNG.",
        )

    # Other non-supported raster formats (BMP, TIFF, ...): re-encode via Pillow.
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            if _img.mode not in ("RGB", "RGBA", "L"):
                _img = _img.convert("RGBA")
            _img.save(out_path, format="PNG")
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path, "image/png", None
    except Exception as _exc:
        logger.warning("Failed to normalize %s image to PNG: %s",
                       detected_mime, _exc)
    return (
        None,
        None,
        f"Image format {detected_mime!r} is not supported by the vision API "
        f"and could not be converted to PNG (install Pillow for raster "
        f"conversion). Convert it to PNG or JPEG and try again.",
    )


def _is_retryable_download_error(error: Exception) -> bool:
    """Return True only for transient image-download failures worth retrying.

    Non-retryable (fail-fast):
      - httpx.HTTPStatusError with a 4xx status other than 429 (404/403/410/...):
        the resource is missing or forbidden; retrying can't change that.
      - PermissionError: blocked by website policy / SSRF guard.
      - ValueError: image too large or blocked redirect — deterministic.

    Retryable (transient):
      - httpx 429 (rate limited) and 5xx (server-side) errors.
      - Connection/timeout/transport errors (httpx.TransportError) and any
        other unclassified exception, which may be a flaky network blip.
    """
    if isinstance(error, (PermissionError, ValueError)):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if 400 <= status < 500 and status != 429:
            return False
        return True
    return True


async def _download_image(image_url: str, destination: Path, max_retries: int = 3) -> Path:
    """
    Download an image from a URL to a local destination (async) with retry logic.
    
    Args:
        image_url (str): The URL of the image to download
        destination (Path): The path where the image should be saved
        max_retries (int): Maximum number of retry attempts (default: 3)
        
    Returns:
        Path: The path to the downloaded image
        
    Raises:
        Exception: If download fails after all retries
    """
    import asyncio
    
    # Create parent directories if they don't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    async def _ssrf_redirect_guard(response):
        """Re-validate each redirect target to prevent redirect-based SSRF.

        Without this, an attacker can host a public URL that 302-redirects
        to http://169.254.169.254/ and bypass the pre-flight is_safe_url check.

        Must be async because httpx.AsyncClient awaits event hooks.
        """
        from tools.url_safety import async_is_safe_url, redirect_target_from_response
        redirect_url = redirect_target_from_response(response)
        if redirect_url and not await async_is_safe_url(redirect_url):
            raise ValueError(
                f"Blocked redirect to private/internal address: {redirect_url}"
            )

    last_error = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(image_url)
            if blocked:
                raise PermissionError(blocked["message"])

            from tools.url_safety import create_ssrf_safe_async_client

            # Download the image with appropriate headers using async httpx
            # Enable follow_redirects to handle image CDNs that redirect (e.g., Imgur, Picsum)
            # SSRF: the client validates DNS at TCP connect time; event_hooks
            # validate each redirect target against private IP ranges.
            async with create_ssrf_safe_async_client(
                timeout=_VISION_DOWNLOAD_TIMEOUT,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                response = await client.get(
                    image_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()

                # Reject overly large images early via Content-Length header.
                cl = response.headers.get("content-length")
                if cl and int(cl) > _VISION_MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image too large ({int(cl)} bytes, max {_VISION_MAX_DOWNLOAD_BYTES})"
                    )

                final_url = str(response.url)
                blocked = check_website_access(final_url)
                if blocked:
                    raise PermissionError(blocked["message"])
                
                # Save the image content (double-check actual size)
                body = response.content
                if len(body) > _VISION_MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image too large ({len(body)} bytes, max {_VISION_MAX_DOWNLOAD_BYTES})"
                    )
                destination.write_bytes(body)
            
            return destination
        except Exception as e:
            last_error = e
            # Error-class-aware retry: only retry transient failures. A 4xx
            # client error (404/403/410, etc.) will never succeed on retry —
            # the resource isn't there or we're not allowed — so burning 3
            # attempts with 2s/4s/8s backoff just inflates latency. 429 (rate
            # limit) and 5xx remain retryable. PermissionError (policy block)
            # and ValueError (too-large / SSRF redirect) are also terminal.
            if not _is_retryable_download_error(e) or attempt >= max_retries - 1:
                logger.error(
                    "Image download failed after %s attempt(s): %s",
                    attempt + 1,
                    str(e)[:100],
                    exc_info=True,
                )
                raise
            wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
            logger.warning("Image download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
            logger.warning("Retrying in %ss...", wait_time)
            await asyncio.sleep(wait_time)

    # The loop always returns on success or re-raises on the final/non-retryable
    # attempt, so reaching here means max_retries was non-positive.
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"_download_image exited retry loop without attempting (max_retries={max_retries})"
    )


def _determine_mime_type(image_path: Path) -> str:
    """
    Determine the MIME type of an image based on its file extension.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        str: The MIME type (defaults to image/jpeg if unknown)
    """
    extension = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml'
    }
    return mime_types.get(extension, 'image/jpeg')


def _image_to_base64_data_url(image_path: Path, mime_type: Optional[str] = None) -> str:
    """
    Convert an image file to a base64-encoded data URL.
    
    Args:
        image_path (Path): Path to the image file
        mime_type (Optional[str]): MIME type of the image (auto-detected if None)
        
    Returns:
        str: Base64-encoded data URL (e.g., "data:image/jpeg;base64,...")
    """
    # Read the image as bytes
    data = image_path.read_bytes()
    
    # Encode to base64
    encoded = base64.b64encode(data).decode("ascii")
    
    # Determine MIME type
    mime = mime_type or _determine_mime_type(image_path)
    
    # Create data URL
    data_url = f"data:{mime};base64,{encoded}"
    
    return data_url


# Absolute hard ceiling for vision API payloads (20 MB) — above this, no major
# provider accepts the image and we reject outright.
_MAX_BASE64_BYTES = 20 * 1024 * 1024

# Proactive embed cap (4 MB).  This is the size we resize an image DOWN to
# before embedding it into conversation history, regardless of the 20 MB hard
# ceiling.  Anthropic's per-image base64 limit is 5 MB; once an oversized image
# is baked into history (e.g. a vision tool-result), it is re-sent on every
# subsequent turn and permanently wedges the session with a 400 that retries
# can't clear (the bad bytes are immutable history).  Capping at embed time —
# with headroom under 5 MB — is the only durable fix.  Matches the post-failure
# shrink target in agent.conversation_compression so behaviour is consistent
# whether we resize proactively or reactively.
_EMBED_TARGET_BYTES = 4 * 1024 * 1024

# Proactive embed dimension cap (px, longest side).  Anthropic enforces an
# 8000px per-side ceiling INDEPENDENTLY of the 5 MB byte cap — a tall full-page
# screenshot can be well under 5 MB yet far over 8000px (e.g. 1200×12000 at
# 0.06 MB), so the byte-only embed check above lets it slip into immutable
# history un-resized and the session bricks on a non-retryable 400.  We cap at
# 7900 (headroom under 8000) so the proactive resize shrinks tall small-byte
# images before they are embedded.
_EMBED_MAX_DIMENSION = 7900

# Target size when auto-resizing on API failure (5 MB).  After a provider
# rejects an image, we downscale to this target and retry once.
_RESIZE_TARGET_BYTES = 5 * 1024 * 1024

# Longest side (px) sent to the vision LLM (NOL-253).  Independent of the
# 7900px EMBED ceiling above, which only guards Anthropic's hard 8000px
# reject: this one is a performance cap.  Every cloud route we serve
# downscales internally to roughly 2048px or less before the model sees the
# image (OpenAI fits high-detail images inside 2048x2048; Anthropic caps
# the long side around 1568), so pixels beyond ~2048 never reach the model
# — they are pure upload bytes and provider-side tiling latency.  The
# NOL-151 run measured the result on montage-built contact sheets shipped
# at full resolution: 6 of 15 QC vision_analyze calls burned their whole
# per-attempt budget and returned nothing.
# Resolution: env HERMES_VISION_MAX_DIMENSION → config
# auxiliary.vision.max_dimension → 2048.  Values <= 0 disable the cap
# (full-resolution sends).
_VISION_SEND_MAX_DIMENSION = 2048


def _resolve_vision_max_dimension(
    default: int = _VISION_SEND_MAX_DIMENSION,
) -> int:
    """Resolve the longest-side pixel cap for vision sends.  Best-effort:
    any config read or parse failure falls back to *default*."""
    try:
        env_val = os.getenv("HERMES_VISION_MAX_DIMENSION", "").strip()
        if env_val:
            try:
                return int(float(env_val))
            except ValueError:
                pass

        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "max_dimension")
        if val is not None:
            return int(float(val))
    except Exception:
        pass
    return default


def _is_image_size_error(error: Exception) -> bool:
    """Detect if an API error is related to image or payload size."""
    err_str = str(error).lower()
    return any(hint in err_str for hint in (
        "too large", "payload", "413", "content_too_large",
        "request_too_large", "image_url", "invalid_request",
        "exceeds", "size limit",
    ))


def _image_exceeds_dimension(image_path: Path, max_dimension: int) -> bool:
    """True if the image's longest side exceeds ``max_dimension`` px.

    Anthropic enforces an 8000px per-side cap independently of the 5 MB byte
    cap, so a tall small-byte screenshot can pass every byte check yet trip a
    non-retryable 400.  Returns False (don't force a resize) when Pillow is
    unavailable or the file can't be read as an image — the byte-based checks
    still apply, and we never want a missing soft dependency to break the
    embed path.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            return max(_img.size) > max_dimension
    except Exception:
        return False


def _resize_image_for_vision(image_path: Path, mime_type: Optional[str] = None,
                              max_base64_bytes: int = _RESIZE_TARGET_BYTES,
                              max_dimension: Optional[int] = None) -> str:
    """Convert an image to a base64 data URL, auto-resizing if too large.

    Tries Pillow first to progressively downscale oversized images.  If Pillow
    is not installed or resizing still exceeds the limit, falls back to the raw
    bytes and lets the caller handle the size check.

    Args:
        max_dimension: If set, images whose longest side exceeds this pixel
            count are forcibly downscaled even if they're under the byte
            budget.  Anthropic enforces an 8000 px per-side cap independently
            of the 5 MB byte cap.

    Returns the base64 data URL string.
    """
    # Quick file-size estimate: base64 expands by ~4/3, plus data URL header.
    # Skip the expensive full-read + encode if Pillow can resize directly.
    file_size = image_path.stat().st_size
    estimated_b64 = (file_size * 4) // 3 + 100  # ~header overhead
    needs_resize_for_bytes = estimated_b64 > max_base64_bytes

    # Check pixel dimensions even if bytes are fine.
    needs_resize_for_dims = False
    if max_dimension is not None:
        try:
            from PIL import Image as _PILQuick
            with _PILQuick.open(image_path) as _quick_img:
                if max(_quick_img.size) > max_dimension:
                    needs_resize_for_dims = True
        except Exception:
            pass  # can't check; Pillow path below will handle or skip

    if not needs_resize_for_bytes and not needs_resize_for_dims:
        # Small enough — just encode directly.
        data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        if len(data_url) <= max_base64_bytes:
            return data_url
    else:
        data_url = None  # defer full encode; try Pillow resize first

    # Attempt auto-resize with Pillow (soft dependency)
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        # Pillow is a lazy-installable soft dependency. Try a best-effort
        # install (respects security.allow_lazy_installs; no-op if disabled or
        # offline), then re-import. If it still isn't importable, fall back to
        # the raw bytes and let the caller raise the size error.
        try:
            from tools.lazy_deps import ensure as _ensure_dep
            # prompt=False: never raise a blocking input() prompt mid-session.
            # Under the interactive CLI prompt_toolkit owns stdin, so a bare
            # input() deadlocks the terminal (#40490). The install is already
            # gated by security.allow_lazy_installs, so reaching here is opt-in.
            _ensure_dep("tool.vision", prompt=False)
            from PIL import Image
            import io as _io
        except Exception:
            logger.info("Pillow not installed — cannot auto-resize oversized image")
            if data_url is None:
                data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
            return data_url  # caller will raise the size error

    logger.info("Image file is %.1f MB (estimated base64 %.1f MB, limit %.1f MB, max_dimension=%s), auto-resizing...",
                file_size / (1024 * 1024), estimated_b64 / (1024 * 1024),
                max_base64_bytes / (1024 * 1024), max_dimension)

    mime = mime_type or _determine_mime_type(image_path)
    # Choose output format: JPEG for photos (smaller), PNG for transparency
    pil_format = "PNG" if mime == "image/png" else "JPEG"
    out_mime = "image/png" if pil_format == "PNG" else "image/jpeg"

    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.info("Pillow cannot open image for resizing: %s", exc)
        if data_url is None:
            data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        return data_url  # fall through to size-check in caller
    # Convert RGBA to RGB for JPEG output
    if pil_format == "JPEG" and img.mode in {"RGBA", "P"}:
        img = img.convert("RGB")

    # Strategy: halve dimensions until both base64 fits AND pixel dimensions
    # are within limits, up to 4 rounds.
    # For JPEG, also try reducing quality at each size step.
    # For PNG, quality is irrelevant — only dimension reduction helps.
    quality_steps = (85, 70, 50) if pil_format == "JPEG" else (None,)
    prev_dims = (img.width, img.height)
    candidate = None  # will be set on first loop iteration

    def _dims_ok(w: int, h: int) -> bool:
        """True if both pixel dimensions are within the limit."""
        if max_dimension is None:
            return True
        return max(w, h) <= max_dimension

    for attempt in range(5):
        if attempt > 0:
            # Proportional scaling: halve the longer side and scale the
            # shorter side to preserve aspect ratio (min dimension 64).
            scale = 0.5
            new_w = max(int(img.width * scale), 64)
            new_h = max(int(img.height * scale), 64)
            # Re-derive the scale from whichever dimension hit the floor
            # so both axes shrink by the same factor.
            if new_w == 64 and img.width > 0:
                effective_scale = 64 / img.width
                new_h = max(int(img.height * effective_scale), 64)
            elif new_h == 64 and img.height > 0:
                effective_scale = 64 / img.height
                new_w = max(int(img.width * effective_scale), 64)
            # Stop if dimensions can't shrink further
            if (new_w, new_h) == prev_dims:
                break
            img = img.resize((new_w, new_h), Image.LANCZOS)
            prev_dims = (new_w, new_h)
            logger.info("Resized to %dx%d (attempt %d)", new_w, new_h, attempt)

        for q in quality_steps:
            buf = _io.BytesIO()
            save_kwargs = {"format": pil_format}
            if q is not None:
                save_kwargs["quality"] = q
            img.save(buf, **save_kwargs)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            candidate = f"data:{out_mime};base64,{encoded}"
            if len(candidate) <= max_base64_bytes and _dims_ok(img.width, img.height):
                logger.info("Auto-resized image fits: %.1f MB (quality=%s, %dx%d)",
                            len(candidate) / (1024 * 1024), q,
                            img.width, img.height)
                return candidate

    # If we still can't get it small enough, return the best attempt
    # and let the caller decide
    if candidate is not None:
        logger.warning("Auto-resize could not fit image under %.1f MB (best: %.1f MB)",
                       max_base64_bytes / (1024 * 1024), len(candidate) / (1024 * 1024))
        return candidate

    # Shouldn't reach here, but fall back to full encode
    return data_url or _image_to_base64_data_url(image_path, mime_type=mime_type)


# ---------------------------------------------------------------------------
# Native fast path: short-circuit the auxiliary LLM when the active main model
# supports native vision. Instead of asking a separate LLM to describe the
# image and returning text, we load the image, base64-encode it, and return a
# multimodal tool-result envelope. The agent loop unwraps the envelope into an
# OpenAI-style content list on the `tool` role; provider adapters (anthropic,
# codex_responses, chat_completions) translate that into Anthropic
# tool_result image blocks / Responses input_image / OpenAI image_url tool
# content. The main model then "sees" the pixels directly on its next turn.
# ---------------------------------------------------------------------------


def _supports_media_in_tool_results(provider: str, model: str) -> bool:
    """Whether the given provider+model combination accepts image content
    inside a tool-result message.

    Providers covered today (per spec docs verified Apr-2026):

      * Anthropic Messages API (``anthropic`` provider, plus aggregators that
        proxy Claude — ``openrouter``, ``nous``, ``vertex``, ``bedrock``):
        ``tool_result`` blocks accept ``image`` content blocks.
      * OpenAI Chat Completions: tool messages accept array content with
        ``image_url`` parts.
      * OpenAI Responses (``openai-codex``): ``function_call_output.output``
        accepts an array of ``input_text``/``input_image`` items.
      * Gemini 3 (and proxied via aggregators): supports multimodal tool
        results. Older Gemini does NOT.

    For unknown / legacy providers we conservatively return False — the
    caller falls back to the legacy aux-LLM text path.  The check is relaxed
    when the provider's ``ProviderProfile`` declares ``supports_vision=True``.
    """
    if not isinstance(provider, str):
        return False
    p = provider.strip().lower()
    if not p:
        return False

    # Aggregators that route to multiple vendors — assume support since
    # users on these aggregators are typically using vision-capable
    # frontier models. Falling back to text would be a regression for
    # them.
    _AGGREGATORS = {
        "openrouter", "nous", "vertex", "bedrock", "anthropic-vertex",
        "google-vertex",
    }
    if p in _AGGREGATORS:
        return True

    # Native Anthropic
    if p in {"anthropic", "claude", "anthropic-direct"}:
        return True

    # OpenAI Chat Completions and Responses
    if p in {"openai", "openai-chat", "openai-codex", "azure-openai"}:
        return True

    # Gemini — gate on model name; older Gemini variants did not support
    # multimodal functionResponse. Gemini 3.x does.
    if p in {"google", "gemini", "google-gemini", "google-vertex-gemini"}:
        if not isinstance(model, str):
            return False
        m = model.strip().lower()
        if "gemini-3" in m or "gemini-pro-3" in m or "gemini-flash-3" in m:
            return True
        return False

    # Check the provider's registered profile for the supports_vision flag.
    # This covers vision-capable providers like xiaomi, minimax, etc. that
    # aren't in the hardcoded list above.
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(p)
        if profile is not None and profile.supports_vision:
            return True
    except Exception:
        pass

    # Other vision-capable provider stacks. Conservative default: False.
    # Add explicit entries here as we verify each provider's tool-result
    # multimodal support empirically.
    return False


def _should_use_native_vision_fast_path() -> bool:
    """Whether vision tools should attach the image to the main model directly
    instead of routing through the auxiliary vision LLM.

    True when image routing resolves to ``native`` AND either the provider is
    known to accept images inside tool results, or the user explicitly declared
    the model vision-capable via the ``model.supports_vision`` config override.
    The override is the escape hatch for custom/local providers that aren't in
    the static allowlist. Best-effort: any resolution failure returns False so
    the caller falls back to the legacy aux-LLM path.
    """
    try:
        from agent.auxiliary_client import _read_main_provider, _read_main_model
        from agent.image_routing import decide_image_input_mode, _lookup_supports_vision
        from hermes_cli.config import load_config

        provider = _read_main_provider()
        model = _read_main_model()
        cfg = load_config()
        if decide_image_input_mode(provider, model, cfg) != "native":
            return False
        return (
            _supports_media_in_tool_results(provider, model)
            or _lookup_supports_vision(provider, model, cfg) is True
        )
    except Exception as exc:
        logger.debug("Native vision fast-path check failed: %s", exc)
        return False


def _build_native_vision_tool_result(
    image_url: str,
    question: str,
    image_data_url: str,
    image_size_bytes: int,
) -> Dict[str, Any]:
    """Build the multimodal tool-result envelope returned by the fast path.

    Shape:
      {
        "_multimodal": True,
        "content": [
          {"type": "text", "text": "<short note + the user's question>"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ],
        "text_summary": "<plain-text fallback>",
        "meta": {"image_url": ..., "size_bytes": N},
      }

    The text part exists for two reasons: (1) it gives the model an
    instruction to act on now that the pixels are in context, and
    (2) providers that don't support multimodal tool results can fall back
    to ``text_summary``.
    """
    # The tool-result text part is intentionally minimal. The model already
    # has the user's original question in context; this just acknowledges
    # the image is now visible and reminds it what it was asked.
    text_part = (
        "Image loaded into your context — you can see it natively now. "
        "Use your built-in vision to answer the user."
    )
    if isinstance(question, str) and question.strip():
        text_part += f"\n\nQuestion: {question.strip()}"

    summary = (
        f"Image attached natively for the main model "
        f"({image_size_bytes / 1024:.1f} KB). "
        "Answer using built-in vision."
    )

    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
        "text_summary": summary,
        "meta": {
            "image_url": image_url[:200],
            "size_bytes": image_size_bytes,
            "native_vision": True,
        },
    }


@contextlib.asynccontextmanager
async def _vision_concurrency_slot():
    """Deprecated no-op shim kept for backward compatibility.

    The fan-out cap was narrowed to the CPU-bound encode/resize burst only
    (see :data:`_vision_cpu_executor` / :func:`_run_encode_on_cpu_executor`).
    Holding a slot across the whole analysis serialized legitimate multi-image
    workflows behind the slow LLM call, which is exactly what we don't want.
    This context manager no longer gates anything; encode/resize is bounded
    where it actually runs. Retained only so any external caller importing it
    keeps working.
    """
    yield


async def _vision_analyze_native(
    image_url: str,
    question: str,
    task_id: Optional[str] = None,
) -> Any:
    """Fast path for vision-capable main models.

    Loads the image (data: / http(s) / file:// / local path / sandbox-container
    path) via the unified resolver, base64-encodes it, and returns a multimodal
    tool-result envelope. The agent loop unwraps it; provider adapters serialize
    it into the right tool-result-with-image shape for each backend.

    Returns:
        A ``_multimodal`` envelope dict on success.
        A JSON error string on failure (matches the existing tool-result
        contract so the agent loop displays errors normally).
    """
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url is required", success=False)

    temp_image_path: Optional[Path] = None
    should_cleanup = False
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Resolve the source to raw bytes through the single resolver (unifies
        # data:/http/file/local/container and enforces terminal-backend
        # confinement). Materialize to a temp file so the existing path-based
        # encode/resize/embed-cap pipeline below is reused verbatim.
        from tools.image_source import (
            ImageResolutionError,
            ResolveContext,
            resolve_image_source,
        )

        try:
            resolved = await resolve_image_source(image_url, ResolveContext(task_id=task_id))
        except ImageResolutionError as exc:
            return tool_error(str(exc), success=False)

        detected_mime_type = resolved.mime
        image_size_bytes = len(resolved.data)
        temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_image_path = temp_dir / f"temp_image_{uuid.uuid4()}.img"
        await asyncio.to_thread(temp_image_path.write_bytes, resolved.data)
        should_cleanup = True

        # Normalize unsupported formats (SVG, BMP, ...) to PNG BEFORE embedding.
        # Anthropic only accepts jpeg/png/gif/webp; an unsupported media_type
        # baked into immutable history wedges the session with a 400 on every
        # resume.  Convert here so it can never enter history. Offloaded — the
        # rasterizers/Pillow are blocking.
        normalized_path, detected_mime_type, _norm_err = await asyncio.to_thread(
            _normalize_to_supported_image, temp_image_path, detected_mime_type,
        )
        if _norm_err or normalized_path is None:
            return tool_error(
                _norm_err or "Image normalization failed.", success=False,
            )
        if normalized_path != temp_image_path:
            # We created a temp PNG — swap to it and ensure it's cleaned up.
            if should_cleanup and temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except Exception:
                    pass
            temp_image_path = normalized_path
            should_cleanup = True
            image_size_bytes = temp_image_path.stat().st_size

        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url,
            temp_image_path, mime_type=detected_mime_type,
        )

        # Proactive embed cap: this image gets baked into conversation
        # history and re-sent on every subsequent turn.  Anthropic rejects
        # any single base64 image over 5 MB OR over 8000px per side with a
        # 400, and because history is immutable, an oversized embed
        # permanently wedges the session — retries can't clear bytes (or
        # pixels) that are already in the request.  Resize DOWN to the embed
        # target (4 MB / 7900px, headroom under both ceilings) whenever the
        # payload exceeds either limit, not just at the 20 MB hard ceiling.
        _over_bytes = len(image_data_url) > _EMBED_TARGET_BYTES
        _over_dims = await _run_encode_on_cpu_executor(
            _image_exceeds_dimension, temp_image_path, _EMBED_MAX_DIMENSION,
        )
        if _over_bytes or _over_dims:
            image_data_url = await _run_encode_on_cpu_executor(
                _resize_image_for_vision,
                temp_image_path, mime_type=detected_mime_type,
                max_base64_bytes=_EMBED_TARGET_BYTES,
                max_dimension=_EMBED_MAX_DIMENSION,
            )
            # If even resizing can't get under the absolute hard ceiling,
            # there's nothing more we can do — reject rather than embed a
            # session-wedging payload.
            if len(image_data_url) > _MAX_BASE64_BYTES:
                return tool_error(
                    f"Image too large for vision API: base64 payload is "
                    f"{len(image_data_url) / (1024 * 1024):.1f} MB "
                    f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) "
                    f"even after resizing. Install Pillow "
                    f"(`pip install Pillow`) for better auto-resize, "
                    f"or compress the image manually.",
                    success=False,
                )

        return _build_native_vision_tool_result(
            image_url=image_url,
            question=question,
            image_data_url=image_data_url,
            image_size_bytes=image_size_bytes,
        )

    except Exception as exc:
        logger.warning("Native vision fast path failed: %s", exc)
        return tool_error(f"Native vision failed: {exc}", success=False)
    finally:
        # Only delete temp files we created — never user-provided paths.
        if should_cleanup and temp_image_path is not None:
            try:
                if temp_image_path.exists():
                    temp_image_path.unlink()
            except Exception:
                pass


async def vision_analyze_tool(
    image_url: str,
    user_prompt: str,
    model: str = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Analyze an image from a URL or local file path using vision AI.
    
    This tool accepts either an HTTP/HTTPS URL or a local file path. For URLs,
    it downloads the image first. In both cases, the image is converted to
    base64 (downscaled to the configured send cap when oversized) and processed
    by the configured auxiliary vision route (``auxiliary.vision``).
    
    The user_prompt parameter is expected to be pre-formatted by the calling
    function (typically model_tools.py) to include both full description
    requests and specific questions.
    
    Args:
        image_url (str): The URL or local file path of the image to analyze.
                         Accepts http://, https:// URLs or absolute/relative file paths.
        user_prompt (str): The pre-formatted prompt for the vision model
        model (str): The vision model to use (default: the configured
                     auxiliary vision model)
    
    Returns:
        str: JSON string containing the analysis results with the following structure:
             {
                 "success": bool,
                 "analysis": str (defaults to error message if None)
             }
    
    Raises:
        Exception: If download fails, analysis fails, or API key is not set
        
    Note:
        - For URLs, temporary images are stored under $HERMES_HOME/cache/vision/ and cleaned up
        - For local file paths, the file is used directly and NOT deleted
        - Supports common image formats (JPEG, PNG, GIF, WebP, etc.)
    """
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data = {
        "parameters": {
            "image_url": image_url,
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "model": model
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model,
        "image_size_bytes": 0
    }
    
    temp_image_path = None
    # Track whether we should clean up the file after processing.
    # Local files (e.g. from the image cache) should NOT be deleted.
    should_cleanup = True
    detected_mime_type = None
    # The per-attempt budget actually used for the LLM call (full or probe),
    # so the timeout error text reports the budget that really applied.
    effective_vision_timeout = None
    probe_budget_active = False

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Analyzing image: %s", image_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])

        # Resolve the source to raw bytes through the single resolver (unifies
        # data:/http/file/local/container and enforces terminal-backend
        # confinement). Materialize to a temp file so the existing path-based
        # encode/resize pipeline below is reused verbatim.
        from tools.image_source import (
            ImageResolutionError,
            ResolveContext,
            resolve_image_source,
        )

        try:
            resolved = await resolve_image_source(image_url, ResolveContext(task_id=task_id))
        except ImageResolutionError as exc:
            raise ValueError(str(exc))

        detected_mime_type = resolved.mime
        temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_image_path = temp_dir / f"temp_image_{uuid.uuid4()}.img"
        await asyncio.to_thread(temp_image_path.write_bytes, resolved.data)
        should_cleanup = True

        # Get image file size for logging
        image_size_bytes = len(resolved.data)
        image_size_kb = image_size_bytes / 1024
        logger.info("Image ready (%.1f KB)", image_size_kb)
        # Normalize unsupported formats (SVG, BMP, ...) to PNG. Vision providers
        # reject these media types; convert before encoding. Offloaded — the
        # rasterizers/Pillow are blocking.
        normalized_path, detected_mime_type, _norm_err = await asyncio.to_thread(
            _normalize_to_supported_image, temp_image_path, detected_mime_type,
        )
        if _norm_err or normalized_path is None:
            raise ValueError(_norm_err or "Image normalization failed.")
        if normalized_path != temp_image_path:
            if should_cleanup and temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except Exception:
                    pass
            temp_image_path = normalized_path
            should_cleanup = True

        # Convert image to base64.  Offloaded to the bounded vision CPU
        # executor so a fan-out of encodes can't saturate every core and
        # starve the event loop.
        logger.info("Converting image to base64...")
        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url, temp_image_path, mime_type=detected_mime_type)
        data_size_kb = len(image_data_url) / 1024
        logger.info("Image converted to base64 (%.1f KB)", data_size_kb)

        # Preflight send cap (NOL-253).  This path used to send full
        # resolution first and downscale only after a provider SIZE
        # rejection — but the failure mode that actually burns run budget is
        # not a 413, it is a TIMEOUT: montage-built contact sheets shipped
        # at full resolution made 6 of 15 QC calls burn their whole
        # per-attempt budget in the NOL-151 run.  Cloud vision routes
        # downscale internally to ~2048px anyway, so oversized sends buy no
        # fidelity — only upload bytes and tiling latency.  Downscale BEFORE
        # the first call whenever the payload exceeds the embed byte target
        # or the send dimension cap.
        _send_max_dim = _resolve_vision_max_dimension()
        _over_bytes = len(image_data_url) > _EMBED_TARGET_BYTES
        _over_dims = False
        if _send_max_dim > 0:
            _over_dims = await _run_encode_on_cpu_executor(
                _image_exceeds_dimension, temp_image_path, _send_max_dim,
            )
        if _over_bytes or _over_dims:
            image_data_url = await _run_encode_on_cpu_executor(
                _resize_image_for_vision,
                temp_image_path, mime_type=detected_mime_type,
                max_base64_bytes=_EMBED_TARGET_BYTES,
                max_dimension=_send_max_dim if _send_max_dim > 0 else None,
            )
            logger.info(
                "Preflight-resized vision payload to %.1f KB "
                "(max_dimension=%s)",
                len(image_data_url) / 1024,
                _send_max_dim if _send_max_dim > 0 else "off",
            )

        # Hard limit (20 MB) — no provider accepts payloads this large.
        # Reached only when Pillow is unavailable or resizing failed.
        if len(image_data_url) > _MAX_BASE64_BYTES:
            raise ValueError(
                f"Image too large for vision API: base64 payload is "
                f"{len(image_data_url) / (1024 * 1024):.1f} MB "
                f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) "
                f"even after resizing. "
                f"Install Pillow (`pip install Pillow`) for better auto-resize, "
                f"or compress the image manually."
            )

        debug_call_data["image_size_bytes"] = image_size_bytes
        
        # Use the prompt as provided (model_tools.py now handles full description formatting)
        comprehensive_prompt = user_prompt
        
        # Prepare the message with base64-encoded image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comprehensive_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
        
        logger.info("Processing image with vision model...")
        
        # Call the vision API via centralized router.
        vision_timeout = _resolve_vision_timeout()
        # Degraded mode (NOL-253): once the consecutive-timeout guard has
        # ordered the strategy switch, spend probe-seconds discovering
        # recovery, not full budgets.  Any success resets the streak and
        # restores the full budget on the next call.
        _streak_at_call = _current_vision_timeout_streak()
        if _streak_at_call >= _VISION_TIMEOUT_SWITCH_AT:
            _probe_timeout = _resolve_vision_probe_timeout()
            if 0 < _probe_timeout < vision_timeout:
                logger.info(
                    "Vision degraded mode: %d consecutive timeouts — "
                    "capping this attempt at %.0fs (probe budget, "
                    "full budget %.0fs)",
                    _streak_at_call, _probe_timeout, vision_timeout,
                )
                vision_timeout = _probe_timeout
                probe_budget_active = True
        effective_vision_timeout = vision_timeout
        vision_temperature = 0.1
        try:
            from hermes_cli.config import cfg_get, load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass
        call_kwargs = {
            "task": "vision",
            "messages": messages,
            "temperature": vision_temperature,
            "max_tokens": 2000,
            "timeout": vision_timeout,
        }
        if model:
            call_kwargs["model"] = model
        _load_auxiliary_client()
        # Try full-size image first; on size-related rejection, downscale and retry.
        try:
            response = await async_call_llm(**call_kwargs)
        except Exception as _api_err:
            if (_is_image_size_error(_api_err)
                    and len(image_data_url) > _RESIZE_TARGET_BYTES):
                logger.info(
                    "API rejected image (%.1f MB, likely too large); "
                    "auto-resizing to ~%.0f MB and retrying...",
                    len(image_data_url) / (1024 * 1024),
                    _RESIZE_TARGET_BYTES / (1024 * 1024),
                )
                image_data_url = await _run_encode_on_cpu_executor(
                    _resize_image_for_vision,
                    temp_image_path, mime_type=detected_mime_type)
                messages[0]["content"][1]["image_url"]["url"] = image_data_url
                response = await async_call_llm(**call_kwargs)
            else:
                raise
        
        # Extract the analysis — fall back to reasoning if content is empty
        analysis = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response) — but never
        # with a second full budget (NOL-253): the first attempt already
        # proved the route responsive (it returned, just empty), so the
        # retry either succeeds quickly or is not worth another full burn.
        if not analysis:
            _retry_timeout = call_kwargs["timeout"]
            _probe_timeout = _resolve_vision_probe_timeout()
            if 0 < _probe_timeout < _retry_timeout:
                _retry_timeout = _probe_timeout
            logger.warning(
                "Vision LLM returned empty content, retrying once "
                "(timeout %.0fs)", _retry_timeout)
            retry_kwargs = {**call_kwargs, "timeout": _retry_timeout}
            response = await async_call_llm(**retry_kwargs)
            analysis = extract_content_or_reasoning(response)

        analysis_length = len(analysis)
        
        logger.info("Image analysis completed (%s characters)", analysis_length)
        
        # Prepare successful response
        result = {
            "success": True,
            "analysis": analysis or "There was a problem with the request and the image could not be analyzed."
        }
        
        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length

        # A successful call proves the route is serving requests again —
        # the consecutive-timeout streak resets (NOL-197).
        _reset_vision_timeout_streak()

        # Log debug information
        _debug.log_call("vision_analyze_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        # Detect vision capability errors — give the model a clear message
        # so it can inform the user instead of a cryptic API error.
        err_str = str(e).lower()
        if _is_vision_timeout(e):
            # Streak-aware timeout guidance (NOL-197): the second
            # consecutive full-budget burn orders the strategy switch
            # instead of letting the model resize and resubmit a third
            # resolution of an image the route cannot analyze in budget.
            # Reports the budget that actually applied to this attempt —
            # the reduced probe budget when degraded mode was active
            # (NOL-253), the full budget otherwise.
            streak = _record_vision_timeout()
            analysis = _vision_timeout_analysis(
                e, streak,
                effective_vision_timeout
                if effective_vision_timeout is not None
                else _resolve_vision_timeout(),
                probe_active=probe_budget_active)
        elif any(hint in err_str for hint in (
            "402", "insufficient", "payment required", "credits", "billing",
        )):
            analysis = (
                "Insufficient credits or payment required. Please top up your "
                f"API provider account and try again. Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "does not support", "not support image",
            "content_policy", "multimodal",
            "unrecognized request argument", "image input",
        )):
            analysis = (
                f"{model} does not support vision or our request was not "
                f"accepted by the server. Error: {e}"
            )
        elif "invalid_request" in err_str or "image_url" in err_str:
            analysis = (
                "The vision API rejected the image. This can happen when the "
                "image is in an unsupported format, corrupted, or still too "
                "large after auto-resize. Try a smaller JPEG/PNG and retry. "
                f"Error: {e}"
            )
        else:
            analysis = (
                "There was a problem with the request and the image could not "
                f"be analyzed. Error: {e}"
            )
        
        # Prepare error response
        result = {
            "success": False,
            "error": error_msg,
            "analysis": analysis,
        }
        
        debug_call_data["error"] = error_msg
        _debug.log_call("vision_analyze_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    finally:
        # Clean up temporary image file (but NOT local/cached files)
        if should_cleanup and temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
                logger.debug("Cleaned up temporary image file")
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete temporary file: %s", cleanup_error, exc_info=True
                )


def check_vision_requirements() -> bool:
    """Check if the configured runtime vision path can resolve a client.

    Mirrors the fallback chain that ``call_llm(task="vision")`` actually uses
    at runtime: first the explicit ``auxiliary.vision.provider`` (if any),
    and if that fails, the auto chain (main provider → openrouter → nous).
    Without the auto-fallback step the tool would disappear from the model's
    tool list whenever the explicit provider name was unresolvable, even
    when the auto chain would have served the request (issue #31179).
    """
    try:
        from agent.auxiliary_client import resolve_vision_provider_client
    except ImportError:
        return False
    try:
        _provider, client, _model = resolve_vision_provider_client()
        if client is not None:
            return True
        # Same fallback to "auto" that call_llm performs when the configured
        # provider can't be resolved.
        _provider, client, _model = resolve_vision_provider_client(provider="auto")
        return client is not None
    except Exception:
        return False



if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("👁️ Vision Tools Module")
    print("=" * 40)
    
    # Check if vision model is available
    api_available = check_vision_requirements()
    
    if not api_available:
        print("❌ No auxiliary vision model available")
        print("Configure a supported multimodal backend (OpenRouter, Nous, Codex, Anthropic, or a custom OpenAI-compatible endpoint).")
        sys.exit(1)
    else:
        print("✅ Vision model available")
    
    print("🛠️ Vision tools ready for use!")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/vision_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set VISION_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from vision_tools import vision_analyze_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      result = await vision_analyze_tool(")
    print("          image_url='https://example.com/image.jpg',")
    print("          user_prompt='What do you see in this image?'")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nExample prompts:")
    print("  - 'What architectural style is this building?'")
    print("  - 'Describe the emotions and mood in this image'")
    print("  - 'What text can you read in this image?'")
    print("  - 'Identify any safety hazards visible'")
    print("  - 'What products or brands are shown?'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export VISION_TOOLS_DEBUG=true")
    print("  # Debug logs capture all vision analysis calls and results")
    print("  # Logs saved to: ./logs/vision_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

VISION_ANALYZE_SCHEMA = {
    "name": "vision_analyze",
    "description": (
        "Load an image into the conversation so you can see it. Accepts a "
        "URL, local file path, or data URL. When your active model has "
        "native vision, the image is attached to your context directly "
        "and you read the pixels yourself on the next turn — call this "
        "any time the user references an image (filepath in their message, "
        "URL in tool output, screenshot from the browser, etc.). For "
        "non-vision models, falls back to an auxiliary vision model that "
        "returns a text description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Image URL (http/https), local file path, or data: URL to load."
            },
            "question": {
                "type": "string",
                "description": "Your specific question or request about the image. Optional context the model uses on the next turn after seeing the image."
            }
        },
        "required": ["image_url", "question"]
    }
}


async def _handle_vision_analyze(args: Dict[str, Any], **kw: Any) -> str:
    image_url = args.get("image_url", "")
    question = args.get("question", "")
    task_id = kw.get("task_id")

    # The fan-out cap lives inside the encode/resize step (offloaded to the
    # bounded _vision_cpu_executor), NOT around the whole analysis — so a
    # legitimate multi-image workflow keeps full request concurrency while the
    # CPU bursts that actually starve the loop are bounded to host cores.
    #
    # Fast path: when native image routing is in effect for the active main
    # model (provider accepts images in tool results, or the user set the
    # model.supports_vision override), short-circuit the auxiliary LLM and
    # return the image bytes as a multimodal tool-result envelope. The main
    # model sees the pixels directly on its next turn — no aux call, no
    # information loss, no extra latency.
    if _should_use_native_vision_fast_path():
        logger.info("vision_analyze: native fast path")
        return await _vision_analyze_native(image_url, question, task_id=task_id)

    # Legacy path: aux LLM describes the image and we return its text.
    full_prompt = (
        "Fully describe and explain everything about this image, then answer the "
        f"following question:\n\n{question}"
    )
    # Prefer config.yaml auxiliary.vision.model; env var is a legacy override.
    model = None
    try:
        from hermes_cli.config import cfg_get, load_config
        _cfg = load_config()
        _vmodel = cfg_get(_cfg, "auxiliary", "vision", "model")
        if _vmodel:
            model = str(_vmodel).strip() or None
    except Exception:
        pass
    if not model:
        model = os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    return await vision_analyze_tool(image_url, full_prompt, model, task_id=task_id)


registry.register(
    name="vision_analyze",
    toolset="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=_handle_vision_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="👁️",
)


# ---------------------------------------------------------------------------
# Video Analysis Tool
# ---------------------------------------------------------------------------

# Extension → MIME. avi/mkv fall back to mp4.
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/mov",
    ".avi": "video/mp4",
    ".mkv": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}

_MAX_VIDEO_BASE64_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
_VIDEO_SIZE_WARN_BYTES = 20 * 1024 * 1024

# Gemini's API rejects requests whose body exceeds ~20 MB, and a base64 data
# URL inside a ``file`` part travels inline in that body — so a local file
# sent to Gemini is bounded well below the 50 MB cap above. 19 MB leaves
# headroom for the prompt and JSON framing, which permits roughly 14 MB of
# source video after base64's 4/3 expansion.
_GEMINI_INLINE_MAX_BASE64_BYTES = 19 * 1024 * 1024


def _gemini_inline_source_cap_mb() -> float:
    """Source-video MB that fit under ``_GEMINI_INLINE_MAX_BASE64_BYTES``.

    Read at call time (not import time) so the figure in user-facing errors
    tracks the cap wherever it is tuned or patched.
    """
    return (_GEMINI_INLINE_MAX_BASE64_BYTES // 4 * 3) / (1024 * 1024)

# Gemini samples ~2.8 fps when ``video_metadata.fps`` is omitted (measured
# through the proxy: 91 video-tokens/s). ``fps: 1`` measured 32 tokens/s — a
# reproducible ~2.8x saving with no visible loss for "what happens in this
# clip" questions. Fractional fps is NOT a reliable lever (0.5 cost more
# than 1; 0.2 differed between identical runs).
_GEMINI_VIDEO_DEFAULT_FPS = 1

_VIDEO_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "video/*,*/*;q=0.8",
}


def _detect_video_mime_type(video_path: Path) -> Optional[str]:
    """Return a video MIME type based on file extension, or None if unsupported."""
    ext = video_path.suffix.lower()
    return _VIDEO_MIME_TYPES.get(ext)


def _video_to_base64_data_url(video_path: Path, mime_type: Optional[str] = None) -> str:
    """Convert a video file to a base64-encoded data URL."""
    data = video_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = mime_type or _VIDEO_MIME_TYPES.get(video_path.suffix.lower(), "video/mp4")
    return f"data:{mime};base64,{encoded}"


def _is_gemini_model(model: Optional[str]) -> bool:
    """True when ``model`` names a Gemini model under any provider prefix.

    Matches ``gemini-3.8-flash``, ``google/gemini-2.5-flash``,
    ``gemini/gemini-3-flash``, ``vertex_ai/gemini-3.8-flash``, ... (case-
    insensitive). ``None``/empty means "provider default" and keeps the
    legacy wire shape.
    """
    if not model or not isinstance(model, str):
        return False
    return "gemini" in model.lower()


def _normalize_video_fps(value: Any) -> Optional[float]:
    """Coerce a configured ``fps`` into a positive number.

    ``None``, ``0``, negatives, NaN and unparsable junk all mean "omit
    ``video_metadata.fps`` and let Gemini sample at its own rate". Integral
    values come back as ``int`` so the wire carries ``{"fps": 1}``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if fps != fps or fps <= 0:  # NaN or non-positive
        return None
    return int(fps) if fps.is_integer() else fps


def _video_part_for_model(
    model: Optional[str],
    file_data: str,
    mime: str,
    fps: Optional[float] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the chat-completions content part that carries the video.

    Gemini (LiteLLM's ``gemini/`` and ``vertex_ai/`` routes) only attaches
    media that arrives as an OpenAI ``file`` part: ``file_data`` is the video
    (an https URL Google fetches server-side, or a ``data:`` URL that is
    inlined), ``format`` its MIME type, ``detail`` maps to the per-part
    ``media_resolution`` on Gemini 3+, and ``video_metadata.fps`` sets the
    frame-sampling rate. A ``video_url`` part is *silently dropped* by that
    transformation — the request succeeds with no video attached and the
    model answers from the text prompt alone.

    Every other provider (vLLM, OpenRouter, Qwen-VL, ...) keeps the
    ``video_url`` shape it has always received.
    """
    if not _is_gemini_model(model):
        return {"type": "video_url", "video_url": {"url": file_data}}
    file_part: Dict[str, Any] = {"file_data": file_data, "format": mime}
    if detail:
        file_part["detail"] = str(detail).strip().lower()
    fps_value = _normalize_video_fps(fps)
    if fps_value is not None:
        file_part["video_metadata"] = {"fps": fps_value}
    return {"type": "file", "file": file_part}


def _gemini_rejects_sampling_overrides(model: Optional[str]) -> bool:
    """True for gemini-3.8-flash and later Flash generations (any prefix).

    Those models reject ``temperature``/``top_p`` outright and the proxy
    strips them, so the tool does not put them on the wire at all. Every
    other model keeps the configured temperature.
    """
    if not _is_gemini_model(model):
        return False
    try:
        from agent.gemini_native_adapter import is_gemini_flash_38_or_later
    except Exception:
        return False
    # Strip any ``<provider>/`` prefix (openrouter/google/, vertex_ai/, ...):
    # the adapter helper only knows the google/ and gemini/ spellings.
    return is_gemini_flash_38_or_later(model.rsplit("/", 1)[-1])


def _video_mime_from_url(video_url: str) -> Optional[str]:
    """MIME type from the URL path's extension, or None when unknown."""
    try:
        path = urlparse(video_url).path or ""
    except ValueError:
        return None
    return _VIDEO_MIME_TYPES.get(Path(path).suffix.lower())


async def _probe_remote_video_mime(video_url: str, timeout: float = 15.0) -> str:
    """Best-effort MIME type for a remote video that Google will fetch itself.

    A HEAD request validates every redirect/final target against both SSRF and
    website policies. Its ``video/*`` Content-Type is used when available,
    otherwise the URL extension wins. Falls back to ``video/mp4``: Gemini
    requires *some* MIME type on a ``fileData`` part and treats mp4 as the
    generic container — it is also what works for YouTube watch URLs, which
    report ``text/html``.
    """
    mime = _video_mime_from_url(video_url)
    try:
        from tools.url_safety import (
            async_is_safe_url,
            create_ssrf_safe_async_client,
            redirect_target_from_response,
        )

        async def _redirect_guard(response):
            targets = [str(response.url)]
            redirect_url = redirect_target_from_response(response)
            if redirect_url:
                targets.append(redirect_url)
            for target in targets:
                if not await async_is_safe_url(target):
                    raise PermissionError(
                        f"Blocked redirect to private/internal address: {target}"
                    )
                blocked = check_website_access(target)
                if blocked:
                    raise PermissionError(blocked["message"])

        async with create_ssrf_safe_async_client(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [_redirect_guard]},
        ) as client:
            response = await client.head(video_url, headers=dict(_VIDEO_FETCH_HEADERS))
        content_type = (
            (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        )
        if content_type.startswith("video/"):
            return content_type
    except PermissionError:
        raise
    except Exception as e:  # MIME detection is advisory; security checks are not
        logger.debug("Video MIME probe failed for %s: %s", video_url[:80], e)
    return mime or "video/mp4"


async def _download_video(video_url: str, destination: Path, max_retries: int = 3) -> Path:
    """Download video from URL with SSRF protection and retry."""
    import asyncio

    destination.parent.mkdir(parents=True, exist_ok=True)

    async def _ssrf_redirect_guard(response):
        from tools.url_safety import async_is_safe_url, redirect_target_from_response
        redirect_url = redirect_target_from_response(response)
        if redirect_url and not await async_is_safe_url(redirect_url):
            raise ValueError(
                f"Blocked redirect to private/internal address: {redirect_url}"
            )

    last_error = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])

            from tools.url_safety import create_ssrf_safe_async_client

            async with create_ssrf_safe_async_client(
                timeout=60.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                response = await client.get(video_url, headers=dict(_VIDEO_FETCH_HEADERS))
                response.raise_for_status()

                cl = response.headers.get("content-length")
                if cl and int(cl) > _MAX_VIDEO_BASE64_BYTES:
                    raise ValueError(
                        f"Video too large ({int(cl)} bytes, max {_MAX_VIDEO_BASE64_BYTES})"
                    )

                final_url = str(response.url)
                blocked = check_website_access(final_url)
                if blocked:
                    raise PermissionError(blocked["message"])

                body = response.content
                if len(body) > _MAX_VIDEO_BASE64_BYTES:
                    raise ValueError(
                        f"Video too large ({len(body)} bytes, max {_MAX_VIDEO_BASE64_BYTES})"
                    )
                destination.write_bytes(body)

            return destination
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.warning("Video download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Video download failed after %s attempts: %s",
                    max_retries, str(e)[:100], exc_info=True,
                )

    if last_error is None:
        raise RuntimeError(
            f"_download_video exited retry loop without attempting (max_retries={max_retries})"
        )
    raise last_error


async def video_analyze_tool(
    video_url: str,
    user_prompt: str,
    model: str = None,
    fps: Optional[float] = _GEMINI_VIDEO_DEFAULT_FPS,
    detail: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Analyze a video via multimodal LLM. Returns JSON {success, analysis}.

    ``fps`` / ``detail`` only affect Gemini models (see
    ``_video_part_for_model``): ``fps`` is sent as ``video_metadata.fps``
    (``None``/``0`` omits it so Gemini samples at its own rate) and ``detail``
    as the per-part ``media_resolution``. ``provider`` is an explicit
    auxiliary provider override (``auxiliary.video.provider``).
    """
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    gemini = _is_gemini_model(model)
    debug_call_data = {
        "parameters": {
            "video_url": video_url,
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "model": model,
            "fps": fps,
            "detail": detail,
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model,
        "video_size_bytes": 0,
        "video_part_type": "file" if gemini else "video_url",
        "video_passthrough": False,
    }

    temp_video_path = None
    should_cleanup = True

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Analyzing video: %s", video_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])

        # Resolve local path vs remote URL
        resolved_url = video_url
        if resolved_url.startswith("file://"):
            resolved_url = resolved_url[len("file://"):]
        local_path = Path(os.path.expanduser(resolved_url))

        video_file_data: Optional[str] = None  # what goes on the wire
        detected_mime: Optional[str] = None
        video_size_bytes = 0

        if local_path.is_file():
            from agent.file_safety import raise_if_read_blocked
            raise_if_read_blocked(str(local_path))
            logger.info("Using local video file: %s", video_url)
            temp_video_path = local_path
            should_cleanup = False
        elif await _validate_image_url_async(video_url):
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])
            if gemini and video_url.lower().startswith("https://"):
                # Gemini fetches an https ``file_data`` URL server-side, so
                # there is nothing to download, base64-encode or size-cap
                # here. The SSRF and website-policy checks above still gate
                # the URL exactly as they gate a download.
                detected_mime = await _probe_remote_video_mime(video_url)
                video_file_data = video_url
                debug_call_data["video_passthrough"] = True
                logger.info("Passing video URL through for Gemini to fetch (%s)", detected_mime)
            else:
                temp_dir = get_hermes_dir("cache/video", "temp_video_files")
                temp_video_path = temp_dir / f"temp_video_{uuid.uuid4()}.mp4"
                await _download_video(video_url, temp_video_path)
                should_cleanup = True
        else:
            raise ValueError(
                "Invalid video source. Provide an HTTP/HTTPS URL or a valid local file path."
            )

        if video_file_data is None:
            video_size_bytes = temp_video_path.stat().st_size
            video_size_mb = video_size_bytes / (1024 * 1024)
            logger.info("Video ready (%.1f MB)", video_size_mb)

            detected_mime = _detect_video_mime_type(temp_video_path)
            if not detected_mime:
                raise ValueError(
                    f"Unsupported video format: '{temp_video_path.suffix}'. "
                    f"Supported: {', '.join(sorted(_VIDEO_MIME_TYPES.keys()))}"
                )

            if video_size_bytes > _VIDEO_SIZE_WARN_BYTES:
                logger.warning("Video is %.1f MB — may be slow or rejected", video_size_mb)

            video_data_url = _video_to_base64_data_url(temp_video_path, mime_type=detected_mime)
            data_size_mb = len(video_data_url) / (1024 * 1024)

            if gemini and len(video_data_url) > _GEMINI_INLINE_MAX_BASE64_BYTES:
                raise ValueError(
                    f"Video too large to inline for Gemini: base64 payload is "
                    f"{data_size_mb:.1f} MB (limit "
                    f"{_GEMINI_INLINE_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                    f"Gemini accepts roughly {_gemini_inline_source_cap_mb():.0f} MB "
                    "of source video after base64 expansion; pass an https URL to let "
                    "Google fetch it, or compress/trim the video and retry."
                )
            if len(video_data_url) > _MAX_VIDEO_BASE64_BYTES:
                raise ValueError(
                    f"Video too large for API: base64 payload is {data_size_mb:.1f} MB "
                    f"(limit {_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                    f"Compress or trim the video and retry."
                )
            video_file_data = video_data_url

        debug_call_data["video_size_bytes"] = video_size_bytes

        fps_value = _normalize_video_fps(fps)
        if gemini and fps_value is not None and fps_value < 1:
            logger.warning(
                "auxiliary.video.fps=%s is fractional; measured non-reproducible "
                "through Gemini — use 1 for a predictable saving",
                fps_value,
            )
        video_part = _video_part_for_model(
            model, video_file_data, detected_mime, fps=fps_value, detail=detail,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    video_part,
                ],
            }
        ]

        vision_timeout = _resolve_vision_timeout(default=180.0, floor=180.0)
        vision_temperature = 0.1
        try:
            from hermes_cli.config import cfg_get, load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass

        call_kwargs = {
            "task": "vision",
            "messages": messages,
            "max_tokens": 4000,
            "timeout": vision_timeout,
        }
        if _gemini_rejects_sampling_overrides(model):
            # gemini-3.8-flash+ rejects temperature/top_p and the proxy strips
            # them anyway — keep them off the wire entirely.
            logger.debug("%s rejects sampling overrides; not sending temperature", model)
        else:
            call_kwargs["temperature"] = vision_temperature
        if model:
            call_kwargs["model"] = model
        if provider:
            call_kwargs["provider"] = provider

        _load_auxiliary_client()
        response = await async_call_llm(**call_kwargs)
        analysis = extract_content_or_reasoning(response)

        if not analysis:
            logger.warning("Empty video response, retrying once")
            response = await async_call_llm(**call_kwargs)
            analysis = extract_content_or_reasoning(response)

        analysis_length = len(analysis) if analysis else 0
        logger.info("Video analysis completed (%s characters)", analysis_length)

        result = {
            "success": True,
            "analysis": analysis or "There was a problem with the request and the video could not be analyzed.",
        }

        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length
        _debug.log_call("video_analyze_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Error analyzing video: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        err_str = str(e).lower()
        if any(hint in err_str for hint in (
            "402", "insufficient", "payment required", "credits", "billing",
        )):
            analysis = (
                "Insufficient credits or payment required. Please top up your "
                f"API provider account and try again. Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "does not support", "not support video",
            "content_policy", "multimodal",
            "unrecognized request argument", "video input",
            "video_url",
        )):
            analysis = (
                f"The model does not support video analysis or the request was "
                f"rejected. Ensure you're using a video-capable model "
                f"(e.g. gemini-3.8-flash, set via auxiliary.video.model). Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "too large", "payload", "413", "content_too_large",
            "request_too_large", "exceeds", "size limit",
        )):
            analysis = (
                "The video is too large for the API. Try compressing or trimming "
                f"the video (max ~50 MB; Gemini accepts roughly {_gemini_inline_source_cap_mb():.0f} MB "
                f"of local source video after base64 expansion, or an https URL). Error: {e}"
            )
        else:
            analysis = (
                "There was a problem with the request and the video could not "
                f"be analyzed. Error: {e}"
            )

        result = {
            "success": False,
            "error": error_msg,
            "analysis": analysis,
        }

        debug_call_data["error"] = error_msg
        _debug.log_call("video_analyze_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    finally:
        if should_cleanup and temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
                logger.debug("Cleaned up temporary video file")
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete temporary file: %s", cleanup_error, exc_info=True
                )


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video from a URL or local file path using a multimodal AI model. "
        "Sends the video to a video-capable model (e.g. Gemini) for understanding. "
        "Use this for video files — for images, use vision_analyze instead. "
        "Supports mp4, webm, mov, avi, mkv, mpeg formats. "
        "Note: large videos (>20 MB) may be slow; max ~50 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "Video URL (http/https) or local file path to analyze.",
            },
            "question": {
                "type": "string",
                "description": "Your specific question about the video. The AI will describe what happens in the video and answer your question.",
            },
        },
        "required": ["video_url", "question"],
    },
}


def _resolve_video_settings() -> Dict[str, Any]:
    """Resolve ``video_analyze`` settings from config.yaml ``auxiliary.video``.

    ``model`` falls back to ``auxiliary.vision.model`` and then to the legacy
    ``AUXILIARY_VIDEO_MODEL`` / ``AUXILIARY_VISION_MODEL`` env vars.
    ``provider`` is honored only when set on ``auxiliary.video`` (otherwise
    the vision task's provider applies). ``fps`` defaults to
    ``_GEMINI_VIDEO_DEFAULT_FPS``; an explicit ``0``/``null`` omits it.
    ``detail`` passes through when set. ``fps``/``detail`` only matter for
    Gemini models.
    """
    model = None
    provider = None
    fps: Any = _GEMINI_VIDEO_DEFAULT_FPS
    detail = None
    try:
        from hermes_cli.config import cfg_get, load_config
        _cfg = load_config()
        _vmodel = cfg_get(_cfg, "auxiliary", "video", "model") or cfg_get(_cfg, "auxiliary", "vision", "model")
        if _vmodel:
            model = str(_vmodel).strip() or None
        _vprovider = cfg_get(_cfg, "auxiliary", "video", "provider")
        if _vprovider:
            provider = str(_vprovider).strip() or None
        fps = cfg_get(_cfg, "auxiliary", "video", "fps", default=_GEMINI_VIDEO_DEFAULT_FPS)
        _detail = cfg_get(_cfg, "auxiliary", "video", "detail")
        if _detail:
            detail = str(_detail).strip().lower() or None
    except Exception:
        pass
    if not model:
        model = os.getenv("AUXILIARY_VIDEO_MODEL", "").strip() or os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    if provider and provider.lower() == "auto":
        provider = None
    return {
        "model": model,
        "provider": provider,
        "fps": _normalize_video_fps(fps),
        "detail": detail,
    }


def _handle_video_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    video_url = args.get("video_url", "")
    question = args.get("question", "")
    full_prompt = (
        "Fully describe and explain everything happening in this video, "
        "including visual content, motion, audio cues, text overlays, and scene "
        f"transitions. Then answer the following question:\n\n{question}"
    )
    # Prefer config.yaml auxiliary.video.* (model falling back to vision);
    # env vars are a legacy override for the model.
    settings = _resolve_video_settings()
    return video_analyze_tool(
        video_url,
        full_prompt,
        settings["model"],
        fps=settings["fps"],
        detail=settings["detail"],
        provider=settings["provider"],
    )


registry.register(
    name="video_analyze",
    toolset="video",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="🎬",
)
