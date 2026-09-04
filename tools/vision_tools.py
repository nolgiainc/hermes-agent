#!/usr/bin/env python3
"""Vision tools: ``vision_analyze`` (image) and ``video_analyze``.

Images resolve through :mod:`tools.image_source`, are normalized to a provider-supported format
(:mod:`tools.vision_tools_image_prep`), then either attach natively to a vision-capable main
model (multimodal tool-result envelope) or are described by the auxiliary vision LLM router.
"""

import base64
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, NamedTuple, Optional
from urllib.parse import parse_qs, urlparse
import httpx

# ``agent.auxiliary_client`` costs ~50 ms cold; only the handlers need it. Both names stay
# module attributes so tests can patch ``tools.vision_tools.async_call_llm`` (truthy-skip: mocks win).
async_call_llm: Any = None
extract_content_or_reasoning: Any = None


def _load_auxiliary_client() -> None:
    global async_call_llm, extract_content_or_reasoning
    if async_call_llm is None or extract_content_or_reasoning is None:
        from agent import auxiliary_client as _aux
        async_call_llm = async_call_llm or _aux.async_call_llm
        extract_content_or_reasoning = extract_content_or_reasoning or _aux.extract_content_or_reasoning


from hermes_constants import get_hermes_dir
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access
from tools.vision_tools_image_prep import (
    _VISION_MAX_VALIDATED_AGGREGATE_PIXELS,
    _VISION_MAX_VALIDATED_FRAME_COUNT,
    _crop_image_region,
    _determine_mime_type,
    _image_exceeds_dimension,
    _normalize_to_supported_image,
    _validate_raster_image_decodable)

logger = logging.getLogger(__name__)

_debug = DebugSession("vision_tools", env_var="VISION_TOOLS_DEBUG")


def _cfg_auxiliary(*keys: str, default=None):
    """``auxiliary.<keys...>`` from config.yaml; ``default`` when config is unavailable."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), "auxiliary", *keys, default=default)
    except Exception:
        return default


def _read_vision_setting(env_var: str, key: str, cast, minimum=None):
    """Env var → ``auxiliary.vision.<key>`` → None. Values that fail ``cast`` or fall below
    ``minimum`` are skipped in favor of the next source (a cap can never be disabled by a bad value)."""
    def _accept(raw):
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            return None
        return val if minimum is None or val >= minimum else None
    val = _accept(os.getenv(env_var, "").strip() or None)
    return val if val is not None else _accept(_cfg_auxiliary("vision", key))


# HTTP download timeout (separate from ``auxiliary.vision.timeout``, which governs the LLM call).
_VISION_DOWNLOAD_TIMEOUT = _read_vision_setting("HERMES_VISION_DOWNLOAD_TIMEOUT", "download_timeout", float)
if _VISION_DOWNLOAD_TIMEOUT is None:
    _VISION_DOWNLOAD_TIMEOUT = 30.0

# Hard cap on downloaded media (50 MB): bounds memory/disk against attacker-hosted files.
_VISION_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


# Per-attempt timeout for the vision/video LLM call itself.  Separate from
# ``auxiliary.vision.download_timeout`` above, which governs only the HTTP
# media fetch.  Resolution: env HERMES_VISION_TIMEOUT → config.yaml
# auxiliary.vision.timeout → the caller's default.
#
# 60s, not the old 120s: every other auxiliary task defaults to 30s
# (_DEFAULT_AUX_TIMEOUT), and a healthy provider returns a vision analysis
# in well under 20s.  A vision call still running at 60s is stalled, not
# slow, and a stall here is pure dead wall-clock inside a user-visible agent
# run.  Local VLMs (llama.cpp, ollama) that legitimately need longer raise it
# with one line of config or the env var.
_VISION_DEFAULT_TIMEOUT = 60.0


def _resolve_vision_timeout(
    default: float = _VISION_DEFAULT_TIMEOUT, *, floor: float = 0.0
) -> float:
    """Resolve the per-attempt vision LLM timeout, never below *floor*.

    ``floor`` exists for ``video_analyze_tool``, whose payloads are an order of
    magnitude larger than a single image: it keeps a low image-tuned config
    value from starving video analysis.  Best-effort throughout — any config
    read or parse failure falls back to *default*.  Also imported by the
    browser vision tools (``tools/browser_tool.py``, ``tools/browser_camofox.py``).
    """
    val = _read_vision_setting("HERMES_VISION_TIMEOUT", "timeout", float)
    return max(default if val is None else val, floor)


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


def _current_vision_timeout_streak() -> int:
    """Read the consecutive-timeout streak without mutating it."""
    with _vision_timeout_streak_lock:
        return _vision_timeout_streak


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
    val = _read_vision_setting("HERMES_VISION_PROBE_TIMEOUT", "probe_timeout", float)
    return default if val is None else val


# CPU-burst cap: a turn can fan out dozens of vision_analyze calls (base64 encode + Pillow
# resize each) that would saturate every core and starve the shared event loop. Only the CPU
# burst is capped (LLM calls stay concurrent), on a dedicated executor sized to usable cores.
# Must be a threading primitive — each call runs via model_tools._run_async on a PER-THREAD
# loop, so an asyncio semaphore cannot coordinate; the default executor is shared with the
# gateway/web server and is deliberately NOT used.
def _detect_host_cpus() -> int:
    """Usable CPU count (``sched_getaffinity`` honors cpuset pinning), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_vision_cpu_workers() -> int:
    """HERMES_VISION_MAX_CONCURRENCY → ``auxiliary.vision.max_concurrency`` → host cores (< 1 ignored)."""
    val = _read_vision_setting("HERMES_VISION_MAX_CONCURRENCY", "max_concurrency", int, minimum=1)
    return val or _detect_host_cpus()


_VISION_CPU_WORKERS = _resolve_vision_cpu_workers()

_vision_cpu_executor = ThreadPoolExecutor(
    max_workers=_VISION_CPU_WORKERS, thread_name_prefix="vision-encode",
)


async def _run_encode_on_cpu_executor(fn, *args, **kwargs):
    """Run a sync encode/resize callable on the bounded vision CPU executor (never the LLM call)."""
    import functools
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_vision_cpu_executor, functools.partial(fn, *args, **kwargs))


async def _validate_image_url_async(url: str) -> bool:
    """HTTP(S) shape check (scheme, netloc; extension-less CDN URLs pass) + SSRF guard with DNS
    off the event loop."""
    if not (isinstance(url, str) and url.startswith(("http://", "https://")) and urlparse(url).netloc):
        return False
    from tools.url_safety import async_is_safe_url
    return await async_is_safe_url(url)


def _is_retryable_download_error(error: Exception) -> bool:
    """Transient failures only: 429, 5xx, transport errors and anything unclassified. Fail-fast on
    other 4xx, PermissionError (policy/SSRF block) and ValueError (too large / blocked redirect)."""
    if isinstance(error, (PermissionError, ValueError)):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or not 400 <= status < 500
    return True


_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class _MediaTooLargeError(ValueError):
    """A download over ``max_bytes``; carries the size and limit so callers can phrase the error."""

    def __init__(self, media_label: str, size: int, limit: int):
        super().__init__(f"{media_label} too large ({size} bytes, max {limit})")
        self.size = size
        self.limit = limit


async def _ssrf_redirect_guard(response):
    """Re-validate each redirect target (a public URL 302ing to 169.254.169.254 would otherwise
    bypass the pre-flight check). Async because httpx.AsyncClient awaits hooks."""
    from tools.url_safety import async_is_safe_url, redirect_target_from_response
    redirect_url = redirect_target_from_response(response)
    if redirect_url and not await async_is_safe_url(redirect_url):
        raise ValueError(f"Blocked redirect to private/internal address: {redirect_url}")


async def _download_media(
    url: str, destination: Path, max_retries: int, *,
    media_label: str, accept: str, max_bytes: int, timeout: float, retry_all: bool,
    response_out: Optional[dict] = None) -> Path:
    """SSRF-safe streaming download with exponential backoff (2s/4s/8s). ``retry_all=False`` (images)
    retries only :func:`_is_retryable_download_error` errors — a 404/403 never succeeds on retry.
    Size overruns raise :class:`_MediaTooLargeError`. ``response_out`` (a dict) receives the
    response's ``content_type`` header when given."""
    from utils import atomic_replace
    from tools.url_safety import create_ssrf_safe_async_client
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(url)
            if blocked:
                raise PermissionError(blocked["message"])

            # follow_redirects for CDNs; the client validates DNS at connect
            # time and the hook re-validates each redirect target.
            async with create_ssrf_safe_async_client(
                timeout=timeout, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client, client.stream(
                "GET", url, headers={"User-Agent": _DOWNLOAD_USER_AGENT, "Accept": accept},
            ) as response:
                response.raise_for_status()

                # Content-Length gives an early reject but servers can omit or lie,
                # so the streaming cap below is authoritative.
                cl = response.headers.get("content-length")
                try:
                    declared_size = int(cl) if cl else None
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise _MediaTooLargeError(media_label, declared_size, max_bytes)
                blocked = check_website_access(str(response.url))
                if blocked:
                    raise PermissionError(blocked["message"])
                if response_out is not None:
                    response_out["content_type"] = response.headers.get("content-type") or ""

                # Stream to a temp file, atomically moved onto destination on success.
                tmp_destination = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                bytes_written = 0
                try:
                    with tmp_destination.open("wb") as f:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            bytes_written += len(chunk)
                            if bytes_written > max_bytes:
                                raise _MediaTooLargeError(media_label, bytes_written, max_bytes)
                            f.write(chunk)
                    atomic_replace(tmp_destination, destination)
                except Exception:
                    try:
                        tmp_destination.unlink(missing_ok=True)
                    except OSError:
                        logger.debug("Could not delete partial download: %s", tmp_destination, exc_info=True)
                    raise
            return destination
        except Exception as e:
            last_error = e
            if attempt >= max_retries - 1 or (not retry_all and not _is_retryable_download_error(e)):
                logger.error("%s download failed after %s attempt(s): %s",
                             media_label, attempt + 1, str(e)[:100], exc_info=True)
                if not retry_all:
                    raise
                break
            wait_time = 2 ** (attempt + 1)
            logger.warning("%s download failed (attempt %s/%s): %s",
                           media_label, attempt + 1, max_retries, str(e)[:50])
            if not retry_all:
                logger.warning("Retrying in %ss...", wait_time)
            await asyncio.sleep(wait_time)
    # Reaching here means max_retries was non-positive (or video exhausted retries).
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"_download_{media_label.lower()} exited retry loop without attempting (max_retries={max_retries})")


async def _download_image(image_url: str, destination: Path, max_retries: int = 3) -> Path:
    """Download an image with SSRF protection and error-class-aware retry."""
    return await _download_media(
        image_url, destination, max_retries,
        media_label="Image", accept="image/*,*/*;q=0.8",
        max_bytes=_VISION_MAX_DOWNLOAD_BYTES, timeout=_VISION_DOWNLOAD_TIMEOUT, retry_all=False)


def _image_to_base64_data_url(image_path: Path, mime_type: Optional[str] = None) -> str:
    """``data:<mime>;base64,...`` for a file (MIME from extension when not given)."""
    mime = mime_type or _determine_mime_type(image_path)
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


# Absolute hard ceiling for vision payloads (20 MB): no major provider accepts more.
_MAX_BASE64_BYTES = 20 * 1024 * 1024

# Proactive embed caps for history reuse: the native path bakes the data URL into the tool
# result, re-sent every later turn (a 4 MB embed cost ~100-260K billed tokens). Anthropic
# downsamples to a 1568px long edge anyway, so pixels past that cost wire bytes for no fidelity.
# The 20 MB hard ceiling / Anthropic 5 MB reject-cap still apply as safety nets; those are one-shot viewing
# limits, not history-reuse sizes. A 4 MB / 7900px embed was observed at ~400K chars and ~100–260K billed
# tokens per image (#92699), so we size for model reading instead: 256 KB keeps a 1568px screenshot cheap
# enough to ride the session (PNGs that exceed it are downscaled further by the byte-budget ladder), well
# under every provider's per-image limit.
_EMBED_TARGET_BYTES = 256 * 1024
_EMBED_MAX_DIMENSION = 1568

# Target when auto-resizing after a provider size rejection (retry once).
_RESIZE_TARGET_BYTES = 5 * 1024 * 1024

# Longest side (px) sent to the aux vision LLM (NOL-253).  Independent of
# _EMBED_MAX_DIMENSION above, which sizes native history embeds: this one is a
# performance cap on the aux-LLM send.  Every cloud route we serve downscales
# internally to roughly 2048px or less before the model sees the image (OpenAI
# fits high-detail images inside 2048x2048; Anthropic caps the long side
# around 1568), so pixels beyond ~2048 never reach the model — they are pure
# upload bytes and provider-side tiling latency.  The NOL-151 run measured the
# result on montage-built contact sheets shipped at full resolution: 6 of 15
# QC vision_analyze calls burned their whole per-attempt budget and returned
# nothing.
# Resolution: env HERMES_VISION_MAX_DIMENSION → config
# auxiliary.vision.max_dimension → 2048.  Values <= 0 disable the cap
# (full-resolution sends).
_VISION_SEND_MAX_DIMENSION = 2048

# Byte target for the aux-LLM preflight downscale (NOL-253). This is the 4 MB
# embed target the preflight was written and measured against; it is NOT
# _EMBED_TARGET_BYTES (256 KB), which sizes NATIVE embeds that ride the
# session history every turn — the aux path sends the image once and keeps
# only the text, so that history-reuse budget does not apply to it.
_VISION_SEND_TARGET_BYTES = 4 * 1024 * 1024


def _resolve_vision_max_dimension(
    default: int = _VISION_SEND_MAX_DIMENSION,
) -> int:
    """Resolve the longest-side pixel cap for vision sends.  Best-effort:
    any config read or parse failure falls back to *default*."""
    val = _read_vision_setting("HERMES_VISION_MAX_DIMENSION", "max_dimension", lambda v: int(float(v)))
    return default if val is None else val


_SIZE_ERROR_HINTS = (
    "too large", "payload", "413", "content_too_large",
    "request_too_large", "exceeds", "size limit",
)


def _is_image_size_error(error: Exception) -> bool:
    """Detect if an API error is related to image or payload size."""
    err_str = str(error).lower()
    return any(hint in err_str for hint in _SIZE_ERROR_HINTS + ("image_url", "invalid_request"))


def _build_scale_note(scale_info: Optional[dict], crop_offset: Optional[dict]) -> Optional[str]:
    """Coordinate-mapping disclosure for downscale and/or region crop; ``None`` when neither applied."""
    parts = []
    if scale_info:
        ow, oh = scale_info["orig_width"], scale_info["orig_height"]
        nw, nh = scale_info["new_width"], scale_info["new_height"]
        fx, fy = (ow / nw if nw else 1.0), (oh / nh if nh else 1.0)
        axes = (f"any coordinates you report by {fx:.2f}" if f"{fx:.2f}" == f"{fy:.2f}"
                else f"any x coordinates you report by {fx:.2f} and any y coordinates by {fy:.2f}")
        parts.append(f"Image downscaled from {ow}x{oh} to {nw}x{nh} for vision; "
                     f"multiply {axes} to map back to the original image.")
    if crop_offset:
        parts.append(
            f"Analysis was performed on a cropped region of the original "
            f"image starting at offset ({crop_offset['x']}, "
            f"{crop_offset['y']}); coordinates are relative to that crop "
            f"origin — add the offset to map back to the full image.")
    return " ".join(parts) if parts else None


def _import_pillow_for_resize():
    """Return ``PIL.Image`` or None (Pillow is a lazy-installable soft dependency). ``prompt=False``:
    a blocking input() deadlocks the CLI where prompt_toolkit owns stdin; the install is already
    gated by security.allow_lazy_installs."""
    try:
        from PIL import Image
    except ImportError:
        try:
            from tools.lazy_deps import ensure as _ensure_dep
            # prompt=False: never raise a blocking input() prompt mid-session. Under the interactive CLI
            # prompt_toolkit owns stdin, so a bare input() deadlocks the terminal (#40490). The install is
            # already gated by security.allow_lazy_installs, so reaching here is opt-in.
            _ensure_dep("tool.vision", prompt=False)
            from PIL import Image
        except Exception:
            return None
    return Image


def _resize_image_for_vision(image_path: Path, mime_type: Optional[str] = None,
                              max_base64_bytes: int = _RESIZE_TARGET_BYTES,
                              max_dimension: Optional[int] = None,
                              scale_out: Optional[dict] = None,
                              force_jpeg: bool = False) -> str:
    """Base64 data URL, progressively downscaled with Pillow while over budget.

    Halves dimensions (aspect-preserving, 64px floor) up to 4 times; JPEG also walks a
    quality ladder (85/70/50) per step. Without Pillow, or if it still doesn't fit, returns
    the best attempt (or raw bytes) and lets the caller apply the size check.
    ``max_dimension``: force a downscale above this long edge even when bytes fit
    (Anthropic's 8000px cap is independent of bytes). ``force_jpeg``: re-encode PNG as JPEG
    when resizing — halving PNG dimensions destroys text legibility on dense screenshots.

    Args: max_dimension: If set, images whose longest side exceeds this pixel count are forcibly downscaled
    even if they're under the byte budget. Anthropic enforces an 8000 px per-side cap independently of the 5
    MB byte cap. force_jpeg: Re-encode as JPEG even for PNG input when a resize is needed. History-reuse
    embeds (#92699) opt in so a text-heavy screenshot keeps its readable resolution and shrinks via JPEG
    quality instead. Images already under both caps are returned unchanged (still PNG).
    """
    file_size = image_path.stat().st_size
    estimated_b64 = (file_size * 4) // 3 + 100  # base64 ~4/3 + data URL header
    data_url = None
    if estimated_b64 <= max_base64_bytes and not (
        max_dimension is not None and _image_exceeds_dimension(image_path, max_dimension)
    ):
        data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        if len(data_url) <= max_base64_bytes:
            return data_url

    def _raw() -> str:
        return data_url or _image_to_base64_data_url(image_path, mime_type=mime_type)
    Image = _import_pillow_for_resize()
    if Image is None:
        logger.info("Pillow not installed — cannot auto-resize oversized image")
        return _raw()  # caller will raise the size error
    logger.info("Image file is %.1f MB (estimated base64 %.1f MB, limit %.1f MB, max_dimension=%s), auto-resizing...",
                file_size / (1024 * 1024), estimated_b64 / (1024 * 1024),
                max_base64_bytes / (1024 * 1024), max_dimension)
    # JPEG for photos (smaller), PNG for transparency — unless force_jpeg.
    is_png = (mime_type or _determine_mime_type(image_path)) == "image/png" and not force_jpeg
    pil_format, out_mime = ("PNG", "image/png") if is_png else ("JPEG", "image/jpeg")
    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.info("Pillow cannot open image for resizing: %s", exc)
        return _raw()

    # JPEG cannot encode alpha/palette modes (force_jpeg routes PNGs here).
    if not is_png and img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")
    quality_steps = (None,) if is_png else (85, 70, 50)
    orig_dims = prev_dims = (img.width, img.height)
    candidate = None

    def _record_scale(w: int, h: int) -> None:
        if scale_out is not None and (w, h) != orig_dims:
            scale_out.update(orig_width=orig_dims[0], orig_height=orig_dims[1], new_width=w, new_height=h)
    for attempt in range(5):
        if attempt > 0:
            # Halve, then re-derive from whichever axis hit the 64px floor so both shrink equally.
            new_w, new_h = max(int(img.width * 0.5), 64), max(int(img.height * 0.5), 64)
            if new_w == 64 and img.width > 0:
                new_h = max(int(img.height * (64 / img.width)), 64)
            elif new_h == 64 and img.height > 0:
                new_w = max(int(img.width * (64 / img.height)), 64)
            if (new_w, new_h) == prev_dims:
                break
            img = img.resize((new_w, new_h), Image.LANCZOS)
            prev_dims = (new_w, new_h)
            logger.info("Resized to %dx%d (attempt %d)", new_w, new_h, attempt)
        dims_ok = max_dimension is None or max(img.width, img.height) <= max_dimension
        for q in quality_steps:
            buf = BytesIO()
            img.save(buf, format=pil_format, **({} if q is None else {"quality": q}))
            candidate = f"data:{out_mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
            if len(candidate) <= max_base64_bytes and dims_ok:
                logger.info("Auto-resized image fits: %.1f MB (quality=%s, %dx%d)",
                            len(candidate) / (1024 * 1024), q, img.width, img.height)
                _record_scale(img.width, img.height)
                return candidate
    if candidate is None:
        return _raw()
    logger.warning("Auto-resize could not fit image under %.1f MB (best: %.1f MB)",
                   max_base64_bytes / (1024 * 1024), len(candidate) / (1024 * 1024))
    _record_scale(img.width, img.height)
    return candidate


# Native fast path: a vision-capable main model gets the image bytes as a multimodal
# tool-result envelope (the agent loop unwraps it into an OpenAI-style content list on the
# `tool` role); no aux LLM, no information loss. Providers whose tool results accept image
# content: Anthropic Messages (and aggregators proxying Claude — assume support), OpenAI
# Chat/Responses. Gemini is gated on model: only 3.x supports multimodal functionResponse.
_TOOL_RESULT_MEDIA_PROVIDERS = frozenset({
    "openrouter", "nous", "vertex", "bedrock", "anthropic-vertex", "google-vertex",
    "anthropic", "claude", "anthropic-direct",
    "openai", "openai-chat", "openai-codex", "azure-openai",
})
_GEMINI_PROVIDERS = frozenset({"google", "gemini", "google-gemini", "google-vertex-gemini"})


def _profile_rejects_tool_media(provider: str) -> bool:
    """Hard veto: the provider's ``ProviderProfile`` declares
    ``supports_vision_tool_messages=False`` — images are accepted in user
    messages but list-type tool-result content is rejected with 400
    (xiaomi/MiMo "text is not set"). ``supports_vision`` alone must not
    override this, or the multimodal tool-result envelope 400s every turn
    and the image never enters context (#89981).
    """
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(str(provider or "").strip().lower())
        return profile is not None and profile.supports_vision_tool_messages is False
    except Exception:
        return False


def _supports_media_in_tool_results(provider: str, model: str) -> bool:
    """Whether provider+model accepts image content inside a tool-result message. Unknown
    providers are False (caller falls back to aux-LLM text) unless their ``ProviderProfile``
    declares ``supports_vision``; ``supports_vision_tool_messages=False`` is a hard veto."""
    p = provider.strip().lower() if isinstance(provider, str) else ""
    if not p or _profile_rejects_tool_media(p):
        return False
    if p in _TOOL_RESULT_MEDIA_PROVIDERS:
        return True
    if p in _GEMINI_PROVIDERS:
        m = model.strip().lower() if isinstance(model, str) else ""
        return any(tag in m for tag in ("gemini-3", "gemini-pro-3", "gemini-flash-3"))
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(p)
        return profile is not None and bool(profile.supports_vision)
    except Exception:
        return False


def _should_use_native_vision_fast_path() -> bool:
    """True when image routing resolves to ``native`` AND the provider accepts images in tool
    results, or the user set the ``model.supports_vision`` override (escape hatch for
    custom/local providers). Any failure → False."""
    try:
        from agent.auxiliary_client import _read_main_provider, _read_main_model
        from agent.image_routing import decide_image_input_mode, _lookup_supports_vision
        from hermes_cli.config import load_config
        provider = _read_main_provider()
        model = _read_main_model()
        cfg = load_config()
        if decide_image_input_mode(provider, model, cfg) != "native":
            return False
        # The profile veto applies ahead of the capability lookup too: a
        # model marked vision-capable by models.dev / custom_providers must
        # not re-open the multimodal-envelope route the profile rejects.
        if _profile_rejects_tool_media(provider):
            return False
        return (
            _supports_media_in_tool_results(provider, model)
            or _lookup_supports_vision(provider, model, cfg) is True)
    except Exception as exc:
        logger.debug("Native vision fast-path check failed: %s", exc)
        return False


def _build_native_vision_tool_result(
    image_url: str, question: str, image_data_url: str, image_size_bytes: int,
    scale_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Multimodal tool-result envelope. The text part is intentionally minimal (the model already
    has the question); ``text_summary`` is the fallback for providers without multimodal tool results."""
    text_part = (
        "Image loaded into your context — you can see it natively now. "
        "Use your built-in vision to answer the user.")
    if isinstance(question, str) and question.strip():
        text_part += f"\n\nQuestion: {question.strip()}"
    if scale_note:
        text_part += f"\n\nNote: {scale_note}"
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": image_data_url}}],
        "text_summary": (
            f"Image attached natively for the main model ({image_size_bytes / 1024:.1f} KB). "
            "Answer using built-in vision."),
        "meta": {"image_url": image_url[:200], "size_bytes": image_size_bytes, "native_vision": True}}


def _unlink_quietly(path: Optional[Path]) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


class _ImagePrepError(ValueError):
    """Raised by :func:`_prepare_image`; the message is user-facing."""


class _PreparedImage(NamedTuple):
    """Temp image ready to encode; ``path`` is owned by the caller (delete it)."""
    path: Path
    mime: Optional[str]
    size_bytes: int
    crop_offset: dict


async def _prepare_image(
    image_url: str, task_id: Optional[str], region: Optional[list], *, validate_decode: bool,
) -> _PreparedImage:
    """Resolve → materialize → normalize → (validate) → (crop). Raises ``_ImagePrepError``.
    Unsupported formats (SVG, BMP) become PNG BEFORE encoding — an unsupported media_type baked
    into immutable history would 400 on every resume. The crop runs BEFORE any downscale so the
    region keeps the full resolution budget. On error no temp file is left."""
    from tools.image_source import ImageResolutionError, ResolveContext, resolve_image_source
    try:
        resolved = await resolve_image_source(image_url, ResolveContext(task_id=task_id))
    except ImageResolutionError as exc:
        raise _ImagePrepError(str(exc)) from exc
    temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"temp_image_{uuid.uuid4()}.img"
    await asyncio.to_thread(path.write_bytes, resolved.data)
    mime, size_bytes, crop_offset = resolved.mime, len(resolved.data), {}
    try:
        normalized_path, mime, norm_err = await asyncio.to_thread(_normalize_to_supported_image, path, mime)
        if norm_err or normalized_path is None:
            raise _ImagePrepError(norm_err or "Image normalization failed.")
        if normalized_path != path:
            _unlink_quietly(path)
            path = normalized_path
            size_bytes = path.stat().st_size
        if validate_decode:
            decode_error = await _run_encode_on_cpu_executor(
                _validate_raster_image_decodable, path,
                _VISION_MAX_VALIDATED_FRAME_COUNT, _VISION_MAX_VALIDATED_AGGREGATE_PIXELS)
            if decode_error:
                raise _ImagePrepError(decode_error)
        if region is not None:
            cropped_path, cropped_mime, crop_err = await asyncio.to_thread(
                _crop_image_region, path, region, offset_out=crop_offset)
            if crop_err or cropped_path is None:
                raise _ImagePrepError(crop_err or "Region crop failed.")
            _unlink_quietly(path)
            path, mime, size_bytes = cropped_path, cropped_mime, cropped_path.stat().st_size
    except BaseException:
        _unlink_quietly(path)
        raise
    return _PreparedImage(path, mime, size_bytes, crop_offset)


def _too_large_message(image_data_url: str) -> str:
    return (
        f"Image too large for vision API: base64 payload is {len(image_data_url) / (1024 * 1024):.1f} MB "
        f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) even after resizing. Install Pillow "
        f"(`pip install Pillow`) for better auto-resize, or compress the image manually.")


async def _resize_prepared(prepared: _PreparedImage, scale_info: dict, **kwargs) -> str:
    """Run :func:`_resize_image_for_vision` on the CPU executor for a prepared image."""
    return await _run_encode_on_cpu_executor(
        _resize_image_for_vision, prepared.path, mime_type=prepared.mime, scale_out=scale_info, **kwargs,
    )


async def _vision_analyze_native(
    image_url: str, question: str, task_id: Optional[str] = None, region: Optional[list] = None,
) -> Any:
    """Fast path for vision-capable main models: a ``_multimodal`` envelope dict on success,
    or a JSON error string (the normal tool-result contract) on failure."""
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url is required", success=False)
    prepared: Optional[_PreparedImage] = None
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)
        try:
            prepared = await _prepare_image(image_url, task_id, region, validate_decode=True)
        except _ImagePrepError as exc:
            return tool_error(str(exc), success=False)
        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url, prepared.path, mime_type=prepared.mime)
        # Proactive embed cap: this image is re-sent on every later turn, so resize DOWN to the
        # history-reuse target whenever the byte or long-edge cap is exceeded, not just at 20 MB.
        _scale_info: dict = {}
        # Anthropic still rejects >5 MB / >8000px with a non-retryable 400, but those are one-shot viewing
        # limits — history embeds are sized smaller so repeated vision_analyze turns don't blow the context
        # (#92699).
        _over_dims = await _run_encode_on_cpu_executor(
            _image_exceeds_dimension, prepared.path, _EMBED_MAX_DIMENSION)
        if len(image_data_url) > _EMBED_TARGET_BYTES or _over_dims:
            image_data_url = await _resize_prepared(
                prepared, _scale_info,
                max_base64_bytes=_EMBED_TARGET_BYTES, max_dimension=_EMBED_MAX_DIMENSION, force_jpeg=True)
            # Reject rather than embed a session-wedging payload.
            if len(image_data_url) > _MAX_BASE64_BYTES:
                return tool_error(_too_large_message(image_data_url), success=False)
        return _build_native_vision_tool_result(
            image_url=image_url, question=question, image_data_url=image_data_url,
            image_size_bytes=prepared.size_bytes,
            scale_note=_build_scale_note(_scale_info or None, prepared.crop_offset or None))
    except Exception as exc:
        logger.warning("Native vision fast path failed: %s", exc)
        return tool_error(f"Native vision failed: {exc}", success=False)
    finally:
        # Only delete temp files we created — never user-provided paths.
        if prepared is not None:
            _unlink_quietly(prepared.path)


def _aux_call_kwargs(messages: list, model: Optional[str], default_timeout: float, *,
                     min_timeout: Optional[float] = None, timeout: Optional[float] = None) -> dict:
    """``async_call_llm`` kwargs: the per-attempt vision timeout (env HERMES_VISION_TIMEOUT →
    ``auxiliary.vision.timeout`` → ``default_timeout``, never below ``min_timeout`` — video's
    floor) and ``auxiliary.vision.temperature`` (default 0.1). An explicit ``timeout`` — an
    already-resolved budget such as the degraded-mode probe budget — wins outright. Local vision
    models (llama.cpp, ollama) can take well over 30s, hence the generous defaults. ``max_tokens``
    is never forwarded (max-tokens-knob policy)."""
    if timeout is None:
        timeout = _resolve_vision_timeout(default=default_timeout, floor=min_timeout or 0.0)
    temperature = 0.1
    try:
        _vtemp = _cfg_auxiliary("vision", "temperature")
        if _vtemp is not None:
            temperature = float(_vtemp)
    except Exception:
        pass
    return {"task": "vision", "messages": messages, "temperature": temperature, "timeout": timeout,
            **({"model": model} if model else {})}


def _media_messages(user_prompt: str, part: dict) -> list:
    """Single user message: text + one media content ``part`` (``image_url``/``video_url``/``file``)."""
    return [{"role": "user", "content": [{"type": "text", "text": user_prompt}, part]}]


# Aux-LLM error classification: first matching hint set wins (billing → capability → size/format).
_BILLING_HINTS = ("402", "insufficient", "payment required", "credits", "billing")
_IMAGE_ERROR_RULES = (
    (_BILLING_HINTS,
     "Insufficient credits or payment required. Please top up your "
     "API provider account and try again. Error: {e}"),
    (("does not support", "not support image", "content_policy", "multimodal",
      "unrecognized request argument", "image input"),
     "{model} does not support vision or our request was not "
     "accepted by the server. Error: {e}"),
    (("invalid_request", "image_url"),
     "The vision API rejected the image. This can happen when the "
     "image is in an unsupported format, corrupted, or still too "
     "large after auto-resize. Try a smaller JPEG/PNG and retry. "
     "Error: {e}"),
)
_VIDEO_ERROR_RULES = (
    (_BILLING_HINTS, _IMAGE_ERROR_RULES[0][1]),
    (("does not support", "not support video", "content_policy", "multimodal",
      "unrecognized request argument", "video input", "video_url"),
     "The model does not support video analysis or the request was "
     "rejected. Ensure you're using a video-capable model "
     "(e.g. gemini-3.8-flash, set via auxiliary.video.model). Error: {e}"),
    (_SIZE_ERROR_HINTS,
     "The video is too large for the API. Try compressing or trimming "
     "the video (max ~50 MB; Gemini accepts roughly {gemini_cap_mb:.0f} MB "
     "of source video after base64 expansion -- local files and non-YouTube "
     "URLs are both inlined; only YouTube URLs are fetched by Google). Error: {e}"),
)
# kind -> (debug tool name, error rules)
_ANALYSIS_KINDS = {
    "image": ("vision_analyze_tool", _IMAGE_ERROR_RULES),
    "video": ("video_analyze_tool", _VIDEO_ERROR_RULES)}


def _debug_call_data(kind: str, source: str, user_prompt: str, model) -> dict:
    prompt = user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt
    return {
        "parameters": {f"{kind}_url": source, "user_prompt": prompt, "model": model},
        "error": None, "success": False, "analysis_length": 0, "model_used": model, f"{kind}_size_bytes": 0}


async def _call_vision_llm(call_kwargs: dict, empty_log: str, response=None, *,
                           retry_kwargs: Optional[dict] = None):
    """Aux vision LLM analysis text, retrying once on empty content (reasoning-only response).
    ``response``: an already-made first call (the image path handles size-error retries itself).
    ``retry_kwargs``: kwargs for the empty-content retry (default: ``call_kwargs``)."""
    _load_auxiliary_client()
    if response is None:
        response = await async_call_llm(**call_kwargs)
    analysis = extract_content_or_reasoning(response)
    if not analysis:
        logger.warning(empty_log)
        analysis = extract_content_or_reasoning(
            await async_call_llm(**(call_kwargs if retry_kwargs is None else retry_kwargs)))
    return analysis


async def _run_analysis(
    kind: str, source: str, user_prompt: str, model: Optional[str],
    stage: Callable[[str, dict, list], Awaitable[tuple]]) -> str:
    """Aux-LLM analysis skeleton for image/video: interrupt check → ``stage`` → JSON result.

    ``stage(user_prompt, debug_call_data, temp_paths)`` returns ``(analysis, scale_note)``; temp
    files it appends to ``temp_paths`` are deleted in ``finally``. Returns JSON
    ``{"success": bool, "analysis": str}`` (``analysis`` carries the error explanation on failure).
    Image analyses also drive the consecutive-timeout streak (NOL-197): success resets it, a
    timeout advances it and gets streak-aware guidance instead of the generic error text.
    """
    tool_name, rules = _ANALYSIS_KINDS[kind]
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data = _debug_call_data(kind, source, user_prompt, model)
    temp_paths: list = []

    def finish(result: dict) -> str:
        _debug.log_call(tool_name, debug_call_data)
        _debug.save()
        return json.dumps(result, indent=2, ensure_ascii=False)

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)
        logger.info("Analyzing %s: %s", kind, source[:60])
        logger.info("User prompt: %s", user_prompt[:100])
        analysis, scale_note = await stage(user_prompt, debug_call_data, temp_paths)
        analysis_length = len(analysis) if analysis else 0
        logger.info("%s analysis completed (%s characters)", kind.capitalize(), analysis_length)
        analysis = analysis or f"There was a problem with the request and the {kind} could not be analyzed."
        result = {"success": True, "analysis": f"[{scale_note}] {analysis}" if scale_note else analysis}
        if scale_note:
            result["scale_note"] = scale_note
        debug_call_data.update(success=True, analysis_length=analysis_length)
        if kind == "image":
            # A successful call proves the route is serving requests again —
            # the consecutive-timeout streak resets (NOL-197).
            _reset_vision_timeout_streak()
        return finish(result)
    except Exception as e:
        error_msg = f"Error analyzing {kind}: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        if kind == "image" and _is_vision_timeout(e):
            # Streak-aware timeout guidance (NOL-197): the second consecutive
            # full-budget burn orders the strategy switch instead of letting
            # the model resize and resubmit a third resolution of an image the
            # route cannot analyze in budget. Reports the budget that actually
            # applied to this attempt — the reduced probe budget when degraded
            # mode was active (NOL-253), the full budget otherwise.
            streak = _record_vision_timeout()
            budget = debug_call_data.get("vision_timeout")
            analysis = _vision_timeout_analysis(
                e, streak, _resolve_vision_timeout() if budget is None else budget,
                probe_active=bool(debug_call_data.get("probe_budget_active")))
        else:
            err_str = str(e).lower()
            template = next(
                (tpl for hints, tpl in rules if any(hint in err_str for hint in hints)),
                f"There was a problem with the request and the {kind} could not be analyzed. Error: {{e}}")
            analysis = template.format(e=e, model=model, gemini_cap_mb=_gemini_inline_source_cap_mb())
        debug_call_data["error"] = error_msg
        return finish({"success": False, "error": error_msg, "analysis": analysis})
    finally:
        for path in temp_paths:
            if path and path.exists():
                try:
                    path.unlink()
                    logger.debug("Cleaned up temporary %s file", kind)
                except Exception as cleanup_error:
                    logger.warning("Could not delete temporary file: %s", cleanup_error, exc_info=True)


async def vision_analyze_tool(
    image_url: str, user_prompt: str, model: str = None,
    task_id: Optional[str] = None, region: Optional[list] = None) -> str:
    """Describe an image (URL, local path, data: URL) with the auxiliary vision LLM. ``user_prompt``
    is pre-formatted by the caller. The payload is downscaled to the configured send cap
    (``auxiliary.vision.max_dimension``) before the first call and the call runs under the
    per-attempt vision budget (``auxiliary.vision.timeout``, probe budget once the route is
    degraded). Temp images live under $HERMES_HOME/cache/vision/."""
    async def stage(prompt: str, debug_call_data: dict, temp_paths: list) -> tuple:
        prepared = await _prepare_image(image_url, task_id, region, validate_decode=False)
        temp_paths.append(prepared.path)
        logger.info("Image ready (%.1f KB)", prepared.size_bytes / 1024)
        logger.info("Converting image to base64...")
        image_data_url = await _run_encode_on_cpu_executor(
            _image_to_base64_data_url, prepared.path, mime_type=prepared.mime)
        logger.info("Image converted to base64 (%.1f KB)", len(image_data_url) / 1024)
        _scale_info: dict = {}
        # Preflight send cap (NOL-253). This path used to send full resolution first and downscale
        # only after a provider SIZE rejection — but the failure mode that actually burns run
        # budget is not a 413, it is a TIMEOUT: montage-built contact sheets shipped at full
        # resolution made 6 of 15 QC calls burn their whole per-attempt budget in the NOL-151 run.
        # Cloud vision routes downscale internally to ~2048px anyway, so oversized sends buy no
        # fidelity — only upload bytes and tiling latency. Downscale BEFORE the first call whenever
        # the payload exceeds the send byte target or the send dimension cap.
        _send_max_dim = _resolve_vision_max_dimension()
        _over_dims = _send_max_dim > 0 and await _run_encode_on_cpu_executor(
            _image_exceeds_dimension, prepared.path, _send_max_dim)
        if len(image_data_url) > _VISION_SEND_TARGET_BYTES or _over_dims:
            image_data_url = await _resize_prepared(
                prepared, _scale_info, max_base64_bytes=_VISION_SEND_TARGET_BYTES,
                max_dimension=_send_max_dim if _send_max_dim > 0 else None)
            logger.info("Preflight-resized vision payload to %.1f KB (max_dimension=%s)",
                        len(image_data_url) / 1024, _send_max_dim if _send_max_dim > 0 else "off")
        # Hard limit (20 MB) — reached only when Pillow is unavailable or resizing failed.
        if len(image_data_url) > _MAX_BASE64_BYTES:
            raise ValueError(_too_large_message(image_data_url))
        debug_call_data["image_size_bytes"] = prepared.size_bytes
        messages = _media_messages(prompt, {"type": "image_url", "image_url": {"url": image_data_url}})
        logger.info("Processing image with vision model...")
        vision_timeout = _resolve_vision_timeout()
        # Degraded mode (NOL-253): once the consecutive-timeout guard has ordered the strategy
        # switch, spend probe-seconds discovering recovery, not full budgets. Any success resets
        # the streak and restores the full budget on the next call.
        probe_budget_active = False
        _streak_at_call = _current_vision_timeout_streak()
        if _streak_at_call >= _VISION_TIMEOUT_SWITCH_AT:
            _probe_timeout = _resolve_vision_probe_timeout()
            if 0 < _probe_timeout < vision_timeout:
                logger.info(
                    "Vision degraded mode: %d consecutive timeouts — capping this attempt at %.0fs "
                    "(probe budget, full budget %.0fs)", _streak_at_call, _probe_timeout, vision_timeout)
                vision_timeout = _probe_timeout
                probe_budget_active = True
        # The budget that really applied, so a timeout's error text reports it (see _run_analysis).
        debug_call_data.update(vision_timeout=vision_timeout, probe_budget_active=probe_budget_active)
        call_kwargs = _aux_call_kwargs(messages, model, _VISION_DEFAULT_TIMEOUT, timeout=vision_timeout)
        _load_auxiliary_client()
        try:
            response = await async_call_llm(**call_kwargs)
        except Exception as _api_err:
            if not (_is_image_size_error(_api_err) and len(image_data_url) > _RESIZE_TARGET_BYTES):
                raise
            logger.info(
                "API rejected image (%.1f MB, likely too large); auto-resizing to ~%.0f MB and retrying...",
                len(image_data_url) / (1024 * 1024), _RESIZE_TARGET_BYTES / (1024 * 1024))
            image_data_url = await _resize_prepared(prepared, _scale_info)
            messages[0]["content"][1]["image_url"]["url"] = image_data_url
            response = await async_call_llm(**call_kwargs)
        # Retry once on empty content — but never with a second full budget (NOL-253): the first
        # attempt already proved the route responsive (it returned, just empty), so the retry
        # either succeeds quickly or is not worth another full burn.
        _retry_timeout = call_kwargs["timeout"]
        _probe_timeout = _resolve_vision_probe_timeout()
        if 0 < _probe_timeout < _retry_timeout:
            _retry_timeout = _probe_timeout
        analysis = await _call_vision_llm(
            call_kwargs, f"Vision LLM returned empty content, retrying once (timeout {_retry_timeout:.0f}s)",
            response, retry_kwargs={**call_kwargs, "timeout": _retry_timeout})
        return analysis, _build_scale_note(_scale_info or None, prepared.crop_offset or None)
    return await _run_analysis("image", image_url, user_prompt, model, stage)


def check_vision_requirements() -> bool:
    """True when ``call_llm(task="vision")`` could resolve a client.

    Mirrors its fallback chain: explicit ``auxiliary.vision.provider``, then auto (main
    provider → openrouter → nous) — without the auto step the tool would vanish whenever
    the explicit name was unresolvable. Probe mode skips real SDK client construction.

    See #31179.
    """
    try:
        from agent.auxiliary_client import aux_probe_mode, resolve_vision_provider_client
        with aux_probe_mode():
            return any(
                resolve_vision_provider_client(**kw)[1] is not None for kw in ({}, {"provider": "auto"})
            )
    except Exception:
        return False


from tools.registry import registry, tool_error

VISION_ANALYZE_SCHEMA = {
    "name": "vision_analyze",
    # Routing mechanics are deliberately absent (the route is automatic and the
    # native result says so itself); region keeps its pre-effect guidance — a
    # model that doesn't know crops keep full resolution never zooms.
    "description": (
        # Dieted (#95681): routing mechanics (native attach vs aux-model text fallback) removed — the route
        # is automatic and the native path's own tool result says "you can see it natively now"; the schema
        # doesn't need to predict plumbing.
        "Load an image into the conversation so you can see it. Call it "
        "any time the user references an image — then answer from what "
        "you see."
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
                "description": "Your question or request about the image."
            },
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional [x1, y1, x2, y2] crop in ORIGINAL-image pixel "
                    "coordinates, applied before any downscaling — the crop "
                    "keeps full resolution. Load the full image first, then "
                    "re-call with a region to zoom into small text or fine "
                    "detail."
                )
            }
        },
        "required": ["image_url", "question"]
    }
}


def _configured_aux_model(sections: tuple, env_vars: tuple) -> Optional[str]:
    """First non-empty ``auxiliary.<section>.model`` from config.yaml, else the first non-empty
    env var (legacy override), else None."""
    for section in sections:
        _vmodel = _cfg_auxiliary(section, "model")
        if _vmodel:
            if str(_vmodel).strip():
                return str(_vmodel).strip()
            break
    return next((v for v in (os.getenv(e, "").strip() for e in env_vars) if v), None)


async def _handle_vision_analyze(args: Dict[str, Any], **kw: Any) -> str:
    image_url, question, region = args.get("image_url", ""), args.get("question", ""), args.get("region")
    task_id = kw.get("task_id")
    # No concurrency gate around the whole analysis — the CPU burst is bounded inside the
    # encode/resize step, so multi-image fan-out keeps full request concurrency.
    if _should_use_native_vision_fast_path():
        logger.info("vision_analyze: native fast path")
        return await _vision_analyze_native(image_url, question, task_id=task_id, region=region)

    # Legacy path: aux LLM describes the image and we return its text.
    full_prompt = (
        "Fully describe and explain everything about this image, then answer the "
        f"following question:\n\n{question}")
    model = _configured_aux_model(("vision",), ("AUXILIARY_VISION_MODEL",))
    return await vision_analyze_tool(image_url, full_prompt, model, task_id=task_id, region=region)


registry.register(
    name="vision_analyze",
    toolset="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=_handle_vision_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="👁️")


# --- video_analyze --------------------------------------------------------
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


def _gemini_inline_source_cap_bytes() -> int:
    """Source-video bytes that fit under ``_GEMINI_INLINE_MAX_BASE64_BYTES``.

    Read at call time (not import time) so the figure tracks the cap wherever
    it is tuned or patched. It is also the download cap for every non-YouTube
    URL sent to Gemini, which the tool inlines exactly like a local file.
    """
    return _GEMINI_INLINE_MAX_BASE64_BYTES // 4 * 3


def _gemini_inline_source_cap_mb() -> float:
    """``_gemini_inline_source_cap_bytes`` in MB, for user-facing errors."""
    return _gemini_inline_source_cap_bytes() / (1024 * 1024)


def _gemini_inline_too_large_error(source_bytes: int) -> ValueError:
    """The one error for a clip Gemini cannot take inline (local or downloaded).

    Google fetches only YouTube URLs itself; every other source travels
    inline in the request body, so the remedy is the same either way. The
    Files API is not wired, so it is deliberately not suggested.
    """
    return ValueError(
        f"Video too large to inline for Gemini: {source_bytes / (1024 * 1024):.1f} MB "
        f"of source video (base64 limit "
        f"{_GEMINI_INLINE_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB). "
        f"Gemini accepts roughly {_gemini_inline_source_cap_mb():.0f} MB of source "
        "video after base64 expansion; trim the clip or pass a YouTube URL "
        "(the only URLs Google fetches itself) and retry."
    )


# Gemini samples ~2.8 fps when ``video_metadata.fps`` is omitted (measured
# through the proxy: 91 video-tokens/s). ``fps: 1`` measured 32 tokens/s — a
# reproducible ~2.8x saving with no visible loss for "what happens in this
# clip" questions. Fractional fps is NOT a reliable lever (0.5 cost more
# than 1; 0.2 differed between identical runs).
_GEMINI_VIDEO_DEFAULT_FPS = 1

_VIDEO_FETCH_HEADERS = {
    "User-Agent": _DOWNLOAD_USER_AGENT,
    "Accept": "video/*,*/*;q=0.8",
}

_INVALID_VIDEO_SOURCE = "Invalid video source. Provide an HTTP/HTTPS URL or a valid local file path."


def _detect_video_mime_type(video_path: Path) -> Optional[str]:
    """Video MIME type from extension, or None if unsupported."""
    return _VIDEO_MIME_TYPES.get(video_path.suffix.lower())


def _unsupported_video_format(suffix: str) -> str:
    return f"Unsupported video format: '{suffix}'. Supported: {', '.join(sorted(_VIDEO_MIME_TYPES.keys()))}"


def _video_to_base64_data_url(video_path: Path, mime_type: Optional[str] = None) -> str:
    mime = mime_type or _detect_video_mime_type(video_path) or "video/mp4"
    return f"data:{mime};base64,{base64.b64encode(video_path.read_bytes()).decode('ascii')}"


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
    (a YouTube URL Google fetches server-side, or a ``data:`` URL that is
    inlined -- see ``_is_youtube_url`` for why nothing else is passed
    through), ``format`` its MIME type, ``detail`` maps to the per-part
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


_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})


def _is_youtube_url(url: Any) -> bool:
    """True for a YouTube video URL that Gemini can fetch itself.

    Google resolves ``file_data`` URLs server-side ONLY for YouTube:
    ``youtube.com/watch?v=<id>``, ``youtube.com/shorts/<id>`` (with or
    without ``www.``/``m.``) and ``youtu.be/<id>``. Any other https URL --
    a signed Google Cloud Storage link included -- comes back from the
    Gemini API as ``403 PERMISSION_DENIED "The caller does not have
    permission"`` after a long stall, so everything else is downloaded and
    inlined by ``video_analyze_tool`` instead.
    """
    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme.lower() not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host == "youtu.be":
        return bool(path.strip("/"))
    if host not in _YOUTUBE_HOSTS:
        return False
    if path == "/watch":
        return bool(parse_qs(parsed.query).get("v", [""])[0])
    return path.startswith("/shorts/") and bool(path[len("/shorts/"):].strip("/"))


def _resolve_downloaded_video_mime(video_url: str, content_type: Optional[str]) -> str:
    """MIME type for a downloaded clip: Content-Type, then URL extension, then mp4.

    The server's ``video/*`` Content-Type is authoritative when present. The
    URL path extension is the next best evidence (signed URLs keep it, e.g.
    ``.../clip.webm?X-Goog-Signature=...``). ``video/mp4`` is the last
    resort: Gemini requires *some* MIME type on the part and treats mp4 as
    the generic container, and the temp file's own ``.mp4`` suffix says
    nothing about what was fetched.
    """
    return content_type or _video_mime_from_url(video_url) or "video/mp4"


async def _probe_remote_video_mime(video_url: str, timeout: float = 15.0) -> str:
    """Best-effort MIME type for a YouTube URL that Google will fetch itself.

    Only YouTube URLs are passed through (``_is_youtube_url``); every other
    URL is downloaded and its MIME resolved from the response instead.

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


class _VideoTooLargeError(_MediaTooLargeError):
    """A clip over the download cap; carries the size so callers can phrase the error."""

    def __init__(self, size: int, limit: int):
        super().__init__("Video", size, limit)


class _DownloadedVideo(NamedTuple):
    """Where ``_download_video`` put the clip and what the server said it was."""

    path: Path
    content_type: Optional[str]  # the response's ``video/*`` type, else None


async def _download_video(
    video_url: str,
    destination: Path,
    max_retries: int = 3,
    max_bytes: int = _MAX_VIDEO_BASE64_BYTES,
) -> _DownloadedVideo:
    """Download a video with SSRF protection and error-class-aware retry.

    ``max_bytes`` bounds the clip: the ``Content-Length`` header is checked
    first and the streamed body again chunk by chunk; either overrun raises
    ``_VideoTooLargeError``. Gemini callers pass
    ``_gemini_inline_source_cap_bytes()`` so a clip that could never be
    inlined is refused at the header stage. Deterministic failures (too
    large, policy block, SSRF redirect, 4xx other than 429) are not retried.
    """
    response_info: dict = {}
    try:
        await _download_media(
            video_url, destination, max_retries,
            media_label="Video", accept=_VIDEO_FETCH_HEADERS["Accept"],
            max_bytes=max_bytes, timeout=60.0, retry_all=False, response_out=response_info)
    except _MediaTooLargeError as e:
        raise _VideoTooLargeError(e.size, e.limit) from e
    content_type = (response_info.get("content_type") or "").split(";", 1)[0].strip().lower()
    return _DownloadedVideo(destination, content_type if content_type.startswith("video/") else None)


async def _materialize_video(
    video_url: str, task_id: Optional[str], temp_paths: list, *,
    max_bytes: int = _MAX_VIDEO_BASE64_BYTES) -> tuple:
    """``(local path, MIME type)`` for a terminal-backend path, local file, or HTTP(S) URL. Only files
    created here are appended to ``temp_paths`` — never user-provided paths. Terminal-backend reads use
    the shared media resolver with ``permitted=("video",)`` — the exact pipeline vision_analyze uses
    (media-cache host reads, bounded in-sandbox exec-read, credential-read guard, 50MB cap). Downloads
    are capped at ``max_bytes`` and take their MIME from the response (Content-Type → URL extension →
    mp4); files take it from the extension (``None`` when unsupported)."""
    from tools.image_source import (
        ImageResolutionError, ResolveContext, _is_local_terminal_backend, resolve_image_source,
    )
    source = video_url.removeprefix("file://")
    local_path = Path(os.path.expanduser(source))
    lowered = (video_url or "").strip().lower()
    path_like = bool(lowered) and not lowered.startswith(("http://", "https://", "data:"))
    if not _is_local_terminal_backend() and path_like:
        logger.info("Reading video source via terminal backend: %s", video_url)
        suffix = Path(source).suffix.lower()
        if suffix not in _VIDEO_MIME_TYPES:
            raise ValueError(_unsupported_video_format(suffix))
        try:
            resolved = await resolve_image_source(
                video_url, ResolveContext(task_id=task_id), permitted=("video",))
        except ImageResolutionError as exc:
            raise ValueError(f"Could not read video from terminal backend: {exc}") from exc
        temp_dir = get_hermes_dir("cache/video", "temp_video_files")
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"terminal_video_{uuid.uuid4()}{suffix}"
        path.write_bytes(resolved.data)
        temp_paths.append(path)
        return path, _detect_video_mime_type(path)
    if local_path.is_file():
        from agent.file_safety import raise_if_read_blocked
        raise_if_read_blocked(str(local_path))
        logger.info("Using local video file: %s", video_url)
        return local_path, _detect_video_mime_type(local_path)
    if await _validate_image_url_async(video_url):
        blocked = check_website_access(video_url)
        if blocked:
            raise PermissionError(blocked["message"])
        path = get_hermes_dir("cache/video", "temp_video_files") / f"temp_video_{uuid.uuid4()}.mp4"
        temp_paths.append(path)
        downloaded = await _download_video(video_url, path, max_bytes=max_bytes)
        return downloaded.path, _resolve_downloaded_video_mime(video_url, downloaded.content_type)
    raise ValueError(_INVALID_VIDEO_SOURCE)


async def video_analyze_tool(
    video_url: str, user_prompt: str, model: str = None, task_id: Optional[str] = None,
    fps: Optional[float] = _GEMINI_VIDEO_DEFAULT_FPS, detail: Optional[str] = None,
    provider: Optional[str] = None) -> str:
    """Analyze a video via multimodal LLM. Returns JSON {success, analysis}.

    Gemini models get the clip as a ``file`` part. A YouTube URL is passed
    through for Google to fetch; every other http(s) URL and every local
    file is inlined as a base64 data URL, capped at
    ``_gemini_inline_source_cap_bytes()`` (~14 MB of source video). Other
    models keep the download + ``video_url`` data-URL path (~50 MB cap).

    ``fps`` / ``detail`` only affect Gemini models (see
    ``_video_part_for_model``): ``fps`` is sent as ``video_metadata.fps``
    (``None``/``0`` omits it so Gemini samples at its own rate) and ``detail``
    as the per-part ``media_resolution``. ``provider`` is an explicit
    auxiliary provider override (``auxiliary.video.provider``).
    """
    gemini = _is_gemini_model(model)

    async def stage(prompt: str, debug_call_data: dict, temp_paths: list) -> tuple:
        debug_call_data["parameters"].update(fps=fps, detail=detail)
        debug_call_data.update(video_part_type="file" if gemini else "video_url", video_passthrough=False)
        video_size_bytes = 0
        if gemini and _is_youtube_url(video_url):
            # Google fetches YouTube ``file_data`` URLs server-side, so there is nothing to
            # download, base64-encode or size-cap here. The SSRF and website-policy checks
            # still gate the URL exactly as they gate a download.
            if not await _validate_image_url_async(video_url):
                raise ValueError(_INVALID_VIDEO_SOURCE)
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])
            detected_mime = await _probe_remote_video_mime(video_url)
            video_file_data = video_url
            debug_call_data["video_passthrough"] = True
            logger.info("Passing YouTube URL through for Gemini to fetch (%s)", detected_mime)
        else:
            # Every other URL is downloaded and inlined. Google does NOT fetch arbitrary https
            # URLs: a signed GCS link in ``file_data`` returns 403 PERMISSION_DENIED after ~50 s,
            # while the same clip inlined is answered in ~5 s. For Gemini the download is capped
            # at what can be inlined so an oversized clip is refused at the Content-Length stage
            # with the same error a local file gets.
            try:
                temp_video_path, detected_mime = await _materialize_video(
                    video_url, task_id, temp_paths,
                    max_bytes=_gemini_inline_source_cap_bytes() if gemini else _MAX_VIDEO_BASE64_BYTES)
            except _VideoTooLargeError as e:
                if gemini:
                    raise _gemini_inline_too_large_error(e.size) from e
                raise
            video_size_bytes = temp_video_path.stat().st_size
            video_size_mb = video_size_bytes / (1024 * 1024)
            logger.info("Video ready (%.1f MB)", video_size_mb)
            if not detected_mime:
                raise ValueError(_unsupported_video_format(temp_video_path.suffix))
            if video_size_bytes > _VIDEO_SIZE_WARN_BYTES:
                logger.warning("Video is %.1f MB — may be slow or rejected", video_size_mb)
            video_data_url = _video_to_base64_data_url(temp_video_path, mime_type=detected_mime)
            if gemini and len(video_data_url) > _GEMINI_INLINE_MAX_BASE64_BYTES:
                raise _gemini_inline_too_large_error(video_size_bytes)
            if len(video_data_url) > _MAX_VIDEO_BASE64_BYTES:
                raise ValueError(
                    f"Video too large for API: base64 payload is {len(video_data_url) / (1024 * 1024):.1f} MB "
                    f"(limit {_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                    f"Compress or trim the video and retry.")
            video_file_data = video_data_url
        debug_call_data["video_size_bytes"] = video_size_bytes
        fps_value = _normalize_video_fps(fps)
        if gemini and fps_value is not None and fps_value < 1:
            logger.warning(
                "auxiliary.video.fps=%s is fractional; measured non-reproducible "
                "through Gemini — use 1 for a predictable saving", fps_value)
        video_part = _video_part_for_model(model, video_file_data, detected_mime, fps=fps_value, detail=detail)
        messages = _media_messages(prompt, video_part)
        call_kwargs = _aux_call_kwargs(messages, model, 180.0, min_timeout=180.0)
        if _gemini_rejects_sampling_overrides(model):
            # gemini-3.8-flash+ rejects temperature/top_p and the proxy strips
            # them anyway — keep them off the wire entirely.
            logger.debug("%s rejects sampling overrides; not sending temperature", model)
            call_kwargs.pop("temperature", None)
        if provider:
            call_kwargs["provider"] = provider
        analysis = await _call_vision_llm(call_kwargs, "Empty video response, retrying once")
        return analysis, None
    return await _run_analysis("video", video_url, user_prompt, model, stage)


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video from a URL or local file path using a multimodal AI model. "
        "Sends the video to a video-capable model (e.g. Gemini) for understanding. "
        "Use this for video files — for images, use vision_analyze instead. "
        "Supports mp4, webm, mov, avi, mkv, mpeg formats. "
        "Note: large videos (>20 MB) may be slow; max ~50 MB. With Gemini models "
        "only YouTube URLs are fetched by Google; any other URL or local file is "
        "inlined and must be under ~14 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "Video URL (http/https, including YouTube links) or local file path to analyze.",
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
    video_url, question = args.get("video_url", ""), args.get("question", "")
    full_prompt = (
        "Fully describe and explain everything happening in this video, "
        "including visual content, motion, audio cues, text overlays, and scene "
        f"transitions. Then answer the following question:\n\n{question}")
    # Prefer config.yaml auxiliary.video.* (model falling back to vision);
    # env vars are a legacy override for the model.
    settings = _resolve_video_settings()
    return video_analyze_tool(
        video_url, full_prompt, settings["model"], task_id=kw.get("task_id"),
        fps=settings["fps"], detail=settings["detail"], provider=settings["provider"])


registry.register(
    name="video_analyze",
    toolset="video",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="🎬")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import contextlib  # noqa: F401,E402
import sys  # noqa: F401,E402
import threading  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
