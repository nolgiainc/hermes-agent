"""Nolgia platform asset resolver: rewrite ``MEDIA:<path>`` tags to ``asset:<uuid>``.

Chat connectors (Telegram, Discord, Slack, …) intercept ``MEDIA:<path>``
tags in the agent's reply and upload the referenced file themselves — the
tag never reaches the human. The Nolgia platform relay is different: it
talks to this gateway over the HTTP API (``POST /v1/runs`` + status
polling) and persists the reply text verbatim as a chat transcript. A
pod-local path like ``MEDIA:/opt/data/cache/screenshots/shot.png`` is dead
text for a web user — there is no shared filesystem between the agent pod
and the web app.

This module closes that gap on API egress: each ``MEDIA:`` tag whose file
exists and is a platform-supported media type is uploaded to the Nolgia
asset library (the same signed three-step flow the platform's own tooling
uses: ``POST /assets/uploads`` → ``PUT`` bytes to the signed URL →
``POST /assets/uploads/{id}/complete``) and the tag is rewritten to
``asset:<uuid>``, which the web chat hydrates into a rich media card.
Anything that cannot be uploaded (missing file, non-media extension,
over the platform size cap, upload failure, per-message budget exceeded)
degrades to the bare filename so the reply never carries a dead local
path.

Enabled only when the pod carries platform credentials
(``NOLGIA_API_URL`` + ``NOLGIA_TOKEN``, injected by the deployment chart)
and ``NOLGIA_MEDIA_RESOLVER`` is not ``"0"`` (the escape hatch). Without
them the API server behaves exactly like upstream.

Stdlib-only on purpose: this must import (and fail soft) in any gateway
process, with or without optional extras installed.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple

from gateway.platforms.base import (
    MEDIA_DELIVERY_EXTS,
    MEDIA_EXTENSIONLESS_TAG_RE,
    MEDIA_TAG_CLEANUP_RE,
    _normalize_media_tag_path,
    validate_media_delivery_path,
)

logger = logging.getLogger(__name__)


_MIB = 1024 * 1024
_GIB = 1024 * _MIB

# Extension → (content_type, max_bytes). EXACTLY the platform's
# CreateAssetUploadRequest ``content_type`` enum (nolgia-api openapi.yaml)
# and its documented size limits: 100 MiB images, 50 GiB video, 5 GiB
# audio. Anything not in this map is not uploadable and degrades to the
# bare filename. Deliberately NOT derived from MEDIA_DELIVERY_EXTS — the
# platform accepts a much narrower set than connectors do.
_EXT_CONTENT_TYPES: Dict[str, Tuple[str, int]] = {
    ".png": ("image/png", 100 * _MIB),
    ".jpg": ("image/jpeg", 100 * _MIB),
    ".jpeg": ("image/jpeg", 100 * _MIB),
    ".webp": ("image/webp", 100 * _MIB),
    ".mp4": ("video/mp4", 50 * _GIB),
    ".mov": ("video/quicktime", 50 * _GIB),
    ".webm": ("video/webm", 50 * _GIB),
    ".mp3": ("audio/mpeg", 5 * _GIB),
    ".wav": ("audio/wav", 5 * _GIB),
    ".ogg": ("audio/ogg", 5 * _GIB),
    ".m4a": ("audio/mp4", 5 * _GIB),
}

# Upload-slot creation is a small JSON round trip; the signed PUT streams
# the actual bytes (video can be large — give it the long leash); complete
# verifies the object server-side and can also take a while.
_CREATE_TIMEOUT_SECONDS = 60.0
_PUT_TIMEOUT_SECONDS = 600.0
_COMPLETE_TIMEOUT_SECONDS = 300.0

# Per-message guards: a reply stuffed with tags must not turn one egress
# into an unbounded upload storm. Beyond either limit, remaining tags
# degrade to the bare filename (still never a local path). The wall budget
# is enforced both BETWEEN uploads (none starts past the deadline) and
# WITHIN the signed PUT (see _DeadlineFile): the PUT timeout is
# socket-inactivity only, so a huge-but-under-cap file streaming steadily
# would otherwise hold the egress — and the run.completed event built from
# it — far past the platform relay's turn budget.
_MAX_UPLOADS_PER_MESSAGE = 6
_UPLOAD_WALL_BUDGET_SECONDS = 120.0

# (resolved path, mtime_ns, size) → asset id. Repeat deliveries of the
# same unchanged file (within a message or across turns in this process)
# reuse the already-uploaded asset instead of re-uploading.
_asset_cache: Dict[Tuple[str, int, int], str] = {}
_asset_cache_lock = threading.Lock()

# All extensions the shared tag matcher recognizes (lowercased, for
# whitespace-boundary prefix candidates in multiword spans below).
_DELIVERABLE_EXT_SUFFIXES: Tuple[str, ...] = tuple(ext.lower() for ext in MEDIA_DELIVERY_EXTS)

# Whitespace inside a tag path (the matcher never lets a path cross a
# newline, so "whitespace" here is space/tab runs).
_INTRA_PATH_WHITESPACE_RE = re.compile(r"[^\S\n]+")


def resolver_enabled() -> bool:
    """True when platform credentials are present and the resolver isn't opted out.

    ``NOLGIA_API_URL`` + ``NOLGIA_TOKEN`` are injected into the gateway
    container by the platform deployment chart, so presence of both is the
    platform-mode signal (env is deployment-owned and changes on every
    roll, unlike the PVC-persisted config.yaml). ``NOLGIA_MEDIA_RESOLVER=0``
    is the operator escape hatch.
    """
    if os.environ.get("NOLGIA_MEDIA_RESOLVER", "").strip() == "0":
        return False
    return bool(os.environ.get("NOLGIA_API_URL", "").strip()) and bool(
        os.environ.get("NOLGIA_TOKEN", "").strip()
    )


def _api_base() -> str:
    """Platform API base with the ``/v1`` suffix (mirrors the chart's sync-skills.py)."""
    url = os.environ.get("NOLGIA_API_URL", "").strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def _http_json(method: str, path: str, payload: Optional[dict], timeout: float) -> dict:
    """Authenticated JSON round trip against the platform API."""
    headers = {"Authorization": "Bearer " + os.environ.get("NOLGIA_TOKEN", "").strip()}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(_api_base() + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


class _UploadDeadlineExceeded(Exception):
    """Raised mid-PUT when the per-message upload wall budget expires."""


class _DeadlineFile:
    """File-like PUT body that aborts ``read()`` once a monotonic deadline passes.

    ``http.client`` streams a file body in fixed-size chunks via ``read()``,
    so checking the deadline here bounds the whole signed PUT by wall clock.
    The PUT's ``timeout`` alone cannot: it is a socket-inactivity timeout,
    and a large file trickling steadily never trips it — one near-cap video
    could otherwise stream for tens of minutes while the run stays
    "running" past the relay's turn budget.
    """

    def __init__(self, handle: BinaryIO, deadline: float) -> None:
        self._handle = handle
        self._deadline = deadline

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self._deadline:
            raise _UploadDeadlineExceeded(
                f"per-message upload wall budget ({_UPLOAD_WALL_BUDGET_SECONDS:.0f}s) "
                "expired mid-transfer"
            )
        return self._handle.read(size)


def _upload_asset(path: Path, content_type: str, size_bytes: int, deadline: float) -> Optional[str]:
    """Run the platform's signed upload flow for one file; return the asset id.

    ``deadline`` is the caller's per-message wall budget (monotonic); the
    streamed PUT aborts as soon as it passes. Any failure logs a warning
    and returns None — the caller degrades the tag to a bare filename.
    Never raises.
    """
    try:
        slot = _http_json(
            "POST",
            "/assets/uploads",
            {"filename": path.name, "content_type": content_type, "size_bytes": size_bytes},
            _CREATE_TIMEOUT_SECONDS,
        )
        upload_url = slot.get("upload_url")
        upload_id = slot.get("upload_id")
        if not upload_url or not upload_id:
            logger.warning(
                "[nolgia_assets] upload slot for %s missing upload_url/upload_id", path.name
            )
            return None
        # Signed PUT straight to storage: Content-Type must EXACTLY match the
        # declared value (the URL signature covers it) and there is no
        # Authorization header. Stream from disk; Content-Length is explicit
        # so urllib doesn't fall back to chunked transfer encoding.
        with open(path, "rb") as file_handle:
            put_request = urllib.request.Request(
                upload_url,
                data=_DeadlineFile(file_handle, deadline),
                method="PUT",
                headers={"Content-Type": content_type, "Content-Length": str(size_bytes)},
            )
            with urllib.request.urlopen(put_request, timeout=_PUT_TIMEOUT_SECONDS) as response:
                response.read()
        asset = _http_json(
            "POST", f"/assets/uploads/{upload_id}/complete", None, _COMPLETE_TIMEOUT_SECONDS
        )
        asset_id = (asset or {}).get("id")
        if not asset_id:
            logger.warning("[nolgia_assets] upload complete for %s returned no asset id", path.name)
            return None
        return str(asset_id)
    except Exception as exc:
        logger.warning("[nolgia_assets] asset upload failed for %s: %s", path.name, exc)
        return None


class _MessageBudget:
    """Per-message upload counters (count + wall clock).

    ``exhausted()`` gates STARTING an upload; the same ``deadline`` is
    threaded into the PUT body stream (``_DeadlineFile``) so an in-flight
    upload cannot outlive the budget either.
    """

    def __init__(self) -> None:
        self.uploads = 0
        self.deadline = time.monotonic() + _UPLOAD_WALL_BUDGET_SECONDS

    def exhausted(self) -> bool:
        return self.uploads >= _MAX_UPLOADS_PER_MESSAGE or time.monotonic() >= self.deadline


def _resolve_tag(raw_path: str, budget: _MessageBudget) -> str:
    """Replacement text for one MEDIA: tag — ``asset:<uuid>`` or a bare filename.

    Never returns a local path and never raises.
    """
    # Filename-only fallback for anything that can't be uploaded. Uses the
    # same quote/punctuation normalization the tag matchers assume so
    # ``MEDIA:"/opt/data/my file.png"`` degrades to ``my file.png``.
    normalized = _normalize_media_tag_path(raw_path)
    fallback = Path(normalized).name if normalized else ""

    # Same safety gate as every connector delivery path: absolute, resolves
    # to an existing regular file, not under the credential/system denylist.
    safe_path = validate_media_delivery_path(raw_path)
    if not safe_path:
        return fallback
    resolved = Path(safe_path)
    fallback = resolved.name or fallback

    mapping = _EXT_CONTENT_TYPES.get(resolved.suffix.lower())
    if mapping is None:
        return fallback
    content_type, max_bytes = mapping

    try:
        stat = resolved.stat()
    except OSError:
        return fallback
    if stat.st_size > max_bytes:
        logger.warning(
            "[nolgia_assets] %s exceeds the platform size cap (%d > %d bytes)",
            resolved.name,
            stat.st_size,
            max_bytes,
        )
        return fallback

    cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    with _asset_cache_lock:
        cached = _asset_cache.get(cache_key)
    if cached:
        return f"asset:{cached}"

    if budget.exhausted():
        logger.warning(
            "[nolgia_assets] per-message upload budget exhausted; degrading %s to filename",
            resolved.name,
        )
        return fallback

    budget.uploads += 1
    asset_id = _upload_asset(resolved, content_type, stat.st_size, budget.deadline)
    if not asset_id:
        return fallback
    with _asset_cache_lock:
        _asset_cache[cache_key] = asset_id
    logger.info("[nolgia_assets] uploaded %s as asset:%s", resolved.name, asset_id)
    return f"asset:{asset_id}"


def _replace_known_ext_tag(match: "re.Match[str]", budget: _MessageBudget) -> str:
    """Replacement for one ``MEDIA_TAG_CLEANUP_RE`` match.

    The matcher's bare-path branch supports unquoted multiword paths
    (``\\S+(?:[^\\S\\n]+\\S+)*?``), so a single match can lazily run across
    spaces and swallow prose — or a SECOND ``MEDIA:`` tag — on the same
    line (quoted paths can't: the quote bounds them). When that happens,
    resolve the shortest whitespace-boundary prefix that ends in a
    deliverable extension and validates as a real file, then re-process
    the rest of the span (it may hold more tags). If nothing validates,
    degrade the first token and re-process the rest.
    """
    raw = match.group("path")
    if raw[:1] in "`\"'" or not _INTRA_PATH_WHITESPACE_RE.search(raw):
        return _resolve_tag(raw, budget)

    # parts = [token, ws, token, ws, ..., token]; a prefix candidate ends on
    # a token boundary (odd part counts), the rest keeps its separator.
    parts = re.split(r"([^\S\n]+)", raw)
    for end in range(1, len(parts) + 1, 2):
        prefix = "".join(parts[:end])
        if not prefix.lower().endswith(_DELIVERABLE_EXT_SUFFIXES):
            continue
        if not validate_media_delivery_path(prefix):
            continue
        return _resolve_tag(prefix, budget) + _resolve_text("".join(parts[end:]), budget)

    # Nothing up to any boundary is a real deliverable file: degrade the
    # first token to its filename and re-process the rest of the span so a
    # trailing tag still gets its own resolution.
    return _resolve_tag(parts[0], budget) + _resolve_text("".join(parts[1:]), budget)


def _resolve_text(text: str, budget: _MessageBudget) -> str:
    """Run both shared tag matchers over ``text`` under one message budget."""
    # Case-insensitive fast path: the shared matchers are IGNORECASE
    # (models hand-write ``media:`` variants and connector delivery honors
    # them), so this guard must never skip a tag the matchers would rewrite.
    if not text or "media:" not in text.lower():
        return text

    def _replace_known_ext(match: "re.Match[str]") -> str:
        return _replace_known_ext_tag(match, budget)

    def _replace_extensionless(match: "re.Match[str]") -> str:
        # MEDIA_EXTENSIONLESS_TAG_RE consumes trailing whitespace; keep it so
        # the words on either side of the tag don't fuse together.
        whole = match.group(0)
        trailing = whole[len(whole.rstrip()):]
        return _resolve_tag(match.group("path"), budget) + trailing

    resolved = MEDIA_TAG_CLEANUP_RE.sub(_replace_known_ext, text)
    return MEDIA_EXTENSIONLESS_TAG_RE.sub(_replace_extensionless, resolved)


def resolve_media_tags_to_assets(text: str) -> str:
    """Rewrite every ``MEDIA:<path>`` tag in ``text`` for platform egress.

    Runs the two shared tag matchers from ``gateway.platforms.base`` (the
    known-extension matcher first, then the extension-less companion) so
    this resolver can never drift from what connector delivery recognizes
    as a tag. Each match becomes ``asset:<uuid>`` on successful upload or
    the bare filename otherwise — the result never contains a pod-local
    path. Blocking (network + disk I/O): call it off the event loop.
    """
    # Must stay case-insensitive like the IGNORECASE matchers (hermes tools
    # emit uppercase ``MEDIA:``, models hand-write lowercase). Upstream's
    # _resolve_media_to_data_urls keeps its case-sensitive guard for parity;
    # this platform-only path deliberately does not mirror it.
    if not text or "media:" not in text.lower():
        return text
    try:
        return _resolve_text(text, _MessageBudget())
    except Exception:
        logger.warning("[nolgia_assets] MEDIA: tag resolution failed", exc_info=True)
        return text
