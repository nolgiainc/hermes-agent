"""NOL-253: contact-sheet QC calls must not burn run budget.

The NOL-151 live run (session 8050af4b) measured 6 of 15 vision_analyze QC
calls timing out at ~68s each on montage-built contact sheets — ~7 minutes
of a 16.6-minute post-generation tail spent on calls that returned nothing.
The NOL-197 streak guard rewrote the error TEXT, but the text is advisory:
the same run drove the streak to 7, every member burning a full budget.

These tests pin the two mechanical guarantees that close the hole:

* the REQUEST SHAPE is bounded before the first attempt — payloads over
  the embed byte target or the send dimension cap are downscaled (cloud
  vision routes downscale to ~2048px internally, so an oversized send buys
  upload and tiling latency, not fidelity);
* the BURN is bounded once the route is degraded — after the consecutive-
  timeout guard orders the strategy switch, calls run with a reduced probe
  budget until one succeeds (success restores the full budget), and the
  empty-content retry never gets a second full budget.

Fully offline: the LLM call is mocked, images are data: URLs.
"""

import base64
import io
import json
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.vision_tools import (
    _VISION_DEFAULT_PROBE_TIMEOUT,
    _VISION_DEFAULT_TIMEOUT,
    _VISION_SEND_MAX_DIMENSION,
    _VISION_SEND_TARGET_BYTES,
    _reset_vision_timeout_streak,
    _resolve_vision_max_dimension,
    _resolve_vision_probe_timeout,
    vision_analyze_tool,
)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_TIMEOUT_ERR = Exception("Request timed out.")

# A tiny "JPEG" (magic bytes only) as a data: URL — resolves to bytes with
# no network and passes the resolver's magic-byte sniff, same trick as
# tests/tools/test_vision_timeout_streak.py.
_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")
_TINY_DATA_URL = f"data:image/jpeg;base64,{_JPEG_B64}"


@pytest.fixture(autouse=True)
def _clean_streak():
    _reset_vision_timeout_streak()
    yield
    _reset_vision_timeout_streak()


def _contact_sheet_data_url(cols=4, rows=4, tile=1024) -> str:
    """A representative montage-built contact sheet: cols x rows frames.

    Distinct per-tile colors keep the JPEG non-degenerate; at the default
    4x4 with 1024px tiles the sheet is 4096x4096 — the class of dense
    composite the NOL-151 run shipped at full resolution.
    """
    sheet = Image.new("RGB", (cols * tile, rows * tile))
    for r in range(rows):
        for c in range(cols):
            color = ((r * 67) % 256, (c * 41) % 256, ((r + c) * 29) % 256)
            sheet.paste(Image.new("RGB", (tile, tile), color),
                        (c * tile, r * tile))
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _decode_sent_image(llm_mock) -> Image.Image:
    """The image the mocked LLM actually received, decoded."""
    messages = llm_mock.call_args.kwargs["messages"]
    sent_url = messages[0]["content"][1]["image_url"]["url"]
    b64 = sent_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _sent_url(llm_mock) -> str:
    messages = llm_mock.call_args.kwargs["messages"]
    return messages[0]["content"][1]["image_url"]["url"]


def _success_mock():
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "Frames 1-16 look consistent."
    response.choices = [choice]
    return AsyncMock(return_value=response)


def _empty_response():
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = ""
    response.choices = [choice]
    return response


async def _analyze(image_url, llm_mock):
    with patch("tools.vision_tools.async_call_llm", llm_mock):
        raw = await vision_analyze_tool(image_url, "QC this sheet",
                                        "test/model")
    return json.loads(raw)


async def _analyze_tiny(llm_mock):
    """Streak-focused helper: tiny payload, encode patched out."""
    with (
        patch(
            "tools.vision_tools._image_to_base64_data_url",
            return_value="data:image/jpeg;base64,abc",
        ),
        patch("tools.vision_tools.async_call_llm", llm_mock),
    ):
        raw = await vision_analyze_tool(_TINY_DATA_URL, "describe",
                                        "test/model")
    return json.loads(raw)


class TestPreflightDownscale:
    @pytest.mark.asyncio
    async def test_contact_sheet_is_downscaled_before_send(self):
        llm = _success_mock()
        result = await _analyze(_contact_sheet_data_url(), llm)
        assert result["success"] is True
        assert llm.await_count == 1  # first attempt, no size-error retry
        sent = _decode_sent_image(llm)
        assert max(sent.size) <= _VISION_SEND_MAX_DIMENSION
        assert len(_sent_url(llm)) <= _VISION_SEND_TARGET_BYTES

    @pytest.mark.asyncio
    async def test_downscale_preserves_aspect_ratio(self):
        llm = _success_mock()
        await _analyze(_contact_sheet_data_url(cols=8, rows=2), llm)
        sent = _decode_sent_image(llm)
        assert max(sent.size) <= _VISION_SEND_MAX_DIMENSION
        width, height = sent.size
        assert width / height == pytest.approx(4.0, rel=0.05)

    @pytest.mark.asyncio
    async def test_small_image_is_sent_untouched(self):
        small = _contact_sheet_data_url(cols=1, rows=1, tile=800)
        llm = _success_mock()
        await _analyze(small, llm)
        # Under both caps: the exact original bytes go out, no re-encode.
        assert _sent_url(llm) == small

    @pytest.mark.asyncio
    async def test_dimension_cap_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_MAX_DIMENSION", "0")
        llm = _success_mock()
        await _analyze(_contact_sheet_data_url(), llm)
        sent = _decode_sent_image(llm)
        assert max(sent.size) == 4096  # full resolution preserved

    @pytest.mark.asyncio
    async def test_qc_pass_completes_within_wall_clock_budget(self):
        """Regression check (NOL-253 scope item 3): a QC pass on a
        representative contact sheet — resolve, preflight downscale,
        (mocked) call — completes in seconds, not a run-budget-sized
        chunk.  20s is a generous CI allowance; locally this runs in
        well under 5s."""
        llm = _success_mock()
        start = time.monotonic()
        result = await _analyze(_contact_sheet_data_url(), llm)
        elapsed = time.monotonic() - start
        assert result["success"] is True
        assert elapsed < 20.0


class TestResolverKnobs:
    def test_probe_timeout_default(self):
        assert _resolve_vision_probe_timeout() == _VISION_DEFAULT_PROBE_TIMEOUT

    def test_probe_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_PROBE_TIMEOUT", "7.5")
        assert _resolve_vision_probe_timeout() == 7.5

    def test_probe_timeout_malformed_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_PROBE_TIMEOUT", "soon-ish")
        assert _resolve_vision_probe_timeout() == _VISION_DEFAULT_PROBE_TIMEOUT

    def test_max_dimension_default(self):
        assert _resolve_vision_max_dimension() == _VISION_SEND_MAX_DIMENSION

    def test_max_dimension_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_MAX_DIMENSION", "1024")
        assert _resolve_vision_max_dimension() == 1024

    def test_max_dimension_malformed_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_MAX_DIMENSION", "huge")
        assert _resolve_vision_max_dimension() == _VISION_SEND_MAX_DIMENSION


class TestDegradedModeProbeBudget:
    @pytest.mark.asyncio
    async def test_full_budget_before_the_switch(self):
        llm = AsyncMock(side_effect=_TIMEOUT_ERR)
        await _analyze_tiny(llm)  # streak 0 -> 1
        assert llm.call_args.kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT
        llm2 = AsyncMock(side_effect=_TIMEOUT_ERR)
        await _analyze_tiny(llm2)  # streak 1 -> 2: still a full budget
        assert llm2.call_args.kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT

    @pytest.mark.asyncio
    async def test_probe_budget_after_the_switch(self):
        for _ in range(2):
            await _analyze_tiny(AsyncMock(side_effect=_TIMEOUT_ERR))
        llm = AsyncMock(side_effect=_TIMEOUT_ERR)
        result = await _analyze_tiny(llm)
        assert llm.call_args.kwargs["timeout"] == _VISION_DEFAULT_PROBE_TIMEOUT
        # The error text reports the budget that actually applied and says
        # degraded mode is on.
        assert "15s per-attempt budget" in result["analysis"]
        assert "Degraded mode is active" in result["analysis"]
        assert "STOP" in result["analysis"]

    @pytest.mark.asyncio
    async def test_success_restores_the_full_budget(self):
        for _ in range(2):
            await _analyze_tiny(AsyncMock(side_effect=_TIMEOUT_ERR))
        ok = await _analyze_tiny(_success_mock())
        assert ok["success"] is True
        llm = AsyncMock(side_effect=_TIMEOUT_ERR)
        await _analyze_tiny(llm)
        assert llm.call_args.kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT

    @pytest.mark.asyncio
    async def test_probe_cap_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_PROBE_TIMEOUT", "0")
        for _ in range(2):
            await _analyze_tiny(AsyncMock(side_effect=_TIMEOUT_ERR))
        llm = AsyncMock(side_effect=_TIMEOUT_ERR)
        result = await _analyze_tiny(llm)
        assert llm.call_args.kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT
        assert "Degraded mode" not in result["analysis"]

    @pytest.mark.asyncio
    async def test_worst_case_streak_budget_is_bounded(self):
        """The NOL-151 shape: 7 consecutive timeouts.  With the probe cap
        the budgets requested sum to 2 full + 5 probe attempts — ~195s at
        defaults — instead of 7 full budgets (~420s)."""
        budgets = []
        for _ in range(7):
            llm = AsyncMock(side_effect=_TIMEOUT_ERR)
            await _analyze_tiny(llm)
            budgets.append(llm.call_args.kwargs["timeout"])
        assert budgets == (
            [_VISION_DEFAULT_TIMEOUT] * 2
            + [_VISION_DEFAULT_PROBE_TIMEOUT] * 5
        )
        assert sum(budgets) <= 200.0


class TestEmptyContentRetry:
    @pytest.mark.asyncio
    async def test_retry_is_capped_at_the_probe_budget(self):
        llm = AsyncMock(return_value=_empty_response())
        result = await _analyze_tiny(llm)
        assert llm.await_count == 2  # one retry, exactly
        first, retry = llm.call_args_list
        assert first.kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT
        assert retry.kwargs["timeout"] == _VISION_DEFAULT_PROBE_TIMEOUT
        # Still the structured success-with-fallback-text shape.
        assert result["success"] is True
        assert "could not be analyzed" in result["analysis"]

    @pytest.mark.asyncio
    async def test_retry_keeps_full_budget_when_probe_disabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_PROBE_TIMEOUT", "0")
        llm = AsyncMock(return_value=_empty_response())
        await _analyze_tiny(llm)
        assert llm.await_count == 2
        assert llm.call_args_list[1].kwargs["timeout"] == _VISION_DEFAULT_TIMEOUT
