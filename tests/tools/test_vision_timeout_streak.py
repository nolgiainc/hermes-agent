"""Consecutive vision-timeout loop guard (NOL-197).

Measured on a live production pod: 6 of 15 vision_analyze calls timed out
at ~68s each, and after every one the agent downsized the same contact
sheet and resubmitted — full res, 1400px and 1024px all burned the entire
per-attempt budget (~7 minutes of a 16.6-minute post-generation tail)
before the strategy switched to single frames, which mostly succeed. The
structured error alone never said "stop resizing"; these tests pin the
guard that does:

* the FIRST timeout's analysis names the budget and allows one more,
  smaller attempt;
* the SECOND consecutive timeout orders the switch outright — stop
  resizing, analyze single frames or record the check as unavailable;
* any successful vision call resets the streak (the route recovered);
* non-timeout errors neither advance nor reset it — only evidence about
  the route's ability to serve a request within budget moves the counter.

Fully offline: the LLM call is mocked, the image is a data: URL.
"""

import base64
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import tools.vision_tools as vision_tools
from tools.vision_tools import (
    _is_vision_timeout,
    _reset_vision_timeout_streak,
    vision_analyze_tool,
)

# A tiny "JPEG" (magic bytes only) as a data: URL — resolves to bytes with
# no network and passes the resolver's magic-byte sniff, same trick as
# tests/tools/test_vision_tools.py.
_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")
_DATA_URL = f"data:image/jpeg;base64,{_JPEG_B64}"

_TIMEOUT_ERR = Exception("Request timed out.")


@pytest.fixture(autouse=True)
def _clean_streak():
    _reset_vision_timeout_streak()
    yield
    _reset_vision_timeout_streak()


async def _analyze(llm_mock):
    with (
        patch(
            "tools.vision_tools._image_to_base64_data_url",
            return_value="data:image/jpeg;base64,abc",
        ),
        patch("tools.vision_tools.async_call_llm", llm_mock),
    ):
        raw = await vision_analyze_tool(_DATA_URL, "describe", "test/model")
    return json.loads(raw)


def _timeout_mock():
    return AsyncMock(side_effect=_TIMEOUT_ERR)


def _success_mock():
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "A perfectly ordinary frame."
    response.choices = [choice]
    return AsyncMock(return_value=response)


class TestTimeoutPredicate:
    def test_matches_timed_out_message(self):
        assert _is_vision_timeout(Exception("Request timed out."))

    def test_matches_timeout_typed_exceptions(self):
        class FakeAPITimeoutError(Exception):
            pass

        assert _is_vision_timeout(FakeAPITimeoutError("anything"))

    def test_rejects_other_errors(self):
        assert not _is_vision_timeout(ValueError("400 invalid_request"))


class TestTimeoutStreak:
    @pytest.mark.asyncio
    async def test_first_timeout_allows_one_more_attempt(self):
        result = await _analyze(_timeout_mock())
        assert result["success"] is False
        assert "timed out after its" in result["analysis"]
        assert "per-attempt" in result["analysis"]
        assert "One more attempt" in result["analysis"]
        assert "STOP" not in result["analysis"]

    @pytest.mark.asyncio
    async def test_second_consecutive_timeout_orders_the_switch(self):
        await _analyze(_timeout_mock())
        result = await _analyze(_timeout_mock())
        assert result["success"] is False
        assert "STOP" in result["analysis"]
        assert "do not resize" in result["analysis"]
        assert "single small frames" in result["analysis"]
        assert "QC unavailable" in result["analysis"]
        assert "2nd consecutive" in result["analysis"]

    @pytest.mark.asyncio
    async def test_streak_keeps_counting_past_the_switch(self):
        for _ in range(3):
            result = await _analyze(_timeout_mock())
        assert "3rd consecutive" in result["analysis"]
        assert "STOP" in result["analysis"]

    @pytest.mark.asyncio
    async def test_success_resets_the_streak(self):
        await _analyze(_timeout_mock())
        ok = await _analyze(_success_mock())
        assert ok["success"] is True
        result = await _analyze(_timeout_mock())
        assert "STOP" not in result["analysis"]
        assert "One more attempt" in result["analysis"]

    @pytest.mark.asyncio
    async def test_non_timeout_error_does_not_advance_the_streak(self):
        await _analyze(AsyncMock(side_effect=ValueError("boom")))
        result = await _analyze(_timeout_mock())
        # Still the first-timeout message: the ValueError contributed
        # nothing to the consecutive-timeout count.
        assert "STOP" not in result["analysis"]
        assert "One more attempt" in result["analysis"]

    @pytest.mark.asyncio
    async def test_non_timeout_error_keeps_the_generic_wording(self):
        result = await _analyze(AsyncMock(side_effect=ValueError("boom")))
        assert result["success"] is False
        assert "could not be analyzed" in result["analysis"]
        assert "STOP" not in result["analysis"]

    @pytest.mark.asyncio
    async def test_error_shape_is_unchanged(self):
        """Callers already branch on {success, error, analysis} — the
        guard only rewrites the analysis text."""
        result = await _analyze(_timeout_mock())
        assert set(result) == {"success", "error", "analysis"}
        assert vision_tools  # namespace import used for patch targets
