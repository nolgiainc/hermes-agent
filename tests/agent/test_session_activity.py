"""Unit tests for the shared session activity observation contract."""

import sys
from types import SimpleNamespace

import pytest

from agent.session_activity import (
    ACTIVITY_DESCRIPTION_MAX,
    ActivityProvenance,
    bound_activity_description,
    build_activity_snapshot,
    format_iteration_progress,
    normalize_activity_provenance,
    reset_session_activity_persist_window,
)

@pytest.mark.parametrize(
    "max_iterations, expected",
    [
        (sys.maxsize, "iteration 3"),  # AIAgent's default: unbounded, so no ceiling is shown
        (None, "iteration 3"),
        (250, "iteration 3/250"),  # a real budget (e.g. delegation.max_iterations) keeps N/M
    ],
)
def test_format_iteration_progress_hides_unbounded_ceiling(max_iterations, expected):
    out = format_iteration_progress(3, max_iterations)
    assert out == expected
    assert str(sys.maxsize) not in out

def test_bound_activity_description_truncates():
    long = "x" * (ACTIVITY_DESCRIPTION_MAX + 80)
    out = bound_activity_description(long)
    assert len(out) == ACTIVITY_DESCRIPTION_MAX
    assert out.endswith("…")


def test_reset_session_activity_persist_window_clears_rate_limit():
    agent = SimpleNamespace(_session_activity_last_persist_mono=1234.5)
    reset_session_activity_persist_window(agent)
    # -inf, not 0.0: the due-check subtracts from time.monotonic(), which
    # counts from boot — on a freshly-booted host with uptime under the
    # heartbeat interval a 0.0 reset would still read as "just persisted"
    # and swallow the forced write. -inf is due on any clock.
    assert agent._session_activity_last_persist_mono == float("-inf")
    import time
    from agent.session_activity import (
        SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS,
    )
    assert (
        time.monotonic() - agent._session_activity_last_persist_mono
        >= SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS
    )


def test_reset_session_activity_persist_window_swallows_missing_attr():
    reset_session_activity_persist_window(object())


def test_normalize_activity_provenance_defaults_to_unknown():
    assert normalize_activity_provenance(None) is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("") is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("not-a-real-source") is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("agent.activity") is ActivityProvenance.UNKNOWN
    assert (
        normalize_activity_provenance(ActivityProvenance.AGENT_COMPRESSION)
        is ActivityProvenance.AGENT_COMPRESSION
    )
    assert (
        normalize_activity_provenance("agent.compression_timeout")
        is ActivityProvenance.AGENT_COMPRESSION_TIMEOUT
    )

def test_build_activity_snapshot_includes_compat_aliases():
    snap = build_activity_snapshot(
        last_activity_at=100.0,
        last_activity_description="starting API call #1",
        last_activity_provenance=ActivityProvenance.UNKNOWN,
        now=110.0,
        extra={"api_call_count": 1},
    )
    assert snap["last_activity_at"] == 100.0
    assert snap["last_activity_description"] == "starting API call #1"
    assert snap["last_activity_provenance"] == "unknown"
    assert snap["seconds_since_activity"] == 10.0
    assert snap["last_activity_ts"] == 100.0
    assert snap["last_activity_desc"] == "starting API call #1"
    assert snap["description"] == "starting API call #1"
    assert snap["api_call_count"] == 1
    assert "phase" not in snap
    assert "last_progress_at" not in snap
