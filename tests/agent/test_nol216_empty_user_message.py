"""Regression tests for NOL-216 — an empty-content USER message wedges a session.

The admin agent hard-wedged when an empty-content ``user`` message sat at
position 715 of the conversation history. Every turn re-sent the full history;
Moonshot (kimi-k3's provider) rejected the empty user message with a
NON-RETRYABLE HTTP 400 ("the message at position N with role 'user' must not be
empty"), so the turn died and the persisted poison re-wedged every subsequent
turn — the NOL-106 permanent-wedge shape, here on the user side and previously
unguarded.

The fix generalizes the NOL-106 assembly guard in ``sanitize_api_messages``: an
empty-content user message (like an empty assistant) is dropped at the final
pre-API chokepoint, so a PRE-EXISTING poisoned history self-heals on its next
turn with NO session-data surgery. A multimodal user turn whose text is empty
but which carries image/other content parts is NOT empty and is kept.
"""

from run_agent import AIAgent


def _has_empty_content_message(messages):
    """True if any message would trip a provider's 'must not be empty' 400.

    Only user/assistant text-content messages are considered: an assistant that
    carries tool_calls or reasoning is a legitimate empty-``content`` turn.
    """
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and (
            (isinstance(m.get("tool_calls"), list) and m.get("tool_calls"))
            or any(m.get(k) for k in ("reasoning_content", "reasoning"))
        ):
            continue
        content = m.get("content")
        if content is None:
            return True
        if isinstance(content, str) and not content.strip():
            return True
        if isinstance(content, list) and not any(part for part in content):
            return True
    return False


class TestEmptyUserMessageStripped:

    def test_empty_user_in_middle_is_stripped_turn_proceeds(self):
        """The incident shape: an empty user message mid-history.

        A position-N empty user between real turns. The sanitizer drops it; the
        assembled payload carries NO empty-content message and the real turns
        survive, so the turn can proceed.
        """
        history = [
            {"role": "user", "content": "deploy the site"},
            {"role": "assistant", "content": "On it — running the deploy."},
            {"role": "user", "content": ""},           # position N — the poison
            {"role": "assistant", "content": "Deploy finished cleanly."},
            {"role": "user", "content": "Continue"},
        ]

        out = AIAgent._sanitize_api_messages(history)

        assert not _has_empty_content_message(out), (
            "dispatched payload must contain no empty-content message"
        )
        user_contents = [m.get("content") for m in out if m.get("role") == "user"]
        assert "deploy the site" in user_contents
        assert "Continue" in user_contents
        assert "" not in user_contents  # the empty user is gone

    def test_whitespace_only_user_is_stripped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "   \n\t  "},   # whitespace only
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert not _has_empty_content_message(out)
        assert all(
            not (m.get("role") == "user" and not str(m.get("content")).strip())
            for m in out
        )

    def test_trailing_empty_user_is_stripped(self):
        """A bare empty re-send at the tail (a 'Continue' that serialized empty)."""
        history = [
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "All green."},
            {"role": "user", "content": ""},
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert not _has_empty_content_message(out)
        assert out and (out[-1]["role"] != "user" or out[-1]["content"].strip())

    def test_none_content_user_is_stripped(self):
        history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": None},
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert not _has_empty_content_message(out)
        assert all(
            m.get("content") is not None for m in out if m.get("role") == "user"
        )

    def test_empty_user_between_two_users_leaves_no_adjacent_users(self):
        """Dropping an empty user between two users leaves no ``user, user`` seam."""
        history = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": ""},           # dropped
            {"role": "user", "content": "second"},
        ]
        out = AIAgent._sanitize_api_messages(history)

        roles = [m.get("role") for m in out]
        assert not any(
            roles[i] == "user" and roles[i + 1] == "user"
            for i in range(len(roles) - 1)
        )
        users = [m for m in out if m.get("role") == "user"]
        assert len(users) == 1
        assert "first" in users[0]["content"] and "second" in users[0]["content"]

    def test_multimodal_user_empty_text_but_image_is_kept(self):
        """A user turn with empty text but an image part is NOT empty — keep it."""
        history = [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ]
        out = AIAgent._sanitize_api_messages(history)
        users = [m for m in out if m.get("role") == "user"]
        assert len(users) == 1, "multimodal user turn must be preserved"

    def test_valid_history_is_untouched(self):
        """No empty messages → nothing dropped, order preserved."""
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert [m.get("content") for m in out] == ["a", "b", "c"]
