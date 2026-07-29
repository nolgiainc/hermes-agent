"""Regression tests for NOL-106 — colliding provider tool_call ids.

The admin agent hard-wedged when Moonshot/Kimi minted the same ``tool_call``
ids across two turns (a per-conversation counter that resets/overlaps,
especially across an interrupted turn). Hermes stored the ids verbatim and
relied on them for uniqueness at request assembly; the dedupe pass then
filtered the later turn's (duplicate) calls, leaving

    {"role": "assistant", "content": "", "tool_calls": []}

which Moonshot rejects with a non-retryable HTTP 400 ("the message at
position N with role 'assistant' must not be empty") — permanently wedging
the session.

Two independent defenses are covered:

  * Layer 1 — ``_namespace_tool_call_ids`` rewrites provider-minted ids to be
    globally unique per response at ingestion, so persisted history never
    stores colliding ids in the first place.

  * Layer 2 — ``_sanitize_api_messages`` guarantees it can NEVER emit an empty
    assistant message, and repairs PRE-EXISTING poisoned histories (written by
    the old code) so a wedged session un-sticks without manual DB surgery.
"""

import types

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers mirroring the normalized-response shape (agent.transports.types)
# ---------------------------------------------------------------------------

def _tool_call(call_id, name="search", arguments="{}", provider_data=None):
    """A minimal normalized ToolCall with a mutable ``id`` field."""
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
        type="function",
        provider_data=provider_data,
    )


def _response(tool_calls):
    return types.SimpleNamespace(tool_calls=list(tool_calls))


# ---------------------------------------------------------------------------
# Layer 1 — source-side uniqueness
# ---------------------------------------------------------------------------

class TestNamespaceProviderToolCallIds:

    def test_colliding_ids_across_turns_become_unique(self):
        """Two turns that reuse the SAME provider id must end up unique.

        Simulates Moonshot returning ``tool_call_201`` on two separate turns
        (the 5251/5258 shape). After namespacing, the persisted ids differ, so
        the assembly-time dedupe can never collapse the second turn's calls.
        """
        turn_a = _response([_tool_call("tool_call_201")])
        turn_b = _response([_tool_call("tool_call_201")])

        AIAgent._namespace_tool_call_ids(turn_a)
        AIAgent._namespace_tool_call_ids(turn_b)

        id_a = turn_a.tool_calls[0].id
        id_b = turn_b.tool_calls[0].id

        # Provider prefix preserved for debuggability, but globally unique.
        assert id_a != id_b
        assert id_a.startswith("tool_call_201")
        assert id_b.startswith("tool_call_201")

    def test_within_turn_ids_stay_unique_and_paired(self):
        """A parallel-call turn keeps one namespaced id per call."""
        turn = _response([
            _tool_call("tool_call_1", name="a"),
            _tool_call("tool_call_2", name="b"),
            _tool_call("tool_call_1", name="c"),  # provider re-used within turn
        ])
        AIAgent._namespace_tool_call_ids(turn)
        ids = [tc.id for tc in turn.tool_calls]
        assert len(set(ids)) == 3  # all distinct after namespacing

    def test_codex_style_ids_left_untouched(self):
        """Codex/Responses calls (call_id/response_item_id) are not rewritten."""
        turn = _response([
            _tool_call(
                "call_abc123",
                provider_data={"call_id": "call_abc123",
                               "response_item_id": "fc_abc123"},
            ),
        ])
        AIAgent._namespace_tool_call_ids(turn)
        assert turn.tool_calls[0].id == "call_abc123"

    def test_missing_or_empty_id_left_for_fallback(self):
        """No provider id → left alone (deterministic fallback owns that case)."""
        turn = _response([_tool_call(""), _tool_call(None)])
        rewritten = AIAgent._namespace_tool_call_ids(turn)
        assert rewritten == 0

    def test_idempotent_no_double_namespacing(self):
        """Running twice must not stack a second suffix."""
        turn = _response([_tool_call("tool_call_9")])
        AIAgent._namespace_tool_call_ids(turn)
        once = turn.tool_calls[0].id
        AIAgent._namespace_tool_call_ids(turn)
        assert turn.tool_calls[0].id == once

    def test_no_tool_calls_is_safe(self):
        assert AIAgent._namespace_tool_call_ids(_response([])) == 0
        assert AIAgent._namespace_tool_call_ids(types.SimpleNamespace()) == 0

    def test_namespaced_pair_survives_assembly(self):
        """End-to-end: after namespacing, a 2-turn history assembles cleanly.

        Builds the persisted-shape history the way the loop would (assistant
        tool_calls[].id + matching tool.tool_call_id, both carrying the SAME
        namespaced id), then runs the real assembly guard. No empty assistant,
        no orphan, both turns preserved.
        """
        turn_a = _response([_tool_call("tool_call_201")])
        turn_b = _response([_tool_call("tool_call_201")])
        AIAgent._namespace_tool_call_ids(turn_a)
        AIAgent._namespace_tool_call_ids(turn_b)
        id_a = turn_a.tool_calls[0].id
        id_b = turn_b.tool_calls[0].id

        history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": id_a, "type": "function",
                             "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": id_a, "content": "res-a"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": id_b, "type": "function",
                             "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": id_b, "content": "res-b"},
        ]

        out = AIAgent._sanitize_api_messages(history)

        assistants = [m for m in out if m.get("role") == "assistant"]
        assert len(assistants) == 2  # both turns survive — no collapse
        for a in assistants:
            assert a.get("tool_calls"), "assistant must keep its tool_calls"
        tool_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
        assert tool_ids == {id_a, id_b}


# ---------------------------------------------------------------------------
# Layer 2 — assembly guard sanitizes a PRE-EXISTING poisoned history
# ---------------------------------------------------------------------------

class TestAssemblyGuardEmptyAssistant:

    def _poisoned_history(self):
        """The exact 5251/5258 collision shape, as the OLD code persisted it.

        row 5251 declared tool_call_201/202/203 and got its results; row 5258
        re-declared the SAME ids (duplicate) and got duplicate results; then an
        interrupted-turn system note follows.
        """
        return [
            {"role": "user", "content": "task"},
            # row 5251 — the real turn.
            {"role": "assistant", "content": "",
             "tool_calls": [
                 {"id": "tool_call_201", "type": "function",
                  "function": {"name": "search", "arguments": "{}"}},
                 {"id": "tool_call_202", "type": "function",
                  "function": {"name": "read", "arguments": "{}"}},
                 {"id": "tool_call_203", "type": "function",
                  "function": {"name": "grep", "arguments": "{}"}},
             ]},
            {"role": "tool", "tool_call_id": "tool_call_201", "content": "A1"},
            {"role": "tool", "tool_call_id": "tool_call_202", "content": "A2"},
            {"role": "tool", "tool_call_id": "tool_call_203", "content": "A3"},
            # row 5258 — collides: SAME ids re-issued from the reset counter.
            {"role": "assistant", "content": "",
             "tool_calls": [
                 {"id": "tool_call_201", "type": "function",
                  "function": {"name": "search", "arguments": "{}"}},
                 {"id": "tool_call_202", "type": "function",
                  "function": {"name": "read", "arguments": "{}"}},
                 {"id": "tool_call_203", "type": "function",
                  "function": {"name": "grep", "arguments": "{}"}},
             ]},
            {"role": "tool", "tool_call_id": "tool_call_201", "content": "B1"},
            {"role": "tool", "tool_call_id": "tool_call_202", "content": "B2"},
            {"role": "tool", "tool_call_id": "tool_call_203", "content": "B3"},
            {"role": "user",
             "content": "[System note: pending tool outputs from an interrupted "
                        "turn. IGNORE those pending results.] continue"},
        ]

    def test_no_empty_assistant_in_output(self):
        out = AIAgent._sanitize_api_messages(self._poisoned_history())
        for m in out:
            if m.get("role") == "assistant":
                has_content = isinstance(m.get("content"), str) and m["content"].strip()
                has_tool_calls = bool(m.get("tool_calls"))
                assert has_content or has_tool_calls, (
                    "assembly must never emit an assistant with empty content "
                    "AND empty/absent tool_calls"
                )

    def test_no_orphan_tool_messages(self):
        out = AIAgent._sanitize_api_messages(self._poisoned_history())
        declared = set()
        for m in out:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    declared.add(AIAgent._get_tool_call_id_static(tc))
        for m in out:
            if m.get("role") == "tool":
                assert m.get("tool_call_id") in declared, "orphan tool result"

    def test_surviving_messages_are_the_correct_ones(self):
        """The FIRST id-holder (5251) and its results survive; the collided
        duplicate turn (5258) is dropped along with its duplicate results."""
        out = AIAgent._sanitize_api_messages(self._poisoned_history())

        assistants = [m for m in out if m.get("role") == "assistant"]
        assert len(assistants) == 1
        surviving_ids = [AIAgent._get_tool_call_id_static(tc)
                         for tc in assistants[0]["tool_calls"]]
        assert surviving_ids == ["tool_call_201", "tool_call_202", "tool_call_203"]

        # Exactly one tool result per declared call, from the first (A*) turn.
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 3
        assert {m["content"] for m in tool_msgs} == {"A1", "A2", "A3"}

        # The user turns (task + interrupted-turn note) are preserved.
        assert out[0] == {"role": "user", "content": "task"}
        assert out[-1]["role"] == "user"

    def test_reasoning_only_assistant_is_preserved(self):
        """A thinking-only assistant carries payload some providers require
        (kimi/deepseek reasoning replay) and must NOT be dropped by the guard."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert any(m.get("role") == "assistant" for m in out)
