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

  * Layer 1 — ``_namespace_tool_call_ids`` de-collides provider-minted ids at
    ingestion: an id that duplicates one already in the in-context history (or
    an earlier call in the same response) is rewritten to be unique. Unique-id
    providers (incl. strict-format ones like Mistral, and Anthropic's
    interleaved-thinking path) are left untouched.

  * Layer 2 — ``_sanitize_api_messages`` guarantees it can NEVER emit an empty
    assistant message, repairs PRE-EXISTING poisoned histories (written by the
    old code), and never leaves two adjacent user turns behind.
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


def _response(tool_calls, anthropic_content_blocks=None):
    return types.SimpleNamespace(
        tool_calls=list(tool_calls),
        anthropic_content_blocks=anthropic_content_blocks,
    )


# ---------------------------------------------------------------------------
# Layer 1 — collision-triggered source-side uniqueness
# ---------------------------------------------------------------------------

class TestNamespaceProviderToolCallIds:

    def test_colliding_ids_across_turns_become_unique(self):
        """A later turn that reuses an id already in history is de-collided.

        Simulates Moonshot returning ``tool_call_201`` on two separate turns
        (the 5251/5258 shape). ``existing_ids`` carries the earlier turn's id,
        so the second turn's duplicate is rewritten — the assembly-time dedupe
        can never collapse it.
        """
        turn_a = _response([_tool_call("tool_call_201")])
        AIAgent._namespace_tool_call_ids(turn_a, set())
        id_a = turn_a.tool_calls[0].id
        assert id_a == "tool_call_201"  # first sighting kept verbatim

        turn_b = _response([_tool_call("tool_call_201")])  # provider reuses id
        AIAgent._namespace_tool_call_ids(turn_b, {id_a})
        id_b = turn_b.tool_calls[0].id

        assert id_b != id_a
        assert id_b.startswith("tool_call_201")  # prefix preserved for debugging

    def test_within_turn_duplicate_ids_de_collided(self):
        """A parallel-call turn that repeats an id gets one unique id per call."""
        turn = _response([
            _tool_call("tool_call_1", name="a"),
            _tool_call("tool_call_2", name="b"),
            _tool_call("tool_call_1", name="c"),  # provider re-used within turn
        ])
        AIAgent._namespace_tool_call_ids(turn, set())
        ids = [tc.id for tc in turn.tool_calls]
        assert len(set(ids)) == 3  # all distinct after de-collision
        assert ids[0] == "tool_call_1"  # first kept verbatim
        assert ids[2].startswith("tool_call_1") and ids[2] != "tool_call_1"

    def test_unique_ids_are_left_untouched(self):
        """Well-behaved providers (unique ids) are NOT reformatted.

        Guards the Mistral case: a strict provider requiring a fixed id schema
        must keep its verbatim id across turns, because nothing collides.
        """
        turn_a = _response([_tool_call("abc123def")])  # Mistral-style 9-char
        AIAgent._namespace_tool_call_ids(turn_a, set())
        assert turn_a.tool_calls[0].id == "abc123def"

        turn_b = _response([_tool_call("xyz789ghi")])
        rewritten = AIAgent._namespace_tool_call_ids(turn_b, {"abc123def"})
        assert rewritten == 0
        assert turn_b.tool_calls[0].id == "xyz789ghi"  # untouched, format intact

    def test_anthropic_interleaved_thinking_skipped(self):
        """Turns carrying anthropic_content_blocks are skipped entirely.

        The authoritative tool_use ids live in the verbatim ordered block list
        (which this function does not rewrite); rewriting the parallel
        tool_calls ids would desync them and 400 Anthropic on replay.
        """
        turn = _response(
            [_tool_call("toolu_01", name="a"), _tool_call("toolu_01", name="b")],
            anthropic_content_blocks=[{"type": "tool_use", "id": "toolu_01"}],
        )
        rewritten = AIAgent._namespace_tool_call_ids(turn, {"toolu_01"})
        assert rewritten == 0
        assert [tc.id for tc in turn.tool_calls] == ["toolu_01", "toolu_01"]

    def test_codex_style_ids_left_untouched(self):
        """Codex/Responses calls (call_id/response_item_id) are not rewritten."""
        turn = _response([
            _tool_call(
                "call_abc123",
                provider_data={"call_id": "call_abc123",
                               "response_item_id": "fc_abc123"},
            ),
        ])
        AIAgent._namespace_tool_call_ids(turn, {"call_abc123"})
        assert turn.tool_calls[0].id == "call_abc123"

    def test_missing_or_empty_id_left_for_fallback(self):
        """No provider id → left alone (deterministic fallback owns that case)."""
        turn = _response([_tool_call(""), _tool_call(None)])
        assert AIAgent._namespace_tool_call_ids(turn, set()) == 0

    def test_idempotent_no_double_namespacing(self):
        """Re-running on the already-de-collided object does not stack a suffix."""
        turn = _response([_tool_call("tool_call_9")])
        AIAgent._namespace_tool_call_ids(turn, {"tool_call_9"})  # collides -> rewrite
        once = turn.tool_calls[0].id
        assert once.startswith("tool_call_9") and once != "tool_call_9"
        AIAgent._namespace_tool_call_ids(turn, {"tool_call_9"})  # id no longer collides
        assert turn.tool_calls[0].id == once

    def test_no_tool_calls_is_safe(self):
        assert AIAgent._namespace_tool_call_ids(_response([]), set()) == 0
        assert AIAgent._namespace_tool_call_ids(types.SimpleNamespace(), set()) == 0

    def test_de_collided_pair_survives_assembly(self):
        """End-to-end: after de-collision, a 2-turn history assembles cleanly.

        Builds the persisted-shape history the way the loop would (assistant
        tool_calls[].id + matching tool.tool_call_id, both carrying the SAME
        de-collided id), then runs the real assembly guard. No empty assistant,
        no orphan, both turns preserved.
        """
        turn_a = _response([_tool_call("tool_call_201")])
        AIAgent._namespace_tool_call_ids(turn_a, set())
        id_a = turn_a.tool_calls[0].id
        turn_b = _response([_tool_call("tool_call_201")])
        AIAgent._namespace_tool_call_ids(turn_b, {id_a})
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
        """Every surviving turn is provider-safe and correctly paired.

        Mechanism (post upstream sync b0ab2e163a): upstream's
        ``_dedupe_tool_call_ids`` now tracks OUTSTANDING (unanswered) calls
        rather than every id ever seen — llama.cpp mints one constant id for
        every call it ever returns, so a seen-once-drop-forever rule deleted
        every tool result after the first (upstream's
        ``test_sanitize_dedup_pass_rearms_constant_llamacpp_id``). An id
        re-issued AFTER its earlier call was answered is therefore a legitimate
        new call: row 5258 keeps its tool_calls and its own results instead of
        being collapsed, and no empty assistant is produced for this shape.
        The assembly guard stays the backstop for any payload-less assistant
        that does reach it (``test_guard_drops_payloadless_final_assistant``).
        """
        out = AIAgent._sanitize_api_messages(self._poisoned_history())

        assistants = [m for m in out if m.get("role") == "assistant"]
        assert len(assistants) == 2
        for a in assistants:
            assert [AIAgent._get_tool_call_id_static(tc) for tc in a["tool_calls"]] == [
                "tool_call_201", "tool_call_202", "tool_call_203",
            ]

        # Each assistant is immediately followed by exactly the 3 results that
        # answer it, in order: A* pair with 5251, B* with 5258 — positional
        # pairing, never cross-wired.
        # Positional (not ``list.index``) lookup: both assistant dicts compare equal.
        idx_a, idx_b = [i for i, m in enumerate(out) if m.get("role") == "assistant"]
        assert [m["content"] for m in out[idx_a + 1:idx_a + 4]] == ["A1", "A2", "A3"]
        assert [m["content"] for m in out[idx_b + 1:idx_b + 4]] == ["B1", "B2", "B3"]
        assert all(m.get("role") == "tool" for m in out[idx_a + 1:idx_a + 4] + out[idx_b + 1:idx_b + 4])
        assert len([m for m in out if m.get("role") == "tool"]) == 6

        # The user turns (task + interrupted-turn note) are preserved.
        assert out[0] == {"role": "user", "content": "task"}
        assert out[-1]["role"] == "user"

    def test_guard_drops_payloadless_final_assistant(self):
        """The assembly guard's assistant branch: a payload-less assistant that
        nothing upstream of it repairs. ``repair_empty_non_final_messages``
        deliberately skips the FINAL message, so an interrupted turn that left
        ``{"role": "assistant", "content": ""}`` at the tail (with or without an
        empty ``tool_calls`` array) would otherwise ship exactly the Moonshot
        "must not be empty" 400 shape. The guard drops it; the real turns stay.
        """
        for tail in (
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": []},
        ):
            out = AIAgent._sanitize_api_messages([{"role": "user", "content": "hi"}, tail])
            assert out == [{"role": "user", "content": "hi"}], out

    def test_reasoning_only_assistant_is_preserved(self):
        """A thinking-only assistant carries payload some providers require
        (kimi/deepseek reasoning replay) and must NOT be dropped by the guard."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
        ]
        out = AIAgent._sanitize_api_messages(history)
        assert any(m.get("role") == "assistant" for m in out)

    def test_user_empty_assistant_user_is_healed_without_empty_content(self):
        """A ``user -> empty assistant -> user`` history must not reach the
        provider carrying an empty-content turn or a same-role adjacency.

        Mechanism (post upstream sync): ``repair_empty_non_final_messages``
        substitutes an honest ``[response interrupted]`` placeholder into the
        empty non-final assistant turn — preserving role alternation — rather
        than dropping it and merging the newly-adjacent users. Either shape is
        provider-safe; the invariants below are what actually matter.
        """
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},  # no payload — healed
            {"role": "user", "content": "second"},
        ]
        out = AIAgent._sanitize_api_messages(history)

        # No empty-content message survives to the wire.
        for m in out:
            if m.get("role") in ("user", "assistant") and not m.get("tool_calls"):
                content = m.get("content")
                assert isinstance(content, str) and content.strip(), (
                    f"empty-content {m.get('role')} turn reached the payload"
                )
        # No same-role user adjacency for strict providers.
        roles = [m.get("role") for m in out]
        assert not any(
            roles[i] == "user" and roles[i + 1] == "user"
            for i in range(len(roles) - 1)
        )
        # Both real user turns survive, in order.
        user_contents = [m["content"] for m in out if m.get("role") == "user"]
        assert user_contents == ["first", "second"]
        # The healed assistant turn carries the documented placeholder.
        assistants = [m for m in out if m.get("role") == "assistant"]
        assert len(assistants) == 1
        assert assistants[0]["content"] == "[response interrupted]"
