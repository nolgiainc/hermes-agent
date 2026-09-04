"""Tests for the native Google AI Studio Gemini adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class DummyResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload











def test_parallel_tool_results_merge_into_one_user_content():
    """Gemini requires strict user/model alternation; two consecutive `user`
    contents are rejected with HTTP 400. Parallel tool calls produce two tool
    results in a row, so their functionResponses must be grouped into a single
    user content instead of two consecutive ones."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "Read a.txt and b.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "AAA"},
        {"role": "tool", "tool_call_id": "call_2", "content": "BBB"},
    ]

    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]

    # No two adjacent contents may share a role.
    assert all(roles[i] != roles[i - 1] for i in range(1, len(roles))), roles
    assert roles == ["user", "model", "user"]

    # Both parallel functionResponses land in the single trailing user content.
    response_parts = [
        p for p in contents[2]["parts"] if "functionResponse" in p
    ]
    outputs = [p["functionResponse"]["response"]["output"] for p in response_parts]
    assert outputs == ["AAA", "BBB"]


def test_consecutive_user_messages_merge_for_gemini_alternation():
    """Back-to-back user messages must also be merged, not sent as two
    consecutive user contents."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]
    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model"], roles


def test_file_parts_translate_to_native_gemini_video_parts():
    from agent.gemini_native_adapter import _extract_multimodal_parts

    encoded = "AAEC"
    parts = _extract_multimodal_parts([
        {
            "type": "file",
            "file": {
                "file_data": f"data:video/mp4;base64,{encoded}",
                "format": "video/mp4",
                "video_metadata": {"fps": 1},
                "detail": "low",
            },
        },
        {
            "type": "file",
            "file": {
                "file_data": "https://cdn.example.com/clip.webm",
                "format": "video/webm",
            },
        },
    ])

    assert parts == [
        {
            "inlineData": {"mimeType": "video/mp4", "data": encoded},
            "videoMetadata": {"fps": 1},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_LOW"},
        },
        {
            "fileData": {
                "mimeType": "video/webm",
                "fileUri": "https://cdn.example.com/clip.webm",
            },
        },
    ]


def test_translate_native_response_surfaces_reasoning_and_tool_calls():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "thinking..."},
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.reasoning == "thinking..."
    assert choice.message.tool_calls[0].function.name == "search"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"q": "hermes"}


def test_native_client_uses_x_goog_api_key_and_native_models_endpoint(monkeypatch):
    from agent.gemini_native_adapter import GeminiNativeClient

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            recorded["headers"] = headers
            return DummyResponse(
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "hello"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            )

        def close(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: DummyHTTP())

    client = GeminiNativeClient(api_key="AIza-test", base_url="https://generativelanguage.googleapis.com/v1beta")
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert recorded["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert recorded["headers"]["x-goog-api-key"] == "AIza-test"
    assert "Authorization" not in recorded["headers"]
    assert response.choices[0].message.content == "hello"








def test_native_client_accepts_injected_http_client():
    from agent.gemini_native_adapter import GeminiNativeClient

    injected = SimpleNamespace(close=lambda: None)
    client = GeminiNativeClient(api_key="AIza-test", http_client=injected)
    assert client._http is injected


def test_native_client_rejects_empty_api_key_with_actionable_message():
    """Empty/whitespace api_key must raise at construction, not produce a cryptic
    Google GFE 'Error 400 (Bad Request)!!1' HTML page on the first request."""
    from agent.gemini_native_adapter import GeminiNativeClient

    for bad in ("", "   ", None):
        with pytest.raises(RuntimeError) as excinfo:
            GeminiNativeClient(api_key=bad)  # type: ignore[arg-type]
        msg = str(excinfo.value)
        assert "GOOGLE_API_KEY" in msg and "GEMINI_API_KEY" in msg
        assert "aistudio.google.com" in msg


@pytest.mark.asyncio
async def test_async_native_client_streams_without_requiring_async_iterator_from_sync_client():
    from agent.gemini_native_adapter import AsyncGeminiNativeClient

    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason=None)])
    sync_stream = iter([chunk])

    def _advance(iterator):
        try:
            return False, next(iterator)
        except StopIteration:
            return True, None

    sync_client = SimpleNamespace(
        api_key="AIza-test",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: sync_stream)),
        _advance_stream_iterator=_advance,
        close=lambda: None,
    )

    async_client = AsyncGeminiNativeClient(sync_client)
    stream = await async_client.chat.completions.create(stream=True)
    collected = []
    async for item in stream:
        collected.append(item)
    assert collected == [chunk]


def test_stream_event_translation_emits_tool_call_delta_with_stable_index():
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}}
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    first = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    second = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)

    assert first[0].choices[0].delta.tool_calls[0].index == 0
    assert second[0].choices[0].delta.tool_calls[0].index == 0
    assert first[0].choices[0].delta.tool_calls[0].id == second[0].choices[0].delta.tool_calls[0].id
    assert first[0].choices[0].delta.tool_calls[0].function.arguments == '{"q": "abc"}'
    assert second[0].choices[0].delta.tool_calls[0].function.arguments == ""
    assert first[-1].choices[0].finish_reason == "tool_calls"










# ---------------------------------------------------------------------------
# X-Goog-Api-Client header tests
# ---------------------------------------------------------------------------










@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.8-flash", True),
        ("google/gemini-3.8-flash", True),
        ("models/gemini-3.8-flash", True),
        ("gemini-3.8-flash-lite", True),
        ("gemini-3.9-flash", True),
        ("gemini-4-flash", True),
        ("gemini-4.0-flash", True),
        ("gemini-4.2-flash-preview", True),
        ("gemini-3.7-flash", False),
        ("gemini-3.6-flash", False),
        ("gemini-3.5-flash-lite", False),
        ("gemini-3-flash-preview", False),
        ("gemini-2.5-flash", False),
        ("gemini-3.8-pro", False),
        ("gemini-3.1-pro-preview", False),
        ("gemma-3-27b-it", False),
        ("", False),
    ],
)
def test_is_gemini_flash_38_or_later(model, expected):
    """Version comparison, not a prefix match: true from 3.8 Flash onward
    (3.9, 4.x, -lite / -preview variants), false for 3.7 and earlier and
    for every Pro model."""
    from agent.gemini_native_adapter import is_gemini_flash_38_or_later

    assert is_gemini_flash_38_or_later(model) is expected


def test_build_gemini_request_drops_sampling_params_for_gemini_38_flash():
    """Google's gemini-3.8-flash guidance: strip temperature / top_p / top_k —
    the model tunes its own sampling. Earlier Flash generations keep them."""
    from agent.gemini_native_adapter import build_gemini_request

    messages = [{"role": "user", "content": "Hi"}]
    strict = build_gemini_request(model="gemini-3.8-flash", messages=messages, temperature=0.7, top_p=0.9)
    assert "temperature" not in strict["generationConfig"]
    assert "topP" not in strict["generationConfig"]
    # The output-ceiling default is unaffected by the gate.
    assert strict["generationConfig"]["maxOutputTokens"] > 0

    legacy = build_gemini_request(model="gemini-3.6-flash", messages=messages, temperature=0.7, top_p=0.9)
    assert legacy["generationConfig"]["temperature"] == 0.7
    assert legacy["generationConfig"]["topP"] == 0.9


def test_build_gemini_request_clamps_minimal_thinking_level_for_gemini_38_flash():
    """gemini-3.8-flash rejects thinkingLevel "minimal" (low/medium/high only);
    a caller-supplied "minimal" lands as "low" there and is passed through
    untouched on earlier generations. Documented levels are never rewritten."""
    from agent.gemini_native_adapter import build_gemini_request

    messages = [{"role": "user", "content": "Hi"}]
    minimal = {"includeThoughts": True, "thinkingLevel": "minimal"}

    strict = build_gemini_request(model="gemini-3.8-flash", messages=messages, thinking_config=minimal)
    assert strict["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "low"}

    legacy = build_gemini_request(model="gemini-3.6-flash", messages=messages, thinking_config=minimal)
    assert legacy["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"

    high = build_gemini_request(
        model="gemini-3.8-flash",
        messages=messages,
        thinking_config={"includeThoughts": True, "thinkingLevel": "high"},
    )
    assert high["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_native_client_strips_sampling_params_for_gemini_38_flash(monkeypatch):
    """End to end through the client: the caller's model id must reach
    build_gemini_request so the gate fires on the wire request, and the
    aggregator-style ``google/`` prefix must not defeat it."""
    from agent.gemini_native_adapter import GeminiNativeClient

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            return DummyResponse(
                payload={
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            )

        def close(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: DummyHTTP())

    client = GeminiNativeClient(api_key="AIza-test", base_url="https://generativelanguage.googleapis.com/v1beta")
    client.chat.completions.create(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
        top_p=0.8,
    )

    assert recorded["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert "temperature" not in recorded["json"]["generationConfig"]
    assert "topP" not in recorded["json"]["generationConfig"]
