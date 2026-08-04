"""Tests for the AWS Bedrock Converse API adapter.

Covers:
  - AWS credential detection and region resolution
  - Message format conversion (OpenAI → Converse and back)
  - Tool definition conversion
  - Response normalization (non-streaming and streaming)
  - Model discovery with caching
  - Edge cases: empty messages, consecutive roles, image content
"""

import json
import logging
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def _mock_botocore_session(*, return_value=None, side_effect=None):
    """Patch botocore.session even when botocore is not installed."""
    botocore_mod = ModuleType("botocore")
    session_mod = ModuleType("botocore.session")
    session_mod.get_session = MagicMock(return_value=return_value, side_effect=side_effect)
    botocore_mod.session = session_mod
    with patch.dict("sys.modules", {"botocore": botocore_mod, "botocore.session": session_mod}):
        yield session_mod.get_session


@contextmanager
def _mock_botocore_config():
    """Patch botocore.config even when botocore is not installed.

    ``_get_bedrock_runtime_client`` imports ``botocore.config.Config`` to build
    a budgeted client, but boto3/botocore sit behind the optional ``bedrock``
    extra and are absent from CI. The cache-keying contract those tests assert
    is botocore-independent, so stub the one symbol the code path needs rather
    than skip the assertions everywhere the extra is missing. Tests that assert
    on real ``Config`` semantics use ``pytest.importorskip`` instead.
    """
    class _StubConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    botocore_mod = ModuleType("botocore")
    config_mod = ModuleType("botocore.config")
    config_mod.Config = _StubConfig
    botocore_mod.config = config_mod
    with patch.dict("sys.modules", {"botocore": botocore_mod, "botocore.config": config_mod}):
        yield _StubConfig


# ---------------------------------------------------------------------------
# AWS credential detection
# ---------------------------------------------------------------------------

class TestResolveAwsAuthEnvVar:
    """Test AWS credential environment variable detection.

    Mirrors OpenClaw's resolveAwsSdkEnvVarName() priority order.
    """



    def test_requires_both_access_key_and_secret(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Only access key, no secret → should not match
        env = {"AWS_ACCESS_KEY_ID": "AKIA..."}
        assert resolve_aws_auth_env_var(env) != "AWS_ACCESS_KEY_ID"




    def test_returns_none_when_no_aws_auth(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Mock botocore to return no credentials (covers EC2 IMDS fallback)
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert resolve_aws_auth_env_var({}) is None



class TestHasAwsCredentials:
    def test_true_with_profile(self):
        from agent.bedrock_adapter import has_aws_credentials
        assert has_aws_credentials({"AWS_PROFILE": "default"}) is True

    def test_false_with_empty_env(self):
        from agent.bedrock_adapter import has_aws_credentials
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert has_aws_credentials({}) is False


class TestResolveBedrocRegion:
    def test_prefers_aws_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        env = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-west-2"}
        assert resolve_bedrock_region(env) == "eu-west-1"


    def test_defaults_to_us_east_1(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = None
        with _mock_botocore_session(return_value=mock_session):
            assert resolve_bedrock_region({}) == "us-east-1"




# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

class TestConvertToolsToConverse:
    """Test OpenAI → Bedrock Converse tool definition conversion."""

    def test_converts_single_tool(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
        }]
        result = convert_tools_to_converse(tools)
        assert len(result) == 1
        spec = result[0]["toolSpec"]
        assert spec["name"] == "read_file"
        assert spec["description"] == "Read a file from disk"
        assert spec["inputSchema"]["json"]["type"] == "object"
        assert "path" in spec["inputSchema"]["json"]["properties"]


    def test_empty_tools(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        assert convert_tools_to_converse([]) == []
        assert convert_tools_to_converse(None) == []



# ---------------------------------------------------------------------------
# Message conversion: OpenAI → Converse
# ---------------------------------------------------------------------------

class TestConvertMessagesToConverse:
    """Test OpenAI message format → Bedrock Converse format conversion."""

    def test_extracts_system_prompt(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assert system is not None
        assert len(system) == 1
        assert system[0]["text"] == "You are a helpful assistant."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


    def test_assistant_with_tool_calls(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "I'll read that file.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/test.txt"}',
                    },
                }],
            },
        ]
        system, msgs = convert_messages_to_converse(messages)
        # 3 messages: user, assistant, trailing user (Converse requires last=user)
        assert len(msgs) == 3
        assistant_content = msgs[1]["content"]
        # Should have text block + toolUse block
        assert any("text" in b for b in assistant_content)
        tool_use_blocks = [b for b in assistant_content if "toolUse" in b]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["toolUse"]["name"] == "read_file"
        assert tool_use_blocks[0]["toolUse"]["toolUseId"] == "call_123"
        assert tool_use_blocks[0]["toolUse"]["input"] == {"path": "/tmp/test.txt"}

    def test_tool_result_becomes_user_message(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Read it"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents here"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        # Tool result should be in a user-role message
        tool_result_msg = [m for m in msgs if m["role"] == "user" and any(
            "toolResult" in b for b in m["content"]
        )]
        assert len(tool_result_msg) == 1
        tr = [b for b in tool_result_msg[0]["content"] if "toolResult" in b][0]
        assert tr["toolResult"]["toolUseId"] == "call_1"
        assert tr["toolResult"]["content"][0]["text"] == "file contents here"





    def test_empty_content_gets_placeholder(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [{"role": "user", "content": ""}]
        system, msgs = convert_messages_to_converse(messages)
        # Empty string should get a space placeholder
        assert msgs[0]["content"][0]["text"].strip() != "" or msgs[0]["content"][0]["text"] == " "




# ---------------------------------------------------------------------------
# Response normalization: Converse → OpenAI
# ---------------------------------------------------------------------------

class TestNormalizeConverseResponse:
    """Test Bedrock Converse response → OpenAI format conversion."""

    def test_text_response(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, world!"}],
                },
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_cache_tokens_folded_into_prompt_tokens(self):
        """Converse's inputTokens excludes cache read/write tokens (unlike
        OpenAI's prompt_tokens). normalize_converse_response must add them
        back into prompt_tokens/total_tokens and surface the Anthropic-named
        fields so normalize_usage() picks them up via its existing fallback."""
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 20,
                "cacheReadInputTokens": 900,
                "cacheWriteInputTokens": 300,
            },
        }
        result = normalize_converse_response(response)
        assert result.usage.prompt_tokens == 50 + 900 + 300
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 50 + 900 + 300 + 20
        assert result.usage.cache_read_input_tokens == 900
        assert result.usage.cache_creation_input_tokens == 300

    def test_tool_use_response(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "I'll read that file."},
                        {
                            "toolUse": {
                                "toolUseId": "call_abc",
                                "name": "read_file",
                                "input": {"path": "/tmp/test.txt"},
                            },
                        },
                    ],
                },
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 15},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].message.content == "I'll read that file."
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].function.name == "read_file"
        assert json.loads(tool_calls[0].function.arguments) == {"path": "/tmp/test.txt"}






# ---------------------------------------------------------------------------
# Streaming response normalization
# ---------------------------------------------------------------------------

class TestNormalizeConverseStreamEvents:
    """Test Bedrock ConverseStream event → OpenAI format conversion."""

    def test_text_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": ", world!"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ]}
        result = normalize_converse_stream_events(events)
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3

    def test_tool_use_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {
                "toolUse": {"toolUseId": "call_1", "name": "read_file"},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '{"path":'},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '"/tmp/f"}'},
            }}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 8}}},
        ]}
        result = normalize_converse_stream_events(events)
        assert result.choices[0].finish_reason == "tool_calls"
        tc = result.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].id == "call_1"
        assert tc[0].function.name == "read_file"
        assert json.loads(tc[0].function.arguments) == {"path": "/tmp/f"}




# ---------------------------------------------------------------------------
# build_converse_kwargs
# ---------------------------------------------------------------------------

class TestBuildConverseKwargs:
    """Test the high-level kwargs builder for Converse API calls."""

    def test_basic_kwargs(self):
        from agent.bedrock_adapter import build_converse_kwargs
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-6-20250514-v1:0",
            messages=messages,
            max_tokens=1024,
        )
        assert kwargs["modelId"] == "anthropic.claude-sonnet-4-6-20250514-v1:0"
        assert kwargs["inferenceConfig"]["maxTokens"] == 1024
        assert kwargs["system"] is not None
        assert len(kwargs["messages"]) >= 1

    def test_includes_tools(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {
            "name": "test", "description": "Test", "parameters": {},
        }}]
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" in kwargs
        assert len(kwargs["toolConfig"]["tools"]) == 1









    def test_cache_point_added_for_supported_model(self):
        """Claude and Nova on the Converse path get cachePoint markers on
        system, tools, and the message before the newest turn."""
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {
            "name": "test", "description": "Test", "parameters": {},
        }}]
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        kwargs = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-6-20250514-v1:0",
            messages=messages,
            tools=tools,
        )
        assert kwargs["system"][-1] == {"cachePoint": {"type": "default"}}
        assert kwargs["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
        # Second-to-last converse message (the assistant "Reply" turn) carries
        # the checkpoint; the newest "Second" turn does not.
        marked = kwargs["messages"][-2]["content"]
        assert marked[-1] == {"cachePoint": {"type": "default"}}
        assert kwargs["messages"][-1]["content"][-1] != {"cachePoint": {"type": "default"}}

    def test_no_cache_point_for_unsupported_model(self):
        from agent.bedrock_adapter import build_converse_kwargs
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        kwargs = build_converse_kwargs(model="meta.llama3-70b-instruct-v1:0", messages=messages)
        assert {"cachePoint": {"type": "default"}} not in kwargs["system"]
        for m in kwargs["messages"]:
            assert {"cachePoint": {"type": "default"}} not in m["content"]


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

class TestDiscoverBedrockModels:
    """Test Bedrock model discovery with mocked AWS API calls."""




    def test_provider_filter(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-v2",
                    "modelName": "Claude v2",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
                {
                    "modelId": "amazon.titan-text",
                    "modelName": "Titan",
                    "providerName": "Amazon",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1", provider_filter=["anthropic"])

        assert len(models) == 1
        assert models[0]["id"] == "anthropic.claude-v2"

    def test_caches_results(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [{
                "modelId": "test-model",
                "modelName": "Test",
                "providerName": "Test",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "modelLifecycle": {"status": "ACTIVE"},
            }],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            first = discover_bedrock_models("us-east-1")
            second = discover_bedrock_models("us-east-1")

        # Should only call the API once (second call uses cache)
        assert mock_client.list_foundation_models.call_count == 1
        assert first == second



    def test_handles_api_error_gracefully(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        with patch("agent.bedrock_adapter._get_bedrock_control_client", side_effect=Exception("No creds")):
            models = discover_bedrock_models("us-east-1")

        assert models == []


class TestExtractProviderFromArn:
    def test_extracts_anthropic(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"
        assert _extract_provider_from_arn(arn) == "anthropic"


    def test_returns_empty_for_invalid_arn(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        assert _extract_provider_from_arn("not-an-arn") == ""
        assert _extract_provider_from_arn("") == ""


# ---------------------------------------------------------------------------
# Client cache management
# ---------------------------------------------------------------------------

class TestClientCache:
    def test_reset_clears_caches(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            _bedrock_control_client_cache,
            reset_client_cache,
        )
        _bedrock_runtime_client_cache["test"] = "dummy"
        _bedrock_control_client_cache["test"] = "dummy"
        reset_client_cache()
        assert len(_bedrock_runtime_client_cache) == 0
        assert len(_bedrock_control_client_cache) == 0


# ---------------------------------------------------------------------------
# Streaming with callbacks
# ---------------------------------------------------------------------------

class TestStreamConverseWithCallbacks:
    """Test real-time streaming with delta callbacks."""

    def test_cache_tokens_folded_into_prompt_tokens(self):
        """The streaming path must fold cacheRead/WriteInputTokens into
        prompt_tokens the same way the non-streaming path does (see
        TestNormalizeConverseResponse.test_cache_tokens_folded_into_prompt_tokens)."""
        from agent.bedrock_adapter import stream_converse_with_callbacks
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {
                "inputTokens": 50,
                "outputTokens": 20,
                "cacheReadInputTokens": 900,
                "cacheWriteInputTokens": 300,
            }}},
        ]}
        result = stream_converse_with_callbacks(events)
        assert result.usage.prompt_tokens == 50 + 900 + 300
        assert result.usage.total_tokens == 50 + 900 + 300 + 20
        assert result.usage.cache_read_input_tokens == 900
        assert result.usage.cache_creation_input_tokens == 300


    def test_text_deltas_suppressed_when_tool_use_present(self):
        """Text deltas should NOT fire when tool_use blocks are present."""
        from agent.bedrock_adapter import stream_converse_with_callbacks
        deltas = []
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Let me check."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {
                "toolUse": {"toolUseId": "c1", "name": "search"},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {
                "toolUse": {"input": '{"q":"test"}'},
            }}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}
        result = stream_converse_with_callbacks(
            events, on_text_delta=lambda t: deltas.append(t),
        )
        # Text delta for "Let me check." should fire (before tool_use was seen)
        assert "Let me check." in deltas
        # But the result should still have both text and tool calls
        assert result.choices[0].message.content == "Let me check."
        assert len(result.choices[0].message.tool_calls) == 1





# ---------------------------------------------------------------------------
# Guardrail config in build_converse_kwargs
# ---------------------------------------------------------------------------

class TestGuardrailConfig:
    """Test that guardrail configuration is correctly passed through."""

    def test_guardrail_included_in_kwargs(self):
        from agent.bedrock_adapter import build_converse_kwargs
        guardrail = {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1",
            "streamProcessingMode": "async",
            "trace": "enabled",
        }
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config=guardrail,
        )
        assert kwargs["guardrailConfig"] == guardrail

    def test_no_guardrail_when_none(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config=None,
        )
        assert "guardrailConfig" not in kwargs

    def test_no_guardrail_when_empty_dict(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config={},
        )
        # Empty dict is falsy, should not be included
        assert "guardrailConfig" not in kwargs


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestBedrockErrorClassification:
    """Test Bedrock-specific error classification."""

    def test_context_overflow_validation_exception(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error(
            "ValidationException: input is too long for model"
        ) == "context_overflow"









class TestBedrockContextLength:
    """Test Bedrock model context length lookup."""











    def test_unknown_model_gets_default(self):
        from agent.bedrock_adapter import get_bedrock_context_length, BEDROCK_DEFAULT_CONTEXT_LENGTH
        assert get_bedrock_context_length("unknown.model-v1:0") == BEDROCK_DEFAULT_CONTEXT_LENGTH



    def test_no_region_skips_probe_uses_table(self):
        # Default call (no region) must NOT hit the network — returns the
        # static table value.  Guards backward compatibility for callers that
        # still invoke get_bedrock_context_length(model_id) with one arg.
        from agent.bedrock_adapter import get_bedrock_context_length
        with patch("agent.bedrock_adapter.probe_bedrock_context_length") as mock_probe:
            assert get_bedrock_context_length("anthropic.claude-opus-4-6") == 1_000_000
            mock_probe.assert_not_called()


class TestBedrockContextProbe:
    """Test the live context-window probe that reads the real window from
    Bedrock's 'prompt is too long' validation error."""

    def _client_raising(self, message):
        client = MagicMock()
        client.converse.side_effect = Exception(message)
        return client



    def test_probe_returns_none_when_client_unavailable(self):
        from agent.bedrock_adapter import probe_bedrock_context_length
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   side_effect=RuntimeError("boto3 missing")):
            assert probe_bedrock_context_length("any.model", "eu-central-1") is None

    def test_probe_result_beats_static_table(self):
        # A successful probe (1M) must override the stale table value (200K
        # via the 'anthropic.claude-opus-4' substring match).
        from agent.bedrock_adapter import get_bedrock_context_length
        err = "prompt is too long: 5000032 tokens > 1000000 maximum"
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   return_value=self._client_raising(err)):
            assert get_bedrock_context_length(
                "eu.anthropic.claude-opus-4-8",
                region="eu-central-1") == 1_000_000



# ---------------------------------------------------------------------------
# Tool-calling capability detection
# ---------------------------------------------------------------------------

class TestModelSupportsToolUse:
    """Test non-tool-calling model detection."""

    def test_claude_supports_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.anthropic.claude-sonnet-4-6") is True




    def test_deepseek_r1_no_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.deepseek.r1-v1:0") is False






class TestBuildConverseKwargsToolStripping:
    """Test that tools are stripped for non-tool-calling models."""

    def test_tools_included_for_claude(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {"name": "test", "description": "t", "parameters": {}}}]
        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" in kwargs

    def test_tools_stripped_for_deepseek_r1(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {"name": "test", "description": "t", "parameters": {}}}]
        kwargs = build_converse_kwargs(
            model="us.deepseek.r1-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" not in kwargs


# ---------------------------------------------------------------------------
# Dual-path model routing
# ---------------------------------------------------------------------------

class TestIsAnthropicBedrockModel:
    """Test Claude model detection for dual-path routing."""

    def test_us_claude_sonnet(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.anthropic.claude-sonnet-4-6") is True



    def test_nova_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.amazon.nova-pro-v1:0") is False





    def test_au_inference_profile(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("au.anthropic.claude-haiku-4-5-20251001-v1:0") is True
        assert is_anthropic_bedrock_model("au.anthropic.claude-sonnet-4-6") is True



class TestEmptyTextBlockFix:
    """Test that empty/whitespace-only text blocks are replaced with a
    non-whitespace placeholder (not a literal space, which is itself
    whitespace and gets rejected by the same Bedrock validation rule)."""

    def test_none_content_gets_placeholder(self):
        from agent.bedrock_adapter import _convert_content_to_converse, _EMPTY_TEXT_PLACEHOLDER
        blocks = _convert_content_to_converse(None)
        assert blocks[0]["text"] == _EMPTY_TEXT_PLACEHOLDER
        assert blocks[0]["text"].strip()



    def test_real_text_preserved(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        blocks = _convert_content_to_converse("Hello")
        assert blocks[0]["text"] == "Hello"




# ---------------------------------------------------------------------------
# Stale-connection detection and per-region client invalidation
# ---------------------------------------------------------------------------

class TestInvalidateRuntimeClient:
    """Per-region eviction used to discard dead/stale bedrock-runtime clients."""

    def test_evicts_only_the_target_region(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            invalidate_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        _bedrock_runtime_client_cache["us-east-1"] = "dead-client"
        _bedrock_runtime_client_cache["us-west-2"] = "live-client"

        evicted = invalidate_runtime_client("us-east-1")

        assert evicted is True
        assert "us-east-1" not in _bedrock_runtime_client_cache
        assert _bedrock_runtime_client_cache["us-west-2"] == "live-client"

    def test_returns_false_when_region_not_cached(self):
        from agent.bedrock_adapter import invalidate_runtime_client, reset_client_cache
        reset_client_cache()
        assert invalidate_runtime_client("eu-west-1") is False

    def test_evicts_budgeted_variants_of_the_target_region(self):
        """A stale connection pool belongs to the region's endpoint, not to
        one timeout budget — every ``region#t<seconds>`` client for that
        region has to go too, or the next budgeted call reuses a dead socket.
        """
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            invalidate_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        _bedrock_runtime_client_cache["us-east-1"] = "dead-default"
        _bedrock_runtime_client_cache["us-east-1#t30"] = "dead-30s"
        _bedrock_runtime_client_cache["us-east-1#t60"] = "dead-60s"
        _bedrock_runtime_client_cache["us-west-2#t30"] = "live-30s"

        assert invalidate_runtime_client("us-east-1") is True

        assert sorted(_bedrock_runtime_client_cache) == ["us-west-2#t30"]

    def test_returns_true_when_only_a_budgeted_variant_is_cached(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            invalidate_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        _bedrock_runtime_client_cache["eu-west-1#t15"] = "dead"

        assert invalidate_runtime_client("eu-west-1") is True
        assert not _bedrock_runtime_client_cache


class TestBedrockRuntimeClientTimeoutBudget:
    """NOL-135: botocore fixes connect/read timeouts at client construction,
    so honouring a caller's per-attempt budget means resolving a client built
    for that budget rather than the shared default one.
    """

    @staticmethod
    def _fake_boto3(captured):
        class _FakeBoto3:
            def client(self, service, **kwargs):
                captured.append((service, kwargs))
                return f"client-{len(captured)}"
        return _FakeBoto3()

    def test_default_client_is_unchanged_and_keyed_by_bare_region(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            _get_bedrock_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        captured: list = []
        with patch("agent.bedrock_adapter._require_boto3",
                   return_value=self._fake_boto3(captured)):
            _get_bedrock_runtime_client("us-east-1")

        assert list(_bedrock_runtime_client_cache) == ["us-east-1"]
        service, kwargs = captured[0]
        assert service == "bedrock-runtime"
        assert kwargs == {"region_name": "us-east-1"}, (
            "an unbudgeted call must keep botocore's own defaults"
        )

    def test_budget_produces_a_separate_client_with_matching_config(self):
        pytest.importorskip("botocore", reason="botocore required for Config assertions")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            _get_bedrock_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        captured: list = []
        with patch("agent.bedrock_adapter._require_boto3",
                   return_value=self._fake_boto3(captured)):
            _get_bedrock_runtime_client("us-east-1")
            _get_bedrock_runtime_client("us-east-1", timeout=30.0)

        assert sorted(_bedrock_runtime_client_cache) == ["us-east-1", "us-east-1#t30"]
        _, budgeted_kwargs = captured[1]
        config = budgeted_kwargs["config"]
        assert config.connect_timeout == 30.0
        assert config.read_timeout == 30.0
        # botocore retries read timeouts by default, which would silently
        # multiply the caller's budget. Hermes owns the retry decision.
        assert config.retries == {"total_max_attempts": 1}

    def test_same_budget_reuses_one_client(self):
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        captured: list = []
        with patch("agent.bedrock_adapter._require_boto3",
                   return_value=self._fake_boto3(captured)), _mock_botocore_config():
            first = _get_bedrock_runtime_client("us-east-1", timeout=30.0)
            second = _get_bedrock_runtime_client("us-east-1", timeout=30.0)

        assert first is second
        assert len(captured) == 1

    def test_distinct_budgets_do_not_share_a_client(self):
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        captured: list = []
        with patch("agent.bedrock_adapter._require_boto3",
                   return_value=self._fake_boto3(captured)), _mock_botocore_config():
            slow = _get_bedrock_runtime_client("us-east-1", timeout=60.0)
            fast = _get_bedrock_runtime_client("us-east-1", timeout=5.0)

        assert slow != fast
        assert len(captured) == 2

    def test_call_converse_forwards_the_budget_to_the_client_factory(self):
        from agent.bedrock_adapter import call_converse

        seen: dict = {}

        def _fake_factory(region, timeout=None):
            seen["region"] = region
            seen["timeout"] = timeout
            client = MagicMock()
            client.converse.return_value = {"output": {"message": {"content": []}}}
            return client

        with patch("agent.bedrock_adapter._get_bedrock_runtime_client", _fake_factory), \
             patch("agent.bedrock_adapter.normalize_converse_response",
                   return_value="normalized"):
            result = call_converse(
                region="us-east-1",
                model="anthropic.claude-3-haiku-20240307-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                timeout=12.5,
            )

        assert result == "normalized"
        assert seen == {"region": "us-east-1", "timeout": 12.5}


class TestIsStaleConnectionError:
    """Classifier that decides whether an exception warrants client eviction."""



    def test_detects_botocore_read_timeout(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import is_stale_connection_error
        from botocore.exceptions import ReadTimeoutError
        exc = ReadTimeoutError(endpoint_url="https://bedrock.example")
        assert is_stale_connection_error(exc) is True


    def test_detects_library_internal_assertion_error(self):
        """A bare AssertionError raised from inside urllib3/botocore signals
        a corrupted connection-pool invariant and should trigger eviction."""
        from agent.bedrock_adapter import is_stale_connection_error

        # Fabricate an AssertionError whose traceback's last frame belongs
        # to a module named "urllib3.connectionpool". We do this by exec'ing
        # a tiny `assert False` under a fake globals dict — the resulting
        # frame's ``f_globals["__name__"]`` is what the classifier inspects.
        fake_globals = {"__name__": "urllib3.connectionpool"}
        try:
            exec("def _boom():\n    assert False\n_boom()", fake_globals)
        except AssertionError as exc:
            assert is_stale_connection_error(exc) is True
        else:
            pytest.fail("AssertionError not raised")



    def test_ignores_unrelated_exceptions(self):
        from agent.bedrock_adapter import is_stale_connection_error
        assert is_stale_connection_error(ValueError("bad input")) is False
        assert is_stale_connection_error(KeyError("missing")) is False


class TestCallConverseInvalidatesOnStaleError:
    """call_converse / call_converse_stream evict the cached client when the
    boto3 call raises a stale-connection error — so the next invocation
    reconnects instead of reusing the dead socket."""


    def test_converse_stream_evicts_client_on_stale_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse_stream,
            reset_client_cache,
        )
        from botocore.exceptions import ConnectionClosedError

        reset_client_cache()
        dead_client = MagicMock()
        dead_client.converse_stream.side_effect = ConnectionClosedError(
            endpoint_url="https://bedrock.example",
        )
        _bedrock_runtime_client_cache["us-east-1"] = dead_client

        with pytest.raises(ConnectionClosedError):
            call_converse_stream(
                region="us-east-1",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert "us-east-1" not in _bedrock_runtime_client_cache

    def test_converse_does_not_evict_on_non_stale_error(self):
        """Non-stale errors (e.g. ValidationException) leave the client cache alone."""
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse,
            reset_client_cache,
        )
        from botocore.exceptions import ClientError

        reset_client_cache()
        live_client = MagicMock()
        live_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "bad"}},
            operation_name="Converse",
        )
        _bedrock_runtime_client_cache["us-east-1"] = live_client

        with pytest.raises(ClientError):
            call_converse(
                region="us-east-1",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert _bedrock_runtime_client_cache.get("us-east-1") is live_client, (
            "validation errors do not indicate a dead connection — keep the client"
        )



class TestStreamingAccessDeniedDetection:
    """is_streaming_access_denied_error() recognizes IAM denials of
    bedrock:InvokeModelWithResponseStream (InvokeModel-only policies)."""

    def _denied_client_error(self):
        from botocore.exceptions import ClientError
        return ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": (
                        "User: arn:aws:iam::123456789012:user/x is not "
                        "authorized to perform: "
                        "bedrock:InvokeModelWithResponseStream on resource: "
                        "arn:aws:bedrock:us-east-1::foundation-model/"
                        "anthropic.claude-3-sonnet-20240229-v1:0"
                    ),
                }
            },
            operation_name="ConverseStream",
        )

    def test_matches_access_denied_client_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import is_streaming_access_denied_error
        assert is_streaming_access_denied_error(self._denied_client_error()) is True




    def test_ignores_unrelated_errors(self):
        from agent.bedrock_adapter import is_streaming_access_denied_error
        assert is_streaming_access_denied_error(ValueError("boom")) is False
        assert is_streaming_access_denied_error(
            RuntimeError("stream not supported")
        ) is False


class TestCallConverseStreamIamFallback:
    """call_converse_stream() falls back to converse() when IAM denies the
    streaming action — InvokeModel-only policies keep working."""

    def test_falls_back_to_converse_on_streaming_denial(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse_stream,
            reset_client_cache,
        )
        from botocore.exceptions import ClientError

        reset_client_cache()
        client = MagicMock()
        client.converse_stream.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": (
                        "User is not authorized to perform: "
                        "bedrock:InvokeModelWithResponseStream"
                    ),
                }
            },
            operation_name="ConverseStream",
        )
        client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        _bedrock_runtime_client_cache["us-east-1"] = client

        result = call_converse_stream(
            region="us-east-1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "hi"}],
        )

        client.converse.assert_called_once()
        assert result.choices[0].message.content == "hi"
        # Not a stale connection — client stays cached.
        assert _bedrock_runtime_client_cache.get("us-east-1") is client


# ---------------------------------------------------------------------------
# boto3 version check
# ---------------------------------------------------------------------------


class TestRequireBoto3VersionCheck:
    """Test that _require_boto3() rejects boto3 versions older than 1.34.59."""

    def test_raises_runtime_error_when_boto3_too_old(self):
        """boto3 < 1.34.59 should raise RuntimeError with upgrade instructions."""
        from agent.bedrock_adapter import _require_boto3

        fake_boto3 = MagicMock()
        fake_boto3.__version__ = "1.34.46"
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with pytest.raises(RuntimeError, match="does not support converse_stream"):
                _require_boto3()

    def test_accepts_boto3_at_minimum_version(self):
        """boto3 == 1.34.59 should be accepted."""
        from agent.bedrock_adapter import _require_boto3

        fake_boto3 = MagicMock()
        fake_boto3.__version__ = "1.34.59"
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            result = _require_boto3()
            assert result is fake_boto3



class TestImageBase64Decoding:
    """Image data URLs must be decoded to raw bytes before passing to Converse API.

    boto3 re-encodes at the wire layer, so passing the base64 string directly
    results in double-encoding. Bedrock rejects with 'Failed to sanitize image'.
    Ref: #33317.
    """

    def test_data_url_decoded_to_bytes(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        import base64

        # A tiny 1x1 red PNG
        raw_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        content = [{"type": "image_url", "image_url": {"url": data_url}}]
        blocks = _convert_content_to_converse(content)

        assert len(blocks) == 1
        img_block = blocks[0]["image"]
        assert img_block["format"] == "png"
        # Must be raw bytes, not a base64 string
        assert isinstance(img_block["source"]["bytes"], bytes)
        assert img_block["source"]["bytes"] == raw_png

    def test_invalid_base64_falls_back_to_encode(self):
        from agent.bedrock_adapter import _convert_content_to_converse

        data_url = "data:image/jpeg;base64,NOT_VALID_BASE64!!!"
        content = [{"type": "image_url", "image_url": {"url": data_url}}]
        blocks = _convert_content_to_converse(content)

        # Should not crash — falls back to encoding the string as bytes
        assert len(blocks) == 1
        assert isinstance(blocks[0]["image"]["source"]["bytes"], bytes)


class TestBearerTokenRoutesToConverse:
    """Bearer Token users must go through Converse API, not AnthropicBedrock SDK.

    The AnthropicBedrock SDK only supports SigV4 signing — it cannot use
    AWS_BEARER_TOKEN_BEDROCK. Ref: #28156.
    """

    def _resolve(self, monkeypatch, *, bearer: bool):
        import os

        from hermes_cli import runtime_provider as rp

        if bearer:
            monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer-token-123")
        else:
            monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ or not bearer

        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {
                "default": "us.anthropic.claude-sonnet-4-6",
                "provider": "bedrock",
            },
        )
        monkeypatch.setattr(rp, "load_config", lambda: {"bedrock": {}})
        return rp.resolve_runtime_provider(requested="bedrock")

    def test_bearer_token_forces_converse_for_claude(self, monkeypatch):
        """Claude model + Bearer Token → bedrock_converse, not anthropic_messages."""
        runtime = self._resolve(monkeypatch, bearer=True)
        assert runtime["api_mode"] == "bedrock_converse"
        assert "bedrock_anthropic" not in runtime

    def test_sigv4_claude_still_uses_anthropic_bedrock_sdk(self, monkeypatch):
        """Without a bearer token, Claude keeps the AnthropicBedrock SDK path."""
        runtime = self._resolve(monkeypatch, bearer=False)
        assert runtime["api_mode"] == "anthropic_messages"
        assert runtime.get("bedrock_anthropic") is True


# ---------------------------------------------------------------------------
# Reasoning config → additionalModelRequestFields (NOL-139)
# ---------------------------------------------------------------------------

class TestBuildConverseKwargsReasoning:
    """A reasoning config must be honoured or reported — never a silent no-op.

    Before NOL-139, ``build_converse_kwargs`` had no way to receive a
    reasoning config at all: ``_build_call_kwargs(provider="bedrock",
    reasoning_config=...)`` produced ``extra_body={"reasoning": {...}}`` and
    the Bedrock shim adapter never read ``extra_body``, so the setting had no
    effect and nothing logged the drop.
    """

    @staticmethod
    def _clear_warned():
        from agent import bedrock_adapter as ba
        ba._REASONING_UNSUPPORTED_WARNED.clear()

    def test_adaptive_claude_gets_thinking_and_output_config(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-opus-4-7-20260101-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_config={"enabled": True, "effort": "high"},
        )

        fields = kwargs["additionalModelRequestFields"]
        assert fields["thinking"] == {
            "type": "adaptive", "display": "summarized",
        }
        assert fields["output_config"] == {"effort": "high"}

    def test_xhigh_downgrades_to_max_on_claude_4_6(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="anthropic.claude-opus-4-6-20251101-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )

        fields = kwargs["additionalModelRequestFields"]
        assert fields["output_config"] == {"effort": "max"}

    def test_legacy_claude_gets_manual_budget_and_max_tokens_bump(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            reasoning_config={"enabled": True, "effort": "high"},
        )

        fields = kwargs["additionalModelRequestFields"]
        assert fields["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        # Anthropic requires max_tokens > budget_tokens for manual thinking
        # and rejects non-default sampling params alongside it.
        assert kwargs["inferenceConfig"]["maxTokens"] == 16000 + 4096
        assert "temperature" not in kwargs["inferenceConfig"]
        assert "topP" not in kwargs["inferenceConfig"]

    def test_reasoning_disabled_sends_no_thinking_fields(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-opus-4-7-20260101-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_config={"enabled": False},
        )

        assert "additionalModelRequestFields" not in kwargs

    def test_non_claude_model_warns_loudly_and_once(self, caplog):
        from agent import bedrock_adapter as ba

        self._clear_warned()
        try:
            with caplog.at_level(logging.WARNING, logger=ba.logger.name):
                for _ in range(2):
                    kwargs = ba.build_converse_kwargs(
                        model="amazon.nova-lite-v1:0",
                        messages=[{"role": "user", "content": "Hi"}],
                        reasoning_config={"enabled": True, "effort": "high"},
                    )
        finally:
            self._clear_warned()

        assert "additionalModelRequestFields" not in kwargs
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "reasoning config cannot be honoured" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"expected exactly one deduped warning, got {len(warnings)}"
        )
        assert "amazon.nova-lite-v1:0" in warnings[0].getMessage()

    def test_haiku_warns_instead_of_silent_noop(self, caplog):
        from agent import bedrock_adapter as ba

        self._clear_warned()
        try:
            with caplog.at_level(logging.WARNING, logger=ba.logger.name):
                kwargs = ba.build_converse_kwargs(
                    model="anthropic.claude-3-haiku-20240307-v1:0",
                    messages=[{"role": "user", "content": "Hi"}],
                    reasoning_config={"enabled": True},
                )
        finally:
            self._clear_warned()

        assert "additionalModelRequestFields" not in kwargs
        assert any(
            "reasoning config cannot be honoured" in r.getMessage()
            for r in caplog.records
        )

    def test_raw_additional_fields_pass_through(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-opus-4-7-20260101-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            additional_model_request_fields={"vendor_field": {"a": 1}},
        )

        fields = kwargs["additionalModelRequestFields"]
        assert fields["vendor_field"] == {"a": 1}

    def test_translated_reasoning_wins_over_raw_duplicate(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-opus-4-7-20260101-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_config={"enabled": True, "effort": "low"},
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": 1},
                "vendor_field": True,
            },
        )

        fields = kwargs["additionalModelRequestFields"]
        assert fields["thinking"] == {
            "type": "adaptive", "display": "summarized",
        }
        assert fields["vendor_field"] is True

    def test_call_converse_forwards_reasoning_to_the_wire(self):
        """Full-path regression: the field must reach client.converse().

        Fails if ANY layer (call_converse → build_converse_kwargs) drops the
        reasoning config or the raw passthrough fields.
        """
        from agent import bedrock_adapter as ba

        captured = {}

        class _FakeClient:
            def converse(self, **kwargs):
                captured.update(kwargs)
                return {
                    "output": {"message": {"role": "assistant", "content": [
                        {"text": "ok"},
                    ]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                }

        with patch.object(
            ba, "_get_bedrock_runtime_client", return_value=_FakeClient(),
        ):
            ba.call_converse(
                region="us-east-1",
                model="us.anthropic.claude-opus-4-7-20260101-v1:0",
                messages=[{"role": "user", "content": "Hi"}],
                reasoning_config={"enabled": True, "effort": "medium"},
                additional_model_request_fields={"vendor_field": 7},
            )

        fields = captured["additionalModelRequestFields"]
        assert fields["thinking"]["type"] == "adaptive"
        assert fields["output_config"] == {"effort": "medium"}
        assert fields["vendor_field"] == 7
