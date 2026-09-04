"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    _detect_video_mime_type,
    _download_video,
    _gemini_inline_source_cap_bytes,
    _gemini_rejects_sampling_overrides,
    _is_gemini_model,
    _is_youtube_url,
    _normalize_video_fps,
    _probe_remote_video_mime,
    _resolve_downloaded_video_mime,
    _resolve_video_settings,
    _video_mime_from_url,
    _video_part_for_model,
    _video_to_base64_data_url,
    _DownloadedVideo,
    _VideoTooLargeError,
    _handle_video_analyze,
    _MAX_VIDEO_BASE64_BYTES,
    video_analyze_tool,
    VIDEO_ANALYZE_SCHEMA,
)


# ---------------------------------------------------------------------------
# _detect_video_mime_type
# ---------------------------------------------------------------------------


class TestDetectVideoMimeType:
    """Extension-based MIME detection for video files."""

    def test_mp4(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_webm(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/webm"


    def test_case_insensitive(self, tmp_path):
        p = tmp_path / "clip.MP4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"


# ---------------------------------------------------------------------------
# _video_to_base64_data_url
# ---------------------------------------------------------------------------


class TestVideoToBase64DataUrl:
    """Base64 encoding of video files."""

    def test_produces_data_url(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        assert result.startswith("data:video/mp4;base64,")


    def test_default_mime_for_unknown_ext(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        # Falls back to video/mp4
        assert result.startswith("data:video/mp4;base64,")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestVideoAnalyzeSchema:
    """Schema structure is correct."""

    def test_schema_name(self):
        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"


    def test_schema_description_mentions_video(self):
        assert "video" in VIDEO_ANALYZE_SCHEMA["description"].lower()


# ---------------------------------------------------------------------------
# _handle_video_analyze handler
# ---------------------------------------------------------------------------


class TestHandleVideoAnalyze:
    """Tests for the registry handler wrapper."""

    def test_returns_awaitable(self, tmp_path, monkeypatch):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "test"})
            result = _handle_video_analyze({"video_url": str(video_file), "question": "what is this?"})
            # Should return an awaitable (coroutine)
            assert asyncio.iscoroutine(result)
            # Clean up the unawaited coroutine
            result.close()


    def test_falls_back_to_vision_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-flash")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-flash"


# ---------------------------------------------------------------------------
# video_analyze_tool — integration-style tests with mocked LLM
# ---------------------------------------------------------------------------


class TestVideoAnalyzeTool:
    """Core video analysis function tests."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_file_success(self, tmp_path, monkeypatch):
        """Analyze a local video file — happy path."""
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A short video showing a demo."

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="A short video showing a demo."):
                result = self._run(video_analyze_tool(str(video), "What is this?"))

        data = json.loads(result)
        assert data["success"] is True
        assert "demo" in data["analysis"].lower()

    def test_local_file_read_guard_blocks_env_via_video_extension(self, tmp_path):
        """A .env file symlinked with a video extension must still be blocked.

        _detect_video_mime_type only checks the file extension, not file
        content, so without a read guard a model could point video_url at
        any credential-store file (renamed/symlinked to look like a video)
        and have its raw bytes base64-encoded and sent to the vision
        provider. Regression for the shared agent.file_safety chokepoint
        added to video_analyze_tool's local-file branch.
        """
        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")
        disguised = tmp_path / "video.mp4"
        disguised.symlink_to(secret)

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(str(disguised), "What is this?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "secret-bearing environment file" in data["error"]
        mock_llm.assert_not_awaited()


    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()


    def test_api_message_format(self, tmp_path):
        """Verify the message sent to LLM uses video_url content type."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            return mock_response

        with patch("tools.vision_tools.async_call_llm", side_effect=capture_llm):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                self._run(video_analyze_tool(str(video), "Describe this"))

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "video_url"
        assert "video_url" in content[1]
        assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")


# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------


class TestVideoToolsetRegistration:
    """Verify the tool is registered correctly."""

    def test_registered_in_video_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("video_analyze")
        assert entry is not None
        assert entry.toolset == "video"
        assert entry.is_async is True
        assert entry.emoji == "🎬"


    def test_in_video_toolset_definition(self):
        """Toolset 'video' should contain video_analyze."""
        from toolsets import TOOLSETS
        assert "video" in TOOLSETS
        assert "video_analyze" in TOOLSETS["video"]["tools"]


# ---------------------------------------------------------------------------
# Provider-aware wire shape (Gemini file part vs legacy video_url part)
# ---------------------------------------------------------------------------

_HTTPS_MP4 = "https://cdn.example.com/clips/demo.mp4"
_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# What nolgia-api hands out for every asset: a V4-signed GCS URL. Google does
# NOT fetch these for a Gemini ``file_data`` part (403 PERMISSION_DENIED).
_SIGNED_GCS_URL = (
    "https://storage.googleapis.com/nolgia-assets/renders/clip.mp4"
    "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
    "&X-Goog-Credential=svc%40proj.iam.gserviceaccount.com%2F20260903%2Fauto%2Fstorage%2Fgoog4_request"
    "&X-Goog-Date=20260903T120000Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=abcdef0123456789"
)
_MB = 1024 * 1024


class TestIsGeminiModel:
    @pytest.mark.parametrize("model", [
        "gemini-3.8-flash",
        "google/gemini-2.5-flash",
        "gemini/gemini-3-flash",
        "vertex_ai/gemini-3.8-flash",
        "openrouter/google/gemini-3-flash-preview",
        "GEMINI-3.8-FLASH",
    ])
    def test_gemini_under_any_prefix(self, model):
        assert _is_gemini_model(model) is True

    @pytest.mark.parametrize("model", [None, "", "openai/gpt-4o", "qwen/qwen3-vl", "claude-opus-4.7"])
    def test_non_gemini(self, model):
        assert _is_gemini_model(model) is False


class TestNormalizeVideoFps:
    @pytest.mark.parametrize("value,expected", [
        (1, 1), (1.0, 1), ("1", 1), (2, 2), (0.5, 0.5), ("0.5", 0.5),
    ])
    def test_positive_values(self, value, expected):
        got = _normalize_video_fps(value)
        assert got == expected
        if expected == int(expected):
            assert isinstance(got, int)

    @pytest.mark.parametrize("value", [None, 0, 0.0, "0", -1, "junk", True, float("nan")])
    def test_omit_values(self, value):
        assert _normalize_video_fps(value) is None


class TestVideoPartForModel:
    """The pure decision: Gemini gets a `file` part, everyone else `video_url`."""

    def test_gemini_https_url_file_part(self):
        part = _video_part_for_model("gemini-3.8-flash", _HTTPS_MP4, "video/mp4", fps=1)
        assert part == {
            "type": "file",
            "file": {
                "file_data": _HTTPS_MP4,
                "format": "video/mp4",
                "video_metadata": {"fps": 1},
            },
        }

    @pytest.mark.parametrize("model", [
        "google/gemini-2.5-flash", "gemini/gemini-3-flash", "vertex_ai/gemini-3.8-flash",
    ])
    def test_gemini_any_prefix(self, model):
        part = _video_part_for_model(model, _HTTPS_MP4, "video/mp4")
        assert part["type"] == "file"
        assert part["file"]["file_data"] == _HTTPS_MP4

    def test_gemini_data_url_file_part(self):
        data_url = "data:video/webm;base64,AAAA"
        part = _video_part_for_model("gemini-3.8-flash", data_url, "video/webm", fps=None)
        assert part["file"] == {"file_data": data_url, "format": "video/webm"}

    @pytest.mark.parametrize("fps", [None, 0, 0.0, "0"])
    def test_fps_zero_or_none_omits_video_metadata(self, fps):
        part = _video_part_for_model("gemini-3.8-flash", _HTTPS_MP4, "video/mp4", fps=fps)
        assert "video_metadata" not in part["file"]

    def test_detail_passthrough(self):
        part = _video_part_for_model("gemini-3.8-flash", _HTTPS_MP4, "video/mp4", fps=1, detail=" Low ")
        assert part["file"]["detail"] == "low"
        assert part["file"]["video_metadata"] == {"fps": 1}

    def test_detail_empty_is_omitted(self):
        part = _video_part_for_model("gemini-3.8-flash", _HTTPS_MP4, "video/mp4", detail="")
        assert "detail" not in part["file"]

    @pytest.mark.parametrize("model", [None, "", "openai/gpt-4o", "qwen/qwen3-vl-235b"])
    def test_non_gemini_keeps_video_url(self, model):
        data_url = "data:video/mp4;base64,AAAA"
        part = _video_part_for_model(model, data_url, "video/mp4", fps=1, detail="low")
        # Exactly the legacy shape — no fps/detail/format leak into other providers.
        assert part == {"type": "video_url", "video_url": {"url": data_url}}


class TestGeminiRejectsSamplingOverrides:
    @pytest.mark.parametrize("model", [
        "gemini-3.8-flash", "google/gemini-3.8-flash", "gemini/gemini-3.8-flash",
        "vertex_ai/gemini-3.8-flash", "openrouter/google/gemini-3.9-flash-lite",
    ])
    def test_flash_38_plus(self, model):
        assert _gemini_rejects_sampling_overrides(model) is True

    @pytest.mark.parametrize("model", [
        None, "", "google/gemini-2.5-flash", "gemini-3.7-flash", "gemini-3-pro", "openai/gpt-4o",
    ])
    def test_everything_else_keeps_temperature(self, model):
        assert _gemini_rejects_sampling_overrides(model) is False


class TestVideoMimeFromUrl:
    @pytest.mark.parametrize("url,expected", [
        (_HTTPS_MP4, "video/mp4"),
        ("https://cdn.example.com/a/b.WEBM?sig=1", "video/webm"),
        ("https://cdn.example.com/clip.mov#t=1", "video/mov"),
        ("https://www.youtube.com/watch?v=abc123", None),
        ("https://cdn.example.com/stream", None),
    ])
    def test_extension_mapping(self, url, expected):
        assert _video_mime_from_url(url) == expected


class TestIsYoutubeUrl:
    """Only YouTube URLs are handed to Google to fetch; everything else is inlined."""

    @pytest.mark.parametrize("url", [
        _YOUTUBE_URL,
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/shorts/dQw4w9WgXcQ",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
        "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",
    ])
    def test_youtube_forms(self, url):
        assert _is_youtube_url(url) is True

    @pytest.mark.parametrize("url", [
        _SIGNED_GCS_URL,
        _HTTPS_MP4,
        "https://cdn.example.com/stream",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?list=PL123",
        "https://www.youtube.com/shorts/",
        "https://www.youtube.com/feed/subscriptions",
        "https://youtu.be/",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
        "https://evil.example/www.youtube.com/watch?v=dQw4w9WgXcQ",
        "ftp://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "/tmp/clip.mp4",
        "",
        None,
        42,
    ])
    def test_everything_else(self, url):
        assert _is_youtube_url(url) is False


class TestResolveDownloadedVideoMime:
    """Content-Type wins, then the URL extension, then video/mp4."""

    @pytest.mark.parametrize("url,content_type,expected", [
        (_SIGNED_GCS_URL, "video/webm", "video/webm"),          # server beats extension
        (_SIGNED_GCS_URL, None, "video/mp4"),                    # extension survives the query
        ("https://cdn.example.com/a/b.WEBM?sig=1", None, "video/webm"),
        ("https://storage.googleapis.com/b/o/stream?X-Goog-Signature=1", None, "video/mp4"),
        ("https://cdn.example.com/stream", "video/mov", "video/mov"),
    ])
    def test_precedence(self, url, content_type, expected):
        assert _resolve_downloaded_video_mime(url, content_type) == expected


class _FakeGetClient:
    """Stands in for the SSRF-safe httpx client in ``_download_video``."""

    def __init__(self, body=b"", headers=None, error=None, response_url=None, fail_times=0):
        self._body = body
        self._headers = headers or {}
        self._error = error
        self._response_url = response_url
        self._fail_times = fail_times
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        self.calls.append((url, headers))
        if self._error and (self._fail_times == 0 or len(self.calls) <= self._fail_times):
            raise self._error
        return SimpleNamespace(
            headers=self._headers,
            url=self._response_url or url,
            content=self._body,
            status_code=200,
            raise_for_status=lambda: None,
        )


class TestDownloadVideo:
    """Size caps, Content-Type capture and retry policy of ``_download_video``."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _download(self, client, url, dest, **kwargs):
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client), \
             patch("asyncio.sleep", new=AsyncMock()) as sleep:
            result = self._run(_download_video(url, dest, **kwargs))
        return result, sleep

    def test_returns_path_and_video_content_type(self, tmp_path):
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 64, {"content-type": "video/webm; charset=binary", "content-length": "64"})
        result, _ = self._download(client, _SIGNED_GCS_URL, dest)
        assert result == _DownloadedVideo(dest, "video/webm")
        assert dest.read_bytes() == b"\x00" * 64
        assert client.calls[0][0] == _SIGNED_GCS_URL

    def test_non_video_content_type_is_none(self, tmp_path):
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 8, {"content-type": "application/octet-stream"})
        result, _ = self._download(client, _SIGNED_GCS_URL, dest)
        assert result.content_type is None
        assert result.path == dest

    def test_content_length_over_cap_is_refused_once(self, tmp_path):
        """Header pre-check: refused before the body is touched, never retried."""
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 8, {"content-length": str(40 * _MB), "content-type": "video/mp4"})
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client), \
             patch("asyncio.sleep", new=AsyncMock()) as sleep:
            with pytest.raises(_VideoTooLargeError) as info:
                self._run(_download_video(_SIGNED_GCS_URL, dest, max_bytes=_gemini_inline_source_cap_bytes()))
        assert info.value.size == 40 * _MB
        assert info.value.limit == _gemini_inline_source_cap_bytes()
        assert isinstance(info.value, ValueError)
        assert not dest.exists()
        assert len(client.calls) == 1
        sleep.assert_not_awaited()

    def test_body_over_cap_is_refused_once(self, tmp_path):
        """Body check: a server that omits Content-Length is still capped."""
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 256, {"content-type": "video/mp4"})
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client), \
             patch("asyncio.sleep", new=AsyncMock()) as sleep:
            with pytest.raises(_VideoTooLargeError) as info:
                self._run(_download_video(_HTTPS_MP4, dest, max_bytes=64))
        assert info.value.size == 256
        assert not dest.exists()
        assert len(client.calls) == 1
        sleep.assert_not_awaited()

    def test_default_cap_is_the_50mb_hard_limit(self, tmp_path):
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 8, {"content-length": str(_MAX_VIDEO_BASE64_BYTES + 1)})
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client), \
             patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(_VideoTooLargeError) as info:
                self._run(_download_video(_HTTPS_MP4, dest))
        assert info.value.limit == _MAX_VIDEO_BASE64_BYTES

    def test_transient_error_is_retried_then_succeeds(self, tmp_path):
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 8, {"content-type": "video/mp4"}, error=RuntimeError("blip"), fail_times=2)
        result, sleep = self._download(client, _HTTPS_MP4, dest)
        assert result.content_type == "video/mp4"
        assert len(client.calls) == 3
        assert sleep.await_count == 2

    def test_policy_block_is_not_retried(self, tmp_path):
        dest = tmp_path / "temp_video_x.mp4"
        client = _FakeGetClient(b"\x00" * 8, {"content-type": "video/mp4"})
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client), \
             patch("tools.vision_tools.check_website_access", return_value={"message": "blocked by policy"}), \
             patch("asyncio.sleep", new=AsyncMock()) as sleep:
            with pytest.raises(PermissionError, match="blocked by policy"):
                self._run(_download_video(_HTTPS_MP4, dest))
        assert client.calls == []
        sleep.assert_not_awaited()


class _FakeHeadClient:
    def __init__(self, content_type=None, error=None, response_url=None, response_hook=None):
        self._ct = content_type
        self._error = error
        self._response_url = response_url
        self._response_hook = response_hook
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def head(self, url, headers=None):
        self.calls.append((url, headers))
        if self._error:
            raise self._error
        response = SimpleNamespace(
            headers={"content-type": self._ct} if self._ct else {},
            url=self._response_url or url,
            status_code=200,
        )
        if self._response_hook:
            await self._response_hook(response)
        return response


class TestProbeRemoteVideoMime:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_extension_is_fallback_after_policy_probe(self):
        client = _FakeHeadClient("application/octet-stream")
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client):
            assert self._run(_probe_remote_video_mime(_HTTPS_MP4)) == "video/mp4"
        assert client.calls and client.calls[0][0] == _HTTPS_MP4

    def test_final_url_is_checked_against_website_policy(self):
        client = None

        def make_client(**kwargs):
            nonlocal client
            hook = kwargs["event_hooks"]["response"][0]
            client = _FakeHeadClient(
                "video/mp4",
                response_url="https://blocked.example/clip.mp4",
                response_hook=hook,
            )
            return client

        with patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=make_client), \
             patch("tools.url_safety.async_is_safe_url", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools.check_website_access", side_effect=lambda url: (
                 {"message": "blocked by policy"} if "blocked.example" in url else None
             )):
            with pytest.raises(PermissionError, match="blocked by policy"):
                self._run(_probe_remote_video_mime(_HTTPS_MP4))

    def test_head_content_type_used_when_video(self):
        client = _FakeHeadClient("video/webm; charset=binary")
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client):
            got = self._run(_probe_remote_video_mime("https://cdn.example.com/stream"))
        assert got == "video/webm"
        assert client.calls and client.calls[0][0] == "https://cdn.example.com/stream"

    def test_non_video_content_type_falls_back_to_mp4(self):
        client = _FakeHeadClient("text/html; charset=utf-8")
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client):
            got = self._run(_probe_remote_video_mime("https://www.youtube.com/watch?v=abc123"))
        assert got == "video/mp4"

    def test_probe_error_falls_back_to_mp4(self):
        client = _FakeHeadClient(error=RuntimeError("boom"))
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=client):
            got = self._run(_probe_remote_video_mime("https://cdn.example.com/stream"))
        assert got == "video/mp4"


# ---------------------------------------------------------------------------
# video_analyze_tool — Gemini wire shape end to end (mocked LLM, no network)
# ---------------------------------------------------------------------------


def _capture_llm(store):
    async def capture(**kwargs):
        store.update(kwargs)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        return mock_response
    return capture


def _fake_download(content_type="video/mp4", size=64):
    """A ``_download_video`` stand-in that writes ``size`` bytes and reports ``content_type``."""

    async def fake(url, destination, max_retries=3, max_bytes=_MAX_VIDEO_BASE64_BYTES):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * size)
        return _DownloadedVideo(destination, content_type)
    return fake


class TestVideoAnalyzeToolGemini:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_youtube_url_passes_through_without_download(self):
        """Gemini + YouTube: URL goes straight into file_data; nothing is downloaded."""
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._probe_remote_video_mime", new=AsyncMock(return_value="video/mp4")) as probe, \
             patch("tools.vision_tools._download_video", new_callable=AsyncMock) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool(_YOUTUBE_URL, "Describe this", "gemini-3.8-flash"))

        assert json.loads(result)["success"] is True
        mock_dl.assert_not_awaited()
        probe.assert_awaited_once_with(_YOUTUBE_URL)
        content = captured["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Describe this"}
        assert content[1] == {
            "type": "file",
            "file": {
                "file_data": _YOUTUBE_URL,
                "format": "video/mp4",
                "video_metadata": {"fps": 1},
            },
        }
        assert captured["model"] == "gemini-3.8-flash"
        # gemini-3.8-flash rejects sampling overrides: temperature stays off the wire.
        assert "temperature" not in captured

    @pytest.mark.parametrize("url", [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    ])
    def test_every_youtube_form_passes_through(self, url):
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._probe_remote_video_mime", new=AsyncMock(return_value="video/mp4")), \
             patch("tools.vision_tools._download_video", new_callable=AsyncMock) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            self._run(video_analyze_tool(url, "Describe", "gemini-3.8-flash"))
        mock_dl.assert_not_awaited()
        assert captured["messages"][0]["content"][1]["file"]["file_data"] == url

    def test_youtube_url_fps_and_detail_from_caller(self):
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._probe_remote_video_mime", new=AsyncMock(return_value="video/mp4")), \
             patch("tools.vision_tools._download_video", new_callable=AsyncMock), \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            self._run(video_analyze_tool(
                _YOUTUBE_URL, "Describe", "google/gemini-2.5-flash", fps=0, detail="low", provider="gemini",
            ))
        file_part = captured["messages"][0]["content"][1]["file"]
        assert "video_metadata" not in file_part
        assert file_part["detail"] == "low"
        assert captured["provider"] == "gemini"
        # Pre-3.8 Gemini still gets the configured temperature.
        assert captured["temperature"] == 0.1

    def test_signed_gcs_url_is_downloaded_and_inlined(self):
        """The live bug: a signed GCS URL in file_data 403s at Google. Inline it instead."""
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._probe_remote_video_mime", new_callable=AsyncMock) as probe, \
             patch("tools.vision_tools._download_video", side_effect=_fake_download("video/mp4", 64)) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool(_SIGNED_GCS_URL, "Review this render", "gemini-3.8-flash"))

        assert json.loads(result)["success"] is True
        probe.assert_not_awaited()
        assert mock_dl.call_count == 1
        args, kwargs = mock_dl.call_args
        assert args[0] == _SIGNED_GCS_URL
        # Gemini downloads are capped at what can be inlined, not the 50 MB hard cap.
        assert kwargs["max_bytes"] == _gemini_inline_source_cap_bytes()
        part = captured["messages"][0]["content"][1]
        assert part["type"] == "file"
        assert part["file"]["file_data"] == (
            "data:video/mp4;base64," + base64.b64encode(b"\x00" * 64).decode("ascii")
        )
        assert part["file"]["format"] == "video/mp4"
        assert part["file"]["video_metadata"] == {"fps": 1}

    def test_downloaded_format_prefers_response_content_type(self):
        captured = {}
        url = "https://storage.googleapis.com/b/o/stream?X-Goog-Signature=1"
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._download_video", side_effect=_fake_download("video/webm")), \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            self._run(video_analyze_tool(url, "Describe", "gemini-3.8-flash"))
        file_part = captured["messages"][0]["content"][1]["file"]
        assert file_part["format"] == "video/webm"
        assert file_part["file_data"].startswith("data:video/webm;base64,")

    def test_downloaded_format_falls_back_to_url_extension_then_mp4(self):
        for url, expected in [
            ("https://cdn.example.com/clip.webm?X-Goog-Signature=1", "video/webm"),
            ("https://storage.googleapis.com/b/o/stream?X-Goog-Signature=1", "video/mp4"),
        ]:
            captured = {}
            with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
                 patch("tools.vision_tools._download_video", side_effect=_fake_download(None)), \
                 patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
                 patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                self._run(video_analyze_tool(url, "Describe", "gemini-3.8-flash"))
            file_part = captured["messages"][0]["content"][1]["file"]
            assert file_part["format"] == expected, url
            assert file_part["file_data"].startswith(f"data:{expected};base64,"), url

    def test_https_url_over_gemini_cap_errors_without_request(self):
        """Content-Length says 40 MB: the clear inline error, and no LLM call."""
        cap = _gemini_inline_source_cap_bytes()

        async def too_large(url, destination, max_retries=3, max_bytes=_MAX_VIDEO_BASE64_BYTES):
            assert max_bytes == cap
            raise _VideoTooLargeError(40 * _MB, max_bytes)

        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._download_video", side_effect=too_large) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(_SIGNED_GCS_URL, "Describe", "gemini-3.8-flash"))
        data = json.loads(result)
        assert data["success"] is False
        assert mock_dl.call_count == 1
        assert "Video too large to inline for Gemini: 40.0 MB of source video" in data["error"]
        assert "roughly 14 MB of source video after base64 expansion" in data["error"]
        assert "trim the clip or pass a YouTube URL" in data["error"]
        assert "Files API" not in data["error"]
        assert "too large" in data["analysis"].lower()
        assert "only YouTube URLs are fetched by Google" in data["analysis"]
        mock_llm.assert_not_awaited()

    def test_https_download_over_gemini_cap_after_fetch_errors_without_request(self, monkeypatch):
        """Backstop: a body that slipped past the header check is still refused post-base64."""
        monkeypatch.setattr("tools.vision_tools._GEMINI_INLINE_MAX_BASE64_BYTES", 64)
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._download_video", side_effect=_fake_download("video/mp4", 256)), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(_SIGNED_GCS_URL, "Describe", "gemini-3.8-flash"))
        data = json.loads(result)
        assert data["success"] is False
        assert "Video too large to inline for Gemini" in data["error"]
        assert "trim the clip or pass a YouTube URL" in data["error"]
        mock_llm.assert_not_awaited()

    def test_private_url_still_blocked_before_passthrough(self):
        """The SSRF guard runs even though Google (not Hermes) would fetch the URL."""
        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool("https://127.0.0.1/clip.mp4", "Describe", "gemini-3.8-flash"))
        data = json.loads(result)
        assert data["success"] is False
        assert "invalid video source" in data["error"].lower()
        mock_llm.assert_not_awaited()

    def test_website_policy_block_applies_to_passthrough(self):
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools.check_website_access", return_value={"message": "blocked by policy"}), \
             patch("tools.vision_tools._download_video", new_callable=AsyncMock) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(_YOUTUBE_URL, "Describe", "gemini-3.8-flash"))
        data = json.loads(result)
        assert data["success"] is False
        assert "blocked by policy" in data["error"]
        mock_dl.assert_not_awaited()
        mock_llm.assert_not_awaited()

    def test_plain_http_url_is_downloaded_and_inlined(self):
        """A non-YouTube http:// URL takes the download + data URL path like any other."""
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._download_video", side_effect=_fake_download("video/mp4")) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            self._run(video_analyze_tool("http://cdn.example.com/clip.mp4", "Describe", "gemini-3.8-flash"))
        assert mock_dl.call_count == 1
        file_part = captured["messages"][0]["content"][1]["file"]
        assert file_part["file_data"].startswith("data:video/mp4;base64,")
        assert file_part["format"] == "video/mp4"

    def test_local_file_is_inlined_as_file_part(self, tmp_path):
        video = tmp_path / "demo.webm"
        video.write_bytes(b"\x00" * 128)
        captured = {}
        with patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool(str(video), "Describe", "gemini-3.8-flash"))
        assert json.loads(result)["success"] is True
        part = captured["messages"][0]["content"][1]
        assert part["type"] == "file"
        assert part["file"]["file_data"].startswith("data:video/webm;base64,")
        assert part["file"]["format"] == "video/webm"
        assert part["file"]["video_metadata"] == {"fps": 1}

    def test_local_file_over_gemini_inline_cap_errors(self, tmp_path, monkeypatch):
        video = tmp_path / "big.mp4"
        video.write_bytes(b"\x00" * 256)
        monkeypatch.setattr("tools.vision_tools._GEMINI_INLINE_MAX_BASE64_BYTES", 64)
        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(str(video), "Describe", "gemini-3.8-flash"))
        data = json.loads(result)
        assert data["success"] is False
        assert "roughly 0 MB of source video after base64 expansion" in data["error"]
        assert "trim the clip or pass a YouTube URL" in data["error"]
        assert "https URL" not in data["error"]
        assert "too large" in data["analysis"].lower()
        mock_llm.assert_not_awaited()

    def test_gemini_inline_cap_does_not_apply_to_other_models(self, tmp_path, monkeypatch):
        video = tmp_path / "big.mp4"
        video.write_bytes(b"\x00" * 256)
        monkeypatch.setattr("tools.vision_tools._GEMINI_INLINE_MAX_BASE64_BYTES", 64)
        captured = {}
        with patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool(str(video), "Describe", "openai/gpt-4o"))
        assert json.loads(result)["success"] is True
        assert captured["messages"][0]["content"][1]["type"] == "video_url"
        assert captured["temperature"] == 0.1

    @pytest.mark.parametrize("url", [_HTTPS_MP4, _SIGNED_GCS_URL, _YOUTUBE_URL])
    def test_non_gemini_urls_still_download_to_video_url(self, url):
        """Regression guard: vLLM/OpenRouter users keep the download + video_url path (50 MB cap)."""
        captured = {}
        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._probe_remote_video_mime", new_callable=AsyncMock) as probe, \
             patch("tools.vision_tools._download_video", side_effect=_fake_download("video/mp4")) as mock_dl, \
             patch("tools.vision_tools.async_call_llm", side_effect=_capture_llm(captured)), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            self._run(video_analyze_tool(url, "Describe", "qwen/qwen3-vl-235b"))
        probe.assert_not_awaited()
        assert mock_dl.call_count == 1
        assert mock_dl.call_args.kwargs["max_bytes"] == _MAX_VIDEO_BASE64_BYTES
        part = captured["messages"][0]["content"][1]
        assert part == {"type": "video_url", "video_url": {"url": part["video_url"]["url"]}}
        assert part["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_non_gemini_over_cap_keeps_generic_error(self):
        async def too_large(url, destination, max_retries=3, max_bytes=_MAX_VIDEO_BASE64_BYTES):
            raise _VideoTooLargeError(60 * _MB, max_bytes)

        with patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)), \
             patch("tools.vision_tools._download_video", side_effect=too_large), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(_HTTPS_MP4, "Describe", "qwen/qwen3-vl-235b"))
        data = json.loads(result)
        assert data["success"] is False
        assert f"Video too large ({60 * _MB} bytes, max {_MAX_VIDEO_BASE64_BYTES})" in data["error"]
        assert "inline for Gemini" not in data["error"]
        mock_llm.assert_not_awaited()

    def test_unsupported_model_error_suggests_gemini_38(self, tmp_path):
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 16)
        with patch("tools.vision_tools.async_call_llm", new=AsyncMock(side_effect=RuntimeError("model does not support video input"))):
            result = self._run(video_analyze_tool(str(video), "Describe", "openai/gpt-4o"))
        data = json.loads(result)
        assert data["success"] is False
        assert "gemini-3.8-flash" in data["analysis"]
        assert "auxiliary.video.model" in data["analysis"]


# ---------------------------------------------------------------------------
# _resolve_video_settings / handler config plumbing
# ---------------------------------------------------------------------------


class TestResolveVideoSettings:
    def test_defaults_without_config(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")
        with patch("hermes_cli.config.load_config", return_value={}):
            settings = _resolve_video_settings()
        assert settings == {"model": None, "provider": None, "fps": 1, "detail": None}

    def test_reads_auxiliary_video_block(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")
        config = {"auxiliary": {"video": {
            "model": " gemini-3.8-flash ", "provider": "gemini", "fps": 2, "detail": "High",
        }}}
        with patch("hermes_cli.config.load_config", return_value=config):
            settings = _resolve_video_settings()
        assert settings == {"model": "gemini-3.8-flash", "provider": "gemini", "fps": 2, "detail": "high"}

    @pytest.mark.parametrize("fps", [0, None, "0"])
    def test_fps_zero_or_null_omits(self, fps, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")
        config = {"auxiliary": {"video": {"model": "gemini-3.8-flash", "fps": fps}}}
        with patch("hermes_cli.config.load_config", return_value=config):
            assert _resolve_video_settings()["fps"] is None

    def test_model_falls_back_to_vision_then_env(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-2.5-flash")
        config = {"auxiliary": {"vision": {"model": "google/gemini-3-flash", "provider": "openrouter"}}}
        with patch("hermes_cli.config.load_config", return_value=config):
            settings = _resolve_video_settings()
        # vision.model wins over env; vision.provider is NOT lifted (the vision task applies it itself).
        assert settings["model"] == "google/gemini-3-flash"
        assert settings["provider"] is None
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _resolve_video_settings()["model"] == "google/gemini-2.5-flash"

    def test_provider_auto_is_dropped(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")
        config = {"auxiliary": {"video": {"provider": "auto"}}}
        with patch("hermes_cli.config.load_config", return_value=config):
            assert _resolve_video_settings()["provider"] is None


class TestHandleVideoAnalyzePassesSettings:
    def test_handler_forwards_fps_detail_provider(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")
        config = {"auxiliary": {"video": {
            "model": "gemini-3.8-flash", "provider": "gemini", "fps": 0, "detail": "low",
        }}}
        with patch("hermes_cli.config.load_config", return_value=config), \
             patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": _HTTPS_MP4, "question": "what?"})
            )
        args, kwargs = mock_tool.call_args
        assert args[0] == _HTTPS_MP4
        assert "what?" in args[1]
        assert args[2] == "gemini-3.8-flash"
        assert kwargs == {"fps": None, "detail": "low", "provider": "gemini"}


# ---------------------------------------------------------------------------
# Offline proof against LiteLLM's Gemini converter (skipped when litellm is
# not importable — see the PR that introduced the file part for the output
# captured with the nolgia litellm fork)
# ---------------------------------------------------------------------------


class TestGeminiConverterOfflineProof:
    """Run the emitted content through LiteLLM's Gemini message converter."""

    def _convert(self, content):
        transformation = pytest.importorskip("litellm.llms.vertex_ai.gemini.transformation")
        contents = transformation._gemini_convert_messages_with_history(
            messages=[{"role": "user", "content": content}],
            model="gemini-3.8-flash",
            custom_llm_provider="gemini",
        )
        return contents[0]["parts"]

    def test_file_part_yields_file_data_with_video_metadata(self):
        part = _video_part_for_model("gemini-3.8-flash", _HTTPS_MP4, "video/mp4", fps=1)
        parts = self._convert([{"type": "text", "text": "Describe"}, part])
        media = [p for p in parts if "file_data" in p or "fileData" in p]
        assert len(media) == 1, parts
        file_data = media[0].get("file_data") or media[0].get("fileData")
        assert (file_data.get("file_uri") or file_data.get("fileUri")) == _HTTPS_MP4
        assert (file_data.get("mime_type") or file_data.get("mimeType")) == "video/mp4"
        video_metadata = media[0].get("video_metadata") or media[0].get("videoMetadata")
        assert video_metadata == {"fps": 1}

    def test_legacy_video_url_part_is_silently_dropped(self):
        """Documents the bug: a video_url part produces NO media part at all."""
        parts = self._convert([
            {"type": "text", "text": "Describe"},
            {"type": "video_url", "video_url": {"url": _HTTPS_MP4}},
        ])
        assert parts == [{"text": "Describe"}]
