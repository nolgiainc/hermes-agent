"""MEDIA: tag → platform asset resolution on API egress (Nolgia fork).

The Nolgia platform relay drives this gateway over ``POST /v1/runs`` (plus
the legacy ``/v1/chat/completions`` fallback) and persists the reply text
verbatim as a web-chat transcript. A pod-local ``MEDIA:<path>`` tag is
dead text for a web user — there is no shared filesystem — so in platform
mode (``NOLGIA_API_URL`` + ``NOLGIA_TOKEN`` present) egress uploads each
tagged file to the platform asset library and rewrites the tag to
``asset:<uuid>``; anything non-uploadable degrades to the bare filename.
Without the platform env every path keeps its exact upstream behavior.
"""

import asyncio
import base64
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms import nolgia_assets  # noqa: E402
from gateway.platforms.api_server import (  # noqa: E402
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)

_ASSET_ID = "12345678-1234-5678-1234-567812345678"
_ASSET_REF = f"asset:{_ASSET_ID}"


@pytest.fixture(autouse=True)
def _clear_asset_cache():
    """The upload-dedupe cache is process-global; isolate every test."""
    with nolgia_assets._asset_cache_lock:
        nolgia_assets._asset_cache.clear()
    yield
    with nolgia_assets._asset_cache_lock:
        nolgia_assets._asset_cache.clear()


@pytest.fixture
def platform_env(monkeypatch):
    """Simulate the platform deployment env (chart-injected on real pods)."""
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
    monkeypatch.setenv("NOLGIA_TOKEN", "test-token")


@pytest.fixture
def upload_calls(monkeypatch):
    """Stub the network upload; record (path, content_type, size) per call."""
    calls = []

    def _fake_upload(path, content_type, size_bytes):
        calls.append((str(path), content_type, size_bytes))
        return _ASSET_ID

    monkeypatch.setattr(nolgia_assets, "_upload_asset", _fake_upload)
    return calls


def _write_png(tmp_path, name="shot.png"):
    p = tmp_path / name
    p.write_bytes(_PNG_BYTES)
    return p


# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------


class TestResolverEnabled:
    def test_disabled_without_platform_env(self):
        # tests/conftest.py's hermetic env guarantees no ambient NOLGIA_*.
        assert not nolgia_assets.resolver_enabled()

    def test_disabled_with_only_one_var(self, monkeypatch):
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
        assert not nolgia_assets.resolver_enabled()
        monkeypatch.delenv("NOLGIA_API_URL")
        monkeypatch.setenv("NOLGIA_TOKEN", "test-token")
        assert not nolgia_assets.resolver_enabled()

    def test_enabled_with_both_vars(self, platform_env):
        assert nolgia_assets.resolver_enabled()

    def test_escape_hatch(self, platform_env, monkeypatch):
        monkeypatch.setenv("NOLGIA_MEDIA_RESOLVER", "0")
        assert not nolgia_assets.resolver_enabled()

    def test_api_base_appends_v1(self, monkeypatch):
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
        assert nolgia_assets._api_base() == "https://api.nolgia.test/v1"
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test/v1")
        assert nolgia_assets._api_base() == "https://api.nolgia.test/v1"
        monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test/v1/")
        assert nolgia_assets._api_base() == "https://api.nolgia.test/v1"


# ---------------------------------------------------------------------------
# Tag resolution (unit; upload stubbed)
# ---------------------------------------------------------------------------


class TestResolveMediaTagsToAssets:
    def test_media_tag_becomes_asset_ref(self, platform_env, upload_calls, tmp_path):
        p = _write_png(tmp_path)
        out = nolgia_assets.resolve_media_tags_to_assets(f"Here you go: MEDIA:{p} done")
        assert out == f"Here you go: {_ASSET_REF} done"
        assert "MEDIA:" not in out
        assert str(tmp_path) not in out
        assert upload_calls == [(str(p.resolve()), "image/png", len(_PNG_BYTES))]

    def test_content_type_map_matches_platform_enum(self, platform_env, upload_calls, tmp_path):
        expectations = {
            "clip.mp4": "video/mp4",
            "clip.mov": "video/quicktime",
            "clip.webm": "video/webm",
            "sound.mp3": "audio/mpeg",
            "sound.wav": "audio/wav",
            "sound.ogg": "audio/ogg",
            "sound.m4a": "audio/mp4",
            "img.jpg": "image/jpeg",
            "img.jpeg": "image/jpeg",
            "img.webp": "image/webp",
        }
        for name, content_type in expectations.items():
            f = tmp_path / name
            f.write_bytes(b"data")
            out = nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{f}")
            assert out == _ASSET_REF, name
            assert upload_calls[-1][1] == content_type, name

    def test_missing_file_degrades_to_basename(self, platform_env, upload_calls):
        out = nolgia_assets.resolve_media_tags_to_assets(
            "see MEDIA:/nonexistent/dir/shot2.png ok"
        )
        assert out == "see shot2.png ok"
        assert upload_calls == []

    def test_non_uploadable_extension_degrades_to_basename(
        self, platform_env, upload_calls, tmp_path
    ):
        # .html and .gif are connector-deliverable (MEDIA_DELIVERY_EXTS) but
        # not in the platform's content_type enum — degrade, don't upload.
        page = tmp_path / "page.html"
        page.write_text("<html></html>", encoding="utf-8")
        gif = tmp_path / "anim.gif"
        gif.write_bytes(b"GIF89a")
        out = nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{page} and MEDIA:{gif}")
        assert out == "page.html and anim.gif"
        assert upload_calls == []

    def test_quoted_and_backticked_tags(self, platform_env, upload_calls, tmp_path):
        spaced = tmp_path / "my shot.png"
        spaced.write_bytes(_PNG_BYTES)
        assert nolgia_assets.resolve_media_tags_to_assets(f'MEDIA:"{spaced}"') == _ASSET_REF
        p = _write_png(tmp_path)
        out = nolgia_assets.resolve_media_tags_to_assets(f"see `MEDIA:{p}` above")
        assert out == f"see {_ASSET_REF} above"

    def test_extensionless_tag_degrades_to_basename(self, platform_env, upload_calls, tmp_path):
        # Extension-less files (Caddyfile-style) are never uploadable media;
        # both the existing and the missing variant degrade to the filename
        # and surrounding spacing survives the extension-less matcher's
        # trailing-whitespace consumption.
        binfile = tmp_path / "output"
        binfile.write_bytes(b"data")
        out = nolgia_assets.resolve_media_tags_to_assets(f"result MEDIA:{binfile} end")
        assert out == "result output end"
        out2 = nolgia_assets.resolve_media_tags_to_assets("result MEDIA:/no/such/binfile end")
        assert out2 == "result binfile end"
        assert upload_calls == []

    def test_repeat_tag_uploads_once(self, platform_env, upload_calls, tmp_path):
        p = _write_png(tmp_path)
        out = nolgia_assets.resolve_media_tags_to_assets(f"a MEDIA:{p} b MEDIA:{p} c")
        assert out == f"a {_ASSET_REF} b {_ASSET_REF} c"
        assert len(upload_calls) == 1

    def test_two_tags_same_line_both_resolve(self, platform_env, upload_calls, tmp_path):
        # The shared matcher's unquoted multiword-path support lets one
        # match lazily span a second tag on the same line; the resolver must
        # still resolve each file separately.
        p = _write_png(tmp_path)
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"vid")
        out = nolgia_assets.resolve_media_tags_to_assets(f"x MEDIA:{p} then MEDIA:{clip} y")
        assert out == f"x {_ASSET_REF} then {_ASSET_REF} y"
        assert [c[1] for c in upload_calls] == ["image/png", "video/mp4"]

    def test_two_missing_tags_same_line_degrade_separately(self, platform_env, upload_calls):
        out = nolgia_assets.resolve_media_tags_to_assets("MEDIA:/no/a.png and MEDIA:/no/b.png")
        assert out == "a.png and b.png"
        assert upload_calls == []

    def test_upload_failure_degrades_to_basename(self, platform_env, monkeypatch, tmp_path):
        monkeypatch.setattr(nolgia_assets, "_upload_asset", lambda *args: None)
        p = _write_png(tmp_path)
        out = nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{p}")
        assert out == "shot.png"

    def test_oversize_file_degrades_to_basename(
        self, platform_env, upload_calls, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(nolgia_assets._EXT_CONTENT_TYPES, ".png", ("image/png", 1))
        p = _write_png(tmp_path)
        out = nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{p}")
        assert out == "shot.png"
        assert upload_calls == []

    def test_per_message_upload_cap(self, platform_env, monkeypatch, tmp_path):
        calls = []

        def _fake_upload(path, content_type, size_bytes):
            calls.append(str(path))
            # Distinct ids so the cache can't mask the cap.
            return f"00000000-0000-0000-0000-{len(calls):012d}"

        monkeypatch.setattr(nolgia_assets, "_upload_asset", _fake_upload)
        files = []
        for index in range(nolgia_assets._MAX_UPLOADS_PER_MESSAGE + 1):
            f = tmp_path / f"m{index}.jpg"
            f.write_bytes(b"jpeg")
            files.append(f)
        text = "\n".join(f"MEDIA:{f}" for f in files)
        out = nolgia_assets.resolve_media_tags_to_assets(text)
        lines = out.splitlines()
        assert len(calls) == nolgia_assets._MAX_UPLOADS_PER_MESSAGE
        assert all(line.startswith("asset:") for line in lines[:-1])
        assert lines[-1] == f"m{nolgia_assets._MAX_UPLOADS_PER_MESSAGE}.jpg"

    def test_text_without_tags_passthrough(self, platform_env, upload_calls):
        assert nolgia_assets.resolve_media_tags_to_assets("plain text") == "plain text"
        assert nolgia_assets.resolve_media_tags_to_assets("") == ""
        prose = "Our SOCIAL MEDIA: strategy is fine"
        assert nolgia_assets.resolve_media_tags_to_assets(prose) == prose
        assert upload_calls == []

    def test_credential_shaped_path_never_uploaded(self, platform_env, upload_calls):
        # validate_media_delivery_path's denylist (and existence check)
        # rejects it; the reply keeps only the filename.
        out = nolgia_assets.resolve_media_tags_to_assets("MEDIA:~/.ssh/id_rsa.png")
        assert out == "id_rsa.png"
        assert upload_calls == []


# ---------------------------------------------------------------------------
# Endpoint egress (runs path + chat completions fallback)
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_api_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


def _mock_agent(final_response: str) -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": final_response}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


async def _poll_run_until_completed(cli, run_id):
    status = None
    for _ in range(60):
        status_resp = await cli.get(f"/v1/runs/{run_id}")
        assert status_resp.status == 200
        status = await status_resp.json()
        if status["status"] == "completed":
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"run never completed: {status}")


class TestRunsPathPlatformAssetEgress:
    @pytest.mark.asyncio
    async def test_run_output_and_completed_event_carry_asset_refs(
        self, platform_env, upload_calls, tmp_path
    ):
        """GET /v1/runs/{id} output AND the run.completed event body must
        carry asset:<uuid> and no MEDIA: tag — this is exactly the egress
        the platform relay persists as the web-chat transcript."""
        png = _write_png(tmp_path)
        adapter = _make_adapter()
        app = _create_api_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(f"done MEDIA:{png}")

                resp = await cli.post("/v1/runs", json={"input": "screenshot please"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                status = await _poll_run_until_completed(cli, run_id)
                assert status["output"] == f"done {_ASSET_REF}"
                assert "MEDIA:" not in status["output"]
                assert str(tmp_path) not in status["output"]

                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                events_body = await events_resp.text()
                assert "run.completed" in events_body
                assert _ASSET_REF in events_body
                assert "MEDIA:" not in events_body
        assert len(upload_calls) == 1

    @pytest.mark.asyncio
    async def test_unresolvable_tag_never_leaks_local_path(
        self, platform_env, upload_calls, tmp_path
    ):
        missing = tmp_path / "gone.png"  # never written
        adapter = _make_adapter()
        app = _create_api_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(f"see MEDIA:{missing} ok")

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                status = await _poll_run_until_completed(cli, run_id)

        assert status["output"] == "see gone.png ok"
        assert str(tmp_path) not in status["output"]
        assert upload_calls == []

    @pytest.mark.asyncio
    async def test_without_platform_env_runs_output_is_upstream_raw(self, tmp_path):
        """No NOLGIA_* env → the runs path keeps its exact upstream
        behavior (raw final_response, no data-URL injection either)."""
        text = "done MEDIA:/nonexistent/shot.png"
        adapter = _make_adapter()
        app = _create_api_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(text)

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                status = await _poll_run_until_completed(cli, run_id)

        assert status["output"] == text


class TestChatCompletionsPlatformAssetEgress:
    @pytest.mark.asyncio
    async def test_non_streaming_uses_asset_resolver_not_data_urls(
        self, platform_env, upload_calls, tmp_path
    ):
        """In platform mode the legacy relay fallback must upload assets
        instead of inlining multi-MB base64 data URLs."""
        png = _write_png(tmp_path)
        adapter = _make_adapter()
        app = _create_api_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(f"Here: MEDIA:{png}")

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        assert content == f"Here: {_ASSET_REF}"
        assert "data:image/png;base64," not in content
        assert "MEDIA:" not in content
        assert len(upload_calls) == 1

    @pytest.mark.asyncio
    async def test_without_platform_env_keeps_data_url_behavior(self, tmp_path):
        png = _write_png(tmp_path)
        adapter = _make_adapter()
        app = _create_api_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(f"Here: MEDIA:{png}")

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        assert "data:image/png;base64," in content
        assert "asset:" not in content
