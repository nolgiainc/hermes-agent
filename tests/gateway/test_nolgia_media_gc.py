"""Confirmed-in-library media GC (Nolgia fork, NOL-516).

Every generation is persisted to the platform library (GCS) server-side; the
pod-local copy is redundant afterwards and its accumulation is what filled
agent PVCs. This module deletes those local copies — but ONLY when the
library is proven to hold the same bytes. The bar these tests defend is
asymmetric on purpose: a confirmed file must be deleted, and an unconfirmed
file must NEVER be, for every flavour of "unconfirmed" (no ledger entry, API
error, non-ready asset, size mismatch, hash mismatch, file changed since
confirmation, GC disabled).
"""

import base64
import hashlib

import pytest

from gateway.platforms import nolgia_assets, nolgia_media_gc

_ASSET_ID = "12345678-1234-5678-1234-567812345678"
_OTHER_ASSET_ID = "87654321-4321-8765-4321-876543218765"


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    """Point HERMES_HOME (and thus the GC ledger) at a per-test tmp dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(nolgia_media_gc, "get_hermes_home", lambda: tmp_path)
    nolgia_media_gc._reset_ledger_for_tests()
    yield
    nolgia_media_gc._reset_ledger_for_tests()


@pytest.fixture
def platform_env(monkeypatch):
    """The chart-injected platform env that turns GC on."""
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.nolgia.test")
    monkeypatch.setenv("NOLGIA_TOKEN", "test-token")
    monkeypatch.delenv("NOLGIA_MEDIA_GC", raising=False)
    monkeypatch.delenv("NOLGIA_MEDIA_GC_ON_UPLOAD", raising=False)
    monkeypatch.delenv("NOLGIA_MEDIA_GC_MIN_AGE_HOURS", raising=False)


def _write_media(path, payload=b"video-bytes-0123456789", age_hours=None):
    """Create a media file, optionally back-dated past the sweep age gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if age_hours is not None:
        import os
        import time

        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
    return path


def _md5_b64(payload):
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


def _stat_pair(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class TestConfigGates:
    def test_disabled_without_platform_env(self, monkeypatch):
        monkeypatch.delenv("NOLGIA_API_URL", raising=False)
        monkeypatch.delenv("NOLGIA_TOKEN", raising=False)
        assert nolgia_media_gc.media_gc_enabled() is False

    def test_enabled_by_default_in_platform_mode(self, platform_env):
        assert nolgia_media_gc.media_gc_enabled() is True

    def test_master_escape_hatch(self, platform_env, monkeypatch):
        monkeypatch.setenv("NOLGIA_MEDIA_GC", "0")
        assert nolgia_media_gc.media_gc_enabled() is False

    def test_on_upload_escape_hatch_keeps_master_on(self, platform_env, monkeypatch):
        monkeypatch.setenv("NOLGIA_MEDIA_GC_ON_UPLOAD", "0")
        assert nolgia_media_gc.media_gc_enabled() is True
        assert nolgia_media_gc._on_upload_delete_enabled() is False

    def test_sweep_interval_has_a_floor(self, platform_env, monkeypatch):
        monkeypatch.setenv("NOLGIA_MEDIA_GC_INTERVAL_SECONDS", "1")
        assert nolgia_media_gc.sweep_interval_seconds() >= 60.0

    def test_bad_config_values_fall_back_to_defaults(self, platform_env, monkeypatch):
        monkeypatch.setenv("NOLGIA_MEDIA_GC_MIN_AGE_HOURS", "not-a-number")
        assert nolgia_media_gc._min_age_seconds() == pytest.approx(6 * 3600)


class TestPostUploadHook:
    """The strongest signal: POST /assets/uploads/{id}/complete returned an id."""

    def test_confirmed_upload_deletes_local_file(self, platform_env, tmp_path):
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)

        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert not path.exists()

    def test_deletion_is_logged_with_path_size_and_basis(
        self, platform_env, tmp_path, caplog
    ):
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)

        with caplog.at_level("INFO", logger="gateway.platforms.nolgia_media_gc"):
            nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        message = "\n".join(record.getMessage() for record in caplog.records)
        assert str(path) in message
        assert str(size) in message
        assert _ASSET_ID in message
        assert "confirmed in library" in message

    def test_file_changed_since_confirmation_is_kept(self, platform_env, tmp_path):
        """Re-render between upload and hook: those bytes are NOT in the library."""
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)
        _write_media(path, payload=b"a-completely-different-render")

        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert path.exists()

    def test_disabled_gc_never_deletes(self, platform_env, monkeypatch, tmp_path):
        monkeypatch.setenv("NOLGIA_MEDIA_GC", "0")
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)

        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert path.exists()

    def test_on_upload_opt_out_keeps_file_but_records_ledger(
        self, platform_env, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("NOLGIA_MEDIA_GC_ON_UPLOAD", "0")
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)

        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert path.exists()
        assert (
            nolgia_media_gc._get_ledger().uploaded_asset_for(str(path), size, mtime_ns)
            == _ASSET_ID
        )

    def test_file_outside_hermes_home_is_kept(self, platform_env, tmp_path):
        """A user's own file outside the agent workspace is never GC'd."""
        outside = tmp_path.parent / "outside-home"
        path = _write_media(outside / "user-clip.mp4")
        size, mtime_ns = _stat_pair(path)
        try:
            nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_non_media_extension_is_kept(self, platform_env, tmp_path):
        path = _write_media(tmp_path / "gen" / "notes.txt")
        size, mtime_ns = _stat_pair(path)

        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert path.exists()

    def test_deleted_path_still_resolves_to_its_asset(self, platform_env, tmp_path):
        """A later MEDIA: tag for a GC'd file resolves to the library asset."""
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        size, mtime_ns = _stat_pair(path)
        nolgia_media_gc.on_confirmed_upload(path, size, mtime_ns, _ASSET_ID)

        assert nolgia_media_gc.deleted_asset_reference(str(path)) == _ASSET_ID

    def test_unknown_path_has_no_asset_reference(self, platform_env, tmp_path):
        assert nolgia_media_gc.deleted_asset_reference(str(tmp_path / "nope.mp4")) is None


class TestLedgerPersistence:
    def test_ledger_survives_process_restart(self, platform_env, tmp_path):
        path = tmp_path / "gen" / "clip.mp4"
        nolgia_media_gc._get_ledger().record_uploaded(str(path), 42, 4242, _ASSET_ID)

        nolgia_media_gc._reset_ledger_for_tests()  # simulate a pod roll

        assert (
            nolgia_media_gc._get_ledger().uploaded_asset_for(str(path), 42, 4242)
            == _ASSET_ID
        )

    def test_ledger_identity_includes_size_and_mtime(self, platform_env, tmp_path):
        path = tmp_path / "gen" / "clip.mp4"
        nolgia_media_gc._get_ledger().record_uploaded(str(path), 42, 4242, _ASSET_ID)
        ledger = nolgia_media_gc._get_ledger()

        assert ledger.uploaded_asset_for(str(path), 99, 4242) is None
        assert ledger.uploaded_asset_for(str(path), 42, 9999) is None


class TestSweeperLedgerBasis:
    """Basis 1: our own confirmed-upload ledger, re-verified against the API."""

    def test_ledger_hit_with_ready_asset_is_deleted(
        self, platform_env, tmp_path, monkeypatch
    ):
        path = _write_media(tmp_path / "gen" / "old.mp4", age_hours=48)
        size, mtime_ns = _stat_pair(path)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), size, mtime_ns, _ASSET_ID
        )
        monkeypatch.setattr(
            nolgia_media_gc,
            "_api_get",
            lambda p, t: {"id": _ASSET_ID, "status": "ready", "size_bytes": size},
        )

        deleted, freed = nolgia_media_gc.sweep_once(tmp_path)

        assert (deleted, freed) == (1, size)
        assert not path.exists()

    def test_api_error_keeps_the_file(self, platform_env, tmp_path, monkeypatch):
        path = _write_media(tmp_path / "gen" / "old.mp4", age_hours=48)
        size, mtime_ns = _stat_pair(path)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), size, mtime_ns, _ASSET_ID
        )

        def _boom(p, t):
            raise OSError("platform API unreachable")

        monkeypatch.setattr(nolgia_media_gc, "_api_get", _boom)

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()

    def test_non_ready_asset_keeps_the_file(self, platform_env, tmp_path, monkeypatch):
        path = _write_media(tmp_path / "gen" / "old.mp4", age_hours=48)
        size, mtime_ns = _stat_pair(path)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), size, mtime_ns, _ASSET_ID
        )
        monkeypatch.setattr(
            nolgia_media_gc,
            "_api_get",
            lambda p, t: {"id": _ASSET_ID, "status": "pending", "size_bytes": size},
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()

    def test_size_mismatch_on_recheck_keeps_the_file(
        self, platform_env, tmp_path, monkeypatch
    ):
        path = _write_media(tmp_path / "gen" / "old.mp4", age_hours=48)
        size, mtime_ns = _stat_pair(path)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), size, mtime_ns, _ASSET_ID
        )
        monkeypatch.setattr(
            nolgia_media_gc,
            "_api_get",
            lambda p, t: {"id": _ASSET_ID, "status": "ready", "size_bytes": size + 1},
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()


class TestSweeperContentBasis:
    """Basis 2: size + MD5 match against the GCS object behind the asset."""

    def test_content_match_is_deleted(self, platform_env, tmp_path, monkeypatch):
        payload = b"generated-video-bytes"
        path = _write_media(tmp_path / "gen" / "downloaded.mp4", payload, age_hours=48)
        monkeypatch.setattr(
            nolgia_media_gc,
            "_build_library_size_index",
            lambda: {
                len(payload): [
                    {"id": _ASSET_ID, "signed_url": "https://gcs.test/o", "size_bytes": len(payload)}
                ]
            },
        )
        monkeypatch.setattr(
            nolgia_media_gc,
            "_gcs_object_md5",
            lambda url: (_md5_b64(payload), len(payload)),
        )

        deleted, freed = nolgia_media_gc.sweep_once(tmp_path)

        assert (deleted, freed) == (1, len(payload))
        assert not path.exists()

    def test_hash_mismatch_keeps_the_file(self, platform_env, tmp_path, monkeypatch):
        """Same size, different bytes — the library does NOT hold this file."""
        payload = b"locally-edited-render"
        path = _write_media(tmp_path / "gen" / "edited.mp4", payload, age_hours=48)
        monkeypatch.setattr(
            nolgia_media_gc,
            "_build_library_size_index",
            lambda: {
                len(payload): [
                    {"id": _OTHER_ASSET_ID, "signed_url": "https://gcs.test/o", "size_bytes": len(payload)}
                ]
            },
        )
        monkeypatch.setattr(
            nolgia_media_gc,
            "_gcs_object_md5",
            lambda url: (_md5_b64(b"a-different-object!!!"), len(payload)),
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()

    def test_no_size_match_in_library_keeps_the_file(
        self, platform_env, tmp_path, monkeypatch
    ):
        path = _write_media(tmp_path / "gen" / "never-uploaded.mp4", age_hours=48)
        monkeypatch.setattr(nolgia_media_gc, "_build_library_size_index", lambda: {})

        def _no_probe(url):  # pragma: no cover - must never be reached
            raise AssertionError("hash probe attempted without a size match")

        monkeypatch.setattr(nolgia_media_gc, "_gcs_object_md5", _no_probe)

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()

    def test_unavailable_gcs_hash_keeps_the_file(
        self, platform_env, tmp_path, monkeypatch
    ):
        payload = b"generated-video-bytes"
        path = _write_media(tmp_path / "gen" / "clip.mp4", payload, age_hours=48)
        monkeypatch.setattr(
            nolgia_media_gc,
            "_build_library_size_index",
            lambda: {
                len(payload): [
                    {"id": _ASSET_ID, "signed_url": "https://gcs.test/o", "size_bytes": len(payload)}
                ]
            },
        )
        monkeypatch.setattr(nolgia_media_gc, "_gcs_object_md5", lambda url: None)

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()


class TestSweeperScope:
    def test_recent_media_is_never_swept(self, platform_env, tmp_path, monkeypatch):
        """Fresh output may still be in use by the turn that made it."""
        payload = b"just-rendered"
        path = _write_media(tmp_path / "gen" / "fresh.mp4", payload)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), *_stat_pair(path), _ASSET_ID
        )

        def _no_api(p, t):  # pragma: no cover - must never be reached
            raise AssertionError("age gate did not hold")

        monkeypatch.setattr(nolgia_media_gc, "_api_get", _no_api)

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()

    def test_protected_trees_are_not_swept(self, platform_env, tmp_path, monkeypatch):
        """Logs/sessions/skills/git trees hold no reclaimable generated media."""
        protected = [
            _write_media(tmp_path / "sessions" / "s1" / "a.mp4", age_hours=48),
            _write_media(tmp_path / "skills" / "demo" / "b.png", age_hours=48),
            _write_media(tmp_path / "logs" / "c.mp4", age_hours=48),
            _write_media(tmp_path / "repo" / ".git" / "d.png", age_hours=48),
            _write_media(tmp_path / "ability-versions" / "x" / "e.mp4", age_hours=48),
        ]
        monkeypatch.setattr(
            nolgia_media_gc,
            "_build_library_size_index",
            lambda: pytest.fail("protected tree entered the confirmation path"),
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert all(p.exists() for p in protected)

    def test_non_media_files_are_never_swept(self, platform_env, tmp_path, monkeypatch):
        keep = [
            _write_media(tmp_path / "work" / "notes.md", age_hours=48),
            _write_media(tmp_path / "work" / "state.db", age_hours=48),
            _write_media(tmp_path / "work" / "script.py", age_hours=48),
        ]
        monkeypatch.setattr(
            nolgia_media_gc,
            "_build_library_size_index",
            lambda: pytest.fail("non-media file entered the confirmation path"),
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert all(p.exists() for p in keep)

    def test_disabled_gc_sweeps_nothing(self, platform_env, monkeypatch, tmp_path):
        monkeypatch.setenv("NOLGIA_MEDIA_GC", "0")
        path = _write_media(tmp_path / "gen" / "old.mp4", age_hours=48)
        nolgia_media_gc._get_ledger().record_uploaded(
            str(path), *_stat_pair(path), _ASSET_ID
        )

        assert nolgia_media_gc.sweep_once(tmp_path) == (0, 0)
        assert path.exists()


class TestGcsHashProbe:
    def test_parses_md5_and_total_size_from_ranged_get(self, monkeypatch):
        """GCS answers a 1-byte range with the object's md5 and full size."""

        class _Headers:
            def get_all(self, name):
                assert name == "x-goog-hash"
                return ["crc32c=0Y5Djg==", "md5=l8HcH9B7ViE7qdBGRsg7HA=="]

            def get(self, name, default=""):
                return "bytes 0-0/3587674" if name == "Content-Range" else default

        class _Response:
            headers = _Headers()

            def read(self):
                return b"x"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            nolgia_media_gc.urllib.request, "urlopen", lambda *a, **k: _Response()
        )

        assert nolgia_media_gc._gcs_object_md5("https://gcs.test/o") == (
            "l8HcH9B7ViE7qdBGRsg7HA==",
            3587674,
        )

    def test_probe_failure_returns_none(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr(nolgia_media_gc.urllib.request, "urlopen", _boom)

        assert nolgia_media_gc._gcs_object_md5("https://gcs.test/o") is None


class TestUploadIntegration:
    """The egress upload path hands confirmed files to GC and reuses the ledger."""

    def test_successful_upload_deletes_the_local_file(
        self, platform_env, tmp_path, monkeypatch
    ):
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        monkeypatch.setattr(
            nolgia_assets, "_upload_asset", lambda *a, **k: _ASSET_ID
        )

        result = nolgia_assets.resolve_media_tags_to_assets(f"here: MEDIA:{path}")

        assert result == f"here: asset:{_ASSET_ID}"
        assert not path.exists()

    def test_failed_upload_never_deletes(self, platform_env, tmp_path, monkeypatch):
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        monkeypatch.setattr(nolgia_assets, "_upload_asset", lambda *a, **k: None)

        result = nolgia_assets.resolve_media_tags_to_assets(f"here: MEDIA:{path}")

        assert result == "here: clip.mp4"
        assert path.exists()

    def test_tag_for_gc_deleted_file_resolves_to_the_asset(
        self, platform_env, tmp_path, monkeypatch
    ):
        """A second turn referencing the deleted path still gets a media card."""
        path = _write_media(tmp_path / "gen" / "clip.mp4")
        monkeypatch.setattr(
            nolgia_assets, "_upload_asset", lambda *a, **k: _ASSET_ID
        )
        nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{path}")
        assert not path.exists()

        def _no_upload(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("re-upload attempted for a deleted file")

        monkeypatch.setattr(nolgia_assets, "_upload_asset", _no_upload)

        assert (
            nolgia_assets.resolve_media_tags_to_assets(f"again MEDIA:{path}")
            == f"again asset:{_ASSET_ID}"
        )

    def test_missing_unknown_file_still_degrades_to_basename(
        self, platform_env, tmp_path
    ):
        missing = tmp_path / "gen" / "never-existed.mp4"

        assert (
            nolgia_assets.resolve_media_tags_to_assets(f"MEDIA:{missing}")
            == "never-existed.mp4"
        )
