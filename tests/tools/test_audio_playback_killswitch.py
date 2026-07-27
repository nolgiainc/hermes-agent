"""HERMES_DISABLE_AUDIO_PLAYBACK must silence every speaker path.

Regression guard for the suite-wide hermetic invariant 5 (tests/conftest.py):
the default TTS provider (edge) is keyless, so without this kill-switch the
gateway tests that drive the real ``stream_tts_to_speaker`` pipeline
synthesize real audio and play it out loud on the developer's machine.
"""

import os
import queue
import threading
import wave
from unittest.mock import MagicMock, patch

import pytest


def _write_tiny_wav(path: str) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 80)


def test_suite_sets_killswitch():
    """The conftest hermetic environment must set the kill-switch for every test."""
    assert os.environ.get("HERMES_DISABLE_AUDIO_PLAYBACK") == "1"


def test_play_audio_file_noop_when_disabled(tmp_path, monkeypatch):
    """play_audio_file must not touch sounddevice or spawn a player process."""
    import tools.voice_mode as voice_mode

    wav = tmp_path / "tone.wav"
    _write_tiny_wav(str(wav))

    monkeypatch.setenv("HERMES_DISABLE_AUDIO_PLAYBACK", "1")
    with patch.object(voice_mode.subprocess, "Popen") as mock_popen, \
         patch.object(voice_mode, "_import_audio") as mock_audio:
        assert voice_mode.play_audio_file(str(wav)) is False
        mock_popen.assert_not_called()
        mock_audio.assert_not_called()


def test_play_audio_file_reaches_players_when_enabled(tmp_path, monkeypatch):
    """Sanity: without the kill-switch the same call proceeds to a player."""
    import tools.voice_mode as voice_mode

    wav = tmp_path / "tone.wav"
    _write_tiny_wav(str(wav))

    monkeypatch.delenv("HERMES_DISABLE_AUDIO_PLAYBACK", raising=False)
    with patch.object(voice_mode, "_import_audio", side_effect=ImportError), \
         patch.object(voice_mode.subprocess, "Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        voice_mode.play_audio_file(str(wav))
        mock_popen.assert_called()


def test_play_beep_noop_when_disabled(monkeypatch):
    import tools.voice_mode as voice_mode

    monkeypatch.setenv("HERMES_DISABLE_AUDIO_PLAYBACK", "1")
    with patch.object(voice_mode, "_import_audio") as mock_audio:
        voice_mode.play_beep()
        mock_audio.assert_not_called()


def test_stream_tts_display_only_when_disabled(monkeypatch):
    """stream_tts_to_speaker must not synthesize or play — display only."""
    import tools.tts_tool as tts_tool
    import tools.voice_mode as voice_mode

    monkeypatch.setenv("HERMES_DISABLE_AUDIO_PLAYBACK", "1")
    text_q = queue.Queue()
    stop_evt = threading.Event()
    done_evt = threading.Event()
    spoken = []

    text_q.put("This sentence must never be audible. ")
    text_q.put(None)

    with patch.object(tts_tool, "text_to_speech_tool") as mock_tts, \
         patch.object(voice_mode, "play_audio_file") as mock_play:
        tts_tool.stream_tts_to_speaker(
            text_q, stop_evt, done_evt,
            display_callback=lambda t: spoken.append(t),
        )

    assert done_evt.is_set()
    assert any("never be audible" in s for s in spoken)
    mock_tts.assert_not_called()
    mock_play.assert_not_called()
