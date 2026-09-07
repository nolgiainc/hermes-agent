"""The registered terminal tool must distinguish executable binaries from scripts.

A compiled ``nolgia`` binary as the first command token must reach the backend even when its
machine code decodes to NUL-bearing paths and lifecycle text. Real scripts containing gateway
lifecycle commands must remain blocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.terminal_tool as terminal_tool
from tools.registry import registry


ELF_BYTES = (
    b"\x7fELF\x02\x01\x01\x00"
    + bytes(64)
    + b"/opt/data/bin/nolgia\x00\x01 --run\nhermes gateway restart\x00"
    + b"\x90" * 4096
)


@pytest.fixture
def terminal_env(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "nolgia"
    binary.parent.mkdir()
    binary.write_bytes(ELF_BYTES)
    binary.chmod(0o755)

    class FakeEnv:
        env: dict = {}

        def __init__(self):
            self.cwd = str(tmp_path)
            self.calls: list[str] = []

        def execute(self, command, **_kwargs):
            self.calls.append(command)
            if command.startswith("head -c"):
                return {
                    "output": binary.read_bytes().decode("utf-8", errors="replace"),
                    "returncode": 0,
                }
            return {"output": "nolgia-video-prompting  v1", "returncode": 0}

    from tools import process_registry

    fake_env = FakeEnv()
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": fake_env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(tmp_path),
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    return binary, fake_env


def test_registered_terminal_runs_binary_first_token(terminal_env):
    binary, fake_env = terminal_env
    command = f"{binary} skills show nolgia-video-prompting"

    result = json.loads(
        registry.dispatch("terminal", {"command": command}, task_id="binary-first-token")
    )

    assert result == {
        "output": "nolgia-video-prompting  v1",
        "exit_code": 0,
        "error": None,
    }
    assert fake_env.calls == [command]


def test_registered_terminal_still_blocks_lifecycle_script(terminal_env):
    binary, fake_env = terminal_env
    wrapper = binary.parent / "restart.sh"
    wrapper.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")

    result = json.loads(
        registry.dispatch(
            "terminal",
            {"command": f"bash {wrapper}"},
            task_id="lifecycle-script",
        )
    )

    assert "Blocked" in result["error"]
    assert fake_env.calls == []
