"""The terminal tool must run a command whose first token is a compiled binary
(nolgiainc/nolgia-agent#228; prod cust-2774ee97, 2026-09-06).

``/opt/data/bin/nolgia skills show <ability>`` failed twice with
``Failed to execute command: open: embedded null character in path``. The gateway lifecycle guard
treated the 12 MB ELF as a "referenced script": the local read skipped it as binary, the terminal
tool's ``read_remote_script`` fallback then handed back the binary's *decoded* bytes, the recursion
tokenized machine code into a NUL-bearing path, and ``os.open`` raised ``ValueError`` past an
``OSError``-only handler — out of ``terminal_tool`` as a tool failure. The model fell back to
``execute_code``, where the sandbox had no ``NOLGIA_TOKEN`` (``test_nolgia_sandbox_env_grant.py``).

Upstream hardened the guard per syscall and then made it total (never raises). This module pins
the incident's exact shape at the TERMINAL-TOOL boundary — an ELF as the first token, the
binary-decoded fallback, a NUL-bearing path token — so a future guard change that lets any of
those escape fails here, not on a customer pod.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cron.lifecycle_guard import (
    _read_referenced_script,
    contains_gateway_lifecycle_command_or_referenced_script,
)
from tools.terminal_tool_guards import gateway_lifecycle_block

# ELF magic + header padding, then machine-code-shaped bytes that tokenize into a NUL-bearing
# absolute path and a literal lifecycle phrase. Neither may be scanned: a binary the user executes
# is not a referenced *shell script*.
ELF_BYTES = (
    b"\x7fELF\x02\x01\x01\x00" + bytes(64)
    + b"/opt/data/bin/nolgia\x00\x01 --run\nhermes gateway restart\x00"
    + b"\x90" * 4096
)


@pytest.fixture
def nolgia_binary(tmp_path) -> Path:
    binary = tmp_path / "bin" / "nolgia"
    binary.parent.mkdir()
    binary.write_bytes(ELF_BYTES)
    binary.chmod(0o755)
    return binary


class _FakeEnv:
    """Local-backend stand-in. ``execute`` records calls; a ``head -c`` read (the guard's
    remote fallback) returns the binary's decoded bytes, exactly what the fallback returned
    before the upstream hardening."""

    env: dict = {}

    def __init__(self, cwd: str, binary: Path):
        self.cwd = cwd
        self._binary = binary
        self.calls: list[str] = []

    def execute(self, command, **_kwargs):
        self.calls.append(command)
        if command.startswith("head -c"):
            return {"output": self._binary.read_bytes().decode("utf-8", errors="replace"),
                    "returncode": 0}
        return {"output": "nolgia-video-prompting  v1", "returncode": 0}


def _inside_supervised_gateway(monkeypatch):
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


def test_guard_lets_a_binary_first_token_through_without_raising(monkeypatch, tmp_path, nolgia_binary):
    _inside_supervised_gateway(monkeypatch)
    fake_env = _FakeEnv(str(tmp_path), nolgia_binary)

    blocked = gateway_lifecycle_block(
        command=f"{nolgia_binary} skills show nolgia-video-prompting",
        env=fake_env, env_type="local", cwd=str(tmp_path), workdir=None, session_key="tf-k",
    )

    assert blocked is None
    # The local read recognised the ELF, so the remote fallback was never asked to `cat` it.
    assert fake_env.calls == []


def test_terminal_tool_runs_the_binary_instead_of_failing(monkeypatch, tmp_path, nolgia_binary):
    """End to end: the incident command reaches the backend and returns its output, not
    ``Failed to execute command: ...``."""
    import tools.terminal_tool as tt

    _inside_supervised_gateway(monkeypatch)
    fake_env = _FakeEnv(str(tmp_path), nolgia_binary)
    monkeypatch.setattr(tt, "_active_environments", {"default": fake_env})
    monkeypatch.setattr(tt, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(
        tt, "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60, "lifetime_seconds": 3600},
    )
    command = f"{nolgia_binary} skills show nolgia-video-prompting"

    result = json.loads(tt.terminal_tool(command=command))

    assert result["exit_code"] == 0, result
    assert result["error"] is None
    assert result["output"] == "nolgia-video-prompting  v1"
    assert fake_env.calls == [command]


def test_binary_decoded_fallback_text_is_a_verdict_not_a_crash(nolgia_binary):
    """The pre-hardening fallback shape: the callback returns the ELF's decoded bytes (NUL
    preserved by ``errors="replace"``). Nothing to scan — and the lifecycle phrase embedded in
    the machine code must not false-positive either."""
    decoded = ELF_BYTES.decode("utf-8", errors="replace")

    verdict = contains_gateway_lifecycle_command_or_referenced_script(
        "/opt/data/bin/nolgia skills show nolgia-video-prompting",
        cwd="/opt/data",
        read_remote_script=lambda _path: decoded,
    )

    assert verdict is False


@pytest.mark.parametrize(
    "path",
    [
        Path("/opt/data/bin/nolgia\x00\x01"),
        Path("\x00"),
    ],
)
def test_read_referenced_script_treats_a_nul_path_as_nothing_to_scan(path):
    """``os.open`` raises ``ValueError`` (not ``OSError``) on an embedded NUL — the exact line of
    the prod trace. It must read as "not a script", never propagate."""
    assert _read_referenced_script(path) == (None, False)


def test_a_real_shell_script_next_to_the_binary_is_still_scanned(monkeypatch, tmp_path, nolgia_binary):
    """Skipping binaries must not blunt the guard: a wrapper script that restarts the gateway is
    still blocked from the same directory."""
    _inside_supervised_gateway(monkeypatch)
    wrapper = nolgia_binary.parent / "restart.sh"
    wrapper.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
    fake_env = _FakeEnv(str(tmp_path), nolgia_binary)

    blocked = gateway_lifecycle_block(
        command=f"bash {wrapper}",
        env=fake_env, env_type="local", cwd=str(tmp_path), workdir=None, session_key="tf-k",
    )

    assert blocked is not None
    assert "Blocked" in blocked
