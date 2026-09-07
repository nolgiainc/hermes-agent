"""nolgiainc/nolgia-agent#228 at the model-facing boundary: the REGISTERED tools.

Prod cust-2774ee97 (2026-09-06) asked a customer to paste a PAT. Two runtime shapes, each pinned
through ``tools.registry.registry.dispatch`` so the contract the model sees is what is asserted:

1. ``terminal`` with the compiled ``nolgia`` binary as the first token failed with
   ``Failed to execute command: open: embedded null character in path`` — the gateway lifecycle
   guard walked into the ELF as a "referenced script", the remote-read fallback handed back its
   decoded bytes, and a NUL-bearing path token crashed ``os.open`` past an ``OSError``-only
   handler. The registered tool must run such a command and still block a real lifecycle script.
2. ``execute_code`` scrubs the child env to an allowlist, so sandbox-spawned ``nolgia`` calls ran
   anonymous (401). The grant is ordinary operator configuration — ``terminal.env_passthrough:
   [NOLGIA_TOKEN, NOLGIA_API_URL]``, seeded by the nolgia-agent chart — and the registered tool
   must show a real child exactly those two names (run-scoped token substituted, NOL-413) with
   the grant, and neither without it.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.config as config_module
import tools.env_passthrough as env_passthrough
import tools.terminal_tool as terminal_tool
# Importing registers the handler; a bare process has no execute_code in the registry otherwise.
import tools.code_execution_tool  # noqa: F401
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.code_kernel import shutdown_all_kernels
from tools.registry import registry

# ELF magic + header padding, then machine-code-shaped bytes that tokenize into a NUL-bearing
# absolute path and a literal lifecycle phrase. Neither may be scanned: a binary the user executes
# is not a referenced *shell script*.
ELF_BYTES = (
    b"\x7fELF\x02\x01\x01\x00"
    + bytes(64)
    + b"/opt/data/bin/nolgia\x00\x01 --run\nhermes gateway restart\x00"
    + b"\x90" * 4096
)

# What the nolgia-agent chart seeds on an unattended platform pod: the grant plus approvals off
# (the pod mirrors HERMES_YOLO_MODE=1 in config so headless sessions never wait on a prompt).
GRANTED_CONFIG = """\
terminal:
  env_passthrough:
    - NOLGIA_TOKEN
    - NOLGIA_API_URL
approvals:
  mode: "off"
code_execution:
  mode: strict
  timeout: 30
"""

# The same pod before the chart carried the grant (the #228 failure mode).
UNGRANTED_CONFIG = """\
approvals:
  mode: "off"
code_execution:
  mode: strict
  timeout: 30
"""

SANDBOX_PROBE = (
    "import json, os\n"
    "print(json.dumps({name: os.environ.get(name) for name in (\n"
    "    'NOLGIA_TOKEN', 'NOLGIA_API_URL', 'OPENAI_API_KEY', 'API_SERVER_KEY')}))"
)


@pytest.fixture(autouse=True)
def fresh_sandbox_state():
    shutdown_all_kernels()
    env_passthrough._config_passthrough = None
    env_passthrough.clear_env_passthrough()
    config_module._RAW_CONFIG_CACHE.clear()
    reset_session_vars()
    yield
    shutdown_all_kernels()
    env_passthrough._config_passthrough = None
    env_passthrough.clear_env_passthrough()
    config_module._RAW_CONFIG_CACHE.clear()
    reset_session_vars()


# --- terminal ---------------------------------------------------------------------------------


@pytest.fixture
def terminal_env(tmp_path, monkeypatch):
    """Registered terminal tool inside a supervised gateway, on a fake local backend whose
    ``head -c`` read (the guard's remote fallback) returns the binary's decoded bytes — exactly
    what the fallback returned before the upstream hardening."""
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


@pytest.mark.parametrize(
    "shape",
    ["elf-first-token", "nul-in-path-token"],
)
def test_registered_terminal_runs_binary_shaped_commands(terminal_env, shape):
    """The incident command (a 12 MB ELF as the first token) and its residue (a NUL-bearing
    path token, how the recursion re-tokenized machine code) both reach the backend."""
    binary, fake_env = terminal_env
    command = (
        f"{binary} skills show nolgia-video-prompting"
        if shape == "elf-first-token"
        else "bash ./run\x00me.sh"
    )

    result = json.loads(
        registry.dispatch("terminal", {"command": command}, task_id=f"binary-{shape}")
    )

    assert result == {
        "output": "nolgia-video-prompting  v1",
        "exit_code": 0,
        "error": None,
    }
    assert fake_env.calls == [command]


def test_registered_terminal_still_blocks_lifecycle_script(terminal_env):
    """Skipping binaries must not blunt the guard: a wrapper script that restarts the gateway,
    sitting next to the binary, is still blocked before the backend sees it."""
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


# --- execute_code -----------------------------------------------------------------------------


def _pod(tmp_path, monkeypatch, config_text: str) -> None:
    """A customer pod's env, reduced to the names whose fate the scrub decides: the platform
    pair, the LiteLLM virtual key and the relay secret (both must keep being withheld)."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("NOLGIA_TOKEN", "nol_pod_scoped_pat")
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.stg.nolgia.ai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-litellm-virtual-key")
    monkeypatch.setenv("API_SERVER_KEY", "relay-shared-secret")


def _observed_in_sandbox(task_id: str) -> dict:
    tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_turn")
    try:
        result = json.loads(
            registry.dispatch(
                "execute_code", {"code": SANDBOX_PROBE, "reset": True}, task_id=task_id
            )
        )
    finally:
        clear_session_vars(tokens)
    assert result["status"] == "success", result
    return json.loads(result["output"])


def test_registered_execute_code_observes_only_config_granted_nolgia_env(tmp_path, monkeypatch):
    _pod(tmp_path, monkeypatch, GRANTED_CONFIG)

    assert _observed_in_sandbox("nolgia-env-grant") == {
        "NOLGIA_TOKEN": "nolt_run_turn",  # the bound run token, not the pod-wide bearer
        "NOLGIA_API_URL": "https://api.stg.nolgia.ai",
        "OPENAI_API_KEY": None,
        "API_SERVER_KEY": None,
    }


def test_registered_execute_code_without_the_grant_is_anonymous(tmp_path, monkeypatch):
    """The #228 failure mode, kept explicit so the chart dependency is: without the seeded
    grant the sandbox sees neither name — a bound run token must not smuggle one in."""
    _pod(tmp_path, monkeypatch, UNGRANTED_CONFIG)

    assert _observed_in_sandbox("nolgia-env-ungranted") == {
        "NOLGIA_TOKEN": None,
        "NOLGIA_API_URL": None,
        "OPENAI_API_KEY": None,
        "API_SERVER_KEY": None,
    }
