"""nolgia-agent#228 at the model-facing boundary: the REGISTERED tools.

The unit-level pins live in ``test_terminal_binary_first_token_guard.py`` and
``test_nolgia_sandbox_env_grant.py``. This module drives the same two shapes through
``tools.registry.registry.dispatch`` — the registered ``terminal`` handler with a compiled
``nolgia`` binary as the first token, and the registered ``execute_code`` handler with a real
local kernel reading the config-granted environment — so the contract the model sees is
exercised end to end, not only its parts.
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


# --- execute_code -------------------------------------------------------------------------


# What the nolgia-agent chart seeds on an unattended platform pod: the grant plus approvals off
# (the pod mirrors HERMES_YOLO_MODE=1 in config so headless sessions never wait on a prompt).
SEEDED_CONFIG = """\
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


def test_registered_execute_code_observes_only_config_granted_nolgia_env(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(SEEDED_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    # The platform pod runs unattended (chart: HERMES_YOLO_MODE=1 / approvals.mode "off");
    # without it the execute_code approval gate blocks arbitrary code in a headless process.
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    monkeypatch.setenv("NOLGIA_TOKEN", "nol_pod_scoped_pat")
    monkeypatch.setenv("NOLGIA_API_URL", "https://api.stg.nolgia.ai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-litellm-virtual-key")
    monkeypatch.setenv("API_SERVER_KEY", "relay-shared-secret")

    tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_turn")
    try:
        result = json.loads(
            registry.dispatch(
                "execute_code",
                {
                    "code": (
                        "import json, os\n"
                        "print(json.dumps({name: os.environ.get(name) for name in (\n"
                        "    'NOLGIA_TOKEN', 'NOLGIA_API_URL', 'OPENAI_API_KEY', "
                        "'API_SERVER_KEY')}))"
                    ),
                    "reset": True,
                },
                task_id="nolgia-env-grant",
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["status"] == "success", result
    observed = json.loads(result["output"])
    assert observed == {
        "NOLGIA_TOKEN": "nolt_run_turn",
        "NOLGIA_API_URL": "https://api.stg.nolgia.ai",
        "OPENAI_API_KEY": None,
        "API_SERVER_KEY": None,
    }
