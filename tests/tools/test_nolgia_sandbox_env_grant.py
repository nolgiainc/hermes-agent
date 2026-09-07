"""The registered execute_code tool must apply configured Nolgia environment grants.

This exercises the real config, registry handler, fresh local kernel, child-environment scrub, and
run-scoped token substitution as one model-facing contract.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.config as config_module
import tools.env_passthrough as env_passthrough
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.code_kernel import shutdown_all_kernels
from tools.registry import registry


SEEDED_CONFIG = """\
terminal:
  env_passthrough:
    - NOLGIA_TOKEN
    - NOLGIA_API_URL
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
