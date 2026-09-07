"""execute_code sandbox grant for the Nolgia platform CLI (nolgiainc/nolgia-agent#228).

On a Nolgia customer pod the ``nolgia`` CLI authenticates from ``NOLGIA_TOKEN`` and talks to
``NOLGIA_API_URL`` (both pod env, chart ``deployment.yaml``). The ``terminal`` tool inherits the
pod env wholesale, but ``execute_code`` builds its child env through ``_scrub_child_env``, which
keeps only allowlisted names — so unless something GRANTS the two names, every ``nolgia`` call a
sandbox script spawns is anonymous (401) and the model concludes the customer must supply a PAT.

The platform grants them from the seeded ``config.yaml`` (nolgia-agent chart)::

    terminal:
      env_passthrough:
        - NOLGIA_TOKEN
        - NOLGIA_API_URL

These tests pin the whole chain the chart relies on — ``config.yaml`` ->
``tools.env_passthrough`` -> ``_scrub_child_env`` -> ``apply_run_scoped_nolgia_token`` — so a
runtime change that breaks any link fails here rather than on a customer pod. No scrub code is
special-cased for the platform: the grant is ordinary operator configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import tools.env_passthrough as env_passthrough
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.code_execution_env import _build_child_env, _scrub_child_env
from tools.environments.local import apply_run_scoped_nolgia_token

# What a customer pod's main container carries (chart deployment.yaml), reduced to the names
# whose fate the scrub decides. OPENAI_API_KEY is the LiteLLM virtual key and API_SERVER_KEY the
# relay secret: both must keep being withheld from sandbox scripts.
POD_ENV = {
    "PATH": "/opt/data/bin:/usr/local/bin:/usr/bin",
    "HOME": "/opt/data",
    "HERMES_HOME": "/opt/data",
    "NOLGIA_TOKEN": "nol_pod_scoped_pat",
    "NOLGIA_API_URL": "https://api.stg.nolgia.ai",
    "OPENAI_API_KEY": "sk-litellm-virtual-key",
    "API_SERVER_KEY": "relay-shared-secret",
}

SEEDED_CONFIG = (
    "model:\n"
    "  default: or-smart\n"
    "terminal:\n"
    "  env_passthrough:\n"
    "    - NOLGIA_TOKEN\n"
    "    - NOLGIA_API_URL\n"
)


@pytest.fixture(autouse=True)
def _fresh_passthrough_state():
    """The config allowlist is cached per process and the skill allowlist per context."""
    import hermes_cli.config as config_module

    env_passthrough._config_passthrough = None
    env_passthrough.clear_env_passthrough()
    config_module._RAW_CONFIG_CACHE.clear()
    reset_session_vars()
    yield
    env_passthrough._config_passthrough = None
    env_passthrough.clear_env_passthrough()
    config_module._RAW_CONFIG_CACHE.clear()
    reset_session_vars()


@pytest.fixture
def config_path(tmp_path, monkeypatch) -> Path:
    """Point every raw-config read at a scratch ``config.yaml`` (absent until a test writes it)."""
    import hermes_cli.config as config_module

    path = tmp_path / "config.yaml"
    monkeypatch.setattr(config_module, "get_config_path", lambda: path)
    return path


def test_without_the_grant_the_sandbox_is_anonymous(config_path):
    """The #228 failure mode: no ``terminal.env_passthrough`` in config.yaml, no skill declaring
    the names — the scrub drops the token (secret substring) AND the API URL (no safe prefix), so
    a sandbox ``nolgia`` call is anonymous against the default API host."""
    env = _scrub_child_env(dict(POD_ENV))
    assert "NOLGIA_TOKEN" not in env
    assert "NOLGIA_API_URL" not in env
    assert env["PATH"] == POD_ENV["PATH"]


def test_seeded_env_passthrough_reaches_the_execute_code_scrub(config_path):
    config_path.write_text(SEEDED_CONFIG, encoding="utf-8")

    env = _scrub_child_env(dict(POD_ENV))

    assert env["NOLGIA_TOKEN"] == POD_ENV["NOLGIA_TOKEN"]
    assert env["NOLGIA_API_URL"] == POD_ENV["NOLGIA_API_URL"]
    # The grant is exactly two names: the provider key and the relay secret stay withheld.
    assert "OPENAI_API_KEY" not in env
    assert "API_SERVER_KEY" not in env


def test_grant_is_not_refused_as_a_hermes_provider_credential(config_path):
    """``env_passthrough`` refuses Hermes-managed provider credentials (GHSA-rhgp-j443-p4rf) from
    config and skills alike. The platform names are not provider credentials, so the config grant
    is honoured — and the skill-frontmatter route stays open as an alternative."""
    for name in ("NOLGIA_TOKEN", "NOLGIA_API_URL"):
        assert env_passthrough._is_hermes_provider_credential(name) is False

    config_path.write_text(SEEDED_CONFIG, encoding="utf-8")
    assert env_passthrough.is_env_passthrough("NOLGIA_TOKEN") is True
    assert env_passthrough.is_env_passthrough("NOLGIA_API_URL") is True
    # Control: a real provider credential listed the same way is still refused.
    config_path.write_text(
        "terminal:\n  env_passthrough:\n    - OPENAI_API_KEY\n", encoding="utf-8"
    )
    env_passthrough._config_passthrough = None
    assert env_passthrough.is_env_passthrough("OPENAI_API_KEY") is False

    env_passthrough.register_env_passthrough(["NOLGIA_TOKEN", "NOLGIA_API_URL"])
    assert env_passthrough.get_all_passthrough() >= {"NOLGIA_TOKEN", "NOLGIA_API_URL"}


def test_granted_token_is_substituted_by_the_bound_run_token(config_path, tmp_path, monkeypatch):
    """NOL-413 inside the sandbox: once the pod GRANTS ``NOLGIA_TOKEN``, a bound run token replaces
    it in the real child-env builder, so sandbox-spawned ``nolgia`` calls attribute to the causing
    turn. The API URL rides along unchanged."""
    config_path.write_text(SEEDED_CONFIG, encoding="utf-8")
    for name, value in POD_ENV.items():
        monkeypatch.setenv(name, value)

    tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_turn")
    try:
        child_env = _build_child_env(
            rpc_endpoint=str(tmp_path / "rpc.sock"),
            rpc_token="rpc-token",
            tmpdir=str(tmp_path),
            child_python=sys.executable,
        )
    finally:
        clear_session_vars(tokens)

    assert child_env["NOLGIA_TOKEN"] == "nolt_run_turn"
    assert child_env["NOLGIA_API_URL"] == POD_ENV["NOLGIA_API_URL"]
    assert "OPENAI_API_KEY" not in child_env
    assert "API_SERVER_KEY" not in child_env


def test_unbound_turn_keeps_the_pod_token(config_path):
    """No run token bound (CLI/cron/relay without ``nolgia_token``): the granted pod-wide value is
    what the sandbox spends — same as the terminal tool today."""
    config_path.write_text(SEEDED_CONFIG, encoding="utf-8")

    env = _scrub_child_env(dict(POD_ENV))
    apply_run_scoped_nolgia_token(env)

    assert env["NOLGIA_TOKEN"] == POD_ENV["NOLGIA_TOKEN"]


def test_substitution_never_introduces_a_withheld_token(config_path):
    """Without the grant, a bound run token must not smuggle ``NOLGIA_TOKEN`` into the sandbox:
    substitution-only, by design (the grant policy lives in config, not in the bridge)."""
    tokens = set_session_vars(platform="api_server", nolgia_token="nolt_run_turn")
    try:
        env = _scrub_child_env(dict(POD_ENV))
        apply_run_scoped_nolgia_token(env)
    finally:
        clear_session_vars(tokens)

    assert "NOLGIA_TOKEN" not in env
