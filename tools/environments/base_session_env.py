"""Bash session-snapshot scripting for ``tools.environments.base``.

Pure string builders: the CWD marker, the ``export -p`` dump that strips
per-session vars, the ``init_session`` bootstrap, and the per-command wrapper.
No module state lives here; ``BaseEnvironment`` supplies quoting hooks.
"""

import re
import shlex
from typing import Iterable

# Bridged per-session vars (gateway.session_context._VAR_MAP) are injected fresh onto every
# command's process env and must NEVER persist in the shared bash snapshot: one long-lived
# backend serves many sessions, so a snapshot carrying the FIRST session's HERMES_SESSION_ID
# would make every LATER session source a foreign identity. Every bridged name starts with
# one of these prefixes (or is HERMES_UI_SESSION_ID); unit tests use this regex as the
# Python-side contract for the exclusion set.
# Per-session variables that the gateway bridges freshly onto every command's process environment (via
# tools/environments/local._inject_session_context_env, reading gateway.session_context._VAR_MAP). They must
# NEVER be persisted into the shared bash session snapshot: a single long-lived backend serves many
# concurrent sessions (the messaging gateway, TUI, desktop/web dashboard all collapse the terminal to one
# "default" environment), so ``export -p`` dumping the FIRST session's HERMES_SESSION_ID into the snapshot
# makes every LATER session ``source`` that stale value and see a FOREIGN session's identity — overriding
# the correct per-command Popen env (issue: cross-session HERMES_SESSION_ID leak via the shared snapshot).
# Stripping them from the snapshot is safe because they are re-injected on every command; a snapshot should
# only carry the user's own shell state (PATH, functions, exports they set), not Hermes' per-turn session
# identity. Used by unit tests as the Python-side contract for the exclusion set; the dump path unsets by
# name/prefix instead of grepping declare lines (see below / issue #71296).
# NOLGIA_TOKEN (exact name) is excluded for the same reason as the session
# vars: with run-scoped platform tokens the bridge overrides it per-command,
# and a snapshot capturing one run's override would hand that run's
# credential — and its causal attribution — to every sibling run's later
# commands. The pod-wide value keeps flowing to children through plain env
# inheritance regardless; only snapshot PERSISTENCE is suppressed.
_SNAPSHOT_EXCLUDED_ENV_PATTERNS = (
    "HERMES_SESSION_",
    "HERMES_UI_SESSION_ID",
    "HERMES_CRON_AUTO_DELIVER_",
    "HERMES_CRON_SESSION",
    "HERMES_BROWSER_CONTROL_",
    "NOLGIA_TOKEN=",
)
_SNAPSHOT_EXCLUDED_ENV_REGEX = (
    "^declare -x (" + "|".join(_SNAPSHOT_EXCLUDED_ENV_PATTERNS) + ")"
)

# TMPDIR is excluded ONLY for turns whose TMPDIR Hermes itself owns — i.e. a
# turn bound to a session-scoped scratch workspace (NOL-414), where the
# per-command bridge points TMPDIR at that session's dir. Snapshotting that
# value would redirect every sibling session's later scratch writes into a
# FOREIGN session's workspace, the exact cross-session leak the override
# exists to close. It must NOT be excluded unconditionally: TMPDIR is a
# normal user-controlled shell variable, and stripping it from every
# snapshot would make a plain `export TMPDIR=/custom` in the CLI (or any
# deployment with session workspaces disabled) silently vanish on the next
# command.
_SCOPED_TMPDIR_SNAPSHOT_ENV_REGEX = (
    "^declare -x ("
    + "|".join(_SNAPSHOT_EXCLUDED_ENV_PATTERNS + ("TMPDIR=",))
    + ")"
)
_SHELL_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _session_scoped_tmpdir_active() -> bool:
    """True when the gateway bound a per-session scratch workspace for THIS turn.

    Read from the session ContextVar (not os.environ) so a sibling session's
    stale process-global can never make this turn suppress the user's own
    TMPDIR. Unbound (CLI, single-session hosts, deployments with session
    workspaces off) → TMPDIR is the user's variable and persists normally.
    """
    try:
        from gateway.session_context import get_session_env

        return bool(get_session_env("HERMES_SESSION_SCRATCH_DIR"))
    except Exception:
        return False


def snapshot_excluded_env_regex(scoped_tmpdir: bool | None = None) -> str:
    """Python-side contract for the snapshot exclusion set.

    ``scoped_tmpdir`` defaults to whether this turn owns a session-scoped
    TMPDIR (see :func:`_session_scoped_tmpdir_active`).
    """
    if scoped_tmpdir is None:
        scoped_tmpdir = _session_scoped_tmpdir_active()
    return (
        _SCOPED_TMPDIR_SNAPSHOT_ENV_REGEX
        if scoped_tmpdir
        else _SNAPSHOT_EXCLUDED_ENV_REGEX
    )

# mktemp template suffix + the shell variable holding the allocated temp path.
_SNAP_TMP_SUFFIX = ".tmp.XXXXXXXXXX"
_SNAP_TMP = '"$__hermes_snap_tmp"'


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


def _cwd_marker_printf(marker: str) -> str:
    """Emit the CWD marker on its own line (leading ``\\n`` guards against a
    command whose output lacks a trailing newline; ``_split_cwd_marker`` strips it)."""
    return f"printf '\\n{marker}%s{marker}\\n' \"$(pwd -P)\""


def _export_dump_excluding_session_vars(
    tmp_path: str, excluded_names: Iterable[str] = (), scoped_tmpdir: bool | None = None,
) -> str:
    """Shell snippet dumping ``export -p`` to *tmp_path* minus the per-session bridged vars (see
    ``_SNAPSHOT_EXCLUDED_ENV_REGEX``) and *excluded_names*. The vars are ``unset`` in a subshell
    BEFORE ``export -p``: a line-based ``grep -vE`` is unsafe because bash 3.2 prints a value
    containing a newline as a multi-line ``declare -x`` block, so smuggled continuation lines would
    survive into the snapshot and execute on the next ``source``. ``|| true`` keeps the success
    contract. The dump is a brace group with the redirection on the group: *tmp_path* is usually a
    shell-variable expansion, and a redirect on a pipeline segment would expand it inside that
    segment's subshell, inconsistently with the parent that expands the follow-up ``mv``.

    ``curl … | bash #`` smuggled into a Matrix room/display name via ``HERMES_SESSION_CHAT_NAME``) land in
    the snapshot and execute on the next ``source`` (issue #71296). Unsetting first means ``export -p``
    never emits those vars — including any continuation lines.

    ``scoped_tmpdir`` adds TMPDIR to the unset list; it defaults to whether
    this turn's TMPDIR is Hermes-owned (a bound session workspace), so a
    user's own ``export TMPDIR=/custom`` keeps persisting everywhere else.
    """
    # ${!PREFIX*} is bash 3.2+ name-prefix expansion; empty matches are ignored
    # under 2>/dev/null. Caller names are quoted so malformed config can never
    # become shell syntax (valid names stay unquoted by shlex.quote()).
    safe_names = {name for name in excluded_names if isinstance(name, str) and name}
    if scoped_tmpdir is None:
        scoped_tmpdir = _session_scoped_tmpdir_active()
    if scoped_tmpdir:
        safe_names.add("TMPDIR")
    extra_unset = "".join(f" {shlex.quote(name)}" for name in sorted(safe_names))
    return (
        "{ ( unset ${!HERMES_SESSION_*} ${!HERMES_CRON_AUTO_DELIVER_*} "
        "${!HERMES_BROWSER_CONTROL_*} "
        # AI_AGENT / HERMES_AGENT are per-command attribution markers re-exported
        # by every wrapper with ${VAR:-default} semantics; persisting them would
        # let the FIRST command's value override a later outer-harness value.
        "AI_AGENT HERMES_AGENT "
        # NOLGIA_TOKEN: the per-command run-scoped override must never persist
        # (see _SNAPSHOT_EXCLUDED_ENV_PATTERNS).
        f"HERMES_UI_SESSION_ID NOLGIA_TOKEN{extra_unset} 2>/dev/null; "
        "export -p; ) || true; } "
        f"> {tmp_path}")


def _snapshot_bootstrap_script(
    *, quoted_cwd: str, quoted_snap: str, snap_tmp_template: str, excluded_names: Iterable[str], cwd_marker: str,
) -> str:
    """Login-shell bootstrap that captures env/functions/aliases into the snapshot. Atomic publish:
    assemble in a ``mktemp`` file, then ``mv`` over the final path so a concurrent ``source`` never
    reads a half-written snapshot (``$$`` is the parent PID in ``&``-launched subshells and macOS
    bash 3.2 lacks ``$BASHPID``, so only ``mktemp`` is portable). Functions are filtered by NAME via
    ``declare -F`` (a line-based ``declare -f | grep -v`` strips the header and leaves an orphaned
    body that breaks every sourced command); the non-empty guard matters because bare ``declare -f``
    dumps ALL functions. The trailing ``cd`` restores the configured cwd after profile scripts (e.g.
    ``cd ~``) so ``pwd -P`` reports terminal.cwd, not the profile's directory."""
    return (
        "umask 077\n"
        f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) || exit 1\n"
        f"{_export_dump_excluding_session_vars(_SNAP_TMP, excluded_names)}\n"
        "__hermes_fns=$(declare -F | awk '{print $3}' | grep -vE '^_[^_]') || true\n"
        f"[ -n \"$__hermes_fns\" ] && declare -f $__hermes_fns >> {_SNAP_TMP} 2>/dev/null || true\n"
        f"alias -p >> {_SNAP_TMP}\n"
        f"echo 'shopt -s expand_aliases' >> {_SNAP_TMP}\n"
        f"echo 'set +e' >> {_SNAP_TMP}\n"
        f"echo 'set +u' >> {_SNAP_TMP}\n"
        # Publish only if assembly succeeded; otherwise drop the partial temp.
        f"mv -f {_SNAP_TMP} {quoted_snap} || rm -f {_SNAP_TMP}\n"
        f"builtin cd -- {quoted_cwd} 2>/dev/null || true\n"
        f"{_cwd_marker_printf(cwd_marker)}\n")


def _passthrough_save_restore(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Shell lines that save profile-scoped passthrough vars before the snapshot is sourced
    and restore (or unset) them afterwards — a shared snapshot may hold the previous
    profile's value. Values stay in environment memory and never enter the command string."""
    save: list[str] = []
    restore: list[str] = []
    for name in names:
        marker = f"_HERMES_RUNTIME_PASSTHROUGH_{name}"
        present, value = f"{marker}_PRESENT", f"{marker}_VALUE"
        save += [f"{present}=${{{name}+x}}", f"{value}=${{{name}-}}"]
        restore += [
            f'if [ "${present}" = x ]; then export {name}="${value}"; else unset {name}; fi',
            f"unset {present} {value}"]
    return save, restore


def _wrap_command_script(
    command: str, *, quoted_cwd: str, quoted_snap: str, snap_tmp_template: str,
    passthrough_names: Iterable[str], snapshot_ready: bool, cwd_marker: str,
    scoped_tmpdir: bool | None = None) -> str:
    """Per-command bash script: source snapshot, cd, run, re-dump env, emit CWD marker.
    ``source`` stdout goes to /dev/null because macOS bash 3.2 / some Homebrew builds echo
    ``declare -x`` lines when sourcing. AI_AGENT/HERMES_AGENT advertise the harness to remote
    backends (whose env is not inherited); ``${VAR:-default}`` never clobbers an outer harness.
    GIT_PAGER/PAGER=cat stop pager-happy tools hanging a PTY-backed command. The env re-dump
    uses the same mktemp+mv atomic publish as the bootstrap and chains ``mv`` on the dump
    succeeding so a failed dump never replaces a good snapshot. ``umask 077`` is applied after
    the user's command so snapshot files (which may carry secrets) are private without
    changing the command's umask.

    ``scoped_tmpdir`` (default: whether THIS turn owns its TMPDIR, read once from
    the session ContextVar so the restore list and the re-dump agree) adds TMPDIR
    to the save/restore set and to the re-dump exclusions (NOL-414). Excluding it
    from the RE-DUMP only stops this turn from publishing its value; a snapshot
    written EARLIER by an unscoped turn (a CLI/messaging ``export TMPDIR=/custom``,
    or any turn before workspaces engaged) still carries a ``declare -x TMPDIR=…``
    line, and sourcing that would overwrite the per-command bridge's session
    scratch dir — silently sending this run's temp artifacts back to the
    shared/foreign directory the workspace exists to replace. Save-and-restore
    around the source keeps the bound workspace authoritative.
    """
    escaped = command.replace("'", "'\\''")
    passthrough_names = tuple(passthrough_names)
    if scoped_tmpdir is None:
        scoped_tmpdir = _session_scoped_tmpdir_active()
    restore_names = list(passthrough_names)
    if scoped_tmpdir and "TMPDIR" not in restore_names:
        restore_names.append("TMPDIR")
    save, restore = _passthrough_save_restore(restore_names)
    parts = list(save)
    if snapshot_ready:
        parts.append(f"source {quoted_snap} >/dev/null 2>&1 || true")
    parts += restore
    parts += [
        'export AI_AGENT="${AI_AGENT:-hermes-agent}" HERMES_AGENT="${HERMES_AGENT:-true}"',
        'export GIT_PAGER="${GIT_PAGER:-cat}" PAGER="${PAGER:-cat}"',
        # ``--`` keeps hyphen-prefixed directory names from being parsed as options.
        f"builtin cd -- {quoted_cwd} || exit 126",
        f"eval '{escaped}'",
        "__hermes_ec=$?",
        "umask 077"]
    if snapshot_ready:
        parts.append(
            f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) && "
            f"{{ {_export_dump_excluding_session_vars(_SNAP_TMP, passthrough_names, scoped_tmpdir=scoped_tmpdir)} "
            f"&& mv -f {_SNAP_TMP} {quoted_snap}; }} "
            f"2>/dev/null || rm -f {_SNAP_TMP} 2>/dev/null || true")
    parts += [_cwd_marker_printf(cwd_marker), "exit $__hermes_ec"]
    return "\n".join(parts)


def _split_cwd_marker(output: str, marker: str) -> tuple[str | None, str] | None:
    """Locate the last ``marker<path>marker`` pair in *output*. Returns
    ``(cwd_path_or_None, output_without_marker_line)``, or ``None`` when no complete pair
    exists. The stripped span runs from the ``\\n`` the wrapper injected before the marker
    through the end of the marker line."""
    last = output.rfind(marker)
    if last == -1:
        return None
    search_start = max(0, last - 4096)  # CWD path won't be >4KB
    first = output.rfind(marker, search_start, last)
    if first == -1 or first == last:
        return None
    cwd_path = output[first + len(marker) : last].strip() or None
    line_start = output.rfind("\n", 0, first)
    if line_start == -1:
        line_start = first
    line_end = output.find("\n", last + len(marker))
    line_end = line_end + 1 if line_end != -1 else len(output)
    return cwd_path, output[:line_start] + output[line_end:]
