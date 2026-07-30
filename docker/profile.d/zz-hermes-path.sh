# shellcheck shell=sh
# Restore the image's intended PATH in login shells (Nolgia fork addition).
#
# Debian's /etc/profile unconditionally resets PATH, which silently drops
# every entry the Dockerfile's ENV PATH adds (/opt/hermes/bin,
# /opt/hermes/.venv/bin, /opt/data/.local/bin). The agent's exec tool
# captures a login-shell environment snapshot once per session
# (tools/environments/base.py init_session, `bash -l`), so every agent
# command inherited the clobbered PATH: `python3` stopped resolving to the
# hermes venv and `import PIL` failed mid-run on pods whose image already
# carried Pillow (NOL-199, session 8050af4b — the venv had PIL 12.2.0 the
# whole time; only login shells couldn't see it).
#
# This snippet is sourced from /etc/profile AFTER its PATH assignment, so
# the prepend wins for root and non-root alike. Guarded so re-sourcing
# (nested login shells, `su -l`) can't stack duplicates.
case ":$PATH:" in
    *:/opt/hermes/.venv/bin:*) ;;
    *) PATH="/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin:$PATH" ;;
esac
export PATH
