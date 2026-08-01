"""Guard: committed skill docs must match what the generator produces.

``website/scripts/generate-skill-docs.py`` renders every
``skills/**/SKILL.md`` and ``optional-skills/**/SKILL.md`` into a Docusaurus
page under ``website/docs/user-guide/skills/``, and also rewrites the two
reference catalogs and the sidebar. All of those outputs are committed.

Nothing verified that the committed copies still matched their sources.
``docs-site-checks.yml`` *regenerates* the pages before it lints diagrams and
builds the site, so CI kept building a site from freshly-generated content
while the committed pages drifted further behind with every SKILL.md edit
that forgot to re-run the generator. By the time NOL-272 was filed the drift
covered ~20 pages, and two skills (``tldraw-offline``, ``pinecone-research``)
had no committed page at all — the docs site silently omitted them.

This closes the loop from the Python suite rather than from a workflow, so
no CI config change is required: ``scripts/ci/classify_changes.py`` already
routes any ``skills/`` or ``optional-skills/`` edit into the ``python`` lane,
and says why in its own docstring — "the skill-doc tests read that tree, so a
doc-looking edit can still break Python". Editing a SKILL.md therefore runs
this test.

When it fails, the fix is always the same:

    python3 website/scripts/generate-skill-docs.py

...and commit the result.

The generator is deterministic and idempotent (running it twice produces
byte-identical output), which is what makes an exact-match comparison a
stable signal rather than a flaky one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "website" / "scripts" / "generate-skill-docs.py"
SIDEBARS = REPO_ROOT / "website" / "sidebars.ts"

REGEN_HINT = (
    "Run `python3 website/scripts/generate-skill-docs.py` and commit the result."
)

# Cap how many paths a failure message lists — a first-time regeneration can
# touch dozens of files and an unbounded dump buries the hint.
_MAX_LISTED = 15


def _format_paths(paths: list[str]) -> str:
    shown = "\n".join(f"  - {p}" for p in paths[:_MAX_LISTED])
    if len(paths) > _MAX_LISTED:
        shown += f"\n  ... and {len(paths) - _MAX_LISTED} more"
    return shown


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the real generator against a throwaway repo root.

    The generator writes in place, resolving its repo root from its own
    ``__file__``. Copying the script into a sandbox therefore redirects every
    write there, so this test never dirties the working tree (and passes just
    as well on an already-dirty one).

    ``skills/`` and ``optional-skills/`` are symlinked rather than copied:
    together they are ~16MB, the generator only reads them, and it derives
    every path with ``relative_to`` on unresolved paths, so symlinks behave
    identically to real directories here.
    """
    root = tmp_path_factory.mktemp("skill-docs-freshness")

    for tree in ("skills", "optional-skills"):
        (root / tree).symlink_to(REPO_ROOT / tree, target_is_directory=True)

    scripts_dir = root / "website" / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / GENERATOR.name
    shutil.copy2(GENERATOR, script)

    (root / "website" / "docs" / "reference").mkdir(parents=True)
    (root / "website" / "docs" / "user-guide" / "skills").mkdir(parents=True)
    # write_sidebar() reads the existing sidebar and patches the skills region,
    # so it needs the committed file as its starting point.
    shutil.copy2(SIDEBARS, root / "website" / "sidebars.ts")

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, (
        f"generate-skill-docs.py failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return root


def _generated_outputs(sandbox: Path) -> list[Path]:
    """Every file the generator wrote, excluding our copy of the script."""
    scripts_dir = sandbox / "website" / "scripts"
    return sorted(
        p
        for p in (sandbox / "website").rglob("*")
        if p.is_file() and scripts_dir not in p.parents
    )


def test_generator_produced_output(sandbox: Path) -> None:
    """Guard the guard: if the sandbox produced nothing, the comparisons below
    would pass vacuously and the drift check would be silently dead."""
    outputs = _generated_outputs(sandbox)
    assert len(outputs) > 100, (
        f"expected the generator to emit a page per skill, got {len(outputs)} files. "
        "The sandbox scaffold is probably wrong, not the docs."
    )


def test_every_skill_has_a_committed_docs_page(sandbox: Path) -> None:
    """A newly added skill must ship its generated page in the same commit.

    This is the failure mode that hid `tldraw-offline` and `pinecone-research`
    from the docs site entirely (NOL-272).
    """
    missing = [
        str(p.relative_to(sandbox))
        for p in _generated_outputs(sandbox)
        if not (REPO_ROOT / p.relative_to(sandbox)).exists()
    ]
    assert not missing, (
        f"{len(missing)} generated docs file(s) are missing from the repo:\n"
        f"{_format_paths(missing)}\n{REGEN_HINT}"
    )


def test_committed_docs_match_their_skill_sources(sandbox: Path) -> None:
    """Editing a SKILL.md without re-running the generator leaves the published
    page showing stale text, versions, and catalog rows."""
    stale = []
    for produced in _generated_outputs(sandbox):
        rel = produced.relative_to(sandbox)
        committed = REPO_ROOT / rel
        if not committed.exists():
            continue  # reported by the missing-page test above
        if produced.read_bytes() != committed.read_bytes():
            stale.append(str(rel))

    assert not stale, (
        f"{len(stale)} committed docs file(s) no longer match their SKILL.md sources:\n"
        f"{_format_paths(stale)}\n{REGEN_HINT}"
    )
