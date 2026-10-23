"""Which `uv pip check` incompatibilities the project's lock resolved on purpose.

A `[tool.uv] override-dependencies` entry replaces a dependency's requirement
during resolution, but the installed distribution still declares the original,
so `uv pip check` reports the overridden package as incompatible. uv.lock records
the overrides it applied (`[manifest] overrides`) and the version it resolved for
each, so a report line is expected exactly when it names an overridden
dependency installed at the locked version. Everything else still fails.

Stdlib only: the bootstrap runner loads pm.environment on a system python that
may predate tomllib.
"""
from __future__ import annotations

from pathlib import Path
import re

# The incompatibility lines uv prints, e.g.
# The package `browser-harness` requires `pillow==12.3.0`, but `13.0.0` is installed
_REPORT_LINE = re.compile(r"^The package `[^`]+` requires ")
_VERSION_MISMATCH = re.compile(
    r"^The package `[^`]+` requires `([A-Za-z0-9][A-Za-z0-9._-]*)[^`]*`, but `([^`]+)` is installed$"
)
_FOUND = re.compile(r"^Found (\d+) incompatibilit(?:y|ies)$", re.MULTILINE)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_lock(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Override names and (name, version) packages from uv.lock text."""
    try:
        import tomllib
    except ModuleNotFoundError:
        manifest = re.search(r"(?ms)^\[manifest\]\s*$(.*?)(?=^\[)", text + "\n[")
        overrides = re.search(r"(?ms)^overrides\s*=\s*\[(.*?)^\]", manifest.group(1)) if manifest else None
        names = re.findall(r'\{\s*name\s*=\s*"([^"]+)"', overrides.group(1)) if overrides else []
        packages = re.findall(r'(?m)^\[\[package\]\]\s*\nname = "([^"]+)"\s*\nversion = "([^"]+)"', text)
        return names, packages
    data = tomllib.loads(text)
    names = [override["name"] for override in data.get("manifest", {}).get("overrides", [])]
    packages = [(package["name"], package["version"]) for package in data.get("package", []) if "version" in package]
    return names, packages


def overridden_versions(lock: Path) -> dict[str, set[str]]:
    """Locked versions of every dependency the lock's manifest overrides."""
    names, packages = _parse_lock(lock.read_text(encoding="utf-8"))
    overridden = {_normalize(name) for name in names}
    versions: dict[str, set[str]] = {}
    for name, version in packages:
        if _normalize(name) in overridden:
            versions.setdefault(_normalize(name), set()).add(version)
    return versions


def unexpected_incompatibilities(report: str, lock: Path | None) -> list[str] | None:
    """The report lines *lock* does not account for.

    None when the report cannot be read line by line (no lines, or fewer than the
    "Found N" total uv printed), so the caller fails on the raw output instead of
    trusting a partial parse.
    """
    lines = [line.strip() for line in report.splitlines() if _REPORT_LINE.match(line.strip())]
    found = _FOUND.search(report)
    if not lines or (found and int(found.group(1)) != len(lines)):
        return None
    if lock is None or not lock.is_file():
        return lines
    expected = overridden_versions(lock)
    unexpected = []
    for line in lines:
        mismatch = _VERSION_MISMATCH.match(line)
        if not (mismatch and mismatch.group(2) in expected.get(_normalize(mismatch.group(1)), ())):
            unexpected.append(line)
    return unexpected
