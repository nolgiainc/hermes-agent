#!/usr/bin/env python3
"""Collect public post metadata from a competitor social profile.

Primary path: shell out to yt-dlp (metadata only, no video downloads) and
normalize the entries into a platform-agnostic post schema. Secondary path:
ingest founder-provided metrics (JSON or CSV) for platforms that block
anonymous scraping.

Usage:
    # Live collection (requires yt-dlp on PATH; public metadata only)
    python3 collect_profile.py "https://www.youtube.com/@SomeChannel" --limit 30 -o posts.json

    # Normalize an existing yt-dlp dump (offline; one JSON object per line)
    python3 collect_profile.py --from-json dump.jsonl --platform youtube -o posts.json

    # Ingest founder-pasted metrics (see references/manual-input.md for template)
    python3 collect_profile.py --manual metrics.csv --platform instagram -o posts.json

Output: JSON document {"platform": ..., "profile": ..., "collected_at": ...,
"posts": [...]} using the normalized post schema:
    {id, url, title, upload_date, duration_s, views, likes, comments, shares}
Missing metrics are null, never 0 — 0 means "measured as zero".

This script performs no LLM/API calls. Network access happens only via yt-dlp
and only when a profile URL is passed on the command line.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys

# Platforms where anonymous metadata collection is generally reliable,
# unreliable, or blocked. Keep honest: this drives the guidance the agent
# gives the founder when collection fails.
PLATFORM_SUPPORT = {
    "youtube": "reliable",     # channels, /videos, /shorts: public metadata works unauthenticated
    "tiktok": "unreliable",    # public profiles often work via yt-dlp, but breaks intermittently
    "instagram": "blocked",    # login-walled for anonymous clients; use --manual
    "x": "blocked",            # API-only; use the xurl skill or --manual
    "unknown": "unreliable",
}

_INT_FIELDS = ("views", "likes", "comments", "shares")


def detect_platform(url: str) -> str:
    """Best-effort platform detection from a profile or post URL."""
    host_match = re.search(r"https?://(?:www\.|m\.)?([^/]+)", url.strip(), re.IGNORECASE)
    host = (host_match.group(1) if host_match else url).lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok.com" in host:
        return "tiktok"
    if "instagram.com" in host:
        return "instagram"
    if "twitter.com" in host or host in ("x.com", "mobile.x.com"):
        return "x"
    return "unknown"


def build_ytdlp_command(url: str, limit: int) -> list[str]:
    """yt-dlp invocation: flat metadata for the newest `limit` uploads, no downloads."""
    return [
        "yt-dlp",
        "--skip-download",
        "--dump-json",
        "--playlist-end", str(limit),
        "--ignore-errors",
        "--no-warnings",
        url,
    ]


def _to_int(value) -> int | None:
    """Coerce a count to int; None (not 0) when the platform withheld it."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def _iso_date(value) -> str | None:
    """Normalize yt-dlp upload_date (YYYYMMDD) or ISO-ish strings to YYYY-MM-DD."""
    if not value:
        return None
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}", raw):
        try:
            return _dt.datetime.strptime(raw, "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
    return match.group(1) if match else None


def normalize_entry(platform: str, raw: dict) -> dict | None:
    """Map one yt-dlp JSON entry onto the normalized post schema.

    Returns None for non-post entries (channel/playlist wrapper objects).
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("_type") in ("playlist", "multi_video") or "entries" in raw:
        return None
    post_id = raw.get("id") or raw.get("display_id")
    if not post_id:
        return None
    return {
        "platform": platform,
        "id": str(post_id),
        "url": raw.get("webpage_url") or raw.get("url") or raw.get("original_url"),
        "title": raw.get("title") or raw.get("description") or "",
        "upload_date": _iso_date(raw.get("upload_date") or raw.get("timestamp_iso")),
        "duration_s": _to_int(raw.get("duration")),
        "views": _to_int(raw.get("view_count")),
        "likes": _to_int(raw.get("like_count")),
        "comments": _to_int(raw.get("comment_count")),
        "shares": _to_int(raw.get("repost_count")),
    }


def parse_ytdlp_output(platform: str, text: str) -> list[dict]:
    """Parse yt-dlp --dump-json output (one JSON object per line)."""
    posts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        post = normalize_entry(platform, raw)
        if post:
            posts.append(post)
    return posts


def parse_manual_rows(platform: str, rows: list[dict]) -> list[dict]:
    """Normalize founder-provided rows (from the manual-input template)."""
    posts = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        lowered = {str(k).strip().lower(): v for k, v in row.items()}
        post = {
            "platform": lowered.get("platform") or platform,
            "id": str(lowered.get("id") or lowered.get("url") or f"manual-{i + 1}"),
            "url": lowered.get("url"),
            "title": lowered.get("title") or lowered.get("caption") or "",
            "upload_date": _iso_date(lowered.get("upload_date") or lowered.get("date")),
            "duration_s": _to_int(lowered.get("duration_s") or lowered.get("duration")),
            "views": _to_int(lowered.get("views")),
            "likes": _to_int(lowered.get("likes")),
            "comments": _to_int(lowered.get("comments")),
            "shares": _to_int(lowered.get("shares")),
        }
        if post["url"] or any(post[f] is not None for f in _INT_FIELDS):
            posts.append(post)
    return posts


def load_manual_file(platform: str, path: str) -> list[dict]:
    """Load a founder-provided JSON (list of objects) or CSV metrics file."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("posts", [])
    else:
        rows = list(csv.DictReader(text.splitlines()))
    return parse_manual_rows(platform, rows)


def run_ytdlp(url: str, limit: int) -> str:
    """Run yt-dlp for live collection. Raises RuntimeError with guidance on failure."""
    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed. Install with: uv pip install yt-dlp "
            "(or use --manual with founder-provided metrics)."
        )
    result = subprocess.run(
        build_ytdlp_command(url, limit),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 and not result.stdout.strip():
        platform = detect_platform(url)
        support = PLATFORM_SUPPORT.get(platform, "unreliable")
        hint = (
            "This platform blocks anonymous collection — do not attempt credentialed "
            "scraping. Ask the founder for metrics via the manual-input template."
            if support == "blocked"
            else "Collection from this platform is intermittent. Retry once; if it "
            "still fails, fall back to the manual-input template."
        )
        raise RuntimeError(
            f"yt-dlp failed for {url} ({platform}, support={support}). {hint}\n"
            f"stderr (last lines):\n" + "\n".join(result.stderr.splitlines()[-5:])
        )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="Public profile URL (YouTube/TikTok work best)")
    parser.add_argument("--limit", type=int, default=30, help="Max recent posts to collect (default 30)")
    parser.add_argument("--from-json", help="Normalize an existing yt-dlp --dump-json output file (offline)")
    parser.add_argument("--manual", help="Ingest founder-provided metrics file (JSON or CSV)")
    parser.add_argument("--platform", help="Platform override (required with --from-json/--manual if not detectable)")
    parser.add_argument("-o", "--output", help="Write JSON here instead of stdout")
    args = parser.parse_args(argv)

    if sum(bool(x) for x in (args.url, args.from_json, args.manual)) != 1:
        parser.error("provide exactly one of: profile URL, --from-json FILE, --manual FILE")

    if args.url:
        platform = args.platform or detect_platform(args.url)
        profile = args.url
        try:
            posts = parse_ytdlp_output(platform, run_ytdlp(args.url, args.limit))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not posts:
            print(
                f"No posts collected from {args.url}. The profile may be empty, private, "
                "or the platform may be blocking anonymous access — see the platform "
                "support notes in SKILL.md and consider the manual-input fallback.",
                file=sys.stderr,
            )
            return 3
    elif args.from_json:
        platform = args.platform or "unknown"
        with open(args.from_json, encoding="utf-8") as fh:
            posts = parse_ytdlp_output(platform, fh.read())
        profile = f"from-json:{args.from_json}"
    else:
        platform = args.platform or "unknown"
        posts = load_manual_file(platform, args.manual)
        profile = f"manual:{args.manual}"

    document = {
        "platform": platform,
        "profile": profile,
        "collected_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "post_count": len(posts),
        "posts": posts,
    }
    payload = json.dumps(document, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"Wrote {len(posts)} posts to {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
