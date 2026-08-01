#!/usr/bin/env python3
"""Score competitor posts and rank top performers. Pure stdlib, fully offline.

Input: one or more JSON documents produced by collect_profile.py (or matching
its normalized schema). Output: the same posts annotated with performance
metrics, ranked, with account-level baselines — as JSON or a Markdown table.

Usage:
    python3 score_posts.py posts.json -o scored.json
    python3 score_posts.py posts_a.json posts_b.json --markdown
    python3 score_posts.py posts.json --top 10 --markdown

Metrics per post (null-safe — missing platform metrics never crash scoring):
    engagement_rate   (likes + comments + shares) / views, when views known
    views_vs_median   views / account median views (outlier multiple)
    performance_index views_vs_median weighted by engagement vs account median;
                      falls back to a likes-based percentile when views are
                      unavailable (common on manually-entered X/Instagram data)

No network access. No LLM/API calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys


def _engagement_actions(post: dict) -> int | None:
    """Sum of known engagement actions; None if none of them were measured."""
    parts = [post.get(k) for k in ("likes", "comments", "shares")]
    known = [p for p in parts if isinstance(p, (int, float))]
    if not known:
        return None
    return int(sum(known))


def engagement_rate(post: dict) -> float | None:
    """View-based engagement rate; None when views or all actions are unknown."""
    views = post.get("views")
    actions = _engagement_actions(post)
    if not isinstance(views, (int, float)) or views <= 0 or actions is None:
        return None
    return round(actions / views, 6)


def _median(values: list) -> float | None:
    cleaned = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _percentile_rank(value: float, values: list[float]) -> float:
    """Fraction of values <= value; 0..1. Stable for small samples."""
    if not values:
        return 0.5
    below = sum(1 for v in values if v <= value)
    return below / len(values)


def score_account(posts: list[dict]) -> dict:
    """Score one account's posts against its own baseline.

    Returns {"baseline": {...}, "posts": [annotated posts sorted best-first]}.
    Comparing a post to its own account's median controls for audience size,
    so a 50K-view outlier on a 5K-median channel outranks a 100K-view video
    on a 200K-median channel.
    """
    median_views = _median([p.get("views") for p in posts])
    rates = [r for r in (engagement_rate(p) for p in posts) if r is not None]
    median_er = _median(rates)
    like_values = [
        float(p["likes"]) for p in posts if isinstance(p.get("likes"), (int, float)) and p["likes"] > 0
    ]

    annotated = []
    for post in posts:
        entry = dict(post)
        er = engagement_rate(post)
        views = post.get("views")

        views_vs_median = None
        if isinstance(views, (int, float)) and views > 0 and median_views:
            views_vs_median = round(views / median_views, 3)

        if views_vs_median is not None:
            index = views_vs_median
            if er is not None and median_er:
                # Reward above-baseline engagement, cap the multiplier so a
                # tiny-view post with 3 likes can't dominate the ranking.
                index *= min(max(er / median_er, 0.25), 4.0)
            basis = "views"
        elif isinstance(post.get("likes"), (int, float)) and post["likes"] > 0 and like_values:
            # No view counts (manual X/Instagram entries): likes percentile.
            index = round(2.0 * _percentile_rank(float(post["likes"]), like_values), 3)
            basis = "likes-percentile"
        else:
            index = None
            basis = "unscored"

        entry["engagement_rate"] = er
        entry["views_vs_median"] = views_vs_median
        entry["performance_index"] = round(index, 3) if index is not None else None
        entry["score_basis"] = basis
        annotated.append(entry)

    annotated.sort(key=lambda p: (p["performance_index"] is not None, p["performance_index"] or 0.0), reverse=True)
    scored = [p for p in annotated if p["performance_index"] is not None]
    top_quartile_cut = max(1, len(scored) // 4) if scored else 0
    for i, post in enumerate(annotated):
        post["rank"] = i + 1
        post["top_performer"] = post["performance_index"] is not None and i < top_quartile_cut

    return {
        "baseline": {
            "post_count": len(posts),
            "median_views": median_views,
            "median_engagement_rate": median_er,
            "scored_count": len(scored),
        },
        "posts": annotated,
    }


def load_documents(paths: list[str]) -> dict[str, list[dict]]:
    """Load collect_profile.py documents, grouped per profile."""
    accounts: dict[str, list[dict]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            profile, posts = path, data
        else:
            profile = data.get("profile") or path
            posts = data.get("posts", [])
        accounts.setdefault(profile, []).extend(posts)
    return accounts


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def render_markdown(results: dict[str, dict], top: int | None) -> str:
    """Human-readable ranked tables, one section per account."""
    lines = []
    for profile, result in results.items():
        base = result["baseline"]
        lines.append(f"## {profile}")
        lines.append("")
        lines.append(
            f"Posts: {base['post_count']} ({base['scored_count']} scored) | "
            f"median views: {_fmt(base['median_views'])} | "
            f"median ER: {_fmt((base['median_engagement_rate'] or 0) * 100 if base['median_engagement_rate'] else None, '%')}"
        )
        lines.append("")
        lines.append("| # | Title | Date | Views | ER | vs median | Index | Top |")
        lines.append("|---|---|---|---|---|---|---|---|")
        posts = result["posts"][:top] if top else result["posts"]
        for post in posts:
            er = post["engagement_rate"]
            title = (post.get("title") or "")[:60].replace("|", "\\|")
            lines.append(
                "| {rank} | {title} | {date} | {views} | {er} | {vsm} | {idx} | {topmark} |".format(
                    rank=post["rank"],
                    title=title or "(untitled)",
                    date=post.get("upload_date") or "—",
                    views=_fmt(post.get("views")),
                    er=_fmt(er * 100 if er is not None else None, "%"),
                    vsm=_fmt(post.get("views_vs_median"), "x"),
                    idx=_fmt(post.get("performance_index")),
                    topmark="*" if post["top_performer"] else "",
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="JSON files from collect_profile.py")
    parser.add_argument("--top", type=int, help="Limit output to the top N posts per account")
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table instead of JSON")
    parser.add_argument("-o", "--output", help="Write output here instead of stdout")
    args = parser.parse_args(argv)

    accounts = load_documents(args.inputs)
    if not any(accounts.values()):
        print("No posts found in input files.", file=sys.stderr)
        return 2

    results = {profile: score_account(posts) for profile, posts in accounts.items()}

    if args.markdown:
        payload = render_markdown(results, args.top)
    else:
        if args.top:
            results = {
                profile: {"baseline": r["baseline"], "posts": r["posts"][: args.top]}
                for profile, r in results.items()
            }
        payload = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
