---
name: competitor-social-analysis
description: "Use when analyzing competitor social profiles: collect posts, score performance, break down creative style, propose Nolgia content ideas."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [competitor-analysis, social-media, content-strategy, video-breakdown, nolgia]
    related_skills: [xurl, youtube-content]
---

# Competitor Social Analysis

Turn competitor social profiles into a performance breakdown and a Nolgia-ready
content plan. The founder provides profile URL(s) (e.g. Higgsfield, InVideo);
this skill collects recent posts, scores what actually performed, breaks down
the creative patterns behind the winners, and proposes video ideas mapped to
Nolgia generation presets.

## When to use

Use when the user asks to analyze a competitor's social presence, find out what
content is working for another brand, break down a competitor's video style
(hooks, editing, fonts, voiceover, color grade), or build a content calendar
from proven competitor patterns.

## What this skill does NOT do

- No login-walled scraping. If a platform blocks anonymous access, say so and
  use the manual-input fallback — never ask for, store, or use credentials to
  work around a wall, and never drive a logged-in browser session for scraping.
- No engagement automation (following, liking, mass-viewing competitors).
- No copying. Output is pattern analysis and original ideas, not scripts to
  re-shoot competitor content beat-for-beat.

## Setup

Live collection uses `yt-dlp` (metadata only, no video downloads):

```bash
uv pip install yt-dlp
```

`yt-dlp` is optional — every step below has an offline path fed by
founder-provided metrics (see "Manual-input fallback").

## Platform support (be honest with the user)

| Platform | Anonymous collection | What to do |
| --- | --- | --- |
| YouTube (channels, `/videos`, `/shorts`) | Reliable | `collect_profile.py` with the channel URL |
| TikTok (public profiles) | Intermittent — works some days, breaks others | Try once, retry once, then fall back to manual input |
| Instagram / Reels | Blocked for anonymous clients | Manual input only — founder pastes metrics or screenshots |
| X / Twitter | API-only | Use the `xurl` skill if configured; otherwise manual input |

When collection fails on a blocked platform, tell the user plainly that the
platform prevents anonymous scraping and hand them the template from
`SKILL_DIR/references/manual-input.md`. Do not burn time on evasion.

## Helper scripts

`SKILL_DIR` is the directory containing this SKILL.md file.

```bash
# 1. Collect recent posts from a public profile (newest ~30)
python3 SKILL_DIR/scripts/collect_profile.py "https://www.youtube.com/@competitor" --limit 30 -o competitor.json

# Ingest founder-provided metrics for blocked platforms (JSON or CSV)
python3 SKILL_DIR/scripts/collect_profile.py --manual metrics.csv --platform instagram -o competitor_ig.json

# Normalize a pre-existing yt-dlp --dump-json capture (offline)
python3 SKILL_DIR/scripts/collect_profile.py --from-json dump.jsonl --platform youtube -o competitor.json

# 2. Score and rank (multiple files = one section per account)
python3 SKILL_DIR/scripts/score_posts.py competitor.json competitor_ig.json --markdown
python3 SKILL_DIR/scripts/score_posts.py competitor.json -o scored.json
```

Both scripts are stdlib-only and make no model/API calls. `collect_profile.py`
touches the network only when given a live profile URL (via yt-dlp).

## Workflow

1. **Intake.** Confirm competitor profile URL(s), the platforms in scope, and
   the goal (style study, trend monitoring, content-calendar input). Default
   window: the 30 most recent posts per profile.
2. **Collect.** Run `collect_profile.py` per profile. On failure, follow the
   platform table above — retry once for intermittent platforms, go straight
   to manual input for blocked ones. Never mix platforms in one file; collect
   per-profile so baselines stay honest.
3. **Score.** Run `score_posts.py` across the collected files. Posts are ranked
   against their own account's median (`performance_index`), so small and large
   accounts compare fairly. Treat `top_performer: true` posts as the study set.
   Sanity-check the baseline block — a median of 3 posts is a weak signal; say
   so in the report.
4. **Creative breakdown.** For each top performer, fill in the template at
   `SKILL_DIR/references/breakdown-template.md`: hook, structure/pacing,
   editing style, fonts/captions, voiceover, color grade, sound, CTA.
   - Metadata alone (title, duration, date) supports hypotheses about hooks and
     packaging. Everything visual (fonts, grade, editing) requires actually
     watching the videos or reading founder-provided screenshots — see the cost
     note below before doing that.
   - Then synthesize *patterns across winners* — that section of the template
     is the actual deliverable; per-post notes are evidence.
5. **Content ideas.** Map each winning pattern to Nolgia output using
   `SKILL_DIR/references/nolgia-presets.md`. Every idea must cite its
   competitor evidence (which posts prove the pattern) and name the preset slug
   that would produce it. 5–10 ideas is the sweet spot; rank by evidence
   strength.
6. **Deliver.** One report: scored table (from `--markdown`), pattern
   synthesis, ideas table mapped to presets, and a short "what we could not
   measure" section listing platforms/metrics that were blocked or missing.

## Cost and authorization

- Metadata collection and scoring are free — no model calls, run them freely.
- **Watching competitor videos (or analyzing screenshots/frames) uses
  vision-model spend. Get explicit user approval for the batch before
  analyzing, and state roughly how many videos you intend to watch.** Under a
  spend freeze, deliver steps 1–3 plus a metadata-only breakdown, and flag the
  visual analysis as pending authorization.
- Prefer analyzing the top 5–8 posts, not the whole collection.

## Manual-input fallback

When a platform is blocked (or the founder simply pastes analytics), use
`SKILL_DIR/references/manual-input.md`: it defines the CSV/JSON columns
`collect_profile.py --manual` accepts and how to transcribe metrics from
screenshots. Founder-provided screenshots of competitor posts are also valid
input for the creative breakdown — treat them as the "watch" step for that
post.

## Compliance guardrails

- Public metadata only; respect robots/rate limits (yt-dlp defaults are fine —
  do not add aggressive retry loops).
- Never impersonate a logged-in user or rotate IPs/user-agents to evade blocks.
- Competitor content is studied for patterns. Do not reproduce competitor
  scripts, watermarks, or branded assets in Nolgia output.

## Error handling

| Symptom | Cause | Fix |
| --- | --- | --- |
| `yt-dlp is not installed` | Missing dependency | `uv pip install yt-dlp`, or switch to `--manual` |
| Collector exits 2 with "blocked" guidance | Platform walls anonymous access | Manual-input fallback; do not retry endlessly |
| Collector exits 3 (no posts) | Empty/private profile or soft block | Verify the URL in a browser; ask the founder for metrics |
| Scorer prints "No posts found" | Empty input files | Re-run collection; check the right file paths were passed |
| All `performance_index` null | No views/likes in the data | Data too sparse to rank — ask for views or likes at minimum |
| TikTok worked yesterday, fails today | Intermittent anonymous access | Retry once, then manual input; note the gap in the report |

## Output format (report skeleton)

```markdown
# Competitor social analysis — <competitors> — <date>

## Performance summary
<score_posts.py --markdown output>

## What the winners have in common
<pattern synthesis from breakdown-template.md>

## Content ideas for Nolgia
| # | Idea | Evidence (competitor posts) | Pattern | Nolgia preset | Notes |

## What we could not measure
<blocked platforms, missing metrics, unwatched videos pending approval>
```
