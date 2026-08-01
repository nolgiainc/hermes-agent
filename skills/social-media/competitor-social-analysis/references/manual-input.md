# Manual-input fallback

For platforms that block anonymous collection (Instagram/Reels, X without API
access, TikTok on a bad day), the founder provides the numbers and
`collect_profile.py --manual` ingests them. Screenshots of the competitor's
profile grid or per-post analytics are the usual source.

## CSV template

Hand the founder exactly this (or fill it yourself while reading their
screenshots):

```csv
url,title,date,views,likes,comments,shares,duration
https://www.instagram.com/reel/ABC123/,"Hook text or caption",2026-07-20,120000,8400,312,,-
https://www.instagram.com/reel/DEF456/,"Second reel caption",2026-07-24,45000,2100,98,,-
```

- `url` — post URL (used as the ID; required)
- `title` — caption or on-screen hook text
- `date` — upload date, `YYYY-MM-DD`
- `views`, `likes`, `comments`, `shares` — leave **empty** when the platform
  hides the number. Empty means "unknown"; `0` means "actually zero". The
  scorer handles missing views by falling back to a likes-based ranking.
- `duration` — seconds, if known

JSON is also accepted: a list of objects with the same keys (or a
`{"posts": [...]}` wrapper).

## Transcribing from screenshots

1. Ask for: the profile grid (shows relative view counts) plus per-post
   screenshots of the posts that look strongest, and the account's follower
   count.
2. Transcribe abbreviated counts precisely: `1.2M` → `1200000`, `45.3K` →
   `45300`. When a count is truncated (e.g. "10K+"), record the floor and note
   it in the report.
3. Reading metrics off screenshots is cheap; but treating screenshots as the
   *creative breakdown* source (fonts, grade, editing) is vision-model work —
   the cost note in SKILL.md applies.
4. Record which posts came from screenshots in the final report's "what we
   could not measure" section: founder-curated screenshots skew toward
   winners, so the account median from manual data is optimistic and the
   `views_vs_median` multiples are conservative. Say this in the report.
