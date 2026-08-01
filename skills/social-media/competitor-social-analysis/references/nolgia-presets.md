# Mapping content ideas to Nolgia presets

Every content idea must name the Nolgia preset that would produce it. Anchor on
the customer-facing preset catalog below; it is what the founder can generate
with today.

## Source of truth

The live catalog is the `presets` table in nolgia-api, served by `GET /presets`
and managed at `/admin/presets`. The seed definitions live in
`nolgia-api/db/migrations/000029_seed_presets.up.sql` (+ `000033` for
`vfx-my-footage`), and the CI-gated canonical slug list is in
`nolgia-api/internal/handlers/presets_store_test.go`. **If you can reach the
API or repo, refresh this table before relying on it** — presets are added and
promoted over time, and this file is a snapshot.

There is also an internal, admin-only production catalog (37 workflow recipes,
`nolgia-agent/overlays/nolgia-admin/skills/creative/nolgia-preset-catalog/`)
with motion/marketing/e-commerce recipes not yet on the public page. When an
idea has no good customer-facing preset, check that catalog and mark the idea
`internal-recipe:` — it needs an admin-side run, not a customer generation.

## Customer-facing presets (snapshot, 12)

| Slug | Output | Style | Competitor patterns that map here |
| --- | --- | --- | --- |
| `ugc-ad` | video | Creator-style talking head, hook text, timed captions, CTA end-card | Talking-head hooks, creator testimonials, paid-social style Reels/Shorts |
| `short-film` | video | Scripted multi-shot scenes, consistent characters | Narrative/storytelling content, mini-documentary arcs |
| `animated-cartoon` | video | 2D cartoon, bold outlines, warm palette | Mascot content, kids-adjacent explainers, comic skits |
| `ugc-try-on` | video | Selfie-style mirror try-on, natural light, vertical | Fashion/beauty try-on content, fit checks |
| `ugc-unboxing` | video | First-reaction unboxing, close-up hands, vertical | Unboxings, first-impression reviews, haul content |
| `vfx-my-footage` | video | Restyle uploaded footage, keeps motion, transforms look | Trend formats that restyle real clips (anime-fy, era swaps) |
| `product-demo` | video | Clean studio shots, slow orbits, hands-on features | Feature walkthroughs, before/after tool demos |
| `ecommerce-product-photos` | image | Seamless background, studio light, catalog-ready | Product carousel posts, catalog refresh content |
| `commercial` | video | Polished multi-shot: hero, lifestyle beats, branded close | High-production brand spots, launch films |
| `social-media-clip` | video | Punchy loop-worthy vertical, scroll-stopping first second | Trend clips, loops, meme-adjacent shortform |
| `music-video` | video | Stylized multi-scene, synced cuts, bold looks | Music-driven edits, aesthetic montages, sound-led trends |
| `explainer-video` | video | Friendly 2D motion graphics, animated icons/charts | How-it-works content, stat posts, feature announcements |

## Idea format

```markdown
| # | Idea | Evidence | Pattern | Preset | Notes |
| 1 | "We rebuilt <viral thing> in 60s" | HF posts #2, #5 (3.1x, 2.4x median) | result-first hook + fast punch-in edit | `social-media-clip` | word-pop captions, <20s |
```

Rules:

- **Evidence first.** Each idea cites the competitor posts (and their
  performance multiples) that prove the pattern. No evidence, no idea.
- **Adapt, don't clone.** The idea must be a Nolgia-native concept using the
  *pattern* (hook formula, pacing, format), never a re-shoot of a competitor
  script.
- **Respect the preset's lane.** Presets carry guardrails server-side; pick the
  preset whose scope actually covers the idea rather than forcing the closest
  match. If nothing fits, say so and tag `internal-recipe:` or propose it as a
  new-preset candidate for the founder.
- **Note the format constraints** the evidence demands (duration band,
  orientation, caption style) so the generating session can honor them.
