"""Invariant + helper-script tests for skills/social-media/competitor-social-analysis (NOL-259).

All tests are offline: yt-dlp is never invoked (live collection is exercised
through the --from-json / --manual paths and mocks), and no model/API calls
are made anywhere.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "social-media" / "competitor-social-analysis"
SCRIPTS_DIR = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import collect_profile
import score_posts


def _frontmatter(skill_md: Path) -> dict:
    import yaml

    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{skill_md} has no YAML frontmatter"
    return yaml.safe_load(match.group(1))


# ---------------------------------------------------------------------------
# Skill invariants
# ---------------------------------------------------------------------------


class TestSkillInvariants:
    def test_skill_md_exists_with_frontmatter(self):
        fm = _frontmatter(SKILL_DIR / "SKILL.md")
        assert fm["name"] == "competitor-social-analysis"
        assert fm["description"]
        assert "linux" in fm["platforms"]

    def test_reference_files_exist(self):
        for name in ("breakdown-template.md", "nolgia-presets.md", "manual-input.md"):
            assert (SKILL_DIR / "references" / name).exists(), f"missing references/{name}"

    def test_helper_scripts_exist(self):
        for name in ("collect_profile.py", "score_posts.py"):
            assert (SCRIPTS_DIR / name).exists(), f"missing scripts/{name}"

    def test_generated_docs_page_exists(self):
        page = (
            REPO
            / "website"
            / "docs"
            / "user-guide"
            / "skills"
            / "bundled"
            / "social-media"
            / "social-media-competitor-social-analysis.md"
        )
        assert page.exists(), f"missing {page}; run website/scripts/generate-skill-docs.py"

    def test_skill_is_honest_about_blocked_platforms(self):
        body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "manual-input" in body, "SKILL.md must document the manual fallback"
        assert "Blocked" in body, "SKILL.md must state which platforms block scraping"
        assert "No login-walled scraping" in body

    def test_skill_requires_spend_authorization_for_vision(self):
        body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "approval" in body.lower(), "vision analysis must require explicit approval"

    def test_presets_reference_lists_canonical_slugs(self):
        body = (SKILL_DIR / "references" / "nolgia-presets.md").read_text(encoding="utf-8")
        # Canonical customer-facing slug list (nolgia-api presets_store_test.go).
        for slug in (
            "ugc-ad",
            "short-film",
            "animated-cartoon",
            "ugc-try-on",
            "ugc-unboxing",
            "vfx-my-footage",
            "product-demo",
            "ecommerce-product-photos",
            "commercial",
            "social-media-clip",
            "music-video",
            "explainer-video",
        ):
            assert f"`{slug}`" in body, f"nolgia-presets.md missing preset {slug}"
        assert "source of truth" in body.lower(), "must point at the live catalog"

    def test_scripts_are_stdlib_only_and_offline(self):
        # No third-party imports and no model/API clients in the helper scripts.
        for name in ("collect_profile.py", "score_posts.py"):
            text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            for forbidden in ("anthropic", "openai", "requests", "httpx", "urllib.request"):
                assert forbidden not in text, f"{name} must not import {forbidden}"


# ---------------------------------------------------------------------------
# collect_profile.py
# ---------------------------------------------------------------------------


class TestDetectPlatform:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/@higgsfield", "youtube"),
            ("https://youtube.com/@x/shorts", "youtube"),
            ("https://www.tiktok.com/@invideo.io", "tiktok"),
            ("https://www.instagram.com/higgsfieldai/", "instagram"),
            ("https://x.com/higgsfield_ai", "x"),
            ("https://twitter.com/someone", "x"),
            ("https://example.com/profile", "unknown"),
        ],
    )
    def test_detection(self, url, expected):
        assert collect_profile.detect_platform(url) == expected


class TestBuildYtdlpCommand:
    def test_metadata_only_flags(self):
        cmd = collect_profile.build_ytdlp_command("https://youtube.com/@a", 25)
        assert cmd[0] == "yt-dlp"
        assert "--skip-download" in cmd
        assert "--dump-json" in cmd
        assert cmd[cmd.index("--playlist-end") + 1] == "25"


YTDLP_FIXTURE = "\n".join(
    [
        json.dumps(
            {
                "id": "vid1",
                "title": "How we made this in 60 seconds",
                "webpage_url": "https://youtube.com/watch?v=vid1",
                "upload_date": "20260710",
                "duration": 58,
                "view_count": 150000,
                "like_count": 9000,
                "comment_count": 410,
            }
        ),
        "not json at all",
        json.dumps({"_type": "playlist", "id": "chan", "entries": []}),
        json.dumps(
            {
                "id": "vid2",
                "title": "Weekly update",
                "webpage_url": "https://youtube.com/watch?v=vid2",
                "upload_date": "20260715",
                "duration": 300,
                "view_count": 8000,
                "like_count": None,
            }
        ),
    ]
)


class TestNormalization:
    def test_parse_ytdlp_output_skips_garbage_and_wrappers(self):
        posts = collect_profile.parse_ytdlp_output("youtube", YTDLP_FIXTURE)
        assert [p["id"] for p in posts] == ["vid1", "vid2"]

    def test_entry_fields_mapped(self):
        post = collect_profile.parse_ytdlp_output("youtube", YTDLP_FIXTURE)[0]
        assert post["platform"] == "youtube"
        assert post["upload_date"] == "2026-07-10"
        assert post["views"] == 150000
        assert post["likes"] == 9000
        assert post["comments"] == 410

    def test_missing_counts_are_none_not_zero(self):
        post = collect_profile.parse_ytdlp_output("youtube", YTDLP_FIXTURE)[1]
        assert post["likes"] is None
        assert post["comments"] is None
        assert post["shares"] is None

    def test_manual_rows_with_comma_counts_and_blanks(self):
        rows = [
            {"url": "https://instagram.com/p/a/", "title": "Reel A", "date": "2026-07-01", "views": "1,200,000", "likes": "45.3", "comments": ""},
        ]
        posts = collect_profile.parse_manual_rows("instagram", rows)
        assert posts[0]["views"] == 1200000
        assert posts[0]["comments"] is None
        assert posts[0]["platform"] == "instagram"

    def test_manual_rows_skip_empty(self):
        assert collect_profile.parse_manual_rows("instagram", [{"title": ""}]) == []


class TestCollectMain:
    def test_from_json_end_to_end(self, tmp_path):
        src = tmp_path / "dump.jsonl"
        src.write_text(YTDLP_FIXTURE, encoding="utf-8")
        out = tmp_path / "posts.json"
        rc = collect_profile.main(["--from-json", str(src), "--platform", "youtube", "-o", str(out)])
        assert rc == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["platform"] == "youtube"
        assert doc["post_count"] == 2

    def test_manual_csv_end_to_end(self, tmp_path):
        src = tmp_path / "metrics.csv"
        src.write_text(
            "url,title,date,views,likes,comments\n"
            "https://instagram.com/p/a/,Reel A,2026-07-01,120000,8400,312\n"
            "https://instagram.com/p/b/,Reel B,2026-07-05,45000,2100,\n",
            encoding="utf-8",
        )
        out = tmp_path / "posts.json"
        rc = collect_profile.main(["--manual", str(src), "--platform", "instagram", "-o", str(out)])
        assert rc == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["post_count"] == 2
        assert doc["posts"][1]["comments"] is None

    def test_requires_exactly_one_input(self, capsys):
        with pytest.raises(SystemExit):
            collect_profile.main([])

    def test_missing_ytdlp_gives_guidance(self, monkeypatch, capsys):
        monkeypatch.setattr(collect_profile.shutil, "which", lambda _: None)
        rc = collect_profile.main(["https://youtube.com/@a"])
        assert rc == 2
        assert "yt-dlp is not installed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# score_posts.py
# ---------------------------------------------------------------------------


def _post(pid, views=None, likes=None, comments=None, shares=None, title=""):
    return {
        "platform": "youtube",
        "id": pid,
        "url": f"https://example.test/{pid}",
        "title": title or pid,
        "upload_date": "2026-07-01",
        "duration_s": 60,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
    }


class TestEngagementRate:
    def test_normal(self):
        assert score_posts.engagement_rate(_post("a", views=1000, likes=90, comments=10)) == 0.1

    def test_no_views_is_none(self):
        assert score_posts.engagement_rate(_post("a", likes=90)) is None

    def test_no_actions_is_none(self):
        assert score_posts.engagement_rate(_post("a", views=1000)) is None


class TestScoreAccount:
    def test_outlier_ranks_first_and_is_top_performer(self):
        posts = [
            _post("mid1", views=10000, likes=500, comments=50),
            _post("viral", views=100000, likes=9000, comments=900),
            _post("mid2", views=9000, likes=400, comments=40),
            _post("flop", views=2000, likes=30, comments=2),
        ]
        result = score_posts.score_account(posts)
        ranked = result["posts"]
        assert ranked[0]["id"] == "viral"
        assert ranked[0]["top_performer"] is True
        assert ranked[-1]["id"] == "flop"
        assert result["baseline"]["median_views"] == 9500.0

    def test_likes_percentile_fallback_when_no_views(self):
        posts = [_post("a", likes=100), _post("b", likes=5000), _post("c", likes=800)]
        result = score_posts.score_account(posts)
        best = result["posts"][0]
        assert best["id"] == "b"
        assert best["score_basis"] == "likes-percentile"

    def test_unscorable_posts_rank_last_not_crash(self):
        posts = [_post("a", views=1000, likes=100), _post("empty")]
        ranked = score_posts.score_account(posts)["posts"]
        assert ranked[-1]["id"] == "empty"
        assert ranked[-1]["performance_index"] is None
        assert ranked[-1]["score_basis"] == "unscored"

    def test_small_view_high_er_cannot_dominate(self):
        posts = [
            _post("tiny", views=50, likes=40),  # 80% ER on 50 views
            _post("real", views=50000, likes=2000, comments=200),
            _post("base1", views=10000, likes=400),
            _post("base2", views=12000, likes=500),
        ]
        ranked = score_posts.score_account(posts)["posts"]
        assert ranked[0]["id"] == "real"


class TestScoreMain:
    def test_end_to_end_json_and_markdown(self, tmp_path, capsys):
        doc = {
            "platform": "youtube",
            "profile": "https://youtube.com/@competitor",
            "posts": [
                _post("a", views=10000, likes=500, comments=50, title="Post | A"),
                _post("b", views=90000, likes=8000, comments=700),
            ],
        }
        src = tmp_path / "posts.json"
        src.write_text(json.dumps(doc), encoding="utf-8")

        rc = score_posts.main([str(src)])
        assert rc == 0
        results = json.loads(capsys.readouterr().out)
        assert "https://youtube.com/@competitor" in results

        rc = score_posts.main([str(src), "--markdown", "--top", "1"])
        assert rc == 0
        md = capsys.readouterr().out
        assert "| # | Title | Date |" in md
        assert "Post \\| A" not in md.split("\n\n")[0]  # top-1 keeps only the winner

    def test_empty_inputs_error(self, tmp_path, capsys):
        src = tmp_path / "empty.json"
        src.write_text(json.dumps({"profile": "p", "posts": []}), encoding="utf-8")
        assert score_posts.main([str(src)]) == 2
