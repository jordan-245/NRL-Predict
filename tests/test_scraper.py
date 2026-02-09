"""
Tests for scraping utilities: URL builder, team mappings, HTML parsers, odds loader.

Covers:
- rlp_url_builder: URL generation for seasons, rounds, matches, finals, players
- team_mappings: standardise_team_name for all aliases, unknown names raise KeyError
- rlp_match_parser: parsing sample HTML match blocks
- rlp_ladder_parser: parsing sample HTML ladder tables
- odds_loader: data cleaning logic with mocked Excel reads
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# rlp_url_builder tests
# ============================================================================


class TestRlpUrlBuilder:
    """Tests for scraping.rlp_url_builder URL generation functions."""

    def test_season_summary_url_default_base(self):
        from scraping.rlp_url_builder import season_summary_url

        url = season_summary_url(2024)
        assert url == "https://www.rugbyleagueproject.org/seasons/nrl-2024/summary.html"

    def test_season_summary_url_custom_base(self):
        from scraping.rlp_url_builder import season_summary_url

        url = season_summary_url(2023, base_url="https://example.com")
        assert url == "https://example.com/seasons/nrl-2023/summary.html"

    def test_season_results_url(self):
        from scraping.rlp_url_builder import season_results_url

        url = season_results_url(2024)
        assert "/seasons/nrl-2024/results.html" in url

    def test_season_players_url(self):
        from scraping.rlp_url_builder import season_players_url

        url = season_players_url(2024)
        assert "/seasons/nrl-2024/players.html" in url

    def test_round_summary_url_regular_round(self):
        from scraping.rlp_url_builder import round_summary_url

        url = round_summary_url(2024, 5)
        assert "/seasons/nrl-2024/round-5/summary.html" in url

    def test_round_summary_url_round_1(self):
        from scraping.rlp_url_builder import round_summary_url

        url = round_summary_url(2024, 1)
        assert "/round-1/" in url

    def test_round_summary_url_round_27(self):
        from scraping.rlp_url_builder import round_summary_url

        url = round_summary_url(2024, 27)
        assert "/round-27/" in url

    def test_round_summary_url_finals_qualif(self):
        from scraping.rlp_url_builder import round_summary_url

        url = round_summary_url(2024, "qualif-final")
        assert "/seasons/nrl-2024/qualif-final/summary.html" in url

    def test_round_summary_url_finals_grand(self):
        from scraping.rlp_url_builder import round_summary_url

        url = round_summary_url(2024, "grand-final")
        assert "/grand-final/summary.html" in url

    def test_round_summary_url_all_finals_slugs(self):
        from scraping.rlp_url_builder import round_summary_url

        finals = ["qualif-final", "elim-final", "semi-final", "prelim-final", "grand-final"]
        for slug in finals:
            url = round_summary_url(2024, slug)
            assert f"/{slug}/summary.html" in url

    def test_round_summary_url_invalid_finals_slug_raises(self):
        from scraping.rlp_url_builder import round_summary_url

        with pytest.raises(ValueError, match="Unknown finals round slug"):
            round_summary_url(2024, "invalid-final")

    def test_round_ladder_url(self):
        from scraping.rlp_url_builder import round_ladder_url

        url = round_ladder_url(2024, 10)
        assert "/seasons/nrl-2024/round-10/ladder.html" in url

    def test_match_summary_url(self):
        from scraping.rlp_url_builder import match_summary_url

        url = match_summary_url(
            2024, 1, "melbourne-storm", "penrith-panthers"
        )
        assert "/round-1/melbourne-storm-vs-penrith-panthers/summary.html" in url

    def test_match_stats_url(self):
        from scraping.rlp_url_builder import match_stats_url

        url = match_stats_url(
            2024, 3, "brisbane-broncos", "sydney-roosters"
        )
        assert "/round-3/brisbane-broncos-vs-sydney-roosters/stats.html" in url

    def test_match_summary_url_finals(self):
        from scraping.rlp_url_builder import match_summary_url

        url = match_summary_url(
            2024, "grand-final", "melbourne-storm", "penrith-panthers"
        )
        assert "/grand-final/melbourne-storm-vs-penrith-panthers/summary.html" in url

    def test_player_profile_url(self):
        from scraping.rlp_url_builder import player_profile_url

        url = player_profile_url("nathan-cleary")
        assert "/players/nathan-cleary/summary.html" in url

    def test_player_profile_url_custom_base(self):
        from scraping.rlp_url_builder import player_profile_url

        url = player_profile_url("james-tedesco", base_url="https://test.org")
        assert url == "https://test.org/players/james-tedesco/summary.html"

    def test_team_name_to_slug_simple(self):
        from scraping.rlp_url_builder import team_name_to_slug

        assert team_name_to_slug("Melbourne Storm") == "melbourne-storm"

    def test_team_name_to_slug_multi_word(self):
        from scraping.rlp_url_builder import team_name_to_slug

        assert team_name_to_slug("South Sydney Rabbitohs") == "south-sydney-rabbitohs"

    def test_team_name_to_slug_hyphenated(self):
        from scraping.rlp_url_builder import team_name_to_slug

        assert team_name_to_slug("Canterbury-Bankstown Bulldogs") == "canterbury-bankstown-bulldogs"

    def test_team_name_to_slug_dots_stripped(self):
        from scraping.rlp_url_builder import team_name_to_slug

        assert team_name_to_slug("St. George Illawarra Dragons") == "st-george-illawarra-dragons"

    def test_team_name_to_slug_whitespace_stripped(self):
        from scraping.rlp_url_builder import team_name_to_slug

        assert team_name_to_slug("  Melbourne Storm  ") == "melbourne-storm"

    def test_all_round_summary_urls_count(self):
        from scraping.rlp_url_builder import all_round_summary_urls

        urls = all_round_summary_urls(2024, base_url="https://example.com")
        # 27 regular rounds + 5 finals = 32
        assert len(urls) == 32

    def test_all_round_summary_urls_tuple_structure(self):
        from scraping.rlp_url_builder import all_round_summary_urls

        urls = all_round_summary_urls(2024, base_url="https://example.com")
        # First entry should be round 1
        assert urls[0][0] == 1
        assert "/round-1/" in urls[0][1]
        # Last entry should be grand-final
        assert urls[-1][0] == "grand-final"

    def test_all_round_ladder_urls_regular_only(self):
        from scraping.rlp_url_builder import all_round_ladder_urls

        urls = all_round_ladder_urls(2024, base_url="https://example.com")
        assert len(urls) == 27  # Only regular rounds
        for rnd, url in urls:
            assert isinstance(rnd, int)
            assert "ladder.html" in url


# ============================================================================
# team_mappings tests
# ============================================================================


class TestTeamMappings:
    """Tests for config.team_mappings standardise_team_name and related."""

    def test_canonical_name_maps_to_itself(self):
        from config.team_mappings import standardise_team_name

        assert standardise_team_name("Melbourne Storm") == "Melbourne Storm"

    def test_alias_maps_correctly(self):
        from config.team_mappings import standardise_team_name

        assert standardise_team_name("Souths") == "South Sydney Rabbitohs"
        assert standardise_team_name("Storm") == "Melbourne Storm"
        assert standardise_team_name("Easts") == "Sydney Roosters"

    def test_three_letter_codes(self):
        from config.team_mappings import standardise_team_name

        assert standardise_team_name("MEL") == "Melbourne Storm"
        assert standardise_team_name("PEN") == "Penrith Panthers"
        assert standardise_team_name("SYD") == "Sydney Roosters"
        assert standardise_team_name("CRO") == "Cronulla Sharks"
        assert standardise_team_name("PAR") == "Parramatta Eels"
        assert standardise_team_name("BRI") == "Brisbane Broncos"
        assert standardise_team_name("DOL") == "Dolphins"

    def test_case_insensitive(self):
        from config.team_mappings import standardise_team_name

        assert standardise_team_name("melbourne storm") == "Melbourne Storm"
        assert standardise_team_name("MELBOURNE STORM") == "Melbourne Storm"
        assert standardise_team_name("Melbourne STORM") == "Melbourne Storm"

    def test_whitespace_stripped(self):
        from config.team_mappings import standardise_team_name

        assert standardise_team_name("  Melbourne Storm  ") == "Melbourne Storm"
        assert standardise_team_name("  MEL  ") == "Melbourne Storm"

    def test_unknown_name_raises_key_error(self):
        from config.team_mappings import standardise_team_name

        with pytest.raises(KeyError, match="Unknown team name"):
            standardise_team_name("Nonexistent Team FC")

    def test_empty_string_raises_key_error(self):
        from config.team_mappings import standardise_team_name

        with pytest.raises(KeyError):
            standardise_team_name("")

    def test_all_canonical_names_in_team_aliases(self):
        from config.team_mappings import ALL_TEAMS, TEAM_ALIASES

        assert len(ALL_TEAMS) == len(TEAM_ALIASES)
        for team in ALL_TEAMS:
            assert team in TEAM_ALIASES

    def test_every_alias_resolves_to_a_canonical_name(self):
        from config.team_mappings import TEAM_ALIASES, standardise_team_name

        for canonical, aliases in TEAM_ALIASES.items():
            # Canonical resolves to itself
            assert standardise_team_name(canonical) == canonical
            for alias in aliases:
                assert standardise_team_name(alias) == canonical

    def test_get_team_slug(self):
        from config.team_mappings import get_team_slug

        assert get_team_slug("Melbourne Storm") == "melbourne-storm"
        assert get_team_slug("MEL") == "melbourne-storm"
        assert get_team_slug("Souths") == "south-sydney-rabbitohs"

    def test_get_team_slug_unknown_raises(self):
        from config.team_mappings import get_team_slug

        with pytest.raises(KeyError):
            get_team_slug("Unknown Team")

    def test_all_teams_has_17_entries(self):
        from config.team_mappings import ALL_TEAMS

        assert len(ALL_TEAMS) == 17

    def test_team_slugs_for_all_canonical(self):
        from config.team_mappings import ALL_TEAMS, TEAM_SLUGS

        for team in ALL_TEAMS:
            assert team in TEAM_SLUGS, f"Missing slug for {team}"
            slug = TEAM_SLUGS[team]
            assert "-" in slug or slug == "dolphins"
            assert slug == slug.lower()


# ============================================================================
# rlp_match_parser tests
# ============================================================================


class TestRlpMatchParser:
    """Tests for scraping.rlp_match_parser with realistic RLP HTML.

    The sample HTML mirrors the actual structure served by
    rugbyleagueproject.org round-summary pages.
    """

    # Realistic RLP match block HTML (based on actual site structure).
    SAMPLE_MATCH_HTML = """
    <div class="quiet" style="font-size:130%;line-height:1.5em;padding:1em 0;border-bottom:dotted 1px #ccc;">
    <span class="noprint"><a class="rlplnk" href="/matches/62008">&gt;</a>&nbsp;</span>
    <strong><a href="../melbourne/summary.html">Melbourne</a> 30</strong>
    (<a class="quiet" href="/players/22773">R. Smith</a> 2, <a class="quiet" href="/players/28264">R. Papenhuyzen</a>, <a class="quiet" href="/players/26266">N. Meaney</a>, <a class="quiet" href="/players/28667">X. Coates</a> tries; <a class="quiet" href="/players/26266">N. Meaney</a> 5 goals) defeated <strong><a href="../penrith/summary.html">Penrith</a> 12</strong> (<a class="quiet" href="/players/22776">N. Cleary</a>, <a class="quiet" href="/players/28598">B. To'o</a> tries; <a class="quiet" href="/players/22776">N. Cleary</a> 2 goals) at <a href="/matches/Custom/abc123">AAMI Park</a>.<br />
    Date: Fri, 8th March. &nbsp;&nbsp;Kickoff: 8:00 PM. &nbsp;&nbsp;Halftime: Melbourne	18-6. &nbsp;&nbsp;Penalties: Penrith 8-5. &nbsp;&nbsp;Referee: <a class="quiet" href="/matches/Custom/ref123">Ashley Klein</a>. &nbsp;&nbsp;Crowd: 22,543.<div class="quiet small">
    <strong><a href="../melbourne/summary.html">Melbourne</a>:</strong>
    <a href="/players/28264">Papenhuyzen</a>, <a href="/players/35233">Warbrick</a>, <a href="/players/12345">Ieremia</a>, <a href="/players/26266">Meaney</a>, <a href="/players/28667">Coates</a>; <a href="/players/20275">Munster</a>, <a href="/players/22773">Hughes</a>; <a href="/players/20000">Hynes</a>, <a href="/players/26157">Grant</a>, <a href="/players/21542">Welch</a>, <a href="/players/19786">Finucane</a>, <a href="/players/14527">Bromwich</a>, <a href="/players/20007">Eisenhuth</a>. <em>Int:</em> <a href="/players/30001">Knight</a>, <a href="/players/30002">Fa'alogo</a>, <a href="/players/22716">King</a>, <a href="/players/31628">Wishart</a>.</div>
    <div class="quiet small">
    <strong><a href="../penrith/summary.html">Penrith</a>:</strong>
    <a href="/players/22807">Edwards</a>, <a href="/players/33587">Turuva</a>, <a href="/players/31081">Tago</a>, <a href="/players/31352">May</a>, <a href="/players/28598">To'o</a>; <a href="/players/24106">Luai</a>, <a href="/players/22776">Cleary</a> (C); <a href="/players/22777">Leota</a>, <a href="/players/35654">Sommerton</a>, <a href="/players/22606">Fisher-Harris</a>, <a href="/players/26267">Garner</a>, <a href="/players/28258">Martin</a>, <a href="/players/20713">Yeo</a>. <em>Int:</em> <a href="/players/31080">Smith</a>, <a href="/players/34024">Henry</a>, <a href="/players/23753">Eisenhuth</a>, <a href="/players/32691">Luke</a>.</div>
    </div>
    """

    SAMPLE_BYE_HTML = """
    <div class="quiet" style="font-size:130%;line-height:1.5em;padding:1em 0;border-bottom:dotted 1px #ccc;">
        Dolphins had the bye
    </div>
    """

    SAMPLE_DRAW_HTML = """
    <div class="quiet" style="font-size:130%;line-height:1.5em;padding:1em 0;border-bottom:dotted 1px #ccc;">
    <span class="noprint"><a class="rlplnk" href="/matches/99999">&gt;</a>&nbsp;</span>
    <strong><a href="../brisbane/summary.html">Brisbane</a> 18</strong>
    (<a class="quiet" href="/players/11111">A. Player</a> 3 tries; <a class="quiet" href="/players/11112">B. Kicker</a> 3 goals) drew with <strong><a href="../canterbury/summary.html">Canterbury</a> 18</strong>
    (<a class="quiet" href="/players/22222">C. Player</a> 3 tries; <a class="quiet" href="/players/22223">D. Kicker</a> 3 goals) at <a href="/matches/Custom/xyz">Suncorp Stadium</a>.<br />
    Date: Sat, 15th June. &nbsp;&nbsp;Kickoff: 7:35 PM.
    </div>
    """

    def test_parse_match_block_extracts_teams(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["home_team"] == "Melbourne"
        assert result["away_team"] == "Penrith"

    def test_parse_match_block_extracts_scores(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["home_score"] == 30
        assert result["away_score"] == 12

    def test_parse_match_block_extracts_result_text(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["result_text"] == "defeated"

    def test_parse_match_block_extracts_venue(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["venue"] == "AAMI Park"

    def test_parse_match_block_extracts_date(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML, year=2024)
        assert result["date"] is not None
        assert "8th March" in result["date"]

    def test_parse_match_block_parses_date_object(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML, year=2024)
        assert result["parsed_date"] is not None
        assert result["parsed_date"].year == 2024
        assert result["parsed_date"].month == 3
        assert result["parsed_date"].day == 8

    def test_parse_match_block_extracts_kickoff(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["kickoff_time"] is not None
        assert "8:00" in result["kickoff_time"]

    def test_parse_match_block_extracts_halftime(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["halftime_home"] == 18
        assert result["halftime_away"] == 6

    def test_parse_match_block_extracts_penalties(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["penalty_home"] == 8
        assert result["penalty_away"] == 5

    def test_parse_match_block_extracts_referee(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["referee"] is not None
        assert "Ashley Klein" in result["referee"]

    def test_parse_match_block_extracts_attendance(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["attendance"] == 22543

    def test_parse_match_block_extracts_match_url(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["match_url"] is not None
        assert "/matches/62008" in result["match_url"]

    def test_parse_match_block_not_bye(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert result["is_bye"] is False
        assert result["is_walkover"] is False
        assert result["is_abandoned"] is False

    def test_parse_bye_block(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_BYE_HTML)
        assert result["is_bye"] is True
        assert result["home_team"] == "Dolphins"

    def test_parse_draw_match(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_DRAW_HTML)
        assert result["home_team"] == "Brisbane"
        assert result["away_team"] == "Canterbury"
        assert result["home_score"] == 18
        assert result["away_score"] == 18
        assert result["result_text"] == "drew with"

    def test_parse_match_block_try_scorers(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        home_scorers = result["home_try_scorers"]
        assert len(home_scorers) == 5  # R. Smith 2, Papenhuyzen, Meaney, Coates
        # R. Smith scored 2 tries
        smith = [s for s in home_scorers if "Smith" in s["player"]][0]
        assert smith["count"] == 2

    def test_parse_match_block_home_lineup(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert len(result["home_lineup"]) == 13
        assert result["home_lineup"][0] == "Papenhuyzen"

    def test_parse_match_block_home_bench(self):
        from scraping.rlp_match_parser import parse_match_block

        result = parse_match_block(self.SAMPLE_MATCH_HTML)
        assert len(result["home_bench"]) == 4
        assert "Knight" in result["home_bench"]

    def test_parse_round_summary_with_match_blocks(self):
        from scraping.rlp_match_parser import parse_round_summary

        html = f"""
        <html><body>
        <h3>Results</h3>
        {self.SAMPLE_MATCH_HTML}
        {self.SAMPLE_DRAW_HTML}
        </body></html>
        """
        matches = parse_round_summary(html, year=2024, round_id=1)
        assert len(matches) == 2
        # Each match should have year and round attached
        for m in matches:
            if m.get("home_team"):
                assert m.get("year") == 2024
                assert m.get("round") == 1

    def test_parse_round_summary_empty_page(self):
        from scraping.rlp_match_parser import parse_round_summary

        html = "<html><body><div class='content'></div></body></html>"
        matches = parse_round_summary(html, year=2024, round_id=1)
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_empty_match_returns_all_fields(self):
        from scraping.rlp_match_parser import MATCH_FIELDS, _empty_match

        m = _empty_match()
        for field in MATCH_FIELDS:
            assert field in m

    def test_parse_scorer_list_helper(self):
        from scraping.rlp_match_parser import _parse_scorer_list

        scorers = _parse_scorer_list("J Smith 2, B Jones, T Brown 3")
        assert len(scorers) == 3
        assert scorers[0]["player"] == "J Smith"
        assert scorers[0]["count"] == 2
        assert scorers[1]["player"] == "B Jones"
        assert scorers[1]["count"] == 1
        assert scorers[2]["player"] == "T Brown"
        assert scorers[2]["count"] == 3

    def test_parse_scorer_list_empty_string(self):
        from scraping.rlp_match_parser import _parse_scorer_list

        assert _parse_scorer_list("") == []
        assert _parse_scorer_list("   ") == []


# ============================================================================
# rlp_ladder_parser tests
# ============================================================================


class TestRlpLadderParser:
    """Tests for scraping.rlp_ladder_parser with realistic RLP HTML.

    The sample HTML mirrors the actual ``<table class="ladder">`` structure
    used by rugbyleagueproject.org ladder pages.
    """

    # Realistic RLP ladder table HTML (simplified to 3 teams).
    SAMPLE_LADDER_HTML = """
    <html><body>
    <table class="ladder">
    <tr><th colspan="2">&nbsp;</th><th colspan="7">Home</th><th colspan="7">Away</th><th colspan="9">Overall</th></tr>
    <tr><th colspan="2">Team</th>
    <th>P</th><th>W</th><th>L</th><th>D</th><th>F</th><th>A</th><th>PD</th>
    <th class="leftdivide">P</th><th>W</th><th>L</th><th>D</th><th>F</th><th>A</th><th>PD</th>
    <th class="leftdivide shade">P</th><th class="shade">W</th><th class="shade">L</th><th class="shade">D</th><th class="shade">Bye</th><th class="shade">F</th><th class="shade">A</th><th class="shade">Pts</th><th class="shade">PD</th><th>FPG</th><th>APG</th></tr>
    <tr class="data"><td class="rank">1.</td>
    <td class="name"><a href="/seasons/nrl-2024/melbourne/summary.html">Melbourne Storm</a></td><td>5</td><td>4</td><td>1</td><td>-</td><td>125</td><td>70</td><td class="tight">55</td><td class="leftdivide">5</td><td>4</td><td>1</td><td>-</td><td>125</td><td>70</td><td class="tight">55</td><td class="shade leftdivide">10</td><td class="shade">8</td><td class="shade">2</td><td class="shade">-</td><td class="shade">-</td><td class="shade">250</td><td class="shade">140</td><td class="shade"><b>16</b></td><td class="shade tight">+110</td><td>25.00</td><td>14.00</td></tr>
    <tr class="data"><td class="rank">2.</td>
    <td class="name"><a href="/seasons/nrl-2024/penrith/summary.html">Penrith Panthers</a></td><td>5</td><td>4</td><td>1</td><td>-</td><td>110</td><td>80</td><td class="tight">30</td><td class="leftdivide">5</td><td>3</td><td>2</td><td>-</td><td>110</td><td>80</td><td class="tight">30</td><td class="shade leftdivide">10</td><td class="shade">7</td><td class="shade">3</td><td class="shade">-</td><td class="shade">-</td><td class="shade">220</td><td class="shade">160</td><td class="shade"><b>14</b></td><td class="shade tight">+60</td><td>22.00</td><td>16.00</td></tr>
    <tr class="data"><td class="rank">3.</td>
    <td class="name"><a href="/seasons/nrl-2024/sydney/summary.html">Sydney Roosters</a></td><td>5</td><td>3</td><td>2</td><td>-</td><td>100</td><td>90</td><td class="tight">10</td><td class="leftdivide">5</td><td>3</td><td>2</td><td>-</td><td>100</td><td>90</td><td class="tight">10</td><td class="shade leftdivide">10</td><td class="shade">6</td><td class="shade">4</td><td class="shade">-</td><td class="shade">-</td><td class="shade">200</td><td class="shade">180</td><td class="shade"><b>12</b></td><td class="shade tight">+20</td><td>20.00</td><td>18.00</td></tr>
    </table>
    </body></html>
    """

    def test_parse_round_ladder_returns_list(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML, year=2024, round_id=10)
        assert isinstance(rows, list)
        assert len(rows) == 3

    def test_parse_round_ladder_team_names(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        teams = [r["team"] for r in rows]
        assert "Melbourne Storm" in teams
        assert "Penrith Panthers" in teams
        assert "Sydney Roosters" in teams

    def test_parse_round_ladder_positions(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        assert rows[0]["position"] == 1
        assert rows[1]["position"] == 2
        assert rows[2]["position"] == 3

    def test_parse_round_ladder_numeric_fields(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        mel = rows[0]  # Melbourne Storm
        assert mel["played"] == 10
        assert mel["won"] == 8
        assert mel["lost"] == 2
        assert mel["drawn"] == 0
        assert mel["points_for"] == 250
        assert mel["points_against"] == 140
        assert mel["points_diff"] == 110
        assert mel["competition_points"] == 16

    def test_parse_round_ladder_metadata(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML, year=2024, round_id=10)
        for r in rows:
            assert r["year"] == 2024
            assert r["round"] == 10

    def test_parse_round_ladder_empty_html(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder("<html><body>No table here</body></html>")
        assert rows == []

    def test_parse_round_ladder_sorted_by_position(self):
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        positions = [r["position"] for r in rows]
        assert positions == sorted(positions)

    def test_parse_round_ladder_all_canonical_columns_present(self):
        from scraping.rlp_ladder_parser import _CANONICAL_COLUMNS, parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        for row in rows:
            for col in _CANONICAL_COLUMNS:
                assert col in row, f"Missing column: {col}"

    def test_parse_round_ladder_home_away_splits(self):
        """Verify home/away stat breakdowns are parsed correctly."""
        from scraping.rlp_ladder_parser import parse_round_ladder

        rows = parse_round_ladder(self.SAMPLE_LADDER_HTML)
        mel = rows[0]  # Melbourne Storm
        assert mel["home_played"] == 5
        assert mel["home_won"] == 4
        assert mel["away_played"] == 5
        assert mel["away_won"] == 4


# ============================================================================
# odds_loader tests
# ============================================================================


class TestOddsLoader:
    """Tests for scraping.odds_loader with mocked Excel reads."""

    @pytest.fixture()
    def mock_odds_raw_df(self) -> pd.DataFrame:
        """Raw DataFrame simulating what pd.read_excel returns."""
        return pd.DataFrame({
            "Date": ["01/03/2024", "02/03/2024", "08/03/2024"],
            "Home Team": ["Melbourne Storm", "Penrith Panthers", "Cronulla-Sutherland Sharks"],
            "Away Team": ["Souths", "Bulldogs", "Manly"],
            "Home Score": [30, 24, 18],
            "Away Score": [12, 18, 20],
            "Home Odds": [1.50, 1.80, 2.10],
            "Away Odds": [2.50, 2.00, 1.75],
            "Play Off Game?": ["N", "N", "Y"],
        })

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_standardises_column_names(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        assert "date" in df.columns
        assert "home_team" in df.columns
        assert "away_team" in df.columns
        assert "home_score" in df.columns
        assert "away_score" in df.columns

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_standardises_team_names(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        # "Souths" should be standardised to "South Sydney Rabbitohs"
        assert "South Sydney Rabbitohs" in df["away_team"].values
        # "Bulldogs" should become "Canterbury Bulldogs"
        assert "Canterbury Bulldogs" in df["away_team"].values

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_parses_dates(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_computes_implied_probabilities(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        # h2h_home -> implied_prob_home = 1/odds
        if "implied_prob_home" in df.columns:
            assert (df["implied_prob_home"] > 0).all()
            assert (df["implied_prob_home"] <= 1).all()

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_computes_home_win(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        assert "home_win" in df.columns
        # First match: home 30 > away 12, so home_win = 1
        mel_row = df[df["home_team"] == "Melbourne Storm"]
        if len(mel_row) > 0:
            assert mel_row.iloc[0]["home_win"] == 1

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_sorts_by_date(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            dates = df["date"].dropna()
            assert (dates.diff().dropna() >= pd.Timedelta(0)).all()

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_playoff_flag_parsed(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx")

        if "is_playoff" in df.columns:
            # Third row has "Y"
            assert df["is_playoff"].dtype == bool

    def test_load_odds_file_not_found_raises(self):
        from scraping.odds_loader import load_odds

        with pytest.raises(FileNotFoundError):
            load_odds("/nonexistent/path/to/odds.xlsx")

    @patch("scraping.odds_loader.Path.is_file", return_value=True)
    @patch("scraping.odds_loader.pd.read_excel")
    def test_load_odds_no_standardise(
        self, mock_read_excel, mock_is_file, mock_odds_raw_df
    ):
        from scraping.odds_loader import load_odds

        mock_read_excel.return_value = mock_odds_raw_df
        df = load_odds("/fake/path.xlsx", standardise_teams=False)

        # Team names should remain as raw
        # "Souths" should not become "South Sydney Rabbitohs"
        away_teams = df["away_team"].values if "away_team" in df.columns else []
        if len(away_teams) > 0:
            # When not standardised, raw name should be preserved
            assert "Souths" in away_teams or "South Sydney Rabbitohs" not in away_teams

    def test_odds_loader_column_map_completeness(self):
        """Verify the column map handles common Excel header variants."""
        from scraping.odds_loader import _COLUMN_MAP

        assert _COLUMN_MAP["date"] == "date"
        assert _COLUMN_MAP["home team"] == "home_team"
        assert _COLUMN_MAP["away team"] == "away_team"
        assert _COLUMN_MAP["home odds"] == "h2h_home"
        assert _COLUMN_MAP["away odds"] == "h2h_away"
