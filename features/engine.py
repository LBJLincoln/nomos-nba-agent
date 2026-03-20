#!/usr/bin/env python3
"""
NBA Quant Feature Engine — 2000+ Features with Genetic Selection
=================================================================
Generates ~2000+ feature candidates across 25 categories, then uses
genetic algorithm to select optimal 150-300 features.

Categories:
  1. ROLLING PERFORMANCE (6 windows × 8 stats × 2 teams = 96)
  2. FOUR FACTORS (8 features × 2 windows × 2 teams = 32)
  3. PACE & EFFICIENCY (12 features × 2 teams = 24)
  4. SCORING PROFILE (10 features × 2 teams = 20)
  5. MOMENTUM & STREAKS (16 features)
  6. REST & SCHEDULE (20 features)
  7. OPPONENT-ADJUSTED (12 features × 2 teams = 24)
  8. MATCHUP & H2H (18 features)
  9. MARKET MICROSTRUCTURE (30+ features — CLV, line movement, steam)
  10. CONTEXT & SITUATIONAL (20 features)
  11. REFEREE FEATURES (10 features — bias, foul rates, tendencies)
  12. PLAYER IMPACT (16 features — star usage, injuries, depth)
  13. QUARTER-LEVEL PATTERNS (14 features — Q1/Q3/Q4 trends)
  14. DEFENSIVE MATCHUP ADVANCED (12 features — paint/perimeter/rim)
  15. POLYMARKET & PREDICTION MARKETS (8 features — market wisdom)
  16. INTERACTION & POLYNOMIAL (200+ features — pairwise, ratios, squared)
  17. ADVANCED ROLLING STATISTICS (160+ features — EWMA, volatility, z-scores)
  18. SEASON TRAJECTORY & CONTEXT (80+ features — pythagorean, playoff pace)
  19. LINEUP & ROTATION ANALYTICS (60+ features — lineup quality, depth)
  20. GAME THEORY & META FEATURES (80+ features — calibration, feedback)
  21. ENVIRONMENTAL & EXTERNAL (40+ features — conference, tanking, revenge)
  22. CROSS-WINDOW MOMENTUM (630 features — trend deltas, acceleration)
  23. ADVANCED MARKET MICROSTRUCTURE II (60+ features — multi-book)
  24. POWER RATING COMPOSITES (60+ features — multi-Elo, RAPTOR)
  25. FATIGUE & LOAD MANAGEMENT (80+ features — cumulative load, degradation)
  ≈ 2000+ feature candidates

Architecture inspired by:
  - Starlizard: 500+ features, genetic selection, real-time adjustment
  - Priomha Capital: 17% annual ROI, market microstructure focus
  - Becker/Kalshi: Maker advantage, longshot bias exploitation
  - Dean Oliver: Four Factors framework
  - NBA Second Spectrum: Player tracking features

THIS SCRIPT MUST RUN ON HF SPACES (16GB RAM) — NOT on VM.
"""

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math

# ── Team mappings ──
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Arena coordinates for travel distance (lat, lon)
ARENA_COORDS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.976),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.187), "MIL": (43.045, -87.917), "MIN": (44.980, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}

# Arena altitudes (feet) — Denver is the key outlier
ARENA_ALTITUDE = {
    "DEN": 5280, "UTA": 4226, "PHX": 1086, "OKC": 1201, "SAS": 650,
    "DAL": 430, "HOU": 43, "MEM": 337, "ATL": 1050, "CHA": 751,
    "IND": 715, "CHI": 594, "MIL": 617, "MIN": 830, "DET": 600,
    "CLE": 653, "BOS": 141, "NYK": 33, "BKN": 33, "PHI": 39,
    "WAS": 25, "MIA": 6, "ORL": 82, "NOP": 7, "TOR": 250,
    "POR": 50, "SAC": 30, "GSW": 12, "LAL": 305, "LAC": 305,
}

# Timezone offsets from ET
TIMEZONE_ET = {
    "ATL": 0, "BOS": 0, "BKN": 0, "CHA": 0, "CHI": -1, "CLE": 0,
    "DAL": -1, "DEN": -2, "DET": 0, "GSW": -3, "HOU": -1, "IND": 0,
    "LAC": -3, "LAL": -3, "MEM": -1, "MIA": 0, "MIL": -1, "MIN": -1,
    "NOP": -1, "NYK": 0, "OKC": -1, "ORL": 0, "PHI": 0, "PHX": -2,
    "POR": -3, "SAC": -3, "SAS": -1, "TOR": 0, "UTA": -2, "WAS": 0,
}

WINDOWS = [3, 5, 7, 10, 15, 20]  # Rolling windows


def resolve(name):
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    if len(name) == 3 and name.isupper():
        return name
    for full, abbr in TEAM_MAP.items():
        if name in full:
            return abbr
    return name[:3].upper() if name else None


def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two coordinates."""
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


class NBAFeatureEngine:
    """
    Generates 2000+ features for each game from historical data.

    Usage:
        engine = NBAFeatureEngine()
        X, y, feature_names = engine.build(games)
        # X.shape = (n_games, ~2000)
    """

    def __init__(self, include_market=True):
        self.include_market = include_market
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Pre-compute all feature names for documentation."""
        names = []

        # 1. ROLLING PERFORMANCE (96 features)
        for prefix in ["h", "a"]:
            for w in WINDOWS:
                names.append(f"{prefix}_wp{w}")       # Win %
                names.append(f"{prefix}_pd{w}")        # Point diff
                names.append(f"{prefix}_ppg{w}")       # Points per game
                names.append(f"{prefix}_papg{w}")      # Points allowed per game
                names.append(f"{prefix}_margin{w}")    # Avg margin
                names.append(f"{prefix}_close{w}")     # Close game % (margin <= 5)
                names.append(f"{prefix}_blowout{w}")   # Blowout % (margin >= 15)
                names.append(f"{prefix}_ou_avg{w}")    # Over/under average

        # 2. FOUR FACTORS (Dean Oliver) — per window (64 features)
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                names.append(f"{prefix}_efg{w}")       # Effective FG%
                names.append(f"{prefix}_tov_rate{w}")   # Turnover rate
                names.append(f"{prefix}_orb_rate{w}")   # Offensive rebound rate
                names.append(f"{prefix}_ft_rate{w}")    # Free throw rate
                names.append(f"{prefix}_opp_efg{w}")    # Opponent eFG%
                names.append(f"{prefix}_opp_tov{w}")    # Opponent TOV rate
                names.append(f"{prefix}_opp_orb{w}")    # Opponent ORB rate
                names.append(f"{prefix}_opp_ft{w}")     # Opponent FT rate

        # 3. PACE & EFFICIENCY (48 features)
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                names.append(f"{prefix}_ortg{w}")      # Offensive rating (pts/100 poss)
                names.append(f"{prefix}_drtg{w}")      # Defensive rating
                names.append(f"{prefix}_netrtg{w}")    # Net rating
                names.append(f"{prefix}_pace{w}")      # Pace (possessions/48min)
                names.append(f"{prefix}_ts{w}")        # True shooting %
                names.append(f"{prefix}_poss{w}")      # Avg possessions
                names.append(f"{prefix}_ast_rate{w}")  # Assist rate
                names.append(f"{prefix}_stl_rate{w}")  # Steal rate
                names.append(f"{prefix}_blk_rate{w}")  # Block rate
                names.append(f"{prefix}_tov_pct{w}")   # Turnover %
                names.append(f"{prefix}_oreb_pct{w}")  # Off rebound %
                names.append(f"{prefix}_dreb_pct{w}")  # Def rebound %

        # 4. SCORING PROFILE (40 features)
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                names.append(f"{prefix}_3par{w}")      # 3-point attempt rate
                names.append(f"{prefix}_3p_pct{w}")    # 3-point %
                names.append(f"{prefix}_2p_pct{w}")    # 2-point %
                names.append(f"{prefix}_ft_pct{w}")    # Free throw %
                names.append(f"{prefix}_paint_pts{w}") # Paint points avg
                names.append(f"{prefix}_fb_pts{w}")    # Fast break points avg
                names.append(f"{prefix}_bench_pts{w}") # Bench points avg
                names.append(f"{prefix}_2nd_pts{w}")   # Second chance points
                names.append(f"{prefix}_pitp{w}")      # Points in the paint
                names.append(f"{prefix}_pts_off_tov{w}") # Points off turnovers

        # 5. MOMENTUM & STREAKS (32 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_streak")            # Current W/L streak
            names.append(f"{prefix}_streak_abs")        # Absolute streak length
            names.append(f"{prefix}_last5_vs_season")   # Last 5 wp - season wp (form delta)
            names.append(f"{prefix}_last3_vs_last10")   # Short vs medium term momentum
            names.append(f"{prefix}_home_wp")           # Home record
            names.append(f"{prefix}_away_wp")           # Away record
            names.append(f"{prefix}_ha_split")          # Home-away split
            names.append(f"{prefix}_ats_wp5")           # Against the spread win % (last 5)
            names.append(f"{prefix}_ou_record5")        # Over/under record (last 5)
            names.append(f"{prefix}_scoring_trend")     # PPG trend (last 5 vs last 20)
            names.append(f"{prefix}_defense_trend")     # PAPG trend
            names.append(f"{prefix}_clutch_wp")         # Win % in games decided by <= 5
            names.append(f"{prefix}_blowout_rate")      # % of games decided by 15+
            names.append(f"{prefix}_comeback_rate")      # % trailing at half but winning
            names.append(f"{prefix}_consistency")       # StdDev of point diff (lower = more consistent)
            names.append(f"{prefix}_recent_margin_std") # Variance in recent margins

        # 6. REST & SCHEDULE (24 features)
        names.extend([
            "h_rest_days", "a_rest_days",
            "rest_advantage",                       # h_rest - a_rest
            "h_b2b", "a_b2b",                      # Back-to-back (0/1)
            "h_3in4", "a_3in4",                    # 3 games in 4 days
            "h_4in6", "a_4in6",                    # 4 games in 6 days
            "h_travel_dist", "a_travel_dist",      # Miles traveled (last game → this)
            "travel_advantage",                     # h_travel - a_travel (negative = home rested)
            "h_altitude", "a_altitude",            # Arena altitude
            "altitude_delta",                       # Home altitude - away altitude (DEN advantage)
            "h_tz_shift", "a_tz_shift",            # Timezone hours shifted
            "tz_advantage",                         # Timezone advantage
            "h_games_7d", "a_games_7d",            # Games played in last 7 days
            "h_miles_7d", "a_miles_7d",            # Total miles in last 7 days
            "schedule_density_diff",                # h_games_7d - a_games_7d
            "combined_fatigue",                     # Composite fatigue score
        ])

        # 7. OPPONENT-ADJUSTED (24 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_sos5")             # Strength of schedule (last 5)
            names.append(f"{prefix}_sos10")            # SOS (last 10)
            names.append(f"{prefix}_sos_season")       # Season SOS
            names.append(f"{prefix}_wp_vs_above500")   # Win% vs winning teams
            names.append(f"{prefix}_wp_vs_below500")   # Win% vs losing teams
            names.append(f"{prefix}_wp_vs_top10")      # Win% vs top 10
            names.append(f"{prefix}_wp_vs_bot10")      # Win% vs bottom 10
            names.append(f"{prefix}_pd_vs_top10")      # Point diff vs top 10
            names.append(f"{prefix}_pd_vs_bot10")      # Point diff vs bottom 10
            names.append(f"{prefix}_opp_avg_ortg")     # Avg opponent ORtg (last 10)
            names.append(f"{prefix}_opp_avg_drtg")     # Avg opponent DRtg (last 10)
            names.append(f"{prefix}_margin_vs_quality") # Margin corr with opp quality

        # 8. MATCHUP & HEAD-TO-HEAD (20 features)
        names.extend([
            "h2h_wp",                              # H2H win % (last 3 seasons)
            "h2h_last3_wp",                        # H2H last 3 meetings
            "h2h_avg_margin",                      # Average margin in H2H
            "h2h_home_wp",                         # H2H home team win %
            "pace_delta",                          # Home pace - Away pace (style clash)
            "style_mismatch",                      # ORtg vs opponent DRtg gap
            "3pt_matchup",                         # Home 3P% vs Away opp 3P%
            "paint_matchup",                       # Home paint pts vs Away opp paint pts
            "tempo_mismatch",                      # Absolute pace difference
            "defensive_matchup",                   # How well defenses match up
            "rebound_edge",                        # Total rebound rate diff
            "turnover_edge",                       # Turnover rate diff
            "free_throw_edge",                     # FT rate diff
            "bench_depth_diff",                    # Bench scoring diff
            "consistency_matchup",                 # Lower variance team
            "fatigue_adjusted_rating",             # NetRtg adjusted for rest/travel
            "elo_home", "elo_away",               # Elo ratings
            "elo_diff",                           # Elo difference + home court
            "elo_recent_change",                   # Elo change last 10 games
        ])

        # 9. MARKET MICROSTRUCTURE (32 features)
        if self.include_market:
            names.extend([
                "opening_spread", "current_spread",
                "spread_movement",                     # current - opening (line movement)
                "spread_movement_abs",                 # Absolute movement
                "reverse_line_movement",               # 1 if line moved opposite to public
                "opening_total", "current_total",
                "total_movement",
                "opening_ml_home", "current_ml_home",
                "ml_movement",                         # Moneyline movement
                "implied_prob_home", "implied_prob_away",
                "model_vs_market",                     # Our model prob - market implied prob
                "edge_magnitude",                      # Absolute value of edge
                "books_disagreement",                  # Max - Min implied prob across books
                "sharp_line",                          # Pinnacle-implied probability
                "public_pct_home",                     # % of public bets on home
                "public_money_pct_home",               # % of money on home
                "smart_money_indicator",               # Money% - Bet% divergence
                "steam_move",                          # 1 if sharp steam move detected
                "clv_recent_avg",                      # Avg CLV of our recent bets
                "market_efficiency",                   # How much line has moved (settled)
                "opening_overround",                   # Total implied prob > 1
                "best_available_odds_home",            # Best odds across books
                "best_available_odds_away",
                "odds_range_home",                     # Max - min odds (disagreement)
                "odds_range_away",
                "time_to_close",                       # Hours until game
                "late_money_direction",                # Direction of late sharp money
                "closing_line_estimate",               # Estimated closing line
                "historical_clv_vs_book",              # Our historical CLV vs this book
                "longshot_flag",                       # 1 if any side > 5.0 odds
            ])

        # 10. CONTEXT & SITUATIONAL (24 features)
        names.extend([
            "home_court_adv",                      # 1.0 always (baseline)
            "season_phase",                        # 0-1 (early → late)
            "month_sin", "month_cos",              # Cyclical month encoding
            "day_of_week",                         # 0-6 (encoded)
            "is_weekend",                          # 1 if Sat/Sun
            "is_national_tv",                      # 1 if ESPN/TNT (higher motivation)
            "h_games_played", "a_games_played",    # Season games played
            "h_season_pct", "a_season_pct",        # % of season completed
            "playoff_race",                        # 1 if both teams in playoff contention
            "tanking_flag",                        # 1 if either team eliminated
            "rivalry",                             # 1 if divisional rivalry
            "h_division", "a_division",            # Division encoding (0-5)
            "same_division",                       # 1 if same division
            "same_conference",                     # 1 if same conference
            "conference_game",                     # 1 if cross-conference
            "power_rank_diff",                     # Power ranking difference
            "vegas_home_fav",                      # 1 if home is favored
            "combined_record",                     # Combined win% (quality indicator)
            "game_importance_score",               # Playoff implications
            "total_expected",                      # Expected total points
        ])

        # 11. REFEREE FEATURES (10 features) — NEW 2026
        names.extend([
            "ref_home_foul_bias",              # Avg (home_fouls - away_fouls) for this crew
            "ref_total_fouls_avg",             # Avg total fouls called per game
            "ref_foul_rate_vs_league",         # This crew's foul rate vs league avg
            "ref_home_ft_advantage",           # Avg FTA differential (home-away) for crew
            "ref_experience_games",            # Total games officiated this season
            "ref_over_tendency",               # % of games going over total for crew
            "ref_close_game_bias",             # Home win % in close games for crew
            "ref_tech_foul_rate",              # Technical fouls per game for crew
            "ref_home_win_rate",               # Home team win % with this crew
            "ref_pace_impact",                 # Avg pace delta vs league avg for crew
        ])

        # 12. PLAYER IMPACT FEATURES (16 features) — NEW 2026
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_star_usage_rate")      # Top 2 players usage rate combined
            names.append(f"{prefix}_star_minutes_load")    # Top 2 players avg minutes last 5
            names.append(f"{prefix}_injury_impact_score")  # Weighted injury severity (0-1)
            names.append(f"{prefix}_injured_war_lost")     # WAR of injured players
            names.append(f"{prefix}_lineup_continuity")    # % same starting lineup last 5
            names.append(f"{prefix}_bench_depth_rating")   # Bench net rating last 10
            names.append(f"{prefix}_star_rest_status")     # 1 if star on B2B, 0.5 if 1 rest day
            names.append(f"{prefix}_rotation_depth")       # Number of players with 10+ min

        # 13. QUARTER-LEVEL PATTERNS (14 features) — NEW 2026
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_q1_margin_avg")        # Avg Q1 margin (last 10)
            names.append(f"{prefix}_q3_margin_avg")        # Avg Q3 margin (comeback indicator)
            names.append(f"{prefix}_q4_clutch_netrtg")     # Net rating last 5 min, close games
            names.append(f"{prefix}_half_adjustment")      # Q3 performance vs Q1 (coaching adj)
            names.append(f"{prefix}_comeback_win_pct")     # Win% when trailing after Q3
            names.append(f"{prefix}_blowout_hold_pct")     # % holding 10+ pt leads
            names.append(f"{prefix}_garbage_time_margin")  # Avg margin change in Q4 blowouts

        # 14. DEFENSIVE MATCHUP ADVANCED (12 features) — NEW 2026
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_paint_defense_rating")  # Opp points in paint allowed
            names.append(f"{prefix}_perimeter_defense")     # Opp 3pt% allowed
            names.append(f"{prefix}_transition_defense")    # Opp fast break pts allowed
            names.append(f"{prefix}_shot_contest_rate")     # % of shots contested
            names.append(f"{prefix}_deflections_per_game")  # Deflections avg
            names.append(f"{prefix}_rim_protection_rate")   # FG% allowed at rim

        # 15. POLYMARKET & PREDICTION MARKET (8 features) — NEW 2026
        names.extend([
            "polymarket_home_prob",             # Polymarket implied probability
            "polymarket_volume",               # Trading volume (confidence indicator)
            "polymarket_line_movement",        # Movement in last 6 hours
            "polymarket_vs_books",             # Polymarket prob - books prob (divergence)
            "prediction_market_consensus",     # Avg of multiple prediction markets
            "market_wisdom_confidence",        # How much markets agree (1 - std)
            "smart_vs_public_divergence",      # Sharp money vs public bets
            "closing_line_value_history",      # Our historical CLV performance
        ])

        # =====================================================================
        # CATEGORIES 16-25: ADVANCED FEATURE EXPANSION (1400+ new features)
        # These features are registered here for genetic selection.
        # Computation is handled in build_v2() (separate step).
        # =====================================================================

        # 16. INTERACTION & POLYNOMIAL FEATURES (210 features)
        # Pairwise interactions between key stats
        inter_pairs = [
            ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"), ("h_wp3", "a_wp3"),
            ("h_ortg10", "a_drtg10"), ("h_ortg5", "a_drtg5"),
            ("h_drtg10", "a_ortg10"), ("h_drtg5", "a_ortg5"),
            ("h_netrtg10", "a_netrtg10"), ("h_netrtg5", "a_netrtg5"),
            ("h_pace10", "a_pace10"), ("h_pace5", "a_pace5"),
            ("h_ppg10", "a_ppg10"), ("h_ppg5", "a_ppg5"),
            ("h_margin10", "a_margin10"), ("h_margin5", "a_margin5"),
            ("h_efg10", "a_efg10"), ("h_efg5", "a_efg5"),
            ("h_ts10", "a_ts10"), ("h_ts5", "a_ts5"),
            ("h_tov_rate10", "a_tov_rate10"), ("h_tov_rate5", "a_tov_rate5"),
            ("h_orb_rate10", "a_orb_rate10"), ("h_orb_rate5", "a_orb_rate5"),
            ("h_3p_pct10", "a_3p_pct10"), ("h_3p_pct5", "a_3p_pct5"),
            ("h_ft_rate10", "a_ft_rate10"), ("h_ft_rate5", "a_ft_rate5"),
            ("h_pd10", "a_pd10"), ("h_pd5", "a_pd5"),
            ("h_blowout10", "a_blowout10"), ("h_close10", "a_close10"),
            ("h_wp10", "elo_diff"), ("a_wp10", "elo_diff"),
            ("h_ortg10", "elo_diff"), ("h_streak", "a_streak"),
            ("h_wp10", "h_rest_days"), ("a_wp10", "a_rest_days"),
            ("h_netrtg10", "rest_advantage"), ("h_ortg10", "h_pace10"),
            ("a_ortg10", "a_pace10"), ("h_drtg10", "a_3p_pct10"),
            ("h_wp10", "h_sos10"), ("a_wp10", "a_sos10"),
            ("h_efg10", "h_pace10"), ("a_efg10", "a_pace10"),
            ("h_margin10", "h_consistency"), ("a_margin10", "a_consistency"),
            ("h_ppg10", "a_papg10"), ("a_ppg10", "h_papg10"),
            ("h_bench_pts10", "a_bench_pts10"),
            ("h_fb_pts10", "a_fb_pts10"),
            ("h_opp_efg10", "a_efg10"), ("a_opp_efg10", "h_efg10"),
            # Additional interaction pairs for higher coverage
            ("h_wp20", "a_wp20"), ("h_wp15", "a_wp15"),
            ("h_ortg10", "h_efg10"), ("a_ortg10", "a_efg10"),
            ("h_drtg10", "h_opp_efg10"), ("a_drtg10", "a_opp_efg10"),
            ("h_pace10", "h_3p_pct10"), ("a_pace10", "a_3p_pct10"),
            ("h_ast_rate10", "a_tov_pct10"), ("a_ast_rate10", "h_tov_pct10"),
            ("h_stl_rate10", "a_tov_pct10"), ("a_stl_rate10", "h_tov_pct10"),
            ("h_blk_rate10", "a_paint_pts10"), ("a_blk_rate10", "h_paint_pts10"),
            ("h_oreb_pct10", "a_dreb_pct10"), ("a_oreb_pct10", "h_dreb_pct10"),
            ("h_3par10", "a_perimeter_defense"), ("a_3par10", "h_perimeter_defense"),
            ("h_wp10", "a_consistency"), ("a_wp10", "h_consistency"),
            ("h_margin10", "elo_diff"), ("a_margin10", "elo_diff"),
            ("h_ppg10", "h_pace10"), ("a_ppg10", "a_pace10"),
            ("h_papg10", "h_drtg10"), ("a_papg10", "a_drtg10"),
            ("h_wp10", "h_home_wp"), ("a_wp10", "a_away_wp"),
            ("h_streak", "h_wp10"), ("a_streak", "a_wp10"),
            ("h_clutch_wp", "a_clutch_wp"), ("h_comeback_rate", "a_comeback_rate"),
            ("h_scoring_trend", "a_defense_trend"),
            ("a_scoring_trend", "h_defense_trend"),
            ("h_ou_avg10", "a_ou_avg10"),
            ("h_netrtg10", "h_consistency"), ("a_netrtg10", "a_consistency"),
            ("h_efg10", "a_opp_efg10"), ("a_efg10", "h_opp_efg10"),
            ("h_ts10", "h_3p_pct10"), ("a_ts10", "a_3p_pct10"),
            ("h_ft_rate10", "h_ft_pct10"), ("a_ft_rate10", "a_ft_pct10"),
            ("h_pts_off_tov10", "a_tov_rate10"), ("a_pts_off_tov10", "h_tov_rate10"),
            ("h_2nd_pts10", "h_oreb_pct10"), ("a_2nd_pts10", "a_oreb_pct10"),
            ("h_wp10", "season_phase"), ("a_wp10", "season_phase"),
            ("elo_diff", "rest_advantage"), ("elo_diff", "travel_advantage"),
            ("current_spread", "h_wp10"), ("current_spread", "a_wp10"),
            ("current_spread", "elo_diff"), ("h_sos10", "a_sos10"),
            ("h_wp_vs_above500", "a_wp_vs_above500"),
            ("h_wp_vs_top10", "a_wp_vs_top10"),
        ]
        for x, y_feat in inter_pairs:
            names.append(f"inter_{x}_{y_feat}")

        # Ratio features (30 features)
        ratio_pairs = [
            ("h_ortg10", "a_drtg10"), ("h_ortg5", "a_drtg5"),
            ("a_ortg10", "h_drtg10"), ("a_ortg5", "h_drtg5"),
            ("h_pace10", "a_pace10"), ("h_pace5", "a_pace5"),
            ("h_efg10", "a_efg10"), ("h_efg5", "a_efg5"),
            ("h_ts10", "a_ts10"), ("h_ts5", "a_ts5"),
            ("h_ppg10", "a_ppg10"), ("h_ppg5", "a_ppg5"),
            ("h_margin10", "a_margin10"), ("h_margin5", "a_margin5"),
            ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"),
            ("h_3p_pct10", "a_3p_pct10"), ("h_3p_pct5", "a_3p_pct5"),
            ("h_ft_rate10", "a_ft_rate10"), ("h_ft_rate5", "a_ft_rate5"),
            ("h_orb_rate10", "a_orb_rate10"), ("h_orb_rate5", "a_orb_rate5"),
            ("h_tov_rate10", "a_tov_rate10"), ("h_tov_rate5", "a_tov_rate5"),
            ("h_bench_pts10", "a_bench_pts10"), ("h_bench_pts5", "a_bench_pts5"),
            ("h_papg10", "a_papg10"), ("h_papg5", "a_papg5"),
            ("h_opp_efg10", "a_opp_efg10"), ("h_opp_efg5", "a_opp_efg5"),
        ]
        for x, y_feat in ratio_pairs:
            names.append(f"ratio_{x}_{y_feat}")

        # Squared terms (30 features)
        sq_features = [
            "h_wp10", "a_wp10", "h_wp5", "a_wp5",
            "h_ortg10", "a_ortg10", "h_drtg10", "a_drtg10",
            "h_netrtg10", "a_netrtg10", "h_margin10", "a_margin10",
            "elo_diff", "spread_movement", "current_spread",
            "h_pace10", "a_pace10", "h_efg10", "a_efg10",
            "h_ppg10", "a_ppg10", "h_ts10", "a_ts10",
            "rest_advantage", "travel_advantage",
            "h_streak", "a_streak", "h_sos10", "a_sos10",
            "h_consistency", "a_consistency",
        ]
        for feat in sq_features:
            names.append(f"sq_{feat}")

        # Trend delta features: short_window - long_window (100 features)
        trend_stats = ["wp", "ppg", "margin", "ortg", "drtg",
                       "efg", "ts", "pace", "pd", "papg"]
        trend_window_pairs = [
            (3, 7), (3, 10), (3, 15), (3, 20),
            (5, 10), (5, 15), (5, 20),
            (7, 15), (7, 20), (10, 20),
        ]
        for prefix in ["h", "a"]:
            for stat in trend_stats:
                for w1, w2 in trend_window_pairs:
                    names.append(f"trend_{prefix}_{stat}_w{w1}_w{w2}")

        # 17. ADVANCED ROLLING STATISTICS (168 features)
        # EWMA features: 3 alphas × 7 stats × 2 teams = 42
        ewma_stats = ["ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
        ewma_alphas = ["01", "03", "05"]
        for prefix in ["h", "a"]:
            for stat in ewma_stats:
                for alpha in ewma_alphas:
                    names.append(f"{prefix}_ewma_{stat}_a{alpha}")

        # Rolling volatility (std dev): 3 windows × 6 stats × 2 teams = 36
        vol_stats = ["margin", "ppg", "papg", "ortg", "drtg", "pace"]
        vol_windows = [5, 10, 20]
        for prefix in ["h", "a"]:
            for stat in vol_stats:
                for w in vol_windows:
                    names.append(f"{prefix}_vol_{stat}_{w}")

        # Rolling min/max: 2 windows × 4 stats × 2 (min/max) × 2 teams = 32
        minmax_stats = ["ppg", "papg", "margin", "ortg"]
        minmax_windows = [5, 10]
        for prefix in ["h", "a"]:
            for stat in minmax_stats:
                for w in minmax_windows:
                    names.append(f"{prefix}_min_{stat}_{w}")
                    names.append(f"{prefix}_max_{stat}_{w}")

        # Z-scores relative to season avg: 10 stats × 2 teams = 20
        zscore_stats = ["ppg", "papg", "margin", "ortg", "drtg",
                        "efg", "ts", "pace", "3p_pct", "ft_rate"]
        for prefix in ["h", "a"]:
            for stat in zscore_stats:
                names.append(f"{prefix}_zscore_{stat}")

        # Rolling skew and kurtosis: 4 stats × 2 teams × 2 = 16
        for prefix in ["h", "a"]:
            for stat in ["margin", "ppg", "ortg", "drtg"]:
                names.append(f"{prefix}_skew_{stat}_10")
                names.append(f"{prefix}_kurtosis_{stat}_10")

        # Range (max - min): 4 stats × 2 windows × 2 teams = 16
        for prefix in ["h", "a"]:
            for stat in ["ppg", "margin", "ortg", "drtg"]:
                for w in [5, 10]:
                    names.append(f"{prefix}_range_{stat}_{w}")

        # Coefficient of variation: 3 stats × 2 teams = 6
        for prefix in ["h", "a"]:
            for stat in ["ppg", "margin", "ortg"]:
                names.append(f"{prefix}_cv_{stat}_10")

        # 18. SEASON TRAJECTORY & CONTEXT (84 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_pyth_wp")                  # Pythagorean win expectation
            names.append(f"{prefix}_pyth_vs_actual")           # Pythagorean - actual (luck measure)
            names.append(f"{prefix}_win_pace_82")              # Projected wins over 82 games
            names.append(f"{prefix}_playoff_pace_delta")       # Win pace vs playoff threshold (42 wins)
            names.append(f"{prefix}_games_behind_1st")         # Games behind conf leader
            names.append(f"{prefix}_games_behind_8th")         # Games behind 8th seed (playoff line)
            names.append(f"{prefix}_games_ahead_lottery")      # Games ahead of lottery (worst record)
            names.append(f"{prefix}_sos_remaining")            # Strength of remaining schedule
            names.append(f"{prefix}_conf_rank")                # Conference ranking
            names.append(f"{prefix}_div_rank")                 # Division ranking
            names.append(f"{prefix}_is_playoff_team")          # 1 if currently in top 10
            names.append(f"{prefix}_playin_range")             # 1 if 7th-12th seed
            names.append(f"{prefix}_pre_allstar_wp")           # Win% before All-Star break
            names.append(f"{prefix}_post_allstar_wp")          # Win% after All-Star break
            names.append(f"{prefix}_allstar_delta")            # Post - pre All-Star performance
            names.append(f"{prefix}_pre_deadline_wp")          # Win% before trade deadline
            names.append(f"{prefix}_post_deadline_wp")         # Win% after trade deadline
            names.append(f"{prefix}_deadline_delta")           # Post - pre deadline performance
            names.append(f"{prefix}_monthly_wp_trend")         # Month-over-month win% change
            names.append(f"{prefix}_monthly_ortg_trend")       # Month-over-month ORtg change
            names.append(f"{prefix}_monthly_drtg_trend")       # Month-over-month DRtg change
            names.append(f"{prefix}_season_half_improvement")  # 2nd half vs 1st half win%
            names.append(f"{prefix}_regression_indicator")     # How far from league avg (mean reversion)
            names.append(f"{prefix}_hot_cold_regime")          # 1=hot, 0=neutral, -1=cold (last 15 games)
            names.append(f"{prefix}_clinch_status")            # 0=eliminated, 1=alive, 2=clinched
            names.append(f"{prefix}_games_remaining")          # Games left in regular season
            names.append(f"{prefix}_wp_last30")                # Win% last 30 games
            names.append(f"{prefix}_wp_last30_vs_season")      # Last 30 - season (late form)
            names.append(f"{prefix}_home_road_trend_5")        # Recent home/road split trend
            names.append(f"{prefix}_scoring_variance_trend")   # Is scoring becoming more consistent?
            names.append(f"{prefix}_first_half_margin_avg")    # Avg 1st half margin last 10
            names.append(f"{prefix}_second_half_margin_avg")   # Avg 2nd half margin last 10
            names.append(f"{prefix}_half_margin_delta")        # 2nd half - 1st half margin trend
            names.append(f"{prefix}_record_vs_spread")         # ATS record (season)
            names.append(f"{prefix}_ats_trend_10")             # ATS trend last 10
            names.append(f"{prefix}_over_rate_season")         # % overs this season
            names.append(f"{prefix}_pt_diff_close_vs_all")     # Point diff in close games vs all
            names.append(f"{prefix}_record_after_loss")        # Win% after a loss (resilience)
            names.append(f"{prefix}_record_after_win")         # Win% after a win (consistency)
            names.append(f"{prefix}_record_after_b2b")         # Win% day after B2B
            names.append(f"{prefix}_blowout_bounce_back")      # Win% after losing by 15+
            names.append(f"{prefix}_overtime_record")          # Win% in OT games
        # game-level trajectory
        names.extend([
            "trajectory_wp_diff",                # h_wp_last30 - a_wp_last30
            "trajectory_pyth_diff",              # h_pyth - a_pyth
        ])

        # 19. LINEUP & ROTATION ANALYTICS (64 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_starting5_netrtg")         # Estimated starting 5 net rating
            names.append(f"{prefix}_starting5_ortg")           # Starting 5 offensive rating
            names.append(f"{prefix}_starting5_drtg")           # Starting 5 defensive rating
            names.append(f"{prefix}_bench_netrtg")             # Bench unit net rating
            names.append(f"{prefix}_bench_ortg")               # Bench unit offensive rating
            names.append(f"{prefix}_bench_drtg")               # Bench unit defensive rating
            names.append(f"{prefix}_starter_bench_gap")        # Starter netrtg - bench netrtg
            names.append(f"{prefix}_minutes_entropy")          # Entropy of minutes distribution
            names.append(f"{prefix}_minutes_gini")             # Gini coefficient of minutes
            names.append(f"{prefix}_top_scorer_dependency")    # Top scorer pts / team pts
            names.append(f"{prefix}_top2_scorer_dependency")   # Top 2 scorers pts / team pts
            names.append(f"{prefix}_top_scorer_minutes")       # Top scorer avg minutes
            names.append(f"{prefix}_key_player_availability")  # Weighted availability of top 5
            names.append(f"{prefix}_key_player_impact")        # RPM of available key players
            names.append(f"{prefix}_lineup_stability_10")      # Lineup stability last 10
            names.append(f"{prefix}_lineup_stability_5")       # Lineup stability last 5
            names.append(f"{prefix}_rotation_size")            # Players with 15+ min avg
            names.append(f"{prefix}_two_man_combo_best")       # Best 2-man combo net rating
            names.append(f"{prefix}_two_man_combo_worst")      # Worst 2-man combo net rating
            names.append(f"{prefix}_three_pt_shooters_count")  # Players shooting > 35% from 3
            names.append(f"{prefix}_rim_protector_rating")     # Best rim protector impact
            names.append(f"{prefix}_playmaker_rating")         # Best playmaker assist rate
            names.append(f"{prefix}_defensive_versatility")    # Positions that can switch
            names.append(f"{prefix}_size_advantage")           # Avg height/weight vs league
            names.append(f"{prefix}_speed_advantage")          # Pace proxy from lineup composition
            names.append(f"{prefix}_experience_avg")           # Avg years in NBA for rotation
            names.append(f"{prefix}_age_avg")                  # Avg age of rotation players
            names.append(f"{prefix}_youth_factor")             # % of minutes to players < 24 yrs
            names.append(f"{prefix}_veteran_factor")           # % of minutes to players > 30 yrs
            names.append(f"{prefix}_clutch_player_rating")     # Best closer rating
            names.append(f"{prefix}_injury_adjusted_depth")    # Effective depth with injuries
            names.append(f"{prefix}_recent_lineup_change")     # 1 if lineup changed in last 3
        # matchup-level lineup features
        names.extend([
            "lineup_netrtg_diff",                # h_starting5_netrtg - a_starting5_netrtg
            "bench_quality_diff",                # h_bench_netrtg - a_bench_netrtg
        ])

        # 20. GAME THEORY & META FEATURES (82 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_model_accuracy_10")        # Our model accuracy for this team (last 10)
            names.append(f"{prefix}_model_accuracy_30")        # Our model accuracy for this team (last 30)
            names.append(f"{prefix}_model_calibration_bias")   # Over/under prediction tendency
            names.append(f"{prefix}_model_avg_edge")           # Avg edge when predicting this team
            names.append(f"{prefix}_model_roi_this_team")      # ROI on bets involving this team
            names.append(f"{prefix}_feat_importance_rank")     # How feature-important this team is
            names.append(f"{prefix}_prediction_confidence")    # Avg model confidence for this team
            names.append(f"{prefix}_upset_rate_as_fav")        # % upsets when favored
            names.append(f"{prefix}_upset_rate_as_dog")        # % upsets when underdog
            names.append(f"{prefix}_market_overreaction")      # Market moves too much after wins/losses
            names.append(f"{prefix}_public_bias_team")         # How much public over/undervalues
            names.append(f"{prefix}_contrarian_value")         # Inverse of public betting %
            names.append(f"{prefix}_line_sensitivity")         # How much line moves with news
            names.append(f"{prefix}_steam_target_freq")        # How often targeted by steam moves
            names.append(f"{prefix}_sharp_favorite")           # How often sharps bet this team
            names.append(f"{prefix}_model_disagreement")       # Our prob vs market prob (team-specific)
        # game-level meta features
        names.extend([
            "meta_model_vs_market_abs",          # Absolute model-market divergence
            "meta_model_vs_market_direction",    # Sign of model-market divergence
            "meta_model_confidence",             # How confident our model is (max prob)
            "meta_market_confidence",            # How confident market is (implied prob range)
            "meta_consensus_strength",           # Agreement across multiple model variants
            "meta_historical_matchup_accuracy",  # Our past accuracy on this matchup type
            "meta_bankroll_adjusted_edge",       # Edge × Kelly fraction
            "meta_risk_adjusted_value",          # Edge / volatility of this bet type
            "meta_opp_strategy_fast",            # Opponent likely pace (fast indicator)
            "meta_opp_strategy_slow",            # Opponent likely pace (slow indicator)
            "meta_opp_strategy_defensive",       # Opponent defensive game plan indicator
            "meta_opp_strategy_three_heavy",     # Opponent 3pt heavy game plan
            "meta_game_type_cluster",            # Cluster ID for similar historical games
            "meta_cluster_home_win_rate",        # Historical home win% in this cluster
            "meta_recent_model_drift",           # How much model predictions have shifted
            "meta_feature_regime",               # Which feature set is currently predictive
            "meta_market_regime",                # Current market efficiency regime
            "meta_total_edge_composite",         # Weighted sum of all edge signals
        ])

        # 21. ENVIRONMENTAL & EXTERNAL (44 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_conf_standing_pct")        # Conf standing percentile
            names.append(f"{prefix}_div_standing_pct")         # Division standing percentile
            names.append(f"{prefix}_lottery_odds_proxy")       # Tanking incentive (worse record = higher)
            names.append(f"{prefix}_tank_indicator")           # 1 if bottom 5, eliminated
            names.append(f"{prefix}_playoff_urgency")          # How much each game matters for playoff
            names.append(f"{prefix}_revenge_game")             # 1 if lost badly (15+) to this opp last meeting
            names.append(f"{prefix}_revenge_intensity")        # Margin of last loss to this opponent
            names.append(f"{prefix}_coach_exp_years")          # Coach experience in years
            names.append(f"{prefix}_coach_win_rate")           # Coach career win%
            names.append(f"{prefix}_coach_playoff_rate")       # Coach playoff appearance rate
            names.append(f"{prefix}_coach_vs_opp_coach")       # H2H record coach vs opposing coach
            names.append(f"{prefix}_coach_adjustment_rating")  # In-game adjustment quality proxy
            names.append(f"{prefix}_public_team_popularity")   # Market popularity (bet volume proxy)
            names.append(f"{prefix}_media_attention")          # National TV game frequency
            names.append(f"{prefix}_roster_turnover")          # % new players this season
            names.append(f"{prefix}_chemistry_index")          # Games together for starting 5
        # game-level environmental
        names.extend([
            "conf_standing_diff",                # h_conf_standing - a_conf_standing
            "div_standing_diff",                 # Division standing differential
            "both_playoff_contenders",           # 1 if both in playoff race
            "both_tanking",                      # 1 if both eliminated
            "upset_potential",                   # Dog win probability from features
            "rivalry_intensity",                 # Combined history + division + stakes
            "public_side_home",                  # 1 if public heavily on home
            "contrarian_signal",                 # 1 if going against public is +EV
            "weather_travel_factor",             # Season-based weather impact on travel
            "altitude_fatigue_compound",         # altitude_delta × travel_dist interaction
            "timezone_circadian_impact",         # Timezone shift × game time interaction
            "arena_noise_factor",               # Home court quality proxy (attendance %)
        ])

        # 22. CROSS-WINDOW MOMENTUM (630 features)
        # For each key stat, compute change between every pair of windows
        # 15 pairs × 10 stats × 2 teams × 2 (delta + acceleration) = 600
        # Plus 30 additional composite features = 630
        cross_stats = ["wp", "ppg", "margin", "ortg", "drtg",
                       "efg", "ts", "pace", "papg", "pd"]
        window_pairs = [
            (3, 5), (3, 7), (3, 10), (3, 15), (3, 20),
            (5, 7), (5, 10), (5, 15), (5, 20),
            (7, 10), (7, 15), (7, 20),
            (10, 15), (10, 20),
            (15, 20),
        ]
        for prefix in ["h", "a"]:
            for stat in cross_stats:
                for w1, w2 in window_pairs:
                    names.append(f"xw_{prefix}_{stat}_{w1}vs{w2}")       # Delta: w1 - w2
                    names.append(f"xw_accel_{prefix}_{stat}_{w1}vs{w2}") # Acceleration

        # Composite cross-window features
        for prefix in ["h", "a"]:
            names.append(f"xw_{prefix}_wp_shortterm_trend")    # Avg of all short-window deltas
            names.append(f"xw_{prefix}_wp_longterm_trend")     # Avg of all long-window deltas
            names.append(f"xw_{prefix}_margin_volatility_trend")  # Volatility change across windows
            names.append(f"xw_{prefix}_ortg_improvement_rate") # Rate of offensive improvement
            names.append(f"xw_{prefix}_drtg_improvement_rate") # Rate of defensive improvement
            names.append(f"xw_{prefix}_overall_trajectory")    # Composite trajectory score
            names.append(f"xw_{prefix}_form_acceleration")     # Is improvement accelerating?
            names.append(f"xw_{prefix}_peak_window")           # Which window shows best form (encoded)
            names.append(f"xw_{prefix}_trough_window")         # Which window shows worst form
            names.append(f"xw_{prefix}_consistency_across_windows")  # Std of values across windows
            names.append(f"xw_{prefix}_trend_agreement")       # Do all windows agree on direction?
            names.append(f"xw_{prefix}_breakout_signal")       # Short windows >> long windows
            names.append(f"xw_{prefix}_decline_signal")        # Short windows << long windows
            names.append(f"xw_{prefix}_mean_reversion_signal") # Extreme deviation from avg
            names.append(f"xw_{prefix}_momentum_strength")     # Strength of current momentum
        # NOTE: 2 teams × (10 stats × 15 pairs × 2 + 15 composites) = 630

        # 23. ADVANCED MARKET MICROSTRUCTURE II (62 features)
        if self.include_market:
            # Multi-book comparison (per book: Pinnacle, DraftKings, FanDuel, BetMGM, Caesars)
            books = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]
            for book in books:
                names.append(f"mkt2_{book}_spread")                # Spread from this book
                names.append(f"mkt2_{book}_ml_home")               # ML from this book
                names.append(f"mkt2_{book}_total")                 # Total from this book
                names.append(f"mkt2_{book}_implied_home")          # Implied prob from this book
                names.append(f"mkt2_{book}_line_move")             # Line movement for this book
                names.append(f"mkt2_{book}_reverse_move")          # Reverse line movement flag
                names.append(f"mkt2_{book}_historical_accuracy")   # This book's historical closing accuracy
            # Cross-book features
            names.extend([
                "mkt2_spread_range",                 # Max spread - min spread across books
                "mkt2_ml_range",                     # Max ML - min ML across books
                "mkt2_total_range",                  # Max total - min total across books
                "mkt2_implied_prob_range",            # Range of implied probabilities
                "mkt2_consensus_spread",              # Median spread across books
                "mkt2_consensus_total",               # Median total across books
                "mkt2_pinnacle_vs_avg",               # Pinnacle spread vs average (sharp indicator)
                "mkt2_sharp_vs_soft_spread",          # Pinnacle vs avg of soft books
                "mkt2_sharp_vs_soft_total",           # Pinnacle total vs avg of soft books
                "mkt2_time_since_open",               # Hours since market opened
                "mkt2_early_move_magnitude",          # Total line move in first 12 hours
                "mkt2_late_move_magnitude",           # Total line move in last 4 hours
                "mkt2_early_vs_late_direction",       # 1 if same direction, -1 if reversed
                "mkt2_public_vs_sharp_ratio",         # Public money% / sharp money%
                "mkt2_steam_count_24h",               # Number of steam moves in 24h
                "mkt2_contrarian_opportunity",        # 1 if public on one side, sharp on other
                "mkt2_market_maturity",               # How settled the line is (low volatility)
                "mkt2_opening_value_home",            # Our edge vs opening line
                "mkt2_current_value_home",            # Our edge vs current line
                "mkt2_value_trend",                   # Edge increasing or decreasing
                "mkt2_best_line_home",                # Best available line for home
                "mkt2_best_line_away",                # Best available line for away
                "mkt2_juice_home",                    # Vig on home side
                "mkt2_juice_away",                    # Vig on away side
                "mkt2_vig_differential",              # Difference in vig (market lean)
                "mkt2_hold_pct",                      # Total market hold percentage
                "mkt2_line_freeze_indicator",         # 1 if line hasn't moved (suspicious)
            ])

        # 24. POWER RATING COMPOSITES (64 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_elo_standard")             # Standard Elo rating
            names.append(f"{prefix}_elo_margin_adj")           # Margin-of-victory adjusted Elo
            names.append(f"{prefix}_elo_recency_weighted")     # More weight on recent games
            names.append(f"{prefix}_elo_home_adj")             # Home-court adjusted Elo
            names.append(f"{prefix}_elo_sos_adj")              # SOS-adjusted Elo
            names.append(f"{prefix}_elo_conf_adj")             # Conference-adjusted Elo
            names.append(f"{prefix}_elo_pace_adj")             # Pace-adjusted Elo
            names.append(f"{prefix}_raptor_composite")         # RAPTOR-style composite estimate
            names.append(f"{prefix}_raptor_offense")           # Offensive RAPTOR component
            names.append(f"{prefix}_raptor_defense")           # Defensive RAPTOR component
            names.append(f"{prefix}_power_rank_ortg")          # Power rank by offense
            names.append(f"{prefix}_power_rank_drtg")          # Power rank by defense
            names.append(f"{prefix}_power_rank_netrtg")        # Power rank by net rating
            names.append(f"{prefix}_power_rank_srs")           # Simple Rating System
            names.append(f"{prefix}_power_rank_composite")     # Weighted composite power rank
            names.append(f"{prefix}_power_rank_trend")         # Power rank change last 10
            names.append(f"{prefix}_power_conf_adjusted")      # Power rank adjusted for conference strength
            names.append(f"{prefix}_power_stability")          # Std dev of power rank last 20
            names.append(f"{prefix}_power_percentile")         # Power rank percentile in league
            names.append(f"{prefix}_rating_confidence")        # How stable is the rating (games played)
            names.append(f"{prefix}_bayesian_rating")          # Bayesian rating (prior + observed)
            names.append(f"{prefix}_glicko_rating")            # Glicko-style rating with uncertainty
            names.append(f"{prefix}_glicko_rd")                # Rating deviation (uncertainty)
            names.append(f"{prefix}_trueskill_mu")             # TrueSkill-style mean
            names.append(f"{prefix}_trueskill_sigma")          # TrueSkill-style uncertainty
        # Pairwise power differentials
        names.extend([
            "power_elo_std_diff",                # Standard Elo diff
            "power_elo_margin_diff",             # Margin Elo diff
            "power_elo_recency_diff",            # Recency Elo diff
            "power_raptor_diff",                 # RAPTOR composite diff
            "power_raptor_off_diff",             # RAPTOR offense diff
            "power_raptor_def_diff",             # RAPTOR defense diff
            "power_srs_diff",                    # SRS diff
            "power_composite_diff",              # Composite power diff
            "power_conf_adj_diff",               # Conference-adjusted diff
            "power_bayesian_diff",               # Bayesian rating diff
            "power_glicko_diff",                 # Glicko diff
            "power_trueskill_diff",              # TrueSkill diff
            "power_max_diff",                    # Max diff across all rating systems
            "power_avg_diff",                    # Avg diff across all rating systems
        ])

        # 25. FATIGUE & LOAD MANAGEMENT (82 features)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_cumul_games_played")       # Total games played this season
            names.append(f"{prefix}_cumul_minutes_total")      # Total team minutes this season
            names.append(f"{prefix}_avg_minutes_per_game")     # Avg minutes per game
            names.append(f"{prefix}_star_minutes_cumul")       # Cumulative minutes for top 2 players
            names.append(f"{prefix}_star_minutes_pct_season")  # Star minutes as % of season capacity
            names.append(f"{prefix}_travel_miles_season")      # Total miles traveled this season
            names.append(f"{prefix}_travel_miles_30d")         # Miles traveled last 30 days
            names.append(f"{prefix}_travel_miles_7d")          # Miles traveled last 7 days
            names.append(f"{prefix}_travel_intensity")         # Miles per game last 10
            names.append(f"{prefix}_rest_pattern_consistency") # Std dev of rest days between games
            names.append(f"{prefix}_rest_deficit_season")      # Cumulative rest deficit vs league avg
            names.append(f"{prefix}_b2b_count_season")         # Back-to-backs this season
            names.append(f"{prefix}_b2b_count_30d")            # Back-to-backs in last 30 days
            names.append(f"{prefix}_3in4_count_season")        # 3-in-4 stretches this season
            names.append(f"{prefix}_dense_schedule_flag")      # 1 if 4+ games in last 7 days
            names.append(f"{prefix}_road_trip_length")         # Current consecutive road games
            names.append(f"{prefix}_home_stand_length")        # Current consecutive home games
            names.append(f"{prefix}_road_trip_fatigue")        # Road games × avg travel per game
            names.append(f"{prefix}_load_management_prob")     # Probability of star rest
            names.append(f"{prefix}_season_fatigue_curve")     # Expected performance dropoff (game #)
            names.append(f"{prefix}_relative_fatigue")         # This team's fatigue vs league avg
            names.append(f"{prefix}_fatigue_adjusted_ortg")    # ORtg adjusted for fatigue
            names.append(f"{prefix}_fatigue_adjusted_drtg")    # DRtg adjusted for fatigue
            names.append(f"{prefix}_fatigue_adjusted_wp")      # Win% adjusted for fatigue
            names.append(f"{prefix}_recovery_quality")         # Performance after rest (historical)
            names.append(f"{prefix}_b2b_performance_drop")     # Avg performance drop on B2B
            names.append(f"{prefix}_altitude_fatigue_cumul")   # Cumulative altitude adjustment
            names.append(f"{prefix}_timezone_changes_season")  # Total timezone changes this season
            names.append(f"{prefix}_circadian_disruption")     # Recent timezone shift impact
            names.append(f"{prefix}_early_season_load")        # Heavy early schedule flag
            names.append(f"{prefix}_late_season_load")         # Heavy late schedule flag
            names.append(f"{prefix}_minutes_distribution_health") # Are starters being overworked?
            names.append(f"{prefix}_injury_risk_score")        # Fatigue-based injury risk proxy
            names.append(f"{prefix}_stamina_rating")           # Team stamina (Q4 performance vs Q1)
            names.append(f"{prefix}_clutch_fatigue")           # Performance in Q4 on B2B or heavy schedule
            names.append(f"{prefix}_fresh_vs_tired_ratio")     # Win% well-rested / win% fatigued
            names.append(f"{prefix}_optimal_rest_indicator")   # 1 if ideal rest pattern
            names.append(f"{prefix}_wear_and_tear_index")      # Composite fatigue accumulation
        # Game-level fatigue differentials
        names.extend([
            "fatigue_cumul_diff",                # h_cumul_games - a_cumul_games
            "fatigue_travel_diff",               # h_travel_season - a_travel_season
            "fatigue_rest_quality_diff",          # h_rest_consistency - a_rest_consistency
            "fatigue_load_diff",                  # h_wear_tear - a_wear_tear
            "fatigue_star_load_diff",             # h_star_minutes_pct - a_star_minutes_pct
            "fatigue_b2b_count_diff",             # h_b2b_count_30d - a_b2b_count_30d
            "fatigue_adjusted_spread",            # Current spread adjusted for fatigue
            "fatigue_composite_edge",             # Composite fatigue advantage
        ])

        self.feature_names = names

    def build(self, games, market_data=None, referee_data=None, player_data=None, quarter_data=None):
        """
        Build feature matrix from historical games.

        Args:
            games: List of game dicts with home/away teams, scores, stats
            market_data: Optional dict of game_id → market features

        Returns:
            X: numpy array (n_games, n_features)
            y: numpy array (n_games,) — 1 if home win
            feature_names: list of feature names
        """
        # State trackers
        team_results = defaultdict(list)  # team → [(date, win, margin, opp, stats_dict)]
        team_last = {}                     # team → last game date
        team_elo = defaultdict(lambda: 1500.0)
        team_home_results = defaultdict(list)
        team_away_results = defaultdict(list)
        h2h_results = defaultdict(list)    # (team1, team2) → results

        # ── Category 24: Multi-ELO state trackers ──
        team_elo_margin = defaultdict(lambda: 1500.0)     # Margin-adjusted ELO
        team_elo_offense = defaultdict(lambda: 1500.0)    # Offensive ELO
        team_elo_defense = defaultdict(lambda: 1500.0)    # Defensive ELO
        team_elo_recency = defaultdict(lambda: 1500.0)    # Recency-weighted ELO
        team_elo_history = defaultdict(list)               # team → [elo_after_each_game]
        team_home_margin_sum = defaultdict(float)          # For home court advantage
        team_home_games_count = defaultdict(int)
        # ── Category 18: Season-level trackers (precomputed per game via records) ──
        # These are derived from team_results on-the-fly (no extra state needed)

        X, y = [], []
        n_market = 32 if self.include_market else 0

        for game in games:
            # Parse game
            hr, ar = game.get("home_team", ""), game.get("away_team", "")
            if "home" in game and isinstance(game["home"], dict):
                h, a = game["home"], game.get("away", {})
                hs = h.get("pts")
                as_ = a.get("pts")
                if not hr:
                    hr = h.get("team_name", "")
                if not ar:
                    ar = a.get("team_name", "")
                h_stats = h
                a_stats = a
            else:
                hs = game.get("home_score")
                as_ = game.get("away_score")
                h_stats = game.get("home_stats", {})
                a_stats = game.get("away_stats", {})

            if hs is None or as_ is None:
                continue
            hs, as_ = int(hs), int(as_)
            home, away = resolve(hr), resolve(ar)
            if not home or not away:
                continue
            gd = game.get("game_date", game.get("date", ""))[:10]

            hr_ = team_results[home]
            ar_ = team_results[away]

            # Skip if not enough history for any features
            if len(hr_) < 3 or len(ar_) < 3:
                # Still record this game for future reference
                self._record_game(team_results, team_last, team_elo,
                                  team_home_results, team_away_results,
                                  h2h_results, home, away, hs, as_, gd,
                                  h_stats, a_stats)
                # Update multi-ELO systems (Cat 24)
                self._update_multi_elo(
                    home, away, hs, as_, h_stats, a_stats,
                    team_elo_margin, team_elo_offense, team_elo_defense,
                    team_elo_recency, team_elo_history,
                    team_home_margin_sum, team_home_games_count)
                continue

            # ── Build feature vector ──
            row = []

            # 1. ROLLING PERFORMANCE (96 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in WINDOWS:
                    row.append(self._wp(tr, w))
                    row.append(self._pd(tr, w))
                    row.append(self._ppg(tr, w))
                    row.append(self._papg(tr, w))
                    row.append(self._avg_margin(tr, w))
                    row.append(self._close_pct(tr, w))
                    row.append(self._blowout_pct(tr, w))
                    row.append(self._ou_avg(tr, w))

            # 2. FOUR FACTORS (64 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in [5, 10]:
                    row.append(self._efg(tr, w))
                    row.append(self._tov_rate(tr, w))
                    row.append(self._orb_rate(tr, w))
                    row.append(self._ft_rate(tr, w))
                    row.append(self._opp_efg(tr, w))
                    row.append(self._opp_tov_rate(tr, w))
                    row.append(self._opp_orb_rate(tr, w))
                    row.append(self._opp_ft_rate(tr, w))

            # 3. PACE & EFFICIENCY (48 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in [5, 10]:
                    row.append(self._ortg(tr, w))
                    row.append(self._drtg(tr, w))
                    row.append(self._netrtg(tr, w))
                    row.append(self._pace(tr, w))
                    row.append(self._ts(tr, w))
                    row.append(self._avg_poss(tr, w))
                    row.append(self._ast_rate(tr, w))
                    row.append(self._stl_rate(tr, w))
                    row.append(self._blk_rate(tr, w))
                    row.append(self._tov_pct(tr, w))
                    row.append(self._oreb_pct(tr, w))
                    row.append(self._dreb_pct(tr, w))

            # 4. SCORING PROFILE (40 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in [5, 10]:
                    row.append(self._stat_avg(tr, w, "3par"))
                    row.append(self._stat_avg(tr, w, "fg3_pct"))
                    row.append(self._stat_avg(tr, w, "fg2_pct"))
                    row.append(self._stat_avg(tr, w, "ft_pct"))
                    row.append(self._stat_avg(tr, w, "paint_pts"))
                    row.append(self._stat_avg(tr, w, "fb_pts"))
                    row.append(self._stat_avg(tr, w, "bench_pts"))
                    row.append(self._stat_avg(tr, w, "2nd_pts"))
                    row.append(self._stat_avg(tr, w, "pitp"))
                    row.append(self._stat_avg(tr, w, "pts_off_tov"))

            # 5. MOMENTUM & STREAKS (32 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                row.append(self._streak(tr))
                row.append(abs(self._streak(tr)))
                row.append(self._wp(tr, 5) - self._wp(tr, 82))
                row.append(self._wp(tr, 3) - self._wp(tr, 10))
                h_rec = team_home_results if prefix == "h" else team_away_results
                t_key = home if prefix == "h" else away
                row.append(self._wp(h_rec.get(t_key, []), 82))
                row.append(self._wp(team_away_results.get(t_key, []), 82))
                row.append(self._wp(team_home_results.get(t_key, []), 82) -
                          self._wp(team_away_results.get(t_key, []), 82))
                row.append(self._ats_wp(tr, 5))
                row.append(self._ou_record(tr, 5))
                row.append(self._ppg(tr, 5) - self._ppg(tr, 20))
                row.append(self._papg(tr, 5) - self._papg(tr, 20))
                row.append(self._clutch_wp(tr))
                row.append(self._blowout_pct(tr, 82))
                row.append(self._comeback_rate(tr))
                row.append(self._consistency(tr, 10))
                row.append(self._consistency(tr, 5))

            # 6. REST & SCHEDULE (24 features)
            h_rest = self._rest_days(home, gd, team_last)
            a_rest = self._rest_days(away, gd, team_last)
            row.extend([
                min(h_rest, 7), min(a_rest, 7),
                h_rest - a_rest,
                1.0 if h_rest <= 1 else 0.0,
                1.0 if a_rest <= 1 else 0.0,
                self._n_in_m(hr_, gd, 3, 4),
                self._n_in_m(ar_, gd, 3, 4),
                self._n_in_m(hr_, gd, 4, 6),
                self._n_in_m(ar_, gd, 4, 6),
                self._travel_dist(hr_, home),
                self._travel_dist(ar_, away),
                self._travel_dist(hr_, home) - self._travel_dist(ar_, away),
                ARENA_ALTITUDE.get(home, 500),
                ARENA_ALTITUDE.get(away, 500),
                ARENA_ALTITUDE.get(home, 500) - ARENA_ALTITUDE.get(away, 500),
                abs(TIMEZONE_ET.get(home, 0) - TIMEZONE_ET.get(self._last_location(hr_), 0)),
                abs(TIMEZONE_ET.get(away, 0) - TIMEZONE_ET.get(self._last_location(ar_), 0)),
                (abs(TIMEZONE_ET.get(home, 0) - TIMEZONE_ET.get(self._last_location(hr_), 0)) -
                 abs(TIMEZONE_ET.get(away, 0) - TIMEZONE_ET.get(self._last_location(ar_), 0))),
                self._games_in_window(hr_, gd, 7),
                self._games_in_window(ar_, gd, 7),
                self._miles_in_window(hr_, gd, 7, home),
                self._miles_in_window(ar_, gd, 7, away),
                self._games_in_window(hr_, gd, 7) - self._games_in_window(ar_, gd, 7),
                self._fatigue_score(hr_, gd, home, h_rest) - self._fatigue_score(ar_, gd, away, a_rest),
            ])

            # 7. OPPONENT-ADJUSTED (24 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                row.append(self._sos(tr, team_results, 5))
                row.append(self._sos(tr, team_results, 10))
                row.append(self._sos(tr, team_results, 82))
                row.append(self._wp_vs_quality(tr, team_results, above=True))
                row.append(self._wp_vs_quality(tr, team_results, above=False))
                row.append(self._wp_vs_topN(tr, team_results, 10, top=True))
                row.append(self._wp_vs_topN(tr, team_results, 10, top=False))
                row.append(self._pd_vs_topN(tr, team_results, 10, top=True))
                row.append(self._pd_vs_topN(tr, team_results, 10, top=False))
                row.append(self._avg_opp_stat(tr, team_results, "ortg", 10))
                row.append(self._avg_opp_stat(tr, team_results, "drtg", 10))
                row.append(self._margin_vs_quality_corr(tr, team_results))

            # 8. MATCHUP & HEAD-TO-HEAD (20 features)
            h2h = h2h_results.get((home, away), []) + h2h_results.get((away, home), [])
            row.extend([
                self._h2h_wp(h2h, home),
                self._h2h_wp(h2h[-3:], home) if len(h2h) >= 3 else 0.5,
                self._h2h_margin(h2h, home),
                self._h2h_home_wp(h2h),
                self._pace(hr_, 10) - self._pace(ar_, 10),
                self._ortg(hr_, 10) - self._drtg(ar_, 10),
                self._stat_avg(hr_, 10, "fg3_pct") - self._stat_avg(ar_, 10, "opp_fg3_pct"),
                self._stat_avg(hr_, 10, "paint_pts") - self._stat_avg(ar_, 10, "opp_paint_pts"),
                abs(self._pace(hr_, 10) - self._pace(ar_, 10)),
                self._drtg(hr_, 10) - self._drtg(ar_, 10),
                self._oreb_pct(hr_, 10) - self._oreb_pct(ar_, 10),
                self._tov_rate(hr_, 10) - self._tov_rate(ar_, 10),
                self._ft_rate(hr_, 10) - self._ft_rate(ar_, 10),
                self._stat_avg(hr_, 10, "bench_pts") - self._stat_avg(ar_, 10, "bench_pts"),
                self._consistency(hr_, 10) - self._consistency(ar_, 10),
                self._netrtg(hr_, 10) * (1 - 0.05 * max(0, self._games_in_window(hr_, gd, 7) - 3)),
                team_elo[home],
                team_elo[away],
                team_elo[home] - team_elo[away] + 50,  # +50 for home court
                (team_elo[home] - 1500) - self._elo_10_ago(hr_, team_elo, home),
            ])

            # 9. MARKET MICROSTRUCTURE (32 features) — filled from market_data or zeros
            if self.include_market:
                mkt = (market_data or {}).get(game.get("id", gd), {})
                row.extend([
                    mkt.get("opening_spread", 0),
                    mkt.get("current_spread", 0),
                    mkt.get("current_spread", 0) - mkt.get("opening_spread", 0),
                    abs(mkt.get("current_spread", 0) - mkt.get("opening_spread", 0)),
                    mkt.get("reverse_line_movement", 0),
                    mkt.get("opening_total", 220),
                    mkt.get("current_total", 220),
                    mkt.get("current_total", 220) - mkt.get("opening_total", 220),
                    mkt.get("opening_ml_home", -110),
                    mkt.get("current_ml_home", -110),
                    mkt.get("current_ml_home", -110) - mkt.get("opening_ml_home", -110),
                    mkt.get("implied_prob_home", 0.5),
                    mkt.get("implied_prob_away", 0.5),
                    0,  # model_vs_market — filled post-prediction
                    0,  # edge_magnitude
                    mkt.get("books_disagreement", 0.05),
                    mkt.get("sharp_line", 0.5),
                    mkt.get("public_pct_home", 0.5),
                    mkt.get("public_money_pct_home", 0.5),
                    mkt.get("smart_money_indicator", 0),
                    mkt.get("steam_move", 0),
                    mkt.get("clv_recent_avg", 0),
                    mkt.get("market_efficiency", 0),
                    mkt.get("opening_overround", 1.05),
                    mkt.get("best_odds_home", 1.9),
                    mkt.get("best_odds_away", 1.9),
                    mkt.get("odds_range_home", 0.1),
                    mkt.get("odds_range_away", 0.1),
                    mkt.get("time_to_close", 24),
                    mkt.get("late_money_direction", 0),
                    mkt.get("closing_line_estimate", 0.5),
                    mkt.get("historical_clv", 0),
                    1 if mkt.get("best_odds_home", 1.9) > 5.0 or mkt.get("best_odds_away", 1.9) > 5.0 else 0,
                ])

            # 10. CONTEXT & SITUATIONAL (24 features)
            try:
                dt = datetime.strptime(gd, "%Y-%m-%d")
                month = dt.month
                dow = dt.weekday()
            except:
                month = 1
                dow = 2
                dt = None

            sp = max(0, min(1, (month - 10) / 7)) if month >= 10 else max(0, min(1, (month + 2) / 7))
            row.extend([
                1.0,  # home court
                sp,
                math.sin(2 * math.pi * month / 12),
                math.cos(2 * math.pi * month / 12),
                dow / 6.0,
                1.0 if dow >= 5 else 0.0,
                0,  # national TV (needs external data)
                min(len(hr_), 82) / 82.0,
                min(len(ar_), 82) / 82.0,
                min(len(hr_), 82) / 82.0,
                min(len(ar_), 82) / 82.0,
                1.0 if self._wp(hr_, 82) > 0.5 and self._wp(ar_, 82) > 0.5 else 0.0,
                1.0 if self._wp(hr_, 82) < 0.3 or self._wp(ar_, 82) < 0.3 else 0.0,
                1.0 if self._is_rivalry(home, away) else 0.0,
                self._division(home),
                self._division(away),
                1.0 if self._division(home) == self._division(away) else 0.0,
                1.0 if self._conference(home) == self._conference(away) else 0.0,
                1.0 if self._conference(home) != self._conference(away) else 0.0,
                self._wp(hr_, 82) - self._wp(ar_, 82),
                1.0 if self._wp(hr_, 82) > self._wp(ar_, 82) else 0.0,
                self._wp(hr_, 82) + self._wp(ar_, 82),
                0.5,  # game importance (needs standings)
                self._ppg(hr_, 10) + self._ppg(ar_, 10),
            ])

            # 11. REFEREE FEATURES (10 features)
            ref = (referee_data or {}).get(game.get("id", gd), {})
            row.extend([
                ref.get("home_foul_bias", 0.0),
                ref.get("total_fouls_avg", 42.0),
                ref.get("foul_rate_vs_league", 1.0),
                ref.get("home_ft_advantage", 0.0),
                ref.get("experience_games", 40) / 82.0,
                ref.get("over_tendency", 0.5),
                ref.get("close_game_bias", 0.5),
                ref.get("tech_foul_rate", 0.3),
                ref.get("home_win_rate", 0.58),
                ref.get("pace_impact", 0.0),
            ])

            # 12. PLAYER IMPACT FEATURES (16 features)
            for prefix, team_key in [("h", home), ("a", away)]:
                pd_ = (player_data or {}).get(team_key, {})
                row.append(pd_.get("star_usage_rate", 0.55))
                row.append(pd_.get("star_minutes_load", 34.0) / 48.0)
                row.append(pd_.get("injury_impact_score", 0.0))
                row.append(pd_.get("injured_war_lost", 0.0))
                row.append(pd_.get("lineup_continuity", 0.8))
                row.append(pd_.get("bench_depth_rating", 0.0) / 10.0)
                rest = self._rest_days(team_key, gd, team_last)
                row.append(1.0 if rest <= 1 else (0.5 if rest <= 2 else 0.0))
                row.append(pd_.get("rotation_depth", 8) / 15.0)

            # 13. QUARTER-LEVEL PATTERNS (14 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                qd_ = (quarter_data or {}).get(home if prefix == "h" else away, {})
                row.append(qd_.get("q1_margin_avg", 0.0))
                row.append(qd_.get("q3_margin_avg", 0.0))
                row.append(qd_.get("q4_clutch_netrtg", 0.0) / 10.0)
                row.append(qd_.get("half_adjustment", 0.0))
                row.append(qd_.get("comeback_win_pct", 0.3))
                row.append(qd_.get("blowout_hold_pct", 0.7))
                row.append(qd_.get("garbage_time_margin", 0.0))

            # 14. DEFENSIVE MATCHUP ADVANCED (12 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                row.append(self._stat_avg(tr, 10, "opp_paint_pts") / 50.0)
                row.append(self._stat_avg(tr, 10, "opp_fg3_pct"))
                row.append(self._stat_avg(tr, 10, "fb_pts") / 20.0)  # proxy: own FB pts
                row.append(0.6)  # contest rate placeholder
                row.append(self._stat_avg(tr, 10, "stl_rate") * 5)  # proxy: deflections
                row.append(0.55)  # rim protection placeholder

            # 15. POLYMARKET & PREDICTION MARKET (8 features)
            pmkt = (market_data or {}).get(game.get("id", gd), {})
            row.extend([
                pmkt.get("polymarket_home_prob", 0.5),
                pmkt.get("polymarket_volume", 0.5),
                pmkt.get("polymarket_line_movement", 0.0),
                pmkt.get("polymarket_vs_books", 0.0),
                pmkt.get("prediction_market_consensus", 0.5),
                pmkt.get("market_wisdom_confidence", 0.5),
                pmkt.get("smart_vs_public_divergence", 0.0),
                pmkt.get("closing_line_value_history", 0.0),
            ])

            # ── CATEGORIES 16-25: ADVANCED FEATURE COMPUTATION ──
            # Build name→index lookup for values already computed (cats 1-15)
            _cat15_end = len(row)
            _name_idx = {}
            for _i, _n in enumerate(self.feature_names):
                if _i < _cat15_end:
                    _name_idx[_n] = _i

            def _val(name):
                """Fast lookup of already-computed feature value by name."""
                idx = _name_idx.get(name)
                if idx is not None:
                    return row[idx]
                return 0.0

            # 16. INTERACTION & POLYNOMIAL FEATURES
            # 16a. Pairwise interaction products
            inter_pairs = [
                ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"), ("h_wp3", "a_wp3"),
                ("h_ortg10", "a_drtg10"), ("h_ortg5", "a_drtg5"),
                ("h_drtg10", "a_ortg10"), ("h_drtg5", "a_ortg5"),
                ("h_netrtg10", "a_netrtg10"), ("h_netrtg5", "a_netrtg5"),
                ("h_pace10", "a_pace10"), ("h_pace5", "a_pace5"),
                ("h_ppg10", "a_ppg10"), ("h_ppg5", "a_ppg5"),
                ("h_margin10", "a_margin10"), ("h_margin5", "a_margin5"),
                ("h_efg10", "a_efg10"), ("h_efg5", "a_efg5"),
                ("h_ts10", "a_ts10"), ("h_ts5", "a_ts5"),
                ("h_tov_rate10", "a_tov_rate10"), ("h_tov_rate5", "a_tov_rate5"),
                ("h_orb_rate10", "a_orb_rate10"), ("h_orb_rate5", "a_orb_rate5"),
                ("h_3p_pct10", "a_3p_pct10"), ("h_3p_pct5", "a_3p_pct5"),
                ("h_ft_rate10", "a_ft_rate10"), ("h_ft_rate5", "a_ft_rate5"),
                ("h_pd10", "a_pd10"), ("h_pd5", "a_pd5"),
                ("h_blowout10", "a_blowout10"), ("h_close10", "a_close10"),
                ("h_wp10", "elo_diff"), ("a_wp10", "elo_diff"),
                ("h_ortg10", "elo_diff"), ("h_streak", "a_streak"),
                ("h_wp10", "h_rest_days"), ("a_wp10", "a_rest_days"),
                ("h_netrtg10", "rest_advantage"), ("h_ortg10", "h_pace10"),
                ("a_ortg10", "a_pace10"), ("h_drtg10", "a_3p_pct10"),
                ("h_wp10", "h_sos10"), ("a_wp10", "a_sos10"),
                ("h_efg10", "h_pace10"), ("a_efg10", "a_pace10"),
                ("h_margin10", "h_consistency"), ("a_margin10", "a_consistency"),
                ("h_ppg10", "a_papg10"), ("a_ppg10", "h_papg10"),
                ("h_bench_pts10", "a_bench_pts10"),
                ("h_fb_pts10", "a_fb_pts10"),
                ("h_opp_efg10", "a_efg10"), ("a_opp_efg10", "h_efg10"),
                ("h_wp20", "a_wp20"), ("h_wp15", "a_wp15"),
                ("h_ortg10", "h_efg10"), ("a_ortg10", "a_efg10"),
                ("h_drtg10", "h_opp_efg10"), ("a_drtg10", "a_opp_efg10"),
                ("h_pace10", "h_3p_pct10"), ("a_pace10", "a_3p_pct10"),
                ("h_ast_rate10", "a_tov_pct10"), ("a_ast_rate10", "h_tov_pct10"),
                ("h_stl_rate10", "a_tov_pct10"), ("a_stl_rate10", "h_tov_pct10"),
                ("h_blk_rate10", "a_paint_pts10"), ("a_blk_rate10", "h_paint_pts10"),
                ("h_oreb_pct10", "a_dreb_pct10"), ("a_oreb_pct10", "h_dreb_pct10"),
                ("h_3par10", "a_perimeter_defense"), ("a_3par10", "h_perimeter_defense"),
                ("h_wp10", "a_consistency"), ("a_wp10", "h_consistency"),
                ("h_margin10", "elo_diff"), ("a_margin10", "elo_diff"),
                ("h_ppg10", "h_pace10"), ("a_ppg10", "a_pace10"),
                ("h_papg10", "h_drtg10"), ("a_papg10", "a_drtg10"),
                ("h_wp10", "h_home_wp"), ("a_wp10", "a_away_wp"),
                ("h_streak", "h_wp10"), ("a_streak", "a_wp10"),
                ("h_clutch_wp", "a_clutch_wp"), ("h_comeback_rate", "a_comeback_rate"),
                ("h_scoring_trend", "a_defense_trend"),
                ("a_scoring_trend", "h_defense_trend"),
                ("h_ou_avg10", "a_ou_avg10"),
                ("h_netrtg10", "h_consistency"), ("a_netrtg10", "a_consistency"),
                ("h_efg10", "a_opp_efg10"), ("a_efg10", "h_opp_efg10"),
                ("h_ts10", "h_3p_pct10"), ("a_ts10", "a_3p_pct10"),
                ("h_ft_rate10", "h_ft_pct10"), ("a_ft_rate10", "a_ft_pct10"),
                ("h_pts_off_tov10", "a_tov_rate10"), ("a_pts_off_tov10", "h_tov_rate10"),
                ("h_2nd_pts10", "h_oreb_pct10"), ("a_2nd_pts10", "a_oreb_pct10"),
                ("h_wp10", "season_phase"), ("a_wp10", "season_phase"),
                ("elo_diff", "rest_advantage"), ("elo_diff", "travel_advantage"),
                ("current_spread", "h_wp10"), ("current_spread", "a_wp10"),
                ("current_spread", "elo_diff"), ("h_sos10", "a_sos10"),
                ("h_wp_vs_above500", "a_wp_vs_above500"),
                ("h_wp_vs_top10", "a_wp_vs_top10"),
            ]
            for x_name, y_name in inter_pairs:
                row.append(_val(x_name) * _val(y_name))

            # 16b. Ratio features
            ratio_pairs = [
                ("h_ortg10", "a_drtg10"), ("h_ortg5", "a_drtg5"),
                ("a_ortg10", "h_drtg10"), ("a_ortg5", "h_drtg5"),
                ("h_pace10", "a_pace10"), ("h_pace5", "a_pace5"),
                ("h_efg10", "a_efg10"), ("h_efg5", "a_efg5"),
                ("h_ts10", "a_ts10"), ("h_ts5", "a_ts5"),
                ("h_ppg10", "a_ppg10"), ("h_ppg5", "a_ppg5"),
                ("h_margin10", "a_margin10"), ("h_margin5", "a_margin5"),
                ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"),
                ("h_3p_pct10", "a_3p_pct10"), ("h_3p_pct5", "a_3p_pct5"),
                ("h_ft_rate10", "a_ft_rate10"), ("h_ft_rate5", "a_ft_rate5"),
                ("h_orb_rate10", "a_orb_rate10"), ("h_orb_rate5", "a_orb_rate5"),
                ("h_tov_rate10", "a_tov_rate10"), ("h_tov_rate5", "a_tov_rate5"),
                ("h_bench_pts10", "a_bench_pts10"), ("h_bench_pts5", "a_bench_pts5"),
                ("h_papg10", "a_papg10"), ("h_papg5", "a_papg5"),
                ("h_opp_efg10", "a_opp_efg10"), ("h_opp_efg5", "a_opp_efg5"),
            ]
            for x_name, y_name in ratio_pairs:
                denom = _val(y_name)
                row.append(_val(x_name) / (denom + 0.001) if abs(denom) > 0.0001 else 1.0)

            # 16c. Squared terms
            sq_features = [
                "h_wp10", "a_wp10", "h_wp5", "a_wp5",
                "h_ortg10", "a_ortg10", "h_drtg10", "a_drtg10",
                "h_netrtg10", "a_netrtg10", "h_margin10", "a_margin10",
                "elo_diff", "spread_movement", "current_spread",
                "h_pace10", "a_pace10", "h_efg10", "a_efg10",
                "h_ppg10", "a_ppg10", "h_ts10", "a_ts10",
                "rest_advantage", "travel_advantage",
                "h_streak", "a_streak", "h_sos10", "a_sos10",
                "h_consistency", "a_consistency",
            ]
            for feat in sq_features:
                v = _val(feat)
                row.append(v * v)

            # 16d. Trend delta features: short_window - long_window
            trend_stats = ["wp", "ppg", "margin", "ortg", "drtg",
                           "efg", "ts", "pace", "pd", "papg"]
            trend_window_pairs = [
                (3, 7), (3, 10), (3, 15), (3, 20),
                (5, 10), (5, 15), (5, 20),
                (7, 15), (7, 20), (10, 20),
            ]
            for prefix in ["h", "a"]:
                for stat in trend_stats:
                    for w1, w2 in trend_window_pairs:
                        row.append(_val(f"{prefix}_{stat}{w1}") - _val(f"{prefix}_{stat}{w2}"))

            # ══════════════════════════════════════════════════════════
            # 17. ADVANCED ROLLING STATISTICS (168 features)
            # EWMA, volatility, z-scores, skew/kurtosis, range, CV
            # ══════════════════════════════════════════════════════════

            # Stat name → record accessor (record = (date, win, margin, opp, stats_dict))
            _STAT_KEY_17 = {
                "ppg": lambda r: r[4].get("pts", 100),
                "margin": lambda r: r[2],
                "ortg": lambda r: r[4].get("ortg", 100),
                "drtg": lambda r: r[4].get("drtg", 100),
                "efg": lambda r: r[4].get("efg_pct", 0.5),
                "ts": lambda r: r[4].get("ts_pct", 0.5),
                "pace": lambda r: r[4].get("pace", 100),
                "papg": lambda r: r[4].get("opp_pts", 100),
                "3p_pct": lambda r: r[4].get("fg3_pct", 0.36),
                "ft_rate": lambda r: r[4].get("ft_rate", 0.25),
            }

            def _extract(tr, stat, n):
                """Extract last n values for a stat from team records."""
                s = tr[-n:] if n <= len(tr) else tr
                fn = _STAT_KEY_17.get(stat)
                return [fn(r) for r in s] if fn and s else []

            def _ewma_val(values, alpha):
                """EWMA via manual recurrence — no pandas dependency."""
                if not values:
                    return 0.0
                result = values[0]
                for v in values[1:]:
                    result = alpha * v + (1.0 - alpha) * result
                return result

            def _std_17(values):
                """Sample standard deviation."""
                n_v = len(values)
                if n_v < 2:
                    return 0.0
                m = sum(values) / n_v
                return math.sqrt(sum((v - m) ** 2 for v in values) / (n_v - 1))

            _ALPHA_MAP_17 = {"01": 0.1, "03": 0.3, "05": 0.5}

            # Pre-extract all needed stat series for both teams (avoid redundant slicing)
            _tc = {}
            for _pfx, _tr in [("h", hr_), ("a", ar_)]:
                _tc[_pfx] = {}
                for _st in ["ppg", "margin", "ortg", "drtg", "efg", "ts", "pace",
                             "papg", "3p_pct", "ft_rate"]:
                    _tc[_pfx][_st] = {}
                    for _w in [5, 10, 20, 82]:
                        _tc[_pfx][_st][_w] = _extract(_tr, _st, _w)

            for _pfx in ["h", "a"]:
                _c = _tc[_pfx]

                # ── EWMA: 7 stats × 3 alphas = 21 per team ──
                for _st in ["ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]:
                    _v20 = _c[_st][20]
                    for _ak in ["01", "03", "05"]:
                        row.append(_ewma_val(_v20, _ALPHA_MAP_17[_ak]))

                # ── Rolling volatility (std): 6 stats × 3 windows = 18 per team ──
                for _st in ["margin", "ppg", "papg", "ortg", "drtg", "pace"]:
                    for _w in [5, 10, 20]:
                        row.append(_std_17(_c[_st][_w]))

                # ── Rolling min/max: 4 stats × 2 windows × 2 = 16 per team ──
                for _st in ["ppg", "papg", "margin", "ortg"]:
                    for _w in [5, 10]:
                        _vals = _c[_st][_w]
                        if _vals:
                            row.append(min(_vals))
                            row.append(max(_vals))
                        else:
                            row.extend([0.0, 0.0])

                # ── Z-scores vs season avg: 10 stats = 10 per team ──
                for _st in ["ppg", "papg", "margin", "ortg", "drtg",
                             "efg", "ts", "pace", "3p_pct", "ft_rate"]:
                    _season = _c[_st][82]
                    _recent = _c[_st][5]
                    if len(_season) >= 5 and _recent:
                        _s_mean = sum(_season) / len(_season)
                        _s_var = sum((v - _s_mean) ** 2 for v in _season) / len(_season)
                        _s_std = math.sqrt(_s_var) if _s_var > 0 else 1e-6
                        _r_mean = sum(_recent) / len(_recent)
                        row.append((_r_mean - _s_mean) / _s_std)
                    else:
                        row.append(0.0)

                # ── Skew & kurtosis (window=10): 4 stats × 2 = 8 per team ──
                for _st in ["margin", "ppg", "ortg", "drtg"]:
                    _vals = _c[_st][10]
                    if len(_vals) >= 3:
                        _nv = len(_vals)
                        _m = sum(_vals) / _nv
                        _m2 = sum((v - _m) ** 2 for v in _vals) / _nv
                        _sd = math.sqrt(_m2) if _m2 > 0 else 1e-6
                        _m3 = sum((v - _m) ** 3 for v in _vals) / _nv
                        _m4 = sum((v - _m) ** 4 for v in _vals) / _nv
                        row.append(_m3 / (_sd ** 3) if _sd > 1e-9 else 0.0)  # skew
                        row.append((_m4 / (_sd ** 4)) - 3.0 if _sd > 1e-9 else 0.0)  # kurtosis
                    else:
                        row.extend([0.0, 0.0])

                # ── Range (max - min): 4 stats × 2 windows = 8 per team ──
                for _st in ["ppg", "margin", "ortg", "drtg"]:
                    for _w in [5, 10]:
                        _vals = _c[_st][_w]
                        row.append((max(_vals) - min(_vals)) if _vals else 0.0)

                # ── Coefficient of variation: 3 stats = 3 per team ──
                for _st in ["ppg", "margin", "ortg"]:
                    _vals = _c[_st][10]
                    if len(_vals) >= 2:
                        _m = sum(_vals) / len(_vals)
                        _sd = _std_17(_vals)
                        row.append(_sd / abs(_m) if abs(_m) > 1e-6 else 0.0)
                    else:
                        row.append(0.0)

            # ── Locate category boundaries once (cached after first game) ──
            if not hasattr(self, '_cat_bounds'):
                self._cat_bounds = {}
                for _i, _n in enumerate(self.feature_names):
                    if _n == "h_pyth_wp" and 18 not in self._cat_bounds:
                        self._cat_bounds[18] = _i
                    if _n == "h_starting5_netrtg" and 19 not in self._cat_bounds:
                        self._cat_bounds[19] = _i
                    if _n == "h_elo_standard" and 24 not in self._cat_bounds:
                        self._cat_bounds[24] = _i
                    if _n == "h_cumul_games_played" and 25 not in self._cat_bounds:
                        self._cat_bounds[25] = _i

            _cat25_start = self._cat_bounds.get(25, len(self.feature_names))

            # ================================================================
            # 18. SEASON TRAJECTORY & CONTEXT (86 features) — REAL COMPUTATION
            # ================================================================
            for prefix, tr, team_key in [("h", hr_, home), ("a", ar_, away)]:
                n_games = len(tr)
                home_tr = team_home_results.get(team_key, [])
                away_tr = team_away_results.get(team_key, [])

                # ── Pythagorean win expectation ──
                total_pts = sum(r[4].get("pts", 100) for r in tr)
                total_opp = sum(r[4].get("opp_pts", 100) for r in tr)
                _e = 13.91
                pts_exp = total_pts ** _e if total_pts > 0 else 1.0
                opp_exp = total_opp ** _e if total_opp > 0 else 1.0
                pyth_wp = pts_exp / (pts_exp + opp_exp) if (pts_exp + opp_exp) > 0 else 0.5
                row.append(pyth_wp)

                # Pythagorean vs actual (luck measure)
                actual_wp = self._wp(tr, n_games)
                row.append(pyth_wp - actual_wp)

                # Win pace over 82 games
                win_pace = actual_wp * 82.0
                row.append(win_pace / 82.0)

                # Playoff pace delta (vs 42-win threshold)
                row.append((win_pace - 42.0) / 82.0)

                # Games behind 1st in conference
                conf = self._conference(team_key)
                conf_wps = []
                for t, recs in team_results.items():
                    if recs and self._conference(t) == conf:
                        conf_wps.append((t, self._wp(recs, len(recs))))
                conf_wps.sort(key=lambda x: x[1], reverse=True)
                best_wp = conf_wps[0][1] if conf_wps else 0.5
                row.append(max(0, (best_wp - actual_wp) * n_games) / 82.0)

                # Games behind 8th seed
                eighth_wp = conf_wps[7][1] if len(conf_wps) >= 8 else 0.5
                row.append((eighth_wp - actual_wp) * n_games / 82.0)

                # Games ahead of lottery (worst record)
                worst_wp = conf_wps[-1][1] if conf_wps else 0.5
                row.append((actual_wp - worst_wp) * n_games / 82.0)

                # Strength of remaining schedule
                row.append(self._sos(tr, team_results, min(n_games, 20)))

                # Conference ranking
                team_rank = 15
                for idx_r, (t, _) in enumerate(conf_wps):
                    if t == team_key:
                        team_rank = idx_r + 1
                        break
                row.append(team_rank / 15.0)

                # Division ranking
                div_code = self._division(team_key)
                div_wps = [(t, w) for t, w in conf_wps if self._division(t) == div_code]
                div_rank = 5
                for idx_r, (t, _) in enumerate(div_wps):
                    if t == team_key:
                        div_rank = idx_r + 1
                        break
                row.append(div_rank / 5.0)

                # Is playoff team (top 10 in conf)
                row.append(1.0 if team_rank <= 10 else 0.0)

                # Play-in range (7th-12th)
                row.append(1.0 if 7 <= team_rank <= 12 else 0.0)

                # Pre All-Star win% (first 55 games proxy)
                pre_asg = tr[:min(55, n_games)]
                row.append(self._wp(pre_asg, len(pre_asg)) if pre_asg else 0.5)

                # Post All-Star win%
                post_asg = tr[55:] if n_games > 55 else []
                row.append(self._wp(post_asg, len(post_asg)) if post_asg else 0.5)

                # All-Star delta
                pre_wp = self._wp(pre_asg, len(pre_asg)) if pre_asg else 0.5
                post_wp = self._wp(post_asg, len(post_asg)) if post_asg else 0.5
                row.append(post_wp - pre_wp)

                # Pre trade deadline win% (first 45 games proxy)
                pre_dl = tr[:min(45, n_games)]
                row.append(self._wp(pre_dl, len(pre_dl)) if pre_dl else 0.5)

                # Post deadline win%
                post_dl = tr[45:] if n_games > 45 else []
                row.append(self._wp(post_dl, len(post_dl)) if post_dl else 0.5)

                # Deadline delta
                pre_dl_wp = self._wp(pre_dl, len(pre_dl)) if pre_dl else 0.5
                post_dl_wp = self._wp(post_dl, len(post_dl)) if post_dl else 0.5
                row.append(post_dl_wp - pre_dl_wp)

                # Monthly win% trend
                if n_games >= 30:
                    row.append(self._wp(tr[-15:], 15) - self._wp(tr[-30:-15], 15))
                else:
                    row.append(0.0)

                # Monthly ORtg trend
                if n_games >= 30:
                    row.append(self._ortg(tr, 15) - self._stat_avg(tr[-30:-15], 15, "ortg"))
                else:
                    row.append(0.0)

                # Monthly DRtg trend
                if n_games >= 30:
                    row.append(self._drtg(tr, 15) - self._stat_avg(tr[-30:-15], 15, "drtg"))
                else:
                    row.append(0.0)

                # Season half improvement: 2nd half vs 1st half win%
                half = n_games // 2
                if half >= 5:
                    row.append(self._wp(tr[half:], n_games - half) - self._wp(tr[:half], half))
                else:
                    row.append(0.0)

                # Regression indicator (distance from .500)
                row.append(actual_wp - 0.5)

                # Hot/cold regime (last 15: 1=hot, 0=neutral, -1=cold)
                last15_wp = self._wp(tr, 15)
                regime = 1.0 if last15_wp >= 0.667 else (-1.0 if last15_wp <= 0.333 else 0.0)
                row.append(regime)

                # Clinch status (0=eliminated, 1=alive, 2=clinched)
                games_rem = max(0, 82 - n_games)
                max_wins = actual_wp * n_games + games_rem
                if max_wins / 82.0 < eighth_wp and eighth_wp > 0.3:
                    clinch = 0.0
                elif actual_wp * n_games > best_wp * 82:
                    clinch = 2.0
                else:
                    clinch = 1.0
                row.append(clinch / 2.0)

                # Games remaining
                row.append(games_rem / 82.0)

                # Win% last 30 games
                wp_last30 = self._wp(tr, 30)
                row.append(wp_last30)

                # Win% last 30 vs season (form delta)
                row.append(wp_last30 - actual_wp)

                # Home/road trend last 5
                rec_h = home_tr[-5:] if len(home_tr) >= 5 else home_tr
                rec_a = away_tr[-5:] if len(away_tr) >= 5 else away_tr
                h_wp_5 = self._wp(rec_h, len(rec_h)) if rec_h else 0.5
                a_wp_5 = self._wp(rec_a, len(rec_a)) if rec_a else 0.5
                row.append(h_wp_5 - a_wp_5)

                # Scoring variance trend
                if n_games >= 20:
                    row.append((self._consistency(tr[-20:-10], 10) - self._consistency(tr, 10)) / 15.0)
                else:
                    row.append(0.0)

                # First half margin avg last 10 (proxy: half margin)
                last10 = tr[-10:]
                avg_m10 = sum(r[2] for r in last10) / max(len(last10), 1)
                row.append(avg_m10 * 0.5 / 15.0)

                # Second half margin avg last 10 (proxy)
                row.append(avg_m10 * 0.5 / 15.0)

                # Half margin delta (proxy: 0)
                row.append(0.0)

                # ATS record (placeholder)
                row.append(0.5)

                # ATS trend last 10 (placeholder)
                row.append(0.5)

                # Over rate season (placeholder)
                row.append(0.5)

                # Point diff: close games vs all
                close_games = [r for r in tr if abs(r[2]) <= 5]
                close_pd = sum(r[2] for r in close_games) / max(len(close_games), 1)
                all_pd = sum(r[2] for r in tr) / max(n_games, 1)
                row.append((close_pd - all_pd) / 15.0)

                # Record after loss (resilience)
                after_loss = [tr[i] for i in range(1, len(tr)) if not tr[i - 1][1]]
                row.append(self._wp(after_loss, len(after_loss)) if after_loss else 0.5)

                # Record after win (consistency)
                after_win = [tr[i] for i in range(1, len(tr)) if tr[i - 1][1]]
                row.append(self._wp(after_win, len(after_win)) if after_win else 0.5)

                # Record after B2B
                after_b2b = [tr[i] for i in range(1, len(tr)) if self._game_rest(tr[i], tr[:i + 1]) <= 1]
                row.append(self._wp(after_b2b, len(after_b2b)) if after_b2b else 0.5)

                # Blowout bounce-back
                after_blow = [tr[i] for i in range(1, len(tr)) if not tr[i - 1][1] and tr[i - 1][2] <= -15]
                row.append(self._wp(after_blow, len(after_blow)) if after_blow else 0.5)

                # Overtime record (proxy: very close games)
                ot_proxy = [r for r in tr if abs(r[2]) <= 3]
                row.append(self._wp(ot_proxy, len(ot_proxy)) if ot_proxy else 0.5)

            # Game-level trajectory differentials (2 features)
            h_wp30 = self._wp(hr_, 30)
            a_wp30 = self._wp(ar_, 30)
            row.append(h_wp30 - a_wp30)

            h_pts_s = sum(r[4].get("pts", 100) for r in hr_)
            h_opp_s = sum(r[4].get("opp_pts", 100) for r in hr_)
            a_pts_s = sum(r[4].get("pts", 100) for r in ar_)
            a_opp_s = sum(r[4].get("opp_pts", 100) for r in ar_)
            _e = 13.91
            h_pyth = (h_pts_s ** _e) / ((h_pts_s ** _e) + (h_opp_s ** _e)) if h_pts_s > 0 else 0.5
            a_pyth = (a_pts_s ** _e) / ((a_pts_s ** _e) + (a_opp_s ** _e)) if a_pts_s > 0 else 0.5
            row.append(h_pyth - a_pyth)

            # ═══════════════════════════════════════════════════════════
            # CATEGORIES 19-23: REAL COMPUTATION (replacing zeros)
            # ═══════════════════════════════════════════════════════════
            _cat19_start = len(row)

            # 19. LINEUP & ROTATION (66 features) — zeros (needs player data)
            row.extend([0.0] * 66)

            # 20. GAME THEORY & META (50 features) — zeros (needs model history)
            row.extend([0.0] * 50)

            # ──────────────────────────────────────────────────────
            # 21. ENVIRONMENTAL & EXTERNAL (44 features) — COMPUTED
            # ──────────────────────────────────────────────────────
            for _pfx21, _tr21, _tk21 in [("h", hr_, home), ("a", ar_, away)]:
                _ngp21 = len(_tr21)
                _wp21 = self._wp(_tr21, _ngp21)

                # Conference standings
                _conf21 = self._conference(_tk21)
                _conf_wps21 = []
                for _t21, _r21 in team_results.items():
                    if _r21 and self._conference(_t21) == _conf21:
                        _conf_wps21.append((_t21, self._wp(_r21, len(_r21))))
                _conf_wps21.sort(key=lambda x: x[1], reverse=True)

                _rank21 = 15
                for _ir21, (_t21, _) in enumerate(_conf_wps21):
                    if _t21 == _tk21:
                        _rank21 = _ir21 + 1
                        break

                row.append(_rank21 / 15.0)                              # conf_standing_pct
                row.append(self._division(_tk21))                       # div_standing_pct
                row.append(max(0, (15 - _rank21)) / 15.0)             # lottery_odds_proxy
                row.append(1.0 if _rank21 >= 12 and _wp21 < 0.35 else 0.0)  # tank_indicator

                # Playoff urgency
                _urg = 0.0
                if 6 <= _rank21 <= 12:
                    _urg = 1.0 - abs(_rank21 - 9) / 6.0
                elif _rank21 <= 5:
                    _urg = 0.3
                row.append(_urg)                                        # playoff_urgency

                # Revenge game
                _opp21 = away if _pfx21 == "h" else home
                _h2h21 = h2h_results.get((_tk21, _opp21), []) + h2h_results.get((_opp21, _tk21), [])
                _rev = 0.0
                _rev_m = 0.0
                if _h2h21:
                    _last_h2h = _h2h21[-1]
                    _m_for_team = _last_h2h[2] if _last_h2h[4].get("team") == _tk21 else -_last_h2h[2]
                    if _m_for_team <= -15:
                        _rev = 1.0
                        _rev_m = abs(_m_for_team) / 30.0
                row.append(_rev)                                        # revenge_game
                row.append(_rev_m)                                      # revenge_intensity

                # Coach/roster/media data not available (9 features)
                row.extend([0.0] * 9)

            # Game-level environmental (12 features)
            _hrank_g = 15
            _arank_g = 15
            for _conf_g in [self._conference(home), self._conference(away)]:
                _cwps_g = []
                for _t, _recs in team_results.items():
                    if _recs and self._conference(_t) == _conf_g:
                        _cwps_g.append((_t, self._wp(_recs, len(_recs))))
                _cwps_g.sort(key=lambda x: x[1], reverse=True)
                for _ir, (_t, _) in enumerate(_cwps_g):
                    if _t == home and self._conference(home) == _conf_g:
                        _hrank_g = _ir + 1
                    if _t == away and self._conference(away) == _conf_g:
                        _arank_g = _ir + 1

            row.append((_hrank_g - _arank_g) / 15.0)                   # conf_standing_diff
            row.append(self._division(home) - self._division(away))    # div_standing_diff
            row.append(1.0 if _hrank_g <= 10 and _arank_g <= 10 else 0.0)  # both_playoff_contenders
            _hwp_g = self._wp(hr_, len(hr_))
            _awp_g = self._wp(ar_, len(ar_))
            row.append(1.0 if (_hrank_g >= 12 and _hwp_g < 0.35 and
                               _arank_g >= 12 and _awp_g < 0.35) else 0.0)  # both_tanking
            _fav = max(_hwp_g, _awp_g)
            _dog = min(_hwp_g, _awp_g)
            row.append(_dog / max(_fav, 0.01))                          # upset_potential
            _rival21 = 1.0 if self._division(home) == self._division(away) else 0.0
            _h2h_ct = len(h2h_results.get((home, away), []) + h2h_results.get((away, home), []))
            row.append(_rival21 * 0.5 + min(_h2h_ct / 10.0, 0.5))    # rivalry_intensity
            row.extend([0.0, 0.0])                                      # public_side_home, contrarian (no data)
            row.append(sp * self._travel_dist(hr_, home) / 3000.0)     # weather_travel_factor
            row.append((ARENA_ALTITUDE.get(home, 500) - ARENA_ALTITUDE.get(away, 500)) *
                       self._travel_dist(ar_, away) / max(5280 * 3000, 1))  # altitude_fatigue_compound
            _tz_sh = abs(TIMEZONE_ET.get(away, 0) - TIMEZONE_ET.get(self._last_location(ar_), 0))
            row.append(_tz_sh * (1.0 if a_rest <= 1 else 0.3))        # timezone_circadian_impact
            row.append(0.0)                                              # arena_noise_factor (no data)

            # ──────────────────────────────────────────────────────
            # 22. CROSS-WINDOW MOMENTUM (630 features) — COMPUTED
            # ──────────────────────────────────────────────────────
            _cw_stats = ["wp", "ppg", "margin", "ortg", "drtg",
                         "efg", "ts", "pace", "papg", "pd"]
            _cw_pairs = [
                (3, 5), (3, 7), (3, 10), (3, 15), (3, 20),
                (5, 7), (5, 10), (5, 15), (5, 20),
                (7, 10), (7, 15), (7, 20),
                (10, 15), (10, 20),
                (15, 20),
            ]
            _cw_windows = [3, 5, 7, 10, 15, 20]
            _CW_FN = {
                "wp": self._wp, "ppg": self._ppg, "margin": self._avg_margin,
                "ortg": self._ortg, "drtg": self._drtg, "efg": self._efg,
                "ts": self._ts, "pace": self._pace, "papg": self._papg,
                "pd": self._pd,
            }

            # Pre-compute all stat values at all windows for both teams
            _cw_cache = {}
            for _pfx22, _tr22 in [("h", hr_), ("a", ar_)]:
                _sv22 = {}
                for _s in _cw_stats:
                    _fn = _CW_FN[_s]
                    for _w in _cw_windows:
                        _sv22[(_s, _w)] = _fn(_tr22, _w)
                _cw_cache[_pfx22] = _sv22

            # Delta/acceleration features (order: h then a, matching feature names)
            for _pfx22 in ["h", "a"]:
                _sv22 = _cw_cache[_pfx22]
                for _s in _cw_stats:
                    for _w1, _w2 in _cw_pairs:
                        _d = _sv22[(_s, _w1)] - _sv22[(_s, _w2)]
                        row.append(_d)                                  # delta
                        row.append(_d / max(_w2 - _w1, 1))            # acceleration

            # Composite features (15 per team, order: h then a)
            for _pfx22 in ["h", "a"]:
                _sv22 = _cw_cache[_pfx22]

                # Collect deltas by stat
                _wp_d22 = [(_w1, _w2, _sv22[("wp", _w1)] - _sv22[("wp", _w2)]) for _w1, _w2 in _cw_pairs]
                _margin_d22 = [_sv22[("margin", _w1)] - _sv22[("margin", _w2)] for _w1, _w2 in _cw_pairs]
                _ortg_d22 = [_sv22[("ortg", _w1)] - _sv22[("ortg", _w2)] for _w1, _w2 in _cw_pairs]
                _drtg_d22 = [_sv22[("drtg", _w1)] - _sv22[("drtg", _w2)] for _w1, _w2 in _cw_pairs]
                _wp_vals = [d for _, _, d in _wp_d22]
                _short_wp = [d for w1, _, d in _wp_d22 if w1 <= 5]
                _long_wp = [d for _, w2, d in _wp_d22 if w2 >= 15]

                # 1. shortterm_trend
                row.append(sum(_short_wp) / max(len(_short_wp), 1))
                # 2. longterm_trend
                row.append(sum(_long_wp) / max(len(_long_wp), 1))
                # 3. margin_volatility_trend
                _md_m = sum(_margin_d22) / max(len(_margin_d22), 1)
                _md_s = (sum((_d - _md_m)**2 for _d in _margin_d22) / max(len(_margin_d22), 1)) ** 0.5
                row.append(_md_s / 10.0)
                # 4. ortg_improvement_rate
                row.append(sum(_ortg_d22) / max(len(_ortg_d22), 1) / 5.0)
                # 5. drtg_improvement_rate (negative = improving defense)
                row.append(-sum(_drtg_d22) / max(len(_drtg_d22), 1) / 5.0)
                # 6. overall_trajectory
                _wp_avg = sum(_wp_vals) / max(len(_wp_vals), 1)
                _mg_avg = sum(_margin_d22) / max(len(_margin_d22), 1)
                _or_avg = sum(_ortg_d22) / max(len(_ortg_d22), 1)
                row.append(0.4 * _wp_avg + 0.3 * _mg_avg / 10.0 + 0.3 * _or_avg / 5.0)
                # 7. form_acceleration
                _vshort = [d for w1, _, d in _wp_d22 if w1 == 3]
                _med22 = [d for w1, _, d in _wp_d22 if w1 == 7]
                row.append(sum(_vshort) / max(len(_vshort), 1) -
                          sum(_med22) / max(len(_med22), 1))
                # 8. peak_window
                _wp_at = [_sv22[("wp", _w)] for _w in _cw_windows]
                row.append(_wp_at.index(max(_wp_at)) / 5.0)
                # 9. trough_window
                row.append(_wp_at.index(min(_wp_at)) / 5.0)
                # 10. consistency_across_windows
                _wm22 = sum(_wp_at) / len(_wp_at)
                row.append((sum((_v - _wm22)**2 for _v in _wp_at) / len(_wp_at)) ** 0.5)
                # 11. trend_agreement
                _signs = [1 if d > 0.01 else (-1 if d < -0.01 else 0) for d in _wp_vals]
                _nz = [s for s in _signs if s != 0]
                row.append(abs(sum(_nz)) / max(len(_nz), 1) if _nz else 0.0)
                # 12. breakout_signal
                _sh_a = sum(_short_wp) / max(len(_short_wp), 1)
                _lg_a = sum(_long_wp) / max(len(_long_wp), 1)
                row.append(max(0, _sh_a - _lg_a) * 5.0)
                # 13. decline_signal
                row.append(max(0, _lg_a - _sh_a) * 5.0)
                # 14. mean_reversion_signal
                row.append(abs(_sv22[("wp", 5)] - 0.5) - abs(_sv22[("wp", 20)] - 0.5))
                # 15. momentum_strength
                row.append(abs(_sv22[("wp", 3)] - _sv22[("wp", 20)]))

            # 23. MARKET II (62 features if market) — zeros (needs multi-book data)
            if self.include_market:
                row.extend([0.0] * 62)

            # Safety: pad/truncate to match expected cat 19-23 count
            _expected_19_23 = self._cat_bounds.get(24, _cat25_start) - self._cat_bounds.get(19, _cat25_start)
            _actual_19_23 = len(row) - _cat19_start
            if _actual_19_23 < _expected_19_23:
                row.extend([0.0] * (_expected_19_23 - _actual_19_23))
            elif _actual_19_23 > _expected_19_23:
                del row[_cat19_start + _expected_19_23:]

            # ================================================================
            # 24. POWER RATING COMPOSITES (64 features) — REAL COMPUTATION
            # ================================================================
            for prefix, team_key in [("h", home), ("a", away)]:
                tr = team_results[team_key]
                n_gp = len(tr)

                # Standard ELO (normalized)
                row.append((team_elo[team_key] - 1500.0) / 400.0)

                # Margin-adjusted ELO
                row.append((team_elo_margin[team_key] - 1500.0) / 400.0)

                # Recency-weighted ELO
                row.append((team_elo_recency[team_key] - 1500.0) / 400.0)

                # Home-court adjusted ELO
                hca_bonus = 0.0
                if team_home_games_count[team_key] > 0:
                    hca_bonus = team_home_margin_sum[team_key] / team_home_games_count[team_key]
                row.append((team_elo[team_key] + hca_bonus * 2.0 - 1500.0) / 400.0)

                # SOS-adjusted ELO
                sos_val = self._sos(tr, team_results, min(n_gp, 20))
                row.append((team_elo[team_key] * (0.5 + sos_val) - 1500.0) / 400.0)

                # Conference-adjusted ELO
                conf_elos = [team_elo[t] for t in team_results
                             if team_results[t] and self._conference(t) == self._conference(team_key)]
                conf_avg_elo = sum(conf_elos) / max(len(conf_elos), 1)
                row.append((team_elo[team_key] - conf_avg_elo) / 200.0)

                # Pace-adjusted ELO
                pace_val = self._pace(tr, 10) if tr else 100.0
                row.append((team_elo[team_key] - 1500.0) * (pace_val / 100.0) / 400.0)

                # RAPTOR composite (offensive + defensive ELO blend)
                off_elo = team_elo_offense[team_key]
                def_elo = team_elo_defense[team_key]
                raptor_comp = (off_elo + def_elo) / 2.0
                row.append((raptor_comp - 1500.0) / 400.0)

                # RAPTOR offense
                row.append((off_elo - 1500.0) / 400.0)

                # RAPTOR defense
                row.append((def_elo - 1500.0) / 400.0)

                # Power rank by offense (ORtg season)
                ortg_s = self._ortg(tr, n_gp) if tr else 100.0
                row.append(ortg_s / 120.0)

                # Power rank by defense (DRtg season, lower=better)
                drtg_s = self._drtg(tr, n_gp) if tr else 110.0
                row.append(1.0 - drtg_s / 120.0)

                # Power rank by net rating
                netrtg_s = ortg_s - drtg_s
                row.append(netrtg_s / 20.0)

                # SRS (Simple Rating System)
                avg_m = self._pd(tr, n_gp) if tr else 0.0
                row.append((avg_m + (sos_val - 0.5) * 10) / 20.0)

                # Composite power rank
                composite = (
                    (team_elo[team_key] - 1500) / 400 * 0.3 +
                    netrtg_s / 20 * 0.3 +
                    (raptor_comp - 1500) / 400 * 0.2 +
                    (avg_m + (sos_val - 0.5) * 10) / 20 * 0.2
                )
                row.append(composite)

                # Power rank trend (ELO change over last 10 games)
                elo_hist = team_elo_history.get(team_key, [])
                if len(elo_hist) >= 10:
                    row.append((elo_hist[-1] - elo_hist[-10]) / 100.0)
                elif len(elo_hist) >= 2:
                    row.append((elo_hist[-1] - elo_hist[0]) / 100.0)
                else:
                    row.append(0.0)

                # Power conf adjusted
                row.append((team_elo[team_key] - conf_avg_elo + netrtg_s) / 30.0)

                # Power stability (std dev of ELO last 20)
                if len(elo_hist) >= 5:
                    recent_elo = elo_hist[-20:]
                    elo_m = sum(recent_elo) / len(recent_elo)
                    elo_std = (sum((e - elo_m) ** 2 for e in recent_elo) / len(recent_elo)) ** 0.5
                    row.append(elo_std / 100.0)
                else:
                    row.append(0.5)

                # Power percentile in league
                all_elos = [team_elo[t] for t in team_results if team_results[t]]
                if all_elos:
                    row.append(sum(1 for e in all_elos if e <= team_elo[team_key]) / len(all_elos))
                else:
                    row.append(0.5)

                # Rating confidence (games played)
                row.append(min(1.0, n_gp / 30.0))

                # Bayesian rating
                prior_w = 10.0
                bayesian = (prior_w * 1500.0 + n_gp * team_elo[team_key]) / (prior_w + max(n_gp, 1))
                row.append((bayesian - 1500.0) / 400.0)

                # Glicko-style rating
                row.append((team_elo[team_key] - 1500.0) / 400.0)

                # Glicko RD (uncertainty)
                glicko_rd = max(30.0, 350.0 - max(n_gp, 1) * 5.0)
                row.append(glicko_rd / 350.0)

                # TrueSkill mu
                row.append((team_elo[team_key] - 1500.0) / 400.0)

                # TrueSkill sigma
                row.append(max(0.1, 1.0 - max(n_gp, 1) / 82.0))

            # Pairwise power differentials (14 features)
            h_elo_s = team_elo[home]
            a_elo_s = team_elo[away]
            row.append((h_elo_s - a_elo_s) / 400.0)                                     # elo_std_diff
            row.append((team_elo_margin[home] - team_elo_margin[away]) / 400.0)          # elo_margin_diff
            row.append((team_elo_recency[home] - team_elo_recency[away]) / 400.0)        # elo_recency_diff
            h_rap = (team_elo_offense[home] + team_elo_defense[home]) / 2.0
            a_rap = (team_elo_offense[away] + team_elo_defense[away]) / 2.0
            row.append((h_rap - a_rap) / 400.0)                                          # raptor_diff
            row.append((team_elo_offense[home] - team_elo_offense[away]) / 400.0)        # raptor_off_diff
            row.append((team_elo_defense[home] - team_elo_defense[away]) / 400.0)        # raptor_def_diff
            h_srs = self._pd(hr_, len(hr_)) + (self._sos(hr_, team_results, 20) - 0.5) * 10
            a_srs = self._pd(ar_, len(ar_)) + (self._sos(ar_, team_results, 20) - 0.5) * 10
            row.append((h_srs - a_srs) / 20.0)                                          # srs_diff
            h_comp = ((h_elo_s - 1500) / 400 * 0.3 + self._netrtg(hr_, len(hr_)) / 20 * 0.3 +
                      (h_rap - 1500) / 400 * 0.2 + h_srs / 20 * 0.2)
            a_comp = ((a_elo_s - 1500) / 400 * 0.3 + self._netrtg(ar_, len(ar_)) / 20 * 0.3 +
                      (a_rap - 1500) / 400 * 0.2 + a_srs / 20 * 0.2)
            row.append(h_comp - a_comp)                                                  # composite_diff
            h_ca = [team_elo[t] for t in team_results
                    if team_results[t] and self._conference(t) == self._conference(home)]
            a_ca = [team_elo[t] for t in team_results
                    if team_results[t] and self._conference(t) == self._conference(away)]
            h_ca_avg = sum(h_ca) / max(len(h_ca), 1)
            a_ca_avg = sum(a_ca) / max(len(a_ca), 1)
            row.append(((h_elo_s - h_ca_avg) - (a_elo_s - a_ca_avg)) / 200.0)           # conf_adj_diff
            h_gp_d = max(len(hr_), 1)
            a_gp_d = max(len(ar_), 1)
            h_bay = (10 * 1500 + h_gp_d * h_elo_s) / (10 + h_gp_d)
            a_bay = (10 * 1500 + a_gp_d * a_elo_s) / (10 + a_gp_d)
            row.append((h_bay - a_bay) / 400.0)                                         # bayesian_diff
            row.append((h_elo_s - a_elo_s) / 400.0)                                     # glicko_diff
            row.append((h_elo_s - a_elo_s) / 400.0)                                     # trueskill_diff
            _all_diffs = [
                (h_elo_s - a_elo_s) / 400.0,
                (team_elo_margin[home] - team_elo_margin[away]) / 400.0,
                (team_elo_recency[home] - team_elo_recency[away]) / 400.0,
                (h_rap - a_rap) / 400.0,
            ]
            row.append(max(_all_diffs, key=abs))                                         # max_diff
            row.append(sum(_all_diffs) / len(_all_diffs))                                # avg_diff

            # 25. FATIGUE & LOAD MANAGEMENT
            for prefix, tr, team_key in [("h", hr_, home), ("a", ar_, away)]:
                # Cumulative games played this season
                cumul_gp = len(tr)
                row.append(cumul_gp / 82.0)

                # Cumulative minutes total (proxy: games × 48)
                row.append(cumul_gp * 48.0 / (82.0 * 48.0))

                # Avg minutes per game (proxy: constant 48 team mins)
                row.append(48.0 / 48.0)

                # Star minutes cumulative (proxy from player data)
                pd_ = (player_data or {}).get(team_key, {})
                star_min = pd_.get("star_minutes_load", 34.0) * cumul_gp
                row.append(star_min / (82.0 * 40.0))  # Normalized

                # Star minutes as pct of season capacity
                row.append(star_min / max(82.0 * 40.0, 1))

                # Travel miles this season
                season_miles = self._total_miles_season(tr, team_key)
                row.append(season_miles / 50000.0)  # Normalize

                # Travel miles last 30 days
                row.append(self._miles_in_window(tr, gd, 30, team_key) / 15000.0)

                # Travel miles last 7 days
                row.append(self._miles_in_window(tr, gd, 7, team_key) / 5000.0)

                # Travel intensity: miles per game last 10
                miles_10g = self._miles_last_n_games(tr, 10, team_key)
                games_10 = min(len(tr), 10)
                row.append((miles_10g / max(games_10, 1)) / 1000.0)

                # Rest pattern consistency: std of rest days between last 10 games
                rest_days_list = self._recent_rest_days(tr, 10)
                if len(rest_days_list) >= 2:
                    rm = sum(rest_days_list) / len(rest_days_list)
                    rest_std = (sum((r - rm) ** 2 for r in rest_days_list) / len(rest_days_list)) ** 0.5
                else:
                    rest_std = 1.0
                row.append(rest_std / 3.0)

                # Rest deficit vs league avg (~1.2 days between games)
                avg_rest = sum(rest_days_list) / len(rest_days_list) if rest_days_list else 1.5
                row.append((avg_rest - 1.2) / 2.0)

                # B2B count this season
                b2b_season = self._count_b2b_in_window(tr, gd, 300)
                row.append(b2b_season / 20.0)

                # B2B count last 30 days
                b2b_30d = self._count_b2b_in_window(tr, gd, 30)
                row.append(b2b_30d / 5.0)

                # 3-in-4 count this season
                three_in_4 = self._count_dense_stretches(tr, gd, 300, 3, 4)
                row.append(three_in_4 / 15.0)

                # Dense schedule flag: 4+ games in last 7 days
                g7 = self._games_in_window(tr, gd, 7)
                row.append(1.0 if g7 >= 4 else 0.0)

                # Road trip length (consecutive away games)
                road_len = self._consecutive_away(tr)
                row.append(road_len / 7.0)

                # Home stand length (consecutive home games)
                home_len = self._consecutive_home(tr)
                row.append(home_len / 7.0)

                # Road trip fatigue: road games × avg travel
                row.append(road_len * (miles_10g / max(games_10, 1)) / 5000.0)

                # Load management probability (high fatigue + star minutes)
                this_rest = self._rest_days(team_key, gd, team_last)
                load_mgmt = 0.0
                if this_rest <= 1 and star_min / max(cumul_gp, 1) > 36:
                    load_mgmt = 0.7
                elif g7 >= 4:
                    load_mgmt = 0.4
                elif cumul_gp > 70:
                    load_mgmt = 0.3
                row.append(load_mgmt)

                # Season fatigue curve: expected dropoff based on game number
                row.append(max(0.0, (cumul_gp - 50) / 82.0))

                # Relative fatigue vs league avg (~41 games at midseason)
                # Approximate league avg as season_pct * 82
                row.append((cumul_gp - 41 * sp) / 20.0 if sp > 0 else 0.0)

                # Fatigue-adjusted ORtg
                fatigue_penalty = 0.02 * max(0, g7 - 3) + 0.01 * (1 if this_rest <= 1 else 0)
                ortg_val = self._ortg(tr, 10)
                row.append(ortg_val * (1.0 - fatigue_penalty) / 110.0)

                # Fatigue-adjusted DRtg (higher = worse defense when tired)
                drtg_val = self._drtg(tr, 10)
                row.append(drtg_val * (1.0 + fatigue_penalty * 0.5) / 110.0)

                # Fatigue-adjusted win%
                wp_val = self._wp(tr, 10)
                row.append(wp_val * (1.0 - fatigue_penalty))

                # Recovery quality: win% when rested 2+ days (historical)
                rested_games = [r for r in tr[-30:] if self._game_rest(r, tr) >= 2]
                row.append(self._wp(rested_games, len(rested_games)) if rested_games else 0.5)

                # B2B performance drop
                b2b_games = [r for r in tr[-30:] if self._game_rest(r, tr) <= 1]
                non_b2b = [r for r in tr[-30:] if self._game_rest(r, tr) > 1]
                b2b_margin = self._pd(b2b_games, len(b2b_games)) if b2b_games else 0.0
                non_b2b_margin = self._pd(non_b2b, len(non_b2b)) if non_b2b else 0.0
                row.append((b2b_margin - non_b2b_margin) / 10.0)

                # Altitude fatigue cumulative (games at high altitude recently)
                high_alt_games = sum(1 for r in tr[-10:]
                                     if ARENA_ALTITUDE.get(r[3], 500) > 3000)
                row.append(high_alt_games / 10.0)

                # Timezone changes this season
                tz_changes = self._count_tz_changes(tr)
                row.append(tz_changes / 40.0)

                # Circadian disruption (recent timezone impact)
                row.append(abs(TIMEZONE_ET.get(team_key, 0) -
                              TIMEZONE_ET.get(self._last_location(tr), 0)) / 3.0)

                # Early season load: heavy in first 30 games
                early_games = [r for r in tr[:30]]
                early_b2b = sum(1 for i in range(1, len(early_games))
                               if self._days_between(early_games[i-1][0], early_games[i][0]) <= 1)
                row.append(early_b2b / 10.0 if len(early_games) >= 20 else 0.0)

                # Late season load: games 60+
                late_games = tr[60:] if len(tr) > 60 else []
                late_density = len(late_games) / max(1, (cumul_gp - 60)) if cumul_gp > 60 else 0.0
                row.append(late_density)

                # Minutes distribution health (proxy: consistency of scoring)
                scoring_std = self._consistency(tr, 10) / 15.0
                row.append(max(0, 1.0 - scoring_std))

                # Injury risk score (fatigue-based proxy)
                injury_risk = (g7 / 5.0) * 0.3 + (1 if this_rest <= 1 else 0) * 0.3 + \
                             (cumul_gp / 82.0) * 0.2 + (star_min / max(cumul_gp * 40, 1)) * 0.2
                row.append(min(1.0, injury_risk))

                # Stamina rating: Q4 vs Q1 performance proxy (use margin trend)
                row.append(self._wp(tr[-5:], 5) - self._wp(tr[-15:], 15) + 0.5)

                # Clutch fatigue: performance in close games on heavy schedule
                recent_close = [r for r in tr[-10:] if abs(r[2]) <= 5]
                row.append(self._wp(recent_close, len(recent_close)) if recent_close else 0.5)

                # Fresh vs tired ratio
                rested_wp = self._wp(rested_games, len(rested_games)) if rested_games else 0.5
                tired_wp = self._wp(b2b_games, len(b2b_games)) if b2b_games else 0.5
                row.append(rested_wp / max(tired_wp, 0.1))

                # Optimal rest indicator: 2-3 days rest, no recent travel
                row.append(1.0 if 2 <= this_rest <= 3 and g7 <= 3 else 0.0)

                # Wear and tear index: composite fatigue accumulation
                wear = (cumul_gp / 82.0) * 0.25 + \
                       (b2b_30d / 5.0) * 0.2 + \
                       (g7 / 5.0) * 0.2 + \
                       (season_miles / 50000.0) * 0.15 + \
                       (1 if this_rest <= 1 else 0) * 0.2
                row.append(min(1.0, wear))

            # Game-level fatigue differentials (8 features)
            # Access the per-team fatigue values just computed
            # Home team values start at _cat25_start, away at _cat25_start + 38
            _h25 = _cat25_start
            _a25 = _cat25_start + 38  # 38 per-team features
            row.append(row[_h25] - row[_a25])                      # cumul_games diff
            row.append(row[_h25 + 5] - row[_a25 + 5])              # travel_season diff
            row.append(row[_h25 + 9] - row[_a25 + 9])              # rest_consistency diff
            row.append(row[_h25 + 37] - row[_a25 + 37])            # wear_tear diff
            row.append(row[_h25 + 4] - row[_a25 + 4])              # star_minutes_pct diff
            row.append(row[_h25 + 12] - row[_a25 + 12])            # b2b_count_30d diff
            # Fatigue-adjusted spread
            mkt_spread = _val("current_spread")
            h_wear = row[_h25 + 37]
            a_wear = row[_a25 + 37]
            row.append(mkt_spread + (h_wear - a_wear) * 2.0)       # fatigue-adjusted spread
            # Fatigue composite edge
            row.append((a_wear - h_wear) * 0.5 + (row[_a25 + 9] - row[_h25 + 9]) * 0.3 +
                       (row[_a25 + 12] - row[_h25 + 12]) * 0.2)   # composite edge

            X.append(row)
            y.append(1 if hs > as_ else 0)

            # Record this game
            self._record_game(team_results, team_last, team_elo,
                              team_home_results, team_away_results,
                              h2h_results, home, away, hs, as_, gd,
                              h_stats, a_stats)
            # Update multi-ELO systems (Cat 24)
            self._update_multi_elo(
                home, away, hs, as_, h_stats, a_stats,
                team_elo_margin, team_elo_offense, team_elo_defense,
                team_elo_recency, team_elo_history,
                team_home_margin_sum, team_home_games_count)

        X = np.nan_to_num(np.array(X, dtype=np.float64))
        y = np.array(y, dtype=np.int32)

        # Verify dimensions
        expected = len(self.feature_names)
        if X.shape[1] != expected:
            print(f"WARNING: Expected {expected} features, got {X.shape[1]}")
            # Pad or truncate
            if X.shape[1] < expected:
                pad = np.zeros((X.shape[0], expected - X.shape[1]))
                X = np.hstack([X, pad])
            else:
                X = X[:, :expected]

        return X, y, self.feature_names

    # ── Helper methods ──

    def _wp(self, records, n):
        s = records[-n:]
        return sum(1 for x in s if x[1]) / len(s) if s else 0.5

    def _pd(self, records, n):
        s = records[-n:]
        return sum(x[2] for x in s) / len(s) if s else 0.0

    def _ppg(self, records, n):
        s = records[-n:]
        return sum(x[4].get("pts", 100) for x in s) / len(s) if s else 100.0

    def _papg(self, records, n):
        s = records[-n:]
        return sum(x[4].get("opp_pts", 100) for x in s) / len(s) if s else 100.0

    def _avg_margin(self, records, n):
        return self._ppg(records, n) - self._papg(records, n)

    def _close_pct(self, records, n):
        s = records[-n:]
        if not s:
            return 0.5
        return sum(1 for x in s if abs(x[2]) <= 5) / len(s)

    def _blowout_pct(self, records, n):
        s = records[-n:]
        if not s:
            return 0.0
        return sum(1 for x in s if abs(x[2]) >= 15) / len(s)

    def _ou_avg(self, records, n):
        s = records[-n:]
        return sum(x[4].get("pts", 100) + x[4].get("opp_pts", 100) for x in s) / len(s) if s else 200.0

    def _streak(self, records):
        if not records:
            return 0
        s = 0
        last = records[-1][1]
        for x in reversed(records):
            if x[1] == last:
                s += 1
            else:
                break
        return s if last else -s

    def _efg(self, records, n):
        return self._stat_avg(records, n, "efg_pct")

    def _tov_rate(self, records, n):
        return self._stat_avg(records, n, "tov_rate")

    def _orb_rate(self, records, n):
        return self._stat_avg(records, n, "oreb_pct")

    def _ft_rate(self, records, n):
        return self._stat_avg(records, n, "ft_rate")

    def _opp_efg(self, records, n):
        return self._stat_avg(records, n, "opp_efg_pct")

    def _opp_tov_rate(self, records, n):
        return self._stat_avg(records, n, "opp_tov_rate")

    def _opp_orb_rate(self, records, n):
        return self._stat_avg(records, n, "opp_oreb_pct")

    def _opp_ft_rate(self, records, n):
        return self._stat_avg(records, n, "opp_ft_rate")

    def _ortg(self, records, n):
        return self._stat_avg(records, n, "ortg")

    def _drtg(self, records, n):
        return self._stat_avg(records, n, "drtg")

    def _netrtg(self, records, n):
        return self._ortg(records, n) - self._drtg(records, n)

    def _pace(self, records, n):
        return self._stat_avg(records, n, "pace")

    def _ts(self, records, n):
        return self._stat_avg(records, n, "ts_pct")

    def _avg_poss(self, records, n):
        return self._stat_avg(records, n, "poss")

    def _ast_rate(self, records, n):
        return self._stat_avg(records, n, "ast_rate")

    def _stl_rate(self, records, n):
        return self._stat_avg(records, n, "stl_rate")

    def _blk_rate(self, records, n):
        return self._stat_avg(records, n, "blk_rate")

    def _tov_pct(self, records, n):
        return self._stat_avg(records, n, "tov_pct")

    def _oreb_pct(self, records, n):
        return self._stat_avg(records, n, "oreb_pct")

    def _dreb_pct(self, records, n):
        return self._stat_avg(records, n, "dreb_pct")

    def _stat_avg(self, records, n, key):
        s = records[-n:]
        if not s:
            return 0.0
        vals = [x[4].get(key, 0) for x in s if key in x[4]]
        return sum(vals) / len(vals) if vals else 0.0

    def _rest_days(self, team, game_date, team_last):
        last = team_last.get(team)
        if not last or not game_date:
            return 3
        try:
            d1 = datetime.strptime(game_date[:10], "%Y-%m-%d")
            d2 = datetime.strptime(last[:10], "%Y-%m-%d")
            return max(0, (d1 - d2).days)
        except:
            return 3

    def _n_in_m(self, records, game_date, n, m):
        """Return 1.0 if team played n games in last m days."""
        if not game_date:
            return 0.0
        try:
            gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            count = sum(1 for r in records[-10:]
                       if (gd - datetime.strptime(r[0][:10], "%Y-%m-%d")).days <= m)
            return 1.0 if count >= n else 0.0
        except:
            return 0.0

    def _travel_dist(self, records, team):
        """Distance from last game location to current arena."""
        if not records:
            return 0
        last_opp = records[-1][3]
        # If last game was home, team is at own arena
        last_loc = last_opp if not records[-1][4].get("is_home", False) else team
        if last_loc not in ARENA_COORDS or team not in ARENA_COORDS:
            return 0
        c1 = ARENA_COORDS[last_loc]
        c2 = ARENA_COORDS[team]
        return haversine(c1[0], c1[1], c2[0], c2[1])

    def _last_location(self, records):
        if not records:
            return "ATL"
        return records[-1][3] if not records[-1][4].get("is_home", False) else records[-1][3]

    def _games_in_window(self, records, game_date, days):
        if not game_date or not records:
            return 0
        try:
            gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            return sum(1 for r in records[-15:]
                      if (gd - datetime.strptime(r[0][:10], "%Y-%m-%d")).days <= days
                      and (gd - datetime.strptime(r[0][:10], "%Y-%m-%d")).days > 0)
        except:
            return 0

    def _miles_in_window(self, records, game_date, days, team):
        """Total travel miles in the last N days."""
        if not records or not game_date:
            return 0
        try:
            gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            total = 0
            for i, r in enumerate(records[-15:]):
                rd = datetime.strptime(r[0][:10], "%Y-%m-%d")
                if 0 < (gd - rd).days <= days:
                    # Distance from previous game
                    if i > 0:
                        prev = records[-15:][i-1]
                        loc1 = prev[3]
                        loc2 = r[3]
                        if loc1 in ARENA_COORDS and loc2 in ARENA_COORDS:
                            total += haversine(*ARENA_COORDS[loc1], *ARENA_COORDS[loc2])
            return total
        except:
            return 0

    def _fatigue_score(self, records, game_date, team, rest):
        """Composite fatigue: games + travel + rest."""
        g7 = self._games_in_window(records, game_date, 7)
        m7 = self._miles_in_window(records, game_date, 7, team) / 1000  # Scale
        b2b = 1 if rest <= 1 else 0
        return g7 * 0.3 + m7 * 0.3 + b2b * 0.4

    def _total_miles_season(self, records, team):
        """Total travel miles for the season (approximate from game locations)."""
        total = 0.0
        prev_loc = team  # Start at home
        for r in records:
            game_loc = team if r[4].get("is_home", False) else r[3]
            if prev_loc in ARENA_COORDS and game_loc in ARENA_COORDS:
                total += haversine(*ARENA_COORDS[prev_loc], *ARENA_COORDS[game_loc])
            prev_loc = game_loc
        return total

    def _miles_last_n_games(self, records, n, team):
        """Travel miles across last n games."""
        recent = records[-n:]
        total = 0.0
        prev_loc = team
        for r in recent:
            game_loc = team if r[4].get("is_home", False) else r[3]
            if prev_loc in ARENA_COORDS and game_loc in ARENA_COORDS:
                total += haversine(*ARENA_COORDS[prev_loc], *ARENA_COORDS[game_loc])
            prev_loc = game_loc
        return total

    def _recent_rest_days(self, records, n):
        """List of rest days between last n games."""
        recent = records[-n:]
        rests = []
        for i in range(1, len(recent)):
            days = self._days_between(recent[i-1][0], recent[i][0])
            rests.append(max(0, days))
        return rests

    def _days_between(self, date1, date2):
        """Days between two date strings."""
        try:
            d1 = datetime.strptime(date1[:10], "%Y-%m-%d")
            d2 = datetime.strptime(date2[:10], "%Y-%m-%d")
            return abs((d2 - d1).days)
        except:
            return 2

    def _count_b2b_in_window(self, records, game_date, days):
        """Count back-to-back instances in the last N days."""
        if not records or not game_date:
            return 0
        try:
            gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            recent = [r for r in records if (gd - datetime.strptime(r[0][:10], "%Y-%m-%d")).days <= days
                      and (gd - datetime.strptime(r[0][:10], "%Y-%m-%d")).days > 0]
            count = 0
            for i in range(1, len(recent)):
                if self._days_between(recent[i-1][0], recent[i][0]) <= 1:
                    count += 1
            return count
        except:
            return 0

    def _count_dense_stretches(self, records, game_date, window_days, n_games, n_days):
        """Count how many times team played n_games in n_days within window."""
        if not records or not game_date:
            return 0
        try:
            gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            dates = []
            for r in records:
                rd = datetime.strptime(r[0][:10], "%Y-%m-%d")
                if 0 < (gd - rd).days <= window_days:
                    dates.append(rd)
            dates.sort()
            count = 0
            for i in range(len(dates) - n_games + 1):
                span = (dates[i + n_games - 1] - dates[i]).days
                if span <= n_days:
                    count += 1
            return count
        except:
            return 0

    def _consecutive_away(self, records):
        """Count current consecutive away games (from most recent)."""
        count = 0
        for r in reversed(records):
            if not r[4].get("is_home", False):
                count += 1
            else:
                break
        return count

    def _consecutive_home(self, records):
        """Count current consecutive home games (from most recent)."""
        count = 0
        for r in reversed(records):
            if r[4].get("is_home", False):
                count += 1
            else:
                break
        return count

    def _game_rest(self, game_record, all_records):
        """Get rest days before a specific game record."""
        idx = None
        for i, r in enumerate(all_records):
            if r[0] == game_record[0]:
                idx = i
                break
        if idx is None or idx == 0:
            return 3
        return self._days_between(all_records[idx - 1][0], game_record[0])

    def _count_tz_changes(self, records):
        """Count timezone changes across the season."""
        if len(records) < 2:
            return 0
        changes = 0
        for i in range(1, len(records)):
            loc1 = records[i-1][3] if not records[i-1][4].get("is_home", False) else records[i-1][4].get("team", "ATL")
            loc2 = records[i][3] if not records[i][4].get("is_home", False) else records[i][4].get("team", "ATL")
            tz1 = TIMEZONE_ET.get(loc1, 0)
            tz2 = TIMEZONE_ET.get(loc2, 0)
            if tz1 != tz2:
                changes += 1
        return changes

    def _sos(self, records, all_results, n):
        rec = records[-n:]
        if not rec:
            return 0.5
        opp_wps = []
        for r in rec:
            opp = r[3]
            if all_results[opp]:
                opp_wps.append(self._wp(all_results[opp], 82))
        return sum(opp_wps) / len(opp_wps) if opp_wps else 0.5

    def _wp_vs_quality(self, records, all_results, above=True):
        if not records:
            return 0.5
        relevant = []
        for r in records:
            opp_wp = self._wp(all_results[r[3]], 82)
            if (above and opp_wp > 0.5) or (not above and opp_wp <= 0.5):
                relevant.append(r)
        return self._wp(relevant, len(relevant)) if relevant else 0.5

    def _wp_vs_topN(self, records, all_results, n, top=True):
        if not records:
            return 0.5
        # Get top/bottom N teams by win%
        team_wps = {t: self._wp(r, 82) for t, r in all_results.items() if r}
        sorted_teams = sorted(team_wps.items(), key=lambda x: x[1], reverse=True)
        if top:
            target = {t for t, _ in sorted_teams[:n]}
        else:
            target = {t for t, _ in sorted_teams[-n:]}
        relevant = [r for r in records if r[3] in target]
        return self._wp(relevant, len(relevant)) if relevant else 0.5

    def _pd_vs_topN(self, records, all_results, n, top=True):
        if not records:
            return 0.0
        team_wps = {t: self._wp(r, 82) for t, r in all_results.items() if r}
        sorted_teams = sorted(team_wps.items(), key=lambda x: x[1], reverse=True)
        target = {t for t, _ in (sorted_teams[:n] if top else sorted_teams[-n:])}
        relevant = [r for r in records if r[3] in target]
        return self._pd(relevant, len(relevant)) if relevant else 0.0

    def _avg_opp_stat(self, records, all_results, stat, n):
        rec = records[-n:]
        if not rec:
            return 0.0
        vals = []
        for r in rec:
            opp_recs = all_results[r[3]]
            if opp_recs:
                vals.append(self._stat_avg(opp_recs, 10, stat))
        return sum(vals) / len(vals) if vals else 0.0

    def _margin_vs_quality_corr(self, records, all_results):
        if len(records) < 10:
            return 0.0
        margins = []
        opp_wps = []
        for r in records[-20:]:
            margins.append(r[2])
            opp_wps.append(self._wp(all_results[r[3]], 82))
        if len(set(opp_wps)) <= 1:
            return 0.0
        m_mean = sum(margins) / len(margins)
        o_mean = sum(opp_wps) / len(opp_wps)
        num = sum((m - m_mean) * (o - o_mean) for m, o in zip(margins, opp_wps))
        den = (sum((m - m_mean)**2 for m in margins) * sum((o - o_mean)**2 for o in opp_wps)) ** 0.5
        return num / den if den > 0 else 0.0

    def _ats_wp(self, records, n):
        return 0.5  # Needs spread data

    def _ou_record(self, records, n):
        return 0.5  # Needs total data

    def _clutch_wp(self, records):
        close = [r for r in records if abs(r[2]) <= 5]
        return self._wp(close, len(close)) if close else 0.5

    def _comeback_rate(self, records):
        return 0.5  # Needs halftime data

    def _consistency(self, records, n):
        s = records[-n:]
        if len(s) < 3:
            return 0.0
        margins = [x[2] for x in s]
        mean = sum(margins) / len(margins)
        return (sum((m - mean)**2 for m in margins) / len(margins)) ** 0.5

    def _h2h_wp(self, h2h, team):
        if not h2h:
            return 0.5
        wins = sum(1 for r in h2h if (r[1] and r[4].get("team") == team) or
                   (not r[1] and r[4].get("team") != team))
        return wins / len(h2h) if h2h else 0.5

    def _h2h_margin(self, h2h, team):
        if not h2h:
            return 0.0
        margins = [r[2] if r[4].get("team") == team else -r[2] for r in h2h]
        return sum(margins) / len(margins) if margins else 0.0

    def _h2h_home_wp(self, h2h):
        if not h2h:
            return 0.5
        home_wins = sum(1 for r in h2h if r[4].get("is_home", False) and r[1])
        home_games = sum(1 for r in h2h if r[4].get("is_home", False))
        return home_wins / home_games if home_games else 0.5

    def _elo_10_ago(self, records, team_elo, team):
        # Approximate: current elo minus recent changes
        return 0.0

    def _record_game(self, team_results, team_last, team_elo,
                     team_home_results, team_away_results,
                     h2h_results, home, away, hs, as_, gd,
                     h_stats, a_stats):
        """Record game for state tracking."""
        margin = hs - as_
        home_win = hs > as_

        # Parse stats from game data
        hs_d = self._parse_stats(h_stats, hs, as_, is_home=True)
        as_d = self._parse_stats(a_stats, as_, hs, is_home=False)
        hs_d["team"] = home
        as_d["team"] = away

        team_results[home].append((gd, home_win, margin, away, hs_d))
        team_results[away].append((gd, not home_win, -margin, home, as_d))
        team_home_results[home].append((gd, home_win, margin, away, hs_d))
        team_away_results[away].append((gd, not home_win, -margin, home, as_d))
        h2h_results[(home, away)].append((gd, home_win, margin, away, hs_d))
        team_last[home] = gd
        team_last[away] = gd

        # Update Elo
        K = 20
        expected_home = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
        result = 1.0 if home_win else 0.0
        team_elo[home] += K * (result - expected_home)
        team_elo[away] += K * ((1 - result) - (1 - expected_home))

    def _update_multi_elo(self, home, away, hs, as_, h_stats, a_stats,
                          team_elo_margin, team_elo_offense, team_elo_defense,
                          team_elo_recency, team_elo_history,
                          team_home_margin_sum, team_home_games_count):
        """Update multi-ELO rating systems for Category 24 (Power Ratings).

        Called after _record_game to update:
        - Margin-adjusted ELO (MOV capped at 20 pts)
        - Offensive ELO (based on points scored)
        - Defensive ELO (based on points allowed)
        - Recency-weighted ELO (K=30, decays faster)
        - ELO history (for trend / momentum features)
        - Home court advantage tracking
        """
        margin = hs - as_
        home_win = hs > as_

        # Parse stats for ORtg/DRtg
        if isinstance(h_stats, dict):
            h_ortg = h_stats.get("ortg", hs * 100 / max(h_stats.get("poss", 100), 1))
            h_drtg = h_stats.get("drtg", as_ * 100 / max(h_stats.get("poss", 100), 1))
        else:
            h_ortg = hs
            h_drtg = as_
        if isinstance(a_stats, dict):
            a_ortg = a_stats.get("ortg", as_ * 100 / max(a_stats.get("poss", 100), 1))
            a_drtg = a_stats.get("drtg", hs * 100 / max(a_stats.get("poss", 100), 1))
        else:
            a_ortg = as_
            a_drtg = hs

        result = 1.0 if home_win else 0.0
        HCA = 50  # Home court advantage in ELO terms

        # ── Margin-adjusted ELO (MOV capped, multiplier) ──
        K_m = 20
        mov_mult = min(abs(margin), 20) / 10.0  # Cap at 20pt margin
        e_m = 1 / (1 + 10 ** ((team_elo_margin[away] - team_elo_margin[home] - HCA) / 400))
        team_elo_margin[home] += K_m * mov_mult * (result - e_m)
        team_elo_margin[away] += K_m * mov_mult * ((1 - result) - (1 - e_m))

        # ── Offensive ELO (based on scoring performance) ──
        K_o = 15
        # Home offense result: fraction of total points scored by home
        off_result = hs / max(hs + as_, 1)
        e_off = 1 / (1 + 10 ** ((team_elo_offense[away] - team_elo_offense[home]) / 400))
        team_elo_offense[home] += K_o * (off_result - e_off)
        team_elo_offense[away] += K_o * ((1 - off_result) - (1 - e_off))

        # ── Defensive ELO (lower opponent scoring = better) ──
        K_d = 15
        # Home defense result: inverse — fewer points allowed = better
        def_result = as_ / max(hs + as_, 1)  # away scored / total — lower is better for home D
        def_result = 1.0 - def_result  # Flip: home defense success
        e_def = 1 / (1 + 10 ** ((team_elo_defense[away] - team_elo_defense[home]) / 400))
        team_elo_defense[home] += K_d * (def_result - e_def)
        team_elo_defense[away] += K_d * ((1 - def_result) - (1 - e_def))

        # ── Recency-weighted ELO (higher K for faster adaptation) ──
        K_r = 30
        e_r = 1 / (1 + 10 ** ((team_elo_recency[away] - team_elo_recency[home] - HCA) / 400))
        team_elo_recency[home] += K_r * (result - e_r)
        team_elo_recency[away] += K_r * ((1 - result) - (1 - e_r))

        # ── ELO history (for trend/momentum tracking) ──
        team_elo_history[home].append(team_elo_margin[home])
        team_elo_history[away].append(team_elo_margin[away])

        # ── Home court advantage tracking ──
        team_home_margin_sum[home] += margin
        team_home_games_count[home] += 1

    def _parse_stats(self, stats, pts, opp_pts, is_home=True):
        """Extract stats from game data, with defaults."""
        if not isinstance(stats, dict):
            stats = {}
        d = {
            "pts": pts, "opp_pts": opp_pts, "is_home": is_home,
            "ortg": stats.get("ortg", pts * 100 / max(stats.get("poss", 100), 1)),
            "drtg": stats.get("drtg", opp_pts * 100 / max(stats.get("poss", 100), 1)),
            "pace": stats.get("pace", (pts + opp_pts) / 2.0),
            "poss": stats.get("poss", (pts + opp_pts) / 2.0),
        }
        # Four Factors
        fga = stats.get("fga", max(pts / 1.1, 80))
        fg3m = stats.get("fg3m", pts * 0.3 / 3)
        fta = stats.get("fta", pts * 0.2)
        tov = stats.get("tov", 13)
        orb = stats.get("oreb", 10)
        drb = stats.get("dreb", 34)
        opp_drb = stats.get("opp_dreb", 34)

        d["efg_pct"] = stats.get("efg_pct", (pts / 2.0) / max(fga, 1))
        d["tov_rate"] = stats.get("tov_rate", tov / max(fga + 0.44 * fta + tov, 1))
        d["oreb_pct"] = stats.get("oreb_pct", orb / max(orb + opp_drb, 1))
        d["ft_rate"] = stats.get("ft_rate", fta / max(fga, 1))
        d["ts_pct"] = stats.get("ts_pct", pts / max(2 * (fga + 0.44 * fta), 1))

        # Opponent Four Factors (estimated)
        d["opp_efg_pct"] = stats.get("opp_efg_pct", d["efg_pct"] * 0.95)
        d["opp_tov_rate"] = stats.get("opp_tov_rate", d["tov_rate"])
        d["opp_oreb_pct"] = stats.get("opp_oreb_pct", d["oreb_pct"])
        d["opp_ft_rate"] = stats.get("opp_ft_rate", d["ft_rate"])

        # Shooting profile
        d["3par"] = stats.get("3par", fg3m * 3 / max(pts, 1))
        d["fg3_pct"] = stats.get("fg3_pct", 0.36)
        d["fg2_pct"] = stats.get("fg2_pct", 0.52)
        d["ft_pct"] = stats.get("ft_pct", 0.78)
        d["paint_pts"] = stats.get("paint_pts", pts * 0.4)
        d["fb_pts"] = stats.get("fb_pts", pts * 0.1)
        d["bench_pts"] = stats.get("bench_pts", pts * 0.3)
        d["2nd_pts"] = stats.get("2nd_pts", pts * 0.1)
        d["pitp"] = stats.get("pitp", pts * 0.4)
        d["pts_off_tov"] = stats.get("pts_off_tov", pts * 0.12)

        # Rates
        d["ast_rate"] = stats.get("ast_rate", 0.6)
        d["stl_rate"] = stats.get("stl_rate", 0.08)
        d["blk_rate"] = stats.get("blk_rate", 0.05)
        d["tov_pct"] = d["tov_rate"]
        d["dreb_pct"] = stats.get("dreb_pct", drb / max(drb + orb, 1))

        # Opponent shooting
        d["opp_fg3_pct"] = stats.get("opp_fg3_pct", 0.36)
        d["opp_paint_pts"] = stats.get("opp_paint_pts", opp_pts * 0.4)

        return d

    def _is_rivalry(self, home, away):
        """Check if divisional rivalry."""
        return self._division(home) == self._division(away)

    def _division(self, team):
        divisions = {
            "ATL": 2, "CHA": 2, "MIA": 2, "ORL": 2, "WAS": 2,  # Southeast
            "BOS": 0, "BKN": 0, "NYK": 0, "PHI": 0, "TOR": 0,  # Atlantic
            "CHI": 1, "CLE": 1, "DET": 1, "IND": 1, "MIL": 1,  # Central
            "DAL": 3, "HOU": 3, "MEM": 3, "NOP": 3, "SAS": 3,  # Southwest
            "DEN": 4, "MIN": 4, "OKC": 4, "POR": 4, "UTA": 4,  # Northwest
            "GSW": 5, "LAC": 5, "LAL": 5, "PHX": 5, "SAC": 5,  # Pacific
        }
        return divisions.get(team, 0) / 5.0

    def _conference(self, team):
        east = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
                "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
        return 0 if team in east else 1


# ── Genetic Feature Selection ──

def genetic_feature_selection(X, y, feature_names, n_generations=50,
                               population_size=100, target_features=200):
    """
    Use genetic algorithm to find optimal feature subset.

    Chromosome: binary vector (1=include, 0=exclude)
    Fitness: negative Brier score (minimize) on walk-forward CV

    Args:
        X: Full feature matrix (n_games, ~580)
        y: Labels
        feature_names: List of feature names
        n_generations: Number of GA generations
        population_size: Population size
        target_features: Target number of features to select

    Returns:
        selected_indices: Indices of selected features
        selected_names: Names of selected features
        fitness_history: Fitness per generation
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss
    import random

    try:
        import xgboost as xgb
        model_cls = lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model_cls = lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )

    n_features = X.shape[1]
    tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for speed
    random.seed(42)

    def fitness(chromosome):
        """Evaluate chromosome fitness = negative Brier score."""
        selected = [i for i, bit in enumerate(chromosome) if bit]
        if len(selected) < 10 or len(selected) > 400:
            return -0.30  # Penalty for too few or too many
        X_sub = X[:, selected]
        briers = []
        for ti, vi in tscv.split(X_sub):
            try:
                m = model_cls()
                m.fit(X_sub[ti], y[ti])
                p = m.predict_proba(X_sub[vi])[:, 1]
                briers.append(brier_score_loss(y[vi], p))
            except:
                briers.append(0.30)
        return -np.mean(briers)

    def crossover(parent1, parent2):
        """Two-point crossover."""
        pt1 = random.randint(0, n_features - 1)
        pt2 = random.randint(pt1, n_features - 1)
        child = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        return child

    def mutate(chromosome, rate=0.02):
        """Flip random bits."""
        return [1 - bit if random.random() < rate else bit for bit in chromosome]

    # Initialize population — biased toward target_features count
    population = []
    for _ in range(population_size):
        prob = target_features / n_features
        chromo = [1 if random.random() < prob else 0 for _ in range(n_features)]
        population.append(chromo)

    best_fitness = -1.0
    best_chromosome = None
    fitness_history = []

    print(f"Genetic Feature Selection: {n_features} candidates → ~{target_features} target")
    print(f"Population: {population_size}, Generations: {n_generations}")

    for gen in range(n_generations):
        # Evaluate fitness
        scores = [fitness(c) for c in population]

        # Track best
        gen_best = max(scores)
        gen_best_idx = scores.index(gen_best)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_chromosome = population[gen_best_idx][:]

        n_selected = sum(best_chromosome) if best_chromosome else 0
        fitness_history.append(-gen_best)
        print(f"  Gen {gen+1}/{n_generations}: Best Brier={-gen_best:.4f} "
              f"(features: {n_selected})")

        # Selection (tournament)
        new_pop = [best_chromosome[:]]  # Elitism: keep best
        while len(new_pop) < population_size:
            # Tournament selection
            contestants = random.sample(list(zip(population, scores)), 5)
            p1 = max(contestants, key=lambda x: x[1])[0]
            contestants = random.sample(list(zip(population, scores)), 5)
            p2 = max(contestants, key=lambda x: x[1])[0]

            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    # Extract selected features
    selected_indices = [i for i, bit in enumerate(best_chromosome) if bit]
    selected_names = [feature_names[i] for i in selected_indices]

    print(f"\nSelected {len(selected_indices)} features (Brier: {-best_fitness:.4f})")
    return selected_indices, selected_names, fitness_history


if __name__ == "__main__":
    print(f"NBA Feature Engine initialized")
    engine = NBAFeatureEngine(include_market=False)
    print(f"Feature candidates: {len(engine.feature_names)}")
    print(f"\nCategories:")
    categories = defaultdict(int)
    for name in engine.feature_names:
        if name.startswith(("h_wp", "a_wp", "h_pd", "a_pd", "h_ppg", "a_ppg",
                           "h_papg", "a_papg", "h_margin", "a_margin",
                           "h_close", "a_close", "h_blowout", "a_blowout",
                           "h_ou_avg", "a_ou_avg")):
            categories["1. Rolling Performance"] += 1
        elif "efg" in name or "tov_rate" in name or "orb_rate" in name or "ft_rate" in name:
            categories["2. Four Factors"] += 1
        elif "ortg" in name or "drtg" in name or "pace" in name or "ts" in name:
            categories["3. Pace & Efficiency"] += 1
        elif "3par" in name or "3p_pct" in name or "paint" in name or "bench" in name:
            categories["4. Scoring Profile"] += 1
        elif "streak" in name or "momentum" in name or "trend" in name:
            categories["5. Momentum"] += 1
        elif "rest" in name or "b2b" in name or "travel" in name or "fatigue" in name:
            categories["6. Rest & Schedule"] += 1
        elif "sos" in name or "vs_top" in name or "vs_above" in name:
            categories["7. Opponent-Adjusted"] += 1
        elif "h2h" in name or "matchup" in name or "elo" in name:
            categories["8. Matchup & H2H"] += 1
        elif "spread" in name or "ml_" in name or "clv" in name or "steam" in name:
            categories["9. Market Microstructure"] += 1
        else:
            categories["10. Context & Situational"] += 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

