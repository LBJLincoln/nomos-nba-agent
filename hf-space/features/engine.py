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

            X.append(row)
            y.append(1 if hs > as_ else 0)

            # Record this game
            self._record_game(team_results, team_last, team_elo,
                              team_home_results, team_away_results,
                              h2h_results, home, away, hs, as_, gd,
                              h_stats, a_stats)

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
