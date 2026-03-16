#!/usr/bin/env python3
"""
NBA Quant Feature Engine — 500+ Features with Genetic Selection
================================================================
Generates ~580 feature candidates across 10 categories, then uses
genetic algorithm (DEAP) to select optimal 150-300 features.

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
  ≈ 580+ feature candidates

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
    Generates 580+ features for each game from historical data.

    Usage:
        engine = NBAFeatureEngine()
        X, y, feature_names = engine.build(games)
        # X.shape = (n_games, ~580)
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

        self.feature_names = names

    def build(self, games, market_data=None):
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
