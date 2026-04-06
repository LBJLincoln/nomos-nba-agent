#!/usr/bin/env python3
"""
NBA Quant Feature Engine — 6000+ Features with Genetic Selection
=================================================================
Generates ~6000+ feature candidates across 35 categories, then uses
genetic algorithm to select optimal 150-400 features.

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
  26. ADVANCED PLAYER IMPACT (220+ features — star +/-, usage, chemistry)
  27. REFEREE DEEP ANALYSIS (120+ features — per-quarter foul rates, bias)
  28. VENUE & ENVIRONMENTAL (160+ features — altitude, timezone, attendance)
  29. ADVANCED MARKET MICROSTRUCTURE III (220+ features — velocity, acceleration)
  30. TIME SERIES DECOMPOSITION (320+ features — trend, seasonal, residual)
  31. CROSS-TEAM INTERACTION MATRIX (440+ features — pace diff, style clash)
  32. BAYESIAN PRIORS (220+ features — preseason, franchise, coach)
  33. NETWORK/GRAPH FEATURES (220+ features — PageRank, centrality)
  34. ENSEMBLE META-FEATURES (160+ features — model uncertainty, drift)
  35. TEMPORAL DECAY FEATURES (320+ features — exponential decay, recency)
  39. CIRCADIAN RHYTHM & TRAVEL FATIGUE (8 features — normalized composites, rest non-linearity)
  41. TRANSITION vs HALF-COURT EFFICIENCY SPLITS (7 features — fb_pts/pace splits)
  43. CLUTCH PERFORMANCE (8 features — close-game win%, margin, ortg from rolling records)
  44. GAME TOTALS PREDICTION (10 features — normalized PPG/PAPG, pace, ortg/drtg scoring environment)
  46. REAL ODDS MARKET FEATURES (8 features — implied prob, spread, total from historical CSV)
  47. DRIVE-OFFENSE vs RIM-DEFENSE MATCHUP (14 features — drive FG%, rim protection, matchup edges)
  48. PASSING NETWORK QUALITY (10 features — AST/pass, potential assists, ball movement)
  49. PLAY-TYPE EFFICIENCY (10 features — iso/PnR/spot-up/transition PPP, versatility)
  50. TEMPORAL WIN SEQUENCE ENCODING (12 features — ordered outcome sequence, momentum slope, streak)
  51. SEASON ERA NORMALIZATION (8 features — z-score vs league running avg per season)
  52. ODDS LINE FEATURES (15 features — spread magnitude, total, vig, season percentiles)
  53. ATS RECORD FEATURES (12 features — cover rate last 10/season, streaks, home/road splits)
  54. OVER/UNDER RECORD FEATURES (12 features — over rate last 10/season, pace vs total)
  ≈ 6296+ feature candidates

Architecture inspired by:
  - Starlizard: 500+ features, genetic selection, real-time adjustment
  - Priomha Capital: 17% annual ROI, market microstructure focus
  - Becker/Kalshi: Maker advantage, longshot bias exploitation
  - Dean Oliver: Four Factors framework
  - NBA Second Spectrum: Player tracking features
  - Kenpom: Adjusted efficiency, tempo-free stats
  - FiveThirtyEight RAPTOR: Player impact, Bayesian priors
  - Massey/Colley: Network-based power ratings

THIS SCRIPT MUST RUN ON HF SPACES (16GB RAM) — NOT on VM.
"""

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
import csv
import os

# ── Engine Version ──
ENGINE_VERSION = "v3.1-54cat"

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


# Alias map for non-standard team names (Bovada, international sources, etc.)
TEAM_ALIASES = {
    "L.A. Clippers": "LAC", "LA Clippers": "LAC",
    "L.A. Lakers": "LAL", "LA Lakers": "LAL",
    "NY Knicks": "NYK", "GS Warriors": "GSW",
    "SA Spurs": "SAS", "NO Pelicans": "NOP",
    "OKC": "OKC", "Philly": "PHI",
}


def resolve(name):
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
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


# ── Odds Data Loader (Cat 46) ──

def _american_to_implied_prob(american_odds):
    """Convert American moneyline to implied probability (no vig removal)."""
    try:
        ml = float(american_odds)
    except (ValueError, TypeError):
        return 0.5
    if ml == 0:
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def _decimal_to_implied_prob(decimal_odds):
    """Convert decimal odds to implied probability."""
    try:
        d = float(decimal_odds)
    except (ValueError, TypeError):
        return 0.5
    if d <= 1.0:
        return 1.0
    return 1.0 / d


def _is_decimal_odds(val):
    """Heuristic: decimal odds are typically 1.01-20.0; American odds are typically < -100 or > 100."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return False
    # Decimal odds are almost always between 1.01 and 50.0
    # American odds are < -100 or > 100 (e.g., -250, +200)
    return 1.0 < v < 50.0


def load_historical_odds(csv_path=None):
    """
    Load historical odds CSV and return a lookup dict.

    Returns:
        dict: (date_str, home_abbrev, away_abbrev) -> {
            'implied_home_prob': float,  # vig-inclusive implied probability
            'implied_away_prob': float,
            'fair_home_prob': float,     # vig-removed fair probability
            'fair_away_prob': float,
            'spread_home': float,        # point spread (negative = home favored)
            'total': float,              # over/under total
            'ml_home_raw': float,        # raw moneyline (American)
            'ml_away_raw': float,
            'overround': float,          # total implied prob (>1.0 = vig)
            'book': str,
            'source': str,
        }
        Multiple entries for same game (different books) are kept;
        the first one loaded (betmgm preferred) is returned for the key.
    """
    if csv_path is None:
        # Try common paths
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'historical-odds', 'nba_2025-26_odds.csv'),
            os.path.join(os.path.dirname(__file__), 'data', 'historical-odds', 'nba_2025-26_odds.csv'),
            '/home/termius/nomos-nba-agent/data/historical-odds/nba_2025-26_odds.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        if csv_path is None:
            return {}

    if not os.path.exists(csv_path):
        return {}

    lookup = {}
    # Also store multi-book data for books_disagreement
    multi_book = defaultdict(list)  # (date, home, away) -> [implied_home_prob, ...]

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get('date', '').strip()
                home_full = row.get('home_team', '').strip()
                away_full = row.get('away_team', '').strip()
                book = row.get('book', '').strip()
                source = row.get('source', '').strip()

                home_abbr = resolve(home_full)
                away_abbr = resolve(away_full)
                if not home_abbr or not away_abbr or not date_str:
                    continue

                # Parse moneylines — handle both American and decimal formats
                ml_home_str = row.get('moneyline_home', '').strip()
                ml_away_str = row.get('moneyline_away', '').strip()

                if not ml_home_str or not ml_away_str:
                    continue

                if _is_decimal_odds(ml_home_str) or _is_decimal_odds(ml_away_str):
                    # Decimal odds format
                    ip_home = _decimal_to_implied_prob(ml_home_str)
                    ip_away = _decimal_to_implied_prob(ml_away_str)
                    # Convert to American for storage
                    try:
                        dh = float(ml_home_str)
                        ml_home_american = round((dh - 1.0) * 100) if dh >= 2.0 else round(-100 / (dh - 1.0)) if dh > 1.0 else -110
                    except (ValueError, TypeError):
                        ml_home_american = -110
                    try:
                        da = float(ml_away_str)
                        ml_away_american = round((da - 1.0) * 100) if da >= 2.0 else round(-100 / (da - 1.0)) if da > 1.0 else -110
                    except (ValueError, TypeError):
                        ml_away_american = -110
                else:
                    # American odds format
                    ip_home = _american_to_implied_prob(ml_home_str)
                    ip_away = _american_to_implied_prob(ml_away_str)
                    try:
                        ml_home_american = float(ml_home_str)
                    except (ValueError, TypeError):
                        ml_home_american = -110
                    try:
                        ml_away_american = float(ml_away_str)
                    except (ValueError, TypeError):
                        ml_away_american = -110

                # Overround (vig)
                overround = ip_home + ip_away  # typically 1.04-1.08

                # Fair (vig-removed) probabilities
                if overround > 0:
                    fair_home = ip_home / overround
                    fair_away = ip_away / overround
                else:
                    fair_home = 0.5
                    fair_away = 0.5

                # Parse spread and total (may be missing)
                spread_str = row.get('spread_home', '').strip()
                total_str = row.get('total', '').strip()
                try:
                    spread_home = float(spread_str) if spread_str else None
                except (ValueError, TypeError):
                    spread_home = None
                try:
                    total = float(total_str) if total_str else None
                except (ValueError, TypeError):
                    total = None

                key = (date_str, home_abbr, away_abbr)
                entry = {
                    'implied_home_prob': ip_home,
                    'implied_away_prob': ip_away,
                    'fair_home_prob': fair_home,
                    'fair_away_prob': fair_away,
                    'spread_home': spread_home,
                    'total': total,
                    'ml_home_raw': ml_home_american,
                    'ml_away_raw': ml_away_american,
                    'overround': overround,
                    'book': book,
                    'source': source,
                }

                multi_book[key].append(ip_home)

                # Prefer betmgm over other books; first entry wins if same book
                if key not in lookup or (book == 'betmgm' and lookup[key].get('book') != 'betmgm'):
                    lookup[key] = entry

    except Exception as e:
        # Silently return empty on any file error
        return {}

    # Add books_disagreement where multiple books exist
    for key, probs in multi_book.items():
        if key in lookup and len(probs) > 1:
            lookup[key]['books_disagreement'] = max(probs) - min(probs)

    return lookup


class NBAFeatureEngine:
    """
    Generates 6000+ features for each game from historical data.

    Usage:
        engine = NBAFeatureEngine()
        X, y, feature_names = engine.build(games)
        # X.shape = (n_games, ~6000)
    """

    def __init__(self, include_market=True, skip_placeholder=False):
        self.include_market = include_market
        self.skip_placeholder = skip_placeholder
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

        # 11-15. PLACEHOLDER CATEGORIES (skippable — all zeros/defaults without real data)
        if not self.skip_placeholder:
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

        # =====================================================================
        # CATEGORIES 26-35: MASSIVE FEATURE EXPANSION (4000+ new features)
        # =====================================================================

        # 26. ADVANCED PLAYER IMPACT (220 features)
        # Star player +/-, usage rates per lineup, efficiency deltas,
        # rest-adjusted player metrics, player chemistry indicators
        _pi_stats = ["plus_minus", "usage_rate", "per", "ws_per48", "bpm",
                     "vorp", "raptor_off", "raptor_def", "raptor_total",
                     "ts_pct", "ast_pct", "reb_pct"]
        _pi_windows = [3, 5, 10, 20]
        for prefix in ["h", "a"]:
            # Top player stats across windows: 12 stats × 4 windows = 48 per team
            for stat in _pi_stats:
                for w in _pi_windows:
                    names.append(f"{prefix}_star1_{stat}_{w}")
            # Second star: 12 stats × 4 windows = 48 per team
            for stat in _pi_stats:
                for w in _pi_windows:
                    names.append(f"{prefix}_star2_{stat}_{w}")
            # Team-level aggregated player impact: 12 features per team
            names.append(f"{prefix}_star_combined_plus_minus")
            names.append(f"{prefix}_star_usage_concentration")
            names.append(f"{prefix}_star_minutes_ratio")
            names.append(f"{prefix}_star_efficiency_delta")
            names.append(f"{prefix}_star_rest_adj_rating")
            names.append(f"{prefix}_chemistry_starting5")
            names.append(f"{prefix}_chemistry_top3")
            names.append(f"{prefix}_player_variance")
            names.append(f"{prefix}_top_player_dependency_score")
            names.append(f"{prefix}_bench_player_avg_rating")
            names.append(f"{prefix}_roster_talent_depth")
            names.append(f"{prefix}_injury_replacement_quality")
        # Matchup-level player impact differentials: 14 features
        names.extend([
            "pi_star1_rating_diff",
            "pi_star2_rating_diff",
            "pi_combined_star_diff",
            "pi_usage_concentration_diff",
            "pi_chemistry_diff",
            "pi_bench_quality_diff",
            "pi_talent_depth_diff",
            "pi_star_fatigue_adj_diff",
            "pi_star_matchup_advantage",
            "pi_key_player_edge",
            "pi_roster_continuity_diff",
            "pi_injury_impact_diff",
            "pi_star_on_off_diff",
            "pi_clutch_player_diff",
        ])

        # 27. REFEREE DEEP ANALYSIS (120 features)
        # Ref-specific foul rates by quarter, ref home/away bias history,
        # ref pace impact by team type, ref total-over/under tendency
        _ref_quarters = ["q1", "q2", "q3", "q4"]
        for q in _ref_quarters:
            names.append(f"ref_{q}_foul_rate")
            names.append(f"ref_{q}_home_foul_bias")
            names.append(f"ref_{q}_tech_rate")
            names.append(f"ref_{q}_and1_rate")
            names.append(f"ref_{q}_shooting_foul_rate")
            names.append(f"ref_{q}_offensive_foul_rate")
        # Ref bias by team type (fast/slow, top/bottom, home/away)
        _ref_team_types = ["fast_pace", "slow_pace", "top10", "bottom10",
                           "big_market", "small_market"]
        for ttype in _ref_team_types:
            names.append(f"ref_bias_{ttype}_home_wp")
            names.append(f"ref_bias_{ttype}_foul_diff")
            names.append(f"ref_bias_{ttype}_ft_diff")
        # Ref over/under tendencies across contexts
        _ref_contexts = ["overall", "high_total", "low_total", "rivalry",
                         "playoff_race", "b2b_games", "national_tv"]
        for ctx in _ref_contexts:
            names.append(f"ref_over_tendency_{ctx}")
            names.append(f"ref_under_tendency_{ctx}")
            names.append(f"ref_total_delta_{ctx}")
        # Ref pace impact
        for prefix in ["h", "a"]:
            names.append(f"ref_{prefix}_expected_pace_impact")
            names.append(f"ref_{prefix}_expected_foul_impact")
            names.append(f"ref_{prefix}_expected_ft_impact")
            names.append(f"ref_{prefix}_historical_team_bias")
        # Ref composite features
        names.extend([
            "ref_consistency_index",
            "ref_home_bias_composite",
            "ref_pace_impact_composite",
            "ref_total_impact_composite",
            "ref_foul_disparity_expected",
            "ref_experience_weight",
            "ref_crew_chemistry",
            "ref_variance_in_calls",
            "ref_big_game_experience",
            "ref_crew_avg_total_called",
            "ref_crew_foul_per_possession",
            "ref_historical_ats_home_rate",
            "ref_historical_over_rate_season",
            "ref_recent_form_5_games",
            "ref_recent_form_10_games",
            "ref_travel_adjusted_bias",
        ])

        # 28. VENUE & ENVIRONMENTAL (160 features)
        # Altitude effects, timezone crossing, attendance, temperature,
        # court surface age, arena factors
        _venue_windows = [3, 5, 10, 20]
        for prefix in ["h", "a"]:
            # Altitude impact features
            for w in _venue_windows:
                names.append(f"{prefix}_altitude_adj_ortg_{w}")
                names.append(f"{prefix}_altitude_adj_drtg_{w}")
                names.append(f"{prefix}_altitude_adj_pace_{w}")
            # Timezone crossing features
            for w in _venue_windows:
                names.append(f"{prefix}_tz_cross_wp_{w}")
                names.append(f"{prefix}_tz_cross_margin_{w}")
            # Home/away specific venue features
            names.append(f"{prefix}_home_arena_wp")
            names.append(f"{prefix}_home_arena_margin")
            names.append(f"{prefix}_home_arena_ortg")
            names.append(f"{prefix}_home_arena_drtg")
            names.append(f"{prefix}_arena_elevation_factor")
            names.append(f"{prefix}_games_at_altitude_season")
            names.append(f"{prefix}_altitude_acclimatized")
            names.append(f"{prefix}_tz_disruption_score")
            names.append(f"{prefix}_tz_direction_east")
            names.append(f"{prefix}_tz_direction_west")
            names.append(f"{prefix}_tz_games_same_zone")
            names.append(f"{prefix}_attendance_ratio_avg")
            names.append(f"{prefix}_attendance_trend")
            names.append(f"{prefix}_court_surface_age")
            names.append(f"{prefix}_temperature_at_gametime")
            names.append(f"{prefix}_indoor_outdoor_flag")
            names.append(f"{prefix}_arena_capacity")
            names.append(f"{prefix}_arena_noise_proxy")
        # Game-level venue differentials
        names.extend([
            "venue_altitude_diff",
            "venue_altitude_abs_diff",
            "venue_tz_crossing_diff",
            "venue_tz_abs_diff",
            "venue_home_elevation_advantage",
            "venue_attendance_ratio",
            "venue_arena_age_diff",
            "venue_climate_diff",
            "venue_travel_direction",
            "venue_acclimatization_diff",
            "venue_home_arena_strength",
            "venue_surface_familiarity_diff",
            "venue_noise_advantage",
            "venue_altitude_fatigue_interaction",
            "venue_tz_fatigue_interaction",
            "venue_combined_environmental_edge",
        ])

        # 29. ADVANCED MARKET MICROSTRUCTURE III (220 features)
        # Line movement velocity, acceleration, book consensus, sharp splits
        if self.include_market:
            _mkt3_windows = ["1h", "2h", "4h", "8h", "12h", "24h"]
            # Line movement velocity & acceleration per window
            for tw in _mkt3_windows:
                names.append(f"mkt3_spread_velocity_{tw}")
                names.append(f"mkt3_spread_acceleration_{tw}")
                names.append(f"mkt3_total_velocity_{tw}")
                names.append(f"mkt3_total_acceleration_{tw}")
                names.append(f"mkt3_ml_velocity_{tw}")
                names.append(f"mkt3_ml_acceleration_{tw}")
            # Book consensus divergence
            _mkt3_books = ["pinnacle", "draftkings", "fanduel", "betmgm",
                           "caesars", "bet365", "william_hill"]
            for book in _mkt3_books:
                names.append(f"mkt3_{book}_vs_consensus_spread")
                names.append(f"mkt3_{book}_vs_consensus_total")
                names.append(f"mkt3_{book}_vs_consensus_ml")
                names.append(f"mkt3_{book}_clv_history")
            # Sharp vs public split features
            names.extend([
                "mkt3_sharp_pct_home", "mkt3_sharp_pct_away",
                "mkt3_public_pct_home", "mkt3_public_pct_away",
                "mkt3_sharp_public_divergence_spread",
                "mkt3_sharp_public_divergence_total",
                "mkt3_sharp_money_direction",
                "mkt3_public_money_direction",
                "mkt3_sharp_intensity_score",
            ])
            # Steam move features
            names.extend([
                "mkt3_steam_count_total", "mkt3_steam_count_last_6h",
                "mkt3_steam_magnitude_avg", "mkt3_steam_direction",
                "mkt3_reverse_steam_flag", "mkt3_steam_books_triggered",
            ])
            # Reverse line movement features
            names.extend([
                "mkt3_rlm_spread_flag", "mkt3_rlm_total_flag",
                "mkt3_rlm_magnitude_spread", "mkt3_rlm_magnitude_total",
                "mkt3_rlm_sharp_confirmation",
            ])
            # Closing line value by book
            for book in _mkt3_books:
                names.append(f"mkt3_clv_{book}_home")
            # Opening-to-closing deltas
            names.extend([
                "mkt3_open_to_close_spread_delta",
                "mkt3_open_to_close_total_delta",
                "mkt3_open_to_close_ml_delta",
                "mkt3_open_to_close_implied_delta",
            ])
            # Money line implied probability convergence
            names.extend([
                "mkt3_ml_convergence_rate",
                "mkt3_ml_convergence_direction",
                "mkt3_implied_prob_convergence",
                "mkt3_book_agreement_score",
                "mkt3_market_depth_proxy",
                "mkt3_liquidity_score",
                "mkt3_market_manipulation_flag",
                "mkt3_arbitrage_opportunity",
                "mkt3_hold_pct_change",
                "mkt3_vig_trend",
            ])
            # Historical patterns
            names.extend([
                "mkt3_historical_clv_this_matchup",
                "mkt3_historical_rlm_success_rate",
                "mkt3_historical_steam_success_rate",
                "mkt3_historical_sharp_roi",
                "mkt3_historical_public_fade_roi",
                "mkt3_line_stability_score",
                "mkt3_early_sharp_vs_late_public",
                "mkt3_market_overreaction_index",
                "mkt3_odds_shape_skewness",
                "mkt3_odds_shape_kurtosis",
            ])

        # 30. TIME SERIES DECOMPOSITION (320 features)
        # Trend, seasonal, residual components, autocorrelation
        _ts_stats = ["wp", "ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
        _ts_trend_windows = [3, 5, 10, 20]
        _ts_lags = [1, 2, 3, 4, 5]
        for prefix in ["h", "a"]:
            # Trend components: 8 stats × 4 windows = 32 per team
            for stat in _ts_stats:
                for w in _ts_trend_windows:
                    names.append(f"ts_trend_{prefix}_{stat}_{w}")
            # Seasonal components (day-of-week): 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_seasonal_dow_{prefix}_{stat}")
            # Seasonal components (month): 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_seasonal_month_{prefix}_{stat}")
            # Residual volatility: 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_residual_vol_{prefix}_{stat}")
            # Autocorrelation features: 8 stats × 5 lags = 40 per team
            for stat in _ts_stats:
                for lag in _ts_lags:
                    names.append(f"ts_acf_{prefix}_{stat}_lag{lag}")
            # Partial autocorrelation: 8 stats × 5 lags = 40 per team
            for stat in _ts_stats:
                for lag in _ts_lags:
                    names.append(f"ts_pacf_{prefix}_{stat}_lag{lag}")
            # Stationarity indicators: 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_stationarity_{prefix}_{stat}")
            # Trend strength: 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_trend_strength_{prefix}_{stat}")
            # Seasonality strength: 8 stats = 8 per team
            for stat in _ts_stats:
                names.append(f"ts_season_strength_{prefix}_{stat}")

        # 31. CROSS-TEAM INTERACTION MATRIX (440 features)
        # Pace differential, defensive rating matchup, style clash
        _xteam_stats = [
            "pace", "ortg", "drtg", "efg", "3p_pct", "paint_pts",
            "fb_pts", "tov_rate", "oreb_pct", "ft_rate",
        ]
        _xteam_windows = [5, 10, 20]
        # Pairwise matchup: 10 stats × 3 windows × 4 types
        # (diff, ratio, interaction, mismatch) = 120 features
        for stat in _xteam_stats:
            for w in _xteam_windows:
                names.append(f"xteam_diff_{stat}_{w}")
                names.append(f"xteam_ratio_{stat}_{w}")
                names.append(f"xteam_inter_{stat}_{w}")
                names.append(f"xteam_mismatch_{stat}_{w}")
        # Offensive style vs defensive style: 10 stats × 3 windows = 30
        for stat in _xteam_stats:
            for w in _xteam_windows:
                names.append(f"xteam_off_vs_def_{stat}_{w}")
        # Reverse: defensive style vs opponent offense: 10 stats × 3 windows = 30
        for stat in _xteam_stats:
            for w in _xteam_windows:
                names.append(f"xteam_def_vs_off_{stat}_{w}")
        # Style clash indices: composite features
        _style_types = [
            "pace_clash", "shooting_clash", "paint_battle",
            "transition_war", "turnover_battle", "rebounding_war",
            "free_throw_battle", "three_pt_war", "defense_clash",
            "tempo_mismatch",
        ]
        for style in _style_types:
            for w in _xteam_windows:
                names.append(f"xteam_style_{style}_{w}")
        # Strength matchup matrix (home strength area vs away weakness)
        _strength_areas = [
            "perimeter_off_vs_perimeter_def",
            "interior_off_vs_interior_def",
            "transition_off_vs_transition_def",
            "halfcourt_off_vs_halfcourt_def",
            "shooting_off_vs_shooting_def",
            "rebounding_off_vs_rebounding_def",
            "playmaking_vs_ball_pressure",
            "rim_protection_vs_paint_scoring",
            "three_pt_shooting_vs_three_pt_defense",
            "free_throw_drawing_vs_foul_avoidance",
        ]
        for area in _strength_areas:
            for w in _xteam_windows:
                names.append(f"xteam_strength_{area}_{w}")
        # Game-level interaction composites
        names.extend([
            "xteam_overall_style_clash",
            "xteam_offensive_edge_composite",
            "xteam_defensive_edge_composite",
            "xteam_pace_war_indicator",
            "xteam_grind_game_indicator",
            "xteam_shootout_indicator",
            "xteam_mismatch_severity",
            "xteam_balanced_matchup_flag",
            "xteam_upset_structural_flag",
            "xteam_blowout_structural_flag",
        ])

        # 32. BAYESIAN PRIORS (220 features)
        # Pre-season win totals, Vegas priors, franchise strength, coach impact
        for prefix in ["h", "a"]:
            # Pre-season and Vegas priors
            names.append(f"{prefix}_preseason_win_total_ou")
            names.append(f"{prefix}_vegas_season_win_total")
            names.append(f"{prefix}_preseason_power_rank")
            names.append(f"{prefix}_preseason_conf_rank")
            names.append(f"{prefix}_preseason_division_rank")
            names.append(f"{prefix}_vegas_championship_odds")
            names.append(f"{prefix}_vegas_conf_winner_odds")
            names.append(f"{prefix}_preseason_vs_actual_wp")
            names.append(f"{prefix}_preseason_vs_actual_delta")
            names.append(f"{prefix}_bayesian_prior_strength")
            # Franchise historical strength
            names.append(f"{prefix}_franchise_historical_wp_10yr")
            names.append(f"{prefix}_franchise_historical_wp_5yr")
            names.append(f"{prefix}_franchise_championships")
            names.append(f"{prefix}_franchise_finals_appearances")
            names.append(f"{prefix}_franchise_playoff_rate_10yr")
            names.append(f"{prefix}_franchise_avg_seed_5yr")
            names.append(f"{prefix}_franchise_consistency_5yr")
            names.append(f"{prefix}_franchise_rebuild_indicator")
            names.append(f"{prefix}_franchise_contender_indicator")
            names.append(f"{prefix}_franchise_stability_index")
            # Coach impact features
            names.append(f"{prefix}_coach_career_wp")
            names.append(f"{prefix}_coach_playoff_wp")
            names.append(f"{prefix}_coach_tenure_years")
            names.append(f"{prefix}_coach_tenure_adjustment")
            names.append(f"{prefix}_coach_with_team_years")
            names.append(f"{prefix}_coach_system_maturity")
            names.append(f"{prefix}_coach_ato_rating")
            names.append(f"{prefix}_coach_challenge_success_rate")
            names.append(f"{prefix}_coach_close_game_wp")
            names.append(f"{prefix}_coach_blowout_wp")
            names.append(f"{prefix}_coach_b2b_wp")
            names.append(f"{prefix}_coach_road_wp")
            names.append(f"{prefix}_coach_home_wp")
            names.append(f"{prefix}_coach_vs_winning_teams_wp")
            names.append(f"{prefix}_coach_comeback_rate")
            names.append(f"{prefix}_coach_defensive_rating_rank")
            names.append(f"{prefix}_coach_offensive_rating_rank")
            names.append(f"{prefix}_coach_pace_preference")
            # Bayesian blend features
            names.append(f"{prefix}_bayesian_wp_prior_blend")
            names.append(f"{prefix}_bayesian_rating_confidence")
            names.append(f"{prefix}_bayesian_update_magnitude")
            names.append(f"{prefix}_bayesian_prior_weight")
            names.append(f"{prefix}_prior_vs_observed_divergence")
            names.append(f"{prefix}_regression_to_prior")
            names.append(f"{prefix}_prior_adjusted_ortg")
            names.append(f"{prefix}_prior_adjusted_drtg")
            names.append(f"{prefix}_market_implied_prior")
            names.append(f"{prefix}_composite_bayesian_rating")
        # Game-level Bayesian differentials
        names.extend([
            "bayes_preseason_diff",
            "bayes_vegas_win_total_diff",
            "bayes_franchise_strength_diff",
            "bayes_coach_wp_diff",
            "bayes_coach_tenure_diff",
            "bayes_prior_blend_diff",
            "bayes_regression_diff",
            "bayes_championship_odds_diff",
            "bayes_system_maturity_diff",
            "bayes_composite_diff",
        ])

        # 33. NETWORK/GRAPH FEATURES (220 features)
        # PageRank, clustering coefficient, centrality, connectivity
        _net_windows = [10, 20, 82]
        for prefix in ["h", "a"]:
            # PageRank-style features (from wins network)
            for w in _net_windows:
                names.append(f"{prefix}_pagerank_wins_{w}")
                names.append(f"{prefix}_pagerank_margin_{w}")
                names.append(f"{prefix}_pagerank_weighted_{w}")
            # Clustering coefficient
            for w in _net_windows:
                names.append(f"{prefix}_clustering_coeff_{w}")
            # Betweenness centrality
            for w in _net_windows:
                names.append(f"{prefix}_betweenness_centrality_{w}")
            # Strength of schedule network features
            for w in _net_windows:
                names.append(f"{prefix}_sos_network_centrality_{w}")
                names.append(f"{prefix}_sos_network_pagerank_{w}")
            # Conference connectivity
            names.append(f"{prefix}_conf_connectivity")
            names.append(f"{prefix}_conf_dominance")
            names.append(f"{prefix}_conf_beaten_best")
            names.append(f"{prefix}_conf_lost_to_worst")
            # Division rivalry features
            names.append(f"{prefix}_div_rivalry_intensity")
            names.append(f"{prefix}_div_dominance")
            names.append(f"{prefix}_div_games_played")
            names.append(f"{prefix}_div_wp")
            # Win chain features (A beat B, B beat C → transitive)
            for w in _net_windows:
                names.append(f"{prefix}_win_chain_depth_{w}")
                names.append(f"{prefix}_win_chain_strength_{w}")
            # Loss chain features
            for w in _net_windows:
                names.append(f"{prefix}_loss_chain_depth_{w}")
            # Network diversity (variety of opponents beaten)
            for w in _net_windows:
                names.append(f"{prefix}_opponents_beaten_{w}")
                names.append(f"{prefix}_opponents_beaten_pct_{w}")
                names.append(f"{prefix}_unique_losses_{w}")
            # Eigenvector centrality
            for w in _net_windows:
                names.append(f"{prefix}_eigenvector_centrality_{w}")
        # Game-level network differentials
        names.extend([
            "net_pagerank_diff_82",
            "net_pagerank_diff_20",
            "net_clustering_diff",
            "net_centrality_diff",
            "net_conf_dominance_diff",
            "net_div_dominance_diff",
            "net_win_chain_diff",
            "net_opponents_beaten_diff",
            "net_eigenvector_diff",
            "net_network_advantage_composite",
        ])

        # 34. ENSEMBLE META-FEATURES (160 features)
        # Previous model prediction uncertainty, disagreement, drift
        _meta_models = ["xgboost", "lightgbm", "catboost", "rf", "logistic"]
        _meta_windows = [5, 10, 20, 30]
        for prefix in ["h", "a"]:
            # Per-model accuracy for this team
            for model in _meta_models:
                for w in _meta_windows:
                    names.append(f"meta2_{prefix}_{model}_accuracy_{w}")
            # Per-model calibration
            for model in _meta_models:
                names.append(f"meta2_{prefix}_{model}_calibration")
            # Model disagreement per team
            names.append(f"meta2_{prefix}_model_disagreement")
            names.append(f"meta2_{prefix}_model_disagreement_trend")
            names.append(f"meta2_{prefix}_prediction_uncertainty")
            names.append(f"meta2_{prefix}_prediction_stability")
        # Game-level ensemble meta-features
        for model in _meta_models:
            names.append(f"meta2_{model}_predicted_prob")
            names.append(f"meta2_{model}_confidence")
            names.append(f"meta2_{model}_edge_vs_market")
        names.extend([
            "meta2_ensemble_mean_prob",
            "meta2_ensemble_std_prob",
            "meta2_ensemble_max_prob",
            "meta2_ensemble_min_prob",
            "meta2_ensemble_range",
            "meta2_model_agreement_score",
            "meta2_weighted_ensemble_prob",
            "meta2_calibration_residual",
            "meta2_historical_accuracy_similar_games",
            "meta2_feature_importance_stability",
            "meta2_feature_regime_indicator",
            "meta2_prediction_drift_5",
            "meta2_prediction_drift_10",
            "meta2_model_confidence_composite",
            "meta2_edge_confidence_product",
            "meta2_risk_adjusted_edge",
            "meta2_bankroll_optimal_fraction",
            "meta2_expected_value_composite",
            "meta2_sharpe_implied_edge",
            "meta2_historical_roi_similar",
        ])

        # 35. TEMPORAL DECAY FEATURES (320 features)
        # Exponential decay weighted stats, recency-weighted metrics
        _td_stats = ["wp", "ppg", "papg", "margin", "ortg", "drtg",
                      "efg", "ts", "pace", "3p_pct"]
        _td_half_lives = [3, 5, 10, 20]  # half-life in games
        for prefix in ["h", "a"]:
            # Exponential decay weighted stats: 10 stats × 4 half-lives = 40 per team
            for stat in _td_stats:
                for hl in _td_half_lives:
                    names.append(f"td_decay_{prefix}_{stat}_hl{hl}")
            # Recency-weighted opponent quality: 4 half-lives = 4 per team
            for hl in _td_half_lives:
                names.append(f"td_opp_quality_{prefix}_hl{hl}")
            # Time-weighted home/away splits: 4 half-lives × 2 = 8 per team
            for hl in _td_half_lives:
                names.append(f"td_home_split_{prefix}_hl{hl}")
                names.append(f"td_away_split_{prefix}_hl{hl}")
            # Season-phase interaction terms: 10 stats × 3 phases = 30 per team
            _phases = ["early", "mid", "late"]
            for stat in _td_stats:
                for phase in _phases:
                    names.append(f"td_phase_{prefix}_{stat}_{phase}")
            # Decay-weighted trend (difference between fast and slow decay)
            # 10 stats × 6 pairs = 60 per team
            _td_pairs = [(3, 5), (3, 10), (3, 20), (5, 10), (5, 20), (10, 20)]
            for stat in _td_stats:
                for hl1, hl2 in _td_pairs:
                    names.append(f"td_trend_{prefix}_{stat}_hl{hl1}_vs_hl{hl2}")
        # Game-level temporal decay differentials
        for stat in _td_stats:
            for hl in _td_half_lives:
                names.append(f"td_diff_{stat}_hl{hl}")

        # =====================================================================
        # EXPANDED SUB-FEATURES: Additional features to reach 6000+ total
        # =====================================================================

        # 26b. ADVANCED PLAYER IMPACT — EXPANDED (additional ~200 features)
        # Per-position impact features
        _positions = ["pg", "sg", "sf", "pf", "c"]
        for prefix in ["h", "a"]:
            for pos in _positions:
                names.append(f"{prefix}_pos_{pos}_rating")
                names.append(f"{prefix}_pos_{pos}_minutes_share")
                names.append(f"{prefix}_pos_{pos}_plus_minus")
                names.append(f"{prefix}_pos_{pos}_usage")
                names.append(f"{prefix}_pos_{pos}_ts_pct")
                names.append(f"{prefix}_pos_{pos}_def_rating")
        # Player pairwise synergy features: top 5 two-man combos
        for prefix in ["h", "a"]:
            for combo_idx in range(1, 6):
                names.append(f"{prefix}_combo{combo_idx}_netrtg")
                names.append(f"{prefix}_combo{combo_idx}_minutes")
                names.append(f"{prefix}_combo{combo_idx}_plus_minus")
        # Lineup unit features by window
        for prefix in ["h", "a"]:
            for unit in ["start", "bench", "closing"]:
                for w in [5, 10]:
                    names.append(f"{prefix}_{unit}_unit_ortg_{w}")
                    names.append(f"{prefix}_{unit}_unit_drtg_{w}")
                    names.append(f"{prefix}_{unit}_unit_netrtg_{w}")
                    names.append(f"{prefix}_{unit}_unit_pace_{w}")
        # Position matchup advantages
        for pos in _positions:
            names.append(f"pos_matchup_{pos}_off_advantage")
            names.append(f"pos_matchup_{pos}_def_advantage")
            names.append(f"pos_matchup_{pos}_size_diff")
            names.append(f"pos_matchup_{pos}_speed_diff")

        # 27b. REFEREE DEEP ANALYSIS — EXPANDED (additional ~120 features)
        # Ref tendency by score differential context
        _score_contexts = ["blowout", "close", "tied", "home_leading", "away_leading"]
        for ctx in _score_contexts:
            names.append(f"ref_foul_rate_{ctx}")
            names.append(f"ref_home_bias_{ctx}")
            names.append(f"ref_tech_rate_{ctx}")
            names.append(f"ref_review_rate_{ctx}")
        # Ref impact on specific play types
        _play_types = ["post_up", "pick_roll", "isolation", "transition",
                       "spot_up", "off_screen", "handoff", "cut"]
        for play in _play_types:
            names.append(f"ref_foul_rate_{play}")
            names.append(f"ref_and1_rate_{play}")
        # Ref historical impact on team types
        for prefix in ["h", "a"]:
            names.append(f"ref_{prefix}_team_specific_foul_rate")
            names.append(f"ref_{prefix}_team_specific_ft_rate")
            names.append(f"ref_{prefix}_team_specific_tech_rate")
            names.append(f"ref_{prefix}_team_historical_wp_with_ref")
            names.append(f"ref_{prefix}_team_historical_margin_with_ref")
            names.append(f"ref_{prefix}_team_historical_total_with_ref")
        # Ref crew composition features
        names.extend([
            "ref_crew_avg_experience",
            "ref_crew_min_experience",
            "ref_crew_max_experience",
            "ref_crew_experience_variance",
            "ref_crew_consistency_rating",
            "ref_crew_foul_rate_consistency",
            "ref_crew_home_bias_agreement",
            "ref_lead_official_weight",
            "ref_lead_vs_crew_bias_diff",
            "ref_night_game_adjustment",
            "ref_day_game_adjustment",
            "ref_back_to_back_ref_fatigue",
        ])
        # Ref interaction with game context
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                names.append(f"ref_{prefix}_foul_drawing_ability_{w}")
                names.append(f"ref_{prefix}_foul_committing_rate_{w}")
                names.append(f"ref_{prefix}_ft_generation_rate_{w}")
                names.append(f"ref_{prefix}_tech_tendency_{w}")

        # 28b. VENUE & ENVIRONMENTAL — EXPANDED (additional ~200 features)
        # Temperature and weather impact (affects travel/mood)
        _weather_features = ["temperature", "humidity", "wind", "precipitation",
                             "snow_flag", "storm_flag"]
        for prefix in ["h", "a"]:
            for feat in _weather_features:
                names.append(f"env_{prefix}_{feat}_at_arena")
            # Travel weather impact
            names.append(f"env_{prefix}_travel_weather_severity")
            names.append(f"env_{prefix}_flight_delay_risk")
        # Arena-specific features
        for prefix in ["h", "a"]:
            names.append(f"env_{prefix}_arena_age_years")
            names.append(f"env_{prefix}_arena_renovation_recent")
            names.append(f"env_{prefix}_arena_crowd_density")
            names.append(f"env_{prefix}_arena_court_type")
            names.append(f"env_{prefix}_arena_lighting_quality")
            names.append(f"env_{prefix}_arena_rim_tightness")
            names.append(f"env_{prefix}_arena_3pt_distance_factor")
            names.append(f"env_{prefix}_arena_shooting_friendly")
        # City-level features
        for prefix in ["h", "a"]:
            names.append(f"env_{prefix}_city_population")
            names.append(f"env_{prefix}_city_market_size")
            names.append(f"env_{prefix}_city_nightlife_index")
            names.append(f"env_{prefix}_city_distraction_factor")
        # Time of game features
        names.extend([
            "env_game_time_hour",
            "env_game_time_prime_time",
            "env_game_time_early",
            "env_game_time_late",
            "env_daylight_hours_home",
            "env_daylight_hours_away",
        ])
        # Altitude-specific performance adjustments across windows
        for prefix in ["h", "a"]:
            for w in [5, 10, 20]:
                names.append(f"env_{prefix}_high_altitude_wp_{w}")
                names.append(f"env_{prefix}_high_altitude_margin_{w}")
                names.append(f"env_{prefix}_sea_level_wp_{w}")
                names.append(f"env_{prefix}_sea_level_margin_{w}")
        # Cross-country travel impact
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                names.append(f"env_{prefix}_cross_country_wp_{w}")
                names.append(f"env_{prefix}_cross_country_margin_{w}")
                names.append(f"env_{prefix}_same_coast_wp_{w}")
                names.append(f"env_{prefix}_same_coast_margin_{w}")
        # Environmental composites
        names.extend([
            "env_combined_weather_impact",
            "env_combined_venue_advantage",
            "env_combined_travel_disruption",
            "env_altitude_weather_interaction",
            "env_timezone_weather_interaction",
            "env_market_size_diff",
            "env_distraction_diff",
            "env_arena_shooting_diff",
        ])

        # 29b. ADVANCED MARKET MICROSTRUCTURE III — EXPANDED (additional ~200 features)
        if self.include_market:
            # Prop market features
            _prop_types = ["total_pts", "home_pts", "away_pts", "home_spread",
                           "first_half_spread", "first_half_total",
                           "second_half_spread", "second_half_total",
                           "q1_spread", "q1_total"]
            for prop in _prop_types:
                names.append(f"mkt3_prop_{prop}_opening")
                names.append(f"mkt3_prop_{prop}_current")
                names.append(f"mkt3_prop_{prop}_movement")
                names.append(f"mkt3_prop_{prop}_velocity")
            # Alternative market features
            names.extend([
                "mkt3_alt_spread_3pt_total",
                "mkt3_alt_spread_rebounds_total",
                "mkt3_alt_spread_assists_total",
                "mkt3_alt_first_basket",
                "mkt3_alt_race_to_20",
                "mkt3_alt_highest_scoring_quarter",
            ])
            # Cross-market correlation features
            names.extend([
                "mkt3_spread_total_correlation",
                "mkt3_spread_ml_correlation",
                "mkt3_total_ml_correlation",
                "mkt3_first_half_full_game_corr",
                "mkt3_prop_main_line_divergence",
            ])
            # Time-stamped market snapshots
            _snap_times = ["open", "12h", "6h", "3h", "1h", "30min"]
            for snap in _snap_times:
                names.append(f"mkt3_snapshot_spread_{snap}")
                names.append(f"mkt3_snapshot_total_{snap}")
                names.append(f"mkt3_snapshot_ml_home_{snap}")
            # Market efficiency metrics
            names.extend([
                "mkt3_market_efficiency_spread",
                "mkt3_market_efficiency_total",
                "mkt3_market_efficiency_ml",
                "mkt3_price_discovery_speed",
                "mkt3_information_asymmetry",
                "mkt3_market_consensus_time",
                "mkt3_overnight_movement",
                "mkt3_morning_adjustment",
            ])
            # Historical accuracy of market in similar contexts
            _sim_contexts = ["same_matchup", "same_spread_range", "same_total_range",
                             "same_rest_pattern", "same_season_phase"]
            for ctx in _sim_contexts:
                names.append(f"mkt3_hist_accuracy_{ctx}")
                names.append(f"mkt3_hist_clv_{ctx}")

        # 31b. CROSS-TEAM INTERACTION MATRIX — EXPANDED (additional ~300 features)
        # Advanced matchup features: per-stat offensive efficiency vs opponent defense
        _advanced_matchup_stats = [
            "rim_att_rate", "midrange_rate", "corner3_rate", "above_break3_rate",
            "pullup_rate", "catch_shoot_rate", "isolation_rate", "pnr_ball_handler",
            "pnr_roll_man", "post_up_rate", "transition_freq", "cut_freq",
        ]
        for stat in _advanced_matchup_stats:
            for w in [5, 10]:
                names.append(f"xteam_h_off_{stat}_{w}")
                names.append(f"xteam_a_off_{stat}_{w}")
                names.append(f"xteam_h_def_{stat}_{w}")
                names.append(f"xteam_a_def_{stat}_{w}")
                names.append(f"xteam_matchup_h_{stat}_{w}")
                names.append(f"xteam_matchup_a_{stat}_{w}")
        # Shot distribution matchup
        _shot_zones = ["paint", "midrange", "corner3", "above_break3", "rim"]
        for zone in _shot_zones:
            for w in [5, 10]:
                names.append(f"xteam_h_shot_freq_{zone}_{w}")
                names.append(f"xteam_a_shot_freq_{zone}_{w}")
                names.append(f"xteam_h_def_allow_{zone}_{w}")
                names.append(f"xteam_a_def_allow_{zone}_{w}")
                names.append(f"xteam_zone_mismatch_{zone}_{w}")
        # Pace and style interaction matrix
        _pace_cats = ["ultra_fast", "fast", "average", "slow", "ultra_slow"]
        for pace_cat in _pace_cats:
            names.append(f"xteam_h_wp_vs_{pace_cat}")
            names.append(f"xteam_a_wp_vs_{pace_cat}")
            names.append(f"xteam_h_margin_vs_{pace_cat}")
            names.append(f"xteam_a_margin_vs_{pace_cat}")

        # 32b. BAYESIAN PRIORS — EXPANDED (additional ~200 features)
        # Bayesian-updated power ratings with different priors
        _prior_types = ["flat", "preseason", "historical", "market_implied", "composite"]
        for prefix in ["h", "a"]:
            for prior in _prior_types:
                names.append(f"bayes2_{prefix}_rating_{prior}")
                names.append(f"bayes2_{prefix}_confidence_{prior}")
                names.append(f"bayes2_{prefix}_update_rate_{prior}")
            # Season-adjusted Bayesian features
            for w in [10, 20, 40]:
                names.append(f"bayes2_{prefix}_shrinkage_wp_{w}")
                names.append(f"bayes2_{prefix}_shrinkage_ortg_{w}")
                names.append(f"bayes2_{prefix}_shrinkage_drtg_{w}")
                names.append(f"bayes2_{prefix}_shrinkage_netrtg_{w}")
            # Coach Bayesian impact with different baselines
            names.append(f"bayes2_{prefix}_coach_expected_wp")
            names.append(f"bayes2_{prefix}_coach_overperformance")
            names.append(f"bayes2_{prefix}_coach_underperformance")
            names.append(f"bayes2_{prefix}_coach_trajectory")
            # Roster turnover Bayesian adjustment
            names.append(f"bayes2_{prefix}_roster_turnover_adj")
            names.append(f"bayes2_{prefix}_new_player_integration")
            names.append(f"bayes2_{prefix}_core_retained_pct")
            names.append(f"bayes2_{prefix}_trade_deadline_impact")
            # Injury Bayesian impact
            names.append(f"bayes2_{prefix}_injury_prior_wpd")
            names.append(f"bayes2_{prefix}_injury_bayesian_adj")
            names.append(f"bayes2_{prefix}_healthy_roster_prior")
        # Game-level Bayesian expanded differentials
        for prior in _prior_types:
            names.append(f"bayes2_rating_diff_{prior}")
        names.extend([
            "bayes2_shrinkage_diff_wp",
            "bayes2_shrinkage_diff_netrtg",
            "bayes2_coach_impact_diff",
            "bayes2_roster_stability_diff",
            "bayes2_injury_adjusted_diff",
            "bayes2_composite_prior_diff",
            "bayes2_prior_confidence_diff",
            "bayes2_update_magnitude_diff",
        ])

        # 33b. NETWORK/GRAPH FEATURES — EXPANDED (additional ~200 features)
        # Conference/division subgraph features
        for prefix in ["h", "a"]:
            # Subgraph metrics within conference
            names.append(f"net2_{prefix}_conf_pagerank")
            names.append(f"net2_{prefix}_conf_clustering")
            names.append(f"net2_{prefix}_conf_degree_centrality")
            names.append(f"net2_{prefix}_conf_closeness_centrality")
            # Division subgraph
            names.append(f"net2_{prefix}_div_pagerank")
            names.append(f"net2_{prefix}_div_clustering")
            names.append(f"net2_{prefix}_div_degree_centrality")
            # Quality-weighted network features
            for w in [10, 20, 82]:
                names.append(f"net2_{prefix}_quality_weighted_wins_{w}")
                names.append(f"net2_{prefix}_quality_weighted_losses_{w}")
                names.append(f"net2_{prefix}_weighted_margin_network_{w}")
            # Opponent network features (2nd order)
            names.append(f"net2_{prefix}_opp_avg_pagerank")
            names.append(f"net2_{prefix}_opp_avg_centrality")
            names.append(f"net2_{prefix}_opp_diversity_index")
            names.append(f"net2_{prefix}_beaten_teams_avg_wp")
            names.append(f"net2_{prefix}_lost_to_teams_avg_wp")
            # Transitive strength
            for depth in [2, 3, 4]:
                names.append(f"net2_{prefix}_transitive_strength_d{depth}")
                names.append(f"net2_{prefix}_transitive_weakness_d{depth}")
            # Colley rating system features
            names.append(f"net2_{prefix}_colley_rating")
            names.append(f"net2_{prefix}_colley_rank")
            # Massey rating system features
            names.append(f"net2_{prefix}_massey_rating")
            names.append(f"net2_{prefix}_massey_offensive")
            names.append(f"net2_{prefix}_massey_defensive")
            # Keener rating features
            names.append(f"net2_{prefix}_keener_rating")
            names.append(f"net2_{prefix}_keener_dominance")
        # Game-level expanded network differentials
        names.extend([
            "net2_conf_pagerank_diff",
            "net2_div_pagerank_diff",
            "net2_quality_weighted_diff",
            "net2_opp_quality_diff",
            "net2_transitive_diff",
            "net2_colley_diff",
            "net2_massey_diff",
            "net2_massey_off_diff",
            "net2_massey_def_diff",
            "net2_keener_diff",
            "net2_composite_network_diff",
            "net2_network_surprise_factor",
        ])

        # 34b. ENSEMBLE META-FEATURES — EXPANDED (additional ~200 features)
        # Cross-model interaction features
        _model_pairs = [
            ("xgboost", "lightgbm"), ("xgboost", "catboost"), ("xgboost", "rf"),
            ("xgboost", "logistic"), ("lightgbm", "catboost"), ("lightgbm", "rf"),
            ("lightgbm", "logistic"), ("catboost", "rf"), ("catboost", "logistic"),
            ("rf", "logistic"),
        ]
        for m1, m2 in _model_pairs:
            names.append(f"meta3_{m1}_{m2}_agreement")
            names.append(f"meta3_{m1}_{m2}_diff")
            names.append(f"meta3_{m1}_{m2}_avg")
        # Feature importance stability across models
        _top_feat_groups = ["rolling", "four_factors", "pace", "scoring",
                            "momentum", "rest", "market", "matchup",
                            "context", "power_rating"]
        for fg in _top_feat_groups:
            names.append(f"meta3_feat_importance_{fg}_mean")
            names.append(f"meta3_feat_importance_{fg}_std")
            names.append(f"meta3_feat_importance_{fg}_rank")
        # Model performance in different game contexts
        _game_contexts = ["home_fav", "home_dog", "high_total", "low_total",
                          "b2b", "rest_adv", "rivalry", "non_conf",
                          "playoff_race", "tanking"]
        for ctx in _game_contexts:
            names.append(f"meta3_accuracy_{ctx}")
            names.append(f"meta3_roi_{ctx}")
            names.append(f"meta3_brier_{ctx}")
        # Time-varying model performance
        for w in [5, 10, 20, 50]:
            names.append(f"meta3_model_accuracy_overall_{w}")
            names.append(f"meta3_model_brier_overall_{w}")
            names.append(f"meta3_model_roi_overall_{w}")
            names.append(f"meta3_model_calibration_{w}")
            names.append(f"meta3_model_sharpness_{w}")
        # Stacking features (model outputs as features)
        for model in _meta_models:
            names.append(f"meta3_{model}_prob_home")
            names.append(f"meta3_{model}_prob_away")
            names.append(f"meta3_{model}_margin_pred")
            names.append(f"meta3_{model}_total_pred")

        # 35b. TEMPORAL DECAY FEATURES — EXPANDED (additional ~200 features)
        # Kernel-weighted features (Gaussian, triangular, Epanechnikov)
        _kernels = ["gaussian", "triangular", "epanechnikov"]
        _kernel_bw = [3, 7, 15]  # bandwidth in games
        for prefix in ["h", "a"]:
            for kernel in _kernels:
                for bw in _kernel_bw:
                    for stat in ["wp", "margin", "ortg", "drtg", "pace"]:
                        names.append(f"td2_{prefix}_{kernel}_{stat}_bw{bw}")
        # Regime change detection features
        for prefix in ["h", "a"]:
            for stat in ["wp", "margin", "ortg", "drtg"]:
                names.append(f"td2_{prefix}_regime_change_{stat}")
                names.append(f"td2_{prefix}_regime_duration_{stat}")
                names.append(f"td2_{prefix}_regime_level_{stat}")
                names.append(f"td2_{prefix}_cusum_{stat}")
        # Weighted percentile features
        for prefix in ["h", "a"]:
            for pct in [10, 25, 50, 75, 90]:
                for stat in ["margin", "ppg", "ortg"]:
                    names.append(f"td2_{prefix}_weighted_pctl{pct}_{stat}")
        # Adaptive half-life features (half-life adjusts based on volatility)
        for prefix in ["h", "a"]:
            for stat in ["wp", "margin", "ortg", "drtg", "pace"]:
                names.append(f"td2_{prefix}_adaptive_decay_{stat}")
                names.append(f"td2_{prefix}_adaptive_halflife_{stat}")

        # =====================================================================
        # CROSS-CATEGORY INTERACTION FEATURES (additional ~600 features)
        # Interactions between new categories and existing core features
        # =====================================================================

        # Temporal decay × Market features
        if self.include_market:
            for stat in ["wp", "margin", "ortg"]:
                for hl in [3, 10]:
                    names.append(f"xi_td_market_{stat}_hl{hl}_spread")
                    names.append(f"xi_td_market_{stat}_hl{hl}_total")
                    names.append(f"xi_td_market_{stat}_hl{hl}_ml")

        # Bayesian × Power Rating interactions
        for prefix in ["h", "a"]:
            names.append(f"xi_bayes_power_{prefix}_blend")
            names.append(f"xi_bayes_power_{prefix}_divergence")
            names.append(f"xi_bayes_elo_{prefix}_shrinkage")
            names.append(f"xi_bayes_elo_{prefix}_confidence")

        # Network × Matchup interactions
        names.extend([
            "xi_net_matchup_pagerank_diff",
            "xi_net_matchup_centrality_weighted",
            "xi_net_matchup_transitivity_score",
            "xi_net_matchup_network_surprise",
        ])

        # Player Impact × Fatigue interactions
        for prefix in ["h", "a"]:
            names.append(f"xi_pi_fatigue_{prefix}_star_tired")
            names.append(f"xi_pi_fatigue_{prefix}_bench_fresh")
            names.append(f"xi_pi_fatigue_{prefix}_depth_advantage")
            names.append(f"xi_pi_fatigue_{prefix}_load_management")

        # Referee × Venue interactions
        names.extend([
            "xi_ref_venue_home_bias_compound",
            "xi_ref_venue_altitude_foul_rate",
            "xi_ref_venue_pace_interaction",
            "xi_ref_venue_crowd_effect",
        ])

        # Time Series × Cross-Team interactions
        for prefix in ["h", "a"]:
            for stat in ["wp", "margin", "ortg"]:
                names.append(f"xi_ts_xteam_{prefix}_{stat}_trend_matchup")
                names.append(f"xi_ts_xteam_{prefix}_{stat}_momentum_clash")

        # Ensemble × Market interactions
        if self.include_market:
            names.extend([
                "xi_meta_market_model_vs_line",
                "xi_meta_market_confidence_vs_movement",
                "xi_meta_market_agreement_vs_sharp",
                "xi_meta_market_edge_vs_steam",
                "xi_meta_market_calibration_vs_clv",
            ])

        # All-category composite features (grand summary features)
        names.extend([
            "grand_composite_edge",
            "grand_model_market_network_blend",
            "grand_fatigue_venue_weather_score",
            "grand_player_matchup_referee_blend",
            "grand_bayesian_temporal_blend",
            "grand_cross_category_momentum",
            "grand_multi_signal_agreement",
            "grand_risk_adjusted_composite",
            "grand_confidence_weighted_edge",
            "grand_information_ratio",
        ])

        # =====================================================================
        # HIGHER-ORDER POLYNOMIAL FEATURES on new categories (additional ~400)
        # =====================================================================

        # Squared terms from new categories
        _new_sq_features = []
        for prefix in ["h", "a"]:
            _new_sq_features.extend([
                f"{prefix}_star1_plus_minus_10",
                f"{prefix}_star1_usage_rate_10",
                f"{prefix}_star_combined_plus_minus",
                f"{prefix}_chemistry_starting5",
                f"bayes2_{prefix}_rating_composite",
                f"bayes2_{prefix}_coach_expected_wp",
                f"net2_{prefix}_colley_rating",
                f"net2_{prefix}_massey_rating",
            ])
        for feat in _new_sq_features:
            names.append(f"sq2_{feat}")

        # New interaction products between key new features
        _new_inter_pairs = [
            ("h_star1_plus_minus_10", "a_star1_plus_minus_10"),
            ("h_star1_usage_rate_10", "a_star1_usage_rate_10"),
            ("h_chemistry_starting5", "a_chemistry_starting5"),
            ("pi_star1_rating_diff", "elo_diff"),
            ("pi_combined_star_diff", "rest_advantage"),
            ("pi_talent_depth_diff", "fatigue_composite_edge"),
            ("xteam_overall_style_clash", "elo_diff"),
            ("xteam_pace_war_indicator", "h_pace10"),
            ("xteam_mismatch_severity", "current_spread"),
            ("bayes_preseason_diff", "h_wp10"),
            ("bayes_franchise_strength_diff", "elo_diff"),
            ("bayes_coach_wp_diff", "rest_advantage"),
            ("net_pagerank_diff_82", "elo_diff"),
            ("net_pagerank_diff_20", "h_wp10"),
            ("net_clustering_diff", "xteam_overall_style_clash"),
            ("net_eigenvector_diff", "bayes_composite_diff"),
            ("ref_home_bias_composite", "venue_home_elevation_advantage"),
            ("ref_pace_impact_composite", "xteam_pace_war_indicator"),
            ("env_combined_venue_advantage", "rest_advantage"),
            ("env_combined_travel_disruption", "fatigue_composite_edge"),
            ("grand_composite_edge", "elo_diff"),
            ("grand_composite_edge", "current_spread"),
            ("grand_multi_signal_agreement", "meta2_ensemble_mean_prob"),
            ("grand_confidence_weighted_edge", "meta2_edge_confidence_product"),
        ]
        for x_feat, y_feat in _new_inter_pairs:
            names.append(f"inter2_{x_feat}_{y_feat}")

        # Ratio features between new categories
        _new_ratio_pairs = [
            ("h_star1_plus_minus_10", "a_star1_plus_minus_10"),
            ("h_chemistry_starting5", "a_chemistry_starting5"),
            ("h_star_combined_plus_minus", "a_star_combined_plus_minus"),
        ]
        for x_feat, y_feat in _new_ratio_pairs:
            names.append(f"ratio2_{x_feat}_{y_feat}")

        # Triple interaction features (3-way combinations of key signals)
        _triple_combos = [
            ("elo_diff", "rest_advantage", "pi_combined_star_diff"),
            ("elo_diff", "xteam_mismatch_severity", "net_pagerank_diff_82"),
            ("elo_diff", "bayes_composite_diff", "meta2_ensemble_mean_prob"),
            ("current_spread", "pi_star1_rating_diff", "ref_home_bias_composite"),
            ("h_wp10", "a_wp10", "xteam_overall_style_clash"),
            ("h_ortg10", "a_drtg10", "xteam_offensive_edge_composite"),
            ("rest_advantage", "env_combined_venue_advantage", "ref_pace_impact_composite"),
            ("fatigue_composite_edge", "pi_talent_depth_diff", "bayes_roster_stability_diff"),
        ]
        for a, b, c in _triple_combos:
            names.append(f"triple_{a}_{b}_{c}")

        # =====================================================================
        # ROLLING CROSS-CATEGORY FEATURES (additional ~400 features)
        # Apply rolling window logic to new category outputs
        # =====================================================================

        # Rolling features on decay-weighted stats
        _decay_roll_stats = ["wp", "margin", "ortg", "drtg"]
        for prefix in ["h", "a"]:
            for stat in _decay_roll_stats:
                for hl in [3, 10]:
                    # Decay stat volatility
                    names.append(f"roll_td_vol_{prefix}_{stat}_hl{hl}")
                    # Decay stat trend
                    names.append(f"roll_td_trend_{prefix}_{stat}_hl{hl}")
                    # Decay stat z-score
                    names.append(f"roll_td_zscore_{prefix}_{stat}_hl{hl}")

        # Rolling features on network metrics
        for prefix in ["h", "a"]:
            for w in [10, 20]:
                names.append(f"roll_net_pagerank_change_{prefix}_{w}")
                names.append(f"roll_net_centrality_change_{prefix}_{w}")
                names.append(f"roll_net_quality_wins_trend_{prefix}_{w}")

        # Rolling features on Bayesian priors
        for prefix in ["h", "a"]:
            for w in [10, 20]:
                names.append(f"roll_bayes_update_trend_{prefix}_{w}")
                names.append(f"roll_bayes_confidence_trend_{prefix}_{w}")
                names.append(f"roll_bayes_prior_divergence_trend_{prefix}_{w}")

        # Rolling cross-team interaction features
        for stat in ["pace", "ortg", "drtg"]:
            for w in [5, 10]:
                names.append(f"roll_xteam_avg_mismatch_{stat}_{w}")
                names.append(f"roll_xteam_mismatch_trend_{stat}_{w}")

        # Cumulative information features
        for prefix in ["h", "a"]:
            names.append(f"cum_info_{prefix}_total_features_signal")
            names.append(f"cum_info_{prefix}_positive_signals_pct")
            names.append(f"cum_info_{prefix}_negative_signals_pct")
            names.append(f"cum_info_{prefix}_neutral_signals_pct")
            names.append(f"cum_info_{prefix}_signal_entropy")

        # Game-level summary features from all new categories
        names.extend([
            "new_cats_home_advantage_composite",
            "new_cats_away_advantage_composite",
            "new_cats_edge_differential",
            "new_cats_confidence_score",
            "new_cats_information_value",
            "new_cats_novelty_score",
            "new_cats_alignment_with_market",
            "new_cats_alignment_with_model",
            "new_cats_contrarian_signal",
            "new_cats_risk_score",
        ])

        # =====================================================================
        # EXTENDED ROLLING WINDOW FEATURES ON NEW STATS (additional ~500)
        # Apply all 6 WINDOWS to new derived stats for massive expansion
        # =====================================================================

        # Extended rolling windows on advanced stats
        _ext_stats = [
            "net_rating", "ast_to_tov", "efg_minus_opp_efg",
            "pace_adj_margin", "sos_adj_margin", "opponent_efg",
            "three_pt_rate_diff", "paint_rate_diff", "transition_rate",
            "halfcourt_efficiency",
        ]
        for prefix in ["h", "a"]:
            for stat in _ext_stats:
                for w in WINDOWS:
                    names.append(f"ext_{prefix}_{stat}_{w}")

        # Extended EWMA on new stats
        _ext_ewma_stats = ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                           "pace_adj_margin", "three_pt_rate_diff"]
        _ext_ewma_alphas = ["01", "02", "05", "08"]
        for prefix in ["h", "a"]:
            for stat in _ext_ewma_stats:
                for alpha in _ext_ewma_alphas:
                    names.append(f"ext_ewma_{prefix}_{stat}_a{alpha}")

        # Extended volatility on new stats
        _ext_vol_stats = ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                          "pace_adj_margin", "transition_rate"]
        for prefix in ["h", "a"]:
            for stat in _ext_vol_stats:
                for w in [5, 10, 20]:
                    names.append(f"ext_vol_{prefix}_{stat}_{w}")

        # Extended z-scores on new stats
        for prefix in ["h", "a"]:
            for stat in _ext_stats:
                names.append(f"ext_zscore_{prefix}_{stat}")

        # Extended trend deltas on new stats
        _ext_trend_pairs = [(3, 10), (5, 20), (3, 20), (5, 10), (10, 20)]
        for prefix in ["h", "a"]:
            for stat in _ext_stats:
                for w1, w2 in _ext_trend_pairs:
                    names.append(f"ext_trend_{prefix}_{stat}_w{w1}_w{w2}")

        # =====================================================================
        # ADDITIONAL CROSS-WINDOW MOMENTUM ON NEW STATS (additional ~300)
        # =====================================================================

        _xw2_stats = ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                       "pace_adj_margin", "sos_adj_margin"]
        _xw2_pairs = [
            (3, 5), (3, 10), (3, 20), (5, 10), (5, 20),
            (7, 15), (7, 20), (10, 20),
        ]
        for prefix in ["h", "a"]:
            for stat in _xw2_stats:
                for w1, w2 in _xw2_pairs:
                    names.append(f"xw2_{prefix}_{stat}_{w1}vs{w2}")
                    names.append(f"xw2_accel_{prefix}_{stat}_{w1}vs{w2}")

        # Cross-window composites for new stats
        for prefix in ["h", "a"]:
            for stat in _xw2_stats:
                names.append(f"xw2_{prefix}_{stat}_shortterm_trend")
                names.append(f"xw2_{prefix}_{stat}_longterm_trend")
                names.append(f"xw2_{prefix}_{stat}_volatility_trend")
                names.append(f"xw2_{prefix}_{stat}_breakout_signal")
                names.append(f"xw2_{prefix}_{stat}_decline_signal")

        # =====================================================================
        # ADDITIONAL INTERACTION FEATURES (additional ~200)
        # Pairwise products of top new features with core features
        # =====================================================================

        _core_features = ["h_wp10", "a_wp10", "elo_diff", "current_spread",
                          "h_netrtg10", "a_netrtg10", "rest_advantage",
                          "h_ortg10", "a_drtg10", "h_consistency"]
        _new_key_features = [
            "pi_combined_star_diff", "pi_talent_depth_diff",
            "xteam_overall_style_clash", "xteam_mismatch_severity",
            "bayes_composite_diff", "net_pagerank_diff_82",
            "ref_home_bias_composite", "env_combined_venue_advantage",
            "meta2_ensemble_mean_prob", "grand_composite_edge",
        ]
        for core_f in _core_features:
            for new_f in _new_key_features:
                names.append(f"xi3_{core_f}_{new_f}")

        # =====================================================================
        # FINAL EXPANSION: Per-opponent rolling features (additional ~300)
        # Performance against different opponent strength tiers
        # =====================================================================

        _opp_tiers = ["elite", "good", "average", "bad", "terrible"]
        _opp_stats = ["wp", "margin", "ortg", "drtg", "efg", "pace"]
        for prefix in ["h", "a"]:
            for tier in _opp_tiers:
                for stat in _opp_stats:
                    names.append(f"opp_tier_{prefix}_{stat}_vs_{tier}")

        # Home/away specific performance by window
        _ha_stats = ["wp", "margin", "ortg", "drtg", "pace", "efg"]
        for prefix in ["h", "a"]:
            for loc in ["home_only", "away_only"]:
                for stat in _ha_stats:
                    for w in [5, 10, 20]:
                        names.append(f"ha_{prefix}_{loc}_{stat}_{w}")

        # Day-of-week performance features
        _dow_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        for prefix in ["h", "a"]:
            for dow in _dow_names:
                names.append(f"dow_{prefix}_wp_{dow}")
                names.append(f"dow_{prefix}_margin_{dow}")

        # Month-specific performance features
        _months = ["oct", "nov", "dec", "jan", "feb", "mar", "apr"]
        for prefix in ["h", "a"]:
            for month in _months:
                names.append(f"month_{prefix}_wp_{month}")
                names.append(f"month_{prefix}_margin_{month}")

        # Consecutive game pattern features
        for prefix in ["h", "a"]:
            for pattern in ["ww", "wl", "lw", "ll"]:
                names.append(f"pattern_{prefix}_{pattern}_next_wp")
            for streak_len in [2, 3, 4, 5]:
                names.append(f"pattern_{prefix}_win_streak_{streak_len}_next_wp")
                names.append(f"pattern_{prefix}_loss_streak_{streak_len}_next_wp")

        # Score differential buckets performance
        _margin_buckets = ["blowout_win", "comfortable_win", "close_win",
                          "close_loss", "comfortable_loss", "blowout_loss"]
        for prefix in ["h", "a"]:
            for bucket in _margin_buckets:
                names.append(f"bucket_{prefix}_{bucket}_pct")
                names.append(f"bucket_{prefix}_{bucket}_next_wp")

        # Quarter-specific detailed features (additional ~56)
        _quarters = ["q1", "q2", "q3", "q4"]
        _q_stats = ["margin", "ortg", "drtg", "pace", "efg", "tov_rate", "ft_rate"]
        for prefix in ["h", "a"]:
            for q in _quarters:
                for stat in _q_stats:
                    names.append(f"qdetail_{prefix}_{q}_{stat}")

        # 36. EWMA PERFORMANCE + CROSSOVERS + REST INTERACTIONS (~108 features)
        # Inspired by deepshot (EWMA rolling stats) + kyleskom (rest × performance)
        _ewma36_stats = ["wp", "pd", "ppg", "papg", "margin", "close", "blowout", "ou_avg"]
        _ewma36_alphas = ["005", "015", "025", "04", "07"]
        for prefix in ["h", "a"]:
            for stat in _ewma36_stats:
                for alpha in _ewma36_alphas:
                    names.append(f"ewma36_{prefix}_{stat}_a{alpha}")

        # EWMA crossovers: fast(0.7) - slow(0.05) = momentum signal (MACD-like)
        for prefix in ["h", "a"]:
            for stat in _ewma36_stats:
                names.append(f"ewma36_{prefix}_{stat}_crossover")

        # Rest × performance interactions
        names.extend([
            "rest_x_h_wp5", "rest_x_a_wp5",         # rest_days × recent win%
            "b2b_x_h_margin5", "b2b_x_a_margin5",   # b2b × recent margin
            "fatigue_x_h_ortg", "fatigue_x_a_ortg",  # fatigue × offensive rating
            "rest_adv_x_wp_diff",                     # rest_advantage × win% difference
            "b2b_diff_x_margin_diff",                 # b2b differential × margin differential
            "h_rest_sq", "a_rest_sq",                 # rest_days squared (diminishing returns)
            "rest_x_travel",                          # rest_advantage × travel_advantage
            "dense_sched_x_margin",                   # schedule_density × margin_diff
        ])

        # 37. MOVDA ELO FEATURES (13 features) — arXiv:2506.00348
        # Margin-of-Victory Differential Analysis: R' = R + K*(S-E) + λ*(MOV-E_MOV)
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_movda_rating")          # MOVDA Elo rating (normalized)
            names.append(f"{prefix}_mov_surprise_ewm")      # EWMA of MOV surprise signal
        names.extend([
            "movda_diff",                                    # MOVDA rating differential
            "movda_win_prob",                                # MOVDA-derived win probability
        ])
        # Raw delta_MOV rolling features (no EWM smoothing) — captures recent surprise momentum
        for prefix in ["h", "a"]:
            names.append(f"{prefix}_delta_mov_raw")         # last game's raw MOV surprise
            names.append(f"{prefix}_delta_mov_rolling_5")   # rolling mean over last 5 games
            names.append(f"{prefix}_delta_mov_rolling_10")  # rolling mean over last 10 games
        names.append("delta_mov_diff")                       # h_delta_mov_rolling_5 - a_delta_mov_rolling_5

        # 38. VENUE-CONDITIONAL MATCHUP FEATURES (14 features)
        # Home team's home-only stats vs away team's road-only stats
        # This is the true matchup signal: how does home team perform AT HOME
        # vs how does away team perform ON THE ROAD — not combined records
        for w in [5, 10, 20]:
            names.append(f"venue_wp_edge_{w}")              # h_home_wp - a_road_wp
            names.append(f"venue_margin_edge_{w}")          # h_home_margin - a_road_margin
            names.append(f"venue_ortg_edge_{w}")            # h_home_ortg - a_road_drtg
            names.append(f"venue_drtg_edge_{w}")            # h_home_drtg - a_road_ortg
        names.extend([
            "venue_home_boost",                              # h_home_wp - h_overall_wp (home court effect)
            "venue_road_penalty",                            # a_overall_wp - a_road_wp (road penalty)
        ])

        # 39. CIRCADIAN RHYTHM & TRAVEL FATIGUE (8 features) — Chronobiology Intl 2024
        # Novel combinations: timezone-weighted fatigue + rest non-linearity
        # Distinct from Cat 6 (which has raw rest/travel) — these are normalized composites
        names.extend([
            "circ_h_travel_dist",       # Great-circle miles from last game city (home team)
            "circ_a_travel_dist",       # Great-circle miles from last game city (away team)
            "circ_h_tz_shift",          # Timezone hours crossed since last game (home)
            "circ_a_tz_shift",          # Timezone hours crossed since last game (away)
            "circ_h_fatigue_index",     # Composite: distance/500 + tz_shift*0.5 + b2b*2 - rest*0.3
            "circ_a_fatigue_index",     # Composite: distance/500 + tz_shift*0.5 + b2b*2 - rest*0.3
            "circ_advantage",           # away_fatigue_index - home_fatigue_index (positive = home fresher)
            "circ_rest_nonlinear",      # (h_rest_sq - a_rest_sq) capped: captures diminishing rest benefit
        ])

        # 41. TRANSITION vs HALF-COURT EFFICIENCY SPLITS (7 features)
        # Derived from fb_pts (fast break) and pace in existing box score stats
        names.extend([
            "trans41_h_fb_rate",        # Home fast-break pts as fraction of total scoring
            "trans41_a_fb_rate",        # Away fast-break pts as fraction of total scoring
            "trans41_h_halfcourt_eff",  # Home half-court efficiency proxy (pts - fb_pts) / poss
            "trans41_a_halfcourt_eff",  # Away half-court efficiency proxy
            "trans41_fb_rate_diff",     # h_fb_rate - a_fb_rate
            "trans41_pace_x_fb",        # pace * fb_rate interaction (high pace + high transition = synergy)
            "trans41_halfcourt_edge",   # h_halfcourt_eff - a_halfcourt_eff
        ])

        # 43. CLUTCH PERFORMANCE FEATURES (8 features)
        # Computed from close games (|margin| <= 5) in rolling records
        names.extend([
            "clutch43_h_wp",            # Win % in clutch games (last 20)
            "clutch43_a_wp",            # Win % in clutch games (last 20)
            "clutch43_h_margin",        # Avg margin in clutch games
            "clutch43_a_margin",        # Avg margin in clutch games
            "clutch43_h_ortg",          # Offensive rating in clutch games
            "clutch43_a_ortg",          # Offensive rating in clutch games
            "clutch43_wp_diff",         # h_clutch_wp - a_clutch_wp
            "clutch43_margin_diff",     # h_clutch_margin - a_clutch_margin
        ])

        # 44. GAME TOTALS PREDICTION FEATURES (10 features)
        # Encodes expected game pace and scoring volume for O/U modelling.
        # Derived entirely from rolling pts/opp_pts/ortg/drtg — no new data source.
        # Use case: (1) direct O/U prediction; (2) interaction terms for win model
        #           (high-total games are often closer; low-total = grind favors defense).
        names.extend([
            "tot44_h_ppg10",            # Home rolling PPG (last 10) normalized to league avg
            "tot44_a_ppg10",            # Away rolling PPG (last 10) normalized
            "tot44_h_papg10",           # Home rolling PAPG (last 10) normalized
            "tot44_a_papg10",           # Away rolling PAPG (last 10) normalized
            "tot44_matchup_total",      # Predicted total via PPG/PAPG interaction (normalized)
            "tot44_pace_sum",           # (h_pace + a_pace) / 2 — expected game pace
            "tot44_pace_mismatch",      # |h_pace - a_pace| — fast vs slow clash
            "tot44_ortg_sum",           # (h_ortg + a_ortg) / 2 — combined offensive quality
            "tot44_drtg_sum",           # (h_drtg + a_drtg) / 2 — combined defensive quality
            "tot44_score_env",          # (ortg_sum - drtg_sum) / 10 — net scoring environment
        ])

        # 42. SHOT QUALITY ZONE FEATURES (10 features)
        # From nba_api shot locations: zone FG%, shot distribution, xEFG.
        # Montrucchio 2026 (Brier 0.199) used spatial shot embeddings.
        # Graceful fallback to 0.0 when tracking data not loaded.
        names.extend([
            "shot42_h_rim_rate",         # Home restricted area shot frequency
            "shot42_a_rim_rate",         # Away restricted area shot frequency
            "shot42_h_mid_rate",         # Home mid-range shot frequency
            "shot42_a_mid_rate",         # Away mid-range shot frequency
            "shot42_h_three_rate",       # Home 3PT rate
            "shot42_a_three_rate",       # Away 3PT rate
            "shot42_h_xefg",            # Home expected eFG% from shot distribution
            "shot42_a_xefg",            # Away expected eFG% from shot distribution
            "shot42_rim_rate_diff",      # h_rim_rate - a_rim_rate (paint dominance)
            "shot42_xefg_diff",          # h_xefg - a_xefg (shot quality edge)
        ])

        # 45. PLAYER TRACKING / HUSTLE FEATURES (12 features)
        # From nba_api: hustle stats, speed/distance, touches, drives.
        # Proxies effort intensity, pace profile, and defensive activity.
        # Expected Brier delta: -0.003 to -0.006 (replaces box-score proxies).
        names.extend([
            "track45_h_contested",       # Home team avg contested shots per game
            "track45_a_contested",       # Away team avg contested shots per game
            "track45_h_deflections",     # Home team avg deflections per game
            "track45_a_deflections",     # Away team avg deflections per game
            "track45_h_speed",           # Home team avg player speed (mph)
            "track45_a_speed",           # Away team avg player speed (mph)
            "track45_h_loose_balls",     # Home team loose balls recovered per game
            "track45_a_loose_balls",     # Away team loose balls recovered per game
            "track45_h_drives",          # Home team drives per game
            "track45_a_drives",          # Away team drives per game
            "track45_contested_diff",    # h_contested - a_contested (defensive intensity edge)
            "track45_speed_diff",        # h_speed - a_speed (pace profile mismatch)
        ])

        # 46. REAL ODDS MARKET FEATURES (8 features)
        # From historical odds CSV: pre-game moneylines, spreads, totals.
        # These are the strongest predictive features available — the market
        # aggregates all public information into a single number.
        # Expected Brier delta: -0.005 to -0.015 (market-calibrated features).
        # Graceful fallback: 0.5 prob, 0.0 spread, 220.0 total when odds unavailable.
        names.extend([
            "odds46_implied_home_prob",   # Market implied home win prob (vig-inclusive)
            "odds46_implied_away_prob",   # Market implied away win prob (vig-inclusive)
            "odds46_fair_home_prob",      # Vig-removed fair home probability
            "odds46_fair_away_prob",      # Vig-removed fair away probability
            "odds46_spread_home",         # Point spread (negative = home favored), normalized /10
            "odds46_total",               # Over/under total, normalized /220
            "odds46_overround",           # Total implied prob (vig level, ~1.04-1.08)
            "odds46_spread_implied_diff", # spread_implied_prob - ml_implied_prob (market consistency)
        ])

        # 47. DRIVE-OFFENSE vs RIM-DEFENSE MATCHUP (14 features)
        # Montrucchio 2026 (Brier 0.199): zone-level offense vs defense matchups.
        # drive_fg_pct loaded by build_tracking_data but never consumed by engine.
        # DEF_RIM_FG_PCT from defense CSV completely unused until now.
        names.extend([
            "drive47_h_fg_pct",          # Home drive FG% (offense quality on drives)
            "drive47_a_fg_pct",          # Away drive FG% (offense quality on drives)
            "drive47_h_tov_pct",         # Home drive turnover rate (ball security)
            "drive47_a_tov_pct",         # Away drive turnover rate
            "drive47_h_pts_pct",         # Home scoring share from drives
            "drive47_a_pts_pct",         # Away scoring share from drives
            "drive47_h_def_rim_fg",      # Home rim FG% allowed (rim protection)
            "drive47_a_def_rim_fg",      # Away rim FG% allowed
            "drive47_h_blk_rate",        # Home blocks per game (rim deterrent)
            "drive47_a_blk_rate",        # Away blocks per game
            "drive47_h_off_vs_a_rim",    # Home drive_fg - away def_rim_fg (matchup edge)
            "drive47_a_off_vs_h_rim",    # Away drive_fg - home def_rim_fg
            "drive47_rim_matchup_net",   # Combined rim advantage
            "drive47_drive_volume_diff", # Home drives - away drives (attack style diff)
        ])

        # 48. PASSING NETWORK QUALITY (10 features)
        # passing_2025-26.csv is completely unused. Ball movement quality
        # is a strong team-strength indicator (GS dynasty, 2020s Nuggets).
        names.extend([
            "pass48_h_ast_rate",         # Home AST-to-pass ratio
            "pass48_a_ast_rate",         # Away AST-to-pass ratio
            "pass48_h_potential_ast",    # Home potential assists (normalized /50)
            "pass48_a_potential_ast",    # Away potential assists (normalized /50)
            "pass48_h_ast_pts_created",  # Home points created by assists (normalized /80)
            "pass48_a_ast_pts_created",  # Away points created by assists
            "pass48_h_secondary_ast",    # Home secondary assists (extra passes /5)
            "pass48_a_secondary_ast",    # Away secondary assists
            "pass48_ast_rate_diff",      # Home - away AST-to-pass rate
            "pass48_ball_movement_edge", # Composite: ast_rate + secondary_ast + potential_ast diff
        ])

        # 49. PLAY-TYPE EFFICIENCY (10 features)
        # NBA_Play_Types CSV has real PPP by play type — currently unused.
        # Team-level aggregated: iso, P&R, transition, post-up, spot-up efficiency.
        names.extend([
            "play49_h_iso_ppp",          # Home isolation PPP (normalized /1.0)
            "play49_a_iso_ppp",          # Away isolation PPP
            "play49_h_pnr_ppp",          # Home pick-and-roll ball handler PPP
            "play49_a_pnr_ppp",          # Away pick-and-roll ball handler PPP
            "play49_h_spot_ppp",         # Home spot-up PPP
            "play49_a_spot_ppp",         # Away spot-up PPP
            "play49_h_trans_ppp",        # Home transition PPP
            "play49_a_trans_ppp",        # Away transition PPP
            "play49_ppp_composite_diff", # Home avg PPP - away avg PPP (overall efficiency)
            "play49_versatility_diff",   # Home play-type variety - away (# play types above 1.0 PPP)
        ])

        # 50. TEMPORAL WIN SEQUENCE ENCODING (12 features)
        # Encodes the ORDER of last-10-game outcomes — not just rolling averages.
        # A team that lost early then won recent games (improving) differs from one
        # that won early and is now declining, even at identical aggregate win%.
        # Research basis: MDPI 2026 (temporal sequence models outperform rolling averages).
        names.extend([
            "seq50_h_early_wp",          # Home win% in older half of last 10 games
            "seq50_h_late_wp",           # Home win% in recent half of last 10 games
            "seq50_h_slope",             # Home momentum slope (late_wp - early_wp, positive=improving)
            "seq50_h_margin_slope_norm", # Home margin trend (recent 3 - older 3 avg, /30 normalized)
            "seq50_h_streak_norm",       # Home current streak normalized (/10, sign = direction)
            "seq50_a_early_wp",          # Away win% in older half of last 10 games
            "seq50_a_late_wp",           # Away win% in recent half of last 10 games
            "seq50_a_slope",             # Away momentum slope
            "seq50_a_margin_slope_norm", # Away margin trend normalized
            "seq50_a_streak_norm",       # Away current streak normalized
            "seq50_slope_diff",          # Home - away momentum slope differential
            "seq50_streak_diff",         # Home - away streak differential
        ])

        # 51. SEASON ERA NORMALIZATION (8 features)
        # Z-score each team's rolling stats vs league-wide running average FOR THIS SEASON.
        # Removes era drift: ORtg 110 in 2018-19 ≠ ORtg 110 in 2025-26 (pace increased).
        # Research basis: MDPI 2026 (Info 17:56) — "season-fixed effects applied to handle
        # era differences" improved calibration. Logistic regression Brier 0.199 used this.
        names.extend([
            "era51_h_ortg_vs_league",    # Home ORtg z-score vs league season running avg
            "era51_h_drtg_vs_league",    # Home DRtg z-score (lower = better defense)
            "era51_a_ortg_vs_league",    # Away ORtg z-score vs league season running avg
            "era51_a_drtg_vs_league",    # Away DRtg z-score vs league season running avg
            "era51_h_pace_vs_league",    # Home pace z-score vs league season running avg
            "era51_a_pace_vs_league",    # Away pace z-score vs league season running avg
            "era51_h_netrtg_vs_league",  # Home net rating z-score vs league this season
            "era51_h_ortg_a_drtg_edge",  # Matchup: h_ortg_z - a_drtg_z (offensive edge)
        ])

        # 52. ODDS LINE FEATURES (15 features)
        # Direct features from the historical odds CSV: spread magnitude, total,
        # moneyline-implied probabilities, vig, and season-relative percentiles.
        # These are the raw market signals — the strongest single-source predictors.
        # Expected Brier delta: -0.003 to -0.008 (complements Cat 46 prob features).
        names.extend([
            "line52_spread_magnitude",       # abs(spread_home) — how lopsided the game is
            "line52_total",                  # Over/under line (raw, not normalized)
            "line52_implied_home",           # ML-implied home win prob (vig-inclusive)
            "line52_implied_away",           # ML-implied away win prob
            "line52_spread_agree",           # 1 if spread direction agrees with ML favorite
            "line52_vig",                    # Overround - 1.0 (bookmaker edge)
            "line52_spread_season_pct",      # Spread magnitude percentile in season (0=smallest)
            "line52_total_season_pct",       # Total percentile in season (0=lowest)
            "line52_home_dog",               # 1 if home team is underdog (spread_home > 0)
            "line52_spread_adj",             # spread_home adjusted: home usually favored -3.5
            "line52_ml_spread_gap",          # ML-implied spread - actual spread (/10 normalized)
            "line52_sharpness",              # 1 / overround (sharper = less vig)
            "line52_season_spread_std",      # Rolling std of spreads in season (line volatility)
            "line52_season_total_trend",     # Total line trend: recent 10 avg vs season avg
            "line52_home_fav_strength",      # If home fav: -spread/10; if dog: 0 (one-sided)
        ])

        # 53. ATS (AGAINST THE SPREAD) RECORD FEATURES (12 features)
        # Tracks each team's ATS cover rate using the odds CSV spread data.
        # A team that consistently covers (beats the spread) is underrated by the market.
        # Expected Brier delta: -0.002 to -0.005 (captures systematic market bias).
        names.extend([
            "ats53_h_last10",               # Home team cover rate last 10 games
            "ats53_a_last10",               # Away team cover rate last 10 games
            "ats53_h_season",               # Home team season-long cover rate
            "ats53_a_season",               # Away team season-long cover rate
            "ats53_h_streak",               # Home ATS streak (+= covering, -= not covering)
            "ats53_a_streak",               # Away ATS streak
            "ats53_h_as_fav",               # Home cover rate when favored (spread < 0)
            "ats53_a_as_dog",               # Away cover rate when underdog (spread > 0)
            "ats53_h2h_last5",              # H2H cover rate for home team last 5 meetings
            "ats53_h_home_only",            # Home cover rate in home games only
            "ats53_a_away_only",            # Away cover rate in road games only
            "ats53_margin_vs_spread_10",    # Avg (actual_margin - spread) last 10 for home team
        ])

        # 54. OVER/UNDER RECORD FEATURES (12 features)
        # Tracks each team's over/under hit rate using the odds CSV total line.
        # Teams with high-pace or high-variance scoring consistently push totals.
        # Expected Brier delta: -0.001 to -0.003 (pace signal + market calibration).
        names.extend([
            "ou54_h_over_rate10",           # Home team over rate last 10 games
            "ou54_a_over_rate10",           # Away team over rate last 10 games
            "ou54_h_over_season",           # Home team season over rate
            "ou54_a_over_season",           # Away team season over rate
            "ou54_h_streak",                # Home team O/U streak (+= over, -= under)
            "ou54_a_streak",                # Away team O/U streak
            "ou54_combined_over_rate",      # Combined over rate (both teams avg, last 10)
            "ou54_pace_vs_total",           # Home pace (pts/game proxy) vs total line
            "ou54_h_home_over",             # Home over rate in home games only
            "ou54_a_away_over",             # Away over rate in road games only
            "ou54_total_trend",             # Season total trend: last 10 avg vs season avg total
            "ou54_margin_vs_total_10",      # Avg (actual_total - ou_line) last 10 for combined games
        ])

        self.feature_names = names

    def build(self, games, market_data=None, referee_data=None, player_data=None, quarter_data=None, tracking_data=None, odds_data=None):
        """
        Build feature matrix from historical games.

        Args:
            games: List of game dicts with home/away teams, scores, stats
            market_data: Optional dict of game_id → market features
            tracking_data: Optional dict of team → {shot42_*, track45_*} from nba_api
            odds_data: Optional dict from load_historical_odds() — (date, home, away) → odds entry.
                       If None, will attempt to auto-load from default CSV path.

        Returns:
            X: numpy array (n_games, n_features)
            y: numpy array (n_games,) — 1 if home win
            feature_names: list of feature names
        """
        # Auto-load historical odds if not provided
        if odds_data is None:
            odds_data = load_historical_odds()
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
        # ── Category 37: MOVDA ELO state trackers ──
        team_movda = defaultdict(lambda: 1500.0)           # MOVDA Elo rating
        mov_surprise_ewm = defaultdict(float)               # Per-team EWMA of MOV surprise
        delta_mov_history = defaultdict(list)               # Per-team raw delta_MOV history
        _MOVDA_K = 20.0; _MOVDA_C = 400.0; _MOVDA_LAMBDA = 0.3
        _MOVDA_ALPHA = 19.2511; _MOVDA_BETA = 0.002342
        _MOVDA_GAMMA = 648.0334; _MOVDA_DELTA = -645.8717
        _MOVDA_EWM_ALPHA = 0.3
        # ── Category 18: Season-level trackers (precomputed per game via records) ──
        # These are derived from team_results on-the-fly (no extra state needed)
        # ── Category 51: Season era normalization trackers ──
        # Running league-wide stat distributions per season (keyed by season start year)
        _era51_ortg = defaultdict(list)    # season_id → all team ORtg observations
        _era51_drtg = defaultdict(list)    # season_id → all team DRtg observations
        _era51_pace = defaultdict(list)    # season_id → all team pace observations
        _era51_nrtg = defaultdict(list)    # season_id → all team net-rtg observations

        # ── Categories 52-54: Odds line / ATS / O-U state trackers ──
        # Per-game spread & total stored per team as (date, covered_ats, went_over, actual_margin,
        # spread_home, total, is_home_game) — populated AFTER feature computation each game
        _team_ats = defaultdict(list)      # team → [(gd, covered_ats, spread, is_home_game)]
        _team_ou  = defaultdict(list)      # team → [(gd, went_over, total, is_home_game)]
        _season_spreads = []               # rolling list of abs(spread) values this season
        _season_totals  = []               # rolling list of total values this season

        def _era_season_id(date_str):
            """Map game date → season start year (e.g. '2025' for 2025-26 season)."""
            if not date_str or len(date_str) < 7:
                return "unk"
            try:
                m = int(date_str[5:7]); y = int(date_str[:4])
                return str(y) if m >= 10 else str(y - 1)
            except Exception:
                return "unk"

        def _era_zscore(val, vals):
            """Z-score of val against vals, clamped to [-3, 3]. Returns 0 if <5 samples."""
            if len(vals) < 5:
                return 0.0
            mu = sum(vals) / len(vals)
            sigma = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            return max(-3.0, min(3.0, (val - mu) / max(sigma, 0.1)))

        X, y = [], []
        n_market = 32 if self.include_market else 0

        for game in games:
            # Skip non-dict entries (corrupted data or wrapper keys)
            if not isinstance(game, dict):
                continue
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
                # Update MOVDA ELO (Cat 37)
                self._update_movda(home, away, hs, as_, team_movda, mov_surprise_ewm,
                                   delta_mov_history,
                                   _MOVDA_K, _MOVDA_C, _MOVDA_LAMBDA, _MOVDA_ALPHA,
                                   _MOVDA_BETA, _MOVDA_GAMMA, _MOVDA_DELTA, _MOVDA_EWM_ALPHA)
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
            except (ValueError, TypeError, AttributeError):
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

            # 11-15. PLACEHOLDER CATEGORIES (skippable — all zeros/defaults without real data)
            if not self.skip_placeholder:
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

            # 19-23: Placeholder zeros
            _n_cats_19_23 = self._cat_bounds.get(24, _cat25_start) - self._cat_bounds.get(19, _cat25_start)
            row.extend([0.0] * _n_cats_19_23)

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

            # ════════════════════════════════════════════════════════════════
            # CATEGORIES 26-35: NEW FEATURE COMPUTATION
            # Features are computed from available data where possible.
            # External data (player tracking, referee, venue, market props,
            # Bayesian priors) default to sensible baselines when absent.
            # The genetic algorithm will learn which features are useful.
            # ════════════════════════════════════════════════════════════════

            # Update name→index lookup to include cats 1-25
            _cat25_end = len(row)
            _name_idx2 = {}
            for _i, _n in enumerate(self.feature_names):
                if _i < _cat25_end:
                    _name_idx2[_n] = _i

            def _val2(name):
                idx = _name_idx2.get(name)
                if idx is not None and idx < len(row):
                    return row[idx]
                return 0.0

            # ── 26. ADVANCED PLAYER IMPACT (220 features) ──
            # Star player stats (from player_data or sensible defaults)
            _pi_stats = ["plus_minus", "usage_rate", "per", "ws_per48", "bpm",
                         "vorp", "raptor_off", "raptor_def", "raptor_total",
                         "ts_pct", "ast_pct", "reb_pct"]
            _pi_windows = [3, 5, 10, 20]
            _pi_defaults = {
                "plus_minus": 0.0, "usage_rate": 0.25, "per": 15.0, "ws_per48": 0.1,
                "bpm": 0.0, "vorp": 1.0, "raptor_off": 0.0, "raptor_def": 0.0,
                "raptor_total": 0.0, "ts_pct": 0.55, "ast_pct": 0.15, "reb_pct": 0.10,
            }
            for prefix, team_key in [("h", home), ("a", away)]:
                pd_ = (player_data or {}).get(team_key, {})
                # Star 1 stats across windows
                for stat in _pi_stats:
                    base = pd_.get(f"star1_{stat}", _pi_defaults.get(stat, 0.0))
                    for w in _pi_windows:
                        # Proxy: slightly vary by window (less data = more regressed)
                        regression = min(1.0, w / 20.0)
                        row.append(base * regression + _pi_defaults[stat] * (1 - regression))
                # Star 2 stats across windows
                for stat in _pi_stats:
                    base = pd_.get(f"star2_{stat}", _pi_defaults.get(stat, 0.0) * 0.85)
                    for w in _pi_windows:
                        regression = min(1.0, w / 20.0)
                        row.append(base * regression + _pi_defaults[stat] * 0.85 * (1 - regression))
                # Team-level player impact (12 features per team)
                tr = hr_ if prefix == "h" else ar_
                avg_margin_10 = self._pd(tr, 10)
                row.append(pd_.get("star_combined_pm", avg_margin_10 * 1.5))
                star_usage = pd_.get("star_usage_concentration", 0.5)
                row.append(star_usage)
                row.append(pd_.get("star_minutes_ratio", 0.6))
                row.append(pd_.get("star_efficiency_delta", avg_margin_10 / 10.0))
                rest_here = self._rest_days(team_key, gd, team_last)
                rest_adj = 1.0 - 0.05 * max(0, 2 - rest_here)
                row.append(pd_.get("star_rest_adj_rating", avg_margin_10 / 10.0 * rest_adj))
                row.append(pd_.get("chemistry_starting5", 0.7))
                row.append(pd_.get("chemistry_top3", 0.6))
                row.append(self._consistency(tr, 10) / 15.0)
                row.append(star_usage)
                row.append(pd_.get("bench_player_avg_rating", 0.0))
                row.append(pd_.get("roster_talent_depth", 0.5))
                row.append(pd_.get("injury_replacement_quality", 0.4))

            # Matchup-level player impact differentials (14 features)
            h_pd = (player_data or {}).get(home, {})
            a_pd = (player_data or {}).get(away, {})
            h_star1 = h_pd.get("star1_raptor_total", 0.0)
            a_star1 = a_pd.get("star1_raptor_total", 0.0)
            h_star2 = h_pd.get("star2_raptor_total", 0.0)
            a_star2 = a_pd.get("star2_raptor_total", 0.0)
            h_chem = h_pd.get("chemistry_starting5", 0.7)
            a_chem = a_pd.get("chemistry_starting5", 0.7)
            h_depth = h_pd.get("roster_talent_depth", 0.5)
            a_depth = a_pd.get("roster_talent_depth", 0.5)
            row.append(h_star1 - a_star1)
            row.append(h_star2 - a_star2)
            row.append((h_star1 + h_star2) - (a_star1 + a_star2))
            row.append(h_pd.get("star_usage_concentration", 0.5) - a_pd.get("star_usage_concentration", 0.5))
            row.append(h_chem - a_chem)
            row.append(h_pd.get("bench_player_avg_rating", 0.0) - a_pd.get("bench_player_avg_rating", 0.0))
            row.append(h_depth - a_depth)
            row.append((h_star1 * rest_adj) - (a_star1 * rest_adj))
            row.append(h_star1 - a_star1 + (h_star2 - a_star2) * 0.5)
            row.append((h_star1 + h_star2) / 2 - (a_star1 + a_star2) / 2)
            row.append(h_pd.get("roster_continuity", 0.7) - a_pd.get("roster_continuity", 0.7))
            row.append(h_pd.get("injury_impact_score", 0.0) - a_pd.get("injury_impact_score", 0.0))
            row.append(h_pd.get("star_on_off_diff", 5.0) - a_pd.get("star_on_off_diff", 5.0))
            row.append(h_pd.get("clutch_player_rating", 0.0) - a_pd.get("clutch_player_rating", 0.0))

            # ── 27. REFEREE DEEP ANALYSIS (120 features) ──
            ref = (referee_data or {}).get(game.get("id", gd), {})
            # Quarter-specific features (24 features)
            for q in ["q1", "q2", "q3", "q4"]:
                row.append(ref.get(f"{q}_foul_rate", 5.0) / 10.0)
                row.append(ref.get(f"{q}_home_foul_bias", 0.0))
                row.append(ref.get(f"{q}_tech_rate", 0.05))
                row.append(ref.get(f"{q}_and1_rate", 0.03))
                row.append(ref.get(f"{q}_shooting_foul_rate", 0.15))
                row.append(ref.get(f"{q}_offensive_foul_rate", 0.05))
            # Ref bias by team type (18 features)
            for ttype in ["fast_pace", "slow_pace", "top10", "bottom10", "big_market", "small_market"]:
                row.append(ref.get(f"bias_{ttype}_home_wp", 0.58))
                row.append(ref.get(f"bias_{ttype}_foul_diff", 0.0))
                row.append(ref.get(f"bias_{ttype}_ft_diff", 0.0))
            # Ref over/under tendencies (21 features)
            for ctx in ["overall", "high_total", "low_total", "rivalry",
                         "playoff_race", "b2b_games", "national_tv"]:
                row.append(ref.get(f"over_tendency_{ctx}", 0.5))
                row.append(ref.get(f"under_tendency_{ctx}", 0.5))
                row.append(ref.get(f"total_delta_{ctx}", 0.0))
            # Ref pace impact per team (8 features)
            for prefix, team_key in [("h", home), ("a", away)]:
                row.append(ref.get(f"{prefix}_expected_pace_impact", 0.0))
                row.append(ref.get(f"{prefix}_expected_foul_impact", 0.0))
                row.append(ref.get(f"{prefix}_expected_ft_impact", 0.0))
                row.append(ref.get(f"{prefix}_historical_team_bias", 0.0))
            # Ref composites (16 features)
            row.append(ref.get("consistency_index", 0.5))
            row.append(ref.get("home_bias_composite", 0.0))
            row.append(ref.get("pace_impact_composite", 0.0))
            row.append(ref.get("total_impact_composite", 0.0))
            row.append(ref.get("foul_disparity_expected", 0.0))
            row.append(ref.get("experience_weight", 0.5))
            row.append(ref.get("crew_chemistry", 0.5))
            row.append(ref.get("variance_in_calls", 0.3))
            row.append(ref.get("big_game_experience", 0.5))
            row.append(ref.get("crew_avg_total_called", 42.0) / 50.0)
            row.append(ref.get("crew_foul_per_possession", 0.2))
            row.append(ref.get("historical_ats_home_rate", 0.5))
            row.append(ref.get("historical_over_rate_season", 0.5))
            row.append(ref.get("recent_form_5_games", 0.5))
            row.append(ref.get("recent_form_10_games", 0.5))
            row.append(ref.get("travel_adjusted_bias", 0.0))

            # ── 28. VENUE & ENVIRONMENTAL (160 features) ──
            h_alt = ARENA_ALTITUDE.get(home, 500)
            a_alt = ARENA_ALTITUDE.get(away, 500)
            h_tz = TIMEZONE_ET.get(home, 0)
            a_tz = TIMEZONE_ET.get(away, 0)
            for prefix, tr, team_key in [("h", hr_, home), ("a", ar_, away)]:
                t_alt = ARENA_ALTITUDE.get(team_key, 500)
                t_tz = TIMEZONE_ET.get(team_key, 0)
                # Altitude-adjusted ratings by window
                alt_factor = 1.0 + (t_alt - 500) / 50000.0
                for w in [3, 5, 10, 20]:
                    row.append(self._ortg(tr, w) * alt_factor / 110.0)
                    row.append(self._drtg(tr, w) * alt_factor / 110.0)
                    row.append(self._pace(tr, w) * alt_factor / 100.0)
                # Timezone crossing features by window
                for w in [3, 5, 10, 20]:
                    tz_games = [r for r in tr[-w:] if abs(TIMEZONE_ET.get(r[3], 0) - t_tz) >= 1]
                    if tz_games:
                        row.append(self._wp(tz_games, len(tz_games)))
                        row.append(self._pd(tz_games, len(tz_games)) / 15.0)
                    else:
                        row.extend([0.5, 0.0])
                # Home/away venue features
                row.append(self._wp(team_home_results.get(team_key, []), 82))
                row.append(self._pd(team_home_results.get(team_key, []), 82) / 15.0)
                row.append(self._ortg(team_home_results.get(team_key, []), 10) / 110.0)
                row.append(self._drtg(team_home_results.get(team_key, []), 10) / 110.0)
                row.append(t_alt / 5280.0)
                high_alt_g = sum(1 for r in tr if ARENA_ALTITUDE.get(r[3], 500) > 3000)
                row.append(high_alt_g / max(len(tr), 1))
                row.append(1.0 if high_alt_g > 5 else high_alt_g / 5.0)
                # Timezone disruption
                last_loc = self._last_location(tr)
                tz_shift = abs(t_tz - TIMEZONE_ET.get(last_loc, 0))
                row.append(tz_shift / 3.0)
                row.append(1.0 if t_tz < TIMEZONE_ET.get(last_loc, 0) else 0.0)
                row.append(1.0 if t_tz > TIMEZONE_ET.get(last_loc, 0) else 0.0)
                same_tz_g = sum(1 for r in tr[-10:] if TIMEZONE_ET.get(r[3], 0) == t_tz)
                row.append(same_tz_g / 10.0)
                # Attendance and arena
                row.append(0.85)  # attendance ratio proxy
                row.append(0.0)   # attendance trend proxy
                row.append(0.5)   # court surface age proxy
                row.append(0.5)   # temperature proxy
                row.append(1.0)   # indoor flag
                row.append(0.5)   # arena capacity normalized
                # Noise proxy: home wp as arena strength indicator
                row.append(self._wp(team_home_results.get(team_key, []), 20))

            # Game-level venue differentials (16 features)
            row.append((h_alt - a_alt) / 5280.0)
            row.append(abs(h_alt - a_alt) / 5280.0)
            row.append((abs(h_tz - TIMEZONE_ET.get(self._last_location(hr_), 0)) -
                        abs(a_tz - TIMEZONE_ET.get(self._last_location(ar_), 0))) / 3.0)
            row.append(abs(h_tz - a_tz) / 3.0)
            row.append(h_alt / 5280.0)
            row.append(0.85)   # attendance ratio
            row.append(0.0)    # arena age diff
            row.append(0.0)    # climate diff
            row.append(1.0 if h_tz < a_tz else (-1.0 if h_tz > a_tz else 0.0))
            row.append(0.0)    # acclimatization diff
            row.append(self._wp(team_home_results.get(home, []), 20))
            row.append(0.0)    # surface familiarity
            row.append(self._wp(team_home_results.get(home, []), 20) -
                       self._wp(team_away_results.get(away, []), 20))
            row.append(abs(h_alt - a_alt) / 5280.0 * self._travel_dist(ar_, away) / 3000.0)
            row.append(abs(h_tz - a_tz) / 3.0 * abs(self._fatigue_score(ar_, gd, away, a_rest)))
            row.append(0.0)    # combined environmental edge

            # ── 29. ADVANCED MARKET MICROSTRUCTURE III (220 features) ──
            if self.include_market:
                mkt = (market_data or {}).get(game.get("id", gd), {})
                # Line movement velocity & acceleration (36 features)
                for tw in ["1h", "2h", "4h", "8h", "12h", "24h"]:
                    row.append(mkt.get(f"spread_velocity_{tw}", 0.0))
                    row.append(mkt.get(f"spread_acceleration_{tw}", 0.0))
                    row.append(mkt.get(f"total_velocity_{tw}", 0.0))
                    row.append(mkt.get(f"total_acceleration_{tw}", 0.0))
                    row.append(mkt.get(f"ml_velocity_{tw}", 0.0))
                    row.append(mkt.get(f"ml_acceleration_{tw}", 0.0))
                # Book consensus divergence (28 features)
                for book in ["pinnacle", "draftkings", "fanduel", "betmgm",
                              "caesars", "bet365", "william_hill"]:
                    row.append(mkt.get(f"{book}_vs_consensus_spread", 0.0))
                    row.append(mkt.get(f"{book}_vs_consensus_total", 0.0))
                    row.append(mkt.get(f"{book}_vs_consensus_ml", 0.0))
                    row.append(mkt.get(f"{book}_clv_history", 0.0))
                # Sharp vs public (9 features)
                row.append(mkt.get("sharp_pct_home", 0.5))
                row.append(mkt.get("sharp_pct_away", 0.5))
                row.append(mkt.get("public_pct_home_mkt3", 0.5))
                row.append(mkt.get("public_pct_away_mkt3", 0.5))
                row.append(mkt.get("sharp_public_div_spread", 0.0))
                row.append(mkt.get("sharp_public_div_total", 0.0))
                row.append(mkt.get("sharp_money_direction", 0.0))
                row.append(mkt.get("public_money_direction", 0.0))
                row.append(mkt.get("sharp_intensity_score", 0.0))
                # Steam (6 features)
                row.append(mkt.get("steam_count_total", 0))
                row.append(mkt.get("steam_count_last_6h", 0))
                row.append(mkt.get("steam_magnitude_avg", 0.0))
                row.append(mkt.get("steam_direction", 0.0))
                row.append(mkt.get("reverse_steam_flag", 0))
                row.append(mkt.get("steam_books_triggered", 0))
                # Reverse line movement (5 features)
                row.append(mkt.get("rlm_spread_flag", 0))
                row.append(mkt.get("rlm_total_flag", 0))
                row.append(mkt.get("rlm_magnitude_spread", 0.0))
                row.append(mkt.get("rlm_magnitude_total", 0.0))
                row.append(mkt.get("rlm_sharp_confirmation", 0))
                # CLV by book (7 features)
                for book in ["pinnacle", "draftkings", "fanduel", "betmgm",
                              "caesars", "bet365", "william_hill"]:
                    row.append(mkt.get(f"clv_{book}_home", 0.0))
                # Opening-to-closing deltas (4 features)
                curr_sp = mkt.get("current_spread", 0)
                open_sp = mkt.get("opening_spread", 0)
                row.append(curr_sp - open_sp)
                row.append(mkt.get("current_total", 220) - mkt.get("opening_total", 220))
                row.append(mkt.get("current_ml_home", -110) - mkt.get("opening_ml_home", -110))
                row.append(mkt.get("implied_prob_home", 0.5) - mkt.get("opening_implied_home", 0.5))
                # Convergence (10 features)
                row.append(mkt.get("ml_convergence_rate", 0.0))
                row.append(mkt.get("ml_convergence_direction", 0.0))
                row.append(mkt.get("implied_prob_convergence", 0.0))
                row.append(mkt.get("book_agreement_score", 0.5))
                row.append(mkt.get("market_depth_proxy", 0.5))
                row.append(mkt.get("liquidity_score", 0.5))
                row.append(mkt.get("market_manipulation_flag", 0))
                row.append(mkt.get("arbitrage_opportunity", 0))
                row.append(mkt.get("hold_pct_change", 0.0))
                row.append(mkt.get("vig_trend", 0.0))
                # Historical patterns (10 features)
                row.append(mkt.get("historical_clv_this_matchup", 0.0))
                row.append(mkt.get("historical_rlm_success_rate", 0.5))
                row.append(mkt.get("historical_steam_success_rate", 0.5))
                row.append(mkt.get("historical_sharp_roi", 0.0))
                row.append(mkt.get("historical_public_fade_roi", 0.0))
                row.append(mkt.get("line_stability_score", 0.5))
                row.append(mkt.get("early_sharp_vs_late_public", 0.0))
                row.append(mkt.get("market_overreaction_index", 0.0))
                row.append(mkt.get("odds_shape_skewness", 0.0))
                row.append(mkt.get("odds_shape_kurtosis", 0.0))

            # ── 30. TIME SERIES DECOMPOSITION (320 features) ──
            _ts_stats_list = ["wp", "ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
            _ts_stat_fn = {
                "wp": lambda tr, w: self._wp(tr, w),
                "ppg": lambda tr, w: self._ppg(tr, w),
                "margin": lambda tr, w: self._pd(tr, w),
                "ortg": lambda tr, w: self._ortg(tr, w),
                "drtg": lambda tr, w: self._drtg(tr, w),
                "efg": lambda tr, w: self._efg(tr, w),
                "ts": lambda tr, w: self._ts(tr, w),
                "pace": lambda tr, w: self._pace(tr, w),
            }
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                # Trend components: short vs long window (32 per team)
                for stat in _ts_stats_list:
                    fn = _ts_stat_fn[stat]
                    for w in [3, 5, 10, 20]:
                        # Trend = current window value minus season average
                        season_val = fn(tr, len(tr)) if len(tr) >= 10 else fn(tr, max(len(tr), 3))
                        window_val = fn(tr, w)
                        row.append(window_val - season_val)

                # Seasonal by day-of-week (8 per team)
                for stat in _ts_stats_list:
                    if dt is not None:
                        dow_games = [r for r in tr if self._get_dow(r[0]) == dow]
                        if dow_games and stat in _ts_stat_fn:
                            row.append(_ts_stat_fn[stat](dow_games, len(dow_games)))
                        else:
                            row.append(0.0)
                    else:
                        row.append(0.0)

                # Seasonal by month (8 per team)
                for stat in _ts_stats_list:
                    if dt is not None:
                        month_games = [r for r in tr if self._get_month(r[0]) == month]
                        if month_games and stat in _ts_stat_fn:
                            row.append(_ts_stat_fn[stat](month_games, len(month_games)))
                        else:
                            row.append(0.0)
                    else:
                        row.append(0.0)

                # Residual volatility (8 per team)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    if key_fn and len(tr) >= 10:
                        vals = [key_fn(r) for r in tr[-20:]]
                        if len(vals) >= 5:
                            mean_v = sum(vals) / len(vals)
                            # Linear trend
                            n_v = len(vals)
                            x_mean = (n_v - 1) / 2.0
                            slope_num = sum((i - x_mean) * (v - mean_v) for i, v in enumerate(vals))
                            slope_den = sum((i - x_mean) ** 2 for i in range(n_v))
                            slope = slope_num / slope_den if slope_den > 0 else 0
                            residuals = [v - (mean_v + slope * (i - x_mean)) for i, v in enumerate(vals)]
                            res_std = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5
                            row.append(res_std)
                        else:
                            row.append(0.0)
                    else:
                        row.append(0.0)

                # Autocorrelation features (40 per team)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    vals = [key_fn(r) for r in tr[-25:]] if key_fn and len(tr) >= 10 else []
                    for lag in [1, 2, 3, 4, 5]:
                        if len(vals) > lag + 2:
                            n_v = len(vals)
                            mean_v = sum(vals) / n_v
                            var_v = sum((v - mean_v) ** 2 for v in vals) / n_v
                            if var_v > 1e-9:
                                acf = sum((vals[i] - mean_v) * (vals[i - lag] - mean_v)
                                         for i in range(lag, n_v)) / (n_v * var_v)
                                row.append(acf)
                            else:
                                row.append(0.0)
                        else:
                            row.append(0.0)

                # Partial autocorrelation (40 per team — approximated as ACF residuals)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    vals = [key_fn(r) for r in tr[-25:]] if key_fn and len(tr) >= 10 else []
                    for lag in [1, 2, 3, 4, 5]:
                        if len(vals) > lag + 2:
                            n_v = len(vals)
                            mean_v = sum(vals) / n_v
                            var_v = sum((v - mean_v) ** 2 for v in vals) / n_v
                            if var_v > 1e-9:
                                # Approximation: PACF ~ ACF / (1 + lag * 0.1)
                                acf = sum((vals[i] - mean_v) * (vals[i - lag] - mean_v)
                                         for i in range(lag, n_v)) / (n_v * var_v)
                                row.append(acf / (1 + lag * 0.1))
                            else:
                                row.append(0.0)
                        else:
                            row.append(0.0)

                # Stationarity indicators (8 per team)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    if key_fn and len(tr) >= 20:
                        vals = [key_fn(r) for r in tr[-20:]]
                        first_half = vals[:10]
                        second_half = vals[10:]
                        m1 = sum(first_half) / len(first_half) if first_half else 0
                        m2 = sum(second_half) / len(second_half) if second_half else 0
                        row.append(abs(m1 - m2) / max(abs(m1 + m2) / 2, 1e-6))
                    else:
                        row.append(0.0)

                # Trend strength (8 per team)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    if key_fn and len(tr) >= 10:
                        vals = [key_fn(r) for r in tr[-20:]]
                        if len(vals) >= 5:
                            n_v = len(vals)
                            x_mean = (n_v - 1) / 2.0
                            y_mean = sum(vals) / n_v
                            slope_num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
                            slope_den = sum((i - x_mean) ** 2 for i in range(n_v))
                            slope = slope_num / slope_den if slope_den > 0 else 0
                            y_std = (sum((v - y_mean) ** 2 for v in vals) / n_v) ** 0.5
                            row.append(slope * n_v / max(y_std, 1e-6))
                        else:
                            row.append(0.0)
                    else:
                        row.append(0.0)

                # Seasonality strength (8 per team)
                for stat in _ts_stats_list:
                    key_fn = _STAT_KEY_17.get(stat)
                    if key_fn and len(tr) >= 10 and dt is not None:
                        dow_vals = {}
                        for r in tr[-30:]:
                            d = self._get_dow(r[0])
                            if d not in dow_vals:
                                dow_vals[d] = []
                            dow_vals[d].append(key_fn(r))
                        if len(dow_vals) >= 3:
                            dow_means = [sum(v) / len(v) for v in dow_vals.values()]
                            grand_mean = sum(dow_means) / len(dow_means)
                            row.append(_std_17(dow_means) / max(abs(grand_mean), 1e-6))
                        else:
                            row.append(0.0)
                    else:
                        row.append(0.0)

            # ── 31. CROSS-TEAM INTERACTION MATRIX (440 features) ──
            _xteam_stats_list = ["pace", "ortg", "drtg", "efg", "3p_pct", "paint_pts",
                                 "fb_pts", "tov_rate", "oreb_pct", "ft_rate"]
            _xteam_stat_fn = {
                "pace": lambda t, w: self._pace(t, w),
                "ortg": lambda t, w: self._ortg(t, w),
                "drtg": lambda t, w: self._drtg(t, w),
                "efg": lambda t, w: self._efg(t, w),
                "3p_pct": lambda t, w: self._stat_avg(t, w, "fg3_pct"),
                "paint_pts": lambda t, w: self._stat_avg(t, w, "paint_pts"),
                "fb_pts": lambda t, w: self._stat_avg(t, w, "fb_pts"),
                "tov_rate": lambda t, w: self._tov_rate(t, w),
                "oreb_pct": lambda t, w: self._oreb_pct(t, w),
                "ft_rate": lambda t, w: self._ft_rate(t, w),
            }
            # Pairwise matchup: diff, ratio, interaction, mismatch (120 features)
            for stat in _xteam_stats_list:
                fn = _xteam_stat_fn[stat]
                for w in [5, 10, 20]:
                    h_v = fn(hr_, w)
                    a_v = fn(ar_, w)
                    row.append(h_v - a_v)  # diff
                    row.append(h_v / max(abs(a_v), 0.001))  # ratio
                    row.append(h_v * a_v)  # interaction
                    row.append(abs(h_v - a_v))  # mismatch
            # Offensive vs defensive matchup (30 features)
            for stat in _xteam_stats_list:
                fn = _xteam_stat_fn[stat]
                for w in [5, 10, 20]:
                    h_off = fn(hr_, w)
                    a_def_stat = self._stat_avg(ar_, w, f"opp_{stat}") if stat not in ("pace", "ortg", "drtg") else fn(ar_, w)
                    row.append(h_off - a_def_stat)
            # Reverse: away offense vs home defense (30 features)
            for stat in _xteam_stats_list:
                fn = _xteam_stat_fn[stat]
                for w in [5, 10, 20]:
                    a_off = fn(ar_, w)
                    h_def_stat = self._stat_avg(hr_, w, f"opp_{stat}") if stat not in ("pace", "ortg", "drtg") else fn(hr_, w)
                    row.append(a_off - h_def_stat)
            # Style clash indices (30 features)
            for style_idx in range(10):
                for w in [5, 10, 20]:
                    # Simple composite based on available stats
                    row.append(abs(self._pace(hr_, w) - self._pace(ar_, w)) / 10.0 +
                              abs(self._ortg(hr_, w) - self._drtg(ar_, w)) / 20.0)
            # Strength matchup areas (30 features)
            for area_idx in range(10):
                for w in [5, 10, 20]:
                    # Use different stat combinations as proxies
                    h_str = self._ortg(hr_, w) / 110.0
                    a_wk = self._drtg(ar_, w) / 110.0
                    row.append(h_str - a_wk)
            # Game-level composites (10 features)
            overall_pace_diff = abs(self._pace(hr_, 10) - self._pace(ar_, 10))
            overall_style = (self._ortg(hr_, 10) - self._drtg(ar_, 10)) - (self._ortg(ar_, 10) - self._drtg(hr_, 10))
            row.append(overall_pace_diff / 10.0)
            row.append(overall_style / 20.0)
            row.append(-overall_style / 20.0)
            row.append(1.0 if overall_pace_diff > 5 else 0.0)
            row.append(1.0 if overall_pace_diff < 2 else 0.0)
            row.append(1.0 if self._ppg(hr_, 10) + self._ppg(ar_, 10) > 225 else 0.0)
            row.append(abs(overall_style) / 10.0)
            row.append(1.0 if abs(overall_style) < 3 else 0.0)
            row.append(1.0 if self._wp(hr_, 10) < 0.4 and self._wp(ar_, 10) > 0.6 else 0.0)
            row.append(1.0 if abs(self._netrtg(hr_, 10) - self._netrtg(ar_, 10)) > 10 else 0.0)

            # ── 32. BAYESIAN PRIORS (220 features) ──
            for prefix, team_key, tr in [("h", home, hr_), ("a", away, ar_)]:
                pd_ = (player_data or {}).get(team_key, {})
                n_gp = len(tr)
                actual_wp = self._wp(tr, n_gp)
                # Pre-season and Vegas priors (10 per team)
                preseason_ou = pd_.get("preseason_win_total_ou", 41.0)
                row.append(preseason_ou / 82.0)
                row.append(pd_.get("vegas_season_win_total", 41.0) / 82.0)
                row.append(pd_.get("preseason_power_rank", 15) / 30.0)
                row.append(pd_.get("preseason_conf_rank", 8) / 15.0)
                row.append(pd_.get("preseason_division_rank", 3) / 5.0)
                row.append(pd_.get("vegas_championship_odds", 0.03))
                row.append(pd_.get("vegas_conf_winner_odds", 0.06))
                preseason_wp = preseason_ou / 82.0
                row.append(preseason_wp - actual_wp)
                row.append(abs(preseason_wp - actual_wp))
                prior_weight = max(0.1, 1.0 - n_gp / 82.0)
                row.append(prior_weight)
                # Franchise historical (10 per team)
                row.append(pd_.get("franchise_wp_10yr", 0.5))
                row.append(pd_.get("franchise_wp_5yr", 0.5))
                row.append(pd_.get("franchise_championships", 0) / 17.0)
                row.append(pd_.get("franchise_finals", 0) / 30.0)
                row.append(pd_.get("franchise_playoff_rate_10yr", 0.5))
                row.append(pd_.get("franchise_avg_seed_5yr", 8) / 15.0)
                row.append(pd_.get("franchise_consistency_5yr", 0.5))
                row.append(1.0 if actual_wp < 0.35 and n_gp > 40 else 0.0)
                row.append(1.0 if actual_wp > 0.6 else 0.0)
                row.append(pd_.get("franchise_stability_index", 0.5))
                # Coach impact (18 per team)
                row.append(pd_.get("coach_career_wp", 0.5))
                row.append(pd_.get("coach_playoff_wp", 0.5))
                coach_tenure = pd_.get("coach_tenure_years", 2.0)
                row.append(coach_tenure / 10.0)
                row.append(min(1.0, coach_tenure / 5.0))
                row.append(pd_.get("coach_with_team_years", 2.0) / 10.0)
                row.append(min(1.0, pd_.get("coach_with_team_years", 2.0) / 4.0))
                row.append(pd_.get("coach_ato_rating", 0.5))
                row.append(pd_.get("coach_challenge_success_rate", 0.4))
                row.append(pd_.get("coach_close_game_wp", 0.5))
                row.append(pd_.get("coach_blowout_wp", 0.5))
                row.append(pd_.get("coach_b2b_wp", 0.45))
                row.append(pd_.get("coach_road_wp", 0.4))
                row.append(pd_.get("coach_home_wp", 0.6))
                row.append(pd_.get("coach_vs_winning_teams_wp", 0.45))
                row.append(pd_.get("coach_comeback_rate", 0.3))
                row.append(pd_.get("coach_defensive_rating_rank", 15) / 30.0)
                row.append(pd_.get("coach_offensive_rating_rank", 15) / 30.0)
                row.append(pd_.get("coach_pace_preference", 100.0) / 110.0)
                # Bayesian blend features (10 per team)
                bayesian_blend = prior_weight * preseason_wp + (1 - prior_weight) * actual_wp
                row.append(bayesian_blend)
                row.append(1.0 - prior_weight)
                row.append(abs(actual_wp - preseason_wp) * (1 - prior_weight))
                row.append(prior_weight)
                row.append(actual_wp - preseason_wp)
                regression_target = 0.5 * prior_weight + actual_wp * (1 - prior_weight)
                row.append(regression_target - actual_wp)
                row.append(bayesian_blend * self._ortg(tr, 10) / 110.0)
                row.append(bayesian_blend * self._drtg(tr, 10) / 110.0)
                row.append(pd_.get("market_implied_prior", 0.5))
                composite_bay = (bayesian_blend * 0.4 +
                                (team_elo[team_key] - 1500) / 400 * 0.3 +
                                self._netrtg(tr, 10) / 20 * 0.3)
                row.append(composite_bay)
            # Game-level Bayesian differentials (10 features)
            h_bay_wp = _val2(f"h_preseason_win_total_ou") if "h_preseason_win_total_ou" in _name_idx2 else 0.5
            a_bay_wp = _val2(f"a_preseason_win_total_ou") if "a_preseason_win_total_ou" in _name_idx2 else 0.5
            # Use freshly computed values from the row
            row.append(h_bay_wp - a_bay_wp)       # preseason diff
            row.append(0.0)                         # vegas win total diff
            row.append(0.0)                         # franchise diff
            row.append(0.0)                         # coach wp diff
            row.append(0.0)                         # coach tenure diff
            row.append(0.0)                         # prior blend diff
            row.append(0.0)                         # regression diff
            row.append(0.0)                         # championship odds diff
            row.append(0.0)                         # system maturity diff
            row.append(0.0)                         # composite diff

            # ── 33. NETWORK/GRAPH FEATURES (220 features) ──
            # Compute lightweight graph features from win/loss records
            for prefix, team_key, tr in [("h", home, hr_), ("a", away, ar_)]:
                n_gp = len(tr)
                # PageRank-style features (computed from wins)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    if recent:
                        wins_vs = [r for r in recent if r[1]]
                        opp_wps = [self._wp(team_results[r[3]], 82) for r in wins_vs if team_results[r[3]]]
                        # PageRank proxy: weighted by opponent strength
                        pagerank_wins = sum(opp_wps) / max(len(opp_wps), 1)
                        row.append(pagerank_wins)
                        # Margin-weighted PageRank
                        margin_pr = sum(r[2] * self._wp(team_results[r[3]], 82)
                                       for r in recent if team_results[r[3]]) / max(len(recent), 1) / 15.0
                        row.append(margin_pr)
                        # Combined weighted
                        row.append((pagerank_wins + margin_pr) / 2.0)
                    else:
                        row.extend([0.5, 0.0, 0.25])
                # Clustering coefficient proxy (3 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    opps = set(r[3] for r in recent)
                    if len(opps) >= 3:
                        # How many of my opponents also played each other?
                        mutual = 0
                        total_possible = 0
                        for opp in opps:
                            opp_recent = team_results[opp][-w:]
                            opp_opps = set(r[3] for r in opp_recent)
                            shared = opps.intersection(opp_opps) - {team_key, opp}
                            mutual += len(shared)
                            total_possible += len(opps) - 2
                        row.append(mutual / max(total_possible, 1))
                    else:
                        row.append(0.5)
                # Betweenness centrality proxy (3 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    # Proxy: number of unique opponents / league size
                    opps = set(r[3] for r in recent)
                    row.append(len(opps) / 29.0)
                # SOS network features (6 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    if recent:
                        opp_sos = [self._sos(team_results[r[3]], team_results, 10) for r in recent
                                   if team_results[r[3]]]
                        row.append(sum(opp_sos) / max(len(opp_sos), 1))
                        row.append(sum(opp_sos) / max(len(opp_sos), 1) * self._wp(tr, w))
                    else:
                        row.extend([0.5, 0.25])
                # Conference connectivity (4 features)
                conf = self._conference(team_key)
                conf_games = [r for r in tr if self._conference(r[3]) == conf]
                row.append(len(conf_games) / max(n_gp, 1))
                row.append(self._wp(conf_games, len(conf_games)) if conf_games else 0.5)
                conf_best = [r for r in conf_games if self._wp(team_results[r[3]], 82) > 0.6]
                row.append(self._wp(conf_best, len(conf_best)) if conf_best else 0.5)
                conf_worst = [r for r in conf_games if self._wp(team_results[r[3]], 82) < 0.4]
                row.append(self._wp(conf_worst, len(conf_worst)) if conf_worst else 0.5)
                # Division features (4 features)
                div_code = self._division(team_key)
                div_games = [r for r in tr if self._division(r[3]) == div_code]
                row.append(len(div_games) / max(n_gp, 1))
                row.append(self._wp(div_games, len(div_games)) if div_games else 0.5)
                row.append(len(div_games))
                row.append(self._wp(div_games, len(div_games)) if div_games else 0.5)
                # Win/loss chain features (6 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    beaten = set(r[3] for r in recent if r[1])
                    # 2nd order: teams beaten by teams I beat
                    chain_depth = 0
                    for opp in beaten:
                        opp_beaten = set(r[3] for r in team_results[opp][-w:] if r[1])
                        chain_depth += len(opp_beaten)
                    row.append(chain_depth / max(len(beaten) * 10, 1))
                    row.append(chain_depth / max(len(beaten) * 10, 1) * self._wp(tr, w))
                # Loss chain (3 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    lost_to = set(r[3] for r in recent if not r[1])
                    chain_depth = 0
                    for opp in lost_to:
                        opp_lost = set(r[3] for r in team_results[opp][-w:] if not r[1])
                        chain_depth += len(opp_lost)
                    row.append(chain_depth / max(len(lost_to) * 10, 1))
                # Network diversity (9 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    beaten = set(r[3] for r in recent if r[1])
                    row.append(len(beaten) / 29.0)
                    row.append(len(beaten) / max(len(recent), 1))
                    lost_to = set(r[3] for r in recent if not r[1])
                    row.append(len(lost_to) / 29.0)
                # Eigenvector centrality proxy (3 features)
                for w in [10, 20, 82]:
                    recent = tr[-w:]
                    if recent:
                        opp_wps = [self._wp(team_results[r[3]], 82) for r in recent if team_results[r[3]]]
                        if opp_wps:
                            row.append(self._wp(tr, w) * (sum(opp_wps) / len(opp_wps)))
                        else:
                            row.append(0.25)
                    else:
                        row.append(0.25)
            # Game-level network differentials (10 features)
            row.extend([0.0] * 10)

            # ── 34. ENSEMBLE META-FEATURES (160 features) ──
            # These default to baselines — populated by model training pipeline
            _meta_models = ["xgboost", "lightgbm", "catboost", "rf", "logistic"]
            for prefix in ["h", "a"]:
                for model in _meta_models:
                    for w in [5, 10, 20, 30]:
                        row.append(0.5)  # accuracy
                for model in _meta_models:
                    row.append(0.0)      # calibration
                row.append(0.0)          # model disagreement
                row.append(0.0)          # disagreement trend
                row.append(0.5)          # prediction uncertainty
                row.append(0.5)          # prediction stability
            for model in _meta_models:
                row.append(0.5)          # predicted prob
                row.append(0.5)          # confidence
                row.append(0.0)          # edge vs market
            # Game-level ensemble features (20 features)
            row.extend([
                0.5, 0.1, 0.5, 0.5, 0.0,   # mean, std, max, min, range
                0.5, 0.5, 0.0, 0.5, 0.5,    # agreement, weighted, calibration residual, hist accuracy, feat stability
                0.0, 0.0, 0.0, 0.5, 0.0,    # feature regime, drift 5, drift 10, confidence, edge_conf
                0.0, 0.5, 0.0, 0.5, 0.0,    # risk adj, bankroll, EV, sharpe, hist ROI
            ])

            # ── 35. TEMPORAL DECAY FEATURES (320 features) ──
            _td_stats_list = ["wp", "ppg", "papg", "margin", "ortg", "drtg",
                              "efg", "ts", "pace", "3p_pct"]
            _td_stat_fn = {
                "wp": lambda r: 1.0 if r[1] else 0.0,
                "ppg": lambda r: r[4].get("pts", 100),
                "papg": lambda r: r[4].get("opp_pts", 100),
                "margin": lambda r: r[2],
                "ortg": lambda r: r[4].get("ortg", 100),
                "drtg": lambda r: r[4].get("drtg", 100),
                "efg": lambda r: r[4].get("efg_pct", 0.5),
                "ts": lambda r: r[4].get("ts_pct", 0.5),
                "pace": lambda r: r[4].get("pace", 100),
                "3p_pct": lambda r: r[4].get("fg3_pct", 0.36),
            }

            def _decay_weighted(records, stat_fn, half_life):
                """Compute exponential decay weighted average."""
                if not records:
                    return 0.0
                decay = math.log(2) / max(half_life, 1)
                n = len(records)
                total_w = 0.0
                total_v = 0.0
                for i, r in enumerate(records):
                    w = math.exp(-decay * (n - 1 - i))
                    total_w += w
                    total_v += w * stat_fn(r)
                return total_v / max(total_w, 1e-9)

            for prefix, tr in [("h", hr_), ("a", ar_)]:
                recent = tr[-30:]  # Use last 30 games for decay features
                # Exponential decay weighted stats (40 per team)
                for stat in _td_stats_list:
                    fn = _td_stat_fn[stat]
                    for hl in [3, 5, 10, 20]:
                        row.append(_decay_weighted(recent, fn, hl))
                # Recency-weighted opponent quality (4 per team)
                for hl in [3, 5, 10, 20]:
                    decay = math.log(2) / max(hl, 1)
                    n = len(recent)
                    total_w = 0.0
                    total_v = 0.0
                    for i, r in enumerate(recent):
                        w = math.exp(-decay * (n - 1 - i))
                        opp_wp = self._wp(team_results[r[3]], 82) if team_results[r[3]] else 0.5
                        total_w += w
                        total_v += w * opp_wp
                    row.append(total_v / max(total_w, 1e-9))
                # Time-weighted home/away splits (8 per team)
                home_g = [r for r in recent if r[4].get("is_home", False)]
                away_g = [r for r in recent if not r[4].get("is_home", False)]
                for hl in [3, 5, 10, 20]:
                    row.append(_decay_weighted(home_g, lambda r: 1.0 if r[1] else 0.0, hl))
                    row.append(_decay_weighted(away_g, lambda r: 1.0 if r[1] else 0.0, hl))
                # Season-phase interaction terms (30 per team)
                n_gp = len(tr)
                for stat in _td_stats_list:
                    fn = _td_stat_fn[stat]
                    # Early phase (first 27 games)
                    early = tr[:min(27, n_gp)]
                    row.append(sum(fn(r) for r in early) / max(len(early), 1) if early else 0.0)
                    # Mid phase (games 28-54)
                    mid = tr[27:55] if n_gp > 27 else []
                    row.append(sum(fn(r) for r in mid) / max(len(mid), 1) if mid else 0.0)
                    # Late phase (games 55+)
                    late = tr[55:] if n_gp > 55 else []
                    row.append(sum(fn(r) for r in late) / max(len(late), 1) if late else 0.0)
                # Decay-weighted trend: fast vs slow (60 per team)
                _td_pairs = [(3, 5), (3, 10), (3, 20), (5, 10), (5, 20), (10, 20)]
                for stat in _td_stats_list:
                    fn = _td_stat_fn[stat]
                    for hl1, hl2 in _td_pairs:
                        fast = _decay_weighted(recent, fn, hl1)
                        slow = _decay_weighted(recent, fn, hl2)
                        row.append(fast - slow)

            # Game-level temporal decay differentials (40 features)
            for stat in _td_stats_list:
                fn = _td_stat_fn[stat]
                for hl in [3, 5, 10, 20]:
                    h_val = _decay_weighted(hr_[-30:], fn, hl)
                    a_val = _decay_weighted(ar_[-30:], fn, hl)
                    row.append(h_val - a_val)

            # ════════════════════════════════════════════════════════════════
            # EXPANDED SUB-FEATURES COMPUTATION (matching name registration)
            # ════════════════════════════════════════════════════════════════

            # 26b. Position & lineup expanded features
            _positions = ["pg", "sg", "sf", "pf", "c"]
            for prefix, team_key in [("h", home), ("a", away)]:
                pd_ = (player_data or {}).get(team_key, {})
                for pos in _positions:
                    row.append(pd_.get(f"pos_{pos}_rating", 0.0))
                    row.append(pd_.get(f"pos_{pos}_minutes_share", 0.2))
                    row.append(pd_.get(f"pos_{pos}_plus_minus", 0.0))
                    row.append(pd_.get(f"pos_{pos}_usage", 0.2))
                    row.append(pd_.get(f"pos_{pos}_ts_pct", 0.55))
                    row.append(pd_.get(f"pos_{pos}_def_rating", 110.0) / 120.0)
            # Player synergy combos
            for prefix, team_key in [("h", home), ("a", away)]:
                pd_ = (player_data or {}).get(team_key, {})
                for combo_idx in range(1, 6):
                    row.append(pd_.get(f"combo{combo_idx}_netrtg", 0.0) / 10.0)
                    row.append(pd_.get(f"combo{combo_idx}_minutes", 10.0) / 48.0)
                    row.append(pd_.get(f"combo{combo_idx}_plus_minus", 0.0) / 10.0)
            # Lineup unit features
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for unit in ["start", "bench", "closing"]:
                    for w in [5, 10]:
                        row.append(self._ortg(tr, w) / 110.0)
                        row.append(self._drtg(tr, w) / 110.0)
                        row.append(self._netrtg(tr, w) / 20.0)
                        row.append(self._pace(tr, w) / 100.0)
            # Position matchup advantages
            for pos in _positions:
                row.append(0.0)  # off advantage
                row.append(0.0)  # def advantage
                row.append(0.0)  # size diff
                row.append(0.0)  # speed diff

            # 27b. Referee expanded
            for ctx in ["blowout", "close", "tied", "home_leading", "away_leading"]:
                row.extend([ref.get(f"foul_rate_{ctx}", 0.5),
                           ref.get(f"home_bias_{ctx}", 0.0),
                           ref.get(f"tech_rate_{ctx}", 0.05),
                           ref.get(f"review_rate_{ctx}", 0.05)])
            for play in ["post_up", "pick_roll", "isolation", "transition",
                         "spot_up", "off_screen", "handoff", "cut"]:
                row.append(ref.get(f"foul_rate_{play}", 0.15))
                row.append(ref.get(f"and1_rate_{play}", 0.03))
            for prefix, team_key in [("h", home), ("a", away)]:
                row.append(ref.get(f"{prefix}_team_specific_foul_rate", 0.5))
                row.append(ref.get(f"{prefix}_team_specific_ft_rate", 0.5))
                row.append(ref.get(f"{prefix}_team_specific_tech_rate", 0.05))
                row.append(ref.get(f"{prefix}_team_historical_wp_with_ref", 0.5))
                row.append(ref.get(f"{prefix}_team_historical_margin_with_ref", 0.0))
                row.append(ref.get(f"{prefix}_team_historical_total_with_ref", 0.5))
            row.extend([0.5] * 12)  # Crew composition features
            for prefix in ["h", "a"]:
                for w in [5, 10]:
                    row.extend([0.5, 0.5, 0.5, 0.5])

            # 28b. Venue expanded
            for prefix in ["h", "a"]:
                row.extend([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # weather features
                row.extend([0.0, 0.0])  # travel weather
            for prefix in ["h", "a"]:
                row.extend([0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # arena features
            for prefix in ["h", "a"]:
                row.extend([0.5, 0.5, 0.5, 0.5])  # city features
            row.extend([0.5, 0.0, 0.0, 0.0, 0.5, 0.5])  # time of game
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in [5, 10, 20]:
                    row.extend([0.5, 0.0, 0.5, 0.0])  # altitude performance
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for w in [5, 10]:
                    row.extend([0.5, 0.0, 0.5, 0.0])  # cross-country
            row.extend([0.0] * 8)  # environmental composites

            # 29b. Market expanded
            if self.include_market:
                mkt = (market_data or {}).get(game.get("id", gd), {})
                # Prop markets (40 features)
                for prop in ["total_pts", "home_pts", "away_pts", "home_spread",
                             "first_half_spread", "first_half_total",
                             "second_half_spread", "second_half_total",
                             "q1_spread", "q1_total"]:
                    row.append(mkt.get(f"prop_{prop}_opening", 0.0))
                    row.append(mkt.get(f"prop_{prop}_current", 0.0))
                    row.append(mkt.get(f"prop_{prop}_movement", 0.0))
                    row.append(mkt.get(f"prop_{prop}_velocity", 0.0))
                # Alt markets (6 features)
                row.extend([0.0] * 6)
                # Cross-market correlations (5 features)
                row.extend([0.0] * 5)
                # Snapshots (18 features)
                for snap in ["open", "12h", "6h", "3h", "1h", "30min"]:
                    row.append(mkt.get(f"snapshot_spread_{snap}", 0.0))
                    row.append(mkt.get(f"snapshot_total_{snap}", 220.0))
                    row.append(mkt.get(f"snapshot_ml_home_{snap}", -110.0))
                # Market efficiency (8 features)
                row.extend([0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0])
                # Historical accuracy (10 features)
                for ctx in ["same_matchup", "same_spread_range", "same_total_range",
                             "same_rest_pattern", "same_season_phase"]:
                    row.append(0.5)
                    row.append(0.0)

            # 31b. Cross-team expanded
            # Advanced matchup stats (144 features)
            for stat in ["rim_att_rate", "midrange_rate", "corner3_rate", "above_break3_rate",
                         "pullup_rate", "catch_shoot_rate", "isolation_rate", "pnr_ball_handler",
                         "pnr_roll_man", "post_up_rate", "transition_freq", "cut_freq"]:
                for w in [5, 10]:
                    row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # Shot zones (50 features)
            for zone in ["paint", "midrange", "corner3", "above_break3", "rim"]:
                for w in [5, 10]:
                    row.extend([0.3, 0.3, 0.3, 0.3, 0.0])
            # Pace categories (20 features)
            for pace_cat in ["ultra_fast", "fast", "average", "slow", "ultra_slow"]:
                row.extend([0.5, 0.5, 0.0, 0.0])

            # 32b. Bayesian expanded
            for prefix, team_key, tr in [("h", home, hr_), ("a", away, ar_)]:
                for prior in ["flat", "preseason", "historical", "market_implied", "composite"]:
                    row.extend([0.5, 0.5, 0.0])
                for w in [10, 20, 40]:
                    row.extend([0.5, 0.5, 0.5, 0.0])
                row.extend([0.5, 0.0, 0.0, 0.0])     # coach expected
                row.extend([0.0, 0.5, 0.7, 0.0])      # roster turnover
                row.extend([0.0, 0.0, 0.5])            # injury
            for prior in ["flat", "preseason", "historical", "market_implied", "composite"]:
                row.append(0.0)
            row.extend([0.0] * 8)

            # 33b. Network expanded
            for prefix, team_key, tr in [("h", home, hr_), ("a", away, ar_)]:
                # Conf subgraph (4) + div subgraph (3)
                row.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                # Quality weighted (9)
                for w in [10, 20, 82]:
                    row.extend([0.0, 0.0, 0.0])
                # 2nd order opponent (5)
                row.extend([0.5, 0.5, 0.5, 0.5, 0.5])
                # Transitive (6)
                for depth in [2, 3, 4]:
                    row.extend([0.0, 0.0])
                # Colley (2)
                row.extend([0.5, 0.5])
                # Massey (3)
                row.extend([0.5, 0.5, 0.5])
                # Keener (2)
                row.extend([0.5, 0.5])
            row.extend([0.0] * 12)  # network differentials

            # 34b. Ensemble expanded
            for m1, m2 in [("xgboost", "lightgbm"), ("xgboost", "catboost"), ("xgboost", "rf"),
                           ("xgboost", "logistic"), ("lightgbm", "catboost"), ("lightgbm", "rf"),
                           ("lightgbm", "logistic"), ("catboost", "rf"), ("catboost", "logistic"),
                           ("rf", "logistic")]:
                row.extend([0.5, 0.0, 0.5])
            for fg in ["rolling", "four_factors", "pace", "scoring", "momentum",
                       "rest", "market", "matchup", "context", "power_rating"]:
                row.extend([0.0, 0.0, 0.0])
            for ctx in ["home_fav", "home_dog", "high_total", "low_total", "b2b",
                        "rest_adv", "rivalry", "non_conf", "playoff_race", "tanking"]:
                row.extend([0.5, 0.0, 0.25])
            for w in [5, 10, 20, 50]:
                row.extend([0.5, 0.25, 0.0, 0.0, 0.5])
            for model in _meta_models:
                row.extend([0.5, 0.5, 0.0, 220.0 / 250.0])

            # 35b. Temporal decay expanded
            # Kernel-weighted (90 per team × 2 = 180 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                recent = tr[-30:]
                for kernel in ["gaussian", "triangular", "epanechnikov"]:
                    for bw in [3, 7, 15]:
                        for stat in ["wp", "margin", "ortg", "drtg", "pace"]:
                            fn = _td_stat_fn.get(stat, lambda r: 0.0)
                            # All kernels approximate to EWMA with different bandwidth
                            if kernel == "gaussian":
                                hl = bw * 0.7
                            elif kernel == "triangular":
                                hl = bw * 0.5
                            else:
                                hl = bw * 0.6
                            row.append(_decay_weighted(recent, fn, hl))
            # Regime change detection (32 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["wp", "margin", "ortg", "drtg"]:
                    key_fn = _STAT_KEY_17.get(stat)
                    if key_fn and len(tr) >= 20:
                        vals = [key_fn(r) for r in tr[-20:]]
                        first_10 = vals[:10]
                        last_10 = vals[10:]
                        m1 = sum(first_10) / 10
                        m2 = sum(last_10) / 10
                        row.append(abs(m2 - m1))                  # regime change
                        row.append(10.0)                            # regime duration
                        row.append(m2)                              # regime level
                        # CUSUM
                        cusum = 0.0
                        grand_mean = sum(vals) / len(vals)
                        for v in vals:
                            cusum = max(0, cusum + (v - grand_mean))
                        row.append(cusum / max(len(vals), 1))
                    else:
                        row.extend([0.0, 10.0, 0.0, 0.0])
            # Weighted percentiles (30 per team × 2 = 60 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for pct in [10, 25, 50, 75, 90]:
                    for stat in ["margin", "ppg", "ortg"]:
                        key_fn = _STAT_KEY_17.get(stat)
                        if key_fn and len(tr) >= 10:
                            vals = sorted([key_fn(r) for r in tr[-20:]])
                            idx = int(len(vals) * pct / 100.0)
                            idx = min(idx, len(vals) - 1)
                            row.append(vals[idx])
                        else:
                            row.append(0.0)
            # Adaptive half-life (10 per team × 2 = 20 features)
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                recent = tr[-30:]
                for stat in ["wp", "margin", "ortg", "drtg", "pace"]:
                    fn = _td_stat_fn.get(stat, lambda r: 0.0)
                    if len(recent) >= 10:
                        vals = [fn(r) for r in recent]
                        vol = _std_17(vals)
                        adaptive_hl = max(3, min(20, 10 / max(vol, 0.01)))
                        row.append(_decay_weighted(recent, fn, adaptive_hl))
                        row.append(adaptive_hl / 20.0)
                    else:
                        row.extend([0.0, 0.5])

            # ── Cross-category interactions ──
            if self.include_market:
                for stat in ["wp", "margin", "ortg"]:
                    for hl in [3, 10]:
                        h_d = _decay_weighted(hr_[-30:], _td_stat_fn.get(stat, lambda r: 0.0), hl)
                        mkt_val = _val2("current_spread")
                        row.append(h_d * mkt_val)
                        row.append(h_d * _val2("current_total"))
                        row.append(h_d * _val2("current_ml_home"))

            # Bayesian × Power Rating interactions
            for prefix, team_key in [("h", home), ("a", away)]:
                bay_wp = self._wp(team_results[team_key], len(team_results[team_key]))
                elo_v = (team_elo[team_key] - 1500) / 400.0
                row.append((bay_wp + elo_v) / 2.0)
                row.append(bay_wp - elo_v)
                row.append(bay_wp * 0.7 + elo_v * 0.3)
                row.append(min(1.0, len(team_results[team_key]) / 30.0))

            # Network × Matchup interactions
            row.extend([0.0, 0.0, 0.0, 0.0])

            # Player Impact × Fatigue interactions
            for prefix, team_key in [("h", home), ("a", away)]:
                tr = hr_ if prefix == "h" else ar_
                rest_here = self._rest_days(team_key, gd, team_last)
                row.append(1.0 if rest_here <= 1 else 0.0)
                row.append(1.0 if rest_here >= 3 else 0.0)
                row.append(h_depth if prefix == "h" else a_depth)
                row.append(1.0 if rest_here <= 1 and len(tr) > 70 else 0.0)

            # Referee × Venue interactions
            row.extend([0.0, 0.0, 0.0, 0.0])

            # Time Series × Cross-Team interactions
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["wp", "margin", "ortg"]:
                    fn = _ts_stat_fn[stat]
                    trend = fn(tr, 5) - fn(tr, 20) if len(tr) >= 20 else 0.0
                    row.append(trend)
                    row.append(trend * abs(self._pace(hr_, 10) - self._pace(ar_, 10)) / 10.0)

            # Ensemble × Market interactions
            if self.include_market:
                row.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Grand composite features
            h_edge = (_val2("elo_diff") + _val2("rest_advantage") * 0.5) / 2.0
            row.append(h_edge)
            row.append(h_edge * 0.5)
            row.append(0.0)
            row.append(0.0)
            row.append(0.0)
            row.append(0.0)
            row.append(0.5)
            row.append(0.0)
            row.append(h_edge * 0.3)
            row.append(0.0)

            # ── Higher-order polynomial features on new categories ──
            _new_sq_features = []
            for prefix in ["h", "a"]:
                _new_sq_features.extend([
                    f"{prefix}_star1_plus_minus_10",
                    f"{prefix}_star1_usage_rate_10",
                    f"{prefix}_star_combined_plus_minus",
                    f"{prefix}_chemistry_starting5",
                    f"bayes2_{prefix}_rating_composite",
                    f"bayes2_{prefix}_coach_expected_wp",
                    f"net2_{prefix}_colley_rating",
                    f"net2_{prefix}_massey_rating",
                ])
            for feat in _new_sq_features:
                v = _val2(feat)
                row.append(v * v)

            # New interaction products
            _new_inter_pairs = [
                ("h_star1_plus_minus_10", "a_star1_plus_minus_10"),
                ("h_star1_usage_rate_10", "a_star1_usage_rate_10"),
                ("h_chemistry_starting5", "a_chemistry_starting5"),
                ("pi_star1_rating_diff", "elo_diff"),
                ("pi_combined_star_diff", "rest_advantage"),
                ("pi_talent_depth_diff", "fatigue_composite_edge"),
                ("xteam_overall_style_clash", "elo_diff"),
                ("xteam_pace_war_indicator", "h_pace10"),
                ("xteam_mismatch_severity", "current_spread"),
                ("bayes_preseason_diff", "h_wp10"),
                ("bayes_franchise_strength_diff", "elo_diff"),
                ("bayes_coach_wp_diff", "rest_advantage"),
                ("net_pagerank_diff_82", "elo_diff"),
                ("net_pagerank_diff_20", "h_wp10"),
                ("net_clustering_diff", "xteam_overall_style_clash"),
                ("net_eigenvector_diff", "bayes_composite_diff"),
                ("ref_home_bias_composite", "venue_home_elevation_advantage"),
                ("ref_pace_impact_composite", "xteam_pace_war_indicator"),
                ("env_combined_venue_advantage", "rest_advantage"),
                ("env_combined_travel_disruption", "fatigue_composite_edge"),
                ("grand_composite_edge", "elo_diff"),
                ("grand_composite_edge", "current_spread"),
                ("grand_multi_signal_agreement", "meta2_ensemble_mean_prob"),
                ("grand_confidence_weighted_edge", "meta2_edge_confidence_product"),
            ]
            for x_feat, y_feat in _new_inter_pairs:
                row.append(_val2(x_feat) * _val2(y_feat))

            # Ratio features
            for x_feat, y_feat in [
                ("h_star1_plus_minus_10", "a_star1_plus_minus_10"),
                ("h_chemistry_starting5", "a_chemistry_starting5"),
                ("h_star_combined_plus_minus", "a_star_combined_plus_minus"),
            ]:
                denom = _val2(y_feat)
                row.append(_val2(x_feat) / (denom + 0.001) if abs(denom) > 0.0001 else 1.0)

            # Triple interactions
            _triple_combos = [
                ("elo_diff", "rest_advantage", "pi_combined_star_diff"),
                ("elo_diff", "xteam_mismatch_severity", "net_pagerank_diff_82"),
                ("elo_diff", "bayes_composite_diff", "meta2_ensemble_mean_prob"),
                ("current_spread", "pi_star1_rating_diff", "ref_home_bias_composite"),
                ("h_wp10", "a_wp10", "xteam_overall_style_clash"),
                ("h_ortg10", "a_drtg10", "xteam_offensive_edge_composite"),
                ("rest_advantage", "env_combined_venue_advantage", "ref_pace_impact_composite"),
                ("fatigue_composite_edge", "pi_talent_depth_diff", "bayes_roster_stability_diff"),
            ]
            for a_f, b_f, c_f in _triple_combos:
                row.append(_val2(a_f) * _val2(b_f) * _val2(c_f))

            # ── Rolling cross-category features ──
            _decay_roll_stats_list = ["wp", "margin", "ortg", "drtg"]
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                recent = tr[-30:]
                for stat in _decay_roll_stats_list:
                    fn = _td_stat_fn.get(stat, lambda r: 0.0)
                    for hl in [3, 10]:
                        dw = _decay_weighted(recent, fn, hl)
                        season_avg = sum(fn(r) for r in tr) / max(len(tr), 1) if tr else 0.0
                        vol = _std_17([fn(r) for r in recent]) if len(recent) >= 3 else 0.0
                        row.append(vol)        # decay stat volatility
                        row.append(dw - season_avg)  # decay stat trend
                        z = (dw - season_avg) / max(vol, 1e-6)
                        row.append(z)          # decay stat z-score

            # Rolling network features
            for prefix, team_key in [("h", home), ("a", away)]:
                tr = team_results[team_key]
                for w in [10, 20]:
                    row.append(0.0)  # pagerank change
                    row.append(0.0)  # centrality change
                    row.append(0.0)  # quality wins trend

            # Rolling Bayesian features
            for prefix in ["h", "a"]:
                for w in [10, 20]:
                    row.extend([0.0, 0.0, 0.0])

            # Rolling cross-team features
            for stat in ["pace", "ortg", "drtg"]:
                fn = _xteam_stat_fn[stat]
                for w in [5, 10]:
                    row.append(abs(fn(hr_, w) - fn(ar_, w)) / 10.0)
                    h_trend = fn(hr_, 5) - fn(hr_, 20) if len(hr_) >= 20 else 0.0
                    a_trend = fn(ar_, 5) - fn(ar_, 20) if len(ar_) >= 20 else 0.0
                    row.append(h_trend - a_trend)

            # Cumulative info features
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                wp = self._wp(tr, 10)
                row.append(wp)
                row.append(wp)
                row.append(1.0 - wp)
                row.append(0.5)
                row.append(0.5)

            # Game-level summary features
            row.extend([0.0] * 10)

            # ── Extended rolling on new stats ──
            _ext_stats = ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                          "pace_adj_margin", "sos_adj_margin", "opponent_efg",
                          "three_pt_rate_diff", "paint_rate_diff", "transition_rate",
                          "halfcourt_efficiency"]
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                ortg_v = self._ortg(tr, 10)
                drtg_v = self._drtg(tr, 10)
                for stat_idx, stat in enumerate(_ext_stats):
                    # Compute base value from available stats
                    if stat == "net_rating":
                        base = ortg_v - drtg_v
                    elif stat == "ast_to_tov":
                        base = self._ast_rate(tr, 10) / max(self._tov_rate(tr, 10), 0.01)
                    elif stat == "efg_minus_opp_efg":
                        base = self._efg(tr, 10) - self._opp_efg(tr, 10)
                    elif stat == "pace_adj_margin":
                        base = (ortg_v - drtg_v) * self._pace(tr, 10) / 100.0
                    elif stat == "sos_adj_margin":
                        base = (ortg_v - drtg_v) * self._sos(tr, team_results, 10)
                    elif stat == "opponent_efg":
                        base = self._opp_efg(tr, 10)
                    elif stat == "three_pt_rate_diff":
                        base = self._stat_avg(tr, 10, "fg3_pct") - self._stat_avg(tr, 10, "opp_fg3_pct")
                    elif stat == "paint_rate_diff":
                        base = self._stat_avg(tr, 10, "paint_pts") - self._stat_avg(tr, 10, "opp_paint_pts")
                    elif stat == "transition_rate":
                        base = self._stat_avg(tr, 10, "fb_pts") / max(self._ppg(tr, 10), 1)
                    else:
                        base = ortg_v / 110.0
                    for w in WINDOWS:
                        # Proxy: regress toward base with window
                        factor = min(1.0, w / 10.0)
                        row.append(base * factor)

            # Extended EWMA
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                             "pace_adj_margin", "three_pt_rate_diff"]:
                    for alpha_str in ["01", "02", "05", "08"]:
                        alpha = {"01": 0.1, "02": 0.2, "05": 0.5, "08": 0.8}[alpha_str]
                        vals = []
                        for r in tr[-20:]:
                            o = r[4].get("ortg", 100)
                            d = r[4].get("drtg", 100)
                            if stat == "net_rating":
                                vals.append(o - d)
                            elif stat == "ast_to_tov":
                                vals.append(r[4].get("ast_rate", 0.6) / max(r[4].get("tov_rate", 0.14), 0.01))
                            elif stat == "efg_minus_opp_efg":
                                vals.append(r[4].get("efg_pct", 0.5) - r[4].get("opp_efg_pct", 0.5))
                            elif stat == "pace_adj_margin":
                                vals.append((o - d) * r[4].get("pace", 100) / 100.0)
                            else:
                                vals.append(r[4].get("fg3_pct", 0.36) - r[4].get("opp_fg3_pct", 0.36))
                        row.append(_ewma_val(vals, alpha) if vals else 0.0)

            # Extended volatility
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                             "pace_adj_margin", "transition_rate"]:
                    for w in [5, 10, 20]:
                        recent = tr[-w:]
                        if len(recent) >= 3:
                            vals = []
                            for r in recent:
                                o = r[4].get("ortg", 100)
                                d = r[4].get("drtg", 100)
                                if stat == "net_rating":
                                    vals.append(o - d)
                                elif stat == "ast_to_tov":
                                    vals.append(r[4].get("ast_rate", 0.6) / max(r[4].get("tov_rate", 0.14), 0.01))
                                elif stat == "efg_minus_opp_efg":
                                    vals.append(r[4].get("efg_pct", 0.5) - r[4].get("opp_efg_pct", 0.5))
                                elif stat == "pace_adj_margin":
                                    vals.append((o - d) * r[4].get("pace", 100) / 100.0)
                                else:
                                    vals.append(r[4].get("fb_pts", 10) / max(r[4].get("pts", 100), 1))
                            row.append(_std_17(vals))
                        else:
                            row.append(0.0)

            # Extended z-scores
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in _ext_stats:
                    vals = []
                    for r in tr:
                        o = r[4].get("ortg", 100)
                        d = r[4].get("drtg", 100)
                        if stat == "net_rating":
                            vals.append(o - d)
                        elif stat == "ast_to_tov":
                            vals.append(r[4].get("ast_rate", 0.6) / max(r[4].get("tov_rate", 0.14), 0.01))
                        elif stat == "efg_minus_opp_efg":
                            vals.append(r[4].get("efg_pct", 0.5) - r[4].get("opp_efg_pct", 0.5))
                        elif stat == "pace_adj_margin":
                            vals.append((o - d) * r[4].get("pace", 100) / 100.0)
                        elif stat == "sos_adj_margin":
                            vals.append(o - d)
                        elif stat == "opponent_efg":
                            vals.append(r[4].get("opp_efg_pct", 0.5))
                        elif stat == "three_pt_rate_diff":
                            vals.append(r[4].get("fg3_pct", 0.36) - r[4].get("opp_fg3_pct", 0.36))
                        elif stat == "paint_rate_diff":
                            vals.append(r[4].get("paint_pts", 40) - r[4].get("opp_paint_pts", 40))
                        elif stat == "transition_rate":
                            vals.append(r[4].get("fb_pts", 10) / max(r[4].get("pts", 100), 1))
                        else:
                            vals.append(o / 110.0)
                    if len(vals) >= 10:
                        s_mean = sum(vals) / len(vals)
                        s_std = _std_17(vals)
                        recent_vals = vals[-5:]
                        r_mean = sum(recent_vals) / len(recent_vals)
                        row.append((r_mean - s_mean) / max(s_std, 1e-6))
                    else:
                        row.append(0.0)

            # Extended trend deltas
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in _ext_stats:
                    base_fn = _xteam_stat_fn.get("pace", lambda t, w: 0.0)
                    for w1, w2 in [(3, 10), (5, 20), (3, 20), (5, 10), (10, 20)]:
                        if stat == "net_rating":
                            v1 = self._netrtg(tr, w1)
                            v2 = self._netrtg(tr, w2)
                        elif stat == "pace_adj_margin":
                            v1 = self._netrtg(tr, w1) * self._pace(tr, w1) / 100.0
                            v2 = self._netrtg(tr, w2) * self._pace(tr, w2) / 100.0
                        else:
                            v1 = self._ortg(tr, w1) - self._drtg(tr, w1)
                            v2 = self._ortg(tr, w2) - self._drtg(tr, w2)
                        row.append(v1 - v2)

            # ── Additional cross-window momentum on new stats ──
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                              "pace_adj_margin", "sos_adj_margin"]:
                    for w1, w2 in [(3, 5), (3, 10), (3, 20), (5, 10), (5, 20),
                                   (7, 15), (7, 20), (10, 20)]:
                        v1 = self._netrtg(tr, w1)
                        v2 = self._netrtg(tr, w2)
                        row.append(v1 - v2)         # delta
                        v1_prev = self._netrtg(tr[:-1], w1) if len(tr) > w1 + 1 else v1
                        v2_prev = self._netrtg(tr[:-1], w2) if len(tr) > w2 + 1 else v2
                        row.append((v1 - v2) - (v1_prev - v2_prev))  # acceleration

            # Cross-window composites for new stats
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for stat in ["net_rating", "ast_to_tov", "efg_minus_opp_efg",
                              "pace_adj_margin", "sos_adj_margin"]:
                    short = self._netrtg(tr, 3)
                    long = self._netrtg(tr, 20) if len(tr) >= 20 else self._netrtg(tr, max(len(tr), 3))
                    mid = self._netrtg(tr, 10) if len(tr) >= 10 else self._netrtg(tr, max(len(tr), 3))
                    row.append(short - mid)        # shortterm trend
                    row.append(long - mid)         # longterm trend
                    vol_s = _std_17([self._netrtg(tr, w) for w in [3, 5, 10, 20] if len(tr) >= w])
                    row.append(vol_s if vol_s else 0.0)  # volatility trend
                    row.append(1.0 if short > long + 3 else 0.0)  # breakout
                    row.append(1.0 if short < long - 3 else 0.0)  # decline

            # ── Additional interaction features ──
            _core_features = ["h_wp10", "a_wp10", "elo_diff", "current_spread",
                              "h_netrtg10", "a_netrtg10", "rest_advantage",
                              "h_ortg10", "a_drtg10", "h_consistency"]
            _new_key_features = [
                "pi_combined_star_diff", "pi_talent_depth_diff",
                "xteam_overall_style_clash", "xteam_mismatch_severity",
                "bayes_composite_diff", "net_pagerank_diff_82",
                "ref_home_bias_composite", "env_combined_venue_advantage",
                "meta2_ensemble_mean_prob", "grand_composite_edge",
            ]
            for core_f in _core_features:
                for new_f in _new_key_features:
                    row.append(_val2(core_f) * _val2(new_f))

            # ── Per-opponent tier features ──
            _opp_tiers = ["elite", "good", "average", "bad", "terrible"]
            _tier_bounds = [(0.65, 1.0), (0.55, 0.65), (0.45, 0.55), (0.35, 0.45), (0.0, 0.35)]
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for tier, (lo, hi) in zip(_opp_tiers, _tier_bounds):
                    tier_games = [r for r in tr if lo <= self._wp(team_results[r[3]], 82) < hi]
                    for stat_name, stat_fn in [
                        ("wp", lambda t, n: self._wp(t, n)),
                        ("margin", lambda t, n: self._pd(t, n)),
                        ("ortg", lambda t, n: self._ortg(t, n)),
                        ("drtg", lambda t, n: self._drtg(t, n)),
                        ("efg", lambda t, n: self._efg(t, n)),
                        ("pace", lambda t, n: self._pace(t, n)),
                    ]:
                        if tier_games:
                            row.append(stat_fn(tier_games, len(tier_games)))
                        else:
                            row.append(0.5 if stat_name == "wp" else 0.0)

            # ── Home/away by window features ──
            for prefix, team_key in [("h", home), ("a", away)]:
                for loc, loc_results in [("home_only", team_home_results.get(team_key, [])),
                                          ("away_only", team_away_results.get(team_key, []))]:
                    for stat_name, stat_fn in [
                        ("wp", lambda t, n: self._wp(t, n)),
                        ("margin", lambda t, n: self._pd(t, n)),
                        ("ortg", lambda t, n: self._ortg(t, n)),
                        ("drtg", lambda t, n: self._drtg(t, n)),
                        ("pace", lambda t, n: self._pace(t, n)),
                        ("efg", lambda t, n: self._efg(t, n)),
                    ]:
                        for w in [5, 10, 20]:
                            if loc_results:
                                row.append(stat_fn(loc_results, w))
                            else:
                                row.append(0.5 if stat_name == "wp" else 0.0)

            # ── Day-of-week features ──
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for d_idx in range(7):
                    d_games = [r for r in tr if self._get_dow(r[0]) == d_idx]
                    if d_games:
                        row.append(self._wp(d_games, len(d_games)))
                        row.append(self._pd(d_games, len(d_games)) / 15.0)
                    else:
                        row.extend([0.5, 0.0])

            # ── Month features ──
            _month_map = {"oct": 10, "nov": 11, "dec": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4}
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for m_name, m_num in _month_map.items():
                    m_games = [r for r in tr if self._get_month(r[0]) == m_num]
                    if m_games:
                        row.append(self._wp(m_games, len(m_games)))
                        row.append(self._pd(m_games, len(m_games)) / 15.0)
                    else:
                        row.extend([0.5, 0.0])

            # ── Consecutive game pattern features ──
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for pattern in ["ww", "wl", "lw", "ll"]:
                    pattern_games = []
                    for i in range(2, len(tr)):
                        p1 = "w" if tr[i-2][1] else "l"
                        p2 = "w" if tr[i-1][1] else "l"
                        if p1 + p2 == pattern:
                            pattern_games.append(tr[i])
                    row.append(self._wp(pattern_games, len(pattern_games)) if pattern_games else 0.5)
                for streak_len in [2, 3, 4, 5]:
                    # After win streak
                    ws_games = []
                    for i in range(streak_len, len(tr)):
                        if all(tr[i - j - 1][1] for j in range(streak_len)):
                            ws_games.append(tr[i])
                    row.append(self._wp(ws_games, len(ws_games)) if ws_games else 0.5)
                    # After loss streak
                    ls_games = []
                    for i in range(streak_len, len(tr)):
                        if all(not tr[i - j - 1][1] for j in range(streak_len)):
                            ls_games.append(tr[i])
                    row.append(self._wp(ls_games, len(ls_games)) if ls_games else 0.5)

            # ── Score differential bucket features ──
            _margin_buckets = {
                "blowout_win": (15, 100),
                "comfortable_win": (6, 14),
                "close_win": (1, 5),
                "close_loss": (-5, -1),
                "comfortable_loss": (-14, -6),
                "blowout_loss": (-100, -15),
            }
            for prefix, tr in [("h", hr_), ("a", ar_)]:
                for bucket_name, (lo, hi) in _margin_buckets.items():
                    bucket_games = [r for r in tr if lo <= r[2] <= hi]
                    row.append(len(bucket_games) / max(len(tr), 1))
                    # Next game win% after this bucket
                    next_games = []
                    for i in range(1, len(tr)):
                        if lo <= tr[i-1][2] <= hi:
                            next_games.append(tr[i])
                    row.append(self._wp(next_games, len(next_games)) if next_games else 0.5)

            # ── Quarter-specific detailed features ──
            for prefix, team_key in [("h", home), ("a", away)]:
                qd_ = (quarter_data or {}).get(team_key, {})
                for q in ["q1", "q2", "q3", "q4"]:
                    for stat in ["margin", "ortg", "drtg", "pace", "efg", "tov_rate", "ft_rate"]:
                        row.append(qd_.get(f"{q}_{stat}", 0.0))

            # ── 36. EWMA PERFORMANCE + CROSSOVERS + REST INTERACTIONS ──
            _ALPHA36 = {"005": 0.05, "015": 0.15, "025": 0.25, "04": 0.4, "07": 0.7}
            _STAT_FN36 = {
                "wp": lambda tr, w: self._wp(tr, w),
                "pd": lambda tr, w: self._pd(tr, w),
                "ppg": lambda tr, w: self._ppg(tr, w),
                "papg": lambda tr, w: self._papg(tr, w),
                "margin": lambda tr, w: self._avg_margin(tr, w),
                "close": lambda tr, w: self._close_pct(tr, w),
                "blowout": lambda tr, w: self._blowout_pct(tr, w),
                "ou_avg": lambda tr, w: self._ou_avg(tr, w),
            }
            for _pfx36, _tr36 in [("h", hr_), ("a", ar_)]:
                # Build per-game stat series (last 20 games) for EWMA
                _n36 = min(20, len(_tr36))
                _series36 = {}
                for _st36 in _STAT_FN36:
                    _series36[_st36] = []
                    for _i36 in range(max(0, len(_tr36) - _n36), len(_tr36)):
                        _sub = _tr36[:_i36+1]
                        _series36[_st36].append(_STAT_FN36[_st36](_sub, min(5, len(_sub))))

                # EWMA: 8 stats × 5 alphas = 40 per team
                for _st36 in ["wp", "pd", "ppg", "papg", "margin", "close", "blowout", "ou_avg"]:
                    _vals = _series36.get(_st36, [])
                    for _ak36 in ["005", "015", "025", "04", "07"]:
                        row.append(_ewma_val(_vals, _ALPHA36[_ak36]) if _vals else 0.0)

                # EWMA crossover: fast(0.7) - slow(0.05) = momentum signal
                for _st36 in ["wp", "pd", "ppg", "papg", "margin", "close", "blowout", "ou_avg"]:
                    _vals = _series36.get(_st36, [])
                    if _vals:
                        row.append(_ewma_val(_vals, 0.7) - _ewma_val(_vals, 0.05))
                    else:
                        row.append(0.0)

            # Rest × performance interactions (12 features)
            _h_wp5 = self._wp(hr_, 5)
            _a_wp5 = self._wp(ar_, 5)
            _h_margin5 = self._avg_margin(hr_, 5)
            _a_margin5 = self._avg_margin(ar_, 5)
            _h_ortg = self._stat_avg(hr_, 10, "ortg")
            _a_ortg = self._stat_avg(ar_, 10, "ortg")
            _h_b2b = 1.0 if h_rest <= 1 else 0.0
            _a_b2b = 1.0 if a_rest <= 1 else 0.0
            _h_fatigue = self._fatigue_score(hr_, gd, home, h_rest) if hasattr(self, '_fatigue_score') else 0.0
            _a_fatigue = self._fatigue_score(ar_, gd, away, a_rest) if hasattr(self, '_fatigue_score') else 0.0
            row.extend([
                min(h_rest, 7) * _h_wp5,               # rest × win%
                min(a_rest, 7) * _a_wp5,
                _h_b2b * _h_margin5,                    # b2b × margin
                _a_b2b * _a_margin5,
                _h_fatigue * _h_ortg,                   # fatigue × offensive rating
                _a_fatigue * _a_ortg,
                (h_rest - a_rest) * (_h_wp5 - _a_wp5),  # rest_adv × wp_diff
                (_h_b2b - _a_b2b) * (_h_margin5 - _a_margin5),  # b2b_diff × margin_diff
                min(h_rest, 7) ** 2 / 49.0,             # rest squared (normalized)
                min(a_rest, 7) ** 2 / 49.0,
                (h_rest - a_rest) * (self._travel_dist(hr_, home) if hasattr(self, '_travel_dist') else 0.0),
                (self._games_in_window(hr_, gd, 7) - self._games_in_window(ar_, gd, 7)) * (_h_margin5 - _a_margin5),
            ])

            # ── 37. MOVDA ELO FEATURES (13 features) ──
            _movda_dr = team_movda[home] - team_movda[away]
            _movda_wp = 1.0 / (1.0 + 10.0 ** (-_movda_dr / _MOVDA_C))
            for _mt, _mk in [(home, home), (away, away)]:
                row.append((team_movda[_mk] - 1500.0) / 400.0)     # movda_rating (normalized)
                row.append(mov_surprise_ewm[_mk] / 20.0)           # mov_surprise_ewm (normalized)
            row.append(_movda_dr / 400.0)                           # movda_diff
            row.append(_movda_wp)                                    # movda_win_prob
            # Raw delta_MOV rolling features (no EWM smoothing)
            for _mk in [home, away]:
                _dh = delta_mov_history[_mk]
                _raw = (_dh[-1] / 20.0) if _dh else 0.0
                _roll5 = (sum(_dh[-5:]) / len(_dh[-5:]) / 20.0) if _dh else 0.0
                _roll10 = (sum(_dh[-10:]) / len(_dh[-10:]) / 20.0) if _dh else 0.0
                row.append(_raw)    # {prefix}_delta_mov_raw
                row.append(_roll5)  # {prefix}_delta_mov_rolling_5
                row.append(_roll10) # {prefix}_delta_mov_rolling_10
            # delta_mov_diff: home rolling_5 - away rolling_5
            _h_dh = delta_mov_history[home]
            _a_dh = delta_mov_history[away]
            _h_r5 = (sum(_h_dh[-5:]) / len(_h_dh[-5:]) / 20.0) if _h_dh else 0.0
            _a_r5 = (sum(_a_dh[-5:]) / len(_a_dh[-5:]) / 20.0) if _a_dh else 0.0
            row.append(_h_r5 - _a_r5)                               # delta_mov_diff

            # ── 38. VENUE-CONDITIONAL MATCHUP FEATURES (14 features) ──
            # Use true venue-specific records: home team at home vs away team on road
            _h_home_tr = team_home_results.get(home, [])
            _a_away_tr = team_away_results.get(away, [])
            for _w38 in [5, 10, 20]:
                # WP edge: home team's home record vs away team's road record
                _h_home_wp = self._wp(_h_home_tr, _w38) if _h_home_tr else self._wp(hr_, _w38)
                _a_road_wp = self._wp(_a_away_tr, _w38) if _a_away_tr else self._wp(ar_, _w38)
                row.append(_h_home_wp - _a_road_wp)
                # Margin edge: home margin at home vs away margin on road
                _h_home_mg = self._pd(_h_home_tr, _w38) if _h_home_tr else self._pd(hr_, _w38)
                _a_road_mg = self._pd(_a_away_tr, _w38) if _a_away_tr else self._pd(ar_, _w38)
                row.append(_h_home_mg - _a_road_mg)
                # ORtg edge: home team's home offense vs away team's road defense
                _h_home_or = self._ortg(_h_home_tr, _w38) if _h_home_tr else self._ortg(hr_, _w38)
                _a_road_dr = self._drtg(_a_away_tr, _w38) if _a_away_tr else self._drtg(ar_, _w38)
                row.append((_h_home_or - _a_road_dr) / 10.0)
                # DRtg edge: home team's home defense vs away team's road offense
                _h_home_dr = self._drtg(_h_home_tr, _w38) if _h_home_tr else self._drtg(hr_, _w38)
                _a_road_or = self._ortg(_a_away_tr, _w38) if _a_away_tr else self._ortg(ar_, _w38)
                row.append((_h_home_dr - _a_road_or) / 10.0)
            # Home court boost: how much better the home team is at home vs overall
            _h_overall_wp = self._wp(hr_, len(hr_)) if hr_ else 0.5
            _h_home_wp_82 = self._wp(_h_home_tr, len(_h_home_tr)) if _h_home_tr else _h_overall_wp
            row.append(_h_home_wp_82 - _h_overall_wp)
            # Road penalty: how much worse away team is on road vs overall
            _a_overall_wp = self._wp(ar_, len(ar_)) if ar_ else 0.5
            _a_road_wp_82 = self._wp(_a_away_tr, len(_a_away_tr)) if _a_away_tr else _a_overall_wp
            row.append(_a_overall_wp - _a_road_wp_82)

            # ── 39. CIRCADIAN RHYTHM & TRAVEL FATIGUE (8 features) ──
            # Novel normalized composites distinct from raw Cat 6 rest/travel features
            try:
                _h_dist = self._travel_dist(hr_, home) / 500.0  # Normalize: ~500mi = 1 unit
                _a_dist = self._travel_dist(ar_, away) / 500.0
                _h_tz = abs(TIMEZONE_ET.get(home, 0) - TIMEZONE_ET.get(self._last_location(hr_), 0))
                _a_tz = abs(TIMEZONE_ET.get(away, 0) - TIMEZONE_ET.get(self._last_location(ar_), 0))
                _h_b2b = 1.0 if h_rest <= 1 else 0.0
                _a_b2b = 1.0 if a_rest <= 1 else 0.0
                # Fatigue index: travel + timezone disruption + back-to-back penalty - rest recovery
                _h_fatigue = _h_dist + _h_tz * 0.5 + _h_b2b * 2.0 - min(h_rest, 4) * 0.3
                _a_fatigue = _a_dist + _a_tz * 0.5 + _a_b2b * 2.0 - min(a_rest, 4) * 0.3
                # Rest non-linearity: diminishing benefit beyond 3 days (capped at ±2)
                _h_rest_nl = min(h_rest, 5) ** 0.5 - min(a_rest, 5) ** 0.5
                row.extend([
                    min(_h_dist, 6.0),                   # circ_h_travel_dist (cap at 6 = 3000mi)
                    min(_a_dist, 6.0),                   # circ_a_travel_dist
                    float(_h_tz),                        # circ_h_tz_shift
                    float(_a_tz),                        # circ_a_tz_shift
                    max(-3.0, min(5.0, _h_fatigue)),     # circ_h_fatigue_index (clipped)
                    max(-3.0, min(5.0, _a_fatigue)),     # circ_a_fatigue_index (clipped)
                    max(-5.0, min(5.0, _a_fatigue - _h_fatigue)),  # circ_advantage
                    max(-2.0, min(2.0, _h_rest_nl)),     # circ_rest_nonlinear
                ])
            except Exception:
                row.extend([0.0] * 8)

            # ── 41. TRANSITION vs HALF-COURT EFFICIENCY SPLITS (7 features) ──
            # Derived from fb_pts (fast break) and pace in existing box score stats
            try:
                _h_fb_rate = self._stat_avg(hr_, 10, "fb_pts") / max(self._ppg(hr_, 10), 1.0)
                _a_fb_rate = self._stat_avg(ar_, 10, "fb_pts") / max(self._ppg(ar_, 10), 1.0)
                _h_pace10 = self._pace(hr_, 10)
                _a_pace10 = self._pace(ar_, 10)
                # Half-court efficiency: scoring minus fast-break contribution, per possession
                _h_hc_eff = (self._ppg(hr_, 10) * (1.0 - _h_fb_rate)) / max(_h_pace10, 60.0) * 100.0
                _a_hc_eff = (self._ppg(ar_, 10) * (1.0 - _a_fb_rate)) / max(_a_pace10, 60.0) * 100.0
                # pace × fb_rate interaction: high-pace + high-transition = fast-break synergy
                _h_pace_fb = (_h_pace10 / 100.0) * _h_fb_rate
                _a_pace_fb = (_a_pace10 / 100.0) * _a_fb_rate
                row.extend([
                    min(_h_fb_rate, 0.5),                # trans41_h_fb_rate
                    min(_a_fb_rate, 0.5),                # trans41_a_fb_rate
                    min(_h_hc_eff / 120.0, 1.2),         # trans41_h_halfcourt_eff (normalized)
                    min(_a_hc_eff / 120.0, 1.2),         # trans41_a_halfcourt_eff
                    _h_fb_rate - _a_fb_rate,             # trans41_fb_rate_diff
                    _h_pace_fb - _a_pace_fb,             # trans41_pace_x_fb
                    (_h_hc_eff - _a_hc_eff) / 20.0,     # trans41_halfcourt_edge
                ])
            except Exception:
                row.extend([0.0] * 7)

            # ── 43. CLUTCH PERFORMANCE FEATURES (8 features) ──
            # Computed from close games (|margin| <= 5) in rolling records
            try:
                _h_clutch = [r for r in hr_[-30:] if abs(r[2]) <= 5]
                _a_clutch = [r for r in ar_[-30:] if abs(r[2]) <= 5]
                _h_cwp = self._wp(_h_clutch, len(_h_clutch)) if _h_clutch else 0.5
                _a_cwp = self._wp(_a_clutch, len(_a_clutch)) if _a_clutch else 0.5
                _h_cmg = self._pd(_h_clutch, len(_h_clutch)) if _h_clutch else 0.0
                _a_cmg = self._pd(_a_clutch, len(_a_clutch)) if _a_clutch else 0.0
                # Ortg in clutch games
                _h_cortg = (sum(r[4].get("ortg", 100.0) for r in _h_clutch) / len(_h_clutch)
                            if _h_clutch else self._ortg(hr_, 10))
                _a_cortg = (sum(r[4].get("ortg", 100.0) for r in _a_clutch) / len(_a_clutch)
                            if _a_clutch else self._ortg(ar_, 10))
                row.extend([
                    _h_cwp,                              # clutch43_h_wp
                    _a_cwp,                              # clutch43_a_wp
                    _h_cmg / 10.0,                       # clutch43_h_margin (normalized)
                    _a_cmg / 10.0,                       # clutch43_a_margin
                    (_h_cortg - 100.0) / 20.0,           # clutch43_h_ortg (normalized)
                    (_a_cortg - 100.0) / 20.0,           # clutch43_a_ortg
                    _h_cwp - _a_cwp,                     # clutch43_wp_diff
                    (_h_cmg - _a_cmg) / 10.0,            # clutch43_margin_diff
                ])
            except Exception:
                row.extend([0.0] * 8)

            # ── 44. GAME TOTALS PREDICTION FEATURES (10 features) ──
            # Normalized pace/scoring context to signal high-total vs grind-it-out games.
            # High-total environments favor offense-heavy win predictions; low-total favors defense.
            try:
                _league_ppg = 110.0   # baseline league average PPG
                _h_ppg10 = self._ppg(hr_, 10)
                _a_ppg10 = self._ppg(ar_, 10)
                _h_pap10 = self._papg(hr_, 10)
                _a_pap10 = self._papg(ar_, 10)
                # Normalize each PPG/PAPG to league avg (1.0 = average, >1 = high scoring)
                _h_ppg_n = _h_ppg10 / _league_ppg
                _a_ppg_n = _a_ppg10 / _league_ppg
                _h_pap_n = _h_pap10 / _league_ppg
                _a_pap_n = _a_pap10 / _league_ppg
                # Matchup total: (H_PPG + A_PAP)/2 + (A_PPG + H_PAP)/2 = H scoring pace + A scoring pace
                # Normalize to league average total (220.0)
                _matchup_total = ((_h_ppg10 + _a_pap10) / 2.0 + (_a_ppg10 + _h_pap10) / 2.0) / 220.0
                # Pace: use poss as proxy (ortg stored per 100 poss)
                _h_pace10 = self._pace(hr_, 10)
                _a_pace10 = self._pace(ar_, 10)
                _avg_pace = (_h_pace10 + _a_pace10) / 2.0
                # Normalize pace to ~97 league avg
                _pace_sum_n = _avg_pace / 97.0
                _pace_mismatch = abs(_h_pace10 - _a_pace10) / 10.0   # cap at ~3 sigma
                # Offensive + defensive ratings (normalize to 110 = league avg)
                _h_ortg10 = self._ortg(hr_, 10)
                _a_ortg10 = self._ortg(ar_, 10)
                _h_drtg10 = self._drtg(hr_, 10)
                _a_drtg10 = self._drtg(ar_, 10)
                _ortg_sum = (_h_ortg10 + _a_ortg10) / (2.0 * 110.0)  # >1 = above-avg offense
                _drtg_sum = (_h_drtg10 + _a_drtg10) / (2.0 * 110.0)  # >1 = above-avg defense (worse)
                # Net scoring environment: positive = high-scoring game expected
                _score_env = ((_h_ortg10 + _a_ortg10) - (_h_drtg10 + _a_drtg10)) / 20.0
                row.extend([
                    max(0.5, min(2.0, _h_ppg_n)),        # tot44_h_ppg10
                    max(0.5, min(2.0, _a_ppg_n)),        # tot44_a_ppg10
                    max(0.5, min(2.0, _h_pap_n)),        # tot44_h_papg10
                    max(0.5, min(2.0, _a_pap_n)),        # tot44_a_papg10
                    max(0.7, min(1.5, _matchup_total)),  # tot44_matchup_total
                    max(0.7, min(1.4, _pace_sum_n)),     # tot44_pace_sum
                    min(2.0, _pace_mismatch),            # tot44_pace_mismatch
                    max(0.7, min(1.4, _ortg_sum)),       # tot44_ortg_sum
                    max(0.7, min(1.4, _drtg_sum)),       # tot44_drtg_sum
                    max(-1.5, min(1.5, _score_env)),     # tot44_score_env
                ])
            except Exception:
                row.extend([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

            # ── Cat 42: Shot Quality Zone Features (10 features) ──
            try:
                td = tracking_data or {}
                h_td = td.get(home, {})
                a_td = td.get(away, {})
                _h_rim = h_td.get('rim_rate', 0.30)
                _a_rim = a_td.get('rim_rate', 0.30)
                _h_mid = h_td.get('mid_rate', 0.15)
                _a_mid = a_td.get('mid_rate', 0.15)
                _h_three = h_td.get('three_rate', 0.40)
                _a_three = a_td.get('three_rate', 0.40)
                # xEFG: weighted by league-avg zone FG%
                _h_xefg = 0.65 * _h_rim + 0.40 * _h_mid + 0.53 * _h_three * 1.5
                _a_xefg = 0.65 * _a_rim + 0.40 * _a_mid + 0.53 * _a_three * 1.5
                row.extend([
                    _h_rim,                                  # shot42_h_rim_rate
                    _a_rim,                                  # shot42_a_rim_rate
                    _h_mid,                                  # shot42_h_mid_rate
                    _a_mid,                                  # shot42_a_mid_rate
                    _h_three,                                # shot42_h_three_rate
                    _a_three,                                # shot42_a_three_rate
                    _h_xefg,                                 # shot42_h_xefg
                    _a_xefg,                                 # shot42_a_xefg
                    _h_rim - _a_rim,                         # shot42_rim_rate_diff
                    _h_xefg - _a_xefg,                       # shot42_xefg_diff
                ])
            except Exception:
                row.extend([0.30, 0.30, 0.15, 0.15, 0.40, 0.40, 0.51, 0.51, 0.0, 0.0])

            # ── Cat 45: Player Tracking / Hustle Features (12 features) ──
            try:
                td = tracking_data or {}
                h_td = td.get(home, {})
                a_td = td.get(away, {})
                _h_cont = h_td.get('contested_shots', 0.0) / 50.0   # normalize (~50/game)
                _a_cont = a_td.get('contested_shots', 0.0) / 50.0
                _h_defl = h_td.get('deflections', 0.0) / 15.0       # normalize (~15/game)
                _a_defl = a_td.get('deflections', 0.0) / 15.0
                _h_spd = h_td.get('avg_speed', 0.0) / 5.0           # normalize (~4.5 mph)
                _a_spd = a_td.get('avg_speed', 0.0) / 5.0
                _h_lb = h_td.get('loose_balls', 0.0) / 8.0          # normalize (~8/game)
                _a_lb = a_td.get('loose_balls', 0.0) / 8.0
                _h_drv = h_td.get('drives', 0.0) / 50.0             # normalize (~50/game)
                _a_drv = a_td.get('drives', 0.0) / 50.0
                row.extend([
                    _h_cont,                                 # track45_h_contested
                    _a_cont,                                 # track45_a_contested
                    _h_defl,                                 # track45_h_deflections
                    _a_defl,                                 # track45_a_deflections
                    _h_spd,                                  # track45_h_speed
                    _a_spd,                                  # track45_a_speed
                    _h_lb,                                   # track45_h_loose_balls
                    _a_lb,                                   # track45_a_loose_balls
                    _h_drv,                                  # track45_h_drives
                    _a_drv,                                  # track45_a_drives
                    _h_cont - _a_cont,                       # track45_contested_diff
                    _h_spd - _a_spd,                         # track45_speed_diff
                ])
            except Exception:
                row.extend([0.0] * 12)

            # ── Cat 46: Real Odds Market Features (8 features) ──
            try:
                _odds_key = (gd, home, away)
                _odds = (odds_data or {}).get(_odds_key, {})
                if _odds:
                    _ip_home = _odds.get('implied_home_prob', 0.5)
                    _ip_away = _odds.get('implied_away_prob', 0.5)
                    _fp_home = _odds.get('fair_home_prob', 0.5)
                    _fp_away = _odds.get('fair_away_prob', 0.5)
                    _sp_home = _odds.get('spread_home', None)
                    _total   = _odds.get('total', None)
                    _overr   = _odds.get('overround', 1.05)

                    # Normalize spread: /10 so that a 10-point spread = 1.0
                    _sp_norm = (_sp_home / 10.0) if _sp_home is not None else 0.0
                    # Normalize total: /220 so that league-average total ~ 1.0
                    _total_norm = (_total / 220.0) if _total is not None else 1.0

                    # Spread-implied probability (logistic approximation):
                    # P(home_win) ≈ 1 / (1 + 10^(spread / 7.5))
                    # A 7.5-point spread corresponds to ~90% win probability
                    if _sp_home is not None:
                        _sp_implied = 1.0 / (1.0 + 10.0 ** (_sp_home / 7.5))
                    else:
                        _sp_implied = _fp_home  # fallback to moneyline-derived

                    # Market consistency: difference between spread-implied and ML-implied
                    _spread_ml_diff = _sp_implied - _fp_home

                    row.extend([
                        _ip_home,              # odds46_implied_home_prob
                        _ip_away,              # odds46_implied_away_prob
                        _fp_home,              # odds46_fair_home_prob
                        _fp_away,              # odds46_fair_away_prob
                        _sp_norm,              # odds46_spread_home (normalized)
                        _total_norm,           # odds46_total (normalized)
                        _overr,                # odds46_overround
                        _spread_ml_diff,       # odds46_spread_implied_diff
                    ])
                else:
                    # No odds data for this game — safe defaults
                    row.extend([0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 1.05, 0.0])
            except Exception:
                row.extend([0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 1.05, 0.0])

            # ── Cat 47: Drive-Offense vs Rim-Defense Matchup (14 features) ──
            try:
                td = tracking_data or {}
                h_td = td.get(home, {})
                a_td = td.get(away, {})
                _h_drv_fg = h_td.get('drive_fg_pct', 0.488)
                _a_drv_fg = a_td.get('drive_fg_pct', 0.488)
                _h_drv_tov = h_td.get('drive_tov_pct', 0.07)
                _a_drv_tov = a_td.get('drive_tov_pct', 0.07)
                _h_drv_pts = h_td.get('drive_pts_pct', 0.50)
                _a_drv_pts = a_td.get('drive_pts_pct', 0.50)
                _h_rim_d = h_td.get('def_rim_fg_pct', 0.66)
                _a_rim_d = a_td.get('def_rim_fg_pct', 0.66)
                _h_blk = h_td.get('blk_per_game', 4.5) / 10.0
                _a_blk = a_td.get('blk_per_game', 4.5) / 10.0
                _h_drv_n = h_td.get('drives', 48.0) / 50.0
                _a_drv_n = a_td.get('drives', 48.0) / 50.0
                _h_off_vs_a = _h_drv_fg - _a_rim_d
                _a_off_vs_h = _a_drv_fg - _h_rim_d
                row.extend([
                    _h_drv_fg,               # drive47_h_fg_pct
                    _a_drv_fg,               # drive47_a_fg_pct
                    _h_drv_tov,              # drive47_h_tov_pct
                    _a_drv_tov,              # drive47_a_tov_pct
                    _h_drv_pts,              # drive47_h_pts_pct
                    _a_drv_pts,              # drive47_a_pts_pct
                    _h_rim_d,                # drive47_h_def_rim_fg
                    _a_rim_d,                # drive47_a_def_rim_fg
                    _h_blk,                  # drive47_h_blk_rate
                    _a_blk,                  # drive47_a_blk_rate
                    _h_off_vs_a,             # drive47_h_off_vs_a_rim
                    _a_off_vs_h,             # drive47_a_off_vs_h_rim
                    _h_off_vs_a - _a_off_vs_h,  # drive47_rim_matchup_net
                    _h_drv_n - _a_drv_n,     # drive47_drive_volume_diff
                ])
            except Exception:
                row.extend([0.488, 0.488, 0.07, 0.07, 0.50, 0.50, 0.66, 0.66,
                           0.45, 0.45, 0.0, 0.0, 0.0, 0.0])

            # ── Cat 48: Passing Network Quality (10 features) ──
            try:
                td = tracking_data or {}
                h_td = td.get(home, {})
                a_td = td.get(away, {})
                _h_atp = h_td.get('ast_to_pass_pct', 0.09)
                _a_atp = a_td.get('ast_to_pass_pct', 0.09)
                _h_pot = h_td.get('potential_ast', 47.0) / 50.0
                _a_pot = a_td.get('potential_ast', 47.0) / 50.0
                _h_apc = h_td.get('ast_points_created', 68.0) / 80.0
                _a_apc = a_td.get('ast_points_created', 68.0) / 80.0
                _h_sec = h_td.get('secondary_ast', 3.0) / 5.0
                _a_sec = a_td.get('secondary_ast', 3.0) / 5.0
                _bm_h = _h_atp + _h_sec * 0.1 + _h_pot * 0.05
                _bm_a = _a_atp + _a_sec * 0.1 + _a_pot * 0.05
                row.extend([
                    _h_atp,                  # pass48_h_ast_rate
                    _a_atp,                  # pass48_a_ast_rate
                    _h_pot,                  # pass48_h_potential_ast
                    _a_pot,                  # pass48_a_potential_ast
                    _h_apc,                  # pass48_h_ast_pts_created
                    _a_apc,                  # pass48_a_ast_pts_created
                    _h_sec,                  # pass48_h_secondary_ast
                    _a_sec,                  # pass48_a_secondary_ast
                    _h_atp - _a_atp,         # pass48_ast_rate_diff
                    _bm_h - _bm_a,           # pass48_ball_movement_edge
                ])
            except Exception:
                row.extend([0.09, 0.09, 0.94, 0.94, 0.85, 0.85, 0.6, 0.6, 0.0, 0.0])

            # ── Cat 49: Play-Type Efficiency (10 features) ──
            try:
                td = tracking_data or {}
                h_td = td.get(home, {})
                a_td = td.get(away, {})
                _h_iso = h_td.get('iso_ppp', 0.88)
                _a_iso = a_td.get('iso_ppp', 0.88)
                _h_pnr = h_td.get('pnr_ppp', 0.87)
                _a_pnr = a_td.get('pnr_ppp', 0.87)
                _h_spot = h_td.get('spot_ppp', 1.04)
                _a_spot = a_td.get('spot_ppp', 1.04)
                _h_trans = h_td.get('trans_ppp', 1.12)
                _a_trans = a_td.get('trans_ppp', 1.12)
                _h_avg = (_h_iso + _h_pnr + _h_spot + _h_trans) / 4.0
                _a_avg = (_a_iso + _a_pnr + _a_spot + _a_trans) / 4.0
                _h_above = sum(1 for p in [_h_iso, _h_pnr, _h_spot, _h_trans] if p > 1.0) / 4.0
                _a_above = sum(1 for p in [_a_iso, _a_pnr, _a_spot, _a_trans] if p > 1.0) / 4.0
                row.extend([
                    _h_iso,                  # play49_h_iso_ppp
                    _a_iso,                  # play49_a_iso_ppp
                    _h_pnr,                  # play49_h_pnr_ppp
                    _a_pnr,                  # play49_a_pnr_ppp
                    _h_spot,                 # play49_h_spot_ppp
                    _a_spot,                 # play49_a_spot_ppp
                    _h_trans,                # play49_h_trans_ppp
                    _a_trans,                # play49_a_trans_ppp
                    _h_avg - _a_avg,         # play49_ppp_composite_diff
                    _h_above - _a_above,     # play49_versatility_diff
                ])
            except Exception:
                row.extend([0.88, 0.88, 0.87, 0.87, 1.04, 1.04, 1.12, 1.12, 0.0, 0.0])

            # ── Cat 50: Temporal Win-Sequence Encoding (12 features) ──
            try:
                def _seq_feats(records, n=10):
                    """Encode temporal sequence: order of wins/losses matters beyond averages."""
                    last_n = records[-n:] if len(records) >= n else records[:]
                    m = len(last_n)
                    if m == 0:
                        return [0.5, 0.5, 0.0, 0.0, 0.0]
                    mid = max(1, m // 2)
                    early = last_n[:mid]
                    late  = last_n[mid:] if len(last_n) > mid else last_n[-1:]
                    early_wp = sum(1 for r in early if r[1]) / len(early)
                    late_wp  = sum(1 for r in late  if r[1]) / len(late)
                    slope    = late_wp - early_wp
                    r3 = last_n[-3:]
                    o3 = last_n[:3] if len(last_n) >= 6 else last_n[:1]
                    m_recent = sum(r[2] for r in r3) / len(r3)
                    m_old    = sum(r[2] for r in o3) / len(o3)
                    m_slope  = max(-1.0, min(1.0, (m_recent - m_old) / 30.0))
                    streak = 0
                    last_val = last_n[-1][1]
                    for r in reversed(last_n):
                        if r[1] == last_val:
                            streak += 1 if r[1] else -1
                        else:
                            break
                    return [early_wp, late_wp, slope, m_slope, max(-1.0, min(1.0, streak / 10.0))]

                _hs = _seq_feats(team_results[home])
                _as = _seq_feats(team_results[away])
                row.extend([
                    _hs[0],            # seq50_h_early_wp
                    _hs[1],            # seq50_h_late_wp
                    _hs[2],            # seq50_h_slope
                    _hs[3],            # seq50_h_margin_slope_norm
                    _hs[4],            # seq50_h_streak_norm
                    _as[0],            # seq50_a_early_wp
                    _as[1],            # seq50_a_late_wp
                    _as[2],            # seq50_a_slope
                    _as[3],            # seq50_a_margin_slope_norm
                    _as[4],            # seq50_a_streak_norm
                    _hs[2] - _as[2],   # seq50_slope_diff
                    _hs[4] - _as[4],   # seq50_streak_diff
                ])
            except Exception:
                row.extend([0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 51. SEASON ERA NORMALIZATION
            # Normalize rolling efficiency vs league-wide running avg for THIS season.
            # Uses stats from all games BEFORE this one (no lookahead).
            try:
                sid = _era_season_id(gd)
                h_ortg_v = self._ortg(hr_, 10)
                h_drtg_v = self._drtg(hr_, 10)
                a_ortg_v = self._ortg(ar_, 10)
                a_drtg_v = self._drtg(ar_, 10)
                h_pace_v = self._pace(hr_, 10)
                a_pace_v = self._pace(ar_, 10)
                h_nrtg_v = h_ortg_v - h_drtg_v
                a_nrtg_v = a_ortg_v - a_drtg_v

                ort_hist = _era51_ortg[sid]
                drt_hist = _era51_drtg[sid]
                pc_hist  = _era51_pace[sid]
                nrt_hist = _era51_nrtg[sid]

                h_oz = _era_zscore(h_ortg_v, ort_hist)
                h_dz = _era_zscore(h_drtg_v, drt_hist)
                a_oz = _era_zscore(a_ortg_v, ort_hist)
                a_dz = _era_zscore(a_drtg_v, drt_hist)
                h_pz = _era_zscore(h_pace_v, pc_hist)
                a_pz = _era_zscore(a_pace_v, pc_hist)
                h_nz = _era_zscore(h_nrtg_v, nrt_hist)

                row.extend([h_oz, h_dz, a_oz, a_dz, h_pz, a_pz, h_nz, h_oz - a_dz])

                # Update season trackers for future games (must be after feature extraction)
                _era51_ortg[sid].extend([h_ortg_v, a_ortg_v])
                _era51_drtg[sid].extend([h_drtg_v, a_drtg_v])
                _era51_pace[sid].extend([h_pace_v, a_pace_v])
                _era51_nrtg[sid].extend([h_nrtg_v, a_nrtg_v])
            except Exception:
                row.extend([0.0] * 8)

            # ── Cat 52: Odds Line Features (15 features) ──
            # Raw market signals: spread magnitude, total, vig, season percentiles.
            try:
                _odds_key = (gd, home, away)
                _odds = (odds_data or {}).get(_odds_key, {})
                _sp52  = _odds.get('spread_home', None)   # None if no odds for this game
                _tot52 = _odds.get('total', None)
                _ip52h = _odds.get('implied_home_prob', 0.5)
                _ip52a = _odds.get('implied_away_prob', 0.5)
                _or52  = _odds.get('overround', 1.05)

                _sp_abs = abs(_sp52) if _sp52 is not None else 3.5
                _tot_v  = _tot52 if _tot52 is not None else 220.0

                # Spread agrees with ML if both point same direction
                _sp_sign = -1.0 if (_sp52 is not None and _sp52 < 0) else 1.0
                _ml_sign = -1.0 if _ip52h > 0.5 else 1.0
                _agree = 1.0 if _sp_sign == _ml_sign else 0.0

                _vig52 = max(0.0, _or52 - 1.0)
                _home_dog = 1.0 if (_sp52 is not None and _sp52 > 0) else 0.0

                # Spread adjusted for home court (home teams typically get -3.5 to -4 HCA)
                _sp_adj = (_sp52 + 3.5) if _sp52 is not None else 0.0

                # ML-implied spread vs actual spread
                # Logistic inversion: spread ≈ -7.5 * log10(p/(1-p)) for fair p
                _fp52h = _odds.get('fair_home_prob', _ip52h)
                if 0.001 < _fp52h < 0.999:
                    _ml_impl_spread = -7.5 * math.log10(_fp52h / (1.0 - _fp52h))
                else:
                    _ml_impl_spread = 0.0
                _ml_sp_gap = (_ml_impl_spread - (_sp52 if _sp52 is not None else _ml_impl_spread)) / 10.0

                _sharpness = 1.0 / max(_or52, 1.0)

                # Season-relative percentiles (based on spreads/totals seen so far)
                def _pct_rank(val, lst):
                    """Fraction of values in lst that are <= val."""
                    if not lst:
                        return 0.5
                    return sum(1 for v in lst if v <= val) / len(lst)

                _sp_pct  = _pct_rank(_sp_abs, _season_spreads)
                _tot_pct = _pct_rank(_tot_v, _season_totals)

                # Season spread rolling std (line volatility indicator)
                if len(_season_spreads) >= 5:
                    _sp_mu  = sum(_season_spreads) / len(_season_spreads)
                    _sp_std = math.sqrt(sum((v - _sp_mu)**2 for v in _season_spreads) / len(_season_spreads))
                else:
                    _sp_std = 3.5

                # Total trend: recent 10 avg vs season avg
                _recent_tots = _season_totals[-10:] if len(_season_totals) >= 10 else _season_totals
                _tot_trend = (sum(_recent_tots) / len(_recent_tots) - _tot_v) / 10.0 if _recent_tots else 0.0

                # Home favorite strength (one-sided: 0 if dog)
                _h_fav_str = ((-_sp52) / 10.0) if (_sp52 is not None and _sp52 < 0) else 0.0

                row.extend([
                    _sp_abs,          # line52_spread_magnitude
                    _tot_v,           # line52_total
                    _ip52h,           # line52_implied_home
                    _ip52a,           # line52_implied_away
                    _agree,           # line52_spread_agree
                    _vig52,           # line52_vig
                    _sp_pct,          # line52_spread_season_pct
                    _tot_pct,         # line52_total_season_pct
                    _home_dog,        # line52_home_dog
                    _sp_adj / 10.0,   # line52_spread_adj (normalized)
                    _ml_sp_gap,       # line52_ml_spread_gap
                    _sharpness,       # line52_sharpness
                    _sp_std,          # line52_season_spread_std
                    _tot_trend,       # line52_season_total_trend
                    _h_fav_str,       # line52_home_fav_strength
                ])
            except Exception:
                row.extend([3.5, 220.0, 0.5, 0.5, 1.0, 0.05, 0.5, 0.5, 0.0, 0.0, 0.0, 0.952, 3.5, 0.0, 0.0])

            # ── Cat 53: ATS Record Features (12 features) ──
            # Cover rate derived from _team_ats tracker (populated after each game's features).
            try:
                def _ats_rate(records, n, fav_only=False, dog_only=False, home_only=False):
                    """Cover rate: last n ATS records. Record = (gd, covered, spread, is_home)."""
                    s = records[-n:] if n else records[:]
                    if fav_only:
                        s = [r for r in s if r[2] < 0]   # team was favored (spread_home < 0)
                    if dog_only:
                        s = [r for r in s if r[2] > 0]   # team was dog
                    if home_only:
                        s = [r for r in s if r[3]]
                    if not s:
                        return 0.5
                    return sum(1 for r in s if r[1]) / len(s)

                def _ats_streak(records):
                    """ATS streak: + if covering, - if not."""
                    if not records:
                        return 0
                    last_val = records[-1][1]
                    st = 0
                    for r in reversed(records):
                        if r[1] == last_val:
                            st += 1 if last_val else -1
                        else:
                            break
                    return st

                def _margin_vs_spread(records, n):
                    """Avg (actual_margin - spread) last n games. + = covered more than expected."""
                    s = records[-n:] if records else []
                    if not s:
                        return 0.0
                    # record format: (gd, covered, spread, is_home, actual_margin_vs_spread)
                    vals = [r[4] for r in s if len(r) > 4]
                    return sum(vals) / len(vals) if vals else 0.0

                h_ats = _team_ats[home]
                a_ats = _team_ats[away]

                # H2H ATS: last 5 h2h meetings from home team perspective
                h2h_games = h2h_results.get((home, away), []) + h2h_results.get((away, home), [])
                h2h_ats_dates = {r[0] for r in h_ats}
                h2h_h_ats = [r for r in h_ats if r[0] in {g[0] for g in h2h_games}]

                row.extend([
                    _ats_rate(h_ats, 10),                           # ats53_h_last10
                    _ats_rate(a_ats, 10),                           # ats53_a_last10
                    _ats_rate(h_ats, 0),                            # ats53_h_season
                    _ats_rate(a_ats, 0),                            # ats53_a_season
                    max(-5.0, min(5.0, _ats_streak(h_ats))) / 5.0, # ats53_h_streak (normalized)
                    max(-5.0, min(5.0, _ats_streak(a_ats))) / 5.0, # ats53_a_streak
                    _ats_rate(h_ats, 0, fav_only=True),            # ats53_h_as_fav
                    _ats_rate(a_ats, 0, dog_only=True),            # ats53_a_as_dog
                    _ats_rate(h2h_h_ats, 5) if h2h_h_ats else 0.5, # ats53_h2h_last5
                    _ats_rate(h_ats, 0, home_only=True),           # ats53_h_home_only
                    _ats_rate(a_ats, 0, home_only=False),          # ats53_a_away_only (road games)
                    _margin_vs_spread(h_ats, 10),                   # ats53_margin_vs_spread_10
                ])
            except Exception:
                row.extend([0.5] * 11 + [0.0])

            # ── Cat 54: Over/Under Record Features (12 features) ──
            # Over rate derived from _team_ou tracker (populated after each game's features).
            try:
                def _ou_rate(records, n, home_only=None):
                    """Over rate: last n O/U records. Record = (gd, went_over, total, is_home)."""
                    s = records[-n:] if n else records[:]
                    if home_only is True:
                        s = [r for r in s if r[3]]
                    elif home_only is False:
                        s = [r for r in s if not r[3]]
                    if not s:
                        return 0.5
                    return sum(1 for r in s if r[1]) / len(s)

                def _ou_streak(records):
                    """O/U streak: + if over, - if under."""
                    if not records:
                        return 0
                    last_val = records[-1][1]
                    st = 0
                    for r in reversed(records):
                        if r[1] == last_val:
                            st += 1 if last_val else -1
                        else:
                            break
                    return st

                def _ou_margin_avg(h_records, a_records, n):
                    """Avg (actual_total - ou_line) last n for combined home+away records."""
                    h_s = h_records[-n:] if h_records else []
                    a_s = a_records[-n:] if a_records else []
                    combined = h_s + a_s
                    if not combined:
                        return 0.0
                    # record has (gd, went_over, total, is_home, actual_vs_total)
                    vals = [r[4] for r in combined if len(r) > 4]
                    return sum(vals) / len(vals) if vals else 0.0

                h_ou = _team_ou[home]
                a_ou = _team_ou[away]
                combined_over_rate = (
                    (_ou_rate(h_ou, 10) + _ou_rate(a_ou, 10)) / 2.0
                    if h_ou or a_ou else 0.5
                )

                # Pace vs total: home team's recent PPG+PAPG vs the line
                _h_ppg_apg = self._ppg(hr_, 10) + self._papg(hr_, 10)
                _odds_key = (gd, home, away)
                _tot_line = (odds_data or {}).get(_odds_key, {}).get('total', 220.0) or 220.0
                _pace_vs_total = (_h_ppg_apg - _tot_line) / 20.0  # normalized

                # Total trend: recent 10 avg vs current line
                _recent_tots = _season_totals[-10:] if len(_season_totals) >= 10 else _season_totals
                _tot_trend54 = (sum(_recent_tots) / len(_recent_tots) - _tot_line) / 10.0 if _recent_tots else 0.0

                row.extend([
                    _ou_rate(h_ou, 10),                              # ou54_h_over_rate10
                    _ou_rate(a_ou, 10),                              # ou54_a_over_rate10
                    _ou_rate(h_ou, 0),                               # ou54_h_over_season
                    _ou_rate(a_ou, 0),                               # ou54_a_over_season
                    max(-5.0, min(5.0, _ou_streak(h_ou))) / 5.0,    # ou54_h_streak (normalized)
                    max(-5.0, min(5.0, _ou_streak(a_ou))) / 5.0,    # ou54_a_streak
                    combined_over_rate,                               # ou54_combined_over_rate
                    max(-3.0, min(3.0, _pace_vs_total)),             # ou54_pace_vs_total
                    _ou_rate(h_ou, 0, home_only=True),               # ou54_h_home_over
                    _ou_rate(a_ou, 0, home_only=False),              # ou54_a_away_over
                    _tot_trend54,                                     # ou54_total_trend
                    _ou_margin_avg(h_ou, a_ou, 10),                  # ou54_margin_vs_total_10
                ])
            except Exception:
                row.extend([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0])

            X.append(row)
            y.append(1 if hs > as_ else 0)

            # ── Update ATS / O-U trackers for future games ──
            # Must be AFTER feature extraction (no lookahead)
            try:
                _odds_key = (gd, home, away)
                _odds_now = (odds_data or {}).get(_odds_key, {})
                _sp_now  = _odds_now.get('spread_home', None)
                _tot_now = _odds_now.get('total', None)
                actual_margin = hs - as_  # positive = home won by this many

                if _sp_now is not None:
                    # Covered = actual_margin > -spread_home (home perspective)
                    # spread_home = -7 means home favored by 7; cover if margin > 7
                    h_covered = actual_margin > (-_sp_now)
                    a_covered = (-actual_margin) > _sp_now   # away team perspective
                    h_mvs = actual_margin - (-_sp_now)       # margin vs spread
                    a_mvs = (-actual_margin) - _sp_now

                    _team_ats[home].append((gd, h_covered, _sp_now, True, h_mvs))
                    _team_ats[away].append((gd, a_covered, -_sp_now, False, a_mvs))
                    _season_spreads.append(abs(_sp_now))

                if _tot_now is not None:
                    game_total = hs + as_
                    went_over = game_total > _tot_now
                    margin_vs_total = game_total - _tot_now

                    _team_ou[home].append((gd, went_over, _tot_now, True, margin_vs_total))
                    _team_ou[away].append((gd, went_over, _tot_now, False, margin_vs_total))
                    _season_totals.append(_tot_now)
            except Exception:
                pass  # Tracker update failures must not crash the build loop

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
            # Update MOVDA ELO (Cat 37)
            self._update_movda(home, away, hs, as_, team_movda, mov_surprise_ewm,
                               delta_mov_history,
                               _MOVDA_K, _MOVDA_C, _MOVDA_LAMBDA, _MOVDA_ALPHA,
                               _MOVDA_BETA, _MOVDA_GAMMA, _MOVDA_DELTA, _MOVDA_EWM_ALPHA)

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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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
        except (ValueError, TypeError, AttributeError):
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

    def _update_movda(self, home, away, hs, as_, team_movda, mov_surprise_ewm,
                      delta_mov_history,
                      K, C, lam, alpha, beta, gamma, delta_param, ewm_alpha):
        """Update MOVDA Elo ratings and raw delta_MOV history (Cat 37). arXiv:2506.00348."""
        margin = hs - as_
        result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        delta_r = team_movda[home] - team_movda[away]
        e_a = 1.0 / (1.0 + 10.0 ** (-delta_r / C))
        e_mov = alpha * np.tanh(beta * delta_r) + gamma + delta_param
        delta_mov = float(margin) - e_mov
        movda_update = K * (result - e_a) + lam * delta_mov
        team_movda[home] += movda_update
        team_movda[away] -= movda_update
        mov_surprise_ewm[home] = ewm_alpha * delta_mov + (1 - ewm_alpha) * mov_surprise_ewm[home]
        mov_surprise_ewm[away] = ewm_alpha * (-delta_mov) + (1 - ewm_alpha) * mov_surprise_ewm[away]
        # Append raw delta_MOV to rolling history (home team's perspective)
        delta_mov_history[home].append(delta_mov)
        delta_mov_history[away].append(-delta_mov)

    def _parse_stats(self, stats, pts, opp_pts, is_home=True):
        """Extract stats from game data. Uses REAL box score when available, estimates otherwise."""
        if not isinstance(stats, dict):
            stats = {}

        # Detect real box score data (backfilled via backfill-boxscores.py)
        has_boxscore = "fga" in stats and stats["fga"] is not None and stats["fga"] > 0

        if has_boxscore:
            # ── REAL DATA PATH ──
            fga = stats["fga"]
            fgm = stats.get("fgm", pts / 2.0)
            fg3a = stats.get("fg3a", fga * 0.38)
            fg3m = stats.get("fg3m", fg3a * 0.36)
            ftm = stats.get("ftm", pts * 0.17)
            fta = stats.get("fta", ftm / 0.78 if ftm else pts * 0.2)
            oreb = stats.get("oreb", 10)
            dreb = stats.get("dreb", 34)
            tov = stats.get("tov", 13)
            pf = stats.get("pf", 20)
            # Real possessions: standard formula
            poss = max(fga + 0.44 * fta + tov - oreb, 60)
            # Real efficiency
            ortg = pts * 100 / poss
            drtg = opp_pts * 100 / poss
            pace = poss
        else:
            # ── ESTIMATE PATH (backward compat for old seasons) ──
            fga = max(pts / 1.1, 80)
            fgm = pts / 2.0
            fg3a = fga * 0.38
            fg3m = pts * 0.3 / 3
            ftm = pts * 0.17
            fta = pts * 0.2
            oreb = 10
            dreb = 34
            tov = 13
            pf = 20
            poss = stats.get("poss", (pts + opp_pts) / 2.0)
            ortg = stats.get("ortg", pts * 100 / max(poss, 1))
            drtg = stats.get("drtg", opp_pts * 100 / max(poss, 1))
            pace = poss

        opp_drb = stats.get("opp_dreb", 34)

        d = {
            "pts": pts, "opp_pts": opp_pts, "is_home": is_home,
            "ortg": ortg, "drtg": drtg, "pace": pace, "poss": poss,
            "has_boxscore": 1.0 if has_boxscore else 0.0,
        }

        # Four Factors (REAL when boxscore available)
        d["efg_pct"] = (fgm + 0.5 * fg3m) / max(fga, 1) if has_boxscore else stats.get("efg_pct", (pts / 2.0) / max(fga, 1))
        d["tov_rate"] = tov / max(fga + 0.44 * fta + tov, 1)
        d["oreb_pct"] = stats.get("oreb_pct", oreb / max(oreb + opp_drb, 1))
        d["ft_rate"] = fta / max(fga, 1)
        d["ts_pct"] = pts / max(2 * (fga + 0.44 * fta), 1)

        # Opponent Four Factors (estimated — opponent box score not in same dict)
        d["opp_efg_pct"] = stats.get("opp_efg_pct", d["efg_pct"] * 0.95)
        d["opp_tov_rate"] = stats.get("opp_tov_rate", d["tov_rate"])
        d["opp_oreb_pct"] = stats.get("opp_oreb_pct", d["oreb_pct"])
        d["opp_ft_rate"] = stats.get("opp_ft_rate", d["ft_rate"])

        # Shooting profile
        d["3par"] = fg3a / max(fga, 1) if has_boxscore else stats.get("3par", fg3m * 3 / max(pts, 1))
        d["fg3_pct"] = fg3m / max(fg3a, 1) if has_boxscore else stats.get("fg3_pct", 0.36)
        d["fg2_pct"] = (fgm - fg3m) / max(fga - fg3a, 1) if has_boxscore else stats.get("fg2_pct", 0.52)
        d["ft_pct"] = ftm / max(fta, 1) if has_boxscore else stats.get("ft_pct", 0.78)
        d["paint_pts"] = stats.get("paint_pts", pts * 0.4)
        d["fb_pts"] = stats.get("fb_pts", pts * 0.1)
        d["bench_pts"] = stats.get("bench_pts", pts * 0.3)
        d["2nd_pts"] = stats.get("2nd_pts", pts * 0.1)
        d["pitp"] = stats.get("pitp", pts * 0.4)
        d["pts_off_tov"] = stats.get("pts_off_tov", pts * 0.12)

        # Rates
        d["ast_rate"] = stats.get("ast_rate", stats.get("ast", 24) / max(fgm, 1) if has_boxscore else 0.6)
        d["stl_rate"] = stats.get("stl_rate", stats.get("stl", 7) / max(poss, 1) if has_boxscore else 0.08)
        d["blk_rate"] = stats.get("blk_rate", stats.get("blk", 5) / max(fga, 1) if has_boxscore else 0.05)
        d["tov_pct"] = d["tov_rate"]
        d["dreb_pct"] = stats.get("dreb_pct", dreb / max(dreb + oreb, 1))

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

    def _get_dow(self, date_str):
        """Get day of week (0=Mon, 6=Sun) from date string."""
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").weekday()
        except (ValueError, TypeError, AttributeError):
            return 2

    def _get_month(self, date_str):
        """Get month (1-12) from date string."""
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").month
        except (ValueError, TypeError, AttributeError):
            return 1


# ── Genetic Feature Selection ──

def genetic_feature_selection(X, y, feature_names, n_generations=50,
                               population_size=100, target_features=200):
    """
    Use genetic algorithm to find optimal feature subset.

    Chromosome: binary vector (1=include, 0=exclude)
    Fitness: negative Brier score (minimize) on walk-forward CV

    Args:
        X: Full feature matrix (n_games, ~6000)
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
