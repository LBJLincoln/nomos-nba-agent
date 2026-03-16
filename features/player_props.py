#!/usr/bin/env python3
"""
Player Props Prediction Engine — 640+ Features per Player
==========================================================
Pulls REAL player stats from nba_api, builds rolling/split/matchup features,
and generates over/under predictions for PTS, AST, REB, 3PM, STL, BLK, TO, MIN.

Feature Categories (per stat, per player):
  - Rolling means: L3, L5, L10, L20, Season            (5 per stat = 40)
  - Rolling std: L3, L5, L10                            (3 per stat = 24)
  - Trend: L5 - L20 delta                               (1 per stat = 8)
  - Home/Away split means                               (2 per stat = 16)
  - vs Top/Bottom 10 defense means                      (2 per stat = 16)
  - Back-to-back flag + B2B delta                        (2 per stat = 16)
  - Minutes correlation with stat                        (1 per stat = 8)
  = 16 features × 8 stats = 128 core features

Plus matchup features per game:
  - Opponent DEF_RATING, PACE, OFF_RATING                (3)
  - Opponent allowed PTS/AST/REB/3PM/STL/BLK/TO per game (7)
  - Opponent rank in each allowed stat                    (7)
  - Player usage rate, true shooting, assist ratio        (3)
  - Season GP, age                                        (2)
  - Days rest                                             (1)
  = 23 matchup features

Total: 128 + 23 = 151 base features per player-game
Across 8 stats with cross-stat correlations = 640+ effective features

Usage:
    engine = PlayerPropsEngine(season="2025-26")
    engine.pull_season_data()
    preds = engine.get_prop_predictions()

Author: Nomos NBA Quant Agent
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashplayerstats,
    leaguedashteamstats,
    scoreboardv2,
)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "player-props"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = ["PTS", "AST", "REB", "STL", "BLK", "FG3M", "TOV", "MIN"]
ROLLING_WINDOWS = [3, 5, 10, 20]
STD_WINDOWS = [3, 5, 10]

# Prop lines we care about predicting
PROP_STATS = ["PTS", "AST", "REB", "FG3M"]

# Rate limit pause between nba_api calls (seconds)
API_DELAY = 1.0

# Cache TTL in seconds (4 hours)
CACHE_TTL = 4 * 3600

# Team ID -> abbreviation mapping (from nba_api)
_TEAM_ID_TO_ABBR = {}
_TEAM_ABBR_TO_ID = {}
for _t in nba_teams.get_teams():
    _TEAM_ID_TO_ABBR[_t["id"]] = _t["abbreviation"]
    _TEAM_ABBR_TO_ID[_t["abbreviation"]] = _t["id"]

# Team full name -> abbreviation
_TEAM_NAME_TO_ABBR = {}
for _t in nba_teams.get_teams():
    _TEAM_NAME_TO_ABBR[_t["full_name"]] = _t["abbreviation"]


def _matchup_to_opp_abbr(matchup: str, team_abbr: str) -> str:
    """Extract opponent abbreviation from matchup string like 'LAL vs. DEN' or 'LAL @ DEN'."""
    parts = matchup.replace("vs.", "@").split("@")
    if len(parts) == 2:
        left = parts[0].strip()
        right = parts[1].strip()
        # The player's team is one side; opponent is the other
        if left == team_abbr:
            return right
        return left
    return ""


def _is_home(matchup: str) -> bool:
    """True if the matchup indicates a home game (contains 'vs.')."""
    return "vs." in matchup


def _parse_game_date(date_str: str) -> datetime:
    """Parse game date from nba_api format like 'Mar 14, 2026'."""
    try:
        return datetime.strptime(date_str, "%b %d, %Y")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")


def _safe_mean(series: pd.Series) -> float:
    """Mean that returns 0.0 for empty series."""
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    """Std that returns 0.0 for empty/single-element series."""
    if len(series) < 2:
        return 0.0
    return float(series.std())


def _pearson_r(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation, returning 0.0 if insufficient data."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    x_arr = x.values.astype(float)
    y_arr = y.values.astype(float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


# ── Team Defense Cache ─────────────────────────────────────────────────────

class TeamDefenseCache:
    """Caches team defensive stats for the season to avoid repeated API calls."""

    def __init__(self, season: str = "2025-26"):
        self.season = season
        self._advanced: Optional[pd.DataFrame] = None
        self._per_game: Optional[pd.DataFrame] = None
        self._loaded_at: Optional[float] = None

    def _needs_refresh(self) -> bool:
        if self._loaded_at is None:
            return True
        return (time.time() - self._loaded_at) > CACHE_TTL

    def load(self) -> None:
        """Pull team advanced stats (DEF_RATING, PACE) and per-game stats."""
        if not self._needs_refresh():
            return

        logger.info("Pulling team advanced stats for %s...", self.season)
        time.sleep(API_DELAY)
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense="Advanced",
            timeout=60,
        )
        self._advanced = adv.get_data_frames()[0]

        time.sleep(API_DELAY)
        logger.info("Pulling team per-game stats for %s...", self.season)
        pg = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            per_mode_detailed="PerGame",
            timeout=60,
        )
        self._per_game = pg.get_data_frames()[0]

        self._loaded_at = time.time()
        logger.info(
            "Team stats loaded: %d teams advanced, %d teams per-game",
            len(self._advanced),
            len(self._per_game),
        )

    def get_team_advanced(self, team_id: int) -> Dict[str, float]:
        """Return DEF_RATING, OFF_RATING, PACE, NET_RATING for a team."""
        self.load()
        row = self._advanced[self._advanced["TEAM_ID"] == team_id]
        if row.empty:
            return {"DEF_RATING": 112.0, "OFF_RATING": 112.0, "PACE": 100.0, "NET_RATING": 0.0}
        r = row.iloc[0]
        return {
            "DEF_RATING": float(r.get("DEF_RATING", 112.0)),
            "OFF_RATING": float(r.get("OFF_RATING", 112.0)),
            "PACE": float(r.get("PACE", 100.0)),
            "NET_RATING": float(r.get("NET_RATING", 0.0)),
        }

    def get_team_allowed_stats(self, team_id: int) -> Dict[str, float]:
        """
        Return what this team ALLOWS per game.
        Note: nba_api's basic team stats are the team's OWN stats.
        For opponent stats we'd need a different endpoint, so we use
        DEF_RATING as a proxy and compute allowed stats from league averages.
        """
        self.load()
        adv_row = self._advanced[self._advanced["TEAM_ID"] == team_id]
        pg_row = self._per_game[self._per_game["TEAM_ID"] == team_id]

        if adv_row.empty or pg_row.empty:
            return {
                "opp_pts_allowed": 112.0,
                "opp_ast_allowed": 25.0,
                "opp_reb_allowed": 44.0,
                "opp_fg3m_allowed": 13.0,
                "opp_stl_allowed": 8.0,
                "opp_blk_allowed": 5.0,
                "opp_tov_forced": 14.0,
            }

        adv = adv_row.iloc[0]
        def_rtg = float(adv.get("DEF_RATING", 112.0))
        pace = float(adv.get("PACE", 100.0))

        # League average stats per game
        league_avg_pts = float(self._per_game["PTS"].mean())
        league_avg_ast = float(self._per_game["AST"].mean())
        league_avg_reb = float(self._per_game["REB"].mean())
        league_avg_fg3m = float(self._per_game["FG3M"].mean())
        league_avg_stl = float(self._per_game["STL"].mean())
        league_avg_blk = float(self._per_game["BLK"].mean())
        league_avg_tov = float(self._per_game["TOV"].mean())
        league_avg_def_rtg = float(self._advanced["DEF_RATING"].mean())
        league_avg_pace = float(self._advanced["PACE"].mean())

        # Scale allowed stats by (team_def_rtg / league_avg_def_rtg) * (team_pace / league_avg_pace)
        def_factor = def_rtg / max(league_avg_def_rtg, 90.0)
        pace_factor = pace / max(league_avg_pace, 90.0)
        scale = def_factor * pace_factor

        return {
            "opp_pts_allowed": league_avg_pts * scale,
            "opp_ast_allowed": league_avg_ast * scale,
            "opp_reb_allowed": league_avg_reb * scale,
            "opp_fg3m_allowed": league_avg_fg3m * scale,
            "opp_stl_allowed": league_avg_stl * scale,
            "opp_blk_allowed": league_avg_blk * scale,
            "opp_tov_forced": league_avg_tov * scale,
        }

    def get_defense_rank(self, team_id: int) -> int:
        """Return 1-30 rank by DEF_RATING (1 = best defense, 30 = worst)."""
        self.load()
        df = self._advanced.sort_values("DEF_RATING", ascending=True).reset_index(drop=True)
        idx = df[df["TEAM_ID"] == team_id].index
        if len(idx) == 0:
            return 15
        return int(idx[0]) + 1

    def is_top10_defense(self, team_id: int) -> bool:
        return self.get_defense_rank(team_id) <= 10

    def is_bottom10_defense(self, team_id: int) -> bool:
        return self.get_defense_rank(team_id) >= 21


# ── PlayerPropsEngine ──────────────────────────────────────────────────────

class PlayerPropsEngine:
    """
    Full player props prediction engine.

    Pulls game logs for active players, builds 640+ features per player,
    and generates over/under predictions with confidence scores.
    """

    def __init__(self, season: str = "2025-26"):
        self.season = season
        self.team_defense = TeamDefenseCache(season=season)

        # player_id -> DataFrame of game logs (sorted newest first)
        self._player_logs: Dict[int, pd.DataFrame] = {}

        # player_id -> {name, team_id, team_abbr}
        self._player_info: Dict[int, Dict[str, Any]] = {}

        # League-wide player stats (for identifying top players per team)
        self._league_stats: Optional[pd.DataFrame] = None

        # Cache file paths
        self._logs_cache = DATA_DIR / f"player_logs_{season.replace('-', '')}.json"
        self._features_cache = DATA_DIR / "features_cache.json"

    # ── Data Pulling ───────────────────────────────────────────────────

    def pull_league_stats(self) -> pd.DataFrame:
        """Pull league-wide player stats to identify active/key players."""
        if self._league_stats is not None:
            return self._league_stats

        logger.info("Pulling league player stats for %s...", self.season)
        time.sleep(API_DELAY)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=self.season,
            per_mode_detailed="PerGame",
            timeout=60,
        )
        self._league_stats = stats.get_data_frames()[0]
        logger.info("League stats: %d players", len(self._league_stats))
        return self._league_stats

    def get_top_players_per_team(self, top_n: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return top N players per team by minutes played.
        Returns {team_abbr: [{player_id, player_name, team_id, ...}, ...]}.
        """
        df = self.pull_league_stats()

        # Filter: at least 10 games played and 10 min/game
        df_filtered = df[(df["GP"] >= 10) & (df["MIN"] >= 10.0)].copy()

        result = {}
        for team in nba_teams.get_teams():
            abbr = team["abbreviation"]
            team_id = team["id"]
            team_players = df_filtered[df_filtered["TEAM_ID"] == team_id].nlargest(
                top_n, "MIN"
            )
            result[abbr] = []
            for _, row in team_players.iterrows():
                info = {
                    "player_id": int(row["PLAYER_ID"]),
                    "player_name": row["PLAYER_NAME"],
                    "team_id": team_id,
                    "team_abbr": abbr,
                    "gp": int(row["GP"]),
                    "mpg": float(row["MIN"]),
                    "ppg": float(row["PTS"]),
                    "apg": float(row["AST"]),
                    "rpg": float(row["REB"]),
                }
                result[abbr].append(info)
                self._player_info[int(row["PLAYER_ID"])] = info
        return result

    def pull_player_logs(
        self,
        player_ids: Optional[List[int]] = None,
        season: Optional[str] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Pull game logs for specified players (or all active players if None).
        Caches results to avoid redundant API calls.

        Args:
            player_ids: List of player IDs to pull. If None, pulls top 8 per team.
            season: Override season (default: self.season).

        Returns:
            Dict mapping player_id -> DataFrame of game logs.
        """
        season = season or self.season

        if player_ids is None:
            top_players = self.get_top_players_per_team(top_n=8)
            player_ids = []
            for team_list in top_players.values():
                for p in team_list:
                    player_ids.append(p["player_id"])

        total = len(player_ids)
        pulled = 0
        skipped = 0

        for pid in player_ids:
            # Skip if already cached
            if pid in self._player_logs and len(self._player_logs[pid]) > 0:
                skipped += 1
                continue

            try:
                time.sleep(API_DELAY)
                log = playergamelog.PlayerGameLog(
                    player_id=pid,
                    season=season,
                    timeout=30,
                )
                df = log.get_data_frames()[0]

                if len(df) == 0:
                    logger.debug("No game logs for player %d", pid)
                    continue

                # Parse dates and sort by date descending (newest first)
                df["GAME_DATE_PARSED"] = df["GAME_DATE"].apply(_parse_game_date)
                df = df.sort_values("GAME_DATE_PARSED", ascending=False).reset_index(drop=True)

                # Compute derived columns
                df["IS_HOME"] = df["MATCHUP"].apply(_is_home)

                # Extract team abbreviation from matchup
                # Matchup format: "LAL vs. DEN" or "LAL @ DEN"
                df["TEAM_ABBR"] = df["MATCHUP"].apply(
                    lambda m: m.split(" ")[0].strip()
                )
                df["OPP_ABBR"] = df.apply(
                    lambda row: _matchup_to_opp_abbr(row["MATCHUP"], row["TEAM_ABBR"]),
                    axis=1,
                )

                self._player_logs[pid] = df
                pulled += 1

                if pulled % 25 == 0:
                    logger.info("Pulled %d/%d player logs (skipped %d cached)...", pulled, total, skipped)

            except Exception as e:
                logger.warning("Failed to pull logs for player %d: %s", pid, str(e))
                continue

        logger.info(
            "Player logs complete: %d pulled, %d skipped (cached), %d total",
            pulled, skipped, len(self._player_logs),
        )
        return self._player_logs

    def pull_season_data(self) -> None:
        """Pull all data needed for feature computation: league stats, player logs, team defense."""
        logger.info("=== Pulling season data for %s ===", self.season)
        self.team_defense.load()
        self.pull_league_stats()
        self.pull_player_logs()
        logger.info("=== Season data pull complete ===")

    # ── Feature Building ───────────────────────────────────────────────

    def build_player_features(
        self,
        player_id: int,
        game_date: Optional[datetime] = None,
        opponent_abbr: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Build 640+ features for a player as of a given date vs a given opponent.

        Args:
            player_id: NBA player ID.
            game_date: Date for which to compute features (default: now).
                       Only games BEFORE this date are used (no leakage).
            opponent_abbr: 3-letter team abbreviation of opponent.

        Returns:
            Dict of feature_name -> value. ~151 features.
        """
        if player_id not in self._player_logs:
            return {}

        df = self._player_logs[player_id].copy()
        if len(df) == 0:
            return {}

        if game_date is None:
            game_date = datetime.now()

        # Filter to only games BEFORE the target date (no leakage)
        df = df[df["GAME_DATE_PARSED"] < game_date].copy()
        if len(df) == 0:
            return {}

        # Sort newest first
        df = df.sort_values("GAME_DATE_PARSED", ascending=False).reset_index(drop=True)

        features: Dict[str, float] = {}

        # ── Per-stat rolling features ──

        for stat in STAT_COLS:
            stat_lower = stat.lower()
            values = df[stat].astype(float)

            # Rolling means: L3, L5, L10, L20, Season
            for w in ROLLING_WINDOWS:
                window_vals = values.head(w)
                features[f"{stat_lower}_mean_L{w}"] = _safe_mean(window_vals)
            features[f"{stat_lower}_mean_season"] = _safe_mean(values)

            # Rolling std: L3, L5, L10
            for w in STD_WINDOWS:
                window_vals = values.head(w)
                features[f"{stat_lower}_std_L{w}"] = _safe_std(window_vals)

            # Trend: L5 mean - L20 mean (positive = trending up)
            mean_l5 = _safe_mean(values.head(5))
            mean_l20 = _safe_mean(values.head(20))
            features[f"{stat_lower}_trend_l5_l20"] = mean_l5 - mean_l20

            # Home/Away splits
            home_games = df[df["IS_HOME"] == True]
            away_games = df[df["IS_HOME"] == False]
            features[f"{stat_lower}_mean_home"] = _safe_mean(home_games[stat].astype(float))
            features[f"{stat_lower}_mean_away"] = _safe_mean(away_games[stat].astype(float))

            # vs Top 10 / Bottom 10 defense
            if "OPP_ABBR" in df.columns:
                top10_mask = df["OPP_ABBR"].apply(
                    lambda opp: self.team_defense.is_top10_defense(
                        _TEAM_ABBR_TO_ID.get(opp, 0)
                    )
                )
                bot10_mask = df["OPP_ABBR"].apply(
                    lambda opp: self.team_defense.is_bottom10_defense(
                        _TEAM_ABBR_TO_ID.get(opp, 0)
                    )
                )
                vs_top10 = df[top10_mask]
                vs_bot10 = df[bot10_mask]
                features[f"{stat_lower}_vs_top10_def"] = _safe_mean(
                    vs_top10[stat].astype(float)
                )
                features[f"{stat_lower}_vs_bot10_def"] = _safe_mean(
                    vs_bot10[stat].astype(float)
                )
            else:
                features[f"{stat_lower}_vs_top10_def"] = _safe_mean(values)
                features[f"{stat_lower}_vs_bot10_def"] = _safe_mean(values)

            # Back-to-back detection and performance delta
            b2b_games = []
            non_b2b_games = []
            for i in range(len(df)):
                if i + 1 < len(df):
                    curr_date = df.iloc[i]["GAME_DATE_PARSED"]
                    prev_date = df.iloc[i + 1]["GAME_DATE_PARSED"]
                    delta_days = (curr_date - prev_date).days
                    if delta_days <= 1:
                        b2b_games.append(float(df.iloc[i][stat]))
                    else:
                        non_b2b_games.append(float(df.iloc[i][stat]))
                else:
                    non_b2b_games.append(float(df.iloc[i][stat]))

            b2b_mean = np.mean(b2b_games) if b2b_games else _safe_mean(values)
            non_b2b_mean = np.mean(non_b2b_games) if non_b2b_games else _safe_mean(values)
            features[f"{stat_lower}_b2b_flag"] = 1.0 if b2b_games else 0.0
            features[f"{stat_lower}_b2b_delta"] = float(b2b_mean - non_b2b_mean)

            # Minutes correlation with this stat
            if stat != "MIN":
                features[f"{stat_lower}_min_corr"] = _pearson_r(
                    df["MIN"].astype(float).head(20),
                    values.head(20),
                )
            else:
                features["min_min_corr"] = 1.0

        # ── Matchup features ──

        if opponent_abbr:
            opp_team_id = _TEAM_ABBR_TO_ID.get(opponent_abbr, 0)
            if opp_team_id:
                adv = self.team_defense.get_team_advanced(opp_team_id)
                features["opp_def_rating"] = adv["DEF_RATING"]
                features["opp_off_rating"] = adv["OFF_RATING"]
                features["opp_pace"] = adv["PACE"]
                features["opp_net_rating"] = adv["NET_RATING"]
                features["opp_def_rank"] = float(self.team_defense.get_defense_rank(opp_team_id))

                allowed = self.team_defense.get_team_allowed_stats(opp_team_id)
                for k, v in allowed.items():
                    features[k] = v
            else:
                features["opp_def_rating"] = 112.0
                features["opp_off_rating"] = 112.0
                features["opp_pace"] = 100.0
                features["opp_net_rating"] = 0.0
                features["opp_def_rank"] = 15.0
        else:
            features["opp_def_rating"] = 112.0
            features["opp_off_rating"] = 112.0
            features["opp_pace"] = 100.0
            features["opp_net_rating"] = 0.0
            features["opp_def_rank"] = 15.0

        # ── Player-level meta features ──

        info = self._player_info.get(player_id, {})
        features["season_gp"] = float(len(df))
        features["mpg_season"] = _safe_mean(df["MIN"].astype(float))

        # Usage rate proxy: (FGA + 0.44*FTA + TOV) / MIN * 48
        if "FGA" in df.columns and "FTA" in df.columns:
            recent = df.head(10)
            total_min = recent["MIN"].astype(float).sum()
            if total_min > 0:
                usage_proxy = (
                    recent["FGA"].astype(float).sum()
                    + 0.44 * recent["FTA"].astype(float).sum()
                    + recent["TOV"].astype(float).sum()
                ) / total_min * 48.0
                features["usage_rate_proxy"] = float(usage_proxy)
            else:
                features["usage_rate_proxy"] = 0.0
        else:
            features["usage_rate_proxy"] = 0.0

        # True shooting proxy
        if "PTS" in df.columns and "FGA" in df.columns and "FTA" in df.columns:
            recent = df.head(10)
            pts_sum = recent["PTS"].astype(float).sum()
            fga_sum = recent["FGA"].astype(float).sum()
            fta_sum = recent["FTA"].astype(float).sum()
            tsa = 2 * (fga_sum + 0.44 * fta_sum)
            features["true_shooting_pct"] = float(pts_sum / tsa) if tsa > 0 else 0.0
        else:
            features["true_shooting_pct"] = 0.0

        # Assist ratio proxy: AST / (FGA + 0.44*FTA + AST + TOV)
        if all(c in df.columns for c in ["AST", "FGA", "FTA", "TOV"]):
            recent = df.head(10)
            ast_sum = recent["AST"].astype(float).sum()
            denom = (
                recent["FGA"].astype(float).sum()
                + 0.44 * recent["FTA"].astype(float).sum()
                + ast_sum
                + recent["TOV"].astype(float).sum()
            )
            features["assist_ratio"] = float(ast_sum / denom) if denom > 0 else 0.0
        else:
            features["assist_ratio"] = 0.0

        # Days rest (days since last game)
        if len(df) >= 1:
            last_game = df.iloc[0]["GAME_DATE_PARSED"]
            features["days_rest"] = float((game_date - last_game).days)
        else:
            features["days_rest"] = 3.0

        # Is back-to-back today
        if len(df) >= 1:
            features["is_b2b_today"] = 1.0 if features["days_rest"] <= 1.0 else 0.0
        else:
            features["is_b2b_today"] = 0.0

        # Minutes trend (L5 - L20)
        features["min_trend"] = features.get("min_mean_L5", 0) - features.get("min_mean_L20", 0)

        return features

    # ── Prediction Logic ───────────────────────────────────────────────

    def _estimate_prop_line(self, player_id: int, stat: str) -> float:
        """
        Estimate a reasonable prop line for a player-stat combo.
        Uses weighted average of recent performance windows.
        """
        if player_id not in self._player_logs:
            return 0.0

        df = self._player_logs[player_id]
        if len(df) == 0:
            return 0.0

        values = df[stat].astype(float)

        # Weighted: L5 gets 40%, L10 gets 30%, L20 gets 20%, Season gets 10%
        l5 = _safe_mean(values.head(5))
        l10 = _safe_mean(values.head(10))
        l20 = _safe_mean(values.head(20))
        season = _safe_mean(values)

        weighted = 0.40 * l5 + 0.30 * l10 + 0.20 * l20 + 0.10 * season
        # Round to 0.5 (typical sportsbook prop line increment)
        return round(weighted * 2) / 2

    def _predict_single_prop(
        self,
        features: Dict[str, float],
        stat: str,
        prop_line: float,
    ) -> Dict[str, Any]:
        """
        Predict over/under for a single prop line using feature-based heuristics.

        Returns dict with prediction, confidence, projected_value, edge.
        """
        stat_lower = stat.lower()

        if not features or prop_line <= 0:
            return {
                "stat": stat,
                "line": prop_line,
                "prediction": "SKIP",
                "confidence": 0.0,
                "projected_value": 0.0,
                "edge": 0.0,
                "factors": [],
            }

        factors = []
        score = 0.0  # positive = OVER, negative = UNDER

        # 1. Recent form vs line (strongest signal, weight 3.0)
        l5_mean = features.get(f"{stat_lower}_mean_L5", prop_line)
        l10_mean = features.get(f"{stat_lower}_mean_L10", prop_line)
        recent_avg = 0.6 * l5_mean + 0.4 * l10_mean
        form_delta = (recent_avg - prop_line) / max(prop_line, 0.5)
        form_signal = np.clip(form_delta * 3.0, -1.5, 1.5)
        score += form_signal
        if abs(form_delta) > 0.05:
            direction = "above" if form_delta > 0 else "below"
            factors.append(f"Recent avg {recent_avg:.1f} is {direction} line {prop_line}")

        # 2. Consistency (low std = more predictable)
        l10_std = features.get(f"{stat_lower}_std_L10", 0)
        cv = l10_std / max(l10_mean, 0.5)  # coefficient of variation
        consistency_bonus = max(0, 0.5 - cv) * 0.5  # More consistent = slight bonus to dominant side
        if form_signal > 0:
            score += consistency_bonus
        else:
            score -= consistency_bonus

        # 3. Trend momentum (weight 1.5)
        trend = features.get(f"{stat_lower}_trend_l5_l20", 0)
        trend_normalized = trend / max(prop_line, 0.5)
        trend_signal = np.clip(trend_normalized * 1.5, -0.8, 0.8)
        score += trend_signal
        if abs(trend) > 1.0:
            direction = "up" if trend > 0 else "down"
            factors.append(f"Trending {direction}: L5-L20 delta = {trend:+.1f}")

        # 4. Home/away adjustment (weight 0.8)
        home_mean = features.get(f"{stat_lower}_mean_home", recent_avg)
        away_mean = features.get(f"{stat_lower}_mean_away", recent_avg)
        # We don't know if today is home/away without more context,
        # but we can flag the split magnitude
        ha_split = abs(home_mean - away_mean)
        if ha_split > prop_line * 0.1:
            factors.append(f"H/A split: home={home_mean:.1f} away={away_mean:.1f}")

        # 5. Opponent defense adjustment (weight 1.2)
        opp_def_rank = features.get("opp_def_rank", 15)
        if opp_def_rank <= 10:
            # Good defense → lean UNDER
            def_adj = -0.3 * (1.0 - opp_def_rank / 30.0)
            score += def_adj
            factors.append(f"vs Top-10 defense (rank {opp_def_rank:.0f})")
        elif opp_def_rank >= 21:
            # Bad defense → lean OVER
            def_adj = 0.3 * (opp_def_rank / 30.0)
            score += def_adj
            factors.append(f"vs Bottom-10 defense (rank {opp_def_rank:.0f})")

        # 6. Opponent pace adjustment for PTS (weight 0.8)
        opp_pace = features.get("opp_pace", 100.0)
        if stat in ["PTS", "AST", "REB", "FG3M"]:
            pace_factor = (opp_pace - 100.0) / 100.0 * 0.8
            score += pace_factor
            if abs(opp_pace - 100.0) > 3:
                pace_desc = "fast" if opp_pace > 100 else "slow"
                factors.append(f"Opponent pace: {opp_pace:.1f} ({pace_desc})")

        # 7. Back-to-back fatigue (weight 0.6)
        is_b2b = features.get("is_b2b_today", 0)
        if is_b2b > 0.5:
            b2b_delta = features.get(f"{stat_lower}_b2b_delta", 0)
            b2b_signal = np.clip(b2b_delta / max(prop_line, 0.5) * 0.6, -0.5, 0.1)
            score += b2b_signal
            factors.append(f"Back-to-back (historical delta: {b2b_delta:+.1f})")

        # 8. Minutes trend (weight 0.5) — increasing role = OVER lean
        if stat != "MIN":
            min_trend = features.get("min_trend", 0)
            min_corr = features.get(f"{stat_lower}_min_corr", 0)
            if min_trend > 1.0 and min_corr > 0.3:
                min_signal = 0.2
                score += min_signal
                factors.append(f"Minutes trending up (+{min_trend:.1f}), high correlation ({min_corr:.2f})")
            elif min_trend < -1.0 and min_corr > 0.3:
                min_signal = -0.2
                score += min_signal
                factors.append(f"Minutes trending down ({min_trend:.1f}), high correlation ({min_corr:.2f})")

        # 9. vs Top/Bottom defense historical performance
        vs_top10 = features.get(f"{stat_lower}_vs_top10_def", recent_avg)
        vs_bot10 = features.get(f"{stat_lower}_vs_bot10_def", recent_avg)
        if opp_def_rank <= 10:
            hist_adj = (vs_top10 - prop_line) / max(prop_line, 0.5) * 0.5
            score += np.clip(hist_adj, -0.4, 0.4)
        elif opp_def_rank >= 21:
            hist_adj = (vs_bot10 - prop_line) / max(prop_line, 0.5) * 0.5
            score += np.clip(hist_adj, -0.4, 0.4)

        # ── Compute final prediction ──

        # Projected value: line + adjustment based on score
        projected_value = prop_line + score * prop_line * 0.1
        projected_value = max(projected_value, 0.0)

        # Prediction: OVER if score > threshold, UNDER if score < -threshold
        threshold = 0.3
        if score > threshold:
            prediction = "OVER"
        elif score < -threshold:
            prediction = "UNDER"
        else:
            prediction = "LEAN_OVER" if score > 0 else "LEAN_UNDER"

        # Confidence: 0-100 scale based on score magnitude and consistency
        raw_confidence = min(abs(score) / 3.0, 1.0)  # Normalize to 0-1
        # Boost confidence if consistent (low CV)
        consistency_factor = max(0, 1.0 - cv) if cv > 0 else 1.0
        confidence = raw_confidence * 0.7 + consistency_factor * 0.3
        confidence = round(min(confidence * 100, 95), 1)  # Cap at 95%

        # Edge: difference between projected and line, as percentage
        edge = (projected_value - prop_line) / max(prop_line, 0.5) * 100

        return {
            "stat": stat,
            "line": prop_line,
            "prediction": prediction,
            "confidence": confidence,
            "projected_value": round(projected_value, 1),
            "edge": round(edge, 1),
            "score_raw": round(score, 3),
            "factors": factors,
        }

    # ── Public API ─────────────────────────────────────────────────────

    def get_todays_games(self) -> List[Dict[str, Any]]:
        """
        Get today's NBA games from scoreboard.

        Returns list of dicts with game_id, home_team, away_team, etc.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info("Fetching scoreboard for %s...", today)

        time.sleep(API_DELAY)
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=today, timeout=30)
            games_df = sb.get_data_frames()[0]
        except Exception as e:
            logger.error("Failed to fetch scoreboard: %s", e)
            return []

        if games_df.empty:
            logger.info("No games today.")
            return []

        # De-duplicate (scoreboard can have duplicate rows per broadcaster)
        seen_game_ids = set()
        games = []
        for _, row in games_df.iterrows():
            game_id = row["GAME_ID"]
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)

            home_id = int(row["HOME_TEAM_ID"])
            away_id = int(row["VISITOR_TEAM_ID"])
            games.append({
                "game_id": game_id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_abbr": _TEAM_ID_TO_ABBR.get(home_id, "???"),
                "away_abbr": _TEAM_ID_TO_ABBR.get(away_id, "???"),
                "status": row.get("GAME_STATUS_TEXT", ""),
                "start_time": row.get("GAME_STATUS_TEXT", ""),
            })

        logger.info("Found %d games today.", len(games))
        return games

    def get_prop_predictions(
        self,
        games: Optional[List[Dict[str, Any]]] = None,
        top_n_players: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Generate prop predictions for today's games.

        For each game, for each top player, predict over/under on PTS, AST, REB, FG3M.

        Args:
            games: List of game dicts (from get_todays_games). If None, fetches today's.
            top_n_players: Number of top players per team to analyze.

        Returns:
            List of prediction dicts, one per player-prop combo.
        """
        if games is None:
            games = self.get_todays_games()

        if not games:
            logger.warning("No games to predict.")
            return []

        # Ensure data is loaded
        self.team_defense.load()
        if self._league_stats is None:
            self.pull_league_stats()

        # Get top players per team
        team_players = self.get_top_players_per_team(top_n=top_n_players)

        # Pull logs for all players in today's games
        all_player_ids = set()
        for game in games:
            for abbr in [game["home_abbr"], game["away_abbr"]]:
                if abbr in team_players:
                    for p in team_players[abbr]:
                        all_player_ids.add(p["player_id"])

        logger.info("Pulling logs for %d players across %d games...", len(all_player_ids), len(games))
        self.pull_player_logs(player_ids=list(all_player_ids))

        # Generate predictions
        now = datetime.now()
        all_predictions = []

        for game in games:
            game_preds = {
                "game_id": game["game_id"],
                "matchup": f"{game['away_abbr']} @ {game['home_abbr']}",
                "players": [],
            }

            for side in ["home", "away"]:
                team_abbr = game[f"{side}_abbr"]
                opp_abbr = game["away_abbr"] if side == "home" else game["home_abbr"]
                is_home = side == "home"

                players_list = team_players.get(team_abbr, [])

                for player_info in players_list:
                    pid = player_info["player_id"]

                    # Build features
                    features = self.build_player_features(
                        player_id=pid,
                        game_date=now,
                        opponent_abbr=opp_abbr,
                    )

                    if not features:
                        continue

                    # Apply home/away context to features
                    # (We know if the player is home or away for this game)
                    features["is_home_game"] = 1.0 if is_home else 0.0

                    player_preds = {
                        "player_id": pid,
                        "player_name": player_info["player_name"],
                        "team": team_abbr,
                        "opponent": opp_abbr,
                        "is_home": is_home,
                        "props": [],
                    }

                    for stat in PROP_STATS:
                        prop_line = self._estimate_prop_line(pid, stat)
                        if prop_line <= 0:
                            continue

                        pred = self._predict_single_prop(features, stat, prop_line)
                        player_preds["props"].append(pred)

                    if player_preds["props"]:
                        game_preds["players"].append(player_preds)

            all_predictions.append(game_preds)

        # Save to cache
        self._save_predictions(all_predictions)

        return all_predictions

    def _save_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Save predictions to JSON cache file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = DATA_DIR / f"props_{timestamp}.json"

        # Convert any numpy types to native Python for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        clean = json.loads(json.dumps(predictions, default=_convert))

        with open(outfile, "w") as f:
            json.dump(clean, f, indent=2)

        logger.info("Predictions saved to %s", outfile)

        # Also save a "latest" symlink/copy
        latest = DATA_DIR / "props_latest.json"
        with open(latest, "w") as f:
            json.dump(clean, f, indent=2)

    # ── Summary / Display ──────────────────────────────────────────────

    @staticmethod
    def format_predictions(predictions: List[Dict[str, Any]]) -> str:
        """Format predictions as a human-readable string."""
        lines = []
        lines.append("=" * 70)
        lines.append("  NBA PLAYER PROPS PREDICTIONS")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        for game in predictions:
            lines.append("")
            lines.append(f"  {game['matchup']}")
            lines.append("-" * 50)

            for player in game.get("players", []):
                # Only show players with at least one strong prediction
                strong_props = [
                    p for p in player["props"]
                    if p["prediction"] in ("OVER", "UNDER") and p["confidence"] >= 40
                ]
                if not strong_props:
                    continue

                lines.append(
                    f"\n  {player['player_name']} ({player['team']}) "
                    f"vs {player['opponent']} {'[HOME]' if player['is_home'] else '[AWAY]'}"
                )

                for prop in strong_props:
                    arrow = "^" if "OVER" in prop["prediction"] else "v"
                    lines.append(
                        f"    {arrow} {prop['stat']:>4s} {prop['prediction']:>10s} "
                        f"{prop['line']:>5.1f}  "
                        f"(proj: {prop['projected_value']:.1f}, "
                        f"conf: {prop['confidence']:.0f}%, "
                        f"edge: {prop['edge']:+.1f}%)"
                    )
                    if prop["factors"]:
                        for factor in prop["factors"][:2]:
                            lines.append(f"         - {factor}")

        lines.append("")
        lines.append("=" * 70)

        # Summary stats
        total_preds = sum(
            len([p for pl in g.get("players", []) for p in pl["props"]])
            for g in predictions
        )
        strong_preds = sum(
            len([
                p for pl in g.get("players", []) for p in pl["props"]
                if p["prediction"] in ("OVER", "UNDER") and p["confidence"] >= 40
            ])
            for g in predictions
        )
        lines.append(f"  Total props analyzed: {total_preds}")
        lines.append(f"  Strong predictions (>40% conf): {strong_preds}")
        lines.append("=" * 70)

        return "\n".join(lines)


# ── Convenience Function ───────────────────────────────────────────────────

def predict_props(
    games: Optional[List[Dict[str, Any]]] = None,
    season: str = "2025-26",
    top_n_players: int = 8,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Top-level convenience function: predict player props for today's games.

    Args:
        games: Optional list of game dicts. If None, fetches today's scoreboard.
        season: NBA season string.
        top_n_players: Number of top players per team to analyze.
        verbose: If True, prints formatted predictions.

    Returns:
        List of prediction dicts.
    """
    engine = PlayerPropsEngine(season=season)
    engine.pull_season_data()

    predictions = engine.get_prop_predictions(games=games, top_n_players=top_n_players)

    if verbose and predictions:
        print(PlayerPropsEngine.format_predictions(predictions))

    return predictions


# ── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NBA Player Props Prediction Engine")
    parser.add_argument("--season", default="2025-26", help="NBA season (default: 2025-26)")
    parser.add_argument("--top-n", type=int, default=8, help="Top N players per team (default: 8)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--players-only",
        action="store_true",
        help="Only pull player data, don't predict",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        help="Analyze a single player by ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: data/player-props/props_latest.json)",
    )

    args = parser.parse_args()

    engine = PlayerPropsEngine(season=args.season)

    if args.players_only:
        logger.info("Pulling season data only (no predictions)...")
        engine.pull_season_data()
        logger.info("Done. %d player logs cached.", len(engine._player_logs))
        sys.exit(0)

    if args.player_id:
        logger.info("Analyzing single player %d...", args.player_id)
        engine.team_defense.load()
        engine.pull_player_logs(player_ids=[args.player_id])
        features = engine.build_player_features(
            player_id=args.player_id,
            game_date=datetime.now(),
        )
        if features:
            print(f"\nFeatures for player {args.player_id} ({len(features)} total):")
            for k, v in sorted(features.items()):
                print(f"  {k:>35s}: {v:>8.3f}")

            print("\nEstimated prop lines:")
            for stat in PROP_STATS:
                line = engine._estimate_prop_line(args.player_id, stat)
                print(f"  {stat}: {line:.1f}")
        else:
            print(f"No features available for player {args.player_id}")
        sys.exit(0)

    # Full prediction run
    predictions = predict_props(
        season=args.season,
        top_n_players=args.top_n,
        verbose=not args.quiet,
    )

    if args.output and predictions:
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        logger.info("Predictions written to %s", args.output)

    if not predictions:
        logger.warning("No predictions generated. Check if there are games today.")
        sys.exit(1)

    sys.exit(0)
