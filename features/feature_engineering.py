import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.power_ratings import (
    get_team, get_rest_adjustment, get_travel_adjustment,
    get_altitude_adjustment, get_injury_adjustment
)

class FeatureEngineer:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.team_stats = {}
        self.power_cache = {}

    def update_team_stats(self, game):
        """Update rolling statistics for each team"""
        home = game['home_team']
        away = game['away_team']

        # Initialize if not exists
        if home not in self.team_stats:
            self.team_stats[home] = self._init_team_stats()
        if away not in self.team_stats:
            self.team_stats[away] = self._init_team_stats()

        # Update with game results
        self._update_stats(home, game['home_score'], game['away_score'], game['home_win'])
        self._update_stats(away, game['away_score'], game['home_score'], not game['home_win'])

    def _init_team_stats(self):
        return {
            'games': 0,
            'pts': 0,
            'opp_pts': 0,
            'wins': 0,
            'pace': 100,
            'off_rtg': 110,
            'def_rtg': 110,
            'last_10': []
        }

    def _update_stats(self, team, pts, opp_pts, win):
        stats = self.team_stats[team]
        stats['games'] += 1
        stats['pts'] += pts
        stats['opp_pts'] += opp_pts
        stats['wins'] += 1 if win else 0
        stats['last_10'].append(pts)
        if len(stats['last_10']) > 10:
            stats['last_10'].pop(0)

        # Update pace and ratings with decay
        games = stats['games']
        if games > 1:
            stats['pace'] = self.decay * stats['pace'] + (1 - self.decay) * (pts + opp_pts) / 2 * 100 / 48
            stats['off_rtg'] = self.decay * stats['off_rtg'] + (1 - self.decay) * pts * 100 / ((pts + opp_pts) / 2)
            stats['def_rtg'] = self.decay * stats['def_rtg'] + (1 - self.decay) * opp_pts * 100 / ((pts + opp_pts) / 2)

    def build_features(self, game):
        """Create comprehensive feature vector for a game"""
        home = game['home_team']
        away = game['away_team']

        # Basic stats
        home_stats = self.team_stats.get(home, self._init_team_stats())
        away_stats = self.team_stats.get(away, self._init_team_stats())

        # Power ratings
        home_pr = self._get_power_rating(home, game['date'])
        away_pr = self._get_power_rating(away, game['date'])

        # Situational factors
        rest_adj = get_rest_adjustment(game['home_rest_days'] - game['away_rest_days'])
        travel_adj = get_travel_adjustment(
            game['away_city'], game['home_city']
        )
        altitude_adj = get_altitude_adjustment(
            game['home_city'], game['away_city']
        )

        # Recent performance
        home_10_avg = np.mean(home_stats['last_10']) if home_stats['last_10'] else home_stats['pts']
        away_10_avg = np.mean(away_stats['last_10']) if away_stats['last_10'] else away_stats['pts']

        # Feature vector
        features = [
            # Power ratings
            home_pr['offensive'], home_pr['defensive'], home_pr['overall'],
            away_pr['offensive'], away_pr['defensive'], away_pr['overall'],

            # Recent performance
            home_stats['off_rtg'], home_stats['def_rtg'], home_10_avg,
            away_stats['off_rtg'], away_stats['def_rtg'], away_10_avg,

            # Situational
            rest_adj, travel_adj, altitude_adj,

            # Meta
            home_stats['wins'] / max(home_stats['games'], 1),
            away_stats['wins'] / max(away_stats['games'], 1)
        ]

        return np.array(features)

    def _get_power_rating(self, team, date):
        """Get cached or calculate power rating"""
        if team not in self.power_cache:
            self.power_cache[team] = get_team(team)
        return self.power_cache[team]

    def get_feature_names(self):
        """Return feature names for interpretation"""
        return [
            'home_off_rtg', 'home_def_rtg', 'home_overall',
            'away_off_rtg', 'away_def_rtg', 'away_overall',
            'home_recent_off', 'home_recent_def', 'home_recent_score',
            'away_recent_off', 'away_recent_def', 'away_recent_score',
            'rest_adj', 'travel_adj', 'altitude_adj',
            'home_win_pct', 'away_win_pct'
        ]
