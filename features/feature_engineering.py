import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

def calculate_rest_days(game_date: datetime, team_schedule: List[Dict]) -> int:
    """Calculate rest days since last game for a team"""
    last_game_date = None
    for game in team_schedule:
        if game['date'] < game_date:
            last_game_date = game['date']
    return (game_date - last_game_date).days if last_game_date else 999

def calculate_travel_distance(from_city: str, to_city: str) -> float:
    """Calculate travel distance between cities"""
    # Simple placeholder - replace with actual distance calculation
    return 0.0

def calculate_home_court_advantage(team_abbr: str, home_abbr: str) -> float:
    """Calculate home court advantage factor"""
    return 1.0 if team_abbr == home_abbr else 0.0

def calculate_injury_impact(injuries: List[str]) -> float:
    """Calculate injury impact score"""
    return len(injuries) * 0.05

def calculate_recent_performance(team_abbr: str, recent_games: List[Dict], window: int = 5) -> float:
    """Calculate recent performance metric"""
    if len(recent_games) < window:
        return 0.0
    recent_stats = recent_games[-window:]
    return np.mean([game['pts'] for game in recent_stats])

def calculate_matchup_history(home_team: str, away_team: str, history: List[Dict]) -> float:
    """Calculate historical matchup advantage"""
    head_to_head = [game for game in history if 
                   (game['home_team'] == home_team and game['away_team'] == away_team) or
                   (game['home_team'] == away_team and game['away_team'] == home_team)]
    if not head_to_head:
        return 0.0
    home_wins = sum(1 for game in head_to_head if game['home_team'] == home_team and game['home_score'] > game['away_score'])
    return home_wins / len(head_to_head)
