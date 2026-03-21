import numpy as np
import pandas as pd
from datetime import datetime
from features.helpers.pace_adjusted import compute_pace_adjusted_points

def compute_head_to_head_record(games: pd.DataFrame, team1_id: int, team2_id: int, season: str = None) -> dict:
    """
    Compute head-to-head record between two teams.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points', 'game_date', 'season']
    team1_id (int): First team ID
    team2_id (int): Second team ID
    season (str): Optional season filter (e.g., '2023-24')
    
    Returns:
    dict: Head-to-head statistics
    """
    # Filter games between the two teams
    matchup_games = games[
        ((games['team_id'] == team1_id) & (games['opponent_id'] == team2_id)) |
        ((games['team_id'] == team2_id) & (games['opponent_id'] == team1_id))
    ]
    
    if season:
        matchup_games = matchup_games[matchup_games['season'] == season]
    
    if matchup_games.empty:
        return {
            'games_played': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'team1_win_pct': 0.0,
            'team2_win_pct': 0.0,
            'team1_avg_margin': 0.0,
            'team2_avg_margin': 0.0,
            'last_game_date': None,
            'current_streak': 0,
            'team1_won_last': False
        }
    
    # Determine which team is team1 and which is team2 in each game
    results = []
    for _, game in matchup_games.iterrows():
        if game['team_id'] == team1_id:
            team1_points = game['points']
            team2_points = game['opponent_points']
            team1_won = game['points'] > game['opponent_points']
        else:
            team1_points = game['opponent_points']
            team2_points = game['points']
            team1_won = game['opponent_points'] > game['points']
        
        results.append({
            'team1_points': team1_points,
            'team2_points': team2_points,
            'team1_won': team1_won,
            'game_date': game['game_date']
        })
    
    results_df = pd.DataFrame(results)
    results_df['margin'] = results_df['team1_points'] - results_df['team2_points']
    
    # Calculate statistics
    games_played = len(results_df)
    team1_wins = results_df['team1_won'].sum()
    team2_wins = games_played - team1_wins
    team1_win_pct = team1_wins / games_played if games_played > 0 else 0.0
    team2_win_pct = team2_wins / games_played if games_played > 0 else 0.0
    
    team1_avg_margin = results_df['margin'].mean()
    team2_avg_margin = -team1_avg_margin
    
    # Current streak
    streak = 0
    last_result = results_df.iloc[-1]['team1_won']
    for i in range(len(results_df) - 1, -1, -1):
        if results_df.iloc[i]['team1_won'] == last_result:
            streak += 1
        else:
            break
    
    # Last game info
    last_game_date = results_df.iloc[-1]['game_date'] if not results_df.empty else None
    team1_won_last = results_df.iloc[-1]['team1_won'] if not results_df.empty else False
    
    return {
        'games_played': games_played,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'team1_win_pct': team1_win_pct,
        'team2_win_pct': team2_win_pct,
        'team1_avg_margin': team1_avg_margin,
        'team2_avg_margin': team2_avg_margin,
        'last_game_date': last_game_date,
        'current_streak': streak,
        'team1_won_last': team1_won_last,
        'recent_results': results_df[['team1_points', 'team2_points', 'team1_won']].tail(5).to_dict('records')
    }

def compute_season_series(games: pd.DataFrame, team1_id: int, team2_id: int, season: str) -> dict:
    """
    Compute season series between two teams.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points', 'game_date', 'season']
    team1_id (int): First team ID
    team2_id (int): Second team ID
    season (str): Season to analyze
    
    Returns:
    dict: Season series statistics
    """
    season_games = games[games['season'] == season].copy()
    
    # Get all games between these teams in the season
    series_games = season_games[
        ((season_games['team_id'] == team1_id) & (season_games['opponent_id'] == team2_id)) |
        ((season_games['team_id'] == team2_id) & (season_games['opponent_id'] == team1_id))
    ]
    
    if series_games.empty:
        return {
            'games_played': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'team1_win_pct': 0.0,
            'team2_win_pct': 0.0,
            'team1_home_wins': 0,
            'team2_home_wins': 0,
            'team1_away_wins': 0,
            'team2_away_wins': 0,
            'home_advantage': 0.0,
            'series_winner': None,
            'games_remaining': 2 - len(series_games)  # NBA teams play 2 games against division opponents
        }
    
    # Determine home/away for each game
    series_results = []
    for _, game in series_games.iterrows():
        if game['team_id'] == team1_id:
            is_home = True
            team1_points = game['points']
            team2_points = game['opponent_points']
            team1_won = game['points'] > game['opponent_points']
        else:
            is_home = False
            team1_points = game['opponent_points']
            team2_points = game['points']
            team1_won = game['opponent_points'] > game['points']
        
        series_results.append({
            'team1_points': team1_points,
            'team2_points': team2_points,
            'team1_won': team1_won,
            'is_home': is_home
        })
    
    series_df = pd.DataFrame(series_results)
    
    # Calculate statistics
    games_played = len(series_df)
    team1_wins = series_df['team1_won'].sum()
    team2_wins = games_played - team1_wins
    team1_win_pct = team1_wins / games_played
    team2_win_pct = team2_wins / games_played
    
    team1_home_wins = series_df[(series_df['team1_won']) & (series_df['is_home'])].shape[0]
    team2_home_wins = series_df[(~series_df['team1_won']) & (~series_df['is_home'])].shape[0]
    team1_away_wins = series_df[(series_df['team1_won']) & (~series_df['is_home'])].shape[0]
    team2_away_wins = series_df[(~series_df['team1_won']) & (series_df['is_home'])].shape[0]
    
    home_advantage = (team1_home_wins + team2_home_wins) / games_played - 0.5
    
    series_winner = 'team1' if team1_wins > team2_wins else 'team2' if team2_wins > team1_wins else 'tie'
    
    return {
        'games_played': games_played,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'team1_win_pct': team1_win_pct,
        'team2_win_pct': team2_win_pct,
        'team1_home_wins': team1_home_wins,
        'team2_home_wins': team2_home_wins,
        'team1_away_wins': team1_away_wins,
        'team2_away_wins': team2_away_wins,
        'home_advantage': home_advantage,
        'series_winner': series_winner,
        'games_remaining': 2 - games_played
    }

def compute_style_matchup(games: pd.DataFrame, team1_id: int, team2_id: int) -> dict:
    """
    Compute style matchup between two teams (pace vs defense).
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'pace', 'defensive_rating', 'points', 'opponent_points']
    team1_id (int): First team ID
    team2_id (int): Second team ID
    
    Returns:
    dict: Style matchup statistics
    """
    team1_games = games[games['team_id'] == team1_id].copy()
    team2_games = games[games['team_id'] == team2_id].copy()
    
    if team1_games.empty or team2_games.empty:
        return {
            'team1_pace': 100.0,
            'team2_pace': 100.0,
            'team1_defense': 110.0,
            'team2_defense': 110.0,
            'pace_differential': 0.0,
            'defense_differential': 0.0,
            'expected_pace': 100.0,
            'style_mismatch': 0.0,
            'offensive_impact': 0.0,
            'defensive_impact': 0.0
        }
    
    # Calculate team statistics
    team1_pace = team1_games['pace'].mean()
    team2_pace = team2_games['pace'].mean()
    team1_defense = team1_games['defensive_rating'].mean()
    team2_defense = team2_games['defensive_rating'].mean()
    
    # Pace differential
    pace_diff = team1_pace - team2_pace
    
    # Defense differential
    defense_diff = team1_defense - team2_defense
    
    # Expected pace (harmonic mean adjusted for team tendencies)
    expected_pace = (team1_pace + team2_pace) / 2.0
    
    # Style mismatch indicator
    pace_mismatch = abs(pace_diff) / max(team1_pace, team2_pace)
    defense_mismatch = abs(defense_diff) / max(team1_defense, team2_defense)
    style_mismatch = (pace_mismatch + defense_mismatch) / 2.0
    
    # Offensive/defensive impact
    team1_offensive_impact = team1_games['offensive_rating'].mean() - team2_defense
    team2_offensive_impact = team2_games['offensive_rating'].mean() - team1_defense
    
    team1_defensive_impact = team1_defense - team2_games['offensive_rating'].mean()
    team2_defensive_impact = team2_defense - team1_games['offensive_rating'].mean()
    
    return {
        'team1_pace': team1_pace,
        'team2_pace': team2_pace,
        'team1_defense': team1_defense,
        'team2_defense': team2_defense,
        'pace_differential': pace_diff,
        'defense_differential': defense_diff,
        'expected_pace': expected_pace,
        'style_mismatch': style_mismatch,
        'offensive_impact': (team1_offensive_impact + team2_offensive_impact) / 2.0,
        'defensive_impact': (team1_defensive_impact + team2_defensive_impact) / 2.0
    }

def compute_home_away_splits_against_opponent(games: pd.DataFrame, team_id: int, opponent_id: int) -> dict:
    """
    Compute home/away splits for a team against a specific opponent.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'home', 'points', 'opponent_points', 'game_date']
    team_id (int): Team ID to analyze
    opponent_id (int): Opponent ID
    
    Returns:
    dict: Home/away split statistics
    """
    team_games = games[games['team_id'] == team_id].copy()
    matchup_games = team_games[team_games['opponent_id'] == opponent_id]
    
    if matchup_games.empty:
        return {
            'home_games': 0,
            'away_games': 0,
            'home_wins': 0,
            'away_wins': 0,
            'home_win_pct': 0.0,
            'away_win_pct': 0.0,
            'home_avg_margin': 0.0,
            'away_avg_margin': 0.0,
            'home_advantage': 0.0,
            'last_home_game': None,
            'last_away_game': None
        }
    
    home_games = matchup_games[matchup_games['home'] == 1]
    away_games = matchup_games[matchup_games['home'] == 0]
    
    # Home game stats
    home_wins = (home_games['points'] > home_games['opponent_points']).sum()
    home_games_played = len(home_games)
    home_win_pct = home_wins / home_games_played if home_games_played > 0 else 0.0
    home_avg_margin = (home_games['points'] - home_games['opponent_points']).mean()
    
    # Away game stats
    away_wins = (away_games['points'] > away_games['opponent_points']).sum()
    away_games_played = len(away_games)
    away_win_pct = away_wins / away_games_played if away_games_played > 0 else 0.0
    away_avg_margin = (away_games['points'] - away_games['opponent_points']).mean()
    
    # Home advantage metric
    home_advantage = home_win_pct - away_win_pct
    
    # Last games
    last_home_game = home_games.sort_values('game_date', ascending=False).iloc[0] if not home_games.empty else None
    last_away_game = away_games.sort_values('game_date', ascending=False).iloc[0] if not away_games.empty else None
    
    return {
        'home_games': home_games_played,
        'away_games': away_games_played,
        'home_wins': home_wins,
        'away_wins': away_wins,
        'home_win_pct': home_win_pct,
        'away_win_pct': away_win_pct,
        'home_avg_margin': home_avg_margin,
        'away_avg_margin': away_avg_margin,
        'home_advantage': home_advantage,
        'last_home_game': last_home_game.to_dict() if last_home_game is not None else None,
        'last_away_game': last_away_game.to_dict() if last_away_game is not None else None
    }

def compute_matchup_quality_score(games: pd.DataFrame, team1_id: int, team2_id: int) -> float:
    """
    Compute a quality score for a matchup based on historical performance and style factors.
    
    Parameters:
    games (pd.DataFrame): Game data
    team1_id (int): First team ID
    team2_id (int): Second team ID
    
    Returns:
    float: Matchup quality score (0-100, higher = more competitive/interesting)
    """
    # Get basic head-to-head stats
    h2h = compute_head_to_head_record(games, team1_id, team2_id)
    
    if h2h['games_played'] == 0:
        return 50.0  # Neutral score for no history
    
    # Style matchup
    style = compute_style_matchup(games, team1_id, team2_id)
    
    # Home/away splits
    splits = compute_home_away_splits_against_opponent(games, team1_id, team2_id)
    
    # Calculate quality components
    competitiveness = min(h2h['team1_win_pct'], 1 - h2h['team1_win_pct']) * 100  # Closer to 50% = more competitive
    style_interest = (1 - style['style_mismatch']) * 50  # Higher mismatch = more interesting
    home_advantage_effect = abs(splits['home_advantage']) * 20  # Higher home advantage = more predictable
    
    # Combine with weights
    quality_score = (0.4 * competitiveness) + (0.35 * style_interest) + (0.25 * (50 - home_advantage_effect))
    
    return max(0, min(100, quality_score))

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,2,2,1,2],
#     'opponent_id': [2,2,1,1,2,1],
#     'points': [100,110,95,105,120,115],
#     'opponent_points': [95,105,100,110,115,120],
#     'home': [1,0,0,1,1,0],
#     'game_date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20', '2024-01-25']),
#     'season': ['2023-24', '2023-24', '2023-24', '2023-24', '2023-24', '2023-24'],
#     'pace': [100, 98, 102, 101, 99, 103],
#     'defensive_rating': [105, 108, 102, 104, 107, 103]
# })
# 
# h2h = compute_head_to_head_record(games, team1_id=1, team2_id=2)
# series = compute_season_series(games, team1_id=1, team2_id=2, season='2023-24')
# style = compute_style_matchup(games, team1_id=1, team2_id=2)
# splits = compute_home_away_splits_against_opponent(games, team_id=1, opponent_id=2)
# quality = compute_matchup_quality_score(games, team1_id=1, team2_id=2)
