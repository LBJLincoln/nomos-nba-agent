import numpy as np
from typing import List, Dict, Optional

def compute_advanced_stats(team_stats: Dict) -> Dict:
    """
    Compute advanced basketball statistics from raw team data
    """
    stats = {}
    
    # Offensive efficiency
    fga = team_stats.get('FGA', 1)
    stats['off_eff'] = (team_stats.get('PTS', 0) / fga) if fga > 0 else 0
    
    # Defensive efficiency
    opp_fga = team_stats.get('opp_FGA', 1)
    stats['def_eff'] = (team_stats.get('opp_PTS', 0) / opp_fga) if opp_fga > 0 else 0
    
    # True shooting percentage
    fta = team_stats.get('FTA', 1)
    stats['ts_pct'] = team_stats.get('PTS', 0) / (2 * (team_stats.get('FGA', 0) + 0.44 * fta))
    
    # Rebound rate
    oreb = team_stats.get('OREB', 1)
    stats['oreb_pct'] = oreb / (oreb + team_stats.get('opp_DREB', 0))
    
    # Turnover rate
    stats['tov_pct'] = team_stats.get('TOV', 0) / (team_stats.get('FGA', 0) + 0.44 * fta + team_stats.get('TOV', 0))
    
    # Pace estimate
    stats['pace'] = team_stats.get('FGA', 0) + team_stats.get('FTA', 0) + team_stats.get('TOV', 0) + team_stats.get('OREB', 0) + team_stats.get('opp_DREB', 0)
    
    return stats

def compute_matchup_features(home_stats: Dict, away_stats: Dict) -> Dict:
    """
    Compute differential features between home and away teams
    """
    features = {}
    
    # Efficiency differentials
    features['off_eff_diff'] = home_stats.get('off_eff', 0) - away_stats.get('off_eff', 0)
    features['def_eff_diff'] = home_stats.get('def_eff', 0) - away_stats.get('def_eff', 0)
    
    # Shooting differentials
    features['ts_pct_diff'] = home_stats.get('ts_pct', 0) - away_stats.get('ts_pct', 0)
    
    # Rebounding differential
    features['oreb_pct_diff'] = home_stats.get('oreb_pct', 0) - away_stats.get('oreb_pct', 0)
    
    # Turnover differential
    features['tov_pct_diff'] = home_stats.get('tov_pct', 0) - away_stats.get('tov_pct', 0)
    
    # Pace differential
    features['pace_diff'] = home_stats.get('pace', 0) - away_stats.get('pace', 0)
    
    # Rest differential
    features['rest_diff'] = home_stats.get('rest_days', 0) - away_stats.get('rest_days', 0)
    
    # Travel differential (in miles)
    features['travel_diff'] = home_stats.get('travel_miles', 0) - away_stats.get('travel_miles', 0)
    
    return features

def compute_rest_travel_features(team_info: Dict) -> Dict:
    """
    Compute rest and travel features for a team
    """
    features = {}
    
    # Rest days (0 = back-to-back, 1 = single day rest, etc.)
    features['rest_days'] = team_info.get('rest_days', 0)
    
    # Travel distance in miles (0 = no travel, higher = longer trips)
    features['travel_miles'] = team_info.get('travel_miles', 0)
    
    # Travel direction (1 = east, -1 = west, 0 = same region)
    features['travel_direction'] = team_info.get('travel_direction', 0)
    
    # Altitude change (in feet)
    features['altitude_change'] = team_info.get('altitude_change', 0)
    
    return features

def compute_injury_impact_features(injuries: List[Dict]) -> Dict:
    """
    Compute injury impact features
    """
    features = {
        'injury_count': 0,
        'star_injury_count': 0,
        'total_impact': 0.0
    }
    
    for injury in injuries:
        impact = injury.get('impact', 0.0)
        features['injury_count'] += 1
        features['total_impact'] += impact
        
        # Star player threshold (arbitrary but reasonable)
        if impact > 0.15:
            features['star_injury_count'] += 1
    
    return features

def compute_recent_performance_features(recent_games: List[Dict], window: int = 5) -> Dict:
    """
    Compute recent performance metrics
    """
    if len(recent_games) == 0:
        return {}
    
    # Use last 'window' games or all available
    recent = recent_games[-window:]
    
    features = {
        'recent_games': len(recent),
        'recent_win_pct': 0.0,
        'recent_pts_avg': 0.0,
        'recent_opp_pts_avg': 0.0,
        'recent_off_eff': 0.0,
        'recent_def_eff': 0.0
    }
    
    wins = sum(1 for g in recent if g.get('win', False))
    features['recent_win_pct'] = wins / len(recent)
    
    if len(recent) > 0:
        features['recent_pts_avg'] = sum(g.get('pts', 0) for g in recent) / len(recent)
        features['recent_opp_pts_avg'] = sum(g.get('opp_pts', 0) for g in recent) / len(recent)
        
        fga_sum = sum(g.get('fga', 1) for g in recent)
        features['recent_off_eff'] = sum(g.get('pts', 0) for g in recent) / fga_sum if fga_sum > 0 else 0
        
        opp_fga_sum = sum(g.get('opp_fga', 1) for g in recent)
        features['recent_def_eff'] = sum(g.get('opp_pts', 0) for g in recent) / opp_fga_sum if opp_fga_sum > 0 else 0
    
    return features
