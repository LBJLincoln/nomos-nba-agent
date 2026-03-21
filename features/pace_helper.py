import numpy as np
from typing import Dict, List

def compute_pace_adjusted(stats: Dict, team_pace: float, league_pace: float = 100.0) -> Dict:
    """
    Adjust offensive/defensive stats by team pace.
    Formula: pace_adjusted = (raw_stat / team_pace) * league_pace
    """
    adjusted = {}
    for key, value in stats.items():
        if isinstance(value, (int, float)) and team_pace > 0:
            adjusted[f"pace_adj_{key}"] = (value / team_pace) * league_pace
        else:
            adjusted[f"pace_adj_{key}"] = value
    return adjusted

def compute_opponent_strength_decomp(games: List[Dict], team_id: str, 
                                     opponent_power: Dict[str, float]) -> Dict:
    """
    Decompose performance into vs strong/weak opponents.
    Strong: opponent_power >= team_power + 3.0
    Weak: opponent_power <= team_power - 3.0
    """
    team_power = opponent_power.get(team_id, 0)
    strong_games = []
    weak_games = []
    neutral_games = []

    for game in games:
        opp_id = game['opponent_id']
        opp_power = opponent_power.get(opp_id, 0)
        if opp_power >= team_power + 3.0:
            strong_games.append(game)
        elif opp_power <= team_power - 3.0:
            weak_games.append(game)
        else:
            neutral_games.append(game)

    def calc_stats(game_list, label):
        if not game_list:
            return {f"vs_{label}_fgp": np.nan, f"vs_{label}_tpp": np.nan,
                    f"vs_{label}_ftp": np.nan, f"vs_{label}_pts": np.nan}
        fgp = np.mean([g['fgp'] for g in game_list])
        tpp = np.mean([g['tpp'] for g in game_list])
        ftp = np.mean([g['ftp'] for g in game_list])
        pts = np.mean([g['pts'] for g in game_list])
        return {f"vs_{label}_fgp": fgp, f"vs_{label}_tpp": tpp,
                f"vs_{label}_ftp": ftp, f"vs_{label}_pts": pts}

    return {**calc_stats(strong_games, "strong"),
            **calc_stats(weak_games, "weak"),
            **calc_stats(neutral_games, "neutral")}
