import numpy as np
from datetime import datetime, timedelta

def compute_rest_patterns(team_games, current_date):
    """
    Compute rest-related features:
    - Days since last game
    - Back-to-back indicator
    - 3-in-4 nights indicator
    - Average rest over last 5 games
    """
    rest_days = []
    back_to_back = 0
    three_in_four = 0

    for i, game in enumerate(team_games):
        if i == 0:
            continue
        prev_game = team_games[i-1]
        days_between = (game['date'] - prev_game['date']).days

        rest_days.append(days_between)

        # Back-to-back
        if days_between <= 1:
            back_to_back += 1

        # 3-in-4 nights
        if i >= 2:
            three_game_window = team_games[i-2:i+1]
            dates = [g['date'] for g in three_game_window]
            if (max(dates) - min(dates)).days <= 3:
                three_in_four += 1

    current_rest = (current_date - team_games[-1]['date']).days if team_games else 0
    avg_rest = np.mean(rest_days[-5:]) if len(rest_days) >= 5 else np.mean(rest_days) if rest_days else 0

    return {
        'current_rest': current_rest,
        'back_to_back_rate': back_to_back / len(team_games) if team_games else 0,
        'three_in_four_rate': three_in_four / len(team_games) if team_games else 0,
        'avg_rest_last_5': avg_rest
    }
