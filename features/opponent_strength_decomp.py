import numpy as np
from typing import Dict, Optional
from datetime import datetime

def compute_opponent_strength_decomp(team: str, date: datetime, decay: float = 0.95) -> Dict:
    """
    Decompose opponent strength into offensive and defensive components
    Returns offensive and defensive ratings adjusted for strength of schedule
    """
    # Get opponent's recent offensive and defensive efficiency
    # This would typically come from a cached stats database
    # For now, we'll use placeholder logic
    
    # Opponent's offensive efficiency (points per 100 possessions)
    opp_off_eff = 110.0  # Placeholder
    
    # Opponent's defensive efficiency (points allowed per 100 possessions)
    opp_def_eff = 105.0  # Placeholder
    
    # League average efficiency
    league_avg = 108.0
    
    # Strength of schedule adjustment
    sos_off = (opp_off_eff - league_avg) / league_avg
    sos_def = (league_avg - opp_def_eff) / league_avg
    
    return {
        'opp_off_eff': opp_off_eff,
        'opp_def_eff': opp_def_eff,
        'sos_off_adj': 1 + sos_off,
        'sos_def_adj': 1 + sos_def,
        'off_rtg': opp_off_eff * (1 + sos_off),
        'def_rtg': opp_def_eff * (1 + sos_def)
    }
