# Research Proposal: Shot-Chart Spatial Embeddings for NBA Feature Engine
**Date:** 2026-04-02  
**Status:** PROPOSED  
**Expected Brier improvement:** -0.010 to -0.020  
**Priority:** HIGH (highest single-feature-class improvement in 2025-2026 literature)
**Source:** MDPI *Information* Jan 2026 — "Uncertainty-Aware Machine Learning for NBA Forecasting"

---

## Finding

The MDPI 2026 paper achieved **Brier 0.199, Accuracy 0.688** — the best published result on public NBA data — using shot-chart spatial embeddings as the primary novel feature class. An ablation confirmed removing shot-chart features consistently increased Brier score. The published ceiling for pre-game-only public-data NBA models is **~0.199**, and we are currently at **0.22182** — shot-chart embeddings represent the single largest actionable gap.

---

## Data Source

NBA Stats API (free, no key required):
- `https://stats.nba.com/stats/shotchartdetail` — per-player shot locations (x, y, zone, made/missed)
- `https://stats.nba.com/stats/leaguedashteamstats` — team-level shooting by zone

The shot chart uses the standard **14-zone breakdown**:
- Restricted area, paint (non-RA), mid-range (left/center/right), corner 3 (left/right), above-break 3 (left/center/right), backcourt

---

## Feature Engineering

### Step 1: Fetch rolling team shot-zone stats (add to `ops/fetch_nba_data.py`)

```python
import requests
import pandas as pd

SHOT_ZONES = [
    'Restricted Area', 'In The Paint (Non-RA)',
    'Mid-Range', 'Left Corner 3', 'Right Corner 3',
    'Above the Break 3', 'Backcourt'
]

def fetch_team_shot_zones(season='2025-26', last_n_games=10):
    """Fetch team shooting efficiency by zone (trailing N games)."""
    url = 'https://stats.nba.com/stats/leaguedashteamstats'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nba.com'}
    params = {
        'Season': season,
        'SeasonType': 'Regular Season',
        'PerMode': 'PerGame',
        'LastNGames': last_n_games,
        'MeasureType': 'Base',
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    data = r.json()['resultSets'][0]
    return pd.DataFrame(data['rowSet'], columns=data['headers'])
```

### Step 2: Build 28-feature zone embedding per team per game

```python
def build_shot_zone_embedding(team_id, games_df, n_zones=14):
    """
    Returns a 28-feature vector: [zone_i_fga_rate, zone_i_fg_pct] for 14 zones.
    Normalized: fga_rate = zone_fga / total_fga (sums to 1.0).
    """
    zone_stats = games_df[games_df['TEAM_ID'] == team_id]
    features = []
    total_fga = zone_stats['FGA'].sum() + 1e-6  # avoid div by zero
    for zone in SHOT_ZONES:
        zone_rows = zone_stats[zone_stats['SHOT_ZONE_BASIC'] == zone]
        fga = zone_rows['FGA'].sum()
        fgm = zone_rows['FGM'].sum()
        fga_rate = fga / total_fga
        fg_pct = fgm / (fga + 1e-6)
        features += [fga_rate, fg_pct]
    return features  # 28 floats
```

### Step 3: Compute offensive/defensive MATCHUP embedding

```python
def shot_zone_matchup_features(home_team_id, away_team_id, games_df):
    """
    Key insight: combine HOME offensive zone efficiency vs AWAY defensive zone efficiency.
    This captures style-clash mismatches (e.g., corner-3-heavy team vs zone-defense-heavy team).
    """
    home_off = build_shot_zone_embedding(home_team_id, games_df)  # 28 features
    away_off = build_shot_zone_embedding(away_team_id, games_df)  # 28 features
    # Zone advantage = home_fg_pct - away_allowed_fg_pct per zone (14 deltas)
    zone_advantage = [home_off[2*i+1] - away_off[2*i+1] for i in range(14)]  # 14 features
    return home_off + away_off + zone_advantage  # 70 features total
```

---

## Integration into `features/engine.py`

Add as **Category 50: SHOT ZONE SPATIAL EMBEDDINGS** (~70 features):

```python
# In engine.py, add new category block:
# Category 50: Shot Zone Spatial Embeddings
# 70 features: home_team_zone(28) + away_team_zone(28) + matchup_advantage(14)
for team_prefix, team_id in [('home', home_id), ('away', away_id)]:
    shot_vec = get_shot_zone_embedding(team_id, shot_data, n_games=10)
    for i, val in enumerate(shot_vec):
        zone_idx = i // 2
        feat_type = 'fga_rate' if i % 2 == 0 else 'fg_pct'
        candidates.append((f'shot_zone_{team_prefix}_z{zone_idx:02d}_{feat_type}', val))
# Matchup advantage deltas
matchup = shot_zone_matchup_features(home_id, away_id, shot_data)
for i, val in enumerate(matchup[-14:]):
    candidates.append((f'shot_zone_matchup_z{i:02d}_advantage', val))
```

---

## Expected Impact

| Feature Set | Published Brier | Our Current |
|-------------|-----------------|-------------|
| Box-score only | ~0.228 | — |
| + Rolling ELO + rest | ~0.224 | 0.22182 |
| + Shot-chart embeddings | **~0.210-0.215** | target |
| + Isotonic calibration | **~0.199-0.205** | goal |

Brier improvement estimate: **-0.010 to -0.020** from shot-chart features alone.

---

## Implementation Steps

1. Add `fetch_team_shot_zones()` to `ops/fetch_nba_data.py` (or `features/expansion.py`)
2. Cache shot zone data in `data/shot_zones/` — update each day (zone stats shift slowly)
3. Add Category 50 to `features/engine.py` — ~70 new feature candidates
4. GA will naturally select/reject zone features over next evolution cycles
5. Monitor: if best island improves Brier within 50 generations, keep; else prune

## Cross-Project Note

A similar spatial embedding approach could work for political prediction — district-level congressional vote geography could be embedded as a 50-state binary vector × sector alignment score.
