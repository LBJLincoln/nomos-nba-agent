# Improve Features

Read `features/engine.py` — the canonical feature engine (v3.0-35cat, 6129 candidates, 36 categories).

Analyze which features are most selected by evolution (check Supabase `nba_experiments` for `selected_features`).
Identify gaps: what real-world signals are NOT captured?

If adding features:
1. Add feature NAMES in the registration block (before `self.feature_names = names`)
2. Add feature COMPUTATION in the build loop (before `X.append(row)`)
3. Verify: `python3 -c "from features.engine import NBAFeatureEngine; print(len(NBAFeatureEngine().feature_names))"`
4. Copy to hf-space: `cp features/engine.py hf-space/features/engine.py`
5. Verify parity: `sha256sum features/engine.py hf-space/features/engine.py`
6. Deploy all 6 islands with `hf-space/deploy_island.py`

NEVER run ML on VM. Feature engineering only.
