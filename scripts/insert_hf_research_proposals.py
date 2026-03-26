#!/usr/bin/env python3
"""
Insert HF March 2026 research proposals into Supabase research_proposals table.
Run: python3 scripts/insert_hf_research_proposals.py
"""
import os
import sys
import json
from pathlib import Path

# Load env
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import urllib.request
import urllib.parse

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ayqviqmxifzmhphiqfmj.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "")

proposals = [
    {
        "agent_source": "research",
        "category": "model_architecture",
        "technique": "TabICLv2 Specialist Island",
        "description": "Add TabICLv2 (arXiv 2602.11139, Feb 2026 SOTA, MIT license) as 7th HF Space specialist. New SOTA tabular model beating RealTabPFN-2.5 without tuning on TabArena+TALENT. pip3 install tabicl. from tabicl import TabICLClassifier. Our 9k-game dataset is in optimal range. 10x faster than TabPFN v2. Integrate as additional ensemble member alongside XGBoost/extra_trees.",
        "expected_brier_delta": -0.006,
        "effort_hours": 6,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "feature_engineering",
        "technique": "Circadian Advantage Features (4 features)",
        "description": "Add circadian_adv_home (time zones crossed eastward by home team last 5 days), circadian_adv_away, net_circadian_delta, eastward_penalty_away (binary). Chronobiology Int 2024 study of 25,016 NBA games proves eastward jet lag disrupts performance systematically (body clock adjusts 1 hour per 24 hours). Quantifies directional travel effect vs our current binary b2b flag.",
        "expected_brier_delta": -0.003,
        "effort_hours": 6,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "feature_engineering",
        "technique": "5-Year Exponential ELO (team_elo_5y)",
        "description": "Add team_elo_5y: 5-season exponentially-decayed ELO (lambda=0.7/season). SHAP analysis in Scientific Reports 2025 NBA paper (He & Choi, Hanyang University, PMC12357926) identified team_elo_5_y as the #1 most predictive feature. Formula: elo_5y(t) = sum(elo(season_t-i) * 0.7^i for i=0..4). Captures dynasty momentum over multiple seasons.",
        "expected_brier_delta": -0.004,
        "effort_hours": 5,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "calibration",
        "technique": "Venn-Abers Post-Hoc Calibration (fix broken isotonic)",
        "description": "URGENT: Current calibration HURTS our model (XGBoost 0.2206 raw vs 0.2240 calibrated = +0.003 damage). Replace with Venn-Abers (arXiv 2502.05676, ICML 2025, van der Laan & Alaa). pip install venn-abers. Finite-sample validity guarantees. Outputs [p0,p1] interval - width as bet-sizing signal. Apply multicalibration per subgroup: b2b games, high-altitude arenas, end-of-season.",
        "expected_brier_delta": -0.004,
        "effort_hours": 8,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "feature_engineering",
        "technique": "LLM-FE Claude-Driven Feature Discovery Loop",
        "description": "Implement LLM-FE (arXiv 2503.14434, ICLR 2026 Workshop). Workflow: Claude receives top-20 SHAP features + validation Brier, proposes 5 NBA-specific Python feature transformations, evaluate each on validation Brier, keep improvements, repeat 10 iterations. Claude has basketball domain knowledge for semantically meaningful features (opponent-adjusted eFG%, etc). Code: github.com/nikhilsab/LLMFE",
        "expected_brier_delta": -0.007,
        "effort_hours": 12,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "model_architecture",
        "technique": "TabPFN-2.5 as Ensemble Member",
        "description": "Add Prior-Labs/tabpfn_2_5 (#1 trending HF tabular model 2026-03-24, 31.6k downloads) as additional ensemble member. pip install tabpfn. 100% win rate vs default XGBoost on our dataset size (9k games, 94 features). Use subsample <=8000 training rows. Blend with extra_trees+xgboost. Note: non-commercial license - use for research/HF Spaces development only.",
        "expected_brier_delta": -0.005,
        "effort_hours": 4,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "market_microstructure",
        "technique": "Line Movement CLV Feature",
        "description": "Add consensus_line_move = opening_book_spread minus closing_book_spread as feature AND bet-sizing signal. Polymarket research shows both books and prediction markets are weakest in 40-60% probability zone (accuracy drops below 70%). This is our primary alpha zone. Add market_prob_spread = abs(our_prob - book_implied_prob). Use as Kelly multiplier: higher divergence = higher edge claim.",
        "expected_brier_delta": -0.002,
        "effort_hours": 3,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "feature_engineering",
        "technique": "L-RAPM Lineup Differential Features",
        "description": "Add lineup RAPM differential features from arXiv 2601.15000 (Jan 2026): projected_lineup_ortg_diff (home minus away projected starting lineup offensive rating), projected_lineup_drtg_diff, lineup_uncertainty (variance across plausible lineups), key_player_out_flag (binary if player with RAPM >+2.0 is questionable/out). Data: nba_api + Basketball-Reference injury reports. Most valuable for games with recent injuries.",
        "expected_brier_delta": -0.004,
        "effort_hours": 10,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "model_architecture",
        "technique": "MOVDA Elo Rating Replacement",
        "description": "Replace elo_diff/elo_home/elo_away with MOVDA ratings (arXiv 2506.00348). Uses MOV deviation from expected to update ratings: expected_MOV = k*tanh(alpha*rating_diff + home_adv). Tested on 13,619 NBA games 2013-2023. Reduces Brier by 1.54% vs TrueSkill, 0.66% vs standard ELO. Converges 13.5% faster. ~100 lines Python via scipy.optimize for tanh fitting.",
        "expected_brier_delta": -0.003,
        "effort_hours": 6,
        "status": "proposed"
    },
    {
        "agent_source": "research",
        "category": "model_architecture",
        "technique": "Bi-Objective NSGA-II Feature Selection",
        "description": "Convert single-objective GA (minimize Brier) to bi-objective NSGA-II (minimize Brier + minimize feature count) using pymoo. AF-NSGA-II: sparse initialization (60% initial pop <=40 features), adaptive crossover (geometric if Hamming >0.5). Prevents correlated noise features. Based on MDPI Electronics 2026. Deploy on S15 wide search space. Expected to find Pareto-optimal subsets with better generalization.",
        "expected_brier_delta": -0.004,
        "effort_hours": 12,
        "status": "proposed"
    }
]

def insert_proposals():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    url = f"{SUPABASE_URL}/rest/v1/research_proposals"

    inserted = 0
    skipped = 0

    for p in proposals:
        data = json.dumps(p).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                if status in (200, 201):
                    inserted += 1
                    print(f"[OK] Inserted: {p['technique'][:60]}")
                else:
                    print(f"[WARN] Status {status} for: {p['technique'][:60]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if "duplicate" in body.lower() or "unique" in body.lower():
                skipped += 1
                print(f"[SKIP] Duplicate: {p['technique'][:60]}")
            else:
                print(f"[ERROR] HTTP {e.code}: {body} — {p['technique'][:60]}")
        except Exception as ex:
            print(f"[ERROR] {ex} — {p['technique'][:60]}")

    print(f"\nDone: {inserted} inserted, {skipped} skipped")
    return inserted

if __name__ == "__main__":
    if not SUPABASE_KEY:
        print("ERROR: SUPABASE_API_KEY not set")
        sys.exit(1)
    print(f"Inserting {len(proposals)} proposals into Supabase...")
    print(f"URL: {SUPABASE_URL}/rest/v1/research_proposals")
    inserted = insert_proposals()
    sys.exit(0 if inserted > 0 else 1)
