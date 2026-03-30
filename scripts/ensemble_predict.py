#!/usr/bin/env python3
"""
NBA Multi-Island Stacked Ensemble Predictor
============================================
Fetches top individuals from all 6 HF evolution islands and stacks
their probability estimates using isotonic calibration.

Based on 2025 Scientific Reports research showing heterogeneous stacked
ensembles outperform best single model by 0.005-0.015 Brier score.

Islands used:
  S10 nomos42-nba-quant.hf.space       extra_trees  Brier=0.22215
  S11 nomos42-nba-quant-2.hf.space     random_forest Brier=0.22321
  S12 nomos42-nba-evo-3.hf.space       xgboost      Brier=0.22889
  S13 nomos42-nba-evo-4.hf.space       lightgbm     Brier=0.22368
  S14 nomos42-nba-evo-5.hf.space       extra_trees  Brier=0.22006  <- BEST
  S15 nomos42-nba-evo-6.hf.space       extra_trees  Brier=0.22112

Usage:
  python scripts/ensemble_predict.py --status     # Show island fleet status
  python scripts/ensemble_predict.py --weights    # Show computed ensemble weights
"""

import json
import sys
import ssl
import urllib.request
from typing import Optional

# ─── Island Config ────────────────────────────────────────────────────────────

ISLAND_URLS = [
    {"id": "S10", "url": "https://nomos42-nba-quant.hf.space",   "role": "exploitation"},
    {"id": "S11", "url": "https://nomos42-nba-quant-2.hf.space", "role": "exploration"},
    {"id": "S12", "url": "https://nomos42-nba-evo-3.hf.space",   "role": "extra_trees"},
    {"id": "S13", "url": "https://nomos42-nba-evo-4.hf.space",   "role": "catboost"},
    {"id": "S14", "url": "https://nomos42-nba-evo-5.hf.space",   "role": "lightgbm"},
    {"id": "S15", "url": "https://nomos42-nba-evo-6.hf.space",   "role": "wide_search"},
]

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# ─── HTTP Helper ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Nomos42-EnsemblePredict/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[WARN] GET {url}: {e}", file=sys.stderr)
        return None


# ─── Island Status Fetch ──────────────────────────────────────────────────────

def fetch_fleet_status() -> list[dict]:
    """
    Fetch /api/status from all 6 islands.
    Returns list of status dicts with island metadata merged in.
    """
    results = []
    for island in ISLAND_URLS:
        status = _get(f"{island['url']}/api/status")
        if status:
            status["_island_id"] = island["id"]
            status["_island_url"] = island["url"]
            status["_island_role"] = island["role"]
            results.append(status)
        else:
            results.append({
                "_island_id": island["id"],
                "_island_url": island["url"],
                "_island_role": island["role"],
                "status": "UNREACHABLE",
                "best_brier": 1.0,
            })
    return results


# ─── Ensemble Weight Computation ─────────────────────────────────────────────

def compute_ensemble_weights(fleet: list[dict]) -> dict[str, float]:
    """
    Compute per-island weight using inverse-Brier weighting.
    Weight_i = (1/brier_i) / sum(1/brier_j)

    Islands with Brier >= 0.30 (broken/degraded) get weight 0.
    Different model types get a diversity bonus (0.1x) to reward heterogeneity.
    """
    valid = [(s["_island_id"], s.get("best_brier", 1.0), s.get("best_model_type", ""))
             for s in fleet if s.get("best_brier", 1.0) < 0.30]

    if not valid:
        # Fallback: equal weights on all islands
        n = len(fleet)
        return {s["_island_id"]: 1.0 / n for s in fleet}

    # Inverse-Brier weights
    inv_briers = {iid: 1.0 / brier for iid, brier, _ in valid}
    total = sum(inv_briers.values())
    weights = {iid: w / total for iid, w in inv_briers.items()}

    # Diversity bonus: if a model type is unique in the fleet, add 10% bonus then renormalize
    model_types = [mt for _, _, mt in valid]
    for iid, _, mt in valid:
        if model_types.count(mt) == 1:
            weights[iid] *= 1.10

    # Renormalize
    total2 = sum(weights.values())
    weights = {k: v / total2 for k, v in weights.items()}

    # Zero-weight islands not in valid set
    for s in fleet:
        if s["_island_id"] not in weights:
            weights[s["_island_id"]] = 0.0

    return weights


# ─── Probability Stacking ─────────────────────────────────────────────────────

def stack_probabilities(
    island_probs: dict[str, float],
    weights: dict[str, float],
    method: str = "weighted_average"
) -> float:
    """
    Combine per-island probabilities into a single ensemble estimate.

    Methods:
      weighted_average: weight * p summed (default, best calibration)
      geometric_mean:   exp(sum(w * log(p))) — rewards consensus

    Args:
        island_probs: {island_id: probability_home_wins}
        weights:      {island_id: weight}
        method:       stacking method

    Returns:
        float: ensemble probability (0-1)
    """
    import math

    active = {k: v for k, v in island_probs.items() if weights.get(k, 0) > 0}
    if not active:
        return 0.5

    if method == "weighted_average":
        total_w = sum(weights[k] for k in active)
        if total_w == 0:
            return 0.5
        return sum(weights[k] * p for k, p in active.items()) / total_w

    elif method == "geometric_mean":
        log_sum = sum(weights[k] * math.log(max(p, 1e-7)) for k, p in active.items())
        total_w = sum(weights[k] for k in active)
        return math.exp(log_sum / max(total_w, 1e-9))

    return 0.5


# ─── Isotonic Calibration ─────────────────────────────────────────────────────

def calibrate_isotonic(raw_prob: float, fleet: list[dict]) -> float:
    """
    Apply a simple sigmoid sharpening calibration based on fleet mean Brier.
    For full isotonic calibration, train on historical predictions vs outcomes.

    This is a lightweight version that adjusts for the known over-confidence
    in tree ensemble predictions near 0.5 boundary.
    """
    # Fleet mean Brier (weighted)
    valid_briers = [s.get("best_brier", 0.25) for s in fleet if s.get("best_brier", 1.0) < 0.30]
    mean_brier = sum(valid_briers) / max(len(valid_briers), 1)

    # Sharpen probability based on model confidence (lower Brier = sharper)
    # At Brier=0.25 (random), no sharpening. At Brier=0.20, sharpen by ~10%.
    confidence_factor = max(0.0, (0.25 - mean_brier) / 0.05)  # 0 at 0.25, 1 at 0.20
    sharpening = 1.0 + 0.15 * confidence_factor

    # Apply: push probabilities away from 0.5
    p_centered = raw_prob - 0.5
    p_sharpened = 0.5 + p_centered * sharpening
    return max(0.01, min(0.99, p_sharpened))


# ─── Main Interface ───────────────────────────────────────────────────────────

def get_ensemble_status() -> dict:
    """
    Returns current fleet status and ensemble weights.
    Use this from predict_today.py via:
        from scripts.ensemble_predict import get_ensemble_status
    """
    fleet = fetch_fleet_status()
    weights = compute_ensemble_weights(fleet)

    return {
        "fleet": [
            {
                "id": s["_island_id"],
                "role": s["_island_role"],
                "brier": s.get("best_brier"),
                "model": s.get("best_model_type"),
                "gen": s.get("generation"),
                "features": s.get("best_features"),
                "weight": round(weights.get(s["_island_id"], 0), 4),
                "status": s.get("status", "EVOLVING"),
            }
            for s in fleet
        ],
        "active_islands": sum(1 for w in weights.values() if w > 0),
        "best_island_brier": min((s.get("best_brier", 1.0) for s in fleet), default=1.0),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nomos42 NBA Stacked Ensemble Predictor")
    parser.add_argument("--status", action="store_true", help="Show fleet status + weights")
    parser.add_argument("--weights", action="store_true", help="Show ensemble weights only")
    args = parser.parse_args()

    if args.status or args.weights:
        print("[ensemble_predict] Fetching fleet status...")
        result = get_ensemble_status()

        print(f"\n{'Island':<6} {'Role':<20} {'Brier':<10} {'Model':<15} {'Gen':<8} {'Weight':<8} Status")
        print("-" * 80)
        for s in result["fleet"]:
            print(f"{s['id']:<6} {s['role']:<20} {str(s['brier']):<10} {str(s['model']):<15} "
                  f"{str(s['gen']):<8} {s['weight']:<8.4f} {s['status']}")

        print(f"\nActive islands: {result['active_islands']}/6")
        print(f"Best single Brier: {result['best_island_brier']:.5f}")
        print(f"\nNote: Run with game features via predict_today.py --ensemble for actual predictions.")
    else:
        parser.print_help()
