#!/usr/bin/env python3
"""
=============================================================================
 PERFORMANCE REPORT — NBA Model V11
 Génère un rapport complet sur les prédictions historiques :
   - ROI / Profit & Loss
   - Brier Score & Log-Loss évolutifs
   - Courbe de calibration (prédit vs réel)
   - Taux de couverture par tranche de Win Rate
=============================================================================
 Usage : python performance_report.py
=============================================================================
"""

import json, os, math
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics import log_loss, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] sklearn requis → pip install scikit-learn")

HISTORY_FILE = "nba_history.json"
BANKROLL     = 100.0


def load_completed(path: str = HISTORY_FILE) -> list:
    if not os.path.exists(path):
        print(f"[ERROR] {path} introuvable")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return [m for m in data if m.get("outcome") is not None]


def compute_roi(entries: list) -> dict:
    total_staked = 0.0
    total_return = 0.0
    wins = 0

    for e in entries:
        stake = e.get("kelly_units", 1.0) or 1.0
        odds  = e.get("odds", 0) or 0
        outcome = e["outcome"]

        if odds <= 0:
            continue

        total_staked += stake
        if outcome == 1:
            total_return += stake * odds
            wins += 1

    profit = total_return - total_staked
    roi    = (profit / total_staked * 100) if total_staked > 0 else 0

    return {
        "n":             len(entries),
        "wins":          wins,
        "win_rate_real": wins / len(entries) if entries else 0,
        "total_staked":  round(total_staked, 2),
        "total_return":  round(total_return, 2),
        "profit":        round(profit, 2),
        "roi_pct":       round(roi, 2),
    }


def compute_calibration_buckets(entries: list, n_buckets: int = 5) -> list:
    """Découpe les prédictions en tranches et compare prédit vs réel."""
    buckets = defaultdict(list)
    step    = 1.0 / n_buckets

    for e in entries:
        p    = e["predicted_win_rate"]
        idx  = min(int(p / step), n_buckets - 1)
        buckets[idx].append(e["outcome"])

    result = []
    for i in range(n_buckets):
        lo   = i * step
        hi   = lo + step
        vals = buckets[i]
        if vals:
            result.append({
                "range":        f"{lo:.0%}–{hi:.0%}",
                "n":            len(vals),
                "predicted_avg": round((lo + hi) / 2, 3),
                "real_win_rate": round(sum(vals) / len(vals), 3),
            })
    return result


def rolling_brier(entries: list, window: int = 20) -> list:
    """Brier Score glissant sur les `window` derniers matchs."""
    scores = []
    for i in range(window, len(entries) + 1):
        chunk = entries[i - window:i]
        preds = [e["predicted_win_rate"] for e in chunk]
        outs  = [e["outcome"] for e in chunk]
        bs    = sum((p - o) ** 2 for p, o in zip(preds, outs)) / len(chunk)
        scores.append({"match_idx": i, "brier": round(bs, 4)})
    return scores


def print_report(entries: list):
    n = len(entries)
    if n == 0:
        print("[INFO] Aucun match complété dans l'historique")
        return

    roi     = compute_roi(entries)
    cal     = compute_calibration_buckets(entries)
    preds   = [e["predicted_win_rate"] for e in entries]
    outs    = [e["outcome"] for e in entries]
    brier   = sum((p - o)**2 for p, o in zip(preds, outs)) / n
    logloss = -sum(o * math.log(max(p, 1e-9)) + (1-o) * math.log(max(1-p, 1e-9))
                   for p, o in zip(preds, outs)) / n

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE REPORT — NBA Model V11")
    print(f"  {n} matchs complétés")
    print(f"{'='*60}")

    print(f"\n📈 ROI & BANKROLL")
    print(f"  Matchs      : {roi['n']}")
    print(f"  Win Rate    : {roi['win_rate_real']:.1%}  (prédit moy: {sum(preds)/n:.1%})")
    print(f"  Misé        : {roi['total_staked']:.1f}u")
    print(f"  Retour      : {roi['total_return']:.1f}u")
    print(f"  Profit      : {roi['profit']:+.1f}u")
    print(f"  ROI         : {roi['roi_pct']:+.1f}%")

    print(f"\n📐 MÉTRIQUES DE CALIBRATION")
    bs_label = "Exceptionnel" if brier < 0.15 else ("Bon" if brier < 0.22 else "À améliorer")
    ll_label = "Bon" if logloss < 0.60 else "Overconfident"
    print(f"  Brier Score : {brier:.4f}  → {bs_label}")
    print(f"  Log-Loss    : {logloss:.4f}  → {ll_label}")

    print(f"\n📊 COURBE DE CALIBRATION (Prédit vs Réel)")
    print(f"  {'Tranche':<12} {'N':>4}  {'Prédit':>8}  {'Réel':>8}  {'Écart':>8}")
    print(f"  {'─'*48}")
    for b in cal:
        gap   = b["real_win_rate"] - b["predicted_avg"]
        arrow = "🔴 overconf" if gap < -0.05 else ("🟢 OK" if abs(gap) <= 0.05 else "🔵 underconf")
        print(f"  {b['range']:<12} {b['n']:>4}  {b['predicted_avg']:>7.1%}  "
              f"{b['real_win_rate']:>7.1%}  {gap:>+7.1%}  {arrow}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    entries = load_completed()
    print_report(entries)
    rb = rolling_brier(entries)
    if rb:
        last = rb[-1]
        print(f"  Brier Score (L20) : {last['brier']:.4f} au match #{last['match_idx']}")
