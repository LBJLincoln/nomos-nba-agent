#!/usr/bin/env python3
"""
=============================================================================
 PLATT SCALING TRAINER — NBA Model V11
 Lance ce script une fois que nba_history.json contient >= 50 matchs
 avec le champ "outcome" rempli (1 = home gagne, 0 = away gagne).
=============================================================================
 Usage :
   1. Après chaque match, ouvre nba_history.json et remplis "outcome"
   2. Une fois >= 50 entrées remplies, lance :  python train_platt.py
   3. Copie les valeurs A et B affichées dans la fonction calibrate() du V11
=============================================================================
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

HISTORY_FILE   = "nba_history.json"
MIN_SAMPLES    = 50

def train_platt():
    if not os.path.exists(HISTORY_FILE):
        print(f"[ERROR] {HISTORY_FILE} introuvable")
        return

    with open(HISTORY_FILE, "r") as f:
        raw = json.load(f)

    data = [m for m in raw if m.get("outcome") is not None]
    print(f"[INFO] {len(data)} matchs avec outcome / {len(raw)} total")

    if len(data) < MIN_SAMPLES:
        print(f"[WAIT] Pas assez de données ({len(data)}/{MIN_SAMPLES})")
        return

    X = np.array([m["predicted_win_rate"] for m in data]).reshape(-1, 1)
    y = np.array([m["outcome"] for m in data])

    model = LogisticRegression()
    model.fit(X, y)

    A = model.coef_[0][0]
    B = model.intercept_[0]

    # Métriques de calibration
    y_proba  = model.predict_proba(X)[:, 1]
    brier    = brier_score_loss(y, y_proba)
    logloss  = log_loss(y, y_proba)
    accuracy = (model.predict(X) == y).mean()

    print(f"\n{'='*50}")
    print(f"  PLATT SCALING — RÉSULTATS")
    print(f"{'='*50}")
    print(f"  Échantillon  : {len(data)} matchs")
    print(f"  Coefficient A: {A:.4f}  ({'overconfident' if A < 1 else 'underconfident'})")
    print(f"  Intercept  B : {B:.4f}")
    print(f"  Accuracy     : {accuracy:.1%}")
    print(f"  Brier Score  : {brier:.4f}  (< 0.25 = bon)")
    print(f"  Log-Loss     : {logloss:.4f}  (< 0.60 = bon)")
    print(f"{'='*50}")
    print(f"\n  → Injecter dans calibrate() du V11 :")
    print(f"     A = {A:.4f}")
    print(f"     B = {B:.4f}")

    # Sauvegarde des coefficients
    coeffs = {"A": A, "B": B, "n_samples": len(data),
               "brier": brier, "logloss": logloss, "accuracy": accuracy}
    with open("platt_coefficients.json", "w") as f:
        json.dump(coeffs, f, indent=4)
    print(f"\n  Coefficients sauvegardés → platt_coefficients.json")

if __name__ == "__main__":
    train_platt()
