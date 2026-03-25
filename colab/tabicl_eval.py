"""
TabICLv2 Evaluation for NBA Prediction — Run on Google Colab T4
================================================================
Expected: -0.005 to -0.008 Brier improvement over tree-based models.
TabICLv2 (MIT, soda-inria) beats TabPFN-2.5 on TabArena/TALENT.

Usage on Colab:
    !pip install tabicl scikit-learn numpy pandas requests
    %run tabicl_eval.py
"""

import numpy as np
import json
import requests
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

# ── Step 1: Fetch evolved feature set from S10 ──
S10_URL = "https://lbjlincoln-nomos-nba-quant.hf.space"

def get_s10_best():
    """Get the best individual's feature set from S10."""
    resp = requests.get(f"{S10_URL}/api/results", timeout=30)
    data = resp.json()
    best = data["best"]
    print(f"S10 best: brier={best['brier']}, model={best['model_type']}, features={best['n_features']}")
    return best["selected_features"]

def get_training_data():
    """Fetch feature matrix from S10's cached data."""
    resp = requests.get(f"{S10_URL}/api/training_data", timeout=60)
    if resp.status_code != 200:
        print(f"Training data endpoint returned {resp.status_code}")
        print("Fallback: you need to upload X.npy and y.npy to Colab")
        return None, None, None
    data = resp.json()
    X = np.array(data["X"])
    y = np.array(data["y"])
    feature_names = data.get("feature_names", [])
    return X, y, feature_names


# ── Step 2: Evaluate TabICLv2 ──
def evaluate_tabicl(X, y, feature_names=None, n_splits=5):
    """Walk-forward evaluation of TabICLv2 vs baseline tree models."""
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("Install TabICLv2: pip install tabicl")
        return

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    import xgboost as xgb

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    models = {
        "tabicl": lambda: TabICLClassifier(),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "xgboost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.02, random_state=42, n_jobs=-1, tree_method="hist"),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {name: [] for name in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model_fn in models.items():
            try:
                model = model_fn()
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                brier = brier_score_loss(y_test, probs)
                results[name].append(brier)
                print(f"  Fold {fold+1}/{n_splits} | {name}: brier={brier:.5f}")
            except Exception as e:
                print(f"  Fold {fold+1}/{n_splits} | {name}: FAILED ({e})")
                results[name].append(0.30)

    print("\n" + "="*60)
    print("RESULTS (walk-forward, lower = better)")
    print("="*60)
    for name, briers in sorted(results.items(), key=lambda x: np.mean(x[1])):
        mean_b = np.mean(briers)
        std_b = np.std(briers)
        print(f"  {name:20s}: {mean_b:.5f} +/- {std_b:.5f}")

    # Delta vs best tree
    best_tree = min(np.mean(results["random_forest"]), np.mean(results["extra_trees"]), np.mean(results["xgboost"]))
    tabicl_mean = np.mean(results["tabicl"])
    delta = tabicl_mean - best_tree
    print(f"\nTabICLv2 vs best tree: {delta:+.5f} ({'BETTER' if delta < 0 else 'WORSE'})")

    return results


# ── Step 3: Stacking meta-learner ──
def evaluate_stacking_with_tabicl(X, y, n_splits=5):
    """Use TabICLv2 as meta-learner on top of tree model predictions."""
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("Install TabICLv2: pip install tabicl")
        return

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    import xgboost as xgb

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    base_models = {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "et": ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "xgb": xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.02, random_state=42, n_jobs=-1, tree_method="hist"),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    stack_briers = []
    base_briers = {name: [] for name in base_models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Split training into base-train and meta-train (80/20 chronological)
        split_pt = int(len(train_idx) * 0.8)
        base_train_idx = train_idx[:split_pt]
        meta_train_idx = train_idx[split_pt:]

        # Train base models and get meta-features
        meta_train_features = []
        meta_test_features = []

        for name, model in base_models.items():
            from sklearn.base import clone
            m = clone(model)
            m.fit(X[base_train_idx], y[base_train_idx])

            # Meta-features: base model predictions
            meta_train_features.append(m.predict_proba(X[meta_train_idx])[:, 1])
            meta_test_features.append(m.predict_proba(X_test)[:, 1])

            # Track base model performance
            test_probs = m.predict_proba(X_test)[:, 1]
            base_briers[name].append(brier_score_loss(y_test, test_probs))

        # Stack: meta-features = [rf_prob, et_prob, xgb_prob]
        X_meta_train = np.column_stack(meta_train_features)
        X_meta_test = np.column_stack(meta_test_features)
        y_meta_train = y[meta_train_idx]

        # TabICLv2 as meta-learner
        try:
            meta_model = TabICLClassifier()
            meta_model.fit(X_meta_train, y_meta_train)
            stack_probs = meta_model.predict_proba(X_meta_test)[:, 1]
            stack_brier = brier_score_loss(y_test, stack_probs)
            stack_briers.append(stack_brier)
            print(f"  Fold {fold+1}: stacking={stack_brier:.5f} | rf={base_briers['rf'][-1]:.5f} et={base_briers['et'][-1]:.5f} xgb={base_briers['xgb'][-1]:.5f}")
        except Exception as e:
            print(f"  Fold {fold+1}: stacking FAILED ({e})")
            stack_briers.append(0.30)

    print("\n" + "="*60)
    print("STACKING RESULTS")
    print("="*60)
    print(f"  TabICLv2 stacking: {np.mean(stack_briers):.5f}")
    for name, briers in base_briers.items():
        print(f"  {name} alone:       {np.mean(briers):.5f}")
    best_base = min(np.mean(v) for v in base_briers.values())
    delta = np.mean(stack_briers) - best_base
    print(f"\n  Stacking vs best base: {delta:+.5f} ({'BETTER' if delta < 0 else 'WORSE'})")

    return stack_briers


if __name__ == "__main__":
    print("="*60)
    print("TabICLv2 NBA Evaluation")
    print("="*60)

    # Try to get data from S10
    features = get_s10_best()
    X, y, feature_names = get_training_data()

    if X is not None:
        print(f"\nData: {X.shape[0]} games x {X.shape[1]} features")
        print("\n--- Direct comparison ---")
        evaluate_tabicl(X, y, feature_names)
        print("\n--- Stacking with TabICLv2 meta-learner ---")
        evaluate_stacking_with_tabicl(X, y)
    else:
        print("\nUpload X.npy and y.npy manually, then run:")
        print("  X = np.load('X.npy'); y = np.load('y.npy')")
        print("  evaluate_tabicl(X, y)")
