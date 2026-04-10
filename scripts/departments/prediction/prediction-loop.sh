#!/bin/bash
# Department: PREDICTION — Karpathy Loop
# Pattern: evaluate → measure Brier → count features → check daily accuracy → output
# Metric: brier_score, feature_count, daily_accuracy, eval_count
# Max runtime: 5 minutes
set -uo pipefail

DEPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$(dirname "$DEPT_DIR")")")"

OUTPUT_DIR="$ROOT/data/departments/prediction"
OUTPUT_FILE="$OUTPUT_DIR/karpathy-output.json"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ITER_FILE="$OUTPUT_DIR/.iteration"
ITERATION=$(cat "$ITER_FILE" 2>/dev/null || echo 0)
ITERATION=$((ITERATION + 1))
echo "$ITERATION" > "$ITER_FILE"

echo "[PREDICTION] Starting Karpathy loop at $TIMESTAMP (iter=$ITERATION)" >&2

# ── 1. Measure Brier from evaluation files ──────────────────────────────────
BRIER_STATS=$(python3 - "$ROOT" <<'PYEOF'
import json, sys, os
from pathlib import Path

root = Path(sys.argv[1])
eval_dir = root / "data" / "eval"
result = {"best_brier": None, "latest_brier": None, "eval_count": 0, "evals": []}

if not eval_dir.exists():
    print(json.dumps(result))
    sys.exit(0)

evals = sorted(eval_dir.glob("eval-*.json"), key=lambda f: f.name, reverse=True)
result["eval_count"] = len(evals)

for ef in evals[:20]:
    try:
        d = json.loads(ef.read_text())
        brier = d.get("brier", d.get("brier_score", d.get("metrics", {}).get("brier")))
        if brier is not None:
            result["evals"].append({"file": ef.name, "brier": round(brier, 5)})
            if result["best_brier"] is None or brier < result["best_brier"]:
                result["best_brier"] = round(brier, 5)
            if result["latest_brier"] is None:
                result["latest_brier"] = round(brier, 5)
    except Exception:
        pass

print(json.dumps(result))
PYEOF
)

# ── 2. Count features ───────────────────────────────────────────────────────
FEATURE_COUNT=$(python3 - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
result = {"npz_files": 0, "engine_exists": False, "feature_categories": 0}

features_dir = root / "data" / "features"
if features_dir.exists():
    result["npz_files"] = len(list(features_dir.glob("*.npz")))

for eng_path in [root / "features" / "engine.py", root / "hf-space" / "features" / "engine.py"]:
    if eng_path.exists():
        result["engine_exists"] = True
        text = eng_path.read_text()
        import re
        cats = re.findall(r'^\s*\d+\.', text, re.MULTILINE)
        result["feature_categories"] = len(cats)
        break

print(json.dumps(result))
PYEOF
)

# ── 3. Check daily accuracy (latest picks vs results) ───────────────────────
ACCURACY_STATS=$(python3 - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
result = {"games_evaluated": 0, "correct": 0, "accuracy": None}

picks_file = root / "data" / "latest-picks.json"
if not picks_file.exists():
    # Try mon-ipad path
    picks_file = Path("/home/termius/mon-ipad/data/nba-agent/latest-picks.json")

if picks_file.exists():
    try:
        data = json.loads(picks_file.read_text())
        games = data.get("games", [])
        for g in games:
            actual = g.get("result", g.get("actual_winner"))
            pred = g.get("prediction", g.get("predicted_winner"))
            if actual and pred:
                result["games_evaluated"] += 1
                if actual == pred:
                    result["correct"] += 1
        if result["games_evaluated"] > 0:
            result["accuracy"] = round(result["correct"] / result["games_evaluated"], 4)
    except Exception:
        pass

print(json.dumps(result))
PYEOF
)

# ── 4. Check evolution state ────────────────────────────────────────────────
EVO_STATE=$(python3 - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
evo_dir = root / "data" / "evolution-state"
result = {"generation": 0, "population_size": 0, "best_fitness": None}

if evo_dir.exists():
    state_files = sorted(evo_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if state_files:
        try:
            d = json.loads(state_files[0].read_text())
            result["generation"] = d.get("generation", 0)
            result["population_size"] = d.get("population_size", len(d.get("population", [])))
            result["best_fitness"] = d.get("best_fitness", d.get("best_brier"))
        except Exception:
            pass

print(json.dumps(result))
PYEOF
)

# ── 5. Determine health ────────────────────────────────────────────────────
BEST_BRIER=$(echo "$BRIER_STATS" | python3 -c "import json,sys; v=json.load(sys.stdin).get('best_brier'); print(v if v else 'null')" 2>/dev/null || echo "null")
EVAL_COUNT=$(echo "$BRIER_STATS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('eval_count',0))" 2>/dev/null || echo 0)

STATUS="active"
if [ "$EVAL_COUNT" -eq 0 ] 2>/dev/null; then
    STATUS="warning"
fi

echo "[PREDICTION] Brier=$BEST_BRIER, evals=$EVAL_COUNT, status=$STATUS" >&2

# ── 6. Write output ────────────────────────────────────────────────────────
cat > "$OUTPUT_FILE" <<EOF
{
  "department": "prediction",
  "timestamp": "$TIMESTAMP",
  "iteration": $ITERATION,
  "status": "$STATUS",
  "brier": $BRIER_STATS,
  "features": $FEATURE_COUNT,
  "accuracy": $ACCURACY_STATS,
  "evolution": $EVO_STATE,
  "best_brier": $BEST_BRIER
}
EOF

echo "[PREDICTION] Karpathy loop complete — $OUTPUT_FILE" >&2

# Console output for guardian
python3 - "$OUTPUT_FILE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({
    "status": d["status"],
    "department": d["department"],
    "metric": "brier_score",
    "best_brier": d.get("best_brier"),
    "eval_count": d.get("brier", {}).get("eval_count", 0),
    "feature_categories": d.get("features", {}).get("feature_categories", 0),
}))
PYEOF
