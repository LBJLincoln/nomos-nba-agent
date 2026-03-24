#!/bin/bash
# Run genetic evolution in a GitHub Codespace (4 cores, 16GB RAM)
# This runs the same evolution loop as HF Spaces but with more compute.
#
# Usage:
#   gh codespace create -r LBJLincoln/nomos-nba-agent -b main --machine standardLinux
#   gh codespace ssh -- bash .devcontainer/run_evolution.sh
#
# Or from within the Codespace terminal:
#   bash .devcontainer/run_evolution.sh

set -euo pipefail

echo "=== NBA Quant Evolution — Codespace Runner ==="
echo "Cores: $(nproc) | RAM: $(free -h | awk '/Mem:/ {print $2}')"

# Install dependencies
pip install -q scikit-learn>=1.5 xgboost lightgbm catboost numpy pandas scipy supabase 2>&1 | tail -1

# Set role for this runner
export SPACE_ROLE="exploitation"

# Run the evolution loop directly (headless, no Gradio)
echo "Starting evolution (exploitation mode)..."
python3 -c "
import sys, os
sys.path.insert(0, 'hf-space')

# Import the evolution engine
from app import evolution_loop

# Run evolution (this blocks and evolves forever)
print('[CODESPACE] Starting evolution loop...')
evolution_loop()
"
