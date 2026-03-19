#!/bin/bash
# Push the kernel to Kaggle for execution
# Requires: pip install kaggle, KAGGLE_USERNAME and KAGGLE_KEY in env
cd "$(dirname "$0")"
kaggle kernels push -p .
echo "Kernel pushed. Check status: kaggle kernels status lbjlincoln26/nba-quant-gpu-runner"
