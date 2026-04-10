#!/bin/bash
# Kaggle GPU Evolution Manager
# Usage: ./kaggle-manage.sh [status|push-cpu|push-gpu|output|list]

KERNEL_ID="alexismoret6/nba-quant-gpu-evolution-v2"
NOTEBOOK_DIR="/tmp/kaggle-push"
NOTEBOOK_SRC="/home/lahargnedebartoli/nomos-nba-agent/colab/nba_gpu_v2_kaggle.ipynb"

case "${1:-status}" in
  status)
    echo "=== Kaggle Notebook Status ==="
    kaggle kernels status "$KERNEL_ID" 2>&1
    ;;

  push-cpu)
    echo "=== Pushing with Internet ON, GPU OFF (Phase 1: setup) ==="
    mkdir -p "$NOTEBOOK_DIR"
    cat > "$NOTEBOOK_DIR/kernel-metadata.json" << 'EOF'
{
  "id": "alexismoret6/nba-quant-gpu-v2",
  "title": "NBA Quant GPU Evolution v2",
  "code_file": "nba_gpu_v2_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
    cp "$NOTEBOOK_SRC" "$NOTEBOOK_DIR/nba_gpu_v2_kaggle.ipynb"
    cd "$NOTEBOOK_DIR" && kaggle kernels push -p . 2>&1
    ;;

  push-gpu)
    echo "=== Pushing with GPU ON, Internet OFF (Phase 2: evolution) ==="
    mkdir -p "$NOTEBOOK_DIR"
    cat > "$NOTEBOOK_DIR/kernel-metadata.json" << 'EOF'
{
  "id": "alexismoret6/nba-quant-gpu-v2",
  "title": "NBA Quant GPU Evolution v2",
  "code_file": "nba_gpu_v2_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
    cp "$NOTEBOOK_SRC" "$NOTEBOOK_DIR/nba_gpu_v2_kaggle.ipynb"
    cd "$NOTEBOOK_DIR" && kaggle kernels push -p . 2>&1
    ;;

  output)
    echo "=== Downloading Output ==="
    mkdir -p /tmp/kaggle-output
    kaggle kernels output "$KERNEL_ID" -p /tmp/kaggle-output 2>&1
    echo "Output saved to /tmp/kaggle-output/"
    ls -la /tmp/kaggle-output/
    ;;

  list)
    echo "=== My Kaggle Notebooks ==="
    kaggle kernels list --mine --sort-by dateRun 2>&1
    ;;

  *)
    echo "Usage: $0 [status|push-cpu|push-gpu|output|list]"
    ;;
esac
