#!/usr/bin/env python3
"""Deploy NBA Quant AI to lbjlincoln/nomos-nba-quant HF Space.

Uploads all files from hf-space/ dir, configures secrets, restarts.

Usage:
    source .env.local
    python3 hf-space/deploy.py
"""

import os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

SPACE_ID = "lbjlincoln/nomos-nba-quant"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_2")
LOCAL_DIR = Path(__file__).parent

SECRETS = {
    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
    "SUPABASE_API_KEY": os.environ.get("SUPABASE_API_KEY", ""),
    "ODDS_API_KEY": os.environ.get("ODDS_API_KEY", ""),
    "VM_CALLBACK_URL": os.environ.get("VM_CALLBACK_URL", "http://34.136.180.66:8080"),
}


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: source .env.local")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    print(f"Deploying NBA Quant AI to {SPACE_ID}...")

    operations = []
    skip = {"__pycache__", ".pyc", "node_modules", ".git", "deploy.py"}

    for fp in LOCAL_DIR.rglob("*"):
        if fp.is_dir():
            continue
        if any(s in str(fp) for s in skip):
            continue
        rel = fp.relative_to(LOCAL_DIR)
        print(f"  + {rel}")
        operations.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(fp)))

    if not operations:
        print("ERROR: No files found")
        sys.exit(1)

    print(f"\nUploading {len(operations)} files...")
    try:
        api.create_commit(
            repo_id=SPACE_ID, repo_type="space", operations=operations,
            commit_message="Deploy: RunLogger + auto-cut + 2058 features (25 categories)",
        )
        print("Upload OK!")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"Space not found, creating {SPACE_ID}...")
            api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker",
                           space_hardware="cpu-basic", private=False)
            api.create_commit(
                repo_id=SPACE_ID, repo_type="space", operations=operations,
                commit_message="Deploy: RunLogger + auto-cut + 2058 features (25 categories)",
            )
        else:
            raise

    print("\nConfiguring secrets...")
    for key, value in SECRETS.items():
        if value:
            try:
                api.add_space_secret(SPACE_ID, key, value)
                print(f"  Set {key}")
            except Exception as e:
                print(f"  WARN: {key}: {e}")
        else:
            print(f"  SKIP {key} (empty)")

    print("\nRestarting Space...")
    try:
        api.restart_space(SPACE_ID)
    except Exception as e:
        print(f"  Restart: {e}")

    print(f"\nDone! Space: https://lbjlincoln-nomos-nba-quant.hf.space")
    print(f"Monitor: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
