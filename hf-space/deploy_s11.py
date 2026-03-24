#!/usr/bin/env python3
"""Deploy S11 as 2nd evolution island (exploration mode).

Deploys the SAME code as S10 but with SPACE_ROLE=exploration secret.
S11 uses higher mutation, wider feature search, more model diversity.

Usage:
    source .env.local
    python3 hf-space/deploy_s11.py
"""

import os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

SPACE_ID = "lbjlincoln/nomos-nba-quant-2"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_2")
LOCAL_DIR = Path(__file__).parent

# Same secrets as S10 + SPACE_ROLE override
SECRETS = {
    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
    "SUPABASE_API_KEY": os.environ.get("SUPABASE_API_KEY", ""),
    "SUPABASE_PASSWORD": os.environ.get("SUPABASE_PASSWORD", ""),
    "SUPABASE_URL_2": os.environ.get("SUPABASE_URL_2", ""),
    "SUPABASE_ANON_KEY_2": os.environ.get("SUPABASE_ANON_KEY_2", ""),
    "SUPABASE_PASSWORD_2": os.environ.get("SUPABASE_PASSWORD_2", ""),
    "SUPABASE_POOLER_2": os.environ.get("SUPABASE_POOLER_2", ""),
    "ODDS_API_KEY": os.environ.get("ODDS_API_KEY", ""),
    "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "ADMIN_TELEGRAM_ID": os.environ.get("ADMIN_TELEGRAM_ID", ""),
    # S11 differentiator: exploration mode
    "SPACE_ROLE": "exploration",
}

# Files to skip (S11 doesn't need deploy scripts or experiment_runner)
SKIP = {"__pycache__", ".pyc", "node_modules", ".git", "deploy.py", "deploy_s11.py"}


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: source .env.local")
        sys.exit(1)

    # Engine parity check
    root_engine = (LOCAL_DIR.parent / "features" / "engine.py").read_bytes()
    hf_engine = (LOCAL_DIR / "features" / "engine.py").read_bytes()
    if root_engine != hf_engine:
        print("ERROR: features/engine.py diverged! Fix parity first.")
        sys.exit(1)
    print("Engine parity check: OK")

    api = HfApi(token=HF_TOKEN)
    print(f"Deploying S11 (EXPLORATION island) to {SPACE_ID}...")

    operations = []
    for fp in LOCAL_DIR.rglob("*"):
        if fp.is_dir():
            continue
        if any(s in str(fp) for s in SKIP):
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
            commit_message="feat: S11 as exploration island (higher mut, wider features)",
        )
        print("Upload OK!")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"Space not found, creating {SPACE_ID}...")
            api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker",
                           space_hardware="cpu-basic", private=False)
            api.create_commit(
                repo_id=SPACE_ID, repo_type="space", operations=operations,
                commit_message="feat: S11 as exploration island (higher mut, wider features)",
            )
        else:
            raise

    print("\nConfiguring secrets (including SPACE_ROLE=exploration)...")
    for key, value in SECRETS.items():
        if value:
            try:
                api.add_space_secret(SPACE_ID, key, value)
                print(f"  Set {key}")
            except Exception as e:
                print(f"  WARN: {key}: {e}")

    print("\nRestarting Space...")
    try:
        api.restart_space(SPACE_ID)
    except Exception as e:
        print(f"  Restart: {e}")

    print(f"\nDone! S11 (exploration): https://lbjlincoln-nomos-nba-quant-2.hf.space")
    print(f"Monitor: https://huggingface.co/spaces/{SPACE_ID}")
    print("\nS11 config: SPACE_ROLE=exploration → mut=0.15, cx=0.70, feat=80, tournament=5")


if __name__ == "__main__":
    main()
