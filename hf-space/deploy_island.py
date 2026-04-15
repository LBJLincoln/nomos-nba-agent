#!/usr/bin/env python3
"""Deploy an evolution island to any HF Space.

Usage:
    source .env.local
    python3 hf-space/deploy_island.py SPACE_ID ROLE [HF_TOKEN_VAR]

Examples:
    python3 hf-space/deploy_island.py LBJLincoln26/nba-evo-3 extra_trees_specialist HF_TOKEN_NBA
    python3 hf-space/deploy_island.py LBJLincoln26/nba-evo-4 catboost_specialist HF_TOKEN_NBA

Roles:
    exploitation       — S10 default: mut=0.09, feat=63 (proven optimal)
    exploration        — S11 default: mut=0.15, feat=80 (wider search)
    extra_trees_specialist — extra_trees only, low mut, narrow features
    catboost_specialist    — catboost focus, medium mut
    neural_specialist     — neural models only (MLP, TabNet, FT-Transformer)
    wide_search           — high mut=0.20, feat=120, all model types
"""

import os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

LOCAL_DIR = Path(__file__).parent
SKIP = {"__pycache__", ".pyc", "node_modules", ".git", "deploy.py", "deploy_s11.py", "deploy_island.py"}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    space_id = sys.argv[1]
    role = sys.argv[2]
    token_var = sys.argv[3] if len(sys.argv) > 3 else "HF_TOKEN"
    token = os.environ.get(token_var)

    if not token:
        print(f"ERROR: {token_var} not set. Run: source .env.local")
        sys.exit(1)

    # Engine parity check
    root_engine = (LOCAL_DIR.parent / "features" / "engine.py").read_bytes()
    hf_engine = (LOCAL_DIR / "features" / "engine.py").read_bytes()
    if root_engine != hf_engine:
        print("ERROR: features/engine.py diverged!")
        sys.exit(1)
    print("Engine parity: OK")

    api = HfApi(token=token)
    print(f"Deploying island ({role}) to {space_id}...")

    operations = []
    for fp in LOCAL_DIR.rglob("*"):
        if fp.is_dir():
            continue
        if any(s in str(fp) for s in SKIP):
            continue
        rel = fp.relative_to(LOCAL_DIR)
        print(f"  + {rel}")
        operations.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(fp)))

    print(f"\nUploading {len(operations)} files...")
    api.create_commit(
        repo_id=space_id, repo_type="space", operations=operations,
        commit_message=f"feat: evolution island ({role})",
    )
    print("Upload OK!")

    # Core secrets (DB only — minimal for evolution)
    secrets = {
        "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_API_KEY": os.environ.get("SUPABASE_API_KEY", ""),
        "SUPABASE_PASSWORD": os.environ.get("SUPABASE_PASSWORD", ""),
        "SPACE_ROLE": role,
    }

    print("\nConfiguring secrets...")
    for key, value in secrets.items():
        if value:
            try:
                api.add_space_secret(space_id, key, value)
                print(f"  Set {key}")
            except Exception as e:
                print(f"  WARN: {key}: {e}")

    print("\nRestarting...")
    try:
        api.restart_space(space_id)
    except Exception as e:
        print(f"  Restart: {e}")

    owner = space_id.split("/")[0].lower()
    name = space_id.split("/")[1]
    print(f"\nDone! https://{owner}-{name}.hf.space")


if __name__ == "__main__":
    main()
