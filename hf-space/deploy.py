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
    # ── Core DB ──
    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
    "SUPABASE_API_KEY": os.environ.get("SUPABASE_API_KEY", ""),
    "SUPABASE_PASSWORD": os.environ.get("SUPABASE_PASSWORD", ""),
    "SUPABASE_URL_2": os.environ.get("SUPABASE_URL_2", ""),
    "SUPABASE_ANON_KEY_2": os.environ.get("SUPABASE_ANON_KEY_2", ""),
    "SUPABASE_PASSWORD_2": os.environ.get("SUPABASE_PASSWORD_2", ""),
    "SUPABASE_POOLER_2": os.environ.get("SUPABASE_POOLER_2", ""),
    # ── Neo4j ──
    "NEO4J_URI": os.environ.get("NEO4J_URI", ""),
    "NEO4J_USER": os.environ.get("NEO4J_USER", ""),
    "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", ""),
    "NEO4J_URI_2": os.environ.get("NEO4J_URI_2", ""),
    "NEO4J_USER_2": os.environ.get("NEO4J_USER_2", ""),
    "NEO4J_PASSWORD_2": os.environ.get("NEO4J_PASSWORD_2", ""),
    # ── Pinecone ──
    "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
    "PINECONE_API_KEY_2": os.environ.get("PINECONE_API_KEY_2", ""),
    "PINECONE_HOST": os.environ.get("PINECONE_HOST", ""),
    # ── Sports / Odds ──
    "ODDS_API_KEY": os.environ.get("ODDS_API_KEY", ""),
    # ── LLM Keys (for CrewAI, agents, research) ──
    "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
    "OPENROUTER_KEY_QUANTITATIVE": os.environ.get("OPENROUTER_KEY_QUANTITATIVE", ""),
    "OPENROUTER_KEY_SPARE": os.environ.get("OPENROUTER_KEY_SPARE", ""),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    "GROQ_API_KEY_2": os.environ.get("GROQ_API_KEY_2", ""),
    "GROQ_API_KEY_3": os.environ.get("GROQ_API_KEY_3", ""),
    "XAI_API_KEY": os.environ.get("XAI_API_KEY", ""),
    "COHERE_API_KEY": os.environ.get("COHERE_API_KEY", ""),
    "KIMI_API_KEY": os.environ.get("KIMI_API_KEY", ""),
    # LiteLLM removed — direct provider calls only
    # ── Telegram ──
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "ADMIN_TELEGRAM_ID": os.environ.get("ADMIN_TELEGRAM_ID", ""),
    "TELEGRAM_CHANNEL_ID": os.environ.get("TELEGRAM_CHANNEL_ID", ""),
    # ── Search / Research ──
    "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", ""),
    "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", ""),
    "EXA_API_KEY": os.environ.get("EXA_API_KEY", ""),
    "JINA_API_KEY": os.environ.get("JINA_API_KEY", ""),
    # ── GitHub ──
    "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
    "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
    # ── HuggingFace ──
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "HF_TOKEN_2": os.environ.get("HF_TOKEN_2", ""),
    "HF_TOKEN_3": os.environ.get("HF_TOKEN_3", ""),
    # ── Infrastructure ──
    "VM_CALLBACK_URL": os.environ.get("VM_CALLBACK_URL", "http://34.136.180.66:8080"),
    "VM_HOST": os.environ.get("VM_HOST", ""),
    "REDIS_URL": os.environ.get("REDIS_URL", ""),
    "REMOTE_CONTROL_KEY": os.environ.get("REMOTE_CONTROL_KEY", ""),
    # ── Google ──
    "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
}


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: source .env.local")
        sys.exit(1)

    # ── Engine parity check ──
    root_engine = (LOCAL_DIR.parent / "features" / "engine.py").read_bytes()
    hf_engine = (LOCAL_DIR / "features" / "engine.py").read_bytes()
    if root_engine != hf_engine:
        print("ERROR: features/engine.py and hf-space/features/engine.py have diverged!")
        print("Fix: cp hf-space/features/engine.py features/engine.py")
        print("Or:  cp features/engine.py hf-space/features/engine.py")
        sys.exit(1)
    print("Engine parity check: OK")

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
            commit_message="feat: real box scores + skip_placeholder + checkpoint/rollback + Brier-dominant fitness",
        )
        print("Upload OK!")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"Space not found, creating {SPACE_ID}...")
            api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker",
                           space_hardware="cpu-basic", private=False)
            api.create_commit(
                repo_id=SPACE_ID, repo_type="space", operations=operations,
                commit_message="feat: real box scores + skip_placeholder + checkpoint/rollback + Brier-dominant fitness",
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
