#!/usr/bin/env python3
"""
Sync NBA Agent metrics to mon-ipad control tower.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
SYNC_DIR = Path("/home/termius/mon-ipad/data/nba-agent")

def sync():
    """Push latest metrics to mon-ipad."""
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    # 1. Metrics summary
    metrics_file = ROOT / "logs" / "metrics.jsonl"
    if metrics_file.exists():
        lines = [l for l in metrics_file.read_text().strip().split("\n") if l]
        recent = lines[-50:]
        entries = [json.loads(l) for l in recent]

        avg_accuracy = sum(e.get("accuracy", 0) for e in entries) / len(entries) if entries else 0
        avg_latency = sum(e.get("avg_latency_ms", 0) for e in entries) / len(entries) if entries else 0

        summary = {
            "total_evals": len(lines),
            "recent_avg_accuracy": round(avg_accuracy, 1),
            "recent_avg_latency_ms": round(avg_latency),
            "last_10": entries[-10:],
            "synced_at": ts,
        }
        (SYNC_DIR / "metrics-summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Metrics: {len(lines)} total, avg accuracy {avg_accuracy:.1f}%")

    # 2. Errors
    errors_file = ROOT / "logs" / "errors.jsonl"
    if errors_file.exists():
        lines = errors_file.read_text().strip().split("\n")
        recent_errors = [json.loads(l) for l in lines[-20:] if l]
        (SYNC_DIR / "recent-errors.json").write_text(json.dumps(recent_errors, indent=2))
        print(f"Errors: {len(lines)} total")

    # 3. Latest eval
    eval_dir = ROOT / "data" / "eval"
    if eval_dir.exists():
        eval_files = sorted(eval_dir.glob("eval-*.json"))
        if eval_files:
            latest = json.loads(eval_files[-1].read_text())
            (SYNC_DIR / "latest-eval.json").write_text(json.dumps({
                "accuracy": latest.get("accuracy"),
                "total": latest.get("total"),
                "passed": latest.get("passed"),
                "category": latest.get("category"),
                "avg_latency_ms": latest.get("avg_latency_ms"),
                "timestamp": latest.get("timestamp"),
            }, indent=2))

    print(f"Synced to {SYNC_DIR}")

if __name__ == "__main__":
    sync()
