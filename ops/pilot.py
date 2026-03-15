#!/usr/bin/env python3
"""
Pilot — Receive commands from mon-ipad control tower.
Protocol: mon-ipad writes ops/commands.json → pilot reads, executes, writes ops/results.json
"""

import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

def load_env():
    env_file = ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

load_env()

COMMANDS_FILE = ROOT / "ops" / "commands.json"
RESULTS_FILE = ROOT / "ops" / "results.json"
HISTORY_FILE = ROOT / "ops" / "command-history.jsonl"

def run_cmd(cmd, timeout=300):
    """Execute a shell command and return result."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(ROOT), env={**os.environ},
        )
        return {
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}

def handle_command(cmd):
    """Process a command from mon-ipad."""
    action = cmd.get("action", "")
    params = cmd.get("params", {})
    ts = datetime.now(timezone.utc).isoformat()

    if action == "eval":
        category = params.get("category", "")
        max_q = params.get("max_q", 10)
        args = ["python3", str(ROOT / "agents" / "nba-agent.py"), "--quick"]
        if category:
            args.extend(["--category", category])
        return run_cmd(args, timeout=300)

    elif action == "eval_full":
        return run_cmd(["python3", str(ROOT / "agents" / "nba-agent.py"), "--eval"], timeout=600)

    elif action == "ask":
        question = params.get("question", "What is the NBA?")
        return run_cmd(["python3", str(ROOT / "agents" / "nba-agent.py"), "--ask", question], timeout=120)

    elif action == "ingest":
        source = params.get("source", "all")
        return run_cmd(["python3", str(ROOT / "ops" / "ingest-nba.py"), "--source", source], timeout=300)

    elif action == "start_daemon":
        interval = params.get("interval", 300)
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_dir / "daemon.log", "a")
        subprocess.Popen(
            ["python3", "-u", str(ROOT / "agents" / "nba-agent.py"), "--daemon", str(interval)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env={**os.environ},
        )
        return {"stdout": f"Daemon started (interval={interval}s)", "returncode": 0}

    elif action == "status":
        # Collect status
        metrics_file = ROOT / "logs" / "metrics.jsonl"
        metrics_count = 0
        last_accuracy = None
        if metrics_file.exists():
            lines = metrics_file.read_text().strip().split("\n")
            metrics_count = len(lines)
            if lines and lines[-1]:
                try:
                    last = json.loads(lines[-1])
                    last_accuracy = last.get("accuracy")
                except:
                    pass

        errors_file = ROOT / "logs" / "errors.jsonl"
        errors_count = sum(1 for _ in open(errors_file)) if errors_file.exists() else 0

        eval_dir = ROOT / "data" / "eval"
        eval_count = len(list(eval_dir.glob("eval-*.json"))) if eval_dir.exists() else 0

        return {
            "stdout": json.dumps({
                "metrics_entries": metrics_count,
                "last_accuracy": last_accuracy,
                "errors": errors_count,
                "eval_runs": eval_count,
                "timestamp": ts,
            }),
            "returncode": 0,
        }

    else:
        return {"stdout": "", "stderr": f"Unknown action: {action}", "returncode": 1}

def poll_loop(interval=5):
    """Poll for commands from mon-ipad."""
    print(f"NBA Pilot listening for commands (poll every {interval}s)")
    last_mtime = 0

    while True:
        try:
            if COMMANDS_FILE.exists():
                mtime = COMMANDS_FILE.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    cmd = json.loads(COMMANDS_FILE.read_text())
                    cmd_id = cmd.get("id", "unknown")
                    print(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Command: {cmd.get('action')} (id={cmd_id})")

                    result = handle_command(cmd)

                    response = {
                        "command_id": cmd_id,
                        "action": cmd.get("action"),
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    RESULTS_FILE.write_text(json.dumps(response, indent=2))

                    # Append to history
                    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(HISTORY_FILE, "a") as f:
                        f.write(json.dumps({"cmd": cmd, "result": result, "ts": datetime.now(timezone.utc).isoformat()}) + "\n")

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(interval)

if __name__ == "__main__":
    poll_loop()
