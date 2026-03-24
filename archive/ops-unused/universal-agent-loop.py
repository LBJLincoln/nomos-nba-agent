#!/usr/bin/env python3
"""Universal Agentic Self-Improvement Loop — Deployable to ANY repo.

Karpathy closed-loop: MEASURE → FIND WEAKEST → EXECUTE → RE-MEASURE → KEEP/REVERT → LEARN

Each repo gets its own instance with custom metrics.
Results are logged to both local JSONL and central mon-ipad for cross-repo dashboard.

Usage:
    python3 ops/universal-agent-loop.py --once           # Single cycle
    python3 ops/universal-agent-loop.py --daemon 600     # Loop every 10min
    python3 ops/universal-agent-loop.py --config repo-config.json  # Custom config
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ─── Defaults ─────────────────────────────────────────────────────────────────

CENTRAL_DIR = Path("/home/termius/mon-ipad/data/cross-repo")
REPO_DIR = Path(os.getcwd())
LOCAL_LOG = REPO_DIR / "data" / "self-improve.jsonl"
LOCAL_STATE = REPO_DIR / "data" / "self-improve-state.json"

# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    prefix = {"INFO": "[+]", "WARN": "[!]", "ERROR": "[X]", "OK": "[✓]"}.get(level, "[*]")
    print(f" {ts} {prefix} {msg}")


# ─── Measurement Primitives ──────────────────────────────────────────────────

def git_health(repo_path):
    """Measure git health."""
    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path, timeout=10)
        return r.stdout.strip()
    try:
        status = run(["git", "status", "--porcelain"])
        commits_7d = len(run(["git", "log", "--oneline", "--since=7 days ago"]).splitlines())
        last_commit = run(["git", "log", "-1", "--format=%ci"])
        total = int(run(["git", "rev-list", "--count", "HEAD"]) or 0)
        return {
            "clean": not bool(status),
            "commits_7d": commits_7d,
            "last_commit": last_commit[:19],
            "total_commits": total,
            "dirty_files": len(status.splitlines()) if status else 0,
        }
    except Exception as e:
        return {"clean": False, "commits_7d": 0, "error": str(e)[:100]}


def count_files(repo_path, pattern):
    """Count files matching glob pattern."""
    try:
        return len(list(Path(repo_path).rglob(pattern)))
    except Exception:
        return 0


def run_tests(repo_path, test_cmd):
    """Run test command, return pass/fail and details."""
    try:
        r = subprocess.run(
            test_cmd, shell=True, capture_output=True, text=True,
            cwd=repo_path, timeout=120,
        )
        output = (r.stdout + r.stderr)[-2000:]
        passed = r.returncode == 0
        # Try to extract test count
        test_count = 0
        for line in output.splitlines():
            if "passed" in line.lower() or "tests" in line.lower():
                for word in line.split():
                    if word.isdigit():
                        test_count = int(word)
                        break
        return {"passed": passed, "returncode": r.returncode, "test_count": test_count, "output_tail": output[-500:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": -1, "error": "timeout"}
    except Exception as e:
        return {"passed": False, "returncode": -1, "error": str(e)[:200]}


def measure_repo(repo_path, config):
    """Measure all configured metrics for a repo. Returns dict of metric_name → {value, target, gap}."""
    metrics = {}

    # Git health (universal)
    git = git_health(repo_path)
    metrics["git_activity"] = {
        "value": min(git.get("commits_7d", 0) * 10, 100),  # 10 commits/week = 100%
        "target": 80,
        "details": git,
    }

    # Test coverage (if test_cmd configured)
    test_cmd = config.get("test_cmd")
    if test_cmd:
        test_result = run_tests(repo_path, test_cmd)
        metrics["tests_passing"] = {
            "value": 100 if test_result["passed"] else 0,
            "target": 100,
            "details": test_result,
        }

    # File count metrics (configured per repo)
    for fm in config.get("file_metrics", []):
        count = count_files(repo_path, fm["pattern"])
        metrics[fm["name"]] = {
            "value": min(count / max(fm.get("target_count", 10), 1) * 100, 100),
            "target": 80,
            "details": {"count": count, "pattern": fm["pattern"]},
        }

    # Custom script metrics
    for cm in config.get("custom_metrics", []):
        try:
            r = subprocess.run(
                cm["cmd"], shell=True, capture_output=True, text=True,
                cwd=repo_path, timeout=60,
            )
            val = 0
            try:
                val = float(r.stdout.strip().splitlines()[-1])
            except Exception:
                val = 100 if r.returncode == 0 else 0
            metrics[cm["name"]] = {"value": val, "target": cm.get("target", 80), "details": {"output": r.stdout[-200:]}}
        except Exception as e:
            metrics[cm["name"]] = {"value": 0, "target": cm.get("target", 80), "details": {"error": str(e)[:100]}}

    # Compute gaps
    for m in metrics.values():
        m["gap"] = max(m["target"] - m["value"], 0)

    return metrics


# ─── Improvement Engine ──────────────────────────────────────────────────────

def find_weakest(metrics):
    """Find the metric with the biggest gap."""
    worst = None
    worst_gap = -1
    for name, m in metrics.items():
        if m["gap"] > worst_gap:
            worst_gap = m["gap"]
            worst = name
    return worst, worst_gap


def build_improvement_prompt(repo_name, weakest_metric, metrics, config):
    """Generate a targeted Claude Code CLI prompt based on the weakest metric."""
    base_rules = """RULES:
- Make EXACTLY ONE small, focused improvement (under 20 lines changed)
- Do NOT rewrite entire files
- Do NOT touch CLAUDE.md, .env, or credentials
- If you create a test, make it actually testable
- Commit your change with a descriptive message
"""

    prompts = config.get("improvement_prompts", {})

    if weakest_metric in prompts:
        specific = prompts[weakest_metric]
    elif weakest_metric == "tests_passing":
        specific = f"Tests are failing in {repo_name}. Read the test output, find the failing test, and fix it. If the fix requires changing source code (not just the test), do that."
    elif weakest_metric == "git_activity":
        specific = f"This repo ({repo_name}) hasn't had commits recently. Find ONE small improvement: fix a typo, add a missing type hint, improve an error message, or add a simple test."
    elif "test" in weakest_metric or "coverage" in weakest_metric:
        specific = f"Add ONE new test for the most critical untested function in {repo_name}. Keep it simple and focused."
    else:
        specific = f"Improve the '{weakest_metric}' metric in {repo_name}. Make one small, measurable change."

    details = metrics.get(weakest_metric, {}).get("details", {})
    context = json.dumps(details, indent=2, default=str)[:500]

    return f"""{specific}

Current state of '{weakest_metric}': {metrics.get(weakest_metric, {}).get('value', 0):.0f}% (target: {metrics.get(weakest_metric, {}).get('target', 80)}%)
Details: {context}

{base_rules}"""


def execute_improvement(repo_path, prompt, timeout=300):
    """Run Claude Code CLI to make an improvement."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--dangerously-skip-permissions", prompt],
            capture_output=True, text=True, timeout=timeout, cwd=repo_path,
        )
        output = (result.stdout or "")[:2000]

        # Check if changes were made
        git_st = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=repo_path,
        )
        has_changes = bool(git_st.stdout.strip())

        return {
            "success": result.returncode == 0,
            "has_changes": has_changes,
            "output": output,
            "duration": timeout,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "has_changes": False, "output": "TIMEOUT", "duration": timeout}
    except Exception as e:
        return {"success": False, "has_changes": False, "output": str(e)[:500], "duration": 0}


def git_commit_push(repo_path, message):
    """Commit all changes and push."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"{message}\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"],
            cwd=repo_path, capture_output=True,
        )
        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_path, capture_output=True, text=True,
        )
        return push.returncode == 0
    except Exception:
        return False


def git_revert(repo_path):
    """Revert all uncommitted changes."""
    try:
        subprocess.run(["git", "checkout", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)
        return True
    except Exception:
        return False


# ─── State Management ────────────────────────────────────────────────────────

def load_state():
    if LOCAL_STATE.exists():
        try:
            return json.loads(LOCAL_STATE.read_text())
        except Exception:
            pass
    return {"cycles": 0, "improvements": 0, "reverts": 0, "failures": 0, "consecutive_failures": 0}


def save_state(state):
    LOCAL_STATE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    LOCAL_STATE.write_text(json.dumps(state, indent=2))


def log_cycle(result):
    """Log cycle result to local JSONL and central cross-repo dir."""
    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_LOG, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")

    # Also write to central cross-repo directory
    CENTRAL_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = result.get("repo", "unknown")
    central_file = CENTRAL_DIR / f"{repo_name}.jsonl"
    with open(central_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")

    # Update central status snapshot
    status_file = CENTRAL_DIR / f"{repo_name}-status.json"
    status = {
        "repo": repo_name,
        "last_cycle": result.get("timestamp"),
        "loop_active": True,
        "metrics": result.get("metrics_before", {}),
        "last_action": result.get("action"),
        "last_result": result.get("outcome"),
        "total_cycles": result.get("total_cycles", 0),
        "total_improvements": result.get("total_improvements", 0),
    }
    status_file.write_text(json.dumps(status, indent=2, default=str))


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_cycle(repo_path, config):
    """Run one Karpathy cycle: measure → find weakest → improve → re-measure → keep/revert."""
    repo_name = config.get("name", Path(repo_path).name)
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()

    log(f"=== CYCLE #{state['cycles']+1} for {repo_name} ===")

    # 1. MEASURE (before)
    log("Measuring...")
    metrics_before = measure_repo(repo_path, config)
    avg_before = sum(m["value"] for m in metrics_before.values()) / max(len(metrics_before), 1)

    metric_strs = [f"{k}={v['value']:.0f}%" for k, v in metrics_before.items()]
    log(f"  Metrics: {', '.join(metric_strs)}")
    log(f"  Average: {avg_before:.1f}%")

    # 2. FIND WEAKEST
    weakest, gap = find_weakest(metrics_before)
    if gap <= 0:
        log("All metrics at target! Nothing to improve.", "OK")
        state["cycles"] += 1
        save_state(state)
        log_cycle({"timestamp": now, "repo": repo_name, "action": "skip", "outcome": "all_at_target",
                    "metrics_before": {k: v["value"] for k, v in metrics_before.items()},
                    "total_cycles": state["cycles"], "total_improvements": state["improvements"]})
        return True

    log(f"  Weakest: {weakest} (gap={gap:.0f}%)")

    # 3. PLAN + EXECUTE
    prompt = build_improvement_prompt(repo_name, weakest, metrics_before, config)
    log(f"Executing improvement for '{weakest}'...")

    exec_result = execute_improvement(repo_path, prompt, timeout=config.get("timeout", 300))

    if not exec_result["has_changes"]:
        log("No changes made by Claude CLI", "WARN")
        state["cycles"] += 1
        state["failures"] += 1
        state["consecutive_failures"] += 1
        save_state(state)
        log_cycle({"timestamp": now, "repo": repo_name, "action": weakest, "outcome": "no_changes",
                    "metrics_before": {k: v["value"] for k, v in metrics_before.items()},
                    "total_cycles": state["cycles"], "total_improvements": state["improvements"]})

        # Auto-stop after 3 consecutive failures
        if state["consecutive_failures"] >= 3:
            log("3 consecutive failures — stopping to avoid thrashing", "ERROR")
            return False
        return True

    # 4. RE-MEASURE (after)
    log("Re-measuring...")
    metrics_after = measure_repo(repo_path, config)
    avg_after = sum(m["value"] for m in metrics_after.values()) / max(len(metrics_after), 1)

    delta = avg_after - avg_before
    target_delta = metrics_after.get(weakest, {}).get("value", 0) - metrics_before.get(weakest, {}).get("value", 0)

    log(f"  Before: {avg_before:.1f}% → After: {avg_after:.1f}% (delta: {delta:+.1f}%)")
    log(f"  Target '{weakest}': {target_delta:+.1f}%")

    # 5. KEEP or REVERT
    if delta >= 0 and target_delta >= 0:
        # Improvement or neutral — KEEP
        pushed = git_commit_push(repo_path, f"improve({weakest}): Auto-improvement by agentic loop")
        log(f"KEPT — committed and {'pushed' if pushed else 'push failed'}", "OK")
        state["improvements"] += 1
        state["consecutive_failures"] = 0
        outcome = "improved"
    elif delta < -5:
        # Regression — REVERT
        git_revert(repo_path)
        log(f"REVERTED — regression detected (delta={delta:+.1f}%)", "WARN")
        state["reverts"] += 1
        outcome = "reverted"
    else:
        # Minor regression but target improved — KEEP with warning
        pushed = git_commit_push(repo_path, f"improve({weakest}): Auto-improvement by agentic loop")
        log(f"KEPT (marginal) — target improved but avg slightly down", "WARN")
        state["improvements"] += 1
        state["consecutive_failures"] = 0
        outcome = "kept_marginal"

    state["cycles"] += 1
    save_state(state)

    log_cycle({
        "timestamp": now, "repo": repo_name, "action": weakest, "outcome": outcome,
        "metrics_before": {k: v["value"] for k, v in metrics_before.items()},
        "metrics_after": {k: v["value"] for k, v in metrics_after.items()},
        "delta": round(delta, 2), "target_delta": round(target_delta, 2),
        "total_cycles": state["cycles"], "total_improvements": state["improvements"],
    })

    return True


# ─── Repo Configs (built-in) ─────────────────────────────────────────────────

BUILTIN_CONFIGS = {
    "rag-website": {
        "name": "rag-website",
        "test_cmd": "npx vitest run --reporter=verbose 2>&1 || true",
        "timeout": 300,
        "file_metrics": [
            {"name": "test_files", "pattern": "**/*.test.{ts,tsx}", "target_count": 20},
            {"name": "components", "pattern": "src/components/**/*.tsx", "target_count": 40},
        ],
        "improvement_prompts": {
            "test_files": "Add ONE new vitest test for the most important untested React component in src/components/. Use vitest + @testing-library/react. Test the component renders without error and has correct text content.",
            "tests_passing": "Read the failing vitest test output. Fix the test or the source code to make it pass. Only fix ONE thing.",
            "git_activity": "Find ONE small improvement in src/app/ pages: fix a typo, improve accessibility (aria-label), or add proper alt text to an image. Under 10 lines.",
        },
    },
    "rag-data-ingestion": {
        "name": "rag-data-ingestion",
        "test_cmd": "python3 -m pytest tests/ -x --tb=short 2>&1 | tail -30 || true",
        "timeout": 300,
        "file_metrics": [
            {"name": "test_files", "pattern": "tests/test_*.py", "target_count": 20},
            {"name": "source_files", "pattern": "src/**/*.py", "target_count": 15},
        ],
        "improvement_prompts": {
            "test_files": "Add ONE new pytest test for the most critical untested function in src/. Test realistic inputs and edge cases.",
            "tests_passing": "Read failing pytest output. Fix the simplest failing test. Change source code if the test is correct.",
        },
    },
    "rag-dashboard": {
        "name": "rag-dashboard",
        "test_cmd": "echo 100",  # No test framework yet
        "timeout": 300,
        "file_metrics": [
            {"name": "html_pages", "pattern": "**/*.html", "target_count": 5},
            {"name": "js_files", "pattern": "**/*.js", "target_count": 10},
        ],
        "improvement_prompts": {
            "html_pages": "Read index.html or control-panel.html. Fix ONE thing: accessibility (aria attributes), semantic HTML, or add a missing meta tag. Under 10 lines.",
            "git_activity": "Find ONE improvement in the dashboard HTML/JS: fix a console.error handler, add loading states, or improve error messages.",
        },
    },
    "nomos-nba-agent": {
        "name": "nomos-nba-agent",
        "test_cmd": "python3 -m pytest tests/ -x --tb=short 2>&1 | tail -30 || true",
        "timeout": 300,
        "file_metrics": [
            {"name": "test_files", "pattern": "tests/test_*.py", "target_count": 10},
            {"name": "model_files", "pattern": "models/*.py", "target_count": 6},
        ],
        "improvement_prompts": {
            "test_files": "Add ONE new pytest test for a function in models/ (kelly.py, predictor.py, odds_analyzer.py, or power_ratings.py). Test with realistic NBA data.",
            "tests_passing": "Read failing test output. Fix the test or the model code. Only ONE fix.",
        },
    },
}


def get_config(repo_path, config_path=None):
    """Load config from file or use built-in."""
    if config_path and Path(config_path).exists():
        return json.loads(Path(config_path).read_text())

    # Try built-in
    repo_name = Path(repo_path).name
    if repo_name in BUILTIN_CONFIGS:
        return BUILTIN_CONFIGS[repo_name]

    # Fallback: minimal config
    return {
        "name": repo_name,
        "test_cmd": None,
        "timeout": 300,
        "file_metrics": [
            {"name": "python_files", "pattern": "**/*.py", "target_count": 20},
        ],
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal Agentic Self-Improvement Loop")
    parser.add_argument("--repo", default=os.getcwd(), help="Repo path (default: cwd)")
    parser.add_argument("--config", help="Config JSON file path")
    parser.add_argument("--once", action="store_true", help="Run single cycle")
    parser.add_argument("--daemon", type=int, metavar="INTERVAL", help="Loop with interval (seconds)")
    parser.add_argument("--max-cycles", type=int, default=100, help="Max cycles before stopping")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    config = get_config(repo_path, args.config)

    log(f"Universal Agent Loop — {config.get('name', 'unknown')}")
    log(f"Repo: {repo_path}")
    log(f"Mode: {'once' if args.once else f'daemon ({args.daemon}s interval)'}")

    cycle = 0
    while cycle < args.max_cycles:
        try:
            ok = run_cycle(repo_path, config)
            if not ok:
                log("Stopping due to repeated failures", "ERROR")
                break
        except KeyboardInterrupt:
            log("Interrupted by user")
            break
        except Exception as e:
            log(f"Cycle error: {e}", "ERROR")

        cycle += 1
        if args.once:
            break

        interval = args.daemon or 600
        log(f"Sleeping {interval}s until next cycle...")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            log("Interrupted")
            break

    log(f"Done after {cycle} cycles")


if __name__ == "__main__":
    main()
