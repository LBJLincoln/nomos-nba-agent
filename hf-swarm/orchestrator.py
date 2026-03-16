#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — Swarm Orchestrator
=========================================
Runs 4 AI coding agents + ML training in parallel, 24/7.

Agents:
  1. Claude Code CLI  — strategic planning, complex analysis
  2. Gemini CLI        — free autonomous coding (headless + YOLO)
  3. Kimi Code CLI     — cheap high-volume code improvement
  4. OpenClaw          — automation hub, skill-based tasks

Training:
  - Karpathy continuous training loop (9+ models, Optuna)
  - Walk-forward backtesting
  - Feature engineering research

All agents share a workspace and iterate on the same codebase.
Results sync to VM data server → website.
"""

import os, sys, json, time, threading, subprocess, signal
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import gradio as gr

# ── Config ──
WORKSPACE = Path("/app/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/app/data")

# Agent config
AGENTS = {
    "claude": {
        "name": "Claude Code CLI",
        "cmd": "claude",
        "available": False,
        "status": "CHECKING",
        "cycles": 0,
        "last_run": "never",
        "logs": deque(maxlen=100),
    },
    "gemini": {
        "name": "Gemini CLI",
        "cmd": "gemini",
        "available": False,
        "status": "CHECKING",
        "cycles": 0,
        "last_run": "never",
        "logs": deque(maxlen=100),
    },
    "kimi": {
        "name": "Kimi Code CLI",
        "cmd": "kimi",
        "available": False,
        "status": "CHECKING",
        "cycles": 0,
        "last_run": "never",
        "logs": deque(maxlen=100),
    },
    "openclaw": {
        "name": "OpenClaw",
        "cmd": "openclaw",
        "available": False,
        "status": "CHECKING",
        "cycles": 0,
        "last_run": "never",
        "logs": deque(maxlen=100),
    },
}

# Training state
training_state = {
    "cycle": 0,
    "best_brier": 1.0,
    "best_model": "none",
    "status": "STARTING",
    "games": 0,
    "models_trained": 0,
    "logs": deque(maxlen=200),
}

# Shared improvement log
improvement_log = deque(maxlen=500)


def log(msg, target="system"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    improvement_log.append(entry)
    if target in AGENTS:
        AGENTS[target]["logs"].append(entry)
    elif target == "training":
        training_state["logs"].append(entry)


def check_agent_availability():
    """Check which AI coding agents are installed and available."""
    for key, agent in AGENTS.items():
        try:
            result = subprocess.run(
                [agent["cmd"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                agent["available"] = True
                agent["status"] = "READY"
                log(f"{agent['name']}: AVAILABLE ({result.stdout.strip()[:50]})", key)
            else:
                # Try --help as fallback
                result2 = subprocess.run(
                    [agent["cmd"], "--help"],
                    capture_output=True, text=True, timeout=10
                )
                if result2.returncode == 0:
                    agent["available"] = True
                    agent["status"] = "READY"
                    log(f"{agent['name']}: AVAILABLE", key)
                else:
                    agent["status"] = "NOT INSTALLED"
                    log(f"{agent['name']}: not available", key)
        except FileNotFoundError:
            agent["status"] = "NOT INSTALLED"
            log(f"{agent['name']}: not found in PATH", key)
        except Exception as e:
            agent["status"] = f"ERROR: {e}"
            log(f"{agent['name']}: error checking: {e}", key)


# ── NBA Improvement Tasks ──

NBA_TASKS = [
    {
        "id": "optimize_features",
        "prompt": """Analyze the NBA prediction model features in /app/workspace/app.py.
The build_features() function creates 24 features. Research and add at least 5 more
high-impact features: pace-adjusted efficiency, clutch performance stats,
travel fatigue (haversine distance between arenas), rest days impact,
and strength of schedule quality. Implement them and verify the feature count increases.
Do NOT break existing functionality.""",
        "priority": 1,
        "agent_pref": "gemini",  # Free, autonomous
    },
    {
        "id": "optuna_deeper",
        "prompt": """The Optuna hyperparameter search in /app/workspace/app.py uses 25 trials.
Increase to 50 trials and add Bayesian optimization with TPE sampler.
Also add hyperparameter search for CatBoost and Random Forest (currently only XGB and LGBM).
Add early stopping to avoid wasting trials on bad configurations.
Test that training still completes within 30 minutes.""",
        "priority": 2,
        "agent_pref": "kimi",  # Cheap for heavy iteration
    },
    {
        "id": "calibration_improvement",
        "prompt": """Improve model calibration in /app/workspace/app.py.
Currently uses isotonic calibration. Add:
1. Platt scaling comparison
2. Beta calibration (requires calibration library)
3. Temperature scaling
4. Ensemble calibration (calibrate the stacking output)
Compare all methods via Brier score on validation set.
Keep the best method. Target: 5% Brier improvement.""",
        "priority": 3,
        "agent_pref": "claude",  # Complex analysis
    },
    {
        "id": "backtest_framework",
        "prompt": """Add walk-forward backtesting to /app/workspace/app.py.
After training, run expanding-window backtest:
- Start with 2 seasons, predict next month
- Track ROI, Sharpe ratio, max drawdown
- Compare Kelly sizing vs flat betting
- Log results to /app/data/results/backtest-latest.json
This validates the model actually makes money, not just accurate predictions.""",
        "priority": 4,
        "agent_pref": "gemini",
    },
    {
        "id": "research_papers",
        "prompt": """Search the web for the latest NBA prediction research papers (2025-2026).
Find at least 3 new techniques we haven't implemented:
- Uncertainty-aware models (MC Dropout, conformal prediction)
- Graph neural networks for player interactions
- Attention-based temporal models
- Market microstructure (CLV, line movement patterns)
Write a summary to /app/data/results/research-findings.json with
technique name, paper reference, expected Brier improvement, implementation difficulty.""",
        "priority": 5,
        "agent_pref": "gemini",  # Free web search
    },
    {
        "id": "data_quality",
        "prompt": """Audit data quality in /app/data/historical/games-*.json files.
Check for: duplicate games, missing scores, team name inconsistencies,
date gaps, impossible scores. Fix any issues found.
Report stats: total games per season, home win %, average score.
Save audit report to /app/data/results/data-audit.json""",
        "priority": 6,
        "agent_pref": "kimi",
    },
]


def run_agent_task(agent_key, task):
    """Run a single improvement task using the specified agent."""
    agent = AGENTS[agent_key]
    if not agent["available"]:
        log(f"Skipping {agent['name']} — not available", agent_key)
        return False

    agent["status"] = f"WORKING: {task['id']}"
    log(f"Starting task '{task['id']}' with {agent['name']}", agent_key)

    try:
        if agent_key == "claude":
            # Claude Code CLI: use -p for non-interactive
            cmd = [
                "claude", "-p", task["prompt"],
                "--output-format", "text",
                "--max-turns", "20",
            ]
            env = os.environ.copy()
            env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

        elif agent_key == "gemini":
            # Gemini CLI: headless + YOLO
            cmd = [
                "gemini", "-p", task["prompt"],
                "--yolo",
            ]
            env = os.environ.copy()
            env["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

        elif agent_key == "kimi":
            # Kimi CLI
            cmd = [
                "kimi", "--prompt", task["prompt"],
            ]
            env = os.environ.copy()

        elif agent_key == "openclaw":
            # OpenClaw CLI (if available)
            cmd = [
                "openclaw", "run", "--prompt", task["prompt"],
            ]
            env = os.environ.copy()
        else:
            return False

        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=1800,  # 30 min max per task
            cwd=str(WORKSPACE),
            env=env,
        )

        output = result.stdout[-500:] if result.stdout else ""
        errors = result.stderr[-300:] if result.stderr else ""

        if result.returncode == 0:
            agent["cycles"] += 1
            agent["last_run"] = datetime.now(timezone.utc).isoformat()
            agent["status"] = "IDLE"
            log(f"Task '{task['id']}' COMPLETED by {agent['name']}", agent_key)
            log(f"Output: {output[:200]}", agent_key)
            return True
        else:
            agent["status"] = "ERROR"
            log(f"Task '{task['id']}' FAILED: {errors[:200]}", agent_key)
            return False

    except subprocess.TimeoutExpired:
        agent["status"] = "TIMEOUT"
        log(f"Task '{task['id']}' TIMEOUT (30min)", agent_key)
        return False
    except Exception as e:
        agent["status"] = f"ERROR: {str(e)[:100]}"
        log(f"Task '{task['id']}' ERROR: {e}", agent_key)
        return False


# ── Training Loop (same as existing app.py) ──

def run_training():
    """Import and run training from the workspace app.py."""
    log("Starting ML training cycle...", "training")
    training_state["status"] = "TRAINING"

    try:
        # Import training module from workspace
        sys.path.insert(0, str(WORKSPACE))
        # Use the existing app.py training logic
        import importlib
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        else:
            import app as training_app

        training_app.train_cycle()
        training_state["cycle"] = training_app.state["cycle"]
        training_state["best_brier"] = training_app.state["best_brier"]
        training_state["best_model"] = training_app.state["best_model"]
        training_state["games"] = training_app.state["games_loaded"]
        training_state["models_trained"] = training_app.state["models_trained"]
        training_state["status"] = "IDLE"
        log(f"Training cycle #{training_state['cycle']} done — Best: {training_state['best_brier']:.4f}", "training")
    except Exception as e:
        training_state["status"] = f"ERROR: {str(e)[:100]}"
        log(f"Training error: {e}", "training")


# ── Agentic Improvement Loop ──

def improvement_loop():
    """
    Main loop: alternate between training and agent improvement tasks.

    Cycle:
    1. TRAIN — Run full ML training cycle
    2. IMPROVE — Dispatch tasks to available agents
    3. EVALUATE — Check if improvements helped
    4. REPEAT
    """
    log("=== SWARM ORCHESTRATOR STARTING ===")

    # Copy app.py to workspace if not there
    workspace_app = WORKSPACE / "app.py"
    source_app = Path("/app/app.py")
    if source_app.exists() and not workspace_app.exists():
        import shutil
        shutil.copy2(source_app, workspace_app)
        log("Copied app.py to workspace")

    # Also copy data
    src_data = Path("/app/data/historical")
    dst_data = WORKSPACE / "data" / "historical"
    if src_data.exists():
        dst_data.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in src_data.glob("*.json"):
            dst = dst_data / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
        log(f"Synced {len(list(src_data.glob('*.json')))} data files to workspace")

    check_agent_availability()

    cycle = 0
    task_idx = 0

    while True:
        cycle += 1
        log(f"\n{'='*60}")
        log(f"IMPROVEMENT CYCLE #{cycle}")
        log(f"{'='*60}")

        # Phase 1: ML Training
        log("Phase 1: ML Training...")
        try:
            run_training()
        except Exception as e:
            log(f"Training phase error: {e}")

        # Phase 2: Agent improvement tasks (parallel where possible)
        log("Phase 2: Agent Improvements...")
        available_agents = [k for k, v in AGENTS.items() if v["available"]]

        if available_agents:
            threads = []
            tasks_this_cycle = []

            for agent_key in available_agents:
                if task_idx >= len(NBA_TASKS):
                    task_idx = 0  # Loop back

                task = NBA_TASKS[task_idx]
                task_idx += 1
                tasks_this_cycle.append((agent_key, task))

            # Run agent tasks in parallel
            for agent_key, task in tasks_this_cycle:
                t = threading.Thread(
                    target=run_agent_task,
                    args=(agent_key, task),
                    daemon=True
                )
                threads.append(t)
                t.start()

            # Wait for all (with timeout)
            for t in threads:
                t.join(timeout=2000)

            log(f"Agent tasks done for cycle #{cycle}")
        else:
            log("No agents available — training only mode")

        # Phase 3: Sync results
        log("Phase 3: Syncing results...")
        try:
            sync_results()
        except Exception as e:
            log(f"Sync error: {e}")

        # Wait between cycles
        wait = 1800 if cycle <= 3 else 3600
        log(f"Waiting {wait//60}min before next cycle...")

        for agent in AGENTS.values():
            if agent["available"] and "ERROR" not in agent["status"]:
                agent["status"] = f"WAITING ({wait//60}min)"

        time.sleep(wait)


def sync_results():
    """Sync training results to VM data server."""
    import urllib.request
    vm_url = os.environ.get("VM_URL", "http://34.136.180.66:8080")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": training_state["cycle"],
        "best_brier": training_state["best_brier"],
        "best_model": training_state["best_model"],
        "games": training_state["games"],
        "models_trained": training_state["models_trained"],
        "agents": {k: {"status": v["status"], "cycles": v["cycles"]} for k, v in AGENTS.items()},
    }

    # Save locally
    out = DATA_DIR / "results" / "swarm-status.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"Saved swarm status to {out}")


# ── Gradio Dashboard ──

def get_dashboard():
    lines = [
        "# NOMOS NBA QUANT AI — Swarm Dashboard",
        "",
        "## Agents",
        "| Agent | Status | Cycles | Last Run |",
        "|-------|--------|--------|----------|",
    ]
    for key, agent in AGENTS.items():
        icon = "🟢" if agent["available"] else "🔴"
        lines.append(f"| {icon} {agent['name']} | {agent['status']} | {agent['cycles']} | {agent['last_run'][:19]} |")

    lines.extend([
        "",
        "## ML Training",
        f"- **Cycle**: {training_state['cycle']}",
        f"- **Status**: {training_state['status']}",
        f"- **Best Brier**: {training_state['best_brier']:.4f}",
        f"- **Best Model**: {training_state['best_model']}",
        f"- **Games**: {training_state['games']:,}",
        f"- **Models**: {training_state['models_trained']}",
    ])

    return "\n".join(lines)


def get_logs():
    return "\n".join(list(improvement_log)[-100:])


def get_agent_logs(agent_key):
    if agent_key in AGENTS:
        return "\n".join(list(AGENTS[agent_key]["logs"])[-50:])
    return "Agent not found"


with gr.Blocks(title="NOMOS NBA QUANT — Swarm", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# NOMOS NBA QUANT AI — 4-Agent Swarm")
    gr.Markdown("*Claude Code + Gemini CLI + Kimi Code + OpenClaw — always improving, 24/7*")

    with gr.Row():
        with gr.Column(scale=2):
            dashboard_md = gr.Markdown(get_dashboard)
            gr.Button("Refresh").click(get_dashboard, outputs=dashboard_md)

        with gr.Column(scale=3):
            logs_box = gr.Textbox(label="Improvement Logs", value=get_logs, lines=25)
            gr.Button("Refresh Logs").click(get_logs, outputs=logs_box)

    with gr.Row():
        with gr.Column():
            agent_select = gr.Dropdown(
                choices=list(AGENTS.keys()),
                label="Agent Logs",
                value="claude"
            )
            agent_logs = gr.Textbox(label="Agent Output", lines=15)
            agent_select.change(get_agent_logs, inputs=agent_select, outputs=agent_logs)
            gr.Button("Refresh Agent").click(get_agent_logs, inputs=agent_select, outputs=agent_logs)

    timer = gr.Timer(30)
    timer.tick(get_dashboard, outputs=dashboard_md)
    timer.tick(get_logs, outputs=logs_box)


if __name__ == "__main__":
    # Start improvement loop in background
    loop_thread = threading.Thread(target=improvement_loop, daemon=True)
    loop_thread.start()

    # Launch Gradio
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
