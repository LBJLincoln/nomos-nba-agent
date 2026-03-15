#!/usr/bin/env python3
"""
NBA Claude Code CLI Bridge — Direct AI modifications from your phone.

Send commands via Telegram or HTTP to modify NBA agent code, run predictions,
check bankroll, trigger self-improvement, and more.

Commands:
  /predict BOS NYK     — Run full ensemble prediction
  /bankroll             — Show bankroll status
  /odds                 — Fetch live odds summary
  /picks                — Show latest value picks
  /weights              — Show/adjust ensemble weights
  /improve              — Trigger self-improvement cycle
  /status               — Full system status
  /modify <description> — Ask Claude Code to modify the NBA agent
  /run <script>         — Run a script (e.g., /run self-improve.py --once)
  /eval                 — Show performance history
  /rankings             — NBA power rankings top 15
  /help                 — Show this help

Runs as HTTP server on port 8090 (accessible via SSH tunnel or Telegram bridge).
"""

import os, sys, json, subprocess, time, ssl, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "ops"))

# ── Load env ─────────────────────────────────────────────────
def load_env():
    for env_file in [ROOT / ".env.local", ROOT.parent / "mon-ipad" / ".env.local"]:
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

DATA_DIR = ROOT / "data"
BANKROLL_DIR = DATA_DIR / "bankroll"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PERFORMANCE_DIR = DATA_DIR / "performance"
LOG_FILE = DATA_DIR / "cli-bridge.jsonl"


def log(msg):
    ts = datetime.now(timezone.utc).isoformat()[:19]
    entry = {"ts": ts, "msg": msg}
    print(f"[{ts}] {msg}")
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════

def cmd_predict(args):
    """Run ensemble prediction for a matchup."""
    if len(args) < 2:
        return "Usage: /predict HOME AWAY (e.g., /predict BOS NYK)"
    home, away = args[0].upper(), args[1].upper()
    try:
        sys.path.insert(0, str(ROOT / "models"))
        from models.predictor import ensemble_predict, format_prediction_report
        pred = ensemble_predict(home, away)
        return format_prediction_report(pred)
    except Exception as e:
        return f"Prediction error: {e}"


def cmd_bankroll(args):
    """Show bankroll status."""
    state_file = BANKROLL_DIR / "state.json"
    if not state_file.exists():
        return "No bankroll state found. Run the daemon first."
    state = json.loads(state_file.read_text())
    balance = state.get("balance", 100)
    initial = state.get("initial_balance", 100)
    growth = ((balance / initial) - 1) * 100 if initial > 0 else 0
    return (
        f"💰 BANKROLL STATUS\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Balance:  ${balance:.2f}\n"
        f"Initial:  ${initial:.2f}\n"
        f"Growth:   {growth:+.1f}%\n"
        f"Record:   {state.get('wins', 0)}W-{state.get('losses', 0)}L-{state.get('pushes', 0)}P\n"
        f"Wagered:  ${state.get('total_wagered', 0):.2f}\n"
        f"Profit:   ${state.get('total_profit', 0):.2f}\n"
        f"Peak:     ${state.get('peak_balance', 100):.2f}\n"
        f"Drawdown: {state.get('max_drawdown_pct', 0):.1f}%\n"
        f"Streak:   {state.get('streak_current', 0)}\n"
        f"Last bet: {state.get('last_bet_ts', 'Never')[:16]}"
    )


def cmd_odds(args):
    """Fetch live odds summary."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "models" / "odds_analyzer.py"), "--live"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT)
        )
        output = result.stdout[:2000] if result.stdout else result.stderr[:500]
        return output or "No odds data available"
    except Exception as e:
        return f"Odds fetch error: {e}"


def cmd_picks(args):
    """Show latest value picks."""
    latest = sorted(PREDICTIONS_DIR.glob("picks-*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True)
    if not latest:
        return "No picks found. Run the daemon first."
    picks = json.loads(latest[0].read_text())
    lines = [
        f"🎯 LATEST PICKS ({picks.get('timestamp', '')[:16]})",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"Bankroll: ${picks.get('bankroll', 100):.2f}",
        f"Opportunities: {picks.get('total_opportunities', 0)}",
        f"Selected: {picks.get('selected_bets', 0)}",
        f"Exposure: {picks.get('total_exposure', '0%')}",
        f"Expected EV: ${picks.get('expected_ev', 0):.2f}",
        "",
    ]
    for i, p in enumerate(picks.get("picks", [])[:10], 1):
        status = "✅" if p.get("is_bet") else "❌"
        lines.append(f"{status} {i}. {p.get('description', '')}")
        lines.append(f"   {p.get('bookmaker', '')} @ {p.get('odds', 0)} | "
                     f"Edge {p.get('edge', '0%')} | Stake ${p.get('stake', 0):.2f}")
        if not p.get("is_bet"):
            lines.append(f"   Reason: {p.get('reason', '')}")
    return "\n".join(lines)


def cmd_weights(args):
    """Show ensemble weights."""
    weights_file = DATA_DIR / "ensemble-weights.json"
    if weights_file.exists():
        data = json.loads(weights_file.read_text())
        weights = data.get("weights", {})
        lines = [
            f"⚖️ ENSEMBLE WEIGHTS",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"Updated: {data.get('timestamp', '')[:16]}",
            f"Reason: {data.get('adjustment_reason', 'Manual')}",
            "",
        ]
        for model, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 40) + "░" * (40 - int(w * 40))
            lines.append(f"  {model:<16s} {w:.1%} {bar}")
        return "\n".join(lines)
    else:
        return ("Default weights:\n"
                "  power_ratings: 35%\n"
                "  monte_carlo:   30%\n"
                "  elo:           20%\n"
                "  poisson:       15%")


def cmd_improve(args):
    """Trigger self-improvement cycle."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "ops" / "self-improve.py"), "--once"],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        output = result.stdout[-2000:] if result.stdout else result.stderr[:500]
        return output or "Self-improvement cycle completed (no output)"
    except Exception as e:
        return f"Self-improvement error: {e}"


def cmd_status(args):
    """Full system status."""
    lines = ["📊 NBA QUANT SYSTEM STATUS", "━" * 30]

    # Bankroll
    state_file = BANKROLL_DIR / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        lines.append(f"Bankroll: ${state.get('balance', 100):.2f} ({state.get('wins', 0)}W-{state.get('losses', 0)}L)")
    else:
        lines.append("Bankroll: Not initialized")

    # Daemon status
    pid_file = ROOT.parent / "mon-ipad" / "data" / "nba-daemon.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        try:
            os.kill(int(pid), 0)
            lines.append(f"Daemon: RUNNING (PID {pid})")
        except (ProcessLookupError, ValueError):
            lines.append("Daemon: STOPPED (stale PID)")
    else:
        lines.append("Daemon: NOT RUNNING")

    # Predictions count
    pred_count = len(list(PREDICTIONS_DIR.glob("picks-*.json"))) if PREDICTIONS_DIR.exists() else 0
    lines.append(f"Predictions: {pred_count} files")

    # Research papers
    research_dir = DATA_DIR / "research"
    paper_count = len(list(research_dir.glob("paper-*.json"))) if research_dir.exists() else 0
    lines.append(f"Research: {paper_count} papers")

    # Performance
    perf_file = PERFORMANCE_DIR / "history.jsonl"
    if perf_file.exists():
        last_lines = perf_file.read_text().strip().splitlines()
        if last_lines:
            last = json.loads(last_lines[-1])
            lines.append(f"Last eval: accuracy {last.get('accuracy', 0):.1%}, Brier {last.get('brier', 0):.4f}")

    # Models
    lines.append(f"\nModels: ELO + Power Ratings + Poisson + Monte Carlo (1000 sims)")
    lines.append(f"Kelly: 1/4 fractional, max 5% per bet, min 2% edge")

    return "\n".join(lines)


def cmd_modify(args):
    """Use Claude Code to modify the NBA agent."""
    if not args:
        return "Usage: /modify <description of what to change>"
    description = " ".join(args)
    log(f"MODIFY REQUEST: {description}")

    try:
        # Run Claude Code CLI with the modification request
        prompt = (
            f"You are in the nomos-nba-agent repo at {ROOT}. "
            f"The user wants you to: {description}\n\n"
            f"Key files:\n"
            f"- models/predictor.py: Ensemble prediction (ELO, Poisson, MC, Power)\n"
            f"- models/power_ratings.py: 30 NBA teams, contextual adjustments\n"
            f"- models/kelly.py: Kelly criterion betting\n"
            f"- models/odds_analyzer.py: Live odds from The Odds API\n"
            f"- ops/nba-quant-daemon.py: Main daemon (2h cycles)\n"
            f"- ops/self-improve.py: Prediction vs results feedback\n"
            f"- ops/bankroll-manager.py: $100 bankroll tracker\n\n"
            f"Make the change, test it, and report what you did."
        )

        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"}
        )

        output = result.stdout[-3000:] if result.stdout else ""
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr]: {result.stderr[:500]}"

        return output or "Claude Code completed (no output)"

    except subprocess.TimeoutExpired:
        return "Claude Code timed out (5min limit). Try a simpler request."
    except FileNotFoundError:
        return "Claude Code CLI not found. Using fallback LLM..."
    except Exception as e:
        return f"Modify error: {e}"


def cmd_run(args):
    """Run a script in the ops/ directory."""
    if not args:
        return "Usage: /run <script> [args] (e.g., /run self-improve.py --once)"
    script = args[0]
    script_args = args[1:]

    # Security: only allow scripts in ops/
    script_path = ROOT / "ops" / script
    if not script_path.exists():
        # Try models/
        script_path = ROOT / "models" / script
    if not script_path.exists():
        return f"Script not found: {script}\nAvailable: {', '.join(f.name for f in (ROOT / 'ops').glob('*.py'))}"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + script_args,
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT)
        )
        output = result.stdout[-2000:] if result.stdout else ""
        if result.returncode != 0 and result.stderr:
            output += f"\n[error]: {result.stderr[:500]}"
        return output or "Script completed (no output)"
    except subprocess.TimeoutExpired:
        return f"Script {script} timed out (2min limit)"
    except Exception as e:
        return f"Run error: {e}"


def cmd_eval(args):
    """Show performance history."""
    perf_file = PERFORMANCE_DIR / "history.jsonl"
    if not perf_file.exists():
        return "No performance history yet. Run /improve first."
    lines = ["📈 PERFORMANCE HISTORY", "━" * 50]
    for entry_str in perf_file.read_text().strip().splitlines()[-15:]:
        entry = json.loads(entry_str)
        lines.append(
            f"  {entry['ts'][:16]} | Acc {entry['accuracy']:.1%} | "
            f"Brier {entry['brier']:.4f} | Spread err {entry['spread_err']:.1f} | "
            f"Best: {entry['best_model']}"
        )
    return "\n".join(lines)


def cmd_rankings(args):
    """Show NBA power rankings."""
    try:
        from models.power_ratings import batch_power_rankings
        rankings = batch_power_rankings()
        lines = [f"🏀 NBA POWER RANKINGS", "━" * 45]
        for r in rankings[:15]:
            lines.append(f"  {r['rank']:2d}. {r['team']} {r['team_name']:<28s} {r['adjusted_power']:+6.1f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Rankings error: {e}"


def cmd_help(args):
    """Show help."""
    return (
        "🤖 NBA QUANT CLI BRIDGE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/predict BOS NYK — Ensemble prediction\n"
        "/bankroll — Bankroll status\n"
        "/odds — Live odds summary\n"
        "/picks — Latest value picks\n"
        "/weights — Ensemble weights\n"
        "/improve — Self-improvement cycle\n"
        "/status — Full system status\n"
        "/modify <desc> — Claude Code modification\n"
        "/run <script> — Run ops script\n"
        "/eval — Performance history\n"
        "/rankings — Power rankings\n"
        "/help — This help"
    )


COMMANDS = {
    "predict": cmd_predict,
    "bankroll": cmd_bankroll,
    "odds": cmd_odds,
    "picks": cmd_picks,
    "weights": cmd_weights,
    "improve": cmd_improve,
    "status": cmd_status,
    "modify": cmd_modify,
    "run": cmd_run,
    "eval": cmd_eval,
    "rankings": cmd_rankings,
    "help": cmd_help,
}


def handle_command(text):
    """Parse and execute a command."""
    text = text.strip()
    if not text:
        return cmd_help([])

    # Parse /command args
    if text.startswith("/"):
        text = text[1:]
    parts = text.split()
    cmd = parts[0].lower()
    args = parts[1:]

    handler = COMMANDS.get(cmd)
    if handler:
        return handler(args)
    else:
        # If not a recognized command, treat as a /modify request
        return cmd_modify(parts)


# ══════════════════════════════════════════════════════════════
# HTTP SERVER (for Telegram bridge integration)
# ══════════════════════════════════════════════════════════════

class NBABridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
            command = data.get("command", data.get("text", ""))
            result = handle_command(command)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "nba-cli-bridge"}).encode())
        elif self.path == "/status":
            result = cmd_status([])
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(result.encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(cmd_help([]).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


# ══════════════════════════════════════════════════════════════
# TELEGRAM INTEGRATION
# ══════════════════════════════════════════════════════════════

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID = int(os.environ.get("ADMIN_TELEGRAM_ID", "6582544948"))
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def tg_send(chat_id, text):
    """Send message via Telegram."""
    if not BOT_TOKEN:
        return
    # Truncate to Telegram's 4096 char limit
    if len(text) > 4000:
        text = text[:4000] + "\n...(truncated)"
    data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(f"{TG_API}/sendMessage", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, context=SSL_CTX, timeout=10)
    except Exception:
        # Retry without markdown
        data = json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(f"{TG_API}/sendMessage", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, context=SSL_CTX, timeout=10)
        except Exception:
            pass


def tg_poll():
    """Poll for Telegram messages (NBA-specific commands only)."""
    if not BOT_TOKEN:
        print("[TG] No TELEGRAM_BOT_TOKEN — Telegram polling disabled")
        return

    offset = 0
    print(f"[TG] NBA Bridge listening for Telegram messages...")

    while True:
        try:
            url = f"{TG_API}/getUpdates?offset={offset}&timeout=30"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=35) as resp:
                data = json.loads(resp.read().decode())

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                user_id = msg.get("from", {}).get("id")

                if user_id != ADMIN_ID:
                    continue

                # Only handle NBA-specific commands (prefix /nba)
                if text.startswith("/nba"):
                    nba_cmd = text[4:].strip()
                    if not nba_cmd:
                        nba_cmd = "help"
                    log(f"TG NBA command: {nba_cmd}")
                    result = handle_command(nba_cmd)
                    tg_send(chat_id, f"🏀 NBA Agent:\n{result}")

        except Exception as e:
            if "timed out" not in str(e).lower():
                print(f"[TG] Poll error: {e}")
            time.sleep(2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Claude Code CLI Bridge")
    parser.add_argument("--port", type=int, default=8090, help="HTTP server port")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram polling")
    parser.add_argument("--cmd", nargs="*", help="Run single command and exit")
    args = parser.parse_args()

    if args.cmd:
        # Single command mode
        result = handle_command(" ".join(args.cmd))
        print(result)
        return

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), NBABridgeHandler)
    print(f"[NBA CLI Bridge] HTTP server on port {args.port}")
    print(f"[NBA CLI Bridge] POST /  with {{\"command\": \"/predict BOS NYK\"}}")
    print(f"[NBA CLI Bridge] GET /health | GET /status")

    # Start Telegram polling in background
    if args.telegram or BOT_TOKEN:
        tg_thread = threading.Thread(target=tg_poll, daemon=True)
        tg_thread.start()

    # Save PID
    pid_file = DATA_DIR / "cli-bridge.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[NBA CLI Bridge] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
