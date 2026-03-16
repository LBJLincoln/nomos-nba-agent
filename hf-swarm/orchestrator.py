#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — Genetic Evolution Orchestrator
=====================================================
Replaces the basic 4-agent swarm with real genetic evolution.

Architecture:
  - EvolutionLoop: Population of 50 model configs, evolving 24/7
  - LiteLLM Research Agents: Search latest 2026 NBA quant papers & market microstructure
  - Self-Diagnostic: Every 5 generations, detect weaknesses & adapt
  - Gradio Dashboard: Live evolution stats (generation, Brier, ROI, features)

All computation runs on HF Spaces (16GB RAM). VM is for pilotage only.
"""

import os, sys, json, time, threading, subprocess, signal, traceback
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
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LiteLLM config (S7 proxy — 13-provider fallback)
LITELLM_URL = os.environ.get("LITELLM_URL", "https://lbjlincoln-nomos-rag-engine-7.hf.space")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "sk-litellm-nomos-2026")

# ── State ──
evolution_state = {
    "generation": 0,
    "best_brier": 1.0,
    "best_roi": 0.0,
    "best_sharpe": 0.0,
    "best_composite": 0.0,
    "best_model_type": "none",
    "n_features_selected": 0,
    "n_feature_candidates": 0,
    "population_size": 0,
    "status": "INITIALIZING",
    "cycle": 0,
    "games_loaded": 0,
    "diagnostics": [],
    "research_summary": "No research yet",
    "history": [],
}

# Shared log
improvement_log = deque(maxlen=500)


def log(msg, target="system"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    improvement_log.append(entry)


# ── LiteLLM Research Agent ──

def call_litellm(system_prompt, user_prompt, model="smart", max_tokens=4000, temperature=0.3):
    """Call LiteLLM proxy (S7) for research tasks."""
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_KEY}",
    }
    req = urllib.request.Request(f"{LITELLM_URL}/v1/chat/completions", data=body, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read().decode())
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    except Exception as e:
        log(f"LiteLLM call failed: {e}")
        return ""


def research_nba_quant(generation):
    """
    LiteLLM research agent: search for latest 2026 NBA quant papers
    and market microstructure techniques.
    """
    log(f"[Research Agent] Gen {generation} — Querying latest NBA quant research...")

    system_prompt = """You are an expert NBA quantitative analyst and sports betting researcher.
Your job is to identify cutting-edge techniques from the latest academic papers and industry practice.
Focus on 2025-2026 developments. Be specific about implementation details.
Output JSON with keys: techniques (list of {name, description, expected_brier_improvement, implementation_difficulty, category}), market_insights (list of strings), recommended_features (list of strings)."""

    user_prompt = f"""Research the latest developments in NBA prediction and sports betting quantitative analysis (2025-2026).

Focus areas:
1. **Market Microstructure**: CLV (Closing Line Value) analysis, line movement patterns, steam moves, reverse line movement, sharp money indicators
2. **Advanced ML**: Uncertainty-aware models (MC Dropout, conformal prediction), graph neural networks for player interactions, attention-based temporal models
3. **Feature Engineering**: Pace-adjusted metrics, travel fatigue models (haversine distance), rest day impact, strength of schedule quality, clutch performance
4. **Calibration**: Temperature scaling, Platt scaling, beta calibration, ensemble calibration techniques
5. **Portfolio Theory**: Kelly criterion with uncertainty, fractional Kelly, simultaneous correlated bets, bankroll management with drawdown control

Current system state:
- Generation: {generation}
- Using genetic algorithm to evolve feature subsets + hyperparameters
- ~580 feature candidates across 10 categories
- Walk-forward backtesting with ROI + Brier + Sharpe fitness

What NEW techniques should we explore? What are the biggest alpha opportunities?"""

    response = call_litellm(system_prompt, user_prompt, model="smart", max_tokens=3000)

    if response:
        log(f"[Research Agent] Got {len(response)} chars of research findings")
        # Save to disk
        findings = {
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": response,
        }
        try:
            parsed = json.loads(response)
            findings["parsed"] = parsed
        except json.JSONDecodeError:
            findings["parsed"] = None

        out = RESULTS_DIR / "research-findings.json"
        out.write_text(json.dumps(findings, indent=2, default=str))

        # Update state
        evolution_state["research_summary"] = response[:500]
        return findings
    else:
        log("[Research Agent] No response from LiteLLM")
        return None


def run_self_diagnostic(evo_loop, generation):
    """
    Self-diagnostic: analyze weaknesses in the current evolution state.
    Runs every 5 generations as configured.
    """
    log(f"[Self-Diagnostic] Gen {generation} — Analyzing system weaknesses...")

    issues = []
    best = evo_loop.best_ever

    if best is None:
        issues.append("NO BEST MODEL YET — population has not been evaluated")
        evolution_state["diagnostics"] = issues
        return issues

    # Core fitness checks
    if best.fitness.get("brier", 1.0) > 0.24:
        issues.append(f"BRIER={best.fitness['brier']:.4f} > 0.24 — barely better than baseline 0.25")
    if best.fitness.get("roi", 0.0) < 0.0:
        issues.append(f"ROI={best.fitness['roi']:.1%} — model LOSES money on value bets")
    if best.fitness.get("calibration", 1.0) > 0.05:
        issues.append(f"Calibration error={best.fitness['calibration']:.3f} > 5% — probabilities unreliable")
    if best.fitness.get("sharpe", 0.0) < 0.5:
        issues.append(f"Sharpe={best.fitness['sharpe']:.2f} < 0.5 — poor risk-adjusted returns")

    # Feature diversity checks
    n_feats = best.n_features
    if n_feats < 100:
        issues.append(f"Only {n_feats} features selected — try wider feature search")
    if n_feats > 300:
        issues.append(f"{n_feats} features — potential overfitting, increase mutation pressure")

    # Check feature category distribution
    selected = best.selected_indices()
    feature_names = evo_loop.feature_engine.feature_names
    categories = {"market": 0, "schedule": 0, "matchup": 0, "efficiency": 0, "other": 0}
    for idx in selected:
        if idx < len(feature_names):
            name = feature_names[idx]
            if any(k in name for k in ("market", "spread", "clv", "steam", "line")):
                categories["market"] += 1
            elif any(k in name for k in ("rest", "travel", "fatigue", "b2b")):
                categories["schedule"] += 1
            elif any(k in name for k in ("elo", "h2h", "matchup")):
                categories["matchup"] += 1
            elif any(k in name for k in ("efg", "ortg", "pace", "four_factor")):
                categories["efficiency"] += 1
            else:
                categories["other"] += 1

    if categories["market"] == 0:
        issues.append("ZERO market features — missing microstructure alpha source!")
    if categories["schedule"] < 5:
        issues.append(f"Only {categories['schedule']} schedule features — travel/fatigue edge missed")
    if categories["efficiency"] < 10:
        issues.append(f"Only {categories['efficiency']} efficiency features — core signal weak")

    # Population diversity check
    model_types = [ind.hyperparams.get("model_type", "?") for ind in evo_loop.population]
    unique_types = set(model_types)
    if len(unique_types) < 2:
        issues.append(f"Population converged to single model type: {unique_types} — need more diversity")

    # Convergence check
    if len(evo_loop.history) >= 10:
        recent = evo_loop.history[-10:]
        composites = [h["best_composite"] for h in recent]
        improvement = composites[-1] - composites[0]
        if improvement < 0.001:
            issues.append(f"Stagnant: only {improvement:.4f} improvement over 10 generations — increase mutation rate")

    if issues:
        log(f"[Self-Diagnostic] {len(issues)} issues found:")
        for issue in issues:
            log(f"  - {issue}")

        # Ask LLM for recommendations based on issues
        rec_prompt = f"""Our NBA prediction genetic evolution system has these issues at generation {generation}:
{chr(10).join(f'- {i}' for i in issues)}

Feature category distribution: {json.dumps(categories)}
Current best Brier: {best.fitness.get('brier', 1.0):.4f}
Current best ROI: {best.fitness.get('roi', 0.0):.1%}
Model type: {best.hyperparams.get('model_type', 'unknown')}
Features selected: {n_feats}

Suggest 3 specific, actionable fixes. Be concise."""

        recs = call_litellm(
            "You are an expert ML/quant advisor. Give specific actionable recommendations.",
            rec_prompt,
            model="fast",
            max_tokens=1000,
        )
        if recs:
            log(f"[Self-Diagnostic] LLM Recommendations: {recs[:300]}")
            issues.append(f"LLM RECS: {recs[:500]}")
    else:
        log("[Self-Diagnostic] No issues found — system healthy")

    evolution_state["diagnostics"] = issues[-10:]
    return issues


# ── Main Evolution Orchestrator Loop ──

def evolution_orchestrator_loop():
    """
    Main loop: run genetic evolution 24/7 with periodic research + diagnostics.

    Cycle:
    1. INITIALIZE — Load data, create population
    2. EVOLVE — Run N generations per cycle
    3. RESEARCH — LiteLLM searches latest papers (every 10 gens)
    4. DIAGNOSTIC — Self-analysis of weaknesses (every 5 gens)
    5. SAVE — Persist results to disk
    6. REPEAT
    """
    log("=== GENETIC EVOLUTION ORCHESTRATOR STARTING ===")

    # Import evolution modules
    sys.path.insert(0, "/app")
    try:
        from evolution.loop import EvolutionLoop
        log("EvolutionLoop imported successfully")
    except ImportError as e:
        log(f"CRITICAL: Cannot import EvolutionLoop: {e}")
        log("Falling back to standalone mode...")
        evolution_state["status"] = f"IMPORT ERROR: {e}"
        return

    # Initialize evolution loop
    try:
        evo_loop = EvolutionLoop(
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
        )
        evolution_state["n_feature_candidates"] = evo_loop.n_feature_candidates
        evolution_state["status"] = "LOADING DATA"
        log(f"Feature engine: {evo_loop.n_feature_candidates} candidates")
    except Exception as e:
        log(f"CRITICAL: EvolutionLoop init failed: {e}")
        log(traceback.format_exc()[-500:])
        evolution_state["status"] = f"INIT ERROR: {e}"
        return

    # Load game data
    try:
        games = evo_loop.load_data()
        evolution_state["games_loaded"] = len(games)
        log(f"Games loaded: {len(games)}")
    except Exception as e:
        log(f"Error loading games: {e}")
        evolution_state["status"] = f"DATA ERROR: {e}"
        return

    if len(games) < 100:
        log(f"Only {len(games)} games — need at least 100 for evolution")
        evolution_state["status"] = f"INSUFFICIENT DATA ({len(games)} games)"
        # Still try to run with what we have, or wait for data
        if len(games) < 10:
            log("FATAL: Not enough data. Waiting for data ingestion...")
            return

    # Build feature matrix
    try:
        evolution_state["status"] = "BUILDING FEATURES"
        log("Building feature matrix...")
        X, y, feature_names = evo_loop.feature_engine.build(games)
        log(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    except Exception as e:
        log(f"Feature build error: {e}")
        log(traceback.format_exc()[-500:])
        evolution_state["status"] = f"FEATURE ERROR: {e}"
        return

    # Initialize population
    try:
        evo_loop.initialize_population()
        evolution_state["population_size"] = len(evo_loop.population)
        log(f"Population initialized: {len(evo_loop.population)} individuals")
    except Exception as e:
        log(f"Population init error: {e}")
        evolution_state["status"] = f"POP ERROR: {e}"
        return

    # ── Continuous Evolution ──
    cycle = 0
    GENS_PER_CYCLE = 10  # Run 10 generations, then pause for research/diagnostic/save

    while True:
        cycle += 1
        evolution_state["cycle"] = cycle
        evolution_state["status"] = "EVOLVING"

        log(f"\n{'='*60}")
        log(f"EVOLUTION CYCLE #{cycle} — Starting {GENS_PER_CYCLE} generations")
        log(f"{'='*60}")

        for gen_in_cycle in range(GENS_PER_CYCLE):
            try:
                start = time.time()
                evo_loop.evolve_generation(X, y)
                elapsed = time.time() - start

                # Update shared state
                best = evo_loop.best_ever
                if best:
                    evolution_state["generation"] = evo_loop.generation
                    evolution_state["best_brier"] = best.fitness.get("brier", 1.0)
                    evolution_state["best_roi"] = best.fitness.get("roi", 0.0)
                    evolution_state["best_sharpe"] = best.fitness.get("sharpe", 0.0)
                    evolution_state["best_composite"] = best.fitness.get("composite", 0.0)
                    evolution_state["best_model_type"] = best.hyperparams.get("model_type", "?")
                    evolution_state["n_features_selected"] = best.n_features
                    evolution_state["history"] = evo_loop.history[-20:]

                log(f"Gen {evo_loop.generation}: "
                    f"Brier={evolution_state['best_brier']:.4f} "
                    f"ROI={evolution_state['best_roi']:.1%} "
                    f"Sharpe={evolution_state['best_sharpe']:.2f} "
                    f"Features={evolution_state['n_features_selected']} "
                    f"({elapsed:.1f}s)")

            except Exception as e:
                log(f"Generation error: {e}")
                log(traceback.format_exc()[-300:])
                continue

        # ── Periodic Research (every 10 generations) ──
        if evo_loop.generation > 0 and evo_loop.generation % 10 == 0:
            evolution_state["status"] = "RESEARCHING"
            try:
                research_thread = threading.Thread(
                    target=research_nba_quant,
                    args=(evo_loop.generation,),
                    daemon=True,
                )
                research_thread.start()
                research_thread.join(timeout=180)  # 3 min max for research
            except Exception as e:
                log(f"Research error: {e}")

        # ── Self-Diagnostic (every 5 generations) ──
        if evo_loop.generation > 0 and evo_loop.generation % 5 == 0:
            evolution_state["status"] = "DIAGNOSTIC"
            try:
                run_self_diagnostic(evo_loop, evo_loop.generation)
            except Exception as e:
                log(f"Diagnostic error: {e}")

        # ── Save Results ──
        evolution_state["status"] = "SAVING"
        try:
            evo_loop.save_results()
            log(f"Results saved — Gen {evo_loop.generation}")
        except Exception as e:
            log(f"Save error: {e}")

        # ── Sync summary to swarm-status.json (for external consumption) ──
        try:
            status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": "genetic-evolution",
                "cycle": cycle,
                "generation": evo_loop.generation,
                "best_brier": evolution_state["best_brier"],
                "best_roi": evolution_state["best_roi"],
                "best_sharpe": evolution_state["best_sharpe"],
                "best_composite": evolution_state["best_composite"],
                "best_model_type": evolution_state["best_model_type"],
                "n_features_selected": evolution_state["n_features_selected"],
                "n_feature_candidates": evolution_state["n_feature_candidates"],
                "population_size": evolution_state["population_size"],
                "games_loaded": evolution_state["games_loaded"],
                "diagnostics": evolution_state["diagnostics"],
            }
            (RESULTS_DIR / "swarm-status.json").write_text(json.dumps(status, indent=2, default=str))
        except Exception as e:
            log(f"Status sync error: {e}")

        # Brief pause between cycles
        evolution_state["status"] = "COOLDOWN"
        log(f"Cycle #{cycle} done — cooling down 60s before next cycle...")
        time.sleep(60)


# ── Gradio Dashboard ──

def get_dashboard():
    s = evolution_state
    gen = s["generation"]
    brier = s["best_brier"]
    roi = s["best_roi"]
    sharpe = s["best_sharpe"]
    composite = s["best_composite"]
    model = s["best_model_type"]
    n_feats = s["n_features_selected"]
    n_cands = s["n_feature_candidates"]
    pop = s["population_size"]
    games = s["games_loaded"]
    status = s["status"]
    cycle = s["cycle"]

    lines = [
        "# NOMOS NBA QUANT AI — Genetic Evolution Dashboard",
        "",
        "## System Status",
        f"- **Status**: {status}",
        f"- **Cycle**: {cycle} | **Generation**: {gen}",
        f"- **Games Loaded**: {games:,}",
        f"- **Population**: {pop} individuals",
        f"- **Feature Candidates**: {n_cands}",
        "",
        "## Best Individual (All-Time)",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Brier Score | **{brier:.4f}** (lower = better, baseline 0.25) |",
        f"| ROI | **{roi:.1%}** (positive = profitable) |",
        f"| Sharpe Ratio | **{sharpe:.2f}** (>1 = good) |",
        f"| Composite Fitness | **{composite:.4f}** |",
        f"| Model Type | **{model}** |",
        f"| Features Selected | **{n_feats}** / {n_cands} |",
        "",
    ]

    # Evolution history chart (text-based)
    history = s.get("history", [])
    if history:
        lines.append("## Evolution Progress (last 20 gens)")
        lines.append("| Gen | Brier | ROI | Features | Model | Composite |")
        lines.append("|-----|-------|-----|----------|-------|-----------|")
        for h in history[-20:]:
            lines.append(
                f"| {h.get('generation', '?')} "
                f"| {h.get('best_brier', 0):.4f} "
                f"| {h.get('best_roi', 0):.1%} "
                f"| {h.get('n_features', '?')} "
                f"| {h.get('model_type', '?')} "
                f"| {h.get('best_composite', 0):.4f} |"
            )
        lines.append("")

    # Diagnostics
    diags = s.get("diagnostics", [])
    if diags:
        lines.append("## Self-Diagnostic Issues")
        for d in diags[-5:]:
            lines.append(f"- {d[:200]}")
        lines.append("")

    # Research
    research = s.get("research_summary", "")
    if research and research != "No research yet":
        lines.append("## Latest Research Findings")
        lines.append(f"```\n{research[:600]}\n```")

    return "\n".join(lines)


def get_logs():
    return "\n".join(list(improvement_log)[-100:])


def get_research():
    try:
        f = RESULTS_DIR / "research-findings.json"
        if f.exists():
            data = json.loads(f.read_text())
            return json.dumps(data, indent=2, default=str)[:5000]
    except Exception:
        pass
    return "No research findings yet."


def get_evolution_json():
    try:
        f = RESULTS_DIR / "evolution-status.json"
        if f.exists():
            return f.read_text()[:10000]
    except Exception:
        pass
    return "{}"


with gr.Blocks(title="NOMOS NBA QUANT — Genetic Evolution", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# NOMOS NBA QUANT AI — Genetic Evolution Engine")
    gr.Markdown("*Population of 50 models evolving 24/7 via genetic algorithm + LiteLLM research agents*")

    with gr.Row():
        with gr.Column(scale=2):
            dashboard_md = gr.Markdown(get_dashboard)
            refresh_btn = gr.Button("Refresh Dashboard")
            refresh_btn.click(get_dashboard, outputs=dashboard_md)

        with gr.Column(scale=3):
            logs_box = gr.Textbox(label="Evolution Logs", value=get_logs, lines=25, max_lines=30)
            refresh_logs_btn = gr.Button("Refresh Logs")
            refresh_logs_btn.click(get_logs, outputs=logs_box)

    with gr.Row():
        with gr.Column():
            research_box = gr.Textbox(label="Research Findings", value=get_research, lines=15, max_lines=20)
            refresh_research_btn = gr.Button("Refresh Research")
            refresh_research_btn.click(get_research, outputs=research_box)

        with gr.Column():
            evo_json_box = gr.Textbox(label="Evolution Status (JSON)", value=get_evolution_json, lines=15, max_lines=20)
            refresh_evo_btn = gr.Button("Refresh Evolution Data")
            refresh_evo_btn.click(get_evolution_json, outputs=evo_json_box)

    # Auto-refresh every 30s
    timer = gr.Timer(30)
    timer.tick(get_dashboard, outputs=dashboard_md)
    timer.tick(get_logs, outputs=logs_box)


# ── Launch evolution loop at module level (HF Spaces imports, doesn't run __main__) ──
_evo_thread = threading.Thread(target=evolution_orchestrator_loop, daemon=True, name="EvolutionLoop")
_evo_thread.start()
log("Genetic evolution orchestrator thread started (module-level)")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
