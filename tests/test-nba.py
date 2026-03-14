#!/usr/bin/env python3
"""
NBA Agent Test Suite — Autonomous evaluation with expert-level scenarios.
Tests the agent across 6 categories with strict keyword-based scoring.
"""

import os, sys, json, time, argparse, random
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
from importlib import import_module

# Dynamic import of nba-agent (has hyphen in name)
import importlib.util
spec = importlib.util.spec_from_file_location("nba_agent", ROOT / "agents" / "nba-agent.py")
nba_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nba_agent)

# ── Test Scenarios (expert-level) ─────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "sharp_betting_nba",
        "description": "Sharp NBA betting with CLV and Kelly Criterion",
        "questions": [
            {"q": "Comment calculer le Kelly Criterion pour un pari NBA spread ?", "keywords": ["Kelly", "bankroll", "edge", "formule"]},
            {"q": "What is Closing Line Value and why is it the best predictor of long-term betting success?", "keywords": ["CLV", "closing", "long-term", "market"]},
            {"q": "Explain Tony Bloom's Starlizard approach to NBA betting", "keywords": ["Bloom", "Starlizard", "model", "EV", "volume"]},
            {"q": "What adjustment should you make for NBA back-to-back games in your model?", "keywords": ["back-to-back", "fatigue", "points", "adjust"]},
        ],
    },
    {
        "name": "advanced_analytics",
        "description": "NBA advanced stats used by front offices",
        "questions": [
            {"q": "Compare RAPTOR, EPM, and LEBRON metrics — which is most predictive?", "keywords": ["RAPTOR", "EPM", "LEBRON", "plus-minus"]},
            {"q": "How does Second Spectrum player tracking work in NBA arenas?", "keywords": ["camera", "tracking", "25", "position", "speed"]},
            {"q": "Qu'est-ce que Synergy Sports et comment les coaches l'utilisent ?", "keywords": ["Synergy", "play", "possession", "PPP"]},
            {"q": "What is the DARKO projection system?", "keywords": ["DARKO", "Kalman", "prediction", "daily"]},
        ],
    },
    {
        "name": "goat_debate_expert",
        "description": "Expert-level historical comparisons",
        "questions": [
            {"q": "Jordan vs LeBron — compare with era-adjusted statistics", "keywords": ["Jordan", "LeBron", "era", "adjust", "PPG"]},
            {"q": "Pourquoi Jokic est-il statistiquement le meilleur centre offensif de tous les temps ?", "keywords": ["Jokic", "triple", "assist", "PER", "MVP"]},
            {"q": "Make the case for Bill Russell as the greatest winner in NBA history", "keywords": ["Russell", "11", "titles", "defense", "Celtics"]},
        ],
    },
    {
        "name": "team_building",
        "description": "Front office strategy and team construction",
        "questions": [
            {"q": "What factors best predict NBA draft success?", "keywords": ["age", "college", "measurements", "conference"]},
            {"q": "How did the 2014 Spurs construct the most beautiful offense in NBA history?", "keywords": ["Spurs", "pass", "movement", "spacing"]},
            {"q": "Explain the positional value hierarchy in modern NBA roster construction", "keywords": ["center", "wing", "guard", "value"]},
        ],
    },
    {
        "name": "live_data",
        "description": "Real-time data integration test",
        "questions": [
            {"q": "Quelles sources de donnees live les equipes NBA utilisent-elles ?", "keywords": ["Sportradar", "tracking", "Second Spectrum", "live"]},
            {"q": "What public APIs can I use to get NBA stats?", "keywords": ["balldontlie", "nba_api", "Basketball Reference", "API"]},
        ],
    },
]

# ── Run tests ─────────────────────────────────────────────────────────────────

def run_scenario(scenario):
    """Run a single test scenario."""
    name = scenario["name"]
    questions = scenario["questions"]
    passed = 0
    total = len(questions)
    results = []

    print(f"\n{'─'*50}")
    print(f"SCENARIO: {name} — {scenario['description']}")
    print(f"{'─'*50}")

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{total}] {q['q'][:60]}...")
        result = nba_agent.answer_question(q["q"])

        answer_lower = result["answer"].lower()
        kw = q["keywords"]
        matched = sum(1 for k in kw if k.lower() in answer_lower)
        ratio = matched / len(kw) if kw else 1.0
        ok = ratio >= 0.5

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"    {status} — {matched}/{len(kw)} keywords ({ratio:.0%}) — {result['latency_ms']}ms")

        results.append({
            "question": q["q"],
            "keywords": kw,
            "matched": matched,
            "ratio": ratio,
            "passed": ok,
            "latency_ms": result["latency_ms"],
            "answer_preview": result["answer"][:200],
        })
        time.sleep(0.5)

    accuracy = passed / total * 100 if total else 0
    print(f"\n  Result: {passed}/{total} ({accuracy:.0f}%)")

    return {
        "scenario": name,
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "results": results,
    }

def run_all(scenarios=None):
    """Run all test scenarios."""
    scenarios = scenarios or SCENARIOS
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(f"\n{'='*60}")
    print(f"NBA AGENT TEST SUITE — {ts}")
    print(f"{len(scenarios)} scenarios, {sum(len(s['questions']) for s in scenarios)} questions")
    print(f"{'='*60}")

    all_results = []
    total_passed = 0
    total_questions = 0

    for scenario in scenarios:
        result = run_scenario(scenario)
        all_results.append(result)
        total_passed += result["passed"]
        total_questions += result["total"]

    overall_accuracy = total_passed / total_questions * 100 if total_questions else 0

    report = {
        "timestamp": ts,
        "overall_accuracy": round(overall_accuracy, 1),
        "total_passed": total_passed,
        "total_questions": total_questions,
        "scenarios": all_results,
    }

    # Save report
    out_dir = ROOT / "logs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"test-{ts}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_passed}/{total_questions} ({overall_accuracy:.1f}%)")
    print(f"Report saved: {out_file}")
    print(f"{'='*60}\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Agent Test Suite")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--scenario", type=str, help="Run specific scenario by name")
    parser.add_argument("--quick", action="store_true", help="Run first scenario only")
    args = parser.parse_args()

    if args.scenario:
        match = [s for s in SCENARIOS if s["name"] == args.scenario]
        if match:
            run_scenario(match[0])
        else:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(s['name'] for s in SCENARIOS)}")
    elif args.quick:
        run_scenario(SCENARIOS[0])
    else:
        run_all()
