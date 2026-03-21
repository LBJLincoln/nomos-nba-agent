#!/usr/bin/env python3
"""
Odds Analyzer — Live odds comparison across 24+ bookmakers.

Starlizard-inspired:
- Fetch real odds from The Odds API
- Compare across: Pinnacle, DraftKings, FanDuel, Betway, Unibet, Winamax, etc.
- Calculate implied probability per bookmaker
- Identify +EV opportunities (our model vs market)
- Calculate CLV (Closing Line Value)
- Detect line movement patterns
- Output: ranked opportunity table with EV%, Kelly sizing, confidence
"""

import os, sys, json, ssl, math, hashlib, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.power_ratings import predict_matchup, get_team, NBA_TEAMS
from models.kelly import (
    BetOpportunity, evaluate_bet, evaluate_multiple_bets, MultiKellyResult,
    implied_probability, kelly_fraction, edge_percentage, format_kelly_report,
    FRACTIONAL_KELLY, DEFAULT_BANKROLL
)

# ── Config ────────────────────────────────────────────────────────────────────
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def _load_env():
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

_load_env()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Bookmakers we track (priority order — Pinnacle is the sharpest)
PRIORITY_BOOKMAKERS = [
    "pinnacle",        # Sharpest book — reference line
    "draftkings",      # Major US book
    "fanduel",         # Major US book
    "betmgm",          # Major US book
    "caesars",         # Major US book
    "bet365",          # Major international
    "unibet",          # European book
    "betway",          # European book
    "williamhill",     # Traditional UK book
    "winamax",         # French book (for ParionsSport comparison)
    "bovada",          # US offshore
    "pointsbetus",     # US book
    "betrivers",       # US regional
    "888sport",        # International
    "betonlineag",     # US offshore
    "mybookieag",      # US offshore
    "superbook",       # US book
    "espnbet",         # New US book
    "hardrockbet",     # US regional
    "lowvig",          # Sharp US book
    "betparx",         # US regional
    "fliff",           # Social sportsbook
    "twinspires",      # US book
    "wynnbet",         # US book
]


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def http_get(url, headers=None, timeout=30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


# ══════════════════════════════════════════════════════════════════════════════
# ODDS FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_bovada_as_odds_api_format():
    """Fetch from Bovada free API and convert to Odds API compatible format."""
    url = "https://www.bovada.lv/services/sports/event/coupon/events/A/description/basketball/nba?marketFilterId=def&lang=en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read().decode())
    games = []
    for group in raw:
        for ev in group.get("events", []):
            parts = ev.get("description", "").split(" @ ")
            if len(parts) != 2:
                continue
            game = {
                "id": ev.get("id"),
                "home_team": parts[1].strip(),
                "away_team": parts[0].strip(),
                "commence_time": ev.get("startTime"),
                "bookmakers": [],
            }
            markets = []
            for dg in ev.get("displayGroups", []):
                for mkt in dg.get("markets", []):
                    mtype = mkt.get("description", "").lower()
                    outcomes = []
                    for oc in mkt.get("outcomes", []):
                        o = {"name": oc.get("description", ""),
                             "price": float(oc.get("price", {}).get("decimal", "2.0") or "2.0")}
                        if oc.get("price", {}).get("handicap"):
                            o["point"] = float(oc["price"]["handicap"])
                        outcomes.append(o)
                    if "spread" in mtype:
                        markets.append({"key": "spreads", "outcomes": outcomes})
                    elif "moneyline" in mtype or "money" in mtype:
                        markets.append({"key": "h2h", "outcomes": outcomes})
                    elif "total" in mtype:
                        markets.append({"key": "totals", "outcomes": outcomes})
            if markets:
                game["bookmakers"] = [{"key": "bovada", "title": "Bovada", "markets": markets}]
            games.append(game)
    return games


def fetch_live_odds(markets="h2h,spreads,totals", regions="us,eu,uk,au"):
    """
    Fetch live NBA odds. Bovada primary (FREE), The Odds API fallback.
    Returns list of games with bookmaker odds in Odds API format.
    """
    # 1. Try Bovada first (free, no key needed)
    try:
        games = _fetch_bovada_as_odds_api_format()
        if games:
            print(f"[ODDS] Bovada: {len(games)} games fetched (FREE)")
            return games
    except Exception as e:
        print(f"[ODDS] Bovada failed: {e}")

    # 2. Try The Odds API (needs key with credits)
    if ODDS_API_KEY:
        url = (
            f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
            f"?apiKey={ODDS_API_KEY}"
            f"&regions={regions}"
            f"&markets={markets}"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
        )
        data, status = http_get(url, timeout=15)
        if isinstance(data, list) and data:
            print(f"[ODDS] The Odds API: {len(data)} games fetched")
            return data

    # 3. Try cached odds from data/odds/latest.json
    cached = ROOT / "data" / "odds" / "latest.json"
    if cached.exists():
        try:
            snapshot = json.loads(cached.read_text())
            if snapshot.get("games"):
                print(f"[ODDS] Using cached odds ({snapshot.get('timestamp', 'unknown')})")
                return snapshot["games"]
        except Exception:
            pass

    # 4. Fall back to simulated odds
    print("[ODDS] No live odds available — using simulated odds")
    return _simulate_odds()


def _simulate_odds():
    """
    Generate realistic simulated odds when API is unavailable.
    Uses power ratings to create market-realistic lines.
    """
    import random
    games = []
    # Generate 5-8 simulated games
    team_pairs = [
        ("BOS", "NYK"), ("OKC", "DEN"), ("LAL", "GSW"),
        ("MIL", "CLE"), ("PHX", "DAL"), ("MIN", "SAC"),
        ("MIA", "PHI"), ("MEM", "HOU"),
    ]

    for home_abbrev, away_abbrev in team_pairs[:random.randint(5, 8)]:
        home_team = NBA_TEAMS[home_abbrev]
        away_team = NBA_TEAMS[away_abbrev]

        # Use power ratings to derive realistic odds
        pred = predict_matchup(home_abbrev, away_abbrev)
        if not pred:
            continue

        home_prob = pred["home_win_prob"]
        away_prob = pred["away_win_prob"]

        # Add vig (3-5% per side)
        vig = random.uniform(1.03, 1.06)
        home_odds = round(1.0 / (home_prob * vig) + random.uniform(-0.05, 0.05), 2)
        away_odds = round(1.0 / (away_prob * vig) + random.uniform(-0.05, 0.05), 2)

        # Ensure odds are reasonable
        home_odds = max(1.10, min(home_odds, 8.0))
        away_odds = max(1.10, min(away_odds, 8.0))

        # Generate spread
        spread_val = round(pred["spread"] + random.uniform(-0.5, 0.5), 1)

        # Generate total
        total_val = round(pred["predicted_total"] + random.uniform(-2, 2), 1)

        bookmakers = []
        for bk in PRIORITY_BOOKMAKERS[:8]:
            # Each book has slightly different odds
            noise_h = random.uniform(-0.08, 0.08)
            noise_a = random.uniform(-0.08, 0.08)
            bk_home = round(max(1.10, home_odds + noise_h), 2)
            bk_away = round(max(1.10, away_odds + noise_a), 2)
            spread_noise = random.uniform(-0.5, 0.5)

            bookmakers.append({
                "key": bk,
                "title": bk.replace("_", " ").title(),
                "last_update": datetime.now(timezone.utc).isoformat(),
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home_team["name"], "price": bk_home},
                            {"name": away_team["name"], "price": bk_away},
                        ]
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home_team["name"], "price": 1.91, "point": round(spread_val + spread_noise, 1)},
                            {"name": away_team["name"], "price": 1.91, "point": round(-spread_val - spread_noise, 1)},
                        ]
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.91, "point": total_val},
                            {"name": "Under", "price": 1.91, "point": total_val},
                        ]
                    },
                ]
            })

        games.append({
            "id": hashlib.md5(f"{home_abbrev}-{away_abbrev}".encode()).hexdigest()[:12],
            "sport_key": "basketball_nba",
            "commence_time": datetime.now(timezone.utc).isoformat(),
            "home_team": home_team["name"],
            "away_team": away_team["name"],
            "bookmakers": bookmakers,
        })

    print(f"[ODDS] Generated {len(games)} simulated games (API key not configured)")
    return games


# ══════════════════════════════════════════════════════════════════════════════
# ODDS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_game_odds(game_data):
    """
    Analyze odds for a single game across all bookmakers.

    Returns dict with:
    - best odds per outcome
    - market consensus
    - implied probabilities
    - arbitrage detection
    """
    home = game_data.get("home_team", "")
    away = game_data.get("away_team", "")

    analysis = {
        "game_id": game_data.get("id", ""),
        "home_team": home,
        "away_team": away,
        "commence_time": game_data.get("commence_time", ""),
        "markets": {},
    }

    for market_key in ["h2h", "spreads", "totals"]:
        market_data = {"best_odds": {}, "worst_odds": {}, "all_odds": defaultdict(list), "implied_probs": {}}

        for bookmaker in game_data.get("bookmakers", []):
            bk_name = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                if market.get("key") != market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    point = outcome.get("point")

                    key = name if market_key == "h2h" else f"{name} ({point:+g})" if point is not None else name

                    market_data["all_odds"][key].append({
                        "bookmaker": bk_name,
                        "price": price,
                        "point": point,
                    })

                    # Track best/worst
                    if key not in market_data["best_odds"] or price > market_data["best_odds"][key]["price"]:
                        market_data["best_odds"][key] = {"bookmaker": bk_name, "price": price, "point": point}
                    if key not in market_data["worst_odds"] or price < market_data["worst_odds"][key]["price"]:
                        market_data["worst_odds"][key] = {"bookmaker": bk_name, "price": price, "point": point}

        # Calculate market consensus (average implied probability)
        for key, odds_list in market_data["all_odds"].items():
            avg_price = sum(o["price"] for o in odds_list) / len(odds_list)
            market_data["implied_probs"][key] = round(implied_probability(avg_price), 4)

        # Check for arbitrage
        if market_key == "h2h" and len(market_data["best_odds"]) >= 2:
            best_prices = list(market_data["best_odds"].values())
            if len(best_prices) >= 2:
                arb_margin = sum(1.0 / bp["price"] for bp in best_prices)
                market_data["arbitrage"] = {
                    "margin": round(arb_margin, 4),
                    "is_arb": arb_margin < 1.0,
                    "profit_pct": round((1.0 - arb_margin) * 100, 2) if arb_margin < 1.0 else 0,
                }

        # Convert defaultdict to regular dict for JSON serialization
        market_data["all_odds"] = dict(market_data["all_odds"])
        analysis["markets"][market_key] = market_data

    return analysis


def find_value_bets(games, bankroll=DEFAULT_BANKROLL, kelly_frac=FRACTIONAL_KELLY):
    """
    Compare our power ratings model against market odds to find +EV bets.

    Returns list of value bets sorted by edge.
    """
    opportunities = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Match team names to our power ratings
        home_abbrev = _match_team_name(home)
        away_abbrev = _match_team_name(away)

        if not home_abbrev or not away_abbrev:
            continue

        # Get our model prediction
        prediction = predict_matchup(home_abbrev, away_abbrev)
        if not prediction:
            continue

        our_home_prob = prediction["home_win_prob"]
        our_away_prob = prediction["away_win_prob"]

        # Analyze odds from all bookmakers
        analysis = analyze_game_odds(game)
        h2h = analysis["markets"].get("h2h", {})

        # Find best odds for each outcome
        for outcome_name, best in h2h.get("best_odds", {}).items():
            odds = best["price"]
            bookmaker = best["bookmaker"]

            # Determine which side and our probability
            if _match_team_name(outcome_name) == home_abbrev:
                our_prob = our_home_prob
                selection = "home"
            elif _match_team_name(outcome_name) == away_abbrev:
                our_prob = our_away_prob
                selection = "away"
            else:
                continue

            edge = edge_percentage(odds, our_prob)

            opp = BetOpportunity(
                game_id=game.get("id", ""),
                description=f"{outcome_name} ML @ {odds:.2f}",
                market="h2h",
                selection=selection,
                decimal_odds=odds,
                estimated_prob=our_prob,
                bookmaker=bookmaker,
            )
            opportunities.append(opp)

        # Also check spreads
        spreads = analysis["markets"].get("spreads", {})
        for outcome_name, best in spreads.get("best_odds", {}).items():
            odds = best["price"]
            point = best.get("point")
            bookmaker = best["bookmaker"]

            if point is None:
                continue

            # Adjust probability based on spread vs our predicted spread
            our_spread = prediction["spread"]
            spread_diff = abs(point) - abs(our_spread)

            # If our model says spread should be larger, there's value on the favorite
            # Simple heuristic: each point of spread difference ~ 3% probability shift
            if _match_team_name(outcome_name.split("(")[0].strip()) == home_abbrev:
                adj_prob = our_home_prob + (spread_diff * 0.03) if point < 0 else our_away_prob
            else:
                adj_prob = our_away_prob + (-spread_diff * 0.03)

            adj_prob = max(0.05, min(0.95, adj_prob))

            opp = BetOpportunity(
                game_id=game.get("id", ""),
                description=f"{outcome_name} Spread @ {odds:.2f}",
                market="spread",
                selection="home" if _match_team_name(outcome_name.split("(")[0].strip()) == home_abbrev else "away",
                decimal_odds=odds,
                estimated_prob=adj_prob,
                bookmaker=bookmaker,
            )
            opportunities.append(opp)

    # Evaluate all with Kelly
    if not opportunities:
        return MultiKellyResult(bets=[], total_exposure=0, expected_portfolio_ev=0,
                                bankroll=bankroll, timestamp=datetime.now(timezone.utc).isoformat())

    result = evaluate_multiple_bets(opportunities, bankroll, kelly_frac)

    # Sort by edge (only bets we should take)
    result.bets.sort(key=lambda b: b.edge, reverse=True)

    return result


def _match_team_name(name):
    """Match a full team name to our abbreviation system."""
    name_lower = name.lower().strip()
    for abbrev, team in NBA_TEAMS.items():
        if (name_lower == team["name"].lower() or
            team["name"].lower().endswith(name_lower) or
            name_lower.endswith(team["name"].split()[-1].lower()) or
            team["city"].lower() in name_lower):
            return abbrev
    return None


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def format_odds_table(games, lang="fr"):
    """Format live odds into a comparison table."""
    lines = [
        f"\n{'='*80}",
        f"COTES NBA LIVE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"{'='*80}\n",
    ]

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        analysis = analyze_game_odds(game)

        lines.append(f"  {away} @ {home}")
        lines.append(f"  {'─'*70}")

        h2h = analysis["markets"].get("h2h", {})
        if h2h.get("best_odds"):
            lines.append(f"  Moneyline (meilleure cote):")
            for outcome, best in h2h["best_odds"].items():
                impl = implied_probability(best["price"])
                lines.append(f"    {outcome:<30s} {best['price']:.2f} ({best['bookmaker']:<12s}) [{impl*100:.1f}%]")

        spreads = analysis["markets"].get("spreads", {})
        if spreads.get("best_odds"):
            lines.append(f"  Spreads:")
            for outcome, best in spreads["best_odds"].items():
                lines.append(f"    {outcome:<30s} {best['price']:.2f} ({best['bookmaker']:<12s})")

        totals = analysis["markets"].get("totals", {})
        if totals.get("best_odds"):
            lines.append(f"  Totals:")
            for outcome, best in totals["best_odds"].items():
                lines.append(f"    {outcome:<30s} {best['price']:.2f} ({best['bookmaker']:<12s})")

        # Arbitrage check
        arb = h2h.get("arbitrage", {})
        if arb.get("is_arb"):
            lines.append(f"  *** ARBITRAGE DETECTE: {arb['profit_pct']:.2f}% profit garanti ***")

        lines.append("")

    return "\n".join(lines)


def format_value_bets_report(multi_kelly, lang="fr"):
    """Format value bets into a ranked report."""
    lines = [
        f"\n{'='*80}",
        f"OPPORTUNITES +EV — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"Bankroll: {multi_kelly.bankroll:.2f}$ | Exposition: {multi_kelly.total_exposure*100:.1f}%",
        f"EV portefeuille: {multi_kelly.expected_portfolio_ev:+.2f}$",
        f"{'='*80}\n",
    ]

    bet_count = 0
    for r in multi_kelly.bets:
        if r.is_bet:
            bet_count += 1
            lines.append(
                f"  {bet_count}. {r.opportunity['description']:<40s} "
                f"Edge: {r.edge*100:+.1f}% | Kelly: {r.fractional_kelly*100:.2f}% | "
                f"Mise: {r.recommended_bet:.2f}$"
            )

    if bet_count == 0:
        lines.append("  Aucune opportunite +EV detectee pour le moment.")

    lines.append(f"\n  Total: {bet_count} paris recommandes")
    lines.append(f"{'='*80}")

    # Also show passes
    passes = [r for r in multi_kelly.bets if not r.is_bet]
    if passes:
        lines.append(f"\n  Paris rejetes ({len(passes)}):")
        for r in passes[:5]:
            lines.append(f"    PASS: {r.opportunity['description']:<35s} | {r.reason}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE/LOAD ODDS HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_odds_snapshot(games, analysis_results=None):
    """Save current odds for CLV tracking."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snapshot_dir = ROOT / "data" / "odds"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "games_count": len(games),
        "games": games[:20],  # Limit to avoid huge files
    }

    if analysis_results:
        snapshot["value_bets"] = [
            {
                "description": b.opportunity.get("description", ""),
                "edge": b.edge,
                "kelly": b.fractional_kelly,
                "bet": b.recommended_bet,
                "is_bet": b.is_bet,
            }
            for b in analysis_results.bets
            if b.is_bet
        ]

    out_file = snapshot_dir / f"odds-{ts}.json"
    out_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
    print(f"[ODDS] Snapshot saved: {out_file}")
    return out_file


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Odds Analyzer")
    parser.add_argument("--live", action="store_true", help="Fetch and display live odds")
    parser.add_argument("--value", action="store_true", help="Find value bets")
    parser.add_argument("--bankroll", type=float, default=1000, help="Current bankroll")
    parser.add_argument("--save", action="store_true", help="Save odds snapshot")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    games = fetch_live_odds()

    if args.live:
        print(format_odds_table(games))

    if args.value or not args.live:
        result = find_value_bets(games, args.bankroll)
        if args.json:
            output = {
                "timestamp": result.timestamp,
                "bankroll": result.bankroll,
                "total_exposure": result.total_exposure,
                "expected_ev": result.expected_portfolio_ev,
                "bets": [
                    {
                        "description": b.opportunity.get("description", ""),
                        "edge": b.edge,
                        "kelly": b.fractional_kelly,
                        "bet": b.recommended_bet,
                        "is_bet": b.is_bet,
                        "reason": b.reason,
                    }
                    for b in result.bets
                ],
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_value_bets_report(result))

    if args.save:
        result = find_value_bets(games, args.bankroll)
        save_odds_snapshot(games, result)
