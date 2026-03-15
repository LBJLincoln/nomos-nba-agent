#!/usr/bin/env python3
"""
NBA Expert Data Ingestion — Comprehensive multi-source pipeline
===============================================================
Sources:
  1. Basketball Reference (via Exa.AI domain-filtered search)
  2. Live Odds / Paris Sportifs (The Odds API — ALL bookmakers)
  3. Expert Analytics Datasets (RAPTOR, EPM, LEBRON, DARKO, CtG, Synergy)
  4. Tony Bloom / Starlizard Strategy (EV, Kelly, CLV, line shopping)
  5. Mathematical Models Applied to NBA Betting (Poisson, Elo, Monte Carlo, Bayes)

Storage: Pinecone (website-sectors-jina-1024 index, nba namespace)
Embeddings: Self-hosted Jina (lbjlincoln-nomos-embeddings-api.hf.space)
Modes: one-shot, --daemon continuous
"""

import os, sys, json, time, ssl, urllib.request, urllib.parse, hashlib, math, argparse, signal
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "ingest"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "expert-ingest-state.json"

# Load environment
def load_env():
    """Load .env.local from repo root, then from mon-ipad as fallback."""
    for env_path in [ROOT / ".env.local", Path("/home/termius/mon-ipad/.env.local")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

load_env()

# SSL context (skip verification for HF Spaces)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── API Keys ──────────────────────────────────────────────────────────────────
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX = "website-sectors-jina-1024"
PINECONE_NAMESPACE = "nba"
EMBEDDINGS_URL = os.environ.get(
    "EMBEDDINGS_URL", "https://lbjlincoln-nomos-embeddings-api.hf.space"
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "expert-ingest.log"
METRICS_FILE = LOG_DIR / "expert-ingest-metrics.jsonl"

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def log_metric(data):
    try:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def http_get(url, headers=None, timeout=30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            # Guard against HTML error pages
            if raw.strip().startswith("<!") or raw.strip().startswith("<html"):
                return {"error": "HTML response (not JSON)"}, resp.status
            return json.loads(raw), resp.status
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")[:500]
        except Exception:
            err_body = ""
        return {"error": f"HTTP {e.code}: {e.reason} — {err_body}"}, e.code
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response from {url}: {e}"}, 0
    except Exception as e:
        return {"error": str(e)}, 0

def http_post(url, data, headers=None, timeout=60):
    body = json.dumps(data).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if raw.strip().startswith("<!") or raw.strip().startswith("<html"):
                return {"error": "HTML response (not JSON)"}, resp.status
            return json.loads(raw), resp.status
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")[:500]
        except Exception:
            err_body = ""
        return {"error": f"HTTP {e.code}: {e.reason} — {err_body}"}, e.code
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response from {url}: {e}"}, 0
    except Exception as e:
        return {"error": str(e)}, 0


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS & PINECONE
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text, max_chars=8000):
    """Get 1024-dim embedding from self-hosted Jina API."""
    text = text[:max_chars]
    resp, status = http_post(
        f"{EMBEDDINGS_URL}/api/v1/embeddings",
        {"input": [text], "model": "jina-embeddings-v3"},
        timeout=30,
    )
    if "error" in resp:
        # Fallback endpoint
        resp, status = http_post(
            f"{EMBEDDINGS_URL}/embed",
            {"texts": [text]},
            timeout=30,
        )
        if "error" in resp:
            log(f"Embedding error: {resp['error']}", "ERROR")
            return None
        embeddings = resp.get("embeddings", [])
        return embeddings[0] if embeddings else None

    data = resp.get("data", [])
    if data:
        return data[0].get("embedding")
    return None

def get_embeddings_batch(texts, batch_size=16, max_chars=8000):
    """Batch embedding — returns list of embedding vectors."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:max_chars] for t in texts[i:i + batch_size]]
        resp, status = http_post(
            f"{EMBEDDINGS_URL}/api/v1/embeddings",
            {"input": batch, "model": "jina-embeddings-v3"},
            timeout=60,
        )
        if "error" in resp:
            # Fallback: embed one by one
            for t in batch:
                emb = get_embedding(t, max_chars)
                if emb is None:
                    log(f"Skipping text with failed embedding ({len(t)} chars)", "WARN")
                    continue
                all_embeddings.append(emb)
            continue

        data = resp.get("data", [])
        for item in sorted(data, key=lambda x: x.get("index", 0)):
            emb = item.get("embedding")
            if emb is None:
                log(f"Batch item missing embedding (index={item.get('index')})", "WARN")
                continue
            all_embeddings.append(emb)

        if i + batch_size < len(texts):
            time.sleep(0.5)  # Rate limit

    return all_embeddings

def pinecone_host():
    """Resolve Pinecone host for the index."""
    # Try cached
    host = os.environ.get("PINECONE_SECTORS_HOST", "")
    if host:
        return host

    # Describe index to get host
    resp, status = http_get(
        f"https://api.pinecone.io/indexes/{PINECONE_INDEX}",
        headers={"Api-Key": PINECONE_API_KEY, "X-Pinecone-API-Version": "2025-01"},
        timeout=15,
    )
    if "error" not in resp:
        host = resp.get("host", "")
        if host:
            if not host.startswith("https://"):
                host = f"https://{host}"
            os.environ["PINECONE_SECTORS_HOST"] = host
            return host

    log(f"Could not resolve Pinecone host: {resp}", "ERROR")
    return ""

def upsert_to_pinecone(documents, batch_size=50):
    """Upsert documents with embeddings to Pinecone (nba namespace)."""
    if not PINECONE_API_KEY:
        log("No PINECONE_API_KEY — skipping upsert", "WARN")
        return 0

    host = pinecone_host()
    if not host:
        return 0

    # Prepare texts for embedding
    texts = [doc.get("text", "") for doc in documents]
    log(f"Embedding {len(texts)} documents...")
    embeddings = get_embeddings_batch(texts)

    # Build vectors
    vectors = []
    skipped = 0
    for doc, emb in zip(documents, embeddings):
        if emb is None:
            skipped += 1
            continue
        meta = {
            "text": doc.get("text", "")[:3900],  # Pinecone metadata limit ~40KB
            "source": doc.get("source", ""),
            "category": doc.get("category", ""),
            "title": doc.get("title", ""),
            "sector": "nba",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        # Add any extra metadata
        for k, v in doc.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        vectors.append({
            "id": doc.get("id", hashlib.md5(doc["text"][:200].encode()).hexdigest()[:16]),
            "values": emb,
            "metadata": meta,
        })

    if skipped:
        log(f"Skipped {skipped}/{len(documents)} docs (embedding failed)", "WARN")

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        resp, status = http_post(
            f"{host}/vectors/upsert",
            {"vectors": batch, "namespace": PINECONE_NAMESPACE},
            headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
            timeout=60,
        )
        if "error" in resp:
            log(f"Pinecone upsert error (batch {i // batch_size}): {resp['error']}", "ERROR")
        else:
            count = resp.get("upsertedCount", len(batch))
            total_upserted += count

        if i + batch_size < len(vectors):
            time.sleep(0.3)

    log(f"Upserted {total_upserted}/{len(vectors)} vectors to Pinecone ({PINECONE_INDEX}/{PINECONE_NAMESPACE})")
    return total_upserted


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: BASKETBALL REFERENCE (via Exa.AI domain-filtered search)
# ══════════════════════════════════════════════════════════════════════════════

BBREF_QUERIES = [
    # All-time career leaders
    {"query": "NBA all-time career points leaders scoring list", "topic": "All-time career scoring leaders (points)"},
    {"query": "NBA all-time career assists leaders", "topic": "All-time career assists leaders"},
    {"query": "NBA all-time career rebounds leaders", "topic": "All-time career rebounds leaders"},
    {"query": "NBA all-time career steals leaders", "topic": "All-time career steals leaders"},
    {"query": "NBA all-time career blocks leaders", "topic": "All-time career blocks leaders"},
    {"query": "NBA all-time career three-pointers made leaders 3PT", "topic": "All-time career 3-point leaders"},
    {"query": "NBA all-time career PER leaders efficiency", "topic": "All-time career PER leaders"},
    {"query": "NBA all-time career Win Shares leaders", "topic": "All-time career Win Shares leaders"},
    {"query": "NBA all-time career VORP leaders value", "topic": "All-time career VORP leaders"},
    {"query": "NBA all-time career BPM leaders box plus minus", "topic": "All-time career BPM leaders"},
    # Single-season records
    {"query": "NBA single season points per game record highest PPG", "topic": "Single-season PPG records"},
    {"query": "NBA single season assists per game record APG", "topic": "Single-season APG records"},
    {"query": "NBA single season rebounds per game record RPG", "topic": "Single-season RPG records"},
    {"query": "NBA single season three-pointers made record 3PM", "topic": "Single-season 3PT records"},
    {"query": "NBA single season steals per game record", "topic": "Single-season steals records"},
    {"query": "NBA single season blocks per game record", "topic": "Single-season blocks records"},
    {"query": "NBA single game scoring record most points", "topic": "Single-game scoring records"},
    {"query": "NBA triple double records most career triple doubles", "topic": "Career triple-double records"},
    # Awards history
    {"query": "NBA MVP award winners history complete list", "topic": "NBA MVP award history"},
    {"query": "NBA Finals MVP winners complete list history", "topic": "NBA Finals MVP history"},
    {"query": "NBA Defensive Player of the Year DPOY winners list", "topic": "NBA DPOY award history"},
    {"query": "NBA Rookie of the Year winners history", "topic": "NBA ROY award history"},
    {"query": "NBA Sixth Man of the Year winners list", "topic": "NBA 6MOY award history"},
    {"query": "NBA Most Improved Player winners list", "topic": "NBA MIP award history"},
    {"query": "NBA All-Star Game MVP winners history", "topic": "NBA All-Star Game MVP history"},
    # Team histories
    {"query": "NBA championship winners list all champions history", "topic": "NBA champions history"},
    {"query": "NBA team win-loss records all-time best seasons", "topic": "Best team records all-time"},
    {"query": "NBA franchise history oldest teams relocations", "topic": "NBA franchise histories"},
    {"query": "NBA dynasties greatest teams ever Bulls Warriors Lakers Celtics", "topic": "NBA greatest dynasties"},
    {"query": "NBA 73-9 Warriors 72-10 Bulls best regular season records", "topic": "Best regular season records"},
    # Draft history
    {"query": "NBA Draft number 1 overall picks history all first picks", "topic": "NBA Draft #1 picks history"},
    {"query": "NBA Draft biggest busts worst picks all-time", "topic": "NBA Draft biggest busts"},
    {"query": "NBA Draft greatest steals late picks who became stars", "topic": "NBA Draft greatest steals"},
    {"query": "NBA Draft class rankings best draft classes 1984 1996 2003", "topic": "Best NBA Draft classes"},
    {"query": "NBA international players drafted history foreign players", "topic": "International NBA Draft history"},
    # Playoff records
    {"query": "NBA playoffs all-time scoring leaders points", "topic": "NBA Playoffs scoring leaders"},
    {"query": "NBA Finals records greatest performances games", "topic": "Greatest NBA Finals performances"},
    {"query": "NBA playoff series comebacks 3-1 deficit history", "topic": "NBA Playoff comeback records"},
]

def fetch_bbref_via_exa():
    """Fetch Basketball Reference data via Exa.AI domain-filtered search."""
    if not EXA_API_KEY:
        log("No EXA_API_KEY — skipping Basketball Reference ingestion", "WARN")
        return []

    documents = []
    success = 0
    fail = 0

    for item in BBREF_QUERIES:
        resp, status = http_post(
            "https://api.exa.ai/search",
            {
                "query": item["query"],
                "numResults": 3,
                "useAutoprompt": False,
                "type": "auto",
                "includeDomains": ["basketball-reference.com"],
                "contents": {"text": {"maxCharacters": 4000}},
            },
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            timeout=30,
        )

        results = resp.get("results", []) if "error" not in resp else []

        if not results:
            fail += 1
            log(f"  BBREF miss: {item['topic']}", "DEBUG")
        else:
            for r in results:
                text = r.get("text", "")
                if text and len(text) > 100:
                    doc_id = f"bbref-{hashlib.md5((r.get('url', '') + item['topic']).encode()).hexdigest()[:12]}"
                    documents.append({
                        "id": doc_id,
                        "text": f"[Basketball Reference] {item['topic']}\n\n{text[:4000]}",
                        "source": r.get("url", "basketball-reference.com"),
                        "title": r.get("title", item["topic"]),
                        "category": "basketball_reference",
                        "metadata": {"subcategory": item["topic"]},
                    })
                    success += 1

        time.sleep(1.2)  # Rate limit Exa.AI

    log(f"[BBREF] {len(documents)} docs fetched ({success} pages, {fail} misses)")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: LIVE ODDS / PARIS SPORTIFS (The Odds API)
# ══════════════════════════════════════════════════════════════════════════════

# ALL major bookmakers — US, EU, UK, French
BOOKMAKER_DISPLAY = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
    "caesars": "Caesars", "pinnacle": "Pinnacle", "bet365": "Bet365",
    "unibet": "Unibet", "williamhill": "William Hill", "betway": "Betway",
    "888sport": "888sport", "bovada": "Bovada", "pointsbetus": "PointsBet",
    "betrivers": "BetRivers", "superbook": "SuperBook", "wynnbet": "WynnBet",
    "espnbet": "ESPN Bet", "hardrockbet": "Hard Rock Bet", "fliff": "Fliff",
    # EU / French books
    "betfair": "Betfair Exchange", "betfair_ex_eu": "Betfair Exchange EU",
    "sport888": "888sport EU", "betclic": "Betclic", "winamax": "Winamax",
    "parionssport": "Parions Sport (FDJ)", "pmu": "PMU", "zebet": "ZEBet",
    "france_pari": "France Pari", "netbet": "NetBet",
    # Sharp books
    "pinnacle": "Pinnacle", "matchbook": "Matchbook", "smarkets": "Smarkets",
    "betdaq": "BETDAQ",
}

def fetch_live_odds():
    """Fetch live NBA odds from ALL bookmakers via the-odds-api.com.
    Covers h2h (moneyline), spreads, totals from US + EU + UK + AU regions."""
    if not ODDS_API_KEY:
        log("No ODDS_API_KEY — skipping live odds", "WARN")
        return []

    all_documents = []

    # Fetch all market types
    for markets in ["h2h,spreads,totals", "player_points,player_rebounds,player_assists"]:
        url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
            f"?apiKey={ODDS_API_KEY}"
            f"&regions=us,us2,eu,uk,au"
            f"&markets={markets}"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
        )
        resp, _ = http_get(url, timeout=30)

        if "error" in resp:
            log(f"[ODDS] Error fetching {markets}: {resp['error']}", "ERROR")
            continue

        games = resp if isinstance(resp, list) else []

        for game in games:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")

            # Build comprehensive odds document per game
            game_text_parts = [
                f"NBA Game Odds — {away} @ {home}",
                f"Date: {commence[:10] if commence else 'TBD'}",
                f"Commence: {commence}",
                "",
            ]

            # Collect all bookmaker odds for comparison / line shopping
            bookmaker_data = {}
            for bookmaker in game.get("bookmakers", []):
                bk_key = bookmaker.get("key", "")
                bk_name = BOOKMAKER_DISPLAY.get(bk_key, bk_key)
                last_update = bookmaker.get("last_update", "")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    outcomes = market.get("outcomes", [])

                    if market_key not in bookmaker_data:
                        bookmaker_data[market_key] = []

                    odds_line = f"  {bk_name}: "
                    parts = []
                    for o in outcomes:
                        name = o.get("name", "")
                        price = o.get("price", 0)
                        point = o.get("point", "")
                        desc = o.get("description", "")
                        s = f"{name} {price:.2f}"
                        if point != "":
                            s += f" ({point:+g})"
                        if desc:
                            s += f" [{desc}]"
                        parts.append(s)
                    odds_line += " | ".join(parts)
                    bookmaker_data[market_key].append(odds_line)

            # Format by market
            for market_key, lines in bookmaker_data.items():
                market_label = {
                    "h2h": "Moneyline (1X2)",
                    "spreads": "Spreads (Handicap)",
                    "totals": "Over/Under (Totals)",
                    "player_points": "Player Points Props",
                    "player_rebounds": "Player Rebounds Props",
                    "player_assists": "Player Assists Props",
                }.get(market_key, market_key)
                game_text_parts.append(f"--- {market_label} ---")
                game_text_parts.extend(lines)
                game_text_parts.append("")

            # Implied probability + edge analysis for moneyline
            if "h2h" in bookmaker_data:
                game_text_parts.append("--- Implied Probabilities (Moneyline) ---")
                for line in bookmaker_data.get("h2h", [])[:3]:
                    # Simple display of best lines
                    game_text_parts.append(f"  {line.strip()}")

            game_text = "\n".join(game_text_parts)

            doc_id = f"odds-{hashlib.md5(f'{home}-{away}-{commence}-{markets}'.encode()).hexdigest()[:12]}"
            all_documents.append({
                "id": doc_id,
                "text": game_text[:5000],
                "source": "the-odds-api.com",
                "title": f"NBA Odds: {away} @ {home} ({commence[:10] if commence else 'TBD'})",
                "category": "live_odds",
                "metadata": {
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "bookmaker_count": len(game.get("bookmakers", [])),
                    "markets": markets,
                },
            })

        time.sleep(0.5)

    log(f"[ODDS] {len(all_documents)} odds documents from live games")
    return all_documents


def fetch_historical_odds():
    """Fetch historical/closing lines for CLV analysis (requires paid Odds API tier)."""
    if not ODDS_API_KEY:
        return []

    # Get scores (completed games) for CLV tracking
    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
        f"?apiKey={ODDS_API_KEY}&daysFrom=3&dateFormat=iso"
    )
    resp, _ = http_get(url, timeout=30)

    if "error" in resp:
        log(f"[ODDS-HIST] Scores error: {resp['error']}", "WARN")
        return []

    games = resp if isinstance(resp, list) else []
    documents = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        scores = game.get("scores", [])
        completed = game.get("completed", False)

        if not completed or not scores:
            continue

        # Build result document
        score_text = ""
        for s in scores:
            score_text += f"  {s.get('name', '')}: {s.get('score', 'N/A')}\n"

        game_date = game.get("commence_time", "")[:10]
        text = (
            f"NBA Game Result — {away} @ {home}\n"
            f"Date: {game_date}\n"
            f"Scores:\n{score_text}\n"
            f"Use for CLV analysis: compare pre-game odds vs closing lines vs result."
        )

        commence_t = game.get("commence_time", "")
        result_key = f"{home}-{away}-{commence_t}"
        documents.append({
            "id": f"result-{hashlib.md5(result_key.encode()).hexdigest()[:12]}",
            "text": text,
            "source": "the-odds-api.com/scores",
            "title": f"Result: {away} @ {home}",
            "category": "game_results",
            "metadata": {"home": home, "away": away, "completed": True},
        })

    log(f"[ODDS-HIST] {len(documents)} completed game results")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: EXPERT ANALYTICS DATASETS
# ══════════════════════════════════════════════════════════════════════════════

EXPERT_ANALYTICS_KNOWLEDGE = [
    # ── RAPTOR ────────────────────────────────────────────────────────────────
    {
        "id": "analytics-raptor-deep",
        "text": """RAPTOR (Robust Algorithm using Player Tracking and On/Off Ratings)
Developer: FiveThirtyEight / Nate Silver (2019, replaced CARMELO)
Type: Hybrid box-score + tracking + on-off regularized adjusted plus-minus

COMPONENTS:
- RAPTOR Offensive Rating: offensive impact per 100 possessions vs league average
- RAPTOR Defensive Rating: defensive impact per 100 possessions (negative = better)
- RAPTOR Total = Offense + Defense
- WAR (Wins Above Replacement): converts RAPTOR to total wins contributed

METHODOLOGY:
1. Start with box-score prior (stats predict impact)
2. Add player tracking data (Second Spectrum): speed, distance, shot quality
3. Apply RAPM (Regularized Adjusted Plus-Minus) using ridge regression
4. Adjust for team quality, opponent quality, pace, lineup context
5. Apply Bayesian shrinkage for small sample sizes

SCALE (per 100 possessions):
- MVP-level: +8 to +12 RAPTOR
- All-NBA: +5 to +8
- All-Star: +3 to +5
- Good starter: +1 to +3
- Average: 0
- Below average: -1 to -3
- Replacement: -3 or worse

WAR SCALE (per season):
- Historic: 15+ WAR (e.g., peak LeBron, Curry, Jokic)
- MVP candidate: 10-15 WAR
- All-Star: 5-10 WAR
- Solid starter: 2-5 WAR
- Replacement: 0 WAR

LIMITATIONS:
- Tracking data not available pre-2013
- Small sample issues early in season (needs ~500 minutes)
- Defensive rating noisy for non-rim-protectors
- FiveThirtyEight shut down in 2024 — model frozen

HISTORICAL LEADERS (RAPTOR WAR, single season):
- Nikola Jokic 2021-22: ~19.6 WAR
- Giannis 2019-20: ~16.0 WAR
- LeBron James 2012-13: ~17.2 WAR""",
        "source": "FiveThirtyEight / RAPTOR methodology",
        "category": "advanced_analytics",
    },
    # ── EPM ───────────────────────────────────────────────────────────────────
    {
        "id": "analytics-epm-deep",
        "text": """EPM (Estimated Plus-Minus) — by Taylor Snarr / Dunks & Threes
The current gold standard for NBA player impact measurement (2023-present).

METHODOLOGY:
1. Core: Regularized Adjusted Plus-Minus (RAPM) with ridge regression
2. Box-score prior: Use traditional and tracking stats as Bayesian priors
3. Luck adjustment: Filter for shot quality, 3PT variance, FT variance
4. Opponent adjustment: Weight opponent strength based on full-season performance
5. Lineup adjustment: Control for teammate quality in every minute played

COMPONENTS:
- O-EPM (Offensive EPM): points per 100 possessions above average on offense
- D-EPM (Defensive EPM): points per 100 possessions prevented vs average on defense
- EPM Total = O-EPM + D-EPM

SCALE (points per 100 possessions above average):
- All-time great season: +8 to +12 (Jokic 2023-24: +9.2)
- MVP level: +5 to +8
- All-NBA: +3 to +5
- All-Star fringe: +2 to +3
- Quality starter: +1 to +2
- League average: 0.0
- Below average: -1 to -3
- Replacement level: -3 or worse

EPM vs OTHER METRICS:
- More predictive than PER, WS/48, BPM for future performance
- Competitive with LEBRON for in-season evaluation
- Better than raw +/- (controls for lineup context)
- More transparent methodology than proprietary team models

RECENT LEADERS (2023-24 regular season):
1. Nikola Jokic: +9.2 EPM (historic — one of highest ever recorded)
2. Luka Doncic: +7.3 EPM
3. Shai Gilgeous-Alexander: +6.8 EPM
4. Giannis Antetokounmpo: +5.9 EPM
5. Anthony Edwards: +4.1 EPM

USED BY: Multiple NBA front offices, DFS professionals, media analysts
WEBSITE: dunksandthrees.com/epm""",
        "source": "Dunks & Threes / Taylor Snarr — EPM",
        "category": "advanced_analytics",
    },
    # ── LEBRON ────────────────────────────────────────────────────────────────
    {
        "id": "analytics-lebron-deep",
        "text": """LEBRON (Luck-adjusted player Estimate using a Box prior Regularized ON-off)
Developer: BBall Index (Krishna Narsu, Tim Pelton)
Type: Regularized adjusted on-off with box-score prior

METHODOLOGY:
1. On-off differential: How does the team perform with player on vs off court?
2. Regularization: Ridge regression to stabilize estimates
3. Box-score prior: Traditional + tracking stats as starting estimate
4. Luck adjustment: Control for opponent 3PT% variance, FT variance
5. Minutes weighting: Recent minutes weighted more than early-season

COMPONENTS:
- O-LEBRON: Offensive impact per 100 possessions
- D-LEBRON: Defensive impact per 100 possessions
- LEBRON Total = O-LEBRON + D-LEBRON

DERIVED METRICS:
- WAR: Wins Above Replacement (from LEBRON rating + minutes played)
- Offensive/Defensive Load: How much of team's O/D runs through player
- Versatility Score: Range of offensive/defensive roles filled

ADVANTAGES OVER EPM:
- Better calibrated for small sample sizes (early season)
- Updates daily with new game data
- More granular role classification
- Includes proprietary tracking-derived features

LIMITATIONS:
- Proprietary — methodology not fully transparent
- Some components not publicly available
- Requires BBall Index subscription for full data

SCALE (same as EPM — per 100 possessions):
- Elite: > +5.0 LEBRON
- All-Star: +3.0 to +5.0
- Starter: +1.0 to +3.0
- Rotation: -1.0 to +1.0
- Below replacement: < -3.0

NOTABLE: LEBRON metric was named before being a metric about LeBron James.
Actually stands for 'Luck-adjusted player Estimate using a Box prior Regularized ON-off'
WEBSITE: bfranklinindex.com""",
        "source": "BBall Index — LEBRON metric",
        "category": "advanced_analytics",
    },
    # ── DARKO ─────────────────────────────────────────────────────────────────
    {
        "id": "analytics-darko-deep",
        "text": """DARKO (Daily Adjusted Rating and Kalman Optimized)
Developer: Kostya Medvedovsky (@kmedved)
Type: Bayesian Kalman filter projection model

METHODOLOGY:
1. Kalman Filter: State-space model treating player ability as latent variable
2. Prior: Historical base rates by position, age, draft position, college stats
3. Daily Update: Every game updates the player's estimated true talent
4. Bayesian Shrinkage: New players heavily shrunk toward prior, veterans toward recent data
5. Injury Adjustment: Automatically reduces projection after missed games

WHAT IT PREDICTS:
- Points, rebounds, assists, steals, blocks per game
- Minutes played projection
- Plus-minus impact (DPM — DARKO Plus-Minus)
- Win probability contribution
- Rest-of-season projections

DPM (DARKO Plus-Minus):
- Scale: same as EPM/LEBRON (per 100 possessions vs average)
- Updated DAILY — faster adaptation than other metrics
- Uses Kalman filter so new data is weighted optimally

USE CASES:
1. DFS (Daily Fantasy): Project tonight's stat line
2. Player Props: Model expected points/rebounds/assists vs betting line
3. Trade evaluation: Project player performance in new context
4. Draft: Project rookie development trajectory
5. Injury return: Estimate production after absence

ADVANTAGES:
- Daily updates = fastest-reacting model
- Transparent methodology (academic paper published)
- Free public access for basic projections
- Handles role changes, injuries, trades naturally via Kalman filter

LIMITATIONS:
- Does not use tracking data (box-score + on-off only)
- Projections can be noisy for low-minute players
- Defensive projections less reliable than offensive

WEBSITE: apanacea.io/darko
RECENT TOP DPM (2024-25):
1. Nikola Jokic
2. Shai Gilgeous-Alexander
3. Luka Doncic""",
        "source": "DARKO / Kostya Medvedovsky",
        "category": "advanced_analytics",
    },
    # ── Cleaning the Glass ────────────────────────────────────────────────────
    {
        "id": "analytics-ctg-deep",
        "text": """Cleaning the Glass — Premium NBA Analytics Platform
Founder: Ben Falk (former VP of Basketball Operations, Philadelphia 76ers)
Subscription: ~$20/month

UNIQUE FEATURES:
1. Garbage Time Filter: Removes blowout minutes (score differential > 25 in 4Q)
   - This alone changes many player/team rankings significantly
   - Some players' stats inflate in garbage time (bench players)
   - Contenders' bench stats deflate (they rest starters in blowouts)

2. Contextual Splits:
   - Home vs Away adjusted
   - Score margin buckets (close games, blowouts)
   - Pace-adjusted rates
   - Opponent-adjusted rates

3. Lineup Data:
   - Every 2-5 man lineup combination
   - On-off splits for every player
   - Lineup net rating per 100 possessions
   - Minutes played per lineup

4. Shot Charts by Zone:
   - Restricted area, paint (non-RA), mid-range, corner 3, above-break 3
   - Frequency + efficiency per zone
   - Percentile rankings vs league

5. Key Four Factors (per team, garbage-time filtered):
   - eFG% (Effective Field Goal %)
   - TOV% (Turnover Rate)
   - ORB% (Offensive Rebound Rate)
   - FT Rate (Free Throw Rate)

WHO USES IT:
- NBA front offices (most teams subscribe)
- Agents and player representatives
- Serious media (The Athletic, ESPN)
- Betting analysts and DFS professionals

WHY IT MATTERS FOR BETTING:
- Garbage-time filter reveals true team strength (not inflated by junk minutes)
- Lineup data shows real closing-5 quality
- Shot distribution predicts future efficiency better than raw shooting %

EXAMPLE INSIGHT:
A team might be 20th in raw eFG%, but 10th when filtering garbage time.
This means the market may undervalue them because public stats include blowout minutes.

WEBSITE: cleaningtheglass.com""",
        "source": "Cleaning the Glass / Ben Falk",
        "category": "advanced_analytics",
    },
    # ── Synergy Sports ────────────────────────────────────────────────────────
    {
        "id": "analytics-synergy-deep",
        "text": """Synergy Sports Play-Type Analytics (now Second Spectrum)
THE tool used by NBA coaching staffs for game preparation and scouting.
Used by: 30/30 NBA teams + college programs + international leagues

PLAY TYPE CLASSIFICATIONS:
1. Pick-and-Roll Ball Handler: Ball handler using a screen
2. Pick-and-Roll Roll Man: Screener rolling/popping after screen
3. Isolation: 1-on-1 play with space cleared
4. Post-Up: Back-to-basket play in the paint
5. Spot-Up: Catch-and-shoot or catch-and-drive from spot-up
6. Transition: Fast break and early offense
7. Cut: Off-ball cut to the basket
8. Off-Screen: Coming off a down screen or flare screen
9. Handoff: Dribble handoff plays
10. Putbacks: Offensive rebound putback
11. Miscellaneous: Unclassified plays

FOR EACH PLAY TYPE, SYNERGY TRACKS:
- Points Per Possession (PPP): efficiency of that play type
- Frequency: % of possessions using that play type
- Percentile Ranking: vs all NBA players/teams
- Turnover %: how often that play type leads to turnover
- And-1 %: how often it leads to foul + made basket
- Score %: how often the possession ends in points

DEFENSIVE EQUIVALENT:
- How well a player/team defends each play type
- Matchup data: Player A defending Player B in isolation PPP

BETTING APPLICATIONS:
1. Matchup analysis: Team A's offense type vs Team B's defense type
2. Pace prediction: Play type distribution predicts game pace
3. Player props: Usage in specific play types predicts stat lines
4. In-game betting: Real-time play type shifts

EXAMPLE:
Dallas Mavericks 2024: 25% of possessions = Luka isolation (97th percentile PPP)
Opponent game plan: Force Luka into PnR (he's 60th percentile in PnR)
Result: Dallas winning % correlates with Luka isolation frequency

DATA ACCESS:
- NBA.com/stats has partial Synergy data
- Full access requires team/media license
- Partial data on stats.nba.com tracking pages""",
        "source": "Synergy Sports / Second Spectrum",
        "category": "advanced_analytics",
    },
    # ── Player Tracking / Second Spectrum ─────────────────────────────────────
    {
        "id": "analytics-tracking-deep",
        "text": """NBA Player Tracking Data — Second Spectrum (official provider since 2021)
Previously: SportVU (STATS LLC) from 2013-2017, then Second Spectrum

CAMERA SYSTEM:
- 5+ cameras in each of the 30 NBA arenas
- Captures: 25 frames per second
- Tracks: x,y coordinates of all 10 players + ball
- Resolution: ~1 inch accuracy

RAW DATA CAPTURED:
- Player position (x,y) at 25fps
- Ball position (x,y,z) at 25fps — includes height for shot tracking
- Player speed (mph), acceleration
- Distance covered (miles per game)

DERIVED METRICS (publicly available on stats.nba.com):
1. SPEED & DISTANCE:
   - Average speed (mph): offense vs defense
   - Distance covered per game (miles)
   - Average speed in transition

2. TOUCHES:
   - Total touches per game
   - Front court touches
   - Time of possession (seconds per touch)
   - Dribbles per touch

3. SHOOTING:
   - Closest defender distance at shot release
   - Shot clock remaining at shot
   - Touch time before shooting (catch-and-shoot vs pull-up)
   - Contested vs uncontested shots

4. PASSING:
   - Potential assists
   - Passes made / received
   - Assist-to-pass %
   - Secondary assists

5. REBOUNDING:
   - Rebound chances (contested vs uncontested)
   - Rebounding distance from basket
   - Adjusted rebound rate

6. DEFENSE:
   - Matchup stats (who guards whom)
   - Defensive FG% vs opponent (DFGA, DFG%)
   - Contested shots per game
   - Deflections per game
   - Charges drawn

PROPRIETARY (team-only):
- Shot quality score (qSQ)
- Expected eFG% based on shot quality
- Screen effectiveness metrics
- Off-ball movement scores
- Defensive positioning grades

BETTING APPLICATIONS:
- Pace modeling from tracking data
- Shot quality differentials predict regression
- Defensive matchup data for player props
- Fatigue models from distance/speed data (back-to-backs)""",
        "source": "NBA / Second Spectrum — Player Tracking",
        "category": "advanced_analytics",
    },
    # ── PBP Stats ─────────────────────────────────────────────────────────────
    {
        "id": "analytics-pbpstats-deep",
        "text": """PBP Stats (Play-by-Play Stats) — by Ryan Davis
WEBSITE: pbpstats.com (free + premium tiers)

UNIQUE OFFERINGS:
1. Matchup Data:
   - Defensive matchups: Who guards whom, and their efficiency
   - Partial possessions: How matchups change within a possession
   - Switch tracking: When screens cause switches

2. Shot Quality Metrics:
   - qSQ (quantified Shot Quality): Expected eFG% based on shot location, defender distance, shot clock
   - xeFG%: Expected effective field goal % from shot quality
   - Diff: Actual eFG% minus xeFG% (luck vs skill indicator)

3. Lineup Finder:
   - Any 2-5 man combination stats
   - Lineup net rating with confidence intervals
   - Filtering: minimum minutes, date range, home/away

4. On/Off Court:
   - Player on vs off impact
   - Splits by lineup size (2-man, 3-man combos with player)

5. Possession Tracking:
   - Type of possession end (made shot, missed shot, turnover, free throw)
   - Second chance possessions
   - Transition possession rate

BETTING APPLICATIONS:
- Matchup-based modeling: Predict efficiency based on defensive matchups
- Shot quality regression: Players shooting above/below expected will regress
- Lineup optimization: Which lineups a coach should (and does) play
- Pace and possessions: Predict total possessions in a game

ADVANCED USE (for models):
- Export play-by-play data for your own models
- Build custom lineup net ratings
- Calculate pace estimates from possession counts
- Build defensive matchup matrices

FREE DATA: Game-level play-by-play stats
PREMIUM: Custom queries, bulk data export, advanced lineup data""",
        "source": "PBP Stats / Ryan Davis",
        "category": "advanced_analytics",
    },
]

def ingest_expert_analytics():
    """Return the expert analytics knowledge documents."""
    log(f"[ANALYTICS] Loading {len(EXPERT_ANALYTICS_KNOWLEDGE)} expert analytics documents")
    return EXPERT_ANALYTICS_KNOWLEDGE


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: TONY BLOOM / STARLIZARD STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

BLOOM_STRATEGY_KNOWLEDGE = [
    {
        "id": "bloom-starlizard-overview",
        "text": """Tony Bloom & Starlizard — The World's Most Successful Sports Bettor

BACKGROUND:
- Tony Bloom, born 1970, Brighton UK. "The Lizard"
- Cambridge math graduate, former poker professional
- Won millions at poker before transitioning to sports betting
- Founded Starlizard in ~2006 — sports betting consultancy
- Bought Brighton & Hove Albion FC in 2009 (now Premier League)
- Estimated annual turnover: $1B+ across all sports

STARLIZARD OPERATION:
- 160+ employees: data scientists, mathematicians, statisticians, former traders
- Headquarters: London + offices worldwide
- Covers: Football (primary), NBA, NFL, Tennis, Cricket, and more
- Operates as a "fund" — clients invest capital, Starlizard places bets on their behalf
- Average client ROI: estimated 8-15% annually (on sports betting!)

KEY PRINCIPLES:
1. MATHEMATICAL EDGE ONLY: Never place a bet without quantifiable positive expected value
2. VOLUME OVER SIZE: Thousands of small-edge bets > few large bets
3. SPEED: Get bets placed within seconds of line release (first-mover advantage)
4. AUTOMATION: Algorithms identify opportunities and place bets automatically
5. INFORMATION PROCESSING: Better analysis of PUBLIC information (not insider info)
6. DIVERSIFICATION: Bet across multiple sports, leagues, markets simultaneously
7. LONG-TERM THINKING: Evaluate over 10,000+ bets, not 10

COMPARISON TO HEDGE FUNDS:
- Sharpe Ratio: Starlizard estimated 2-3+ (comparable to top quant funds)
- Risk management: Kelly Criterion-based position sizing
- Drawdown management: Never risk more than defined % of bankroll
- Model validation: Continuous backtesting and out-of-sample testing""",
        "source": "Tony Bloom / Starlizard — Overview",
        "category": "betting_strategy",
    },
    # ── EV Calculations ───────────────────────────────────────────────────────
    {
        "id": "bloom-ev-calculations",
        "text": """Expected Value (EV) Calculations — Applied to NBA Betting

FUNDAMENTAL FORMULA:
EV = (Probability_Win * Profit_if_Win) - (Probability_Lose * Loss_if_Lose)

STEP-BY-STEP NBA EXAMPLE:
1. Your model says Lakers have 55% chance to win
2. Bookmaker odds: Lakers at 2.10 (decimal) = +110 American
3. Implied probability of 2.10 odds = 1/2.10 = 47.6%
4. YOUR probability (55%) > Implied (47.6%) = POSITIVE EV

EV CALCULATION:
EV = (0.55 * 1.10) - (0.45 * 1.00)
EV = 0.605 - 0.45 = +0.155
EV = +15.5% per dollar wagered

MINIMUM EDGE THRESHOLDS (professional standard):
- Moneyline: >= 3% edge (model prob vs implied prob)
- Spreads: >= 2% edge (smaller margin, higher volume)
- Totals: >= 2% edge (more predictable markets)
- Player Props: >= 5% edge (higher variance, lower limits)
- Parlays: >= 8% edge per leg (compounding error)
- Live betting: >= 4% edge (fast-moving, execution risk)

CONVERTING BETWEEN ODDS FORMATS:
- Decimal to Implied%: 1/decimal * 100
- American + to Implied%: 100 / (American + 100) * 100
- American - to Implied%: |American| / (|American| + 100) * 100
- Implied% to Fair Odds (no vig): Remove vig by normalizing both sides to 100%

REMOVING THE VIG (Juice):
Example: Lakers -110 / Celtics -110
- Implied: 52.38% + 52.38% = 104.76% (4.76% overround = the vig)
- True probability: 52.38/104.76 = 50.0% each (coin flip)
- Fair odds: 2.00 each (even money)

PROFESSIONAL TIP: Compare your model probability to the VIG-FREE implied probability,
not the raw bookmaker implied probability. This gives you the TRUE edge.""",
        "source": "Tony Bloom / EV Methodology",
        "category": "betting_strategy",
    },
    # ── Kelly Criterion ───────────────────────────────────────────────────────
    {
        "id": "bloom-kelly-criterion",
        "text": """Kelly Criterion — Bankroll Management for NBA Betting

THE KELLY FORMULA:
f* = (bp - q) / b

Where:
  f* = fraction of bankroll to bet
  b = decimal odds - 1 (net odds)
  p = your estimated probability of winning
  q = 1 - p (probability of losing)

NBA EXAMPLE:
- Model probability: 58% (p = 0.58, q = 0.42)
- Odds: 2.05 decimal (b = 1.05)
- Kelly: (1.05 * 0.58 - 0.42) / 1.05 = (0.609 - 0.42) / 1.05 = 0.189 / 1.05 = 18.0%

PROFESSIONAL MODIFICATIONS:

1. FRACTIONAL KELLY (CRITICAL — used by all pros):
   - Full Kelly is mathematically optimal but TOO VOLATILE
   - 1/4 Kelly: bet 25% of Kelly suggestion (most common among pros)
   - 1/3 Kelly: slightly more aggressive
   - 1/2 Kelly: maximum aggression for pros
   - Starlizard reportedly uses ~1/4 Kelly

2. SIMULTANEOUS BETS:
   - When placing multiple bets same day, Kelly must be adjusted downward
   - Independent bets: reduce each by sqrt(n) where n = number of bets
   - Correlated bets (same game): treat as single bet

3. BANKROLL RULES:
   - MAXIMUM single bet: 3% of bankroll (even if Kelly says more)
   - MAXIMUM daily exposure: 15% of bankroll
   - MINIMUM bankroll: 100 units for adequate Kelly sizing
   - STOP LOSS: If bankroll drops 30%, reassess all models

4. KELLY FOR PLAYER PROPS:
   - Use 1/5 Kelly (higher variance on props)
   - Max bet: 1% of bankroll per prop
   - Correlated props (same player multi): 0.5% max

COMMON MISTAKES:
- Using full Kelly → too volatile, risk of ruin increases dramatically
- Not adjusting for simultaneous bets → overexposure
- Overestimating edge → Kelly amplifies errors (be conservative with p)
- Not accounting for bookmaker limits → can't always get full Kelly down

ADVANCED: GROWTH-OPTIMAL BETTING
Kelly maximizes long-term geometric growth rate of bankroll.
With perfect probability estimates, Kelly doubles money in minimum expected time.
With imperfect estimates (reality), fractional Kelly provides much smoother equity curve.""",
        "source": "Kelly Criterion / Bankroll Management",
        "category": "betting_strategy",
    },
    # ── CLV (Closing Line Value) ──────────────────────────────────────────────
    {
        "id": "bloom-clv-tracking",
        "text": """CLV (Closing Line Value) — The Single Best Predictor of Long-Term Betting Success

WHAT IS CLV:
The difference between the odds you BET AT and the CLOSING LINE (final odds before game starts).
Closing line = most efficient/accurate line because it incorporates ALL information.

WHY CLV MATTERS:
- If you consistently beat the closing line, you WILL be profitable long-term
- If you don't beat the closing line, you will LOSE long-term regardless of recent results
- Pinnacle's closing line is the industry benchmark (sharpest book)
- CLV > 0 over 1000+ bets = you have a genuine edge

HOW TO CALCULATE CLV:

Method 1 — Simple CLV:
CLV = Your_Odds - Closing_Odds
Example: You bet Lakers at 2.15, line closes at 2.00
CLV = 2.15 - 2.00 = +0.15 (positive = good)

Method 2 — Percentage CLV (better):
CLV% = (Your_Implied_Prob - Closing_Implied_Prob) / Closing_Implied_Prob * 100
Example: You bet at 2.15 (46.5%) → closes at 2.00 (50.0%)
CLV% = (50.0 - 46.5) / 50.0 * 100 = +7.0%

Method 3 — No-Vig CLV (most accurate):
1. Remove vig from closing line to get true closing probability
2. Compare to your bet's implied probability
3. This eliminates the bookmaker margin from the comparison

TARGET CLV VALUES:
- +2% average CLV = profitable (good recreational bettor)
- +3-4% average CLV = strong (semi-professional)
- +5%+ average CLV = elite (professional/syndicate level)
- Starlizard reportedly achieves +3-5% average CLV

CLV TRACKING SYSTEM:
1. Record EVERY bet: timestamp, odds, stake, bookmaker
2. Record closing line at Pinnacle for each bet
3. Calculate CLV for each bet
4. Track 30-day rolling CLV average
5. If rolling CLV drops below +1%, reduce stake sizes
6. If rolling CLV is negative for 500+ bets, the model is broken

CLV vs RESULTS:
- Short term (100 bets): Results = mostly luck, CLV = more signal
- Medium term (1000 bets): Results start reflecting edge, CLV is clear signal
- Long term (5000+ bets): Results converge to CLV prediction
- ALWAYS trust CLV over results for model evaluation

NBA SPECIFIC CLV TIPS:
- NBA lines move FAST (high liquidity) — bet early for best CLV
- Injury news moves lines 2-4 points — be first to react
- Back-to-back information is priced in by closing — get there first
- Props have wider closing line variance — easier to beat CLV""",
        "source": "CLV (Closing Line Value) Tracking",
        "category": "betting_strategy",
    },
    # ── Line Shopping ─────────────────────────────────────────────────────────
    {
        "id": "bloom-line-shopping",
        "text": """Line Shopping — The Easiest Edge in Sports Betting

CONCEPT:
Different bookmakers offer different odds on the same event.
Shopping for the best line across ALL available bookmakers = guaranteed higher ROI.

NBA EXAMPLE:
Event: Lakers vs Celtics — Lakers spread -3.5
- DraftKings: -3.5 (-110)    = needs 52.4% to break even
- FanDuel: -3.5 (-108)       = needs 51.9% to break even
- Pinnacle: -3.5 (-105)      = needs 51.2% to break even  ← BEST LINE
- BetMGM: -3.5 (-112)        = needs 52.8% to break even
- Betclic: -3.5 (1.91)       = needs 52.4% to break even
- Winamax: -3.5 (1.92)       = needs 52.1% to break even

Difference between worst (-112) and best (-105): 1.6% edge
Over 1000 bets: this 1.6% difference = significant profit vs loss

SHARP vs SOFT BOOKS:
SHARP (market-making, accept large bets, tight lines):
1. Pinnacle — THE benchmark. Sharpest lines in the world.
2. Circa Sports — US sharp book
3. Bookmaker.eu — Accepts high limits
4. BETDAQ/Matchbook — Betting exchanges (no built-in edge)
5. Betfair Exchange — Largest exchange globally

SOFT (recreational focus, wider margins, limit sharps quickly):
1. DraftKings — Limits winning bettors aggressively
2. FanDuel — Same, but slightly slower to limit
3. BetMGM — Limits fast
4. Caesars — Limits fastest of all US books
5. Unibet, bet365 — EU soft books

FRENCH BOOKMAKERS (Paris Sportifs):
1. Parions Sport (FDJ) — State-owned, widest margins, rarely limits
2. Winamax — Most popular in France, decent odds
3. Betclic — Second most popular, good for NBA
4. PMU — Historically horse racing, expanding to NBA
5. ZEBet — Smaller, sometimes best odds on NBA props
6. NetBet — EU-licensed, available in France
7. Unibet — Available via French license
8. France Pari — Smaller, occasional value on NBA

LINE SHOPPING STRATEGY:
1. Have accounts at 8-12 bookmakers (mix of sharp + soft)
2. Before EVERY bet, check odds at ALL books
3. Use odds comparison sites: oddschecker.com, oddsportal.com
4. Target: always bet at odds >= Pinnacle closing line
5. Maintain separate bankrolls per book (or use an aggregator)
6. Track which books give best NBA lines consistently

TOOLS:
- OddsJam, PositiveEV, OddsShopper — automated line shopping
- The Odds API (api.the-odds-api.com) — programmatic odds comparison
- Action Network — odds tracking + line movement

ROI IMPACT:
A bettor with 2% edge who shops lines can add +1-2% extra ROI.
This turns a marginal winner into a solid professional.""",
        "source": "Line Shopping Strategy",
        "category": "betting_strategy",
    },
    # ── Sharp vs Square Indicators ────────────────────────────────────────────
    {
        "id": "bloom-sharp-square",
        "text": """Sharp vs Square Money Indicators — Reading NBA Line Movements

SHARP MONEY (Professional / Smart Money):
- Large bets placed at opening or on steam moves
- Typically bet early when lines are soft
- Focus on closing line value (CLV)
- Bet into bad numbers, against public sentiment
- Use mathematical models, not gut feeling

SQUARE MONEY (Recreational / Public Money):
- Smaller bets, higher volume
- Bet closer to game time
- Influenced by media narratives, recent results, star players
- Overvalue favorites, overs, primetime games
- Chase recent trends (hot/cold streaks)

LINE MOVEMENT INDICATORS:

1. REVERSE LINE MOVEMENT (RLM):
   - Line moves AGAINST the side receiving more bets
   - Example: 70% of bets on Lakers -5, but line moves to Lakers -4
   - This means SHARP money is on the other side (opponent +4)
   - Very strong sharp indicator

2. STEAM MOVES:
   - Sudden, significant line movement across ALL bookmakers simultaneously
   - Caused by syndicate/sharp money hitting multiple books at once
   - Usually 0.5-1.5 points in minutes
   - Following steam moves is a viable strategy (but speed is critical)

3. OPENING LINE vs CLOSING LINE:
   - Opening: set by market makers with models
   - Movement: driven by money (sharp + public)
   - Closing: most efficient/accurate line
   - Sharp bettors move lines early, public moves lines late

4. TOTALS (Over/Under):
   - Public HEAVILY favors overs (they want to see scoring)
   - If an over total goes DOWN despite heavy over action = sharp money on under
   - NBA unders are historically undervalued by the public

5. TICKET COUNT vs HANDLE:
   - Ticket count: number of individual bets (mostly public)
   - Handle: total dollar amount wagered (sharp money dominates)
   - When 75% of tickets are on Side A but 60% of HANDLE is on Side B = sharp on Side B

NBA-SPECIFIC PATTERNS:
- Sharp money hammers back-to-back road underdogs (public fades tired teams)
- Sharp money takes under on nationally televised games (public bets over for excitement)
- Sharp money bets AGAINST teams on 3+ game winning streaks (mean reversion)
- Sharp money bets road teams getting 7+ points (overreaction to home court)
- Monday/Tuesday NBA unders: fatigue from weekend games, lower pace""",
        "source": "Sharp vs Square Money Indicators",
        "category": "betting_strategy",
    },
]

def ingest_bloom_strategy():
    """Return Tony Bloom / Starlizard strategy knowledge documents."""
    log(f"[BLOOM] Loading {len(BLOOM_STRATEGY_KNOWLEDGE)} strategy documents")
    return BLOOM_STRATEGY_KNOWLEDGE


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: MATHEMATICAL MODELS APPLIED TO NBA BETTING
# ══════════════════════════════════════════════════════════════════════════════

MATH_MODELS_KNOWLEDGE = [
    # ── Poisson Regression ────────────────────────────────────────────────────
    {
        "id": "math-poisson-nba",
        "text": """Poisson Regression for NBA Totals Betting

CONCEPT:
Model the scoring distribution of each team to predict game totals (over/under).
While basketball scores aren't perfectly Poisson (they're slightly over-dispersed),
Poisson-based models still provide excellent predictions for totals.

NBA APPLICATION:
1. Estimate each team's expected points scored in a game
2. Adjust for opponent defensive strength
3. Adjust for pace (possessions per game)
4. Sum expected points for game total prediction
5. Compare to bookmaker total → bet when edge exists

STEP-BY-STEP MODEL:

Step 1 — Offensive Rating (points per 100 possessions):
ORtg_team = Team_Points / Team_Possessions * 100

Step 2 — Defensive Rating:
DRtg_team = Opponent_Points / Team_Possessions * 100

Step 3 — Expected Possessions in matchup:
Expected_Pace = (Pace_A + Pace_B) / 2 * Pace_Factor
Pace_Factor accounts for game context (playoffs = slower, back-to-back = slower)

Step 4 — Expected Points:
Team_A_Expected = ORtg_A * DRtg_B / League_Avg_DRtg * Expected_Pace / 100
Team_B_Expected = ORtg_B * DRtg_A / League_Avg_DRtg * Expected_Pace / 100

Step 5 — Total:
Predicted_Total = Team_A_Expected + Team_B_Expected

Step 6 — Edge:
Edge = Predicted_Total - Bookmaker_Total
If Edge > 3 points → strong signal for over or under

ADJUSTMENTS (Critical):
- Home court: +2.5 points for home team (declining in recent years: ~+1.5 in 2024)
- Back-to-back: -3 to -5 points for tired team
- Altitude: Denver home = +1.5 points (opponents struggle at altitude)
- Rest days: +1 point per extra rest day vs opponent
- Travel: cross-country = -1 point
- Injuries: position-specific impact (star guard > rotation big)
- Pace matchup: two fast teams = total goes up; two slow teams = total goes down

ADVANCED — Negative Binomial:
Since NBA scoring is over-dispersed (variance > mean), Negative Binomial regression
is theoretically more appropriate than Poisson. In practice, difference is small.
Use NB when modeling quarter-by-quarter scoring (more variance at quarter level).

PYTHON IMPLEMENTATION:
- statsmodels.GLM with Poisson/NB family
- Features: ORtg, DRtg, Pace, rest days, travel, injuries, home/away
- Training data: last 3 seasons of games
- Validation: cross-validate by season (no data leakage)""",
        "source": "Poisson Regression / NBA Totals Model",
        "category": "mathematical_models",
    },
    # ── Elo Ratings ───────────────────────────────────────────────────────────
    {
        "id": "math-elo-nba",
        "text": """Elo Ratings for NBA Team Strength — Applied to Betting

ORIGIN:
Arpad Elo created the Elo system for chess. FiveThirtyEight adapted it for NBA.
Simple, elegant, and surprisingly powerful for predicting NBA games.

THE FORMULA:
New_Rating = Old_Rating + K * (Actual - Expected)

Where:
- K = update factor (larger K = more responsive to recent games)
- Actual = 1 (win) or 0 (loss)
- Expected = 1 / (1 + 10^((Opponent_Elo - Team_Elo) / 400))

NBA-SPECIFIC ELO:

1. STARTING RATINGS:
   - New season: carry over 75% of previous season's rating + 25% mean reversion
   - League average: 1500
   - Championship contender: 1700+
   - Tank/rebuild: 1300-

2. K-FACTOR (learning rate):
   - FiveThirtyEight NBA K = 20
   - Higher K early in season (first 15 games), lower later
   - Margin of victory adjustment: K_MOV = K * log(MOV + 1) * (2.2 / ((Elo_diff * 0.001) + 2.2))

3. HOME COURT ADVANTAGE:
   - Add +100 Elo to home team (equivalent to ~3.5 point spread)
   - Declining trend: modern NBA HCA is closer to +70 Elo

4. CONVERTING ELO TO WIN PROBABILITY:
   Win_Prob = 1 / (1 + 10^((Opponent_Elo - Team_Elo) / 400))
   Example: Team at 1700 vs opponent at 1500 (at home: 1700 + 100 = 1800)
   Win_Prob = 1 / (1 + 10^((1500-1800)/400)) = 1 / (1 + 10^(-0.75)) = 84.9%

5. CONVERTING ELO TO SPREAD:
   Spread = (Team_Elo - Opponent_Elo) / 28
   Example: 1800 vs 1500 = 300/28 = 10.7 point spread

6. NBA ELO TIERS (FiveThirtyEight scale):
   - 1750+: All-time great team (2017 Warriors: ~1850)
   - 1650-1750: Title contender
   - 1550-1650: Playoff team
   - 1450-1550: Bubble team / play-in
   - 1350-1450: Lottery team
   - <1350: Historic bad team

BETTING APPLICATION:
1. Calculate your Elo-based win probability
2. Convert bookmaker odds to implied probability
3. If Elo_prob > Implied_prob + margin → positive EV bet
4. Track prediction accuracy over time (FiveThirtyEight NBA Elo: ~68% accuracy)

LIMITATIONS:
- Doesn't account for injuries (must adjust manually)
- Doesn't distinguish roster changes during season
- Margin of victory adjustment can overreact to blowouts
- Doesn't capture matchup-specific advantages

ENHANCEMENTS:
- Add injury adjustment: reduce Elo by player's RPM * minutes lost
- Pace-adjusted: faster teams have higher variance
- Playoff adjustment: different dynamics than regular season""",
        "source": "Elo Rating System / NBA Application",
        "category": "mathematical_models",
    },
    # ── Monte Carlo Simulation ────────────────────────────────────────────────
    {
        "id": "math-monte-carlo-nba",
        "text": """Monte Carlo Simulation for NBA Series & Season Outcomes

CONCEPT:
Simulate thousands (10,000+) of possible outcomes to estimate probabilities.
Essential for: playoff series prediction, season win totals, championship odds.

NBA APPLICATIONS:

1. PLAYOFF SERIES SIMULATION:
   Input: Win probability for each game (adjust for home court)
   Process: Simulate 7-game series 10,000+ times
   Output: Probability of winning series, expected games, sweep probability

   Example — Lakers (55% win prob per game):
   - Game-by-game simulation with home court pattern (2-2-1-1-1)
   - Home games: 60% win prob, Away games: 50%
   - After 10,000 sims: Lakers win series 61.3% of the time
   - Series length distribution: 4 games (12%), 5 (24%), 6 (32%), 7 (32%)

2. SEASON WIN TOTAL SIMULATION:
   Input: Elo ratings for all 30 teams, schedule
   Process: Simulate all 82 games for each team, 10,000+ times
   Output: Win distribution, playoff probability, seeding odds

3. CHAMPIONSHIP SIMULATION:
   Input: Current Elo + remaining schedule + playoff bracket probabilities
   Process: Simulate rest of season → simulate all playoff rounds
   Output: Championship probability for each team

4. PLAYER PROP SIMULATION:
   Input: Player's statistical distributions (mean, std dev per stat)
   Process: Sample from distribution (negative binomial for counting stats)
   Output: Probability of over/under specific lines

IMPLEMENTATION:

```python
import random

def simulate_series(p_home, p_away, n_sims=10000):
    '''Simulate NBA playoff series (2-2-1-1-1 format)'''
    wins = 0
    home_pattern = [True, True, False, False, True, False, True]  # Team A perspective

    for _ in range(n_sims):
        a_wins, b_wins = 0, 0
        for game_idx in range(7):
            if a_wins == 4 or b_wins == 4:
                break
            is_home = home_pattern[game_idx]
            p = p_home if is_home else p_away
            if random.random() < p:
                a_wins += 1
            else:
                b_wins += 1
        if a_wins == 4:
            wins += 1

    return wins / n_sims
```

VARIANCE AND SAMPLE SIZE:
- 1,000 simulations: ~3% error in probability estimates
- 10,000 simulations: ~1% error
- 100,000 simulations: ~0.3% error
- For betting decisions: 10,000 minimum

BETTING APPLICATION:
1. Simulate series outcome probabilities
2. Compare to bookmaker series prices
3. If model_prob > implied_prob + 3% → bet
4. Also: simulate game totals, player props, first-half lines

ADVANCED TECHNIQUES:
- Correlated outcomes: If one game's margin affects next game's momentum
- Injury probability: Include chance of injury reducing team strength mid-series
- Fatigue modeling: Back-to-back travel reduces win probability in later games""",
        "source": "Monte Carlo Simulation / NBA",
        "category": "mathematical_models",
    },
    # ── Bayesian Updating ─────────────────────────────────────────────────────
    {
        "id": "math-bayesian-nba",
        "text": """Bayesian Updating for In-Season NBA Projections

CONCEPT:
Start with a PRIOR belief about each team/player's ability, then UPDATE
as new game data arrives. The combination = POSTERIOR estimate.

BAYES THEOREM:
P(Ability | Data) = P(Data | Ability) * P(Ability) / P(Data)

NBA APPLICATIONS:

1. TEAM STRENGTH UPDATING:
   - Prior (preseason): Last year's rating + offseason changes (trades, draft, coaching)
   - Likelihood: This season's game results
   - Posterior: Updated team rating

   Early season (10 games): Prior dominates (preseason ranking matters most)
   Mid-season (40 games): 50/50 between prior and current data
   Late season (70+ games): Current data dominates

2. PLAYER PERFORMANCE UPDATING:
   - Prior: Career averages, age curve, role projection
   - Likelihood: Current season stats
   - Posterior: True talent estimate

   Example — Player shooting 45% from 3 through 20 games (career 37%):
   Prior: 37% (strong prior from career data)
   Likelihood: 45% (small sample)
   Posterior: ~39% (moves toward current data, but prior pulls it back)
   As games increase, posterior moves closer to current season data.

3. INJURY RETURN PROJECTIONS:
   - Prior: Historical recovery curves for injury type
   - Likelihood: Recent game performance post-injury
   - Posterior: Expected performance level

PRACTICAL IMPLEMENTATION:

Beta-Binomial for shooting:
Prior: Beta(alpha=74, beta=126) for 37% shooter (equivalent to ~200 shot prior)
After 50 makes in 100 attempts this season:
Posterior: Beta(74+50, 126+50) = Beta(124, 176) → 41.3% estimate
The 200-shot prior "anchors" the estimate toward career average.

Adjusting Prior Strength:
- Young player (2 NBA seasons): weak prior (equivalent to ~100 shots)
- Veteran (8+ seasons): strong prior (equivalent to ~500+ shots)
- After major injury/surgery: weaken prior (ability may have changed)
- After trade to new team: weaken prior for team context stats

BETTING APPLICATION:
1. Build preseason priors for all 30 teams (power ratings)
2. After each game, update using Bayesian methods
3. The posterior = your team rating for betting
4. Early season: be cautious (high uncertainty in posterior)
5. Mid-season: posterior is well-calibrated → bet with more confidence
6. Key: The market also updates — you need to update FASTER or BETTER

ADVANTAGES OVER PURE FREQUENTIST:
- Handles small samples naturally (doesn't overreact to 5-game streaks)
- Incorporates domain knowledge (preseason rankings have information!)
- Provides uncertainty estimates (credible intervals, not just point estimates)
- Self-calibrating: as data accumulates, prior influence shrinks automatically

PITFALL:
If your prior is wrong (bad preseason assessment), it can bias estimates for weeks.
Solution: Use mixture priors or increase data weighting early in season.""",
        "source": "Bayesian Updating / NBA Projections",
        "category": "mathematical_models",
    },
    # ── Edge Detection ────────────────────────────────────────────────────────
    {
        "id": "math-edge-detection",
        "text": """Edge Detection — Finding Model Probability vs Market Implied Probability Gaps

THE CORE QUESTION:
"Where does my model disagree with the market, and is my model right?"

PROCESS:

Step 1 — Build Your Model:
- Team rating model (Elo, power ratings, or more sophisticated)
- Incorporate: pace, efficiency, injuries, rest, travel, matchup history
- Output: Win probability for each team in each game

Step 2 — Get Market Probabilities:
- Scrape odds from Pinnacle (sharpest book) or use The Odds API
- Convert to implied probabilities
- Remove the vig to get "true market probability"

Step 3 — Calculate Edge:
Edge = Model_Probability - Market_Probability

Example:
- Your model: Lakers 58.0% to win
- Pinnacle no-vig: Lakers 53.0%
- Edge: +5.0% (significant — above 3% threshold)

Step 4 — Validate Edge is Real:
- Is this a consistent pattern or one-off?
- What is driving the disagreement? (injury news? rest? travel?)
- Does the edge remain if you adjust model parameters by +/-2%?
- Is the edge on a market you've historically been accurate on?

Step 5 — Size the Bet:
- Use Kelly Criterion with the edge size
- Higher confidence in edge → larger fraction of Kelly
- Novel/unusual edge → smaller fraction of Kelly

EDGE CATEGORIES IN NBA:

1. REST/SCHEDULE EDGES (2-4% edge):
   - 3-in-4-nights vs well-rested team
   - 4th game in 5 nights (very strong under signal)
   - Cross-country travel (West Coast team playing Eastern time)

2. INJURY EDGES (3-8% edge but short-lived):
   - Late injury news (star player out, line hasn't moved yet)
   - Returning player (market often overvalues return from long absence)
   - Subtle injuries: not on injury report but playing hurt

3. MATCHUP EDGES (1-3% edge):
   - Style matchup: fast team vs slow team (total prediction)
   - Defensive matchup: team lacks perimeter defense vs 3-point heavy offense
   - Size matchup: small-ball team vs dominant post player

4. PUBLIC PERCEPTION EDGES (1-3% edge):
   - "Public loves the Lakers" — line inflated by casual money
   - National TV games: public bets overs, favorites
   - Teams on winning/losing streaks (mean reversion)

5. MARKET STRUCTURE EDGES (1-2% edge):
   - Line shopping across 10+ books for best price
   - Soft book vs sharp book discrepancies
   - Early-week lines (less efficient) vs closing lines

TRACKING YOUR EDGES:
| Date | Game | Edge Type | Edge Size | Bet | Result | CLV |
|------|------|-----------|-----------|-----|--------|-----|
Track every bet in this format. After 500+ bets, analyze:
- Which edge types are profitable?
- Which edge types are you overestimating?
- Optimal minimum edge threshold by market type?

KEY INSIGHT:
You don't need to be right on every game. You need to be SYSTEMATICALLY right
on the games where you have an edge. The non-edge games, you skip.
Professional bettors skip 80-90% of games — they only bet with genuine edge.""",
        "source": "Edge Detection / Model vs Market",
        "category": "mathematical_models",
    },
    # ── Regression Models ─────────────────────────────────────────────────────
    {
        "id": "math-regression-nba",
        "text": """Regression Models for NBA Prediction

LINEAR REGRESSION FOR SPREADS:
Dependent variable: Point differential (home score - away score)
Independent variables:
- Net rating difference (ORtg - DRtg for each team)
- Pace differential
- Home court advantage indicator
- Rest days differential
- Travel distance
- Injury impact score
- Recent form (last 10 games ATS record)

LOGISTIC REGRESSION FOR WIN PROBABILITY:
Dependent variable: Win (1) or Loss (0)
Same features as linear regression
Output: Probability of home team winning
More appropriate than linear regression for binary outcomes.

RIDGE/LASSO REGRESSION:
- Regularization prevents overfitting on correlated features
- Ridge (L2): shrinks coefficients but keeps all features
- Lasso (L1): sets some coefficients to zero (feature selection)
- Elastic Net: combines both
- NBA application: Many correlated features (ORtg, eFG%, TS% are correlated)

RANDOM FOREST / GRADIENT BOOSTING:
- Non-linear relationships: fatigue effects may not be linear
- Interaction effects: back-to-back + travel = worse than sum of parts
- XGBoost/LightGBM popular for NBA prediction models
- Feature importance tells you which factors matter most

KEY FEATURES FOR NBA PREDICTION MODELS:

Team-level (strongest predictors):
1. Net Rating (offensive - defensive rating per 100 poss): r=0.85 with wins
2. eFG% differential: r=0.78 with wins
3. Turnover rate differential: r=0.45 with wins
4. Free throw rate differential: r=0.35 with wins
5. Offensive rebound rate: r=0.30 with wins

Context features:
6. Days of rest (0, 1, 2, 3+)
7. Home/Away
8. Distance traveled from last game
9. Time zone changes
10. Altitude change (to/from Denver)

Player-level:
11. Missing player impact (RPM or RAPTOR WAR of absent players)
12. Minutes distribution change due to injuries
13. Lineup net rating of expected starting 5

VALIDATION:
- Never train and test on same season data
- Use walk-forward validation (train on seasons 1-3, test on season 4, etc.)
- Out-of-sample accuracy goal: 68-72% for ATS, 72-76% for moneyline
- Compare to closing line accuracy (~73% for Pinnacle)
- If your model < 68% accuracy, it's not ready for betting

OVERFITTING WARNINGS:
- In-sample accuracy of 85%+ with out-of-sample of 65% = overfit
- Too many features relative to sample size = overfit
- Regularization (ridge/lasso) or feature selection to prevent this
- Rule of thumb: need 20-50 observations per feature""",
        "source": "Regression Models / NBA Prediction",
        "category": "mathematical_models",
    },
]

def ingest_math_models():
    """Return mathematical models knowledge documents."""
    log(f"[MATH] Loading {len(MATH_MODELS_KNOWLEDGE)} mathematical models documents")
    return MATH_MODELS_KNOWLEDGE


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 6: EXA.AI WEB CONTENT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

EXA_SEARCH_QUERIES = [
    # Expert analytics sources
    {"query": "NBA RAPTOR ratings player rankings 2024-25 season", "category": "analytics_current"},
    {"query": "EPM estimated plus-minus NBA player rankings 2025", "category": "analytics_current"},
    {"query": "LEBRON metric BBall Index NBA player impact 2025", "category": "analytics_current"},
    {"query": "DARKO projections NBA player daily ratings today", "category": "analytics_current"},
    {"query": "Cleaning the Glass NBA analytics garbage time filter insights", "category": "analytics_current"},
    {"query": "Synergy Sports NBA play type data pick and roll isolation spot up", "category": "analytics_current"},
    {"query": "NBA Second Spectrum tracking data player speed distance 2025", "category": "analytics_current"},
    {"query": "PBP Stats NBA matchup data defensive assignments 2025", "category": "analytics_current"},

    # Betting strategy / professional
    {"query": "NBA professional sports betting strategy closing line value CLV", "category": "betting_professional"},
    {"query": "Tony Bloom Starlizard sports betting operation methodology", "category": "betting_professional"},
    {"query": "NBA Kelly Criterion bankroll management sports betting mathematics", "category": "betting_professional"},
    {"query": "sharp vs square NBA betting money line movement indicators", "category": "betting_professional"},
    {"query": "NBA expected value EV positive betting edge detection model", "category": "betting_professional"},
    {"query": "NBA live betting in-game strategy market efficiency models", "category": "betting_professional"},
    {"query": "NBA player props betting model statistical projection", "category": "betting_professional"},

    # Mathematical models
    {"query": "NBA Elo rating system team strength prediction model FiveThirtyEight", "category": "math_models"},
    {"query": "Poisson regression NBA totals prediction over under model", "category": "math_models"},
    {"query": "Monte Carlo simulation NBA playoff series championship odds", "category": "math_models"},
    {"query": "Bayesian statistics NBA player projection regression to mean", "category": "math_models"},
    {"query": "machine learning NBA game prediction XGBoost neural network", "category": "math_models"},
    {"query": "NBA power ratings algorithm Sagarin Massey BPI methodology", "category": "math_models"},

    # Historical / reference
    {"query": "NBA all-time greatest players statistical ranking GOAT debate analysis", "category": "historical"},
    {"query": "NBA championship history dynasty analysis Lakers Celtics Warriors Bulls", "category": "historical"},
    {"query": "NBA salary cap analytics team building strategy luxury tax", "category": "team_building"},
    {"query": "NBA Draft analytics prospect evaluation statistical model bust probability", "category": "draft"},
    {"query": "NBA three-point revolution analytics spacing efficiency modern basketball", "category": "tactical"},
    {"query": "NBA clutch time statistics fourth quarter performance under pressure", "category": "clutch"},

    # French content / Paris sportifs
    {"query": "paris sportifs NBA strategie pronostics basketball analyse", "category": "paris_sportifs"},
    {"query": "cotes NBA bookmakers francais Winamax Betclic Parions Sport analyse", "category": "paris_sportifs"},
    {"query": "NBA analytics avancees statistiques impact joueurs analyse", "category": "paris_sportifs_analytics"},
]

def search_exa_expert(query, num_results=5, include_domains=None, category="web_content"):
    """Search for expert NBA content via Exa.AI."""
    if not EXA_API_KEY:
        return []

    payload = {
        "query": query,
        "numResults": num_results,
        "useAutoprompt": True,
        "type": "auto",
        "contents": {"text": {"maxCharacters": 3000}},
    }
    if include_domains:
        payload["includeDomains"] = include_domains

    resp, status = http_post(
        "https://api.exa.ai/search",
        payload,
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        timeout=30,
    )

    if "error" in resp:
        log(f"Exa search error for '{query[:50]}': {resp['error']}", "WARN")
        return []

    results = resp.get("results", [])
    documents = []
    for r in results:
        text = r.get("text", "")
        if not text or len(text) < 80:
            continue
        url = r.get("url", "")
        doc_id = f"exa-{hashlib.md5((url + query).encode()).hexdigest()[:12]}"
        documents.append({
            "id": doc_id,
            "text": f"[{r.get('title', 'Web Content')}]\nSource: {url}\n\n{text[:3000]}",
            "source": url,
            "title": r.get("title", ""),
            "category": category,
            "metadata": {"search_query": query},
        })

    return documents

def fetch_exa_expert_content():
    """Run all Exa.AI expert content searches."""
    if not EXA_API_KEY:
        log("No EXA_API_KEY — skipping Exa.AI expert content", "WARN")
        return []

    all_docs = []
    for item in EXA_SEARCH_QUERIES:
        docs = search_exa_expert(
            item["query"],
            num_results=5,
            category=item.get("category", "web_content"),
        )
        all_docs.extend(docs)
        log(f"  Exa: '{item['query'][:50]}...' → {len(docs)} results", "DEBUG")
        time.sleep(1.5)  # Rate limit

    log(f"[EXA] {len(all_docs)} expert content documents from {len(EXA_SEARCH_QUERIES)} queries")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 7: BRAVE SEARCH (complementary web content)
# ══════════════════════════════════════════════════════════════════════════════

BRAVE_QUERIES = [
    "NBA 2024-25 season advanced statistics leaders RAPTOR EPM",
    "NBA betting closing line value professional strategy 2025",
    "NBA player tracking data Second Spectrum analytics",
    "NBA Elo ratings power rankings current season",
    "NBA Draft 2025 mock draft analytics big board",
    "Paris sportifs NBA cotes analyses expert basketball",
]

def fetch_brave_content():
    """Search for NBA content via Brave Search API."""
    if not BRAVE_API_KEY:
        log("No BRAVE_API_KEY — skipping Brave Search", "WARN")
        return []

    all_docs = []
    for query in BRAVE_QUERIES:
        url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count=5"
        resp, status = http_get(
            url,
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            timeout=15,
        )

        if "error" in resp:
            log(f"Brave search error: {resp['error']}", "WARN")
            continue

        results = resp.get("web", {}).get("results", [])
        for r in results:
            title = r.get("title", "")
            description = r.get("description", "")
            url_result = r.get("url", "")
            text = f"{title}\n{description}"
            if len(text) < 50:
                continue

            doc_id = f"brave-{hashlib.md5(url_result.encode()).hexdigest()[:12]}"
            all_docs.append({
                "id": doc_id,
                "text": f"[{title}]\nSource: {url_result}\n\n{description}",
                "source": url_result,
                "title": title,
                "category": "web_content_brave",
                "metadata": {"search_engine": "brave", "query": query},
            })

        time.sleep(1)

    log(f"[BRAVE] {len(all_docs)} results from {len(BRAVE_QUERIES)} queries")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_state():
    """Load ingestion state for daemon mode."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "total_ingested": 0,
        "total_upserted": 0,
        "cycles": 0,
        "last_run": None,
        "source_counts": {},
        "errors": [],
    }

def save_state(state):
    """Persist ingestion state."""
    try:
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    except Exception as e:
        log(f"Failed to save state: {e}", "ERROR")


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(mode="full", skip_pinecone=False):
    """
    Run the full expert ingestion pipeline.

    Modes:
      full     — all sources
      knowledge — built-in knowledge bases only (no API calls)
      live     — live odds + Exa/Brave searches only (API calls)
      odds     — live odds only
      bbref    — Basketball Reference via Exa only
      exa      — Exa expert content only
    """
    all_docs = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    source_counts = {}

    log(f"\n{'='*70}")
    log(f"NBA EXPERT INGESTION — {ts} — Mode: {mode}")
    log(f"{'='*70}")

    # ── 1. Expert Analytics Knowledge ─────────────────────────────────────
    if mode in ("full", "knowledge"):
        docs = ingest_expert_analytics()
        all_docs.extend(docs)
        source_counts["expert_analytics"] = len(docs)

    # ── 2. Tony Bloom / Starlizard Strategy ───────────────────────────────
    if mode in ("full", "knowledge"):
        docs = ingest_bloom_strategy()
        all_docs.extend(docs)
        source_counts["bloom_strategy"] = len(docs)

    # ── 3. Mathematical Models ────────────────────────────────────────────
    if mode in ("full", "knowledge"):
        docs = ingest_math_models()
        all_docs.extend(docs)
        source_counts["math_models"] = len(docs)

    # ── 4. Live Betting Odds ──────────────────────────────────────────────
    if mode in ("full", "live", "odds"):
        docs = fetch_live_odds()
        all_docs.extend(docs)
        source_counts["live_odds"] = len(docs)

        # Also fetch historical/results
        hist_docs = fetch_historical_odds()
        all_docs.extend(hist_docs)
        source_counts["game_results"] = len(hist_docs)

    # ── 5. Basketball Reference (via Exa.AI) ──────────────────────────────
    if mode in ("full", "live", "bbref"):
        docs = fetch_bbref_via_exa()
        all_docs.extend(docs)
        source_counts["basketball_reference"] = len(docs)

    # ── 6. Exa.AI Expert Content ──────────────────────────────────────────
    if mode in ("full", "live", "exa"):
        docs = fetch_exa_expert_content()
        all_docs.extend(docs)
        source_counts["exa_expert_content"] = len(docs)

    # ── 7. Brave Search Content ───────────────────────────────────────────
    if mode in ("full", "live"):
        docs = fetch_brave_content()
        all_docs.extend(docs)
        source_counts["brave_content"] = len(docs)

    # ── Deduplicate by ID ─────────────────────────────────────────────────
    seen_ids = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = doc.get("id", "")
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    dupes = len(all_docs) - len(unique_docs)
    if dupes > 0:
        log(f"Deduplicated: {dupes} duplicates removed")
    all_docs = unique_docs

    # ── Save to disk ──────────────────────────────────────────────────────
    out_file = DATA_DIR / f"expert-ingest-{ts}.json"
    out_file.write_text(json.dumps({
        "timestamp": ts,
        "mode": mode,
        "total_documents": len(all_docs),
        "source_counts": source_counts,
        "documents": all_docs,
    }, indent=2, ensure_ascii=False))

    # ── Upsert to Pinecone ────────────────────────────────────────────────
    upserted = 0
    if not skip_pinecone and all_docs:
        log(f"\nUpserting {len(all_docs)} documents to Pinecone ({PINECONE_INDEX}/{PINECONE_NAMESPACE})...")
        upserted = upsert_to_pinecone(all_docs)

    # ── Summary ───────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"INGESTION COMPLETE")
    log(f"  Total documents: {len(all_docs)}")
    log(f"  Upserted to Pinecone: {upserted}")
    log(f"  Sources:")
    for src, count in source_counts.items():
        log(f"    {src}: {count}")
    log(f"  Saved to: {out_file}")
    log(f"{'='*70}\n")

    # Log metrics
    log_metric({
        "event": "ingestion_complete",
        "mode": mode,
        "total_documents": len(all_docs),
        "upserted": upserted,
        "source_counts": source_counts,
    })

    return all_docs, upserted


# ══════════════════════════════════════════════════════════════════════════════
# DAEMON MODE
# ══════════════════════════════════════════════════════════════════════════════

DAEMON_MODES = {
    "full": {
        "interval": 3600,  # 1 hour
        "description": "Full ingestion cycle every hour",
        "mode": "full",
    },
    "live": {
        "interval": 300,  # 5 minutes
        "description": "Live odds + web search every 5 minutes",
        "mode": "live",
    },
    "odds": {
        "interval": 120,  # 2 minutes
        "description": "Live odds only every 2 minutes",
        "mode": "odds",
    },
    "knowledge": {
        "interval": 86400,  # 24 hours
        "description": "Knowledge base refresh every 24 hours",
        "mode": "knowledge",
    },
}

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log(f"Received signal {signum} — shutting down gracefully...")
    _shutdown = True

def run_daemon(daemon_mode="full"):
    """Run continuous ingestion in daemon mode."""
    global _shutdown

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = DAEMON_MODES.get(daemon_mode, DAEMON_MODES["full"])
    interval = config["interval"]
    ingest_mode = config["mode"]

    # Write PID
    pid_file = ROOT / "data" / "ingest" / "expert-ingest.pid"
    pid_file.write_text(str(os.getpid()))

    log(f"DAEMON STARTED — Mode: {daemon_mode} ({config['description']})")
    log(f"  PID: {os.getpid()}")
    log(f"  Interval: {interval}s")
    log(f"  Pinecone: {PINECONE_INDEX}/{PINECONE_NAMESPACE}")

    state = load_state()
    consecutive_errors = 0

    while not _shutdown:
        try:
            docs, upserted = run_ingestion(mode=ingest_mode)
            state["cycles"] += 1
            state["total_ingested"] += len(docs)
            state["total_upserted"] += upserted
            state["last_run"] = datetime.now(timezone.utc).isoformat()
            state["source_counts"][f"cycle_{state['cycles']}"] = len(docs)
            consecutive_errors = 0
            save_state(state)

        except Exception as e:
            consecutive_errors += 1
            err_msg = f"Cycle error: {e}\n{traceback.format_exc()}"
            log(err_msg, "ERROR")
            state["errors"].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            })
            # Keep only last 20 errors
            state["errors"] = state["errors"][-20:]
            save_state(state)

            if consecutive_errors >= 3:
                log("3 consecutive errors — stopping daemon", "FATAL")
                break

        # Sleep in 10s increments to allow graceful shutdown
        sleep_until = time.time() + interval
        while time.time() < sleep_until and not _shutdown:
            time.sleep(min(10, sleep_until - time.time()))

    # Cleanup
    log("DAEMON STOPPED")
    if pid_file.exists():
        pid_file.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NBA Expert Data Ingestion — Multi-source pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       All sources (knowledge + live odds + Exa + Brave + BBREF)
  knowledge  Built-in knowledge bases only (no API calls)
  live       Live odds + Exa + Brave searches
  odds       Live betting odds only
  bbref      Basketball Reference via Exa only
  exa        Exa.AI expert content only

Daemon modes (--daemon):
  full       Full cycle every 1 hour
  live       Live data every 5 minutes
  odds       Odds-only every 2 minutes
  knowledge  Knowledge refresh every 24 hours

Examples:
  python3 ingest-expert-data.py --mode full
  python3 ingest-expert-data.py --mode odds --no-pinecone
  python3 ingest-expert-data.py --daemon live
  python3 ingest-expert-data.py --daemon odds
        """,
    )
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "knowledge", "live", "odds", "bbref", "exa"],
        help="Ingestion mode (default: full)",
    )
    parser.add_argument(
        "--daemon", metavar="MODE", default=None,
        choices=["full", "live", "odds", "knowledge"],
        help="Run in daemon mode with specified cycle type",
    )
    parser.add_argument(
        "--no-pinecone", action="store_true",
        help="Skip Pinecone upsert (save to disk only)",
    )

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.daemon)
    else:
        run_ingestion(mode=args.mode, skip_pinecone=args.no_pinecone)
