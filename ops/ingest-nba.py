#!/usr/bin/env python3
"""
NBA Data Ingestion — Multi-source expert-level data collection
Sources: balldontlie, odds-api, NBA advanced stats, expert knowledge bases
"""

import os, sys, json, time, ssl, urllib.request, urllib.parse, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))

# Load env
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

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

LITELLM_URL = os.environ.get("LITELLM_BASE_URL", "https://lbjlincoln-nomos-rag-engine-7.hf.space/v1")
LITELLM_KEY = os.environ.get("LITELLM_API_KEY", "sk-litellm-nomos-2026")
BALLDONTLIE_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

def http_get(url, headers=None, timeout=30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
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
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: LIVE BETTING ODDS (The Odds API — covers ALL major books)
# ══════════════════════════════════════════════════════════════════════════════
# Covers: DraftKings, FanDuel, BetMGM, Caesars, PointsBet, BetRivers, Pinnacle,
#         Bet365, Unibet, William Hill, Betway, 888sport, and 40+ more

ODDS_BOOKMAKERS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus",
    "betrivers", "pinnacle", "bet365", "unibet", "williamhill",
    "betway", "888sport", "bovada", "betonlineag", "mybookieag",
    "superbook", "wynnbet", "twinspires", "betus", "lowvig",
    "betparx", "espnbet", "fliff", "hardrockbet"
]

def fetch_live_odds():
    """Fetch live NBA odds from ALL bookmakers via the-odds-api.com"""
    if not ODDS_API_KEY:
        print("[ODDS] No ODDS_API_KEY — set it to fetch live odds")
        return []

    # H2H (moneyline)
    odds_h2h, _ = http_get(
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?apiKey={ODDS_API_KEY}&regions=us,eu,uk,au&markets=h2h,spreads,totals"
        f"&oddsFormat=decimal&dateFormat=iso",
    )

    if "error" in odds_h2h:
        print(f"[ODDS] Error: {odds_h2h['error']}")
        return []

    games = odds_h2h if isinstance(odds_h2h, list) else []
    documents = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")

        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                text = f"NBA Odds — {away} @ {home} ({commence[:10]})\n"
                text += f"Bookmaker: {bk_name} | Market: {market_key}\n"
                for o in outcomes:
                    name = o.get("name", "")
                    price = o.get("price", 0)
                    point = o.get("point", "")
                    text += f"  {name}: {price}" + (f" ({point:+g})" if point else "") + "\n"

                documents.append({
                    "id": f"odds-{hashlib.md5(f'{home}-{away}-{bk_name}-{market_key}'.encode()).hexdigest()[:12]}",
                    "text": text,
                    "source": f"the-odds-api/{bk_name}",
                    "category": "live_odds",
                    "metadata": {
                        "home": home, "away": away, "bookmaker": bk_name,
                        "market": market_key, "commence": commence,
                    },
                })

    print(f"[ODDS] Fetched {len(documents)} odds entries from {len(games)} games")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: NBA ADVANCED STATS (what NBA teams actually use)
# ══════════════════════════════════════════════════════════════════════════════

NBA_ADVANCED_STATS_KNOWLEDGE = [
    # Stats avancees utilisees par les equipes NBA
    {
        "id": "adv-raptor",
        "text": """RAPTOR (Robust Algorithm using Player Tracking and On/Off Ratings)
Developpe par FiveThirtyEight/Nate Silver. Combine box score, on/off data, et player tracking.
Composantes: RAPTOR Offense + RAPTOR Defense = RAPTOR Total (en points par 100 possessions)
WAR RAPTOR = impact total en victoires. Top tier: +8 WAR/saison.
Utilise par: analystes, media, front offices pour comparaisons cross-era.
Avantage: ajuste pour qualite des coequipiers et adversaires.""",
        "source": "FiveThirtyEight/Nate Silver",
        "category": "advanced_stats",
    },
    {
        "id": "adv-epm",
        "text": """EPM (Estimated Plus-Minus) — Dunks & Threes / Taylor Snarr
Le gold standard des metrics modernes. Utilise regularized adjusted plus-minus (RAPM)
enrichi par box score priors. Separé en EPM Offense et EPM Defense.
Echelle: points par 100 possessions au-dessus de la moyenne.
Elite: > +5.0 EPM. All-Star level: > +3.0. Starter: > +1.0.
Jokic 2023-24: +9.2 EPM (historique). Utilise par plusieurs front offices NBA.""",
        "source": "Dunks & Threes / Taylor Snarr",
        "category": "advanced_stats",
    },
    {
        "id": "adv-lebron",
        "text": """LEBRON (Luck-adjusted player Estimate using a Box prior Regularized ON-off)
BBall Index metric. Combine RAPM avec prior box score.
LEBRON = O-LEBRON + D-LEBRON. WAR derivee du LEBRON.
Un des meilleurs predicteurs de performance future.
Avantage: mieux calibre que EPM pour les petits echantillons.
Utilise par: front offices, media, DFS players.""",
        "source": "BBall Index",
        "category": "advanced_stats",
    },
    {
        "id": "adv-darko",
        "text": """DARKO (Daily Adjusted Rating and Kalman Optimized)
Modele predictif bayesien par Kostya Medvedovsky.
Utilise filtres de Kalman pour mettre a jour les projections joueur QUOTIDIENNEMENT.
Predit: points, rebonds, passes, minutes, impact.
Tres utilise pour DFS (Daily Fantasy Sports) et paris props.
Avantage: s'adapte rapidement aux changements (blessures, changement de role).""",
        "source": "DARKO / Kostya Medvedovsky",
        "category": "advanced_stats",
    },
    {
        "id": "adv-tracking",
        "text": """NBA Player Tracking (Second Spectrum / anciennement SportVU)
Camera system installé dans les 30 arenas NBA. Capture 25 frames/seconde.
Données: position x,y de chaque joueur + ballon, vitesse, acceleration, distance parcourue.
Metrics derivees: contested shots, drive frequency, catch-and-shoot %, pull-up %,
touches, time of possession, speed, distance covered, rebounding chances.
Utilise par: TOUTES les equipes NBA pour game planning et player evaluation.
Public via stats.nba.com/players/tracking/.""",
        "source": "NBA.com / Second Spectrum",
        "category": "advanced_stats",
    },
    {
        "id": "adv-synergy",
        "text": """Synergy Sports (maintenant partie de Second Spectrum)
Play-by-play video tagging. Chaque possession classifiee par type:
- Pick-and-Roll Ball Handler / Roll Man
- Isolation, Post-Up, Spot-Up
- Transition, Cut, Off-Screen, Handoff, Putbacks, Miscellaneous
Pour chaque type: PPP (Points Per Possession), frequency, percentile ranking.
THE tool utilise par les coaching staffs pour game prep et scouting.
30/30 equipes NBA utilisent Synergy.""",
        "source": "Synergy Sports / Second Spectrum",
        "category": "advanced_stats",
    },
    {
        "id": "adv-cleaning-glass",
        "text": """Cleaning the Glass (Ben Falk, ancien VP Basketball Ops 76ers)
Site premium d'analytics avancees. Filtre les minutes garbage time.
Metrics cles: eFG%, TS%, TOV%, ORB%, FTr, Assist Rate — tous filtres contextuellement.
Lineup data, on/off splits, shot charts par zone.
Utilise par: front offices, agents, media serieux.
Avantage: filtrage garbage time + contexte (home/away, score margin).""",
        "source": "Cleaning the Glass / Ben Falk",
        "category": "advanced_stats",
    },
    {
        "id": "adv-pbp-stats",
        "text": """PBP Stats (Ryan Davis)
Play-by-play analytics detaillees. Tracking data public.
Matchup data: qui garde qui, efficacite defensive par matchup.
Shot quality metrics (qSQ, xeFG%), expected rebounding.
Lineup combinations avec contexte de minutes jouees.
Utilise par: coaches, DFS, bettors serieux.""",
        "source": "PBP Stats / Ryan Davis",
        "category": "advanced_stats",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: TONY BLOOM / STARLIZARD STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

TONY_BLOOM_KNOWLEDGE = [
    {
        "id": "bloom-001",
        "text": """Tony Bloom Strategy — Le plus grand parieur sportif du monde
Tony Bloom, surnomme "The Lizard", proprietaire de Brighton & Hove Albion FC.
Fondateur de Starlizard, societe de conseil en paris sportifs (~$1B+ de chiffre d'affaires annuel).

PRINCIPES CLES DE LA STRATEGIE BLOOM:
1. MODELES MATHEMATIQUES: Equipe de 160+ data scientists, physiciens, mathematiciens
2. EXPECTED VALUE (EV): Ne JAMAIS parier sans edge mathematique positif
3. VOLUME: Des milliers de paris par jour, petits edges mais volume massif
4. AUTOMATISATION: Algorithmes qui placent les paris automatiquement quand EV > seuil
5. CLOSING LINE VALUE (CLV): Mesure principale de performance — si tu bats la cloture, tu gagnes long terme
6. MARKET MAKING: Comprendre comment les bookmakers fixent les cotes (pouvoir predictif vs flow d'argent)""",
        "source": "Tony Bloom / Starlizard",
        "category": "betting_strategy",
    },
    {
        "id": "bloom-002",
        "text": """Starlizard Methodology — Application au NBA

MODELE NBA DE TYPE STARLIZARD:
1. POWER RATINGS: Chaque equipe a un rating ajuste quotidiennement
   - Back-to-back adjustment: -2 a -4 points
   - Travel adjustment: -0.5 a -1.5 points
   - Injury impact: modele position-specific (star PG > rotation big)
   - Rest days: +0.5 par jour de repos au-dela de 1

2. LINE SHOPPING: Comparer TOUTES les cotes disponibles
   - Pinnacle = marche le plus efficace (sharp book)
   - Si ta ligne > Pinnacle close = potentiellement +EV
   - Minimum 3-5% edge avant de parier

3. BANKROLL MANAGEMENT (Kelly Criterion adapte):
   - Full Kelly trop volatile → utiliser 1/4 Kelly ou 1/3 Kelly
   - Max bet = 3% du bankroll
   - Track ROI sur 1000+ paris minimum avant de juger

4. TIMING: Parier EARLY quand les lignes sont soft, OU LIVE quand le marche surreagit

5. DATA PIPELINE: Automatisation complete
   - Scraping odds en temps reel (toutes les 30s)
   - Calcul EV automatique
   - Alerte quand EV > seuil (ex: 3%)
   - Placement automatique si possible""",
        "source": "Tony Bloom / Starlizard methodology",
        "category": "betting_strategy",
    },
    {
        "id": "bloom-003",
        "text": """Sharps vs Squares — Ce que les equipes NBA de betting font differemment

SHARP BETTING NBA (ce que font les pros):
1. FOCUS SUR TOTALS: Les totals (over/under) sont plus predictibles que les spreads
2. PLAYER PROPS: Edge enorme sur les props avec des modeles de projection
3. LIVE BETTING: Le marche in-game est moins efficient que pre-game
4. CONTRARIAN: Parier contre le public quand les lignes bougent sur steam moves recreationnels
5. CORRELATED PARLAYS: Ex — equipe rapide + over (correles positivement)

METRICS A TRACKER (comme un hedge fund):
- CLV (Closing Line Value): objectif +2% CLV moyen
- ROI: objectif 3-7% long terme
- Yield: profit / turnover
- Sharpe Ratio: risk-adjusted returns
- Drawdown max: jamais > 20% du bankroll
- Win rate: pas important en soi, seul le EV compte

ERREURS A EVITER:
- Parier avec ses emotions (son equipe favorite)
- Chaser les pertes (augmenter les mises apres une perte)
- Ignorer les back-to-backs et fatigue
- Ne pas tracker ses paris systematiquement
- Parier sans edge (recreation ≠ investissement)""",
        "source": "Sharp Betting / Professional NBA Betting",
        "category": "betting_strategy",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: EXPERT KNOWLEDGE (top NBA analysts methodology)
# ══════════════════════════════════════════════════════════════════════════════

NBA_EXPERT_KNOWLEDGE = [
    {
        "id": "expert-zach-lowe",
        "text": """Zach Lowe — ESPN Senior NBA Writer (anciennement Grantland)
Considere comme le meilleur analyste tactique NBA dans les media.
Methodologie: Film study + analytics + interviews coaching staffs.
Specialites: defensive schemes, off-ball movement, lineup construction.
Son podcast "Lowe Post" est reference chez les front offices NBA.
Key insight: L'oeil et les stats doivent converger — si les stats disent une chose
mais le film montre autre chose, c'est les stats qui sont incompletes.""",
        "source": "Expert knowledge / Zach Lowe",
        "category": "expert_methodology",
    },
    {
        "id": "expert-ben-taylor",
        "text": """Ben Taylor — Thinking Basketball (YouTube + Book)
Auteur de "Thinking Basketball: How to watch, enjoy and appreciate the game"
Le meilleur pour les analyses historiques cross-era.
Methodologie: backpicks.com GOAT rankings, film study, statistical context.
Son systeme de ranking integre: scoring efficiency, playmaking, defense,
leadership, clutch, longevity — avec ajustement pour l'ere.
Chaine YouTube: 500K+ subscribers. Reference absolue pour GOAT debates.""",
        "source": "Expert knowledge / Ben Taylor",
        "category": "expert_methodology",
    },
    {
        "id": "expert-seth-partnow",
        "text": """Seth Partnow — ancien VP Analytics Milwaukee Bucks (champion 2021)
Auteur de "The Midrange Theory: Basketball's Evolution in the Age of Analytics"
Travaille maintenant pour The Athletic.
Expertise: comment les equipes NBA utilisent REELLEMENT les analytics.
Key insight: Les equipes ne suivent pas betement les modeles — l'analytics
est un input parmi d'autres (scouting, coaching, culture, market dynamics).
Shot selection, lineup optimization, draft modeling.""",
        "source": "Expert knowledge / Seth Partnow",
        "category": "expert_methodology",
    },
    {
        "id": "expert-kirk-goldsberry",
        "text": """Kirk Goldsberry — Pioneer des shot charts / spatial analytics
Auteur de "SprawlBall" et "The Atlas of NBA Shooters".
Professeur a UT San Antonio devenu consultant NBA.
A revolutionne la visualisation des donnees NBA avec les shot charts par zone.
Travaille avec les San Antonio Spurs.
Key insight: La revolution 3-points est la plus grande transformation
tactique de l'histoire du sport professionnel.""",
        "source": "Expert knowledge / Kirk Goldsberry",
        "category": "expert_methodology",
    },
    {
        "id": "expert-nate-silver",
        "text": """Nate Silver — FiveThirtyEight NBA RAPTOR/Elo models
Statisticien celebre, fondateur de FiveThirtyEight.
A cree le systeme Elo NBA (adaptee du chess Elo) pour predire les matchs.
RAPTOR: sa metric joueur la plus aboutie (replaced CARMELO in 2019).
Approche: bayesian priors + updating avec nouvelles donnees.
Elo NBA: predit ~68% des matchs correctement (reference).
Aussi auteur de "The Signal and the Noise" sur la prediction en general.""",
        "source": "Expert knowledge / Nate Silver",
        "category": "expert_methodology",
    },
    {
        "id": "expert-haralabos-voulgaris",
        "text": """Haralabos Voulgaris — Parieur professionnel NBA devenu front office
Un des plus gros parieurs NBA de l'histoire. A rejoint les Dallas Mavericks
comme Director of Quantitative Research & Development sous Mark Cuban.
Methodologie: modeles de simulation Monte Carlo, injury impact modeling,
referee tendency analysis, pace/style adjustments.
Key insight: L'avantage dans les paris NBA vient du meilleur traitement
de l'information PUBLIQUE, pas de l'acces a l'information privee.
A quitte Dallas en 2021 apres des resultats mitiges en front office.""",
        "source": "Expert knowledge / Haralabos Voulgaris",
        "category": "expert_methodology",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: NBA TEAM DATASETS (what actual teams use)
# ══════════════════════════════════════════════════════════════════════════════

NBA_TEAM_DATASETS = [
    {
        "id": "dataset-second-spectrum",
        "text": """Second Spectrum — Official NBA Tracking Data Provider
Every NBA arena has 5+ cameras tracking all 10 players + ball at 25fps.
Data captured: x,y coordinates, speed, acceleration, distance.
Derived stats: shot quality (qSQ), pass quality, defensive coverage,
screen effectiveness, rebounding positioning, transition speed.
EVERY NBA team has access. Some teams build proprietary models on top.
Cost: included in NBA team licensing (~$1M+/year for enhanced access).
Public data: limited subset on stats.nba.com""",
        "source": "NBA Teams / Second Spectrum",
        "category": "team_datasets",
    },
    {
        "id": "dataset-statsbomb",
        "text": """StatsBomb (primarily soccer but expanding to basketball)
Event-level data with 360 freeze frames.
For NBA context: similar companies provide event-level data:
- Sportradar (official NBA data partner) — play-by-play, live stats
- Genius Sports — betting data, live odds, performance data
- Stats Perform (Opta) — historical stats, AI predictions
NBA teams typically use 3-5 data providers simultaneously.""",
        "source": "NBA Teams / Data Providers",
        "category": "team_datasets",
    },
    {
        "id": "dataset-public-sources",
        "text": """Free/Public NBA Data Sources Used by Analysts:
1. Basketball Reference (basketball-reference.com) — Historical stats since 1946
2. NBA.com/stats — Official tracking data (limited public API)
3. stats.nba.com API — JSON endpoints for all stats categories
4. balldontlie.io — Clean REST API for players, games, stats
5. nba_api Python package — Wrapper for NBA.com endpoints
6. Kaggle NBA datasets — Historical game data, player stats, draft data
7. PBP Stats (pbpstats.com) — Detailed play-by-play
8. Cleaning the Glass — Premium filtered analytics
9. Dunks & Threes — EPM and advanced metrics
10. BBall Index — LEBRON metric and proprietary data""",
        "source": "Public NBA Data Sources",
        "category": "team_datasets",
    },
    {
        "id": "dataset-draft-models",
        "text": """NBA Draft Prediction Models (used by teams):
1. Big Board models: Combine measurements + college stats + conference strength
2. Statistical translation models: Convert college stats to projected NBA stats
   - Kevin Pelton's WARP (Wins Above Replacement Player)
   - Sports Reference's VORP projections
3. Archetype models: Compare prospect body/stats profile to historical players
4. Bust probability models: Predict likelihood of bust based on red flags
   (age, conference, statistical profile, shooting concerns)
5. Positional value: Center production is cheapest to replace (draft guards/wings higher)
Key variables: age at draft, college conference, per-40 minute stats,
true shooting %, assist rate, steal rate, block rate, rebounding rate.""",
        "source": "NBA Draft Analytics",
        "category": "team_datasets",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 6: EXA.AI SEARCH (discover latest NBA content)
# ══════════════════════════════════════════════════════════════════════════════

def search_exa_nba(query, num_results=10):
    """Search for NBA expert content via Exa.AI."""
    if not EXA_API_KEY:
        return []

    resp, status = http_post(
        "https://api.exa.ai/search",
        {
            "query": query,
            "numResults": num_results,
            "useAutoprompt": True,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 2000}},
        },
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        timeout=30,
    )

    if "error" in resp:
        return []

    results = resp.get("results", [])
    documents = []
    for r in results:
        text = r.get("text", "")
        if not text:
            continue
        documents.append({
            "id": f"exa-{hashlib.md5(r.get('url','').encode()).hexdigest()[:12]}",
            "text": text[:2000],
            "source": r.get("url", ""),
            "title": r.get("title", ""),
            "category": "web_content",
        })

    return documents

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def ingest_all():
    """Run full ingestion pipeline."""
    all_docs = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(f"\n{'='*60}")
    print(f"NBA INGESTION — {ts}")
    print(f"{'='*60}\n")

    # 1. Built-in knowledge bases
    print("[1/6] Advanced stats knowledge...")
    for doc in NBA_ADVANCED_STATS_KNOWLEDGE:
        all_docs.append(doc)
    print(f"  → {len(NBA_ADVANCED_STATS_KNOWLEDGE)} docs")

    # 2. Tony Bloom / betting strategy
    print("[2/6] Tony Bloom / Starlizard strategy...")
    for doc in TONY_BLOOM_KNOWLEDGE:
        all_docs.append(doc)
    print(f"  → {len(TONY_BLOOM_KNOWLEDGE)} docs")

    # 3. Expert methodology
    print("[3/6] NBA expert knowledge (Lowe, Taylor, Partnow, Silver, Voulgaris)...")
    for doc in NBA_EXPERT_KNOWLEDGE:
        all_docs.append(doc)
    print(f"  → {len(NBA_EXPERT_KNOWLEDGE)} docs")

    # 4. Team datasets knowledge
    print("[4/6] NBA team datasets documentation...")
    for doc in NBA_TEAM_DATASETS:
        all_docs.append(doc)
    print(f"  → {len(NBA_TEAM_DATASETS)} docs")

    # 5. Live odds
    print("[5/6] Live betting odds (all bookmakers)...")
    odds = fetch_live_odds()
    all_docs.extend(odds)
    print(f"  → {len(odds)} odds entries")

    # 6. Exa.AI expert content search
    print("[6/6] Exa.AI expert content search...")
    exa_queries = [
        "NBA advanced analytics 2025 best practices team building",
        "NBA betting sharp strategy CLV closing line value",
        "Tony Bloom Starlizard sports betting methodology",
        "NBA player tracking data Second Spectrum analytics",
        "RAPTOR EPM LEBRON NBA player impact metrics comparison",
        "NBA draft prediction model machine learning",
    ]
    exa_total = 0
    for q in exa_queries:
        docs = search_exa_nba(q, num_results=5)
        all_docs.extend(docs)
        exa_total += len(docs)
        time.sleep(1)  # Rate limit
    print(f"  → {exa_total} web articles")

    # Save all documents
    out_file = ROOT / "data" / "ingest" / f"ingest-{ts}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "timestamp": ts,
        "total_documents": len(all_docs),
        "sources": {
            "advanced_stats": len(NBA_ADVANCED_STATS_KNOWLEDGE),
            "betting_strategy": len(TONY_BLOOM_KNOWLEDGE),
            "expert_methodology": len(NBA_EXPERT_KNOWLEDGE),
            "team_datasets": len(NBA_TEAM_DATASETS),
            "live_odds": len(odds),
            "web_content": exa_total,
        },
        "documents": all_docs,
    }, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_docs)} documents ingested")
    print(f"Saved to: {out_file}")
    print(f"{'='*60}\n")

    return all_docs

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Data Ingestion")
    parser.add_argument("--source", default="all", choices=["all", "odds", "exa", "knowledge"])
    args = parser.parse_args()

    if args.source == "odds":
        fetch_live_odds()
    elif args.source == "exa":
        for q in ["NBA advanced analytics 2025", "NBA betting strategy sharp"]:
            docs = search_exa_nba(q)
            print(f"  {q}: {len(docs)} results")
    else:
        ingest_all()
