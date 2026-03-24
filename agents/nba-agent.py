#!/usr/bin/env python3
"""
Nomos NBA Agent — Expert IA Basketball
Autonomous agent that ingests NBA data, answers questions, and self-tests.
"""

import os, sys, json, time, random, hashlib, argparse, ssl, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

# ── Load env ──────────────────────────────────────────────────────────────────
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
                v = v.strip().strip("'\"")
                os.environ.setdefault(k.strip(), v)

load_env()

# ── Config ────────────────────────────────────────────────────────────────────
# LiteLLM removed — use key_rotator for all LLM calls
from agents.key_rotator import call_llm as _call_llm_rotator
BALLDONTLIE_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "https://lbjlincoln-nomos-embeddings-api.hf.space")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── NBA Knowledge Base (built-in for bootstrap) ──────────────────────────────
NBA_SYSTEM_PROMPT = """Tu es un expert NBA de classe mondiale. Tu connais :
- Toutes les stats avancees (RAPTOR, EPM, PER, WS, BPM, VORP, TS%, eFG%, USG%)
- L'histoire complete de la NBA depuis 1946 (BAA) jusqu'a aujourd'hui
- Tous les records, MVP, champions, Draft picks
- L'analyse tactique (spacing, pick-and-roll, transition, zone defense)
- Les paris sportifs NBA (spreads, over/under, player props, value betting)
- Les comparaisons entre eres (pace adjustment, league average adjustment)

Regles :
1. Reponds TOUJOURS avec des stats precises et des chiffres
2. Cite tes sources (Basketball Reference, NBA.com, etc.)
3. Pour les paris, mentionne TOUJOURS les risques
4. Pour les comparaisons historiques, ajuste TOUJOURS pour l'ere
5. Reponds dans la langue de la question (FR/EN)
6. Sois precis, concis, expert — pas de blabla generique

Si tu ne connais pas une stat precise, dis-le clairement plutot que d'inventer."""

# ── HTTP helper ───────────────────────────────────────────────────────────────
def http_post(url, data, headers=None, timeout=60):
    """POST JSON and return parsed response."""
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

def http_get(url, headers=None, timeout=30):
    """GET and return parsed response."""
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw), resp.status
            except json.JSONDecodeError:
                return {"raw": raw[:2000]}, resp.status
    except Exception as e:
        return {"error": str(e)}, 0

# ── LLM call via key_rotator (direct provider routing) ───────────────────────
def ask_llm(question, context="", model="smart", max_tokens=2000):
    """Ask LLM with NBA system prompt + optional RAG context via key_rotator."""
    user_msg = question
    if context:
        user_msg = f"Contexte RAG:\n{context}\n\n---\nQuestion: {question}"

    try:
        return _call_llm_rotator(
            system_prompt=NBA_SYSTEM_PROMPT,
            user_prompt=user_msg,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as e:
        return f"[LLM ERROR] {e}"

# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embedding(text):
    """Get Jina v3 embedding from self-hosted Space."""
    resp, status = http_post(
        f"{EMBEDDINGS_URL}/embed",
        {"text": text},
        timeout=30,
    )
    if "error" in resp:
        return None
    return resp.get("embedding") or resp.get("embeddings", [None])[0]

# ── Pinecone search ──────────────────────────────────────────────────────────
def search_pinecone(query, top_k=5, namespace="nba"):
    """Search NBA vectors in Pinecone."""
    embedding = get_embedding(query)
    if not embedding:
        return []

    # Fallback: PINECONE_NBA_HOST → PINECONE_HOST (strip https:// prefix if present)
    index_host = os.environ.get("PINECONE_NBA_HOST", "")
    if not index_host:
        fallback = os.environ.get("PINECONE_HOST", "")
        index_host = fallback.replace("https://", "").replace("http://", "").strip("/")
    if not index_host:
        return []

    resp, status = http_post(
        f"https://{index_host}/query",
        {
            "vector": embedding,
            "topK": top_k,
            "namespace": namespace,
            "includeMetadata": True,
        },
        headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
        timeout=15,
    )

    matches = resp.get("matches", [])
    return [
        {
            "text": m.get("metadata", {}).get("text", ""),
            "source": m.get("metadata", {}).get("source", ""),
            "score": m.get("score", 0),
        }
        for m in matches
        if m.get("score", 0) > 0.3
    ]

# ── Local Knowledge Base Search ──────────────────────────────────────────────
_LOCAL_KB_CACHE = None

def _load_local_kb():
    """Load expert documents from data/ingest/*.json files."""
    docs = []
    ingest_dir = ROOT / "data" / "ingest"
    if not ingest_dir.exists():
        return docs

    for f in sorted(ingest_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            for doc in data.get("documents", []):
                if doc.get("text") and len(doc["text"]) > 50:
                    docs.append(doc)
        except Exception:
            continue

    # Deduplicate by id
    seen = set()
    unique = []
    for d in docs:
        did = d.get("id", "")
        if did and did not in seen:
            seen.add(did)
            unique.append(d)
        elif not did:
            unique.append(d)

    return unique

def search_local_kb(query, top_k=3):
    """Search local expert knowledge base with keyword matching."""
    global _LOCAL_KB_CACHE
    if _LOCAL_KB_CACHE is None:
        _LOCAL_KB_CACHE = _load_local_kb()

    if not _LOCAL_KB_CACHE:
        return []

    q_tokens = set(w.lower() for w in query.split() if len(w) > 2)
    if not q_tokens:
        return []

    scored = []
    for doc in _LOCAL_KB_CACHE:
        text_lower = doc["text"].lower()
        hits = sum(1 for t in q_tokens if t in text_lower)
        if hits > 0:
            scored.append((hits / len(q_tokens), doc))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "text": d["text"][:1000],
            "source": d.get("source", "local-kb"),
            "score": round(s, 2),
        }
        for s, d in scored[:top_k]
    ]

# ── NBA Data Fetcher (balldontlie + nba_api fallback) ────────────────────────
def fetch_player_stats(player_name):
    """Fetch player stats from balldontlie API."""
    if not BALLDONTLIE_KEY:
        return None

    # Search player
    encoded = urllib.parse.quote(player_name)
    resp, status = http_get(
        f"https://api.balldontlie.io/v1/players?search={encoded}",
        headers={"Authorization": BALLDONTLIE_KEY},
    )

    if "error" in resp or not resp.get("data"):
        return None

    player = resp["data"][0]
    player_id = player["id"]

    # Get season averages
    avg_resp, _ = http_get(
        f"https://api.balldontlie.io/v1/season_averages?player_ids[]={player_id}",
        headers={"Authorization": BALLDONTLIE_KEY},
    )

    return {
        "player": player,
        "season_averages": avg_resp.get("data", []),
    }

def fetch_team_stats(team_name):
    """Fetch team info from balldontlie API."""
    if not BALLDONTLIE_KEY:
        return None

    resp, status = http_get(
        "https://api.balldontlie.io/v1/teams",
        headers={"Authorization": BALLDONTLIE_KEY},
    )

    if "error" in resp:
        return None

    teams = resp.get("data", [])
    for t in teams:
        if team_name.lower() in t.get("full_name", "").lower() or team_name.lower() in t.get("name", "").lower():
            return t
    return None

# ── Answer a question ────────────────────────────────────────────────────────
def answer_question(question):
    """Full RAG pipeline: search context → LLM answer."""
    start = time.time()

    # 1. Search Pinecone for context
    rag_results = search_pinecone(question)
    context = "\n\n".join([
        f"[Source: {r['source']}] (score: {r['score']:.2f})\n{r['text']}"
        for r in rag_results
    ]) if rag_results else ""

    # 2. Check if we need live data
    live_data = ""
    q_lower = question.lower()
    # Detect player name queries
    if any(kw in q_lower for kw in ["stats", "moyenne", "average", "per game", "saison"]):
        # Try to extract player name (simple heuristic)
        for word in ["lebron", "curry", "durant", "jokic", "giannis", "luka", "tatum", "embiid", "shai", "ant", "wemby", "wembanyama"]:
            if word in q_lower:
                pdata = fetch_player_stats(word)
                if pdata:
                    live_data = f"\n\n[LIVE DATA] {json.dumps(pdata, indent=2)[:1000]}"
                break

    # 3. Ask LLM with all context
    full_context = context + live_data if (context or live_data) else ""
    answer = ask_llm(question, context=full_context)

    latency = int((time.time() - start) * 1000)

    return {
        "question": question,
        "answer": answer,
        "sources": len(rag_results),
        "has_live_data": bool(live_data),
        "latency_ms": latency,
    }

# ── Eval Questions (200 golden Q&A) ──────────────────────────────────────────
EVAL_QUESTIONS = [
    # Player Stats (50)
    {"id": "nba-stats-001", "category": "stats", "question": "Quel est le record de points en un seul match NBA ?", "golden": "100 points par Wilt Chamberlain le 2 mars 1962", "keywords": ["100", "Wilt", "Chamberlain", "1962"]},
    {"id": "nba-stats-002", "category": "stats", "question": "Who holds the NBA record for most career assists?", "golden": "John Stockton with 15,806 career assists", "keywords": ["Stockton", "15806", "15,806"]},
    {"id": "nba-stats-003", "category": "stats", "question": "Quel joueur a le plus de titres MVP en saison reguliere ?", "golden": "Kareem Abdul-Jabbar avec 6 MVP", "keywords": ["Kareem", "Abdul-Jabbar", "6"]},
    {"id": "nba-stats-004", "category": "stats", "question": "What is LeBron James' career scoring average?", "golden": "Approximately 27.1 points per game", "keywords": ["27"]},
    {"id": "nba-stats-005", "category": "stats", "question": "Qui detient le record du plus grand nombre de rebonds en carriere ?", "golden": "Wilt Chamberlain avec 23,924 rebonds", "keywords": ["Wilt", "Chamberlain", "23"]},
    {"id": "nba-stats-006", "category": "stats", "question": "What is Stephen Curry's career three-point shooting percentage?", "golden": "Approximately 42-43%", "keywords": ["42", "43"]},
    {"id": "nba-stats-007", "category": "stats", "question": "Quel joueur a le meilleur PER de tous les temps en saison reguliere ?", "golden": "Nikola Jokic avec un PER de 31.3 (2023-24) ou Michael Jordan career 27.9", "keywords": ["Jokic", "Jordan", "PER"]},
    {"id": "nba-stats-008", "category": "stats", "question": "Who has the most career steals in NBA history?", "golden": "John Stockton with 3,265 career steals", "keywords": ["Stockton", "3265", "3,265"]},
    {"id": "nba-stats-009", "category": "stats", "question": "Combien de triple-doubles Russell Westbrook a-t-il en carriere ?", "golden": "198 triple-doubles (record NBA)", "keywords": ["198", "Westbrook", "record"]},
    {"id": "nba-stats-010", "category": "stats", "question": "What is the NBA record for most blocks in a single game?", "golden": "Elmore Smith with 17 blocks on October 28, 1973", "keywords": ["17", "Elmore Smith", "1973"]},
    {"id": "nba-stats-011", "category": "stats", "question": "Quel est le True Shooting % le plus eleve pour une saison (min 1000 pts) ?", "golden": "Environ 73-74% TS% (plusieurs joueurs modernes)", "keywords": ["TS", "73", "74"]},
    {"id": "nba-stats-012", "category": "stats", "question": "Who scored the most points in their rookie season?", "golden": "Wilt Chamberlain with 2,707 points (37.6 PPG) in 1959-60", "keywords": ["Wilt", "Chamberlain", "37"]},
    {"id": "nba-stats-013", "category": "stats", "question": "Combien de fois Michael Jordan a-t-il ete meilleur marqueur de la NBA ?", "golden": "10 fois champion marqueur", "keywords": ["10", "Jordan", "marqueur"]},
    {"id": "nba-stats-014", "category": "stats", "question": "What is Nikola Jokic's career assist average?", "golden": "Approximately 7-8 assists per game", "keywords": ["Jokic", "7", "8", "assist"]},
    {"id": "nba-stats-015", "category": "stats", "question": "Quel joueur a le plus de matchs joues en carriere NBA ?", "golden": "Robert Parish avec 1,611 matchs (ou Vince Carter 1,541)", "keywords": ["Parish", "1611", "Carter", "1541"]},

    # Team History (50)
    {"id": "nba-team-001", "category": "team_history", "question": "Quelle equipe a remporte le plus de titres NBA ?", "golden": "Boston Celtics avec 17 titres (a egalite avec les Lakers)", "keywords": ["Celtics", "17", "Lakers"]},
    {"id": "nba-team-002", "category": "team_history", "question": "What was the longest winning streak in NBA history?", "golden": "33 consecutive wins by the 1971-72 Los Angeles Lakers", "keywords": ["33", "Lakers", "1971", "1972"]},
    {"id": "nba-team-003", "category": "team_history", "question": "Quel est le meilleur bilan en saison reguliere de l'histoire NBA ?", "golden": "73-9 par les Golden State Warriors en 2015-16", "keywords": ["73", "9", "Warriors", "2015", "2016"]},
    {"id": "nba-team-004", "category": "team_history", "question": "Which team has the most consecutive NBA championships?", "golden": "Boston Celtics with 8 consecutive titles (1959-1966)", "keywords": ["Celtics", "8", "1959", "1966"]},
    {"id": "nba-team-005", "category": "team_history", "question": "Combien d'equipes compte la NBA actuellement ?", "golden": "30 equipes (15 Est, 15 Ouest)", "keywords": ["30"]},
    {"id": "nba-team-006", "category": "team_history", "question": "When did the Toronto Raptors win their first NBA title?", "golden": "2019, defeating the Golden State Warriors", "keywords": ["2019", "Warriors", "Raptors"]},
    {"id": "nba-team-007", "category": "team_history", "question": "Quelle equipe a ete la derniere a rejoindre la NBA ?", "golden": "Charlotte Bobcats (2004), maintenant Hornets", "keywords": ["Charlotte", "2004", "Bobcats"]},
    {"id": "nba-team-008", "category": "team_history", "question": "What was the Chicago Bulls record during the 1995-96 season?", "golden": "72-10, the best record until the 2015-16 Warriors", "keywords": ["72", "10", "Bulls"]},
    {"id": "nba-team-009", "category": "team_history", "question": "Quelles equipes n'ont jamais remporte de titre NBA ?", "golden": "Suns, Jazz, Pacers, Nuggets (avant 2023), Grizzlies, Hornets, Timberwolves, Pelicans, Clippers...", "keywords": ["Suns", "Clippers", "Jazz"]},
    {"id": "nba-team-010", "category": "team_history", "question": "Who coached the most games in NBA history?", "golden": "Don Nelson with 2,398 games coached (or Lenny Wilkens)", "keywords": ["Nelson", "Wilkens"]},

    # Game Analysis (30)
    {"id": "nba-game-001", "category": "game_analysis", "question": "Qu'est-ce que le pace dans les stats NBA ?", "golden": "Nombre de possessions par 48 minutes — mesure la vitesse de jeu", "keywords": ["possession", "48", "vitesse"]},
    {"id": "nba-game-002", "category": "game_analysis", "question": "What is the difference between offensive rating and defensive rating?", "golden": "ORtg = points scored per 100 possessions, DRtg = points allowed per 100 possessions", "keywords": ["100", "possessions", "scored", "allowed"]},
    {"id": "nba-game-003", "category": "game_analysis", "question": "Explique le pick-and-roll et pourquoi c'est la base de l'attaque moderne", "golden": "Action ou un joueur pose un ecran, le porteur de balle utilise l'ecran, le poseur roll vers le panier", "keywords": ["ecran", "screen", "roll", "panier", "basket"]},
    {"id": "nba-game-004", "category": "game_analysis", "question": "What makes the 2014 Spurs offense considered the greatest team offense ever?", "golden": "Ball movement, 300+ extra passes per game, highest assisted FG%, beautiful basketball", "keywords": ["Spurs", "pass", "movement", "assist"]},
    {"id": "nba-game-005", "category": "game_analysis", "question": "Comment le small-ball a-t-il change la NBA ?", "golden": "Les Warriors de 2015+ ont popularise le 5-out, spacing, 3-point shooting", "keywords": ["Warriors", "spacing", "3-point", "small"]},

    # Betting (30)
    {"id": "nba-bet-001", "category": "betting", "question": "Qu'est-ce que le Kelly Criterion en paris sportifs NBA ?", "golden": "Formule de gestion de bankroll: f* = (bp - q) / b, ou b=cote decimale-1, p=proba victoire, q=1-p", "keywords": ["Kelly", "bankroll", "formule", "proba"]},
    {"id": "nba-bet-002", "category": "betting", "question": "What is CLV (Closing Line Value) in NBA betting?", "golden": "Difference between the odds you bet at and the closing odds — positive CLV indicates long-term profitability", "keywords": ["closing", "odds", "profitability", "CLV"]},
    {"id": "nba-bet-003", "category": "betting", "question": "Comment fonctionnent les spreads NBA ?", "golden": "Le favori doit gagner par plus de X points, l'outsider doit perdre par moins de X points ou gagner", "keywords": ["favori", "points", "favorite", "spread"]},
    {"id": "nba-bet-004", "category": "betting", "question": "What are player props in NBA betting?", "golden": "Bets on individual player performance: points over/under, rebounds, assists, 3-pointers made, etc.", "keywords": ["individual", "player", "over", "under", "performance"]},
    {"id": "nba-bet-005", "category": "betting", "question": "Quelle est la marge typique d'un bookmaker NBA ?", "golden": "3-5% de vig (vigorish/juice) sur les cotes standard, spreads -110/-110", "keywords": ["3", "5", "vig", "juice", "110"]},

    # Predictions / Draft (20)
    {"id": "nba-pred-001", "category": "predictions", "question": "Qui a ete le premier choix de la Draft 2023 ?", "golden": "Victor Wembanyama, selectionne par les San Antonio Spurs", "keywords": ["Wembanyama", "Spurs", "2023"]},
    {"id": "nba-pred-002", "category": "predictions", "question": "What factors predict NBA draft success?", "golden": "College stats, physical measurements, age, conference strength, combine performance, intangibles", "keywords": ["college", "measurements", "age", "combine"]},
    {"id": "nba-pred-003", "category": "predictions", "question": "Quel est l'impact moyen d'une blessure au ligament croise sur la carriere d'un joueur NBA ?", "golden": "Historiquement 12-18 mois de recuperation, baisse de performance de 10-20% dans les stats", "keywords": ["12", "18", "mois", "baisse", "10", "20"]},

    # GOAT Debates (20)
    {"id": "nba-goat-001", "category": "goat", "question": "Compare Michael Jordan et LeBron James en termes de stats de carriere", "golden": "MJ: 30.1 PPG, 6 titres, 5 MVP, 6 Finals MVP. LBJ: 27.1 PPG, 4 titres, 4 MVP, 4 Finals MVP, meilleur marqueur all-time", "keywords": ["Jordan", "LeBron", "30", "27", "MVP", "titres"]},
    {"id": "nba-goat-002", "category": "goat", "question": "Is Steph Curry the greatest shooter of all time?", "golden": "Yes by consensus — career 42-43% from 3, 3,747+ career threes (record), revolutionized the game", "keywords": ["Curry", "42", "43", "3747", "record", "three"]},
    {"id": "nba-goat-003", "category": "goat", "question": "Pourquoi Nikola Jokic est-il considere comme le meilleur centre offensif de l'histoire ?", "golden": "Triple-double machine, 3x MVP, playmaking historique pour un centre, PER record", "keywords": ["Jokic", "MVP", "triple-double", "playmaking", "centre"]},
    {"id": "nba-goat-004", "category": "goat", "question": "Who is the greatest defensive player in NBA history?", "golden": "Bill Russell (11 titles, 5 MVPs), Hakeem Olajuwon, Ben Wallace, or Tim Duncan depending on era", "keywords": ["Russell", "Hakeem", "defensive"]},
    {"id": "nba-goat-005", "category": "goat", "question": "Compare la dynastie Warriors (2015-2022) avec la dynastie Bulls (1991-1998)", "golden": "Warriors: 4 titres/6 Finals, shooting revolution. Bulls: 6 titres/6 Finals, deux three-peats, MJ domination", "keywords": ["Warriors", "Bulls", "4", "6", "three-peat"]},
]

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_answer(result, eval_q):
    """Check if answer contains expected keywords."""
    answer = result["answer"].lower()
    keywords = eval_q.get("keywords", [])
    if not keywords:
        return True, 1.0

    matches = sum(1 for kw in keywords if kw.lower() in answer)
    ratio = matches / len(keywords)
    passed = ratio >= 0.5  # At least half the keywords present
    return passed, ratio

def run_eval(questions=None, category=None, max_q=None):
    """Run evaluation on a set of questions."""
    qs = questions or EVAL_QUESTIONS
    if category:
        qs = [q for q in qs if q["category"] == category]
    if max_q:
        qs = random.sample(qs, min(max_q, len(qs)))

    results = []
    passed = 0
    total = len(qs)

    print(f"\n{'='*60}")
    print(f"NBA EVAL — {total} questions" + (f" (category: {category})" if category else ""))
    print(f"{'='*60}\n")

    for i, q in enumerate(qs, 1):
        print(f"[{i}/{total}] {q['question'][:70]}...")
        result = answer_question(q["question"])
        ok, ratio = evaluate_answer(result, q)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "golden": q.get("golden", ""),
            "answer": result["answer"][:500],
            "keywords_matched": ratio,
            "passed": ok,
            "latency_ms": result["latency_ms"],
            "sources": result["sources"],
        })

        print(f"  {status} ({ratio:.0%} keywords) — {result['latency_ms']}ms\n")
        time.sleep(0.5)  # Rate limit

    accuracy = passed / total * 100 if total else 0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 1),
        "category": category or "all",
        "avg_latency_ms": int(sum(r["latency_ms"] for r in results) / total) if total else 0,
        "results": results,
    }

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_file = ROOT / "data" / "eval" / f"eval-{ts}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} ({accuracy:.1f}%)")
    print(f"Average latency: {summary['avg_latency_ms']}ms")
    print(f"Saved to: {out_file}")
    print(f"{'='*60}\n")

    return summary

# ── Daemon mode ───────────────────────────────────────────────────────────────
def daemon_loop(interval=300, max_q=20):
    """Continuous self-testing loop."""
    cycle = 0
    categories = ["stats", "team_history", "game_analysis", "betting", "predictions", "goat"]

    print(f"NBA Agent daemon started — {max_q}q every {interval}s")

    while True:
        cycle += 1
        cat = categories[(cycle - 1) % len(categories)]
        print(f"\n[CYCLE {cycle}] Category: {cat}")

        try:
            summary = run_eval(category=cat, max_q=max_q)

            # Log to metrics JSONL
            metrics_file = ROOT / "logs" / "metrics.jsonl"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "cycle": cycle,
                    "category": cat,
                    "accuracy": summary["accuracy"],
                    "total": summary["total"],
                    "passed": summary["passed"],
                    "avg_latency_ms": summary["avg_latency_ms"],
                    "timestamp": summary["timestamp"],
                }) + "\n")

            # Sync to mon-ipad
            sync_to_control(summary, cycle)

        except Exception as e:
            print(f"[ERROR] Cycle {cycle}: {e}")
            err_file = ROOT / "logs" / "errors.jsonl"
            err_file.parent.mkdir(parents=True, exist_ok=True)
            with open(err_file, "a") as f:
                f.write(json.dumps({"cycle": cycle, "error": str(e), "ts": datetime.now(timezone.utc).isoformat()}) + "\n")

        print(f"Sleeping {interval}s...")
        time.sleep(interval)

def sync_to_control(summary, cycle):
    """Push results to mon-ipad data directory."""
    sync_dir = Path("/home/termius/mon-ipad/data/nba-agent")
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Latest results
    (sync_dir / "latest-eval.json").write_text(json.dumps({
        "cycle": cycle,
        "accuracy": summary["accuracy"],
        "total": summary["total"],
        "passed": summary["passed"],
        "category": summary["category"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "timestamp": summary["timestamp"],
    }, indent=2))

    # Append to history
    with open(sync_dir / "eval-history.jsonl", "a") as f:
        f.write(json.dumps({
            "cycle": cycle,
            "accuracy": summary["accuracy"],
            "category": summary["category"],
            "timestamp": summary["timestamp"],
        }) + "\n")

# ── Interactive mode ──────────────────────────────────────────────────────────
def interactive():
    """Interactive Q&A mode."""
    print("\nNBA Agent — Expert IA Basketball")
    print("Tape 'quit' pour sortir\n")

    while True:
        try:
            q = input("NBA> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        result = answer_question(q)
        print(f"\n{result['answer']}")
        print(f"\n[{result['latency_ms']}ms | {result['sources']} sources | live={result['has_live_data']}]\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nomos NBA Agent")
    parser.add_argument("--daemon", type=int, metavar="INTERVAL", help="Daemon mode with interval in seconds")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick eval (10 questions)")
    parser.add_argument("--category", type=str, help="Eval category: stats, team_history, game_analysis, betting, predictions, goat")
    parser.add_argument("--ask", type=str, help="Ask a single question")
    args = parser.parse_args()

    if args.daemon:
        daemon_loop(interval=args.daemon, max_q=10)
    elif args.eval:
        run_eval(category=args.category)
    elif args.quick:
        run_eval(max_q=10, category=args.category)
    elif args.ask:
        result = answer_question(args.ask)
        print(f"\n{result['answer']}")
        print(f"\n[{result['latency_ms']}ms | {result['sources']} sources]")
    else:
        interactive()
