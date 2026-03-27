#!/usr/bin/env python3
"""
=============================================================================
 UPDATE RESULTS — NBA Model V11
 Remplit automatiquement le champ "outcome" dans nba_history.json
 en interrogeant l'API gratuite balldontlie.io (pas de clé requise).
=============================================================================
 Usage :
   python update_results.py              # Met à jour tous les matchs
   python update_results.py --dry-run    # Simule sans écrire
=============================================================================
"""

import json, os, time, sys
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("[WARN] requests non installé → pip install requests")

HISTORY_FILE  = "nba_history.json"
API_BASE      = "https://api.balldontlie.io/v1"
SLEEP_BETWEEN = 0.6   # Respecte le rate-limit 60 req/min (tier gratuit)

# Mapping abbreviations → noms API balldontlie
TEAM_ABBR_TO_NAME = {
    "ATL":"Hawks",  "BOS":"Celtics", "BKN":"Nets",   "CHA":"Hornets",
    "CHI":"Bulls",  "CLE":"Cavaliers","DAL":"Mavericks","DEN":"Nuggets",
    "DET":"Pistons","GSW":"Warriors","HOU":"Rockets", "IND":"Pacers",
    "LAC":"Clippers","LAL":"Lakers","MEM":"Grizzlies","MIA":"Heat",
    "MIL":"Bucks",  "MIN":"Timberwolves","NOP":"Pelicans","NYK":"Knicks",
    "OKC":"Thunder","ORL":"Magic",  "PHI":"76ers",   "PHX":"Suns",
    "POR":"Trail Blazers","SAC":"Kings","SAS":"Spurs","TOR":"Raptors",
    "UTA":"Jazz",   "WAS":"Wizards",
}


def fetch_game_result(date_str: str, home_abbr: str, away_abbr: str) -> int | None:
    """
    Retourne 1 si home gagne, 0 si away gagne, None si introuvable.
    date_str format : "YYYY-MM-DD"
    """
    if not REQUESTS_OK:
        return None

    url    = f"{API_BASE}/games"
    params = {"dates[]": date_str, "per_page": 30}

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        games = resp.json().get("data", [])
    except Exception as e:
        print(f"  [API ERROR] {e}")
        return None

    home_name = TEAM_ABBR_TO_NAME.get(home_abbr.upper(), "")
    away_name = TEAM_ABBR_TO_NAME.get(away_abbr.upper(), "")

    for g in games:
        ht = g.get("home_team", {}).get("full_name", "")
        vt = g.get("visitor_team", {}).get("full_name", "")

        if home_name in ht and away_name in vt:
            hs = g.get("home_team_score", 0)
            vs = g.get("visitor_team_score", 0)
            if hs > 0 or vs > 0:                    # Match terminé
                return 1 if hs > vs else 0
            return None                              # Score = 0 → pas encore joué

    return None                                      # Match non trouvé


def update_outcomes(dry_run: bool = False) -> dict:
    if not os.path.exists(HISTORY_FILE):
        print(f"[ERROR] {HISTORY_FILE} introuvable")
        return {}

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    pending  = [e for e in history if e.get("outcome") is None]
    updated  = 0
    skipped  = 0
    not_found = 0

    print(f"\n{'='*55}")
    print(f"  UPDATE RESULTS — {len(pending)} matchs en attente")
    print(f"{'='*55}")

    for entry in pending:
        teams = entry.get("teams", "")
        date  = entry.get("date", "")

        try:
            home_abbr, away_abbr = [t.strip() for t in teams.split("vs")]
        except ValueError:
            print(f"  ⚠️  Format inconnu : {teams}")
            skipped += 1
            continue

        # Ne pas essayer les matchs d'aujourd'hui (résultat pas encore dispo)
        try:
            match_date = datetime.strptime(date, "%Y-%m-%d")
            if match_date.date() >= datetime.today().date():
                print(f"  ⏳ {teams} ({date}) — match futur, ignoré")
                skipped += 1
                continue
        except ValueError:
            pass

        outcome = fetch_game_result(date, home_abbr, away_abbr)
        time.sleep(SLEEP_BETWEEN)

        if outcome is not None:
            result_str = "✅ HOME WIN" if outcome == 1 else "❌ AWAY WIN"
            print(f"  {teams} ({date}) → {result_str}")
            if not dry_run:
                entry["outcome"] = outcome
            updated += 1
        else:
            print(f"  ❓ {teams} ({date}) → résultat non disponible")
            not_found += 1

    if not dry_run and updated > 0:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
        print(f"\n  💾 {HISTORY_FILE} mis à jour ({updated} résultats ajoutés)")

    print(f"\n  Résumé : {updated} mis à jour | {skipped} ignorés | {not_found} introuvables")
    print(f"{'='*55}\n")

    return {"updated": updated, "skipped": skipped, "not_found": not_found}


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("[MODE] Dry-run activé — aucune écriture")
    update_outcomes(dry_run=dry)
