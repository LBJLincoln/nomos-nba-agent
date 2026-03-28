# NBA 2025-26 Historical Odds - Sources

## Overview

`nba_2025-26_odds.csv` contains closing moneyline odds for all NBA regular season
games played Oct 2025 - Mar 2026. Coverage: 1,100+ games.

Columns: `date, home_team, away_team, moneyline_home, moneyline_away, spread_home, total, book, source`

**Moneyline format:**
- `mgm_kaggle` rows: American format (e.g. -250, 200)
- `sbr_scrape` rows: American format (e.g. -140, 115)  
- `local_snapshot_decimal` rows: Decimal format (e.g. 1.40, 3.50) — Pinnacle/Bovada

---

## Source 1: MGM Grand Kaggle Dataset

**Coverage:** Oct 21, 2025 – Feb 12, 2026 (808 games)  
**Bookmaker:** BetMGM (US sportsbook)  
**Type:** Pregame opening odds (not closing, but very close for most games)  
**Format:** American moneylines, spreads, totals, public betting percentages

**Kaggle dataset:**
- `ref`: caseydurfee/mgm-grand-nba-betting-data  
- `file`: all_odds.csv (1.5MB)  
- Updated: 2026-02-15  
- License: CC-BY-SA-4.0

**Download:**
```bash
kaggle datasets download caseydurfee/mgm-grand-nba-betting-data --path /tmp/mgm-nba/ --unzip
```

**Key columns in source:**
- `money_home_odds` / `money_away_odds`: American moneylines
- `spread_home_points`: home team spread
- `total_over_points`: game total
- `money_home_wager_percentage`: public money %
- `money_home_stake_percentage`: public bet ticket %

**Why BetMGM:** BetMGM is a major US market maker with sharp lines close to Pinnacle
for popular games. For CLV analysis, it's a reasonable proxy.

---

## Source 2: SportsBettingReview.com (SBR) Scraper

**Coverage:** Feb 13, 2026 – Mar 28, 2026 (291 games)  
**Bookmakers:** BetMGM (primary), FanDuel, DraftKings, Caesars, Bet365, Fanatics  
**Type:** Closing lines (last posted odds before tip-off)  
**Format:** American moneylines

**URL pattern:**
```
https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/?date=YYYY-MM-DD
```

**Data extraction:**  
Page embeds JSON in `<script id="__NEXT_DATA__">`. Path:
```
props.pageProps.oddsTables[0].oddsTableModel.gameRows[n].oddsViews[n].currentLine.homeOdds
```

**Rate limit:** 2.5 seconds between requests (safe). SBR does not require login.

**Notes:**
- Returns closing lines (currentLine) AND opening lines (openingLine)
- Typically 6 bookmakers per game: BetMGM, FanDuel, Caesars, Bet365, DraftKings, Fanatics
- Pinnacle is NOT listed (US market only)
- Script prefers BetMGM → FanDuel → DraftKings → Caesars → Bet365 → Fanatics

---

## Source 3: Local Odds API Snapshots

**Coverage:** Mar 15–17, 2026 (15 games), Mar 28 (14 games via live-odds.json)  
**Bookmakers:** Pinnacle, FanDuel, Bovada  
**Type:** Pre-game odds at time of snapshot (not closing)  
**Format:** Decimal odds (e.g. 1.40 = -250)

**Location:**
- `/home/termius/nomos-nba-agent/data/odds-YYYYMMDD-HHMM.json` — The Odds API snapshots
- `/home/termius/mon-ipad/data/nba-agent/live-odds.json` — Current day (via nba-daily-odds.py cron)

**Note:** The Odds API free quota (500 req/mo) was exhausted after Mar 17, 2026.
Historical endpoint requires paid plan (~$99/mo). SBR scraper covers this gap for free.

**Conversion from decimal to American:**
```python
def decimal_to_american(dec):
    if dec >= 2.0:
        return int((dec - 1) * 100)   # e.g. 3.5 -> +250
    else:
        return int(-100 / (dec - 1))  # e.g. 1.4 -> -250
```

---

## Gaps and Known Issues

| Period | Status |
|--------|--------|
| Oct 21, 2025 – Feb 12, 2026 | COMPLETE (808 games, MGM) |
| Feb 13–18, 2026 | All-Star break — no NBA games |
| Feb 19, 2026 – Mar 28, 2026 | COMPLETE (291 games, SBR closing lines) |

**Team name normalization:**
- MGM dataset uses city names (e.g. "Houston" not "Houston Rockets")
- SBR uses full names (e.g. "Houston Rockets")
- Both are normalized to full names in the output CSV

**Preseason games (Oct 2-20, 2025):**
Not included. MGM dataset starts Oct 21 (season opener: Lakers vs Warriors).

---

## The Odds API Historical Endpoint (NOT used — paid plan required)

```
GET https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey=KEY&date=YYYY-MM-DDTHH:MM:SSZ
```

- Available from June 6, 2020 (5-min snapshots since Sep 2022)
- Requires paid plan (~$99/mo for "historical" tier)
- Key: ODDS_API_KEY in /home/termius/nomos-nba-agent/.env.local
- Free tier: 500 req/mo, current + future games only

---

## Adding Future Dates

To extend coverage (run after each game day):
```bash
python3 /home/termius/nomos-nba-agent/scripts/scrape_season_odds.py \
  --source sbr \
  --from-date YYYY-MM-DD \
  --to-date YYYY-MM-DD \
  --output /home/termius/nomos-nba-agent/data/historical-odds/nba_2025-26_odds.csv
```

To append to existing file (avoids re-fetching MGM):
```bash
python3 /home/termius/nomos-nba-agent/scripts/scrape_season_odds.py \
  --source sbr \
  --sbr-start YYYY-MM-DD \
  --to-date YYYY-MM-DD \
  --output /tmp/new_odds.csv
# Then manually append to nba_2025-26_odds.csv
```

---

## Other Sources Investigated

| Source | Status | Reason Not Used |
|--------|--------|-----------------|
| sportsbookreviewsonline.com | Dead | Last updated 2022-23, no 2024-25 or 2025-26 |
| OddsPortal | JS-rendered | Requires Selenium/Playwright, complex scraping |
| The Odds API historical | Paid only | $99/mo, free tier = 500 req current games only |
| GitHub repos | Not found | No public repos with 2025-26 moneylines |
| Kaggle: cviaxmiwnptr | Has 2023-25 scores but no moneylines for those years |
| Kaggle: zachht basketball | Soccer/football only, no NBA |
| oliviersportsdata | Sample only, paid full dataset |
| ActionNetwork API | 404/401 for historical dates |
| Pinnacle guest API | Auth required for markets endpoint |
| VegasInsider | JS-rendered |
| OddsShark database | No CSV export, JS-rendered |
