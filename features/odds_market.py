#!/usr/bin/env python3
"""
NBA Odds Market Engine — Real-Time Market Microstructure Analysis
=================================================================
Pulls REAL odds data from The Odds API v4, computes market microstructure
features used by Starlizard/Priomha-class quant models:

  - Opening vs closing line movement (direction + magnitude)
  - Consensus spread / moneyline across all bookmakers
  - Sharp vs square money detection (Pinnacle/Circa vs DK/FD divergence)
  - Steam moves (rapid line movement across 3+ books)
  - Reverse line movement (line opposite to public flow)
  - Juice / overround analysis (market efficiency per book)
  - Best available odds across all books
  - Player prop consensus and line movement
  - Full value bet detection with Kelly sizing

Data Architecture:
  - Live odds saved to data/odds/YYYYMMDD/live-HHMMSS.json
  - Historical odds saved to data/odds/YYYYMMDD/historical.json
  - Player props saved to data/odds/YYYYMMDD/props-{game_id}.json

API: The Odds API v4 — https://the-odds-api.com/liveapi/guides/v4/
"""

import os
import sys
import json
import ssl
import math
import time
import urllib.request
import urllib.parse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("odds_market")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ── SSL context (permissive for HF Spaces / cloud envs) ─────────────────────
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── API Config ───────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "959eab3a6b0b731ef1766579e355f51d")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Rate limit: sleep between API calls to stay within quota
API_SLEEP_SECONDS = 1.2

# ── Market definitions ───────────────────────────────────────────────────────
GAME_MARKETS = ["h2h", "spreads", "totals"]
PLAYER_PROP_MARKETS = [
    "player_points",
    "player_assists",
    "player_rebounds",
    "player_threes",
    "player_blocks",
    "player_steals",
]
ALL_REGIONS = "us,us2,eu,uk,au"

# ── Bookmaker classification ────────────────────────────────────────────────
# Sharp books: low margins, move first, used by professional bettors
SHARP_BOOKS = {
    "pinnacle", "circa", "lowvig", "betcris", "bookmaker",
    "betonlineag", "heritage", "5dimes",
}

# Square books: higher margins, follow sharp lines, retail-heavy
SQUARE_BOOKS = {
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus",
    "espnbet", "betrivers", "hardrockbet", "betparx", "wynnbet",
    "twinspires", "fliff", "superbook",
}

# All tracked bookmakers (priority order)
ALL_BOOKMAKERS = [
    "pinnacle", "circa", "lowvig", "betonlineag",
    "draftkings", "fanduel", "betmgm", "caesars",
    "bet365", "unibet", "betway", "williamhill",
    "bovada", "pointsbetus", "betrivers", "888sport",
    "espnbet", "hardrockbet", "mybookieag", "superbook",
    "betparx", "fliff", "twinspires", "wynnbet",
    "paddypower", "skybet", "boylesports", "betvictor",
    "onexbet", "betclic_fr", "unibet_fr", "parionssport_fr",
    "winamax_fr", "tipico_de", "everygame",
]


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, timeout: int = 30) -> Tuple[Any, int, Dict[str, str]]:
    """
    GET request returning (parsed_json, status_code, response_headers).
    Response headers include API usage info (x-requests-remaining, etc).
    """
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.getheaders()}
            body = json.loads(resp.read().decode("utf-8"))
            return body, resp.status, headers
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        logger.error(f"HTTP {e.code}: {error_body[:300]}")
        return {"error": str(e), "detail": error_body[:500]}, e.code, {}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}, 0, {}


def _log_api_usage(headers: Dict[str, str]):
    """Log API quota usage from response headers."""
    remaining = headers.get("x-requests-remaining", "?")
    used = headers.get("x-requests-used", "?")
    logger.info(f"API quota: {used} used, {remaining} remaining")


# ══════════════════════════════════════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def implied_prob_to_decimal(prob: float) -> float:
    """Convert probability to fair decimal odds (no vig)."""
    if prob <= 0:
        return 100.0
    if prob >= 1.0:
        return 1.0
    return 1.0 / prob


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    else:
        return round(-100 / (decimal_odds - 1.0))


def compute_overround(odds_list: List[float]) -> float:
    """
    Compute overround (vig/juice) from a set of outcome odds.
    Fair market = 1.0 (100%). Typical NBA h2h = 1.04-1.08 (4-8% vig).
    """
    if not odds_list or any(o <= 0 for o in odds_list):
        return 0.0
    return sum(1.0 / o for o in odds_list)


def remove_vig(odds_list: List[float]) -> List[float]:
    """Remove vig to get fair probabilities."""
    overround = compute_overround(odds_list)
    if overround <= 0:
        return [0.5] * len(odds_list)
    return [(1.0 / o) / overround for o in odds_list]


def kelly_fraction(prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion.
    f* = fraction * (b*p - q) / b
    where b = decimal_odds - 1, p = estimated prob, q = 1 - p
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    if full_kelly <= 0:
        return 0.0
    return fraction * full_kelly


# ══════════════════════════════════════════════════════════════════════════════
# ODDS MARKET ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class OddsMarketEngine:
    """
    Real-time odds market microstructure engine.

    Fetches live and historical odds from The Odds API v4, computes
    Starlizard-grade market features for NBA quant modeling.

    Usage:
        engine = OddsMarketEngine()
        odds = engine.fetch_live_odds()
        features = engine.compute_market_features(odds[0])
        value = engine.find_value_bets(model_predictions, odds)
    """

    def __init__(self, api_key: str = None, data_dir: str = None):
        self.api_key = api_key or ODDS_API_KEY
        self.data_dir = Path(data_dir) if data_dir else ROOT / "data" / "odds"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for rate limiting
        self._last_request_time = 0.0

        # In-memory odds history for steam move detection
        self._odds_snapshots: List[Dict] = []

        # Load any existing snapshots from today for context
        self._load_today_snapshots()

    def _load_today_snapshots(self):
        """Load today's previously saved snapshots for line movement tracking."""
        today_dir = self.data_dir / datetime.now(timezone.utc).strftime("%Y%m%d")
        if not today_dir.exists():
            return
        live_files = sorted(today_dir.glob("live-*.json"))
        for f in live_files[-10:]:  # last 10 snapshots max
            try:
                data = json.loads(f.read_text())
                self._odds_snapshots.append(data)
            except Exception:
                continue
        if self._odds_snapshots:
            logger.info(f"Loaded {len(self._odds_snapshots)} previous snapshots from today")

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < API_SLEEP_SECONDS:
            time.sleep(API_SLEEP_SECONDS - elapsed)
        self._last_request_time = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # FETCHING
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_live_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = ALL_REGIONS,
        save: bool = True,
    ) -> List[Dict]:
        """
        Fetch current live odds for all today's NBA games.

        Returns list of game dicts from The Odds API, each containing
        bookmakers with their market outcomes.
        """
        if not self.api_key:
            logger.error("No ODDS_API_KEY configured")
            return []

        self._rate_limit()

        url = (
            f"{ODDS_API_BASE}/sports/{SPORT}/odds/"
            f"?apiKey={self.api_key}"
            f"&regions={regions}"
            f"&markets={markets}"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
        )

        data, status, headers = _http_get(url)
        _log_api_usage(headers)

        if isinstance(data, dict) and "error" in data:
            logger.error(f"API error fetching live odds: {data}")
            return []

        if not isinstance(data, list):
            logger.error(f"Unexpected response type: {type(data)}")
            return []

        logger.info(f"Fetched live odds for {len(data)} games from {_count_bookmakers(data)} bookmakers")

        # Save snapshot
        if save and data:
            self._save_snapshot(data, "live")

        # Add to in-memory history for steam detection
        self._odds_snapshots.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "games": data,
        })
        # Keep last 24 snapshots (~24 hours if fetched hourly)
        if len(self._odds_snapshots) > 24:
            self._odds_snapshots = self._odds_snapshots[-24:]

        return data

    def fetch_player_props(
        self,
        game_id: str,
        markets: str = None,
        regions: str = ALL_REGIONS,
        save: bool = True,
    ) -> List[Dict]:
        """
        Fetch player prop lines for a specific game.

        Args:
            game_id: The Odds API game event ID (e.g., "ffe5b43309eff2a1...")
            markets: Comma-separated prop markets. Defaults to all 6.
            regions: Bookmaker regions to include.

        Returns list of bookmaker prop data for the game.
        """
        if not self.api_key:
            logger.error("No ODDS_API_KEY configured")
            return []

        if markets is None:
            markets = ",".join(PLAYER_PROP_MARKETS)

        self._rate_limit()

        url = (
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{game_id}/odds/"
            f"?apiKey={self.api_key}"
            f"&regions={regions}"
            f"&markets={markets}"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
        )

        data, status, headers = _http_get(url)
        _log_api_usage(headers)

        if isinstance(data, dict) and "error" in data:
            logger.error(f"API error fetching props for {game_id}: {data}")
            return []

        # The API returns the game object with bookmakers containing prop markets
        if isinstance(data, dict):
            bookmakers = data.get("bookmakers", [])
        elif isinstance(data, list) and data:
            bookmakers = data[0].get("bookmakers", []) if data else []
        else:
            bookmakers = []

        prop_count = sum(
            len(m.get("outcomes", []))
            for bk in bookmakers
            for m in bk.get("markets", [])
        )
        logger.info(f"Fetched {prop_count} prop lines for game {game_id[:12]}... from {len(bookmakers)} books")

        if save and bookmakers:
            self._save_snapshot(
                {"game_id": game_id, "bookmakers": bookmakers},
                f"props-{game_id[:12]}",
            )

        return bookmakers

    def fetch_historical_odds(
        self,
        date: str,
        markets: str = "h2h,spreads,totals",
        regions: str = ALL_REGIONS,
        save: bool = True,
    ) -> List[Dict]:
        """
        Fetch historical odds snapshot for a past date (for backtesting).

        Args:
            date: ISO date string, e.g. "2026-03-15T12:00:00Z"
            markets: Markets to fetch.

        Returns list of game dicts with odds as of that timestamp.
        Note: Historical API requires a paid plan on The Odds API.
        """
        if not self.api_key:
            logger.error("No ODDS_API_KEY configured")
            return []

        self._rate_limit()

        url = (
            f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds/"
            f"?apiKey={self.api_key}"
            f"&regions={regions}"
            f"&markets={markets}"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
            f"&date={urllib.parse.quote(date)}"
        )

        data, status, headers = _http_get(url)
        _log_api_usage(headers)

        if isinstance(data, dict) and "error" in data:
            logger.warning(f"Historical API error (may require paid plan): {data}")
            return []

        # Historical endpoint wraps data in {"timestamp": ..., "data": [...]}
        games = []
        if isinstance(data, dict):
            games = data.get("data", [])
            ts = data.get("timestamp", date)
            logger.info(f"Fetched historical odds for {len(games)} games at {ts}")
        elif isinstance(data, list):
            games = data
            logger.info(f"Fetched historical odds for {len(games)} games")

        if save and games:
            date_str = date[:10].replace("-", "")
            self._save_snapshot(
                {"date": date, "games": games},
                f"historical-{date_str}",
            )

        return games

    # ──────────────────────────────────────────────────────────────────────────
    # MARKET FEATURES (per game)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_market_features(self, game: Dict) -> Dict[str, float]:
        """
        Compute full market microstructure feature set for a single game.

        Returns dict of ~40 numeric features suitable for ML model input.

        Features computed:
          - Moneyline implied probabilities (home/away, consensus)
          - Spread consensus and standard deviation
          - Total consensus and standard deviation
          - Sharp vs square divergence (Pinnacle line vs DK/FD)
          - Overround per book type (sharp vs square)
          - Best available odds (home ML, away ML)
          - Line movement from earliest snapshot
          - Steam move indicator
          - Number of bookmakers offering the game
        """
        features = {}
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        bookmakers = game.get("bookmakers", [])
        game_id = game.get("id", "")

        if not bookmakers:
            return self._empty_features()

        # ── Extract all odds by market ──
        h2h_home_odds = []
        h2h_away_odds = []
        spread_home_points = []
        spread_away_points = []
        spread_home_prices = []
        spread_away_prices = []
        total_points = []
        total_over_prices = []
        total_under_prices = []

        sharp_h2h_home = []
        sharp_h2h_away = []
        square_h2h_home = []
        square_h2h_away = []

        sharp_spreads = []
        square_spreads = []
        sharp_totals = []
        square_totals = []

        book_overrounds_h2h = {}

        for bk in bookmakers:
            bk_key = bk.get("key", "")
            is_sharp = bk_key in SHARP_BOOKS
            is_square = bk_key in SQUARE_BOOKS

            for market in bk.get("markets", []):
                mkey = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h" and len(outcomes) >= 2:
                    home_price = None
                    away_price = None
                    for o in outcomes:
                        if o.get("name") == home_team:
                            home_price = o["price"]
                            h2h_home_odds.append(home_price)
                        elif o.get("name") == away_team:
                            away_price = o["price"]
                            h2h_away_odds.append(away_price)

                    if home_price and away_price:
                        overround = compute_overround([home_price, away_price])
                        book_overrounds_h2h[bk_key] = overround

                        if is_sharp:
                            sharp_h2h_home.append(home_price)
                            sharp_h2h_away.append(away_price)
                        if is_square:
                            square_h2h_home.append(home_price)
                            square_h2h_away.append(away_price)

                elif mkey == "spreads" and len(outcomes) >= 2:
                    for o in outcomes:
                        point = o.get("point", 0)
                        price = o.get("price", 1.91)
                        if o.get("name") == home_team:
                            spread_home_points.append(point)
                            spread_home_prices.append(price)
                            if is_sharp:
                                sharp_spreads.append(point)
                            if is_square:
                                square_spreads.append(point)
                        elif o.get("name") == away_team:
                            spread_away_points.append(point)
                            spread_away_prices.append(price)

                elif mkey == "totals" and len(outcomes) >= 2:
                    for o in outcomes:
                        point = o.get("point", 0)
                        price = o.get("price", 1.91)
                        if o.get("name") == "Over":
                            total_points.append(point)
                            total_over_prices.append(price)
                            if is_sharp:
                                sharp_totals.append(point)
                            if is_square:
                                square_totals.append(point)
                        elif o.get("name") == "Under":
                            total_under_prices.append(price)

        # ── Moneyline features ──
        features["n_bookmakers"] = len(bookmakers)
        features["n_books_h2h"] = len(h2h_home_odds)
        features["n_books_spreads"] = len(spread_home_points)
        features["n_books_totals"] = len(total_points)

        if h2h_home_odds:
            features["ml_home_best"] = max(h2h_home_odds)
            features["ml_home_worst"] = min(h2h_home_odds)
            features["ml_home_avg"] = _mean(h2h_home_odds)
            features["ml_home_std"] = _std(h2h_home_odds)
            features["ml_home_implied_prob"] = decimal_to_implied_prob(features["ml_home_avg"])
        else:
            features.update({
                "ml_home_best": 0, "ml_home_worst": 0, "ml_home_avg": 0,
                "ml_home_std": 0, "ml_home_implied_prob": 0.5,
            })

        if h2h_away_odds:
            features["ml_away_best"] = max(h2h_away_odds)
            features["ml_away_worst"] = min(h2h_away_odds)
            features["ml_away_avg"] = _mean(h2h_away_odds)
            features["ml_away_std"] = _std(h2h_away_odds)
            features["ml_away_implied_prob"] = decimal_to_implied_prob(features["ml_away_avg"])
        else:
            features.update({
                "ml_away_best": 0, "ml_away_worst": 0, "ml_away_avg": 0,
                "ml_away_std": 0, "ml_away_implied_prob": 0.5,
            })

        # Consensus implied probabilities (vig-removed)
        if h2h_home_odds and h2h_away_odds:
            avg_home = _mean(h2h_home_odds)
            avg_away = _mean(h2h_away_odds)
            fair_probs = remove_vig([avg_home, avg_away])
            features["consensus_home_prob"] = fair_probs[0]
            features["consensus_away_prob"] = fair_probs[1]
        else:
            features["consensus_home_prob"] = 0.5
            features["consensus_away_prob"] = 0.5

        # Overround (market efficiency)
        if h2h_home_odds and h2h_away_odds:
            avg_overround = _mean(list(book_overrounds_h2h.values())) if book_overrounds_h2h else 1.05
            features["avg_overround_h2h"] = avg_overround
            features["min_overround_h2h"] = min(book_overrounds_h2h.values()) if book_overrounds_h2h else 1.0
            features["max_overround_h2h"] = max(book_overrounds_h2h.values()) if book_overrounds_h2h else 1.1
        else:
            features["avg_overround_h2h"] = 1.05
            features["min_overround_h2h"] = 1.0
            features["max_overround_h2h"] = 1.1

        # ── Sharp vs Square divergence ──
        # This is the KEY feature: when sharp books disagree with square books,
        # it signals informed money. Pinnacle moves first, DK/FD follow.
        if sharp_h2h_home and square_h2h_home:
            sharp_home_avg = _mean(sharp_h2h_home)
            square_home_avg = _mean(square_h2h_home)
            # Positive = sharp books give home BETTER odds (more value on home)
            features["sharp_square_div_home"] = sharp_home_avg - square_home_avg
        else:
            features["sharp_square_div_home"] = 0.0

        if sharp_h2h_away and square_h2h_away:
            sharp_away_avg = _mean(sharp_h2h_away)
            square_away_avg = _mean(square_h2h_away)
            features["sharp_square_div_away"] = sharp_away_avg - square_away_avg
        else:
            features["sharp_square_div_away"] = 0.0

        # Sharp overround vs square overround
        sharp_ors = [v for k, v in book_overrounds_h2h.items() if k in SHARP_BOOKS]
        square_ors = [v for k, v in book_overrounds_h2h.items() if k in SQUARE_BOOKS]
        features["sharp_avg_overround"] = _mean(sharp_ors) if sharp_ors else 1.02
        features["square_avg_overround"] = _mean(square_ors) if square_ors else 1.06

        # ── Spread features ──
        if spread_home_points:
            features["spread_consensus"] = _mean(spread_home_points)
            features["spread_std"] = _std(spread_home_points)
            features["spread_best_home"] = min(spread_home_points)  # most favorable for home
            features["spread_worst_home"] = max(spread_home_points)
            features["spread_range"] = max(spread_home_points) - min(spread_home_points)
        else:
            features.update({
                "spread_consensus": 0, "spread_std": 0,
                "spread_best_home": 0, "spread_worst_home": 0, "spread_range": 0,
            })

        # Sharp vs square spread divergence
        if sharp_spreads and square_spreads:
            features["sharp_square_spread_div"] = _mean(sharp_spreads) - _mean(square_spreads)
        else:
            features["sharp_square_spread_div"] = 0.0

        # ── Total features ──
        if total_points:
            features["total_consensus"] = _mean(total_points)
            features["total_std"] = _std(total_points)
            features["total_range"] = max(total_points) - min(total_points)
        else:
            features.update({"total_consensus": 0, "total_std": 0, "total_range": 0})

        if sharp_totals and square_totals:
            features["sharp_square_total_div"] = _mean(sharp_totals) - _mean(square_totals)
        else:
            features["sharp_square_total_div"] = 0.0

        # ── Line movement features ──
        movement = self._compute_line_movement(game_id, home_team, away_team)
        features.update(movement)

        # ── Steam move detection ──
        steam = self._detect_steam_moves(game_id, home_team, away_team)
        features.update(steam)

        return features

    def compute_prop_features(self, player: str, game_id: str, bookmakers: List[Dict] = None) -> Dict[str, float]:
        """
        Compute market features for a player's prop lines.

        Args:
            player: Player name (e.g., "Trae Young")
            game_id: Game event ID
            bookmakers: Pre-fetched prop bookmakers (or will fetch)

        Returns dict of prop features:
          - Consensus over/under line per market
          - Line spread across books
          - Best over/under odds
          - Juice differential
        """
        if bookmakers is None:
            bookmakers = self.fetch_player_props(game_id)

        features = {}

        for prop_market in PLAYER_PROP_MARKETS:
            short_name = prop_market.replace("player_", "")  # e.g., "points"

            lines = []       # over/under lines
            over_odds = []   # price for over
            under_odds = []  # price for under

            for bk in bookmakers:
                for market in bk.get("markets", []):
                    if market.get("key") != prop_market:
                        continue
                    for outcome in market.get("outcomes", []):
                        desc = outcome.get("description", "")
                        if player.lower() not in desc.lower():
                            continue
                        point = outcome.get("point")
                        price = outcome.get("price", 0)
                        name = outcome.get("name", "")

                        if point is not None:
                            lines.append(point)
                        if name == "Over" and price > 0:
                            over_odds.append(price)
                        elif name == "Under" and price > 0:
                            under_odds.append(price)

            if lines:
                features[f"prop_{short_name}_consensus_line"] = _mean(lines)
                features[f"prop_{short_name}_line_std"] = _std(lines)
                features[f"prop_{short_name}_line_range"] = max(lines) - min(lines)
                features[f"prop_{short_name}_n_books"] = len(lines) / 2  # each book has over+under
            else:
                features[f"prop_{short_name}_consensus_line"] = 0.0
                features[f"prop_{short_name}_line_std"] = 0.0
                features[f"prop_{short_name}_line_range"] = 0.0
                features[f"prop_{short_name}_n_books"] = 0.0

            if over_odds:
                features[f"prop_{short_name}_best_over"] = max(over_odds)
                features[f"prop_{short_name}_avg_over"] = _mean(over_odds)
            else:
                features[f"prop_{short_name}_best_over"] = 0.0
                features[f"prop_{short_name}_avg_over"] = 0.0

            if under_odds:
                features[f"prop_{short_name}_best_under"] = max(under_odds)
                features[f"prop_{short_name}_avg_under"] = _mean(under_odds)
            else:
                features[f"prop_{short_name}_best_under"] = 0.0
                features[f"prop_{short_name}_avg_under"] = 0.0

            # Juice analysis on this prop
            if over_odds and under_odds:
                avg_over = _mean(over_odds)
                avg_under = _mean(under_odds)
                features[f"prop_{short_name}_overround"] = compute_overround([avg_over, avg_under])
                # Positive = market leans under (over is juiced more)
                features[f"prop_{short_name}_juice_lean"] = (
                    decimal_to_implied_prob(avg_over) - decimal_to_implied_prob(avg_under)
                )
            else:
                features[f"prop_{short_name}_overround"] = 1.05
                features[f"prop_{short_name}_juice_lean"] = 0.0

        return features

    def find_value_bets(
        self,
        predictions: Dict[str, Dict],
        odds: List[Dict],
        kelly_frac: float = 0.25,
        min_edge: float = 0.02,
        bankroll: float = 100.0,
    ) -> List[Dict]:
        """
        Compare model probabilities to market implied probabilities.

        Args:
            predictions: Dict keyed by "home_team vs away_team" or game_id.
                Each value has: {"home_win_prob": float, "away_win_prob": float,
                                  "predicted_spread": float, "predicted_total": float}
            odds: Live odds from fetch_live_odds()
            kelly_frac: Fractional Kelly multiplier (0.25 = quarter Kelly)
            min_edge: Minimum edge to consider (0.02 = 2%)
            bankroll: Current bankroll for Kelly sizing

        Returns list of value bet dicts sorted by edge descending.
        """
        value_bets = []

        for game in odds:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            game_id = game.get("id", "")

            # Try to match prediction by various keys
            pred = (
                predictions.get(game_id)
                or predictions.get(f"{home} vs {away}")
                or predictions.get(f"{away} @ {home}")
                or self._fuzzy_match_prediction(predictions, home, away)
            )
            if not pred:
                continue

            model_home_prob = pred.get("home_win_prob", 0.5)
            model_away_prob = pred.get("away_win_prob", 1.0 - model_home_prob)
            model_spread = pred.get("predicted_spread")
            model_total = pred.get("predicted_total")

            # Compute per-game features for context
            features = self.compute_market_features(game)

            # ── Moneyline value ──
            best_home_ml = self.get_best_odds("h2h", home, game)
            best_away_ml = self.get_best_odds("h2h", away, game)

            if best_home_ml:
                market_prob = decimal_to_implied_prob(best_home_ml["price"])
                edge = model_home_prob - market_prob
                if edge > min_edge:
                    stake = kelly_frac_amount(model_home_prob, best_home_ml["price"], kelly_frac, bankroll)
                    value_bets.append({
                        "game": f"{away} @ {home}",
                        "game_id": game_id,
                        "market": "moneyline",
                        "selection": home,
                        "side": "home",
                        "decimal_odds": best_home_ml["price"],
                        "american_odds": decimal_to_american(best_home_ml["price"]),
                        "bookmaker": best_home_ml["bookmaker"],
                        "model_prob": round(model_home_prob, 4),
                        "market_prob": round(market_prob, 4),
                        "edge_pct": round(edge * 100, 2),
                        "expected_value": round(edge * (best_home_ml["price"] - 1) - (1 - model_home_prob), 4),
                        "kelly_fraction": round(kelly_fraction(model_home_prob, best_home_ml["price"], kelly_frac), 4),
                        "recommended_stake": round(stake, 2),
                        "consensus_prob": round(features.get("consensus_home_prob", 0.5), 4),
                        "sharp_square_div": round(features.get("sharp_square_div_home", 0), 4),
                    })

            if best_away_ml:
                market_prob = decimal_to_implied_prob(best_away_ml["price"])
                edge = model_away_prob - market_prob
                if edge > min_edge:
                    stake = kelly_frac_amount(model_away_prob, best_away_ml["price"], kelly_frac, bankroll)
                    value_bets.append({
                        "game": f"{away} @ {home}",
                        "game_id": game_id,
                        "market": "moneyline",
                        "selection": away,
                        "side": "away",
                        "decimal_odds": best_away_ml["price"],
                        "american_odds": decimal_to_american(best_away_ml["price"]),
                        "bookmaker": best_away_ml["bookmaker"],
                        "model_prob": round(model_away_prob, 4),
                        "market_prob": round(market_prob, 4),
                        "edge_pct": round(edge * 100, 2),
                        "expected_value": round(edge * (best_away_ml["price"] - 1) - (1 - model_away_prob), 4),
                        "kelly_fraction": round(kelly_fraction(model_away_prob, best_away_ml["price"], kelly_frac), 4),
                        "recommended_stake": round(stake, 2),
                        "consensus_prob": round(features.get("consensus_away_prob", 0.5), 4),
                        "sharp_square_div": round(features.get("sharp_square_div_away", 0), 4),
                    })

            # ── Spread value ──
            if model_spread is not None:
                spread_bets = self._find_spread_value(
                    game, home, away, model_spread, model_home_prob,
                    kelly_frac, min_edge, bankroll, features,
                )
                value_bets.extend(spread_bets)

            # ── Total value ──
            if model_total is not None:
                total_bets = self._find_total_value(
                    game, home, away, model_total,
                    kelly_frac, min_edge, bankroll, features,
                )
                value_bets.extend(total_bets)

        # Sort by edge descending
        value_bets.sort(key=lambda b: b["edge_pct"], reverse=True)
        return value_bets

    def get_best_odds(
        self,
        market: str,
        selection: str,
        game: Dict,
    ) -> Optional[Dict]:
        """
        Find the best available odds across all bookmakers for a selection.

        Args:
            market: "h2h", "spreads", or "totals"
            selection: Team name, "Over", or "Under"
            game: Single game dict from the API

        Returns {"bookmaker": str, "price": float, "point": float|None} or None.
        """
        best = None

        for bk in game.get("bookmakers", []):
            bk_key = bk.get("key", "")
            for m in bk.get("markets", []):
                if m.get("key") != market:
                    continue
                for outcome in m.get("outcomes", []):
                    if outcome.get("name") != selection:
                        continue
                    price = outcome.get("price", 0)
                    if best is None or price > best["price"]:
                        best = {
                            "bookmaker": bk_key,
                            "price": price,
                            "point": outcome.get("point"),
                        }

        return best

    # ──────────────────────────────────────────────────────────────────────────
    # LINE MOVEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_line_movement(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
    ) -> Dict[str, float]:
        """
        Compute line movement by comparing current odds to the earliest
        snapshot we have for this game.

        Returns features:
          - ml_home_movement: change in home ML odds (positive = drifted out)
          - ml_away_movement: change in away ML odds
          - spread_movement: change in home spread (negative = more favored)
          - total_movement: change in total points line
          - movement_direction: +1 if home became more favored, -1 if away, 0 if stable
          - hours_since_open: how many hours of data we have
        """
        result = {
            "ml_home_movement": 0.0,
            "ml_away_movement": 0.0,
            "spread_movement": 0.0,
            "total_movement": 0.0,
            "movement_direction": 0.0,
            "hours_since_open": 0.0,
        }

        if len(self._odds_snapshots) < 2:
            return result

        # Find this game in the earliest snapshot
        earliest_game = None
        earliest_ts = None
        for snap in self._odds_snapshots:
            games = snap.get("games", [])
            for g in games:
                if g.get("id") == game_id:
                    earliest_game = g
                    earliest_ts = snap.get("timestamp")
                    break
            if earliest_game:
                break

        if not earliest_game:
            return result

        # Compute time span
        if earliest_ts:
            try:
                t0 = datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
                t1 = datetime.now(timezone.utc)
                result["hours_since_open"] = (t1 - t0).total_seconds() / 3600.0
            except Exception:
                pass

        # Extract opening odds (average across books in earliest snapshot)
        open_home_ml = []
        open_away_ml = []
        open_spreads = []
        open_totals = []

        for bk in earliest_game.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    if m["key"] == "h2h":
                        if o.get("name") == home_team:
                            open_home_ml.append(o["price"])
                        elif o.get("name") == away_team:
                            open_away_ml.append(o["price"])
                    elif m["key"] == "spreads":
                        if o.get("name") == home_team and o.get("point") is not None:
                            open_spreads.append(o["point"])
                    elif m["key"] == "totals":
                        if o.get("name") == "Over" and o.get("point") is not None:
                            open_totals.append(o["point"])

        # Compare to current (we compute current averages from the game passed in)
        # Actually we already computed these in compute_market_features, so just
        # compute the opening averages and the diff
        curr_home_ml = []
        curr_away_ml = []
        curr_spreads = []
        curr_totals = []

        # We need to re-extract from the latest snapshot for this game
        latest_game = None
        for snap in reversed(self._odds_snapshots):
            for g in snap.get("games", []):
                if g.get("id") == game_id:
                    latest_game = g
                    break
            if latest_game:
                break

        if latest_game:
            for bk in latest_game.get("bookmakers", []):
                for m in bk.get("markets", []):
                    for o in m.get("outcomes", []):
                        if m["key"] == "h2h":
                            if o.get("name") == home_team:
                                curr_home_ml.append(o["price"])
                            elif o.get("name") == away_team:
                                curr_away_ml.append(o["price"])
                        elif m["key"] == "spreads":
                            if o.get("name") == home_team and o.get("point") is not None:
                                curr_spreads.append(o["point"])
                        elif m["key"] == "totals":
                            if o.get("name") == "Over" and o.get("point") is not None:
                                curr_totals.append(o["point"])

        if open_home_ml and curr_home_ml:
            result["ml_home_movement"] = _mean(curr_home_ml) - _mean(open_home_ml)
        if open_away_ml and curr_away_ml:
            result["ml_away_movement"] = _mean(curr_away_ml) - _mean(open_away_ml)
        if open_spreads and curr_spreads:
            result["spread_movement"] = _mean(curr_spreads) - _mean(open_spreads)
        if open_totals and curr_totals:
            result["total_movement"] = _mean(curr_totals) - _mean(open_totals)

        # Direction: if home ML decreased (shorter odds) = home became more favored
        if result["ml_home_movement"] < -0.03:
            result["movement_direction"] = 1.0   # home more favored
        elif result["ml_home_movement"] > 0.03:
            result["movement_direction"] = -1.0  # away more favored
        else:
            result["movement_direction"] = 0.0

        return result

    def _detect_steam_moves(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
    ) -> Dict[str, float]:
        """
        Detect steam moves: rapid, correlated line movement across 3+ books
        within a short window (< 30 minutes between snapshots).

        A steam move signals large sharp money entering the market.

        Returns:
          - steam_move_detected: 1.0 if steam move found, 0.0 otherwise
          - steam_move_direction: +1 (home), -1 (away), 0 (none)
          - steam_move_magnitude: size of the coordinated move
        """
        result = {
            "steam_move_detected": 0.0,
            "steam_move_direction": 0.0,
            "steam_move_magnitude": 0.0,
        }

        if len(self._odds_snapshots) < 2:
            return result

        # Compare last two snapshots
        prev_snap = self._odds_snapshots[-2]
        curr_snap = self._odds_snapshots[-1]

        # Check time gap: steam moves matter in < 30min windows
        try:
            t_prev = datetime.fromisoformat(prev_snap.get("timestamp", "").replace("Z", "+00:00"))
            t_curr = datetime.fromisoformat(curr_snap.get("timestamp", "").replace("Z", "+00:00"))
            gap_minutes = (t_curr - t_prev).total_seconds() / 60.0
        except Exception:
            gap_minutes = 60  # assume 1 hour if unknown

        # Only detect steam in sub-60-minute windows
        if gap_minutes > 60:
            return result

        prev_game = None
        curr_game = None
        for g in prev_snap.get("games", []):
            if g.get("id") == game_id:
                prev_game = g
                break
        for g in curr_snap.get("games", []):
            if g.get("id") == game_id:
                curr_game = g
                break

        if not prev_game or not curr_game:
            return result

        # Track per-book moneyline changes
        prev_odds = _extract_h2h_by_book(prev_game, home_team, away_team)
        curr_odds = _extract_h2h_by_book(curr_game, home_team, away_team)

        home_movers = 0
        away_movers = 0
        total_magnitude = 0.0

        for bk_key in set(prev_odds.keys()) & set(curr_odds.keys()):
            prev_h = prev_odds[bk_key].get("home", 0)
            curr_h = curr_odds[bk_key].get("home", 0)

            if prev_h > 0 and curr_h > 0:
                diff = curr_h - prev_h
                if diff < -0.05:  # home shortened by 5+ cents
                    home_movers += 1
                    total_magnitude += abs(diff)
                elif diff > 0.05:  # home drifted by 5+ cents
                    away_movers += 1
                    total_magnitude += abs(diff)

        # Steam = 3+ books moving same direction rapidly
        if home_movers >= 3:
            result["steam_move_detected"] = 1.0
            result["steam_move_direction"] = 1.0
            result["steam_move_magnitude"] = total_magnitude / home_movers
        elif away_movers >= 3:
            result["steam_move_detected"] = 1.0
            result["steam_move_direction"] = -1.0
            result["steam_move_magnitude"] = total_magnitude / away_movers

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # SPREAD & TOTAL VALUE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _find_spread_value(
        self,
        game: Dict,
        home: str,
        away: str,
        model_spread: float,
        model_home_prob: float,
        kelly_frac: float,
        min_edge: float,
        bankroll: float,
        features: Dict,
    ) -> List[Dict]:
        """Find value in spread markets by comparing model spread to market spread."""
        bets = []
        game_id = game.get("id", "")
        consensus_spread = features.get("spread_consensus", 0)

        if consensus_spread == 0:
            return bets

        # Each point of spread difference ~ 3% probability shift (empirical NBA estimate)
        PROB_PER_POINT = 0.03

        for bk in game.get("bookmakers", []):
            bk_key = bk.get("key", "")
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                for outcome in m.get("outcomes", []):
                    team_name = outcome.get("name", "")
                    point = outcome.get("point")
                    price = outcome.get("price", 0)

                    if point is None or price <= 1.0:
                        continue

                    # Model spread is from home perspective (negative = home favored)
                    # Market point is the handicap for this team
                    if team_name == home:
                        # Value exists if our spread is more favorable for home than market
                        # e.g., model says -5.0, market says -3.0 => home should be bigger fav
                        spread_edge_points = model_spread - point  # both negative for home fav
                        adj_prob = 0.5 + spread_edge_points * PROB_PER_POINT
                        adj_prob = max(0.05, min(0.95, adj_prob))
                        selection = home
                        side = "home"
                    elif team_name == away:
                        spread_edge_points = (-model_spread) - point
                        adj_prob = 0.5 + spread_edge_points * PROB_PER_POINT
                        adj_prob = max(0.05, min(0.95, adj_prob))
                        selection = away
                        side = "away"
                    else:
                        continue

                    market_prob = decimal_to_implied_prob(price)
                    edge = adj_prob - market_prob

                    if edge > min_edge:
                        stake = kelly_frac_amount(adj_prob, price, kelly_frac, bankroll)
                        bets.append({
                            "game": f"{away} @ {home}",
                            "game_id": game_id,
                            "market": "spread",
                            "selection": f"{selection} {point:+g}",
                            "side": side,
                            "decimal_odds": price,
                            "american_odds": decimal_to_american(price),
                            "bookmaker": bk_key,
                            "model_prob": round(adj_prob, 4),
                            "market_prob": round(market_prob, 4),
                            "edge_pct": round(edge * 100, 2),
                            "expected_value": round(edge * (price - 1) - (1 - adj_prob), 4),
                            "kelly_fraction": round(kelly_fraction(adj_prob, price, kelly_frac), 4),
                            "recommended_stake": round(stake, 2),
                            "model_spread": model_spread,
                            "market_spread": point,
                            "spread_edge_points": round(spread_edge_points, 1),
                        })

        # Deduplicate: keep only best odds per side
        best_by_side = {}
        for b in bets:
            key = b["side"]
            if key not in best_by_side or b["edge_pct"] > best_by_side[key]["edge_pct"]:
                best_by_side[key] = b
        return list(best_by_side.values())

    def _find_total_value(
        self,
        game: Dict,
        home: str,
        away: str,
        model_total: float,
        kelly_frac: float,
        min_edge: float,
        bankroll: float,
        features: Dict,
    ) -> List[Dict]:
        """Find value in totals markets by comparing model total to market total."""
        bets = []
        game_id = game.get("id", "")
        consensus_total = features.get("total_consensus", 0)

        if consensus_total == 0:
            return bets

        PROB_PER_POINT = 0.03  # each point difference ~ 3% probability shift

        for bk in game.get("bookmakers", []):
            bk_key = bk.get("key", "")
            for m in bk.get("markets", []):
                if m.get("key") != "totals":
                    continue
                for outcome in m.get("outcomes", []):
                    ou_name = outcome.get("name", "")  # "Over" or "Under"
                    point = outcome.get("point")
                    price = outcome.get("price", 0)

                    if point is None or price <= 1.0:
                        continue

                    if ou_name == "Over":
                        # Value on over if model total > market total
                        edge_points = model_total - point
                        adj_prob = 0.5 + edge_points * PROB_PER_POINT
                    elif ou_name == "Under":
                        # Value on under if model total < market total
                        edge_points = point - model_total
                        adj_prob = 0.5 + edge_points * PROB_PER_POINT
                    else:
                        continue

                    adj_prob = max(0.05, min(0.95, adj_prob))
                    market_prob = decimal_to_implied_prob(price)
                    edge = adj_prob - market_prob

                    if edge > min_edge:
                        stake = kelly_frac_amount(adj_prob, price, kelly_frac, bankroll)
                        bets.append({
                            "game": f"{away} @ {home}",
                            "game_id": game_id,
                            "market": "total",
                            "selection": f"{ou_name} {point}",
                            "side": ou_name.lower(),
                            "decimal_odds": price,
                            "american_odds": decimal_to_american(price),
                            "bookmaker": bk_key,
                            "model_prob": round(adj_prob, 4),
                            "market_prob": round(market_prob, 4),
                            "edge_pct": round(edge * 100, 2),
                            "expected_value": round(edge * (price - 1) - (1 - adj_prob), 4),
                            "kelly_fraction": round(kelly_fraction(adj_prob, price, kelly_frac), 4),
                            "recommended_stake": round(stake, 2),
                            "model_total": model_total,
                            "market_total": point,
                            "total_edge_points": round(model_total - point, 1),
                        })

        # Keep best per side (over/under)
        best_by_side = {}
        for b in bets:
            key = b["side"]
            if key not in best_by_side or b["edge_pct"] > best_by_side[key]["edge_pct"]:
                best_by_side[key] = b
        return list(best_by_side.values())

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────────────────

    def _fuzzy_match_prediction(
        self,
        predictions: Dict,
        home: str,
        away: str,
    ) -> Optional[Dict]:
        """Fuzzy match a prediction key to home/away team names."""
        home_lower = home.lower()
        away_lower = away.lower()

        for key, pred in predictions.items():
            key_lower = key.lower()
            # Check if both team names appear in the key
            home_words = home_lower.split()
            away_words = away_lower.split()
            # Match on last word (team nickname): "Hawks", "Magic", etc.
            if (home_words and away_words and
                home_words[-1] in key_lower and away_words[-1] in key_lower):
                return pred
        return None

    def _empty_features(self) -> Dict[str, float]:
        """Return zeroed feature dict when no data available."""
        keys = [
            "n_bookmakers", "n_books_h2h", "n_books_spreads", "n_books_totals",
            "ml_home_best", "ml_home_worst", "ml_home_avg", "ml_home_std", "ml_home_implied_prob",
            "ml_away_best", "ml_away_worst", "ml_away_avg", "ml_away_std", "ml_away_implied_prob",
            "consensus_home_prob", "consensus_away_prob",
            "avg_overround_h2h", "min_overround_h2h", "max_overround_h2h",
            "sharp_square_div_home", "sharp_square_div_away",
            "sharp_avg_overround", "square_avg_overround",
            "spread_consensus", "spread_std", "spread_best_home", "spread_worst_home", "spread_range",
            "sharp_square_spread_div",
            "total_consensus", "total_std", "total_range",
            "sharp_square_total_div",
            "ml_home_movement", "ml_away_movement", "spread_movement", "total_movement",
            "movement_direction", "hours_since_open",
            "steam_move_detected", "steam_move_direction", "steam_move_magnitude",
        ]
        return {k: 0.0 for k in keys}

    def _save_snapshot(self, data: Any, prefix: str):
        """Save data to date-stamped directory."""
        now = datetime.now(timezone.utc)
        day_dir = self.data_dir / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        ts = now.strftime("%H%M%S")
        out_file = day_dir / f"{prefix}-{ts}.json"

        payload = {
            "timestamp": now.isoformat(),
            "data": data if isinstance(data, (list, dict)) else str(data),
        }

        try:
            out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            logger.info(f"Saved: {out_file}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

        return out_file


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION: find_all_value_bets
# ══════════════════════════════════════════════════════════════════════════════

def find_all_value_bets(
    model_predictions: Dict[str, Dict],
    live_odds: List[Dict] = None,
    include_props: bool = False,
    bankroll: float = 100.0,
    kelly_frac: float = 0.25,
    min_edge: float = 0.02,
) -> List[Dict]:
    """
    Master function: compare model predictions against live market odds
    to find all value bets across moneyline, spread, total, and optionally
    player props.

    Args:
        model_predictions: Dict keyed by game identifier.
            Each value: {
                "home_win_prob": float,          # Required
                "away_win_prob": float,           # Optional (1 - home_win_prob)
                "predicted_spread": float,        # Optional (negative = home favored)
                "predicted_total": float,         # Optional
                "player_props": {                 # Optional, for prop bets
                    "Player Name": {
                        "points": float,
                        "assists": float,
                        "rebounds": float,
                        ...
                    }
                }
            }
        live_odds: Pre-fetched odds (or will fetch fresh)
        include_props: Also scan player prop markets (costs extra API calls)
        bankroll: Current bankroll for Kelly sizing
        kelly_frac: Fractional Kelly multiplier
        min_edge: Minimum edge threshold

    Returns:
        List of value bet dicts sorted by edge, each containing:
        - game, game_id, market, selection, side
        - decimal_odds, american_odds, bookmaker
        - model_prob, market_prob, edge_pct
        - expected_value, kelly_fraction, recommended_stake
    """
    engine = OddsMarketEngine()

    # Fetch live odds if not provided
    if live_odds is None:
        live_odds = engine.fetch_live_odds()

    if not live_odds:
        logger.warning("No live odds available")
        return []

    # Find game-level value bets (moneyline, spread, total)
    value_bets = engine.find_value_bets(
        model_predictions, live_odds, kelly_frac, min_edge, bankroll,
    )

    # Optionally scan player props
    if include_props:
        for game in live_odds:
            game_id = game.get("id", "")
            home = game.get("home_team", "")
            away = game.get("away_team", "")

            pred = (
                model_predictions.get(game_id)
                or model_predictions.get(f"{home} vs {away}")
                or model_predictions.get(f"{away} @ {home}")
            )
            if not pred or "player_props" not in pred:
                continue

            # Fetch props for this game (costs 1 API call)
            prop_bookmakers = engine.fetch_player_props(game_id)
            if not prop_bookmakers:
                continue

            player_props = pred["player_props"]
            for player_name, player_preds in player_props.items():
                prop_bets = _find_player_prop_value(
                    engine, game, game_id, player_name, player_preds,
                    prop_bookmakers, kelly_frac, min_edge, bankroll,
                )
                value_bets.extend(prop_bets)

    # Final sort by edge
    value_bets.sort(key=lambda b: b["edge_pct"], reverse=True)

    # Log summary
    if value_bets:
        total_edge = sum(b["edge_pct"] for b in value_bets)
        total_stake = sum(b["recommended_stake"] for b in value_bets)
        logger.info(
            f"Found {len(value_bets)} value bets | "
            f"Avg edge: {total_edge/len(value_bets):.1f}% | "
            f"Total recommended stake: ${total_stake:.2f}"
        )
    else:
        logger.info("No value bets found above minimum edge threshold")

    return value_bets


def _find_player_prop_value(
    engine: OddsMarketEngine,
    game: Dict,
    game_id: str,
    player_name: str,
    player_preds: Dict[str, float],
    prop_bookmakers: List[Dict],
    kelly_frac: float,
    min_edge: float,
    bankroll: float,
) -> List[Dict]:
    """Find value in player prop markets for a single player."""
    bets = []
    home = game.get("home_team", "")
    away = game.get("away_team", "")

    PROB_PER_POINT = {
        "points": 0.025,     # each point ~ 2.5% probability shift
        "assists": 0.05,     # each assist ~ 5%
        "rebounds": 0.04,    # each rebound ~ 4%
        "threes": 0.06,      # each three ~ 6%
        "blocks": 0.08,      # each block ~ 8%
        "steals": 0.08,      # each steal ~ 8%
    }

    for prop_market in PLAYER_PROP_MARKETS:
        short_name = prop_market.replace("player_", "")
        model_value = player_preds.get(short_name)
        if model_value is None:
            continue

        prob_per_pt = PROB_PER_POINT.get(short_name, 0.03)

        for bk in prop_bookmakers:
            bk_key = bk.get("key", "")
            for market in bk.get("markets", []):
                if market.get("key") != prop_market:
                    continue
                for outcome in market.get("outcomes", []):
                    desc = outcome.get("description", "")
                    if player_name.lower() not in desc.lower():
                        continue

                    point = outcome.get("point")
                    price = outcome.get("price", 0)
                    ou_name = outcome.get("name", "")

                    if point is None or price <= 1.0:
                        continue

                    if ou_name == "Over":
                        edge_points = model_value - point
                        adj_prob = 0.5 + edge_points * prob_per_pt
                    elif ou_name == "Under":
                        edge_points = point - model_value
                        adj_prob = 0.5 + edge_points * prob_per_pt
                    else:
                        continue

                    adj_prob = max(0.05, min(0.95, adj_prob))
                    market_prob = decimal_to_implied_prob(price)
                    edge = adj_prob - market_prob

                    if edge > min_edge:
                        stake = kelly_frac_amount(adj_prob, price, kelly_frac, bankroll)
                        bets.append({
                            "game": f"{away} @ {home}",
                            "game_id": game_id,
                            "market": f"player_prop_{short_name}",
                            "selection": f"{player_name} {ou_name} {point} {short_name}",
                            "side": ou_name.lower(),
                            "player": player_name,
                            "prop_type": short_name,
                            "decimal_odds": price,
                            "american_odds": decimal_to_american(price),
                            "bookmaker": bk_key,
                            "model_prob": round(adj_prob, 4),
                            "market_prob": round(market_prob, 4),
                            "edge_pct": round(edge * 100, 2),
                            "expected_value": round(edge * (price - 1) - (1 - adj_prob), 4),
                            "kelly_fraction": round(kelly_fraction(adj_prob, price, kelly_frac), 4),
                            "recommended_stake": round(stake, 2),
                            "model_projection": model_value,
                            "market_line": point,
                            "line_edge": round(model_value - point, 1),
                        })

    # Keep best per (prop_type, side) combo
    best_by_key = {}
    for b in bets:
        key = (b["prop_type"], b["side"])
        if key not in best_by_key or b["edge_pct"] > best_by_key[key]["edge_pct"]:
            best_by_key[key] = b
    return list(best_by_key.values())


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kelly_frac_amount(prob: float, decimal_odds: float, frac: float, bankroll: float) -> float:
    """Calculate Kelly stake in currency units, capped at 5% of bankroll."""
    kf = kelly_fraction(prob, decimal_odds, frac)
    max_bet = bankroll * 0.05
    return min(kf * bankroll, max_bet)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _count_bookmakers(games: List[Dict]) -> int:
    """Count unique bookmakers across all games."""
    books = set()
    for g in games:
        for bk in g.get("bookmakers", []):
            books.add(bk.get("key", ""))
    return len(books)


def _extract_h2h_by_book(game: Dict, home: str, away: str) -> Dict[str, Dict]:
    """Extract h2h odds indexed by bookmaker key."""
    result = {}
    for bk in game.get("bookmakers", []):
        bk_key = bk.get("key", "")
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            entry = {}
            for o in m.get("outcomes", []):
                if o.get("name") == home:
                    entry["home"] = o["price"]
                elif o.get("name") == away:
                    entry["away"] = o["price"]
            if entry:
                result[bk_key] = entry
    return result


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def format_value_bets_report(value_bets: List[Dict], bankroll: float = 100.0) -> str:
    """Format value bets into a readable report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "",
        f"{'='*90}",
        f"  VALUE BETS REPORT — {now}",
        f"  Bankroll: ${bankroll:.2f}",
        f"{'='*90}",
        "",
    ]

    if not value_bets:
        lines.append("  No value bets found above minimum edge threshold.")
        lines.append(f"{'='*90}")
        return "\n".join(lines)

    total_stake = 0.0
    for i, bet in enumerate(value_bets, 1):
        market_label = bet["market"].upper()
        edge = bet["edge_pct"]
        stake = bet["recommended_stake"]
        total_stake += stake

        lines.append(
            f"  {i:2d}. [{market_label:>15s}] {bet['selection']:<35s}"
        )
        lines.append(
            f"      Game: {bet['game']}"
        )
        lines.append(
            f"      Odds: {bet['decimal_odds']:.2f} ({bet['american_odds']:+d}) @ {bet['bookmaker']}"
        )
        lines.append(
            f"      Model: {bet['model_prob']*100:.1f}%  |  Market: {bet['market_prob']*100:.1f}%  |  "
            f"Edge: {edge:+.1f}%  |  EV: {bet['expected_value']:+.4f}"
        )
        lines.append(
            f"      Kelly: {bet['kelly_fraction']*100:.2f}%  |  Stake: ${stake:.2f}"
        )
        lines.append("")

    lines.append(f"  {'─'*86}")
    lines.append(f"  Total bets: {len(value_bets)}  |  Total stake: ${total_stake:.2f}  |  Exposure: {total_stake/bankroll*100:.1f}%")
    avg_edge = sum(b["edge_pct"] for b in value_bets) / len(value_bets)
    lines.append(f"  Average edge: {avg_edge:+.1f}%")
    lines.append(f"{'='*90}")

    return "\n".join(lines)


def format_market_features_report(features: Dict[str, float], game_label: str = "") -> str:
    """Format market microstructure features into a readable report."""
    lines = [
        "",
        f"{'='*70}",
        f"  MARKET MICROSTRUCTURE — {game_label}",
        f"{'='*70}",
        "",
        f"  Bookmakers: {features.get('n_bookmakers', 0):.0f} total "
        f"({features.get('n_books_h2h', 0):.0f} h2h, "
        f"{features.get('n_books_spreads', 0):.0f} spreads, "
        f"{features.get('n_books_totals', 0):.0f} totals)",
        "",
        "  MONEYLINE:",
        f"    Home: best {features.get('ml_home_best', 0):.2f} / avg {features.get('ml_home_avg', 0):.2f} / worst {features.get('ml_home_worst', 0):.2f}  (std: {features.get('ml_home_std', 0):.3f})",
        f"    Away: best {features.get('ml_away_best', 0):.2f} / avg {features.get('ml_away_avg', 0):.2f} / worst {features.get('ml_away_worst', 0):.2f}  (std: {features.get('ml_away_std', 0):.3f})",
        f"    Consensus (vig-removed): Home {features.get('consensus_home_prob', 0)*100:.1f}% / Away {features.get('consensus_away_prob', 0)*100:.1f}%",
        "",
        "  SPREAD:",
        f"    Consensus: {features.get('spread_consensus', 0):+.1f}  (std: {features.get('spread_std', 0):.2f}, range: {features.get('spread_range', 0):.1f})",
        "",
        "  TOTAL:",
        f"    Consensus: {features.get('total_consensus', 0):.1f}  (std: {features.get('total_std', 0):.2f}, range: {features.get('total_range', 0):.1f})",
        "",
        "  SHARP vs SQUARE:",
        f"    ML divergence (home): {features.get('sharp_square_div_home', 0):+.3f}  (+ = sharp gives more value on home)",
        f"    ML divergence (away): {features.get('sharp_square_div_away', 0):+.3f}",
        f"    Spread divergence:    {features.get('sharp_square_spread_div', 0):+.2f} pts",
        f"    Total divergence:     {features.get('sharp_square_total_div', 0):+.2f} pts",
        f"    Sharp overround:      {features.get('sharp_avg_overround', 0):.3f}  |  Square: {features.get('square_avg_overround', 0):.3f}",
        "",
        "  JUICE / EFFICIENCY:",
        f"    Avg overround: {features.get('avg_overround_h2h', 0):.3f}  (min: {features.get('min_overround_h2h', 0):.3f}, max: {features.get('max_overround_h2h', 0):.3f})",
        "",
        "  LINE MOVEMENT:",
        f"    Home ML: {features.get('ml_home_movement', 0):+.3f}  |  Away ML: {features.get('ml_away_movement', 0):+.3f}",
        f"    Spread: {features.get('spread_movement', 0):+.2f}  |  Total: {features.get('total_movement', 0):+.2f}",
        f"    Direction: {'HOME' if features.get('movement_direction', 0) > 0 else 'AWAY' if features.get('movement_direction', 0) < 0 else 'STABLE'}",
        f"    Hours tracked: {features.get('hours_since_open', 0):.1f}",
        "",
        "  STEAM MOVES:",
        f"    Detected: {'YES' if features.get('steam_move_detected', 0) > 0 else 'NO'}",
    ]

    if features.get("steam_move_detected", 0) > 0:
        direction = "HOME" if features.get("steam_move_direction", 0) > 0 else "AWAY"
        lines.append(f"    Direction: {direction}  |  Magnitude: {features.get('steam_move_magnitude', 0):.3f}")

    lines.append(f"{'='*70}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NBA Odds Market Engine — Real-time market microstructure analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python features/odds_market.py --live                    # Fetch + display live odds
  python features/odds_market.py --features                # Compute market features for all games
  python features/odds_market.py --value                   # Find value bets (requires predictions)
  python features/odds_market.py --live --save             # Fetch and save snapshot
  python features/odds_market.py --historical 2026-03-15   # Fetch historical odds
        """,
    )

    parser.add_argument("--live", action="store_true", help="Fetch and display live odds")
    parser.add_argument("--features", action="store_true", help="Compute market features for all games")
    parser.add_argument("--value", action="store_true", help="Find value bets (uses dummy predictions for demo)")
    parser.add_argument("--historical", type=str, metavar="DATE", help="Fetch historical odds (YYYY-MM-DD)")
    parser.add_argument("--props", type=str, metavar="GAME_ID", help="Fetch player props for a game")
    parser.add_argument("--save", action="store_true", default=True, help="Save snapshots (default: True)")
    parser.add_argument("--no-save", action="store_true", help="Do not save snapshots")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Current bankroll (default: 100)")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold (default: 0.02)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    save = not args.no_save

    engine = OddsMarketEngine()

    # ── Live odds ──
    if args.live or args.features or args.value:
        print(f"\nFetching live NBA odds...")
        games = engine.fetch_live_odds(save=save)

        if not games:
            print("No games found. NBA may be off today or API issue.")
            return

        if args.live:
            print(f"\n{'='*80}")
            print(f"  LIVE NBA ODDS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
            print(f"  {len(games)} games from {_count_bookmakers(games)} bookmakers")
            print(f"{'='*80}\n")

            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                time_str = game.get("commence_time", "")[:16].replace("T", " ")
                n_books = len(game.get("bookmakers", []))

                best_home = engine.get_best_odds("h2h", home, game)
                best_away = engine.get_best_odds("h2h", away, game)
                best_spread = engine.get_best_odds("spreads", home, game)
                best_total_over = engine.get_best_odds("totals", "Over", game)

                print(f"  {away} @ {home}  ({time_str}, {n_books} books)")
                print(f"  {'─'*70}")

                if best_home and best_away:
                    h_imp = decimal_to_implied_prob(best_home["price"]) * 100
                    a_imp = decimal_to_implied_prob(best_away["price"]) * 100
                    print(f"    ML:  {home:<25s} {best_home['price']:.2f} [{h_imp:.0f}%] @ {best_home['bookmaker']}")
                    print(f"         {away:<25s} {best_away['price']:.2f} [{a_imp:.0f}%] @ {best_away['bookmaker']}")

                if best_spread:
                    print(f"    SPR: {home} {best_spread['point']:+g} @ {best_spread['price']:.2f} ({best_spread['bookmaker']})")

                if best_total_over:
                    print(f"    TOT: O/U {best_total_over['point']} @ {best_total_over['price']:.2f} ({best_total_over['bookmaker']})")

                print()

        if args.features:
            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                features = engine.compute_market_features(game)

                if args.json:
                    print(json.dumps({"game": f"{away} @ {home}", "features": features}, indent=2))
                else:
                    print(format_market_features_report(features, f"{away} @ {home}"))

        if args.value:
            # Demo: generate simple predictions from consensus for demonstration
            # In production, these come from the ML model
            print("\n  [VALUE] Using consensus-based demo predictions (replace with real model)...\n")
            predictions = {}
            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                features = engine.compute_market_features(game)

                # Create slight model edge over consensus for demo
                # In production, this comes from features/engine.py model
                import random
                noise = random.uniform(-0.05, 0.05)
                predictions[f"{away} @ {home}"] = {
                    "home_win_prob": min(0.95, max(0.05, features.get("consensus_home_prob", 0.5) + noise)),
                    "away_win_prob": min(0.95, max(0.05, features.get("consensus_away_prob", 0.5) - noise)),
                    "predicted_spread": features.get("spread_consensus", 0) + random.uniform(-1, 1),
                    "predicted_total": features.get("total_consensus", 220) + random.uniform(-2, 2),
                }

            value_bets = engine.find_value_bets(
                predictions, games,
                kelly_frac=0.25,
                min_edge=args.min_edge,
                bankroll=args.bankroll,
            )

            if args.json:
                print(json.dumps(value_bets, indent=2))
            else:
                print(format_value_bets_report(value_bets, args.bankroll))

    # ── Historical ──
    if args.historical:
        date_str = args.historical
        if len(date_str) == 10:
            date_str += "T12:00:00Z"  # default to noon
        print(f"\nFetching historical odds for {date_str}...")
        games = engine.fetch_historical_odds(date_str, save=save)
        if games:
            print(f"  Retrieved {len(games)} games")
            if args.json:
                print(json.dumps(games[:3], indent=2))
        else:
            print("  No historical data (may require paid API plan)")

    # ── Player props ──
    if args.props:
        print(f"\nFetching player props for game {args.props}...")
        bookmakers = engine.fetch_player_props(args.props, save=save)
        if bookmakers:
            prop_count = sum(
                len(m.get("outcomes", []))
                for bk in bookmakers
                for m in bk.get("markets", [])
            )
            print(f"  {prop_count} prop lines from {len(bookmakers)} books")
            if args.json:
                print(json.dumps(bookmakers[:2], indent=2, default=str))
        else:
            print("  No props available (game may not be upcoming or props not offered)")


if __name__ == "__main__":
    main()
