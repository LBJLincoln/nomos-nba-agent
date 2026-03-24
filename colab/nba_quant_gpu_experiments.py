# %% [markdown]
# # NBA Quant AI — GPU Experiment Runner (Google Colab)
#
# **Purpose**: Run GPU-accelerated model experiments for the NBA Quant AI pipeline.
# Polls the Supabase `nba_experiments` table for pending experiments, trains
# GPU-native models (TabNet, FT-Transformer, NODE, LSTM, MLP, XGBoost-GPU,
# LightGBM-GPU, CatBoost-GPU), evaluates via walk-forward backtesting,
# and writes results back to Supabase.
#
# **Architecture**: This notebook acts as a GPU compute node alongside
# S10 (genetic evolution) and S11 (CPU experiment runner). It picks up
# experiments with `target_space = 'colab'` or `target_space = 'any'`.
#
# **Requirements**: Google Colab with GPU runtime (T4/A100/L4).

# %% [markdown]
# ## 1. Install Dependencies

# %%
# ── pip installs (run once per Colab session) ──
import subprocess, sys

_packages = [
    "psycopg2-binary",
    "xgboost",
    "lightgbm",
    "catboost",
    "pytorch-tabnet",
    "scikit-learn>=1.3",
    "pandas",
    "numpy",
]

for pkg in _packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# PyTorch is pre-installed on Colab, but ensure it's there
try:
    import torch
    print(f"PyTorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch

print("All packages installed.")

# %% [markdown]
# ## 2. Configuration & Supabase Connection

# %%
import os
import json
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

FEATURE_ENGINE_VERSION = "colab-inline-163feat"

# ── DATABASE_URL from Colab secrets or environment ──
DATABASE_URL = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    try:
        from google.colab import userdata
        DATABASE_URL = userdata.get("DATABASE_URL")
        print("DATABASE_URL loaded from Colab secrets.")
    except Exception:
        pass

if not DATABASE_URL:
    print("WARNING: DATABASE_URL not set. Set it via:")
    print("  1. Colab Secrets (key icon in left sidebar) -> add DATABASE_URL")
    print("  2. Or: os.environ['DATABASE_URL'] = 'postgresql://...'")

# ── Constants ──
POLL_INTERVAL = 60           # Poll every 60s
MAX_EVAL_GAMES = 10000       # OOM protection cap
WALK_FORWARD_SPLITS = 3      # Default walk-forward splits
EXPERIMENT_TIMEOUT = 1800    # 30 min max per experiment (matches HF Space S11)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 15                # Early stopping patience

print(f"Device: {DEVICE}")
print(f"Max games: {MAX_EVAL_GAMES}")

# %% [markdown]
# ## 3. Supabase PostgreSQL Connection

# %%
import psycopg2
from psycopg2 import pool as pg_pool

_pg_pool = None


def _get_pg():
    """Lazy PostgreSQL connection pool for Supabase."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    if not DATABASE_URL:
        print("[ERROR] DATABASE_URL not set")
        return None
    try:
        _pg_pool = pg_pool.SimpleConnectionPool(
            1, 3, DATABASE_URL, options="-c search_path=public"
        )
        conn = _pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        _pg_pool.putconn(conn)
        print("[OK] PostgreSQL connected to Supabase")
        return _pg_pool
    except Exception as e:
        print(f"[ERROR] PostgreSQL connection failed: {e}")
        _pg_pool = None
        return None


def _exec_sql(sql, params=None, fetch=True):
    """Execute SQL on Supabase."""
    pool = _get_pg()
    if not pool:
        return None
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            if fetch:
                try:
                    return cur.fetchall()
                except Exception:
                    return True
            return True
    except Exception as e:
        print(f"[SQL ERROR] {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return None
    finally:
        if conn and pool:
            try:
                pool.putconn(conn)
            except Exception:
                pass


def _reconnect_pg():
    """Force reconnection on next call."""
    global _pg_pool
    if _pg_pool:
        try:
            _pg_pool.closeall()
        except Exception:
            pass
    _pg_pool = None


# Test connection
_get_pg()

# %% [markdown]
# ## 4. Data Loading
#
# Load game data from Supabase or a CSV fallback URL.
# The feature engine builds 2000+ features from raw game records.

# %%
def load_games_from_supabase(limit=15000) -> List[dict]:
    """Load games from Supabase nba_games or nba_historical_games table."""
    # Try nba_games first, then historical
    for table in ["nba_games", "nba_historical_games", "games"]:
        rows = _exec_sql(f"""
            SELECT data FROM public.{table}
            ORDER BY id ASC
            LIMIT %s
        """, (limit,))
        if rows and rows is not True and len(rows) > 0:
            games = []
            for row in rows:
                if isinstance(row[0], dict):
                    games.append(row[0])
                elif isinstance(row[0], str):
                    games.append(json.loads(row[0]))
            print(f"Loaded {len(games)} games from {table}")
            return games

    # Fallback: try loading from a JSON column structure
    rows = _exec_sql("""
        SELECT game_date, home_team, away_team, home_pts, away_pts,
               home_stats, away_stats
        FROM public.nba_game_results
        ORDER BY game_date ASC
        LIMIT %s
    """, (limit,))
    if rows and rows is not True and len(rows) > 0:
        games = []
        for row in rows:
            games.append({
                "game_date": str(row[0]),
                "home_team": row[1],
                "away_team": row[2],
                "home": {"team_name": row[1], "pts": row[3], **(row[5] or {})},
                "away": {"team_name": row[2], "pts": row[4], **(row[6] or {})},
            })
        print(f"Loaded {len(games)} games from nba_game_results")
        return games

    print("[WARN] No games found in Supabase")
    return []


def load_games_from_csv(url: str = None) -> List[dict]:
    """Load games from a CSV URL (fallback)."""
    if url is None:
        url = os.environ.get(
            "GAMES_CSV_URL",
            "https://raw.githubusercontent.com/nomos42/nomos-nba-agent/main/data/historical/games-all.csv"
        )
    try:
        df = pd.read_csv(url)
        games = []
        for _, row in df.iterrows():
            games.append({
                "game_date": str(row.get("game_date", row.get("date", ""))),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "home": {
                    "team_name": row.get("home_team", ""),
                    "pts": int(row.get("home_pts", row.get("home_score", 0))),
                },
                "away": {
                    "team_name": row.get("away_team", ""),
                    "pts": int(row.get("away_pts", row.get("away_score", 0))),
                },
            })
        print(f"Loaded {len(games)} games from CSV")
        return games
    except Exception as e:
        print(f"[ERROR] CSV load failed: {e}")
        return []


def load_games_from_json_files(directory: str = "/content/data/historical") -> List[dict]:
    """Load games from local JSON files (if uploaded to Colab)."""
    from pathlib import Path
    games = []
    hist_dir = Path(directory)
    if not hist_dir.exists():
        return games
    for f in sorted(hist_dir.glob("games-*.json")):
        data = json.loads(f.read_text())
        games.extend(data if isinstance(data, list) else data.get("games", []))
    games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
    print(f"Loaded {len(games)} games from {directory}")
    return games

# %% [markdown]
# ## 5. Inline Feature Engine (Lightweight for Colab)
#
# Builds ~200 core features without needing the full 2000+ engine.
# Sufficient for GPU model experiments.

# %%
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

WINDOWS = [3, 5, 7, 10, 15, 20]


def _safe(val, default=0.0):
    """Safely convert to float."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _rolling_stats(results, window):
    """Compute rolling stats from recent results."""
    recent = results[-window:] if len(results) >= window else results
    if not recent:
        return [0.0] * 8
    wins = sum(1 for r in recent if r[1])
    pts = [r[2] for r in recent]  # margin
    scores = [r[3] for r in recent]  # points scored
    return [
        wins / len(recent),                           # win_pct
        float(np.mean(scores)) if scores else 0.0,    # avg_pts
        float(np.mean(pts)) if pts else 0.0,          # avg_margin
        float(np.std(pts)) if len(pts) > 1 else 0.0,  # margin_volatility
        float(np.max(pts)) if pts else 0.0,            # best_margin
        float(np.min(pts)) if pts else 0.0,            # worst_margin
        float(np.median(pts)) if pts else 0.0,         # median_margin
        sum(1 for r in recent if r[1]) / max(len(recent), 1),  # recent_form
    ]


def build_features_inline(games: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build ~200 features inline (no external engine dependency).

    Returns: X (n_games, n_features), y (n_games,), feature_names
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    team_results = defaultdict(list)  # team -> [(date, win, margin, pts, opp)]
    team_last = {}                     # team -> last game date
    team_elo = defaultdict(lambda: 1500.0)

    X, y_out = [], []
    feature_names = []
    first = True

    for game in games:
        home_info = game.get("home", {})
        away_info = game.get("away", {})
        home_team = home_info.get("team_name", game.get("home_team", ""))
        away_team = away_info.get("team_name", game.get("away_team", ""))
        home_pts = _safe(home_info.get("pts", 0))
        away_pts = _safe(away_info.get("pts", 0))
        game_date = game.get("game_date", game.get("date", "2020-01-01"))

        ht = TEAM_MAP.get(home_team, home_team[:3].upper() if home_team else "UNK")
        at = TEAM_MAP.get(away_team, away_team[:3].upper() if away_team else "UNK")

        if not home_team or not away_team or (home_pts == 0 and away_pts == 0):
            continue

        home_win = 1 if home_pts > away_pts else 0
        margin = home_pts - away_pts

        hr = team_results[ht]
        ar = team_results[at]

        if len(hr) < 5 or len(ar) < 5:
            # Update state and skip (not enough history)
            team_results[ht].append((game_date, home_win == 1, margin, home_pts, at))
            team_results[at].append((game_date, home_win == 0, -margin, away_pts, ht))
            team_elo[ht], team_elo[at] = _update_elo(team_elo[ht], team_elo[at], home_win)
            team_last[ht] = game_date
            team_last[at] = game_date
            continue

        row = []
        names = []

        # ── Rolling performance features (6 windows x 8 stats x 2 teams = 96) ──
        for w in WINDOWS:
            h_stats = _rolling_stats(hr, w)
            a_stats = _rolling_stats(ar, w)
            stat_labels = ["winpct", "avg_pts", "avg_margin", "margin_vol",
                           "best_margin", "worst_margin", "median_margin", "form"]
            for i, label in enumerate(stat_labels):
                row.extend([h_stats[i], a_stats[i], h_stats[i] - a_stats[i]])
                names.extend([f"h_{label}_w{w}", f"a_{label}_w{w}", f"d_{label}_w{w}"])

        # ── ELO features (3) ──
        row.extend([team_elo[ht], team_elo[at], team_elo[ht] - team_elo[at]])
        names.extend(["h_elo", "a_elo", "elo_d"])

        # ── Rest features (4) ──
        try:
            gd = datetime.strptime(str(game_date)[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            gd = datetime(2020, 1, 1)

        home_rest = 3.0
        away_rest = 3.0
        if ht in team_last:
            try:
                ld = datetime.strptime(str(team_last[ht])[:10], "%Y-%m-%d")
                home_rest = min((gd - ld).days, 10)
            except (ValueError, TypeError):
                pass
        if at in team_last:
            try:
                ld = datetime.strptime(str(team_last[at])[:10], "%Y-%m-%d")
                away_rest = min((gd - ld).days, 10)
            except (ValueError, TypeError):
                pass

        row.extend([home_rest, away_rest, home_rest - away_rest,
                    1.0 if home_rest <= 1 else 0.0])
        names.extend(["h_rest", "a_rest", "rest_d", "h_b2b"])

        # ── Streak features (4) ──
        h_streak = _streak(hr)
        a_streak = _streak(ar)
        row.extend([h_streak, a_streak, h_streak - a_streak,
                    1.0 if h_streak >= 3 else (- 1.0 if h_streak <= -3 else 0.0)])
        names.extend(["h_streak", "a_streak", "streak_d", "h_hot_cold"])

        # ── Head-to-head (3) ──
        h2h = [r for r in hr if r[4] == at][-10:]
        h2h_wins = sum(1 for r in h2h if r[1]) / max(len(h2h), 1)
        h2h_margin = float(np.mean([r[2] for r in h2h])) if h2h else 0.0
        row.extend([h2h_wins, h2h_margin, len(h2h)])
        names.extend(["h2h_wp", "h2h_mg", "h2h_n"])

        # ── Season context (4) ──
        h_games_played = len(hr)
        a_games_played = len(ar)
        h_season_winpct = sum(1 for r in hr if r[1]) / max(h_games_played, 1)
        a_season_winpct = sum(1 for r in ar if r[1]) / max(a_games_played, 1)
        row.extend([h_season_winpct, a_season_winpct,
                    h_season_winpct - a_season_winpct, h_games_played / 82.0])
        names.extend(["h_swp", "a_swp",
                       "swp_d", "season_progress"])

        # ── Home advantage constant (1) ──
        row.append(1.0)
        names.append("home_court")

        if first:
            feature_names = names
            first = False

        X.append(row)
        y_out.append(home_win)

        # Update state
        team_results[ht].append((game_date, home_win == 1, margin, home_pts, at))
        team_results[at].append((game_date, home_win == 0, -margin, away_pts, ht))
        team_elo[ht], team_elo[at] = _update_elo(team_elo[ht], team_elo[at], home_win)
        team_last[ht] = game_date
        team_last[at] = game_date

    X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    y_arr = np.array(y_out, dtype=np.int32)
    print(f"Feature matrix: {X_arr.shape} ({len(feature_names)} features, {len(y_arr)} games)")
    return X_arr, y_arr, feature_names


def _update_elo(home_elo, away_elo, home_win, K=20, HCA=100):
    """Update Elo ratings."""
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - HCA) / 400))
    home_elo += K * (home_win - exp_home)
    away_elo += K * ((1 - home_win) - (1 - exp_home))
    return home_elo, away_elo


def _streak(results):
    """Compute current win/loss streak (positive = wins, negative = losses)."""
    if not results:
        return 0
    streak = 0
    last_win = results[-1][1]
    for r in reversed(results):
        if r[1] == last_win:
            streak += 1
        else:
            break
    return streak if last_win else -streak

# %% [markdown]
# ## 6. GPU Model Definitions
#
# All models that benefit from GPU acceleration.

# %%
# ═══════════════════════════════════════════════════════════
# 6a. PyTorch MLP with Dropout + BatchNorm
# ═══════════════════════════════════════════════════════════

class NBANet(nn.Module):
    """MLP with BatchNorm, Dropout, and residual connections."""

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NBANetResidual(nn.Module):
    """MLP with residual skip connections."""

    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)  # Residual
            x = torch.relu(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# 6b. LSTM on Rolling Game Sequences
# ═══════════════════════════════════════════════════════════

class NBALSTM(nn.Module):
    """LSTM that processes a sequence of recent game feature vectors."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # Take last timestep
        return self.head(last)


# ═══════════════════════════════════════════════════════════
# 6c. FT-Transformer (Feature Tokenizer + Transformer)
# ═══════════════════════════════════════════════════════════

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.
    Each feature gets its own embedding, then passed through Transformer blocks.
    Simplified implementation following Gorishniy et al. (2021).
    """

    def __init__(self, input_dim, d_token=64, n_heads=4, n_blocks=3, dropout=0.2):
        super().__init__()
        self.d_token = d_token

        # Per-feature linear tokenizer: each feature -> d_token vector
        self.feature_tokenizer = nn.Linear(1, d_token)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_token))

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Tokenize: (batch, n_features) -> (batch, n_features, d_token)
        tokens = self.feature_tokenizer(x.unsqueeze(-1))  # (B, F, d)
        tokens = tokens + self.feature_bias.unsqueeze(0)   # Add per-feature bias

        # Prepend [CLS]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, d)

        # Transformer
        out = self.transformer(tokens)

        # Classification from [CLS] token
        cls_out = out[:, 0, :]
        return self.head(cls_out)


# ═══════════════════════════════════════════════════════════
# 6d. Neural Oblivious Decision Ensemble (NODE)
# ═══════════════════════════════════════════════════════════

class ObliviousDecisionTree(nn.Module):
    """Differentiable oblivious decision tree (one layer of NODE)."""

    def __init__(self, input_dim, n_trees=128, depth=6):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth

        # Feature selection weights (soft)
        self.feature_select = nn.Linear(input_dim, n_trees * depth)
        # Thresholds
        self.thresholds = nn.Parameter(torch.randn(n_trees, depth) * 0.5)
        # Leaf response values (2^depth leaves per tree)
        n_leaves = 2 ** depth
        self.response = nn.Parameter(torch.randn(n_trees, n_leaves) * 0.01)

    def forward(self, x):
        # x: (batch, features)
        batch_size = x.size(0)

        # Compute comparisons: (batch, n_trees, depth)
        feats = self.feature_select(x).view(batch_size, self.n_trees, self.depth)
        comparisons = torch.sigmoid(feats - self.thresholds.unsqueeze(0))

        # Binary path: convert to leaf indices via binary encoding
        # Each comparison gives a "soft" bit. Combine via outer product.
        # Simplified: use Gumbel-softmax style leaf routing
        leaf_probs = comparisons  # (B, T, D) — each in [0,1]

        # Build leaf weights: for depth D, we have 2^D leaves
        # leaf[i] = prod of (comparison[j] if bit j of i is 1 else 1-comparison[j])
        n_leaves = 2 ** self.depth
        leaf_weights = torch.ones(batch_size, self.n_trees, n_leaves, device=x.device)
        for d in range(self.depth):
            bit = (torch.arange(n_leaves, device=x.device) >> d) & 1  # (n_leaves,)
            bit = bit.float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_leaves)
            comp = leaf_probs[:, :, d:d+1]  # (B, T, 1)
            leaf_weights = leaf_weights * (comp * bit + (1 - comp) * (1 - bit))

        # Weighted response: (batch, n_trees)
        response = (leaf_weights * self.response.unsqueeze(0)).sum(dim=-1)

        # Average over trees
        return response.mean(dim=-1, keepdim=True)


class NODEModel(nn.Module):
    """Neural Oblivious Decision Ensemble — stacked layers of ODTs."""

    def __init__(self, input_dim, n_layers=3, n_trees=64, depth=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(n_layers):
            self.layers.append(ObliviousDecisionTree(current_dim, n_trees, depth))
            self.bn_layers.append(nn.BatchNorm1d(1))
            # NODE concatenates ODT output with input for next layer
            current_dim = input_dim  # Keep using original features

        self.head = nn.Linear(n_layers, 1)

    def forward(self, x):
        outputs = []
        for layer, bn in zip(self.layers, self.bn_layers):
            out = layer(x)  # (B, 1)
            out = bn(out)
            outputs.append(out)
        stacked = torch.cat(outputs, dim=-1)  # (B, n_layers)
        return self.head(stacked)


# %% [markdown]
# ## 7. Training Utilities

# %%
def prepare_sequences(X, seq_len=10):
    """Convert flat feature matrix to sequences for LSTM.

    For game i, the sequence is [X[i-seq_len], ..., X[i-1], X[i]].
    Games without enough history are padded with zeros.
    """
    n, d = X.shape
    X_seq = np.zeros((n, seq_len, d), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        length = i - start + 1
        X_seq[i, seq_len - length:] = X[start:i + 1]
    return X_seq


def train_pytorch_model(model, X_train, y_train, X_val, y_val,
                         max_epochs=MAX_EPOCHS, patience=PATIENCE,
                         lr=1e-3, weight_decay=1e-4, is_sequence=False):
    """Train a PyTorch model with early stopping. Returns val probabilities."""
    model = model.to(DEVICE)

    if is_sequence:
        X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        X_va_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    else:
        X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        X_va_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False  # TimeSeriesSplit — no shuffle
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss = criterion(val_logits, torch.tensor(
                y_val, dtype=torch.float32
            ).unsqueeze(1).to(DEVICE)).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    # Get probabilities
    model.eval()
    with torch.no_grad():
        logits = model(X_va_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Free GPU memory
    del X_tr_t, X_va_t, y_tr_t
    torch.cuda.empty_cache()

    return probs


def train_tabnet(X_train, y_train, X_val, y_val, params=None):
    """Train TabNet and return val probabilities."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    default_params = {
        "n_d": 32, "n_a": 32,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "scheduler_params": {"step_size": 15, "gamma": 0.9},
        "mask_type": "entmax",
        "verbose": 0,
        "device_name": DEVICE,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    clf = TabNetClassifier(**default_params)
    clf.fit(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        eval_set=[(X_val.astype(np.float32), y_val)],
        eval_metric=["logloss"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
    )

    probs = clf.predict_proba(X_val.astype(np.float32))[:, 1]
    return probs


# %% [markdown]
# ## 8. Walk-Forward Backtesting Engine

# %%
def _sanitize_for_json(obj):
    """Convert numpy/torch types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().numpy().tolist()
    return obj


def walk_forward_evaluate(model_name: str, X: np.ndarray, y: np.ndarray,
                           params: dict = None, n_splits: int = WALK_FORWARD_SPLITS,
                           feature_indices: list = None) -> dict:
    """
    Walk-forward backtesting for any model type.

    Supports: mlp, mlp_residual, lstm, ft_transformer, node, tabnet,
              xgboost_gpu, lightgbm_gpu, catboost_gpu

    Returns dict with brier, accuracy, log_loss, roi, per-fold details.
    """
    params = params or {}

    # Feature selection
    if feature_indices is not None:
        X = X[:, feature_indices]

    # OOM protection
    if X.shape[0] > MAX_EVAL_GAMES:
        X = X[-MAX_EVAL_GAMES:]
        y = y[-MAX_EVAL_GAMES:]

    # Scale features (all GPU models benefit)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6))

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    all_probs = []
    all_true = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Fold {fold_idx + 1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        try:
            probs = _run_model_fold(
                model_name, X_train, y_train, X_val, y_val, params
            )

            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 0.001, 0.999)

            brier = float(brier_score_loss(y_val, probs))
            ll = float(log_loss(y_val, probs))
            acc = float(accuracy_score(y_val, (probs > 0.5).astype(int)))

            fold_results.append({
                "fold": fold_idx + 1,
                "brier": round(brier, 5),
                "log_loss": round(ll, 5),
                "accuracy": round(acc, 4),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            })

            all_probs.extend(probs.tolist())
            all_true.extend(y_val.tolist())

            print(f"    Brier: {brier:.4f} | Acc: {acc:.4f} | LogLoss: {ll:.4f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            fold_results.append({
                "fold": fold_idx + 1,
                "error": str(e)[:300],
            })

    # Aggregate results
    valid_folds = [f for f in fold_results if "brier" in f]
    if not valid_folds:
        return {"brier": 1.0, "accuracy": 0.0, "log_loss": 10.0,
                "error": "All folds failed", "folds": fold_results}

    avg_brier = float(np.mean([f["brier"] for f in valid_folds]))
    avg_acc = float(np.mean([f["accuracy"] for f in valid_folds]))
    avg_ll = float(np.mean([f["log_loss"] for f in valid_folds]))

    # Overall metrics from concatenated predictions
    if all_probs:
        all_probs_arr = np.array(all_probs)
        all_true_arr = np.array(all_true)
        overall_brier = float(brier_score_loss(all_true_arr, all_probs_arr))
        overall_acc = float(accuracy_score(all_true_arr, (all_probs_arr > 0.5).astype(int)))
        overall_ll = float(log_loss(all_true_arr, all_probs_arr))
    else:
        overall_brier, overall_acc, overall_ll = avg_brier, avg_acc, avg_ll

    # Simple ROI estimate: bet on predicted side if prob > 0.55, -110 odds
    roi = 0.0
    if all_probs:
        bets = 0
        profit = 0.0
        for p, actual in zip(all_probs, all_true):
            if p > 0.55:
                bets += 1
                profit += (100 / 110) if actual == 1 else -1.0
            elif p < 0.45:
                bets += 1
                profit += (100 / 110) if actual == 0 else -1.0
        roi = float(profit / max(bets, 1))

    return {
        "model": model_name,
        "brier": round(overall_brier, 5),
        "accuracy": round(overall_acc, 4),
        "log_loss": round(overall_ll, 5),
        "roi": round(roi, 4),
        "avg_brier": round(avg_brier, 5),
        "avg_accuracy": round(avg_acc, 4),
        "n_splits": n_splits,
        "n_games": X.shape[0],
        "n_features": X.shape[1],
        "folds": fold_results,
        "valid_folds": len(valid_folds),
    }


def _run_model_fold(model_name, X_train, y_train, X_val, y_val, params):
    """Run a single fold for a given model type. Returns val probabilities."""

    # ── PyTorch MLP ──
    if model_name == "mlp":
        hidden = params.get("hidden_dims", (256, 128, 64))
        dropout = params.get("dropout", 0.3)
        model = NBANet(X_train.shape[1], hidden_dims=hidden, dropout=dropout)
        return train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
        )

    # ── PyTorch Residual MLP ──
    if model_name == "mlp_residual":
        model = NBANetResidual(
            X_train.shape[1],
            hidden_dim=params.get("hidden_dim", 128),
            n_blocks=params.get("n_blocks", 3),
            dropout=params.get("dropout", 0.3),
        )
        return train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            lr=params.get("lr", 1e-3),
        )

    # ── LSTM on game sequences ──
    if model_name == "lstm":
        seq_len = params.get("seq_len", 10)
        X_tr_seq = prepare_sequences(X_train, seq_len).astype(np.float32)
        X_va_seq = prepare_sequences(X_val, seq_len).astype(np.float32)
        model = NBALSTM(
            X_train.shape[1],
            hidden_dim=params.get("hidden_dim", 64),
            n_layers=params.get("n_layers", 2),
            dropout=params.get("dropout", 0.2),
        )
        return train_pytorch_model(
            model, X_tr_seq, y_train, X_va_seq, y_val,
            lr=params.get("lr", 1e-3),
            is_sequence=True,
        )

    # ── FT-Transformer ──
    if model_name == "ft_transformer":
        model = FTTransformer(
            X_train.shape[1],
            d_token=params.get("d_token", 64),
            n_heads=params.get("n_heads", 4),
            n_blocks=params.get("n_blocks", 3),
            dropout=params.get("dropout", 0.2),
        )
        return train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            lr=params.get("lr", 5e-4),
            weight_decay=params.get("weight_decay", 1e-5),
        )

    # ── NODE ──
    if model_name == "node":
        model = NODEModel(
            X_train.shape[1],
            n_layers=params.get("n_layers", 3),
            n_trees=params.get("n_trees", 64),
            depth=params.get("depth", 5),
        )
        return train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
        )

    # ── TabNet ──
    if model_name == "tabnet":
        return train_tabnet(X_train, y_train, X_val, y_val, params)

    # ── XGBoost GPU ──
    if model_name == "xgboost_gpu":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            min_child_weight=params.get("min_child_weight", 5),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            tree_method="gpu_hist",
            device="cuda",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)[:, 1]

    # ── LightGBM GPU ──
    if model_name == "lightgbm_gpu":
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            num_leaves=params.get("num_leaves", 31),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            device="gpu",
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)[:, 1]

    # ── CatBoost GPU ──
    if model_name == "catboost_gpu":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=params.get("iterations", 200),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            task_type="GPU",
            devices="0",
            verbose=0,
            random_seed=42,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)[:, 1]

    raise ValueError(f"Unknown model: {model_name}. Supported: mlp, mlp_residual, "
                     f"lstm, ft_transformer, node, tabnet, xgboost_gpu, lightgbm_gpu, catboost_gpu")


# %% [markdown]
# ## 9. Experiment Queue Interface

# %%
def fetch_next_experiment() -> Optional[Dict[str, Any]]:
    """Fetch AND atomically claim the next pending GPU experiment from Supabase.

    Uses CTE with FOR UPDATE SKIP LOCKED to prevent double-pickup when
    multiple Colab instances poll simultaneously.
    """
    rows = _exec_sql("""
        WITH next_exp AS (
            SELECT id FROM public.nba_experiments
            WHERE status = 'pending'
              AND (target_space = 'colab' OR target_space = 'gpu'
                   OR target_space = 'any' OR target_space IS NULL)
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE public.nba_experiments e
        SET status = 'running', started_at = NOW()
        FROM next_exp
        WHERE e.id = next_exp.id
        RETURNING e.id, e.experiment_id, e.agent_name, e.experiment_type,
                  e.description, e.hypothesis, e.params, e.priority,
                  e.status, e.target_space, e.baseline_brier, e.created_at
    """)
    if not rows or rows is True or len(rows) == 0:
        return None
    row = rows[0]
    return {
        "id": row[0],
        "experiment_id": row[1],
        "agent_name": row[2],
        "experiment_type": row[3],
        "description": row[4],
        "hypothesis": row[5],
        "params": row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
        "priority": row[7],
        "status": row[8],
        "target_space": row[9],
        "baseline_brier": float(row[10]) if row[10] else None,
        "created_at": str(row[11]) if row[11] else None,
    }


def claim_experiment(exp_id: int) -> bool:
    """Legacy claim — now handled atomically in fetch_next_experiment(), kept for compatibility."""
    return True


def complete_experiment(exp_id: int, brier: float, accuracy: float,
                        log_loss_val: float, details: dict,
                        status: str = "completed"):
    """Write results back to Supabase.

    IMPORTANT: Cast all values to native Python float/int to avoid
    psycopg2 'can't adapt type numpy.float64' errors.
    """
    clean_details = _sanitize_for_json(details)
    _exec_sql("""
        UPDATE public.nba_experiments
        SET status = %s,
            result_brier = %s,
            result_accuracy = %s,
            result_log_loss = %s,
            result_details = %s,
            feature_engine_version = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (
        str(status),
        float(brier),
        float(accuracy),
        float(log_loss_val),
        json.dumps(clean_details),
        FEATURE_ENGINE_VERSION,
        int(exp_id),
    ), fetch=False)


def fail_experiment(exp_id: int, error_msg: str):
    """Mark experiment as failed."""
    details = {
        "error": str(error_msg)[:2000],
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "runner": "colab_gpu",
    }
    _exec_sql("""
        UPDATE public.nba_experiments
        SET status = 'failed',
            result_details = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (json.dumps(details), int(exp_id)), fetch=False)


# %% [markdown]
# ## 10. Experiment Execution Router

# %%
# All GPU-capable model types
GPU_MODELS = [
    "mlp", "mlp_residual", "lstm", "ft_transformer", "node", "tabnet",
    "xgboost_gpu", "lightgbm_gpu", "catboost_gpu",
]


def run_gpu_experiment(experiment: dict, X: np.ndarray, y: np.ndarray,
                        feature_names: list) -> dict:
    """Execute a single GPU experiment."""
    params = experiment["params"]
    exp_type = experiment["experiment_type"]
    exp_id = experiment["id"]

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {experiment['experiment_id']}")
    print(f"Type: {exp_type} | Agent: {experiment['agent_name']}")
    print(f"Description: {experiment['description'][:200]}")
    print(f"{'=' * 60}")

    # Claim it
    if not claim_experiment(exp_id):
        raise RuntimeError(f"Experiment {exp_id} already claimed")

    start_time = time.time()

    try:
        # ── model_test: Run a specific GPU model ──
        if exp_type == "model_test":
            model_name = params.get("model_type", "mlp")
            if model_name not in GPU_MODELS:
                # Try mapping from S10/S11 names
                model_map = {
                    "xgboost": "xgboost_gpu",
                    "lightgbm": "lightgbm_gpu",
                    "catboost": "catboost_gpu",
                }
                model_name = model_map.get(model_name, model_name)

            feature_indices = params.get("feature_indices")
            n_splits = params.get("n_splits", WALK_FORWARD_SPLITS)

            results = walk_forward_evaluate(
                model_name, X, y,
                params=params.get("hyperparams", params),
                n_splits=n_splits,
                feature_indices=feature_indices,
            )

        # ── gpu_benchmark: Run ALL GPU models and compare ──
        elif exp_type == "gpu_benchmark":
            models_to_test = params.get("models", GPU_MODELS)
            n_splits = params.get("n_splits", WALK_FORWARD_SPLITS)
            feature_indices = params.get("feature_indices")

            all_results = {}
            best_brier = 1.0
            best_model = None

            for model_name in models_to_test:
                print(f"\n--- Benchmarking: {model_name} ---")
                try:
                    model_params = params.get(f"{model_name}_params", {})
                    r = walk_forward_evaluate(
                        model_name, X, y,
                        params=model_params,
                        n_splits=n_splits,
                        feature_indices=feature_indices,
                    )
                    all_results[model_name] = r
                    if r["brier"] < best_brier:
                        best_brier = r["brier"]
                        best_model = model_name
                    print(f"  -> Brier: {r['brier']:.4f} | Acc: {r['accuracy']:.4f}")
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    all_results[model_name] = {"error": str(e)[:300]}

                # Clear GPU memory between models
                torch.cuda.empty_cache()

            results = {
                "brier": float(best_brier),
                "accuracy": float(all_results.get(best_model, {}).get("accuracy", 0.0)),
                "log_loss": float(all_results.get(best_model, {}).get("log_loss", 10.0)),
                "best_model": best_model,
                "all_results": all_results,
                "models_tested": len(models_to_test),
            }

        # ── feature_test: Test feature subset with GPU model ──
        elif exp_type == "feature_test":
            model_name = params.get("model_type", "mlp")
            if model_name not in GPU_MODELS:
                model_name = "mlp"
            feature_indices = params.get("feature_indices")
            n_splits = params.get("n_splits", WALK_FORWARD_SPLITS)

            results = walk_forward_evaluate(
                model_name, X, y,
                params=params.get("hyperparams", {}),
                n_splits=n_splits,
                feature_indices=feature_indices,
            )

        # ── calibration_test with GPU models ──
        elif exp_type == "calibration_test":
            # Run the same model with different post-hoc calibrations
            # (isotonic/sigmoid applied to GPU model output)
            model_name = params.get("model_type", "mlp")
            n_splits = params.get("n_splits", WALK_FORWARD_SPLITS)

            results = walk_forward_evaluate(
                model_name, X, y,
                params=params.get("hyperparams", {}),
                n_splits=n_splits,
            )
            # Note: calibration for neural nets is typically inherent in the
            # BCE loss training. External calibration can be added as future work.

        else:
            raise ValueError(
                f"Unknown experiment type for GPU runner: {exp_type}. "
                f"Supported: model_test, gpu_benchmark, feature_test, calibration_test"
            )

        elapsed = time.time() - start_time

        # Ensure all values are native Python types for Supabase
        brier = float(results.get("brier", 1.0))
        accuracy = float(results.get("accuracy", 0.0))
        ll = float(results.get("log_loss", 10.0))

        results["elapsed_seconds"] = round(elapsed, 1)
        results["games_evaluated"] = min(int(X.shape[0]), MAX_EVAL_GAMES)
        results["feature_candidates"] = int(X.shape[1])
        results["experiment_id"] = experiment["experiment_id"]
        results["agent_name"] = experiment["agent_name"]
        results["runner"] = "colab_gpu"
        results["device"] = DEVICE
        if torch.cuda.is_available():
            results["gpu_name"] = torch.cuda.get_device_name(0)

        # Compare with baseline
        baseline = experiment.get("baseline_brier")
        if baseline:
            results["improvement"] = round(float(baseline) - brier, 5)
            results["improved"] = brier < float(baseline)

        complete_experiment(exp_id, brier, accuracy, ll, results)

        print(f"\nCOMPLETED: Brier={brier:.4f} | Acc={accuracy:.4f} | {elapsed:.1f}s")
        if baseline:
            delta = float(baseline) - brier
            print(f"vs baseline {baseline:.4f}: "
                  f"{'BETTER' if delta > 0 else 'WORSE'} by {abs(delta):.4f}")

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{str(e)[:500]}\n{traceback.format_exc()[-1000:]}"
        fail_experiment(exp_id, error_msg)
        print(f"\nFAILED: {str(e)[:300]} ({elapsed:.1f}s)")
        raise


# %% [markdown]
# ## 11. Load Data & Build Features

# %%
print("Loading game data...")

# Try sources in order: local JSON files -> Supabase -> CSV
games = load_games_from_json_files()
if len(games) < 100:
    games = load_games_from_supabase()
if len(games) < 100:
    games = load_games_from_csv()

if len(games) < 100:
    print("ERROR: Not enough games loaded. Upload data or check Supabase connection.")
    print("You can upload games-*.json files to /content/data/historical/")
else:
    print(f"\nBuilding features from {len(games)} games...")
    X, y, feature_names = build_features_inline(games)

    # Remove zero-variance features
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        X = X[:, valid_mask]
        feature_names = [f for f, v in zip(feature_names, valid_mask) if v]
        print(f"Removed {n_removed} zero-variance features")

    # OOM protection
    if X.shape[0] > MAX_EVAL_GAMES:
        print(f"Capping to {MAX_EVAL_GAMES} most recent games (OOM protection)")
        X = X[-MAX_EVAL_GAMES:]
        y = y[-MAX_EVAL_GAMES:]

    print(f"\nReady: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Home win rate: {y.mean():.3f}")

# %% [markdown]
# ## 12. Quick Sanity Test (Single Model)
#
# Run a quick test to verify everything works before starting the loop.

# %%
if 'X' in dir() and X.shape[0] > 500:
    print("Running quick sanity test with MLP...")
    quick_result = walk_forward_evaluate("mlp", X, y, params={"hidden_dims": (64, 32)}, n_splits=2)
    print(f"\nQuick test result:")
    print(f"  Brier:    {quick_result['brier']:.4f}")
    print(f"  Accuracy: {quick_result['accuracy']:.4f}")
    print(f"  LogLoss:  {quick_result['log_loss']:.4f}")
    print(f"  ROI:      {quick_result['roi']:.4f}")

    torch.cuda.empty_cache()
else:
    print("Skipping sanity test (not enough data)")

# %% [markdown]
# ## 13. Submit a Manual Experiment
#
# Use this cell to submit an experiment directly to the Supabase queue,
# or run one immediately.

# %%
def submit_experiment(model_type: str, description: str = "",
                      params: dict = None, priority: int = 5,
                      baseline_brier: float = None):
    """Submit a GPU experiment to the Supabase queue."""
    exp_id = f"colab-{model_type}-{int(time.time())}"
    params = params or {}
    params["model_type"] = model_type

    result = _exec_sql("""
        INSERT INTO public.nba_experiments
        (experiment_id, agent_name, experiment_type, description, hypothesis,
         params, priority, status, target_space, baseline_brier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 'colab', %s)
        RETURNING id
    """, (
        exp_id,
        "colab_gpu_manual",
        "model_test",
        description or f"GPU test: {model_type}",
        f"Test {model_type} on GPU with walk-forward backtest",
        json.dumps(params),
        priority,
        float(baseline_brier) if baseline_brier else None,
    ))

    if result and result is not True:
        db_id = result[0][0]
        print(f"Experiment submitted: {exp_id} (DB id={db_id})")
        return exp_id
    else:
        print("Failed to submit experiment")
        return None


def run_all_gpu_models(n_splits=3, baseline_brier=0.2205):
    """Submit a gpu_benchmark experiment that tests all GPU models."""
    exp_id = f"colab-benchmark-{int(time.time())}"
    params = {
        "models": GPU_MODELS,
        "n_splits": n_splits,
    }
    result = _exec_sql("""
        INSERT INTO public.nba_experiments
        (experiment_id, agent_name, experiment_type, description, hypothesis,
         params, priority, status, target_space, baseline_brier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 'colab', %s)
        RETURNING id
    """, (
        exp_id,
        "colab_gpu_benchmark",
        "gpu_benchmark",
        f"Full GPU benchmark: {len(GPU_MODELS)} models",
        "Find the best GPU-accelerated model for NBA prediction",
        json.dumps(params),
        10,  # High priority
        float(baseline_brier) if baseline_brier else None,
    ))

    if result and result is not True:
        db_id = result[0][0]
        print(f"Benchmark submitted: {exp_id} (DB id={db_id})")
        print(f"Models: {GPU_MODELS}")
        return exp_id
    else:
        print("Failed to submit benchmark")
        return None


# Example usage (uncomment to submit):
# submit_experiment("ft_transformer", "Test FT-Transformer on NBA data", baseline_brier=0.2205)
# submit_experiment("tabnet", "Test TabNet with attention masking", baseline_brier=0.2205)
# submit_experiment("node", "Test Neural ODT Ensemble", baseline_brier=0.2205)
# run_all_gpu_models()

# %% [markdown]
# ## 14. Autonomous Polling Loop
#
# Run this cell to start polling Supabase for pending experiments.
# The loop runs continuously until interrupted (Ctrl+C or stop button).

# %%
def experiment_loop(max_iterations: int = 0):
    """
    Main autonomous experiment polling loop.

    Args:
        max_iterations: 0 = infinite loop, N = run N experiments then stop
    """
    if 'X' not in dir() and 'X' not in globals():
        print("ERROR: No feature data loaded. Run the data loading cell first.")
        return

    global X, y, feature_names

    print("=" * 60)
    print("COLAB GPU EXPERIMENT RUNNER — STARTING")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Polling every {POLL_INTERVAL}s for experiments")
    print(f"Target spaces: colab, gpu, any, NULL")
    print("=" * 60)

    completed = 0
    failed = 0
    consecutive_errors = 0

    while True:
        try:
            experiment = fetch_next_experiment()

            if experiment:
                pending_count = _count_pending()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Found: {experiment['experiment_id']} "
                      f"(queue depth: {pending_count})")

                try:
                    run_gpu_experiment(experiment, X, y, feature_names)
                    completed += 1
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    failed += 1

                consecutive_errors = 0

                # Check iteration limit
                if max_iterations > 0 and (completed + failed) >= max_iterations:
                    print(f"\nReached {max_iterations} iterations. Stopping.")
                    break

                # Clear GPU memory
                torch.cuda.empty_cache()

            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"No pending experiments (completed={completed}, failed={failed}). "
                      f"Sleeping {POLL_INTERVAL}s...", end="\r")
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\nStopped by user. Completed={completed}, Failed={failed}")
            break

        except Exception as e:
            consecutive_errors += 1
            print(f"\n[ERROR] Poll error #{consecutive_errors}: {e}")

            if consecutive_errors >= 5:
                print("5 consecutive errors — reconnecting to Supabase")
                _reconnect_pg()
                consecutive_errors = 0

            time.sleep(POLL_INTERVAL * 2)

    print(f"\nFinal: {completed} completed, {failed} failed")


def _count_pending() -> int:
    """Count pending GPU experiments."""
    rows = _exec_sql("""
        SELECT COUNT(*) FROM public.nba_experiments
        WHERE status = 'pending'
          AND (target_space = 'colab' OR target_space = 'gpu'
               OR target_space = 'any' OR target_space IS NULL)
    """)
    if rows and rows is not True and len(rows) > 0:
        return int(rows[0][0])
    return 0


# Start the loop (uncomment one):
# experiment_loop()          # Run forever
# experiment_loop(5)         # Run 5 experiments then stop

# %% [markdown]
# ## 15. Quick Run: Test All GPU Models Now
#
# Skip the queue and run all GPU models directly.

# %%
def benchmark_all_models_now(n_splits=3):
    """Run all GPU models directly (no Supabase queue needed)."""
    if 'X' not in dir() and 'X' not in globals():
        print("ERROR: No feature data loaded.")
        return

    results = {}
    print(f"Benchmarking {len(GPU_MODELS)} models on {X.shape[0]} games, "
          f"{X.shape[1]} features, {n_splits} splits\n")

    for model_name in GPU_MODELS:
        print(f"\n{'─' * 40}")
        print(f"Model: {model_name}")
        print(f"{'─' * 40}")
        try:
            r = walk_forward_evaluate(model_name, X, y, n_splits=n_splits)
            results[model_name] = r
            print(f"  Brier:    {r['brier']:.4f}")
            print(f"  Accuracy: {r['accuracy']:.4f}")
            print(f"  LogLoss:  {r['log_loss']:.4f}")
            print(f"  ROI:      {r['roi']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[model_name] = {"error": str(e)}

        torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Brier':>8} {'Accuracy':>10} {'LogLoss':>10} {'ROI':>8}")
    print(f"{'─' * 60}")

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "brier" in v],
        key=lambda x: x[1]["brier"]
    )
    for name, r in sorted_results:
        print(f"{name:<20} {r['brier']:>8.4f} {r['accuracy']:>10.4f} "
              f"{r['log_loss']:>10.4f} {r['roi']:>8.4f}")

    failed = [k for k, v in results.items() if "error" in v]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")

    return results


# Uncomment to run immediately:
# results = benchmark_all_models_now(n_splits=3)
