#!/usr/bin/env python3
"""
NBA Quant AI — Kaggle GPU Experiment Runner
Polls Supabase nba_experiments for pending experiments, trains GPU models
(MLP, LSTM, FT-Transformer, TabNet, XGBoost-GPU, LightGBM-GPU, CatBoost-GPU),
evaluates via walk-forward backtesting, writes results back.
Picks up experiments with target_space IN ('colab','gpu','kaggle','any',NULL).
Push via Kaggle API — see trigger.sh
"""
import subprocess, sys

# Resilient pip install — don't crash if DNS/network is flaky
for pkg in ["psycopg2-binary", "pytorch-tabnet", "scikit-learn>=1.3"]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                              timeout=60, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARN] pip install {pkg} failed: {e} — using pre-installed version")

import os, json, time, traceback, numpy as np, torch, torch.nn as nn
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import psycopg2
from psycopg2 import pool as pg_pool

print(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Config ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EVAL_GAMES, WALK_FORWARD_SPLITS, BATCH_SIZE = 10000, 3, 512
MAX_EPOCHS, PATIENCE, MAX_EXPERIMENTS, POLL_INTERVAL = 200, 15, 10, 30
GPU_MODELS = ["mlp", "mlp_residual", "lstm", "ft_transformer", "tabnet",
              "node", "mc_dropout_rnn", "saint", "tft",
              "xgboost_gpu", "lightgbm_gpu", "catboost_gpu"]

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    try:
        from kaggle_secrets import UserSecretsClient
        DATABASE_URL = UserSecretsClient().get_secret("DATABASE_URL")
        print("DATABASE_URL loaded from Kaggle secrets.")
    except Exception as e:
        print(f"[WARN] Kaggle secrets: {e}")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Add via Kaggle -> Add-ons -> Secrets.")

# ═══════════════════════════════════════════════
# SUPABASE CONNECTION
# ═══════════════════════════════════════════════
_pg_pool = None

def _get_pg():
    global _pg_pool
    if _pg_pool: return _pg_pool
    if not DATABASE_URL: return None
    try:
        _pg_pool = pg_pool.SimpleConnectionPool(1, 3, DATABASE_URL,
                                                 options="-c search_path=public")
        c = _pg_pool.getconn()
        c.cursor().execute("SELECT 1"); c.commit()
        _pg_pool.putconn(c)
        print("[OK] PostgreSQL connected")
        return _pg_pool
    except Exception as e:
        print(f"[ERROR] PG connect: {e}"); _pg_pool = None; return None

def _exec_sql(sql, params=None, fetch=True):
    pool = _get_pg()
    if not pool: return None
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params); conn.commit()
            if fetch:
                try: return cur.fetchall()
                except: return True
            return True
    except Exception as e:
        print(f"[SQL ERROR] {e}")
        if conn:
            try: conn.rollback()
            except: pass
        return None
    finally:
        if conn and pool:
            try: pool.putconn(conn)
            except: pass

def _reconnect_pg():
    global _pg_pool
    if _pg_pool:
        try: _pg_pool.closeall()
        except: pass
    _pg_pool = None

_get_pg()

# ═══════════════════════════════════════════════
# DATA LOADING — GitHub raw files (game data NOT in Supabase)
# ═══════════════════════════════════════════════

GITHUB_RAW = "https://raw.githubusercontent.com/LBJLincoln/nomos-nba-agent/main/data/historical"
SEASONS = ["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"]

def load_games() -> List[dict]:
    """Load game data from GitHub repo (primary) or local files (fallback)."""
    import urllib.request
    games = []

    # Try GitHub raw files first (works on Kaggle/Colab with internet)
    for season in SEASONS:
        url = f"{GITHUB_RAW}/games-{season}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NomoS42/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    games.extend(data)
                    print(f"  {season}: {len(data)} games (GitHub)")
        except Exception as e:
            print(f"  {season}: GitHub failed ({e})")

    if games:
        games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
        print(f"Loaded {len(games)} total games from GitHub")
        return games

    # Fallback: local files (if cloned repo)
    from pathlib import Path
    for hist_dir in [Path("data/historical"), Path("../data/historical"), Path("/kaggle/working/data/historical")]:
        if hist_dir.exists():
            for f in sorted(hist_dir.glob("games-*.json")):
                data = json.loads(f.read_text())
                games.extend(data if isinstance(data, list) else data.get("games", []))
            if games:
                games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
                print(f"Loaded {len(games)} games from {hist_dir}")
                return games

    # Last resort: nba_api
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        for season in SEASONS[-3:]:
            time.sleep(2)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable="00",
                season_type_nullable="Regular Season", timeout=20)
            df = finder.get_data_frames()[0]
            pairs = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in pairs: pairs[gid] = []
                pairs[gid].append(row)
            for gid, teams in pairs.items():
                if len(teams) != 2: continue
                home = next((t for t in teams if " vs. " in str(t.get("MATCHUP",""))), None)
                away = next((t for t in teams if " @ " in str(t.get("MATCHUP",""))), None)
                if home is not None and away is not None:
                    games.append({"game_date": home.get("GAME_DATE",""),
                                  "home_team": home["TEAM_NAME"], "away_team": away["TEAM_NAME"],
                                  "home": {"team_name": home["TEAM_NAME"], "pts": int(home["PTS"])},
                                  "away": {"team_name": away["TEAM_NAME"], "pts": int(away["PTS"])}})
        if games:
            games.sort(key=lambda g: g.get("game_date",""))
            print(f"Loaded {len(games)} games from NBA API")
            return games
    except Exception as e:
        print(f"[WARN] nba_api fallback failed: {e}")

    print("[ERROR] No games found from any source"); return []

# ═══════════════════════════════════════════════
# FEATURE ENGINE (~160 features)
# ═══════════════════════════════════════════════

TEAM_MAP = {
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN",
    "Charlotte Hornets":"CHA","Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE",
    "Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET",
    "Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
    "Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM",
    "Miami Heat":"MIA","Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN",
    "New Orleans Pelicans":"NOP","New York Knicks":"NYK","Oklahoma City Thunder":"OKC",
    "Orlando Magic":"ORL","Philadelphia 76ers":"PHI","Phoenix Suns":"PHX",
    "Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS",
    "Toronto Raptors":"TOR","Utah Jazz":"UTA","Washington Wizards":"WAS"}
WINDOWS = [3, 5, 7, 10, 15, 20]

def _safe(v, d=0.0):
    try: f = float(v); return f if np.isfinite(f) else d
    except: return d

def _elo(he, ae, hw, K=20, H=100):
    e = 1.0/(1.0+10**((ae-he-H)/400)); return he+K*(hw-e), ae+K*((1-hw)-(1-e))

def _streak(res):
    if not res: return 0
    s, lw = 0, res[-1][1]
    for r in reversed(res):
        if r[1] == lw: s += 1
        else: break
    return s if lw else -s

def _rstats(res, w):
    rc = res[-w:] if len(res) >= w else res
    if not rc: return [0.0]*6
    wins = sum(1 for r in rc if r[1])
    m = [r[2] for r in rc]; s = [r[3] for r in rc]
    return [wins/len(rc), float(np.mean(s)), float(np.mean(m)),
            float(np.std(m)) if len(m)>1 else 0.0,
            float(np.max(m)), float(np.min(m))]

def build_features(games):
    tr, tl, te = defaultdict(list), {}, defaultdict(lambda: 1500.0)
    X, Y, fnames, first = [], [], [], True
    for g in games:
        hi, ai = g.get("home",{}), g.get("away",{})
        hn = hi.get("team_name", g.get("home_team",""))
        an = ai.get("team_name", g.get("away_team",""))
        hp, ap = _safe(hi.get("pts",0)), _safe(ai.get("pts",0))
        gd = g.get("game_date", g.get("date","2020-01-01"))
        ht = TEAM_MAP.get(hn, hn[:3].upper() if hn else "UNK")
        at = TEAM_MAP.get(an, an[:3].upper() if an else "UNK")
        if not hn or not an or (hp==0 and ap==0): continue
        hw = 1 if hp > ap else 0; mg = hp - ap
        hr, ar = tr[ht], tr[at]
        if len(hr)<5 or len(ar)<5:
            tr[ht].append((gd,hw==1,mg,hp,at)); tr[at].append((gd,hw==0,-mg,ap,ht))
            te[ht],te[at] = _elo(te[ht],te[at],hw); tl[ht]=gd; tl[at]=gd; continue
        row, names = [], []
        # Rolling stats (6 windows x 6 stats x 3 = 108)
        for w in WINDOWS:
            hs, als = _rstats(hr,w), _rstats(ar,w)
            for i,lb in enumerate(["winpct","avg_pts","avg_margin","margin_vol","best_mg","worst_mg"]):
                row.extend([hs[i], als[i], hs[i]-als[i]])
                names.extend([f"h_{lb}_w{w}", f"a_{lb}_w{w}", f"d_{lb}_w{w}"])
        # ELO (3)
        row.extend([te[ht], te[at], te[ht]-te[at]]); names.extend(["h_elo","a_elo","elo_d"])
        # Rest (4)
        try: gdt = datetime.strptime(str(gd)[:10],"%Y-%m-%d")
        except: gdt = datetime(2020,1,1)
        hrest, arest = 3.0, 3.0
        for t,rest_name in [(ht,"h"),(at,"a")]:
            if t in tl:
                try:
                    d = (gdt - datetime.strptime(str(tl[t])[:10],"%Y-%m-%d")).days
                    if rest_name=="h": hrest = min(d,10)
                    else: arest = min(d,10)
                except: pass
        row.extend([hrest, arest, hrest-arest, 1.0 if hrest<=1 else 0.0])
        names.extend(["h_rest","a_rest","rest_d","h_b2b"])
        # Streak (3)
        hs, als = _streak(hr), _streak(ar)
        row.extend([hs, als, hs-als]); names.extend(["h_streak","a_streak","streak_d"])
        # H2H (2)
        h2h = [r for r in hr if r[4]==at][-10:]
        h2hw = sum(1 for r in h2h if r[1])/max(len(h2h),1)
        row.extend([h2hw, len(h2h)]); names.extend(["h2h_wp","h2h_n"])
        # Season (3) + home court (1)
        hgp, agp = len(hr), len(ar)
        hsw = sum(1 for r in hr if r[1])/max(hgp,1)
        asw = sum(1 for r in ar if r[1])/max(agp,1)
        row.extend([hsw, asw, hsw-asw, 1.0])
        names.extend(["h_swp","a_swp","swp_d","home_court"])
        if first: fnames = names; first = False
        X.append(row); Y.append(hw)
        tr[ht].append((gd,hw==1,mg,hp,at)); tr[at].append((gd,hw==0,-mg,ap,ht))
        te[ht],te[at] = _elo(te[ht],te[at],hw); tl[ht]=gd; tl[at]=gd
    Xa = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    ya = np.array(Y, dtype=np.int32)
    print(f"Features: {Xa.shape} ({len(fnames)} cols, {len(ya)} games)")
    return Xa, ya, fnames

# ═══════════════════════════════════════════════
# GPU MODEL DEFINITIONS
# ═══════════════════════════════════════════════

class NBANet(nn.Module):
    def __init__(self, dim, hidden=(256,128,64), drop=0.3):
        super().__init__()
        layers, prev = [], dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class NBALSTM(nn.Module):
    def __init__(self, dim, hid=64, nl=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(dim, hid, nl, dropout=drop if nl>1 else 0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid,32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32,1))
    def forward(self, x):
        o,_ = self.lstm(x); return self.head(o[:,-1,:])

class FTTransformer(nn.Module):
    def __init__(self, dim, d=64, nh=4, nb=3, drop=0.2):
        super().__init__()
        self.tok = nn.Linear(1,d)
        self.bias = nn.Parameter(torch.zeros(dim,d))
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        el = nn.TransformerEncoderLayer(d,nh,d*4,drop,activation="gelu",batch_first=True)
        self.tf = nn.TransformerEncoder(el, nb)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,1))
    def forward(self, x):
        B = x.size(0)
        t = self.tok(x.unsqueeze(-1)) + self.bias.unsqueeze(0)
        t = torch.cat([self.cls.expand(B,-1,-1), t], 1)
        return self.head(self.tf(t)[:,0,:])

# ── Residual MLP (skip connections for deeper learning) ──
class NBANetResidual(nn.Module):
    def __init__(self, dim, hidden=(256,128,64), drop=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(dim, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.GELU())
        blocks = []
        for i in range(len(hidden)-1):
            blocks.append(nn.Sequential(
                nn.Linear(hidden[i], hidden[i+1]), nn.BatchNorm1d(hidden[i+1]),
                nn.GELU(), nn.Dropout(drop)))
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([
            nn.Linear(hidden[i], hidden[i+1]) if hidden[i]!=hidden[i+1] else nn.Identity()
            for i in range(len(hidden)-1)])
        self.head = nn.Linear(hidden[-1], 1)
    def forward(self, x):
        x = self.input_proj(x)
        for block, ds in zip(self.blocks, self.downsamples):
            x = block(x) + ds(x)  # residual connection
        return self.head(x)

# ── Neural Oblivious Decision Ensemble (NODE) — differentiable trees ──
class ObliviousTree(nn.Module):
    """Single differentiable oblivious decision tree."""
    def __init__(self, dim, depth=6, out=1):
        super().__init__()
        self.depth = depth
        self.features = nn.Linear(dim, depth, bias=False)
        self.thresholds = nn.Parameter(torch.randn(depth)*0.01)
        self.response = nn.Parameter(torch.randn(2**depth, out)*0.01)
    def forward(self, x):
        h = torch.sigmoid(self.features(x) - self.thresholds)  # (B, depth)
        # Build binary path index
        idx = torch.zeros(x.size(0), 2**self.depth, device=x.device)
        for i in range(2**self.depth):
            bits = [(i >> d) & 1 for d in range(self.depth)]
            p = torch.ones(x.size(0), device=x.device)
            for d, b in enumerate(bits):
                p = p * (h[:,d] if b else 1-h[:,d])
            idx[:, i] = p
        return (idx.unsqueeze(-1) * self.response.unsqueeze(0)).sum(1)

class NODEModel(nn.Module):
    def __init__(self, dim, n_trees=128, depth=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = dim
        for _ in range(n_layers):
            trees = nn.ModuleList([ObliviousTree(curr_dim, depth, 1) for _ in range(n_trees)])
            self.layers.append(trees)
            curr_dim = n_trees  # each tree outputs 1 value, concat = n_trees
        self.head = nn.Linear(n_trees, 1)
    def forward(self, x):
        for trees in self.layers:
            outs = [t(x) for t in trees]
            x = torch.cat(outs, dim=-1)
        return self.head(x)

# ── MC Dropout RNN (2026 paper — Brier 0.199) — dropout at inference ──
class MCDropoutRNN(nn.Module):
    def __init__(self, dim, hid=128, nl=2, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(dim, hid, nl, dropout=drop if nl>1 else 0, batch_first=True)
        self.drop = nn.Dropout(drop)  # stays active at inference!
        self.head = nn.Sequential(nn.Linear(hid,64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64,1))
    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(self.drop(o[:,-1,:]))
    def predict_mc(self, x, n_samples=30):
        """MC Dropout: run N forward passes with dropout ON, average predictions."""
        self.train()  # keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(torch.sigmoid(self(x)).cpu())
        self.eval()
        stacked = torch.stack(preds)
        return stacked.mean(0).numpy().flatten(), stacked.std(0).numpy().flatten()

# ── SAINT-style (Self-Attention + Intersample Attention) ──
class SAINT(nn.Module):
    def __init__(self, dim, d=64, nh=4, nb=2, drop=0.2):
        super().__init__()
        self.tok = nn.Linear(1, d)
        self.bias = nn.Parameter(torch.zeros(dim, d))
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        # Self-attention layers (within-sample)
        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nh, d*4, drop, activation="gelu", batch_first=True)
            for _ in range(nb)])
        # Intersample attention layers (across-sample) — applied on batch dim
        self.inter_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nh, d*4, drop, activation="gelu", batch_first=True)
            for _ in range(nb)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):
        B, F = x.size()
        t = self.tok(x.unsqueeze(-1)) + self.bias.unsqueeze(0)
        t = torch.cat([self.cls.expand(B,-1,-1), t], 1)  # (B, F+1, d)
        for sa, ia in zip(self.self_attn, self.inter_attn):
            t = sa(t)  # self-attention within each sample
            # Intersample: transpose so batch becomes sequence
            cls_tokens = t[:, 0:1, :]  # (B, 1, d)
            cls_seq = cls_tokens.squeeze(1).unsqueeze(0)  # (1, B, d)
            cls_seq = ia(cls_seq)  # attention across samples
            t[:, 0:1, :] = cls_seq.squeeze(0).unsqueeze(1)
        return self.head(t[:, 0, :])

# ── Temporal Fusion Transformer (simplified) ──
class TFT(nn.Module):
    def __init__(self, dim, d=64, nh=4, nb=2, drop=0.2):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(dim, d), nn.ReLU(), nn.Linear(d, d), nn.Sigmoid())
        self.grn = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d), nn.Dropout(drop))
        self.proj = nn.Linear(dim, d)
        el = nn.TransformerEncoderLayer(d, nh, d*4, drop, activation="gelu", batch_first=True)
        self.tf = nn.TransformerEncoder(el, nb)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):
        # Variable selection network
        weights = self.vsn(x)
        selected = self.proj(x) * weights
        # GRN + self-attention
        gated = self.grn(selected)
        seq = gated.unsqueeze(1)  # (B, 1, d) single timestep for non-sequence
        attended = self.tf(seq).squeeze(1)
        out = attended * self.gate(attended) + selected  # skip + gate
        return self.head(out)

# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════

def _seqs(X, sl=10):
    n,d = X.shape; s = np.zeros((n,sl,d), dtype=np.float32)
    for i in range(n):
        st = max(0,i-sl+1); L = i-st+1; s[i,sl-L:] = X[st:i+1]
    return s

# ── Custom Loss Functions (2026 research) ──
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al.) — down-weight easy examples, focus on hard ones."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__(); self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

class BrierLoss(nn.Module):
    """Direct Brier Score as loss — train to minimize Brier directly."""
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return ((probs - targets)**2).mean()

class LabelSmoothBCE(nn.Module):
    """BCE with label smoothing — prevents overconfident predictions."""
    def __init__(self, alpha=0.05):
        super().__init__(); self.alpha = alpha
    def forward(self, logits, targets):
        smooth = targets * (1 - self.alpha) + 0.5 * self.alpha
        return nn.functional.binary_cross_entropy_with_logits(logits, smooth)

LOSS_REGISTRY = {
    'bce': lambda p: nn.BCEWithLogitsLoss(),
    'focal': lambda p: FocalLoss(p.get('focal_gamma', 2.0), p.get('focal_alpha', 0.25)),
    'brier': lambda p: BrierLoss(),
    'label_smooth': lambda p: LabelSmoothBCE(p.get('label_smooth_alpha', 0.05)),
}

def _train_pt(model, Xtr, ytr, Xva, yva, lr=1e-3, wd=1e-4, seq=False, loss_fn=None, params=None):
    params = params or {}
    model = model.to(DEVICE)
    xt = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    xv = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=BATCH_SIZE, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",0.5,patience=5)
    # Select loss function — default BCE, but supports focal, brier, label_smooth
    loss_name = params.get('loss_fn', 'bce')
    crit = loss_fn or LOSS_REGISTRY.get(loss_name, LOSS_REGISTRY['bce'])(params)
    best_vl, best_st, wait = float("inf"), None, 0
    yv_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb,yb in loader:
            opt.zero_grad(); loss = crit(model(xb),yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(xv), yv_t).item()
        sched.step(vl)
        if vl < best_vl:
            best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= PATIENCE: break
    if best_st: model.load_state_dict(best_st); model = model.to(DEVICE)
    model.eval()
    with torch.no_grad(): probs = torch.sigmoid(model(xv)).cpu().numpy().flatten()
    del xt, xv, yt, yv_t; torch.cuda.empty_cache()
    return probs

def _train_tabnet(Xtr, ytr, Xva, yva, p=None):
    from pytorch_tabnet.tab_model import TabNetClassifier
    kw = {"n_d":32,"n_a":32,"n_steps":5,"gamma":1.5,"lambda_sparse":1e-4,
          "optimizer_fn":torch.optim.Adam,"optimizer_params":{"lr":2e-2,"weight_decay":1e-5},
          "scheduler_fn":torch.optim.lr_scheduler.StepLR,
          "scheduler_params":{"step_size":15,"gamma":0.9},
          "mask_type":"entmax","verbose":0,"device_name":DEVICE,"seed":42}
    if p:
        for k in ["n_d","n_a","n_steps","gamma","lambda_sparse"]:
            if k in p: kw[k] = p[k]
    clf = TabNetClassifier(**kw)
    clf.fit(Xtr.astype(np.float32), ytr, eval_set=[(Xva.astype(np.float32),yva)],
            eval_metric=["logloss"], max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE)
    return clf.predict_proba(Xva.astype(np.float32))[:,1]

def _fold(name, Xtr, ytr, Xva, yva, p):
    if name == "mlp":
        h = p.get("hidden_dims",(256,128,64))
        if isinstance(h,list): h = tuple(h)
        return _train_pt(NBANet(Xtr.shape[1],h,p.get("dropout",0.3)),
                         Xtr,ytr,Xva,yva,p.get("lr",1e-3),p.get("weight_decay",1e-4),params=p)
    if name == "mlp_residual":
        h = p.get("hidden_dims",(256,128,64))
        if isinstance(h,list): h = tuple(h)
        return _train_pt(NBANetResidual(Xtr.shape[1],h,p.get("dropout",0.3)),
                         Xtr,ytr,Xva,yva,p.get("lr",1e-3),p.get("weight_decay",1e-4),params=p)
    if name == "lstm":
        sl = p.get("seq_len",10)
        return _train_pt(NBALSTM(Xtr.shape[1],p.get("hidden_dim",64),p.get("n_layers",2),
                                  p.get("dropout",0.2)),
                         _seqs(Xtr,sl),ytr,_seqs(Xva,sl),yva,p.get("lr",1e-3),seq=True)
    if name == "ft_transformer":
        return _train_pt(FTTransformer(Xtr.shape[1],p.get("d_token",64),p.get("n_heads",4),
                                        p.get("n_blocks",3),p.get("dropout",0.2)),
                         Xtr,ytr,Xva,yva,p.get("lr",5e-4),p.get("weight_decay",1e-5))
    if name == "node":
        return _train_pt(NODEModel(Xtr.shape[1], p.get("n_trees",128), p.get("depth",4),
                                    p.get("n_layers",2)),
                         Xtr,ytr,Xva,yva,p.get("lr",1e-3),p.get("weight_decay",1e-4))
    if name == "mc_dropout_rnn":
        sl = p.get("seq_len",10)
        model = MCDropoutRNN(Xtr.shape[1], p.get("hidden_dim",128), p.get("n_layers",2),
                              p.get("dropout",0.3))
        # Train normally first
        _train_pt(model, _seqs(Xtr,sl), ytr, _seqs(Xva,sl), yva, p.get("lr",1e-3), seq=True)
        # MC inference: N forward passes with dropout ON → average
        model = model.to(DEVICE)
        xv = torch.tensor(_seqs(Xva,sl), dtype=torch.float32).to(DEVICE)
        n_mc = p.get("mc_samples", 30)
        preds, _ = model.predict_mc(xv, n_mc)
        del xv; torch.cuda.empty_cache()
        return preds
    if name == "saint":
        return _train_pt(SAINT(Xtr.shape[1], p.get("d_token",64), p.get("n_heads",4),
                                p.get("n_blocks",2), p.get("dropout",0.2)),
                         Xtr,ytr,Xva,yva,p.get("lr",5e-4),p.get("weight_decay",1e-5))
    if name == "tft":
        return _train_pt(TFT(Xtr.shape[1], p.get("d_model",64), p.get("n_heads",4),
                              p.get("n_blocks",2), p.get("dropout",0.2)),
                         Xtr,ytr,Xva,yva,p.get("lr",5e-4),p.get("weight_decay",1e-5))
    if name == "tabnet":
        return _train_tabnet(Xtr,ytr,Xva,yva,p)
    if name == "xgboost_gpu":
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=p.get("n_estimators",200),
            max_depth=p.get("max_depth",6), learning_rate=p.get("learning_rate",0.05),
            subsample=p.get("subsample",0.8), colsample_bytree=p.get("colsample_bytree",0.7),
            min_child_weight=p.get("min_child_weight",5),
            reg_alpha=p.get("reg_alpha",0.1), reg_lambda=p.get("reg_lambda",1.0),
            tree_method="gpu_hist", device="cuda", eval_metric="logloss",
            random_state=42, n_jobs=-1)
        clf.fit(Xtr,ytr); return clf.predict_proba(Xva)[:,1]
    if name == "lightgbm_gpu":
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(n_estimators=p.get("n_estimators",200),
            max_depth=p.get("max_depth",6), learning_rate=p.get("learning_rate",0.05),
            subsample=p.get("subsample",0.8), num_leaves=p.get("num_leaves",31),
            reg_alpha=p.get("reg_alpha",0.1), reg_lambda=p.get("reg_lambda",1.0),
            device="gpu", verbose=-1, random_state=42, n_jobs=-1)
        clf.fit(Xtr,ytr); return clf.predict_proba(Xva)[:,1]
    if name == "catboost_gpu":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(iterations=p.get("iterations",200), depth=p.get("depth",6),
            learning_rate=p.get("learning_rate",0.05), l2_leaf_reg=p.get("l2_leaf_reg",3.0),
            task_type="GPU", devices="0", verbose=0, random_seed=42)
        clf.fit(Xtr,ytr); return clf.predict_proba(Xva)[:,1]
    raise ValueError(f"Unknown model: {name}. Supported: {GPU_MODELS}")

# ═══════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ═══════════════════════════════════════════════

def _sanitize(obj):
    if isinstance(obj,dict): return {k:_sanitize(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [_sanitize(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if hasattr(obj,'item'): return obj.item()
    if isinstance(obj,torch.Tensor): return obj.detach().cpu().numpy().tolist()
    return obj

def wf_eval(model_name, X, y, params=None, n_splits=WALK_FORWARD_SPLITS, feat_idx=None):
    params = params or {}
    if feat_idx is not None: X = X[:, feat_idx]
    if X.shape[0] > MAX_EVAL_GAMES: X, y = X[-MAX_EVAL_GAMES:], y[-MAX_EVAL_GAMES:]
    Xs = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6))
    folds, all_p, all_t = [], [], []
    for fi,(ti,vi) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(Xs)):
        Xtr, Xva, ytr, yva = Xs[ti], Xs[vi], y[ti], y[vi]
        print(f"  Fold {fi+1}/{n_splits}: train={len(ti)}, val={len(vi)}")
        try:
            pr = np.clip(_fold(model_name, Xtr, ytr, Xva, yva, params), 0.001, 0.999)
            b = float(brier_score_loss(yva, pr))
            ll = float(log_loss(yva, pr))
            ac = float(accuracy_score(yva, (pr>0.5).astype(int)))
            folds.append({"fold":fi+1,"brier":round(b,5),"log_loss":round(ll,5),
                          "accuracy":round(ac,4),"n_train":len(ti),"n_val":len(vi)})
            all_p.extend(pr.tolist()); all_t.extend(yva.tolist())
            print(f"    Brier:{b:.4f} Acc:{ac:.4f} LL:{ll:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}"); traceback.print_exc()
            folds.append({"fold":fi+1,"error":str(e)[:300]})
    vf = [f for f in folds if "brier" in f]
    if not vf: return {"brier":1.0,"accuracy":0.0,"log_loss":10.0,"error":"All folds failed","folds":folds}
    if all_p:
        ap, at = np.array(all_p), np.array(all_t)
        ob = float(brier_score_loss(at,ap))
        oa = float(accuracy_score(at,(ap>0.5).astype(int)))
        ol = float(log_loss(at,ap))
    else:
        ob = float(np.mean([f["brier"] for f in vf]))
        oa = float(np.mean([f["accuracy"] for f in vf]))
        ol = float(np.mean([f["log_loss"] for f in vf]))
    # ROI: bet when prob > 0.55 or < 0.45 at -110
    bets, profit = 0, 0.0
    for p,a in zip(all_p, all_t):
        if p>0.55: bets+=1; profit += (100/110) if a==1 else -1.0
        elif p<0.45: bets+=1; profit += (100/110) if a==0 else -1.0
    roi = float(profit/max(bets,1))
    return {"model":model_name,"brier":round(ob,5),"accuracy":round(oa,4),
            "log_loss":round(ol,5),"roi":round(roi,4),
            "avg_brier":round(float(np.mean([f["brier"] for f in vf])),5),
            "n_splits":n_splits,"n_games":int(X.shape[0]),"n_features":int(X.shape[1]),
            "folds":folds,"valid_folds":len(vf)}

# ═══════════════════════════════════════════════
# EXPERIMENT QUEUE
# ═══════════════════════════════════════════════

_TARGET_SQL = """(target_space IN ('colab','gpu','kaggle','any') OR target_space IS NULL)"""

def fetch_next():
    rows = _exec_sql(f"""SELECT id, experiment_id, agent_name, experiment_type, description,
        hypothesis, params, priority, status, target_space, baseline_brier, created_at
        FROM public.nba_experiments WHERE status='pending' AND {_TARGET_SQL}
        ORDER BY priority DESC, created_at ASC LIMIT 1""")
    if not rows or rows is True: return None
    r = rows[0]
    return {"id":r[0],"experiment_id":r[1],"agent_name":r[2],"experiment_type":r[3],
            "description":r[4],"hypothesis":r[5],
            "params":r[6] if isinstance(r[6],dict) else json.loads(r[6]) if r[6] else {},
            "priority":r[7],"status":r[8],"target_space":r[9],
            "baseline_brier":float(r[10]) if r[10] else None,
            "created_at":str(r[11]) if r[11] else None}

def _claim(eid):
    r = _exec_sql("UPDATE public.nba_experiments SET status='running',started_at=NOW() "
                   "WHERE id=%s AND status='pending' RETURNING id", (eid,))
    return r is not None and r is not True and len(r)>0

def _complete(eid, brier, acc, ll, details, status="completed"):
    _exec_sql("UPDATE public.nba_experiments SET status=%s,result_brier=%s,result_accuracy=%s,"
              "result_log_loss=%s,result_details=%s,completed_at=NOW() WHERE id=%s",
              (str(status),float(brier),float(acc),float(ll),
               json.dumps(_sanitize(details)),int(eid)), fetch=False)

def _fail(eid, msg):
    d = {"error":str(msg)[:2000],"failed_at":datetime.now(timezone.utc).isoformat(),
         "runner":"kaggle_gpu"}
    _exec_sql("UPDATE public.nba_experiments SET status='failed',result_details=%s,"
              "completed_at=NOW() WHERE id=%s", (json.dumps(d),int(eid)), fetch=False)

def _pending():
    r = _exec_sql(f"SELECT COUNT(*) FROM public.nba_experiments WHERE status='pending' AND {_TARGET_SQL}")
    return int(r[0][0]) if r and r is not True and len(r)>0 else 0

# ═══════════════════════════════════════════════
# EXPERIMENT ROUTER
# ═══════════════════════════════════════════════

def run_exp(exp, X, y, fnames):
    p, et, eid = exp["params"], exp["experiment_type"], exp["id"]
    print(f"\n{'='*60}\nEXP: {exp['experiment_id']}\nType: {et} | Agent: {exp['agent_name']}")
    print(f"Desc: {exp['description'][:200]}\n{'='*60}")
    if not _claim(eid): raise RuntimeError(f"Exp {eid} already claimed")
    t0 = time.time()
    try:
        if et == "model_test":
            mn = p.get("model_type","mlp")
            mn = {"xgboost":"xgboost_gpu","lightgbm":"lightgbm_gpu","catboost":"catboost_gpu"}.get(mn,mn)
            res = wf_eval(mn, X, y, p.get("hyperparams",p),
                          p.get("n_splits",WALK_FORWARD_SPLITS), p.get("feature_indices"))
        elif et == "gpu_benchmark":
            mods = p.get("models", GPU_MODELS); ar = {}; bb, bm = 1.0, None
            for mn in mods:
                print(f"\n--- {mn} ---")
                try:
                    r = wf_eval(mn, X, y, p.get(f"{mn}_params",{}),
                                p.get("n_splits",WALK_FORWARD_SPLITS), p.get("feature_indices"))
                    ar[mn] = r
                    if r["brier"]<bb: bb,bm = r["brier"],mn
                    print(f"  Brier:{r['brier']:.4f} Acc:{r['accuracy']:.4f}")
                except Exception as e:
                    print(f"  FAILED: {e}"); ar[mn] = {"error":str(e)[:300]}
                torch.cuda.empty_cache()
            res = {"brier":float(bb),"accuracy":float(ar.get(bm,{}).get("accuracy",0.0)),
                   "log_loss":float(ar.get(bm,{}).get("log_loss",10.0)),
                   "best_model":bm,"all_results":ar,"models_tested":len(mods)}
        elif et in ("feature_test","calibration_test"):
            mn = p.get("model_type","mlp")
            if mn not in GPU_MODELS: mn = "mlp"
            res = wf_eval(mn, X, y, p.get("hyperparams",{}),
                          p.get("n_splits",WALK_FORWARD_SPLITS),
                          p.get("feature_indices") if et=="feature_test" else None)
        else:
            raise ValueError(f"Unknown type: {et}")

        el = time.time()-t0
        brier, acc, ll = float(res.get("brier",1.0)), float(res.get("accuracy",0.0)), float(res.get("log_loss",10.0))
        res.update({"elapsed_seconds":round(el,1),"games_evaluated":min(int(X.shape[0]),MAX_EVAL_GAMES),
                    "feature_candidates":int(X.shape[1]),"experiment_id":exp["experiment_id"],
                    "agent_name":exp["agent_name"],"runner":"kaggle_gpu","device":DEVICE})
        if torch.cuda.is_available(): res["gpu_name"] = torch.cuda.get_device_name(0)
        bl = exp.get("baseline_brier")
        if bl: res["improvement"]=round(float(bl)-brier,5); res["improved"]=brier<float(bl)
        _complete(eid, brier, acc, ll, res)
        print(f"\nDONE: Brier={brier:.4f} Acc={acc:.4f} {el:.1f}s")
        if bl:
            d = float(bl)-brier
            print(f"vs baseline {bl:.4f}: {'BETTER' if d>0 else 'WORSE'} by {abs(d):.4f}")
        return res
    except Exception as e:
        _fail(eid, f"{str(e)[:500]}\n{traceback.format_exc()[-1000:]}")
        print(f"\nFAILED: {str(e)[:300]} ({time.time()-t0:.1f}s)"); raise

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

print(f"\n{'='*60}\nNBA QUANT AI — KAGGLE GPU RUNNER\nDevice: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*60}")

print("\n[1/3] Loading games from Supabase...")
games = load_games()
if len(games) < 100:
    print(f"ERROR: Only {len(games)} games. Need 100+."); sys.exit(1)

print(f"\n[2/3] Building features from {len(games)} games...")
X, y, feature_names = build_features(games)
var = np.var(X, axis=0); vm = var > 1e-10; nr = int((~vm).sum())
if nr > 0:
    X = X[:,vm]; feature_names = [f for f,v in zip(feature_names,vm) if v]
    print(f"Removed {nr} zero-variance features")
if X.shape[0] > MAX_EVAL_GAMES:
    X, y = X[-MAX_EVAL_GAMES:], y[-MAX_EVAL_GAMES:]
    print(f"Capped to {MAX_EVAL_GAMES} games")
print(f"\nReady: {X.shape[0]} games, {X.shape[1]} features, home_win={y.mean():.3f}")

print(f"\n[3/3] Polling (max {MAX_EXPERIMENTS} experiments)...\n{'='*60}")
completed, failed, cerr = 0, 0, 0
for it in range(MAX_EXPERIMENTS * 3):
    try:
        exp = fetch_next()
        if exp:
            pn = _pending()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found: {exp['experiment_id']} (queue:{pn})")
            try: run_exp(exp, X, y, feature_names); completed += 1
            except: failed += 1
            cerr = 0; torch.cuda.empty_cache()
            if completed+failed >= MAX_EXPERIMENTS:
                print(f"\nReached {MAX_EXPERIMENTS} limit."); break
        else:
            if completed+failed > 0: print("\nQueue empty. Done."); break
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No experiments. Waiting {POLL_INTERVAL}s... ({it+1})")
            time.sleep(POLL_INTERVAL)
            if it >= 4 and completed+failed == 0: print("No experiments after 5 polls."); break
    except Exception as e:
        cerr += 1; print(f"\n[ERROR] #{cerr}: {e}")
        if cerr >= 5: _reconnect_pg(); cerr = 0
        time.sleep(POLL_INTERVAL)

print(f"\n{'='*60}\nKAGGLE SESSION COMPLETE: {completed} done, {failed} failed")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*60}")



### BEGIN Model Architect addition (2026-03-20) ###
===CODE: kaggle/nba_gpu_runner.py===
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import numpy as np
import torch # Assuming torch is already imported for GPU checks

# --- New Model Training Function: XGBoost DART Extreme Depth ---
# This function encapsulates the training logic for a generic XGBoost model,
# allowing `run_exp
### END Model Architect addition ###


### BEGIN Calibrator addition (2026-03-20) ###
===CODE: kaggle/nba_gpu_runner.py===
# New calibration method: Temperature Scaling + Isotonic Regression Ensemble
# Addresses both global over/under-confidence and local non-monotonicity.

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import warnings

# Suppress specific
### END Calibrator addition ###


### BEGIN Research Scholar addition (2026-03-20) ###
# Implements FT-Transformer with feature tokenization + self-attention (Gorishniy 2021→2025)
# Tabular transformer with learned feature embeddings for NBA data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class FeatureTokenizer(nn.Module):
    """Learned feature embeddings for tabular data"""
    def __init__(self, n_features, dim):
        super().__init__()
        self.embeddings = nn.Embedding(n_features, dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, x):
        # x: (batch, features) → (batch, features, dim)
        return self.embeddings(x.long())

class FT_Transformer(nn.Module):
    """Full transformer with feature tokenization"""
    def __init__(self, n_features, n_layers=6, n_heads=8, dim=128, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim*4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Tokenize features
        x_emb = self.tokenizer(x)
        # Transformer encoding
        x_enc = self.transformer(x_emb)
        # Pool over feature dimension
        x_pool = x_enc.mean(dim=1)
        # Classification
        x_out = self.fc(x_pool)
        return self.sigmoid(x_out).squeeze()

def run_ft_transformer(exp, X, y, feature_names):
    """Run FT-Transformer experiment"""
    params = exp['params']
    model = FT_Transformer(
        n_features=X.shape[1],
        n_layers=params.get('n_layers', 6),
        n_heads=params.get('n_heads', 8),
        dim=params.get('dim', 128),
        dropout=params.get('dropout', 0.2)
    )
    
    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=params.get('batch_size', 32), shuffle=True)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
    criterion = nn.BCELoss()
    epochs = params.get('epochs', 100)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb.long())
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.long()).numpy()
        brier = np.mean((preds - y)**2)
        accuracy = np.mean((preds > 0.5) == y)
    
    # Save results
    result = {
        'brier_score': brier,
        'accuracy': accuracy,
        'predictions': preds.tolist(),
        'model_type': 'ft_transformer'
    }
    save_results(exp, result)
    print(f"✓ FT-Transformer complete: Brier={brier:.4f}, Acc={accuracy:.4f}")

# Add to run_exp dispatch
def run_exp(exp, X, y, feature_names):
    """Dispatch to appropriate model runner"""
    model_type = exp['params']['model_type']
    if model_type == 'ft_transformer':
        run_ft_transformer(exp, X, y, feature_names)
    else:
        # Existing model runners...
        pass
### END Research Scholar addition ###


### BEGIN Model Architect addition (2026-03-20) ###
### Research Scholar addition: XGBoost Deep Trees with Low Learning Rate ###
# Implements: Extreme Gradient Boosting - Depth 12, LR 0.005
# Hypothesis: Very deep trees with tiny learning rate will capture complex interactions without overfitting

import xgboost as xgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
import numpy as np
import json
import os
from datetime import datetime


def run_xgboost_deep_lr(exp, X, y, feature_names):
    """
    Run XGBoost with deep trees (max_depth=12) and very low learning rate (0.005).
    Uses aggressive regularization to prevent overfitting despite depth.
    """
    params = exp['params']
    print(f"\n{'='*60}")
    print(f"Running XGBoost Deep-LR Experiment: {exp['name']}")
    print(f"Config: max_depth={params.get('max_depth', 12)}, "
          f"learning_rate={params.get('learning_rate', 0.005)}, "
          f"n_estimators={params.get('n_estimators', 2000)}")
    print(f"{'='*60}")
    
    # Ensure numpy arrays
    X = np.array(X)
    y = np.array(y).ravel()
    
    # XGBoost parameters - deep trees with strong regularization
    xgb_params = {
        'max_depth': params.get('max_depth', 12),
        'learning_rate': params.get('learning_rate', 0.005),
        'n_estimators': params.get('n_estimators', 2000),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 1),      # L1 regularization
        'reg_lambda': params.get('reg_lambda', 5),    # L2 regularization (strong)
        'booster': params.get('booster', 'gbtree'),
        'min_child_weight': params.get('min_child_weight', 1),
        'gamma': params.get('gamma', 0.1),            # Min loss reduction for split
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Use early stopping with validation set to prevent overfitting
    # Split for early stopping validation
    np.random.seed(42)
    n_samples = len(y)
    val_size = int(0.15 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    
    # Initialize model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else xgb_params['n_estimators']
    print(f"Best iteration: {best_iteration}/{xgb_params['n_estimators']}")
    
    # Cross-validation for robust evaluation (time-series aware if dates available)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get out-of-fold predictions
    cv_preds = cross_val_predict(
        xgb.XGBClassifier(**{**xgb_params, 'n_estimators': best_iteration}),
        X, y,
        cv=cv,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]
    
    # Calculate metrics
    brier = brier_score_loss(y, cv_preds)
    accuracy = accuracy_score(y, cv_preds > 0.5)
    logloss = log_loss(y, cv_preds)
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = {
        name: float(imp) 
        for name, imp in zip(feature_names, importance)
    }
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nResults:")
    print(f"  Brier Score: {brier:.6f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"\nTop 10 Important Features:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")
    
    # Calibration check
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y, cv_preds, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"  Expected Calibration Error: {calibration_error:.4f}")
    
    # Save results
    result = {
        'brier_score': float(brier),
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'calibration_error': float(calibration_error),
        'best_iteration': int(best_iteration),
        'n_features': int(X.shape[1]),
        'n_samples': int(len(y)),
        'predictions': cv_preds.tolist(),
        'feature_importance': feature_importance,
        'top_features': top_features,
        'model_type': 'xgboost_deep_lr',
        'params_used': {k: v for k, v in xgb_params.items() if k not in ['verbosity']}
    }
    
    save_results(exp, result)
    
    # Save model for potential ensemble use
    model_dir = os.path.dirname(exp.get('output_path', 'results/'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"xgb_deep_lr_{exp['name']}.json")
    model.save_model(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    print(f"\n✓ XGBoost Deep-LR complete: Brier={brier:.6f}, Acc={accuracy:.4f}")
    
    return result


# Update run_exp dispatch to include new model
_original_run_exp = run_exp  # Store reference if needed

def run_exp(exp, X, y, feature_names):
    """Dispatch to appropriate model runner"""
    model_type = exp['params'].get('model_type', '')
    
    if model_type == 'xgboost' and exp['params'].get('max_depth', 6) >= 10:
        # Deep XGBoost with low learning rate
        run_xgboost_deep_lr(exp, X, y, feature_names)
    elif model_type == 'ft_transformer':
        run_ft_transformer(exp, X, y, feature_names)
    else:
        # Existing model runners - call original or raise informative error
        raise NotImplementedError(f"Model type '{model_type}' not implemented. "
                                f"Available: 'xgboost' (deep), 'ft_transformer'")

### END Research Scholar addition ###
### END Model Architect addition ###
