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
for pkg in ["psycopg2-binary", "pytorch-tabnet", "scikit-learn>=1.3"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

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
# DATA LOADING
# ═══════════════════════════════════════════════

def load_games_from_supabase(limit=15000) -> List[dict]:
    for table in ["nba_games", "nba_historical_games", "games"]:
        rows = _exec_sql(f"SELECT data FROM public.{table} ORDER BY id ASC LIMIT %s",
                         (limit,))
        if rows and rows is not True and len(rows) > 0:
            games = [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]
            print(f"Loaded {len(games)} games from {table}")
            return games
    rows = _exec_sql("""SELECT game_date, home_team, away_team, home_pts, away_pts,
        home_stats, away_stats FROM public.nba_game_results ORDER BY game_date ASC
        LIMIT %s""", (limit,))
    if rows and rows is not True and len(rows) > 0:
        games = [{"game_date": str(r[0]), "home_team": r[1], "away_team": r[2],
                  "home": {"team_name": r[1], "pts": r[3], **(r[5] or {})},
                  "away": {"team_name": r[2], "pts": r[4], **(r[6] or {})}} for r in rows]
        print(f"Loaded {len(games)} games from nba_game_results")
        return games
    print("[WARN] No games found"); return []

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
games = load_games_from_supabase()
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
# Implements XGBoost DART model with extreme regularization and low learning rate for fine-grained probability calibration
import xgboost as xgb
from sklearn.metrics import brier_score_loss

class XGBoostDARTModel:
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return brier_score_loss(y, y_pred)

def run_xgboost_dart_exp(exp, X, y, feature_names):
    """
    Run XGBoost DART experiment
    """
    params = {
        "model_type": "xgboost",
        "booster": "dart",
        "max_depth": 6,
        "learning_rate": 0.005,
        "n_estimators": 3000,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
        "reg_alpha": 1,
        "reg_lambda": 10,
        "min_child_weight": 7,
        "gamma": 1,
        "sample_type": "weighted",
        "normalize_type": "tree",
        "rate_drop": 0.1,
        "skip_drop": 0.5,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42
    }
    model = XGBoostDARTModel(params)
    model.train(X, y)
    y_pred = model.predict(X)
    brier_loss = brier_score_loss(y, y_pred)
    print(f"XGBoost DART Brier Loss: {brier_loss:.4f}")

# Add this to the experiment loop
def run_exp(exp, X, y, feature_names):
    if exp['model_type'] == 'xgboost_dart':
        run_xgboost_dart_exp(exp, X, y, feature_names)
    else:
        # existing experiment code
        pass
### END Model Architect addition ###


### BEGIN Model Architect addition (2026-03-20) ###
===CODE: kaggle/nba_gpu_runner.py===
### BEGIN Model Architect addition (2024-07-20) ###
# Implements TabNet model with increased attention dimensions, more steps, and Entmax masking
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import brier_score_loss
from torch.optim import AdamW

class TabNet
### END Model Architect addition ###


### BEGIN Calibrator addition (2026-03-20) ###
# Implements Platt Scaling + Temperature Scaling ensemble for calibration
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

class PlattTemperatureScalingEnsemble:
    def __init__(self, params):
        self.params = params
        self.platt_scaler = None
        self.temperature_scaler = None
        self.ensemble_method = params["ensemble_method"]
        self.weights = params["weights"]

    def fit(self, X, y):
        platt_scaler = CalibratedClassifierCV(method='sigmoid', cv=5)
        platt_scaler.fit(X, y)

        temperature_scaler = CalibratedClassifierCV(method='isotonic', cv=5)
        temperature_scaler.fit(X, y)

        self.platt_scaler = platt_scaler
        self.temperature_scaler = temperature_scaler

    def predict_proba(self, X):
        platt_pred = self.platt_scaler.predict_proba(X)[:, 1]
        temperature_pred = self.temperature_scaler.predict_proba(X)[:, 1]

        if self.ensemble_method == "weighted_average":
            return self.weights[0] * platt_pred + self.weights[1] * temperature_pred
        else:
            raise ValueError("Invalid ensemble method")

    def evaluate(self, X, y):
        y_pred = self.predict_proba(X)
        return brier_score_loss(y, y_pred)

def run_platt_temperature_scaling_ensemble_exp(exp, X, y, feature_names):
    """
    Run Platt Scaling + Temperature Scaling ensemble experiment
    """
    params = {
        "calibration_method": ["platt", "temperature"],
        "platt_C": [0.1, 1, 10],
        "temperature_T": [0.7, 1, 1.3],
        "ensemble_method": "weighted_average",
        "weights": [0.5, 0.5],
        "cv_folds": 5
    }
    model = PlattTemperatureScalingEnsemble(params)
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    brier_loss = model.evaluate(X, y)
    print(f"Platt Temperature Scaling Ensemble Brier Loss: {brier_loss:.4f}")

# Add this to the experiment loop
def run_exp(exp, X, y, feature_names):
    if exp['model_type'] == 'platt_temperature_scaling_ensemble':
        run_platt_temperature_scaling_ensemble_exp(exp, X, y, feature_names)
    else:
        # existing experiment code
        pass
### END Calibrator addition ###


### BEGIN Model Architect addition (2026-03-20) ###
### BEGIN Model Architect addition (2026-03-20) ###
# Implements XGBoost with extreme depth (16) and aggressive regularization
# Hypothesis: Very deep trees with strong regularization can capture complex interactions without overfitting
import xgboost as xgb
from sklearn.metrics import brier_score_loss


class XGBoostExtremeDepthModel:
    """
    XGBoost model with extreme tree depth and aggressive regularization.
    Designed to capture complex feature interactions while preventing overfitting
    through strong L1/L2 regularization, high min_child_weight, and subsampling.
    """
    
    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.model = None
        
    def _default_params(self):
        return {
            "booster": "gbtree",
            "max_depth": 16,
            "learning_rate": 0.01,
            "n_estimators": 1500,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1,           # L1 regularization
            "reg_lambda": 5,          # L2 regularization (aggressive)
            "min_child_weight": 3,    # Prevent overfitting on small samples
            "gamma": 0.5,             # Minimum loss reduction for split
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",    # GPU-compatible histogram method
            "random_state": 42,
            "n_jobs": -1
        }
    
    def train(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the XGBoost model with optional early stopping.
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds to wait before stopping if no improvement
            verbose: Whether to print training progress
        """
        self.model = xgb.XGBClassifier(**self.params)
        
        fit_params = {
            "verbose": verbose
        }
        
        if eval_set is not None and early_stopping_rounds is not None:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        """
        Predict class probabilities for binary classification.
        
        Returns:
            Probability of positive class (shape: (n_samples,))
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict_class(self, X):
        """
        Predict class labels (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model using Brier score (proper scoring rule for probabilities).
        Lower is better.
        """
        y_pred = self.predict(X)
        return brier_score_loss(y, y_pred)
    
    def feature_importances(self, importance_type='gain'):
        """
        Get feature importances from the trained model.
        
        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.get_booster().get_score(importance_type=importance_type)
    
    def get_params(self):
        """Get current model parameters."""
        return self.params.copy()


def run_xgboost_extreme_depth_exp(exp, X, y, feature_names, X_val=None, y_val=None):
    """
    Run XGBoost extreme depth experiment with aggressive regularization.
    
    Args:
        exp: Experiment configuration dict
        X: Training features
        y: Training targets
        feature_names: List of feature names
        X_val: Optional validation features
        y_val: Optional validation targets
    
    Returns:
        dict with model, predictions, and metrics
    """
    print("=" * 60)
    print("Running XGBoost Extreme Depth + Aggressive Regularization")
    print(f"Params: max_depth=16, learning_rate=0.01, n_estimators=1500")
    print(f"        reg_alpha=1, reg_lambda=5, min_child_weight=3, gamma=0.5")
    print("=" * 60)
    
    # Initialize model with experiment parameters
    model = XGBoostExtremeDepthModel(params=exp.get('params'))
    
    # Setup validation for early stopping if provided
    eval_set = None
    early_stopping_rounds = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        early_stopping_rounds = 50
        print(f"Using validation set with early stopping (patience={early_stopping_rounds})")
    
    # Train model
    model.train(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=True)
    
    # Generate predictions
    y_pred_train = model.predict(X)
    
    # Calculate metrics
    train_brier = brier_score_loss(y, y_pred_train)
    
    results = {
        'model': model,
        'y_pred_train': y_pred_train,
        'train_brier_loss': train_brier,
        'best_iteration': getattr(model.model, 'best_iteration', None),
        'n_estimators_used': getattr(model.model, 'n_estimators', 1500)
    }
    
    print(f"\nTraining Brier Loss: {train_brier:.6f}")
    
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        val_brier = brier_score_loss(y_val, y_pred_val)
        results['y_pred_val'] = y_pred_val
        results['val_brier_loss'] = val_brier
        print(f"Validation Brier Loss: {val_brier:.6f}")
        
        # Log if early stopping was triggered
        if hasattr(model.model, 'best_iteration') and model.model.best_iteration is not None:
            print(f"Early stopping triggered at iteration: {model.model.best_iteration}")
    
    # Feature importance summary
    try:
        importances = model.feature_importances(importance_type='gain')
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 features by gain:")
        for feat, score in top_features:
            idx = int(feat.replace('f', ''))
            feat_name = feature_names[idx] if idx < len(feature_names) else feat
            print(f"  {feat_name}: {score:.4f}")
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
    
    print("=" * 60)
    return results


# Update experiment dispatcher
def run_exp(exp, X, y, feature_names, X_val=None, y_val=None):
    """
    Main experiment runner dispatcher.
    """
    model_type = exp.get('model_type', '')
    
    if exp.get('model_type') == 'xgboost_dart':
        run_xgboost_dart_exp(exp, X, y, feature_names)
    elif model_type == 'xgboost_extreme_depth' or (
        model_type == 'xgboost' and 
        exp.get('params', {}).get('max_depth') == 16
    ):
        return run_xgboost_extreme_depth_exp(exp, X, y, feature_names, X_val, y_val)
    else:
        # existing experiment code
        pass
### END Model Architect addition ###
### END Model Architect addition ###


### BEGIN Research Scholar addition (2026-03-20) ###
```python
# FT-Transformer with Brier loss + Focal loss ensemble implementation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FTTransformerEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, params):
        self.params = params
        self.model = None
        self.feature_tokenization = params["feature_tokenization"]
        self.attention_heads = params["attention_heads"]
        self.brier_loss_weight = params["brier_loss_weight"]
        self.focal_loss_weight = params["focal_loss_weight"]
        self.focal_gamma = params["focal_gamma"]
        self.max_depth = params["max_depth"]
        self.learning_rate = params["learning_rate"]
        self.epochs = params["epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, input_dim):
        class FTTransformer(nn.Module):
            def __init__(self, input_dim, attention_heads, max_depth):
                super().__init__()
                self.token_embedding = nn.Linear(input_dim, 64) if self.feature_tokenization else nn.Identity()
                self.positional_encoding = nn.Embedding(max_depth, 64)
                self.transformer_blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=64, nhead=attention_heads, batch_first=True)
                    for _ in range(max_depth)
                ])
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                if self.feature_tokenization:
                    x = self.token_embedding(x)
                positional_enc = self.positional_encoding(torch.arange(x.size(1), device=x.device).unsqueeze(0))
                x = x + positional_enc
                for block in self.transformer_blocks:
                    x = block(x)
                x = self.fc(x.mean(dim=1))
                return torch.sigmoid(x)

        return FTTransformer(input_dim, self.attention_heads, self.max_depth).to(self.device)

    def brier_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def focal_loss(self, y_true, y_pred, gamma=2):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        return torch.mean((1 - pt) ** gamma * torch.log(pt))

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device).unsqueeze(1)

        self.model = self.build_model(X.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(batch_X).squeeze()
                loss = (self.brier_loss_weight * self.brier_loss(batch_y, y_pred) +
                        self.focal_loss_weight * self.focal_loss(batch_y, y_pred, self.focal_gamma))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step(total_loss)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_tensor).squeeze().cpu().numpy()
        return y_pred

    def evaluate(self, X, y):
        y_pred = self.predict_proba(X)
        return brier_score_loss(y, y_pred)

def run_ft_transformer_ensemble_exp(exp, X, y, feature_names):
    """
    Run FT-Transformer with Brier loss + Focal loss ensemble experiment
    """
    params = {
        "model_type": "ft_transformer",
        "feature_tokenization": True,
        "attention_heads": 8,
        "brier_loss_weight": 0.7,
        "focal_loss_weight": 0.3,
        "focal_gamma": 2,
        "max_depth": 6,
        "learning_rate": 0.001,
        "epochs": 200,
        "ensemble_method": "weighted_average",
        "weights": [0.5, 0.5],
        "cv_folds": 5
    }
    model = FTTransformerEnsemble(params)
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    brier_loss = model.evaluate(X, y)
    print(f"FT-Transformer Ensemble Brier Loss: {brier_loss:.4f}")

# Add this to the experiment loop
def run_exp(exp, X, y, feature_names):
    if exp['model_type'] == 'platt_temperature_scaling_ensemble':
        run_platt_temperature_scaling_ensemble_exp(exp, X, y, feature_names)
    elif exp['model_type'] == 'ft_transformer':
        run_ft_transformer_ensemble_exp(exp, X, y, feature_names)
    else:
        # existing experiment code
        pass
```
### END Research Scholar addition ###


### BEGIN Model Architect addition (2026-03-20) ###
# XGBoost with extreme depth and regularization
# Very deep trees with strong regularization to capture complex interactions without overfitting

import xgboost as xgb
from sklearn.metrics import brier_score_loss

class XGBoostExtremeDepth:
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)

        # Set XGBoost parameters
        xgb_params = {
            'max_depth': self.params['max_depth'],
            'learning_rate': self.params['learning_rate'],
            'n_estimators': self.params['n_estimators'],
            'subsample': self.params['subsample'],
            'colsample_bytree': self.params['colsample_bytree'],
            'reg_alpha': self.params['reg_alpha'],
            'reg_lambda': self.params['reg_lambda'],
            'min_child_weight': self.params['min_child_weight'],
            'gamma': self.params['gamma'],
            'booster': self.params['booster'],
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if 'gpu' in self.params else 'hist',
            'random_state': 42
        }

        # Train model
        self.model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            verbose_eval=100
        )

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def evaluate(self, X, y):
        y_pred = self.predict_proba(X)
        return brier_score_loss(y, y_pred)

# Add to experiment loop
def run_xgboost_extreme_depth_exp(exp, X, y, feature_names):
    params = {
        "model_type": "xgboost",
        "max_depth": 20,
        "learning_rate": 0.01,
        "n_estimators": 1500,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 1,
        "reg_lambda": 5,
        "min_child_weight": 5,
        "gamma": 0.5,
        "booster": "gbtree"
    }
    model = XGBoostExtremeDepth(params)
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    brier_loss = model.evaluate(X, y)
    print(f"XGBoost Extreme Depth Brier Loss: {brier_loss:.4f}")

# Add this to the experiment loop
def run_exp(exp, X, y, feature_names):
    if exp['model_type'] == 'platt_temperature_scaling_ensemble':
        run_platt_temperature_scaling_ensemble_exp(exp, X, y, feature_names)
    elif exp['model_type'] == 'ft_transformer':
        run_ft_transformer_ensemble_exp(exp, X, y, feature_names)
    elif exp['model_type'] == 'xgboost_extreme_depth':
        run_xgboost_extreme_depth_exp(exp, X, y, feature_names)
    else:
        # existing experiment code
        pass
### END Model Architect addition ###
