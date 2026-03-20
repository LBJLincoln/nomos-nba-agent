#!/usr/bin/env python3
"""
NBA Quant Feature Expansion — 2000+ Additional Features
=========================================================
Post-processing expansion module that takes the base ~2058 features from
NBAFeatureEngine and generates 2000+ additional derived features for the
genetic algorithm to select from.

New Feature Categories:
  A. Polynomial Interactions     (500+) — pairwise products of top 30 features + squares
  B. Ratio Features              (120+) — ratios between related stats
  C. Bayesian Features           (60+)  — Bayesian-updated priors for team strength
  D. Temporal Features           (120+) — EMA crossovers, momentum slopes, acceleration
  E. Market Microstructure II    (100+) — CLV decomposition, steam velocity
  F. Cross-Window Features       (200+) — differences & ratios between rolling windows
  G. Referee Advanced            (60+)  — referee-team interactions, pace impact
  H. Cluster Features            (50+)  — team archetype clusters, schedule difficulty

Total expansion: ~2000+ new features → combined ~4000+ for genetic selection.

THIS SCRIPT MUST RUN ON HF SPACES (16GB RAM) — NOT on VM.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict


# ══════════════════════════════════════════════════════════════════
# SAFE MATH UTILITIES
# ══════════════════════════════════════════════════════════════════

def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division with zero/nan protection."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(b) > 1e-8, a / b, fill)
    return np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Safe log transform: log(1 + |x|) * sign(x)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    """Safe square root: sqrt(|x|) * sign(x)."""
    return np.sign(x) * np.sqrt(np.abs(x))


def _clip_extreme(x: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Clip values beyond sigma standard deviations from mean."""
    if x.size == 0:
        return x
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-10:
        return x
    lo = mu - sigma * sd
    hi = mu + sigma * sd
    return np.clip(x, lo, hi)


def _stabilize(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf and clip extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip each column independently
    for j in range(X.shape[1]):
        X[:, j] = _clip_extreme(X[:, j], sigma=6.0)
    return X


# ══════════════════════════════════════════════════════════════════
# FEATURE NAME RESOLVER
# ══════════════════════════════════════════════════════════════════

class _NameIndex:
    """Fast name-to-column-index lookup."""

    def __init__(self, feature_names: List[str]):
        self._map = {name: i for i, name in enumerate(feature_names)}
        self._names = feature_names

    def idx(self, name: str) -> Optional[int]:
        return self._map.get(name)

    def col(self, X: np.ndarray, name: str) -> Optional[np.ndarray]:
        i = self._map.get(name)
        if i is not None and i < X.shape[1]:
            return X[:, i]
        return None

    def has(self, name: str) -> bool:
        return name in self._map


# ══════════════════════════════════════════════════════════════════
# TOP FEATURE DEFINITIONS (for polynomial / interaction expansion)
# ══════════════════════════════════════════════════════════════════

# Top 30 most predictive base features (empirically validated via GA)
TOP30_FEATURES = [
    "h_wp10", "a_wp10", "h_wp5", "a_wp5",
    "h_netrtg10", "a_netrtg10", "h_netrtg5", "a_netrtg5",
    "h_ortg10", "a_ortg10", "h_drtg10", "a_drtg10",
    "h_margin10", "a_margin10", "h_efg10", "a_efg10",
    "elo_diff", "h_ppg10", "a_ppg10", "h_papg10", "a_papg10",
    "h_ts10", "a_ts10", "h_pace10", "a_pace10",
    "rest_advantage", "h_sos10", "a_sos10",
    "h_consistency", "a_consistency",
]

# Extended top 45 features for broader polynomial coverage
TOP45_FEATURES = TOP30_FEATURES + [
    "h_wp3", "a_wp3", "h_wp20", "a_wp20",
    "h_ortg5", "a_ortg5", "h_drtg5", "a_drtg5",
    "h_margin5", "a_margin5", "h_streak", "a_streak",
    "h_tov_rate10", "a_tov_rate10", "h_orb_rate10", "a_orb_rate10",
]

# Top 20 features for squared terms
TOP20_SQUARED = [
    "h_wp10", "a_wp10", "h_netrtg10", "a_netrtg10",
    "h_ortg10", "a_ortg10", "h_drtg10", "a_drtg10",
    "h_margin10", "a_margin10", "elo_diff", "h_efg10", "a_efg10",
    "h_ppg10", "a_ppg10", "h_ts10", "a_ts10",
    "h_pace10", "a_pace10", "rest_advantage",
]


# ══════════════════════════════════════════════════════════════════
# A. POLYNOMIAL INTERACTION FEATURES (500+)
# ══════════════════════════════════════════════════════════════════

def _build_polynomial_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Generate pairwise products of top 30 features (435 pairs)
    plus squares of top 20 features, cube roots, and abs-diff products.

    Returns: (new_columns, new_names)
    """
    cols = []
    names = []

    # Resolve available features
    avail = []
    for name in TOP30_FEATURES:
        c = ni.col(X, name)
        if c is not None:
            avail.append((name, c))

    n_avail = len(avail)

    # A1. All pairwise products: C(n,2)
    for i in range(n_avail):
        for j in range(i + 1, n_avail):
            name_i, col_i = avail[i]
            name_j, col_j = avail[j]
            prod = col_i * col_j
            cols.append(prod)
            names.append(f"poly_{name_i}_x_{name_j}")

    # A2. Squares of top 20
    for name in TOP20_SQUARED:
        c = ni.col(X, name)
        if c is not None:
            cols.append(c ** 2)
            names.append(f"poly_sq_{name}")

    # A3. Cube root transforms of top 20 (captures nonlinearity)
    for name in TOP20_SQUARED:
        c = ni.col(X, name)
        if c is not None:
            cols.append(np.sign(c) * np.abs(c) ** (1.0 / 3.0))
            names.append(f"poly_cbrt_{name}")

    # A4. Absolute difference products: |f_i - f_j| for matched home/away pairs
    matchup_pairs = [
        ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"),
        ("h_netrtg10", "a_netrtg10"), ("h_netrtg5", "a_netrtg5"),
        ("h_ortg10", "a_ortg10"), ("h_drtg10", "a_drtg10"),
        ("h_margin10", "a_margin10"), ("h_efg10", "a_efg10"),
        ("h_ppg10", "a_ppg10"), ("h_papg10", "a_papg10"),
        ("h_ts10", "a_ts10"), ("h_pace10", "a_pace10"),
        ("h_consistency", "a_consistency"), ("h_sos10", "a_sos10"),
    ]
    for n1, n2 in matchup_pairs:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            cols.append(np.abs(c1 - c2))
            names.append(f"poly_absdiff_{n1}_{n2}")
            # Signed difference squared
            diff = c1 - c2
            cols.append(diff * np.abs(diff))
            names.append(f"poly_signsqdiff_{n1}_{n2}")

    # A5. Triple interactions for the most important feature combos
    triple_combos = [
        ("h_wp10", "a_wp10", "elo_diff"),
        ("h_netrtg10", "a_netrtg10", "rest_advantage"),
        ("h_ortg10", "a_drtg10", "h_pace10"),
        ("a_ortg10", "h_drtg10", "a_pace10"),
        ("h_margin10", "a_margin10", "elo_diff"),
        ("h_efg10", "a_efg10", "h_ts10"),
        ("h_wp5", "a_wp5", "rest_advantage"),
        ("h_wp10", "h_sos10", "elo_diff"),
        ("a_wp10", "a_sos10", "elo_diff"),
        ("h_consistency", "a_consistency", "elo_diff"),
        ("h_wp10", "a_drtg10", "rest_advantage"),
        ("a_wp10", "h_drtg10", "rest_advantage"),
        ("h_efg10", "a_tov_rate10", "elo_diff"),
        ("a_efg10", "h_tov_rate10", "elo_diff"),
        ("h_ortg10", "h_pace10", "h_efg10"),
        ("a_ortg10", "a_pace10", "a_efg10"),
        ("h_margin10", "h_consistency", "elo_diff"),
        ("a_margin10", "a_consistency", "elo_diff"),
        ("h_wp3", "a_wp3", "elo_diff"),
        ("h_wp20", "a_wp20", "elo_diff"),
    ]
    for n1, n2, n3 in triple_combos:
        c1, c2, c3 = ni.col(X, n1), ni.col(X, n2), ni.col(X, n3)
        if c1 is not None and c2 is not None and c3 is not None:
            cols.append(c1 * c2 * c3)
            names.append(f"poly_triple_{n1}_{n2}_{n3}")

    # A6. Extended pairwise products using TOP45 (only new pairs not in TOP30)
    extra_avail = []
    top30_set = set(TOP30_FEATURES)
    for name in TOP45_FEATURES:
        if name not in top30_set:
            c = ni.col(X, name)
            if c is not None:
                extra_avail.append((name, c))

    # Products of extra features with top 30
    for name_e, col_e in extra_avail:
        for name_t, col_t in avail:
            cols.append(col_e * col_t)
            names.append(f"poly_ext_{name_e}_x_{name_t}")

    # Products among extra features themselves
    for i in range(len(extra_avail)):
        for j in range(i + 1, len(extra_avail)):
            name_i, col_i = extra_avail[i]
            name_j, col_j = extra_avail[j]
            cols.append(col_i * col_j)
            names.append(f"poly_ext_{name_i}_x_{name_j}")

    # A7. Log-transformed products for top 10 pairs
    log_pairs = [
        ("h_wp10", "a_wp10"), ("h_netrtg10", "a_netrtg10"),
        ("h_ortg10", "a_drtg10"), ("a_ortg10", "h_drtg10"),
        ("h_margin10", "a_margin10"), ("h_efg10", "a_efg10"),
        ("h_ppg10", "a_ppg10"), ("h_ts10", "a_ts10"),
        ("h_pace10", "a_pace10"), ("h_sos10", "a_sos10"),
    ]
    for n1, n2 in log_pairs:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            cols.append(_safe_log(c1) * _safe_log(c2))
            names.append(f"poly_logprod_{n1}_{n2}")
            cols.append(_safe_log(c1) - _safe_log(c2))
            names.append(f"poly_logdiff_{n1}_{n2}")

    # A8. Power 1.5 transforms (between linear and quadratic)
    for name in TOP20_SQUARED[:12]:
        c = ni.col(X, name)
        if c is not None:
            cols.append(np.sign(c) * np.abs(c) ** 1.5)
            names.append(f"poly_pow15_{name}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# B. RATIO FEATURES (120+)
# ══════════════════════════════════════════════════════════════════

def _build_ratio_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Ratios between related stats across teams and windows.
    Different from base engine ratios -- uses log-ratios, multi-window
    comparisons, and offense-vs-defense cross-team ratios.
    """
    cols = []
    names = []

    # B1. Log-ratio of matched home/away stats
    log_ratio_pairs = [
        ("h_wp10", "a_wp10"), ("h_wp5", "a_wp5"), ("h_wp3", "a_wp3"),
        ("h_wp15", "a_wp15"), ("h_wp20", "a_wp20"),
        ("h_ortg10", "a_ortg10"), ("h_ortg5", "a_ortg5"),
        ("h_drtg10", "a_drtg10"), ("h_drtg5", "a_drtg5"),
        ("h_netrtg10", "a_netrtg10"), ("h_netrtg5", "a_netrtg5"),
        ("h_ppg10", "a_ppg10"), ("h_ppg5", "a_ppg5"),
        ("h_papg10", "a_papg10"), ("h_papg5", "a_papg5"),
        ("h_margin10", "a_margin10"), ("h_margin5", "a_margin5"),
        ("h_efg10", "a_efg10"), ("h_efg5", "a_efg5"),
        ("h_ts10", "a_ts10"), ("h_ts5", "a_ts5"),
        ("h_pace10", "a_pace10"), ("h_pace5", "a_pace5"),
        ("h_3p_pct10", "a_3p_pct10"), ("h_3p_pct5", "a_3p_pct5"),
        ("h_ft_rate10", "a_ft_rate10"), ("h_tov_rate10", "a_tov_rate10"),
        ("h_orb_rate10", "a_orb_rate10"), ("h_blk_rate10", "a_blk_rate10"),
        ("h_stl_rate10", "a_stl_rate10"), ("h_ast_rate10", "a_ast_rate10"),
    ]
    for n1, n2 in log_ratio_pairs:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            # Log ratio: log(1+|c1|) - log(1+|c2|) preserving sign
            lr = _safe_log(c1) - _safe_log(c2)
            cols.append(lr)
            names.append(f"lratio_{n1}_{n2}")

    # B2. Cross-team offense vs defense ratios
    cross_off_def = [
        ("h_ortg10", "a_drtg10"), ("h_ortg5", "a_drtg5"),
        ("a_ortg10", "h_drtg10"), ("a_ortg5", "h_drtg5"),
        ("h_efg10", "a_opp_efg10"), ("a_efg10", "h_opp_efg10"),
        ("h_efg5", "a_opp_efg5"), ("a_efg5", "h_opp_efg5"),
        ("h_ppg10", "a_papg10"), ("a_ppg10", "h_papg10"),
        ("h_ppg5", "a_papg5"), ("a_ppg5", "h_papg5"),
        ("h_ts10", "a_opp_efg10"), ("a_ts10", "h_opp_efg10"),
        ("h_3p_pct10", "a_perimeter_defense"), ("a_3p_pct10", "h_perimeter_defense"),
        ("h_tov_rate10", "a_stl_rate10"), ("a_tov_rate10", "h_stl_rate10"),
        ("h_orb_rate10", "a_dreb_pct10"), ("a_orb_rate10", "h_dreb_pct10"),
    ]
    for n1, n2 in cross_off_def:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            cols.append(_safe_div(c1, c2, fill=1.0))
            names.append(f"xratio_{n1}_{n2}")

    # B3. Multi-window ratio: short/long window for same stat (form acceleration)
    window_ratio_stats = ["wp", "ppg", "margin", "ortg", "drtg", "efg", "ts", "pace", "papg"]
    window_ratio_pairs = [(3, 10), (3, 20), (5, 15), (5, 20), (7, 20)]
    for prefix in ["h", "a"]:
        for stat in window_ratio_stats:
            for w_short, w_long in window_ratio_pairs:
                n_short = f"{prefix}_{stat}{w_short}"
                n_long = f"{prefix}_{stat}{w_long}"
                c_short, c_long = ni.col(X, n_short), ni.col(X, n_long)
                if c_short is not None and c_long is not None:
                    cols.append(_safe_div(c_short, c_long, fill=1.0))
                    names.append(f"wratio_{n_short}_{n_long}")

    # B4. Efficiency ratios: offensive output per defensive input
    for prefix in ["h", "a"]:
        ortg = ni.col(X, f"{prefix}_ortg10")
        drtg = ni.col(X, f"{prefix}_drtg10")
        pace = ni.col(X, f"{prefix}_pace10")
        efg = ni.col(X, f"{prefix}_efg10")
        ts = ni.col(X, f"{prefix}_ts10")
        tov = ni.col(X, f"{prefix}_tov_rate10")
        if ortg is not None and drtg is not None:
            cols.append(_safe_div(ortg, drtg, fill=1.0))
            names.append(f"effratio_{prefix}_ortg_drtg")
        if ortg is not None and pace is not None:
            cols.append(_safe_div(ortg, pace, fill=1.0))
            names.append(f"effratio_{prefix}_ortg_pace")
        if efg is not None and tov is not None:
            cols.append(_safe_div(efg, tov + 0.01, fill=1.0))
            names.append(f"effratio_{prefix}_efg_tov")
        if ts is not None and tov is not None:
            cols.append(_safe_div(ts, tov + 0.01, fill=1.0))
            names.append(f"effratio_{prefix}_ts_tov")

    # B5. Volatility-normalized differentials
    vol_norm_stats = ["margin", "ppg", "ortg", "drtg", "pace", "papg"]
    for stat in vol_norm_stats:
        h_val = ni.col(X, f"h_{stat}10")
        a_val = ni.col(X, f"a_{stat}10")
        h_vol = ni.col(X, f"h_vol_{stat}_10")
        a_vol = ni.col(X, f"a_vol_{stat}_10")
        if h_val is not None and a_val is not None:
            diff = h_val - a_val
            if h_vol is not None and a_vol is not None:
                combined_vol = np.sqrt(h_vol ** 2 + a_vol ** 2 + 0.01)
                cols.append(_safe_div(diff, combined_vol, fill=0.0))
                names.append(f"vnorm_diff_{stat}")
            # Also: ratio of volatilities (who is more predictable?)
            if h_vol is not None and a_vol is not None:
                cols.append(_safe_div(h_vol, a_vol + 0.01, fill=1.0))
                names.append(f"vol_ratio_{stat}")

    # B6. EWMA-based ratios
    ewma_stats_r = ["ppg", "margin", "ortg", "drtg", "efg", "ts"]
    for stat in ewma_stats_r:
        for prefix in ["h", "a"]:
            fast = ni.col(X, f"{prefix}_ewma_{stat}_a05")
            slow = ni.col(X, f"{prefix}_ewma_{stat}_a01")
            if fast is not None and slow is not None:
                cols.append(_safe_div(fast, slow + 0.01, fill=1.0))
                names.append(f"ewma_ratio_{prefix}_{stat}")

    # B7. Z-score to volatility ratio (signal/noise)
    zscore_stats = ["ppg", "margin", "ortg", "drtg", "efg", "ts"]
    for prefix in ["h", "a"]:
        for stat in zscore_stats:
            z = ni.col(X, f"{prefix}_zscore_{stat}")
            vol = ni.col(X, f"{prefix}_vol_{stat}_10")
            if z is not None and vol is not None:
                cols.append(_safe_div(z, vol + 0.01, fill=0.0))
                names.append(f"z_vol_ratio_{prefix}_{stat}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# C. BAYESIAN FEATURES (60+)
# ══════════════════════════════════════════════════════════════════

def _build_bayesian_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Bayesian-updated priors for team strength, clutch performance, rest impact.
    Uses conjugate Beta-Binomial updates where win% is treated as a Beta posterior.
    """
    cols = []
    names = []

    # Prior parameters: Beta(alpha0, beta0) = Beta(5, 5) = league average 0.5
    ALPHA0 = 5.0
    BETA0 = 5.0

    # C1. Bayesian win probability at multiple windows
    for prefix in ["h", "a"]:
        for w in [3, 5, 7, 10, 15, 20]:
            wp = ni.col(X, f"{prefix}_wp{w}")
            if wp is not None:
                # Bayesian posterior mean: (alpha0 + wins) / (alpha0 + beta0 + games)
                # Approximate wins = wp * w, games = w
                wins = wp * w
                posterior = (ALPHA0 + wins) / (ALPHA0 + BETA0 + w)
                cols.append(posterior)
                names.append(f"bayes_wp_{prefix}_w{w}")

                # Posterior variance (uncertainty)
                a_post = ALPHA0 + wins
                b_post = BETA0 + (w - wins)
                var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
                cols.append(var)
                names.append(f"bayes_var_{prefix}_w{w}")

    # C2. Bayesian strength differential
    for w in [5, 10, 20]:
        h_wp = ni.col(X, f"h_wp{w}")
        a_wp = ni.col(X, f"a_wp{w}")
        if h_wp is not None and a_wp is not None:
            h_post = (ALPHA0 + h_wp * w) / (ALPHA0 + BETA0 + w)
            a_post = (ALPHA0 + a_wp * w) / (ALPHA0 + BETA0 + w)
            cols.append(h_post - a_post)
            names.append(f"bayes_diff_w{w}")
            # Probability that home is stronger (approximate)
            # Using normal approx: P(h > a) ~ Phi((h_mean - a_mean) / sqrt(h_var + a_var))
            h_a = ALPHA0 + h_wp * w
            h_b = BETA0 + (1 - h_wp) * w
            a_a = ALPHA0 + a_wp * w
            a_b = BETA0 + (1 - a_wp) * w
            h_var = (h_a * h_b) / ((h_a + h_b) ** 2 * (h_a + h_b + 1))
            a_var = (a_a * a_b) / ((a_a + a_b) ** 2 * (a_a + a_b + 1))
            z = _safe_div(h_post - a_post, np.sqrt(h_var + a_var + 1e-10), fill=0.0)
            # Sigmoid approximation of Phi(z)
            prob_stronger = 1.0 / (1.0 + np.exp(-1.7 * z))
            cols.append(prob_stronger)
            names.append(f"bayes_prob_stronger_w{w}")

    # C3. Bayesian clutch performance
    for prefix in ["h", "a"]:
        clutch = ni.col(X, f"{prefix}_clutch_wp")
        if clutch is not None:
            # Assume ~20 clutch games per team per season
            n_clutch = 20
            clutch_post = (3.0 + clutch * n_clutch) / (3.0 + 3.0 + n_clutch)
            cols.append(clutch_post)
            names.append(f"bayes_clutch_{prefix}")

    # C4. Bayesian rest impact (prior: rest has moderate positive effect)
    rest_adv = ni.col(X, "rest_advantage")
    h_wp10 = ni.col(X, "h_wp10")
    a_wp10 = ni.col(X, "a_wp10")
    if rest_adv is not None and h_wp10 is not None and a_wp10 is not None:
        # Rest-adjusted Bayesian strength
        rest_bonus = np.clip(rest_adv * 0.02, -0.1, 0.1)
        cols.append((ALPHA0 + (h_wp10 + rest_bonus) * 10) / (ALPHA0 + BETA0 + 10))
        names.append("bayes_rest_adj_h")
        cols.append((ALPHA0 + (a_wp10 - rest_bonus) * 10) / (ALPHA0 + BETA0 + 10))
        names.append("bayes_rest_adj_a")

    # C5. Bayesian SOS-adjusted strength
    for prefix in ["h", "a"]:
        wp = ni.col(X, f"{prefix}_wp10")
        sos = ni.col(X, f"{prefix}_sos10")
        if wp is not None and sos is not None:
            # Stronger prior if SOS is high (harder schedule = more informative games)
            sos_weight = np.clip(sos * 2, 0.5, 2.0)  # Scale SOS to prior weight
            adj_alpha = ALPHA0 * sos_weight
            adj_beta = BETA0 * sos_weight
            adj_post = (adj_alpha + wp * 10) / (adj_alpha + adj_beta + 10)
            cols.append(adj_post)
            names.append(f"bayes_sos_adj_{prefix}")

    # C6. Bayesian home court advantage estimate
    h_home_wp = ni.col(X, "h_home_wp")
    h_away_wp = ni.col(X, "h_away_wp")
    if h_home_wp is not None and h_away_wp is not None:
        # Bayesian HCA = posterior(home_wp) - posterior(away_wp)
        h_post = (3.0 + h_home_wp * 20) / (3.0 + 3.0 + 20)
        a_post = (3.0 + h_away_wp * 20) / (3.0 + 3.0 + 20)
        cols.append(h_post - a_post)
        names.append("bayes_hca_estimate")

    # C7. Empirical Bayes shrinkage for key stats
    shrink_stats = [
        ("ortg", 105.0, 5.0),  # (stat, league_mean, prior_strength)
        ("drtg", 108.0, 5.0),
        ("efg", 0.50, 10.0),
        ("ts", 0.56, 10.0),
        ("pace", 100.0, 3.0),
    ]
    for stat, league_mean, prior_n in shrink_stats:
        for prefix in ["h", "a"]:
            for w in [5, 10]:
                c = ni.col(X, f"{prefix}_{stat}{w}")
                if c is not None:
                    # Shrinkage: posterior = (prior_n * league_mean + w * observed) / (prior_n + w)
                    shrunk = (prior_n * league_mean + w * c) / (prior_n + w)
                    cols.append(shrunk)
                    names.append(f"bayes_shrink_{prefix}_{stat}_w{w}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# D. TEMPORAL FEATURES (120+)
# ══════════════════════════════════════════════════════════════════

def _build_temporal_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    EMA crossovers, momentum slopes, acceleration (2nd derivative),
    trend reversals, and regime detection.
    """
    cols = []
    names = []

    # D1. EMA crossovers (fast/slow) — MACD-style for NBA stats
    # Use short vs long window as proxy for fast/slow EMA
    crossover_stats = ["wp", "ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
    crossover_pairs = [(3, 10), (5, 15), (5, 20), (7, 20), (3, 7)]
    for prefix in ["h", "a"]:
        for stat in crossover_stats:
            for w_fast, w_slow in crossover_pairs:
                fast = ni.col(X, f"{prefix}_{stat}{w_fast}")
                slow = ni.col(X, f"{prefix}_{stat}{w_slow}")
                if fast is not None and slow is not None:
                    # MACD line
                    macd = fast - slow
                    cols.append(macd)
                    names.append(f"macd_{prefix}_{stat}_f{w_fast}_s{w_slow}")

    # D2. Momentum slope: linear regression slope over windows
    # Approximate using (recent - older) / window_span
    slope_stats = ["wp", "ppg", "margin", "ortg", "drtg", "efg"]
    slope_windows = [(3, 10), (5, 20), (3, 20)]
    for prefix in ["h", "a"]:
        for stat in slope_stats:
            for w_recent, w_total in slope_windows:
                recent = ni.col(X, f"{prefix}_{stat}{w_recent}")
                total = ni.col(X, f"{prefix}_{stat}{w_total}")
                if recent is not None and total is not None:
                    slope = (recent - total) / max(w_total - w_recent, 1)
                    cols.append(slope)
                    names.append(f"slope_{prefix}_{stat}_r{w_recent}_t{w_total}")

    # D3. Acceleration (2nd derivative): change in slope
    # accel = slope(recent) - slope(older)
    accel_stats = ["wp", "margin", "ortg", "drtg"]
    for prefix in ["h", "a"]:
        for stat in accel_stats:
            w3 = ni.col(X, f"{prefix}_{stat}3")
            w7 = ni.col(X, f"{prefix}_{stat}7")
            w15 = ni.col(X, f"{prefix}_{stat}15")
            if w3 is not None and w7 is not None and w15 is not None:
                slope_recent = (w3 - w7) / 4.0
                slope_older = (w7 - w15) / 8.0
                accel = slope_recent - slope_older
                cols.append(accel)
                names.append(f"accel_{prefix}_{stat}_3_7_15")

            w5 = ni.col(X, f"{prefix}_{stat}5")
            w10 = ni.col(X, f"{prefix}_{stat}10")
            w20 = ni.col(X, f"{prefix}_{stat}20")
            if w5 is not None and w10 is not None and w20 is not None:
                slope_r2 = (w5 - w10) / 5.0
                slope_o2 = (w10 - w20) / 10.0
                accel2 = slope_r2 - slope_o2
                cols.append(accel2)
                names.append(f"accel_{prefix}_{stat}_5_10_20")

    # D4. Trend reversal detection
    # If short-term and medium-term trends disagree with long-term
    reversal_stats = ["wp", "margin", "ortg"]
    for prefix in ["h", "a"]:
        for stat in reversal_stats:
            w3 = ni.col(X, f"{prefix}_{stat}3")
            w10 = ni.col(X, f"{prefix}_{stat}10")
            w20 = ni.col(X, f"{prefix}_{stat}20")
            if w3 is not None and w10 is not None and w20 is not None:
                short_trend = w3 - w10
                long_trend = w10 - w20
                # Reversal: short and long trends have opposite signs
                reversal = np.where(short_trend * long_trend < 0, 1.0, 0.0)
                cols.append(reversal)
                names.append(f"reversal_{prefix}_{stat}")
                # Reversal magnitude
                cols.append(np.abs(short_trend - long_trend))
                names.append(f"reversal_mag_{prefix}_{stat}")

    # D5. Regime detection: is the team in a hot/cold/stable regime?
    for prefix in ["h", "a"]:
        wp3 = ni.col(X, f"{prefix}_wp3")
        wp10 = ni.col(X, f"{prefix}_wp10")
        wp20 = ni.col(X, f"{prefix}_wp20")
        if wp3 is not None and wp10 is not None and wp20 is not None:
            # Hot regime: short >> long
            hot = np.where((wp3 > wp10 + 0.1) & (wp10 > wp20 + 0.05), 1.0, 0.0)
            cols.append(hot)
            names.append(f"regime_hot_{prefix}")
            # Cold regime: short << long
            cold = np.where((wp3 < wp10 - 0.1) & (wp10 < wp20 - 0.05), 1.0, 0.0)
            cols.append(cold)
            names.append(f"regime_cold_{prefix}")
            # Stable regime
            stable = np.where(np.abs(wp3 - wp20) < 0.05, 1.0, 0.0)
            cols.append(stable)
            names.append(f"regime_stable_{prefix}")
            # Mean-reverting signal: extreme short-term that should revert
            extreme_up = np.where((wp3 > 0.8) & (wp20 < 0.6), 1.0, 0.0)
            cols.append(extreme_up)
            names.append(f"regime_mean_rev_up_{prefix}")
            extreme_down = np.where((wp3 < 0.2) & (wp20 > 0.4), 1.0, 0.0)
            cols.append(extreme_down)
            names.append(f"regime_mean_rev_down_{prefix}")

    # D6. EWMA crossover signals
    ewma_stats = ["ppg", "margin", "ortg", "drtg", "efg", "ts"]
    for prefix in ["h", "a"]:
        for stat in ewma_stats:
            fast_ewma = ni.col(X, f"{prefix}_ewma_{stat}_a05")
            slow_ewma = ni.col(X, f"{prefix}_ewma_{stat}_a01")
            if fast_ewma is not None and slow_ewma is not None:
                # EWMA crossover signal
                cols.append(fast_ewma - slow_ewma)
                names.append(f"ewma_xover_{prefix}_{stat}")
                # Crossover direction (binary)
                cols.append(np.where(fast_ewma > slow_ewma, 1.0, 0.0))
                names.append(f"ewma_xover_dir_{prefix}_{stat}")

    # D7. Volatility-adjusted momentum
    for prefix in ["h", "a"]:
        for stat in ["margin", "ppg", "ortg"]:
            momentum = ni.col(X, f"{prefix}_{stat}5")
            baseline = ni.col(X, f"{prefix}_{stat}20")
            vol = ni.col(X, f"{prefix}_vol_{stat}_10")
            if momentum is not None and baseline is not None and vol is not None:
                # Sharpe-like ratio: momentum / volatility
                vol_adj = _safe_div(momentum - baseline, vol + 0.01, fill=0.0)
                cols.append(vol_adj)
                names.append(f"vol_adj_mom_{prefix}_{stat}")

    # D8. Z-score momentum: (recent - season_mean) / season_std for more stats
    zscore_mom_stats = ["ppg", "papg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
    for prefix in ["h", "a"]:
        for stat in zscore_mom_stats:
            z = ni.col(X, f"{prefix}_zscore_{stat}")
            if z is not None:
                # Z-score squared (captures extremity regardless of direction)
                cols.append(z ** 2)
                names.append(f"zsq_{prefix}_{stat}")
                # Positive z-score only (above average)
                cols.append(np.maximum(z, 0))
                names.append(f"zpos_{prefix}_{stat}")
                # Negative z-score only (below average)
                cols.append(np.minimum(z, 0))
                names.append(f"zneg_{prefix}_{stat}")

    # D9. Cross-team EWMA differential evolution
    ewma_diff_stats = ["ppg", "margin", "ortg", "drtg", "efg", "ts", "pace"]
    for stat in ewma_diff_stats:
        for alpha in ["01", "03", "05"]:
            h_ewma = ni.col(X, f"h_ewma_{stat}_a{alpha}")
            a_ewma = ni.col(X, f"a_ewma_{stat}_a{alpha}")
            if h_ewma is not None and a_ewma is not None:
                cols.append(h_ewma - a_ewma)
                names.append(f"ewma_gap_{stat}_a{alpha}")

    # D10. Rolling range momentum (is the range expanding or contracting?)
    range_stats = ["ppg", "margin", "ortg", "drtg"]
    for prefix in ["h", "a"]:
        for stat in range_stats:
            r5 = ni.col(X, f"{prefix}_range_{stat}_5")
            r10 = ni.col(X, f"{prefix}_range_{stat}_10")
            if r5 is not None and r10 is not None:
                cols.append(r5 - r10)
                names.append(f"range_delta_{prefix}_{stat}")
                cols.append(_safe_div(r5, r10 + 0.01, fill=1.0))
                names.append(f"range_ratio_{prefix}_{stat}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# E. MARKET MICROSTRUCTURE II (100+)
# ══════════════════════════════════════════════════════════════════

def _build_market_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Advanced market features: CLV decomposition, steam velocity,
    sharp/public divergence, closing line efficiency, market regime.
    """
    cols = []
    names = []

    # Get base market features
    opening_spread = ni.col(X, "opening_spread")
    current_spread = ni.col(X, "current_spread")
    spread_movement = ni.col(X, "spread_movement")
    opening_total = ni.col(X, "opening_total")
    current_total = ni.col(X, "current_total")
    total_movement = ni.col(X, "total_movement")
    implied_home = ni.col(X, "implied_prob_home")
    implied_away = ni.col(X, "implied_prob_away")
    public_pct = ni.col(X, "public_pct_home")
    public_money = ni.col(X, "public_money_pct_home")
    smart_money = ni.col(X, "smart_money_indicator")
    steam_move = ni.col(X, "steam_move")
    time_to_close = ni.col(X, "time_to_close")
    opening_ml = ni.col(X, "opening_ml_home")
    current_ml = ni.col(X, "current_ml_home")

    if opening_spread is not None and current_spread is not None:
        # E1. CLV decomposition
        # Closing line value = current - opening (positive = line moved in our favor)
        clv = current_spread - opening_spread
        cols.append(clv)
        names.append("mktx_clv_spread")

        # CLV as percentage of opening
        cols.append(_safe_div(clv, np.abs(opening_spread) + 0.5, fill=0.0))
        names.append("mktx_clv_pct")

        # CLV velocity: movement per hour
        if time_to_close is not None:
            elapsed = np.clip(24.0 - time_to_close, 1.0, 24.0)
            cols.append(_safe_div(clv, elapsed, fill=0.0))
            names.append("mktx_clv_velocity")

        # Spread movement squared (captures large moves)
        if spread_movement is not None:
            cols.append(spread_movement ** 2 * np.sign(spread_movement))
            names.append("mktx_spread_move_sq")

        # E2. Steam velocity and intensity
        if steam_move is not None and spread_movement is not None:
            cols.append(steam_move * np.abs(spread_movement))
            names.append("mktx_steam_intensity")
            if time_to_close is not None:
                cols.append(steam_move * _safe_div(np.abs(spread_movement), time_to_close + 0.5))
                names.append("mktx_steam_velocity")

        # E3. Sharp/public divergence features
        if public_pct is not None and public_money is not None:
            # Money vs bets divergence (sharp indicator)
            divergence = public_money - public_pct
            cols.append(divergence)
            names.append("mktx_money_bet_div")
            cols.append(np.abs(divergence))
            names.append("mktx_money_bet_div_abs")
            # Contrarian signal: when public is heavily on one side
            cols.append(np.where(np.abs(public_pct - 0.5) > 0.2, 1.0 - public_pct, 0.0))
            names.append("mktx_contrarian_strength")

        if smart_money is not None:
            # Smart money alignment with spread movement
            if spread_movement is not None:
                cols.append(smart_money * spread_movement)
                names.append("mktx_smart_spread_align")
            # Smart money confidence
            cols.append(np.abs(smart_money))
            names.append("mktx_smart_confidence")

        # E4. Market efficiency indicators
        if implied_home is not None:
            # Implied probability gap
            cols.append(np.abs(implied_home - 0.5))
            names.append("mktx_implied_gap")
            # Market certainty (how far from 50/50)
            cols.append(implied_home ** 2 + (1 - implied_home) ** 2)
            names.append("mktx_market_certainty")

        if implied_home is not None and implied_away is not None:
            # Overround (vig)
            overround = implied_home + implied_away - 1.0
            cols.append(overround)
            names.append("mktx_overround")
            # True implied probability (vig-removed)
            true_home = _safe_div(implied_home, implied_home + implied_away, fill=0.5)
            cols.append(true_home)
            names.append("mktx_true_prob_home")

        # E5. Line movement patterns
        if spread_movement is not None:
            # Movement direction indicator
            cols.append(np.sign(spread_movement))
            names.append("mktx_move_direction")
            # Large move flag (>1.5 points)
            cols.append(np.where(np.abs(spread_movement) > 1.5, 1.0, 0.0))
            names.append("mktx_large_move")
            # Reverse line movement (public on one side, line moves other way)
            if public_pct is not None:
                public_side = np.where(public_pct > 0.5, -1.0, 1.0)  # Public favors home = neg spread
                rlm = np.where(spread_movement * public_side > 0, 1.0, 0.0)
                cols.append(rlm)
                names.append("mktx_rlm_detected")

    if opening_total is not None and current_total is not None:
        # E6. Total line features
        total_move = current_total - opening_total
        cols.append(total_move)
        names.append("mktx_total_move")
        cols.append(np.abs(total_move))
        names.append("mktx_total_move_abs")
        # Total > 230 flag (high scoring expected)
        cols.append(np.where(current_total > 230, 1.0, 0.0))
        names.append("mktx_high_total")
        # Total < 210 flag (low scoring expected)
        cols.append(np.where(current_total < 210, 1.0, 0.0))
        names.append("mktx_low_total")

    if opening_ml is not None and current_ml is not None:
        # E7. Moneyline-derived features
        ml_move = current_ml - opening_ml
        cols.append(ml_move)
        names.append("mktx_ml_move")
        # ML to implied probability conversion
        def ml_to_prob(ml):
            pos = np.where(ml > 0, 100.0 / (ml + 100.0), 0.0)
            neg = np.where(ml < 0, np.abs(ml) / (np.abs(ml) + 100.0), 0.0)
            neutral = np.where(ml == 0, 0.5, 0.0)
            return pos + neg + neutral
        open_prob = ml_to_prob(opening_ml)
        curr_prob = ml_to_prob(current_ml)
        cols.append(curr_prob - open_prob)
        names.append("mktx_prob_shift")
        cols.append(np.abs(curr_prob - open_prob))
        names.append("mktx_prob_shift_abs")

    # E8. Market + model interaction features
    elo_diff = ni.col(X, "elo_diff")
    if elo_diff is not None and implied_home is not None:
        # Convert elo_diff to implied probability
        elo_prob = 1.0 / (1.0 + np.exp(-elo_diff / 173.718))
        model_edge = elo_prob - implied_home
        cols.append(model_edge)
        names.append("mktx_model_edge")
        cols.append(np.abs(model_edge))
        names.append("mktx_model_edge_abs")
        # Edge direction
        cols.append(np.sign(model_edge))
        names.append("mktx_edge_direction")

    # E9. Combined market strength signal
    signals = []
    if spread_movement is not None:
        signals.append(np.clip(spread_movement / 3.0, -1, 1))
    if smart_money is not None:
        signals.append(np.clip(smart_money, -1, 1))
    if steam_move is not None:
        signals.append(steam_move)
    if len(signals) >= 2:
        combined = sum(signals) / len(signals)
        cols.append(combined)
        names.append("mktx_combined_signal")
        # Signal agreement (all pointing same direction)
        agreement = np.mean([np.sign(s) for s in signals], axis=0)
        cols.append(agreement)
        names.append("mktx_signal_agreement")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# F. CROSS-WINDOW FEATURES (200+)
# ══════════════════════════════════════════════════════════════════

def _build_cross_window_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Differences and ratios between different rolling windows.
    Unlike base engine cross-window which uses deltas/acceleration,
    this adds: normalized differences, ratio-based momentum, dispersion,
    and multi-stat composite window signals.
    """
    cols = []
    names = []

    stats = ["wp", "ppg", "margin", "ortg", "drtg", "efg", "ts", "pace", "papg"]
    windows = [3, 5, 7, 10, 15, 20]

    for prefix in ["h", "a"]:
        # F1. Normalized window differences (z-score-like)
        # (short - long) / volatility
        for stat in stats:
            for w_short in [3, 5]:
                for w_long in [10, 15, 20]:
                    short_val = ni.col(X, f"{prefix}_{stat}{w_short}")
                    long_val = ni.col(X, f"{prefix}_{stat}{w_long}")
                    vol = ni.col(X, f"{prefix}_vol_{stat}_10")
                    if short_val is not None and long_val is not None:
                        diff = short_val - long_val
                        if vol is not None:
                            norm_diff = _safe_div(diff, vol + 0.001, fill=0.0)
                            cols.append(norm_diff)
                            names.append(f"xwn_{prefix}_{stat}_{w_short}v{w_long}")

        # F2. Window dispersion: how much does the stat vary across windows?
        for stat in stats:
            vals = []
            for w in windows:
                c = ni.col(X, f"{prefix}_{stat}{w}")
                if c is not None:
                    vals.append(c)
            if len(vals) >= 3:
                stacked = np.column_stack(vals)
                # Std across windows (per game)
                dispersion = np.std(stacked, axis=1)
                cols.append(dispersion)
                names.append(f"xw_disp_{prefix}_{stat}")
                # Range across windows
                w_range = np.max(stacked, axis=1) - np.min(stacked, axis=1)
                cols.append(w_range)
                names.append(f"xw_range_{prefix}_{stat}")
                # Is the most recent window the best?
                best_is_recent = np.where(
                    stacked[:, 0] == np.max(stacked, axis=1), 1.0, 0.0
                )
                cols.append(best_is_recent)
                names.append(f"xw_recent_best_{prefix}_{stat}")

    # F3. Home/Away cross-window differential momentum
    for stat in ["wp", "margin", "ortg", "netrtg"]:
        for w_short, w_long in [(3, 10), (5, 20)]:
            h_short = ni.col(X, f"h_{stat}{w_short}")
            h_long = ni.col(X, f"h_{stat}{w_long}")
            a_short = ni.col(X, f"a_{stat}{w_short}")
            a_long = ni.col(X, f"a_{stat}{w_long}")
            if h_short is not None and h_long is not None and a_short is not None and a_long is not None:
                h_momentum = h_short - h_long
                a_momentum = a_short - a_long
                # Momentum advantage: who is improving faster?
                cols.append(h_momentum - a_momentum)
                names.append(f"xw_mom_adv_{stat}_{w_short}v{w_long}")
                # Both improving?
                cols.append(np.where((h_momentum > 0) & (a_momentum > 0), 1.0, 0.0))
                names.append(f"xw_both_up_{stat}_{w_short}v{w_long}")
                # Home improving, away declining?
                cols.append(np.where((h_momentum > 0) & (a_momentum < 0), 1.0, 0.0))
                names.append(f"xw_h_up_a_down_{stat}_{w_short}v{w_long}")

    # F4. Composite momentum scores across multiple stats
    momentum_stats = ["wp", "margin", "ortg", "efg"]
    for prefix in ["h", "a"]:
        mom_signals = []
        for stat in momentum_stats:
            w5 = ni.col(X, f"{prefix}_{stat}5")
            w20 = ni.col(X, f"{prefix}_{stat}20")
            if w5 is not None and w20 is not None:
                mom_signals.append(w5 - w20)
        if len(mom_signals) >= 3:
            stacked_mom = np.column_stack(mom_signals)
            # Composite momentum (average of all stat momentums)
            cols.append(np.mean(stacked_mom, axis=1))
            names.append(f"xw_composite_mom_{prefix}")
            # Momentum agreement (how many stats show same direction)
            agreement = np.mean(np.sign(stacked_mom), axis=1)
            cols.append(agreement)
            names.append(f"xw_mom_agreement_{prefix}")
            # Strongest momentum signal
            cols.append(np.max(np.abs(stacked_mom), axis=1))
            names.append(f"xw_max_mom_{prefix}")

    # F5. More window pairs for differential momentum (more stats, more pairs)
    for stat in ["wp", "margin", "ortg", "drtg", "efg", "ts", "ppg", "papg"]:
        for w_short, w_long in [(3, 7), (3, 15), (5, 10), (7, 15), (7, 20), (10, 20)]:
            h_short = ni.col(X, f"h_{stat}{w_short}")
            h_long = ni.col(X, f"h_{stat}{w_long}")
            a_short = ni.col(X, f"a_{stat}{w_short}")
            a_long = ni.col(X, f"a_{stat}{w_long}")
            if h_short is not None and h_long is not None and a_short is not None and a_long is not None:
                h_mom = h_short - h_long
                a_mom = a_short - a_long
                cols.append(h_mom - a_mom)
                names.append(f"xw5_mom_adv_{stat}_{w_short}v{w_long}")

    # F6. Cross-stat window comparison (does offense and defense improve together?)
    for prefix in ["h", "a"]:
        for w in [5, 10]:
            ortg_mom = None
            drtg_mom = None
            ortg_s = ni.col(X, f"{prefix}_ortg{w}")
            ortg_l = ni.col(X, f"{prefix}_ortg20")
            drtg_s = ni.col(X, f"{prefix}_drtg{w}")
            drtg_l = ni.col(X, f"{prefix}_drtg20")
            if ortg_s is not None and ortg_l is not None:
                ortg_mom = ortg_s - ortg_l
            if drtg_s is not None and drtg_l is not None:
                drtg_mom = drtg_s - drtg_l
            if ortg_mom is not None and drtg_mom is not None:
                # Both improving (offense up, defense down = improving)
                cols.append(ortg_mom - drtg_mom)
                names.append(f"xw6_net_improve_{prefix}_w{w}")
                # Offense improving but defense declining
                cols.append(np.where((ortg_mom > 0) & (drtg_mom > 0), 1.0, 0.0))
                names.append(f"xw6_off_up_def_down_{prefix}_w{w}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# G. REFEREE ADVANCED FEATURES (60+)
# ══════════════════════════════════════════════════════════════════

def _build_referee_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Referee-team interactions, referee clutch tendencies, referee pace impact,
    and ref bias interactions with team playing styles.
    """
    cols = []
    names = []

    # Base referee features
    ref_foul_bias = ni.col(X, "ref_home_foul_bias")
    ref_total_fouls = ni.col(X, "ref_total_fouls_avg")
    ref_over_tend = ni.col(X, "ref_over_tendency")
    ref_home_wr = ni.col(X, "ref_home_win_rate")
    ref_pace_impact = ni.col(X, "ref_pace_impact")
    ref_foul_rate = ni.col(X, "ref_foul_rate_vs_league")
    ref_close_bias = ni.col(X, "ref_close_game_bias")
    ref_tech_rate = ni.col(X, "ref_tech_foul_rate")
    ref_ft_adv = ni.col(X, "ref_home_ft_advantage")

    # G1. Referee × team playing style interactions
    for prefix in ["h", "a"]:
        pace = ni.col(X, f"{prefix}_pace10")
        ft_rate = ni.col(X, f"{prefix}_ft_rate10")
        three_rate = ni.col(X, f"{prefix}_3p_pct10")
        tov_rate = ni.col(X, f"{prefix}_tov_rate10")
        efg = ni.col(X, f"{prefix}_efg10")

        if ref_total_fouls is not None:
            # Foul-prone teams benefit from high-foul refs
            if ft_rate is not None:
                cols.append(ref_total_fouls * ft_rate)
                names.append(f"refx_fouls_x_ft_{prefix}")
            # Fast-paced teams with fast-whistling refs
            if pace is not None and ref_pace_impact is not None:
                cols.append(pace * ref_pace_impact)
                names.append(f"refx_pace_x_ref_pace_{prefix}")

        if ref_foul_bias is not None:
            # Home foul bias × team aggressiveness
            if ft_rate is not None:
                sign = 1.0 if prefix == "h" else -1.0
                cols.append(ref_foul_bias * ft_rate * sign)
                names.append(f"refx_bias_x_ft_{prefix}")
            # Home bias × team efficiency (ref can help efficient teams more)
            if efg is not None:
                sign = 1.0 if prefix == "h" else -1.0
                cols.append(ref_foul_bias * efg * sign)
                names.append(f"refx_bias_x_efg_{prefix}")

        if ref_over_tend is not None:
            # Over-tendency × pace
            if pace is not None:
                cols.append(ref_over_tend * pace)
                names.append(f"refx_over_x_pace_{prefix}")
            # Over-tendency × shooting efficiency
            if three_rate is not None:
                cols.append(ref_over_tend * three_rate)
                names.append(f"refx_over_x_3pt_{prefix}")

    # G2. Referee differential impact (how much does ref help home vs away)
    if ref_foul_bias is not None:
        h_ft_rate = ni.col(X, "h_ft_rate10")
        a_ft_rate = ni.col(X, "a_ft_rate10")
        if h_ft_rate is not None and a_ft_rate is not None:
            # Net referee impact = bias × (home_ft_dependency - away_ft_dependency)
            cols.append(ref_foul_bias * (h_ft_rate - a_ft_rate))
            names.append("refx_net_foul_impact")

    if ref_home_wr is not None:
        # Ref home win rate deviation from average (0.58)
        cols.append(ref_home_wr - 0.58)
        names.append("refx_home_bias_excess")
        # Strong home bias flag
        cols.append(np.where(ref_home_wr > 0.62, 1.0, 0.0))
        names.append("refx_strong_home_bias")
        # Away-friendly ref flag
        cols.append(np.where(ref_home_wr < 0.54, 1.0, 0.0))
        names.append("refx_away_friendly")

    if ref_close_bias is not None:
        # Clutch ref impact × team clutch ability
        h_clutch = ni.col(X, "h_clutch_wp")
        a_clutch = ni.col(X, "a_clutch_wp")
        if h_clutch is not None:
            cols.append(ref_close_bias * h_clutch)
            names.append("refx_clutch_x_h_clutch")
        if a_clutch is not None:
            cols.append((1 - ref_close_bias) * a_clutch)
            names.append("refx_clutch_x_a_clutch")

    if ref_tech_rate is not None:
        # Tech foul rate interactions
        for prefix in ["h", "a"]:
            streak = ni.col(X, f"{prefix}_streak")
            if streak is not None:
                # Frustrated teams (losing streak) + tech-happy refs
                cols.append(ref_tech_rate * np.clip(-streak, 0, 10) / 10.0)
                names.append(f"refx_tech_x_losing_{prefix}")

    # G3. Referee × matchup context
    if ref_over_tend is not None:
        h_pace = ni.col(X, "h_pace10")
        a_pace = ni.col(X, "a_pace10")
        if h_pace is not None and a_pace is not None:
            combined_pace = (h_pace + a_pace) / 2.0
            cols.append(ref_over_tend * combined_pace)
            names.append("refx_over_x_combined_pace")
            # High-pace matchup with over-tendency ref
            cols.append(np.where((combined_pace > 102) & (ref_over_tend > 0.55), 1.0, 0.0))
            names.append("refx_pace_over_flag")

    if ref_ft_adv is not None:
        # FT advantage impact on aggressive vs perimeter teams
        for prefix in ["h", "a"]:
            paint_pts = ni.col(X, f"{prefix}_paint_pts10")
            if paint_pts is not None:
                sign = 1.0 if prefix == "h" else -1.0
                cols.append(ref_ft_adv * paint_pts * sign / 50.0)
                names.append(f"refx_ft_adv_x_paint_{prefix}")

    # G4. Referee composite scores
    if ref_foul_bias is not None and ref_home_wr is not None and ref_close_bias is not None:
        # Overall home advantage from ref crew
        ref_home_score = (
            ref_foul_bias / 5.0 * 0.3 +
            (ref_home_wr - 0.5) * 0.4 +
            (ref_close_bias - 0.5) * 0.3
        )
        cols.append(ref_home_score)
        names.append("refx_composite_home_advantage")
        # Is this a "fair" or "home-biased" crew?
        cols.append(np.where(np.abs(ref_home_score) < 0.02, 1.0, 0.0))
        names.append("refx_fair_crew")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# H. CLUSTER FEATURES (50+)
# ══════════════════════════════════════════════════════════════════

def _build_cluster_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Team archetype clusters, playing style embeddings, schedule difficulty
    clusters, and matchup type indicators. Uses lightweight k-means-style
    clustering without sklearn dependency.
    """
    cols = []
    names = []

    # H1. Team archetype features based on playing style dimensions
    # Define archetype dimensions
    for prefix in ["h", "a"]:
        pace = ni.col(X, f"{prefix}_pace10")
        ortg = ni.col(X, f"{prefix}_ortg10")
        drtg = ni.col(X, f"{prefix}_drtg10")
        three_rate = ni.col(X, f"{prefix}_3p_pct10")
        ft_rate = ni.col(X, f"{prefix}_ft_rate10")
        efg = ni.col(X, f"{prefix}_efg10")
        tov = ni.col(X, f"{prefix}_tov_rate10")

        if pace is not None and ortg is not None and drtg is not None:
            # Offensive archetype: fast & efficient vs slow & grinding
            off_style = np.where(pace > 101, 1.0, 0.0) * np.where(ortg > 110, 1.0, 0.0)
            cols.append(off_style)
            names.append(f"clust_fast_efficient_{prefix}")

            grind_style = np.where(pace < 98, 1.0, 0.0) * np.where(drtg < 108, 1.0, 0.0)
            cols.append(grind_style)
            names.append(f"clust_slow_grind_{prefix}")

            # Balanced: good offense + good defense
            balanced = np.where((ortg > 108) & (drtg < 110), 1.0, 0.0)
            cols.append(balanced)
            names.append(f"clust_balanced_{prefix}")

            # Offense-only: good offense, bad defense
            off_only = np.where((ortg > 110) & (drtg > 112), 1.0, 0.0)
            cols.append(off_only)
            names.append(f"clust_offense_only_{prefix}")

            # Defense-only: bad offense, good defense
            def_only = np.where((ortg < 108) & (drtg < 106), 1.0, 0.0)
            cols.append(def_only)
            names.append(f"clust_defense_only_{prefix}")

            # Elite: top tier both ways
            elite = np.where((ortg > 112) & (drtg < 108), 1.0, 0.0)
            cols.append(elite)
            names.append(f"clust_elite_{prefix}")

            # Tanking: bad both ways
            tank = np.where((ortg < 105) & (drtg > 114), 1.0, 0.0)
            cols.append(tank)
            names.append(f"clust_tank_{prefix}")

        if three_rate is not None and ft_rate is not None:
            # Three-point heavy team
            cols.append(np.where(three_rate > 0.38, 1.0, 0.0))
            names.append(f"clust_3pt_heavy_{prefix}")

            # Paint-dominant team (low 3pt, high FT)
            cols.append(np.where((three_rate < 0.33) & (ft_rate > 0.28), 1.0, 0.0))
            names.append(f"clust_paint_dom_{prefix}")

        if efg is not None and tov is not None:
            # Efficient but turnover-prone
            cols.append(np.where((efg > 0.52) & (tov > 0.14), 1.0, 0.0))
            names.append(f"clust_eff_sloppy_{prefix}")

            # Clean but inefficient
            cols.append(np.where((efg < 0.48) & (tov < 0.12), 1.0, 0.0))
            names.append(f"clust_clean_ineff_{prefix}")

    # H2. Matchup type classification
    h_pace = ni.col(X, "h_pace10")
    a_pace = ni.col(X, "a_pace10")
    h_ortg = ni.col(X, "h_ortg10")
    a_ortg = ni.col(X, "a_ortg10")
    h_drtg = ni.col(X, "h_drtg10")
    a_drtg = ni.col(X, "a_drtg10")

    if h_pace is not None and a_pace is not None:
        avg_pace = (h_pace + a_pace) / 2.0
        # High-pace shootout expected
        cols.append(np.where(avg_pace > 103, 1.0, 0.0))
        names.append("clust_shootout_matchup")
        # Grind-it-out matchup
        cols.append(np.where(avg_pace < 96, 1.0, 0.0))
        names.append("clust_grind_matchup")
        # Pace mismatch
        cols.append(np.abs(h_pace - a_pace))
        names.append("clust_pace_mismatch")
        # Extreme pace mismatch flag
        cols.append(np.where(np.abs(h_pace - a_pace) > 5, 1.0, 0.0))
        names.append("clust_extreme_pace_mismatch")

    if h_ortg is not None and a_ortg is not None and h_drtg is not None and a_drtg is not None:
        # Both elite matchup
        cols.append(np.where(
            (h_ortg > 110) & (a_ortg > 110) & (h_drtg < 110) & (a_drtg < 110),
            1.0, 0.0
        ))
        names.append("clust_elite_vs_elite")

        # Quality gap (big mismatch in overall quality)
        h_net = h_ortg - h_drtg
        a_net = a_ortg - a_drtg
        quality_gap = np.abs(h_net - a_net)
        cols.append(quality_gap)
        names.append("clust_quality_gap")
        # Trap game: big favorite vs bad team (upset potential)
        cols.append(np.where((quality_gap > 10) & (h_net > a_net), 1.0, 0.0))
        names.append("clust_trap_game_home_fav")
        cols.append(np.where((quality_gap > 10) & (a_net > h_net), 1.0, 0.0))
        names.append("clust_trap_game_away_fav")

    # H3. Schedule difficulty clusters
    for prefix in ["h", "a"]:
        sos = ni.col(X, f"{prefix}_sos10")
        rest = ni.col(X, f"{prefix}_rest_days") if prefix == "h" else ni.col(X, f"{prefix[0]}_rest_days")
        games_7d = ni.col(X, f"{prefix}_games_7d") if prefix == "h" else ni.col(X, f"{prefix[0]}_games_7d")

        if sos is not None:
            # Easy schedule stretch
            cols.append(np.where(sos < 0.45, 1.0, 0.0))
            names.append(f"clust_easy_sched_{prefix}")
            # Hard schedule stretch
            cols.append(np.where(sos > 0.55, 1.0, 0.0))
            names.append(f"clust_hard_sched_{prefix}")

    h_rest = ni.col(X, "h_rest_days")
    a_rest = ni.col(X, "a_rest_days")
    if h_rest is not None and a_rest is not None:
        # Rest mismatch categories
        # Well-rested home vs tired away
        cols.append(np.where((h_rest >= 2) & (a_rest <= 1), 1.0, 0.0))
        names.append("clust_rest_adv_home")
        # Tired home vs well-rested away
        cols.append(np.where((h_rest <= 1) & (a_rest >= 2), 1.0, 0.0))
        names.append("clust_rest_adv_away")
        # Both rested
        cols.append(np.where((h_rest >= 2) & (a_rest >= 2), 1.0, 0.0))
        names.append("clust_both_rested")
        # Both tired
        cols.append(np.where((h_rest <= 1) & (a_rest <= 1), 1.0, 0.0))
        names.append("clust_both_tired")

    # H4. Continuous style embeddings (distance from archetype centroids)
    # Centroid-based features: distance from 4 archetype centroids
    archetype_centroids = {
        # (pace, ortg, drtg) centroids
        "uptempo_elite": (104.0, 115.0, 105.0),
        "halfcourt_defense": (96.0, 108.0, 104.0),
        "balanced_good": (100.0, 112.0, 108.0),
        "rebuilding": (100.0, 105.0, 114.0),
    }
    for prefix in ["h", "a"]:
        pace = ni.col(X, f"{prefix}_pace10")
        ortg = ni.col(X, f"{prefix}_ortg10")
        drtg = ni.col(X, f"{prefix}_drtg10")
        if pace is not None and ortg is not None and drtg is not None:
            for arch_name, (c_pace, c_ortg, c_drtg) in archetype_centroids.items():
                # Euclidean distance (normalized by typical range)
                dist = np.sqrt(
                    ((pace - c_pace) / 5.0) ** 2 +
                    ((ortg - c_ortg) / 10.0) ** 2 +
                    ((drtg - c_drtg) / 10.0) ** 2
                )
                cols.append(dist)
                names.append(f"clust_dist_{arch_name}_{prefix}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# I. ADDITIONAL DERIVED FEATURES (200+)
# ══════════════════════════════════════════════════════════════════

def _build_additional_features(X: np.ndarray, ni: _NameIndex) -> Tuple[np.ndarray, List[str]]:
    """
    Miscellaneous high-value derived features that don't fit other categories:
    - Dean Oliver Four Factors composites
    - Power ranking composites
    - Fatigue-adjusted predictions
    - Information-theoretic features
    """
    cols = []
    names = []

    # I1. Four Factors composite scores (Dean Oliver weights)
    # Offense: eFG% (40%) + TOV% (25%) + ORB% (20%) + FT rate (15%)
    for prefix in ["h", "a"]:
        for w in [5, 10]:
            efg = ni.col(X, f"{prefix}_efg{w}")
            tov = ni.col(X, f"{prefix}_tov_rate{w}")
            orb = ni.col(X, f"{prefix}_orb_rate{w}")
            ftr = ni.col(X, f"{prefix}_ft_rate{w}")
            if efg is not None and tov is not None and orb is not None and ftr is not None:
                off_score = efg * 0.40 + (1 - tov) * 0.25 + orb * 0.20 + ftr * 0.15
                cols.append(off_score)
                names.append(f"ff_off_composite_{prefix}_w{w}")

            opp_efg = ni.col(X, f"{prefix}_opp_efg{w}")
            opp_tov = ni.col(X, f"{prefix}_opp_tov{w}")
            opp_orb = ni.col(X, f"{prefix}_opp_orb{w}")
            opp_ftr = ni.col(X, f"{prefix}_opp_ft{w}")
            if opp_efg is not None and opp_tov is not None and opp_orb is not None and opp_ftr is not None:
                def_score = (1 - opp_efg) * 0.40 + opp_tov * 0.25 + (1 - opp_orb) * 0.20 + (1 - opp_ftr) * 0.15
                cols.append(def_score)
                names.append(f"ff_def_composite_{prefix}_w{w}")

    # I2. Four Factors matchup advantages
    for w in [5, 10]:
        for factor, h_name, a_name in [
            ("efg", f"h_efg{w}", f"a_opp_efg{w}"),
            ("tov", f"h_tov_rate{w}", f"a_opp_tov{w}"),
            ("orb", f"h_orb_rate{w}", f"a_opp_orb{w}"),
            ("ftr", f"h_ft_rate{w}", f"a_opp_ft{w}"),
        ]:
            h_val = ni.col(X, h_name)
            a_val = ni.col(X, a_name)
            if h_val is not None and a_val is not None:
                cols.append(h_val - a_val)
                names.append(f"ff_matchup_{factor}_h_w{w}")

        for factor, h_name, a_name in [
            ("efg", f"a_efg{w}", f"h_opp_efg{w}"),
            ("tov", f"a_tov_rate{w}", f"h_opp_tov{w}"),
            ("orb", f"a_orb_rate{w}", f"h_opp_orb{w}"),
            ("ftr", f"a_ft_rate{w}", f"h_opp_ft{w}"),
        ]:
            h_val = ni.col(X, h_name)
            a_val = ni.col(X, a_name)
            if h_val is not None and a_val is not None:
                cols.append(h_val - a_val)
                names.append(f"ff_matchup_{factor}_a_w{w}")

    # I3. Composite power differential features
    elo_diff = ni.col(X, "elo_diff")
    for stat in ["wp10", "netrtg10", "margin10"]:
        h_val = ni.col(X, f"h_{stat}")
        a_val = ni.col(X, f"a_{stat}")
        if h_val is not None and a_val is not None and elo_diff is not None:
            diff = h_val - a_val
            # Combined signal: stats + Elo agreement
            cols.append(diff * np.sign(elo_diff))
            names.append(f"power_agree_{stat}")
            # Disagreement: Elo says one thing, stat says another
            cols.append(np.where(diff * elo_diff < 0, 1.0, 0.0))
            names.append(f"power_disagree_{stat}")

    # I4. Fatigue-quality interaction features
    rest_adv = ni.col(X, "rest_advantage")
    if rest_adv is not None:
        for stat_diff in [("h_wp10", "a_wp10"), ("h_netrtg10", "a_netrtg10"),
                         ("h_ortg10", "a_ortg10")]:
            h_val = ni.col(X, stat_diff[0])
            a_val = ni.col(X, stat_diff[1])
            if h_val is not None and a_val is not None:
                quality_diff = h_val - a_val
                # Rest amplifies quality gap
                cols.append(quality_diff * rest_adv)
                names.append(f"fatigue_x_{stat_diff[0].split('_', 1)[1]}")

    # I5. Consistency-adjusted features
    for prefix in ["h", "a"]:
        wp = ni.col(X, f"{prefix}_wp10")
        cons = ni.col(X, f"{prefix}_consistency")
        if wp is not None and cons is not None:
            # Sharpe-like: win% / consistency
            cols.append(_safe_div(wp - 0.5, cons + 0.01, fill=0.0))
            names.append(f"sharpe_wp_{prefix}")
            # High-floor team: good + consistent
            cols.append(np.where((wp > 0.55) & (cons < 8), 1.0, 0.0))
            names.append(f"high_floor_{prefix}")
            # Volatile team: medium record + inconsistent
            cols.append(np.where((np.abs(wp - 0.5) < 0.1) & (cons > 12), 1.0, 0.0))
            names.append(f"volatile_{prefix}")

    # I6. Z-score based features (relative to league)
    z_features = ["h_wp10", "a_wp10", "h_netrtg10", "a_netrtg10",
                  "h_ortg10", "a_ortg10", "h_drtg10", "a_drtg10"]
    for feat_name in z_features:
        c = ni.col(X, feat_name)
        if c is not None:
            mu = np.nanmean(c)
            sd = np.nanstd(c)
            if sd > 1e-6:
                z = (c - mu) / sd
                cols.append(z)
                names.append(f"zscore_exp_{feat_name}")

    # I7. Percentile rank features
    rank_features = ["h_wp10", "a_wp10", "h_netrtg10", "a_netrtg10", "elo_diff"]
    for feat_name in rank_features:
        c = ni.col(X, feat_name)
        if c is not None and c.shape[0] > 1:
            # Percentile rank (per-sample, looking at the distribution)
            sorted_idx = np.argsort(np.argsort(c))
            pct = sorted_idx / max(c.shape[0] - 1, 1)
            cols.append(pct)
            names.append(f"pctrank_{feat_name}")

    # I8. Sigmoid transforms for probability-like features
    sigmoid_features = [
        ("elo_diff", 173.718),  # Standard Elo-to-prob scaling
        ("h_netrtg10", 5.0),
        ("a_netrtg10", 5.0),
        ("rest_advantage", 2.0),
    ]
    for feat_name, scale in sigmoid_features:
        c = ni.col(X, feat_name)
        if c is not None:
            sigmoid = 1.0 / (1.0 + np.exp(-c / scale))
            cols.append(sigmoid)
            names.append(f"sigmoid_{feat_name}")

    # I9. Min/Max of home vs away for key stats
    minmax_stats = ["wp10", "wp5", "netrtg10", "ortg10", "drtg10",
                   "margin10", "efg10", "ts10", "ppg10"]
    for stat in minmax_stats:
        h = ni.col(X, f"h_{stat}")
        a = ni.col(X, f"a_{stat}")
        if h is not None and a is not None:
            cols.append(np.maximum(h, a))
            names.append(f"max_ha_{stat}")
            cols.append(np.minimum(h, a))
            names.append(f"min_ha_{stat}")
            # Ratio of min to max
            cols.append(_safe_div(np.minimum(h, a), np.maximum(np.abs(h), np.abs(a)) + 0.01, fill=1.0))
            names.append(f"minmax_ratio_{stat}")

    # I10. Multi-stat composite differentials (weighted combinations)
    composite_configs = [
        # (name, [(feature, weight), ...])
        ("power_index", [("wp10", 0.25), ("netrtg10", 0.25), ("margin10", 0.20),
                         ("ortg10", 0.15), ("efg10", 0.15)]),
        ("offense_index", [("ortg10", 0.30), ("efg10", 0.25), ("ts10", 0.20),
                           ("ppg10", 0.15), ("pace10", 0.10)]),
        ("defense_index", [("drtg10", -0.35), ("papg10", -0.25),
                           ("opp_efg10", -0.20), ("tov_rate10", 0.20)]),
        ("form_index", [("wp3", 0.30), ("wp5", 0.25), ("margin5", 0.25),
                        ("streak", 0.20)]),
        ("stability_index", [("wp10", 0.30), ("consistency", -0.30),
                             ("wp20", 0.20), ("sos10", 0.20)]),
    ]
    for comp_name, weights in composite_configs:
        for prefix in ["h", "a"]:
            comp_val = np.zeros(X.shape[0])
            n_found = 0
            for feat, w in weights:
                c = ni.col(X, f"{prefix}_{feat}")
                if c is not None:
                    comp_val += c * w
                    n_found += 1
            if n_found >= 2:
                cols.append(comp_val)
                names.append(f"comp_{comp_name}_{prefix}")

    # Composite differentials
    for comp_name, weights in composite_configs:
        h_val = np.zeros(X.shape[0])
        a_val = np.zeros(X.shape[0])
        h_found, a_found = 0, 0
        for feat, w in weights:
            hc = ni.col(X, f"h_{feat}")
            ac = ni.col(X, f"a_{feat}")
            if hc is not None:
                h_val += hc * w
                h_found += 1
            if ac is not None:
                a_val += ac * w
                a_found += 1
        if h_found >= 2 and a_found >= 2:
            cols.append(h_val - a_val)
            names.append(f"comp_diff_{comp_name}")

    # I11. Interaction of composites with context
    season_phase = ni.col(X, "season_phase")
    elo_diff = ni.col(X, "elo_diff")
    rest_adv = ni.col(X, "rest_advantage")
    context_features = [
        ("season_phase", season_phase),
        ("elo_diff", elo_diff),
        ("rest_advantage", rest_adv),
    ]
    key_stats = ["h_wp10", "a_wp10", "h_netrtg10", "a_netrtg10",
                 "h_margin10", "a_margin10", "h_ortg10", "a_ortg10"]
    for ctx_name, ctx_col in context_features:
        if ctx_col is not None:
            for stat_name in key_stats:
                stat_col = ni.col(X, stat_name)
                if stat_col is not None:
                    cols.append(stat_col * ctx_col)
                    names.append(f"ctx_{stat_name}_x_{ctx_name}")

    # I12. Nonlinear transforms of key differentials
    diff_pairs = [
        ("h_wp10", "a_wp10"), ("h_netrtg10", "a_netrtg10"),
        ("h_ortg10", "a_ortg10"), ("h_drtg10", "a_drtg10"),
        ("h_margin10", "a_margin10"), ("h_efg10", "a_efg10"),
        ("h_ppg10", "a_ppg10"), ("h_ts10", "a_ts10"),
        ("h_pace10", "a_pace10"), ("h_consistency", "a_consistency"),
    ]
    for n1, n2 in diff_pairs:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            diff = c1 - c2
            # Tanh transform (bounded nonlinear)
            cols.append(np.tanh(diff))
            names.append(f"tanh_diff_{n1}_{n2}")
            # ReLU-like: positive advantage only
            cols.append(np.maximum(diff, 0))
            names.append(f"relu_pos_{n1}_{n2}")
            # Negative advantage only
            cols.append(np.minimum(diff, 0))
            names.append(f"relu_neg_{n1}_{n2}")
            # Exponential decay of advantage
            cols.append(np.exp(-np.abs(diff)))
            names.append(f"exp_decay_{n1}_{n2}")

    # I13. Rolling stat interaction features (offense meets defense across teams)
    for w in [5, 10]:
        # Home offense vs Away defense mismatches
        for off_stat, def_stat in [
            (f"h_ortg{w}", f"a_drtg{w}"),
            (f"h_efg{w}", f"a_opp_efg{w}"),
            (f"h_pace{w}", f"a_pace{w}"),
            (f"h_3p_pct{w}", f"a_perimeter_defense"),
            (f"h_ft_rate{w}", f"a_opp_ft{w}"),
        ]:
            oc = ni.col(X, off_stat)
            dc = ni.col(X, def_stat)
            if oc is not None and dc is not None:
                cols.append(oc - dc)
                names.append(f"matchup_diff_{off_stat}_{def_stat}")
                cols.append(oc * dc)
                names.append(f"matchup_prod_{off_stat}_{def_stat}")

        # Away offense vs Home defense mismatches
        for off_stat, def_stat in [
            (f"a_ortg{w}", f"h_drtg{w}"),
            (f"a_efg{w}", f"h_opp_efg{w}"),
            (f"a_3p_pct{w}", f"h_perimeter_defense"),
            (f"a_ft_rate{w}", f"h_opp_ft{w}"),
        ]:
            oc = ni.col(X, off_stat)
            dc = ni.col(X, def_stat)
            if oc is not None and dc is not None:
                cols.append(oc - dc)
                names.append(f"matchup_diff_{off_stat}_{def_stat}")
                cols.append(oc * dc)
                names.append(f"matchup_prod_{off_stat}_{def_stat}")

    # I14. Binned features for key stats (discretization helps tree models)
    bin_features = [
        ("elo_diff", [-200, -100, -50, 0, 50, 100, 200]),
        ("rest_advantage", [-3, -1, 0, 1, 3]),
        ("h_wp10", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("a_wp10", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("h_netrtg10", [-5, -2, 0, 2, 5]),
        ("a_netrtg10", [-5, -2, 0, 2, 5]),
    ]
    for feat_name, thresholds in bin_features:
        c = ni.col(X, feat_name)
        if c is not None:
            for thresh in thresholds:
                cols.append(np.where(c > thresh, 1.0, 0.0))
                names.append(f"bin_{feat_name}_gt_{thresh}")

    # I15. Difference-of-differences: how has the gap between teams changed?
    dod_stats = ["wp", "margin", "ortg", "drtg", "efg", "ppg", "ts", "pace"]
    for stat in dod_stats:
        for w_recent, w_older in [(3, 10), (5, 20), (3, 20), (5, 15)]:
            h_recent = ni.col(X, f"h_{stat}{w_recent}")
            a_recent = ni.col(X, f"a_{stat}{w_recent}")
            h_older = ni.col(X, f"h_{stat}{w_older}")
            a_older = ni.col(X, f"a_{stat}{w_older}")
            if h_recent is not None and a_recent is not None and h_older is not None and a_older is not None:
                gap_recent = h_recent - a_recent
                gap_older = h_older - a_older
                dod = gap_recent - gap_older  # Is the gap widening or closing?
                cols.append(dod)
                names.append(f"dod_{stat}_{w_recent}v{w_older}")

    # I16. Harmonic mean and geometric mean of key feature pairs
    gm_pairs = [
        ("h_wp10", "a_wp10"), ("h_ortg10", "a_ortg10"),
        ("h_efg10", "a_efg10"), ("h_ts10", "a_ts10"),
        ("h_margin10", "a_margin10"), ("h_netrtg10", "a_netrtg10"),
    ]
    for n1, n2 in gm_pairs:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            # Geometric mean (for positive values, safe version)
            c1_pos = np.abs(c1) + 0.01
            c2_pos = np.abs(c2) + 0.01
            gm = np.sqrt(c1_pos * c2_pos)
            cols.append(gm)
            names.append(f"geomean_{n1}_{n2}")
            # Harmonic mean
            hm = _safe_div(2.0 * c1_pos * c2_pos, c1_pos + c2_pos, fill=0.0)
            cols.append(hm)
            names.append(f"harmmean_{n1}_{n2}")

    # I17. Conditional features: stat X given condition Y
    # e.g., home win% when home is rested vs tired
    h_rest = ni.col(X, "h_rest_days")
    a_rest = ni.col(X, "a_rest_days")
    h_b2b = ni.col(X, "h_b2b")
    a_b2b = ni.col(X, "a_b2b")
    h_wp10 = ni.col(X, "h_wp10")
    a_wp10 = ni.col(X, "a_wp10")
    elo_diff = ni.col(X, "elo_diff")
    season_phase = ni.col(X, "season_phase")

    if h_rest is not None and h_wp10 is not None:
        # Home quality when rested
        cols.append(np.where(h_rest >= 2, h_wp10, 0.0))
        names.append("cond_h_wp10_rested")
        cols.append(np.where(h_rest <= 1, h_wp10, 0.0))
        names.append("cond_h_wp10_tired")
    if a_rest is not None and a_wp10 is not None:
        cols.append(np.where(a_rest >= 2, a_wp10, 0.0))
        names.append("cond_a_wp10_rested")
        cols.append(np.where(a_rest <= 1, a_wp10, 0.0))
        names.append("cond_a_wp10_tired")
    if season_phase is not None and elo_diff is not None:
        # Elo reliability changes with season phase (more games = more reliable)
        cols.append(elo_diff * season_phase)
        names.append("cond_elo_x_phase")
        cols.append(elo_diff * (1 - season_phase))
        names.append("cond_elo_x_early_season")

    # I18. Exponential moving average interaction products
    ewma_interact = [
        ("h_ewma_ppg_a05", "a_ewma_ppg_a05"),
        ("h_ewma_margin_a05", "a_ewma_margin_a05"),
        ("h_ewma_ortg_a05", "a_ewma_drtg_a05"),
        ("a_ewma_ortg_a05", "h_ewma_drtg_a05"),
        ("h_ewma_efg_a05", "a_ewma_efg_a05"),
        ("h_ewma_ts_a05", "a_ewma_ts_a05"),
        ("h_ewma_ppg_a01", "a_ewma_ppg_a01"),
        ("h_ewma_margin_a01", "a_ewma_margin_a01"),
        ("h_ewma_ortg_a01", "a_ewma_drtg_a01"),
        ("a_ewma_ortg_a01", "h_ewma_drtg_a01"),
    ]
    for n1, n2 in ewma_interact:
        c1, c2 = ni.col(X, n1), ni.col(X, n2)
        if c1 is not None and c2 is not None:
            cols.append(c1 - c2)
            names.append(f"ewma_diff_{n1}_{n2}")
            cols.append(c1 * c2)
            names.append(f"ewma_prod_{n1}_{n2}")

    # I19. Power rating combination features
    power_feats = ["elo_standard", "elo_margin_adj", "elo_recency_weighted",
                   "raptor_composite", "bayesian_rating", "glicko_rating"]
    for feat in power_feats:
        h_val = ni.col(X, f"h_{feat}")
        a_val = ni.col(X, f"a_{feat}")
        if h_val is not None and a_val is not None:
            cols.append(h_val - a_val)
            names.append(f"pwr_diff_{feat}")
            cols.append(np.abs(h_val - a_val))
            names.append(f"pwr_absdiff_{feat}")
            cols.append(h_val + a_val)
            names.append(f"pwr_sum_{feat}")

    # I20. Skewness and kurtosis interaction features
    for prefix in ["h", "a"]:
        for stat in ["margin", "ppg"]:
            skew = ni.col(X, f"{prefix}_skew_{stat}_10")
            kurt = ni.col(X, f"{prefix}_kurtosis_{stat}_10")
            wp = ni.col(X, f"{prefix}_wp10")
            if skew is not None and wp is not None:
                cols.append(skew * wp)
                names.append(f"skew_x_wp_{prefix}_{stat}")
            if kurt is not None and wp is not None:
                cols.append(kurt * wp)
                names.append(f"kurt_x_wp_{prefix}_{stat}")
            if skew is not None and kurt is not None:
                cols.append(skew * kurt)
                names.append(f"skew_x_kurt_{prefix}_{stat}")

    if not cols:
        return np.zeros((X.shape[0], 0)), []

    return np.column_stack(cols), names


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE PRE-FILTERING
# ══════════════════════════════════════════════════════════════════

def _prefilter_features(X_expanded: np.ndarray, feature_names: List[str],
                        variance_threshold: float = 1e-6,
                        correlation_threshold: float = 0.999) -> Tuple[np.ndarray, List[str]]:
    """
    Remove garbage features before feeding to genetic algorithm:
    1. Zero/near-zero variance features
    2. All-NaN or all-zero columns
    3. Nearly perfectly correlated duplicates (keep first)

    This is a LIGHTWEIGHT pre-filter, not feature selection.
    The GA does the real selection.
    """
    n_orig = X_expanded.shape[1]
    keep = np.ones(n_orig, dtype=bool)

    # Step 1: Remove zero-variance features
    variances = np.var(X_expanded, axis=0)
    keep &= variances > variance_threshold

    # Step 2: Remove all-zero columns
    all_zero = np.all(X_expanded == 0, axis=0)
    keep &= ~all_zero

    # Step 3: Remove near-duplicate columns (high correlation)
    # Only check new features (after base features) to avoid O(n^2) on thousands
    # Use sampling for speed: check correlation on 500 random samples
    kept_indices = np.where(keep)[0]
    if len(kept_indices) > 50 and X_expanded.shape[0] > 100:
        sample_size = min(500, X_expanded.shape[0])
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(X_expanded.shape[0], sample_size, replace=False)
        X_sample = X_expanded[sample_idx][:, kept_indices]

        # Compute correlations in chunks to avoid memory issues
        # Only flag exact duplicates (corr > 0.999)
        to_remove = set()
        chunk_size = 200
        for start in range(0, len(kept_indices), chunk_size):
            end = min(start + chunk_size, len(kept_indices))
            chunk = X_sample[:, start:end]
            chunk_std = np.std(chunk, axis=0)
            valid = chunk_std > 1e-8

            if np.sum(valid) < 2:
                continue

            # Standardize valid columns
            chunk_valid = chunk[:, valid]
            chunk_valid = (chunk_valid - np.mean(chunk_valid, axis=0)) / (np.std(chunk_valid, axis=0) + 1e-10)

            # Compute correlation matrix within chunk
            corr = chunk_valid.T @ chunk_valid / sample_size

            # Find highly correlated pairs
            valid_idx = np.where(valid)[0]
            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    if abs(corr[i, j]) > correlation_threshold:
                        global_j = kept_indices[start + valid_idx[j]]
                        to_remove.add(global_j)

        for idx in to_remove:
            keep[idx] = False

    X_filtered = X_expanded[:, keep]
    filtered_names = [feature_names[i] for i in range(n_orig) if keep[i]]

    n_removed = n_orig - X_filtered.shape[1]
    if n_removed > 0:
        print(f"  Pre-filter: removed {n_removed} garbage features "
              f"({n_orig} -> {X_filtered.shape[1]})")

    return X_filtered, filtered_names


# ══════════════════════════════════════════════════════════════════
# MAIN EXPANSION FUNCTION
# ══════════════════════════════════════════════════════════════════

def expand_features(X: np.ndarray, feature_names: List[str],
                    include_market: bool = True,
                    prefilter: bool = True,
                    verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Expand base ~2058 features to 4000+ features for genetic selection.

    Takes the output of NBAFeatureEngine.build() and generates additional
    derived features across 8+ new categories.

    Args:
        X: Base feature matrix from engine.build() — shape (n_games, ~2058)
        feature_names: List of base feature names from engine.feature_names
        include_market: Whether to include market-derived features
        prefilter: Whether to remove garbage features before returning
        verbose: Print progress

    Returns:
        X_expanded: Expanded feature matrix — shape (n_games, 4000+)
        expanded_names: List of all feature names (base + expanded)
    """
    assert X.shape[1] == len(feature_names), \
        f"X has {X.shape[1]} cols but {len(feature_names)} names"

    ni = _NameIndex(feature_names)
    n_base = X.shape[1]

    if verbose:
        print(f"Feature Expansion: starting with {n_base} base features")

    expansion_blocks = []
    expansion_names = []

    # A. Polynomial Interactions (500+)
    if verbose:
        print("  [A] Polynomial interactions...")
    poly_X, poly_names = _build_polynomial_features(X, ni)
    expansion_blocks.append(poly_X)
    expansion_names.extend(poly_names)
    if verbose:
        print(f"      -> {len(poly_names)} features")

    # B. Ratio Features (120+)
    if verbose:
        print("  [B] Ratio features...")
    ratio_X, ratio_names = _build_ratio_features(X, ni)
    expansion_blocks.append(ratio_X)
    expansion_names.extend(ratio_names)
    if verbose:
        print(f"      -> {len(ratio_names)} features")

    # C. Bayesian Features (60+)
    if verbose:
        print("  [C] Bayesian features...")
    bayes_X, bayes_names = _build_bayesian_features(X, ni)
    expansion_blocks.append(bayes_X)
    expansion_names.extend(bayes_names)
    if verbose:
        print(f"      -> {len(bayes_names)} features")

    # D. Temporal Features (120+)
    if verbose:
        print("  [D] Temporal features...")
    temp_X, temp_names = _build_temporal_features(X, ni)
    expansion_blocks.append(temp_X)
    expansion_names.extend(temp_names)
    if verbose:
        print(f"      -> {len(temp_names)} features")

    # E. Market Microstructure II (100+)
    if include_market:
        if verbose:
            print("  [E] Market microstructure II...")
        mkt_X, mkt_names = _build_market_features(X, ni)
        expansion_blocks.append(mkt_X)
        expansion_names.extend(mkt_names)
        if verbose:
            print(f"      -> {len(mkt_names)} features")

    # F. Cross-Window Features (200+)
    if verbose:
        print("  [F] Cross-window features...")
    xw_X, xw_names = _build_cross_window_features(X, ni)
    expansion_blocks.append(xw_X)
    expansion_names.extend(xw_names)
    if verbose:
        print(f"      -> {len(xw_names)} features")

    # G. Referee Advanced (60+)
    if verbose:
        print("  [G] Referee advanced...")
    ref_X, ref_names = _build_referee_features(X, ni)
    expansion_blocks.append(ref_X)
    expansion_names.extend(ref_names)
    if verbose:
        print(f"      -> {len(ref_names)} features")

    # H. Cluster Features (50+)
    if verbose:
        print("  [H] Cluster features...")
    clust_X, clust_names = _build_cluster_features(X, ni)
    expansion_blocks.append(clust_X)
    expansion_names.extend(clust_names)
    if verbose:
        print(f"      -> {len(clust_names)} features")

    # I. Additional Derived Features (200+)
    if verbose:
        print("  [I] Additional derived...")
    add_X, add_names = _build_additional_features(X, ni)
    expansion_blocks.append(add_X)
    expansion_names.extend(add_names)
    if verbose:
        print(f"      -> {len(add_names)} features")

    # Concatenate all expansion blocks
    valid_blocks = [b for b in expansion_blocks if b.shape[1] > 0]
    if valid_blocks:
        X_new = np.column_stack(valid_blocks)
    else:
        X_new = np.zeros((X.shape[0], 0))

    # Stabilize numerical values
    X_new = _stabilize(X_new)

    # Combine with base features
    X_expanded = np.hstack([X, X_new])
    all_names = list(feature_names) + expansion_names

    if verbose:
        print(f"\n  Total before pre-filter: {len(all_names)} features "
              f"({n_base} base + {len(expansion_names)} expanded)")

    # Pre-filter garbage features
    if prefilter:
        X_expanded, all_names = _prefilter_features(X_expanded, all_names)

    if verbose:
        print(f"  FINAL: {len(all_names)} features ready for genetic selection")

    return X_expanded, all_names


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE: Get expansion feature count without data
# ══════════════════════════════════════════════════════════════════

def estimate_expansion_count(n_base: int = 2058, include_market: bool = True) -> Dict[str, int]:
    """Estimate feature counts per category without actual data."""
    counts = {
        "A. Polynomial Interactions": 1155,   # C(30,2) + ext pairs + sq + cbrt + absdiff + triple + log + pow
        "B. Ratio Features": 131,             # log-ratio + cross + window + efficiency + vol_norm + ewma + z_vol
        "C. Bayesian Features": 57,           # wp + diff + clutch + rest + sos + hca + shrink
        "D. Temporal Features": 187,          # macd + slope + accel + reversal + regime + ewma_xover + vol_adj + zscore + ewma_gap + range
        "E. Market Microstructure II": 30 if include_market else 0,
        "F. Cross-Window Features": 106,      # norm + disp + momentum_diff + composite + extended + cross-stat
        "G. Referee Advanced": 26,            # ref-team interactions + composites
        "H. Cluster Features": 46,            # archetypes + matchup types + schedule + centroids
        "I. Additional Derived": 310,         # composites + context + nonlinear + matchups + bins + dod + means + cond + ewma + power + skew
    }
    total_new = sum(counts.values())
    counts["Base Features"] = n_base
    counts["TOTAL (before pre-filter)"] = n_base + total_new
    return counts


# ══════════════════════════════════════════════════════════════════
# CLI TESTING
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Quant Feature Expansion — Estimate")
    print("=" * 60)

    counts = estimate_expansion_count()
    for cat, count in counts.items():
        if cat in ("Base Features", "TOTAL"):
            print(f"\n  {'─' * 40}")
        print(f"  {cat}: {count}")

    print(f"\n{'=' * 60}")
    print("Testing with synthetic data...")
    print("=" * 60)

    # Create synthetic feature matrix matching engine output
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features.engine import NBAFeatureEngine
    engine = NBAFeatureEngine(include_market=True)
    n_features = len(engine.feature_names)
    n_samples = 200

    # Generate synthetic data (random but realistic ranges)
    rng = np.random.RandomState(42)
    X_fake = rng.randn(n_samples, n_features) * 0.5 + 0.5
    # Ensure some features look like win%
    for i, name in enumerate(engine.feature_names):
        if "wp" in name:
            X_fake[:, i] = np.clip(X_fake[:, i], 0, 1)
        elif "ortg" in name:
            X_fake[:, i] = X_fake[:, i] * 10 + 105
        elif "drtg" in name:
            X_fake[:, i] = X_fake[:, i] * 10 + 108
        elif "pace" in name:
            X_fake[:, i] = X_fake[:, i] * 5 + 100
        elif "efg" in name:
            X_fake[:, i] = np.clip(X_fake[:, i], 0.3, 0.7)

    X_expanded, expanded_names = expand_features(
        X_fake, engine.feature_names, include_market=True, prefilter=True, verbose=True
    )

    print(f"\nResult: {X_expanded.shape[0]} games x {X_expanded.shape[1]} features")
    print(f"NaN count: {np.isnan(X_expanded).sum()}")
    print(f"Inf count: {np.isinf(X_expanded).sum()}")

    # Show feature name distribution
    prefixes = {}
    for n in expanded_names:
        p = n.split("_")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print("\nTop prefixes:")
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"  {p}: {c}")
