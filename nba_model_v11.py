#!/usr/bin/env python3
"""
=============================================================================
 NBA BETTING MODEL — VERSION 11
 Architecture : Log-Odds | EPM/RAPTOR | Kelly Fractionnel | Platt Scaling
 Nouveautés V11 :
   - Amortisseur d'usage cumulé (Bloc A) : évite les Win Rates aberrants
   - Shot Quality / eFG% Regression (Bloc B) : tempère le momentum luck
   - Persistence JSON automatique (nba_history.json) pour Platt Scaling
=============================================================================
 SECTIONS :
   1.  Imports & Configuration
   2.  Constantes & Paramètres Globaux
   3.  Structures de Données (Dataclasses)
   4.  Bloc C  — Qualité Intrinsèque (SRS Rolling Pondéré)
   5.  Bloc A  — Roster & Disponibilités (EPM + Amortisseur Usage V11)
   6.  Bloc B  — Forme Récente, Momentum & Shot Quality (V11)
   7.  Bloc D  — Contexte Situationnel (Home/Away Personnalisé)
   8.  Bloc E  — Matchup & H2H
   9.  Assemblage Log-Odds Final
  10.  Bloc F  — Validation Marché (Filtre Post-Calcul)
  11.  Kelly Criterion Adaptatif
  12.  Calibration Platt Scaling
  13.  Filtres Éliminatoires
  14.  Rapport Final, Output & Persistence JSON (V11)
  15.  Interface Match (Section à Modifier par l'Utilisateur)
=============================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS & CONFIGURATION
# =============================================================================
import math
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

try:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] sklearn non disponible — Platt Scaling désactivé")

# =============================================================================
# SECTION 2 — CONSTANTES & PARAMÈTRES GLOBAUX
# =============================================================================

# ── Formule logistique ────────────────────────────────────────────────────────
SRS_COEFFICIENT       = 0.15
SRS_ROLLING_WEIGHT    = 0.70
SRS_SEASON_WEIGHT     = 0.30

# ── Seuils de décision ───────────────────────────────────────────────────────
WIN_RATE_THRESHOLD    = 0.62
EV_THRESHOLD          = 1.10
BLOWOUT_THRESHOLD     = 0.85
PLATT_MIN_SAMPLES     = 50
SAFETY_MARGIN         = 0.03    # -3% si Win Rate > 80% (avant Platt actif)

# ── Amortisseur d'usage cumulé (V11) ─────────────────────────────────────────
USAGE_CAP_THRESHOLD   = 0.40    # Si usage cumulé absents > 40% → amortissement
USAGE_CAP_FACTOR      = 0.85    # Réduction à 85% du λ brut

# ── Shot Quality eFG% (V11) ──────────────────────────────────────────────────
EFG_LUCK_THRESHOLD    = 0.05    # Seuil de surperformance (5%)
EFG_LUCK_PENALTY      = -0.15   # Pénalité log-odds si surperformance détectée

# ── Kelly ────────────────────────────────────────────────────────────────────
KELLY_HALF            = 0.50
KELLY_QUARTER         = 0.25
KELLY_MAX_UNITS       = 15.0
BANKROLL              = 100.0

# ── Home/Away personnalisé (λ log-odds) ──────────────────────────────────────
HOME_ADVANTAGE = {
    "DEN": +0.34,  "UTA": +0.28,
    "BOS": +0.22,  "NYK": +0.20,  "CLE": +0.20,
    "MIL": +0.18,  "PHI": +0.18,  "MIA": +0.18,
    "CHI": +0.16,  "LAL": +0.16,  "GSW": +0.16,
    "SAS": +0.10,  "ORL": +0.10,
    "WAS": +0.08,  "BKN": +0.08,
    "DEFAULT": +0.14,
}

# ── Table d'impact EPM (log-odds λ) ──────────────────────────────────────────
EPM_IMPACT_TABLE = [
    (5.0,  99.0, -0.55, "Star Elite (EPM >5.0)"),
    (3.0,   5.0, -0.35, "Star (EPM 3.0–5.0)"),
    (1.0,   3.0, -0.20, "All-Star (EPM 1.0–3.0)"),
    (0.0,   1.0, -0.08, "Rotation (EPM <1.0)"),
]

# ── Persistence ───────────────────────────────────────────────────────────────
HISTORY_FILE = "nba_history.json"

# =============================================================================
# SECTION 3 — STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class PlayerAbsence:
    """Représente un joueur absent ou en doute."""
    name:        str
    epm:         float
    usage_rate:  float          # 0–1  (ex: 0.30 = 30%)
    status:      str            # "OUT" | "GTD"
    confirmed:   bool = True    # False → impact réduit à 50%

    def epm_display(self) -> str:
        return f"EPM {self.epm:+.1f} | Usage {self.usage_rate:.0%}"


@dataclass
class TeamData:
    """Données complètes d'une équipe pour un match."""
    name:               str
    abbreviation:       str

    # Bloc C
    srs_last30:         float
    srs_season:         float
    net_rtg:            float
    off_rtg_rank:       int
    def_rtg_rank:       int
    pace:               float

    # Bloc A
    absences:           list = field(default_factory=list)

    # Bloc B
    current_streak:     int   = 0
    l5_record:          tuple = (3, 2)
    efg_season:         float = 0.540   # eFG% saison  (V11)
    efg_last5:          float = 0.540   # eFG% L5      (V11)

    # Bloc D
    is_home:            bool  = False
    is_back_to_back:    bool  = False
    games_last_6_days:  int   = 2
    rest_days:          int   = 1
    playoff_situation:  str   = "neutral"   # "contending"|"eliminated"|"neutral"

    # Bloc E
    to_rate:            float = 13.5
    oreb_rate:          float = 10.0
    h2h_wins_season:    int   = 0
    h2h_games_season:   int   = 0

    @property
    def srs_weighted(self) -> float:
        return SRS_ROLLING_WEIGHT * self.srs_last30 + SRS_SEASON_WEIGHT * self.srs_season

    @property
    def gtd_unconfirmed_count(self) -> int:
        return sum(1 for a in self.absences if a.status == "GTD" and not a.confirmed)

    @property
    def total_usage_out(self) -> float:
        return sum(a.usage_rate for a in self.absences)


@dataclass
class MatchData:
    """Données d'un match complet."""
    home:                TeamData
    away:                TeamData
    date:                str
    tip_off_cet:         str
    odds_home:           float = 0.0
    h2h_era_weight:      float = 1.0
    line_movement_sharp: int   = 0      # +1 home | -1 away | 0 neutre


# =============================================================================
# SECTION 4 — BLOC C : QUALITÉ INTRINSÈQUE
# =============================================================================

def compute_bloc_c(home: TeamData, away: TeamData) -> dict:
    delta_srs  = home.srs_weighted - away.srs_weighted
    logit_base = SRS_COEFFICIENT * delta_srs

    def_rtg_adj = 0.0
    if home.def_rtg_rank <= 5:  def_rtg_adj += 0.24
    elif home.def_rtg_rank >= 26: def_rtg_adj -= 0.24
    if away.def_rtg_rank <= 5:  def_rtg_adj -= 0.24
    elif away.def_rtg_rank >= 26: def_rtg_adj += 0.24

    off_rtg_adj = 0.0
    if home.off_rtg_rank <= 5:  off_rtg_adj += 0.20
    elif home.off_rtg_rank >= 26: off_rtg_adj -= 0.20
    if away.off_rtg_rank <= 5:  off_rtg_adj -= 0.20
    elif away.off_rtg_rank >= 26: off_rtg_adj += 0.20

    lambda_c = logit_base + def_rtg_adj + off_rtg_adj

    return {
        "delta_srs_weighted": round(delta_srs, 2),
        "logit_base":         round(logit_base, 4),
        "def_rtg_adj":        round(def_rtg_adj, 4),
        "off_rtg_adj":        round(off_rtg_adj, 4),
        "pace_diff":          round(home.pace - away.pace, 2),
        "lambda_c":           round(lambda_c, 4),
    }


# =============================================================================
# SECTION 5 — BLOC A : ROSTER & DISPONIBILITÉS (V11 — AMORTISSEUR USAGE)
# =============================================================================

def get_absence_lambda(absence: PlayerAbsence) -> float:
    """λ EPM pour un joueur absent (avant amortisseur)."""
    lam = 0.0
    for epm_min, epm_max, impact, _ in EPM_IMPACT_TABLE:
        if epm_min <= absence.epm < epm_max:
            lam = impact
            break
    if absence.usage_rate > 0.32 and absence.epm >= 3.0:
        lam -= 0.06                          # Usage ball-dominant → majoration
    if absence.status == "GTD" and not absence.confirmed:
        lam *= 0.50                          # GTD non confirmé → impact 50%
    return lam


def get_team_roster_impact(team: TeamData) -> tuple:
    """
    Calcule (lambda_brut, season_outs, usage_total) pour une équipe.
    V11 : amortisseur si usage cumulé absents > 40%.
    """
    raw_lambda    = sum(get_absence_lambda(a) for a in team.absences)
    total_usage   = team.total_usage_out
    season_outs   = sum(1 for a in team.absences if a.status == "OUT")
    gtd_count     = sum(1 for a in team.absences if a.status == "GTD")

    # ── Amortisseur d'usage cumulé (V11) ─────────────────────────────────────
    usage_dampened = False
    if total_usage > USAGE_CAP_THRESHOLD:
        raw_lambda   *= USAGE_CAP_FACTOR
        usage_dampened = True

    # ── Memphis Rule (≥6 OUT pour la saison) ─────────────────────────────────
    if season_outs >= 6:
        raw_lambda *= 1.5

    # ── Multi-GTD Penalty ─────────────────────────────────────────────────────
    if gtd_count >= 4:
        raw_lambda -= 0.24

    return raw_lambda, season_outs, total_usage, gtd_count, usage_dampened


def compute_bloc_a(home: TeamData, away: TeamData) -> dict:
    h_lam, h_outs, h_usage, h_gtd, h_damp = get_team_roster_impact(home)
    a_lam, a_outs, a_usage, a_gtd, a_damp = get_team_roster_impact(away)

    # λ_A net : absences away avantagent home → away_lambda - home_lambda
    lambda_a = a_lam - h_lam

    return {
        "lambda_a":           round(lambda_a, 4),
        "home_absence_lambda": round(h_lam, 4),
        "away_absence_lambda": round(a_lam, 4),
        "home_usage_out":     round(h_usage, 2),
        "away_usage_out":     round(a_usage, 2),
        "home_season_outs":   h_outs,
        "away_season_outs":   a_outs,
        "home_gtd_count":     h_gtd,
        "away_gtd_count":     a_gtd,
        "home_usage_dampened": h_damp,
        "away_usage_dampened": a_damp,
        "memphis_rule_home":  h_outs >= 6,
        "memphis_rule_away":  a_outs >= 6,
    }


# =============================================================================
# SECTION 6 — BLOC B : FORME RÉCENTE, MOMENTUM & SHOT QUALITY (V11)
# =============================================================================

L5_TABLE = {
    (5,0): +0.32, (4,1): +0.16, (3,2): 0.0,
    (2,3): -0.16, (1,4): -0.28, (0,5): -0.36,
}


def compute_bloc_b(team: TeamData) -> dict:
    """Forme + momentum pour une équipe, avec correction eFG% luck (V11)."""
    # ── Streak ────────────────────────────────────────────────────────────────
    streak = team.current_streak
    if streak > 0:
        streak_lambda = min(streak, 5) * 0.12
        if streak > 7:
            streak_lambda -= (streak - 7) * 0.08   # Streak fatigue
    else:
        streak_lambda = max(streak, -5) * 0.12

    # ── L5 ────────────────────────────────────────────────────────────────────
    wins, losses = team.l5_record
    l5_lambda = L5_TABLE.get((min(wins, 5), min(losses, 5)), 0.0)

    # ── Divergence L5 vs momentum ─────────────────────────────────────────────
    divergence_mult = 1.3 if (
        (streak > 0 and wins < 2) or (streak < 0 and wins > 3)
    ) else 1.0

    # ── Shot Quality Regression (V11) ─────────────────────────────────────────
    shooting_luck = 0.0
    efg_diff      = team.efg_last5 - team.efg_season
    if efg_diff > EFG_LUCK_THRESHOLD:
        shooting_luck = EFG_LUCK_PENALTY
        luck_label    = f"⚠️  Surperformance eFG% +{efg_diff:.1%} → regression attendue"
    elif efg_diff < -EFG_LUCK_THRESHOLD:
        shooting_luck = +0.10                 # Sous-performance → légère prime
        luck_label    = f"📉 Sous-performance eFG% {efg_diff:.1%} → rebond attendu"
    else:
        luck_label    = "✅ eFG% dans la norme"

    lambda_b = (streak_lambda + l5_lambda * divergence_mult) + shooting_luck

    return {
        "streak":          streak,
        "streak_lambda":   round(streak_lambda, 4),
        "l5_record":       team.l5_record,
        "l5_lambda":       round(l5_lambda, 4),
        "divergence_mult": divergence_mult,
        "efg_season":      round(team.efg_season, 3),
        "efg_last5":       round(team.efg_last5, 3),
        "efg_diff":        round(efg_diff, 3),
        "shooting_luck":   round(shooting_luck, 4),
        "luck_label":      luck_label,
        "lambda_b":        round(lambda_b, 4),
    }


def compute_bloc_b_net(home: TeamData, away: TeamData) -> dict:
    h = compute_bloc_b(home)
    a = compute_bloc_b(away)
    return {
        "home":          h,
        "away":          a,
        "lambda_b_net":  round(h["lambda_b"] - a["lambda_b"], 4),
    }


# =============================================================================
# SECTION 7 — BLOC D : CONTEXTE SITUATIONNEL
# =============================================================================

def get_home_advantage_lambda(abbr: str) -> float:
    return HOME_ADVANTAGE.get(abbr.upper(), HOME_ADVANTAGE["DEFAULT"])


def compute_bloc_d(home: TeamData, away: TeamData) -> dict:
    components = {}

    # Home/Away personnalisé
    home_adv = get_home_advantage_lambda(home.abbreviation)
    components["home_advantage"] = round(home_adv, 4)

    # Enjeu Playoff
    def playoff_lam(t: TeamData) -> float:
        if t.playoff_situation == "contending":  return +0.20
        if t.playoff_situation == "eliminated":  return -0.20
        return 0.0

    components["playoff_net"] = round(playoff_lam(home) - playoff_lam(away), 4)

    # B2B / Densité (mutuellement exclusifs)
    b2b_lambda     = 0.0
    density_lambda = 0.0

    if home.is_back_to_back: b2b_lambda -= 0.20
    if away.is_back_to_back: b2b_lambda += 0.20

    if not home.is_back_to_back and not away.is_back_to_back:
        if home.games_last_6_days >= 4: density_lambda -= 0.12
        elif home.games_last_6_days == 3: density_lambda -= 0.06
        if away.games_last_6_days >= 4: density_lambda += 0.12
        elif away.games_last_6_days == 3: density_lambda += 0.06

    components["b2b_lambda"]     = round(b2b_lambda, 4)
    components["density_lambda"] = round(density_lambda, 4)

    # Rest advantage
    rest_diff  = home.rest_days - away.rest_days
    rest_lam   = min(rest_diff * 0.10, 0.30)
    components["rest_days_diff"] = rest_diff
    components["rest_lambda"]    = round(rest_lam, 4)

    lambda_d = (
        home_adv
        + components["playoff_net"]
        + b2b_lambda
        + density_lambda
        + rest_lam
    )
    components["lambda_d"] = round(lambda_d, 4)
    return components


# =============================================================================
# SECTION 8 — BLOC E : MATCHUP & H2H
# =============================================================================

def compute_bloc_e(home: TeamData, away: TeamData,
                   h2h_era_weight: float = 1.0) -> dict:
    # H2H saison
    h2h_lambda = 0.0
    if home.h2h_games_season > 0:
        h2h_wins      = min(home.h2h_wins_season, 2)
        away_h2h_wins = min(home.h2h_games_season - home.h2h_wins_season, 2)
        h2h_lambda    = (h2h_wins - away_h2h_wins) * 0.10 * h2h_era_weight

    # Turnover rate différentiel
    to_diff   = home.to_rate - away.to_rate
    to_lambda = math.copysign(0.12, to_diff) if abs(to_diff) >= 4 else (
                math.copysign(0.06, to_diff) if abs(to_diff) >= 2 else 0.0)

    # Rebond offensif différentiel
    oreb_diff   = home.oreb_rate - away.oreb_rate
    oreb_lambda = math.copysign(0.08, oreb_diff) if abs(oreb_diff) >= 3 else (
                  math.copysign(0.04, oreb_diff) if abs(oreb_diff) >= 1.5 else 0.0)

    lambda_e = h2h_lambda + to_lambda + oreb_lambda

    return {
        "h2h_lambda":  round(h2h_lambda, 4),
        "to_diff":     round(to_diff, 2),
        "to_lambda":   round(to_lambda, 4),
        "oreb_diff":   round(oreb_diff, 2),
        "oreb_lambda": round(oreb_lambda, 4),
        "lambda_e":    round(lambda_e, 4),
    }


# =============================================================================
# SECTION 9 — ASSEMBLAGE LOG-ODDS FINAL
# =============================================================================

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_win_rate(match: MatchData) -> dict:
    home, away = match.home, match.away

    bloc_c = compute_bloc_c(home, away)
    bloc_a = compute_bloc_a(home, away)
    bloc_b = compute_bloc_b_net(home, away)
    bloc_d = compute_bloc_d(home, away)
    bloc_e = compute_bloc_e(home, away, match.h2h_era_weight)

    logit_final = (
        bloc_c["lambda_c"]
        + bloc_a["lambda_a"]
        + bloc_b["lambda_b_net"]
        + bloc_d["lambda_d"]
        + bloc_e["lambda_e"]
    )

    win_rate_raw = logistic(logit_final)
    win_rate     = win_rate_raw - SAFETY_MARGIN if win_rate_raw > 0.80 else win_rate_raw

    return {
        "bloc_c":         bloc_c,
        "bloc_a":         bloc_a,
        "bloc_b":         bloc_b,
        "bloc_d":         bloc_d,
        "bloc_e":         bloc_e,
        "logit_final":    round(logit_final, 4),
        "win_rate_raw":   round(win_rate_raw, 4),
        "win_rate":       round(win_rate, 4),
        "safety_applied": win_rate_raw > 0.80,
    }


# =============================================================================
# SECTION 10 — BLOC F : VALIDATION MARCHÉ (POST-CALCUL)
# =============================================================================

def validate_bloc_f(win_rate: float, odds_decimal: float,
                    line_movement: int = 0) -> dict:
    if odds_decimal <= 0:
        return {"ev": None, "ev_valid": False, "bet_valid": False,
                "signal": "⚠️ Cote non renseignée",
                "recommendation": "Renseigner la cote avant décision"}

    ev       = win_rate * odds_decimal
    ev_valid = ev > EV_THRESHOLD
    no_value = odds_decimal < 1.10

    signal = {1: "📈 Sharp money vers HOME",
              -1: "📉 Sharp money vers AWAY"}.get(line_movement, "➡️ Mouvement neutre")

    bet_valid = ev_valid and not no_value and win_rate >= WIN_RATE_THRESHOLD

    if no_value:
        rec = "⛔ Cote sans valeur (<1.10) → passer au spread"
    elif not win_rate >= WIN_RATE_THRESHOLD:
        rec = f"⛔ Win Rate {win_rate:.1%} < seuil {WIN_RATE_THRESHOLD:.0%}"
    elif not ev_valid:
        rec = f"⛔ EV {ev:.3f} < {EV_THRESHOLD} → pas de value"
    else:
        rec = (f"✅ VALID — Évaluer ATS (Win Rate {win_rate:.1%} > {BLOWOUT_THRESHOLD:.0%})"
               if win_rate > BLOWOUT_THRESHOLD
               else f"✅ VALID — ML recommandé (EV {ev:.3f})")

    return {"ev": round(ev, 4), "ev_valid": ev_valid, "signal": signal,
            "no_value": no_value, "bet_valid": bet_valid, "recommendation": rec}


# =============================================================================
# SECTION 11 — KELLY CRITERION ADAPTATIF
# =============================================================================

def compute_kelly(win_rate: float, odds_decimal: float,
                  unconfirmed_gtd_count: int = 0,
                  bankroll: float = BANKROLL) -> dict:
    b = odds_decimal - 1.0
    if b <= 0:
        return {"kelly_fraction": 0, "units": 0, "note": "Cote invalide"}

    kelly_full = max((win_rate * b - (1 - win_rate)) / b, 0.0)
    fraction   = KELLY_QUARTER if unconfirmed_gtd_count >= 1 else KELLY_HALF
    note       = (f"Quart-Kelly ({unconfirmed_gtd_count} GTD non confirmé(s))"
                  if unconfirmed_gtd_count >= 1 else "Demi-Kelly standard")

    units_raw  = kelly_full * fraction * bankroll
    units      = min(units_raw, KELLY_MAX_UNITS)

    return {
        "kelly_full": round(kelly_full, 4),
        "fraction":   fraction,
        "units_raw":  round(units_raw, 2),
        "units":      round(units, 2),
        "capped":     units_raw > KELLY_MAX_UNITS,
        "note":       note,
    }


# =============================================================================
# SECTION 12 — CALIBRATION PLATT SCALING
# =============================================================================

class PlattScaler:
    def __init__(self):
        self.fitted    = False
        self.model     = None
        self.n_samples = 0

    def fit(self, predictions: list, outcomes: list):
        if not SKLEARN_AVAILABLE:
            print("[WARN] sklearn requis")
            return
        if len(predictions) < PLATT_MIN_SAMPLES:
            print(f"[INFO] {len(predictions)}/{PLATT_MIN_SAMPLES} matchs — Platt non activé")
            return
        X          = [[p] for p in predictions]
        self.model = LogisticRegression()
        self.model.fit(X, outcomes)
        self.fitted    = True
        self.n_samples = len(predictions)
        a = self.model.coef_[0][0]
        bias = "overconfident" if a < 1 else "underconfident"
        print(f"[Platt] Actif | n={self.n_samples} | A={a:.3f} ({bias})")

    def calibrate(self, win_rate: float) -> float:
        if not self.fitted or self.model is None:
            return win_rate
        return round(float(self.model.predict_proba([[win_rate]])[0][1]), 4)

    def fit_from_history(self, history_file: str = HISTORY_FILE):
        """Charge et entraîne depuis le JSON de persistence."""
        if not os.path.exists(history_file):
            print(f"[INFO] {history_file} introuvable")
            return
        with open(history_file, "r") as f:
            data = [m for m in json.load(f) if m.get("outcome") is not None]
        if len(data) < PLATT_MIN_SAMPLES:
            print(f"[INFO] {len(data)}/{PLATT_MIN_SAMPLES} matchs avec outcome — Platt non activé")
            return
        preds    = [m["predicted_win_rate"] for m in data]
        outcomes = [m["outcome"] for m in data]
        self.fit(preds, outcomes)


platt_scaler = PlattScaler()


# =============================================================================
# SECTION 13 — FILTRES ÉLIMINATOIRES
# =============================================================================

def apply_elimination_filters(match: MatchData, bloc_a: dict,
                               win_rate: float) -> dict:
    filters = {}

    # Roster vide
    home_total = bloc_a["home_season_outs"] + bloc_a["home_gtd_count"]
    away_total = bloc_a["away_season_outs"] + bloc_a["away_gtd_count"]
    roster_vide = (home_total >= 5) and (away_total >= 5)
    filters["roster_vide"] = {
        "triggered": roster_vide,
        "detail": f"Home {home_total} absents | Away {away_total} absents",
        "action": "⛔ SKIP — rosters insuffisants" if roster_vide else "✅ OK",
    }

    # Seuil Win Rate
    wr_ok = win_rate >= WIN_RATE_THRESHOLD
    filters["win_rate_threshold"] = {
        "triggered": not wr_ok,
        "detail": f"Win Rate {win_rate:.1%} vs seuil {WIN_RATE_THRESHOLD:.0%}",
        "action": "✅ OK" if wr_ok else "⛔ SKIP — Win Rate insuffisant",
    }

    # Star Exception (GTD EPM >5 non confirmé)
    star_gtd = [a.name for a in match.home.absences + match.away.absences
                if a.epm >= 5.0 and a.status == "GTD" and not a.confirmed]
    filters["star_exception"] = {
        "triggered": bool(star_gtd),
        "players":   star_gtd,
        "action":    f"⏳ CONDITIONNEL — attendre {star_gtd}" if star_gtd else "✅ OK",
    }

    # validate_team_abbr
    filters["validate_team_abbr"] = {
        "home":   match.home.abbreviation,
        "away":   match.away.abbreviation,
        "action": "✅ Équipes identifiées",
    }

    # B2B / Densité exclusivité
    b2b_active = match.home.is_back_to_back or match.away.is_back_to_back
    filters["b2b_density_exclusive"] = {
        "b2b_active": b2b_active,
        "action": "✅ Densité désactivée (B2B actif)" if b2b_active else "✅ Densité active",
    }

    hard_skip  = roster_vide or not wr_ok
    conditional = bool(star_gtd)

    return {"filters": filters, "hard_skip": hard_skip,
            "conditional": conditional, "proceed": not hard_skip}


# =============================================================================
# SECTION 14 — RAPPORT FINAL, OUTPUT & PERSISTENCE JSON (V11)
# =============================================================================

def _save_to_history(match: MatchData, final_win_rate: float):
    """Enregistre la prédiction dans nba_history.json pour Platt Scaling futur."""
    entry = {
        "date":               match.date,
        "teams":              f"{match.home.abbreviation} vs {match.away.abbreviation}",
        "predicted_win_rate": round(final_win_rate, 4),
        "odds":               match.odds_home,
        "outcome":            None,   # À remplir : 1 = home gagne, 0 = away
        "recorded_at":        datetime.now().strftime("%Y-%m-%dT%H:%M"),
    }
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def run_model(match: MatchData, save_history: bool = True) -> dict:
    """Pipeline complet V11 — retourne le rapport final structuré."""
    print(f"\n{'='*65}")
    print(f"  NBA MODEL V11 — {match.home.name} vs {match.away.name}")
    print(f"  {match.date}  |  Tip-off {match.tip_off_cet} CET")
    print(f"{'='*65}")

    wr_result  = compute_win_rate(match)
    win_rate   = wr_result["win_rate"]

    # Platt Scaling (si actif)
    win_rate_calibrated = platt_scaler.calibrate(win_rate)
    final_win_rate      = win_rate_calibrated

    # Filtres éliminatoires
    elimination = apply_elimination_filters(match, wr_result["bloc_a"], final_win_rate)

    # Bloc F
    bloc_f = validate_bloc_f(final_win_rate, match.odds_home, match.line_movement_sharp)

    # Kelly
    unconfirmed = match.home.gtd_unconfirmed_count + match.away.gtd_unconfirmed_count
    kelly = compute_kelly(final_win_rate, match.odds_home, unconfirmed) if match.odds_home > 0 else None

    # Étoiles
    if final_win_rate >= 0.90:   stars = "⭐⭐⭐⭐⭐"
    elif final_win_rate >= 0.80: stars = "⭐⭐⭐⭐"
    elif final_win_rate >= 0.72: stars = "⭐⭐⭐"
    elif final_win_rate >= 0.62: stars = "⭐⭐"
    else:                        stars = "⭐ (sous seuil)"

    # ── Affichage ─────────────────────────────────────────────────────────────
    print(f"\n📊 COMPOSANTES LOG-ODDS")
    print(f"  λ_C (SRS/Qualité)        : {wr_result['bloc_c']['lambda_c']:+.4f}")
    print(f"  λ_A (Roster/EPM V11)     : {wr_result['bloc_a']['lambda_a']:+.4f}")
    if wr_result["bloc_a"]["home_usage_dampened"]:
        print(f"      ⚡ Amortisseur HOME appliqué (usage >{USAGE_CAP_THRESHOLD:.0%})")
    if wr_result["bloc_a"]["away_usage_dampened"]:
        print(f"      ⚡ Amortisseur AWAY appliqué (usage >{USAGE_CAP_THRESHOLD:.0%})")
    print(f"  λ_B (Forme+eFG% V11)     : {wr_result['bloc_b']['lambda_b_net']:+.4f}")
    print(f"      Home: {wr_result['bloc_b']['home']['luck_label']}")
    print(f"      Away: {wr_result['bloc_b']['away']['luck_label']}")
    print(f"  λ_D (Contexte)           : {wr_result['bloc_d']['lambda_d']:+.4f}")
    print(f"  λ_E (Matchup/H2H)        : {wr_result['bloc_e']['lambda_e']:+.4f}")
    print(f"  {'─'*43}")
    print(f"  logit_final              : {wr_result['logit_final']:+.4f}")
    print(f"  Win Rate brut            : {wr_result['win_rate_raw']:.1%}")
    if wr_result["safety_applied"]:
        print(f"  Safety margin -3%        : appliquée")
    if platt_scaler.fitted:
        print(f"  Platt calibré            : {win_rate_calibrated:.1%} ({platt_scaler.n_samples} matchs)")

    print(f"\n🎯 WIN RATE FINAL          : {final_win_rate:.1%}  {stars}")

    print(f"\n🔍 FILTRES ÉLIMINATOIRES")
    for k, v in elimination["filters"].items():
        print(f"  {k:<30} → {v['action']}")

    print(f"\n💹 BLOC F — VALIDATION MARCHÉ")
    if match.odds_home > 0:
        print(f"  Cote         : {match.odds_home:.2f}")
        print(f"  EV           : {bloc_f['ev']:.4f} {'✅' if bloc_f['ev_valid'] else '❌'}")
    print(f"  Line mvmt    : {bloc_f['signal']}")
    print(f"  → {bloc_f['recommendation']}")

    if kelly and bloc_f["bet_valid"] and not elimination["hard_skip"]:
        print(f"\n💰 KELLY SIZING")
        print(f"  Kelly brut   : {kelly['kelly_full']:.1%}")
        print(f"  Fraction     : {kelly['fraction']} ({kelly['note']})")
        print(f"  Mise         : {kelly['units']:.1f}u / {BANKROLL:.0f}u bankroll")
        if kelly["capped"]:
            print(f"  ⚠️  Plafond {KELLY_MAX_UNITS:.0f}u appliqué")

    # Persistence JSON
    if save_history:
        _save_to_history(match, final_win_rate)
        print(f"\n💾 Sauvegardé → {HISTORY_FILE}")

    print(f"\n{'='*65}\n")

    return {
        "match":          f"{match.home.name} vs {match.away.name}",
        "tip_off":        match.tip_off_cet,
        "win_rate":       final_win_rate,
        "logit_final":    wr_result["logit_final"],
        "stars":          stars,
        "bet_valid":      bloc_f["bet_valid"] and not elimination["hard_skip"],
        "conditional":    elimination["conditional"],
        "hard_skip":      elimination["hard_skip"],
        "ev":             bloc_f.get("ev"),
        "kelly_units":    kelly["units"] if kelly else None,
        "recommendation": bloc_f["recommendation"],
        "blocs":          wr_result,
        "bloc_f":         bloc_f,
        "elimination":    elimination,
    }


# =============================================================================
# SECTION 15 — INTERFACE MATCH (SECTION À MODIFIER À CHAQUE ANALYSE)
# =============================================================================
# Sources :
#   SRS           → basketball-reference.com > Team Stats > SRS
#   EPM           → dunksandthrees.com
#   BPM proxy     → basketball-reference.com > Advanced Stats
#   TO forcés     → NBA.com > Team Stats > Opp TOV
#   OREB/match    → NBA.com > Team Stats > OREB
#   eFG% saison   → NBA.com > Team Stats > eFG%
#   eFG% L5       → basketball-reference.com > Game Logs (moyenne des 5 derniers)
#   Cotes         → Bookmaker direct (vérifier 30 min avant tip-off)
# =============================================================================

if __name__ == "__main__":

    # Charger Platt si historique disponible
    platt_scaler.fit_from_history()

    # ── EXEMPLE : Denver Nuggets vs Utah Jazz ─────────────────────────────────
    match_den_uta = MatchData(
        date                 = "2026-03-28",
        tip_off_cet          = "02h00",
        odds_home            = 1.25,
        line_movement_sharp  = 0,
        h2h_era_weight       = 0.85,

        home = TeamData(
            name               = "Denver Nuggets",
            abbreviation       = "DEN",
            srs_last30         = +6.8,
            srs_season         = +5.9,
            net_rtg            = +7.2,
            off_rtg_rank       = 4,
            def_rtg_rank       = 8,
            pace               = 99.2,
            absences           = [],
            current_streak     = 4,
            l5_record          = (4, 1),
            efg_season         = 0.558,
            efg_last5          = 0.562,   # Légèrement au-dessus → dans la norme
            is_home            = True,
            is_back_to_back    = False,
            games_last_6_days  = 2,
            rest_days          = 2,
            playoff_situation  = "contending",
            to_rate            = 14.1,
            oreb_rate          = 11.8,
            h2h_wins_season    = 1,
            h2h_games_season   = 2,
        ),

        away = TeamData(
            name               = "Utah Jazz",
            abbreviation       = "UTA",
            srs_last30         = -8.5,
            srs_season         = -7.1,
            net_rtg            = -8.9,
            off_rtg_rank       = 28,
            def_rtg_rank       = 27,
            pace               = 97.8,
            absences           = [
                PlayerAbsence("Lauri Markkanen", epm=3.8, usage_rate=0.28, status="OUT", confirmed=True),
                PlayerAbsence("Jusuf Nurkic",    epm=1.2, usage_rate=0.22, status="OUT", confirmed=True),
                PlayerAbsence("Walker Kessler",  epm=1.5, usage_rate=0.18, status="OUT", confirmed=True),
                PlayerAbsence("John Collins",    epm=0.8, usage_rate=0.20, status="OUT", confirmed=True),
                PlayerAbsence("Jordan Clarkson", epm=0.6, usage_rate=0.25, status="OUT", confirmed=True),
                PlayerAbsence("Keyonte George",  epm=0.4, usage_rate=0.22, status="OUT", confirmed=True),
            ],
            current_streak     = -3,
            l5_record          = (1, 4),
            efg_season         = 0.521,
            efg_last5          = 0.498,   # Sous-performance → rebond attendu
            is_home            = False,
            is_back_to_back    = False,
            games_last_6_days  = 3,
            rest_days          = 1,
            playoff_situation  = "eliminated",
            to_rate            = 11.2,
            oreb_rate          = 8.9,
            h2h_wins_season    = 1,
            h2h_games_season   = 2,
        ),
    )

    result = run_model(match_den_uta)
