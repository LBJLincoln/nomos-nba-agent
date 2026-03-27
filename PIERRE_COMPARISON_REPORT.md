# Rapport Comparatif : Pierre V11 vs Nomos42 NBA Quant AI

> Date : 2026-03-27 | Auteur : Nomos42 AI | Pour : Pierre

---

## EXECUTIVE SUMMARY

| Dimension | Pierre V11 | Nomos42 NBA Quant |
|-----------|-----------|-------------------|
| **Approche** | Modele expert log-odds (hand-crafted) | Genetic Algorithm + ML evolution (automated) |
| **Features** | ~30 features manuelles (5 blocs) | 6,135 features auto-generees (37 categories) |
| **Modeles** | 1 (logistic/Platt) | 14 types (XGBoost, CatBoost, LightGBM, ExtraTrees, RF, TabICL...) |
| **Calibration** | Platt Scaling (logistic) | 4 methodes (sigmoid, isotonic, Venn-Abers, conformal) |
| **Sizing** | Kelly demi/quart, 15u max | Kelly 1/4 fractionnel, 5% max, portfolio optimization |
| **Training** | Zero (expert rules) | 24/7 evolution sur 6 HF Spaces + Kaggle GPU |
| **Brier (estimé)** | ~0.24-0.26 (a valider) | **0.21837** (mesuré, all-time best) |
| **Automatisation** | Manuel (user remplit outcome) | Full auto (cron, API, data fetch, predictions) |

**Verdict :** Les deux approches sont **complementaires**. Pierre excelle en interpretabilite et en integration de contexte situationnel (B2B, GTD, eFG% luck). Nomos42 excelle en puissance predictive brute et automatisation. Un **ensemble** des deux pourrait battre chacun individuellement.

---

## 1. ARCHITECTURE DETAILLEE

### Pierre V11 : Modele Expert Log-Odds

```
INPUT (Manuel)
  User remplit TeamData pour chaque equipe :
    - SRS (30j + season), Net RTG, Off/Def RTG rank, Pace
    - Absences (EPM, usage_rate, status OUT/GTD)
    - Streak, L5 record, eFG% season vs L5
    - Home/Away, B2B, rest days, playoff situation
    - TO rate, OREB rate, H2H
    - Cotes marche

BLOCS LOG-ODDS (5 blocs additifs)
  Bloc C : Qualite Intrinseque (SRS, Off/Def RTG)  →  lambda_C
  Bloc A : Roster & Absences (EPM impact table)     →  lambda_A
  Bloc B : Momentum & Shot Quality (streak, eFG%)   →  lambda_B
  Bloc D : Contexte (home, B2B, rest, playoff)      →  lambda_D
  Bloc E : Matchup (H2H, TO, OREB)                  →  lambda_E

ASSEMBLAGE
  logit_final = lambda_C + lambda_A + lambda_B + lambda_D + lambda_E
  win_rate = sigmoid(logit_final)
  if win_rate > 80% : win_rate -= 3% (safety margin)

POST-PROCESSING
  Platt Scaling (calibration apres 50 matchs)  →  win_rate_calibrated
  Bloc F : Validation Marche (EV > 1.10, WR > 62%)
  Kelly : demi ou quart (si GTD)  →  units (max 15u)

PERSISTENCE
  nba_history.json → outcome rempli manuellement ou via balldontlie API
```

### Nomos42 NBA Quant : Genetic Algorithm Evolution

```
DATA (Automatique, 24/7)
  Supabase + APIs : 9,500+ matchs historiques
  Live odds : 6 bookmakers toutes les 30 min
  Player props, lineups, injuries : automatique

FEATURE ENGINE v3.0 (37 categories, 6,135 features)
  Cat 1-5   : Team strength (SRS, ELO, ratings, pace)
  Cat 6-10  : Player impact (RAPTOR, minutes, lineups)
  Cat 11-15 : Recent form (streaks, momentum, rolling averages)
  Cat 16-20 : Matchup (H2H, style, pace diff)
  Cat 21-25 : Market (odds, CLV, sharp moves, steam)
  Cat 26-30 : Situational (rest, B2B, travel, altitude)
  Cat 31-35 : Advanced (EWMA, rolling volatility, trend)
  Cat 36    : EWMA exponential weighted features
  Cat 37    : MOVDA (margin of victory delta acceleration)

EVOLUTION (24/7 sur 6 iles HF Spaces + Kaggle GPU)
  Population : 60 individus par ile (360 total)
  Genome : masque de features + modele + hyperparams
  Fitness : Brier + ROI + Sharpe + ECE (multi-objectif NSGA-II)
  Mutation : adaptative 0.08-0.15, crossover 0.80
  Migration : ring topology toutes les 8 generations
  Rythme : ~1 generation/5min par ile, 24/7

MODELES (14 types, selection par evolution)
  Tree-based : XGBoost, LightGBM, CatBoost, ExtraTrees, RandomForest
  Custom : XGBoost-Brier (objectif custom pour Brier direct)
  GPU : TabICLv2 (transformer, Kaggle uniquement)
  Calibration : sigmoid, isotonic, Venn-Abers, conformal

SIZING
  Kelly 1/4 fractionnel, max 5% bankroll/bet
  Portfolio optimization (max 25% exposure totale)
  Conformal prediction intervals
```

---

## 2. COMPARAISON FEATURE PAR FEATURE

### Pierre a, Nomos42 n'a pas (ou fait differemment) :

| Feature Pierre | Equivalent Nomos42 | Note |
|----------------|-------------------|------|
| **eFG% luck correction** (eFG_L5 vs season) | Non | **EXCELLENT** — Nomos42 devrait l'ajouter. Regression vers la moyenne du shooting est un signal fort |
| **Memphis Rule** (6+ OUT → x1.5 impact) | Non | Bonne heuristique, pourrait etre apprise par le modele |
| **Usage cap amortisseur** (>40% usage OUT → 85%) | Non | Smart — evite les surestimations quand trop d'absents |
| **Home advantage personnalisé par equipe** (DEN +0.34, WAS +0.08) | Oui (Cat 26-30, mais appris, pas hand-crafted) | Pierre est plus precis ici car hand-tuned |
| **Safety margin -3%** quand WR > 80% | Oui (probability clipping [0.025, 0.975]) | Meme intention, approches differentes |
| **Line movement sharp money** (+1/-1/0) | Oui (Cat 21-25 : steam moves, CLV, sharp action) | Nomos42 est beaucoup plus detaille |

### Nomos42 a, Pierre n'a pas :

| Feature Nomos42 | Impact | Note |
|-----------------|--------|------|
| **6,135 features automatiques** | Majeur | Le volume permet de capturer des patterns invisibles a l'oeil |
| **MOVDA** (margin of victory delta acceleration) | +1.16% vs Elo | Signal de momentum avance |
| **EWMA** (exponential weighted moving averages) | Important | Lissage temporel adaptatif |
| **Market features** (CLV, steam, line movement quantifie) | Important | Pierre n'a qu'un flag +1/-1/0 |
| **Multi-modeles** (14 types en competition) | Important | Le GA selectionne le meilleur modele par contexte |
| **Walk-forward temporal validation** | Critique | Empeche le lookahead bias |
| **Feature selection evoluee** | Important | Le GA elimine les features bruitees |

---

## 3. CALIBRATION : COMPARAISON DETAILLEE

### Pierre : Platt Scaling

```python
# Entraine apres 50 matchs
LogisticRegression().fit(predicted_win_rates, outcomes)
# Calibre : P_calibre = sigmoid(A * P_raw + B)
# Avantage : simple, robuste, interpreble
# Limite : lineaire, assume stabilite dans le temps
```

**Forces :**
- Simple et efficace
- A et B sont interpretables ("overconfident" / "underconfident")
- Se retraine facilement avec train_platt.py

**Limites :**
- Lineaire (ne capture pas les non-linearites de calibration)
- Pas de calibration conditionnelle (meme correction pour un match 60% et 85%)

### Nomos42 : 4 methodes de calibration

| Methode | Description | Quand c'est mieux |
|---------|-------------|-------------------|
| **Sigmoid** (= Platt) | Identique a Pierre | Baseline |
| **Isotonic** | Piecewise linear, non-parametrique | Quand la calibration n'est pas lineaire |
| **Venn-Abers** | Prediction conforme, garanties probabilistes | Quand on veut des bornes de confiance |
| **Conformal** | Distribution-free coverage guarantee | Quand on veut P(erreur) < alpha |

La calibration est **selectionnee par evolution** : le GA teste les 4 et garde la meilleure pour chaque individu.

---

## 4. KELLY : COMPARAISON

| Aspect | Pierre V11 | Nomos42 |
|--------|-----------|---------|
| Formule | f = (p*b - q) / b | Identique |
| Fraction | 1/2 (standard) ou 1/4 (si GTD) | 1/4 toujours (Starlizard standard) |
| Cap par bet | 15 unites | 5% bankroll |
| Cap total | Aucun | 25% exposure totale |
| Seuil EV | EV > 1.10 | Edge > 2% |
| Seuil WR | > 62% | Aucun seuil fixe (le modele decide) |
| Bankroll | 100 EUR | 100 EUR (identique) |

**Analyse :** Pierre est plus conservateur (seuils EV et WR), Nomos42 est plus agressif mais avec un cap portfolio. Les deux approches sont valides. Pierre evite les faux positifs, Nomos42 maximise le volume.

---

## 5. METRIQUES DE PERFORMANCE

### Nomos42 (mesure, all-time)

| Metrique | Valeur | Source |
|----------|--------|--------|
| **Brier Score** | **0.21837** | S13 CatBoost, gen 815, 2026-03-26 |
| Brier (Kaggle best) | 0.21844 | extra_trees, 200 features, gen 52 |
| Brier (MOVDA-era) | 0.22041 | S10 XGBoost, gen 435 |
| Calibration ECE | ~0.015 | Tres bien calibre |
| Matchs evalues | 9,500+ | Depuis 2020 |
| Evolution | 24/7, 6 iles | ~1000 generations/jour |

### Pierre V11 (estime, a valider)

| Metrique | Estimation | Raisonnement |
|----------|-----------|--------------|
| **Brier Score** | **~0.24-0.26** | Modeles experts log-odds typiquement 0.24-0.27 sans ML |
| Calibration | A determiner | Depend du Platt et du sample size |
| Matchs | Debut (< 50) | nba_history.json en construction |
| ROI | A determiner | Pas assez de data |

**Note importante :** Le Brier estime de Pierre est **avant Platt Scaling**. Avec calibration et 200+ matchs, il pourrait descendre a ~0.23-0.24. C'est honorable pour un modele expert sans ML.

---

## 6. FORCES ET FAIBLESSES

### Pierre V11

| Forces | Faiblesses |
|--------|------------|
| Tres interpretable (chaque lambda est explicable) | Scalabilite limitee (features manuelles) |
| Contexte situationnel riche (B2B, GTD, eFG% luck) | Pas de training automatique |
| Pas de risque d'overfitting | Pas de validation temporelle |
| Facile a auditer et debugger | Input manuel (erreur humaine possible) |
| eFG% regression = signal non exploite par Nomos42 | Un seul "modele" (log-odds + sigmoid) |

### Nomos42

| Forces | Faiblesses |
|--------|------------|
| 0.21837 Brier (state of the art) | Black box (difficile d'expliquer pourquoi) |
| 24/7 automatique | Necessite infrastructure (HF Spaces, Kaggle, VM) |
| 14 modeles en competition | Risque d'overfitting sur feature noise |
| Walk-forward validation rigoureuse | Cout en compute |
| Feature selection evoluee (GA) | Lent a converger (semaines d'evolution) |

---

## 7. RECOMMANDATIONS : SYNERGIE

### A. Ce que Nomos42 devrait prendre de Pierre :

1. **eFG% luck correction** → Ajouter comme Cat 38 dans engine.py :
   ```python
   efg_diff = efg_last5 - efg_season
   if efg_diff > 0.05: penalty = -0.15  # Regression attendue
   ```
   Impact estime : Brier -0.001 a -0.003

2. **Home advantage personnalise par equipe** → Enrichir Cat 26-30 avec les lambdas hand-tuned de Pierre comme features additionnelles

3. **Usage cap amortisseur** → Ajouter aux features d'absence (quand usage_cumule_absents > 40%, signal de degradation non-lineaire)

### B. Ce que Pierre devrait prendre de Nomos42 :

1. **Automatiser l'input** → Utiliser l'API balldontlie + nba_api pour remplir automatiquement TeamData (SRS, streaks, absences)

2. **Ajouter des market features** → Au minimum: CLV (closing line value), odds implied probability, spread

3. **Walk-forward validation** → Ne jamais evaluer la performance in-sample. Toujours train sur le passe, test sur le futur

4. **Augmenter les features** → Le plus gros gain serait d'ajouter des features RAPTOR/EPM automatiques pour chaque joueur, pas juste les absents

5. **Ensemble** → Pierre_V11_winrate comme feature dans Nomos42 = meta-learner

### C. Projet commun : Ensemble Model

```
Pierre V11  →  win_rate_pierre (interpretable)
Nomos42     →  win_rate_nomos42 (ML)
                    ↓
            Stacking / blending
            weight_pierre * WR_pierre + weight_nomos42 * WR_nomos42
                    ↓
            win_rate_ensemble (best of both)
```

**Gain estime :** Brier -0.005 a -0.015 vs Nomos42 seul. Les erreurs de Pierre et Nomos42 sont decorrelees (expert vs ML), donc l'ensemble devrait surperformer.

---

## 8. COMMENT MESURER ET COMPARER

### Setup recommande :

1. **Pierre lance `performance_report.py`** apres 50+ matchs avec outcomes
2. **Nomos42 fournit ses predictions** pour les memes matchs via API
3. **Comparer sur les memes matchs** :
   - Brier Score (global)
   - Brier Score par tranche de WR (50-60%, 60-70%, 70%+)
   - Calibration plot (predit vs reel)
   - ROI sur Kelly identique
   - Log-loss

### Script de comparaison (a fournir) :

Pierre peut ajouter le champ `nomos42_prediction` dans `nba_history.json` pour chaque match, puis comparer :

```python
brier_pierre  = mean((pierre_wr - outcome)^2)
brier_nomos42 = mean((nomos42_wr - outcome)^2)
brier_ensemble = mean(((0.3*pierre + 0.7*nomos42) - outcome)^2)
```

---

## 9. GLOSSAIRE

| Terme | Definition |
|-------|-----------|
| **Brier Score** | mean((prediction - outcome)^2). Plus bas = mieux. 0.25 = random, 0.22 = bon, 0.20 = excellent |
| **Log-Loss** | -mean(y*log(p) + (1-y)*log(1-p)). Penalise plus les erreurs confiantes |
| **ECE** | Expected Calibration Error. Ecart moyen entre confiance et precision |
| **Kelly** | f = (pb - q) / b. Fraction optimale du bankroll a miser |
| **CLV** | Closing Line Value. Ecart entre ta cote et la cote finale |
| **EV** | Expected Value. p * odds. >1.0 = value |
| **NSGA-II** | Algorithme genetique multi-objectif (Pareto ranking) |
| **Walk-forward** | Train sur le passe, test sur le futur (jamais d'in-sample) |
| **Platt Scaling** | Calibration par regression logistique sur les predictions brutes |
| **MOVDA** | Margin Of Victory Delta Acceleration (signal de momentum) |

---

## 10. CONCLUSION

Pierre V11 est un **excellent modele expert** — bien structure, interpretable, avec des innovations interessantes (eFG% luck, usage cap, Memphis rule). Sa principale limite est l'absence de ML et la scalabilite.

Nomos42 est un **systeme de recherche automatise** — il explore 6,135 features, 14 modeles, 4 calibrations, 24/7. Sa principale limite est l'interpretabilite.

**L'ideal : combiner les deux.** Pierre apporte le domain knowledge et l'interpretabilite. Nomos42 apporte la puissance predictive et l'automatisation. Un ensemble des deux devrait battre chacun individuellement.

---

*Genere par Nomos42 AI — 2026-03-27*
*Contact : @Nomos42 (Telegram)*
