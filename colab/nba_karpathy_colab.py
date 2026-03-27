# %% [markdown]
# # NBA Quant AI — Karpathy Autoresearch Loop (Google Colab)
#
# **Pattern:** [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
# - Each iteration: modify config -> train 5min -> measure Brier -> keep if better -> loop
# - 12 iterations/hour, ~100/session
# - Google Drive persistence (survives disconnects)
# - TabICL GPU model + tree-based models
#
# **Before running:** Runtime -> GPU (T4) -> Secrets: `HF_TOKEN`, `DATABASE_URL`
#
# **Target:** Beat ATR 0.21837

# %%
# ═══════════════════════════════════════
# CELL 1: SETUP + DRIVE + DEPS
# ═══════════════════════════════════════
import subprocess, sys, os, time, gc, json, warnings, random, math
import numpy as np
warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
DRIVE_DIR = '/content/drive/MyDrive/nba-quant-karpathy'
os.makedirs(DRIVE_DIR, exist_ok=True)

# Secrets
from google.colab import userdata
for key in ['HF_TOKEN', 'DATABASE_URL']:
    try:
        v = userdata.get(key)
        if v: os.environ[key] = v; print(f'{key}: OK')
    except: print(f'{key}: not set')

# Install deps
t0 = time.time()
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'xgboost', 'lightgbm', 'catboost', 'scikit-learn', 'tabicl',
    'psycopg2-binary', 'huggingface_hub'])
print(f'Deps: {time.time()-t0:.0f}s')

import torch
print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Pre-cache TabICL
try:
    from tabicl import TabICLClassifier
    _m = TabICLClassifier()
    _m.fit(np.random.randn(50,5), np.random.randint(0,2,50))
    del _m; gc.collect()
    print('TabICL: cached')
except Exception as e:
    print(f'TabICL: {e}')

# %%
# ═══════════════════════════════════════
# CELL 2: BUILD OR LOAD FEATURES
# ═══════════════════════════════════════
from pathlib import Path
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

FEATURE_CACHE = Path(f'{DRIVE_DIR}/features_cache_v38.npz')

if FEATURE_CACHE.exists():
    print('Loading cached features from Drive...')
    data = np.load(str(FEATURE_CACHE), allow_pickle=True)
    X, y, feature_names = data['X'], data['y'], list(data['feature_names'])
else:
    print('Building features (~30 min, cached to Drive after)...')
    if not os.path.exists('/content/nomos-nba-agent'):
        subprocess.run(['git', 'clone', '--depth=1',
            f'https://{os.environ.get("HF_TOKEN","")}@github.com/LBJLincoln/nomos-nba-agent.git',
            '/content/nomos-nba-agent'], check=True)
    sys.path.insert(0, '/content/nomos-nba-agent/hf-space')
    from features.engine import NBAFeatureEngine
    games = []
    for f in sorted(Path('/content/nomos-nba-agent/hf-space/data/historical').glob('games-*.json')):
        raw = json.loads(f.read_text())
        if isinstance(raw, list): games.extend(raw)
        elif isinstance(raw, dict) and 'games' in raw: games.extend(raw['games'])
    print(f'Games: {len(games)}')
    engine = NBAFeatureEngine()
    X_raw, y_raw, feature_names = engine.build(games)
    X = np.nan_to_num(np.array(X_raw, dtype=np.float32))
    y = np.array(y_raw, dtype=np.int32)
    var = np.var(X, axis=0)
    valid = var > 1e-10
    X, feature_names = X[:, valid], [f for f, v in zip(feature_names, valid) if v]
    np.savez_compressed(str(FEATURE_CACHE), X=X, y=np.array(y), feature_names=np.array(feature_names))
    print(f'Cached: {X.shape}')

# Subsample
if X.shape[0] > 6000:
    X, y = X[-6000:], y[-6000:]
print(f'Ready: {X.shape}')

# %%
# ═══════════════════════════════════════
# CELL 3: KARPATHY LOOP ENGINE
# ═══════════════════════════════════════
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import signal, urllib.request, ssl

STATE_FILE = Path(f'{DRIVE_DIR}/karpathy_state.json')
RESULTS_FILE = Path(f'{DRIVE_DIR}/result.json')
LOG_FILE = Path(f'{DRIVE_DIR}/experiment_log.jsonl')

# ═══ IMMUTABLE EVALUATION HARNESS ═══
_xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_model(model_type, hp):
    if model_type == 'tabicl':
        from tabicl import TabICLClassifier
        return TabICLClassifier()
    elif model_type == 'xgboost':
        return xgb.XGBClassifier(max_depth=hp.get('depth', 6), learning_rate=hp.get('lr', 0.1),
            n_estimators=hp.get('n_est', 200), random_state=42, verbosity=0,
            tree_method='hist', device=_xgb_device)
    elif model_type == 'xgboost_brier':
        def brier_obj(y_true, y_pred):
            grad = 2 * (y_pred - y_true)
            hess = np.full_like(grad, 2.0)
            return grad, hess
        return xgb.XGBClassifier(max_depth=hp.get('depth', 6), learning_rate=hp.get('lr', 0.1),
            n_estimators=hp.get('n_est', 200), random_state=42, objective=brier_obj,
            verbosity=0, tree_method='hist', device=_xgb_device)
    elif model_type == 'lightgbm':
        return lgbm.LGBMClassifier(max_depth=hp.get('depth', 6), learning_rate=hp.get('lr', 0.1),
            n_estimators=hp.get('n_est', 200), random_state=42, verbose=-1)
    elif model_type == 'catboost':
        return CatBoostClassifier(depth=min(hp.get('depth', 6), 10), learning_rate=hp.get('lr', 0.1),
            iterations=hp.get('n_est', 200), random_state=42, verbose=0, task_type='GPU')
    elif model_type == 'extra_trees':
        return ExtraTreesClassifier(n_estimators=hp.get('n_est', 200),
            max_depth=hp.get('depth', None), random_state=42, n_jobs=-1)
    else:
        return RandomForestClassifier(n_estimators=hp.get('n_est', 200), random_state=42, n_jobs=-1)

class _Timeout(Exception): pass
def _timeout_handler(s, f): raise _Timeout()

def evaluate(mask, model_type, hp, timeout=120):
    selected = np.where(mask)[0]
    if len(selected) < 5 or len(selected) > 200: return 1.0
    X_sub = X[:, selected]
    tscv = TimeSeriesSplit(n_splits=2)

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        briers = []
        for tr, te in tscv.split(X_sub):
            model = make_model(model_type, hp)
            model.fit(X_sub[tr], y[tr])
            probs = model.predict_proba(X_sub[te])[:, 1]
            briers.append(brier_score_loss(y[te], probs))
            del model
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return float(np.mean(briers))
    except _Timeout:
        signal.signal(signal.SIGALRM, old); return 0.30
    except:
        signal.alarm(0); signal.signal(signal.SIGALRM, old); return 1.0

# ═══ CONFIG (modifiable) ═══
CONFIG = {
    'population_size': 30,
    'iteration_budget_sec': 300,
    'mutation_rate': 0.09,
    'crossover_rate': 0.80,
    'target_features': 63,
    'model_types': ['tabicl', 'xgboost', 'xgboost_brier', 'extra_trees', 'catboost', 'lightgbm', 'random_forest'],
    'model_weights': [0.30, 0.15, 0.15, 0.15, 0.10, 0.08, 0.07],
}

def random_individual():
    n = X.shape[1]
    mask = np.zeros(n, dtype=bool)
    sel = np.random.choice(n, size=min(CONFIG['target_features'], n), replace=False)
    mask[sel] = True
    mt = np.random.choice(CONFIG['model_types'], p=CONFIG['model_weights'])
    hp = {'depth': random.randint(4, 10), 'lr': round(random.uniform(0.01, 0.3), 3),
          'n_est': random.randint(100, 500)}
    return {'mask': mask, 'model_type': mt, 'hp': hp, 'brier': 1.0}

def mutate(ind):
    new = {'mask': ind['mask'].copy(), 'model_type': ind['model_type'],
           'hp': dict(ind['hp']), 'brier': 1.0}
    n_flip = max(1, int(CONFIG['mutation_rate'] * np.sum(new['mask'])))
    for _ in range(n_flip):
        idx = random.randint(0, len(new['mask'])-1)
        new['mask'][idx] = not new['mask'][idx]
    n_sel = np.sum(new['mask'])
    while n_sel > 200:
        on = np.where(new['mask'])[0]; new['mask'][np.random.choice(on)] = False; n_sel -= 1
    while n_sel < 10:
        off = np.where(~new['mask'])[0]
        if len(off) == 0: break
        new['mask'][np.random.choice(off)] = True; n_sel += 1
    if random.random() < 0.2:
        new['hp']['depth'] = max(4, min(10, new['hp']['depth'] + random.choice([-1,0,1])))
    if random.random() < 0.1:
        new['model_type'] = np.random.choice(CONFIG['model_types'], p=CONFIG['model_weights'])
    return new

def crossover(p1, p2):
    child = {'mask': np.zeros_like(p1['mask']), 'brier': 1.0}
    for i in range(len(child['mask'])):
        child['mask'][i] = p1['mask'][i] if random.random() < 0.5 else p2['mask'][i]
    child['model_type'] = p1['model_type'] if random.random() < 0.5 else p2['model_type']
    child['hp'] = dict(p1['hp'] if random.random() < 0.5 else p2['hp'])
    return child

def fetch_seeds():
    seeds = []
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for name, url in [('S10','https://nomos42-nba-quant.hf.space/api/best'),
                       ('S11','https://nomos42-nba-quant-2.hf.space/api/best'),
                       ('S12','https://nomos42-nba-evo-3.hf.space/api/best'),
                       ('S13','https://nomos42-nba-evo-4.hf.space/api/best'),
                       ('S14','https://nomos42-nba-evo-5.hf.space/api/best'),
                       ('S15','https://nomos42-nba-evo-6.hf.space/api/best')]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Nomos42/1.0'})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
                d = json.loads(r.read())
                if d.get('brier', 1.0) < 0.99:
                    mask = np.zeros(X.shape[1], dtype=bool)
                    for idx in d.get('features', []):
                        if 0 <= idx < X.shape[1]: mask[idx] = True
                    seeds.append({'mask': mask, 'model_type': d.get('model_type', 'xgboost'),
                                  'hp': d.get('hp', {'depth':6,'lr':0.1,'n_est':200}),
                                  'brier': float(d.get('brier', 1.0))})
                    # Also add TabICL variant
                    seeds.append({'mask': mask.copy(), 'model_type': 'tabicl',
                                  'hp': {'depth':6,'lr':0.1,'n_est':200}, 'brier': 1.0})
                    print(f'  {name}: brier={d.get("brier","?")}')
        except: pass
    print(f'Seeds: {len(seeds)}')
    return seeds

# %%
# ═══════════════════════════════════════
# CELL 4: RUN KARPATHY LOOP
# ═══════════════════════════════════════
from datetime import datetime

if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())
    population = [{**ind, 'mask': np.array(ind['mask'], dtype=bool)} for ind in state['population']]
    best_ever = state['best_ever']
    iteration = state['iteration']
    print(f'Resumed: iter={iteration}, best={best_ever:.5f}')
else:
    seeds = fetch_seeds()
    population = seeds[:CONFIG['population_size']]
    while len(population) < CONFIG['population_size']:
        population.append(random_individual())
    best_ever = min((ind['brier'] for ind in population if ind['brier'] < 1.0), default=1.0)
    iteration = 0

SESSION_LIMIT = 12 * 3600  # 12h Colab max
session_start = time.time()

print(f'\n{"="*70}')
print(f'  NBA QUANT AI — KARPATHY LOOP (Colab GPU)')
print(f'  Pop={CONFIG["population_size"]} | Budget={CONFIG["iteration_budget_sec"]}s')
print(f'  TabICL weight: {CONFIG["model_weights"][0]:.0%} | ATR: 0.21837')
print(f'{"="*70}\n')

while time.time() - session_start < SESSION_LIMIT:
    iteration += 1
    t0 = time.time()
    n_evals = improved = 0

    for ind in population:
        if ind['brier'] >= 0.99:
            ind['brier'] = evaluate(ind['mask'], ind['model_type'], ind['hp'])
            n_evals += 1
            if time.time() - t0 > CONFIG['iteration_budget_sec']: break

    population.sort(key=lambda x: x['brier'])
    if population[0]['brier'] < best_ever:
        best_ever = population[0]['brier']; improved = True

    elite_size = max(2, CONFIG['population_size'] // 5)
    elite = population[:elite_size]
    offspring = []
    while len(offspring) < CONFIG['population_size'] - elite_size:
        if random.random() < CONFIG['crossover_rate']:
            p1, p2 = random.sample(elite, 2)
            child = mutate(crossover(p1, p2))
        else:
            child = mutate(random.choice(elite))
        offspring.append(child)
    population = elite + offspring

    dur = time.time() - t0
    tag = '*** NEW BEST ***' if improved else ''
    elapsed = (time.time() - session_start) / 60
    print(f'Iter {iteration}: best={best_ever:.5f} ({population[0]["model_type"]}, {int(np.sum(population[0]["mask"]))}f) | '
          f'{n_evals} evals {dur:.0f}s | {elapsed:.0f}min {tag}')

    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps({'iter': iteration, 'best': best_ever, 'improved': improved,
                           'evals': n_evals, 'dur': round(dur,1), 'ts': datetime.now().isoformat()}) + '\n')

    # Save to Drive EVERY iteration (survives disconnects)
    s = {'population': [{**ind, 'mask': ind['mask'].tolist()} for ind in population],
         'best_ever': best_ever, 'iteration': iteration, 'ts': datetime.now().isoformat()}
    STATE_FILE.write_text(json.dumps(s))
    RESULTS_FILE.write_text(json.dumps({
        'best_brier': best_ever, 'iteration': iteration,
        'model_type': population[0]['model_type'],
        'n_features': int(np.sum(population[0]['mask'])),
        'features': [int(i) for i in np.where(population[0]['mask'])[0]],
        'hp': population[0]['hp']}, indent=2))

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f'\nDONE: {iteration} iterations, best={best_ever:.5f}')
