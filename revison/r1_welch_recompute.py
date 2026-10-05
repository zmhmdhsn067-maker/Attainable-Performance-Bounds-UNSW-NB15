"""
R1-7: recompute every Table 4 Welch test from per-seed values, with exact two-sided p-values.

Per-seed sources (seeds 42, 1, 7, 123, 2024):
  Random Forest   results/e4_random_forest_full.json (accs, f1ms); balanced accuracy and
                  weighted-F1 per seed were printed by e4_random_forest_full.py but not saved,
                  so they are transcribed from that run's console output (4-decimal rounding).
  XGBoost/LightGBM results/multiseed_statistics.json (accs, f1ms)
  Transformer-KAN results/ablation_results.json (e6_seed*)

All SDs here are SAMPLE standard deviations (ddof=1). The submission's Table 3 mixed
conventions: Transformer-KAN SDs were ddof=1, tree-family SDs were np.std's default ddof=0.

These are PROVISIONAL: every family is being re-run under a single logged protocol, and the
final Table 4 will be regenerated from those runs by the same function.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, 'results', p), encoding='utf-8'))

seeds = [42, 1, 7, 123, 2024]
rf = R('e4_random_forest_full.json')
ms = R('multiseed_statistics.json')['multiclass']
ab = R('ablation_results.json')

data = {
    'Random Forest': {
        'accuracy': np.array(rf['accs']), 'macro_f1': np.array(rf['f1ms']),
        'balanced_accuracy': np.array([0.5446, 0.5413, 0.5446, 0.5502, 0.5418]),
    },
    'XGBoost': {'accuracy': np.array(ms['xgboost']['accs']), 'macro_f1': np.array(ms['xgboost']['f1ms'])},
    'LightGBM': {'accuracy': np.array(ms['lightgbm']['accs']), 'macro_f1': np.array(ms['lightgbm']['f1ms'])},
    'Transformer-KAN': {
        k: np.array([ab[f'e6_seed{s}'][src] for s in seeds])
        for k, src in [('accuracy', 'accuracy'), ('macro_f1', 'f1_macro'),
                       ('balanced_accuracy', 'balanced_accuracy')]
    },
}


def welch(a, b):
    a, b = np.asarray(a, float) * 100, np.asarray(b, float) * 100     # percentage points
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    tcrit = stats.t.ppf(0.975, df)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    J = 1 - 3 / (4 * (len(a) + len(b)) - 9)                    # Hedges small-sample correction
    return {'diff_pt': diff, 't': float(t), 'df': float(df), 'p_two_sided': float(p),
            'ci95_pt': [diff - tcrit * se, diff + tcrit * se], 'hedges_g': float(J * diff / sp)}


tests = [
    ('Random Forest', 'XGBoost', 'macro_f1'),
    ('Random Forest', 'LightGBM', 'macro_f1'),
    ('LightGBM', 'XGBoost', 'macro_f1'),
    ('Random Forest', 'Transformer-KAN', 'macro_f1'),
    ('Random Forest', 'Transformer-KAN', 'accuracy'),
    ('Transformer-KAN', 'Random Forest', 'balanced_accuracy'),
]
submitted = {  # Table 4 as submitted: (t, df, p)
    0: (4.29, 8.0, '< 0.001'), 1: (1.04, 7.9, '0.30'), 2: (3.07, 8.0, '0.015'),
    3: (15.38, 7.2, '< 0.001'), 4: (8.84, 4.1, '< 0.001'), 5: (6.50, 4.6, '< 0.001'),
}

rows = []
for i, (a, b, metric) in enumerate(tests):
    w = welch(data[a][metric], data[b][metric])
    w.update({'comparison': f'{a} vs {b}', 'metric': metric,
              'submitted_t_df_p': submitted[i]})
    rows.append(w)

# Holm-Bonferroni across the six tests
order = np.argsort([r['p_two_sided'] for r in rows])
m_ = len(rows)
running = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (m_ - rank) * rows[idx]['p_two_sided'])
    running = max(running, adj)
    rows[idx]['p_holm'] = running

print(f"{'comparison':<36}{'metric':<18}{'diff':>7}{'t':>7}{'df':>6}{'p exact':>11}{'p Holm':>9}   submitted (t, df, p)")
for r in rows:
    print(f"{r['comparison']:<36}{r['metric']:<18}{r['diff_pt']:>+7.2f}{r['t']:>7.2f}{r['df']:>6.1f}"
          f"{r['p_two_sided']:>11.2e}{r['p_holm']:>9.4f}   {r['submitted_t_df_p']}")

print('\nSample SDs (ddof=1) vs population SDs (ddof=0) used for the tree rows of Table 3:')
for fam in ['Random Forest', 'XGBoost', 'LightGBM', 'Transformer-KAN']:
    for metric, v in data[fam].items():
        v = v * 100
        print(f"  {fam:<16}{metric:<18} mean {v.mean():6.2f}  SD(ddof=1) {v.std(ddof=1):.2f}  SD(ddof=0) {v.std(ddof=0):.2f}")

with open(os.path.join(os.path.dirname(__file__), 'results', 'r1_welch_recompute.json'), 'w') as f:
    json.dump(rows, f, indent=2, default=float)
