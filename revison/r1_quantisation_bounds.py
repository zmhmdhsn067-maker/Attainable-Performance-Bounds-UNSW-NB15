"""
Table 11, uniform-quantile rows: bounds for classifiers that are piecewise constant on a
uniform quantile grid of the continuous features (64 / 32 / 16 / 8 bins, edges fitted on the
training partition; categorical columns untouched). Same binning as the submitted
e3_bin_sensitivity.py, but groups are formed by EXACT row equality (not a 64-bit hash) and
the macro-recall bound D and the exact joint macro-F1 bound J are added.

Output: results/r1_quantisation_bounds.json
"""
import os
import sys
import json
import time
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('e', os.path.join(HERE, 'r1_joint_bound_exact.py'))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
b = e.b
CAT_COLS = ['proto', 'service', 'state']


def bin_continuous(train_df, test_df, cont_cols, n_bins):
    """Identical to e3_bin_sensitivity.bin_continuous."""
    te_binned = test_df.copy()
    for col in cont_cols:
        edges = np.unique(np.quantile(train_df[col].values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            te_binned[col] = 0
            continue
        te_binned[col] = np.digitize(test_df[col].values, edges[1:-1])
    return te_binned


def main():
    tr = pd.read_csv(os.path.join(ROOT, 'data', 'UNSW_NB15_training-set.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'data', 'UNSW_NB15_testing-set.csv'))
    for df in (tr, te):
        df.drop(columns=['id', 'label'], inplace=True, errors='ignore')
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')
    names = sorted(tr['attack_cat'].unique())
    y = te.pop('attack_cat').map({c: i for i, c in enumerate(names)}).to_numpy()
    tr.pop('attack_cat')
    cont = [c for c in tr.columns if c not in CAT_COLS]
    out = {}
    for n_bins in (None, 64, 32, 16, 8):
        t0 = time.time()
        X = te if n_bins is None else bin_continuous(tr, te, cont, n_bins)
        M = b.count_matrix(b.exact_group_ids(X), y, len(names))
        _, n_conf = b.conflict_rates(M)
        j = e.joint_milp_aggregated(M, time_limit=900)
        key = 'exact' if n_bins is None else f'{n_bins}_bins'
        out[key] = {'groups': int(M.shape[0]), 'conflicted_groups': n_conf,
                    'bound_A_accuracy': float(b.bound_A(M)), 'bound_D_macro_recall': float(b.bound_D(M)[0]),
                    'joint_macro_f1_lower': j['joint_macro_f1_lower'], 'joint_macro_f1_upper': j['joint_macro_f1_upper'],
                    'joint_exact': j['exact'], 'seconds': time.time() - t0}
        r = out[key]
        print(f"{key:<8} groups {r['groups']:>6}  conflicted {r['conflicted_groups']:>5}  A {100*r['bound_A_accuracy']:.3f}  "
              f"D {100*r['bound_D_macro_recall']:.2f}  J {r['joint_macro_f1_lower']:.4f}–{r['joint_macro_f1_upper']:.4f} "
              f"exact={r['joint_exact']}  {r['seconds']:.0f}s", flush=True)
        with open(os.path.join(HERE, 'results', 'r1_quantisation_bounds.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
