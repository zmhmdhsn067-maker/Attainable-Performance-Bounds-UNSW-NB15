"""
Reviewer 1, comment 4: distinguish the universal (exact-duplicate) bound from bounds that
hold only under an explicit hypothesis-class restriction.

Let H_cuts be the set of classifiers that are piecewise constant on the axis-aligned grid
formed by a fixed set of per-feature cut points. A histogram-based gradient-boosted
ensemble trained with those cuts belongs to H_cuts: every split has the form x_f < c with c
a cut point, so two inputs falling in the same cell receive the same prediction. Hence the
exact-duplicate bound computed on cell indices (instead of raw values) is a valid upper
bound for every member of H_cuts -- but NOT for unrestricted classifiers such as a neural
network, for which only the exact-duplicate bound applies.

This script:
  1. builds XGBoost's own quantile sketch (QuantileDMatrix, max_bin=256) on the training
     split of seed 42, trains the submission's XGBoost configuration on that very matrix,
     and checks that every split threshold in the trained booster is one of the cut points;
  2. maps the test rows to cell indices and recomputes bounds A and D and the per-class F1
     ceilings for H_cuts (the exact joint macro-F1 on the same cells is computed by
     r1_joint_bound_exact.py).
"""
import os
import sys
import json
import importlib.util
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import gpu_boosting_comparison as gb

spec = importlib.util.spec_from_file_location('b', os.path.join(HERE, 'r1_bounds_all_representations.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def main():
    Xtr, Xte, ytr, yte, feats, class_names = gb.load_and_preprocess('multiclass')
    K = len(class_names)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=42, stratify=ytr)

    dtrain = xgb.QuantileDMatrix(Xf, label=yf, max_bin=256)
    dval = xgb.QuantileDMatrix(Xv, label=yv, ref=dtrain)
    params = dict(objective='multi:softprob', num_class=K, max_depth=9, eta=0.08, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, tree_method='hist',
                  device='cuda', eval_metric='mlogloss', seed=42, max_bin=256)
    booster = xgb.train(params, dtrain, num_boost_round=900, evals=[(dval, 'val')],
                        early_stopping_rounds=30, verbose_eval=False)

    indptr, values = dtrain.get_quantile_cut()
    cuts = [np.asarray(values[indptr[f]:indptr[f + 1]], dtype=np.float32) for f in range(Xtr.shape[1])]

    # sanity check: every split threshold used by the trained booster is a cut point
    df = booster.trees_to_dataframe()
    splits = df[df['Feature'] != 'Leaf']
    bad = 0
    for fname, grp in splits.groupby('Feature'):
        f = int(fname[1:]) if fname.startswith('f') else feats.index(fname)
        cs = cuts[f]
        for s in grp['Split'].to_numpy(np.float32):
            if not np.any(np.isclose(cs, s, rtol=0, atol=0)):
                bad += 1
    print(f'booster splits: {len(splits)}  thresholds not in cut set: {bad}', flush=True)

    cells = np.column_stack([np.searchsorted(cuts[f], Xte[:, f], side='right') for f in range(Xte.shape[1])])
    gid = b.exact_group_ids(cells)
    M = b.count_matrix(gid, yte, K)
    conf, n_conf = b.conflict_rates(M)
    C = b.bound_C(M)
    macD, _ = b.bound_D(M)
    res = {
        'hypothesis_class': 'piecewise constant on XGBoost max_bin=256 quantile cells (seed-42 training split)',
        'booster_best_iteration': int(booster.best_iteration),
        'booster_split_thresholds': int(len(splits)),
        'thresholds_not_in_cut_set': int(bad),
        'cuts_per_feature_median': float(np.median([len(c) for c in cuts])),
        'distinct_test_cells': int(M.shape[0]), 'conflicted_cells': n_conf,
        'bound_A_accuracy': float(b.bound_A(M)), 'bound_D_macro_recall': float(macD),
        'mean_of_per_class_F1_ceilings': float(np.mean([c['f1'] for c in C])),
        'per_class_f1_ceiling': {class_names[c]: C[c]['f1'] for c in range(K)},
        'per_class_conflict_rate': {class_names[c]: float(conf[c]) for c in range(K)},
    }
    res['joint_macro_f1'] = {'see': 'results/r1_joint_bound_exact.json, entry H_xgb_quantile_cells_256'}
    print(json.dumps(res, indent=1), flush=True)
    with open(os.path.join(HERE, 'results', 'r1_hypothesis_class_bounds.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
