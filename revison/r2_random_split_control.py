"""
Reviewer 2, comment 6: measure (rather than assert) what a random pooled split does.

For the official partition and for five random stratified re-splits of the pooled corpus
(train+test, 257,673 rows; test size fixed at the official 82,332; seeds 42, 1, 7, 123, 2024):
  * exact-duplicate coverage of test rows by the training split
  * train-only lookup (memorisation) accuracy, overall and on covered rows
  * test-internal accuracy upper bound (bound A)
  * XGBoost (the submission's configuration) accuracy / macro-F1 / balanced accuracy,
    decomposed over covered vs uncovered test rows.
The covered/uncovered decomposition separates memorisation from the other thing a random
split changes -- removal of the official partition's prior shift (Normal 31.9% -> 44.9%).
Validation for early stopping is a stratified 15% carve-out of each training split only.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'results', 'r2_random_split_control.json')
CAT = ['proto', 'service', 'state']
SEEDS = [42, 1, 7, 123, 2024]


def load_raw():
    tr = pd.read_csv(os.path.join(ROOT, 'data', 'UNSW_NB15_training-set.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'data', 'UNSW_NB15_testing-set.csv'))
    for df in (tr, te):
        df.drop(columns=['id', 'label'], inplace=True, errors='ignore')
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')
    return tr, te


def encode(train_df, test_df):
    """Tree-family encoding (identical to gpu_boosting_comparison): label-encode categoricals
    on the training split, unseen test categories -> code 0, float32."""
    tr, te = train_df.copy(), test_df.copy()
    for col in CAT:
        le = LabelEncoder().fit(tr[col].astype(str))
        known = set(le.classes_)
        tr[col] = le.transform(tr[col].astype(str))
        te[col] = le.transform(te[col].astype(str).where(te[col].astype(str).isin(known), le.classes_[0]))
    return tr.to_numpy(np.float32), te.to_numpy(np.float32)


def coverage_and_lookup(Xtr_df, ytr, Xte_df, yte):
    cols = list(Xtr_df.columns)
    t = Xtr_df.copy(); t['_y'] = ytr
    maj = (t.groupby(cols + ['_y'], sort=False).size().reset_index(name='n')
             .sort_values('n', ascending=False).drop_duplicates(cols).drop(columns='n'))
    merged = Xte_df.merge(maj, on=cols, how='left')
    covered = merged['_y'].notna().to_numpy()
    pred = merged['_y'].fillna(pd.Series(ytr).value_counts().idxmax()).to_numpy().astype(int)
    return covered, float((pred == yte).mean()), float((pred[covered] == yte[covered]).mean())


def bound_A(Xte_df, yte):
    t = Xte_df.copy(); t['_y'] = yte
    g = t.groupby(list(Xte_df.columns) + ['_y'], sort=False).size()
    return float(g.groupby(level=list(range(len(Xte_df.columns)))).max().sum() / len(yte))


def xgb_fit_predict(Xtr, ytr, Xte, K, seed):
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
    clf = xgb.XGBClassifier(n_estimators=900, max_depth=9, learning_rate=0.08, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
                            objective='multi:softprob', num_class=K, tree_method='hist',
                            device='cuda', eval_metric='mlogloss', early_stopping_rounds=30,
                            random_state=seed)
    clf.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    return clf.predict(Xte)


def scores(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return {'n': int(len(y)), 'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p))}


def run_split(name, tr_df, te_df, class_names, seed):
    t0 = time.time()
    cidx = {c: i for i, c in enumerate(class_names)}
    ytr = tr_df['attack_cat'].map(cidx).to_numpy(); yte = te_df['attack_cat'].map(cidx).to_numpy()
    Xtr_df = tr_df.drop(columns='attack_cat').reset_index(drop=True)
    Xte_df = te_df.drop(columns='attack_cat').reset_index(drop=True)
    covered, lk_all, lk_cov = coverage_and_lookup(Xtr_df, ytr, Xte_df, yte)
    Xtr, Xte = encode(Xtr_df, Xte_df)
    pred = xgb_fit_predict(Xtr, ytr, Xte, len(class_names), seed)
    r = {'split': name, 'seed': seed, 'coverage': float(covered.mean()),
         'lookup_accuracy_overall': lk_all, 'lookup_accuracy_on_covered': lk_cov,
         'bound_A_test_internal': bound_A(Xte_df, yte),
         'test_normal_share': float((te_df['attack_cat'] == 'Normal').mean()),
         'xgb_all': scores(yte, pred), 'xgb_covered': scores(yte, pred, covered),
         'xgb_uncovered': scores(yte, pred, ~covered)}
    c = r['coverage']
    r['xgb_accuracy_gain_from_covered_rows_pt'] = 100 * c * (r['xgb_covered']['accuracy'] - r['xgb_uncovered']['accuracy'])
    r['seconds'] = time.time() - t0
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                      if not isinstance(v, dict)}), flush=True)
    print('   xgb all / covered / uncovered acc:', round(r['xgb_all']['accuracy'], 4),
          round(r['xgb_covered']['accuracy'], 4), round(r['xgb_uncovered']['accuracy'], 4), flush=True)
    return r


def main():
    tr, te = load_raw()
    class_names = sorted(tr['attack_cat'].unique())
    out = {'official': [], 'random': []}
    for s in SEEDS:                      # official partition: only the early-stopping split varies
        out['official'].append(run_split('official', tr, te, class_names, s))
    pooled = pd.concat([tr, te], ignore_index=True)
    for s in SEEDS:
        rtr, rte = train_test_split(pooled, test_size=len(te), random_state=s, stratify=pooled['attack_cat'])
        out['random'].append(run_split('random_pooled', rtr, rte, class_names, s))

    def agg(rows, key, sub=None):
        v = np.array([(r[key][sub] if sub else r[key]) for r in rows], float)
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}
    out['summary'] = {
        split: {'coverage': agg(out[split], 'coverage'),
                'lookup_accuracy_overall': agg(out[split], 'lookup_accuracy_overall'),
                'bound_A_test_internal': agg(out[split], 'bound_A_test_internal'),
                'xgb_accuracy': agg(out[split], 'xgb_all', 'accuracy'),
                'xgb_macro_f1': agg(out[split], 'xgb_all', 'macro_f1'),
                'xgb_accuracy_uncovered_rows': agg(out[split], 'xgb_uncovered', 'accuracy'),
                'xgb_gain_from_covered_rows_pt': agg(out[split], 'xgb_accuracy_gain_from_covered_rows_pt')}
        for split in ('official', 'random')}
    print(json.dumps(out['summary'], indent=1))
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
