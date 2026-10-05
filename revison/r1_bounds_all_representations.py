"""
Revision analysis for reviewer comments R1-1, R2-1 (jointly attainable macro-F1) and
R1-3 (bounds computed on the representation each classifier actually received).

Differences from the submitted attainable_bounds.py:
  * Groups are formed by EXACT row equality (np.unique over rows / pandas groupby over all
    columns), not by a 64-bit hash, so the manuscript's "no hash collision" statement is
    true by construction. The hash-based group count is reported alongside for comparison.
  * Bounds are computed on three representations:
      R0  raw CSV values (float64 numerics, string categoricals) -- what the submission used
      R1  tree-family input: float32, proto/service/state label-encoded, unseen test
          categories mapped to code 0 (gpu_boosting_comparison.load_and_preprocess)
      R2  Transformer-KAN / Random Forest input: R1 with the 39 continuous columns
          standardised in float32 (evaluate_unsw_nb15_hybrid.load_and_preprocess)
  * Provides the joint macro-F1 program (joint_macro_f1_milp); on UNSW-NB15 the exact optimum is
    computed by r1_joint_bound_exact.py, which removes the symmetry between identical groups.

Joint macro-F1 formulation. For a deterministic classifier h, every group g receives one
label. With TP_c = sum_{g:h(g)=c} n_gc, P_c = sum_{g:h(g)=c} |g| and N_c the class total,
F1_c = 2 TP_c / (N_c + P_c). Pure groups are assigned their own label without loss of
optimality (doing so weakly increases every F1_c). For the conflicted groups, binaries
x_gc select the label; f_c <= 2 TP_c / (N_c + P_c) is linearised exactly as
    (N_c + A_c) f_c + sum_g |g| z_gc - 2 sum_g n_gc x_gc <= 2 A_c,
    z_gc >= f_c + x_gc - 1,  z_gc >= 0,
where A_c is the pure-group mass of class c and z_gc stands for f_c * x_gc (only the lower
McCormick envelope is needed, because z appears only on the restrictive side).
"""
import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import evaluate_unsw_nb15_hybrid as m          # R2 (TKAN / RF input)
import gpu_boosting_comparison as gb            # R1 (tree input)

OUT = os.path.join(HERE, 'results', 'r1_bounds_all_representations.json')


# ─────────────────────────────── grouping ────────────────────────────────
def exact_group_ids(X):
    if isinstance(X, pd.DataFrame):
        return X.groupby(list(X.columns), sort=False, dropna=False).ngroup().to_numpy()
    _, inv = np.unique(np.ascontiguousarray(X), axis=0, return_inverse=True)
    return inv.ravel()


def hash_group_count(X):
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    return int(pd.util.hash_pandas_object(df, index=False).nunique())


def count_matrix(gid, y, K):
    G = int(gid.max()) + 1
    M = np.zeros((G, K), dtype=np.int64)
    np.add.at(M, (gid, y), 1)
    return M


# ─────────────────────────────── bounds ──────────────────────────────────
def bound_A(M):
    return M.max(axis=1).sum() / M.sum()


def bound_D(M):
    N = M.sum(axis=0)
    ratio = M / np.maximum(N, 1)
    choice = ratio.argmax(axis=1)
    rec_num = np.zeros(M.shape[1])
    np.add.at(rec_num, choice, M[np.arange(len(M)), choice])
    rec = rec_num / N
    return float(rec.mean()), rec


def recall_under_A(M):
    N = M.sum(axis=0)
    choice = M.argmax(axis=1)
    rec_num = np.zeros(M.shape[1])
    np.add.at(rec_num, choice, M[np.arange(len(M)), choice])
    return rec_num / N


def bound_C(M):
    """Per-class F1 ceiling by purity-sorted prefix sweep; also returns P/R at optimum."""
    N = M.sum(axis=0)
    sizes = M.sum(axis=1)
    out = []
    for c in range(M.shape[1]):
        a = M[:, c]
        keep = a > 0
        a_k, b_k = a[keep], (sizes - a)[keep]
        order = np.argsort(-(a_k / (a_k + b_k)), kind='stable')
        T = np.cumsum(a_k[order])
        F = np.cumsum(b_k[order])
        f1 = 2 * T / (N[c] + T + F)
        k = int(np.argmax(f1))
        out.append({'f1': float(f1[k]), 'precision': float(T[k] / (T[k] + F[k])),
                    'recall': float(T[k] / N[c]), 'groups_used': k + 1})
    return out


def conflict_rates(M):
    N = M.sum(axis=0)
    conflicted = (M > 0).sum(axis=1) > 1
    return M[conflicted].sum(axis=0) / N, int(conflicted.sum())


def macro_f1_of_assignment(M, assign):
    K = M.shape[1]
    N = M.sum(axis=0)
    sizes = M.sum(axis=1)
    TP = np.zeros(K)
    P = np.zeros(K)
    np.add.at(TP, assign, M[np.arange(len(M)), assign])
    np.add.at(P, assign, sizes)
    f1 = np.where(N + P > 0, 2 * TP / (N + P), 0.0)
    return float(f1.mean()), f1


def joint_macro_f1_milp(M, time_limit=900):
    K = M.shape[1]
    N = M.sum(axis=0).astype(float)
    conflicted = (M > 0).sum(axis=1) > 1
    Mc = M[conflicted].astype(float)
    pure = M[~conflicted]
    A = pure.sum(axis=0).astype(float)          # pure groups: TP = P = their size
    Gc = Mc.shape[0]
    s = Mc.sum(axis=1)

    nx = Gc * K
    ix = lambda g, c: g * K + c
    iz = lambda g, c: nx + g * K + c
    i_f = lambda c: 2 * nx + c
    nvar = 2 * nx + K

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for g in range(Gc):                      # sum_c x_gc = 1
        for c in range(K):
            rows.append(r); cols.append(ix(g, c)); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1
    for c in range(K):                       # class F1 linearisation
        rows.append(r); cols.append(i_f(c)); vals.append(N[c] + A[c])
        for g in range(Gc):
            rows.append(r); cols.append(iz(g, c)); vals.append(s[g])
            if Mc[g, c] > 0:
                rows.append(r); cols.append(ix(g, c)); vals.append(-2.0 * Mc[g, c])
        lb.append(-np.inf); ub.append(2.0 * A[c]); r += 1
    for g in range(Gc):                      # z >= f + x - 1
        for c in range(K):
            rows += [r, r, r]; cols += [iz(g, c), i_f(c), ix(g, c)]; vals += [1.0, -1.0, -1.0]
            lb.append(-1.0); ub.append(np.inf); r += 1

    Acon = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    cobj = np.zeros(nvar); cobj[2 * nx:] = -1.0 / K
    integrality = np.zeros(nvar); integrality[:nx] = 1
    t0 = time.time()
    res = milp(cobj, constraints=LinearConstraint(Acon, lb, ub), integrality=integrality,
               bounds=Bounds(0, 1), options={'time_limit': time_limit, 'mip_rel_gap': 1e-9})
    dt = time.time() - t0

    x = res.x[:nx].reshape(Gc, K)
    conf_assign = x.argmax(axis=1)
    full_assign = M.argmax(axis=1).copy()
    full_assign[conflicted] = conf_assign
    val, f1 = macro_f1_of_assignment(M, full_assign)
    dual = float(-res.mip_dual_bound) if getattr(res, 'mip_dual_bound', None) is not None else None
    absent = int(sum(Mc[g, conf_assign[g]] == 0 for g in range(Gc)))
    return {'status': res.message, 'solve_seconds': dt, 'macro_f1_incumbent': val,
            'macro_f1_dual_upper_bound': dual, 'mip_gap': getattr(res, 'mip_gap', None),
            'per_class_f1_at_joint_optimum': f1.tolist(), 'conflicted_groups': Gc,
            'groups_assigned_to_a_class_absent_from_the_group': absent}


def coordinate_ascent(M, init):
    """Independent heuristic cross-check of the MILP: any assignment is a realisable
    deterministic classifier, so its macro-F1 is a certified lower bound on the joint optimum.
    TP/P are updated incrementally, so each move costs O(K)."""
    K = M.shape[1]
    N = M.sum(axis=0).astype(float)
    sizes = M.sum(axis=1).astype(float)
    conflicted = np.where((M > 0).sum(axis=1) > 1)[0]
    assign = init.copy()
    TP = np.zeros(K); P = np.zeros(K)
    np.add.at(TP, assign, M[np.arange(len(M)), assign])
    np.add.at(P, assign, sizes)
    score = lambda: float(np.mean(np.where(N + P > 0, 2 * TP / (N + P), 0.0)))
    best = score()
    improved = True
    while improved:
        improved = False
        for g in conflicted:
            a = assign[g]
            for b in range(K):
                if b == a:
                    continue
                TP[a] -= M[g, a]; P[a] -= sizes[g]; TP[b] += M[g, b]; P[b] += sizes[g]
                v = score()
                if v > best + 1e-12:
                    best, a, improved = v, b, True
                    assign[g] = b
                else:
                    TP[b] -= M[g, b]; P[b] -= sizes[g]; TP[a] += M[g, a]; P[a] += sizes[g]
    return best


# ─────────────────────────────── lookup ──────────────────────────────────
def lookup_baseline(Xtr, ytr, Xte, yte):
    tr = pd.DataFrame(Xtr) if not isinstance(Xtr, pd.DataFrame) else Xtr.copy()
    te = pd.DataFrame(Xte) if not isinstance(Xte, pd.DataFrame) else Xte.copy()
    cols = list(tr.columns)
    tr['_y'] = ytr
    maj = (tr.groupby(cols + ['_y'], sort=False, dropna=False).size()
             .reset_index(name='n').sort_values('n', ascending=False)
             .drop_duplicates(cols).drop(columns='n'))
    merged = te.merge(maj, on=cols, how='left')
    covered = merged['_y'].notna().to_numpy()
    fallback = pd.Series(ytr).value_counts().idxmax()
    pred = merged['_y'].fillna(fallback).to_numpy().astype(np.int64)
    return {'coverage': float(covered.mean()),
            'accuracy_overall_with_fallback': float((pred == yte).mean()),
            'accuracy_on_covered_rows': float((pred[covered] == yte[covered]).mean())}


def analyse(name, Xtr, ytr, Xte, yte, class_names, gid, joint=None):
    K = len(class_names)
    t0 = time.time()
    M = count_matrix(gid, yte, K)
    conf, n_conf_groups = conflict_rates(M)
    accA = bound_A(M)
    macD, recD = bound_D(M)
    recA = recall_under_A(M)
    C = bound_C(M)
    meanC = float(np.mean([c['f1'] for c in C]))
    res = {
        'representation': name,
        'test_rows': int(len(yte)),
        'distinct_vectors_exact': int(M.shape[0]),
        'distinct_vectors_hash64': hash_group_count(Xte),
        'conflicted_groups': n_conf_groups,
        'bound_A_accuracy': float(accA),
        'bound_D_macro_recall': float(macD),
        'mean_of_per_class_F1_ceilings': meanC,
        'per_class': {class_names[c]: {
            'N': int(M[:, c].sum()), 'conflict_rate': float(conf[c]),
            'recall_A': float(recA[c]), 'recall_D': float(recD[c]),
            'f1_ceiling': C[c]['f1'], 'precision_at_f1_ceiling': C[c]['precision'],
            'recall_at_f1_ceiling': C[c]['recall']} for c in range(K)},
        'lookup': lookup_baseline(Xtr, ytr, Xte, yte),
    }
    if joint is None:
        # The plain formulation below stops at its time limit on UNSW-NB15; the exact joint bound
        # is computed by r1_joint_bound_exact.py (symmetry-free formulation, proven optimal).
        # joint_macro_f1_milp is kept for small instances such as CICIDS2017.
        joint = {'see': 'results/r1_joint_bound_exact.json (r1_joint_bound_exact.py)'}
    res['joint_macro_f1'] = joint
    res['seconds'] = time.time() - t0
    return res


def same_partition(a, b):
    """True iff two group-id vectors induce the same partition of the rows."""
    pairs = pd.DataFrame({'a': a, 'b': b}).drop_duplicates()
    return pairs['a'].is_unique and pairs['b'].is_unique


def main():
    results = {}

    # R0: raw CSV values
    tr = pd.read_csv(m.TRAIN_PATH); te = pd.read_csv(m.TEST_PATH)
    for df in (tr, te):
        df.drop(columns=['id', 'label'], inplace=True, errors='ignore')
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')
    class_names = sorted(tr['attack_cat'].unique())
    cidx = {c: i for i, c in enumerate(class_names)}
    ytr0 = tr.pop('attack_cat').map(cidx).to_numpy()
    yte0 = te.pop('attack_cat').map(cidx).to_numpy()
    seen = {col: set(tr[col].astype(str)) for col in ['proto', 'service', 'state']}
    unseen_mask = np.zeros(len(te), dtype=bool)
    unseen = {}
    for col in ['proto', 'service', 'state']:
        miss = ~te[col].astype(str).isin(seen[col]).to_numpy()
        unseen[col] = int(miss.sum()); unseen_mask |= miss

    # R1: tree-family input;  R2: TKAN / RF input
    Xtr1, Xte1, ytr1, yte1, _, cn1 = gb.load_and_preprocess('multiclass')
    Xtr2, Xte2, ytr2, yte2, _, cn2, *_ = m.load_and_preprocess()
    assert list(cn1) == class_names and list(cn2) == class_names
    assert (yte1 == yte0).all() and (yte2 == yte0).all() and (ytr1 == ytr0).all() and (ytr2 == ytr0).all()

    gid0 = exact_group_ids(te)
    gid1 = exact_group_ids(Xte1)
    gid2 = exact_group_ids(Xte2)
    results['partition_identical'] = {
        'R0_vs_R1': bool(same_partition(gid0, gid1)),
        'R1_vs_R2': bool(same_partition(gid1, gid2)),
        'R0_vs_R2': bool(same_partition(gid0, gid2)),
    }
    print('partition identical:', results['partition_identical'], flush=True)

    r0 = analyse('R0_raw_csv', tr, ytr0, te, yte0, class_names, gid0)
    r0['test_rows_with_unseen_category'] = int(unseen_mask.sum())
    r0['unseen_by_column'] = unseen
    results['R0'] = r0
    print(json.dumps({k: v for k, v in r0.items() if k != 'per_class'}, indent=1), flush=True)

    j1 = r0['joint_macro_f1'] if results['partition_identical']['R0_vs_R1'] else None
    r1 = analyse('R1_tree_input_float32_labelencoded', Xtr1, ytr1, Xte1, yte1, class_names, gid1, j1)
    results['R1'] = r1
    print(json.dumps({k: v for k, v in r1.items() if k not in ('per_class', 'joint_macro_f1')}, indent=1), flush=True)

    j2 = r1['joint_macro_f1'] if results['partition_identical']['R1_vs_R2'] else None
    r2 = analyse('R2_tkan_rf_input_float32_scaled', Xtr2, ytr2, Xte2, yte2, class_names, gid2, j2)
    results['R2'] = r2
    print(json.dumps({k: v for k, v in r2.items() if k not in ('per_class', 'joint_macro_f1')}, indent=1), flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
