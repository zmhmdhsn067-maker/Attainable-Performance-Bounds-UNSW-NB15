"""
Exact jointly attainable macro-F1 (reviewer comments R1-1, R2-1) with a tightened MILP.

The plain McCormick formulation in r1_bounds_all_representations.py is exact, but its LP
relaxation is weak because f_c ranges over [0, 1]; on R0 HiGHS stopped at the 900 s limit
with a 15% gap. Two valid facts tighten it without excluding any optimal solution:
  (i)  f_c <= C_c, the exact per-class F1 ceiling (purity-prefix sweep); hence the mean of
       the ceilings is itself an upper bound on the joint optimum;
  (ii) every label assignment is a realisable deterministic classifier, so the best value LB
       found by multi-start coordinate ascent is attained. Any assignment at least as good as
       LB satisfies, for every class, f_c >= L_c = K*LB - sum_{c' != c} C_c'.
On the box [L_c, C_c] the McCormick envelopes of z_gc = f_c * x_gc are
       z_gc >= L_c x_gc,     z_gc >= C_c x_gc + f_c - C_c,
exact at binary x and much tighter in the relaxation. The objective cut sum_c f_c >= K*LB
removes nothing that could be optimal.

Representations: R0 (raw values; the tree families' R1 induces the identical partition),
R2 (standardised float32 input of the Transformer-KAN and Random Forest) and H (cells of
XGBoost's max_bin=256 quantile grid; bound for classifiers piecewise constant on that grid).
"""
import os
import sys
import json
import time
import itertools
import importlib.util

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('b', os.path.join(HERE, 'r1_bounds_all_representations.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)   # also chdir(ROOT), utf-8 stdout
m, gb = b.m, b.gb
OUT = os.path.join(HERE, 'results', 'r1_joint_bound_tight.json')


# ───────────────────────── lower bound: multi-start ascent ─────────────────────────
def ascent(M, init):
    """First-improvement coordinate ascent over the labels of conflicted groups.
    Returns (macro-F1, assignment); the assignment is a concrete classifier."""
    K = M.shape[1]
    N = M.sum(axis=0).astype(float)
    sizes = M.sum(axis=1).astype(float)
    conflicted = np.where((M > 0).sum(axis=1) > 1)[0]
    assign = init.copy()
    TP = np.zeros(K); P = np.zeros(K)
    np.add.at(TP, assign, M[np.arange(len(M)), assign])
    np.add.at(P, assign, sizes)
    f1 = lambda t, p, n: 2 * t / (n + p) if n + p > 0 else 0.0
    improved = True
    while improved:
        improved = False
        for g in conflicted:
            a = assign[g]
            base_a = f1(TP[a], P[a], N[a])
            loss_a = base_a - f1(TP[a] - M[g, a], P[a] - sizes[g], N[a])
            best_gain, best_b = 1e-13, -1
            for c in range(K):
                if c == a:
                    continue
                gain = f1(TP[c] + M[g, c], P[c] + sizes[g], N[c]) - f1(TP[c], P[c], N[c]) - loss_a
                if gain > best_gain:
                    best_gain, best_b = gain, c
            if best_b >= 0:
                TP[a] -= M[g, a]; P[a] -= sizes[g]
                TP[best_b] += M[g, best_b]; P[best_b] += sizes[g]
                assign[g] = best_b
                improved = True
    val, _ = b.macro_f1_of_assignment(M, assign)
    return val, assign


def ceiling_sets(M):
    """For each class, the groups in its F1-optimal purity prefix."""
    N = M.sum(axis=0); sizes = M.sum(axis=1)
    sets = []
    for c in range(M.shape[1]):
        idx = np.where(M[:, c] > 0)[0]
        a = M[idx, c]; bb = sizes[idx] - a
        order = np.argsort(-(a / (a + bb)), kind='stable')
        T = np.cumsum(a[order]); F = np.cumsum(bb[order])
        k = int(np.argmax(2 * T / (N[c] + T + F)))
        sets.append(set(idx[order[:k + 1]].tolist()))
    return sets


def multi_start_lb(M, restarts=24, seed=0):
    K = M.shape[1]
    rng = np.random.default_rng(seed)
    conflicted = np.where((M > 0).sum(axis=1) > 1)[0]
    starts = {'majority': M.argmax(axis=1)}
    # ceiling-guided start: give each conflicted group to a class whose ceiling uses it
    # (largest count if several); groups no ceiling uses go to the largest class
    sets = ceiling_sets(M)
    init = M.argmax(axis=1).copy()
    big = int(M.sum(axis=0).argmax())
    for g in conflicted:
        want = [c for c in range(K) if g in sets[c]]
        init[g] = max(want, key=lambda c: M[g, c]) if want else big
    starts['ceiling_guided'] = init
    log = {}
    best_v, best_a = -1.0, None
    for name, s in starts.items():
        v, a = ascent(M, s)
        log[name] = v
        if v > best_v:
            best_v, best_a = v, a
    pert = []
    for r in range(restarts):                  # perturb 10% of conflicted groups, re-ascend
        s = best_a.copy()
        pick = rng.choice(conflicted, size=max(1, len(conflicted) // 10), replace=False)
        for g in pick:
            present = np.where(M[g] > 0)[0]
            s[g] = rng.choice(present) if rng.random() < 0.8 else rng.integers(K)
        v, a = ascent(M, s)
        pert.append(v)
        if v > best_v + 1e-13:
            best_v, best_a = v, a
    log['perturbation_restarts'] = restarts
    log['perturbation_best'] = max(pert) if pert else None
    return best_v, best_a, log


# ───────────────────────── tightened exact MILP ─────────────────────────
def joint_milp_tight(M, time_limit=1800, disp=False):
    K = M.shape[1]
    N = M.sum(axis=0).astype(float)
    C = np.array([c['f1'] for c in b.bound_C(M)])
    LB, lb_assign, lb_log = multi_start_lb(M)
    UB_ceil = float(C.mean())
    conflicted = (M > 0).sum(axis=1) > 1
    Mc = M[conflicted].astype(float)
    A = M[~conflicted].sum(axis=0).astype(float)
    Gc = Mc.shape[0]
    s = Mc.sum(axis=1)
    U = C.copy()
    L = np.clip(K * LB - (C.sum() - C) - 1e-9, 0.0, U)

    nx = Gc * K
    gg = np.repeat(np.arange(Gc), K); cc = np.tile(np.arange(K), Gc)
    xi = gg * K + cc; zi = nx + xi; fi = 2 * nx + np.arange(K)
    ncc = Mc[gg, cc]
    R, Cn, V, lo, hi = [], [], [], [], []
    r = 0
    R.append(r + gg); Cn.append(xi); V.append(np.ones(nx))             # sum_c x_gc = 1
    lo.append(np.ones(Gc)); hi.append(np.ones(Gc)); r += Gc
    R.append(r + np.arange(K)); Cn.append(fi); V.append(N + A)          # F1 rows
    R.append(r + cc); Cn.append(zi); V.append(s[gg])
    nz = ncc > 0
    R.append(r + cc[nz]); Cn.append(xi[nz]); V.append(-2.0 * ncc[nz])
    lo.append(np.full(K, -np.inf)); hi.append(2.0 * A); r += K
    ridx = r + np.arange(nx)                                             # z >= U x + f - U
    R += [ridx, ridx, ridx]; Cn += [zi, fi[cc], xi]; V += [np.ones(nx), -np.ones(nx), -U[cc]]
    lo.append(-U[cc]); hi.append(np.full(nx, np.inf)); r += nx
    sel = L[cc] > 0                                                      # z >= L x
    ns = int(sel.sum()); ridx = r + np.arange(ns)
    R += [ridx, ridx]; Cn += [zi[sel], xi[sel]]; V += [np.ones(ns), -L[cc][sel]]
    lo.append(np.zeros(ns)); hi.append(np.full(ns, np.inf)); r += ns
    R.append(np.full(K, r)); Cn.append(fi); V.append(np.ones(K))         # objective cut
    lo.append(np.array([K * LB - 1e-7])); hi.append(np.array([np.inf])); r += 1

    Acon = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cn))),
                      shape=(r, 2 * nx + K)).tocsr()
    lo = np.concatenate(lo); hi = np.concatenate(hi)
    cobj = np.zeros(2 * nx + K); cobj[fi] = -1.0 / K
    integ = np.zeros(2 * nx + K); integ[:nx] = 1
    vlb = np.concatenate([np.zeros(nx), np.zeros(nx), L])
    vub = np.concatenate([np.ones(nx), U[cc], U])
    t0 = time.time()
    res = milp(cobj, constraints=LinearConstraint(Acon, lo, hi), integrality=integ,
               bounds=Bounds(vlb, vub),
               options={'time_limit': time_limit, 'mip_rel_gap': 1e-9, 'disp': disp})
    dt = time.time() - t0

    inc_val, inc_f1 = None, None
    best_assign = lb_assign
    if res.x is not None:
        x = res.x[:nx].reshape(Gc, K)
        full = M.argmax(axis=1).copy(); full[conflicted] = x.argmax(axis=1)
        inc_val, inc_f1 = b.macro_f1_of_assignment(M, full)
        if inc_val > LB:
            best_assign = full
    dual = getattr(res, 'mip_dual_bound', None)
    ub_milp = float(-dual) if dual is not None and np.isfinite(dual) else None
    lower = max(LB, inc_val if inc_val is not None else -1)
    upper = min(UB_ceil, ub_milp) if ub_milp is not None else UB_ceil
    _, f1_best = b.macro_f1_of_assignment(M, best_assign)
    return {
        'status': res.message, 'solve_seconds': dt, 'conflicted_groups': int(Gc),
        'upper_bound_mean_of_ceilings': UB_ceil,
        'lower_bound_multistart_ascent': LB, 'ascent_log': lb_log,
        'box_width_per_class': (U - L).tolist(),
        'milp_incumbent_macro_f1': inc_val, 'milp_dual_upper_bound': ub_milp,
        'joint_macro_f1_lower': lower, 'joint_macro_f1_upper': upper,
        'gap_pt': 100 * (upper - lower),
        'exact': bool(upper - lower < 1e-6),
        'per_class_f1_at_best_assignment': f1_best.tolist(),
        'per_class_f1_ceiling': C.tolist(),
        'groups_assigned_to_absent_class': int(sum(M[g, best_assign[g]] == 0
                                                   for g in np.where(conflicted)[0])),
    }


# ───────────────────────── brute-force verification ─────────────────────────
def brute_force(M):
    K = M.shape[1]
    conflicted = np.where((M > 0).sum(axis=1) > 1)[0]
    base = M.argmax(axis=1)
    best = -1.0
    for combo in itertools.product(range(K), repeat=len(conflicted)):
        a = base.copy(); a[conflicted] = combo
        best = max(best, b.macro_f1_of_assignment(M, a)[0])
    return best


def self_test(n=30, seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(n):
        K = int(rng.integers(3, 5)); Gc = 7 if K == 4 else 9
        rows = []
        for c in range(K):                                   # pure groups, one big class
            for _ in range(int(rng.integers(1, 4))):
                v = np.zeros(K, int); v[c] = int(rng.integers(1, 400 if c == 0 else 30)); rows.append(v)
        for _ in range(Gc):                                  # conflicted groups
            v = np.zeros(K, int)
            cls = rng.choice(K, size=int(rng.integers(2, K + 1)), replace=False)
            v[cls] = rng.integers(1, 25, size=len(cls)); rows.append(v)
        M = np.array(rows)
        bf = brute_force(M)
        j = joint_milp_tight(M, time_limit=60)
        worst = max(worst, abs(j['joint_macro_f1_lower'] - bf), abs(j['joint_macro_f1_upper'] - bf))
    return worst


# ───────────────────────── representations ─────────────────────────
def load_groups():
    tr = pd.read_csv(m.TRAIN_PATH); te = pd.read_csv(m.TEST_PATH)
    for df in (tr, te):
        df.drop(columns=['id', 'label'], inplace=True, errors='ignore')
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')
    class_names = sorted(tr['attack_cat'].unique())
    y = te.pop('attack_cat').map({c: i for i, c in enumerate(class_names)}).to_numpy()
    groups = {'R0_raw_equals_R1_tree_input': b.exact_group_ids(te)}

    Xtr2, Xte2, _, yte2, *_ = m.load_and_preprocess()
    assert (yte2 == y).all()
    groups['R2_tkan_rf_input'] = b.exact_group_ids(Xte2)

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    Xtr1, Xte1, ytr1, yte1, _, _ = gb.load_and_preprocess('multiclass')
    assert (yte1 == y).all()
    Xf, _, yf, _ = train_test_split(Xtr1, ytr1, test_size=0.15, random_state=42, stratify=ytr1)
    d = xgb.QuantileDMatrix(Xf, label=yf, max_bin=256)
    indptr, values = d.get_quantile_cut()
    cuts = [np.asarray(values[indptr[f]:indptr[f + 1]], dtype=np.float32) for f in range(Xte1.shape[1])]
    cells = np.column_stack([np.searchsorted(cuts[f], Xte1[:, f], side='right') for f in range(Xte1.shape[1])])
    groups['H_xgb_quantile_cells_256'] = b.exact_group_ids(cells)
    return groups, y, class_names


def main():
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            results = json.load(f)
    if 'self_test_max_abs_diff_vs_brute_force' not in results:
        t0 = time.time()
        results['self_test_max_abs_diff_vs_brute_force'] = self_test()
        print(f"self-test (30 instances vs brute force): max |diff| = "
              f"{results['self_test_max_abs_diff_vs_brute_force']:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    groups, y, class_names = load_groups()
    K = len(class_names)
    for name, gid in groups.items():
        if name in results and results[name].get('exact'):
            continue
        M = b.count_matrix(gid, y, K)
        print(f'\n=== {name}: {M.shape[0]} groups ===', flush=True)
        j = joint_milp_tight(M, time_limit=1800, disp=True)
        j['per_class_f1_at_best_assignment'] = dict(zip(class_names, j['per_class_f1_at_best_assignment']))
        j['per_class_f1_ceiling'] = dict(zip(class_names, j['per_class_f1_ceiling']))
        j['box_width_per_class'] = dict(zip(class_names, j['box_width_per_class']))
        results[name] = j
        print(json.dumps({k: v for k, v in j.items() if not isinstance(v, dict)}, indent=1), flush=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
