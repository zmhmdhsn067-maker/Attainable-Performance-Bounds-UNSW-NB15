"""
Exact jointly attainable macro-F1, symmetry-free formulation (reviewer comments R1-1, R2-1).

Conflicted groups with identical class-count profiles are interchangeable: on R0 the 575
conflicted groups have only 75 distinct profiles (one occurs 221 times), which makes the
per-group MILP of r1_joint_bound_tight.py highly symmetric. Here the decision is instead the
integer y_tc = number of groups of profile t labelled c (sum_c y_tc = m_t), written in binary,
y_tc = sum_k 2^k b_tck, so that every product f_c * b_tck is linearised exactly by McCormick
envelopes on the box f_c in [L_c, U_c] (U_c = per-class ceiling, L_c from the attained lower
bound; see r1_joint_bound_tight.py for why neither excludes an optimal solution):
    u_tck >= L_c b_tck,   u_tck >= U_c b_tck + f_c - U_c,   u_tck >= 0.
F1 rows are divided by (N_c + A_c) for numerical conditioning:
    f_c + sum_{t,k} 2^k (s_t u_tck - 2 n_tc b_tck) / (N_c + A_c) <= 2 A_c / (N_c + A_c).
Any assignment with the optimal counts attains the optimum, which is recomputed exactly.
"""
import os
import sys
import json
import time
import importlib.util

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t', os.path.join(HERE, 'r1_joint_bound_tight.py'))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
b = t.b
OUT = os.path.join(HERE, 'results', 'r1_joint_bound_exact.json')


def joint_milp_aggregated(M, time_limit=1800, disp=False):
    K = M.shape[1]
    N = M.sum(axis=0).astype(float)
    C = np.array([c['f1'] for c in b.bound_C(M)])
    t0 = time.time()
    LB, lb_assign, lb_log = t.multi_start_lb(M)
    t_lb = time.time() - t0
    conflicted = np.where((M > 0).sum(axis=1) > 1)[0]
    A = np.delete(M, conflicted, axis=0).sum(axis=0).astype(float)
    prof, inv, mult = np.unique(M[conflicted], axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    T = len(prof)
    s = prof.sum(axis=1).astype(float)
    U = C.copy()
    L = np.clip(K * LB - (C.sum() - C) - 1e-9, 0.0, U)
    scale = N + A

    # enumerate (t, c, k) triples
    tt, cc, kk = [], [], []
    for ti in range(T):
        nbits = int(np.ceil(np.log2(mult[ti] + 1)))
        for c in range(K):
            for k in range(nbits):
                tt.append(ti); cc.append(c); kk.append(k)
    tt, cc, kk = np.array(tt), np.array(cc), np.array(kk)
    nb = len(tt)
    w = 2.0 ** kk
    bi = np.arange(nb); ui = nb + bi; fi = 2 * nb + np.arange(K)
    R, Cn, V, lo, hi = [], [], [], [], []
    r = 0
    R.append(r + tt); Cn.append(bi); V.append(w)                           # sum_c y_tc = m_t
    lo.append(mult.astype(float)); hi.append(mult.astype(float)); r += T
    R.append(r + np.arange(K)); Cn.append(fi); V.append(np.ones(K))        # scaled F1 rows
    R.append(r + cc); Cn.append(ui); V.append(w * s[tt] / scale[cc])
    ntc = prof[tt, cc].astype(float); nz = ntc > 0
    R.append(r + cc[nz]); Cn.append(bi[nz]); V.append(-2.0 * w[nz] * ntc[nz] / scale[cc[nz]])
    lo.append(np.full(K, -np.inf)); hi.append(2.0 * A / scale); r += K
    ridx = r + bi                                                            # u >= U b + f - U
    R += [ridx, ridx, ridx]; Cn += [ui, fi[cc], bi]; V += [np.ones(nb), -np.ones(nb), -U[cc]]
    lo.append(-U[cc]); hi.append(np.full(nb, np.inf)); r += nb
    sel = L[cc] > 0                                                          # u >= L b
    ns = int(sel.sum()); ridx = r + np.arange(ns)
    R += [ridx, ridx]; Cn += [ui[sel], bi[sel]]; V += [np.ones(ns), -L[cc][sel]]
    lo.append(np.zeros(ns)); hi.append(np.full(ns, np.inf)); r += ns
    R.append(np.full(K, r)); Cn.append(fi); V.append(np.ones(K))            # objective cut
    lo.append(np.array([K * LB - 1e-7])); hi.append(np.array([np.inf])); r += 1

    Acon = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cn))),
                      shape=(r, 2 * nb + K)).tocsr()
    cobj = np.zeros(2 * nb + K); cobj[fi] = -1.0 / K
    integ = np.zeros(2 * nb + K); integ[:nb] = 1
    vlb = np.concatenate([np.zeros(nb), np.zeros(nb), L])
    vub = np.concatenate([np.ones(nb), U[cc], U])
    t1 = time.time()
    res = milp(cobj, constraints=LinearConstraint(Acon, np.concatenate(lo), np.concatenate(hi)),
               integrality=integ, bounds=Bounds(vlb, vub),
               options={'time_limit': time_limit, 'mip_rel_gap': 1e-9, 'disp': disp})
    dt = time.time() - t1

    best_assign, inc_val = lb_assign, None
    if res.x is not None:
        beta = np.round(res.x[:nb]).astype(int)
        y = np.zeros((T, K), dtype=int)
        np.add.at(y, (tt, cc), beta * (2 ** kk))
        assert (y.sum(axis=1) == mult).all(), 'count constraint violated after rounding'
        full = M.argmax(axis=1).copy()
        for ti in range(T):                     # distribute each profile's groups by the counts
            members = conflicted[inv == ti]
            labels = np.repeat(np.arange(K), y[ti])
            full[members] = labels
        inc_val, _ = b.macro_f1_of_assignment(M, full)
        if inc_val > LB:
            best_assign = full
    dual = getattr(res, 'mip_dual_bound', None)
    ub_milp = float(-dual) if dual is not None and np.isfinite(dual) else None
    lower = max(LB, inc_val if inc_val is not None else -1.0)
    upper = min(float(C.mean()), ub_milp) if ub_milp is not None else float(C.mean())
    _, f1_best = b.macro_f1_of_assignment(M, best_assign)
    TP = np.zeros(K); P = np.zeros(K)
    np.add.at(TP, best_assign, M[np.arange(len(M)), best_assign])
    np.add.at(P, best_assign, M.sum(axis=1))
    return {
        'status': res.message, 'solve_seconds': dt, 'lower_bound_search_seconds': t_lb,
        'conflicted_groups': int(len(conflicted)), 'distinct_conflict_profiles': int(T),
        'binary_variables': int(nb),
        'upper_bound_mean_of_ceilings': float(C.mean()),
        'lower_bound_multistart_ascent': float(LB), 'ascent_log': lb_log,
        'milp_incumbent_macro_f1': inc_val, 'milp_dual_upper_bound': ub_milp,
        'joint_macro_f1_lower': float(lower), 'joint_macro_f1_upper': float(upper),
        'gap_pt': 100 * (upper - lower), 'exact': bool(upper - lower < 1e-6),
        'per_class_f1_at_joint_optimum': f1_best.tolist(),
        'per_class_precision_at_joint_optimum': np.where(P > 0, TP / np.maximum(P, 1), 0).tolist(),
        'per_class_recall_at_joint_optimum': (TP / N).tolist(),
        'per_class_f1_ceiling': C.tolist(),
        'groups_assigned_to_absent_class': int(sum(M[g, best_assign[g]] == 0 for g in conflicted)),
    }


def self_test(n=30, seed=3):
    """Brute force on small instances with repeated conflict profiles."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        K = int(rng.integers(3, 5)); Gc = 7 if K == 4 else 9
        rows = []
        for c in range(K):
            for _ in range(int(rng.integers(1, 4))):
                v = np.zeros(K, int); v[c] = int(rng.integers(1, 400 if c == 0 else 30)); rows.append(v)
        base = []
        for _ in range(int(rng.integers(2, 4))):
            v = np.zeros(K, int)
            cls = rng.choice(K, size=int(rng.integers(2, K + 1)), replace=False)
            v[cls] = rng.integers(1, 25, size=len(cls)); base.append(v)
        for g in range(Gc):
            rows.append(base[int(rng.integers(len(base)))].copy())
        M = np.array(rows)
        bf = t.brute_force(M)
        j = joint_milp_aggregated(M, time_limit=60)
        worst = max(worst, abs(j['joint_macro_f1_lower'] - bf), abs(j['joint_macro_f1_upper'] - bf))
    return worst


def main():
    only = sys.argv[1:]                          # optional: names of representations to run
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            results = json.load(f)
    if 'self_test_max_abs_diff_vs_brute_force' not in results:
        t0 = time.time()
        results['self_test_max_abs_diff_vs_brute_force'] = self_test()
        print(f"self-test (30 instances with repeated profiles, vs brute force): max |diff| = "
              f"{results['self_test_max_abs_diff_vs_brute_force']:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    groups, y, class_names = t.load_groups()
    K = len(class_names)
    for name, gid in groups.items():
        if (only and name not in only) or (name in results and results[name].get('exact')):
            continue
        M = b.count_matrix(gid, y, K)
        print(f'\n=== {name}: {M.shape[0]} groups ===', flush=True)
        j = joint_milp_aggregated(M, time_limit=1800, disp=True)
        for key in ('per_class_f1_at_joint_optimum', 'per_class_precision_at_joint_optimum',
                    'per_class_recall_at_joint_optimum', 'per_class_f1_ceiling'):
            j[key] = dict(zip(class_names, j[key]))
        results[name] = j
        print(json.dumps({k: v for k, v in j.items() if not isinstance(v, dict)}, indent=1), flush=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
