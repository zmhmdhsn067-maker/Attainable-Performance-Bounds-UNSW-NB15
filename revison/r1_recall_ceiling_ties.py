"""
Per-class recall "ceilings" under the accuracy-optimal (bound A) and macro-recall-optimal
(bound D) assignments, reported as RANGES over all optimal assignments.

When a conflicted group has two or more classes tied for the maximum, every choice among
them is optimal for the aggregate objective, but the per-class recalls differ. The submitted
Table 8 reported one arbitrary tie-break. Here, for each class c:
  minimum = ties always broken against c,  maximum = ties always broken in favour of c.
Both are attained by an optimal assignment, so the range is exact. That per-class recall
under an optimal assignment is not even unique strengthens the paper's point that recall
ceilings are not bounds.

Output: results/r1_recall_ceilings.json
"""
import os
import json
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('b', os.path.join(HERE, 'r1_bounds_all_representations.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def recall_range(score, M):
    """score: (G, K) objective per group and class; returns per-class (min, max) recall."""
    N = M.sum(axis=0).astype(float)
    best = score.max(axis=1, keepdims=True)
    tied = np.isclose(score, best, rtol=0, atol=1e-12) & (M > 0)
    unique = tied & (tied.sum(axis=1, keepdims=True) == 1)
    lo = (M * unique).sum(axis=0) / N
    hi = (M * tied).sum(axis=0) / N
    return lo, hi, int((tied.sum(axis=1) > 1).sum())


def main():
    te = pd.read_csv(b.m.TEST_PATH)
    te.drop(columns=['id', 'label'], inplace=True, errors='ignore')
    te['attack_cat'] = te['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')
    names = sorted(te['attack_cat'].unique())
    y = te.pop('attack_cat').map({c: i for i, c in enumerate(names)}).to_numpy()
    M = b.count_matrix(b.exact_group_ids(te), y, len(names))
    N = M.sum(axis=0)
    loA, hiA, tiesA = recall_range(M.astype(float), M)
    loD, hiD, tiesD = recall_range(M / N, M)
    out = {'groups_with_tied_maximum': {'accuracy_optimal': tiesA, 'macro_recall_optimal': tiesD}, 'per_class': {}}
    print(f'groups with a tied maximum: accuracy-optimal {tiesA}, macro-recall-optimal {tiesD}')
    print(f'{"class":<15} {"conflict":>8} {"acc-opt recall":>18} {"macro-rec-opt recall":>22} {"ratio range":>14}')
    conf, _ = b.conflict_rates(M)
    for c, n in enumerate(names):
        r = {'conflict_rate_pct': 100 * conf[c], 'accuracy_optimal_recall_pct': [100 * loA[c], 100 * hiA[c]],
             'macro_recall_optimal_recall_pct': [100 * loD[c], 100 * hiD[c]]}
        ratio = [hiD[c] / hiA[c] if hiA[c] else float('inf'), hiD[c] / loA[c] if loA[c] else float('inf')]
        r['ratio_range'] = ratio
        out['per_class'][n] = r
        print(f'{n:<15} {100*conf[c]:8.2f} {100*loA[c]:8.2f}–{100*hiA[c]:<8.2f} {100*loD[c]:10.2f}–{100*hiD[c]:<10.2f} '
              f'{ratio[0]:6.2f}–{ratio[1]:.2f}')
    with open(os.path.join(HERE, 'results', 'r1_recall_ceilings.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
