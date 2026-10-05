"""
PSO-based per-class sample-weight optimizer for the UNSW-NB15 Transformer-KAN model.

evaluate_unsw_nb15_hybrid.py balances classes by drawing training minibatches from a
class-weighted distribution (oversampling rare/hard classes) instead of loss-level
reweighting -- see `sample_weights` in that file. The per-class weight vector was, until
now, hand-tuned: sqrt-softened + capped 'balanced' weights, with Analysis/Backdoor/DoS
pinned back to the natural (unboosted) floor because a duplicate-feature-vector audit
found 72-80% of THEIR rows share an identical 42-feature vector with a CONFLICTING
official label in the training data -- oversampling those rows just repeats
contradictions, which measurably hurt DoS recall without helping Analysis/Backdoor at all
(both stay ~0.00 regardless of weighting, since the ground truth itself is ambiguous for
most of their rows).

This script replaces that hand-tuning with a search: Particle Swarm Optimization over the
10-dimensional weight vector, using validation F1-macro as the fitness signal (never the
test set -- same train/val/test discipline as the rest of this project).

Compute budget note: a full run (150 epochs, 149K training rows) takes ~45-75 minutes on
this CPU. Real PSO needs dozens of fitness evaluations (swarm x iterations), so evaluating
every particle with a full run would take days. Each fitness evaluation here instead uses
a SHORT, FIXED-length proxy training run (no early stopping) on a stratified SUBSAMPLE of
the training/validation data. This trades fitness-signal precision (especially for
ultra-rare classes like Worms, where the val subsample may contain single-digit examples)
for being able to actually run the search to completion. The result is a *good starting
point*, not a guaranteed optimum -- the discovered weights should still be validated with
a full run before being trusted as final.

Output: results/pso_class_weights.json (class_names, weights, val_f1_macro, per-iteration
history). evaluate_unsw_nb15_hybrid.py automatically uses this file if present (falls back
to the hand-tuned heuristic otherwise) -- see the weight-loading block in run_pipeline().
"""
import os
import json
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import evaluate_unsw_nb15_hybrid as base

SEED = base.SEED
RESULTS_DIR = base.RESULTS_DIR
WEIGHTS_PATH = os.path.join(RESULTS_DIR, 'pso_class_weights.json')

# ── PSO / proxy-training config ──────────────────────────────────────────────────
N_PARTICLES     = 6
N_ITERATIONS    = 6
PROXY_EPOCHS    = 10
TRAIN_SUBSAMPLE = 30000
VAL_SUBSAMPLE   = 8000
W_LO, W_HI      = 0.3, 6.0
INERTIA, C1, C2 = 0.7, 1.6, 1.6

rng = np.random.RandomState(SEED)


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int):
    idx_parts = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        take = min(len(c_idx), max(1, int(round(n * len(c_idx) / len(y)))))
        idx_parts.append(rng.choice(c_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _heuristic_weights(y_tr_full: np.ndarray, class_names: list) -> np.ndarray:
    """Reproduce evaluate_unsw_nb15_hybrid.py's hand-tuned heuristic, used only to
    seed one PSO particle as a warm start (not blindly trusted as the answer).
    Neutral/"no boost" reference is 1.0 in compute_class_weight('balanced')'s own
    convention, not cw.min() (that's whichever weight the most-frequent class landed
    on, which is < 1.0 and actively suppresses a pinned class below its natural rate —
    this collapsed DoS recall to 0.06 in an earlier run before the fix)."""
    cw = base.compute_class_weight('balanced', classes=np.unique(y_tr_full), y=y_tr_full)
    cw = np.sqrt(cw)
    cw = np.clip(cw, cw.min(), cw.min() * base.WEIGHT_CAP_RATIO)
    for cls in ['Analysis', 'Backdoor', 'DoS']:
        if cls in class_names:
            cw[class_names.index(cls)] = 1.0
    return cw


def make_proxy_data():
    (X_tr_full, X_te, y_tr_full, y_te, feat_names, class_names,
     cont_idx, cat_idx, cat_card) = base.load_and_preprocess()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=base.VAL_SIZE, random_state=SEED, stratify=y_tr_full)

    heuristic = _heuristic_weights(y_tr, class_names)

    X_tr_s, y_tr_s = _stratified_subsample(X_tr, y_tr, TRAIN_SUBSAMPLE)
    X_val_s, y_val_s = _stratified_subsample(X_val, y_val, VAL_SUBSAMPLE)

    print(f"  Proxy train subsample: {X_tr_s.shape}  |  Proxy val subsample: {X_val_s.shape}")
    return X_tr_s, y_tr_s, X_val_s, y_val_s, cont_idx, cat_idx, cat_card, class_names, heuristic


def fitness(weight_vec: np.ndarray, data: tuple, n_classes: int, device: str = 'cpu') -> float:
    X_tr_s, y_tr_s, X_val_s, y_val_s, cont_idx, cat_idx, cat_card, class_names, _ = data

    torch.manual_seed(SEED)
    model = base.HybridTransformerKAN(cont_idx, cat_idx, cat_card, n_classes).to(device)

    cw = torch.tensor(weight_vec, dtype=torch.float32)
    sample_weights = cw[torch.as_tensor(y_tr_s, dtype=torch.long)].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=base.LR)
    crit = base.FocalLoss(alpha=None, gamma=base.FOCAL_GAMMA)

    Xtr = torch.as_tensor(X_tr_s).to(device)
    ytr = torch.as_tensor(y_tr_s).to(device)
    Xv  = torch.as_tensor(X_val_s).to(device)

    n = Xtr.shape[0]
    bs = min(base.BATCH_SIZE, n)

    model.train()
    for _ in range(PROXY_EPOCHS):
        perm = torch.multinomial(sample_weights, n, replacement=True)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xv).argmax(1).cpu().numpy()
    return f1_score(y_val_s, preds, average='macro', zero_division=0)


def run_pso():
    print(f"\n{'#' * 72}")
    print(f"  PSO SEARCH: per-class sample-weight vector (proxy-trained fitness)")
    print(f"  particles={N_PARTICLES}  iterations={N_ITERATIONS}  proxy_epochs={PROXY_EPOCHS}")
    print(f"{'#' * 72}")
    t_start = time.time()

    data = make_proxy_data()
    class_names = data[-2]
    heuristic = data[-1]
    n_classes = len(class_names)
    dim = n_classes

    print("  Heuristic warm-start weights:")
    for name, w in zip(class_names, heuristic):
        print(f"    {name:<15} {w:.3f}")

    positions = rng.uniform(W_LO, W_HI, size=(N_PARTICLES, dim))
    positions[0] = np.clip(heuristic, W_LO, W_HI)
    velocities = rng.uniform(-1, 1, size=(N_PARTICLES, dim)) * 0.1

    pbest_pos = positions.copy()
    pbest_val = np.full(N_PARTICLES, -np.inf)
    gbest_pos = positions[0].copy()
    gbest_val = -np.inf

    history = []

    for it in range(N_ITERATIONS):
        for p in range(N_PARTICLES):
            val = fitness(positions[p], data, n_classes)
            if val > pbest_val[p]:
                pbest_val[p] = val
                pbest_pos[p] = positions[p].copy()
            if val > gbest_val:
                gbest_val = val
                gbest_pos = positions[p].copy()
            print(f"  it {it + 1}/{N_ITERATIONS}  particle {p + 1}/{N_PARTICLES}  "
                  f"F1_macro(val)={val:.4f}  gbest={gbest_val:.4f}  "
                  f"elapsed={time.time() - t_start:.0f}s", flush=True)

        r1 = rng.uniform(size=(N_PARTICLES, dim))
        r2 = rng.uniform(size=(N_PARTICLES, dim))
        velocities = (INERTIA * velocities
                      + C1 * r1 * (pbest_pos - positions)
                      + C2 * r2 * (gbest_pos - positions))
        positions = np.clip(positions + velocities, W_LO, W_HI)

        history.append({'iteration': it + 1, 'gbest_f1_macro': float(gbest_val),
                         'gbest_weights': gbest_pos.tolist()})
        print(f"  -- iteration {it + 1} done, gbest F1_macro(val)={gbest_val:.4f} --\n", flush=True)

    print(f"\n  PSO done in {time.time() - t_start:.0f}s. Best proxy val F1_macro={gbest_val:.4f}")
    print("  Best per-class weights found:")
    for name, w in zip(class_names, gbest_pos):
        print(f"    {name:<15} {w:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'class_names': class_names,
            'weights': gbest_pos.tolist(),
            'proxy_val_f1_macro': float(gbest_val),
            'heuristic_weights': heuristic.tolist(),
            'history': history,
        }, f, indent=2)
    print(f"  Saved -> {WEIGHTS_PATH}")
    print("  Run evaluate_unsw_nb15_hybrid.py to validate these weights with a full training run.")


if __name__ == '__main__':
    run_pso()
