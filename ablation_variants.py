"""
E6/E7 infrastructure: 5-seed replication of the canonical hybrid, plus three single-factor
ablation controls (KAN-vs-MLP head, focal-vs-weighted-CE loss, balanced-vs-uniform sampling).

Reuses evaluate_unsw_nb15_hybrid.py's tokenizer/transformer/pooling/training machinery
unchanged; only the classifier head, loss criterion, or sampling strategy is swapped per
config, holding everything else at the canonical (74.23/49.29) configuration fixed.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import evaluate_unsw_nb15_hybrid as m

STATUS_PATH = os.path.join(m.RESULTS_DIR, 'ablation_status.json')
RESULTS_PATH = os.path.join(m.RESULTS_DIR, 'ablation_results.json')

MLP_HIDDEN = 1098  # matched to KANClassifier's 67,104 params (see param-count check)


class MLPClassifier(nn.Module):
    """Parameter-matched MLP replacement for KANClassifier (E7a): same
    LayerNorm -> Linear -> activation -> LayerNorm -> Dropout -> Linear shape,
    GELU instead of the KAN's spline+SiLU edges, ~67,084 params vs KAN's 67,104."""

    def __init__(self, in_features: int, hidden_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.norm_hidden = nn.LayerNorm(hidden_features)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, n_classes)

    def forward(self, x):
        x = self.act(self.fc1(self.norm_in(x)))
        x = self.drop(self.norm_hidden(x))
        return self.fc2(x)


class HybridTransformerVariant(nn.Module):
    """Same as HybridTransformerKAN but with a pluggable classifier head."""

    def __init__(self, cont_idx, cat_idx, cat_cardinalities, n_classes,
                 head_type='kan', d_model=m.D_MODEL, n_heads=m.N_HEADS,
                 n_layers=m.N_TRANSFORMER_LAYERS, clf_hidden=m.CLASSIFIER_HIDDEN,
                 dropout=m.DROPOUT):
        super().__init__()
        self.tokenizer = m.MixedTokenizer(cont_idx, cat_idx, cat_cardinalities, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = m.AttentionPool(d_model, n_heads=n_heads)

        if head_type == 'kan':
            self.head = m.KANClassifier(d_model, clf_hidden, n_classes, dropout=dropout)
        elif head_type == 'mlp':
            self.head = MLPClassifier(d_model, MLP_HIDDEN, n_classes, dropout=dropout)
        else:
            raise ValueError(head_type)

    def forward(self, x):
        tok = self.tokenizer(x)
        tok = self.transformer(tok)
        pooled = self.pool(tok)
        return self.head(pooled)


def canonical_class_weights(y_tr, class_names):
    """Reproduces run_pipeline's canonical (hand-tuned, PSO-independent) weight vector:
    sqrt-softened balanced weights, capped at 8x, with Analysis/Backdoor/DoS pinned to the
    neutral floor of 1.0. This is the 74.23/49.29 canonical configuration's weight source."""
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = np.sqrt(cw)
    cw = np.clip(cw, cw.min(), cw.min() * m.WEIGHT_CAP_RATIO)
    for cls in ['Analysis', 'Backdoor', 'DoS']:
        if cls in class_names:
            cw[class_names.index(cls)] = 1.0
    return cw


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(STATUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)


def run_config(name, seed, head_type, loss_type, sampling, device='cpu'):
    """Runs one full training+eval config and returns its summary dict."""
    print(f"\n{'#'*72}\n  CONFIG: {name}  (seed={seed}, head={head_type}, loss={loss_type}, "
          f"sampling={sampling})\n{'#'*72}", flush=True)

    (X_tr_full, X_te, y_tr_full, y_te, feat_names, class_names,
     cont_idx, cat_idx, cat_card) = m.load_and_preprocess()
    n_classes = int(np.unique(y_tr_full).size)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=m.VAL_SIZE, random_state=seed, stratify=y_tr_full)

    cw = canonical_class_weights(y_tr, class_names)
    class_weights_t = torch.tensor(cw, dtype=torch.float32)

    if sampling == 'balanced':
        sample_weights = class_weights_t[torch.as_tensor(y_tr, dtype=torch.long)]
    elif sampling == 'uniform':
        sample_weights = None
    else:
        raise ValueError(sampling)

    if loss_type == 'focal':
        criterion = None  # train_model's default: FocalLoss(alpha=None, gamma=FOCAL_GAMMA)
    elif loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        raise ValueError(loss_type)

    torch.manual_seed(seed)
    model = HybridTransformerVariant(cont_idx, cat_idx, cat_card, n_classes, head_type=head_type)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_path = os.path.join(m.RESULTS_DIR, f'checkpoint_ablation_{name}.pt')

    t0 = time.time()
    t_l, v_l = m.train_model(model, X_tr, y_tr, X_val, y_val, device,
                              sample_weights=sample_weights, checkpoint_path=ckpt_path,
                              criterion=criterion)
    train_time = time.time() - t0
    best_epoch = int(np.argmin(v_l)) + 1

    clf_res = m.evaluate(model, X_te, y_te, class_names, device)

    result = {
        'name': name, 'seed': seed, 'head_type': head_type, 'loss_type': loss_type,
        'sampling': sampling, 'n_params': n_params,
        'accuracy': clf_res['accuracy'], 'balanced_accuracy': clf_res['balanced_accuracy'],
        'precision_weighted': clf_res['precision_weighted'], 'recall_weighted': clf_res['recall_weighted'],
        'f1_weighted': clf_res['f1_weighted'], 'precision_macro': clf_res['precision_macro'],
        'recall_macro': clf_res['recall_macro'], 'f1_macro': clf_res['f1_macro'],
        'report_dict': clf_res['report_dict'],
        'total_epochs': len(t_l), 'best_epoch': best_epoch,
        'train_time_s': train_time, 'time_to_best_epoch_s': train_time * best_epoch / len(t_l),
        'val_loss_at_epoch30': v_l[29] if len(v_l) >= 30 else None,
        'val_loss_final': v_l[-1],
    }
    print(f"\n  RESULT [{name}]: acc={result['accuracy']:.4f}  bal_acc={result['balanced_accuracy']:.4f}  "
          f"f1_macro={result['f1_macro']:.4f}  f1_weighted={result['f1_weighted']:.4f}  "
          f"params={n_params}  time={train_time:.0f}s  epochs={len(t_l)}  best_ep={best_epoch}", flush=True)
    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)

    configs = []
    # E6: 5-seed replication of the canonical config (kan head, focal loss, balanced sampling)
    for seed in [42, 1, 7, 123, 2024]:
        configs.append((f'e6_seed{seed}', seed, 'kan', 'focal', 'balanced'))
    # E7a: KAN head -> parameter-matched MLP head
    configs.append(('e7a_mlp_head', 42, 'mlp', 'focal', 'balanced'))
    # E7b: focal loss -> weighted cross-entropy
    configs.append(('e7b_weighted_ce', 42, 'kan', 'weighted_ce', 'balanced'))
    # E7d: balanced sampling -> uniform sampling
    configs.append(('e7d_uniform_sampling', 42, 'kan', 'focal', 'uniform'))

    status = load_status()
    results = load_results()

    for name, seed, head_type, loss_type, sampling in configs:
        if status.get(name) == 'done':
            print(f'[skip] {name} already completed', flush=True)
            continue
        result = run_config(name, seed, head_type, loss_type, sampling, device=device)
        results[name] = result
        status[name] = 'done'
        save_results(results)
        save_status(status)
        print(f'[saved] {name}', flush=True)

    print('\nAll configs complete.')
    print('acc mean/std across E6 seeds:')
    e6_accs = [results[f'e6_seed{s}']['accuracy'] for s in [42, 1, 7, 123, 2024]]
    e6_f1ms = [results[f'e6_seed{s}']['f1_macro'] for s in [42, 1, 7, 123, 2024]]
    e6_bals = [results[f'e6_seed{s}']['balanced_accuracy'] for s in [42, 1, 7, 123, 2024]]
    print(f"  accuracy: {np.mean(e6_accs)*100:.2f}% +/- {np.std(e6_accs)*100:.2f}%")
    print(f"  f1_macro: {np.mean(e6_f1ms):.4f} +/- {np.std(e6_f1ms):.4f}")
    print(f"  balanced_accuracy: {np.mean(e6_bals)*100:.2f}% +/- {np.std(e6_bals)*100:.2f}%")


if __name__ == '__main__':
    main()
