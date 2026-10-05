"""
Hybrid Deep Learning for Network Intrusion Detection — UNSW-NB15 (multiclass, attack_cat)

Architecture:

  1. Per-feature numerical tokenizer (FT-Transformer style): each scaled feature gets
     its own learned embedding -> a (n_features, d_model) token sequence.
  2. Transformer encoder (multi-head self-attention over feature tokens): learns
     which feature-pairs matter jointly (replaces any fixed feature ranking).
  3. Mean pool + KAN (Kolmogorov-Arnold Network) classification head: learnable
     B-spline edge functions instead of fixed activations + linear weights.

Environment layout (self-contained, does not touch any other project folder):
  UNSW_NB15_Hybrid/
    data/     UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv  (copies)
    results/  all outputs for this dataset (txt/csv reports + png plots)
    evaluate_unsw_nb15_hybrid.py  (this file)

Target: 'attack_cat' (10 classes: Normal + 9 attack categories). The binary 'label'
column is dropped — it is a deterministic function of attack_cat and would leak the
target trivially if kept as a feature. 'id' is dropped (row counter, not a feature).

Validation is carved out of the TRAINING set only (85/15 stratified split) — the test
set is touched exactly once, at final evaluation.
"""

import os
import sys
import json
import math
import warnings
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 42
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TRAIN_PATH  = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH   = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

TARGET_COL = 'attack_cat'   # multiclass target; 'label' (binary) is dropped as leakage

VAL_SIZE      = 0.15   # carved out of the training set only; test set stays untouched until final eval
CAT_COLS      = ['proto', 'service', 'state']  # categorical -> learned embedding, not scaled linear tokenizer
D_MODEL       = 48
N_HEADS       = 4
N_TRANSFORMER_LAYERS = 3
CLASSIFIER_HIDDEN = 96
KAN_GRID_SIZE = 8      # number of B-spline intervals per KAN edge
KAN_SPLINE_ORDER = 3   # cubic B-splines
EPOCHS        = 150
LR            = 1e-3
BATCH_SIZE    = 8192   # measured fastest CPU throughput point for this model (~150s/epoch vs ~160s at 2048)
DROPOUT       = 0.2
PATIENCE      = 15
EP_THRESHOLD  = 0.01
STABILITY_E   = 5
FOCAL_GAMMA   = 1.0    # focal-loss focusing parameter (lowered from 2.0 — was over-correcting: see WEIGHT_CAP_RATIO note)
WEIGHT_CAP_RATIO = 8   # cap max/min class-weight ratio after sqrt-softening (raw 'balanced' ratio hits ~430:1 here).
                       # 15x stacked with focal loss caused the model to over-fire on rare classes (Worms/Shellcode/
                       # Backdoor recall>0.8 but precision<0.23) while shredding Normal recall (28% of Normal -> Fuzzers).

TASK = 'unsw_nb15_hybrid'

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)   # measured fastest on this CPU (14 logical threads showed worse per-batch time than 4)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def _safe_encode(series: pd.Series, le: LabelEncoder) -> np.ndarray:
    known    = set(le.classes_)
    fallback = le.classes_[0]
    return le.transform(series.apply(lambda x: x if x in known else fallback))


def load_and_preprocess():
    print("  Reading CSV file(s)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    for df in (train_df, test_df):
        df.drop(columns=['id', 'label'], inplace=True, errors='ignore')
        df[TARGET_COL] = df[TARGET_COL].fillna('Normal').astype(str).str.strip().replace('', 'Normal')

    le_target = LabelEncoder()
    y_train   = le_target.fit_transform(train_df.pop(TARGET_COL)).astype(np.int64)
    y_test    = _safe_encode(test_df.pop(TARGET_COL), le_target).astype(np.int64)
    class_names = list(le_target.classes_)

    # proto/service/state are categorical codes with no ordinal meaning (label-encoded
    # alphabetically) — they get a learned lookup embedding in the model, not the scaled
    # linear tokenizer used for genuinely numeric features, so we track their cardinality
    # and keep them OUT of the StandardScaler below.
    cat_cardinalities = []
    for col in CAT_COLS:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            test_df[col]  = _safe_encode(test_df[col].astype(str), le)
            cat_cardinalities.append(len(le.classes_))

    feature_names = list(train_df.columns)
    cat_idx  = [feature_names.index(c) for c in CAT_COLS if c in feature_names]
    cont_idx = [i for i in range(len(feature_names)) if i not in cat_idx]

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    scaler = StandardScaler()
    X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
    X_test[:, cont_idx]  = scaler.transform(X_test[:, cont_idx])

    return (X_train, X_test, y_train, y_test, feature_names, class_names,
            cont_idx, cat_idx, cat_cardinalities)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FT-TOKENIZER + TRANSFORMER + KAN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class NumericTokenizer(nn.Module):
    """Per-feature linear embedding (FT-Transformer style): x_i -> x_i * W_i + b_i.
    Turns a (batch, n_features) row into a (batch, n_features, d_model) token sequence."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class MixedTokenizer(nn.Module):
    """FT-Transformer tokenizer for mixed continuous/categorical tabular data.
    Continuous features get the per-feature linear embedding above. Categorical
    features (proto/service/state — label-encoded integer codes with no ordinal
    meaning) get a learned lookup embedding instead, since x_i * W_i would assume
    a false ordinal relationship between arbitrary category codes."""

    def __init__(self, cont_idx: list, cat_idx: list, cat_cardinalities: list, d_model: int):
        super().__init__()
        self.cont_idx = cont_idx
        self.cat_idx  = cat_idx
        self.cont_tokenizer = NumericTokenizer(len(cont_idx), d_model) if cont_idx else None
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.cont_tokenizer is not None:
            toks.append(self.cont_tokenizer(x[:, self.cont_idx]))
        if self.cat_embeddings:
            cat_x = x[:, self.cat_idx].long()
            cat_toks = torch.stack(
                [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
            toks.append(cat_toks)
        return torch.cat(toks, dim=1)


class KANLinear(nn.Module):
    """Single Kolmogorov-Arnold layer (Liu et al., 2024 / 'efficient-kan' formulation).
    Replaces a fixed activation + linear weight with a learnable univariate B-spline
    function on every input-output edge, plus a residual SiLU base branch. Input is
    tanh-squashed first so activations always land inside the spline grid range,
    regardless of the scale of whatever feeds this layer."""

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = KAN_GRID_SIZE, spline_order: int = KAN_SPLINE_ORDER,
                 scale_noise: float = 0.1, scale_base: float = 1.0, scale_spline: float = 1.0,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
             + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer('grid', grid)

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(grid_size + 1, in_features, out_features) - 0.5) * scale_noise / grid_size
            )
            self.spline_weight.data.copy_(
                scale_spline * self._curve2coeff(self.grid.T[spline_order:-spline_order], noise)
            )

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self._b_splines(x).view(x.size(0), -1),
                               self.spline_weight.view(self.out_features, -1))
        return base_out + spline_out


class AttentionPool(nn.Module):
    """Learnable-query attention pooling: replaces uniform mean pooling over the
    feature-token sequence with a weighted combination the model learns per example.
    Mean pooling gives every one of the 42 feature tokens equal say regardless of
    whether it's diagnostic for the current row; a tree-based baseline (Random Forest)
    effectively does per-example feature selection via its splits and outperformed
    mean-pooling on Normal/Fuzzers separation — attention pooling is the Transformer-
    native equivalent of that selectivity."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class KANClassifier(nn.Module):
    """Two-layer KAN head: pooled token embedding -> hidden spline layer -> class logits.
    LayerNorm before each KANLinear keeps activations near unit variance so the internal
    tanh (see KANLinear.forward) squashes into its informative range instead of saturating
    flat or crowding near zero, whatever the raw scale of the incoming features is."""

    def __init__(self, in_features: int, hidden_features: int, n_classes: int, dropout: float = DROPOUT):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_features)
        self.kan1 = KANLinear(in_features, hidden_features)
        self.norm_hidden = nn.LayerNorm(hidden_features)
        self.drop = nn.Dropout(dropout)
        self.kan2 = KANLinear(hidden_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan1(self.norm_in(x))
        x = self.drop(self.norm_hidden(x))
        return self.kan2(x)


class HybridTransformerKAN(nn.Module):
    """FT-Tokenizer -> Transformer encoder (self-attention over feature tokens) -> KAN classifier."""

    def __init__(self, cont_idx: list, cat_idx: list, cat_cardinalities: list, n_classes: int,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_TRANSFORMER_LAYERS,
                 clf_hidden: int = CLASSIFIER_HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.tokenizer = MixedTokenizer(cont_idx, cat_idx, cat_cardinalities, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttentionPool(d_model, n_heads=n_heads)

        self.kan_head = KANClassifier(d_model, clf_hidden, n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x)          # (B, F, D)
        tok = self.transformer(tok)      # (B, F, D)
        pooled = self.pool(tok)          # (B, D)
        return self.kan_head(pooled)


class FocalLoss(nn.Module):
    """Multiclass focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t).
    Class balance here is handled at the DATA level (train_model samples minibatches
    from a class-balanced distribution, see `sample_weights`), not by an `alpha` loss
    penalty — stacking both compounded into severe over-correction (rare classes hit
    recall>0.8 at precision<0.23, while Normal recall collapsed because ~28% of it got
    pulled into Fuzzers). `gamma` alone still down-weights easy/well-classified examples."""

    def __init__(self, alpha: torch.Tensor = None, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        alpha_t = self.alpha.gather(0, target) if self.alpha is not None else 1.0
        loss = -alpha_t * (1 - p_t).pow(self.gamma) * logp_t
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, f'checkpoint_{TASK}.pt')


def train_model(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 device: str,
                 sample_weights: torch.Tensor = None,
                 checkpoint_path: str = CHECKPOINT_PATH,
                 criterion: nn.Module = None) -> tuple:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    # Ablation hook (E7b): pass a different criterion (e.g. weighted CrossEntropyLoss) to
    # isolate whether focal loss contributes anything beyond the class-balanced sampler.
    crit = criterion if criterion is not None else FocalLoss(alpha=None, gamma=FOCAL_GAMMA)
    crit = crit.to(device) if hasattr(crit, 'to') else crit

    Xtr = torch.as_tensor(X_tr).to(device)
    ytr = torch.as_tensor(y_tr).to(device)
    Xv  = torch.as_tensor(X_val).to(device)
    yv  = torch.as_tensor(y_val).to(device)
    sw  = sample_weights.to(device) if sample_weights is not None else None

    n = Xtr.shape[0]
    train_losses: list = []
    val_losses:   list = []
    best_val   = float('inf')
    best_state = None
    no_improve = 0
    start_ep   = 1

    # Long CPU runs (~1-2h) have repeatedly been cut off by environment/session restarts
    # with nothing to show for the lost wall-clock time. Resume from the last completed
    # epoch instead of starting over if a checkpoint from an interrupted run is present.
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['opt_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        train_losses = ckpt['train_losses']
        val_losses   = ckpt['val_losses']
        best_val     = ckpt['best_val']
        best_state   = ckpt['best_state']
        no_improve   = ckpt['no_improve']
        start_ep     = ckpt['epoch'] + 1
        print(f"    Resumed from checkpoint at epoch {start_ep - 1} "
              f"(best_val={best_val:.5f}, no_improve={no_improve})", flush=True)

    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        # Class-balanced minibatch sampling (with replacement) instead of loss-level
        # class weighting — rebalances by row composition the way Random Forest's
        # balanced_subsample does, rather than distorting gradient magnitude per class.
        perm = torch.multinomial(sw, n, replacement=True) if sw is not None \
            else torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        train_losses.append(total / n)

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"    Epoch {ep:>3}/{EPOCHS}  "
              f"train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}  "
              f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)

        if checkpoint_path:
            torch.save({
                'epoch': ep, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(), 'train_losses': train_losses,
                'val_losses': val_losses, 'best_val': best_val, 'best_state': best_state,
                'no_improve': no_improve,
            }, checkpoint_path)

        if no_improve >= PATIENCE:
            print(f"\n    Early stop at epoch {ep} (patience={PATIENCE})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print()
    return train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
             class_names: list, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(X_test).to(device)).argmax(1).cpu().numpy()

    labels_present = sorted(np.unique(np.concatenate([y_test, preds])))
    target_names = [class_names[i] for i in labels_present]

    report_dict = classification_report(
        y_test, preds, labels=labels_present, target_names=target_names,
        zero_division=0, output_dict=True)
    report_txt = classification_report(
        y_test, preds, labels=labels_present, target_names=target_names,
        zero_division=0)

    return {
        'accuracy':           accuracy_score(y_test, preds),
        'balanced_accuracy':  balanced_accuracy_score(y_test, preds),
        'precision_weighted': precision_score(y_test, preds, average='weighted', zero_division=0),
        'recall_weighted':    recall_score(y_test, preds, average='weighted', zero_division=0),
        'f1_weighted':        f1_score(y_test, preds, average='weighted', zero_division=0),
        'precision_macro':    precision_score(y_test, preds, average='macro', zero_division=0),
        'recall_macro':       recall_score(y_test, preds, average='macro', zero_division=0),
        'f1_macro':           f1_score(y_test, preds, average='macro', zero_division=0),
        'confusion_matrix':   confusion_matrix(y_test, preds, labels=labels_present),
        'cm_labels':          target_names,
        'report_dict':        report_dict,
        'report_txt':         report_txt,
        'preds':              preds,
    }


def compute_convergence(val_losses: list,
                         E: int = STABILITY_E,
                         threshold: float = EP_THRESHOLD) -> dict:
    v = np.array(val_losses, dtype=np.float64)
    ep = int(len(v))
    for e in range(1, len(v)):
        if abs(v[e] - v[e - 1]) < v[e] * threshold:
            ep = e + 1
            break
    return {
        'Ep':        ep,
        'Stability': float(np.std(v[-E:])),
        'AULC':      float(np.mean(v)),
        'Avg_Delta': float(np.mean(np.abs(np.diff(v)))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING & REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(train_losses: list, val_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color='blue', linewidth=1.8, label='Training Loss')
    ax.plot(epochs, val_losses, color='orange', linewidth=1.8, linestyle='--', label='Validation Loss')
    ax.set_title('Transformer-KAN Learning Curve — UNSW-NB15', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'loss_curves_{TASK}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    sz = max(6, len(class_names))
    fig, ax = plt.subplots(figsize=(sz + 2, sz))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix — Transformer-KAN — UNSW-NB15', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'confusion_matrix_{TASK}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {path}")


def print_and_save_results(clf: dict, conv: dict) -> None:
    W = 64
    lines = []
    lines.append('=' * W)
    lines.append('  RESULTS - Transformer-KAN - UNSW-NB15 (train -> test)')
    lines.append('=' * W)
    lines.append(f"  Accuracy            : {clf['accuracy']:.4f}")
    lines.append(f"  Balanced Accuracy   : {clf['balanced_accuracy']:.4f}")
    lines.append(f"  Precision (weighted): {clf['precision_weighted']:.4f}")
    lines.append(f"  Recall    (weighted): {clf['recall_weighted']:.4f}")
    lines.append(f"  F1-Score  (weighted): {clf['f1_weighted']:.4f}")
    lines.append(f"  Precision (macro)   : {clf['precision_macro']:.4f}")
    lines.append(f"  Recall    (macro)   : {clf['recall_macro']:.4f}")
    lines.append(f"  F1-Score  (macro)   : {clf['f1_macro']:.4f}")
    lines.append('-' * W)
    lines.append(f"  Ep  (plateau epoch)    : {conv['Ep']}")
    lines.append(f"  Stability (STD last 5) : {conv['Stability']:.6f}")
    lines.append(f"  AULC (mean val loss)   : {conv['AULC']:.6f}")
    lines.append(f"  Avg Delta              : {conv['Avg_Delta']:.6f}")
    lines.append('=' * W)
    lines.append('  Per-class report:')
    lines.append(clf['report_txt'])
    lines.append('  Confusion matrix (rows=true, cols=pred), labels=' + str(clf['cm_labels']))
    lines.append(np.array2string(clf['confusion_matrix']))

    text = '\n'.join(lines)
    print('\n' + text)

    txt_path = os.path.join(RESULTS_DIR, f'results_{TASK}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"\n  Saved -> {txt_path}")

    report_df = pd.DataFrame(clf['report_dict']).transpose()
    csv_path = os.path.join(RESULTS_DIR, f'classification_report_{TASK}.csv')
    report_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"  Saved -> {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'#' * 72}")
    print(f"  PIPELINE : UNSW-NB15 (Transformer-KAN, train -> test)")
    print(f"{'#' * 72}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device   : {device}")

    print("\n[Phase 1] Loading & Preprocessing")
    (X_tr_full, X_te, y_tr_full, y_te, feat_names, class_names,
     cont_idx, cat_idx, cat_cardinalities) = load_and_preprocess()
    n_classes = int(np.unique(y_tr_full).size)
    print(f"  X_train  : {X_tr_full.shape}  |  X_test : {X_te.shape}")
    print(f"  Classes  : {n_classes} -> {class_names}")

    # Held-out validation split carved out of the TRAINING set only.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_tr_full)
    print(f"  X_train  : {X_tr.shape}  |  X_val : {X_val.shape}  (val carved out of train)")

    # Raw 'balanced' weights hit a ~430:1 max/min ratio on this dataset (Normal: 56,000
    # vs Worms: 130 in train). Used here to build PER-ROW sampling weights (minibatches
    # drawn from a class-balanced distribution, see train_model), not a loss penalty —
    # loss-level weighting stacked with this same ratio previously over-corrected badly
    # (rare-class recall>0.8 at precision<0.23; ~28% of Normal pulled into Fuzzers).
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = np.sqrt(cw)
    cw = np.clip(cw, cw.min(), cw.min() * WEIGHT_CAP_RATIO)

    # Analysis/Backdoor/DoS are pinned to NEUTRAL (natural, count-proportional) sampling:
    # an exact-duplicate-feature-vector audit of the training set found 72-80% of THEIR
    # rows share an identical 42-feature vector with a DIFFERENT official attack_cat label
    # — i.e. the ground truth itself is contradictory for most of these rows. Oversampling
    # them doesn't teach the model anything; it just replays the same contradictions more
    # often. Normal/Fuzzers/Shellcode/Worms have low conflict rates (<12%) and demonstrably
    # benefited from the oversampling boost, so they keep it.
    #
    # The neutral/"no boost" reference weight in compute_class_weight('balanced')'s own
    # convention is 1.0, NOT cw.min(). cw.min() is whatever weight the single MOST FREQUENT
    # class (Normal) landed on — a value strictly below 1.0, because Normal is already
    # over-represented and needs down-weighting. An earlier version of this pin used
    # cw.min() by mistake, which suppressed DoS BELOW its natural training frequency
    # instead of leaving it alone — DoS recall collapsed to 0.06 (worse than doing
    # nothing). Pinning to 1.0 is the correct "leave this class's own count-proportional
    # representation alone" reference point.
    CONFLICT_HEAVY_CLASSES = ['Analysis', 'Backdoor', 'DoS']
    floor = 1.0
    for cls in CONFLICT_HEAVY_CLASSES:
        if cls in class_names:
            cw[class_names.index(cls)] = floor

    # If pso_optimize_weights.py has already found a per-class weight vector, use that
    # instead of the hand-tuned heuristic above (matched by class name, not position, in
    # case class ordering ever differs between runs). Falls back to the heuristic if the
    # file is absent so this script stays runnable standalone.
    pso_weights_path = os.path.join(RESULTS_DIR, 'pso_class_weights.json')
    if os.path.exists(pso_weights_path):
        with open(pso_weights_path, 'r', encoding='utf-8') as f:
            pso_data = json.load(f)
        pso_names = pso_data['class_names']
        pso_w = pso_data['weights']
        for cls, w in zip(pso_names, pso_w):
            if cls in class_names:
                cw[class_names.index(cls)] = w
        print(f"  Using PSO-optimized class weights from {pso_weights_path} "
              f"(proxy val F1_macro={pso_data.get('proxy_val_f1_macro', float('nan')):.4f})")
    else:
        print("  No PSO weights found — using hand-tuned heuristic class weights.")

    class_weights = torch.tensor(cw, dtype=torch.float32)
    sample_weights = class_weights[torch.as_tensor(y_tr, dtype=torch.long)]

    print("\n[Training] FT-Tokenizer -> Transformer -> KAN")
    torch.manual_seed(SEED)
    model = HybridTransformerKAN(cont_idx, cat_idx, cat_cardinalities, n_classes)

    t_l, v_l = train_model(model, X_tr, y_tr, X_val, y_val, device, sample_weights)
    clf_res  = evaluate(model, X_te, y_te, class_names, device)
    conv_res = compute_convergence(v_l)

    print(f"  Acc={clf_res['accuracy']:.4f}  "
          f"P(w)={clf_res['precision_weighted']:.4f}  "
          f"R(w)={clf_res['recall_weighted']:.4f}  "
          f"F1(w)={clf_res['f1_weighted']:.4f}  "
          f"Ep={conv_res['Ep']}")

    print_and_save_results(clf_res, conv_res)
    print(f"\n[Saving plots]")
    plot_loss_curves(t_l, v_l)
    plot_confusion_matrix(clf_res['confusion_matrix'], clf_res['cm_labels'])


if __name__ == '__main__':
    run_pipeline()
    print(f"\nAll done. Outputs saved to:\n  {RESULTS_DIR}")
