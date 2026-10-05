"""
GPU-accelerated algorithm comparison for UNSW-NB15 -- binary (label) AND multiclass
(attack_cat) -- across XGBoost, LightGBM, and CatBoost.

Rationale: exhaustive testing already showed gradient-boosted trees beat every deep
learning variant tried on this tabular dataset (a well-documented general finding, not
specific to this project -- see Grinsztajn et al. 2022). This script extends that finding:
  - Binary task (label: 0=Normal, 1=Attack) -- attack_cat is dropped to avoid leaking
    the multiclass answer into the binary target (attack_cat trivially implies label).
  - Multiclass task (attack_cat, 10 classes) -- label is dropped (existing leakage
    rationale from evaluate_unsw_nb15_hybrid.py: label is a deterministic function of
    attack_cat).
  - GPU acceleration where the library supports it on this machine (RTX 4060, verified
    working for XGBoost's device='cuda'; LightGBM/CatBoost fall back to CPU automatically
    if their GPU build isn't available here -- this script handles that per-library).

Same official train/test split discipline as the rest of this project: validation is
carved out of TRAINING only (85/15 stratified), test set touched once for final metrics.
"""
import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, classification_report)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TRAIN_PATH  = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH   = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
CAT_COLS    = ['proto', 'service', 'state']
SEED        = 42


def _safe_encode(series, le):
    known = set(le.classes_)
    fallback = le.classes_[0]
    return le.transform(series.apply(lambda x: x if x in known else fallback))


def load_and_preprocess(task: str):
    """task: 'binary' (target='label') or 'multiclass' (target='attack_cat')."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    for df in (train_df, test_df):
        df.drop(columns=['id'], inplace=True, errors='ignore')
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace('', 'Normal')

    if task == 'binary':
        train_df.drop(columns=['attack_cat'], inplace=True, errors='ignore')
        test_df.drop(columns=['attack_cat'], inplace=True, errors='ignore')
        target_col = 'label'
        y_train = train_df.pop(target_col).astype(np.int64).values
        y_test  = test_df.pop(target_col).astype(np.int64).values
        class_names = ['Normal', 'Attack']
    else:
        train_df.drop(columns=['label'], inplace=True, errors='ignore')
        test_df.drop(columns=['label'], inplace=True, errors='ignore')
        target_col = 'attack_cat'
        le_t = LabelEncoder()
        y_train = le_t.fit_transform(train_df.pop(target_col)).astype(np.int64)
        y_test  = _safe_encode(test_df.pop(target_col), le_t).astype(np.int64)
        class_names = list(le_t.classes_)

    for col in CAT_COLS:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            test_df[col]  = _safe_encode(test_df[col].astype(str), le)

    feature_names = list(train_df.columns)
    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    return X_train, X_test, y_train, y_test, feature_names, class_names


def evaluate(y_true, preds, class_names, multiclass: bool):
    avg = 'macro' if multiclass else 'binary'
    return {
        'accuracy':          accuracy_score(y_true, preds),
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'f1_macro':          f1_score(y_true, preds, average='macro', zero_division=0),
        'f1_weighted':       f1_score(y_true, preds, average='weighted', zero_division=0),
        'precision_macro':   precision_score(y_true, preds, average='macro', zero_division=0),
        'recall_macro':      recall_score(y_true, preds, average='macro', zero_division=0),
        'report':            classification_report(y_true, preds, target_names=class_names,
                                                     zero_division=0, output_dict=True),
        'report_txt':        classification_report(y_true, preds, target_names=class_names, zero_division=0),
    }


def run_xgb(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, use_gpu=True):
    multiclass = task == 'multiclass'
    params = dict(
        n_estimators=900, max_depth=9, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
        tree_method='hist', eval_metric='mlogloss' if multiclass else 'logloss',
        early_stopping_rounds=30, random_state=SEED,
    )
    if multiclass:
        params['objective'] = 'multi:softprob'
        params['num_class'] = len(class_names)
    else:
        params['objective'] = 'binary:logistic'
    if use_gpu:
        params['device'] = 'cuda'
    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    dt = time.time() - t0
    preds = clf.predict(X_te)
    return evaluate(y_te, preds, class_names, multiclass), dt, clf.best_iteration


def run_lgb(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, use_gpu=True):
    multiclass = task == 'multiclass'
    params = dict(
        n_estimators=1500, max_depth=-1, num_leaves=255, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, verbosity=-1,
    )
    if multiclass:
        params['objective'] = 'multiclass'
        params['num_class'] = len(class_names)
    else:
        params['objective'] = 'binary'
    if use_gpu:
        params['device'] = 'gpu'
    clf = lgb.LGBMClassifier(**params)
    t0 = time.time()
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)])
    except Exception as e:
        if use_gpu:
            print(f'    [LightGBM GPU failed ({e}), falling back to CPU]')
            params['device'] = 'cpu'
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            raise
    dt = time.time() - t0
    preds = clf.predict(X_te)
    return evaluate(y_te, preds, class_names, multiclass), dt, clf.best_iteration_


def run_cat(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, cat_idx, use_gpu=True):
    multiclass = task == 'multiclass'
    params = dict(
        iterations=1500, depth=8, learning_rate=0.08, l2_leaf_reg=3.0,
        loss_function='MultiClass' if multiclass else 'Logloss',
        random_seed=SEED, early_stopping_rounds=30, verbose=False,
    )
    if use_gpu:
        params['task_type'] = 'GPU'

    # CatBoost's native categorical handling (its main advantage) requires the
    # cat_features columns to be int/str dtype -- a plain float32 ndarray with
    # cat_features set raises CatBoostError (numpy arrays are single-dtype, so the
    # categorical columns silently inherit float32 like everything else). Build a
    # per-column-typed DataFrame instead: int32 for categorical columns, float32
    # for the rest.
    def make_pool(X, y=None):
        df = pd.DataFrame(X)
        for i in cat_idx:
            df[i] = df[i].astype(np.int32)
        return cb.Pool(df, label=y, cat_features=cat_idx)

    train_pool = make_pool(X_tr, y_tr)
    val_pool   = make_pool(X_val, y_val)
    test_pool  = make_pool(X_te)

    clf = cb.CatBoostClassifier(**params)
    t0 = time.time()
    try:
        clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
    except Exception as e:
        if use_gpu:
            print(f'    [CatBoost GPU failed ({e}), falling back to CPU]')
            params['task_type'] = 'CPU'
            clf = cb.CatBoostClassifier(**params)
            clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
        else:
            raise
    dt = time.time() - t0
    preds = clf.predict(test_pool).astype(np.int64).ravel()
    return evaluate(y_te, preds, class_names, multiclass), dt, clf.best_iteration_


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for task in ['multiclass', 'binary']:
        print(f"\n{'#'*72}\n  TASK: {task}\n{'#'*72}")
        X_tr_full, X_te, y_tr_full, y_te, feat_names, class_names = load_and_preprocess(task)
        cat_idx = [feat_names.index(c) for c in CAT_COLS if c in feat_names]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, random_state=SEED, stratify=y_tr_full)
        print(f"  X_train={X_tr.shape}  X_val={X_val.shape}  X_test={X_te.shape}  classes={class_names}")

        runners = {
            'xgboost_gpu': lambda: run_xgb(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, use_gpu=True),
            'lightgbm_gpu': lambda: run_lgb(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, use_gpu=True),
            'catboost_gpu': lambda: run_cat(task, X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, cat_idx, use_gpu=True),
        }

        task_results = {}
        for name, fn in runners.items():
            print(f"\n  [{name}]")
            try:
                res, dt, best_iter = fn()
                task_results[name] = {**res, 'train_time_s': dt, 'best_iteration': int(best_iter)}
                print(f"    acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  "
                      f"f1_macro={res['f1_macro']:.4f}  f1_weighted={res['f1_weighted']:.4f}  "
                      f"time={dt:.1f}s  best_iter={best_iter}")
            except Exception as e:
                print(f"    FAILED: {repr(e)}")
                task_results[name] = {'error': repr(e)}

        all_results[task] = task_results

    out_path = os.path.join(RESULTS_DIR, 'gpu_boosting_comparison.json')
    # strip the sklearn report_dict (large) before saving a compact json summary
    summary = {
        task: {
            name: {k: v for k, v in res.items() if k not in ('report',)}
            for name, res in task_results.items()
        }
        for task, task_results in all_results.items()
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for task, task_results in all_results.items():
        print(f"\n  {task.upper()}")
        for name, res in task_results.items():
            if 'error' in res:
                print(f"    {name:<15} FAILED")
            else:
                print(f"    {name:<15} acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  "
                      f"f1_macro={res['f1_macro']:.4f}  f1_weighted={res['f1_weighted']:.4f}")


if __name__ == '__main__':
    main()
