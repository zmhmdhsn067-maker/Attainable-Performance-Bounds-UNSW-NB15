# Attainable-Performance Bounds from Label Conflict in Multiclass Network Intrusion Detection — code and results

Code, configurations, seeds, train/validation split indices, per-seed predictions and table/figure scripts for

> ⟦Authors⟧, "Attainable-Performance Bounds from Label Conflict in Multiclass Network Intrusion Detection: A Controlled Evaluation on UNSW-NB15", *International Journal of Intelligent Engineering and Systems*, ⟦Vol., No., pp., year⟧.

Every number in the paper can be regenerated from this repository. The stored per-seed predictions allow all results tables and figures to be rebuilt without retraining. Trained model weights are not distributed: every model is reproduced by the scripts and seeds below.

## Layout

```
evaluate_unsw_nb15_hybrid.py   preprocessing (representation R2) and the Transformer–KAN model and training loop
ablation_variants.py           MLP head of matched size, model with pluggable head, canonical class weights
gpu_boosting_comparison.py     preprocessing for the tree ensembles (representation R1)
pso_optimize_weights.py        particle-swarm search over class weights (separate experiment, Section 5.5)
data/                          place the dataset files here (see data/README.md)
revision/
  r1_bounds_all_representations.py   exact grouping; bounds A, D, per-class F1 ceilings C; memorisation baseline L;
                                     verification on representations R0, R1, R2
  r1_joint_bound_exact.py            exact joint macro-F1 bound J (mixed-integer program, symmetry-free) + brute-force check
  r1_joint_bound_tight.py            tightened per-group formulation, lower-bound search, brute-force verifier (used by the above)
  r1_hypothesis_class_bounds.py      bounds on XGBoost's own 256-bin quantile grid; checks every split threshold is a cut point
  r1_quantisation_bounds.py          bounds on uniform quantile grids (64/32/16/8 bins)
  r1_recall_ceiling_ties.py          per-class recall ceilings under both objectives, as exact ranges over tied groups
  r2_random_split_control.py         official partition vs five random pooled splits
  r2_bounds_cicids2017.py            the same bound procedure on CICIDS2017
  r1_welch_recompute.py              recomputation of the submitted Welch p-values (review audit)
  results/                           JSON output of the scripts above (+ pso_class_weights.json)
  figures_ijies/                     Fig. 1–6
  Rerun_Results_Summary.md           all model-result tables, as produced by rerun/analyse.py
  rerun/
    common.py      shared splits, single-process lock, metrics, result/prediction writer
    trees.py       Random Forest, XGBoost, LightGBM, CatBoost: hyperparameter selection and all runs
    tkan.py        Transformer–KAN: reference, MLP-head, weighted-CE and uniform-sampling arms
    analyse.py     every model-result table, Welch/Holm/Hedges statistics, error-overlap analysis
    figures.py     Fig. 1–6 (single column, ≥ 10 pt text)
    splits/        split_seed{42,1,7,123,2024}.npz — fitting/validation indices into the official training file
    out/           one JSON (metrics) and one NPZ (predictions) per run
```

Each `out/<run>.npz` holds `test_pred`, `test_proba` (float16), `val_pred` and `val_proba`. Each `out/<run>.json` holds validation and test metrics, per-class scores, the confusion matrix, hyperparameters, fit time and, for the Transformer–KAN, the full training curve. `grid_*.json` and `trees_selected_hyperparameters.json` record the validation-only hyperparameter selection.

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Table 3 (selected hyperparameters) | `revision/rerun/trees.py` | `rerun/out/trees_selected_hyperparameters.json` |
| Tables 4–8, 13 (best F1), 14; error overlap | `revision/rerun/analyse.py` | `Rerun_Results_Summary.md`, `rerun/out/analysis_summary.json` |
| Table 9 (random-split control) | `revision/r2_random_split_control.py` | `results/r2_random_split_control.json` |
| Table 11 rows R0/R1/R2 (A, D, groups) | `revision/r1_bounds_all_representations.py` | `results/r1_bounds_all_representations.json` |
| Joint macro-F1 J (Tables 10, 11, 13; Eqs. 5–8) | `revision/r1_joint_bound_exact.py` | `results/r1_joint_bound_exact.json` |
| Table 11, XGBoost-grid row | `revision/r1_hypothesis_class_bounds.py` | `results/r1_hypothesis_class_bounds.json` |
| Table 11, uniform-grid rows | `revision/r1_quantisation_bounds.py` | `results/r1_quantisation_bounds.json` |
| Table 12 (recall ceilings) | `revision/r1_recall_ceiling_ties.py` | `results/r1_recall_ceilings.json` |
| Table 13 (conflict rates, ceilings, F1 at the joint optimum) | `r1_bounds_all_representations.py`, `r1_joint_bound_exact.py` | as above |
| Table 15 (CICIDS2017) | `revision/r2_bounds_cicids2017.py` | `results/r2_bounds_cicids2017.json` |
| Fig. 1–6 | `revision/rerun/figures.py` | `figures_ijies/fig1.png` … `fig6.png` |
| PSO experiment (Section 5.5) | `pso_optimize_weights.py` | `results/pso_class_weights.json` |

## Requirements

Python 3.13; `pip install -r requirements.txt`. The reported runs used Windows 11 and an NVIDIA RTX 4060 Laptop GPU (8 GB):
- XGBoost and CatBoost ran on the GPU;
- LightGBM (pip build, no GPU support) and Random Forest ran on the CPU;
- the Transformer–KAN ran with the CUDA 13.0 build of PyTorch 2.12.0, and `tkan.py` refuses to run without CUDA.

## Reproducing

First place the data files as described in `data/README.md`.

```bash
# 1. bounds (CPU; minutes)
python revision/r1_bounds_all_representations.py
python revision/r1_joint_bound_exact.py           # ~2 min per representation, plus a 30-instance brute-force self-test
python revision/r1_hypothesis_class_bounds.py     # trains one XGBoost model (GPU)
python revision/r1_quantisation_bounds.py
python revision/r1_recall_ceiling_ties.py

# 2. protocol and second-corpus controls
python revision/r2_random_split_control.py        # 10 XGBoost fits (GPU)
python revision/r2_bounds_cicids2017.py           # needs the CICIDS2017 files

# 3. model runs (resumable; completed runs are skipped)
cd revision/rerun
python trees.py      # hyperparameter grid + 65 runs, about 1 h
python tkan.py       # 20 Transformer–KAN runs, about 5.5 h on the GPU above; starts after trees.py releases its lock

# 4. tables and figures from the stored predictions (no training)
python analyse.py
python figures.py
```

## Protocol in brief

- **Split.** For each seed s ∈ {42, 1, 7, 123, 2024}, the official training partition is split 85/15 into fitting and validation sets, stratified on the attack category with `random_state = s`. The same indices serve every model family and both tasks (`rerun/splits/`). The test partition is used once per trained model.
- **Selection.** Tree hyperparameters are selected by validation macro-F1 on the seed-42 split, over the grids in `trees.py`, and reused for all seeds, tasks and weighting arms. Boosting rounds stop on validation log-loss (patience 30). The Transformer–KAN restores its minimum-validation-loss checkpoint (patience 15, at most 150 epochs).
- **Bounds.** Groups are formed by exact row equality on the stated representation. All bounds are computed on the test partition and are attained by an oracle that knows the test labels; they bound, but do not estimate, what a model trained on the training partition can reach.

## Reproducibility notes

- `rerun/common.py` holds a single-process lock per queue, so a second copy of a run cannot start while one is alive.
- Transformer–KAN checkpoints store the full random-generator state. A run interrupted and resumed reproduced the uninterrupted run exactly on the same machine, using deterministic algorithms where PyTorch provides them.
- Results on other GPUs or driver versions may differ in the last digits. CatBoost on the GPU is not bit-for-bit deterministic.
- Fit times are wall-clock times measured with one job at a time on the machine above.

## Licence

⟦to be chosen by the authors⟧
