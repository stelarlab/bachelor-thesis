# Strip position reconstruction for the ATLAS NSW Micromegas detector

Bachelor thesis  |  **branch: `main` (GNN)** — LMU Munich, 2026  
Supervisors: Prof. Dr. Otmar Biebel, Dr. Fabian Vogel

This repository contains the full analysis code for comparing position reconstruction methods on testbeam data from a Micromegas SM1 prototype (530 V, 29°, Ar:CO2:iC4H10 93:5:2).

## Results

![Model comparison](results/figures/efficiency_bar.png)

| Method | sigma_core | σ₆₈ | efficiency |
|---|---|---|---|
| Charge-mean (Vogel h_residual_6) | 332 μm | — | — |
| XGBoost v8 | 121 μm | 527 μm | 97.1% |
| GNN road TC ⭐ | 113 μm | 305 μm | 93.8% |
| GNN TC v2 | 115 μm | 296 μm | 93.7% |
| GNN multidomain v1 | 110 μm | 411 μm | 94.5% |
| GNN zero-shot (100→200 ns) | 185 μm | 337 μm | 86.7% |

sigma_core = Gaussian core width. σ₆₈ = robust 68% half-width (all events).

![Residuals](results/figures/residuals_all.png)

![Core vs tail robustness](results/figures/sigma_comparison.png)

![XGBoost feature importance](results/figures/feature_importance_gain.png)

## Development chronology

The analysis proceeded in four phases:

**Phase 0 — Calibration** (`scripts/01_diagnose.py`)  
Detector shift and frame transform calibrated on the data. The charge-mean residual has a systematic offset corrected here. This step must run before training.

**Phase 1 — XGBoost baseline** (`src/train_xgboost.py`)  
24 aggregated cluster features (positions, charges, times). Strip-by-strip feature variants (v5–v7) did not improve performance — decision trees cannot model inter-strip dependencies. v8 is the structural limit of this approach.

**Phase 2 — GNN** (`src/train_gnn.py`)  
Strip self-attention model (Transformer encoder). Each strip is a node; the model learns interactions between strips directly. Multiple variants explored:
- `baseline_100ns` — first working GNN, single dataset
- `gnn_road` — road-based strip selection (vs. cluster-based)
- `gnn_tc` / `gnn_tc_v2` — time-corrected centroid as anchor
- `gnn_road_tc` — road selection + TC anchor, best single-domain model
- `run_multidomain_v1–v4` — trained on 100 ns + 200 ns simultaneously

**Phase 3 — Generalization** (`scripts/03_evaluate_zeroshot.py`)  
Zero-shot transfer: model trained on 100 ns evaluated on 200 ns without retraining. TC-anchor variants generalize significantly better than charge-mean anchor variants.

## Structure

```
src/
  data_loader.py      ROOT I/O, strip selection, frame transform
  dataset.py          PyTorch dataset, normalization, TC-centroid anchor
  model.py            strip self-attention GNN (Transformer encoder)
  fits.py             double-Gaussian fit + robust sigma_68
  train_gnn.py        GNN training loop
  evaluate.py         comparison table over all models
  plots.py            figure generation

scripts/
  01_diagnose.py          Phase 0: detector shift, frame calibration
  02_analyse_features.py  feature correlation + XGBoost importance
  03_evaluate_zeroshot.py zero-shot transfer to unseen shaping times

results/figures/        figures
```

> XGBoost models and feature engineering are on the [`xgboost`](../../tree/xgboost) branch.

## Data

Input: `Data_for_Roman_29deg_530V_100ns_x9.root` (not included, available on request).

## Requirements

```
python >= 3.11
torch, torch-geometric, xgboost, uproot, awkward, numpy, matplotlib, scipy, tqdm
```
