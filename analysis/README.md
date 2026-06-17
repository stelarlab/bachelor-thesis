# Analysis — understanding the data and the models

These scripts are not part of the reconstruction pipeline. They exist to understand
*why* the models behave the way they do, and to document the properties of the input
data before any tuning. Each writes a plain-text report and figures to `outputs/`.

All scripts take `--data <file.root>` and import the library from `../src`.

## Scripts

| script | question it answers |
|---|---|
| `diagnose_inputs.py` | what do the raw inputs and the 24 XGBoost features look like? |
| `analyse_gaps_vs_tails.py` | do fragmented clusters (gaps) cause the XGBoost tails? |
| `analyse_time_outliers.py` | do broken strip times cause the XGBoost tails? |
| `analyse_xgb_tails.py` | which features separate tail from core events; what do the trees split on? |

`analyse_*` scripts need `outputs/xgb_predictions.npz` and (for the tree dump)
`outputs/xgb_model.json` from `pipeline/03_train_xgboost.py`.

## Findings (530 V, 100 ns)

**Input shape.** Road clusters have a median of 6 strips. 32.8% of road clusters
contain at least one gap (a missing strip inside the cluster). 9.3% of events have
hits but no strip within the +-5 mm road, and are dropped by the XGBoost feature
builder — this is the origin of the XGBoost / GNN test-set size difference.

**Time outliers.** Strip times sit physically in ~[-60, +160] ns (p0.1..p99.9), but
0.07% of strips carry |t| > 300 ns, down to -5111 ns. These are broken single strips.

**The XGBoost tails are not a data defect.** Two hypotheses were tested against the
residuals and both fail:

- *gaps*: gap events have sigma68 = 565 um vs. 549 um for clean events (1.0x); 34% of
  all tail events are gap clusters, equal to the 33% base rate. No enrichment.
- *time outliers*: only 92 events (0.41%); removing them leaves sigma68 unchanged
  (553 -> 553 um). Too few to matter.

The tail fraction is ~36% in every subset, however the events are split. The wide
XGBoost distribution is therefore intrinsic to the model, not driven by a broken
subset of the data — consistent with a piecewise-constant tree regressor producing
discretisation error on a smooth, continuous target.
