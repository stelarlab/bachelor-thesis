"""XGBoost v8b: 24 features + tuned hyperparameters (12000 rounds, huber=0.3, depth=9)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xgboost as xgb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_events, learn_frame_transform, load_detector_shift
from features import FEATURES, build_features

MAX_NHITS_TRAIN = 50
ROOT = Path(__file__).resolve().parents[1]   # repo root (pipeline/ is one level down)
DATA = ROOT / "Data_for_Roman_29deg_530V_100ns_x9.root"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED  = 42
SPLIT = (0.70, 0.15, 0.15)


def main():
    print("[XGB] loading data ...", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    print(f"[XGB] frame: track_icept = {a:.6f} * out_xpos + {b:+.4f}", flush=True)

    detector_shift = load_detector_shift(OUT)
    if detector_shift != 0.0:
        print(f"[XGB] detector shift: {detector_shift*1000:+.1f} um", flush=True)
    else:
        print("[XGB] no detector shift set", flush=True)

    X, y, mask, cm = build_features(ev, a, b, detector_shift_mm=detector_shift)
    print(f"[XGB] {len(FEATURES)} features", flush=True)

    n_hits_full = ev.n_hits[mask]
    Xv, yv, cmv = X[mask], y[mask], cm[mask]
    n = len(yv)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr = int(SPLIT[0] * n); n_va = int(SPLIT[1] * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    n_tr0, n_va0 = len(tr), len(va)
    tr = tr[n_hits_full[tr] <= MAX_NHITS_TRAIN]
    va = va[n_hits_full[va] <= MAX_NHITS_TRAIN]
    print(f"[XGB] outlier cut: train {n_tr0}->{len(tr)}  val {n_va0}->{len(va)}", flush=True)
    print(f"[XGB] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    y_off = float(yv[tr].mean())

    dtr = xgb.DMatrix(Xv[tr], label=yv[tr] - y_off, feature_names=FEATURES)
    dva = xgb.DMatrix(Xv[va], label=yv[va] - y_off, feature_names=FEATURES)
    dte = xgb.DMatrix(Xv[te], label=yv[te] - y_off, feature_names=FEATURES)

    # v2 hyperparameters vs. v1:
    #   huber_slope  1.0 -> 0.5  (more robust to outliers)
    #   max_depth    8   -> 7    (less overfitting)
    #   eta          0.05-> 0.03 (smaller steps)
    #   num_rounds   1500-> 3000 (early_stopping=80)
    #   min_child_w  5.0 -> 8.0  (better generalisation)
    #   reg_lambda   1.0 -> 2.0  (stronger L2)
    # v8 vs v4: num_boost_round 3000->8000, early_stopping 80->200
    #            huber_slope 0.5->0.3 (sharper core), max_depth 7->9
    params = dict(
        objective="reg:pseudohubererror", huber_slope=0.3,
        tree_method="hist", max_depth=9, eta=0.03,
        subsample=0.85, colsample_bytree=0.80,
        min_child_weight=8.0,
        reg_lambda=2.0,
        seed=SEED,
    )
    booster = xgb.train(
        params, dtr, num_boost_round=12000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=300, verbose_eval=400,
    )
    print(f"[XGB] best_iteration = {booster.best_iteration}", flush=True)

    y_pred = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) + y_off
    np.savez_compressed(
        OUT / "xgb_predictions.npz",
        y_pred=y_pred, y_true=yv[te], charge_mean_pred=cmv[te],
        train_idx=tr, val_idx=va, test_idx=te,
        slope_frame=a, offset_frame=b,
    )
    booster.save_model(str(OUT / "xgb_model.json"))
    print("[XGB] saved: outputs/xgb_predictions.npz", flush=True)


if __name__ == "__main__":
    main()
