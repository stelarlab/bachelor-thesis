"""Train XGBoost position regressor on strip cluster features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xgboost as xgb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_events, learn_frame_transform, load_detector_shift, filter_road_empty
from features import FEATURES, build_features

MAX_NHITS_TRAIN = 50
ROOT = Path(__file__).resolve().parents[1]   # repo root (pipeline/ is one level down)
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED  = 42
SPLIT = (0.70, 0.15, 0.15)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .root file")
    args = p.parse_args()
    DATA = Path(args.data)
    print("[XGB] loading data ...", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    print(f"[XGB] frame: track_icept = {a:.6f} * out_xpos + {b:+.4f}", flush=True)

    detector_shift = load_detector_shift(OUT)
    if detector_shift != 0.0:
        print(f"[XGB] detector shift: {detector_shift*1000:+.1f} um", flush=True)
    else:
        print("[XGB] no detector shift set", flush=True)

    valid_idx = filter_road_empty(ev, a, b, detector_shift_mm=detector_shift)
    print(f"[XGB] road-empty filter: {ev.n_events} -> {len(valid_idx)} events "
          f"({100*(1-len(valid_idx)/ev.n_events):.1f}% removed)", flush=True)

    X, y, mask, cm = build_features(ev, a, b, detector_shift_mm=detector_shift)
    print(f"[XGB] {len(FEATURES)} features", flush=True)

    road_mask = np.zeros(ev.n_events, dtype=bool)
    road_mask[valid_idx] = True
    full_mask = mask & road_mask

    n_hits_full = ev.n_hits[full_mask]
    Xv, yv, cmv = X[full_mask], y[full_mask], cm[full_mask]
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

    # no x_min -> no absolute position, so predict local offset and reconstruct
    label_tr = (yv[tr] - cmv[tr]) / 5.0
    label_va = (yv[va] - cmv[va]) / 5.0
    label_te = (yv[te] - cmv[te]) / 5.0

    dtr = xgb.DMatrix(Xv[tr], label=label_tr, feature_names=FEATURES)
    dva = xgb.DMatrix(Xv[va], label=label_va, feature_names=FEATURES)
    dte = xgb.DMatrix(Xv[te], label=label_te, feature_names=FEATURES)

    # v2: huber_slope 1.0->0.5, max_depth 8->7, eta 0.05->0.03, min_child_w 5->8, reg_lambda 1->2
    # v8: num_boost_round 3000->12000, early_stopping 80->300, huber_slope 0.5->0.3, depth 7->9
    # v9: features 24->22 (x_min removed, abs. time anchored, muTPC full fit, tc_correction added)
    #     label changed to local offset (y - cm) / 5.0 — cm anchor replaces x_min
    # v10: track_slope removed → sigma68 2022 um (vs 577 um) — essential, kept back in
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

    y_pred = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) * 5.0 + cmv[te]
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
