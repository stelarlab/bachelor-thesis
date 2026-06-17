"""Feature correlation analysis and XGBoost importance plots.

Outputs:
    outputs/plots/feature_importance_gain.png
    outputs/plots/feature_importance_weight.png
    outputs/plots/feature_importance_cover.png
    outputs/plots/feature_correlation.png
    outputs/feature_analysis.txt
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_events, learn_frame_transform, load_detector_shift
from features import FEATURES, build_features

ROOT  = Path(__file__).resolve().parents[1]   # repo root (pipeline/ is one level down)
OUT   = ROOT / "outputs"
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
CORR_THRESHOLD = 0.95

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .root file")
    args = p.parse_args()
    DATA = Path(args.data)
    print("[ANALYSE] loading data ...", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    detector_shift = load_detector_shift(OUT)
    X, y, mask, _ = build_features(ev, a, b, detector_shift_mm=detector_shift)
    X_valid = X[mask]
    print(f"[ANALYSE] {mask.sum()} valid events, {len(FEATURES)} features", flush=True)

    df = pd.DataFrame(X_valid, columns=FEATURES)
    corr = df.corr()

    redundant = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            c = abs(corr.iloc[i, j])
            if c > CORR_THRESHOLD:
                redundant.append((FEATURES[i], FEATURES[j], corr.iloc[i, j]))
    redundant.sort(key=lambda x: -abs(x[2]))
    to_remove = list(dict.fromkeys([f2 for _, f2, _ in redundant]))

    txt_path = OUT / "feature_analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Feature correlation analysis (threshold={CORR_THRESHOLD})\n")
        f.write(f"Events: {mask.sum()}, features: {len(FEATURES)}\n\n")
        f.write(f"Highly correlated pairs (|r| > {CORR_THRESHOLD}):\n")
        for f1, f2, r in redundant:
            f.write(f"  r={r:+.3f}  {f1}  <->  {f2}\n")
        f.write(f"\nRecommended to remove:\n")
        for feat in to_remove:
            f.write(f"  - {feat}\n")
    print(f"[ANALYSE] {len(redundant)} redundant pairs -> {txt_path}", flush=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURES, fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("feature correlation matrix (XGBoost, 100ns)")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_correlation.png", dpi=150)
    plt.close()
    print(f"[ANALYSE] Plot -> {PLOTS}/feature_correlation.png", flush=True)

    model_path = OUT / "xgb_model.json"
    if model_path.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        for imp_type in ["gain", "weight", "cover"]:
            scores = model.get_booster().get_score(importance_type=imp_type)
            vals = [scores.get(f"f{i}", 0.0) for i in range(len(FEATURES))]
            order = np.argsort(vals)[::-1]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh([FEATURES[o] for o in order], [vals[o] for o in order])
            ax.set_xlabel(f"Importance ({imp_type})")
            ax.set_title(f"XGBoost Feature Importance ({imp_type})")
            plt.tight_layout()
            plt.savefig(PLOTS / f"feature_importance_{imp_type}.png", dpi=150)
            plt.close()
        print(f"[ANALYSE] importance plots -> {PLOTS}/", flush=True)
    else:
        print("[ANALYSE] no xgb_model.json, skipping importance.", flush=True)

    print("\n[ANALYSE] redundant pairs:")
    for f1, f2, r in redundant:
        print(f"  r={r:+.3f}  {f1}  <->  {f2}")
    print(f"\n[ANALYSE] recommended to remove ({len(to_remove)}): {to_remove}")

if __name__ == "__main__":
    main()
