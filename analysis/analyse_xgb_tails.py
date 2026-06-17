"""XGBoost tree & tail analysis: why is the XGBoost distribution less sharp?

XGBoost has a sharper core but wider tails than the GNN. This script looks at WHICH
events end up in the tails and WHY.

  1. tree dump       -- first trees as text + (if graphviz) as image; shows which
                        features XGBoost splits on first
  2. tail vs. core   -- feature distributions for tail events (|res|>500um) against
                        core events (|res|<100um) -- which features separate them?
  3. residual vs. feature -- 2D hexbin: residual against the most separating features;
                        shows whether the tails appear at specific feature values

Requires: outputs/xgb_model.json and outputs/xgb_predictions.npz (from 03_train_xgboost.py)

Outputs:
    outputs/plots/xgb_tails/tree_<k>.png        (if graphviz available)
    outputs/plots/xgb_tails/tail_vs_core.png
    outputs/plots/xgb_tails/residual_vs_feature.png
    outputs/xgb_tree_dump.txt
    outputs/xgb_tail_report.txt
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_events, learn_frame_transform, load_detector_shift
from features import FEATURES, build_features

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs"
PLOTDIR = OUT / "plots" / "xgb_tails"
PLOTDIR.mkdir(parents=True, exist_ok=True)

TAIL_MM = 0.5    # |res| > 500 um = tail event
CORE_MM = 0.1    # |res| < 100 um = core event
N_TREES_DUMP = 3


def load_model_and_test(DATA):
    model_path = OUT / "xgb_model.json"
    pred_path  = OUT / "xgb_predictions.npz"
    if not model_path.exists() or not pred_path.exists():
        raise FileNotFoundError("xgb_model.json or xgb_predictions.npz missing -- run 03_train_xgboost.py first.")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    d = np.load(pred_path)
    y_pred, y_true = d["y_pred"], d["y_true"]
    test_idx = d["test_idx"]

    # rebuild features -- test_idx indexes into Xv = X[mask] (see 03_train_xgboost.py)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    shift = load_detector_shift(OUT)
    X, _, mask, _ = build_features(ev, a, b, detector_shift_mm=shift)
    Xv = X[mask]
    X_test = Xv[test_idx]
    return booster, X_test, y_pred, y_true


def dump_trees(booster, txt):
    """Text dump of the first trees + image export if graphviz is available."""
    dump = booster.get_dump(with_stats=True)
    txt.append(f"total trees: {len(dump)}")
    txt.append(f"first {N_TREES_DUMP} trees (feature ids = f0..f{len(FEATURES)-1}):")
    fmap = {f"f{i}": name for i, name in enumerate(FEATURES)}
    for k in range(min(N_TREES_DUMP, len(dump))):
        txt.append(f"\n--- tree {k} ---")
        tree_str = dump[k]
        for fid, name in fmap.items():   # replace f-ids with feature names for readability
            tree_str = tree_str.replace(f"[{fid}<", f"[{name}<")
        txt.append(tree_str.rstrip())

    try:
        for k in range(min(N_TREES_DUMP, len(dump))):
            ax = xgb.plot_tree(booster, num_trees=k)
            fig = ax.figure
            fig.set_size_inches(22, 12)
            fig.savefig(PLOTDIR / f"tree_{k}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
        print(f"  -> {PLOTDIR}/tree_0..{min(N_TREES_DUMP, len(dump))-1}.png", flush=True)
    except Exception as exc:
        print(f"  -> graphviz not available ({exc}); text dump only.", flush=True)


def tail_vs_core(X_test, y_pred, y_true, txt):
    res = y_pred - y_true
    tail = np.abs(res) > TAIL_MM
    core = np.abs(res) < CORE_MM
    txt.append("")
    txt.append(f"tail events (|res|>{TAIL_MM*1000:.0f}um): {tail.sum()}  ({tail.mean()*100:.1f}%)")
    txt.append(f"core events (|res|<{CORE_MM*1000:.0f}um): {core.sum()}  ({core.mean()*100:.1f}%)")
    txt.append("")
    txt.append(f"  {'feature':<28s} {'core-mean':>12s} {'tail-mean':>12s} {'delta (std)':>14s}")

    # which features separate tail from core most (standardised difference)?
    diffs = []
    for j, name in enumerate(FEATURES):
        c = X_test[core, j]; t = X_test[tail, j]
        c = c[np.isfinite(c)]; t = t[np.isfinite(t)]
        if c.size == 0 or t.size == 0:
            continue
        pooled_std = np.sqrt(0.5 * (c.std()**2 + t.std()**2)) + 1e-9
        d_std = (t.mean() - c.mean()) / pooled_std
        diffs.append((name, j, c.mean(), t.mean(), d_std))
    diffs.sort(key=lambda x: -abs(x[4]))
    for name, j, cm, tm, d in diffs:
        txt.append(f"  {name:<28s} {cm:>12.3f} {tm:>12.3f} {d:>+14.2f}")

    # plot: top-6 separating features as overlaid histograms
    top = diffs[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (name, j, cm, tm, d) in zip(axes, top):
        c = X_test[core, j]; t = X_test[tail, j]
        c = c[np.isfinite(c)]; t = t[np.isfinite(t)]
        lo, hi = np.percentile(np.concatenate([c, t]), [1, 99])
        bins = np.linspace(lo, hi, 50) if hi > lo else 50
        ax.hist(c, bins=bins, density=True, alpha=0.55, color="tab:green", label="core")
        ax.hist(t, bins=bins, density=True, alpha=0.55, color="tab:red",   label="tail")
        ax.set_title(f"{name}  (delta={d:+.2f}std)", fontsize=9)
        ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    for ax in axes[len(top):]:
        ax.set_visible(False)
    fig.suptitle("tail vs. core events -- top separating features (XGBoost)", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "tail_vs_core.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'tail_vs_core.png'}")
    return [d[1] for d in diffs[:4]]   # indices of the 4 most separating features


def residual_vs_feature(X_test, y_pred, y_true, top_feat_idx):
    res = (y_pred - y_true) * 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, j in zip(axes, top_feat_idx):
        col = X_test[:, j]
        m = np.isfinite(col) & (np.abs(res) < 2000)
        lo, hi = np.percentile(col[m], [1, 99])
        ax.hexbin(col[m], res[m], gridsize=60, cmap="viridis", mincnt=1,
                  extent=(lo, hi, -2000, 2000))
        ax.axhline(0, color="white", lw=0.6, ls=":")
        ax.axhline( 500, color="red", lw=0.8, ls="--")
        ax.axhline(-500, color="red", lw=0.8, ls="--")
        ax.set_xlabel(FEATURES[j]); ax.set_ylabel("residual [um]")
        ax.set_title(FEATURES[j], fontsize=9)
    fig.suptitle("residual vs. separating features (red = +-500 um tail boundary)", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "residual_vs_feature.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'residual_vs_feature.png'}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .root file (same as training)")
    args = p.parse_args()
    DATA = Path(args.data)

    print("[XGB-TAIL] loading model + test set ...", flush=True)
    booster, X_test, y_pred, y_true = load_model_and_test(DATA)
    print(f"[XGB-TAIL] {len(y_true)} test events", flush=True)

    tree_txt = ["XGBoost tree dump", ""]
    print("[XGB-TAIL] dumping trees ...", flush=True)
    dump_trees(booster, tree_txt)
    (OUT / "xgb_tree_dump.txt").write_text("\n".join(tree_txt), encoding="utf-8")
    print(f"[XGB-TAIL] tree dump -> {OUT / 'xgb_tree_dump.txt'}", flush=True)

    tail_txt = ["XGBoost tail analysis", ""]
    print("[XGB-TAIL] tail vs. core ...", flush=True)
    top_idx = tail_vs_core(X_test, y_pred, y_true, tail_txt)
    residual_vs_feature(X_test, y_pred, y_true, top_idx)
    (OUT / "xgb_tail_report.txt").write_text("\n".join(tail_txt), encoding="utf-8")
    print(f"[XGB-TAIL] tail report -> {OUT / 'xgb_tail_report.txt'}", flush=True)
    print("\n".join(tail_txt).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
