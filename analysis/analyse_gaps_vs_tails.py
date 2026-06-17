"""Link cluster gaps to the XGBoost residuals: do the tails come from gaps?

Hypothesis: XGBoost compresses the cluster into aggregated numbers and loses the
structure of fragmented clusters. If true, the tail events (|res|>500um) should be
over-represented among gap clusters.

Mapping: xgb_predictions.npz test_idx indexes into X[mask]; the original event is
np.where(mask)[0][test_idx]. This couples residual and cluster geometry per event.

Outputs:
    outputs/plots/gaps/sigma_by_gap.png       -- sigma68 / tail fraction: gap vs. no gap
    outputs/plots/gaps/res_vs_gapsize.png      -- |residual| vs. largest gap
    outputs/gap_tail_report.txt
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import (
    PITCH_MM, ROAD_MM,
    load_events, learn_frame_transform, load_detector_shift, select_strips_in_road,
)
from features import build_features

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs"
PLOTDIR = OUT / "plots" / "gaps"
PLOTDIR.mkdir(parents=True, exist_ok=True)

GAP_TOL = 1.5 * PITCH_MM
TAIL_MM = 0.5


def gap_features_for_events(ev, orig_idx, a, b, shift):
    """Per original event: number of gaps + largest gap (in missing strips)."""
    n_gaps   = np.zeros(len(orig_idx), dtype=np.int32)
    max_gap  = np.zeros(len(orig_idx), dtype=np.float64)
    n_road   = np.zeros(len(orig_idx), dtype=np.int32)
    for k, i in enumerate(orig_idx):
        xs = np.asarray(ev.hits_x[i], dtype=np.float64) - shift
        qs = np.asarray(ev.hits_q[i], dtype=np.float64)
        ts = np.asarray(ev.hits_t[i], dtype=np.float64)
        track_x = (ev.track_icept[i] - b) / a
        sx, _, _ = select_strips_in_road(xs, qs, ts, track_x, road_mm=ROAD_MM)
        n_road[k] = sx.size
        if sx.size >= 2:
            diffs = np.diff(np.sort(sx))
            gmask = diffs > GAP_TOL
            n_gaps[k] = int(gmask.sum())
            missing = np.round(diffs / PITCH_MM) - 1.0
            max_gap[k] = float(missing[gmask].max()) if gmask.any() else 0.0
    return n_gaps, max_gap, n_road


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    args = p.parse_args()
    DATA = Path(args.data)

    print("[GAPS] loading data + XGBoost predictions ...", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    shift = load_detector_shift(OUT)
    X, y, mask, _ = build_features(ev, a, b, detector_shift_mm=shift)

    d = np.load(OUT / "xgb_predictions.npz")
    test_idx = d["test_idx"]
    res = (d["y_pred"] - d["y_true"]) * 1000.0   # um
    orig_idx = np.where(mask)[0][test_idx]       # original event index per test event

    print(f"[GAPS] {len(orig_idx)} test events, computing gaps ...", flush=True)
    n_gaps, max_gap, n_road = gap_features_for_events(ev, orig_idx, a, b, shift)

    has_gap = n_gaps > 0
    txt = ["Gaps vs. tails -- XGBoost test set", ""]
    txt.append(f"total test events:   {len(res)}")
    txt.append(f"with gap:            {has_gap.sum()}  ({has_gap.mean()*100:.1f}%)")
    txt.append("")

    def stats(r):
        q16, q84 = np.quantile(r, [0.16, 0.84])
        s68 = 0.5 * (q84 - q16)
        tail = np.mean(np.abs(r) > TAIL_MM * 1000)
        return s68, tail

    s68_gap, tail_gap     = stats(res[has_gap])
    s68_clean, tail_clean = stats(res[~has_gap])
    txt.append(f"  {'group':<14s} {'N':>8s} {'sigma68 [um]':>14s} {'tail>500um':>12s}")
    txt.append(f"  {'no gap':<14s} {(~has_gap).sum():>8d} {s68_clean:>14.0f} {tail_clean*100:>11.1f}%")
    txt.append(f"  {'with gap':<14s} {has_gap.sum():>8d} {s68_gap:>14.0f} {tail_gap*100:>11.1f}%")
    txt.append("")
    txt.append(f"  -> gap events: {s68_gap/max(s68_clean,1e-9):.1f}x wider sigma68, "
               f"{tail_gap/max(tail_clean,1e-9):.1f}x more tails.")

    is_tail = np.abs(res) > TAIL_MM * 1000
    if is_tail.sum() > 0:
        frac_tails_from_gaps = has_gap[is_tail].mean()
        txt.append(f"  -> {frac_tails_from_gaps*100:.0f}% of ALL tail events are gap clusters "
                   f"(base rate of gaps: {has_gap.mean()*100:.0f}%).")

    # --- plot 1: sigma68 + tail fraction as bars ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = ["no\ngap", "with\ngap"]
    axes[0].bar(groups, [s68_clean, s68_gap], color=["tab:green", "tab:red"], edgecolor="white")
    axes[0].set_ylabel("sigma68 [um]"); axes[0].set_title("resolution by cluster gaps")
    for i, v in enumerate([s68_clean, s68_gap]):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    axes[1].bar(groups, [tail_clean*100, tail_gap*100], color=["tab:green", "tab:red"], edgecolor="white")
    axes[1].set_ylabel("tail fraction (|res|>500um) [%]"); axes[1].set_title("tails by cluster gaps")
    for i, v in enumerate([tail_clean*100, tail_gap*100]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "sigma_by_gap.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'sigma_by_gap.png'}")

    # --- plot 2: |residual| vs. largest gap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    m = np.abs(res) < 2000
    ax.hexbin(np.clip(max_gap[m], 0, 10), np.abs(res[m]), gridsize=(22, 50),
              cmap="inferno", mincnt=1, extent=(0, 10, 0, 2000))
    ax.axhline(500, color="cyan", lw=1.0, ls="--", label="500 um tail boundary")
    ax.set_xlabel("largest gap in cluster [missing strips]")
    ax.set_ylabel("|residual| [um]")
    ax.set_title("XGBoost: |residual| vs. cluster gap")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "res_vs_gapsize.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'res_vs_gapsize.png'}")

    (OUT / "gap_tail_report.txt").write_text("\n".join(txt), encoding="utf-8")
    print(f"[GAPS] report -> {OUT / 'gap_tail_report.txt'}", flush=True)
    print("\n".join(txt).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
