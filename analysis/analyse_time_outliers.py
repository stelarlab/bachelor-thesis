"""Link time-outlier strips to the XGBoost residuals.

Finding from diagnose_inputs: strip times sit physically in ~[-60, +160] ns
(p0.1..p99.9), but 0.07% of strips have |t|>300 ns -- down to -5111 ns. These are
broken single strips. Question: do they pull the XGBoost prediction into the tails?

A test event counts as a "time outlier" if at least one strip in the road cluster
has |t| > T_OUTLIER ns. We compare sigma68 / tail fraction of clean vs. outlier
events, and check whether a sanity cut would improve the resolution.

Mapping as in analyse_gaps_vs_tails: orig = np.where(mask)[0][test_idx].

Outputs:
    outputs/plots/time/sigma_by_timeoutlier.png
    outputs/plots/time/res_vs_maxabst.png
    outputs/time_outlier_report.txt
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
    ROAD_MM,
    load_events, learn_frame_transform, load_detector_shift, select_strips_in_road,
)
from features import build_features

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs"
PLOTDIR = OUT / "plots" / "time"
PLOTDIR.mkdir(parents=True, exist_ok=True)

T_OUTLIER = 300.0   # ns; physical window is ~[-60,+160] (p0.1..p99.9)
TAIL_MM   = 0.5


def time_props_for_events(ev, orig_idx, a, b, shift):
    """Per test event: max|t| in the road cluster (detect broken strips)."""
    max_abs_t = np.zeros(len(orig_idx), dtype=np.float64)
    for k, i in enumerate(orig_idx):
        xs = np.asarray(ev.hits_x[i], dtype=np.float64) - shift
        qs = np.asarray(ev.hits_q[i], dtype=np.float64)
        ts = np.asarray(ev.hits_t[i], dtype=np.float64)
        track_x = (ev.track_icept[i] - b) / a
        sx, sq, st = select_strips_in_road(xs, qs, ts, track_x, road_mm=ROAD_MM)
        if st.size > 0:
            max_abs_t[k] = float(np.abs(st).max())
    return max_abs_t


def s68_tail(r):
    q16, q84 = np.quantile(r, [0.16, 0.84])
    return 0.5 * (q84 - q16), float(np.mean(np.abs(r) > TAIL_MM * 1000))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    args = p.parse_args()
    DATA = Path(args.data)

    print("[TIME] loading data + XGBoost predictions ...", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    shift = load_detector_shift(OUT)
    X, y, mask, _ = build_features(ev, a, b, detector_shift_mm=shift)

    d = np.load(OUT / "xgb_predictions.npz")
    test_idx = d["test_idx"]
    res = (d["y_pred"] - d["y_true"]) * 1000.0
    orig_idx = np.where(mask)[0][test_idx]

    print(f"[TIME] {len(orig_idx)} test events, computing time properties ...", flush=True)
    max_abs_t = time_props_for_events(ev, orig_idx, a, b, shift)
    bad = max_abs_t > T_OUTLIER

    txt = ["Time outliers vs. tails -- XGBoost test set", ""]
    txt.append(f"threshold: a strip with |t| > {T_OUTLIER:.0f} ns flags the event as outlier")
    txt.append(f"total test events:   {len(res)}")
    txt.append(f"time outliers:       {bad.sum()}  ({bad.mean()*100:.2f}%)")
    txt.append("")

    s68_bad,   tail_bad   = s68_tail(res[bad])   if bad.sum() > 10 else (float("nan"), float("nan"))
    s68_clean, tail_clean = s68_tail(res[~bad])
    txt.append(f"  {'group':<18s} {'N':>8s} {'sigma68 [um]':>14s} {'tail>500um':>12s}")
    txt.append(f"  {'clean time':<18s} {(~bad).sum():>8d} {s68_clean:>14.0f} {tail_clean*100:>11.1f}%")
    txt.append(f"  {'time outlier':<18s} {bad.sum():>8d} {s68_bad:>14.0f} {tail_bad*100:>11.1f}%")
    txt.append("")

    is_tail = np.abs(res) > TAIL_MM * 1000
    if is_tail.sum() > 0:
        frac = bad[is_tail].mean()
        txt.append(f"  -> {frac*100:.1f}% of all tail events are time outliers "
                   f"(base rate: {bad.mean()*100:.2f}%).")
        if bad.mean() > 0:
            txt.append(f"  -> enrichment in the tail: {frac/bad.mean():.1f}x")

    # hypothetical sanity cut: sigma68 with time outliers removed
    txt.append("")
    txt.append(f"  sigma68 all events:          {s68_tail(res)[0]:.0f} um")
    txt.append(f"  sigma68 after cut |t|<{T_OUTLIER:.0f}ns: {s68_clean:.0f} um "
               f"(removes {bad.mean()*100:.2f}% of events)")

    # --- plot 1: bars sigma68 + tail ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = ["clean\ntime", "time\noutlier"]
    axes[0].bar(groups, [s68_clean, s68_bad], color=["tab:green", "tab:red"], edgecolor="white")
    axes[0].set_ylabel("sigma68 [um]"); axes[0].set_title("resolution by time quality")
    for i, v in enumerate([s68_clean, s68_bad]):
        if np.isfinite(v): axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    axes[1].bar(groups, [tail_clean*100, tail_bad*100], color=["tab:green", "tab:red"], edgecolor="white")
    axes[1].set_ylabel("tail fraction (|res|>500um) [%]"); axes[1].set_title("tails by time quality")
    for i, v in enumerate([tail_clean*100, tail_bad*100]):
        if np.isfinite(v): axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "sigma_by_timeoutlier.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'sigma_by_timeoutlier.png'}")

    # --- plot 2: |residual| vs. max|t| in cluster ---
    fig, ax = plt.subplots(figsize=(8, 5))
    m = np.abs(res) < 2000
    ax.scatter(np.clip(max_abs_t[m], 0, 800), np.abs(res[m]), s=3, alpha=0.15, color="tab:purple")
    ax.axhline(500, color="red", lw=1.0, ls="--", label="500 um tail boundary")
    ax.axvline(T_OUTLIER, color="black", lw=1.0, ls=":", label=f"|t|={T_OUTLIER:.0f}ns threshold")
    ax.set_xlabel("max |t| in road cluster [ns]")
    ax.set_ylabel("|residual| [um]")
    ax.set_title("XGBoost: |residual| vs. largest strip time")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "res_vs_maxabst.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'res_vs_maxabst.png'}")

    (OUT / "time_outlier_report.txt").write_text("\n".join(txt), encoding="utf-8")
    print(f"[TIME] report -> {OUT / 'time_outlier_report.txt'}", flush=True)
    print("\n".join(txt).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
