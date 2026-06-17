"""Input diagnostics: raw-data shape and properties of the XGBoost features.

Answers the question "what do my inputs actually look like" before any model tuning:
  1. cluster shape    -- n_strips per event (raw and after road selection)
  2. gaps             -- missing strips inside a cluster (split clusters / dead channels)
  3. feature table    -- range, NaN fraction, zero fraction, std per XGBoost feature
  4. feature histos   -- per-feature distribution (reveals spikes, clipped edges)

Outputs:
    outputs/plots/inputs/nstrips_distribution.png
    outputs/plots/inputs/gap_distribution.png
    outputs/plots/inputs/feature_histograms.png
    outputs/input_feature_report.txt
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
from features import FEATURES, build_features

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs"
PLOTDIR = OUT / "plots" / "inputs"
PLOTDIR.mkdir(parents=True, exist_ok=True)

GAP_TOL = 1.5 * PITCH_MM   # neighbour spacing > 1.5 pitch counts as a gap


def cluster_shape(ev, a, b, detector_shift_mm):
    """Per event: raw n_strips, n_strips in road, number of gaps in the road cluster."""
    n = ev.n_events
    n_raw  = np.zeros(n, dtype=np.int32)
    n_road = np.zeros(n, dtype=np.int32)
    n_gaps = np.full(n, -1, dtype=np.int32)   # -1 = no valid road cluster
    max_gap_strips = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if ev.n_hits[i] == 0:
            continue
        xs = np.asarray(ev.hits_x[i], dtype=np.float64) - detector_shift_mm
        qs = np.asarray(ev.hits_q[i], dtype=np.float64)
        ts = np.asarray(ev.hits_t[i], dtype=np.float64)
        n_raw[i] = xs.size

        track_x = (ev.track_icept[i] - b) / a
        sx, _, _ = select_strips_in_road(xs, qs, ts, track_x, road_mm=ROAD_MM)
        if sx.size == 0:
            continue
        n_road[i] = sx.size
        sx_sorted = np.sort(sx)
        if sx.size >= 2:
            diffs = np.diff(sx_sorted)
            gap_mask = diffs > GAP_TOL
            n_gaps[i] = int(gap_mask.sum())
            missing = np.round(diffs / PITCH_MM) - 1.0   # missing strips per gap
            max_gap_strips[i] = float(missing[gap_mask].max()) if gap_mask.any() else 0.0
        else:
            n_gaps[i] = 0
    return n_raw, n_road, n_gaps, max_gap_strips


def plot_nstrips(n_raw, n_road, txt):
    valid_raw  = n_raw[n_raw > 0]
    valid_road = n_road[n_road > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(0, 31) - 0.5
    for ax, data, title, color in [
        (axes[0], valid_raw,  "n_strips per event (raw)",   "tab:gray"),
        (axes[1], valid_road, "n_strips in road (+-5 mm)",  "tab:blue"),
    ]:
        ax.hist(data[data <= 30], bins=bins, color=color, edgecolor="white")
        ax.set_xlabel("number of strips"); ax.set_ylabel("events")
        ax.set_title(title); ax.set_xlim(0, 30)
        ax.axvline(np.median(data), color="red", ls="--", lw=1.2,
                   label=f"median = {np.median(data):.0f}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "nstrips_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'nstrips_distribution.png'}")

    for name, data in [("raw", valid_raw), ("road", valid_road)]:
        txt.append(f"  n_strips ({name}): median={np.median(data):.0f}  "
                   f"mean={data.mean():.2f}  min={data.min()}  max={data.max()}  "
                   f"p99={np.percentile(data, 99):.0f}")


def plot_gaps(n_gaps, max_gap_strips, txt):
    valid = n_gaps >= 0
    g = n_gaps[valid]
    frac_with_gap = float((g > 0).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(g[g <= 6], bins=np.arange(0, 8) - 0.5, color="tab:orange", edgecolor="white")
    axes[0].set_xlabel("number of gaps in road cluster"); axes[0].set_ylabel("events")
    axes[0].set_title(f"cluster gaps  ({frac_with_gap*100:.1f}% have >=1 gap)")

    mgap = max_gap_strips[valid]
    mgap = mgap[mgap > 0]
    if mgap.size > 0:
        axes[1].hist(mgap[mgap <= 10], bins=np.arange(0, 12) - 0.5, color="tab:red", edgecolor="white")
    axes[1].set_xlabel("largest gap [missing strips]"); axes[1].set_ylabel("events (with gap)")
    axes[1].set_title("size of the largest gap per cluster")
    plt.tight_layout()
    fig.savefig(PLOTDIR / "gap_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'gap_distribution.png'}")

    txt.append(f"  events with >=1 gap in road cluster: {frac_with_gap*100:.1f}%")
    if mgap.size > 0:
        txt.append(f"  largest gap: median={np.median(mgap):.0f}  max={mgap.max():.0f} strips")


def feature_table(X, mask, txt):
    """Range / NaN / zero / std per feature on the valid events."""
    Xv = X[mask]
    txt.append("")
    txt.append(f"  feature properties ({mask.sum()} valid events):")
    txt.append(f"  {'feature':<28s} {'min':>10s} {'max':>10s} {'mean':>10s} "
               f"{'std':>10s} {'%NaN':>6s} {'%==0':>6s}")
    for j, name in enumerate(FEATURES):
        col_all = X[:, j]
        col     = Xv[:, j]
        pct_nan = float(np.isnan(col_all).mean()) * 100.0
        finite  = col[np.isfinite(col)]
        if finite.size == 0:
            txt.append(f"  {name:<28s} {'(all NaN)':>40s}")
            continue
        pct_zero = float((finite == 0).mean()) * 100.0
        txt.append(f"  {name:<28s} {finite.min():>10.3f} {finite.max():>10.3f} "
                   f"{finite.mean():>10.3f} {finite.std():>10.3f} "
                   f"{pct_nan:>5.1f}% {pct_zero:>5.1f}%")


def plot_feature_histograms(X, mask):
    Xv = X[mask]
    ncol = 4
    nrow = int(np.ceil(len(FEATURES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = axes.flatten()
    for j, name in enumerate(FEATURES):
        ax = axes[j]
        col = Xv[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            ax.set_visible(False); continue
        lo, hi = np.percentile(col, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1e-6
        m = (col >= lo) & (col <= hi)   # xlim via mask, not clip -> no edge spike
        ax.hist(col[m], bins=60, color="tab:blue", alpha=0.75)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("XGBoost input features -- distributions (0.5-99.5 percentile)", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "feature_histograms.png", dpi=140)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'feature_histograms.png'}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .root file")
    args = p.parse_args()
    DATA = Path(args.data)
    print(f"[INPUTS] loading: {DATA.name}", flush=True)
    ev = load_events(DATA)
    a, b = learn_frame_transform(ev)
    detector_shift = load_detector_shift(OUT)
    print(f"[INPUTS] frame: a={a:.6f} b={b:+.4f}  shift={detector_shift*1000:+.1f}um", flush=True)

    txt = [f"Input feature diagnostics -- {DATA.name}",
           f"total events: {ev.n_events}", ""]

    print("[INPUTS] cluster shape & gaps ...", flush=True)
    n_raw, n_road, n_gaps, max_gap = cluster_shape(ev, a, b, detector_shift)
    txt.append("cluster shape:")
    plot_nstrips(n_raw, n_road, txt)
    plot_gaps(n_gaps, max_gap, txt)

    n_empty = int((ev.n_hits == 0).sum())
    n_noroad = int(((n_raw > 0) & (n_road == 0)).sum())
    txt.append("")
    txt.append(f"  empty events (n_hits==0):           {n_empty}  ({n_empty/ev.n_events*100:.1f}%)")
    txt.append(f"  events with hits but no road strip: {n_noroad}  ({n_noroad/ev.n_events*100:.1f}%)")

    print("[INPUTS] feature properties ...", flush=True)
    X, y, mask, _ = build_features(ev, a, b, detector_shift_mm=detector_shift)
    feature_table(X, mask, txt)
    plot_feature_histograms(X, mask)

    report = OUT / "input_feature_report.txt"
    report.write_text("\n".join(txt), encoding="utf-8")
    print(f"[INPUTS] report -> {report}", flush=True)
    print("\n".join(txt).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
