"""Phase 0 calibration: detector shift, frame transform, slope correlations.

Writes outputs/detector_shift.json (read by data_loader.load_detector_shift).

Plot 1  cm_residual_gauss.png       -- charge-mean residual, Gaussian fit (mu_shift)
Plot 2  cm_residual_vs_slope.png    -- residual mean vs. track slope (Vogel Eq. 5.22)
Plot 3  frame_residual_vs_nhits.png -- frame residual vs. n_hits (selection bias check)
Plot 4  frame_residual_vs_slope.png -- frame residual vs. track slope (Vogel Eq. 5.30)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import (
    PITCH_MM, ROAD_MM,
    load_events, learn_frame_transform, save_detector_shift, select_strips_in_road,
)

ROOT    = Path(__file__).resolve().parent
OUT     = ROOT / "outputs"
PLOTDIR = OUT / "plots" / "diagnose"
PLOTDIR.mkdir(parents=True, exist_ok=True)

def gauss(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)

def core_stats(residuals_mm: np.ndarray, core_range_mm: float = 0.5, bins: int = 200):
    """Model-free: mu from histogram peak, sigma from 68% half-width.
    """
    r = residuals_mm[np.isfinite(residuals_mm)]

    r_core = r[np.abs(r) < core_range_mm]
    counts, edges = np.histogram(r_core, bins=bins, range=(-core_range_mm, core_range_mm))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu_mode = float(centers[np.argmax(counts)])

    q16, q84 = np.quantile(r, [0.16, 0.84])
    sigma_68 = 0.5 * (q84 - q16)

    mu_median = float(np.median(r))
    return mu_mode, mu_median, sigma_68, counts, centers

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .root file")
    args = p.parse_args()
    DATA = Path(args.data)
    print(f"[DIAG] loading: {DATA.name}", flush=True)
    ev = load_events(DATA)
    a_frame, b_frame = learn_frame_transform(ev)
    print(f"[DIAG] Frame: track_icept = {a_frame:.6f} * out_xpos + {b_frame:+.4f}", flush=True)

    n_ev = ev.n_events
    cm_pred  = np.full(n_ev, np.nan)
    n_strips = np.zeros(n_ev, dtype=np.int32)
    for i in range(n_ev):
        if ev.n_hits[i] == 0:
            continue
        xs = np.asarray(ev.hits_x[i]); qs = np.asarray(ev.hits_q[i]); ts = np.asarray(ev.hits_t[i])
        track_x = (ev.track_icept[i] - b_frame) / a_frame
        sx, sq, st = select_strips_in_road(xs, qs, ts, track_x, road_mm=ROAD_MM)
        if sx.size == 0 or sq.sum() <= 0:
            continue
        x_cm        = (sx * sq).sum() / sq.sum()
        cm_pred[i]  = a_frame * x_cm + b_frame
        n_strips[i] = sx.size

    res_cm     = cm_pred - ev.track_icept
    valid      = np.isfinite(res_cm)
    res_valid  = res_cm[valid]
    slope_v    = ev.track_slope[valid]
    nhits_v    = n_strips[valid]
    print(f"[DIAG] valide Events nach Road-Selektion: {valid.sum()} / {n_ev}", flush=True)

    # no Gaussian fit: CM distribution has heavy tails; use peak position + 68% half-width
    qc_all = np.abs(res_valid) < 2.0
    res_qc = res_valid[qc_all]
    print(f"[DIAG] QC-Filter |res|<2mm: {qc_all.sum()} / {len(res_valid)} Events "
          f"({qc_all.mean()*100:.1f}%)", flush=True)
    mu_mode, mu_median, sigma_68, counts, centers = core_stats(res_qc, core_range_mm=0.5)
    mu_shift_mm = mu_median   # median is more robust than mode for flat distributions
    print(f"[DIAG] Core-Stats auf QC-Events:", flush=True)
    print(f"[DIAG]   Peak-Mode  = {mu_mode*1000:+.1f} um  (Histogramm-Maximum in +-0.5mm)", flush=True)
    print(f"[DIAG]   Median     = {mu_median*1000:+.1f} um  <- wird als Shift verwendet", flush=True)
    q16, q84 = np.quantile(res_qc, [0.16, 0.84])
    sigma_68 = 0.5 * (q84 - q16)
    print(f"[DIAG]   sigma_68   = {sigma_68*1000:.1f} um  (68%-Halbbreite, QC-Events)", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers * 1000, counts, width=(centers[1]-centers[0])*1000,
           color="steelblue", alpha=0.6, label="CM-Residuum (Core +-0.5mm)")
    ax.axvline(mu_mode   * 1000, color="red",    ls="--", lw=1.5, label=f"Mode  = {mu_mode*1000:+.1f} um")
    ax.axvline(mu_median * 1000, color="orange", ls="--", lw=1.5, label=f"Median= {mu_median*1000:+.1f} um")
    ax.axvline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Charge-Mean Residuum  [um]")
    ax.set_ylabel("Eintraege")
    ax.set_title("Phase 0.1 — Detektor-Shift (Diss Eq. 5.21)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "cm_residual_gauss.png", dpi=140); plt.close(fig)
    print(f"[DIAG]   -> {PLOTDIR / 'cm_residual_gauss.png'}")

    # residual mean vs. track slope (Vogel Diss. Eq. 5.22)
    slope_q = slope_v[qc_all]; res_q = res_qc
    s_lo, s_hi = np.quantile(slope_q, [0.02, 0.98])
    bins = np.linspace(s_lo, s_hi, 25)
    centers_s = 0.5 * (bins[:-1] + bins[1:])
    mu_per_bin, sem_per_bin, n_per_bin = [], [], []
    MIN_N_BIN = 200   # ignore bins with fewer events (edge bins are noisy)
    for k in range(len(bins) - 1):
        m = (slope_q >= bins[k]) & (slope_q < bins[k+1])
        if m.sum() < MIN_N_BIN:
            mu_per_bin.append(np.nan); sem_per_bin.append(np.nan); n_per_bin.append(0); continue
        r_bin = res_q[m]
        mu_per_bin.append(np.median(r_bin) * 1000.0)
        sem_per_bin.append(np.std(r_bin) / np.sqrt(m.sum()) * 1000.0)
        n_per_bin.append(int(m.sum()))
    mu_per_bin = np.array(mu_per_bin); sem_per_bin = np.array(sem_per_bin)
    ok = np.isfinite(mu_per_bin)
    if ok.sum() >= 3:
        p1, p0 = np.polyfit(centers_s[ok], mu_per_bin[ok], 1)
        bias_swing_um = float(np.max(mu_per_bin[ok]) - np.min(mu_per_bin[ok]))
    else:
        p1, p0, bias_swing_um = float("nan"), float("nan"), 0.0
    slope_lo, slope_hi = float(slope_q.min()), float(slope_q.max())
    print(f"[DIAG] Slope-Korrelation (Diss Eq. 5.22): p1={p1:+.1f}  p0={p0:+.1f} um", flush=True)
    print(f"[DIAG] Slope-Range: [{slope_lo:+.4f}, {slope_hi:+.4f}]  "
          f"Bins mit N>={MIN_N_BIN}: {ok.sum()}", flush=True)
    print(f"[DIAG] Bias-Swing (Median pro Bin): {bias_swing_um:.0f} um", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers_s, mu_per_bin, yerr=sem_per_bin, fmt="o", color="tab:blue", capsize=3, label="bin-Mittelwert")
    if np.isfinite(p1):
        ax.plot(centers_s, p1 * centers_s + p0, "r--", lw=1.5,
                label=f"Linear-Fit: {p1:+.0f}*slope {p0:+.0f}")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Track-Slope")
    ax.set_ylabel("mean(Residuum)  [um]")
    ax.set_title("Phase 0.2 — Z-Shift-Check (Diss Eq. 5.22)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "cm_residual_vs_slope.png", dpi=140); plt.close(fig)
    print(f"[DIAG]   -> {PLOTDIR / 'cm_residual_vs_slope.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    nh_q = nhits_v[qc_all]
    ax.hexbin(nh_q, res_q * 1000, gridsize=(40, 60), cmap="Blues", mincnt=1, extent=(0, 30, -1500, 1500))
    ax.axhline(0,                  color="black", lw=0.6, ls=":")
    ax.axhline(mu_shift_mm * 1000, color="red",   lw=1.0, ls="--", label=f"mu_shift = {mu_shift_mm*1000:+.0f} um")
    ax.set_xlabel("n_strips in Road")
    ax.set_ylabel("CM-Residuum  [um]")
    ax.set_title("Phase 0.3 — Frame-Fit Bias vs. Cluster-Groesse")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "frame_residual_vs_nhits.png", dpi=140); plt.close(fig)
    print(f"[DIAG]   -> {PLOTDIR / 'frame_residual_vs_nhits.png'}")

    # residual vs. track slope (Vogel Diss. Eq. 5.30)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hexbin(slope_q, res_q * 1000, gridsize=(50, 60), cmap="Greens", mincnt=1)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Track-Slope")
    ax.set_ylabel("CM-Residuum  [um]")
    ax.set_title("Phase 0.4 — Slope-Abhaengigkeit (Diss Eq. 5.30 Z-Rotation)")
    plt.tight_layout()
    fig.savefig(PLOTDIR / "frame_residual_vs_slope.png", dpi=140); plt.close(fig)
    print(f"[DIAG]   -> {PLOTDIR / 'frame_residual_vs_slope.png'}")

    json_path = save_detector_shift(
        OUT, mu_shift_mm,
        sigma_68_mm=sigma_68,
        slope_p1_um_per_slope=(p1 if np.isfinite(p1) else 0.0),
        slope_p0_um=(p0 if np.isfinite(p0) else 0.0),
        n_events_qc=int(qc_all.sum()),
        road_mm=ROAD_MM,
        frame_a=a_frame, frame_b=b_frame,
    )
    print(f"\n[DIAG] gespeichert: {json_path}")
    print(f"[DIAG] -> Phase 1 nutzt mu_shift = {mu_shift_mm*1000:+.1f} um")
    print(f"[DIAG] -> Pitch-Vergleich: |mu_shift| / pitch = {abs(mu_shift_mm)/PITCH_MM:.2f} Strips")

    # significance check: slope bias is real only if swing >> typical bin SEM
    sem_typical = float(np.nanmedian(sem_per_bin[ok])) if ok.sum() > 0 else float("inf")
    swing_significant = bias_swing_um > 3 * sem_typical

    print("\n=== EMPFEHLUNG ===")

    shift_um = mu_shift_mm * 1000
    if abs(shift_um) < 30:
        print(f"  Detektor-Shift (Median QC) = {shift_um:+.0f} um")
        print(f"  -> vernachlaessigbar (<30 um = 0.07 Strips), keine Korrektur noetig")
    else:
        print(f"  Detektor-Shift (Median QC) = {shift_um:+.0f} um -> SHIFT wird in Phase 1 angewandt")
    print(f"  [Info] Peak-Mode = {mu_mode*1000:+.0f} um -- Verteilung hat keinen scharfen Peak,")
    print(f"         das ist normal bei CM mit 29-Grad-Tracks (Drift-Asymmetrie).")
    print(f"  [Info] Nur {qc_all.mean()*100:.0f}% der Events in |res|<2mm -- "
          f"Road-Selektion allein reicht noch nicht, Phase 1 Training wird das verbessern.")
    if swing_significant:
        print(f"  Bias-Swing = {bias_swing_um:.0f} um (>{3:.0f}x SEM={sem_typical:.0f} um) -> Z-Shift signifikant")
        print(f"  -> Phase 2: y_corr = y - ({p1:+.1f}*slope + {p0:+.1f}) [um]")
    else:
        print(f"  Bias-Swing = {bias_swing_um:.0f} um aber SEM={sem_typical:.0f} um")
        print(f"  -> nicht signifikant (Rauschen durch breite CM-Verteilung), kein Z-Shift noetig")

if __name__ == "__main__":
    main()
