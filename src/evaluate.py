"""Evaluate all models from outputs/*_predictions.npz and print comparison table."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fits import fit_residuals

ROOT = Path(__file__).resolve().parents[1]   # repo root (src/ is one level down)
DATA = ROOT / "Data_for_Roman_29deg_530V_100ns_x9.root"
OUT  = ROOT / "outputs"

GNN_COLORS = ["tab:red", "tab:green", "tab:purple", "tab:pink"]

def fit_fabian_h_residual_6():
    f = uproot.open(str(DATA))
    h = f["h_residual_6"]
    edges = h.axis().edges(); centers = 0.5 * (edges[:-1] + edges[1:])
    samples = np.repeat(centers, h.values().astype(int))
    return fit_residuals(samples, fit_range_mm=0.5)

def load_npz(path):
    d = np.load(path)
    res = d["y_pred"] - d["y_true"]
    qc  = np.abs(res) < 2.0
    domain = d["domain"] if "domain" in d else None
    return res, qc, fit_residuals(res[qc], fit_range_mm=0.5), domain

_DOMAIN_LABELS = ["100ns", "200ns", "300ns", "400ns"]

def print_domain_rows(stem, res, qc, domain):

    if domain is None:
        return
    for d_idx in sorted(np.unique(domain)):
        mask = domain == d_idx
        r = res[mask]; q = qc[mask]
        fr = fit_residuals(r[q], fit_range_mm=0.5)
        lbl = _DOMAIN_LABELS[d_idx] if d_idx < len(_DOMAIN_LABELS) else f"domain_{d_idx}"
        print_row(f"  GNN ({stem}) [{lbl}]", fr, q.mean())

def ml_metrics(y_pred, y_true):
    """Standard ML regression metrics — computed on all test events.

    Computed in micrometers for readability.
    Note: RMSE here == RMS in fits.py when bias≈0, but is exact regardless of bias.
    """
    res    = (y_pred - y_true) * 1000.0
    mse    = float(np.mean(res ** 2))
    rmse   = float(np.sqrt(mse))
    mae    = float(np.mean(np.abs(res)))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum(((y_true - y_true.mean()) * 1000.0) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2)

def print_ml_metrics(name, y_pred, y_true):
    m = ml_metrics(y_pred, y_true)
    print(f"  {name:44s}  MSE={m['mse']:8.0f} um^2  RMSE={m['rmse']:6.0f} um  "
          f"MAE={m['mae']:6.0f} um  R2={m['r2']:.4f}")

def print_row(name, fr, frac):
    if fr is None:
        print(f"  {name:44s} {'(n/a)':>48s}")
        return
    f = f"{frac*100:5.1f}%" if frac is not None else "  -  "
    print(f"  {name:44s} {fr.sigma_core_um:7.0f} um  {fr.sigma_weighted_um:6.0f} um  "
          f"{fr.sigma_68_um:6.0f} um  {fr.rms_um:5.0f}  {f:>7s}")

def main():
    print("[EVAL] charge-mean benchmark ...", flush=True)
    fit_fab = fit_fabian_h_residual_6()

    print("[EVAL] XGBoost ...", flush=True)
    xgb_path = OUT / "xgb_predictions.npz"
    res_xgb, qc_xgb, fit_xgb, _ = load_npz(xgb_path)
    d_xgb = np.load(xgb_path)
    res_cm = d_xgb["charge_mean_pred"] - d_xgb["y_true"]
    qc_cm  = np.abs(res_cm) < 2.0
    fit_cm = fit_residuals(res_cm[qc_cm], fit_range_mm=0.5)

    tc_path = OUT / "tc_predictions.npz"
    if tc_path.exists():
        print("[EVAL] time-corrected centroid ...", flush=True)
        res_tc, qc_tc, fit_tc, _ = load_npz(tc_path)
    else:
        res_tc, qc_tc, fit_tc = None, None, None

    gnn_entries = []
    for p in sorted(OUT.glob("*_predictions.npz")):
        stem = p.stem.replace("_predictions", "")
        if stem in ("xgb", "tc"):
            continue
        print(f"[EVAL] {stem} ...", flush=True)
        res, qc, fr, domain = load_npz(p)
        gnn_entries.append((stem, res, qc, fr, domain))

    header = f"  {'Method':44s} {'sigma_core':>9s}  {'sigma_w':>8s}  {'sigma_68':>8s}  {'RMS':>5s}  {'in_2mm':>7s}"
    print(f"\n{header}")
    print("  " + "-" * 90)
    print_row("Charge-mean (Vogel h_residual_6)",  fit_fab, None)
    print_row("Charge-mean (cluster-aware)", fit_cm,  qc_cm.mean())
    print_row("Time-corrected centroid", fit_tc, qc_tc.mean() if qc_tc is not None else None)
    print_row("XGBoost",                            fit_xgb, qc_xgb.mean())
    for stem, res, qc, fr, domain in gnn_entries:
        label = f"GNN ({stem})"
        print_row(label, fr, qc.mean())
        print_domain_rows(stem, res, qc, domain)

    print(f"\n  {'Method':44s}  {'MSE [um^2]':>12s}  {'RMSE [um]':>9s}  {'MAE [um]':>8s}  {'R2':>6s}")
    print("  " + "-" * 90)
    d_xgb_full = np.load(xgb_path)
    print_ml_metrics("Charge-mean (cluster-aware)", d_xgb_full["charge_mean_pred"], d_xgb_full["y_true"])
    print_ml_metrics("XGBoost",                     d_xgb_full["y_pred"],           d_xgb_full["y_true"])
    for stem, res, qc, fr, domain in gnn_entries:
        npz = np.load(OUT / f"{stem}_predictions.npz")
        print_ml_metrics(f"GNN ({stem})", npz["y_pred"], npz["y_true"])

    xlim_um = 2000
    bin_width = 20
    bins = np.arange(-xlim_um, xlim_um + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_base = [
        ("Eigene Charge-Mean",      res_cm,  qc_cm,  "gray"),
        ("Time-Corrected Centroid", res_tc,  qc_tc,  "tab:orange"),
        ("XGBoost",                 res_xgb, qc_xgb, "tab:blue"),
    ]
    for name, res, qc, color in plot_base:
        if res is None:
            continue
        fr = fit_residuals(res[qc], fit_range_mm=0.5)
        mask = np.abs(res * 1000) <= xlim_um
        ax.hist(res[mask] * 1000, bins=bins, histtype="step", lw=1.5, color=color,
                label=f"{name}  σ={fr.sigma_core_um:.0f} µm  eff={qc.mean()*100:.0f}%")
    for (stem, res, qc, fr, domain), color in zip(gnn_entries, GNN_COLORS):
        label = f"GNN ({stem})" if stem != "gnn" else "GNN"
        mask = np.abs(res * 1000) <= xlim_um
        ax.hist(res[mask] * 1000, bins=bins, histtype="step", lw=1.8, color=color,
                label=f"{label}  σ={fr.sigma_core_um:.0f} µm  eff={qc.mean()*100:.0f}%")
    ax.axvline( 2000, color="gray", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(-2000, color="gray", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.axvspan(-xlim_um, -2000, alpha=0.04, color="red")
    ax.axvspan( 2000,  xlim_um, alpha=0.04, color="red")
    ax.set_xlabel("y_pred − y_true  [µm]")
    ax.set_ylabel("entries")
    ax.set_yscale("log")
    ax.set_xlim(-xlim_um, xlim_um)
    ax.legend(fontsize=9)
    ax.set_title("residuals — test set (530 V, 29°)")
    plt.tight_layout()
    fig.savefig(OUT / "residuals_comparison.png", dpi=140)
    plt.close(fig)
    print(f"\n[EVAL] Plot: {OUT / 'residuals_comparison.png'}", flush=True)

if __name__ == "__main__":
    main()
