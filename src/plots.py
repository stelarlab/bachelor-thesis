"""Generate comparison plots into outputs/plots/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import uproot
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fits import fit_residuals

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Datensatz" / "Data_for_Roman_29deg_530V_100ns_x9.root"
OUT  = ROOT / "outputs"
PLOTDIR = OUT / "plots"
PLOTDIR.mkdir(exist_ok=True)

COLORS = {
    "cm":  "gray",
    "tc":  "tab:orange",
    "xgb": "tab:blue",
    "gnn": "tab:red",
    "gnn_alldata": "tab:green",
    "gnn_tc": "tab:purple",
}
LABELS = {
    "cm":         "Charge-Mean",
    "tc":         "Time-Corrected Centroid",
    "xgb":        "XGBoost",
    "gnn":        "GNN",
    "gnn_alldata":"GNN (alle Daten)",
    "gnn_tc":     "GNN + r_tc Feature",
}

def load_all():
    data = {}

    xgb_path = OUT / "xgb_predictions.npz"
    if xgb_path.exists():
        d = np.load(xgb_path)
        data["xgb"] = d["y_pred"] - d["y_true"]
        data["cm"]  = d["charge_mean_pred"] - d["y_true"]

    tc_path = OUT / "tc_predictions.npz"
    if tc_path.exists():
        d = np.load(tc_path)
        data["tc"] = d["y_pred"] - d["y_true"]

    for p in sorted(OUT.glob("*_predictions.npz")):
        stem = p.stem.replace("_predictions", "")
        if stem in ("xgb", "tc"):
            continue
        d = np.load(p)
        data[stem] = d["y_pred"] - d["y_true"]

    return data

def load_histories():
    histories = {}
    for p in sorted(OUT.glob("*_predictions.npz")):
        stem = p.stem.replace("_predictions", "")
        if stem in ("xgb", "tc"):
            continue
        d = np.load(p)
        if "history" in d:
            histories[stem] = d["history"]
    return histories

def plot_residuals(data, filename, xlim_um=2000, title="Residuen-Vergleich"):
    bin_width = 20  # µm per bin
    bins = np.arange(-xlim_um, xlim_um + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["cm", "tc", "xgb"] + [k for k in data if k not in ("cm", "tc", "xgb")]
    for key in order:
        if key not in data:
            continue
        res = data[key]
        qc  = np.abs(res) < 2.0
        if qc.sum() < 10:
            continue
        fr    = fit_residuals(res[qc], fit_range_mm=0.5)
        color = COLORS.get(key, "black")
        label = LABELS.get(key, key)
    
        mask = np.abs(res * 1000) <= xlim_um
        n_outside = (~mask).sum()
        ax.hist(res[mask] * 1000, bins=bins,
                histtype="step", lw=1.8, color=color,
                label=f"{label}  σ={fr.sigma_core_um:.0f} µm  eff={qc.mean()*100:.0f}%")

    ax.axvline( 2000, color="gray", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(-2000, color="gray", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(0, color="black", lw=0.6, ls="--")

    ax.axvspan(-xlim_um, -2000, alpha=0.04, color="red")
    ax.axvspan( 2000,  xlim_um, alpha=0.04, color="red")
    ax.text( 1980, ax.get_ylim()[0] * 1.5, "±2 mm", ha="right", va="bottom",
             fontsize=8, color="gray")
    ax.set_xlabel("y_pred − y_true  [µm]")
    ax.set_ylabel("entries")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)
    ax.set_xlim(-xlim_um, xlim_um)
    plt.tight_layout()
    fig.savefig(PLOTDIR / filename, dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / filename}")

def plot_efficiency(data):
    keys   = [k for k in ["cm", "tc", "xgb"] + sorted(data) if k in data]
    keys   = list(dict.fromkeys(keys))
    effs   = [np.mean(np.abs(data[k]) < 2.0) * 100 for k in keys]
    sigmas = []
    for k in keys:
        res = data[k]; qc = np.abs(res) < 2.0
        sigmas.append(fit_residuals(res[qc], fit_range_mm=0.5).sigma_core_um if qc.sum() > 10 else float("nan"))

    colors = [COLORS.get(k, "tab:gray") for k in keys]
    labels = [LABELS.get(k, k) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].barh(labels, effs, color=colors, edgecolor="white")
    axes[0].set_xlabel("fraction of events within 2 mm  [%]")
    axes[0].set_title("efficiency")
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(effs):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    axes[1].barh(labels, sigmas, color=colors, edgecolor="white")
    axes[1].set_xlabel("σ_core  [µm]")
    axes[1].set_title("core resolution")
    for i, v in enumerate(sigmas):
        if np.isfinite(v):
            axes[1].text(v + 1, i, f"{v:.0f} µm", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTDIR / "efficiency_bar.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'efficiency_bar.png'}")

def plot_history(histories):
    if not histories:
        print("  -> no history data found, skipping.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stem, hist in histories.items():
        color = COLORS.get(stem, "black")
        label = LABELS.get(stem, stem)
        eps   = hist[:, 0]
        axes[0].plot(eps, hist[:, 1], color=color, lw=1.5, label=label)
        axes[0].plot(eps, hist[:, 2], color=color, lw=1.5, ls="--")
        axes[1].plot(eps, hist[:, 3], color=color, lw=1.5, label=label)
    for ax in axes:
        ax.legend(fontsize=8); ax.set_xlabel("epoch")
    axes[0].set_ylabel("Huber loss"); axes[0].set_title("train (—) / val (--) loss")
    axes[1].set_ylabel("RMS [µm]");   axes[1].set_title("val RMS over epochs")
    plt.tight_layout()
    fig.savefig(PLOTDIR / "history_gnn.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'history_gnn.png'}")

def plot_scatter_cm_tc(data):
    if "cm" not in data or "tc" not in data:
        print("  -> cm oder tc fehlt, ueberspringe Scatter-Plot.")
        return
    res_cm = data["cm"]; res_tc = data["tc"]

    n = min(len(res_cm), len(res_tc))
    res_cm, res_tc = res_cm[:n], res_tc[:n]
    qc = (np.abs(res_cm) < 3.0) & (np.abs(res_tc) < 3.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(res_cm[qc] * 1000, res_tc[qc] * 1000,
              gridsize=80, cmap="Blues", mincnt=1)
    lim = 800
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("charge-mean residual  [µm]")
    ax.set_ylabel("time-corrected centroid residual  [µm]")
    ax.set_title("charge-mean vs. time-corrected centroid")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "scatter_cm_vs_tc.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'scatter_cm_vs_tc.png'}")

def plot_residual_vs_nstrips(data):
    xgb_path = OUT / "xgb_predictions.npz"
    if "xgb" not in data or not xgb_path.exists():
        return
    d = np.load(xgb_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(-800, 800, 100)
    for ax, (key, title) in zip(axes, [("xgb", "XGBoost"), ("gnn", "GNN")]):
        if key not in data:
            ax.set_visible(False); continue
        res = data[key] * 1000
        qc  = np.abs(res) < 2000
        ax.hist(res[qc], bins=bins, histtype="stepfilled", alpha=0.6,
                color=COLORS.get(key, "gray"), label="alle Events")

        tail = (np.abs(res) > 500) & (np.abs(res) < 2000)
        ax.hist(res[tail], bins=bins, histtype="step", lw=1.5,
                color="red", label=f"Tails (|res|>500µm): {tail.mean()*100:.1f}%")
        ax.set_xlabel("Residuum [µm]"); ax.set_ylabel("entries")
        ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTDIR / "tail_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  -> {PLOTDIR / 'tail_analysis.png'}")

def main():
    print("[PLOTS] loading ...", flush=True)
    data      = load_all()
    histories = load_histories()

    print("[PLOTS] generating ...", flush=True)
    plot_residuals(data, "residuals_all.png",  xlim_um=1500, title="residuals — test set (530 V, 29°)")
    plot_residuals(data, "residuals_zoom.png", xlim_um=500,  title="residuals — core region ±500 µm")
    plot_efficiency(data)
    plot_history(histories)
    plot_scatter_cm_tc(data)
    plot_residual_vs_nstrips(data)
    print(f"\n[PLOTS] done: {PLOTDIR}", flush=True)

if __name__ == "__main__":
    main()
