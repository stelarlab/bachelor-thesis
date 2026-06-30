"""Batch zero-shot evaluation: apply a trained model to multiple datasets from a config.

Example:
  python pipeline/06_zeroshot_batch.py \\
    --model outputs/gnn_tc_xcorr_v1_model.pt \\
    --norm  outputs/gnn_tc_xcorr_v1_norm.json \\
    --config configs/datasets.yaml \\
    --datasets h8_29deg_530V_100ns \\
    --out-prefix zeroshot_xcorr

  # all datasets:
  python pipeline/06_zeroshot_batch.py ... --all
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_events, learn_frame_transform, load_detector_shift, filter_road_empty
from dataset import HitDataset, Normalization, collate_padded
from model import StripModel
from fits import fit_residuals

ROOT  = Path(__file__).resolve().parents[1]
OUT   = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
PLOTS = OUT  / "plots";   PLOTS.mkdir(exist_ok=True)


def _plot_residuals(res, fr, name, out_prefix, theta_deg):
    xlim_um  = 1500
    bin_width = 20
    bins = np.arange(-xlim_um, xlim_um + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(res[np.abs(res * 1000) <= xlim_um] * 1000, bins=bins,
            histtype="step", lw=1.8, color="tab:red",
            label=f"σ_core={fr.sigma_core_um:.0f} µm  σ₆₈={fr.sigma_68_um:.0f} µm")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("y_pred − y_true  [µm]")
    ax.set_ylabel("entries")
    ax.set_yscale("log")
    ax.set_xlim(-xlim_um, xlim_um)
    ax.legend(fontsize=9)
    ax.set_title(f"zero-shot: {name}  θ={theta_deg:.0f}°")
    plt.tight_layout()
    p = PLOTS / f"{out_prefix}_{name}_residuals.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[Batch]   plot: {p.name}", flush=True)


def _plot_residuals_vs_position(res, y_true, name, out_prefix, theta_deg):
    qc = np.abs(res) < 2.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hexbin(y_true[qc], res[qc] * 1000, gridsize=80, cmap="Blues",
              mincnt=1, vmax=None)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    order   = np.argsort(y_true[qc])
    x_s     = y_true[qc][order]
    r_s     = res[qc][order] * 1000
    w       = max(len(x_s) // 50, 10)
    x_med   = np.convolve(x_s, np.ones(w)/w, mode="valid")
    r_med   = np.convolve(r_s, np.ones(w)/w, mode="valid")
    ax.plot(x_med, r_med, color="red", lw=1.5, label="rolling mean")
    ax.set_xlabel("true position  [mm]")
    ax.set_ylabel("residual  [µm]")
    ax.set_ylim(-500, 500)
    ax.legend(fontsize=8)
    ax.set_title(f"residual vs. position: {name}  θ={theta_deg:.0f}°")
    plt.tight_layout()
    p = PLOTS / f"{out_prefix}_{name}_res_vs_pos.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[Batch]   plot: {p.name}", flush=True)


def eval_dataset(cfg, model, norm, n_strip_feats, device, out_prefix, detector_shift, tc_anchor=True):
    name      = cfg["name"]
    path      = Path(cfg["path"])
    theta_deg = float(cfg.get("theta_deg", 29.0))

    layer     = cfg.get("layer", None)

    print(f"\n[Batch] === {name}  θ={theta_deg}°  layer={layer}  {path.name} ===", flush=True)

    ev = load_events(path, layer=layer)
    a, b = learn_frame_transform(ev)
    all_idx = filter_road_empty(ev, a, b, detector_shift_mm=detector_shift)
    n_total = ev.n_events
    print(f"[Batch]   {n_total} events -> {len(all_idx)} after road-empty filter "
          f"({100*(1-len(all_idx)/n_total):.1f}% removed)", flush=True)

    eval_norm = copy.copy(norm)
    eval_norm.theta_deg = theta_deg

    ds = HitDataset(ev, all_idx, a, b, eval_norm,
                    detector_shift_mm=detector_shift,
                    tc_anchor=tc_anchor, cluster_select=True)
    dl = DataLoader(ds, batch_size=512, shuffle=False,
                    collate_fn=collate_padded, num_workers=0)

    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(
                batch["strip_feats"].to(device),
                batch["mask"].to(device),
                batch["global_feats"].to(device),
            )
            pred_xpos  = out.cpu() * 5.0 + batch["x_med"]
            pred_icept = pred_xpos * a + b
            y_pred_list.append(pred_icept.numpy())
            y_true_list.append(batch["track_icept"].numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    res    = y_pred - y_true
    res   -= np.median(res)

    # efficiency (2 definitions):
    # eff_reco  = events within ±2mm / events that passed road-empty filter (reconstructable)
    # eff_total = events within ±2mm / all tracks (including road-empty)
    qc           = np.abs(res) < 2.0
    eff_reco     = qc.mean() * 100
    eff_total    = (qc.sum() / n_total) * 100

    fr = fit_residuals(res[qc], fit_range_mm=0.5)

    print(f"[Batch]   sigma_core     = {fr.sigma_core_um:.0f} µm", flush=True)
    print(f"[Batch]   sigma_68       = {fr.sigma_68_um:.0f} µm", flush=True)
    print(f"[Batch]   eff (reco)     = {eff_reco:.1f}%  (within ±2mm / reconstructable events)", flush=True)
    print(f"[Batch]   eff (total)    = {eff_total:.1f}%  (within ±2mm / all tracks)", flush=True)

    npz_path = OUT / f"{out_prefix}_{name}_predictions.npz"
    np.savez_compressed(npz_path, y_pred=y_pred, y_true=y_true,
                        theta_deg=theta_deg, dataset_name=np.array(name))
    print(f"[Batch]   saved: {npz_path.name}", flush=True)

    _plot_residuals(res, fr, name, out_prefix, theta_deg)
    _plot_residuals_vs_position(res, y_true, name, out_prefix, theta_deg)

    return dict(
        name=name, theta_deg=theta_deg,
        sigma_core_um=fr.sigma_core_um,
        sigma_68_um=fr.sigma_68_um,
        eff_reco=eff_reco,
        eff_total=eff_total,
        n_reco=len(y_pred),
        n_total=n_total,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, help="Path to _model.pt")
    p.add_argument("--norm",       required=True, help="Path to _norm.json")
    p.add_argument("--config",     required=True, help="Path to configs/datasets.yaml")
    p.add_argument("--datasets",   nargs="+",     help="Dataset names from config")
    p.add_argument("--all",        action="store_true", help="Evaluate all datasets in config")
    p.add_argument("--out-prefix",  default="zeroshot")
    p.add_argument("--no-tc-anchor", action="store_true",
                   help="Use median(x) anchor instead of TC-centroid (must match training).")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    all_cfgs = {c["name"]: c for c in config["datasets"]}

    if args.all:
        selected = list(all_cfgs.values())
    elif args.datasets:
        selected = [all_cfgs[n] for n in args.datasets if n in all_cfgs]
        missing  = [n for n in args.datasets if n not in all_cfgs]
        if missing:
            print(f"[Batch] WARNING: unknown dataset names: {missing}")
    else:
        p.error("Specify --datasets <name> ... or --all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Batch] device = {device}", flush=True)

    norm_loaded = Normalization.load(args.norm)
    norm_is_list = isinstance(norm_loaded, list)
    norm_default = norm_loaded[0] if norm_is_list else norm_loaded
    state = torch.load(args.model, map_location="cpu")
    n_strip_feats  = state["strip_encoder.0.weight"].shape[1]
    n_global_feats = state["global_proj.weight"].shape[1]
    d_model  = state["strip_encoder.0.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("transformer.layers.") and k.endswith(".norm1.weight"))
    n_heads  = d_model // 16  # convention: 4 for d=64, 8 for d=128
    model = StripModel(n_strip_feats=n_strip_feats, n_global_feats=n_global_feats,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    model.load_state_dict(state)
    print(f"[Batch] model: {n_strip_feats} strip feats  {n_global_feats} global feats  "
          f"d={d_model}  heads={n_heads}  layers={n_layers}", flush=True)
    if norm_is_list:
        print(f"[Batch] per-dataset norm detected ({len(norm_loaded)} datasets)", flush=True)

    detector_shift = load_detector_shift(OUT)

    results = []
    for cfg in selected:
        try:
            if norm_is_list:
                norm = Normalization.load_for_dataset(args.norm, cfg["name"])
            else:
                norm = norm_default
            r = eval_dataset(cfg, model, norm, n_strip_feats,
                             device, args.out_prefix, detector_shift,
                             tc_anchor=not args.no_tc_anchor)
            results.append(r)
        except Exception as e:
            print(f"[Batch] ERROR on {cfg['name']}: {e}", flush=True)

    print("\n[Batch] === Summary ===")
    print(f"{'Dataset':<35} {'θ':>5} {'σ_core':>8} {'σ₆₈':>8} "
          f"{'eff_reco':>10} {'eff_total':>11} {'N_reco':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<35} {r['theta_deg']:>4.0f}° "
              f"{r['sigma_core_um']:>7.0f}µm "
              f"{r['sigma_68_um']:>7.0f}µm "
              f"{r['eff_reco']:>9.1f}% "
              f"{r['eff_total']:>10.1f}% "
              f"{r['n_reco']:>8}")


if __name__ == "__main__":
    main()
