"""Batch zero-shot evaluation: apply a trained model to multiple datasets from a config.

Example:
  python pipeline/06_zeroshot_batch.py \\
    --model outputs/gnn_tc_xcorr_v1_model.pt \\
    --norm  outputs/gnn_tc_xcorr_v1_norm.json \\
    --config configs/datasets.yaml \\
    --datasets h4_gif_29deg_530V_100ns h4_gif_29deg_530V_200ns \\
    --out-prefix zeroshot_xcorr_v1

  # or run on all datasets in the config:
  python pipeline/06_zeroshot_batch.py ... --all
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
PLOTS = OUT / "plots";   PLOTS.mkdir(exist_ok=True)


def eval_dataset(cfg: dict, model: StripModel, norm: Normalization,
                 n_strip_feats: int, device: torch.device,
                 out_prefix: str, detector_shift: float) -> dict:
    name      = cfg["name"]
    path      = Path(cfg["path"])
    theta_deg = float(cfg.get("theta_deg", 29.0))

    print(f"\n[Batch] === {name}  θ={theta_deg}°  {path.name} ===", flush=True)

    ev = load_events(path)
    a, b = learn_frame_transform(ev)
    all_idx = filter_road_empty(ev, a, b, detector_shift_mm=detector_shift)
    print(f"[Batch]   {ev.n_events} events -> {len(all_idx)} after road-empty filter "
          f"({100*(1-len(all_idx)/ev.n_events):.1f}% removed)", flush=True)

    # use theta from config, not from training norm — model input features use
    # tan(theta) for x_corr_rel; at inference we apply the correct angle for this dataset
    import copy
    eval_norm = copy.copy(norm)
    eval_norm.theta_deg = theta_deg

    ds = HitDataset(ev, all_idx, a, b, eval_norm,
                    detector_shift_mm=detector_shift,
                    tc_anchor=True, cluster_select=True)
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
    res   -= np.median(res)   # center
    qc     = np.abs(res) < 2.0
    fr     = fit_residuals(res[qc], fit_range_mm=0.5)

    print(f"[Batch]   sigma_core = {fr.sigma_core_um:.0f} µm", flush=True)
    print(f"[Batch]   sigma_68   = {fr.sigma_68_um:.0f} µm", flush=True)
    print(f"[Batch]   efficiency = {qc.mean()*100:.1f}%", flush=True)

    npz_path = OUT / f"{out_prefix}_{name}_predictions.npz"
    np.savez_compressed(npz_path, y_pred=y_pred, y_true=y_true,
                        theta_deg=theta_deg, dataset_name=name)
    print(f"[Batch]   saved: {npz_path.name}", flush=True)

    return dict(
        name=name, theta_deg=theta_deg,
        sigma_core_um=fr.sigma_core_um,
        sigma_68_um=fr.sigma_68_um,
        efficiency=qc.mean() * 100,
        n_events=len(y_pred),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, help="Path to _model.pt")
    p.add_argument("--norm",       required=True, help="Path to _norm.json")
    p.add_argument("--config",     required=True, help="Path to configs/datasets.yaml")
    p.add_argument("--datasets",   nargs="+",     help="Dataset names to evaluate (from config)")
    p.add_argument("--all",        action="store_true", help="Evaluate all datasets in config")
    p.add_argument("--out-prefix", default="zeroshot")
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

    norm = Normalization.load(args.norm)
    state = torch.load(args.model, map_location="cpu")
    n_strip_feats = state["strip_encoder.0.weight"].shape[1]
    n_global_feats = state["global_proj.weight"].shape[1]
    model = StripModel(n_strip_feats=n_strip_feats,
                       n_global_feats=n_global_feats).to(device)
    model.load_state_dict(state)
    print(f"[Batch] model: {n_strip_feats} strip feats  "
          f"{n_global_feats} global feats", flush=True)

    detector_shift = load_detector_shift(OUT)

    results = []
    for cfg in selected:
        try:
            r = eval_dataset(cfg, model, norm, n_strip_feats, device,
                             args.out_prefix, detector_shift)
            results.append(r)
        except Exception as e:
            print(f"[Batch] ERROR on {cfg['name']}: {e}", flush=True)

    print("\n[Batch] === Summary ===")
    print(f"{'Dataset':<35} {'θ':>5} {'σ_core':>8} {'σ₆₈':>8} {'eff':>7} {'N':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<35} {r['theta_deg']:>4.0f}° "
              f"{r['sigma_core_um']:>7.0f}µm "
              f"{r['sigma_68_um']:>7.0f}µm "
              f"{r['efficiency']:>6.1f}% "
              f"{r['n_events']:>7}")


if __name__ == "__main__":
    main()
