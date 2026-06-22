"""Zero-shot transfer: apply a trained model to an unseen dataset without retraining.

The TC anchor lives in the unified dataset.py (tc_anchor flag); dataset_tc.py was
merged into it. tc_anchor is inferred from the model: 5 strip features = TC model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_events, learn_frame_transform, load_detector_shift, filter_road_empty
from dataset import HitDataset, Normalization, collate_padded
from model import StripModel
from fits import fit_residuals

ROOT  = Path(__file__).resolve().parents[1]   # repo root (pipeline/ is one level down)
OUT   = ROOT / "outputs"
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ZeroShot] device = {device}")
    norm_path = Path(args.norm)
    # infer tc_anchor from model: 5 strip features = TC model (r_tc removed in redesign)
    _state = torch.load(args.model, map_location="cpu")
    n_strip_feats_model = _state["strip_encoder.0.weight"].shape[1]

    use_tc = args.tc or (n_strip_feats_model == 5)
    norm = Normalization.load(norm_path)
    print(f"[ZeroShot] model: {n_strip_feats_model} strip feats  use_tc={use_tc}")
    print(f"[ZeroShot] normalization: theta={norm.theta_deg}deg  tmax={norm.tmax_ns}ns")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    print(f"[ZeroShot] dataset: {data_path.name}")
    ev = load_events(data_path)
    a, b = learn_frame_transform(ev)
    detector_shift = load_detector_shift(OUT)
    all_idx = filter_road_empty(ev, a, b, detector_shift_mm=detector_shift)
    print(f"[ZeroShot] {ev.n_events} events  shift={detector_shift*1000:.1f}um  "
          f"road-empty removed: {ev.n_events - len(all_idx)} ({100*(1-len(all_idx)/ev.n_events):.1f}%)")
    ds = HitDataset(ev, all_idx, a, b, norm, detector_shift_mm=detector_shift,
                    tc_anchor=use_tc, cluster_select=True)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_padded, num_workers=0)
    model = StripModel(n_strip_feats=n_strip_feats_model).to(device)
    model.load_state_dict(_state)
    model.eval()
    print(f"[ZeroShot] model loaded: {args.model}")
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["strip_feats"].to(device), batch["mask"].to(device), batch["global_feats"].to(device))
            pred_xpos  = out.cpu() * 5.0 + batch["x_med"]
            pred_icept = pred_xpos * a + b
            y_pred_list.append(pred_icept.numpy())
            y_true_list.append(batch["track_icept"].numpy())
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    res = y_pred - y_true
    qc  = np.abs(res) < 2.0
    fr  = fit_residuals(res[qc], fit_range_mm=0.5)
    # Vogel §5.3.2: tracking error < σ_i ≈ 75 µm (reference chambers BLY).
    # The exact interpolation error at the DUT depends on the setup geometry (Eq. 5.17).
    # 57 µm is an estimate used here; treat sigma_det as approximate.
    SIGMA_TRACK_UM = 57.0
    sigma_det = np.sqrt(max(fr.sigma_core_um**2 - SIGMA_TRACK_UM**2, 0))
    print(f"[ZeroShot] events: {len(y_pred)}")
    print(f"[ZeroShot] efficiency: {qc.mean()*100:.1f}%")
    print(f"[ZeroShot] sigma_core = {fr.sigma_core_um:.0f} um")
    print(f"[ZeroShot] sigma_det  = {sigma_det:.0f} um")
    print(f"[ZeroShot] sigma_w    = {fr.sigma_weighted_um:.0f} um")
    npz_path = OUT / f"{args.out_prefix}_predictions.npz"
    np.savez_compressed(npz_path, y_pred=y_pred, y_true=y_true,
        slope_frame=a, offset_frame=b,
        train_norm_theta=norm.theta_deg, train_norm_tmax=norm.tmax_ns,
        eval_data=str(data_path.name))
    print(f"[ZeroShot] saved: {npz_path}")
    xlim_um = 2000
    bins = np.arange(-xlim_um, xlim_um + 20, 20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(res[np.abs(res*1000)<=xlim_um]*1000, bins=bins, histtype="step", lw=1.8, color="tab:red",
            label=f"zero-shot  sigma_core={fr.sigma_core_um:.0f}um  sigma_det={sigma_det:.0f}um  eff={qc.mean()*100:.0f}%")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("residual  y_pred - y_true  [um]")
    ax.set_ylabel("entries")
    ax.set_yscale("log")
    ax.set_xlim(-xlim_um, xlim_um)
    ax.legend(fontsize=9)
    ax.set_title(f"zero-shot: model ({norm.tmax_ns:.0f}ns/{norm.theta_deg:.0f}deg) -> {data_path.stem}")
    plt.tight_layout()
    plot_path = PLOTS / f"{args.out_prefix}_residuals.png"
    fig.savefig(plot_path, dpi=140); plt.close(fig)
    print(f"[ZeroShot] plot: {plot_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str, required=True)
    p.add_argument("--norm",       type=str, required=True)
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--out-prefix", type=str, default="zeroshot")
    p.add_argument("--tc",         action="store_true")
    run(p.parse_args())

if __name__ == "__main__":
    main()