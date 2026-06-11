from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_events, learn_frame_transform, load_detector_shift
from dataset_tc import HitDataset as HitDatasetTC, Normalization as NormTC, collate_padded as collate_tc
from dataset   import HitDataset, Normalization, collate_padded
from model import StripModel
from fits import fit_residuals

ROOT  = Path(__file__).resolve().parent
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
    norm = NormTC.load(norm_path) if use_tc else Normalization.load(norm_path)
    print(f"[ZeroShot] Modell: {n_strip_feats_model} Strip-Feats  use_tc={use_tc}")
    print(f"[ZeroShot] Normierung: theta={norm.theta_deg}deg  tmax={norm.tmax_ns}ns")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    print(f"[ZeroShot] Datensatz: {data_path.name}")
    ev = load_events(data_path)
    a, b = learn_frame_transform(ev)
    detector_shift = load_detector_shift(OUT)
    print(f"[ZeroShot] {ev.n_events} Events  shift={detector_shift*1000:.1f}um")
    all_idx = np.arange(ev.n_events)
    if use_tc:
        ds = HitDatasetTC(ev, all_idx, a, b, norm, detector_shift_mm=detector_shift)
        dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_tc, num_workers=0)
        n_strip_feats = n_strip_feats_model
    else:
        ds = HitDataset(ev, all_idx, a, b, norm, detector_shift_mm=detector_shift)
        dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_padded, num_workers=0)
        n_strip_feats = n_strip_feats_model
    model = StripModel(n_strip_feats=n_strip_feats).to(device)
    model.load_state_dict(_state)
    model.eval()
    print(f"[ZeroShot] Modell geladen: {args.model}")
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
    sigma_det = np.sqrt(max(fr.sigma_core_um**2 - 57**2, 0))
    print(f"[ZeroShot] Events: {len(y_pred)}")
    print(f"[ZeroShot] Effizienz: {qc.mean()*100:.1f}%")
    print(f"[ZeroShot] sigma_core = {fr.sigma_core_um:.0f} um")
    print(f"[ZeroShot] sigma_det  = {sigma_det:.0f} um")
    print(f"[ZeroShot] sigma_w    = {fr.sigma_weighted_um:.0f} um")
    npz_path = OUT / f"{args.out_prefix}_predictions.npz"
    np.savez_compressed(npz_path, y_pred=y_pred, y_true=y_true,
        slope_frame=a, offset_frame=b,
        train_norm_theta=norm.theta_deg, train_norm_tmax=norm.tmax_ns,
        eval_data=str(data_path.name))
    print(f"[ZeroShot] gespeichert: {npz_path}")
    xlim_um = 2000
    bins = np.arange(-xlim_um, xlim_um + 20, 20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(res[np.abs(res*1000)<=xlim_um]*1000, bins=bins, histtype="step", lw=1.8, color="tab:red",
            label=f"Zero-Shot  sigma_core={fr.sigma_core_um:.0f}um  sigma_det={sigma_det:.0f}um  eff={qc.mean()*100:.0f}%")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Residuum  y_pred - y_true  [um]")
    ax.set_ylabel("Eintraege")
    ax.set_yscale("log")
    ax.set_xlim(-xlim_um, xlim_um)
    ax.legend(fontsize=9)
    ax.set_title(f"Zero-Shot: Modell ({norm.tmax_ns:.0f}ns/{norm.theta_deg:.0f}deg) -> {data_path.stem}")
    plt.tight_layout()
    plot_path = PLOTS / f"{args.out_prefix}_residuals.png"
    fig.savefig(plot_path, dpi=140); plt.close(fig)
    print(f"[ZeroShot] Plot: {plot_path}")

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