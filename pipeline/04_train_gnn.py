"""Train the strip self-attention GNN for position reconstruction.

This script covers all GNN variants through CLI flags:
  --tc-anchor     use TC-centroid anchor instead of median(x)
  --data A B ...  one file = single-domain; multiple files = multi-domain training
  --train-all     merge train+val splits for the final model (test set untouched)

Output files (all under outputs/):
  <prefix>_model.pt       model weights
  <prefix>_norm.json      normalization statistics
  <prefix>_predictions.npz  y_pred, y_true, domain labels, training history

Example — single domain, TC-anchor:
  python train_gnn.py --data Data_100ns.root --tc-anchor --out-prefix gnn_tc

Example — multi-domain:
  python train_gnn.py --data Data_100ns.root Data_200ns.root --out-prefix gnn_multidomain
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_events, learn_frame_transform, load_detector_shift, filter_road_empty
from dataset import HitDataset, Normalization, collate_padded
from model import StripModel

MAX_NHITS_TRAIN = 50   # drop extreme-multiplicity events from training (Fabian, group meeting)

ROOT = Path(__file__).resolve().parents[1]   # repo root (pipeline/ is one level down)
OUT  = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

def _resolve_device(choice: str) -> torch.device:
    """Pick compute device; fall back to CPU if CUDA build doesn’t support the GPU."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps" or (choice == "auto" and
            torch.backends.mps.is_available() and not torch.cuda.is_available()):
        print("[GNN] device = MPS (Apple Silicon)", flush=True)
        return torch.device("mps")
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            sm  = f"sm_{cap[0]}{cap[1]}"
            if sm not in torch.cuda.get_arch_list() and choice == "auto":
                print(f"[GNN] GPU {torch.cuda.get_device_name(0)} ({sm}) not supported — falling back to CPU.", flush=True)
                return torch.device("cpu")
        except Exception as exc:
            print(f"[GNN] CUDA check failed ({exc}) — using CPU.", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    p = argparse.ArgumentParser(description="Train strip self-attention GNN.")
    p.add_argument("--data",         type=str, nargs="+", required=True,
                   help="Path(s) to .root file(s). Multiple = multi-domain training.")
    p.add_argument("--out-prefix",   type=str, default="gnn")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model",      type=int,   default=64)
    p.add_argument("--n-heads",      type=int,   default=4)
    p.add_argument("--n-layers",     type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--device",       choices=["auto","cuda","mps","cpu"], default="auto")
    p.add_argument("--tmax",         type=float, default=100.0,
                   help="Shaping time [ns] of the dataset. Single-domain only.")
    p.add_argument("--theta",        type=float, default=29.0,
                   help="Track incidence angle [deg].")
    p.add_argument("--tc-anchor",    action="store_true",
                   help="Use TC-centroid as anchor (better empirically; default: median).")
    p.add_argument("--no-cluster-select", action="store_true",
                   help="Disable within-road cluster isolation (default: enabled). "
                        "Cluster isolation picks the single connected strip cluster "
                        "closest to the track, removing δ-electron contamination "
                        "inside the road (Vogel §2.1.1).")
    p.add_argument("--train-all",    action="store_true",
                   help="Train on train+val; keep 5%% as internal mini-val for model selection.")
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--huber-delta",  type=float, default=0.2,
                   help="Huber loss delta in road units (default 0.2 ≈ 1 mm). "
                        "Smaller values (e.g. 0.05 ≈ 250 µm) sharpen core optimisation.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    detector_shift = load_detector_shift(OUT)
    if detector_shift != 0.0:
        print(f"[GNN] detector shift (phase 0): {detector_shift*1000:+.1f} um", flush=True)

    all_ev, all_a, all_b, splits = [], [], [], []
    for path_str in args.data:
        data_path = Path(path_str)
        print(f"[GNN] loading: {data_path.name}", flush=True)
        ev = load_events(data_path)
        a, b = learn_frame_transform(ev)
        print(f"[GNN]   frame transform: a={a:.6f}  b={b:+.4f}", flush=True)
        all_ev.append(ev); all_a.append(a); all_b.append(b)

        # Keep only events that have at least one strip inside the road.
        # Events with an empty road (δ-electron background, Vogel §2.1.1) produce
        # an anchor far from the true position; their residuals dominate RMSE while
        # leaving σ₆₈ unaffected.  Filtering here makes both metrics consistent and
        # aligns the GNN test set with XGBoost (which already skips road-empty events).
        valid_idx = filter_road_empty(ev, a, b,
                                      detector_shift_mm=detector_shift,
                                      road_mm=5.0)
        print(f"[GNN]   road-empty filter: {ev.n_events} → {len(valid_idx)} events "
              f"({100*(1-len(valid_idx)/ev.n_events):.1f}% removed)", flush=True)
        rng  = np.random.default_rng(args.seed)
        perm = rng.permutation(len(valid_idx))
        n_tr = int(0.70 * len(valid_idx))
        n_va = int(0.15 * len(valid_idx))

        if args.train_all:
            trva = valid_idx[perm[:n_tr + n_va]]
            n_mini = max(int(0.05 * len(trva)), 1)
            tr_idx = trva[n_mini:]
            va_idx = trva[:n_mini]   # 5% internal mini-val; test set never touched
        else:
            tr_idx = valid_idx[perm[:n_tr]]
            va_idx = valid_idx[perm[n_tr:n_tr + n_va]]
        te_idx = valid_idx[perm[n_tr + n_va:]]

        # Outlier cut on train/val only — test set keeps full distribution
        tr_idx = tr_idx[ev.n_hits[tr_idx] <= MAX_NHITS_TRAIN]
        va_idx = va_idx[ev.n_hits[va_idx] <= MAX_NHITS_TRAIN]
        splits.append({"train": tr_idx, "val": va_idx, "test": te_idx})
        print(f"[GNN]   train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}", flush=True)

    if len(all_ev) > 1:
        norm = Normalization.from_datasets(
            [(ev, sp["train"]) for ev, sp in zip(all_ev, splits)],
            theta_deg=args.theta)
        print(f"[GNN] multi-domain normalization (tmax=-1)", flush=True)
    else:
        norm = Normalization.from_arrays(
            all_ev[0], train_idx=splits[0]["train"],
            theta_deg=args.theta, tmax_ns=args.tmax)
    cluster_select = not args.no_cluster_select
    anchor_name = "TC-centroid" if args.tc_anchor else "median(x)"
    print(f"[GNN] anchor: {anchor_name}  theta={args.theta}deg  "
          f"cluster_select={cluster_select}", flush=True)

    domain_test = np.concatenate([
        np.full(len(sp["test"]), d, dtype=np.int32) for d, sp in enumerate(splits)
    ])

    def make_ds(split_key):
        return ConcatDataset([
            HitDataset(ev, sp[split_key], a, b, norm,
                       detector_shift_mm=detector_shift, tc_anchor=args.tc_anchor,
                       cluster_select=cluster_select)
            for ev, sp, a, b in zip(all_ev, splits, all_a, all_b)
        ])

    dl_tr = DataLoader(make_ds("train"), batch_size=args.batch_size,     shuffle=True,
                       collate_fn=collate_padded, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(make_ds("val"),   batch_size=2*args.batch_size,   shuffle=False,
                       collate_fn=collate_padded, num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(make_ds("test"),  batch_size=2*args.batch_size,   shuffle=False,
                       collate_fn=collate_padded, num_workers=args.num_workers, pin_memory=True)

    device = _resolve_device(args.device)
    print(f"[GNN] device = {device}", flush=True)
    model = StripModel(n_strip_feats=6, n_global_feats=7, d_model=args.d_model, n_heads=args.n_heads,
                       n_layers=args.n_layers, dropout=args.dropout).to(device)
    print(f"[GNN] parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs-10, 1))
    warmup  = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=10)
    sched   = torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], milestones=[10])
    def nll_loss(mu, log_sigma, target):
        # Gaussian NLL: 0.5 * [(y - mu)^2 / sigma^2 + log(sigma^2)]
        # log_sigma clamped to [-6, 4] to keep sigma in [0.002, 55] road-units
        log_sigma = log_sigma.clamp(-6.0, 4.0)
        return (0.5 * ((target - mu) ** 2 * torch.exp(-2 * log_sigma) + 2 * log_sigma)).mean()

    best_sigma68, best_state, best_ep = float("inf"), None, -1
    no_improve = 0
    history = []

    for ep in range(args.epochs):
        t0 = time.time()
        model.train(); tr_loss = 0.0; n = 0
        for batch in tqdm(dl_tr, desc=f"ep {ep:2d} train", ncols=100, leave=False):
            for k in ("strip_feats", "mask", "global_feats", "label_local"):
                batch[k] = batch[k].to(device, non_blocking=True)
            mu, log_sigma = model(batch["strip_feats"], batch["mask"], batch["global_feats"])
            loss = nll_loss(mu, log_sigma, batch["label_local"])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            bs = batch["strip_feats"].size(0)
            tr_loss += loss.item() * bs; n += bs
        tr_loss /= max(n, 1)

        model.eval(); va_loss = 0.0; n = 0
        sq = 0.0; sab = 0.0; sy = 0.0; syy = 0.0
        residuals_mm = []
        with torch.no_grad():
            for batch in tqdm(dl_va, desc=f"ep {ep:2d} val  ", ncols=100, leave=False):
                for k in ("strip_feats", "mask", "global_feats", "label_local", "x_med", "label_xpos"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                mu, log_sigma = model(batch["strip_feats"], batch["mask"], batch["global_feats"])
                va_loss  += nll_loss(mu, log_sigma, batch["label_local"]).item() * batch["strip_feats"].size(0)
                pred_xpos = mu * 5.0 + batch["x_med"]
                y_true    = batch["label_xpos"]
                err       = pred_xpos - y_true
                sq       += (err ** 2).sum().item()
                sab      += err.abs().sum().item()
                sy       += y_true.sum().item()
                syy      += (y_true ** 2).sum().item()
                n        += batch["strip_feats"].size(0)
                residuals_mm.append(err.cpu().numpy())
        va_loss /= max(n, 1)
        nn_      = max(n, 1)
        mse_um2  = (sq / nn_) * 1e6                       # mm² → µm²
        rms_um   = float(np.sqrt(sq / nn_)) * 1000.0      # RMSE [µm]
        mae_um   = (sab / nn_) * 1000.0
        ss_tot   = syy - sy * sy / nn_
        r2       = 1.0 - sq / ss_tot if ss_tot > 1e-12 else float("nan")
        res_all  = np.concatenate(residuals_mm)            # [mm]
        q16, q84 = np.percentile(res_all, [16, 84])
        sigma68_um = 0.5 * (q84 - q16) * 1000.0           # σ₆₈ [µm]
        sched.step()

        history.append((ep, tr_loss, va_loss, rms_um, mse_um2, mae_um, r2, sigma68_um))
        print(f"  ep {ep:2d}: train={tr_loss:.4f}  val={va_loss:.4f}  "
              f"σ₆₈={sigma68_um:.0f}µm  RMSE={rms_um:.0f}µm  "
              f"MAE={mae_um:.0f}µm  R²={r2:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        if sigma68_um < best_sigma68 - 0.5:
            best_sigma68, best_ep = sigma68_um, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"[GNN] early stopping at epoch {ep+1}", flush=True)
            break

    model.load_state_dict(best_state)
    print(f"[GNN] best epoch = {best_ep}  σ₆₈ = {best_sigma68:.0f}µm", flush=True)

    model.eval()
    preds_xpos, pred_sigmas, label_xpos, track_icepts = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dl_te, desc="test", ncols=100):
            for k in ("strip_feats", "mask", "global_feats", "x_med"):
                batch[k] = batch[k].to(device, non_blocking=True)
            mu, log_sigma = model(batch["strip_feats"], batch["mask"], batch["global_feats"])
            preds_xpos.append((mu * 5.0 + batch["x_med"]).cpu().numpy())
            pred_sigmas.append((torch.exp(log_sigma) * 5.0).cpu().numpy())  # sigma in mm
            label_xpos.append(batch["label_xpos"].numpy())
            track_icepts.append(batch["track_icept"].numpy())

    preds_xpos   = np.concatenate(preds_xpos)
    pred_sigmas  = np.concatenate(pred_sigmas)
    track_icepts = np.concatenate(track_icepts)
    _a = np.array(all_a); _b = np.array(all_b)
    preds_track  = preds_xpos * _a[domain_test] + _b[domain_test]

    npz_path  = OUT / f"{args.out_prefix}_predictions.npz"
    pt_path   = OUT / f"{args.out_prefix}_model.pt"
    norm_path = OUT / f"{args.out_prefix}_norm.json"
    np.savez_compressed(npz_path, y_pred=preds_track, y_true=track_icepts,
                        pred_sigma=pred_sigmas,
                        domain=domain_test, history=np.array(history))
    torch.save(model.state_dict(), pt_path)
    norm.save(norm_path)
    print(f"[GNN] saved: {npz_path.name}  {pt_path.name}  {norm_path.name}", flush=True)

if __name__ == "__main__":
    main()
