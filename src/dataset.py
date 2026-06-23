"""PyTorch Dataset for strip-hit events from the ATLAS strip detector.

Each event contains a variable number of strip hits (x, charge, time) inside
a road around the track. The dataset handles road selection, normalization, and
the choice of anchor for the local prediction target.

Two anchor modes:
  - median(x)          : simple positional median, fast and robust
  - TC-centroid (x_tc) : charge-weighted, drift-corrected centroid
                         x_tc = Σ(x_i * q_i)/Σq_i  -  Σ(z_i * q_i)/Σq_i * tan(θ)
                         empirically better; used for all final models

Strip features per hit: [x_norm, q_norm, t_norm, x_rel, z_norm]  →  5 features
Global features per event: [slope_norm, nonprec_norm, log1p(n_strips)]
Label: (true_xpos - anchor) / 5.0  — local offset in road units (~mm)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import EventArrays, V_DRIFT, ROAD_MM, select_strips_in_road, select_cluster_near_track


@dataclass
class Normalization:
    x_mean: float; x_std: float
    q_mean: float; q_std: float
    t_mean: float; t_std: float
    z_mean: float; z_std: float        # z = v_drift * t  [mm]
    slope_mean: float; slope_std: float
    nonprec_mean: float; nonprec_std: float
    theta_deg: float = 29.0            # track incidence angle
    tmax_ns: float = 100.0             # shaping time of the dataset; -1.0 for multi-domain

    @classmethod
    def from_arrays(cls, ev: EventArrays, train_idx=None,
                    theta_deg: float = 29.0, tmax_ns: float = 100.0) -> "Normalization":
        """Compute normalization statistics from training hits only (no test leakage)."""
        hx = ev.hits_x[train_idx] if train_idx is not None else ev.hits_x
        hq = ev.hits_q[train_idx] if train_idx is not None else ev.hits_q
        ht = ev.hits_t[train_idx] if train_idx is not None else ev.hits_t
        sl = ev.track_slope[train_idx] if train_idx is not None else ev.track_slope
        np_ = ev.non_prec[train_idx]   if train_idx is not None else ev.non_prec
        fx = np.asarray(ak.flatten(hx))
        fq = np.asarray(ak.flatten(hq))
        ft = np.asarray(ak.flatten(ht))
        fz = ft * V_DRIFT
        return cls(
            x_mean=float(fx.mean()),  x_std=float(fx.std()  + 1e-9),
            q_mean=float(fq.mean()),  q_std=float(fq.std()  + 1e-9),
            t_mean=float(ft.mean()),  t_std=float(ft.std()  + 1e-9),
            z_mean=float(fz.mean()),  z_std=float(fz.std()  + 1e-9),
            slope_mean=float(sl.mean()),  slope_std=float(sl.std()  + 1e-9),
            nonprec_mean=float(np_.mean()), nonprec_std=float(np_.std() + 1e-9),
            theta_deg=float(theta_deg), tmax_ns=float(tmax_ns),
        )

    @classmethod
    def from_datasets(cls, datasets: list[tuple], theta_deg: float = 29.0) -> "Normalization":
        """Pool statistics across multiple domains for multi-domain training."""
        all_x, all_q, all_t, all_sl, all_np = [], [], [], [], []
        for ev, tr in datasets:
            all_x.append(np.asarray(ak.flatten(ev.hits_x[tr])))
            all_q.append(np.asarray(ak.flatten(ev.hits_q[tr])))
            all_t.append(np.asarray(ak.flatten(ev.hits_t[tr])))
            all_sl.append(ev.track_slope[tr])
            all_np.append(ev.non_prec[tr])
        fx=np.concatenate(all_x); fq=np.concatenate(all_q)
        ft=np.concatenate(all_t); fz=ft*V_DRIFT
        fs=np.concatenate(all_sl); fn=np.concatenate(all_np)
        return cls(
            x_mean=float(fx.mean()), x_std=float(fx.std()+1e-9),
            q_mean=float(fq.mean()), q_std=float(fq.std()+1e-9),
            t_mean=float(ft.mean()), t_std=float(ft.std()+1e-9),
            z_mean=float(fz.mean()), z_std=float(fz.std()+1e-9),
            slope_mean=float(fs.mean()), slope_std=float(fs.std()+1e-9),
            nonprec_mean=float(fn.mean()), nonprec_std=float(fn.std()+1e-9),
            theta_deg=float(theta_deg), tmax_ns=-1.0,   # -1 = multi-domain
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Normalization":
        return cls(**json.loads(Path(path).read_text()))


class HitDataset(Dataset):
    """One sample = one track with its associated strip hits inside the road.

    Args:
        tc_anchor:       if True, use TC-centroid as anchor for x_rel and label_local.
                         Default False → median(x). TC-anchor is used in all final models.
        cluster_select:  if True (default), apply within-road cluster isolation:
                         splits road strips by gaps > 2×pitch and keeps only the cluster
                         whose charge-weighted centroid is closest to the track.
                         Removes δ-electron contamination inside the road (Vogel §2.1.1).
    """
    def __init__(self, ev: EventArrays, indices: np.ndarray,
                 slope_frame: float, offset_frame: float,
                 norm: Normalization, detector_shift_mm: float = 0.0,
                 road_mm: float = ROAD_MM, tc_anchor: bool = False,
                 cluster_select: bool = True):
        super().__init__()
        self.norm = norm
        self.tan_theta = math.tan(math.radians(norm.theta_deg))
        self.slope_frame    = slope_frame
        self.offset_frame   = offset_frame
        self.tc_anchor      = tc_anchor
        self.cluster_select = cluster_select

        self.slope = ev.track_slope[indices].astype(np.float32)
        self.nonp  = ev.non_prec[indices].astype(np.float32)
        self.icept = ev.track_icept[indices].astype(np.float32)
        self.label_xpos = ((self.icept - offset_frame) / slope_frame).astype(np.float32)

        # Strip selection: two-stage pipeline.
        # Stage 1 — road window: keep only strips within ±road_mm of the track.
        #   Removes δ-electron hits that landed far from the primary cluster.
        #   Caller must have applied filter_road_empty first — every sx is non-empty.
        # Stage 2 — cluster isolation (cluster_select=True, default):
        #   Within the road, multiple disconnected clusters may still exist when a
        #   δ-electron lands 1–4 mm from the primary cluster (Vogel §2.1.1).
        #   select_cluster_near_track splits by gaps > 2×pitch and keeps only the
        #   cluster whose charge-weighted centroid is closest to the track.
        #   This is the single most effective way to reduce TC-anchor contamination
        #   from within-road δ-electrons, improving σ₆₈ by ~10–30 µm.
        self.hits_x, self.hits_q, self.hits_t = [], [], []
        for i in indices:
            xs = np.asarray(ev.hits_x[i], dtype=np.float32) - detector_shift_mm
            qs = np.asarray(ev.hits_q[i], dtype=np.float32)
            ts = np.asarray(ev.hits_t[i], dtype=np.float32)
            track_x = float((ev.track_icept[i] - offset_frame) / slope_frame)
            sx, sq, st = select_strips_in_road(xs, qs, ts, track_x, road_mm=road_mm)
            if self.cluster_select and sx.size > 1:
                sx, sq, st = select_cluster_near_track(sx, sq, st, track_x)
            self.hits_x.append(sx)
            self.hits_q.append(sq)
            self.hits_t.append(st)

    def __len__(self):
        return len(self.icept)

    def __getitem__(self, i):
        x, q, t = self.hits_x[i], self.hits_q[i], self.hits_t[i]
        z = t * V_DRIFT   # drift distance [mm]

        if self.tc_anchor:
            # TC-centroid: charge-weighted mean corrected for drift direction
            # x_tc = x_CW - z_CW * tan(θ)  →  projects strips onto the track axis
            q_sum = q.sum()
            x_cw  = float((x * q).sum() / q_sum) if q_sum > 0 else float(x.mean())
            z_cw  = float((z * q).sum() / q_sum) if q_sum > 0 else float(z.mean())
            anchor = x_cw - z_cw * self.tan_theta
        else:
            anchor = float(np.median(x))

        x_norm = (x - self.norm.x_mean) / self.norm.x_std
        q_norm = (q - self.norm.q_mean) / self.norm.q_std
        t_norm = (t - self.norm.t_mean) / self.norm.t_std
        z_norm = (z - self.norm.z_mean) / self.norm.z_std
        x_rel  = (x - anchor).astype(np.float32) / 5.0
        # per-strip muTPC-corrected position projected onto pad plane (Vogel §5.4.1)
        x_corr_rel = ((x - z * self.tan_theta) - anchor).astype(np.float32) / 5.0

        strip = np.stack([x_norm, q_norm, t_norm, x_rel, z_norm, x_corr_rel], axis=1).astype(np.float32)
        glob  = np.array([
            (self.slope[i]  - self.norm.slope_mean)   / self.norm.slope_std,
            (self.nonp[i]   - self.norm.nonprec_mean) / self.norm.nonprec_std,
            np.log1p(len(x)),   # log-compressed strip count → roughly Gaussian
        ], dtype=np.float32)

        label_local = (self.label_xpos[i] - anchor) / 5.0   # consistent with x_rel

        return dict(
            strip_feats  = torch.from_numpy(strip),
            global_feats = torch.from_numpy(glob),
            label_local  = torch.tensor(label_local, dtype=torch.float32),
            x_med        = torch.tensor(anchor,      dtype=torch.float32),
            label_xpos   = torch.tensor(self.label_xpos[i]),
            track_icept  = torch.tensor(self.icept[i]),
        )


def collate_padded(batch: list[dict]) -> dict:
    """Pad variable-length strip sequences to the longest in the batch."""
    B = len(batch)
    N = max(b["strip_feats"].shape[0] for b in batch)
    F = batch[0]["strip_feats"].shape[1]
    feats = torch.zeros(B, N, F)
    mask  = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["strip_feats"].shape[0]
        feats[i, :n] = b["strip_feats"]
        mask[i, :n]  = True
    return dict(
        strip_feats  = feats,
        mask         = mask,
        global_feats = torch.stack([b["global_feats"] for b in batch]),
        label_local  = torch.stack([b["label_local"]  for b in batch]),
        x_med        = torch.stack([b["x_med"]        for b in batch]),
        label_xpos   = torch.stack([b["label_xpos"]   for b in batch]),
        track_icept  = torch.stack([b["track_icept"]  for b in batch]),
    )
