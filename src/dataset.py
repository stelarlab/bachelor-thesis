"""PyTorch Dataset for strip-hit events from the ATLAS strip detector.

Strip features per hit (6):
  q_norm       — charge, normalized globally
  x_rel        — position relative to TC-anchor, in road units (/5 mm)
  x_rel_xmin   — position relative to cluster x_min (/5 mm); no absolute position info
  t_rel        — time relative to t_min of cluster, scaled by global t_std; no trigger offset
  z_norm       — drift distance, normalized globally
  x_corr_rel   — muTPC-corrected position (x - z*tan(θ)) relative to anchor (Vogel §5.4.1)

Global features per event (7):
  slope_norm, nonprec_norm, log1p(n_strips)
  q_sum_norm   — total cluster charge (Landau-distributed)
  q_mean_norm  — mean charge per strip
  q_std_norm   — charge spread (cluster shape)
  tc_corr_norm — TC correction value: t_cw_local * V_DRIFT * TAN_THETA (Vogel Gl. 5.40)

Label: (true_xpos - anchor) / 5.0
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import EventArrays, V_DRIFT, TAN_THETA, ROAD_MM, select_strips_in_road, select_cluster_near_track


@dataclass
class Normalization:
    # per-strip stats
    x_mean: float; x_std: float
    q_mean: float; q_std: float
    t_mean: float; t_std: float
    z_mean: float; z_std: float
    # per-event track stats
    slope_mean: float; slope_std: float
    nonprec_mean: float; nonprec_std: float
    # per-event cluster aggregate stats
    q_sum_mean: float; q_sum_std: float
    q_evmean_mean: float; q_evmean_std: float
    q_evstd_mean: float; q_evstd_std: float
    tc_mean: float; tc_std: float
    # metadata
    theta_deg: float = 29.0
    tmax_ns: float = 100.0

    @classmethod
    def from_arrays(cls, ev: EventArrays, train_idx=None,
                    theta_deg: float = 29.0, tmax_ns: float = 100.0) -> "Normalization":
        hx = ev.hits_x[train_idx] if train_idx is not None else ev.hits_x
        hq = ev.hits_q[train_idx] if train_idx is not None else ev.hits_q
        ht = ev.hits_t[train_idx] if train_idx is not None else ev.hits_t
        sl = ev.track_slope[train_idx] if train_idx is not None else ev.track_slope
        np_ = ev.non_prec[train_idx]   if train_idx is not None else ev.non_prec

        fx = np.asarray(ak.flatten(hx))
        fq = np.asarray(ak.flatten(hq))
        ft = np.asarray(ak.flatten(ht))
        fz = ft * V_DRIFT

        q_sum_ev  = np.asarray(ak.sum(hq, axis=1), dtype=np.float64)
        valid     = q_sum_ev > 0
        q_sum_ev  = q_sum_ev[valid]
        q_mean_ev = np.asarray(ak.mean(hq, axis=1), dtype=np.float64)[valid]
        q_std_ev  = np.asarray(ak.std(hq,  axis=1), dtype=np.float64)[valid]
        t_cw_ev   = np.asarray(ak.sum(ht * hq, axis=1), dtype=np.float64)[valid] / q_sum_ev
        t_min_ev  = np.asarray(ak.min(ht, axis=1),  dtype=np.float64)[valid]
        tc_ev     = (t_cw_ev - t_min_ev) * V_DRIFT * TAN_THETA

        return cls(
            x_mean=float(fx.mean()),  x_std=float(fx.std()  + 1e-9),
            q_mean=float(fq.mean()),  q_std=float(fq.std()  + 1e-9),
            t_mean=float(ft.mean()),  t_std=float(ft.std()  + 1e-9),
            z_mean=float(fz.mean()),  z_std=float(fz.std()  + 1e-9),
            slope_mean=float(sl.mean()),   slope_std=float(sl.std()   + 1e-9),
            nonprec_mean=float(np_.mean()), nonprec_std=float(np_.std() + 1e-9),
            q_sum_mean=float(q_sum_ev.mean()),   q_sum_std=float(q_sum_ev.std()   + 1e-9),
            q_evmean_mean=float(q_mean_ev.mean()), q_evmean_std=float(q_mean_ev.std() + 1e-9),
            q_evstd_mean=float(q_std_ev.mean()),   q_evstd_std=float(q_std_ev.std()   + 1e-9),
            tc_mean=float(tc_ev.mean()),  tc_std=float(tc_ev.std()  + 1e-9),
            theta_deg=float(theta_deg), tmax_ns=float(tmax_ns),
        )

    @classmethod
    def from_datasets(cls, datasets: list[tuple], theta_deg: float = 29.0) -> "Normalization":
        all_x, all_q, all_t, all_sl, all_np = [], [], [], [], []
        all_qs, all_qm, all_qstd, all_tc = [], [], [], []
        for ev, tr in datasets:
            hx = ev.hits_x[tr]; hq = ev.hits_q[tr]; ht = ev.hits_t[tr]
            all_x.append(np.asarray(ak.flatten(hx)))
            all_q.append(np.asarray(ak.flatten(hq)))
            all_t.append(np.asarray(ak.flatten(ht)))
            all_sl.append(ev.track_slope[tr])
            all_np.append(ev.non_prec[tr])
            q_sum_ev = np.asarray(ak.sum(hq, axis=1), dtype=np.float64)
            valid    = q_sum_ev > 0
            q_sum_ev = q_sum_ev[valid]
            t_cw_ev  = np.asarray(ak.sum(ht * hq, axis=1), dtype=np.float64)[valid] / q_sum_ev
            t_min_ev = np.asarray(ak.min(ht, axis=1), dtype=np.float64)[valid]
            all_qs.append(q_sum_ev)
            all_qm.append(np.asarray(ak.mean(hq, axis=1), dtype=np.float64)[valid])
            all_qstd.append(np.asarray(ak.std(hq, axis=1), dtype=np.float64)[valid])
            all_tc.append((t_cw_ev - t_min_ev) * V_DRIFT * TAN_THETA)

        fx=np.concatenate(all_x); fq=np.concatenate(all_q)
        ft=np.concatenate(all_t); fz=ft*V_DRIFT
        fs=np.concatenate(all_sl); fn=np.concatenate(all_np)
        qs=np.concatenate(all_qs); qm=np.concatenate(all_qm)
        qstd=np.concatenate(all_qstd); tc=np.concatenate(all_tc)

        return cls(
            x_mean=float(fx.mean()), x_std=float(fx.std()+1e-9),
            q_mean=float(fq.mean()), q_std=float(fq.std()+1e-9),
            t_mean=float(ft.mean()), t_std=float(ft.std()+1e-9),
            z_mean=float(fz.mean()), z_std=float(fz.std()+1e-9),
            slope_mean=float(fs.mean()), slope_std=float(fs.std()+1e-9),
            nonprec_mean=float(fn.mean()), nonprec_std=float(fn.std()+1e-9),
            q_sum_mean=float(qs.mean()),   q_sum_std=float(qs.std()+1e-9),
            q_evmean_mean=float(qm.mean()), q_evmean_std=float(qm.std()+1e-9),
            q_evstd_mean=float(qstd.mean()), q_evstd_std=float(qstd.std()+1e-9),
            tc_mean=float(tc.mean()), tc_std=float(tc.std()+1e-9),
            theta_deg=float(theta_deg), tmax_ns=-1.0,
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Normalization":
        return cls(**json.loads(Path(path).read_text()))


class HitDataset(Dataset):
    """One sample = one track with its associated strip hits inside the road.

    Args:
        tc_anchor:      use TC-centroid as anchor (default: median(x))
        cluster_select: apply within-road cluster isolation (default: True)
    """
    def __init__(self, ev: EventArrays, indices: np.ndarray,
                 slope_frame: float, offset_frame: float,
                 norm: Normalization, detector_shift_mm: float = 0.0,
                 road_mm: float = ROAD_MM, tc_anchor: bool = False,
                 cluster_select: bool = True):
        super().__init__()
        self.norm = norm
        self.tan_theta = math.tan(math.radians(norm.theta_deg))
        self.slope_frame  = slope_frame
        self.offset_frame = offset_frame
        self.tc_anchor    = tc_anchor
        self.cluster_select = cluster_select

        self.slope = ev.track_slope[indices].astype(np.float32)
        self.nonp  = ev.non_prec[indices].astype(np.float32)
        self.icept = ev.track_icept[indices].astype(np.float32)
        self.label_xpos = ((self.icept - offset_frame) / slope_frame).astype(np.float32)

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
        z = t * V_DRIFT

        q_sum = float(q.sum()) if q.sum() > 0 else 1.0
        x_cw  = float((x * q).sum() / q_sum)
        z_cw  = float((z * q).sum() / q_sum)

        if self.tc_anchor:
            anchor = x_cw - z_cw * self.tan_theta
        else:
            anchor = float(np.median(x))

        t_min  = float(t.min())
        t_cw   = float((t * q).sum() / q_sum)
        tc_val = (t_cw - t_min) * V_DRIFT * self.tan_theta   # Vogel Gl. 5.40

        # strip features — no absolute position, no trigger offset
        q_norm       = (q - self.norm.q_mean) / self.norm.q_std
        x_rel        = (x - anchor) / 5.0
        x_rel_xmin   = (x - float(x.min())) / 5.0
        t_rel        = (t - t_min) / self.norm.t_std
        z_norm       = (z - self.norm.z_mean) / self.norm.z_std
        x_corr_rel   = ((x - z * self.tan_theta) - anchor) / 5.0

        strip = np.stack([q_norm, x_rel, x_rel_xmin, t_rel, z_norm, x_corr_rel],
                         axis=1).astype(np.float32)

        glob = np.array([
            (self.slope[i] - self.norm.slope_mean)      / self.norm.slope_std,
            (self.nonp[i]  - self.norm.nonprec_mean)     / self.norm.nonprec_std,
            np.log1p(len(x)),
            (q_sum              - self.norm.q_sum_mean)     / self.norm.q_sum_std,
            (float(q.mean())    - self.norm.q_evmean_mean)  / self.norm.q_evmean_std,
            (float(q.std())     - self.norm.q_evstd_mean)   / self.norm.q_evstd_std,
            (tc_val             - self.norm.tc_mean)         / self.norm.tc_std,
        ], dtype=np.float32)

        label_local = (self.label_xpos[i] - anchor) / 5.0

        return dict(
            strip_feats  = torch.from_numpy(strip),
            global_feats = torch.from_numpy(glob),
            label_local  = torch.tensor(label_local, dtype=torch.float32),
            x_med        = torch.tensor(anchor,       dtype=torch.float32),
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
