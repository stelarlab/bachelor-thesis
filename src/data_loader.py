"""ROOT I/O, strip selection, frame transform."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from numpy.linalg import lstsq

PITCH_MM  = 0.425   # SM1 strip pitch, Vogel Diss. p. 35
V_DRIFT   = 0.047   # mm/ns, Ar:CO2:iC4H10 93:5:2
TAN_THETA = math.tan(math.radians(29.0))  # tan(29°) ≈ 0.5543
ROAD_MM   = 5.0     # strip selection window around track

@dataclass
class EventArrays:
    hits_x: ak.Array       # strip positions per event [mm]
    hits_q: ak.Array       # strip charges per event [ADC]
    hits_t: ak.Array       # strip times per event [ns]
    n_hits: np.ndarray     # number of strips per event
    track_icept: np.ndarray   # track extrapolation to layer 6, tracker frame [mm]
    track_slope: np.ndarray   # track slope (non-precision direction)
    non_prec: np.ndarray      # non-precision coordinate [mm]

    @property
    def n_events(self) -> int:
        return len(self.track_icept)

def load_events(path: Path | str, max_events: int | None = None) -> EventArrays:
    f = uproot.open(str(path))
    t = f["ana"]
    arr = t.arrays(library="ak") if max_events is None else t.arrays(entry_stop=max_events, library="ak")
    return EventArrays(
        hits_x=arr["out_xpos"],
        hits_q=arr["out_charge"],
        hits_t=arr["out_time"],
        n_hits=np.asarray(ak.num(arr["out_xpos"])),
        track_icept=np.asarray(arr["out_track_icept"]),
        track_slope=np.asarray(arr["out_track_slope"]),
        non_prec=np.asarray(arr["out_non_prec"]),
    )

def select_strips_in_road(xs, qs, ts, track_x, road_mm: float = ROAD_MM):
    # take all strips within ±road_mm of the track; avoids cluster-gap heuristics
    # and suppresses background hits from delta electrons / secondary particles
    if len(xs) == 0:
        return xs, qs, ts
    in_road = np.abs(xs - track_x) < road_mm
    return xs[in_road], qs[in_road], ts[in_road]

def detector_shift_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / "detector_shift.json"

def save_detector_shift(out_dir: Path | str, mu_shift_mm: float, **extra) -> Path:
    p = detector_shift_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mu_shift_mm": float(mu_shift_mm), **{k: float(v) for k, v in extra.items()}}
    p.write_text(json.dumps(payload, indent=2))
    return p

def load_detector_shift(out_dir: Path | str) -> float:
    # returns 0.0 if no shift file exists so the pipeline runs without Phase 0
    p = detector_shift_path(out_dir)
    if not p.exists():
        return 0.0
    return float(json.loads(p.read_text())["mu_shift_mm"])

def select_cluster_near_track(xs, qs, ts, track_x, gap_strips=2):
    # split hits into clusters (max one strip gap), pick the one closest to the track
    if len(xs) == 0:
        return xs, qs, ts
    order = np.argsort(xs)
    xs, qs, ts = xs[order], qs[order], ts[order]
    starts = [0] + [i for i in range(1, len(xs)) if xs[i] - xs[i - 1] > gap_strips * PITCH_MM + 1e-6] + [len(xs)]
    best_d, best = float("inf"), None
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        if qs[a:b].sum() <= 0:
            continue
        c = (xs[a:b] * qs[a:b]).sum() / qs[a:b].sum()
        if abs(c - track_x) < best_d:
            best_d, best = abs(c - track_x), (a, b)
    if best is None:
        return xs[:0], qs[:0], ts[:0]
    a, b = best
    return xs[a:b], qs[a:b], ts[a:b]

def concat_events(*evs: "EventArrays") -> "EventArrays":
    return EventArrays(
        hits_x=ak.concatenate([e.hits_x for e in evs]),
        hits_q=ak.concatenate([e.hits_q for e in evs]),
        hits_t=ak.concatenate([e.hits_t for e in evs]),
        n_hits=np.concatenate([e.n_hits for e in evs]),
        track_icept=np.concatenate([e.track_icept for e in evs]),
        track_slope=np.concatenate([e.track_slope for e in evs]),
        non_prec=np.concatenate([e.non_prec for e in evs]),
    )

def time_corrected_centroid(xs, qs, ts, tan_theta: float = TAN_THETA):
    # Vogel Diss. Eq. 5.39-5.40: x_tc = x_CW + (t_ref - t_CW) * v_drift * tan(θ)
    q_sum = qs.sum()
    if q_sum <= 0:
        return float(np.nanmean(xs))
    x_cm  = float((xs * qs).sum() / q_sum)
    t_cw  = float((ts * qs).sum() / q_sum)
    t_ref = float(ts.mean())
    delta_t = t_ref - t_cw
    return x_cm + delta_t * V_DRIFT * tan_theta

def learn_frame_transform(ev: EventArrays) -> tuple[float, float]:
    # out_xpos (module frame) and out_track_icept (tracker frame) are related by
    # track_icept = a * out_xpos + b; fit on compact 3-8 strip events with MAD trimming
    y_naive = np.full(ev.n_events, np.nan)
    for i in range(ev.n_events):
        if ev.n_hits[i] == 0:
            continue
        xs = np.asarray(ev.hits_x[i]); qs = np.asarray(ev.hits_q[i])
        if qs.sum() > 0:
            y_naive[i] = (xs * qs).sum() / qs.sum()

    keep = np.zeros(ev.n_events, dtype=bool)
    for i in range(ev.n_events):
        if 3 <= ev.n_hits[i] <= 8:
            xs = np.asarray(ev.hits_x[i])
            if xs.max() - xs.min() < 5.0:
                keep[i] = True
    keep &= np.isfinite(y_naive)

    A = np.stack([y_naive[keep], np.ones(keep.sum())], axis=1)
    y = ev.track_icept[keep]
    coef, *_ = lstsq(A, y, rcond=None)
    resid = y - A @ coef
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep2 = np.abs(resid - np.median(resid)) < 5 * 1.4826 * mad
    coef, *_ = lstsq(A[keep2], y[keep2], rcond=None)
    return float(coef[0]), float(coef[1])
