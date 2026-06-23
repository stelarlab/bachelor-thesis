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

PITCH_MM  = 0.425    # SM1 strip pitch [mm], Vogel §2.3.2
V_DRIFT   = 0.04630  # electron drift velocity [mm/ns], Ar:CO₂:iC₄H₁₀ 93:5:2,
                     # Vogel §4.4.2: vD = 4.63 cm/µs from 5 mm gap / 108 ns box width
                     # = 4.63 * 10 mm/cm / 1000 ns/µs = 0.04630 mm/ns exactly
TAN_THETA = math.tan(math.radians(29.0))  # tan(29°) ≈ 0.5543, Vogel §7.2.3
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

def filter_road_empty(ev: "EventArrays", slope_frame: float, offset_frame: float,
                      detector_shift_mm: float = 0.0,
                      road_mm: float = ROAD_MM) -> np.ndarray:
    """Return indices of events that have at least one strip inside the road.

    Events with no strip in the ±road_mm window around the track extrapolation
    cause the TC-anchor to be computed from unrelated strips, producing residuals
    of 10–300 mm that dominate RMSE while σ₆₈ stays unaffected.  These events
    are physically real (δ-electron background, Vogel §2.1.1) but cannot be
    reconstructed — they should be excluded from training *and* evaluation so
    that RMSE reflects true model quality.
    """
    keep = []
    for i in range(ev.n_events):
        if ev.n_hits[i] == 0:
            continue
        xs      = np.asarray(ev.hits_x[i], dtype=np.float32) - detector_shift_mm
        track_x = float((ev.track_icept[i] - offset_frame) / slope_frame)
        if np.any(np.abs(xs - track_x) < road_mm):
            keep.append(i)
    return np.array(keep, dtype=np.int64)

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

def muTPC_clean(xs, ts, n_sigma=3.0):
    """Iterative strip rejection: remove strips > n_sigma from linear muTPC t(x) fit.

    xs must be sorted by position. Returns boolean mask (True = keep).
    Removes δ-electron strips that contaminate timing features within the cluster
    (Vogel §2.1.1). Used by both HitDataset and build_features."""
    mask = np.ones(len(xs), dtype=bool)
    for _ in range(3):
        if mask.sum() < 2:
            break
        slope, intercept = np.polyfit(xs[mask], ts[mask], 1)
        resid = np.abs(ts - (slope * xs + intercept))
        sigma = resid[mask].std()
        if sigma < 1e-9:
            break
        mask = resid < n_sigma * sigma
    return mask


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
