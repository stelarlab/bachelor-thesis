"""Pro Event ein Feature-Vektor mit Cluster-Statistiken fuer XGBoost."""
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from data_loader import EventArrays, PITCH_MM, ROAD_MM, select_strips_in_road


# v4-FINAL Feature-Set (24 Features): aggregierte Cluster-Statistiken
# Strip-by-Strip (v5-v7) hat nicht geholfen -- v4 ist XGBoost-Grenze
#   Gegenueber v1 entfernt (Korrelation > 0.95):
#     q_right_frac, x_max, t_span, q_max
#   Gegenueber v2 entfernt (Fit-Artefakt):
#     tc_correction, tc_position
#   Neu hinzugefuegt:
#     q_asymmetry = (q_right - q_left) / q_sum
#     t_slope     = (t_last - t_first) / x_span
FEATURES = [
    "n_strips", "q_sum", "q_mean", "q_std", "q_min",
    "x_min", "x_span", "x_qmean_local",
    "q_centroid_offset", "q_left_frac", "q_asymmetry", "q_argmax_pos_local",
    "t_mean_qweighted", "t_min", "t_max",
    "t_first_strip", "t_last_strip", "t_argmin_pos_local", "t_slope",
    "track_slope", "non_prec", "n_hits_total", "n_clusters_event",
    "outer_q_sum_outside_cluster",
]


def build_features(ev: EventArrays, slope_frame: float, offset_frame: float,
                   detector_shift_mm: float = 0.0, road_mm: float = ROAD_MM):
    n = ev.n_events
    X = np.full((n, len(FEATURES)), np.nan)

    y = ev.track_icept.copy()
    mask = np.zeros(n, dtype=bool)
    cm_pred = np.full(n, np.nan)
    idx = {k: i for i, k in enumerate(FEATURES)}

    def to_xpos(track_y):
        return (track_y - offset_frame) / slope_frame

    for i in tqdm(range(n), ncols=80, desc="Features"):
        if ev.n_hits[i] == 0:
            continue
        xs = np.asarray(ev.hits_x[i], dtype=np.float64) - detector_shift_mm
        qs = np.asarray(ev.hits_q[i], dtype=np.float64)
        ts = np.asarray(ev.hits_t[i], dtype=np.float64)

        sub_x, sub_q, sub_t = select_strips_in_road(
            xs, qs, ts, to_xpos(ev.track_icept[i]), road_mm=road_mm)
        if sub_x.size == 0 or sub_q.sum() <= 0:
            continue

        outside = (xs < sub_x.min() - 1e-9) | (xs > sub_x.max() + 1e-9)
        order = np.argsort(xs)
        xs_s = xs[order]
        n_clusters = 1 + int((np.diff(xs_s) > 2 * PITCH_MM + 1e-6).sum())

        cm_xpos = (sub_x * sub_q).sum() / sub_q.sum()
        cm_pred[i] = slope_frame * cm_xpos + offset_frame

        x_min_v = float(sub_x.min())
        x_max_v = float(sub_x.max())
        q_sum = float(sub_q.sum())
        x_span_v = x_max_v - x_min_v
        cm_local = cm_xpos - x_min_v
        left = sub_x <= cm_xpos
        right = sub_x > cm_xpos
        q_left_v = float(sub_q[left].sum())
        q_right_v = float(sub_q[right].sum())

        t_cw = float((sub_t * sub_q).sum() / q_sum)
        t_first_v = float(sub_t[np.argmin(sub_x)])
        t_last_v  = float(sub_t[np.argmax(sub_x)])
        t_slope_v = (t_last_v - t_first_v) / x_span_v if x_span_v > 1e-9 else 0.0

        feats = {
            "n_strips": float(sub_x.size),
            "q_sum": q_sum, "q_mean": float(sub_q.mean()), "q_std": float(sub_q.std()),
            "q_min": float(sub_q.min()),
            "x_min": x_min_v, "x_span": x_span_v,
            "x_qmean_local": cm_local,
            "q_centroid_offset": cm_local - 0.5 * x_span_v,
            "q_left_frac": q_left_v / max(q_sum, 1e-9),
            "q_asymmetry": (q_right_v - q_left_v) / max(q_sum, 1e-9),
            "q_argmax_pos_local": float(sub_x[np.argmax(sub_q)] - x_min_v),
            "t_mean_qweighted": t_cw,
            "t_min": float(sub_t.min()), "t_max": float(sub_t.max()),
            "t_first_strip": t_first_v,
            "t_last_strip": t_last_v,
            "t_argmin_pos_local": float(sub_x[np.argmin(sub_t)] - x_min_v),
            "t_slope": t_slope_v,
            "track_slope": float(ev.track_slope[i]), "non_prec": float(ev.non_prec[i]),
            "n_hits_total": float(ev.n_hits[i]), "n_clusters_event": float(n_clusters),
            "outer_q_sum_outside_cluster": float(qs[outside].sum()),
        }
        for k, v in feats.items():
            X[i, idx[k]] = v

        mask[i] = True

    return X, y, mask, cm_pred
