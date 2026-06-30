"""One feature vector of cluster statistics per event, for XGBoost."""
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from data_loader import EventArrays, PITCH_MM, ROAD_MM, V_DRIFT, TAN_THETA, select_strips_in_road, muTPC_clean

FEATURES = [
    "n_strips",
    "q_sum", "q_mean", "q_std",
    "x_span", "x_qmean_local", "q_centroid_offset",
    "q_left_frac", "q_asymmetry",
    "q_argmax_pos_local", "q_argmax_frac",
    "t_span", "t_cw_local", "tc_correction",
    "t_slope", "t_argmin_pos_local",
    "track_slope", "non_prec",
    "n_hits_total", "n_clusters_event",
    "outer_q_sum_outside_cluster",
    "muTPC_slope",
    "n_outlier_strips",
]


def _muTPC_slope(xs, ts):
    if len(xs) < 2:
        return 0.0
    if len(xs) == 2:
        dx = xs[1] - xs[0]
        return float((ts[1] - ts[0]) / dx) if abs(dx) > 1e-9 else 0.0
    A = np.stack([xs, np.ones(len(xs))], axis=1)
    slope, *_ = np.linalg.lstsq(A, ts, rcond=None)
    return float(slope[0])



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
        n_clusters = 1 + int((np.diff(xs[order]) > 2 * PITCH_MM + 1e-6).sum())

        q_sum = float(sub_q.sum())
        cm_xpos = (sub_x * sub_q).sum() / q_sum
        cm_pred[i] = slope_frame * cm_xpos + offset_frame

        x_min_v  = float(sub_x.min())
        x_span_v = float(sub_x.max()) - x_min_v
        cm_local = cm_xpos - x_min_v

        left  = sub_x <= cm_xpos
        right = sub_x > cm_xpos
        q_left_v  = float(sub_q[left].sum())
        q_right_v = float(sub_q[right].sum())

        order_x      = np.argsort(sub_x)
        clean_mask   = muTPC_clean(sub_x[order_x], sub_t[order_x])
        n_outlier_v  = float((~clean_mask).sum())
        cx = sub_x[order_x][clean_mask]
        cq = sub_q[order_x][clean_mask]
        ct = sub_t[order_x][clean_mask]
        if cx.size < 2:
            cx, cq, ct = sub_x[order_x], sub_q[order_x], sub_t[order_x]

        t_min_v      = float(ct.min())
        q_sum_c      = float(cq.sum()) if cq.sum() > 0 else q_sum
        t_cw         = float((ct * cq).sum() / q_sum_c)
        t_span_v     = float(ct.max()) - t_min_v
        t_cw_local_v = t_cw - t_min_v
        tc_corr_v    = t_cw_local_v * V_DRIFT * TAN_THETA
        muTPC_v      = _muTPC_slope(cx, ct)

        q_argmax_local = float(sub_x[np.argmax(sub_q)] - x_min_v)

        feats = {
            "n_strips":                    float(sub_x.size),
            "q_sum":                       q_sum,
            "q_mean":                      float(sub_q.mean()),
            "q_std":                       float(sub_q.std()),
            "x_span":                      x_span_v,
            "x_qmean_local":               cm_local,
            "q_centroid_offset":           cm_local - 0.5 * x_span_v,
            "q_left_frac":                 q_left_v / max(q_sum, 1e-9),
            "q_asymmetry":                 (q_right_v - q_left_v) / max(q_sum, 1e-9),
            "q_argmax_pos_local":          q_argmax_local,
            "q_argmax_frac":               q_argmax_local / x_span_v if x_span_v > 1e-9 else 0.5,
            "t_span":                      t_span_v,
            "t_cw_local":                  t_cw_local_v,
            "tc_correction":               tc_corr_v,
            "t_slope":                     muTPC_v,
            "t_argmin_pos_local":          float(cx[np.argmin(ct)] - x_min_v),
            "track_slope":                 float(ev.track_slope[i]),
            "non_prec":                    float(ev.non_prec[i]),
            "n_hits_total":                float(ev.n_hits[i]),
            "n_clusters_event":            float(n_clusters),
            "outer_q_sum_outside_cluster": float(qs[outside].sum()),
            "muTPC_slope":                 muTPC_v,
            "n_outlier_strips":            n_outlier_v,
        }
        for k, v in feats.items():
            X[i, idx[k]] = v
        mask[i] = True

    return X, y, mask, cm_pred
