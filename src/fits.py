"""Double-Gaussian fit for residual distributions.

Convention follows Fabian Vogel's dissertation (Eq. 5.31/5.32):
  - core Gaussian captures the intrinsic detector resolution
  - tail Gaussian absorbs delta-ray contamination and reconstruction outliers
  - sigma_weighted = integral-weighted combination of both sigmas

All output values are in micrometers [um].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    sigma_core_um:     float   # width of the core Gaussian [um]
    sigma_tail_um:     float   # width of the tail Gaussian [um]
    sigma_weighted_um: float   # integral-weighted sigma (Eq. 5.32) [um]
    mu_core_um:        float   # mean of the core Gaussian [um]
    sigma_68_um:       float   # robust 68% half-width (model-free) [um]
    rms_um:            float   # RMS of all finite residuals [um]
    n_entries:         int


def _gauss(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)

def _double_gauss(x, ac, mc, sc, at, mt, st):
    return _gauss(x, ac, mc, sc) + _gauss(x, at, mt, st)


def fit_residuals(residuals_mm: np.ndarray,
                  fit_range_mm: float = 0.5,
                  bins: int = 240) -> FitResult:
    """Fit a double Gaussian to the residual distribution.

    sigma_68 is always computed model-free from quantiles — it is the
    primary figure-of-merit reported in the thesis.
    """
    r = residuals_mm[np.isfinite(residuals_mm)]
    rms_um = float(np.sqrt(np.mean(r ** 2))) * 1000.0
    s68 = 0.5 * (np.quantile(r, 0.84) - np.quantile(r, 0.16)) * 1000.0 if r.size > 0 else float("nan")

    r = r[np.abs(r) < fit_range_mm]
    counts, edges = np.histogram(r, bins=bins, range=(-fit_range_mm, fit_range_mm))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guess: core width from FWHM of histogram peak
    a0    = float(counts.max())
    above = centers[counts >= a0 / 2]
    s0    = max((above.max() - above.min()) / 2.355, 0.01) if above.size >= 2 else 0.05
    p0     = [a0, 0.0, s0,  0.2*a0, 0.0, max(np.std(r), 4*s0)]
    bounds = ([0, -fit_range_mm, 0.001, 0, -fit_range_mm, 0.02],
              [np.inf, fit_range_mm, fit_range_mm, np.inf, fit_range_mm, 5*fit_range_mm])
    try:
        popt, _ = curve_fit(_double_gauss, centers, counts, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        popt = p0   # fit failed — return initial guess rather than crashing

    ac, mc, sc, at, mt, st = popt
    if sc > st:   # enforce: core = narrower Gaussian
        ac, mc, sc, at, mt, st = at, mt, st, ac, mc, sc

    ic = ac * sc * np.sqrt(2 * np.pi)
    it = at * st * np.sqrt(2 * np.pi)
    sigma_w = (ic*sc + it*st) / (ic + it) * 1000.0 if (ic + it) > 0 else float("nan")

    return FitResult(
        sigma_core_um     = float(sc) * 1000.0,
        sigma_tail_um     = float(st) * 1000.0,
        sigma_weighted_um = float(sigma_w),
        mu_core_um        = float(mc) * 1000.0,
        sigma_68_um       = float(s68),
        rms_um            = rms_um,
        n_entries         = int(np.isfinite(residuals_mm).sum()),
    )
