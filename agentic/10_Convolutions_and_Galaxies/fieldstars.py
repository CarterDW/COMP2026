"""Find compact sources off-center in the Galaxy Zoo cutouts and fit each with a
stretched Moffat:  mean(v) = A * asinh(F * M(r; fwhm, beta) + b),  M of unit flux.
"""
import numpy as np
from scipy import ndimage, optimize

import sims

HALF = 10     # fit stamps are 21 x 21


def candidates(v, min_peak=40, r_min=28):
    """Local maxima off the central galaxy's footprint, away from the edges."""
    g = v.astype(float).mean(axis=-1)
    sky = np.median(g)
    peaks = (g == ndimage.maximum_filter(g, 9)) & (g > sky + min_peak)
    lab, _ = ndimage.label(sims.object_mask(v, grow=1))
    central = lab[60, 60]
    yy, xx = np.nonzero(peaks)
    keep = [(y, x) for y, x in zip(yy, xx)
            if HALF <= y < sims.N - HALF and HALF <= x < sims.N - HALF
            and np.hypot(y - 59.5, x - 59.5) > r_min
            and (central == 0 or lab[y, x] != central)]
    return keep


def stamp(v, y, x):
    return v[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1].astype(float)


def isolated_and_round(v, y, x, max_border=12.0, max_ell=0.15):
    """The stamp's border is at sky level (not inside a big galaxy), and the source is
    round by its second moments within r < 4 px."""
    g = stamp(v, y, x).mean(axis=-1)
    border = np.r_[g[0], g[-1], g[:, 0], g[:, -1]]
    if np.median(border) > max_border:
        return False
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1].astype(float)
    w = np.clip(g - np.median(border), 0, None) * (xx**2 + yy**2 < 16)
    mxx, myy, mxy = (w * xx * xx).sum(), (w * yy * yy).sum(), (w * xx * yy).sum()
    ell = np.hypot(mxx - myy, 2 * mxy) / (mxx + myy)
    return ell < max_ell


def model(p, yy, xx):
    A, logF, b, x0, y0, fwhm, beta = p
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    M = (beta - 1) / (np.pi * a**2) * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a**2) ** (-beta)
    return A * np.arcsinh(np.exp(logF) * M + b)


def fit(v, y, x, A=None, beta=None):
    """Fit one stamp. A and/or beta, when given, are held fixed (global parameters).
    Returns the parameters, the chi^2 sum (pixel units), the rms, and the peak."""
    g = stamp(v, y, x).mean(axis=-1)
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1].astype(float)
    use = (g > 0) & (g < 250)
    p0 = [30.0, np.log(50.0), 0.1, 0.0, 0.0, 3.0, 3.0]
    lo = [1.0, -5, -1, -3, -3, 0.8, 1.2]
    hi = [500.0, 15, 5, 3, 3, 15, 20]
    for k, val in [(0, A), (6, beta)]:
        if val is not None:
            p0[k], lo[k], hi[k] = val, val * (1 - 1e-9), val * (1 + 1e-9)
    res = optimize.least_squares(lambda p: (model(p, yy, xx) - g)[use], p0, bounds=(lo, hi))
    return res.x, np.sum(res.fun**2), np.sqrt(np.mean(res.fun**2)), g.max()
