"""Exact two-body motion: advance (x, v) along its conic for a time dt, using universal variables.

With s the universal anomaly, beta = 2 mu / r0 - v0^2 and the Stumpff-type functions G_k(beta, s),
    r(s) = r0 G0 + eta0 G1 + mu G2,     t(s) = r0 G1 + eta0 G2 + mu G3,     eta0 = x0 . v0,
we solve t(s) = dt for s, then use the f and g functions
    x = f x0 + g v0,   v = fdot x0 + gdot v0,
    f = 1 - mu G2 / r0,  g = dt - mu G3,  fdot = -mu G1 / (r0 r),  gdot = 1 - mu G2 / r.
Works for elliptic (beta > 0), parabolic and hyperbolic (beta < 0) orbits.

Since dt/ds = r > 0, t(s) is monotonic: the root is bracketed and found by Newton steps safeguarded with bisection,
which cannot fail even from a poor first guess (e.g. at the pericenter of a very eccentric orbit). For elliptic
orbits dt is first reduced modulo the period, which is exact.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _stumpff(z):
    """c0..c3 of z = beta s^2: c0 = cos sqrt z, c1 = sin sqrt z / sqrt z, c2 = (1 - c0)/z, c3 = (1 - c1)/z."""
    if abs(z) < 0.1:                                  # series: c_k = sum (-z)^n / (2n + k)!
        c3 = (1 - z / 20 * (1 - z / 42 * (1 - z / 72 * (1 - z / 110 * (1 - z / 156))))) / 6
        c2 = (1 - z / 12 * (1 - z / 30 * (1 - z / 56 * (1 - z / 90 * (1 - z / 132))))) / 2
    elif z > 0:
        sz = np.sqrt(z)
        c2 = (1 - np.cos(sz)) / z
        c3 = (sz - np.sin(sz)) / (z * sz)
    else:
        sz = np.sqrt(-z)
        c2 = (np.cosh(sz) - 1) / (-z)
        c3 = (np.sinh(sz) - sz) / (-z * sz)
    c1 = 1 - z * c3
    c0 = 1 - z * c2
    return c0, c1, c2, c3


@njit(cache=True)
def kepler_drift(x, v, mu, dt):
    """Return (x, v) after time dt on the Kepler orbit about a fixed mass with G M = mu."""
    r0 = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    eta0 = x[0] * v[0] + x[1] * v[1] + x[2] * v[2]
    beta = 2 * mu / r0 - (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    t_target = dt
    if beta > 0:                                      # elliptic: one period is exactly s = 2 pi / sqrt(beta)
        period = 2 * np.pi * mu / beta**1.5
        t_target = dt - period * np.floor(dt / period + 0.5)        # in [-P/2, P/2)
    sign = 1.0 if t_target >= 0 else -1.0             # solve for |t| and restore the sign (t(-s) = -t(s) with eta0 -> -eta0)
    target, eta = abs(t_target), sign * eta0

    def t_of(s):
        c0, c1, c2, c3 = _stumpff(beta * s * s)
        return r0 * s * c1 + eta * s * s * c2 + mu * s * s * s * c3, r0 * c0 + eta * s * c1 + mu * s * s * c2

    lo, hi = 0.0, target / r0                          # t(0) = 0 <= target; grow hi until t(hi) >= target
    if beta > 0:
        hi = min(hi, 2 * np.pi / np.sqrt(beta))       # t at one full period exceeds |t_target| <= P/2
    while t_of(hi)[0] < target:
        hi *= 2.0
    s = target / r0 - eta * target**2 / (2 * r0**3)  # series guess (good when dt << orbital period) ...
    if not (lo < s < hi):
        s = 0.5 * (lo + hi)                           # ... else start mid-bracket
    for _ in range(200):
        t_s, r_s = t_of(s)
        if t_s < target:
            lo = s
        else:
            hi = s
        s_new = s - (t_s - target) / r_s               # Newton step (r_s = dt/ds > 0)
        if not (lo < s_new < hi):
            s_new = 0.5 * (lo + hi)                    # fall back to bisection
        converged = abs(s_new - s) <= 1e-13 * abs(s_new) or hi - lo <= 1e-15 * hi
        s = s_new
        if converged:                                  # one more Newton step: quadratic convergence takes the
            t_s, r_s = t_of(s)                         # 1e-13 iterate to machine precision (a tighter stopping
            s -= (t_s - target) / r_s                  # test would just bisect roundoff noise)
            break
    s *= sign
    c0, c1, c2, c3 = _stumpff(beta * s * s)
    G1, G2, G3 = s * c1, s * s * c2, s * s * s * c3
    r = r0 * c0 + eta0 * G1 + mu * G2
    f = 1 - mu * G2 / r0
    g = t_target - mu * G3                           # s solves the period-reduced time
    fdot = -mu * G1 / (r0 * r)
    gdot = 1 - mu * G2 / r
    return f * x + g * v, fdot * x + gdot * v
