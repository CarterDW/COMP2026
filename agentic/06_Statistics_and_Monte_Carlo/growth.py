"""The linear growth factor D(z) in flat LCDM, with radiation.

Sub-horizon matter perturbations obey, in ln a,

    D'' + (2 + d ln H / d ln a) D' = (3/2) Om(a) D,

where Om(a) is the matter fraction of the total density at a. Radiation adds to
H but does not cluster on these scales, which is what slows growth before
equality. Deep in the radiation era the growing mode is Meszaros's
D = 1 + 3y/2, with y = a / a_eq, and that sets the initial conditions.

Radiation: photons plus three massless neutrinos, omega_r = 4.18e-5.
"""

import numpy as np
from scipy.integrate import solve_ivp

import cosmo

OMEGA_R_H2 = 4.18e-5


def _E2(a, Om, Or):
    """H^2 / H0^2 for flat LCDM with radiation."""
    return Or / a**4 + Om / a**3 + (1 - Om - Or)


def growth(z, Om=0.31, radiation=True, a_start=1e-7):
    """D at redshifts z (array), normalized to D(z=0) = 1."""
    Or = OMEGA_R_H2 / cosmo.H**2 if radiation else 0.0
    a_eq = Or / Om
    y0 = a_start / a_eq if radiation else np.inf

    def rhs(lna, u):
        a = np.exp(lna)
        E2 = _E2(a, Om, Or)
        dlnH = -0.5 * (4 * Or / a**4 + 3 * Om / a**3) / E2
        Om_a = Om / a**3 / E2
        return [u[1], -(2 + dlnH) * u[1] + 1.5 * Om_a * u[0]]

    # Meszaros growing mode, D = 1 + 3y/2, or D = a in pure matter domination.
    u0 = [1 + 1.5 * y0, 1.5 * y0] if radiation else [a_start, a_start]
    a = 1 / (1 + np.atleast_1d(np.asarray(z, float)))
    # Solve once on the sorted, de-duplicated times (z = 0 coincides with the a = 1
    # normalization point), then map back to the caller's order.
    lna, inv = np.unique(np.log(np.concatenate([a, [1.0]])), return_inverse=True)
    sol = solve_ivp(rhs, (np.log(a_start), 0.0), u0, t_eval=lna, rtol=1e-10, atol=1e-30)
    D = sol.y[0][inv]
    return D[:-1] / D[-1]
