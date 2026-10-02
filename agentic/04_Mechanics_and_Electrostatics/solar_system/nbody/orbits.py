"""Analytic two-body results used as references for the simulations."""
import numpy as np
from nbody.units import G


def kepler_period(a, m, M=1.0):
    """Two-body period [yr] for semi-major axis a [AU], masses m and M [Msun]."""
    return 2 * np.pi * np.sqrt(a**3 / (G * (M + m)))
