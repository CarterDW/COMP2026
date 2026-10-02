"""Analytic two-body results used as references for the simulations."""
import numpy as np
from nbody.units import G


def kepler_period(a, m, M=1.0):
    """Two-body period [yr] for semi-major axis a [AU], masses m and M [Msun]."""
    return 2 * np.pi * np.sqrt(a**3 / (G * (M + m)))


def two_body_at_pericenter(m1, m2, a, e):
    """Positions and velocities (2, 3) of a bound pair at pericenter, in its center-of-mass frame.

    The orbit lies in the xy-plane, with pericenter along +x for body 2.
    """
    M = m1 + m2
    r_p = a * (1 - e)
    v_p = np.sqrt(G * M * (1 + e) / (a * (1 - e)))   # vis-viva at pericenter
    rel_pos, rel_vel = np.array([r_p, 0, 0]), np.array([0, v_p, 0])
    pos = np.array([-m2 / M * rel_pos, m1 / M * rel_pos])
    vel = np.array([-m2 / M * rel_vel, m1 / M * rel_vel])
    return pos, vel


FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight():
    """Chenciner-Montgomery three-body choreography: three equal masses chasing each other on a figure-eight.

    Published with G m = 1, so in code units each mass is 1/G. Returns pos, vel (3, 3), mass (3,);
    the orbit repeats after FIGURE_EIGHT_PERIOD.
    """
    x1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    pos = np.array([x1, -x1, np.zeros(3)])
    vel = np.array([-v3 / 2, -v3 / 2, v3])
    return pos, vel, np.full(3, 1 / G)


def orbital_elements(x, u, mu):
    """Semi-major axis, eccentricity and inclination (rad) from heliocentric positions/velocities, shape (N, 3)."""
    r = np.linalg.norm(x, axis=1)
    energy = 0.5 * np.sum(u**2, axis=1) - mu / r
    a = -mu / (2 * energy)
    L = np.cross(x, u)
    e_vec = np.cross(u, L) / mu - x / r[:, None]
    inc = np.arccos(np.clip(L[:, 2] / np.linalg.norm(L, axis=1), -1, 1))
    return a, np.linalg.norm(e_vec, axis=1), inc
