"""Rung 6: resolving close pericenter passages in the hybrid integrator (nbody/hybrid.py)."""
import numpy as np
import pytest
import nbody.hybrid as hy
from nbody.orbits import orbital_elements
from nbody.units import G, M_EARTH, M_JUPITER

M_STAR = 0.816
MU = G * M_STAR
DT = 0.7**1.5 / np.sqrt(M_STAR) / 15                  # the Rung 6 timestep: 15 steps per inner-edge orbit


def embryo_and_giant(a, q):
    e = 1 - q / a
    x = np.array([[q, 0, 0], [6.1, 0, 0]])
    u = np.array([[0, np.sqrt(MU * (1 + e) / q), 0], [0, np.sqrt(MU / 6.1), 0]])
    return x, u, np.array([0.1 * M_EARTH, 2.5 * M_JUPITER])


def embryo_a_history(stepper, refine, a, q, years=300.0):
    """Embryo semi-major axis every 23 coarse steps (~1 yr), integrating with dt = DT / refine."""
    dt = DT / refine
    x, u, m = embryo_and_giant(a, q)
    Q, v = hy.from_heliocentric(x, u, m, M_STAR)
    alive, big = np.ones(2, bool), np.ones(2, bool)
    comp, log, R = np.column_stack([m, np.zeros((2, 3))]), np.zeros((4, 6)), np.full(2, 1e-9)
    rc = hy.critical_radii(Q, v, m, alive, MU, DT)
    out = []
    for k in range(int(years / DT) * refine):
        stepper(Q, v, m, R, alive, big, comp, rc, MU, M_STAR, dt, log, 0, 0.0)
        if k % (23 * refine) == 0:                    # identical sample times for every refine
            xh, uh = hy.to_heliocentric(Q, v, m, M_STAR)
            out.append(orbital_elements(xh[:1], uh[:1], MU)[0][0])
    return np.array(out)


def test_near_circular_orbits_never_trigger_substeps():
    vc = np.sqrt(MU / np.array([0.7, 1.0, 2.0]))
    Q = np.array([[0.7, 0, 0], [0, 1.0, 0], [-2.0, 0, 0]])
    v = np.array([[0, vc[0], 0], [-vc[1], 0, 0], [0, -vc[2], 0]])
    assert hy.pericenter_substeps(Q, v, np.ones(3, bool), MU, DT) == 1


@pytest.mark.parametrize("a, q", [(1.5, 0.2), (2.5, 0.3)])
def test_substeps_match_a_fine_timestep_reference(a, q):
    # An eccentric embryo whose pericenter passage is much faster than one step, perturbed by the 2.5 MJ giant.
    # Measured: plain steps drift by ~1-3% in a; with substeps the history matches dt/16 to ~1e-3.
    reference = embryo_a_history(hy.step, 16, a, q)
    plain = embryo_a_history(hy.step, 1, a, q)
    resolved = embryo_a_history(hy.step_resolving_pericenters, 1, a, q)
    error = lambda hist: np.abs(hist - reference).max() / a
    assert error(resolved) < 2e-3
    assert error(plain) > 5 * error(resolved)
