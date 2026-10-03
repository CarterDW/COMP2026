"""Rung 6: resolving close pericenter passages in the hybrid integrator (per-body substeps, nbody/hybrid.py)."""
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


def plain(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t_now, acc):
    """hy.step with the force-cache argument of step_resolving_pericenters (ignored: step evaluates its own forces)."""
    return hy.step(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t_now)


resolving = hy.step_resolving_pericenters


def embryo_a_history(stepper, refine, a, q, years=300.0):
    """Embryo semi-major axis every 23 coarse steps (~1 yr), integrating with dt = DT / refine."""
    dt = DT / refine
    x, u, m = embryo_and_giant(a, q)
    Q, v = hy.from_heliocentric(x, u, m, M_STAR)
    alive, big = np.ones(2, bool), np.ones(2, bool)
    comp, log, R = np.column_stack([m, np.zeros((2, 3))]), np.zeros((4, 6)), np.full(2, 1e-9)
    rc = hy.critical_radii(Q, v, m, alive, MU, DT)
    acc = hy.far_accelerations(Q, m, alive, big, rc)
    out = []
    for k in range(int(years / DT) * refine):
        stepper(Q, v, m, R, alive, big, comp, rc, MU, M_STAR, dt, log, 0, 0.0, acc)
        if k % (23 * refine) == 0:                    # identical sample times for every refine
            xh, uh = hy.to_heliocentric(Q, v, m, M_STAR)
            out.append(orbital_elements(xh[:1], uh[:1], MU)[0][0])
    return np.array(out)


def test_near_circular_orbits_never_trigger_substeps():
    vc = np.sqrt(MU / np.array([0.7, 1.0, 2.0]))
    Q = np.array([[0.7, 0, 0], [0, 1.0, 0], [-2.0, 0, 0]])
    v = np.array([[0, vc[0], 0], [-vc[1], 0, 0], [0, -vc[2], 0]])
    assert (hy.pericenter_substeps(Q, v, np.ones(3, bool), MU, DT) == 1).all()


def test_without_fast_pericenters_the_step_is_unchanged():
    # Near-circular bodies are never substepped, so step_resolving_pericenters must reproduce step() bit for bit;
    # this also checks its force cache (reusing the closing kick's accelerations for the next opening kick).
    x, u, m = embryo_and_giant(1.0, 0.99)
    runs = []
    for stepper in (plain, resolving):
        Q, v = hy.from_heliocentric(x, u, m, M_STAR)
        alive, big, R = np.ones(2, bool), np.ones(2, bool), np.full(2, 1e-9)
        comp, log = np.column_stack([m, np.zeros((2, 3))]), np.zeros((4, 6))
        rc = hy.critical_radii(Q, v, m, alive, MU, DT)
        acc = hy.far_accelerations(Q, m, alive, big, rc)
        for k in range(2000):
            stepper(Q, v, m, R, alive, big, comp, rc, MU, M_STAR, DT, log, 0, k * DT, acc)
        runs.append(np.concatenate([Q, v]))
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize("a, q", [(1.5, 0.2), (2.5, 0.3)])
def test_substeps_match_a_fine_timestep_reference(a, q):
    # An eccentric embryo whose pericenter passage is much faster than one step, perturbed by the 2.5 MJ giant.
    # Measured: plain steps drift by ~1-3% in a; with substeps the history matches dt/16 to ~1e-3.
    reference = embryo_a_history(plain, 16, a, q)
    unresolved = embryo_a_history(plain, 1, a, q)
    resolved = embryo_a_history(resolving, 1, a, q)
    error = lambda hist: np.abs(hist - reference).max() / a
    assert error(resolved) < 2e-3
    assert error(unresolved) > 5 * error(resolved)
