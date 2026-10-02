"""Rung 1: leapfrog reproduces Kepler orbits and has the properties of a symplectic integrator."""
import numpy as np
import pytest
from nbody.integrators import leapfrog
from nbody.diagnostics import total_energy, angular_momentum, momentum
from nbody.orbits import kepler_period, two_body_at_pericenter, figure_eight, FIGURE_EIGHT_PERIOD

# Sun + a Jupiter-mass planet on an e = 0.5 orbit at 1 AU, no softening.
MASS = np.array([1.0, 1e-3])
A, E = 1.0, 0.5
PERIOD = kepler_period(A, MASS[1], MASS[0])
POS0, VEL0 = two_body_at_pericenter(MASS[0], MASS[1], A, E)


def kepler_run(steps_per_orbit, n_orbits, save_every=1):
    return leapfrog(POS0, VEL0, MASS, 0.0, PERIOD / steps_per_orbit, steps_per_orbit * n_orbits, save_every)


def energy_error(X, V):
    E = np.array([total_energy(x, v, MASS, 0.0) for x, v in zip(X, V)])
    return np.abs(E / E[0] - 1)


def measured_period(t, X, n_orbits):
    """Time for the separation vector to sweep n_orbits * 2 pi, interpolated between snapshots."""
    rel = X[:, 1] - X[:, 0]
    phase = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    return np.interp(2 * np.pi * n_orbits, phase, t) / n_orbits


@pytest.fixture(scope="module")
def long_run():
    return kepler_run(1000, 100, save_every=10)


def test_inputs_not_modified():
    pos, vel = POS0.copy(), VEL0.copy()
    leapfrog(pos, vel, MASS, 0.0, 1e-3, 10)
    assert np.all(pos == POS0) and np.all(vel == VEL0)


def test_kepler_period():
    # The phase error of leapfrog is O(dt^2): ~2e-4 at 1000 steps/orbit, 4x smaller at 2000.
    errs = [abs(measured_period(*kepler_run(n, 20)[:2], 19) / PERIOD - 1) for n in (1000, 2000)]
    assert errs[0] < 3e-4
    assert 3.8 < errs[0] / errs[1] < 4.2


def test_energy_error_bounded(long_run):
    # Symplectic: the energy error oscillates each orbit but does not drift over 100 orbits.
    t, X, V = long_run
    dE = energy_error(X, V)
    tenth = len(dE) // 10
    assert dE.max() < 2e-4
    assert abs(dE[-tenth:].max() / dE[:tenth].max() - 1) < 1e-2


def test_energy_error_scales_as_dt_squared():
    errs = [energy_error(*kepler_run(n, 2)[1:]).max() for n in (500, 1000, 2000)]
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.all((3.8 < ratios) & (ratios < 4.2))


def test_angular_momentum_conserved(long_run):
    # Central pairwise forces conserve L exactly in each kick and drift, so only roundoff remains.
    t, X, V = long_run
    L = np.array([angular_momentum(x, v, MASS)[2] for x, v in zip(X, V)])
    assert np.abs(L / L[0] - 1).max() < 1e-12


def test_momentum_conserved(long_run):
    t, X, V = long_run
    p = np.array([momentum(v, MASS) for v in V])
    p_scale = MASS[1] * np.abs(V[:, 1]).max()
    assert np.abs(p).max() < 1e-12 * p_scale


def test_time_reversible():
    dt, n = PERIOD / 1000, 5000
    t, X, V = leapfrog(POS0, VEL0, MASS, 0.0, dt, n, n)
    t, Xb, Vb = leapfrog(X[-1], -V[-1], MASS, 0.0, dt, n, n)
    assert np.abs(Xb[-1] - POS0).max() < 1e-10
    assert np.abs(-Vb[-1] - VEL0).max() < 1e-10


def test_figure_eight_closes():
    # A genuinely three-body orbit: after one period every body is back where it started.
    pos, vel, mass = figure_eight()
    errs = []
    for n in (1000, 2000):
        t, X, V = leapfrog(pos, vel, mass, 0.0, FIGURE_EIGHT_PERIOD / n, n, n)
        errs.append(np.abs(X[-1] - pos).max())
    assert errs[0] < 1e-4
    assert 3.5 < errs[0] / errs[1] < 4.5
