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


@pytest.mark.parametrize("a, e", [(1.136, 0.918), (2.5, 0.88), (1.0, 0.05), (-3.0, 1.4)])
def test_time_to_radius_inbound_matches_kepler_propagation(a, e):
    # From pericenter, propagate back by tau_t to reach r_target on the way in, then by tau0 more: from there the
    # body must take tau0 to reach r_target (and tau0 + tau_t to reach pericenter). a < 0 is a hyperbola. The
    # e = 0.918 orbit is the one a seed-6 planetesimal had when its fall into pericenter went unresolved.
    # Tolerance 1e-7 relative: the Kepler solver's convergence.
    from nbody.kepler import kepler_drift
    q = a * (1 - e)
    x, u = np.array([q, 0, 0.0]), np.array([0, np.sqrt(MU * (1 + e) / q), 0.0])
    T = 2 * np.pi * np.sqrt(abs(a) ** 3 / MU)
    for tau_t, tau0 in [(0.0, 0.3 * T), (0.02 * T, 0.1 * T), (0.05 * T, 0.001 * T)]:
        x_t, u_t = kepler_drift(x, u, MU, -tau_t)
        X, U = kepler_drift(x_t, u_t, MU, -tau0)
        r = np.linalg.norm(X)
        t = hy.time_to_radius_inbound(r, X @ U / r, U @ U, e, MU, max(q, np.linalg.norm(x_t)))
        assert t == pytest.approx(tau0, rel=1e-7, abs=1e-12 * T)
    X, U = kepler_drift(x, u, MU, 0.01 * T)                       # outbound: an ellipse comes back, a hyperbola never
    r = np.linalg.norm(X)
    expected = 0.99 * T if a > 0 else np.inf
    assert hy.time_to_radius_inbound(r, X @ U / r, U @ U, e, MU, q) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("a, q, refine", [(1.5, 0.2, 16), (2.5, 0.3, 16), (1.136, 0.093, 64)])
def test_substeps_match_a_fine_timestep_reference(a, q, refine):
    # An eccentric embryo whose pericenter passage is much faster than one step, perturbed by the 2.5 MJ giant.
    # Measured: plain steps drift by ~1-3% in a; with substeps the history matches the reference to ~1e-3.
    # The reference must itself resolve pericenter: dt/16 sweeps 0.74 rad per step at q = 0.093, so that case uses dt/64.
    reference = embryo_a_history(plain, refine, a, q)
    unresolved = embryo_a_history(plain, 1, a, q)
    resolved = embryo_a_history(resolving, 1, a, q)
    error = lambda hist: np.abs(hist - reference).max() / a
    assert error(resolved) < 2e-3
    assert error(unresolved) > 5 * error(resolved)


def test_plunging_planetesimal_crossing_an_embryo_keeps_its_orbit():
    # Seed 4's failure: a planetesimal (a = 2.24 AU, q = 0.062 AU) plunging past the innermost embryo (0.62 AU) is
    # flagged as an encounter on most passages. Unless the encounter group carries its own jump term, its fall goes
    # unresolved there and its energy random-walks (measured before the fix: 30% in a over 200 yr, system dE/E 1.7e-5).
    # With it: 6e-3 in a (0.6% of the planetesimal's own energy over ~50 passages) and dE/E 3e-7.
    a, q = 2.24, 0.062
    e = 1 - q / a
    vq = np.sqrt(MU * (1 + e) / q)
    x = np.array([[q, 0, 0], [0.62, 0, 0], [6.1, 0, 0]])
    u = np.array([[0, vq, 0], [0, np.sqrt(MU / 0.62), 0], [0, np.sqrt(MU / 6.1), 0]])
    m = np.array([0.0119 * M_EARTH, 0.4 * M_EARTH, 2.5 * M_JUPITER])

    def history(stepper, refine, years=200.0):
        dt = DT / refine
        Q, v = hy.from_heliocentric(x, u, m, M_STAR)
        alive, big = np.ones(3, bool), np.array([False, True, True])
        R, comp, log = np.full(3, 1e-9), np.column_stack([m, np.zeros((3, 3))]), np.zeros((4, 6))
        rc = hy.critical_radii(Q, v, m, alive, MU, DT)
        acc = hy.far_accelerations(Q, m, alive, big, rc)
        E0 = hy.total_energy(Q, v, m, alive, big, MU, M_STAR)
        out, dE = [], 0.0
        for k in range(int(years / DT) * refine):
            stepper(Q, v, m, R, alive, big, comp, rc, MU, M_STAR, dt, log, 0, 0.0, acc)
            if k % (23 * refine) == 0:
                xh, uh = hy.to_heliocentric(Q, v, m, M_STAR)
                out.append(orbital_elements(xh[:1], uh[:1], MU)[0][0])
                dE = max(dE, abs(hy.total_energy(Q, v, m, alive, big, MU, M_STAR) / E0 - 1))
        return np.array(out), dE

    reference, _ = history(plain, 64)
    resolved, dE = history(resolving, 1)
    assert np.abs(resolved - reference).max() / a < 1.5e-2
    assert dE < 1e-6
