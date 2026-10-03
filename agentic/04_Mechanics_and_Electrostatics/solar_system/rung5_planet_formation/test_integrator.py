"""5b: the exact Kepler solver and the hybrid (Wisdom-Holman + close-encounter) integrator."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import nbody.hybrid as hy
from nbody.kepler import kepler_drift
from nbody.integrators import leapfrog
from nbody.units import G, M_EARTH
from nbody.planets import A, MASS

MU = G * 1.0


def invariants(x, v):
    L = np.cross(x, v)
    return 0.5 * v @ v - MU / np.linalg.norm(x), L, np.cross(v, L) - MU * x / np.linalg.norm(x)


# ---------------------------------------------------------------- Kepler solver

@pytest.mark.parametrize("e", [0.0, 0.5, 0.9, 0.99, 0.9999])
def test_kepler_orbit_closes_and_conserves_invariants(e):
    rp = 1 - e
    vp = np.sqrt(MU * (1 + e) / rp)
    x0, v0 = np.array([rp, 0, 0.0]), np.array([0.0, vp * np.cos(0.3), vp * np.sin(0.3)])
    E0, L0, A0 = invariants(x0, v0)
    P = 2 * np.pi * np.sqrt((-MU / (2 * E0)) ** 3 / MU)
    assert np.abs(kepler_drift(x0, v0, MU, P)[0] - x0).max() < 1e-8 * rp           # one period: back to the start
    x, v = x0.copy(), v0.copy()
    for _ in range(1000):                                                            # many uneven steps
        x, v = kepler_drift(x, v, MU, 1.37e-3 * P)
    E, L, A_vec = invariants(x, v)
    assert abs(E / E0 - 1) < 1e-11
    assert np.abs(L - L0).max() < 1e-12 * np.linalg.norm(L0)
    assert np.abs(A_vec - A0).max() < 1e-11 * MU                                     # Runge-Lenz: orbit orientation fixed


def test_kepler_hyperbolic_matches_direct_integration():
    x0, v0 = np.array([1.0, 0, 0]), np.array([0.0, 1.5 * np.sqrt(2 * MU), 0.3])
    rhs = lambda t, y: np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3])
    ref = solve_ivp(rhs, [0, 2.0], np.concatenate([x0, v0]), rtol=1e-13, atol=1e-14).y[:, -1]
    x, v = kepler_drift(x0, v0, MU, 2.0)
    assert np.abs(x - ref[:3]).max() < 1e-11 and np.abs(v - ref[3:]).max() < 1e-11
    assert np.abs(kepler_drift(x, v, MU, -2.0)[0] - x0).max() < 1e-12             # time-reversible


# ---------------------------------------------------------------- Wisdom-Holman on Jupiter + Saturn

def jupiter_saturn():
    a, m, phase = np.array([A[4], A[5]]), np.array([MASS[4], MASS[5]]), np.array([0.0, 2.0])
    x = np.stack([a * np.cos(phase), a * np.sin(phase), np.zeros(2)], axis=1)
    vc = np.sqrt(MU / a)
    u = np.stack([-vc * np.sin(phase), vc * np.cos(phase), np.array([0.0, 0.01 * vc[1]])], axis=1)
    return x, u, m


def run(x, u, m, dt, n_steps, R=None, big=None):
    """Integrate with the hybrid stepper; returns Q, v, alive, max |dE/E| (energy checked every step), log, n_log."""
    R = np.full(len(m), 1e-9) if R is None else R
    big = np.ones(len(m), bool) if big is None else big
    Q, v = hy.from_heliocentric(x, u, m, 1.0)
    m, alive, log, n_log = m.copy(), np.ones(len(m), bool), np.zeros((10, 6)), 0
    comp = np.column_stack([m, np.zeros((len(m), 3))])          # everything starts as seed solids
    rc = hy.critical_radii(Q, v, m, alive, MU, dt)
    E0, worst = hy.total_energy(Q, v, m, alive, big, MU, 1.0), 0.0
    for k in range(n_steps):
        n_log = hy.step(Q, v, m, R, alive, big, comp, rc, MU, 1.0, dt, log, n_log, k * dt)
        worst = max(worst, abs((hy.total_energy(Q, v, m, alive, big, MU, 1.0) + log[:n_log, 5].sum()) / E0 - 1))
    run.comp = comp                                            # for the composition checks
    return Q, v, m, alive, worst, log, n_log


def test_coordinate_round_trip():
    x, u, m = jupiter_saturn()
    Q, v = hy.from_heliocentric(x, u, m, 1.0)
    x2, u2 = hy.to_heliocentric(Q, v, m, 1.0)
    assert np.allclose(x2, x, rtol=0, atol=1e-15) and np.allclose(u2, u, rtol=1e-14)


def test_energy_error_is_bounded_and_second_order():
    # Measured max |dE/E| over 600 yr: 3.3e-6, 8.1e-7, 2.0e-7 for dt = P_J/20, /40, /80.
    x, u, m = jupiter_saturn()
    errs = [run(x, u, m, 11.86 / n, int(600 / (11.86 / n)))[4] for n in (20, 40, 80)]
    assert errs[0] < 1e-5
    assert all(3.5 < errs[k] / errs[k + 1] < 4.5 for k in range(2))


def test_matches_a_tiny_step_leapfrog_reference():
    # The Rung 1 leapfrog with dt = P_J / 20000 (star as a particle) is the reference. Measured differences after
    # 593 yr: 1.9e-3, 4.8e-4 AU at P/20, P/40 - converging as dt^2.
    x, u, m = jupiter_saturn()
    pos, vel, mm = np.vstack([np.zeros(3), x]), np.vstack([np.zeros(3), u]), np.concatenate([[1.0], m])
    vel -= mm @ vel / mm.sum()
    pos -= mm @ pos / mm.sum()
    t, X, V = leapfrog(pos, vel, mm, 0.0, 11.86 / 20000, 1000000, 1000000)
    reference = X[-1, 1:] - X[-1, 0]
    diffs = [np.abs(run(x, u, m, t[-1] / round(t[-1] / (11.86 / n)), round(t[-1] / (11.86 / n)))[0] - reference).max()
             for n in (20, 40)]
    assert diffs[1] < 1e-3 and 3.5 < diffs[0] / diffs[1] < 4.5


def test_time_reversible():
    x, u, m = jupiter_saturn()
    Q, v, mm, alive, *_ = run(x, u, m, 11.86 / 20, 500)
    xb, ub = hy.to_heliocentric(Q, -v, mm, 1.0)
    Qb, *_ = run(xb, ub, mm, 11.86 / 20, 500)
    assert np.abs(Qb - x).max() < 1e-9


# ---------------------------------------------------------------- close encounters

def test_changeover_function():
    rc, d = 1.0, 1e-7
    assert hy.changeover(0.1, rc) == (0.0, 0.0) and hy.changeover(1.0, rc) == (1.0, 0.0)
    for r in np.linspace(0.12, 0.98, 30):
        numeric = (hy.changeover(r + d, rc)[0] - hy.changeover(r - d, rc)[0]) / (2 * d)
        assert np.isclose(hy.changeover(r, rc)[1], numeric, rtol=1e-6)


def scattering_pair():
    """Two 10 Mearth embryos 0.05 AU apart at 5 AU: inside one Hill radius, they scatter repeatedly."""
    m = np.full(2, 10 * M_EARTH)
    a, phase = np.array([5.0, 5.05]), np.array([0.0, 0.03])
    x = np.stack([a * np.cos(phase), a * np.sin(phase), np.zeros(2)], axis=1)
    vc = np.sqrt(MU / a)
    return x, np.stack([-vc * np.sin(phase), vc * np.cos(phase), np.zeros(2)], axis=1), m


def test_energy_conserved_through_repeated_close_encounters():
    # With the switch, the splitting error of the changeover shell remains: measured 4.3e-5 at dt = P/40 with a
    # 3-Hill-radius changeover (3e-6 with 6 Hill radii; pure Wisdom-Holman, no switch: ~1e-2). Formation studies accept ~1e-4.
    x, u, m = scattering_pair()
    P = 2 * np.pi * np.sqrt(125 / MU)
    Q, v, mm, alive, worst, *_ = run(x, u, m, P / 40, int(300 / (P / 40)))
    assert worst < 1e-4


def test_collision_merges_and_closes_the_energy_budget():
    vk = np.sqrt(MU / 5.0)
    x = np.array([[5.0, 0, 0], [5.0, 0.002, 0.0]])
    u = np.array([[0.0, vk, 0.0], [0.003 * vk, vk, 0.0]])
    m = np.array([5 * M_EARTH, 3 * M_EARTH])
    Q, v, mm, alive, worst, log, n_log = run(x, u, m, 2 * np.pi * np.sqrt(125 / MU) / 40, 40, R=np.full(2, 1e-3))
    assert n_log == 1 and alive.sum() == 1
    assert np.isclose(mm[alive].sum(), m.sum(), rtol=1e-15)
    assert np.allclose(run.comp.sum(axis=1), mm, rtol=1e-14)               # composition columns add up to the mass
    survivor = np.flatnonzero(alive)[0]
    assert np.isclose(run.comp[survivor, 1], m.min(), rtol=1e-14)           # the smaller body arrived as collisions
    # E_now + energy lost in the merger = E_initial. The merger bookkeeping is exact (see the merge test); what remains
    # is ordinary integration error during the approach (measured 1.3e-10).
    assert worst < 1e-8
    survivor_mass, absorbed_mass = log[0, 3], log[0, 4]
    contact = 0.5 * survivor_mass * absorbed_mass / (survivor_mass + absorbed_mass) * (0.003 * vk) ** 2 \
        - G * survivor_mass * absorbed_mass / 2e-3
    assert np.isclose(log[0, 5], contact, rtol=0.02)         # lost energy ~ relative KE + pair potential at contact


def test_merge_conserves_mass_momentum_and_volume():
    Q, v = np.array([[1.0, 0, 0], [1.1, 0.1, 0]]), np.array([[0, 6.0, 0], [0.5, 5.0, 0.1]])
    m, R, alive, big = np.array([2.0, 1.0]), np.array([0.3, 0.2]), np.ones(2, bool), np.array([True, False])
    p0, cm0 = m @ v, m @ Q
    comp = np.array([[1.5, 0.2, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])          # rows sum to the masses 2 and 1
    hy.merge(0, 1, Q, v, m, R, alive, big, comp)
    assert m[0] == 3.0 and not alive[1] and big[0]
    assert np.allclose(comp[0], [1.5, 0.2 + 0.4 + 0.3, 0.4, 0.2]) and np.all(comp[1] == 0)   # gas stays gas
    assert np.allclose(m[0] * v[0], p0, rtol=1e-15) and np.allclose(m[0] * Q[0], cm0, rtol=1e-15)
    assert np.isclose(R[0] ** 3, 0.3**3 + 0.2**3, rtol=1e-14)


def test_planetesimals_ignore_each_other():
    # Two small bodies 5e-4 AU apart: run together, each must follow the same path as when run alone, up to the
    # coupling through the star's reflex motion, ~(m / M_star) v t ~ 1.6e-13 AU here. If they attracted each other,
    # the mutual pull would displace them by ~3e-4 AU over these 60 yr.
    vk = np.sqrt(MU / 5.0)
    x = np.array([[5.0, 0, 0], [5.0, 5e-4, 0]])
    u = np.array([[0, vk, 0], [0, vk, 0.0]])
    m = np.array([1e-15, 1e-15])
    dt = 2 * np.pi * np.sqrt(125 / MU) / 40
    together = run(x, u, m, dt, 100, big=np.zeros(2, bool))[0]
    alone = [run(x[k:k + 1], u[k:k + 1], m[k:k + 1], dt, 100, big=np.zeros(1, bool))[0][0] for k in range(2)]
    assert np.abs(together - np.array(alone)).max() < 1e-12


def test_orbital_elements_recover_a_known_orbit():
    from nbody.orbits import orbital_elements
    a, e, inc = 7.0, 0.3, 0.2
    rp = a * (1 - e)
    vp = np.sqrt(MU * (1 + e) / rp)
    x, u = np.array([[rp, 0, 0.0]]), np.array([[0, vp * np.cos(inc), vp * np.sin(inc)]])
    x2, u2 = kepler_drift(x[0], u[0], MU, 3.7)                 # anywhere along the orbit
    for xx, uu in ((x, u), (x2[None], u2[None])):
        a_, e_, i_ = orbital_elements(xx, uu, MU)
        assert np.allclose([a_[0], e_[0], i_[0]], [a, e, inc], rtol=1e-12)
