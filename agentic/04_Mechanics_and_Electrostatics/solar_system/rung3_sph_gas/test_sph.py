"""Rung 3: SPH gas. Kernel and density first, then conservation, then the Evrard and Jeans tests."""
import os
import numpy as np
import pytest
from nbody.sph import kernel, kernel_derivative, density, smoothing_lengths, hydro_forces, Gas, ETA
from nbody.clouds import lattice_sphere, uniform_sphere, free_fall_time, cycloid_radius
from nbody.integrators import sph_leapfrog
from nbody.gravity import potential_energy
from nbody.diagnostics import kinetic_energy
from nbody.units import G, PC

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- kernel and density

def test_kernel_is_normalized():
    h = 0.7
    r = np.linspace(0, 2 * h, 200001)
    integrand = 4 * np.pi * r**2 * np.array([kernel(x, h) for x in r])
    assert np.isclose(np.trapz(integrand, r), 1.0, rtol=1e-8)


def test_kernel_derivative_matches_finite_difference():
    h, d = 0.7, 1e-6
    for r in np.linspace(0.01, 1.99, 40) * h:
        numeric = (kernel(r + d, h) - kernel(r - d, h)) / (2 * d)
        assert np.isclose(kernel_derivative(r, h), numeric, rtol=1e-6)


def test_lattice_density_and_smoothing_length():
    # Cubic lattice of spacing a: interior density must be m / a^3 and h must settle to ETA * a.
    n, a = 20, 1.0
    g = np.arange(n) * a
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T.astype(float)
    mass = np.ones(len(pos))
    h, iters = smoothing_lengths(pos, mass, np.full(len(pos), 1.0))
    rho = density(pos, mass, h)
    interior = np.all((pos > 5) & (pos < n - 6), axis=1)       # more than 2h from every face
    assert iters < 100
    assert np.allclose(rho[interior], 1.0, rtol=2e-3)
    assert np.allclose(h[interior], ETA * a, rtol=2e-3)


def test_smoothing_lengths_are_self_consistent():
    pos, vel, mass = uniform_sphere(1000, 1.0, 1.0, rng)
    h, iters = smoothing_lengths(pos, mass, np.full(1000, 0.2))
    assert iters < 100
    assert np.allclose(h, ETA * (mass / density(pos, mass, h)) ** (1 / 3), rtol=1e-3)


def test_pair_forces_conserve_momentum_angular_momentum_and_energy():
    # Equal and opposite pair forces along r_ij: total force and torque vanish. The total heating exactly
    # balances the work done by the SPH forces: sum m v.a + sum m du/dt = 0 (summed over all particles).
    pos, vel, mass = uniform_sphere(1000, 1.0, 1.0, rng)
    vel = rng.normal(size=pos.shape) * 0.3
    h, _ = smoothing_lengths(pos, mass, np.full(1000, 0.2))
    rho = density(pos, mass, h)
    P, c = 0.1 * rho, np.full(1000, np.sqrt(0.1))
    acc, dudt, signal = hydro_forces(pos, vel, mass, h, rho, P, c, 1.0, 2.0)
    force = mass[:, None] * acc
    assert np.abs(force.sum(axis=0)).max() < 1e-14 * np.abs(force).sum()
    assert np.abs(np.cross(pos, force).sum(axis=0)).max() < 1e-14 * np.abs(np.cross(pos, force)).sum()
    work = np.sum(mass * np.sum(vel * acc, axis=1))
    assert abs(work + np.sum(mass * dudt)) < 1e-12 * np.abs(mass * np.sum(vel * acc, axis=1)).sum()


def test_gas_requires_explicit_parameters():
    with pytest.raises(AssertionError):
        Gas("isothermal")
    with pytest.raises(AssertionError):
        Gas("adiabatic", cs=1.0)
    with pytest.raises(AssertionError):
        Gas("polytropic", gamma=5 / 3)


# ---------------------------------------------------------------- Evrard adiabatic collapse

@pytest.fixture(scope="module")
def evrard():
    """Evrard (1988): rho ~ 1/r, gamma = 5/3, u = 0.05 G M / R, at rest. G = M = R = 1 units of energy and time."""
    pos, vel, mass = lattice_sphere(1500, 1.0, 1.0)
    pos *= (np.linalg.norm(pos, axis=1) ** 0.5)[:, None]          # r -> r^(3/2): uniform -> rho ~ 1/r
    T = np.sqrt(1 / G)
    res = sph_leapfrog(pos, vel, 0.05 * G, mass, Gas("adiabatic", gamma=5 / 3), 0.01, np.linspace(0, 3, 61) * T)
    K = np.array([kinetic_energy(v, mass) for v in res["vel"]]) / G
    W = np.array([potential_energy(x, mass, 0.01) for x in res["pos"]]) / G
    U = np.array([np.sum(mass * u) for u in res["u"]]) / G
    return res["t"] / T, K, U, W


def test_evrard_initial_energies(evrard):
    t, K, U, W = evrard
    assert K[0] == 0 and np.isclose(U[0], 0.05, rtol=1e-12)
    assert np.isclose(W[0], -2 / 3, rtol=5e-3)          # W = -(3 - a)/(5 - 2a) G M^2 / R for rho ~ r^-a, a = 1


def test_evrard_conserves_total_energy(evrard):
    # Measured 3e-4 over the whole run (GADGET-1 reports 2.3e-3 for this test).
    t, K, U, W = evrard
    E = K + U + W
    assert np.abs(E / E[0] - 1).max() < 2e-3


def test_evrard_bounce_matches_published_runs(evrard):
    # Published (energies in G M^2/R, time in sqrt(R^3/GM)): maximum compression at t = 1.05-1.15 with U_max = 1.41
    # (GADGET-1, 4224 particles) to 1.75 (1D PPM reference). At our lower resolution expect the low end, a bit late.
    t, K, U, W = evrard
    assert 1.0 <= t[U.argmax()] <= 1.3
    assert 1.2 <= U.max() <= 1.8
    assert 0.8 <= t[K.argmax()] <= 1.0                  # infall speed peaks just before the bounce (refs: 0.85-0.90)


# ---------------------------------------------------------------- isothermal collapse and the Jeans criterion

M, R = 1.0, 0.1 * PC
T_FF = free_fall_time(M, R)
EPS = 0.01 * R
RHO0 = M / (4 / 3 * np.pi * R**3)


def isothermal_run(alpha, t_end=2.0, n=1500):
    """Uniform isothermal sphere at rest with U/|W| = alpha. Stops if the peak density reaches 1000 rho0."""
    pos, vel, mass = lattice_sphere(n, M, R)
    W0 = abs(potential_energy(pos, mass, EPS))
    cs = np.sqrt(2 * alpha * W0 / (3 * M))                          # U = (3/2) M cs^2
    res = sph_leapfrog(pos, vel, 0.0, mass, Gas("isothermal", cs=cs), EPS, np.linspace(0, t_end, 21) * T_FF,
                       rho_stop=1e3 * RHO0)
    return res, pos, mass


def test_collapse_guaranteed_below_truelove_bound():
    # Truelove et al. (1998): for U/|W| < 5/pi^2 = 0.507 the inward rarefaction cannot reach the center before
    # it collapses, so collapse is guaranteed.
    res, pos, mass = isothermal_run(0.45)
    assert res["stopped"] and res["t"][-1] < 1.2 * T_FF


def test_no_collapse_below_one_jeans_mass():
    # U/|W| = 1 is exactly one Jeans mass (M/M_J = (U/|W|)^(-3/2), Tohline 1982). At 1.2 the cloud holds 0.76 M_J:
    # it must not run away; it expands. (Measured threshold: U/|W| = 0.71-0.78, rising slowly with N.)
    res, pos, mass = isothermal_run(1.2)
    r50 = [np.median(np.linalg.norm(x, axis=1)) for x in res["pos"]]
    assert not res["stopped"]
    assert res["rho"].max() < 10 * RHO0
    assert r50[-1] > r50[0]


def test_interior_free_falls_until_the_rarefaction_arrives():
    # A uniform sphere has no internal pressure gradient: its core follows the pressure-free cycloid,
    # rho = rho0 / cos^6(eta), even with gas pressure. Measured within 2-6% up to 0.8 t_ff for U/|W| = 0.3.
    res, pos, mass = isothermal_run(0.3, t_end=1.0)
    core = np.linalg.norm(pos, axis=1) < 0.3 * R
    for t, rho in zip(res["t"], res["rho"]):
        if t <= 0.8 * T_FF:
            assert np.isclose(np.median(rho[core]) / RHO0, cycloid_radius(t, T_FF) ** -3, rtol=0.10)


def test_isothermal_energy_budget_closes():
    # Isothermal gas radiates its compressional heat away; with that tallied, K + W + E_rad is conserved.
    res, pos, mass = isothermal_run(0.3, t_end=1.0)
    E = np.array([kinetic_energy(v, mass) + potential_energy(x, mass, EPS) + e
                  for x, v, e in zip(res["pos"], res["vel"], res["E_rad"])])
    assert res["E_rad"][-1] > 0
    assert np.abs(E / E[0] - 1).max() < 5e-3


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="~70 s; run with RUN_SLOW=1")
def test_evrard_matches_gadget1_at_the_same_resolution():
    # Springel, Yoshida & White (2001, New Astron. 6, 79), Fig. 10, SPH with 4224 particles (values read off the
    # figure, +-0.02): U_max 1.41 at t = 1.15, W_min -2.12, and K, U, W = 0.10, 0.51, -1.22 at t = 2 and
    # 0.07, 0.67, -1.35 at t = 3. Ours (4224 particles): 1.42 at 1.14, -2.15, (0.11, 0.51, -1.23), (0.07, 0.66, -1.35).
    pos, vel, mass = lattice_sphere(4224, 1.0, 1.0)
    pos *= (np.linalg.norm(pos, axis=1) ** 0.5)[:, None]
    T = np.sqrt(1 / G)
    res = sph_leapfrog(pos, vel, 0.05 * G, mass, Gas("adiabatic", gamma=5 / 3), 0.01, np.linspace(0, 3, 61) * T)
    t = res["t"] / T
    K = np.array([kinetic_energy(v, mass) for v in res["vel"]]) / G
    W = np.array([potential_energy(x, mass, 0.01) for x in res["pos"]]) / G
    U = np.array([np.sum(mass * u) for u in res["u"]]) / G
    assert abs(U.max() - 1.41) < 0.05 and abs(t[U.argmax()] - 1.15) < 0.1
    assert abs(W.min() + 2.12) < 0.06
    for t_ref, ref in ((2.0, (0.10, 0.51, -1.22)), (3.0, (0.07, 0.67, -1.35))):
        i = np.argmin(abs(t - t_ref))
        assert np.allclose((K[i], U[i], W[i]), ref, atol=0.04)
