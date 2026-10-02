"""Rung 2: a cold, pressureless uniform sphere collapses on the analytic free-fall time."""
import numpy as np
import pytest
from nbody.clouds import uniform_sphere, power_law_sphere, virial_velocities, free_fall_time, cycloid_radius
from nbody.integrators import leapfrog
from nbody.gravity import potential_energy, potentials
from nbody.diagnostics import total_energy, momentum, virial_ratio, lagrangian_radii, unbound_fraction
from nbody.units import G, PC

# A 1 Msun cloud of radius 0.1 pc: t_ff ~ 5.2e5 yr.
M, R = 1.0, 0.1 * PC
T_FF = free_fall_time(M, R)


@pytest.fixture(scope="module")
def collapse():
    """N = 1000, eps = 0.01 R, dt = t_ff / 1000, run to 4 t_ff, snapshot every 0.02 t_ff."""
    pos, vel, mass = uniform_sphere(1000, M, R, np.random.default_rng(0))
    t, X, V = leapfrog(pos, vel, mass, 0.01 * R, T_FF / 1000, 4000, 20)
    return t, X, V, mass, 0.01 * R


def test_uniform_sphere_sampling():
    pos, vel, mass = uniform_sphere(20000, M, R, np.random.default_rng(1))
    assert np.isclose(mass.sum(), M) and np.all(vel == 0)
    assert np.allclose(mass @ pos, 0, atol=1e-12 * R)
    # Uniform density: the radius enclosing a fraction q of the mass is R q^(1/3).
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.allclose(lagrangian_radii(pos, mass, q), R * q ** (1 / 3), rtol=0.02)


def test_potential_energy_of_uniform_sphere():
    # W = -(3/5) G M^2 / R for a continuous uniform sphere. N = 2000 sampling noise is ~0.5%.
    pos, vel, mass = uniform_sphere(2000, M, R, np.random.default_rng(2))
    assert np.isclose(potential_energy(pos, mass, 0.0), -0.6 * G * M**2 / R, rtol=0.02)


def test_free_fall_time_matches_density_form():
    rho = M / (4 / 3 * np.pi * R**3)
    assert np.isclose(T_FF, np.sqrt(3 * np.pi / (32 * G * rho)), rtol=1e-14)


def test_cycloid_endpoints_and_small_time_limit():
    assert cycloid_radius(0.0, T_FF) == 1.0
    assert cycloid_radius(T_FF, T_FF) < 1e-15
    # Series solution of r'' = -G M / r^2 from rest: r/r0 = 1 - x/2 - x^2/12 + O(x^3), with x = G M t^2 / R^3.
    t = 0.01 * T_FF
    x = G * M * t**2 / R**3
    assert np.isclose(cycloid_radius(t, T_FF), 1 - x / 2 - x**2 / 12, rtol=0, atol=1e-11)


def test_shells_follow_the_cycloid(collapse):
    # Up to 0.8 t_ff each Lagrangian shell should track r0 cos^2(eta). Poisson noise in the initial sampling
    # (~1/sqrt(N)) grows as the cloud contracts; measured deviations at N = 1000 are 1-3% (r50), 5-7% (r90).
    t, X, V, mass, eps = collapse
    early = t <= 0.8 * T_FF
    radii = np.array([lagrangian_radii(x, mass, [0.5, 0.9]) for x in X[early]])
    predicted = cycloid_radius(t[early], T_FF)
    assert np.abs(radii[:, 0] / radii[0, 0] / predicted - 1).max() < 0.05
    assert np.abs(radii[:, 1] / radii[0, 1] / predicted - 1).max() < 0.10


def test_collapse_happens_at_free_fall_time(collapse):
    # Minimum half-mass radius at t_ff, slightly late because softening weakens the force near the center.
    t, X, V, mass, eps = collapse
    r50 = np.array([lagrangian_radii(x, mass, [0.5])[0] for x in X])
    assert 1.0 <= t[r50.argmin()] / T_FF <= 1.08


def test_energy_conserved_through_bounce(collapse):
    t, X, V, mass, eps = collapse
    E = np.array([total_energy(x, v, mass, eps) for x, v in zip(X, V)])
    assert np.abs(E / E[0] - 1).max() < 5e-3


def test_momentum_conserved(collapse):
    t, X, V, mass, eps = collapse
    p_scale = np.sum(mass * np.linalg.norm(V[-1], axis=1))
    assert np.abs(momentum(V[-1], mass)).max() < 1e-10 * p_scale


def test_bound_remnant_is_virialized(collapse):
    # The bounce ejects ~20% of the mass. The particles still bound (1/2 v^2 + phi < 0) settle to
    # 2K/|W| = 1; including the escapers gives ~1.5. Averaged over 3-4 t_ff.
    t, X, V, mass, eps = collapse
    ratios = []
    for x, v in zip(X[t >= 3 * T_FF], V[t >= 3 * T_FF]):
        bound = 0.5 * np.sum(v**2, axis=1) + potentials(x, mass, eps) < 0
        ratios.append(virial_ratio(x[bound], v[bound], mass[bound], eps))
    assert abs(np.mean(ratios) - 1) < 0.1


def test_ejected_mass_matches_published_fit(collapse):
    # Joyce, Marcos & Sylos Labini (2009, MNRAS 397, 775) fit the ejected mass of a cold uniform collapse as
    # f = 0.048 + 0.022 ln N, i.e. 20.0% at N = 1000. Across 5 seeds we measure 17.8-23.6% (scatter ~2.4 points),
    # so allow 6 points, about 2.5 times the seed-to-seed scatter.
    t, X, V, mass, eps = collapse
    measured = np.mean([unbound_fraction(x, v, mass, eps) for x, v in zip(X[t >= 3 * T_FF], V[t >= 3 * T_FF])])
    assert abs(measured - (0.048 + 0.022 * np.log(len(mass)))) < 0.06


def test_softening_makes_the_collapse_convergeable():
    # Without softening, the few very close pairs that random sampling always produces orbit each other faster than
    # any fixed timestep can follow (the error blows up by ~0.1 t_ff, well before the bounce): shrinking dt does
    # not rescue energy conservation. With softening, the error falls as dt^2 (16x for 4x smaller dt).
    pos, vel, mass = uniform_sphere(500, M, R, np.random.default_rng(0))

    def max_energy_error(eps, steps_per_tff):
        t, X, V = leapfrog(pos, vel, mass, eps, T_FF / steps_per_tff, int(1.5 * steps_per_tff), steps_per_tff // 50)
        E = np.array([total_energy(x, v, mass, eps) for x, v in zip(X, V)])
        return np.abs(E / E[0] - 1).max()

    unsoftened = [max_energy_error(0.0, n) for n in (500, 2000)]
    softened = [max_energy_error(0.01 * R, n) for n in (500, 2000)]
    assert min(unsoftened) > 1.0                       # > 100% error at both timesteps
    assert softened[0] < 1e-2 and softened[0] / softened[1] > 10


def test_power_law_sphere_sampling():
    # rho ~ r^-alpha means M(<r) ~ r^(3 - alpha): the radius enclosing a fraction q of the mass is R q^(1/(3 - alpha)).
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for alpha in (1.0, 2.0):
        pos, vel, mass = power_law_sphere(20000, M, R, alpha, np.random.default_rng(3))
        assert np.allclose(lagrangian_radii(pos, mass, q), R * q ** (1 / (3 - alpha)), rtol=0.03)


def test_virial_velocities_hit_the_requested_ratio():
    pos, vel, mass = uniform_sphere(500, M, R, np.random.default_rng(4))
    vel = virial_velocities(pos, mass, 0.01 * R, 0.5, np.random.default_rng(5))
    assert np.isclose(virial_ratio(pos, vel, mass, 0.01 * R), 0.5, rtol=1e-12)
    assert np.allclose(momentum(vel, mass), 0, atol=1e-12 * np.sum(mass * np.linalg.norm(vel, axis=1)))


def mass_lost(pos, vel, mass, eps=0.01 * R):
    """Unbound mass fraction after 3 t_ff, at N = 500 and dt = t_ff / 1000."""
    t, X, V = leapfrog(pos, vel, mass, eps, T_FF / 1000, 3000, 3000)
    return unbound_fraction(X[-1], V[-1], mass, eps)


def test_warm_start_prevents_ejection():
    # Sylos Labini (2012): no ejection for an initial virial ratio Q0 >~ 0.5. Measured: <= 0.2% (seeds 0-2),
    # against 13-20% for the same clouds started cold.
    pos, vel, mass = uniform_sphere(500, M, R, np.random.default_rng(0))
    warm = virial_velocities(pos, mass, 0.01 * R, 0.5, np.random.default_rng(1))
    assert mass_lost(pos, vel, mass) > 0.10
    assert mass_lost(pos, warm, mass) < 0.01


def test_centrally_concentrated_cloud_barely_ejects():
    # Sylos Labini (2013): ~1% ejection for rho ~ r^-alpha with alpha >= 2, because shells no longer arrive at the
    # center together (inner ones fall first). Measured: 1.0-2.4% (seeds 0-2).
    pos, vel, mass = power_law_sphere(500, M, R, 2.0, np.random.default_rng(0))
    assert mass_lost(pos, vel, mass) < 0.04
