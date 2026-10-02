"""5a: the viscous disk that hands the Rung 4 disk to planet formation."""
import numpy as np
from nbody.viscous_disk import (LyndenBellPringle, scale_radius_from, viscosity, solid_surface_density, isolation_mass,
                                T_1AU, SNOW_LINE, Z_ROCK, Z_ICE)
from nbody.units import G
from initial_disk import planet_forming_disk, rung4_disk          # this rung's own module (pytest puts its folder on sys.path)

DISK = LyndenBellPringle(0.1, 100.0, 0.8, 1e-3)


def test_temperature_law_gives_viscosity_linear_in_radius():
    # T ~ R^-1/2 and nu = alpha cs^2 / Omega make nu ~ R: the gamma = 1 similarity case.
    assert np.isclose(viscosity(20.0, 0.8, 1e-3) / viscosity(10.0, 0.8, 1e-3), 2.0, rtol=1e-12)


def test_similarity_solution_solves_the_viscous_diffusion_equation():
    # dSigma/dt = (3/R) d/dR [ R^(1/2) d/dR (nu Sigma R^(1/2)) ], checked with central differences.
    R, t, dR, dt = np.linspace(5, 400, 200), 3e6, 1e-3, 10.0
    nu = lambda r: DISK.nu1 * r / DISK.R1
    flux = lambda r: np.sqrt(r) * (nu(r + dR) * DISK.sigma(r + dR, t) * np.sqrt(r + dR)
                                   - nu(r - dR) * DISK.sigma(r - dR, t) * np.sqrt(r - dR)) / (2 * dR)
    rhs = 3 / R * (flux(R + dR) - flux(R - dR)) / (2 * dR)
    lhs = (DISK.sigma(R, t + dt) - DISK.sigma(R, t - dt)) / (2 * dt)
    assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-6 * np.abs(lhs).max())


def test_mass_drains_onto_the_star_while_angular_momentum_is_conserved():
    R = np.geomspace(1e-3, 2e4, 200001)
    J = lambda t: np.trapz(2 * np.pi * R * DISK.sigma(R, t) * np.sqrt(G * DISK.M_star * R), R)
    for t in (0.0, 1e6, 1e7):
        assert np.isclose(np.trapz(2 * np.pi * R * DISK.sigma(R, t), R), DISK.mass(t), rtol=1e-4)
    assert np.isclose(J(1e7), J(0.0), rtol=1e-4) and DISK.mass(1e7) < 0.8 * DISK.mass(0.0)
    dMdt = (DISK.mass(1e6 + 1e3) - DISK.mass(1e6 - 1e3)) / 2e3
    assert np.isclose(DISK.accretion_rate(1e6), -dMdt, rtol=1e-6)


def test_scale_radius_reproduces_mass_and_angular_momentum():
    M, J, M_star = rung4_disk()
    disk = planet_forming_disk()
    R = np.geomspace(1e-3, 3e4, 200001)
    J_fit = np.trapz(2 * np.pi * R * disk.sigma(R, 0.0) * np.sqrt(G * M_star * R), R)
    assert np.isclose(disk.mass(0.0), M, rtol=1e-12) and np.isclose(J_fit, J, rtol=1e-4)


def test_snow_line_and_solids():
    assert np.isclose(T_1AU / np.sqrt(SNOW_LINE), 170.0, rtol=0.01)          # water ice condenses below ~170 K
    s = solid_surface_density(np.array([1.0, 5.0]), np.array([1.0, 1.0]))
    assert np.allclose(s, [Z_ROCK, Z_ICE]) and np.isclose(Z_ICE / Z_ROCK, 4.2, rtol=0.01)


def test_isolation_mass_fills_its_feeding_zone():
    # M_iso is the solid mass in an annulus of width b Hill radii of M_iso itself.
    R, sigma, M_star, b = 5.0, 5e-7, 0.8, 10.0
    M = isolation_mass(R, sigma, M_star, b)
    hill = R * (M / (3 * M_star)) ** (1 / 3)
    assert np.isclose(M, 2 * np.pi * R * b * hill * sigma, rtol=1e-12)
