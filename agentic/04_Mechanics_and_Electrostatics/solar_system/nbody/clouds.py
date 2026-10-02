"""Initial conditions for collapsing clouds, and the analytic pressureless-collapse solution."""
import numpy as np
from nbody.units import G
from nbody.gravity import potential_energy


def power_law_sphere(n, M, R, alpha, rng):
    """n equal-mass particles at rest in a sphere with density rho ~ r^-alpha (alpha < 3), center of mass at 0.

    M(<r) = M (r/R)^(3 - alpha), so a uniform deviate u maps to r = R u^(1/(3 - alpha)).
    Returns pos (n, 3), vel (n, 3), mass (n,).
    """
    assert alpha < 3, "mass diverges at the center for alpha >= 3"
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    r = R * rng.uniform(size=n) ** (1 / (3 - alpha))
    pos = r[:, None] * direction
    pos -= pos.mean(axis=0)
    return pos, np.zeros((n, 3)), np.full(n, M / n)


def uniform_sphere(n, M, R, rng):
    """power_law_sphere with alpha = 0: uniform density."""
    return power_law_sphere(n, M, R, 0.0, rng)


def virial_velocities(pos, mass, eps, Q0, rng):
    """Isotropic Gaussian velocities, zero total momentum, scaled so that 2K/|W| = Q0 exactly."""
    vel = rng.normal(size=pos.shape)
    vel -= np.sum(mass[:, None] * vel, axis=0) / mass.sum()
    kinetic = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    return vel * np.sqrt(Q0 * abs(potential_energy(pos, mass, eps)) / (2 * kinetic))


def free_fall_time(M, R):
    """t_ff = sqrt(3 pi / (32 G rho)) = (pi/2) sqrt(R^3 / (2 G M)) for a uniform sphere."""
    return 0.5 * np.pi * np.sqrt(R**3 / (2 * G * M))


def cycloid_radius(t, t_ff):
    """r(t) / r(0) for any shell of a cold uniform sphere, valid for 0 <= t <= t_ff.

    Each shell falls like a radial Kepler orbit: r = r0 cos^2(eta), t = t_ff (2/pi) (eta + sin(eta) cos(eta)).
    Every shell reaches r = 0 at the same time t_ff, because the mass inside each shell never changes.
    """
    eta = np.linspace(0, np.pi / 2, 20001)
    t_of_eta = t_ff * (2 / np.pi) * (eta + np.sin(eta) * np.cos(eta))
    return np.cos(np.interp(t, t_of_eta, eta)) ** 2


def lattice_sphere(n, M, R):
    """About n equal-mass particles on a cubic lattice inside a sphere: uniform density without Poisson noise.

    SPH needs this: randomly placed particles give density estimates with ~40% scatter, which act as spurious
    pressure forces. Returns pos, vel (zeros), mass, with the exact particle count set by the lattice.
    """
    a = R * (4 * np.pi / 3 / n) ** (1 / 3)
    k = int(np.ceil(R / a)) + 1
    g = (np.arange(-k, k) + 0.5) * a          # offset by a/2: symmetric about the origin
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    pos = pos[np.linalg.norm(pos, axis=1) < R]
    return pos, np.zeros_like(pos), np.full(len(pos), M / len(pos))
