"""Rung 1: the softened force law is right before anything is integrated."""
import numpy as np
from nbody.gravity import accelerations, potential_energy
from nbody.units import G

rng = np.random.default_rng(0)


def test_two_body_newtonian():
    pos = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    mass = np.array([1.0, 3e-6])
    acc = accelerations(pos, mass, eps=0.0)
    assert np.allclose(acc[0], [G * 3e-6 / 4, 0, 0], rtol=1e-14)   # pulled toward body 1, +x
    assert np.allclose(acc[1], [-G * 1.0 / 4, 0, 0], rtol=1e-14)   # pulled toward body 0, -x


def test_softening_finite_at_zero_separation():
    pos = np.zeros((2, 3))
    acc = accelerations(pos, np.ones(2), eps=0.1)
    assert np.all(acc == 0)


def test_plummer_force_at_r_equals_eps():
    eps = 0.1
    pos = np.array([[0.0, 0, 0], [eps, 0, 0]])
    acc = accelerations(pos, np.array([1.0, 0.0]), eps)
    assert np.isclose(acc[1, 0], -G / (2**1.5 * eps**2), rtol=1e-14)


def test_softening_correction_is_three_halves_eps2_over_r2():
    # Far away, |a| / (G m / r^2) = (1 + eps^2/r^2)^(-3/2) ~ 1 - (3/2) (eps/r)^2.
    eps, r = 1e-3, 1.0
    pos = np.array([[0.0, 0, 0], [r, 0, 0]])
    acc = accelerations(pos, np.array([1.0, 0.0]), eps)
    deviation = 1 - abs(acc[1, 0]) / (G / r**2)
    assert np.isclose(deviation, 1.5 * (eps / r)**2, rtol=1e-3)


def test_newtons_third_law():
    pos, mass = rng.normal(size=(50, 3)), rng.uniform(0.1, 1, 50)
    force = mass[:, None] * accelerations(pos, mass, eps=0.01)
    assert np.all(np.abs(force.sum(axis=0)) < 1e-13 * np.abs(force).sum())


def test_force_is_minus_gradient_of_potential():
    # accelerations and potential_energy are coded independently; central differences tie them together.
    pos, mass, eps, h = rng.normal(size=(10, 3)), rng.uniform(0.1, 1, 10), 0.05, 1e-5
    force = mass[:, None] * accelerations(pos, mass, eps)
    grad = np.zeros_like(pos)
    for i in range(10):
        for k in range(3):
            step = np.zeros_like(pos)
            step[i, k] = h
            grad[i, k] = (potential_energy(pos + step, mass, eps) - potential_energy(pos - step, mass, eps)) / (2 * h)
    assert np.allclose(force, -grad, rtol=1e-6, atol=1e-8 * np.abs(force).max())
