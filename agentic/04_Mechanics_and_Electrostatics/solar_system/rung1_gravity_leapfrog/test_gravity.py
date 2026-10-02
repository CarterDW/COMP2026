"""Rung 1: the softened force law is right before anything is integrated."""
import os
import time
import numpy as np
import pytest
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


@pytest.mark.parametrize("n", [50, 600])   # below and above PARALLEL_ABOVE_N: serial and parallel builds
def test_numba_matches_numpy_reference(n):
    from nbody.gravity import accelerations_numpy, potential_energy_numpy
    pos, mass, eps = rng.normal(size=(n, 3)), rng.uniform(0.1, 1, n), 0.02
    assert np.allclose(accelerations(pos, mass, eps), accelerations_numpy(pos, mass, eps), rtol=1e-12, atol=0)
    assert np.isclose(potential_energy(pos, mass, eps), potential_energy_numpy(pos, mass, eps), rtol=1e-12)


@pytest.mark.parametrize("n", [50, 600])
def test_per_particle_potentials_sum_to_potential_energy(n):
    from nbody.gravity import potentials
    pos, mass, eps = rng.normal(size=(n, 3)), rng.uniform(0.1, 1, n), 0.02
    assert np.isclose(0.5 * np.sum(mass * potentials(pos, mass, eps)), potential_energy(pos, mass, eps), rtol=1e-12)


@pytest.mark.skipif(os.cpu_count() < 4, reason="needs several cores to see a parallel speedup")
def test_parallel_build_really_runs_in_parallel():
    # Guards against a numba cache collision that once made the "parallel" build silently run serial code.
    from nbody.gravity import _acc_serial, _acc_parallel
    pos, mass = rng.normal(size=(1000, 3)), np.full(1000, 1e-3)
    _acc_serial(pos, mass, 0.01), _acc_parallel(pos, mass, 0.01)          # compile outside the timing

    def best_time(f):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            f(pos, mass, 0.01)
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best_time(_acc_parallel) < 0.5 * best_time(_acc_serial)
