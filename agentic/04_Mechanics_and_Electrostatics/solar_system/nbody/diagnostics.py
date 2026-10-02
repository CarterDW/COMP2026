"""Conserved quantities. Each takes a single snapshot: pos, vel (N, 3) and mass (N,)."""
import numpy as np
from nbody.gravity import potential_energy


def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass * np.sum(vel**2, axis=-1))


def total_energy(pos, vel, mass, eps):
    return kinetic_energy(vel, mass) + potential_energy(pos, mass, eps)


def momentum(vel, mass):
    return np.sum(mass[:, None] * vel, axis=0)


def angular_momentum(pos, vel, mass):
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)
