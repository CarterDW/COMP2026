"""Conserved quantities. Each takes a single snapshot: pos, vel (N, 3) and mass (N,)."""
import numpy as np
from nbody.gravity import potential_energy, potentials


def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass * np.sum(vel**2, axis=-1))


def total_energy(pos, vel, mass, eps):
    return kinetic_energy(vel, mass) + potential_energy(pos, mass, eps)


def momentum(vel, mass):
    return np.sum(mass[:, None] * vel, axis=0)


def angular_momentum(pos, vel, mass):
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)


def virial_ratio(pos, vel, mass, eps):
    """2K / |W|: equals 1 for a system in virial equilibrium."""
    return 2 * kinetic_energy(vel, mass) / abs(potential_energy(pos, mass, eps))


def lagrangian_radii(pos, mass, fractions):
    """Radii about the center of mass enclosing the given fractions of the total mass."""
    center = np.sum(mass[:, None] * pos, axis=0) / mass.sum()
    r = np.linalg.norm(pos - center, axis=1)
    order = np.argsort(r)
    enclosed = np.cumsum(mass[order]) / mass.sum()
    return r[order][np.searchsorted(enclosed, fractions)]


def unbound_fraction(pos, vel, mass, eps):
    """Mass fraction with positive specific energy (1/2) v^2 + phi, i.e. escaping the rest of the system."""
    unbound = 0.5 * np.sum(vel**2, axis=1) + potentials(pos, mass, eps) > 0
    return mass[unbound].sum() / mass.sum()
