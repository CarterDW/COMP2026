"""Sink particles (simplified from Bate, Bonnell & Price 1995).

A sink replaces gas that has collapsed below the resolution limit with one point mass. Each sink carries mass,
position, velocity and spin (its internal angular momentum about its own center of mass). Creation and accretion
put the sink at the center of mass of everything it swallows, with the summed momentum, and add the leftover
angular momentum to its spin. That conserves mass, linear momentum and total angular momentum exactly.
"""
import numpy as np
from nbody.units import G


class Sinks:
    def __init__(self):
        self.pos, self.vel = np.zeros((0, 3)), np.zeros((0, 3))
        self.mass, self.spin = np.zeros(0), np.zeros((0, 3))

    def __len__(self):
        return len(self.mass)

    def angular_momentum(self):
        """Orbital plus spin angular momentum of all sinks about the origin."""
        return np.sum(self.mass[:, None] * np.cross(self.pos, self.vel), axis=0) + self.spin.sum(axis=0)


def _merge(m0, x0, v0, s0, m, x, v):
    """Combine a body (m0, x0, v0, spin s0) with particles (m, x, v): returns the merged mass, position, velocity
    and spin about the new center of mass, so that mass, momentum and angular momentum are unchanged."""
    M = m0 + m.sum()
    X = (m0 * x0 + m @ x) / M
    V = (m0 * v0 + m @ v) / M
    S = s0 + m0 * np.cross(x0 - X, v0 - V) + np.sum(m[:, None] * np.cross(x - X, v - V), axis=0)
    return M, X, V, S


def try_create_sink(sinks, pos, vel, mass, rho, thermal, r_acc, rho_sink):
    """Turn the densest gas particle and its neighbours within r_acc into a sink, if

    1. its density exceeds rho_sink,
    2. it is not inside an existing sink's accretion radius (that gas is the sink's to accrete), and
    3. the group within r_acc is gravitationally bound: kinetic (in its own frame) + thermal + self-gravity < 0.

    thermal is each particle's specific thermal energy. Returns the indices of gas particles removed (may be empty).
    """
    i = np.argmax(rho)
    if rho[i] < rho_sink:
        return np.zeros(0, dtype=int)
    if len(sinks) and np.min(np.linalg.norm(sinks.pos - pos[i], axis=1)) < 2 * r_acc:
        return np.zeros(0, dtype=int)
    group = np.flatnonzero(np.linalg.norm(pos - pos[i], axis=1) < r_acc)
    m, x, v = mass[group], pos[group], vel[group]
    v_cm = m @ v / m.sum()
    kinetic = 0.5 * np.sum(m * np.sum((v - v_cm) ** 2, axis=1))
    a, b = np.triu_indices(len(group), k=1)
    self_gravity = -G * np.sum(m[a] * m[b] / np.linalg.norm(x[a] - x[b], axis=1))
    if kinetic + np.sum(m * thermal[group]) + self_gravity >= 0:
        return np.zeros(0, dtype=int)
    M, X, V, S = _merge(0.0, np.zeros(3), np.zeros(3), np.zeros(3), m, x, v)
    sinks.mass, sinks.pos = np.append(sinks.mass, M), np.vstack([sinks.pos, X])
    sinks.vel, sinks.spin = np.vstack([sinks.vel, V]), np.vstack([sinks.spin, S])
    return group


def accrete(sinks, pos, vel, mass, r_acc):
    """Each gas particle within r_acc of its nearest sink is accreted if it is bound to that sink and has less
    specific angular momentum than a circular orbit at r_acc (so gas orbiting inside r_acc is not swallowed
    just for being there). Returns the indices of gas particles removed."""
    if len(sinks) == 0:
        return np.zeros(0, dtype=int)
    d = np.linalg.norm(pos[:, None, :] - sinks.pos[None, :, :], axis=2)          # (N_gas, N_sinks)
    nearest = np.argmin(d, axis=1)
    removed = []
    for s in range(len(sinks)):
        cand = np.flatnonzero((nearest == s) & (d[:, s] < r_acc))
        dx, dv = pos[cand] - sinks.pos[s], vel[cand] - sinks.vel[s]
        r = np.linalg.norm(dx, axis=1)
        bound = 0.5 * np.sum(dv**2, axis=1) - G * sinks.mass[s] / r < 0
        low_j = np.linalg.norm(np.cross(dx, dv), axis=1) < np.sqrt(G * sinks.mass[s] * r_acc)
        take = cand[bound & low_j]
        if len(take):
            sinks.mass[s], sinks.pos[s], sinks.vel[s], sinks.spin[s] = _merge(
                sinks.mass[s], sinks.pos[s], sinks.vel[s], sinks.spin[s], mass[take], pos[take], vel[take])
            removed.append(take)
    return np.concatenate(removed) if removed else np.zeros(0, dtype=int)


def jeans_resolution_density(cs, particle_mass, n_ngb=58):
    """Highest density at which SPH still resolves the Jeans mass with >= 2 n_ngb particles (Bate & Burkert 1997).

    M_J = (pi^(5/2) / 6) cs^3 / (G^(3/2) rho^(1/2)). Above this density collapse is numerical, not physical,
    so sinks must form here.
    """
    return ((np.pi**2.5 / 6) * cs**3 / (G**1.5 * 2 * n_ngb * particle_mass)) ** 2
