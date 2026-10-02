"""Direct-sum Plummer-softened gravity: phi_ij = -G m_j / sqrt(r_ij^2 + eps^2).

pos is (N, 3) in AU, mass is (N,) in Msun, eps is the softening length in AU.
"""
import numpy as np
from nbody.units import G


def accelerations(pos, mass, eps):
    """a_i = G sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2), shape (N, 3)."""
    dx = pos[None, :, :] - pos[:, None, :]          # dx[i, j] = x_j - x_i
    r2 = np.sum(dx**2, axis=-1) + eps**2
    np.fill_diagonal(r2, np.inf)                    # no self-force
    return G * np.einsum("ij,ijk->ik", mass[None, :] * r2**-1.5, dx)


def potential_energy(pos, mass, eps):
    """W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2)."""
    i, j = np.triu_indices(len(mass), k=1)
    r2 = np.sum((pos[j] - pos[i])**2, axis=-1) + eps**2
    return -G * np.sum(mass[i] * mass[j] / np.sqrt(r2))
