"""Direct-sum Plummer-softened gravity: phi_ij = -G m_j / sqrt(r_ij^2 + eps^2).

pos is (N, 3) in AU, mass is (N,) in Msun, eps is the softening length in AU.

`accelerations` and `potential_energy` are numba loops; they are what the integrators use. Each loop is
compiled twice, serial and parallel over particles, because starting threads costs ~60 us per call: for
small N (a two-body orbit) that overhead is far larger than the work. Both give the same answer.

The `_numpy` versions are short vectorized references (O(N^2) memory, so N <~ 2000) that the tests
check the fast versions against.
"""
import numpy as np
from numba import njit, prange
from nbody.units import G


def _accelerations(pos, mass, eps):
    """a_i = G sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2), shape (N, 3)."""
    n = len(mass)
    acc = np.zeros((n, 3))
    for i in prange(n):
        ax = ay = az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
            w = mass[j] * (dx * dx + dy * dy + dz * dz + eps * eps) ** -1.5
            ax += w * dx
            ay += w * dy
            az += w * dz
        acc[i, 0], acc[i, 1], acc[i, 2] = G * ax, G * ay, G * az
    return acc


def _potential_energy(pos, mass, eps):
    """W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2)."""
    n = len(mass)
    total = 0.0
    for i in prange(n):
        row = 0.0
        for j in range(i + 1, n):
            dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
            row += mass[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
        total += mass[i] * row
    return -G * total


def _potentials(pos, mass, eps):
    """phi_i = -G sum_{j != i} m_j / sqrt(r_ij^2 + eps^2), shape (N,). W = (1/2) sum_i m_i phi_i."""
    n = len(mass)
    phi = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
            s += mass[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
        phi[i] = -G * s
    return phi


# Only the serial builds are cached to disk: numba names its cache after the Python function, so a cached
# parallel build of the same function would collide with the serial one and silently load serial code.
PARALLEL_ABOVE_N = 256
_acc_serial, _acc_parallel = njit(cache=True)(_accelerations), njit(parallel=True)(_accelerations)
_pot_serial, _pot_parallel = njit(cache=True)(_potential_energy), njit(parallel=True)(_potential_energy)
_phi_serial, _phi_parallel = njit(cache=True)(_potentials), njit(parallel=True)(_potentials)


def accelerations(pos, mass, eps):
    """a_i = G sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2), shape (N, 3)."""
    return (_acc_parallel if len(mass) > PARALLEL_ABOVE_N else _acc_serial)(pos, mass, eps)


def potential_energy(pos, mass, eps):
    """W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2)."""
    return (_pot_parallel if len(mass) > PARALLEL_ABOVE_N else _pot_serial)(pos, mass, eps)


def potentials(pos, mass, eps):
    """Gravitational potential at each particle from all the others, shape (N,)."""
    return (_phi_parallel if len(mass) > PARALLEL_ABOVE_N else _phi_serial)(pos, mass, eps)


def accelerations_numpy(pos, mass, eps):
    dx = pos[None, :, :] - pos[:, None, :]          # dx[i, j] = x_j - x_i
    r2 = np.sum(dx**2, axis=-1) + eps**2
    np.fill_diagonal(r2, np.inf)                    # no self-force
    return G * np.einsum("ij,ijk->ik", mass[None, :] * r2**-1.5, dx)


def potential_energy_numpy(pos, mass, eps):
    i, j = np.triu_indices(len(mass), k=1)
    r2 = np.sum((pos[j] - pos[i])**2, axis=-1) + eps**2
    return -G * np.sum(mass[i] * mass[j] / np.sqrt(r2))
