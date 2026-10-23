"""Checkerboard Metropolis for 2D spin models on an L x L periodic square lattice, in numba.

Arrays are (R, L, L): R independent replicas, each with its own temperature.
Black sites (i + j even) only have white neighbours and vice versa, so all sites
of one colour can be updated at once. Every (replica, row) of a colour is a
separate parallel task.

The XY model (spins are angles theta, H = -sum cos(theta_i - theta_j)) is here
now. The Ising model will reuse the same sweep structure with a different site
move. J = 1, k_B = 1, and one sweep is L^2 attempted moves.

Random numbers come from numba's per-thread generators, seeded by seed() below.
Streams are independent across threads but not bit-reproducible, because which
thread gets which row varies from run to run.
"""

import numpy as np
from numba import njit, prange

TWO_PI = 2 * np.pi


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def _xy_site(th, r, i, j, beta, delta):
    """Metropolis move at site (i, j) of replica r, in place. Returns 1 if accepted.

    Proposal theta -> theta + delta * U(-1, 1) is symmetric, so the acceptance is
    min(1, exp(-beta dE)). The local field h = sum_j (cos, sin)(theta_j) gives
    E_i = -h . (cos, sin)(theta_i).
    """
    L = th.shape[1]
    hx = hy = 0.0
    for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
        t = th[r, a % L, b % L]
        hx += np.cos(t)
        hy += np.sin(t)
    old = th[r, i, j]
    new = old + delta * (2 * np.random.random() - 1)
    dE = hx * (np.cos(old) - np.cos(new)) + hy * (np.sin(old) - np.sin(new))
    if dE <= 0 or np.random.random() < np.exp(-beta * dE):
        th[r, i, j] = new % TWO_PI
        return 1
    return 0


@njit(parallel=True, cache=True)
def xy_sweep(th, beta, delta):
    """One sweep, black then white, in place. beta, delta are (R,). Returns acceptance (R,)."""
    R, L, _ = th.shape
    acc_row = np.zeros((R, L))
    for colour in (0, 1):
        for k in prange(R * L):
            r, i = k // L, k % L
            n = 0
            for j in range((i + colour) % 2, L, 2):
                n += _xy_site(th, r, i, j, beta[r], delta[r])
            acc_row[r, i] += n
    return acc_row.sum(axis=1) / (L * L)


@njit(parallel=True, cache=True)
def xy_site_moves(th, beta, delta, i, j, n_moves):
    """n_moves moves at site (i, j) only, in every replica. For testing the site kernel."""
    for r in prange(th.shape[0]):
        for _ in range(n_moves):
            _xy_site(th, r, i, j, beta[r], delta[r])


def xy_energy(theta):
    """Energy per spin, per replica: -(1/N) sum over the 2N bonds of cos(dtheta)."""
    bonds = sum(np.cos(theta - np.roll(theta, -1, ax)) for ax in (1, 2))
    return -bonds.mean(axis=(1, 2))


def wrap(d):
    """Angle differences wrapped to [-pi, pi)."""
    return (d + np.pi) % TWO_PI - np.pi


def vortices(theta):
    """Integer vortex charge on every plaquette, shape (R, L, L).

    Plaquette (i, j) has corners (i, j), (i, j+1), (i+1, j+1), (i+1, j), with x
    along j and y along i. Going around it anticlockwise, the wrapped angle
    differences sum to 2 pi q, with q in {-1, 0, +1}.
    """
    right = np.roll(theta, -1, 2)
    up_right = np.roll(right, -1, 1)
    up = np.roll(theta, -1, 1)
    loop = wrap(right - theta) + wrap(up_right - right) + wrap(up - up_right) + wrap(theta - up)
    return np.rint(loop / TWO_PI).astype(int)


def tune_delta(theta, beta, n_sweeps=200, target=0.5):
    """Warm up while adapting the step size towards the target acceptance. Returns delta (R,).

    Adapting breaks detailed balance, so delta is frozen after this and the
    measurement sweeps use it unchanged. Capped at pi, where the proposal is uniform.
    """
    delta = np.ones_like(beta)
    for n in range(n_sweeps):
        acc = xy_sweep(theta, beta, delta)
        if n % 10 == 9:
            delta = np.minimum(delta * (acc / target) ** 0.5 + 1e-3, np.pi)
    return delta


def run_xy(L, T, n_warm, n_meas, s, ordered=False):
    """Equilibrate and measure the energy. T is one temperature per replica; s seeds the run.

    Returns (theta, u) with u of shape (n_meas, R): the energy per spin after
    every measurement sweep.
    """
    seed(s)
    beta = 1 / np.asarray(T, float)
    R = len(beta)
    theta = np.zeros((R, L, L)) if ordered else np.random.default_rng(s).uniform(0, TWO_PI, (R, L, L))
    delta = tune_delta(theta, beta)
    for _ in range(n_warm):
        xy_sweep(theta, beta, delta)
    u = np.empty((n_meas, R))
    for n in range(n_meas):
        xy_sweep(theta, beta, delta)
        u[n] = xy_energy(theta)
    return theta, u
