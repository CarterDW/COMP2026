"""Smoothed particle hydrodynamics (Monaghan 1992 "vanilla" SPH), direct O(N^2) neighbour search.

Each gas particle i has a smoothing length h_i: its "size". Fields are kernel-weighted sums over neighbours
within 2 h. Forces use the pairwise-symmetric form

    dv_i/dt = - sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) grad_i W_ij
    du_i/dt =   sum_j m_j (P_i/rho_i^2 + Pi_ij / 2)       v_ij . grad_i W_ij

with grad_i W_ij averaged over h_i and h_j, so that pair forces are equal and opposite: momentum and angular
momentum are conserved exactly. Relabeling i <-> j shows sum_i m_i du_i/dt = -sum_i m_i v_i . dv_i/dt, so total
energy is conserved exactly too. The energy equation deliberately uses only the particle's own P_i: the fully
symmetric form (with P_j/rho_j^2 as well) lets a neighbour's pressure cool a particle to negative u.
Pi_ij is Monaghan artificial viscosity, the dissipation that turns converging flows (shocks) into heat.
"""
import numpy as np
from numba import njit, prange

ETA = 1.2   # h = ETA (m / rho)^(1/3): about 58 neighbours inside 2h


@njit(cache=True)
def kernel(r, h):
    """Cubic spline W(r, h) in 3D, support 2h, normalized to 1."""
    q = r / h
    if q < 1.0:
        return (1.0 - 1.5 * q * q + 0.75 * q * q * q) / (np.pi * h**3)
    if q < 2.0:
        return 0.25 * (2.0 - q) ** 3 / (np.pi * h**3)
    return 0.0


@njit(cache=True)
def kernel_derivative(r, h):
    """dW/dr for the cubic spline."""
    q = r / h
    if q < 1.0:
        return (-3.0 * q + 2.25 * q * q) / (np.pi * h**4)
    if q < 2.0:
        return -0.75 * (2.0 - q) ** 2 / (np.pi * h**4)
    return 0.0


@njit(cache=True)
def _solve_h(dist, nbr_mass, own_mass, h_max):
    """Solve h^3 rho(h) = ETA^3 m, i.e. h = ETA (m / rho(h))^(1/3), by bisection on (0, h_max].

    Each neighbour adds m_j h^3 W(r_j, h) = (m_j / pi) w(r_j / h), which only grows with h, so h^3 rho(h) is
    increasing: the root is unique, and at h -> 0 (only the particle itself) h^3 rho = m/pi < ETA^3 m.
    dist/nbr_mass list the candidate neighbours, including the particle itself at distance 0.
    Returns h and whether the root lies inside (0, h_max] (False: the neighbour list is too short).
    """
    target = ETA**3 * own_mass
    total = 0.0
    for k in range(len(dist)):
        total += nbr_mass[k] * kernel(dist[k], h_max)
    if h_max**3 * total < target:
        return h_max, False
    lo, hi = 0.0, h_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        total = 0.0
        for k in range(len(dist)):
            total += nbr_mass[k] * kernel(dist[k], mid)
        if mid**3 * total < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * hi:
            break
    return 0.5 * (lo + hi), True


@njit(cache=True)
def _density_at(i, pos, mass, h):
    rho = 0.0
    for j in range(len(mass)):
        dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
        rho += mass[j] * kernel(np.sqrt(dx * dx + dy * dy + dz * dz), h)
    return rho


@njit(parallel=True, cache=True)
def density(pos, mass, h):
    """rho_i = sum_j m_j W(r_ij, h_i), including the particle itself."""
    rho = np.zeros(len(mass))
    for i in prange(len(mass)):
        rho[i] = _density_at(i, pos, mass, h[i])
    return rho


@njit(parallel=True, cache=True)
def smoothing_lengths(pos, mass, h_guess):
    """h_i = ETA (m_i / rho_i)^(1/3) for each particle, solved by bisection over all other particles (O(N^2)).

    h_guess is unused (kept so the call matches the tree version). Returns h and a flag that is True if every
    particle's root was found.
    """
    n = len(mass)
    h = np.zeros(n)
    found = np.ones(n, dtype=np.bool_)
    for i in prange(n):
        dist = np.empty(n)
        for j in range(n):
            dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
            dist[j] = np.sqrt(dx * dx + dy * dy + dz * dz)
        h[i], found[i] = _solve_h(dist, mass, mass[i], 1e3 * dist.max())     # every particle is a candidate
    return h, found.all()


@njit(parallel=True, cache=True)
def hydro_forces(pos, vel, mass, h, rho, P, c, alpha, beta):
    """Pressure + artificial-viscosity accelerations (N, 3), du/dt (N,), and each particle's viscous signal
    speed max_j(c_ij + 1.2 (alpha c_ij + beta |mu_ij|)), used for the Courant timestep."""
    n = len(mass)
    acc = np.zeros((n, 3))
    dudt = np.zeros(n)
    signal = np.zeros(n)
    for i in prange(n):
        signal[i] = c[i]
        for j in range(n):
            if j == i:
                continue
            dx, dy, dz = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], pos[i, 2] - pos[j, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r >= 2.0 * max(h[i], h[j]):
                continue
            dWdr = 0.5 * (kernel_derivative(r, h[i]) + kernel_derivative(r, h[j]))
            gx, gy, gz = dWdr * dx / r, dWdr * dy / r, dWdr * dz / r          # grad_i W_ij
            dvx, dvy, dvz = vel[i, 0] - vel[j, 0], vel[i, 1] - vel[j, 1], vel[i, 2] - vel[j, 2]
            v_dot_r = dvx * dx + dvy * dy + dvz * dz

            visc = 0.0
            if v_dot_r < 0.0:                                                 # approaching: viscosity on
                h_ij, c_ij, rho_ij = 0.5 * (h[i] + h[j]), 0.5 * (c[i] + c[j]), 0.5 * (rho[i] + rho[j])
                mu = h_ij * v_dot_r / (r * r + 0.01 * h_ij * h_ij)
                visc = (-alpha * c_ij * mu + beta * mu * mu) / rho_ij
                signal[i] = max(signal[i], c_ij + 1.2 * (alpha * c_ij + beta * abs(mu)))

            A = P[i] / rho[i] ** 2 + P[j] / rho[j] ** 2 + visc
            acc[i, 0] -= mass[j] * A * gx
            acc[i, 1] -= mass[j] * A * gy
            acc[i, 2] -= mass[j] * A * gz
            dudt[i] += mass[j] * (P[i] / rho[i] ** 2 + 0.5 * visc) * (dvx * gx + dvy * gy + dvz * gz)
    return acc, dudt, signal


class Gas:
    """Equation of state and viscosity parameters.

    eos = "isothermal": P = cs^2 rho. Heat from compression/shocks is radiated away instantly; it is tallied
                        as radiated energy so that the total energy budget can still be checked.
    eos = "adiabatic":  P = (gamma - 1) rho u, with u evolved by du/dt. Heat stays in the gas.
    eos = "barotropic": P = cs^2 rho sqrt(1 + (rho/rho_crit)^(4/3)) (Bate, Bonnell & Bromm 2003): isothermal
                        while the cloud can radiate, stiffening to P ~ rho^(5/3) once it turns opaque above
                        rho_crit (the first hydrostatic core). Heating is tallied as radiated, as for isothermal.
                        Optional protostellar heating: with T_floor and T_1au given (K), gas at distance d from
                        the nearest sink has temperature max(T_floor, T_1au (d / 1 AU)^(-1/2)) (Hayashi 1981 uses
                        T_1au = 280 K); cs^2 scales with that temperature, with cs being the value at T_floor.
    """

    def __init__(self, eos, cs=None, gamma=None, rho_crit=None, T_floor=None, T_1au=None, alpha=1.0, beta=2.0):
        assert eos in ("isothermal", "adiabatic", "barotropic"), eos
        assert (eos in ("isothermal", "barotropic")) == (cs is not None), "cs is required for (only) isothermal/barotropic gas"
        assert (eos == "adiabatic") == (gamma is not None), "gamma is required for (only) adiabatic gas"
        assert (eos == "barotropic") == (rho_crit is not None), "rho_crit is required for (only) barotropic gas"
        assert (T_floor is None) == (T_1au is None), "heating needs both T_floor and T_1au"
        assert T_floor is None or eos == "barotropic", "protostellar heating is implemented for barotropic gas"
        self.eos, self.cs, self.gamma, self.rho_crit, self.alpha, self.beta = eos, cs, gamma, rho_crit, alpha, beta
        self.T_floor, self.T_1au = T_floor, T_1au

    def heated(self):
        return self.T_floor is not None

    def pressure_and_sound_speed(self, rho, u, d_sink=None):
        """d_sink: distance (AU) of each particle to its nearest sink; required (only) when heating is on."""
        assert self.heated() == (d_sink is not None), "pass d_sink exactly when protostellar heating is on"
        if self.eos == "isothermal":
            return self.cs**2 * rho, np.full(len(rho), self.cs)
        if self.eos == "barotropic":
            cs2 = self.cs**2 * np.ones(len(rho))
            if self.heated():
                cs2 = cs2 * np.maximum(1.0, self.T_1au / self.T_floor / np.sqrt(d_sink))
            x = (rho / self.rho_crit) ** (4 / 3)
            dP_drho = cs2 * (np.sqrt(1 + x) + (2 / 3) * x / np.sqrt(1 + x))
            return cs2 * rho * np.sqrt(1 + x), np.sqrt(dP_drho)
        assert np.all(u >= 0), "negative internal energy"
        return (self.gamma - 1) * rho * u, np.sqrt(self.gamma * (self.gamma - 1) * u)

    def thermal_energy(self, mass, u):
        """Total U = sum m u. For isothermal gas u = (3/2) cs^2 (ideal monatomic-like thermal energy, constant)."""
        assert self.eos != "barotropic", "thermal energy of barotropic gas is not tracked"
        if self.eos == "isothermal":
            return 1.5 * self.cs**2 * mass.sum()
        return np.sum(mass * u)


# ---------------------------------------------------------------- tree-based versions (same physics, ~O(N log N))
# The functions above test every pair, O(N^2). These find neighbours with a k-d tree and then visit each
# interacting pair once. Tests check that both give the same density, smoothing lengths and forces.

@njit(parallel=True, cache=True)
def _smoothing_lengths_knn(dist, nbr_mass, own_mass, complete):
    """Smoothing lengths and densities from each particle's K nearest neighbours (sorted distances, self first).

    Unless the lists hold every particle (complete), the kernel support 2h must fit inside the list,
    h <= dist[K-1] / 2. Returns h, rho, and for each particle whether its root was found within its list.
    """
    n, K = dist.shape
    h, rho = np.zeros(n), np.zeros(n)
    fits = np.ones(n, dtype=np.bool_)
    for i in prange(n):
        h_max = 1e3 * dist[i, K - 1] if complete else 0.5 * dist[i, K - 1]
        h[i], fits[i] = _solve_h(dist[i], nbr_mass[i], own_mass[i], h_max)
        for k in range(K):
            rho[i] += nbr_mass[i, k] * kernel(dist[i, k], h[i])
    return h, rho, fits


def neighbour_data(pos, mass, h_guess, k=128):
    """Smoothing lengths and densities from a k-d tree, and the (i, j) pairs with r_ij < 2 max(h_i, h_j), each once.

    Every particle starts with its k nearest neighbours; only particles whose kernel does not fit inside their
    list are re-queried with twice as many, so one isolated particle does not inflate everyone's list.
    h_guess is unused (h is solved from scratch by bisection).
    """
    from scipy.spatial import cKDTree
    n = len(mass)
    tree = cKDTree(pos)
    h, rho = np.empty(n), np.empty(n)
    rows_i, rows_j = [], []
    todo = np.arange(n)
    while len(todo):
        k_eff = min(k, n)
        dist, idx = tree.query(pos[todo], k=k_eff, workers=-1)
        dist, idx = dist.reshape(len(todo), k_eff), idx.reshape(len(todo), k_eff)
        h_t, rho_t, fits = _smoothing_lengths_knn(dist, mass[idx], mass[todo], k_eff == n)
        assert fits.any() or k_eff < n, "too few particles for any smoothing length to hold ETA^3 m"
        done = todo[fits]
        h[done], rho[done] = h_t[fits], rho_t[fits]
        gather = dist[fits] < 2 * h_t[fits][:, None]                 # j inside i's kernel
        rows_i.append(np.repeat(done, k_eff).reshape(len(done), k_eff)[gather])
        rows_j.append(idx[fits][gather])
        todo, k = todo[~fits], 2 * k
    i_idx, j_idx = np.concatenate(rows_i), np.concatenate(rows_j)
    keep = i_idx != j_idx
    a, b = np.minimum(i_idx[keep], j_idx[keep]), np.maximum(i_idx[keep], j_idx[keep])
    pairs = np.unique(a.astype(np.int64) * n + b)                   # union of both gather lists, deduplicated
    return h, rho, np.stack([pairs // n, pairs % n], axis=1)


@njit(cache=True)
def hydro_forces_pairs(pairs, pos, vel, mass, h, rho, P, c, alpha, beta):
    """Same as hydro_forces, but looping once over an explicit (i, j) pair list and updating both particles."""
    n = len(mass)
    acc = np.zeros((n, 3))
    dudt = np.zeros(n)
    signal = c.copy()
    for p in range(len(pairs)):
        i, j = pairs[p, 0], pairs[p, 1]
        dx, dy, dz = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], pos[i, 2] - pos[j, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        dWdr = 0.5 * (kernel_derivative(r, h[i]) + kernel_derivative(r, h[j]))
        gx, gy, gz = dWdr * dx / r, dWdr * dy / r, dWdr * dz / r              # grad_i W_ij = -grad_j W_ij
        dvx, dvy, dvz = vel[i, 0] - vel[j, 0], vel[i, 1] - vel[j, 1], vel[i, 2] - vel[j, 2]
        v_dot_r = dvx * dx + dvy * dy + dvz * dz
        v_dot_g = dvx * gx + dvy * gy + dvz * gz
        visc = 0.0
        if v_dot_r < 0.0:
            h_ij, c_ij, rho_ij = 0.5 * (h[i] + h[j]), 0.5 * (c[i] + c[j]), 0.5 * (rho[i] + rho[j])
            mu = h_ij * v_dot_r / (r * r + 0.01 * h_ij * h_ij)
            visc = (-alpha * c_ij * mu + beta * mu * mu) / rho_ij
            s = c_ij + 1.2 * (alpha * c_ij + beta * abs(mu))
            signal[i] = max(signal[i], s)
            signal[j] = max(signal[j], s)
        A = P[i] / rho[i] ** 2 + P[j] / rho[j] ** 2 + visc
        acc[i, 0] -= mass[j] * A * gx
        acc[i, 1] -= mass[j] * A * gy
        acc[i, 2] -= mass[j] * A * gz
        acc[j, 0] += mass[i] * A * gx
        acc[j, 1] += mass[i] * A * gy
        acc[j, 2] += mass[i] * A * gz
        dudt[i] += mass[j] * (P[i] / rho[i] ** 2 + 0.5 * visc) * v_dot_g
        dudt[j] += mass[i] * (P[j] / rho[j] ** 2 + 0.5 * visc) * v_dot_g    # v_ji . grad_j W_ji = v_ij . grad_i W_ij
    return acc, dudt, signal
