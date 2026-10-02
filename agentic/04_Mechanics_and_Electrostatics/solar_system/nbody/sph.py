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
def smoothing_lengths(pos, mass, h_guess, tol=1e-4, max_iter=100):
    """Solve h_i = ETA (m_i / rho_i(h_i))^(1/3) for each particle by damped fixed-point iteration.

    Returns h and the number of iterations the slowest particle needed (== max_iter means not converged).
    """
    n = len(mass)
    h = h_guess.copy()
    iters = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        hi = h[i]
        for k in range(max_iter):
            h_new = ETA * (mass[i] / _density_at(i, pos, mass, hi)) ** (1.0 / 3.0)
            iters[i] = k + 1
            if abs(h_new - hi) < tol * hi:
                hi = h_new
                break
            hi = 0.5 * (hi + h_new)      # damping: plain iteration can oscillate at the cloud's edge
        h[i] = hi
    return h, iters.max()


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
    """

    def __init__(self, eos, cs=None, gamma=None, alpha=1.0, beta=2.0):
        assert eos in ("isothermal", "adiabatic"), eos
        assert (eos == "isothermal") == (cs is not None), "isothermal gas needs cs (and only it)"
        assert (eos == "adiabatic") == (gamma is not None), "adiabatic gas needs gamma (and only it)"
        self.eos, self.cs, self.gamma, self.alpha, self.beta = eos, cs, gamma, alpha, beta

    def pressure_and_sound_speed(self, rho, u):
        if self.eos == "isothermal":
            return self.cs**2 * rho, np.full(len(rho), self.cs)
        assert np.all(u >= 0), "negative internal energy"
        return (self.gamma - 1) * rho * u, np.sqrt(self.gamma * (self.gamma - 1) * u)

    def thermal_energy(self, mass, u):
        """Total U = sum m u. For isothermal gas u = (3/2) cs^2 (ideal monatomic-like thermal energy, constant)."""
        if self.eos == "isothermal":
            return 1.5 * self.cs**2 * mass.sum()
        return np.sum(mass * u)
