"""1D radial transport of pebbles through the inner gas disk, and their conversion into planetesimals.

Pebbles (rock and ice, moving together) live on log-spaced rings between R_MIN and R_OUT on top of the gas disk of
nbody/gas_effects.py (disk_state). Each step:
- Stokes number St = min(ST_GROWTH, St_frag): the Lambrechts & Johansen (2014) pebble size (itself the drift-limited
  size at the growth front, where the pebbles form), broken down where collisions exceed the fragmentation speed,
  St_frag = 0.37 v_frag^2 / (3 alpha_turb cs^2) (Birnstiel et al. 2012, as in Drazkowska et al. 2016), with v_frag
  10 m/s for icy and 1-10 m/s for dry pebbles. The local drift limit caps growth, not pebbles arriving from outside,
  so it is not applied.
- Radial velocity with the dust's back-reaction on the gas (Nakagawa et al. 1986; Ida & Guillot 2016 Eq. 2):
  v_r = (-2 St L^2 eta v_K + L u_g) / (1 + L^2 St^2),  L = 1 / (1 + eps),  eta v_K = (1/2) (H/r)^2 |dlnP/dlnr| v_K,
  u_g = -3 nu_acc / (2 r) the gas inflow (alpha_acc).
- Turbulent diffusion of the pebble-to-gas ratio Z = Sigma_p / Sigma_g with D = alpha_turb cs H / (1 + St^2).
- Midplane ratio eps = Z / sqrt((Pi/5)^2 + alpha_turb / (alpha_turb + St)), Pi = eta v_K / cs: settling against
  turbulence (Youdin & Lithwick 2007) plus the streaming instability's own stirring (Li & Youdin 2021).
- Ice sublimates where T > 170 K (inside the snow line); the vapor leaves with the gas and is tallied.
- Streaming instability: where St >= st_min and Z > Z_crit(St, alpha_turb), pebbles turn into planetesimals at
  zeta per orbit (Drazkowska et al. 2016). Z_crit is Lim et al. (2024) Eq. 19 inside its range (0.01 <= St <= 0.1)
  and Li & Youdin (2021) with turbulence elsewhere (LY21 everywhere when fit = FIT_LY21).

Mass enters through the outer edge (inflow(t), Msun/yr) and leaves through the inner edge onto the star. Units:
AU, Msun, yr. The gas follows dlnP/dlnr = -11/4 (Sigma ~ r^-1, T ~ r^-1/2; the exponential taper is negligible
inside 4 AU), as in nbody/gas_effects.py.
"""
import numpy as np
from numba import njit
from nbody.gas_effects import disk_state, CM_PER_AU, S_PER_YR
from nbody.viscous_disk import SNOW_LINE

R_MIN, R_OUT = 0.1, 4.0               # AU
DLNP = 11 / 4                          # |dlnP/dlnr|
ST_GROWTH = 0.05                       # the pebble Stokes number of the growth-front model (nbody/gas_effects.py)
FIT_LY21, FIT_LIM24 = 0, 1
M_PER_S = 100 * S_PER_YR / CM_PER_AU   # 1 m/s in AU/yr


def grid(n):
    """Cell edges, centers and ring areas of n log-spaced cells."""
    edges = np.geomspace(R_MIN, R_OUT, n + 1)
    return edges, np.sqrt(edges[1:] * edges[:-1]), np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


@njit
def z_crit(st, alpha, pi, fit):
    """Pebble-to-gas ratio above which the streaming instability forms planetesimals."""
    if fit == FIT_LIM24 and 0.01 <= st <= 0.1:
        la, ls = np.log10(alpha), np.log10(st)
        return 10 ** (0.15 * la**2 - 0.24 * ls * la - 1.48 * ls + 1.18 * la)
    ls = np.log10(st)
    eps_crit = 2.5 if st < 0.015 else 10 ** (0.48 * ls**2 + 0.87 * ls - 0.11)
    return eps_crit * np.sqrt((pi / 5) ** 2 + alpha / (alpha + st))


@njit
def drift_velocity(st, eps, eta_vk, u_gas):
    lam = 1 / (1 + eps)
    return (-2 * st * lam**2 * eta_vk + lam * u_gas) / (1 + lam**2 * st**2)


@njit
def local_state(r, t, disk, z, icy, v_frag_dry, v_frag_ice):
    """Gas density, St, eps, radial velocity, diffusivity, Omega and Pi of pebbles with ratio z at radius r."""
    sigma_g, H, omega, nu_turb = disk_state(r, t, disk)
    alpha_turb, alpha_acc = disk[8], disk[4]
    cs = H * omega
    vk = omega * r
    eta_vk = 0.5 * (H / r) ** 2 * DLNP * vk
    v_frag = (v_frag_ice if icy else v_frag_dry) * M_PER_S
    st = min(ST_GROWTH, 0.37 * v_frag**2 / (3 * alpha_turb * cs**2))
    pi = eta_vk / cs
    eps = z / np.sqrt((pi / 5) ** 2 + alpha_turb / (alpha_turb + st))
    u_gas = -1.5 * alpha_acc * cs * H / r
    return sigma_g, st, eps, drift_velocity(st, eps, eta_vk, u_gas), alpha_turb * cs * H / (1 + st**2), omega, pi


@njit
def step(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, dt, disk, inflow_rate, inflow_ice_fraction,
         v_frag_dry, v_frag_ice, zeta, st_min, fit, ledger):
    """Advance the pebble surface densities rock, ice (Msun/AU^2) by dt; planetesimals form into plts_*.

    ledger accumulates (inflow, onto the star, sublimated, into planetesimals) in Msun.
    """
    n = len(centers)
    sig_g, st, v, D, omega, pi = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        icy = centers[i] > SNOW_LINE
        sig_g[i] = disk_state(centers[i], t, disk)[0]
        z = (rock[i] + ice[i]) / sig_g[i]
        s, st[i], eps, v[i], D[i], omega[i], pi[i] = local_state(centers[i], t, disk, z, icy, v_frag_dry, v_frag_ice)
    # mass fluxes (Msun/yr, outward positive) through each edge: donor-cell advection (the velocity and density of
    # the cell the flow comes from, so a jump in drift speed, as at the snow line, cannot dam the faster cell) plus
    # diffusion of Z
    flux_rock, flux_ice = np.zeros(n + 1), np.zeros(n + 1)
    for k in range(1, n):
        up = k if v[k] < 0 else k - 1
        if v[k] < 0 and v[k - 1] > 0:                             # flows diverging from this edge: no advection
            up = -1
        vel = v[up] if up >= 0 else 0.0
        sg = 0.5 * (sig_g[k - 1] + sig_g[k])
        dD = 0.5 * (D[k - 1] + D[k]) * sg / (centers[k] - centers[k - 1])
        ring = 2 * np.pi * edges[k]
        flux_rock[k] = ring * (vel * rock[max(up, 0)] - dD * (rock[k] / sig_g[k] - rock[k - 1] / sig_g[k - 1]))
        flux_ice[k] = ring * (vel * ice[max(up, 0)] - dD * (ice[k] / sig_g[k] - ice[k - 1] / sig_g[k - 1]))
    out = min(v[0], 0.0) * 2 * np.pi * edges[0]                   # inner edge: outflow onto the star only
    flux_rock[0], flux_ice[0] = out * rock[0], out * ice[0]
    flux_rock[n] = -inflow_rate * (1 - inflow_ice_fraction)       # outer edge: the pebble flux from outside
    flux_ice[n] = -inflow_rate * inflow_ice_fraction
    ledger[0] += inflow_rate * dt
    ledger[1] -= (flux_rock[0] + flux_ice[0]) * dt
    for i in range(n):
        rock[i] += (flux_rock[i] - flux_rock[i + 1]) * dt / areas[i]
        ice[i] += (flux_ice[i] - flux_ice[i + 1]) * dt / areas[i]
        if centers[i] < SNOW_LINE:                                # ice sublimates; the vapor leaves with the gas
            ledger[2] += ice[i] * areas[i]
            ice[i] = 0.0
        z = (rock[i] + ice[i]) / sig_g[i]
        if st[i] >= st_min and z > z_crit(st[i], disk[8], pi[i], fit):
            f = zeta * omega[i] / (2 * np.pi) * dt                # fraction converted this step
            plts_rock[i] += f * rock[i]
            plts_ice[i] += f * ice[i]
            ledger[3] += f * (rock[i] + ice[i]) * areas[i]
            rock[i] *= 1 - f
            ice[i] *= 1 - f


@njit
def max_timestep(rock, ice, edges, centers, t, disk, v_frag_dry, v_frag_ice, courant):
    """Courant limit for advection and diffusion."""
    dt = np.inf
    for i in range(len(centers)):
        z = (rock[i] + ice[i]) / disk_state(centers[i], t, disk)[0]
        s = local_state(centers[i], t, disk, z, centers[i] > SNOW_LINE, v_frag_dry, v_frag_ice)
        dr = edges[i + 1] - edges[i]
        dt = min(dt, courant * dr / max(abs(s[3]), 1e-30), courant * dr**2 / (2 * s[4]))
    return dt


@njit
def advance(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, t_end, disk, inflow_rate, inflow_ice_fraction,
            v_frag_dry, v_frag_ice, zeta, st_min, fit, ledger, courant):
    """Step from t to t_end at the Courant limit with a fixed inflow; returns the number of steps."""
    n_steps = 0
    while t < t_end:
        dt = min(max_timestep(rock, ice, edges, centers, t, disk, v_frag_dry, v_frag_ice, courant), t_end - t)
        step(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, dt, disk, inflow_rate, inflow_ice_fraction,
             v_frag_dry, v_frag_ice, zeta, st_min, fit, ledger)
        t += dt
        n_steps += 1
    return n_steps
