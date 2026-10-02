"""Gas-disk effects on embryos and planetesimals, applied after each hybrid step (operator splitting).

The gas disk is the gamma = 1 Lynden-Bell & Pringle solution (nbody/viscous_disk.py) times a dispersal factor
exp(-(t - t0) / TAU_DISPERSAL) for photoevaporation (disk lifetimes of a few Myr, Mamajek 2009).

- Planetesimals: aerodynamic drag (Adachi, Hayashi & Nakazawa 1976), du/dt = -K |du| du with
  K = 3 C_D rho_gas / (8 rho_s s), relative to gas orbiting at v_K (1 - eta), eta = (11/8) (H/R)^2 for
  Sigma ~ R^-1, T ~ R^-1/2. The quadratic decay |du|(t) = |du0| / (1 + K |du0| t) is applied exactly.
- Embryos (big bodies): tidal damping by the disk (Tanaka & Ward 2004),
  t_wave = (M_star / m) (M_star / (Sigma a^2)) (h/a)^4 / Omega, eccentricity e-folds in t_wave / 0.780 and
  inclination in t_wave / 0.544. The radial and vertical velocities relax at twice those rates (each holds half
  of its epicyclic oscillation's energy), as in Cresswell & Nelson (2008). (Type-I migration is
  left out: its strength in real disks is uncertain.)
- Gas accretion onto cores above M_CRIT (Mizuno 1980), following Tanigawa & Tanaka (2016):
  dm/dt = min(m / tau_KH, D Sigma_gap), with tau_KH = 1e9 yr (m / Mearth)^-3 (Ida & Lin 2004),
  D = 0.29 (h/r)^-2 (m/M)^(4/3) r^2 Omega (Tanigawa & Watanabe 2002) and the gas left in the planet's gap
  Sigma_gap = Sigma / (1 + 0.034 K), K = (h/r)^-5 (m/M)^2 / alpha (Kanagawa et al. 2015). All giants share
  the disk's inflow, the global accretion rate of the similarity solution (times the dispersal factor):
  it passes them from the outside in and each takes its rate from what is left. The gas joins with the
  local gas velocity.
- Pebble accretion (optional, disk[6] = 1). Dust grows into pebbles behind a front sweeping outward,
  r_g = (3/16)^(1/3) (G M)^(1/3) (EPS_D Z)^(2/3) t^(2/3), releasing an inward pebble mass flux
  Mdot_F = 2 pi r_g (dr_g/dt) Z Sigma_gas(r_g) (Lambrechts & Johansen 2014). The flux passes the embryos from the
  outside in; each takes Mdot = 2 (tau_s/0.1)^(2/3) Omega r_H^2 Sigma_p with Sigma_p = Mdot_F / (2 pi r v_r) and
  drift speed v_r = 2 tau_s eta v_K, until it reaches the pebble isolation mass 20 Mearth (h/0.05)^3 (M/Msun)
  (Lambrechts et al. 2014). An isolated embryo carves a pressure bump that stops the flux to everything inside it.
- Planetary migration (optional, disk[7] = 1): the non-isothermal type I torque of Paardekooper, Baruteau & Kley
  (2011): Lindblad torque plus barotropic and entropy-related corotation torques, saturated by viscous and thermal
  diffusion. Thermal diffusivity chi = 16 gamma (gamma-1) sigma T^4 / (3 kappa rho^2 H^2 Omega^2) with the Bell & Lin
  (1994) ice-grain opacity kappa = 2e-4 T^2 cm^2/g (valid below ~150 K: everywhere beyond ~3.5 AU here). Gap
  opening reduces the torque to Gamma / (1 + 0.04 K), which blends smoothly into type II (Kanagawa et al. 2018).
  The torque is applied as v_hel -> v_hel exp(Gamma dt / L) (Cresswell & Nelson 2008).
"""
import numpy as np
from numba import njit
from nbody.units import G, M_EARTH, K_OVER_MH
from nbody.viscous_disk import MU_GAS, T_1AU
from nbody.hybrid import step, critical_radii

TAU_DISPERSAL = 2.5e6           # yr
C_D = 1.0                       # drag coefficient of a large body
PLANETESIMAL_SIZE = 50.0 / 1.496e8      # 50 km in AU: drag acts on the real bodies a tracer represents
RHO_SOLID = 1.5 / 5.94e-7       # 1.5 g/cm^3 (icy rock) in Msun/AU^3
RHO_GIANT = 1.33 / 5.94e-7      # Jupiter's mean density, for bodies that are mostly gas
M_CRIT = 10 * M_EARTH
EPS_D = 0.05                    # dust growth efficiency (Lambrechts & Johansen 2014)
STOKES = 0.05                   # pebble Stokes number
Z_PEBBLES = 30.0 / 1700         # solids-to-gas ratio of the outer (icy) disk, as in nbody/viscous_disk.py
GAMMA_GAS = 1.4                 # adiabatic index of H2-dominated disk gas
KAPPA_ICE_CGS = 2e-4            # cm^2 g^-1 K^-2: Bell & Lin (1994) opacity kappa = KAPPA_ICE_CGS T^2 for ice grains
SIGMA_SB_CGS = 5.670374e-5      # erg cm^-2 s^-1 K^-4
CM_PER_AU, S_PER_YR, G_PER_MSUN = 1.495978707e13, 3.15576e7, 1.98841e33
CS2_1AU = K_OVER_MH / MU_GAS * T_1AU           # cs^2 at 1 AU; cs^2 ~ R^-1/2


@njit
def disk_state(R, t, disk):
    """Gas surface density, scale height, Omega, viscosity at radius R and time t.

    disk = (M0, R1, t_nu, M_star, alpha, t0, pebbles_on, migration_on): the similarity solution's parameters, the
    dispersal start, and which optional physics is on.
    """
    M0, R1, t_nu, M_star, alpha, t0 = disk[0], disk[1], disk[2], disk[3], disk[4], disk[5]
    T = 1 + t / t_nu
    sigma = M0 / (2 * np.pi * R1 * R) * T**-1.5 * np.exp(-R / (R1 * T)) * np.exp(-max(t - t0, 0.0) / TAU_DISPERSAL)
    omega = np.sqrt(G * M_star / R**3)
    cs2 = CS2_1AU / np.sqrt(R)
    H = np.sqrt(cs2) / omega
    return sigma, H, omega, alpha * cs2 / omega


@njit
def pebble_flux(t, disk):
    """Mdot_F = 2 pi r_g (dr_g/dt) Z Sigma_gas(r_g) with r_g the pebble growth front (Lambrechts & Johansen 2014)."""
    M_star = disk[3]
    c = (3 / 16) ** (1 / 3) * (G * M_star) ** (1 / 3) * (EPS_D * Z_PEBBLES) ** (2 / 3)
    r_g = c * t ** (2 / 3)
    sigma = disk_state(r_g, t, disk)[0]
    return 2 * np.pi * r_g * (2 / 3) * r_g / t * Z_PEBBLES * sigma


@njit
def pebble_isolation_mass(R, t, disk):
    H = disk_state(R, t, disk)[1]
    return 20 * M_EARTH * (H / R / 0.05) ** 3 * disk[3]


@njit
def pebble_capture_rate(m, R, t, disk, flux):
    """Mass per year a body of mass m at radius R takes from a pebble flux passing it (2D Hill regime)."""
    sigma, H, omega, nu = disk_state(R, t, disk)
    M_star = disk[3]
    r_hill = R * (m / (3 * M_star)) ** (1 / 3)
    v_r = 2 * STOKES * (11 / 8) * (H / R) ** 2 * omega * R
    sigma_p = flux / (2 * np.pi * R * v_r)
    return min(2 * (STOKES / 0.1) ** (2 / 3) * omega * r_hill**2 * sigma_p, flux)


@njit
def accrete_pebbles(Q, v, m, alive, big, t, dt, disk, v_star):
    """Pass the pebble flux from the outside in; each embryo takes its share. Returns True if any mass changed."""
    flux = pebble_flux(t, disk)
    idx = np.flatnonzero(alive & big)
    radius = np.empty(len(idx))
    for k in range(len(idx)):
        radius[k] = np.sqrt(Q[idx[k], 0] ** 2 + Q[idx[k], 1] ** 2)
    changed = False
    for k in np.argsort(-radius):
        i, Rcyl = idx[k], radius[k]
        if flux <= 0.0:
            break
        if m[i] >= pebble_isolation_mass(Rcyl, t, disk):
            flux = 0.0                                   # the isolated embryo's pressure bump stops the pebbles
            break
        rate = pebble_capture_rate(m[i], Rcyl, t, disk, flux)
        omega = np.sqrt(G * disk[3] / Rcyl**3)
        dm = rate * dt                                   # pebbles arrive on circular orbits
        vx, vy = -omega * Q[i, 1] + v_star[0], omega * Q[i, 0] + v_star[1]
        v[i, 0] = (m[i] * v[i, 0] + dm * vx) / (m[i] + dm)
        v[i, 1] = (m[i] * v[i, 1] + dm * vy) / (m[i] + dm)
        v[i, 2] = (m[i] * v[i, 2] + dm * v_star[2]) / (m[i] + dm)
        m[i] += dm
        flux -= rate
        changed = True
    return changed


@njit
def disk_inflow(t, disk):
    """Global accretion rate of the similarity solution, M0 / (2 t_nu) T^(-3/2), times the dispersal factor."""
    M0, t_nu, t0 = disk[0], disk[2], disk[5]
    return 0.5 * M0 / t_nu * (1 + t / t_nu) ** -1.5 * np.exp(-max(t - t0, 0.0) / TAU_DISPERSAL)


@njit
def gas_capture_rate(m, R, t, disk):
    """min(Kelvin-Helmholtz rate, D Sigma_gap) for a core of mass m at radius R (Tanigawa & Tanaka 2016)."""
    sigma, H, omega, nu = disk_state(R, t, disk)
    q, h = m / disk[3], H / R
    K = h**-5 * q**2 / disk[4]
    hydro = 0.29 * h**-2 * q ** (4 / 3) * R**2 * omega * sigma / (1 + 0.034 * K)
    return min(m * (m / M_EARTH) ** 3 / 1e9, hydro)


@njit
def accrete_gas(Q, v, m, R, alive, big, t, dt, disk, v_star):
    """Cores above M_CRIT take gas from the disk inflow, outermost first. Returns True if any mass changed."""
    inflow = disk_inflow(t, disk)
    idx = np.flatnonzero(alive & big & (m >= M_CRIT))
    radius = np.empty(len(idx))
    for k in range(len(idx)):
        radius[k] = np.sqrt(Q[idx[k], 0] ** 2 + Q[idx[k], 1] ** 2)
    changed = False
    for k in np.argsort(-radius):
        if inflow <= 0.0:
            break
        i, Rcyl = idx[k], radius[k]
        rate = min(gas_capture_rate(m[i], Rcyl, t, disk), inflow)
        sigma, H, omega, nu = disk_state(Rcyl, t, disk)
        vg = omega * Rcyl * (1 - (11 / 8) * (H / Rcyl) ** 2)
        dm = rate * dt                                           # gas arrives with the local gas velocity
        M_new = m[i] + dm
        v[i, 0] = (m[i] * v[i, 0] + dm * (-vg * Q[i, 1] / Rcyl + v_star[0])) / M_new
        v[i, 1] = (m[i] * v[i, 1] + dm * (vg * Q[i, 0] / Rcyl + v_star[1])) / M_new
        v[i, 2] = (m[i] * v[i, 2] + dm * v_star[2]) / M_new
        m[i] = M_new
        if m[i] > 2 * M_CRIT:                                    # mostly gas: Jupiter-like density
            R[i] = max(R[i], (3 * m[i] / (4 * np.pi * RHO_GIANT)) ** (1 / 3))
        inflow -= rate
        changed = True
    return changed


@njit
def _saturation_G(p):
    p_break = np.sqrt(8 / (45 * np.pi))
    if p < p_break:
        return (16 / 25) * (45 * np.pi / 8) ** 0.75 * p**1.5
    return 1 - (9 / 25) * (8 / (45 * np.pi)) ** (4 / 3) * p ** (-8 / 3)


@njit
def _saturation_K(p):
    p_break = np.sqrt(28 / (45 * np.pi))
    if p < p_break:
        return (16 / 25) * (45 * np.pi / 28) ** 0.75 * p**1.5
    return 1 - (9 / 25) * (28 / (45 * np.pi)) ** (4 / 3) * p ** (-8 / 3)


@njit
def _saturation_F(p):
    return 1 / (1 + (p / 1.3) ** 2)


@njit
def thermal_diffusivity(R, t, disk):
    """chi = 16 gamma (gamma - 1) sigma T^4 / (3 kappa rho^2 H^2 Omega^2), computed in cgs, returned in AU^2/yr."""
    sigma, H, omega, nu = disk_state(R, t, disk)
    T = T_1AU / np.sqrt(R)
    rho_cgs = sigma / (np.sqrt(2 * np.pi) * H) * G_PER_MSUN / CM_PER_AU**3
    H_cgs, omega_cgs = H * CM_PER_AU, omega / S_PER_YR
    kappa = KAPPA_ICE_CGS * T**2
    chi_cgs = 16 * GAMMA_GAS * (GAMMA_GAS - 1) * SIGMA_SB_CGS * T**4 / (3 * kappa * rho_cgs**2 * H_cgs**2 * omega_cgs**2)
    return chi_cgs / CM_PER_AU**2 * S_PER_YR


@njit
def type1_torque(m, R, t, disk, alpha_slope, beta_slope, chi):
    """Paardekooper et al. (2011) torque on a planet of mass m at R, given -dlnSigma/dlnR, -dlnT/dlnR and chi.

    Returns (total, Lindblad, corotation) torques, before the gap reduction.
    """
    sigma, H, omega, nu = disk_state(R, t, disk)
    q, h = m / disk[3], H / R
    gamma0 = (q / h) ** 2 * sigma * R**4 * omega**2
    Q = 2 * chi / (3 * h**3 * R**2 * omega)
    g = GAMMA_GAS
    # gamma_eff = 2 Q g / (g Q + 1/2 sqrt(2 sqrt(A) - 2 + 2 g^2 Q^2)), A = (g^2 Q^2 + 1)^2 - 16 Q^2 (g - 1).
    # For small Q, 2 sqrt(A) - 2 cancels catastrophically; 2 (A - 1) / (sqrt(A) + 1) is the same and stable.
    A_minus_1 = g**4 * Q**4 + 2 * g * g * Q * Q - 16 * Q * Q * (g - 1)
    inner = 2 * A_minus_1 / (np.sqrt(1 + A_minus_1) + 1) + 2 * g * g * Q * Q
    gamma_eff = 2 * Q * g / (g * Q + 0.5 * np.sqrt(inner))
    xi = beta_slope - (g - 1) * alpha_slope
    lindblad = gamma0 / gamma_eff * (-2.5 - 1.7 * beta_slope + 0.1 * alpha_slope)
    hs_baro = gamma0 / gamma_eff * 1.1 * (1.5 - alpha_slope)
    hs_ent = gamma0 / gamma_eff * 7.9 * xi / gamma_eff
    lin_baro = gamma0 / gamma_eff * 0.7 * (1.5 - alpha_slope)
    lin_ent = gamma0 / gamma_eff * (2.2 - 1.4 / gamma_eff) * xi
    x_s = 1.1 / gamma_eff**0.25 * np.sqrt(q / h)          # horseshoe half-width / R (softening b/h = 0.4)
    p_nu = (2 / 3) * np.sqrt(R**2 * omega * x_s**3 / (2 * np.pi * nu))
    p_chi = np.sqrt(R**2 * omega * x_s**3 / (2 * np.pi * chi))
    Fn, Fc = _saturation_F(p_nu), _saturation_F(p_chi)
    Gn, Gc = _saturation_G(p_nu), _saturation_G(p_chi)
    Kn, Kc = _saturation_K(p_nu), _saturation_K(p_chi)
    corotation = (hs_baro * Fn * Gn + (1 - Kn) * lin_baro
                  + hs_ent * Fn * Fc * np.sqrt(Gn * Gc) + np.sqrt((1 - Kn) * (1 - Kc)) * lin_ent)
    return lindblad + corotation, lindblad, corotation


@njit
def migration_torque(m, R, t, disk):
    """Paardekooper type I torque in our disk, reduced by the gap factor 1 / (1 + 0.04 K) (Kanagawa et al. 2018)."""
    sigma, H, omega, nu = disk_state(R, t, disk)
    T_disk = 1 + t / disk[2]
    alpha_slope = 1 + R / (disk[1] * T_disk)                # Sigma ~ R^-1 exp(-R / (R1 T))
    beta_slope = 0.5                                          # T ~ R^-1/2
    total = type1_torque(m, R, t, disk, alpha_slope, beta_slope, thermal_diffusivity(R, t, disk))[0]
    K = (H / R) ** -5 * (m / disk[3]) ** 2 / disk[4]
    return total / (1 + 0.04 * K)


@njit
def apply_gas(Q, v, m, R, alive, big, t, dt, disk, M_star):
    """Drag, tidal damping and gas accretion over dt. Returns True if any mass changed (rc must be updated).

    Written with scalars only: numba would heap-allocate every small temporary array, once per body per step.
    """
    n = len(m)
    px = py = pz = 0.0
    for i in range(n):
        if alive[i]:
            px += m[i] * v[i, 0]
            py += m[i] * v[i, 1]
            pz += m[i] * v[i, 2]
    vsx, vsy, vsz = -px / M_star, -py / M_star, -pz / M_star            # star; u = v - v_star is heliocentric
    mass_changed = False
    for i in range(n):
        if not alive[i]:
            continue
        x, y, z = Q[i, 0], Q[i, 1], Q[i, 2]
        Rcyl = np.sqrt(x * x + y * y)
        sigma, H, omega, nu = disk_state(Rcyl, t, disk)
        if sigma <= 0.0:
            continue
        vg = omega * Rcyl * (1 - (11 / 8) * (H / Rcyl) ** 2)            # gas: circular, slightly sub-Keplerian
        ugx, ugy = -vg * y / Rcyl, vg * x / Rcyl
        dux, duy, duz = v[i, 0] - vsx - ugx, v[i, 1] - vsy - ugy, v[i, 2] - vsz
        if not big[i]:
            rho = sigma / (np.sqrt(2 * np.pi) * H) * np.exp(-0.5 * (z / H) ** 2)
            K = 3 * C_D * rho / (8 * RHO_SOLID * PLANETESIMAL_SIZE)
            f = 1 / (1 + K * np.sqrt(dux * dux + duy * duy + duz * duz) * dt)   # exact for quadratic drag
            dux, duy, duz = f * dux, f * duy, f * duz
        else:
            t_wave = (M_star / m[i]) * (M_star / (sigma * Rcyl**2)) * (H / Rcyl) ** 4 / omega
            rx, ry = x / Rcyl, y / Rcyl
            ur = (v[i, 0] - vsx) * rx + (v[i, 1] - vsy) * ry               # radial velocity (gas has none)
            c = (1 - np.exp(-dt * 2 * 0.780 / t_wave)) * ur                # radial velocity carries half the
            dux, duy = dux - c * rx, duy - c * ry                         # epicycle: damp it at twice the e rate
            duz *= np.exp(-dt * 2 * 0.544 / t_wave)                         # likewise v_z for the inclination
        v[i, 0], v[i, 1], v[i, 2] = ugx + dux + vsx, ugy + duy + vsy, duz + vsz
        if big[i] and disk[7] > 0:                              # migration: torque on the heliocentric velocity
            hx, hy, hz = v[i, 0] - vsx, v[i, 1] - vsy, v[i, 2] - vsz
            L = m[i] * abs(x * hy - y * hx)
            f = np.exp(migration_torque(m[i], Rcyl, t, disk) * dt / L)
            v[i, 0], v[i, 1], v[i, 2] = hx * f + vsx, hy * f + vsy, hz * f + vsz
    mass_changed = accrete_gas(Q, v, m, R, alive, big, t, dt, disk, np.array([vsx, vsy, vsz]))
    if disk[6] > 0:
        mass_changed = accrete_pebbles(Q, v, m, alive, big, t, dt, disk, np.array([vsx, vsy, vsz])) or mass_changed
    return mass_changed


@njit
def evolve(Q, v, m, R, alive, big, rc, mu, M_star, dt, n_steps, t_start, disk, log, n_log, r_out):
    """n_steps hybrid steps with gas effects. Bodies beyond r_out (AU) or inside 1 AU (accreted by the inner disk
    or star) are removed. Returns n_log; rc is updated in place whenever masses change."""
    t = t_start
    for k in range(n_steps):
        n_before = n_log
        n_log = step(Q, v, m, R, alive, big, rc, mu, M_star, dt, log, n_log, t)
        t += dt
        changed = apply_gas(Q, v, m, R, alive, big, t, dt, disk, M_star)
        for i in range(len(m)):
            if alive[i]:
                r2 = Q[i, 0] ** 2 + Q[i, 1] ** 2 + Q[i, 2] ** 2
                if r2 > r_out**2 or r2 < 1.0:
                    alive[i] = False
                    m[i] = 0.0
        if changed or n_log > n_before:
            rc[:] = critical_radii(Q, v, m, alive, mu, dt)
    return n_log
