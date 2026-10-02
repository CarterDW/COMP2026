"""One-zone protostar model: from the sink's accretion history to hydrogen ignition (sub-grid physics for Rung 4b).

The star is an ideal-gas polytrope of index n = 1.5 (fully convective) with mass M and radius R.
- Central temperature (virial theorem + Lane-Emden structure):  T_c = (mu m_H / k) (G M / R) / [(n+1) xi_1 |theta'(xi_1)|]
- Total energy:  E = -(3 / (5 - n)) G M^2 / (2 R);  contraction releases -dE/dt = L - L_nuclear.

Phases:
1. Accretion. The radius is the larger of the initial accretion radius 2.5 Rsun (Mdot / 1e-5 Msun/yr)^0.2 (Offner et al.
   2009) and the deuterium "birthline": D burning acts as a thermostat holding T_c at T_D = 1.5e6 K (Stahler 1988).
   Luminosity: accretion F_ACC G M Mdot / R, plus the photosphere at the Hayashi temperature, plus D burning of the
   freshly accreted deuterium while on the birthline (the thermostat burns what arrives, so no reserve builds up).
2. Pre-main sequence at fixed mass. Kelvin-Helmholtz contraction. Hayashi track: L = 4 pi R^2 sigma T_HAYASHI^4 while
   that exceeds the main-sequence luminosity; then the Henyey track at L = L_ZAMS(M).
3. Hydrogen ignition. The pp chain gives L_H = L_ZAMS (T_c / T_H)^7 (energy rate ~ rho T^4 with rho ~ M / R^3 and
   T_c ~ M / R). Contraction stops where L_H = L: the zero-age main sequence.
"""
import numpy as np
from scipy.integrate import solve_ivp
from nbody.units import G, R_SUN, L_SUN, SIGMA_SB, K_OVER_MH, MSUN_KG, AU_M, YEAR_S

MU = 0.61                 # mean molecular weight of ionized solar-composition gas
T_D = 1.5e6               # K, deuterium ignition (Offner et al. 2009)
T_H = 1.0e7               # K, hydrogen ignition
T_HAYASHI = 4300.0        # K, Hayashi-track effective temperature near 1 Msun (BHAC15)
F_ACC = 0.5               # fraction of accretion energy radiated (Offner et al. 2009)
L_ZAMS_1MSUN = 0.68       # Lsun, BHAC15 zero-age main sequence at 1 Msun
# Deuterium: D/H = 2e-5 by number, X = 0.7 -> mass fraction 2.8e-5; D + p -> He-3 releases 5.494 MeV.
E_D_PER_MASS = 2.8e-5 / (2 * 1.6735575e-27) * 5.494e6 * 1.602176634e-19 * (YEAR_S / AU_M) ** 2   # code units per Msun


def lane_emden(n):
    """First zero xi_1 of the Lane-Emden solution theta_n and |theta'(xi_1)|."""
    def rhs(xi, y):
        return [y[1], -max(y[0], 0.0) ** n - 2 * y[1] / xi]

    surface = lambda xi, y: y[0]
    surface.terminal = True
    xi0 = 1e-6                                                     # series start: theta = 1 - xi^2/6
    sol = solve_ivp(rhs, [xi0, 20], [1 - xi0**2 / 6, -xi0 / 3], events=surface, rtol=1e-12, atol=1e-14)
    return sol.t_events[0][0], abs(sol.y_events[0][0][1])


def structure_factor(n):
    xi1, dtheta = lane_emden(n)
    return 1 / ((n + 1) * xi1 * dtheta)


F_N15 = structure_factor(1.5)


def central_temperature(M, R):
    return MU / K_OVER_MH * G * M / R * F_N15


def radius_at_central_temperature(M, T):
    return MU / K_OVER_MH * G * M / T * F_N15


def energy(M, R):
    return -(3 / 3.5) * G * M**2 / (2 * R)


def zams_luminosity(M):
    """L ~ M^4 for Sun-like stars, normalized to BHAC15 at 1 Msun."""
    return L_ZAMS_1MSUN * L_SUN * M**4


def photosphere_luminosity(M, R):
    """Hayashi track (fully convective, fixed T_eff) until it falls to L_ZAMS; then the Henyey track at L_ZAMS."""
    return np.maximum(4 * np.pi * R**2 * SIGMA_SB * T_HAYASHI**4, zams_luminosity(M))


def hydrogen_luminosity(M, R):
    return zams_luminosity(M) * (central_temperature(M, R) / T_H) ** 7


def contraction_rate(M, R):
    """dR/dt from -dE/dt = L - L_H with E = -(3/(5-n)) G M^2 / (2R), n = 1.5."""
    return -(2 * 3.5 / 3) * R**2 * (photosphere_luminosity(M, R) - hydrogen_luminosity(M, R)) / (G * M**2)


def hayashi_radius(R0, M, t):
    """Exact solution of dR/dt = -A R^4 (Hayashi track, no fusion): R = (R0^-3 + 3 A t)^(-1/3)."""
    A = (2 * 3.5 / 3) * 4 * np.pi * SIGMA_SB * T_HAYASHI**4 / (G * M**2)
    return (R0**-3 + 3 * A * t) ** (-1 / 3)


def accretion_phase(t, M, mdot):
    """Radius and luminosities on an accretion history (arrays t, M, Mdot; M > 0)."""
    R_init = 2.5 * R_SUN * (np.maximum(mdot, 1e-12) / 1e-5) ** 0.2
    R_birth = radius_at_central_temperature(M, T_D)
    R = np.maximum(R_init, R_birth)
    on_birthline = R_birth >= R_init
    return dict(t=t, M=M, R=R, T_c=central_temperature(M, R), L_acc=F_ACC * G * M * mdot / R,
                L_phot=4 * np.pi * R**2 * SIGMA_SB * T_HAYASHI**4, L_D=np.where(on_birthline, E_D_PER_MASS * mdot, 0.0),
                L_H=np.zeros(len(t)))


def pre_main_sequence(M, R0, t0, t_end):
    """Kelvin-Helmholtz contraction at fixed mass M from radius R0 at time t0 until t_end (or the ZAMS)."""
    sol = solve_ivp(lambda t, y: [contraction_rate(M, y[0])], [t0, t_end], [R0], method="LSODA",
                    rtol=1e-9, atol=1e-12 * R_SUN, dense_output=True)
    t = np.geomspace(t0, t_end, 800)
    R = sol.sol(t)[0]
    return dict(t=t, M=np.full(len(t), M), R=R, T_c=central_temperature(M, R), L_acc=np.zeros(len(t)),
                L_phot=photosphere_luminosity(M, R), L_D=np.zeros(len(t)), L_H=hydrogen_luminosity(M, R))


def effective_temperature(L, R):
    return (L / (4 * np.pi * R**2 * SIGMA_SB)) ** 0.25
