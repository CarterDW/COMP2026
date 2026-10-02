"""Viscous accretion disk: the Lynden-Bell & Pringle (1974) similarity solution with nu ~ R.

The gas temperature T = T_1au (R / AU)^(-1/2) (the Rung 4 protostellar-heating law, Hayashi 1981) and an alpha
viscosity nu = alpha cs^2 / Omega give nu = nu_1 (R / R_1), exactly the gamma = 1 case of the similarity solution:

    Sigma(R, t) = M_0 / (2 pi R_1^2) (R/R_1)^(-1) T^(-3/2) exp(-R / (R_1 T)),   T = 1 + t / t_nu,  t_nu = R_1^2 / (3 nu_1)

It conserves angular momentum while spreading: mass drains onto the star (M(t) = M_0 T^(-1/2)) as the outer disk
expands. Solids follow Hayashi (1981): rock is 7.1/1700 of the gas inside the snow line, rock + ice 30/1700 outside.
"""
import numpy as np
from math import gamma as gamma_fn
from nbody.units import G, K_OVER_MH

MU_GAS = 2.33                 # mean molecular weight of molecular disk gas
T_1AU = 280.0                 # K at 1 AU (Hayashi 1981; the Rung 4 heating law)
SNOW_LINE = 2.7               # AU, where T = 170 K
Z_ROCK, Z_ICE = 7.1 / 1700, 30.0 / 1700


def sound_speed(R):
    return np.sqrt(K_OVER_MH / MU_GAS * T_1AU / np.sqrt(R))


def viscosity(R, M_star, alpha):
    omega = np.sqrt(G * M_star / R**3)
    return alpha * sound_speed(R) ** 2 / omega


def scale_height(R, M_star):
    return sound_speed(R) / np.sqrt(G * M_star / R**3)


def scale_radius_from(M_disk, J_disk, M_star):
    """R_1 such that the T = 1 similarity disk has mass M_disk and (Keplerian) angular momentum J_disk.

    J = int 2 pi R Sigma sqrt(G M R) dR = M_0 sqrt(G M_star R_1) Gamma(3/2).
    """
    return (J_disk / (gamma_fn(1.5) * M_disk)) ** 2 / (G * M_star)


class LyndenBellPringle:
    def __init__(self, M0, R1, M_star, alpha):
        self.M0, self.R1, self.M_star, self.alpha = M0, R1, M_star, alpha
        self.nu1 = viscosity(R1, M_star, alpha)
        self.t_nu = R1**2 / (3 * self.nu1)

    def T(self, t):
        return 1 + t / self.t_nu

    def sigma(self, R, t):
        T = self.T(t)
        return self.M0 / (2 * np.pi * self.R1**2) * (self.R1 / R) * T**-1.5 * np.exp(-R / (self.R1 * T))

    def mass(self, t):
        return self.M0 / np.sqrt(self.T(t))

    def mass_inside(self, R, t):
        return self.mass(t) * (1 - np.exp(-R / (self.R1 * self.T(t))))

    def accretion_rate(self, t):
        """Mass flow onto the star, -dM/dt."""
        return 0.5 * self.M0 * self.T(t) ** -1.5 / self.t_nu


def solid_surface_density(R, sigma_gas):
    return sigma_gas * np.where(R < SNOW_LINE, Z_ROCK, Z_ICE)


def isolation_mass(R, sigma_solid, M_star, b=10.0):
    """Mass an embryo reaches by sweeping an annulus b Hill radii wide: M = (2 pi b R^2 Sigma_s)^(3/2) / (3 M_star)^(1/2)."""
    return (2 * np.pi * b * R**2 * sigma_solid) ** 1.5 / np.sqrt(3 * M_star)
