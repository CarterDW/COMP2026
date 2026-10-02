"""Code units: AU, solar mass, year. In these units G = 4 pi^2 exactly."""
import numpy as np

G = 4 * np.pi**2

# SI reference values (IAU 2012 / 2015, CODATA 2018)
AU_M = 1.495978707e11        # m, exact by definition
GM_SUN_SI = 1.32712440018e20  # m^3 s^-2, known far better than G or M_sun separately
G_SI = 6.67430e-11           # m^3 kg^-1 s^-2
DAY_S = 86400.0

# The year is *defined* so that G = 4 pi^2: a massless body at 1 AU orbits in 1 yr.
# This is the Gaussian year, ~365.2569 days.
YEAR_S = 2 * np.pi * np.sqrt(AU_M**3 / GM_SUN_SI)
MSUN_KG = GM_SUN_SI / G_SI

KM_S = 1e3 * YEAR_S / AU_M   # 1 km/s expressed in AU/yr
PC = 648000 / np.pi          # 1 parsec in AU

M_EARTH = 5.9722e24 / MSUN_KG
M_JUPITER = 1.89813e27 / MSUN_KG
R_EARTH = 6.3710e6 / AU_M
