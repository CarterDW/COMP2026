"""Reference data for the real planets.

a: J2000 mean semi-major axis [AU] (Standish, JPL approximate elements; Earth row is the Earth-Moon barycenter).
period_days: sidereal orbital period [days] (NASA planetary fact sheet).
mass: planet mass [Msun] (Earth row includes the Moon).
"""
import numpy as np
from nbody.units import MSUN_KG

NAMES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

A = np.array([0.38709927, 0.72333566, 1.00000261, 1.52371034,
              5.20288700, 9.53667594, 19.18916464, 30.06992276])

PERIOD_DAYS = np.array([87.969, 224.701, 365.256, 686.980,
                        4332.589, 10759.22, 30685.4, 60189.0])

MASS = np.array([3.3011e23, 4.8675e24, 5.9722e24 + 7.342e22, 6.4171e23,
                 1.89813e27, 5.6834e26, 8.6813e25, 1.02413e26]) / MSUN_KG
