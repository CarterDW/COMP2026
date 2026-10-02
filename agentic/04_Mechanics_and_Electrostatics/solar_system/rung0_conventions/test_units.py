"""Rung 0: the code units reproduce real, independently measured numbers."""
import numpy as np
from nbody import units as u
from nbody import planets as p
from nbody.orbits import kepler_period


def measured_period_yr():
    return p.PERIOD_DAYS * u.DAY_S / u.YEAR_S


def test_year_is_gaussian_year():
    gaussian_year_days = 2 * np.pi / 0.01720209895   # from Gauss's constant k
    assert abs(u.YEAR_S / u.DAY_S / gaussian_year_days - 1) < 1e-9


def test_year_close_to_sidereal_year():
    # Differs at ~1e-6 because Earth has mass and a != 1 AU exactly.
    assert abs(u.YEAR_S / u.DAY_S / 365.256363004 - 1) < 3e-6


def test_earth_orbital_speed():
    v = np.sqrt(u.G / 1.0) / u.KM_S   # circular speed at 1 AU, in km/s
    assert abs(v - 29.78) < 0.01


def test_sun_to_earth_mass_ratio():
    assert abs(1 / u.M_EARTH / 332946 - 1) < 1e-4


def test_parsec_in_au():
    assert abs(u.PC - 206264.806) < 1e-3


def test_kepler_inner_planets_and_jupiter():
    rel = measured_period_yr()[:5] / kepler_period(p.A[:5], p.MASS[:5]) - 1
    assert np.all(np.abs(rel) < 3e-5)


def test_jupiter_needs_its_own_mass():
    # Without the (1 + m) factor Jupiter's period is off 20x more: the data can see Jupiter's mass.
    with_mass = abs(measured_period_yr()[4] / kepler_period(p.A[4], p.MASS[4]) - 1)
    massless = abs(measured_period_yr()[4] / kepler_period(p.A[4], 0.0) - 1)
    assert with_mass < 3e-5 < 4e-4 < massless


def test_kepler_outer_planets():
    # Limited to ~6e-4 by mutual perturbations among the giants (e.g. Jupiter-Saturn near 5:2).
    rel = measured_period_yr()[5:] / kepler_period(p.A[5:], p.MASS[5:]) - 1
    assert np.all(np.abs(rel) < 1e-3)
