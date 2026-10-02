"""Rung 4b: the one-zone protostar model, checked against exact results and the BHAC15 1 Msun track."""
import numpy as np
from nbody import protostar as ps
from nbody.plotstyle import blackbody_rgb
from nbody.units import R_SUN, L_SUN, G


# ---------------------------------------------------------------- stellar structure

def test_lane_emden_n1_is_exact():
    # theta_1 = sin(xi) / xi: first zero at pi, with |theta'(pi)| = 1/pi.
    xi1, dtheta = ps.lane_emden(1.0)
    assert np.isclose(xi1, np.pi, rtol=1e-8) and np.isclose(dtheta, 1 / np.pi, rtol=1e-8)


def test_lane_emden_matches_tabulated_polytropes():
    # Chandrasekhar (1939): n = 1.5: xi1 = 3.65375, |theta'| = 0.20330; n = 3: 6.89685, 0.04243.
    for n, xi_ref, d_ref in ((1.5, 3.65375, 0.20330), (3.0, 6.89685, 0.04243)):
        xi1, dtheta = ps.lane_emden(n)
        assert np.isclose(xi1, xi_ref, atol=1e-5) and np.isclose(dtheta, d_ref, atol=1e-5)


def test_central_temperature_and_birthline():
    # T_c = 7.6e6 K (M/Msun)(Rsun/R) for n = 1.5, mu = 0.61; the deuterium thermostat (1.5e6 K) sits at 5.1 Rsun.
    assert np.isclose(ps.central_temperature(1.0, R_SUN), 7.6e6, rtol=0.01)
    assert np.isclose(ps.radius_at_central_temperature(1.0, ps.T_D) / R_SUN, 5.1, rtol=0.01)


def test_deuterium_luminosity_matches_offner_prescription():
    # Burning accreted D (D/H = 2e-5, 5.494 MeV each) at Mdot = 1e-5 Msun/yr. Offner et al. (2009) use 15 Lsun.
    L_D = ps.E_D_PER_MASS * 1e-5 / L_SUN
    assert 0.75 < L_D / 15 < 1.25


# ---------------------------------------------------------------- contraction

def test_hayashi_contraction_matches_exact_solution(monkeypatch):
    # With fusion switched off and L >> L_ZAMS, the numerical contraction must follow R = (R0^-3 + 3 A t)^(-1/3).
    # (With fusion on, the T_c^7 tail is already 1e-5 to 6e-4 of L here and shifts R by ~5e-5.)
    monkeypatch.setattr(ps, "hydrogen_luminosity", lambda M, R: 0.0 * R)
    M, R0 = 1.0, 3.1 * R_SUN
    pm = ps.pre_main_sequence(M, R0, 0.0 + 1.0, 2e6)
    early = pm["R"] > 2.0 * R_SUN
    assert np.allclose(pm["R"][early], ps.hayashi_radius(R0, M, pm["t"][early] - 1.0), rtol=1e-6)


def test_energy_budget_closes():
    # Radiated energy minus fusion energy equals the gravitational energy released: int (L - L_H) dt = E0 - E_end.
    M, R0 = 1.0, 3.1 * R_SUN
    pm = ps.pre_main_sequence(M, R0, 1.0, 2e8)
    radiated = np.trapz(pm["L_phot"] - pm["L_H"], pm["t"])
    released = ps.energy(M, R0) - ps.energy(M, pm["R"][-1])
    assert np.isclose(radiated, released, rtol=1e-3)


def test_contraction_stops_on_the_main_sequence():
    # The ZAMS is where hydrogen supplies all the luminosity: there T_c = T_H by construction.
    pm = ps.pre_main_sequence(1.0, 3.1 * R_SUN, 1.0, 2e8)
    assert np.isclose(pm["L_H"][-1] / pm["L_phot"][-1], 1.0, rtol=1e-6)
    assert np.isclose(pm["T_c"][-1], ps.T_H, rtol=1e-6)


def test_one_msun_track_against_bhac15():
    # BHAC15 1 Msun (Baraffe et al. 2015): R = 3.1 Rsun and T_c = 2.5e6 K at 0.5 Myr; 1.4 Rsun at ~5 Myr (4.6 Myr later);
    # settles on the ZAMS at 40-50 Myr with R = 0.89 Rsun. A one-zone n = 1.5 star is slightly too compact, so allow
    # 20% on the contraction time and radius and 35% on the ZAMS arrival time. Measured: 2.45e6 K, 4.8 Myr,
    # 0.76 Rsun, 34 Myr.
    pm = ps.pre_main_sequence(1.0, 3.1 * R_SUN, 0.5e6, 2e8)
    t, R = pm["t"], pm["R"] / R_SUN
    assert np.isclose(pm["T_c"][0], 2.5e6, rtol=0.05)
    assert np.isclose(np.interp(-1.4, -R, t) - 0.5e6, 4.6e6, rtol=0.2)
    assert np.isclose(R[-1], 0.89, rtol=0.2)
    t_zams = t[np.argmax(pm["L_H"] / pm["L_phot"] > 0.99)]
    assert 0.65 * 40e6 < t_zams < 1.35 * 50e6


# ---------------------------------------------------------------- accretion phase

def test_accretion_phase_thermostat_and_luminosity():
    t = np.linspace(1e4, 3e4, 50)
    M = np.linspace(0.05, 0.8, 50)
    mdot = np.gradient(M, t)
    ac = ps.accretion_phase(t, M, mdot)
    assert np.all(ac["T_c"] <= ps.T_D * (1 + 1e-12))                # deuterium holds T_c at or below 1.5e6 K
    assert np.allclose(ac["L_acc"], ps.F_ACC * G * M * mdot / ac["R"], rtol=1e-12)
    assert np.all((ac["L_D"] > 0) == (ac["T_c"] >= ps.T_D * (1 - 1e-12)))   # D burns only on the birthline


# ---------------------------------------------------------------- color

def test_blackbody_colors():
    r, g, b = blackbody_rgb(6504)                                    # the sRGB white point (D65) is ~white
    assert min(r, g, b) > 0.95
    r, g, b = blackbody_rgb(3000)                                    # cool: orange
    assert r > g > b
    r, g, b = blackbody_rgb(15000)                                   # hot: blue-white
    assert b > g > r
