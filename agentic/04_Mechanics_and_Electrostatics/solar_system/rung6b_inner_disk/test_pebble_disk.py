"""Rung 6b: the 1D inner-disk pebble model (nbody/pebble_disk.py)."""
import numpy as np
import pytest
import nbody.pebble_disk as pd
from nbody.gas_effects import disk_state
from nbody.units import M_EARTH
from disk_setup import disk_params


def test_thresholds_match_the_published_fits():
    # Li & Youdin (2021) with turbulence at St = 0.05, alpha = 1e-4, Pi = 0.051: eps_crit = 0.37, Z_crit = 0.017.
    # Lim et al. (2024) Eq. 19: Z_crit(0.05, 1e-4) = 0.023, and Z_crit(0.01, 1e-4) = 0.052, inside their measured
    # 0.04-0.065. Below St = 0.01 the Lim fit is out of range and LY21 (eps_crit = 2.5) takes over.
    assert pd.z_crit(0.05, 1e-4, 0.051, pd.FIT_LY21) == pytest.approx(0.37 * np.sqrt((0.051 / 5) ** 2 + 1e-4 / 0.0501), rel=1e-2)
    assert pd.z_crit(0.05, 1e-4, 0.051, pd.FIT_LY21) == pytest.approx(0.017, rel=0.05)
    assert pd.z_crit(0.05, 1e-4, 0.051, pd.FIT_LIM24) == pytest.approx(0.023, rel=0.02)
    assert 0.04 < pd.z_crit(0.01, 1e-4, 0.051, pd.FIT_LIM24) < 0.065
    assert pd.z_crit(0.005, 1e-4, 0.051, pd.FIT_LIM24) == pd.z_crit(0.005, 1e-4, 0.051, pd.FIT_LY21)


def test_drift_velocity_limits():
    # eps -> 0: a test particle, v = (-2 St eta vK + u_g) / (1 + St^2) (Nakagawa et al. 1986).
    # eps -> infinity: the dust carries the gas along and stops drifting. St -> 0: grains move with the gas.
    st, eta_vk, u = 0.05, 0.03, -1e-4
    assert pd.drift_velocity(st, 0.0, eta_vk, u) == pytest.approx((-2 * st * eta_vk + u) / (1 + st**2), rel=1e-14)
    assert abs(pd.drift_velocity(st, 1e6, eta_vk, u)) < 1e-9
    assert pd.drift_velocity(0.0, 0.3, eta_vk, u) == pytest.approx(u / 1.3, rel=1e-14)


def run(disk, n, t0, t1, inflow, ice_fraction, v_frag_dry, zeta, rock0=None):
    edges, centers, areas = pd.grid(n)
    rock = np.zeros(n) if rock0 is None else rock0.copy()
    ice, plts_rock, plts_ice, ledger = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(4)
    t = t0
    while t < t1:
        dt = min(pd.max_timestep(rock, ice, edges, centers, t, disk, v_frag_dry, 10.0, 0.4), t1 - t)
        pd.step(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, dt, disk, inflow, ice_fraction,
                v_frag_dry, 10.0, zeta, 0.01, pd.FIT_LIM24, ledger)
        t += dt
    return rock, ice, plts_rock, plts_ice, ledger, (edges, centers, areas)


def test_steady_drift_matches_the_analytic_profile():
    # A constant, dilute (eps << 1) rock flux with negligible turbulence must settle to Sigma = F / (2 pi r |v_r|),
    # v_r = (-2 St eta vK + u_g) / (1 + St^2). Tolerance 3%: first-order upwinding evaluates the flux half a cell
    # (Delta ln r / 2 = 0.9%) away from the cell center.
    disk = disk_params(alpha_turb=1e-8)
    F = 1e-6 * M_EARTH                                               # 1 Mearth/Myr in Msun/yr: eps ~ 1e-3
    rock, *_, (edges, centers, areas) = run(disk, 200, 0.0, 2e4, F, 0.0, 10.0, 0.0)
    expected = np.empty_like(centers)
    for i, r in enumerate(centers):
        sigma_g, H, omega, _ = disk_state(r, 2e4, disk)
        eta_vk = 0.5 * (H / r) ** 2 * pd.DLNP * omega * r
        u = -1.5 * disk[4] * (H * omega) * H / r
        expected[i] = F / (2 * np.pi * r * abs((-2 * pd.ST_GROWTH * eta_vk + u) / (1 + pd.ST_GROWTH**2)))
    assert np.max(np.abs(rock / expected - 1)) < 0.03


def test_mass_is_conserved_exactly():
    # Initial pebbles + inflow = what is left + onto the star + sublimated + planetesimals, to roundoff, with every
    # process on: icy inflow, sublimation at the snow line, diffusion, back-reaction and a forced conversion
    # (zeta = 1e-2 with a dense start that is above threshold).
    disk = disk_params()
    edges, centers, areas = pd.grid(100)
    rock0 = 0.05 * np.array([disk_state(r, 1e5, disk)[0] for r in centers])
    rock, ice, pr, pi_, ledger, _ = run(disk, 100, 1e5, 1.03e5, 50 * M_EARTH / 1e6, 0.76, 10.0, 1e-2, rock0)
    start = np.sum(rock0 * areas) + ledger[0]
    end = np.sum((rock + ice) * areas) + ledger[1] + ledger[2] + ledger[3]
    assert ledger[2] > 0 and ledger[3] > 0                            # sublimation and conversion both happened
    assert np.sum((pr + pi_) * areas) == pytest.approx(ledger[3], rel=1e-12)
    assert end == pytest.approx(start, rel=1e-12)


def test_snow_line_jam_has_the_analytic_size_and_no_dam_upstream():
    # A steady, dilute icy flux crossing the snow line: outside, Sigma_ice = F_ice / (2 pi r |v_icy|); inside, the
    # ice is gone and the rock slows down (St 1e-3 instead of 0.05), so Sigma_rock = F_rock / (2 pi r |v_dry|), a
    # "traffic jam" ~10x denser. A scheme that averages velocities across the jump dams the last icy cell instead
    # (it once made 90% of the planetesimals in the fiducial run). Negligible turbulence (alpha = 1e-8) with
    # v_frag_dry = 0.01 m/s gives the dry St ~ 1e-3 that 1 m/s gives at alpha = 1e-4. Tolerance 3%, as above.
    disk = disk_params(alpha_turb=1e-8)
    F, ice_fraction, v_dry = 1e-8 * M_EARTH, 0.76, 0.01
    edges, centers, areas = pd.grid(200)
    rock, ice, plts_rock, plts_ice, ledger = (np.zeros(200) for _ in range(5))
    t = 0.0
    while t < 3e5:
        dt = min(pd.max_timestep(rock, ice, edges, centers, t, disk, v_dry, 10.0, 0.4), 3e5 - t)
        pd.step(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, dt, disk, F, ice_fraction, v_dry, 10.0, 0.0,
                0.01, pd.FIT_LIM24, ledger)
        t += dt

    def sigma(r, rate, icy):
        return rate / (2 * np.pi * r * abs(pd.local_state(r, t, disk, 0.0, icy, v_dry, 10.0)[3]))

    outside = (centers > 2.9) & (centers < 3.9)
    inside = (centers > 0.2) & (centers < 2.5)
    ice_expected = np.array([sigma(r, F * ice_fraction, True) for r in centers[outside]])
    rock_expected = np.array([sigma(r, F * (1 - ice_fraction), False) for r in centers[inside]])
    assert np.max(np.abs(ice[outside] / ice_expected - 1)) < 0.03
    assert np.max(np.abs(rock[inside] / rock_expected - 1)) < 0.03
    assert np.all(ice[centers < 2.7] == 0)
    jam = rock[inside].mean() / (F * (1 - ice_fraction) / (2 * np.pi * centers[inside] * abs(
        pd.local_state(1.0, t, disk, 0.0, True, v_dry, 10.0)[3]))).mean()
    assert jam > 5                                                    # the dry rock really is jammed
