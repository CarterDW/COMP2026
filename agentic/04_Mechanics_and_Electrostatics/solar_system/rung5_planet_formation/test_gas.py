"""5d-5e: gas drag, tidal damping, dispersal and gas accretion (nbody/gas_effects.py)."""
import numpy as np
import nbody.hybrid as hy
from nbody.gas_effects import (apply_gas, evolve, disk_state, TAU_DISPERSAL, C_D, RHO_SOLID, PLANETESIMAL_SIZE,
                               M_CRIT)
from nbody.units import G, M_EARTH

M_STAR = 0.816
MU = G * M_STAR
DISK = np.array([0.137, 258.0, 1.1e7, M_STAR, 1e-3, 1e6, 0.0, 0.0, 1e-3])   # (..., t0, pebbles, migration, alpha_turb)


def one_body(x, u, m, big):
    Q, v = hy.from_heliocentric(np.array([x], float), np.array([u], float), np.array([m]), M_STAR)
    return Q, v, np.array([m]), np.full(1, 1e-9), np.ones(1, bool), np.array([big])


def seed(m):
    """Composition array with all mass as seed solids."""
    return np.column_stack([m, np.zeros((len(m), 3))])


def gas_velocity(R, t):
    sigma, H, omega, nu = disk_state(R, t, DISK)
    return omega * R * (1 - (11 / 8) * (H / R) ** 2)


def test_gas_orbits_slightly_slower_than_keplerian():
    R = 5.0
    sigma, H, omega, nu = disk_state(R, 1e6, DISK)
    eta = 1 - gas_velocity(R, 1e6) / (omega * R)
    assert 1e-3 < eta < 1e-2 and np.isclose(eta, 1.375 * (H / R) ** 2)


def test_disk_disperses_exponentially():
    # disk_state = similarity solution x exp(-(t - t0) / TAU_DISPERSAL): dividing out the similarity part leaves 1/e.
    from nbody.viscous_disk import LyndenBellPringle
    lbp = LyndenBellPringle(DISK[0], DISK[1], M_STAR, DISK[4])
    lbp.t_nu = DISK[2]                                     # use exactly the t_nu packed into DISK
    t0, t1 = 1e6, 1e6 + TAU_DISPERSAL
    ratio = disk_state(5.0, t1, DISK)[0] / disk_state(5.0, t0, DISK)[0]
    assert np.isclose(ratio / (lbp.sigma(5.0, t1) / lbp.sigma(5.0, t0)), np.exp(-1), rtol=1e-12)


def test_quadratic_drag_is_applied_exactly():
    # A planetesimal 100 m/s faster than the gas: one drag step must give |du| / (1 + K |du| dt).
    R, t, dt, extra = 5.0, 1e6, 1e4, 100 * 0.000211      # 100 m/s in AU/yr
    vg = gas_velocity(R, t)
    Q, v, m, Rad, alive, big = one_body([R, 0, 0], [0, vg + extra, 0], 1e-15, False)
    sigma, H, omega, nu = disk_state(R, t, DISK)
    K = 3 * C_D * sigma / (np.sqrt(2 * np.pi) * H) / (8 * RHO_SOLID * PLANETESIMAL_SIZE)
    apply_gas(Q, v, m, Rad, alive, big, seed(m), t, dt, DISK, M_STAR)
    u = hy.to_heliocentric(Q, v, m, M_STAR)[1][0]
    assert np.isclose(u[1] - vg, extra / (1 + K * extra * dt), rtol=1e-9)


def test_embryo_eccentricity_damps_on_the_tanaka_ward_time():
    # e must e-fold in t_wave / 0.780 (Tanaka & Ward 2004). Measured over 20 kyr for a 3 Mearth embryo at 5 AU.
    a0, e0, m = 5.0, 0.05, 3 * M_EARTH
    Q, v, mm, Rad, alive, big = one_body([a0 * (1 - e0), 0, 0], [0, np.sqrt(MU * (1 + e0) / (a0 * (1 - e0))), 0], m, True)
    dt = a0 ** 1.5 / np.sqrt(M_STAR) / 40
    rc = hy.critical_radii(Q, v, mm, alive, MU, dt)
    n = int(2e4 / dt)
    evolve(Q, v, mm, Rad, alive, big, seed(mm), rc, MU, M_STAR, dt, n, 1e6, DISK, np.zeros((2, 6)), 0, 200.0)
    x, u = hy.to_heliocentric(Q, v, mm, M_STAR)
    r, E = np.linalg.norm(x[0]), 0.5 * u[0] @ u[0] - MU / np.linalg.norm(x[0])
    a = -MU / (2 * E)
    e = np.sqrt(1 - np.linalg.norm(np.cross(x[0], u[0])) ** 2 / (MU * a))
    sigma, H, omega, nu = disk_state(a0, 1e6 + 1e4, DISK)
    t_wave = (M_STAR / m) * (M_STAR / (sigma * a0**2)) * (H / a0) ** 4 / omega
    assert np.isclose(np.log(e0 / e) / (n * dt), 0.780 / t_wave, rtol=0.1)


def test_gas_accretion_switches_on_above_ten_earth_masses():
    from nbody.gas_effects import gas_capture_rate
    R, t, dt = 5.0, 1e6, 10.0
    vk = np.sqrt(MU / R)
    for m, should_grow in ((9.9 * M_EARTH, False), (30 * M_EARTH, True)):
        Q, v, mm, Rad, alive, big = one_body([R, 0, 0], [0, vk, 0], m, True)
        comp = seed(mm)
        apply_gas(Q, v, mm, Rad, alive, big, comp, t, dt, DISK, M_STAR)
        assert (mm[0] > m) == should_grow
    assert np.isclose(mm[0] - m, gas_capture_rate(m, R, t, DISK) * dt, rtol=1e-12)
    assert np.isclose(comp[0, 3], mm[0] - m, rtol=1e-12) and comp[0, 0] == m        # the gain is booked as gas


def test_gas_capture_follows_tanigawa_tanaka():
    # Kelvin-Helmholtz limited for small cores; for big ones D Sigma / (1 + 0.034 K), which peaks and then falls as
    # the planet's gap deepens (Tanigawa & Tanaka 2016, eqs. 7-10).
    from nbody.gas_effects import gas_capture_rate
    from nbody.units import M_JUPITER
    R, t = 5.0, 1.5e6
    sigma, H, omega, nu = disk_state(R, t, DISK)
    m = 12 * M_EARTH
    assert np.isclose(gas_capture_rate(m, R, t, DISK), m * 12.0**3 / 1e9, rtol=1e-12)
    m = M_JUPITER
    q, h = m / M_STAR, H / R
    expected = 0.29 * h**-2 * q ** (4 / 3) * R**2 * omega * sigma / (1 + 0.034 * h**-5 * q**2 / DISK[8])
    assert np.isclose(gas_capture_rate(m, R, t, DISK), expected, rtol=1e-12)
    assert gas_capture_rate(3 * M_JUPITER, R, t, DISK) < gas_capture_rate(M_JUPITER, R, t, DISK)


def test_giants_share_the_disk_inflow_outside_in():
    from nbody.gas_effects import accrete_gas, disk_inflow, gas_capture_rate
    from nbody.units import M_JUPITER
    t, dt = 1.5e6, 1.0
    x = np.array([[5.0, 0, 0], [10.0, 0, 0]])
    u = np.array([[0, np.sqrt(MU / 5), 0], [0, np.sqrt(MU / 10), 0]])
    m = np.array([0.3 * M_JUPITER, 0.3 * M_JUPITER])
    Q, v = hy.from_heliocentric(x, u, m, M_STAR)
    m0, R = m.copy(), np.full(2, 1e-4)
    accrete_gas(Q, v, m, R, np.ones(2, bool), np.ones(2, bool), seed(m), t, dt, DISK, np.zeros(3))
    inflow = disk_inflow(t, DISK)
    outer = min(gas_capture_rate(m0[1], 10.0, t, DISK), inflow)
    inner = min(gas_capture_rate(m0[0], 5.0, t, DISK), inflow - outer)
    assert np.isclose(m[1] - m0[1], outer * dt, rtol=1e-9) and np.isclose(m[0] - m0[0], inner * dt, rtol=1e-9)
    # Together they never exceed what the disk supplies. (The gain is a difference of two ~3e-4 masses, so allow a
    # few units of roundoff in m rather than a relative tolerance on the tiny gain.)
    assert (m - m0).sum() <= inflow * dt + 4 * np.finfo(float).eps * m.sum()


# ---------------------------------------------------------------- pebble accretion

from nbody.gas_effects import pebble_flux, pebble_capture_rate, pebble_isolation_mass, accrete_pebbles, EPS_D, Z_PEBBLES, STOKES

PEBBLE_DISK = DISK.copy()
PEBBLE_DISK[6] = 1.0


def test_pebble_flux_is_the_solids_swept_up_by_the_growth_front():
    # Mdot_F must equal d/dt of the solid mass inside r_g(t) (gas profile held fixed): checked by finite differences.
    t, dt = 1.5e6, 1e3
    r_g = lambda t: (3 / 16) ** (1 / 3) * (G * M_STAR) ** (1 / 3) * (EPS_D * Z_PEBBLES) ** (2 / 3) * t ** (2 / 3)
    R = np.linspace(1e-3, 1000, 2000001)
    sig = np.array([disk_state(r, t, DISK)[0] for r in R[::1000]])
    solids = lambda rg: np.trapz(np.where(R[::1000] < rg, 2 * np.pi * R[::1000] * Z_PEBBLES * sig, 0), R[::1000])
    finite_diff = (solids(r_g(t + dt)) - solids(r_g(t - dt))) / (2 * dt)
    assert np.isclose(pebble_flux(t, PEBBLE_DISK), finite_diff, rtol=0.02)


def test_capture_rate_formula():
    m, R, t, flux = 5 * M_EARTH, 7.0, 1e6, 1e-9
    sigma, H, omega, nu = disk_state(R, t, DISK)
    v_r = 2 * STOKES * 1.375 * (H / R) ** 2 * omega * R
    expected = 2 * (STOKES / 0.1) ** (2 / 3) * omega * (R * (m / (3 * M_STAR)) ** (1 / 3)) ** 2 * flux / (2 * np.pi * R * v_r)
    assert np.isclose(pebble_capture_rate(m, R, t, PEBBLE_DISK, flux), expected, rtol=1e-12)


def test_outer_embryos_take_their_share_and_isolated_ones_block_the_flux():
    t, dt = 1e6, 1.0
    x = np.array([[5.0, 0, 0], [12.0, 0, 0]])
    u = np.array([[0, np.sqrt(MU / 5), 0], [0, np.sqrt(MU / 12), 0]])
    flux = pebble_flux(t, PEBBLE_DISK)
    for m_outer, blocked in ((5 * M_EARTH, False), (1.01 * pebble_isolation_mass(12.0, t, PEBBLE_DISK), True)):
        m = np.array([3 * M_EARTH, m_outer])
        Q, v = hy.from_heliocentric(x, u, m, M_STAR)
        m0 = m.copy()
        comp = seed(m)
        accrete_pebbles(Q, v, m, np.ones(2, bool), np.ones(2, bool), comp, t, dt, PEBBLE_DISK, np.zeros(3))
        assert np.allclose(comp[:, 2], m - m0, rtol=0, atol=4 * np.finfo(float).eps * m.max())   # booked as pebbles
        outer_take = pebble_capture_rate(m0[1], 12.0, t, PEBBLE_DISK, flux) * dt
        if blocked:
            assert m[1] == m0[1] and m[0] == m0[0]                     # isolated outer embryo: nothing gets through
        else:
            inner_take = pebble_capture_rate(m0[0], 5.0, t, PEBBLE_DISK, flux - outer_take / dt) * dt
            assert np.isclose(m[1] - m0[1], outer_take, rtol=1e-6) and np.isclose(m[0] - m0[0], inner_take, rtol=1e-6)


# ---------------------------------------------------------------- migration (Paardekooper et al. 2011, Kanagawa et al. 2018)

from nbody.gas_effects import (type1_torque, migration_torque, thermal_diffusivity, _saturation_F, _saturation_G,
                               _saturation_K, GAMMA_GAS)

MIG_DISK = DISK.copy()
MIG_DISK[7] = 1.0


def test_saturation_functions_are_continuous_with_the_right_limits():
    for func, p_break in ((_saturation_G, np.sqrt(8 / (45 * np.pi))), (_saturation_K, np.sqrt(28 / (45 * np.pi)))):
        assert np.isclose(func(p_break * (1 - 1e-12)), func(p_break * (1 + 1e-12)), rtol=1e-9)
        assert func(1e-4) < 1e-5 and abs(func(1e4) - 1) < 1e-9          # 0 for fast diffusion, 1 for slow
    assert _saturation_F(0.0) == 1.0 and _saturation_F(1e3) < 1e-5


def gamma_eff_from_torque(chi):
    """Recover gamma_eff from the Lindblad torque, which is Gamma_0 / gamma_eff times a fixed slope factor."""
    m, R, t = 5 * M_EARTH, 5.0, 1e6
    sigma, H, omega, nu = disk_state(R, t, DISK)
    gamma0 = (m / M_STAR / (H / R)) ** 2 * sigma * R**4 * omega**2
    lindblad = type1_torque(m, R, t, DISK, 1.0, 0.5, chi)[1]
    return gamma0 * (-2.5 - 1.7 * 0.5 + 0.1 * 1.0) / lindblad


def test_effective_gamma_has_isothermal_and_adiabatic_limits():
    assert np.isclose(gamma_eff_from_torque(1e20), 1.0, rtol=1e-6)            # fast cooling: isothermal
    assert np.isclose(gamma_eff_from_torque(1e-20), GAMMA_GAS, rtol=1e-6)     # no cooling: adiabatic


def test_saturated_corotation_leaves_only_the_lindblad_torque():
    # Nearly inviscid gas (alpha -> 0) cannot refresh the horseshoe region: the corotation torque saturates to zero.
    inviscid = DISK.copy()
    inviscid[8] = 1e-14
    total, lindblad, corotation = type1_torque(5 * M_EARTH, 5.0, 1e6, inviscid, 1.0, 0.5, 1e-20)
    assert abs(corotation) < 1e-6 * abs(lindblad) and np.isclose(total, lindblad, rtol=1e-6)


def test_type1_rate_is_comparable_to_tanaka_2002():
    # Isothermal reference (Tanaka, Takeuchi & Ward 2002): a / (da/dt) = (M/m)(M/(Sigma r^2)) h^2 / ((2.7 + 1.1 a) Omega).
    # L / |Gamma| is twice that (da/dt = 2 a Gamma / L). The non-isothermal torque differs by its corotation terms and
    # gamma_eff: expect agreement within ~2x (measured 0.78).
    m, R, t = 3 * M_EARTH, 7.0, 1.2e6
    sigma, H, omega, nu = disk_state(R, t, MIG_DISK)
    slope = 1 + R / (258.0 * (1 + t / 1.1e7))
    t_tanaka = 2 * (M_STAR / m) * (M_STAR / (sigma * R**2)) * (H / R) ** 2 / ((2.7 + 1.1 * slope) * omega)
    t_ours = m * np.sqrt(G * M_STAR * R) / abs(migration_torque(m, R, t, MIG_DISK))
    assert migration_torque(m, R, t, MIG_DISK) < 0 and 0.5 < t_ours / t_tanaka < 2


def test_gap_opening_reduces_the_torque():
    from nbody.units import M_JUPITER
    m, R, t = M_JUPITER, 5.0, 1.5e6
    sigma, H, omega, nu = disk_state(R, t, MIG_DISK)
    K = (H / R) ** -5 * (m / M_STAR) ** 2 / DISK[8]
    T_d = 1 + t / DISK[2]
    raw = type1_torque(m, R, t, MIG_DISK, 1 + R / (DISK[1] * T_d), 0.5, thermal_diffusivity(R, t, MIG_DISK))[0]
    assert K > 1000 and np.isclose(migration_torque(m, R, t, MIG_DISK), raw / (1 + 0.04 * K), rtol=1e-12)


def test_planet_migrates_at_the_torque_rate():
    # For a near-circular orbit dL/dt = Gamma gives da/dt = 2 a Gamma / L. A 5 Mearth planet at 10 AU over 20 kyr.
    a0, m = 10.0, 5 * M_EARTH
    Q, v, mm, Rad, alive, big = one_body([a0, 0, 0], [0, np.sqrt(MU / a0), 0], m, True)
    dt = a0 ** 1.5 / np.sqrt(M_STAR) / 40
    rc = hy.critical_radii(Q, v, mm, alive, MU, dt)
    n = int(2e4 / dt)
    evolve(Q, v, mm, Rad, alive, big, seed(mm), rc, MU, M_STAR, dt, n, 1e6, MIG_DISK, np.zeros((2, 6)), 0, 200.0)
    x, u = hy.to_heliocentric(Q, v, mm, M_STAR)
    a = -MU / (2 * (0.5 * u[0] @ u[0] - MU / np.linalg.norm(x[0])))
    rate = 2 * a0 * migration_torque(m, a0, 1e6 + 1e4, MIG_DISK) / (m * np.sqrt(G * M_STAR * a0))
    assert np.isclose((a - a0) / (n * dt), rate, rtol=0.05)


def test_type2_migration_slows_in_proportion_to_midplane_turbulence():
    # Deep in a gap the torque ~ Gamma_I / (0.04 K) with K ~ 1 / alpha_turb: ten times less turbulence, ten times slower.
    from nbody.units import M_JUPITER
    quiet = MIG_DISK.copy()
    quiet[8] = 1e-4
    ratio = migration_torque(M_JUPITER, 5.0, 1.5e6, MIG_DISK) / migration_torque(M_JUPITER, 5.0, 1.5e6, quiet)
    assert 9 < ratio < 11
