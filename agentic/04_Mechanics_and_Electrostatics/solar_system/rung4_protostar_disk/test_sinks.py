"""Rung 4: sink particles, the barotropic equation of state, and the rotating collapse."""
import numpy as np
import pytest
from nbody.sinks import Sinks, accrete, try_create_sink, jeans_resolution_density
from nbody.sph import Gas
from nbody.clouds import lattice_sphere, uniform_sphere
from nbody.integrators import star_formation_leapfrog
from nbody.units import G, KM_S

rng = np.random.default_rng(0)


def totals(gas_pos, gas_vel, gas_mass, sinks):
    """Total mass, momentum, and angular momentum (orbital + sink spin) about the origin."""
    M = gas_mass.sum() + sinks.mass.sum()
    p = gas_mass @ gas_vel + sinks.mass @ sinks.vel
    L = np.sum(gas_mass[:, None] * np.cross(gas_pos, gas_vel), axis=0) + sinks.angular_momentum()
    return M, p, L


def one_sink(mass, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)):
    s = Sinks()
    s.mass, s.pos, s.vel, s.spin = np.array([mass]), np.array([pos], float), np.array([vel], float), np.zeros((1, 3))
    return s


# ---------------------------------------------------------------- equation of state

def test_barotropic_limits():
    # Isothermal far below rho_crit; P ~ rho^(5/3) far above, i.e. dlnP/dlnrho = 1 -> 5/3.
    gas = Gas("barotropic", cs=1.0, rho_crit=1.0)
    rho = np.array([1e-6, 1e-6 * 1.001, 1e6, 1e6 * 1.001])
    P, c = gas.pressure_and_sound_speed(rho, None)
    assert np.isclose(np.log(P[1] / P[0]) / np.log(1.001), 1.0, atol=1e-4)
    assert np.isclose(np.log(P[3] / P[2]) / np.log(1.001), 5 / 3, atol=1e-4)
    # The sound speed is sqrt(dP/drho): compare with a finite difference.
    r, d = 2.0, 1e-6
    numeric = (gas.pressure_and_sound_speed(np.array([r + d]), None)[0] - gas.pressure_and_sound_speed(np.array([r - d]), None)[0]) / (2 * d)
    assert np.isclose(gas.pressure_and_sound_speed(np.array([r]), None)[1][0] ** 2, numeric[0], rtol=1e-8)


def test_jeans_resolution_density_resolves_two_neighbour_masses():
    cs, m = 0.19 * KM_S, 2e-4
    rho = jeans_resolution_density(cs, m)
    jeans_mass = (np.pi**2.5 / 6) * cs**3 / (G**1.5 * np.sqrt(rho))
    assert np.isclose(jeans_mass, 2 * 58 * m, rtol=1e-12)


# ---------------------------------------------------------------- sink creation and accretion conserve everything

def test_accretion_conserves_mass_momentum_and_angular_momentum():
    pos = rng.normal(size=(300, 3)) * 0.5
    vel = rng.normal(size=(300, 3)) * 0.1
    mass = rng.uniform(1e-4, 2e-4, 300)
    sinks = one_sink(1.0, pos=(0.1, -0.2, 0.05), vel=(0.01, 0.02, -0.01))
    before = totals(pos, vel, mass, sinks)
    removed = accrete(sinks, pos, vel, mass, r_acc=0.6)
    keep = np.setdiff1d(np.arange(300), removed)
    after = totals(pos[keep], vel[keep], mass[keep], sinks)
    assert len(removed) > 20
    assert np.isclose(after[0], before[0], rtol=1e-14)
    assert np.allclose(after[1], before[1], atol=1e-15)
    assert np.allclose(after[2], before[2], atol=1e-15)


def test_accretion_only_takes_bound_low_angular_momentum_gas():
    # Three particles inside r_acc. 0: falling in slowly (taken). 1: bound (E < 0) but with more specific angular
    # momentum than a circular orbit at r_acc, so it would orbit rather than fall in (kept). 2: faster than escape
    # speed (unbound: kept).
    M, r_acc = 1.0, 1.0
    v_c = lambda r: np.sqrt(G * M / r)
    pos = np.array([[0.5, 0, 0], [0, 0.9, 0], [0, 0, 0.5]], float)
    vel = np.array([[-0.1 * v_c(0.5), 0, 0], [-1.3 * v_c(0.9), 0, 0], [0, 0, 1.5 * v_c(0.5)]])
    assert 0.5 * (1.3 * v_c(0.9)) ** 2 - G * M / 0.9 < 0                      # particle 1 really is bound
    assert 0.9 * 1.3 * v_c(0.9) > np.sqrt(G * M * r_acc)                      # ... with too much angular momentum
    removed = accrete(one_sink(M), pos, vel, np.full(3, 1e-6), r_acc)
    assert list(removed) == [0]


def test_sink_creation_needs_a_bound_dense_group():
    pos, vel, mass = uniform_sphere(500, 0.1, 1.0, rng)
    rho = np.ones(500)
    rho[0] = 10.0
    cold, hot = np.zeros(500), np.full(500, 100.0)             # specific thermal energy
    assert len(try_create_sink(Sinks(), pos, vel, mass, rho, cold, r_acc=0.5, rho_sink=20.0)) == 0   # not dense enough
    assert len(try_create_sink(Sinks(), pos, vel, mass, rho, hot, r_acc=0.5, rho_sink=5.0)) == 0     # unbound (hot)
    sinks = Sinks()
    before = totals(pos, vel, mass, sinks)
    taken = try_create_sink(sinks, pos, vel, mass, rho, cold, r_acc=0.5, rho_sink=5.0)
    keep = np.setdiff1d(np.arange(500), taken)
    after = totals(pos[keep], vel[keep], mass[keep], sinks)
    assert len(sinks) == 1 and len(taken) > 5
    assert np.isclose(after[0], before[0], rtol=1e-14)
    assert np.allclose(after[1], before[1], atol=1e-15) and np.allclose(after[2], before[2], atol=1e-15)


# ---------------------------------------------------------------- accretion history against analytic infall

def test_accretion_follows_analytic_radial_infall():
    # Cold, viscosity-free gas sphere (mass Mg, radius R) around a central sink Ms. A shell starting at r0 falls
    # under the fixed mass M(r0) = Ms + Mg (r0/R)^3 (inner shells always arrive first, so no shell crossing) and
    # reaches r_acc at t(r0) = sqrt(r0^3 / 2GM) [sqrt(x(1-x)) + arccos(sqrt(x))], x = r_acc / r0.
    # Measured agreement: within 0.005 Msun (one lattice shell) up to 0.9 of the last arrival time.
    Ms, Mg, R, r_acc = 0.5, 0.5, 1000.0, 50.0
    pos, vel, mass = lattice_sphere(2000, Mg, R)

    def arrival(r0):
        M, x = Ms + Mg * (r0 / R) ** 3, r_acc / r0
        return np.sqrt(r0**3 / (2 * G * M)) * (np.sqrt(x * (1 - x)) + np.arccos(np.sqrt(x)))

    T = arrival(R)
    gas = Gas("isothermal", cs=0.001 * KM_S, alpha=0.0, beta=0.0)
    snaps, hist, sinks, status = star_formation_leapfrog(pos, vel, mass, gas, 10.0, np.linspace(0, 0.9, 10) * T,
                                                         r_acc, np.inf, sinks=one_sink(Ms))
    r0 = np.linspace(1.0001 * r_acc, R, 4000)
    for s in snaps:
        predicted = Ms + Mg * (np.interp(s["t"], arrival(r0), r0, left=r_acc) / R) ** 3
        assert abs(s["sink_mass"][0] - predicted) < 0.008
        assert np.isclose(s["mass"].sum() + s["sink_mass"].sum(), Ms + Mg, rtol=1e-13)
    assert status == "done"


# ---------------------------------------------------------------- the rotating collapse itself (small version)

@pytest.fixture(scope="module")
def rotating_collapse():
    """The showcase cloud (1 Msun, 0.01 pc, rho ~ 1/r, 10 K, E_rot/|W| = 0.02, heated) at N = 3000, to 1.5 t_ff."""
    from nbody.clouds import perturbed_lattice_sphere, free_fall_time, solid_body_rotation
    from nbody.sph import ETA
    from nbody.units import PC
    M, R = 1.0, 0.01 * PC
    pos, _, mass = perturbed_lattice_sphere(3000, M, R, 0.1, np.random.default_rng(1))
    pos *= ((np.linalg.norm(pos, axis=1) / R) ** 0.5)[:, None]
    cs = 0.19 * KM_S
    rho_sink = jeans_resolution_density(cs, mass[0])
    r_acc = ETA * (mass[0] / rho_sink) ** (1 / 3)
    eps = 0.5 * r_acc
    vel = solid_body_rotation(pos, mass, eps, 0.02)
    T = free_fall_time(M, R)
    gas = Gas("barotropic", cs=cs, rho_crit=1e-13 / 5.94e-7, T_floor=10.0, T_1au=280.0)
    snaps, hist, sinks, status = star_formation_leapfrog(pos, vel, mass, gas, eps, np.linspace(0, 1.5, 7) * T, r_acc, rho_sink)
    return dict(pos0=pos, vel0=vel, mass0=mass, snaps=snaps, status=status, eps=eps, r_acc=r_acc)


def snapshot_totals(s):
    M = s["mass"].sum() + s["sink_mass"].sum()
    p = s["mass"] @ s["vel"] + s["sink_mass"] @ s["sink_vel"]
    L = (np.sum(s["mass"][:, None] * np.cross(s["pos"], s["vel"]), axis=0)
         + np.sum(s["sink_mass"][:, None] * np.cross(s["sink_pos"], s["sink_vel"]), axis=0) + s["sink_spin"].sum(axis=0))
    return M, p, L


def test_rotating_collapse_forms_a_single_star(rotating_collapse):
    # With the lattice symmetry broken and protostellar heating on, the cloud makes one star (Rung 4 decision).
    final = rotating_collapse["snaps"][-1]
    assert rotating_collapse["status"] == "done"
    assert len(final["sink_mass"]) == 1 and final["sink_mass"][0] > 0.5


def test_rotating_collapse_conserves_mass_momentum_and_angular_momentum(rotating_collapse):
    # Pair forces plus exact sink bookkeeping (orbit + spin): measured to ~1e-15 through every accretion event.
    M0, p0, L0 = snapshot_totals(rotating_collapse["snaps"][0])
    p_scale = np.sum(rotating_collapse["mass0"] * np.linalg.norm(rotating_collapse["vel0"], axis=1))
    for s in rotating_collapse["snaps"]:
        M, p, L = snapshot_totals(s)
        assert abs(M / M0 - 1) < 1e-12
        assert np.linalg.norm(p - p0) < 1e-12 * p_scale
        assert np.linalg.norm(L - L0) < 1e-12 * np.linalg.norm(L0)


def test_gas_collapse_stays_bound(rotating_collapse):
    # The collisionless cold collapse of Rung 2 lost ~20% of its mass. Dissipative gas must keep >= 95% (measured 100%).
    from nbody.disks import bound_fraction
    assert bound_fraction(rotating_collapse["snaps"][-1], rotating_collapse["eps"]) >= 0.95


def test_a_rotationally_supported_disk_forms(rotating_collapse):
    # At this low resolution (r_acc ~ 110 AU) the disk is small, but it must exist outside the accretion radius.
    from nbody.disks import central_frame, disk_members
    final = rotating_collapse["snaps"][-1]
    x, v, M_star = central_frame(final)
    disk = disk_members(x, v, M_star)
    assert final["mass"][disk].sum() > 0.01
    assert np.median(np.linalg.norm(x[disk, :2], axis=1)) > rotating_collapse["r_acc"]
