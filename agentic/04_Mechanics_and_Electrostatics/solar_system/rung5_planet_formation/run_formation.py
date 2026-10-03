"""5f: grow planets in the giant-planet zone (4-30 AU) of the Rung 4 disk, from a disk age of 1 Myr to 3 Myr.

Embryos start at half their local isolation mass, spaced 10 mutual Hill radii apart; the rest of the solids are
planetesimal tracers. Saves data/<tag>.npz (snapshots every 10 kyr and the collision log).

    python path/to/run_formation.py [--pebbles] [--seed N] [--tag NAME] [--resume]

--pebbles   embryos also accrete drifting pebbles (nbody/gas_effects.py)
--migration embryos and giants migrate (Paardekooper et al. 2011 torques with the Kanagawa et al. 2018 gap factor)
--alpha-turb midplane turbulence for gaps and torques near planets (default: alpha_acc = 1e-3, as in the first runs)
--seed      random seed for the initial orbits (default 5); --tag names the output (default from the options)
--resume    continue from data/<tag>_checkpoint.npz, written every 100 kyr

Runs single-threaded (NUMBA_NUM_THREADS=1): at this N, threads cost more than they save, and it lets one run per core
go side by side. Use launch.sh to start runs detached, so closing the app or sleeping the laptop does not kill them.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"                 # before numba is imported
import sys
import time
from pathlib import Path
import numpy as np
import nbody.hybrid as hy
from nbody.gas_effects import evolve, RHO_SOLID
from nbody.viscous_disk import solid_surface_density, isolation_mass
from nbody.units import G, M_EARTH
from initial_disk import planet_forming_disk, T_START

HERE = Path(__file__).parent
(HERE / "data").mkdir(exist_ok=True)
R_IN, R_OUT, N_PLANETESIMALS = 4.0, 30.0, 400


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


PEBBLES = 1.0 if "--pebbles" in sys.argv else 0.0
MIGRATION = 1.0 if "--migration" in sys.argv else 0.0
SEED = int(option("--seed", 5))
TAG = option("--tag", ("formation_pebbles" if PEBBLES else "formation") + ("_migration" if MIGRATION else "")
             + ("" if SEED == 5 else f"_seed{SEED}"))

OUT_FILE = HERE / "data" / f"{TAG}.npz"
CHECKPOINT = HERE / "data" / f"{TAG}_checkpoint.npz"
T_END = 3e6
RHO_EMBRYO = 2.0 / 5.94e-7                       # 2 g/cm^3 in Msun/AU^3
rng = np.random.default_rng(SEED)

disk = planet_forming_disk()
M_star, mu = disk.M_star, G * disk.M_star
grid = np.linspace(R_IN, R_OUT, 4000)
sigma_s = solid_surface_density(grid, disk.sigma(grid, T_START))
M_solid = np.trapz(2 * np.pi * grid * sigma_s, grid)

# Embryos: half the local isolation mass, 10 mutual Hill radii apart.
a_emb, m_emb = [], []
a = R_IN
while a < R_OUT:
    m = 0.5 * isolation_mass(a, np.interp(a, grid, sigma_s), M_star)
    a_emb.append(a)
    m_emb.append(m)
    a *= 1 + 10 * (2 * m / (3 * M_star)) ** (1 / 3)              # next embryo 10 mutual Hill radii further out
a_emb, m_emb = np.array(a_emb), np.array(m_emb)
# Planetesimals: the remaining solid mass, placed following Sigma_s.
m_pl = (M_solid - m_emb.sum()) / N_PLANETESIMALS
cdf = np.cumsum(2 * np.pi * grid * sigma_s)
a_pl = np.interp(rng.uniform(0, cdf[-1], N_PLANETESIMALS), cdf, grid)
print(f"solids 4-30 AU: {M_solid / M_EARTH:.1f} Mearth; {len(a_emb)} embryos ({m_emb.min() / M_EARTH:.2f}-{m_emb.max() / M_EARTH:.2f} Mearth, "
      f"total {m_emb.sum() / M_EARTH:.1f}); {N_PLANETESIMALS} planetesimal tracers of {m_pl / M_EARTH:.3f} Mearth")


def orbits(a, e_rms, i_rms):
    """Heliocentric positions/velocities for orbits with Rayleigh-distributed e, i and random angles."""
    n = len(a)
    e, inc = rng.rayleigh(e_rms, n), rng.rayleigh(i_rms, n)
    node, peri, M = rng.uniform(0, 2 * np.pi, (3, n))
    E = M.copy()
    for _ in range(30):                                          # Kepler's equation
        E = M + e * np.sin(E)
    x_orb = np.stack([a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E), np.zeros(n)], 1)
    n_mean = np.sqrt(mu / a**3)
    v_orb = np.stack([-a * n_mean * np.sin(E), a * n_mean * np.sqrt(1 - e**2) * np.cos(E), np.zeros(n)], 1) / (1 - e * np.cos(E))[:, None]
    x, u = np.empty((n, 3)), np.empty((n, 3))
    for k in range(n):
        cO, sO, cw, sw, ci, si = np.cos(node[k]), np.sin(node[k]), np.cos(peri[k]), np.sin(peri[k]), np.cos(inc[k]), np.sin(inc[k])
        rot = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                        [sw * si, cw * si, ci]])
        x[k], u[k] = rot @ x_orb[k], rot @ v_orb[k]
    return x, u


x_e, u_e = orbits(a_emb, 0.01, 0.005)
x_p, u_p = orbits(a_pl, 0.02, 0.01)
x, u = np.vstack([x_e, x_p]), np.vstack([u_e, u_p])
m = np.concatenate([m_emb, np.full(N_PLANETESIMALS, m_pl)])
R = np.concatenate([(3 * m_emb / (4 * np.pi * RHO_EMBRYO)) ** (1 / 3), np.full(N_PLANETESIMALS, (3 * m_pl / (4 * np.pi * RHO_SOLID)) ** (1 / 3))])
big = np.concatenate([np.ones(len(m_emb), bool), np.zeros(N_PLANETESIMALS, bool)])
alive = np.ones(len(m), bool)
comp = np.column_stack([m, np.zeros((len(m), 3))])   # mass by origin: seed, collisions, pebbles, gas
Q, v = hy.from_heliocentric(x, u, m, M_star)

dt = R_IN**1.5 / np.sqrt(M_star) / 25           # 25 steps per orbit at the inner edge (energy error ~1e-6)
rc = hy.critical_radii(Q, v, m, alive, mu, dt)
ALPHA_TURB = float(option("--alpha-turb", disk.alpha))
disk_params = np.array([disk.M0, disk.R1, disk.t_nu, M_star, disk.alpha, T_START, PEBBLES, MIGRATION, ALPHA_TURB])
print(f"alpha_acc = {disk.alpha:g} (disk evolution, gas inflow), alpha_turb = {ALPHA_TURB:g} (gaps, torques)")
log, n_log = np.zeros((20000, 6)), 0
out_every = 1e4
steps_per_out = int(round(out_every / dt))
snap = dict(t=[], x=[], u=[], m=[], R=[], alive=[], big=[], comp=[])


def record(t):
    xh, uh = hy.to_heliocentric(Q, v, m, M_star)
    for key, value in zip(snap, (t, xh, uh, m, R, alive, big, comp)):
        snap[key].append(np.copy(value))


t, t0 = T_START, time.time()
if "--resume" in sys.argv and CHECKPOINT.exists():
    c = np.load(CHECKPOINT)
    Q, v, m, R, alive, big, comp, rc = (np.array(c[k]) for k in ("Q", "v", "m", "R", "alive", "big", "comp", "rc"))
    n_log, t = int(c["n_log"]), float(c["t"])
    log[:n_log] = c["log"]
    snap = {k: list(c["snap_" + k]) for k in snap}
    print(f"resumed {TAG} at {t / 1e6:.2f} Myr", flush=True)
else:
    record(t)
while t < T_END - 1e-6:
    n_log = evolve(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, steps_per_out, t, disk_params, log, n_log, 200.0)
    t += steps_per_out * dt
    record(t)
    if len(snap["t"]) % 10 == 1:
        np.savez(CHECKPOINT, Q=Q, v=v, m=m, R=R, alive=alive, big=big, comp=comp, rc=rc, n_log=n_log, t=t, log=log[:n_log],
                 **{"snap_" + k: np.array(vals) for k, vals in snap.items()})
        top = np.sort(m[alive & big])[::-1][:5] / M_EARTH
        print(f"t = {t / 1e6:.2f} Myr ({time.time() - t0:.0f} s): {alive.sum()} bodies, {n_log} mergers, "
              f"largest {np.round(top, 1)} Mearth", flush=True)

np.savez(OUT_FILE, **{k: np.array(vals) for k, vals in snap.items()}, log=log[:n_log],
         M_star=M_star, dt=dt, disk_params=disk_params, M_solid=M_solid)
CHECKPOINT.unlink(missing_ok=True)
print(f"done in {time.time() - t0:.0f} s; saved {OUT_FILE}")
