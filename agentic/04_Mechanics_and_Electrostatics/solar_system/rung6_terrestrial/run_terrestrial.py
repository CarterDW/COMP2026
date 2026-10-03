"""Rung 6: late-stage terrestrial planet assembly, gas-free, with the Rung 5 giant (production runs).

Embryos come from our disk (rock inside the snow line, rock + ice outside), each at its isolation mass and 10 mutual
Hill radii apart, from 0.7 to 4 AU; the count follows from the disk. The giant is the Rung 5 survivor. Gravity and
perfect mergers only; bodies inside 0.2 AU (hit the star) or beyond 100 AU (ejected) are removed, with their energy
and mass tallied. Snapshots every 10 kyr; a checkpoint every 100 kyr.

    python path/to/run_terrestrial.py --seed N [--myr T] [--planetesimals N] [--resume]   (launch.sh starts them detached)

--planetesimals N: embryos start at half their isolation mass and the other half of the solids is N planetesimals
(following Sigma_s) that feel embryos and the giant but not each other. Their gravity damps the embryos' eccentricities
(dynamical friction; O'Brien et al. 2006). Steps resolve fast pericenter passages (step_resolving_pericenters).
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"                 # one core per run: many seeds side by side
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
import nbody.hybrid as hy
from nbody.viscous_disk import solid_surface_density, isolation_mass
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "rung5_planet_formation"))
from initial_disk import planet_forming_disk, T_START          # noqa: E402  (Rung 5's disk handoff)


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SEED = int(option("--seed", 1))
T_RUN = float(option("--myr", 20)) * 1e6
N_PL = int(option("--planetesimals", 0))
TAG = f"terrestrial_seed{SEED}" + (f"_pl{N_PL}" if N_PL else "")
OUT, CHECKPOINT = HERE / "data" / f"{TAG}.npz", HERE / "data" / f"{TAG}_checkpoint.npz"
(HERE / "data").mkdir(exist_ok=True)
R_IN, R_OUT, R_STAR_HIT, R_EJECT = 0.7, 4.0, 0.2, 100.0
RHO_ROCK = 3.0 / 5.94e-7                                         # 3 g/cm^3 in Msun/AU^3
rng = np.random.default_rng(SEED)

giant = np.load(HERE.parent / "rung5_planet_formation" / "handoff.npz")
M_star = float(giant["M_star"])
mu = G * M_star
disk = planet_forming_disk()
grid = np.linspace(R_IN, R_OUT, 4000)
sigma_s = solid_surface_density(grid, disk.sigma(grid, T_START))
EMBRYO_SHARE = 0.5 if N_PL else 1.0                              # fraction of the local solids in embryos
a_emb, m_emb = [], []
a = R_IN
while a < R_OUT:
    mi = EMBRYO_SHARE * isolation_mass(a, np.interp(a, grid, sigma_s), M_star)
    a_emb.append(a)
    m_emb.append(mi)
    a *= 1 + 10 * (2 * mi / (3 * M_star)) ** (1 / 3)
a_emb, m_emb = np.array(a_emb), np.array(m_emb)
n_emb = len(a_emb)

# Nearly circular, nearly coplanar embryos at random phases (the seed changes only these); then the giant.
e, inc = rng.rayleigh(0.01, n_emb), rng.rayleigh(0.005, n_emb)
phase = rng.uniform(0, 2 * np.pi, n_emb)
r = a_emb * (1 - e)
vp = np.sqrt(mu * (1 + e) / r)
x = np.stack([r * np.cos(phase), r * np.sin(phase), np.zeros(n_emb)], 1)
u = np.stack([-vp * np.sin(phase), vp * np.cos(phase) * np.cos(inc), vp * np.cos(phase) * np.sin(inc)], 1)
# Planetesimals: the rest of the solid mass between R_IN and R_OUT, placed following Sigma_s, slightly stirred.
M_solid = np.trapz(2 * np.pi * grid * sigma_s, grid)
m_pl = (M_solid - m_emb.sum()) / N_PL if N_PL else 0.0
cdf = np.cumsum(2 * np.pi * grid * sigma_s)
a_pl = np.interp(rng.uniform(0, cdf[-1], N_PL), cdf, grid)
e_pl, i_pl, ph_pl = rng.rayleigh(0.01, N_PL), rng.rayleigh(0.005, N_PL), rng.uniform(0, 2 * np.pi, N_PL)
r_pl = a_pl * (1 - e_pl)
vp_pl = np.sqrt(mu * (1 + e_pl) / r_pl)
x_pl = np.stack([r_pl * np.cos(ph_pl), r_pl * np.sin(ph_pl), np.zeros(N_PL)], 1)
u_pl = np.stack([-vp_pl * np.sin(ph_pl), vp_pl * np.cos(ph_pl) * np.cos(i_pl), vp_pl * np.cos(ph_pl) * np.sin(i_pl)], 1)

# Order: embryos, planetesimals, giant (the giant is always last).
x, u = np.vstack([x, x_pl, giant["x"]]), np.vstack([u, u_pl, giant["u"]])
m = np.concatenate([m_emb, np.full(N_PL, m_pl), giant["m"]])
R = np.concatenate([(3 * m_emb / (4 * np.pi * RHO_ROCK)) ** (1 / 3), np.full(N_PL, (3 * m_pl / (4 * np.pi * RHO_ROCK)) ** (1 / 3)),
                    giant["R"]])
n = len(m)
big = np.concatenate([np.ones(n_emb, bool), np.zeros(N_PL, bool), [True]])
alive = np.ones(n, bool)
comp = np.column_stack([m, np.zeros((n, 3))])
Q, v = hy.from_heliocentric(x, u, m, M_star)
dt = R_IN**1.5 / np.sqrt(M_star) / 15
rc = hy.critical_radii(Q, v, m, alive, mu, dt)


@njit
def run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, n_steps, t0, log, n_log):
    for k in range(n_steps):
        n_before = n_log
        n_log = hy.step_resolving_pericenters(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t0 + k * dt)
        if n_log > n_before:
            rc[:] = hy.critical_radii(Q, v, m, alive, mu, dt)
    return n_log


log, n_log = np.zeros((5000, 6)), 0
E0 = hy.total_energy(Q, v, m, alive, big, mu, M_star)
E_removed, M_star_hit, M_ejected = 0.0, 0.0, 0.0
steps = int(round(1e4 / dt))                                   # one snapshot per 10 kyr
snap = dict(t=[], x=[], u=[], m=[], alive=[])
t = 0.0


def record():
    xh, uh = hy.to_heliocentric(Q, v, m, M_star)
    for key, value in zip(snap, (t, xh, uh, m, alive)):
        snap[key].append(np.copy(value))


if "--resume" in sys.argv and CHECKPOINT.exists():
    c = np.load(CHECKPOINT)
    Q, v, m, R, alive, comp, rc = (np.array(c[k]) for k in ("Q", "v", "m", "R", "alive", "comp", "rc"))
    t, n_log, E_removed = float(c["t"]), int(c["n_log"]), float(c["E_removed"])
    M_star_hit, M_ejected = float(c["M_star_hit"]), float(c["M_ejected"])
    log[:n_log] = c["log"]
    snap = {k: list(c["snap_" + k]) for k in snap}
    print(f"resumed {TAG} at {t / 1e6:.2f} Myr", flush=True)
else:
    print(f"{TAG}: {N_PL} planetesimals of {m_pl / M_EARTH:.4f} Mearth; " if N_PL else f"{TAG}: ", end="")
    print(f"{n_emb} embryos, {m_emb.min() / M_EARTH:.3f}-{m_emb.max() / M_EARTH:.3f} Mearth "
          f"(total {m_emb.sum() / M_EARTH:.2f}), 0.7-{a_emb.max():.2f} AU; giant {giant['m'][0] / M_EARTH:.0f} Mearth "
          f"at {giant['a'][0]:.2f} AU; dt = {dt:.4f} yr", flush=True)
    record()

t_wall = time.time()
while t < T_RUN - 1e-6:
    n_log = run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, steps, t, log, n_log)
    t += steps * dt
    r_now = np.linalg.norm(Q, axis=1)
    gone = alive & ((r_now < R_STAR_HIT) | (r_now > R_EJECT))
    if gone.any():
        E_before = hy.total_energy(Q, v, m, alive, big, mu, M_star)
        for i in np.flatnonzero(gone):
            if r_now[i] < R_STAR_HIT:
                M_star_hit += m[i]
            else:
                M_ejected += m[i]
            alive[i], m[i], comp[i] = False, 0.0, 0.0
        E_removed += E_before - hy.total_energy(Q, v, m, alive, big, mu, M_star)
        rc = hy.critical_radii(Q, v, m, alive, mu, dt)
    record()
    if len(snap["t"]) % 10 == 1:
        dE = (hy.total_energy(Q, v, m, alive, big, mu, M_star) + log[:n_log, 5].sum() + E_removed) / E0 - 1
        np.savez(CHECKPOINT, Q=Q, v=v, m=m, R=R, alive=alive, comp=comp, rc=rc, t=t, n_log=n_log, log=log[:n_log],
                 E_removed=E_removed, M_star_hit=M_star_hit, M_ejected=M_ejected,
                 **{"snap_" + k: np.array(vals) for k, vals in snap.items()})
        top = np.sort(m[:n_emb][alive[:n_emb]])[::-1][:4] / M_EARTH
        print(f"t = {t / 1e6:6.2f} Myr ({time.time() - t_wall:6.0f} s): {int(alive[:n_emb].sum())} embryos, "
              f"{int(alive[n_emb:n_emb + N_PL].sum())} planetesimals, "
              f"{n_log} mergers, largest {np.round(top, 2)} Me, budget dE/E {dE:.1e}", flush=True)

np.savez(OUT, **{k: np.array(vals) for k, vals in snap.items()}, log=log[:n_log], comp=comp, n_emb=n_emb, n_pl=N_PL,
         M_star=M_star, E0=E0, E_removed=E_removed, M_star_hit=M_star_hit, M_ejected=M_ejected, a_emb0=a_emb,
         m_emb0=m_emb, seed=SEED)
CHECKPOINT.unlink(missing_ok=True)
print(f"done in {time.time() - t_wall:.0f} s; saved {OUT}", flush=True)
