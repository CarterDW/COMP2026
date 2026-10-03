"""Rung 6 smoke test: a short (default 0.5 Myr) gas-free run of terrestrial embryos with the Rung 5 giant.

Embryos come from our own disk (Rung 5's viscous disk: rock inside the snow line, rock + ice outside), each at its
isolation mass and 10 mutual Hill radii apart, from 0.7 to 3.5 AU. The giant is the Rung 5 survivor. Gravity and
mergers only. Bodies inside 0.2 AU (hit the star) or beyond 100 AU (ejected) are removed and their energy tallied.

    python path/to/smoke_test.py [T_MYR]            -> data/smoke.npz, then the checks are printed
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
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

(HERE / "data").mkdir(exist_ok=True)
T_RUN = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 0.5e6
R_IN, R_OUT, R_STAR_HIT, R_EJECT = 0.7, 3.5, 0.2, 100.0
RHO_ROCK = 3.0 / 5.94e-7                                         # 3 g/cm^3 in Msun/AU^3
rng = np.random.default_rng(6)

giant = np.load(HERE.parent / "rung5_planet_formation" / "handoff.npz")
M_star = float(giant["M_star"])
mu = G * M_star
disk = planet_forming_disk()
grid = np.linspace(R_IN, R_OUT, 4000)
sigma_s = solid_surface_density(grid, disk.sigma(grid, T_START))
a_emb, m_emb = [], []
a = R_IN
while a < R_OUT:
    mi = isolation_mass(a, np.interp(a, grid, sigma_s), M_star)
    a_emb.append(a)
    m_emb.append(mi)
    a *= 1 + 10 * (2 * mi / (3 * M_star)) ** (1 / 3)
a_emb, m_emb = np.array(a_emb), np.array(m_emb)
n_emb = len(a_emb)

# Nearly circular, nearly coplanar embryos (e, i ~ 0.01) at random phases; then the giant.
e, inc = rng.rayleigh(0.01, n_emb), rng.rayleigh(0.005, n_emb)
phase, node = rng.uniform(0, 2 * np.pi, (2, n_emb))
r = a_emb * (1 - e)
vp = np.sqrt(mu * (1 + e) / r)
x = np.stack([r * np.cos(phase), r * np.sin(phase), np.zeros(n_emb)], 1)
u = np.stack([-vp * np.sin(phase), vp * np.cos(phase) * np.cos(inc), vp * np.cos(phase) * np.sin(inc)], 1)
x, u = np.vstack([x, giant["x"]]), np.vstack([u, giant["u"]])
m = np.concatenate([m_emb, giant["m"]])
R = np.concatenate([(3 * m_emb / (4 * np.pi * RHO_ROCK)) ** (1 / 3), giant["R"]])
n = len(m)
big, alive = np.ones(n, bool), np.ones(n, bool)
comp = np.column_stack([m, np.zeros((n, 3))])
Q, v = hy.from_heliocentric(x, u, m, M_star)
print(f"{n_emb} embryos, {m_emb.min() / M_EARTH:.3f}-{m_emb.max() / M_EARTH:.3f} Mearth (total {m_emb.sum() / M_EARTH:.2f}), "
      f"0.7-{a_emb.max():.2f} AU; giant {giant['m'][0] / M_EARTH:.0f} Mearth at {giant['a'][0]:.2f} AU", flush=True)

dt = R_IN**1.5 / np.sqrt(M_star) / 15
rc = hy.critical_radii(Q, v, m, alive, mu, dt)


@njit
def run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, n_steps, t0, log, n_log):
    for k in range(n_steps):
        n_before = n_log
        n_log = hy.step(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t0 + k * dt)
        if n_log > n_before:
            rc[:] = hy.critical_radii(Q, v, m, alive, mu, dt)
    return n_log


log, n_log = np.zeros((5000, 6)), 0
E0 = hy.total_energy(Q, v, m, alive, big, mu, M_star)
E_removed, M_removed = 0.0, {"star": 0.0, "ejected": 0.0}
out_every = 5000.0
steps = int(round(out_every / dt))
snap = dict(t=[], x=[], u=[], m=[], alive=[])
t, t_wall = 0.0, time.time()


def record():
    xh, uh = hy.to_heliocentric(Q, v, m, M_star)
    for key, value in zip(snap, (t, xh, uh, m, alive)):
        snap[key].append(np.copy(value))


record()
while t < T_RUN - 1e-6:
    n_log = run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, steps, t, log, n_log)
    t += steps * dt
    r_now = np.linalg.norm(Q, axis=1)
    gone = alive & ((r_now < R_STAR_HIT) | (r_now > R_EJECT))
    if gone.any():                                          # remove, and tally the energy they take away
        E_before = hy.total_energy(Q, v, m, alive, big, mu, M_star)
        for i in np.flatnonzero(gone):
            M_removed["star" if r_now[i] < R_STAR_HIT else "ejected"] += m[i]
            alive[i], m[i] = False, 0.0
        E_removed += E_before - hy.total_energy(Q, v, m, alive, big, mu, M_star)
        rc = hy.critical_radii(Q, v, m, alive, mu, dt)
    record()
    if len(snap["t"]) % 10 == 1:
        dE = (hy.total_energy(Q, v, m, alive, big, mu, M_star) + log[:n_log, 5].sum() + E_removed) / E0 - 1
        print(f"t = {t / 1e6:.3f} Myr ({time.time() - t_wall:.0f} s): {int(alive[:n_emb].sum())} embryos left, "
              f"{n_log} mergers, budget dE/E {dE:.1e}", flush=True)

np.savez(HERE / "data" / "smoke.npz", **{k: np.array(vals) for k, vals in snap.items()}, log=log[:n_log],
         comp=comp, n_emb=n_emb, M_star=M_star, E0=E0, E_removed=E_removed, M_star_hit=M_removed["star"],
         M_ejected=M_removed["ejected"], a_emb0=a_emb, m_emb0=m_emb)
print(f"done in {time.time() - t_wall:.0f} s", flush=True)
