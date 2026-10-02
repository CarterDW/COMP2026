"""Rung 4 showcase run: a rotating 1 Msun cloud collapses into a protostar (sink) and a disk.

Saves snapshots and the sink's accretion history to rung4_protostar_disk/data/collapse.npz (~30 min at N = 12000).
Run from anywhere:  python path/to/run_collapse.py [N]
"""
import sys
import time
from pathlib import Path
import numpy as np
from nbody.clouds import perturbed_lattice_sphere, free_fall_time, solid_body_rotation
from nbody.integrators import star_formation_leapfrog
from nbody.sph import Gas, ETA
from nbody.sinks import jeans_resolution_density
from nbody.units import G, KM_S, PC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# Cloud: 1 Msun, R = 0.01 pc (2063 AU), rho ~ 1/r so it collapses inside-out, 10 K, E_rot/|W| = 0.02.
M, R, BETA = 1.0, 0.01 * PC, 0.02
CS = 0.19 * KM_S                                   # isothermal sound speed at 10 K, mu = 2.33
RHO_CRIT = 1e-13 / 5.94e-7                         # 1e-13 g/cm^3 in Msun/AU^3: the gas turns opaque
pos, _, mass = perturbed_lattice_sphere(N, M, R, 0.1, np.random.default_rng(1))
pos *= ((np.linalg.norm(pos, axis=1) / R) ** 0.5)[:, None]          # uniform -> rho ~ 1/r

# Sinks form where SPH stops resolving the Jeans mass; their accretion radius is the smoothing length there.
RHO_SINK = jeans_resolution_density(CS, mass[0])
R_ACC = ETA * (mass[0] / RHO_SINK) ** (1 / 3)
EPS = 0.5 * R_ACC
vel = solid_body_rotation(pos, mass, EPS, BETA)
T_FF = free_fall_time(M, R)
gas = Gas("barotropic", cs=CS, rho_crit=RHO_CRIT, T_floor=10.0, T_1au=280.0)

print(f"N = {len(mass)}, t_ff = {T_FF:.0f} yr, rho_sink = {RHO_SINK * 5.94e-7:.1e} g/cm^3, r_acc = {R_ACC:.1f} AU")
t0 = time.time()
snaps, history, sinks, status = star_formation_leapfrog(pos, vel, mass, gas, EPS, np.linspace(0, 2, 41) * T_FF,
                                                        R_ACC, RHO_SINK)
print(f"{status} after {len(history) - 1} steps, {time.time() - t0:.0f} s; sinks: {np.round(sinks.mass, 3)}")

hist_t = np.array([t for t, _ in history])
hist_m = np.array([m.sum() for _, m in history])             # total sink mass
np.savez(OUT / "collapse.npz", t_ff=T_FF, r_acc=R_ACC, eps=EPS, M=M, R=R, beta=BETA, cs=CS,
         init_pos=pos, init_vel=vel, init_mass=mass, hist_t=hist_t, hist_sink_mass=hist_m,
         snaps=np.array(snaps, dtype=object), allow_pickle=True)
print(f"saved {OUT / 'collapse.npz'}")
