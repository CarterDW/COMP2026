"""Rung 6 check: the giant's mean-motion resonances in a ring of massless test particles (a Kirkwood-gap analog).

300 test particles on circular orbits, 2-4.5 AU, feel only the star and the Rung 5 giant. Particles in a p:q
resonance, at a = a_giant (q/p)^(2/3), have their eccentricities pumped. First-order resonances (2:1) act even on a
circular giant; higher-order ones (3:1, 5:2) scale with the giant's eccentricity, which is tiny (0.003) here.

    python path/to/resonance_test.py [T_KYR]          -> printed checks and plots/resonances.png
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import nbody.hybrid as hy
from nbody.orbits import orbital_elements
from nbody.plotstyle import BLUE, ORANGE, INK, MUTED, style, save
from nbody.units import G

HERE = Path(__file__).parent
T_RUN = float(sys.argv[1]) * 1e3 if len(sys.argv) > 1 else 500e3
N_TEST = 300
giant = np.load(HERE.parent / "rung5_planet_formation" / "handoff.npz")
M_star = float(giant["M_star"])
mu = G * M_star
a_g = float(giant["a"][0])
e_g = float(giant["e"][0])

rng = np.random.default_rng(7)
a0 = np.linspace(2.0, 4.5, N_TEST)
phase = rng.uniform(0, 2 * np.pi, N_TEST)
inc = rng.rayleigh(0.002, N_TEST)
vc = np.sqrt(mu / a0)
x = np.stack([a0 * np.cos(phase), a0 * np.sin(phase), np.zeros(N_TEST)], 1)
u = np.stack([-vc * np.sin(phase), vc * np.cos(phase) * np.cos(inc), vc * np.cos(phase) * np.sin(inc)], 1)
x, u = np.vstack([x, giant["x"]]), np.vstack([u, giant["u"]])
m = np.concatenate([np.full(N_TEST, 1e-15), giant["m"]])
R = np.concatenate([np.full(N_TEST, 1e-12), giant["R"]])
big = np.concatenate([np.zeros(N_TEST, bool), [True]])
alive = np.ones(N_TEST + 1, bool)
comp = np.column_stack([m, np.zeros((len(m), 3))])
Q, v = hy.from_heliocentric(x, u, m, M_star)
dt = 2.0**1.5 / np.sqrt(M_star) / 20
rc = hy.critical_radii(Q, v, m, alive, mu, dt)


@njit
def run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, n_steps, log):
    n_log = 0
    for k in range(n_steps):
        n_log = hy.step(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, k * dt)
    return n_log


t0 = time.time()
n_steps = int(T_RUN / dt)
log = np.zeros((1000, 6))
n_log = run_steps(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, n_steps, log)
xh, uh = hy.to_heliocentric(Q, v, m, M_star)
a_end, e_end, _ = orbital_elements(xh[:N_TEST], uh[:N_TEST], mu)
print(f"{T_RUN / 1e3:.0f} kyr in {time.time() - t0:.0f} s; giant a = {a_g:.2f} AU, e = {e_g:.4f}; {n_log} test particles hit the giant")

RES = {"2:1": (1, 2), "3:1": (1, 3), "5:2": (2, 5), "7:3": (3, 7), "5:3": (3, 5)}
a_res = {k: a_g * (q / p) ** (2 / 3) for k, (q, p) in RES.items()}
live = alive[:N_TEST]
far = np.ones(N_TEST, bool)
for k, ar in a_res.items():
    near = live & (np.abs(a0 - ar) < 0.03)
    far &= np.abs(a0 - ar) > 0.1
    if near.any():
        print(f"  {k} at {ar:.2f} AU: max e of particles that started within 0.03 AU = {e_end[near].max():.3f}")
print(f"  away from resonances (> 0.1 AU from all): median e = {np.median(e_end[live & far]):.4f}, max {e_end[live & far].max():.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogy(a0[live], np.maximum(e_end[live], 1e-5), "o", color=BLUE, ms=3)
for k, ar in a_res.items():
    ax.axvline(ar, color=ORANGE, lw=1, ls=":")
    ax.text(ar, 0.6, k, color=ORANGE, fontsize=9, ha="center")
ax.set_ylim(1e-4, 1)
ax.set_xlabel("starting semi-major axis  [AU]")
ax.set_ylabel(f"eccentricity after {T_RUN / 1e3:.0f} kyr")
ax.set_title(f"Test particles near our giant's resonances (giant: {a_g:.2f} AU, e = {e_g:.3f})", color=INK, fontsize=10)
style(ax)
(HERE / "plots").mkdir(exist_ok=True)
save(fig, HERE / "plots", "resonances.png")
