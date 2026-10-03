"""Checks and a figure for the Rung 6 smoke test (data/smoke.npz).

Run from anywhere:  python path/to/analyze_smoke.py      -> printed checks and plots/smoke_test.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nbody.plotstyle import BLUE, ORANGE, INK, MUTED, GRID, YELLOW, style, save
from nbody.orbits import orbital_elements
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
d = np.load(HERE / "data" / "smoke.npz")
t, x, u, m, alive = d["t"], d["x"], d["u"], d["m"], d["alive"]
n_emb, M_star = int(d["n_emb"]), float(d["M_star"])
mu = G * M_star
a_giant = orbital_elements(x[-1, -1:], u[-1, -1:], mu)[0][0]
emb = np.arange(n_emb)

# 1. Budgets
log = d["log"]
print(f"1. energy: (E_now + merger losses + removed) / E0 - 1 is tracked during the run (see the log)")
M0 = d["m_emb0"].sum()
M_end = m[-1, :n_emb][alive[-1, :n_emb]].sum()
gained_by_giant = m[-1, -1] - m[0, -1]
print(f"   mass: embryos start {M0 / M_EARTH:.3f} Me = survivors {M_end / M_EARTH:.3f} + giant gained {gained_by_giant / M_EARTH:.3f} "
      f"+ star {float(d['M_star_hit']) / M_EARTH:.3f} + ejected {float(d['M_ejected']) / M_EARTH:.3f} "
      f"(residual {(M0 - M_end - gained_by_giant - float(d['M_star_hit']) - float(d['M_ejected'])) / M0:.1e})")

# 2. Stirring
e0 = orbital_elements(x[0, :n_emb], u[0, :n_emb], mu)[1]
a_end, e_end, i_end = orbital_elements(x[-1, :n_emb], u[-1, :n_emb], mu)
live = alive[-1, :n_emb]
print(f"2. eccentricity: median {np.median(e0):.3f} -> {np.median(e_end[live]):.3f}, max {e0.max():.3f} -> {e_end[live].max():.3f}")

# 3. Growth
n_alive = alive[:, :n_emb].sum(axis=1)
largest = (m[:, :n_emb] * alive[:, :n_emb]).max(axis=1) / M_EARTH
print(f"3. growth: {n_alive[0]} -> {n_alive[-1]} embryos ({len(log)} mergers); largest {largest[0]:.2f} -> {largest[-1]:.2f} Me")

# 4. Resonances with the giant: a_res = a_giant (q/p)^(2/3) for a p:q mean-motion resonance
resonances = {"3:1": (1 / 3) ** (2 / 3), "5:2": (2 / 5) ** (2 / 3), "2:1": (1 / 2) ** (2 / 3)}
a_res = {k: a_giant * f for k, f in resonances.items()}
near = lambda lo, hi: live & (a_end > lo) & (a_end < hi)
inner, resonant = near(0.7, 2.5), near(a_res["3:1"] - 0.15, a_res["5:2"] + 0.15)
print(f"4. resonances at {', '.join(f'{k} {v:.2f} AU' for k, v in a_res.items())}: median e inside 2.5 AU "
      f"{np.median(e_end[inner]):.3f} vs. near 3:1-5:2 {np.median(e_end[resonant]) if resonant.any() else float('nan'):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.4, 1, 1]})
ax = axes[0]
ax.scatter(d["a_emb0"], e0, s=20 * (d["m_emb0"] / M_EARTH) ** (2 / 3) * 10, facecolor="none", edgecolor=MUTED, label="start")
ax.scatter(a_end[live], e_end[live], s=20 * (m[-1, :n_emb][live] / M_EARTH) ** (2 / 3) * 10, color=BLUE, label=f"{t[-1] / 1e6:.1f} Myr")
for k, a_r in a_res.items():
    ax.axvline(a_r, color=ORANGE, lw=1, ls=":")
    ax.text(a_r, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.1, k, color=ORANGE, fontsize=8, ha="center")
ax.set_xlabel("semi-major axis  [AU]")
ax.set_ylabel("eccentricity")
ax.set_title("Stirring, and the giant's resonances", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8)
axes[1].plot(t / 1e6, n_alive, color=BLUE, lw=2)
axes[1].set_xlabel("time  [Myr]")
axes[1].set_ylabel("embryos left")
axes[1].set_title("Collisions merge embryos", color=INK, fontsize=10)
axes[2].plot(t / 1e6, largest, color=BLUE, lw=2)
axes[2].axhline(1.0, color=YELLOW, lw=1, ls=":")
axes[2].set_ylim(0, max(1.15, 1.1 * largest.max()))
axes[2].text(t[0] / 1e6, 1.02, "Earth", color=MUTED, fontsize=8)
axes[2].set_xlabel("time  [Myr]")
axes[2].set_ylabel("largest embryo  [Mearth]")
axes[2].set_title("The biggest grows", color=INK, fontsize=10)
for a_ in axes:
    style(a_)
(HERE / "plots").mkdir(exist_ok=True)
save(fig, HERE / "plots", "smoke_test.png")
