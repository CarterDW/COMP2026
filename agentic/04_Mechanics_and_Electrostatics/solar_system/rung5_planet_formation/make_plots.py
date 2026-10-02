"""Rung 5 figures and animation for one run, from data/<tag>.npz (run run_formation.py first).

Run from anywhere:  python path/to/make_plots.py TAG [--handoff]
Figures go to plots/<tag>/. With --handoff, this run's planets are written to handoff.npz for Rung 6.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, GRID, style, save
from nbody.orbits import orbital_elements
from nbody.viscous_disk import solid_surface_density, isolation_mass
from nbody.gas_effects import M_CRIT
from nbody.units import G, M_EARTH
from initial_disk import planet_forming_disk, T_START

HERE = Path(__file__).parent
TAG = sys.argv[1]
PLOTS = HERE / "plots" / TAG
PLOTS.mkdir(parents=True, exist_ok=True)
d = np.load(HERE / "data" / f"{TAG}.npz")
t, M_star = d["t"], float(d["M_star"])
mu = G * M_star
n_snap = len(t)
elements = [orbital_elements(d["x"][k], d["u"][k], mu) for k in range(n_snap)]
alive, big, mass = d["alive"], d["big"], d["m"]
final = n_snap - 1
planets = np.flatnonzero(alive[final] & big[final])
a_f, e_f, i_f = (q[planets] for q in elements[final])
m_f = mass[final, planets] / M_EARTH
order = np.argsort(a_f)
print("final bodies (a [AU], e, i [deg], m [Mearth]):")
for k in order:
    print(f"  {a_f[k]:6.2f}  {e_f[k]:.3f}  {np.degrees(i_f[k]):5.2f}  {m_f[k]:8.2f}")
print(f"{len(d['log'])} mergers; solids 4-30 AU at the start: {float(d['M_solid']) / M_EARTH:.1f} Mearth")

GIANTS = [("Jupiter", 5.20, 317.8), ("Saturn", 9.58, 95.2), ("Uranus", 19.2, 14.5), ("Neptune", 30.1, 17.1)]

# ---------------------------------------------------------------- 1. final system vs isolation mass and the real giants
disk = planet_forming_disk()
grid = np.linspace(4, 30, 300)
m_iso = isolation_mass(grid, solid_surface_density(grid, disk.sigma(grid, T_START)), M_star) / M_EARTH
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.semilogy(grid, m_iso, color=MUTED, lw=1.5, ls="--", label="isolation mass (10 Hill radii of solids)")
ax.axhline(M_CRIT / M_EARTH, color=ORANGE, lw=1, ls=":")
ax.text(29.5, M_CRIT / M_EARTH * 1.12, "core mass for runaway gas accretion (10 Mearth)", color=ORANGE, fontsize=8, ha="right")
start = np.flatnonzero(big[0])
ax.semilogy(elements[0][0][start], mass[0, start] / M_EARTH, "o", mfc="white", mec=BLUE, ms=6, label="embryos at 1 Myr")
ax.semilogy(a_f, m_f, "o", color=BLUE, ms=9, label=f"planets at {t[final] / 1e6:.0f} Myr")
for name, a_g, m_g in GIANTS:
    ax.semilogy(a_g, m_g, "*", color=YELLOW, mec=INK, ms=13)
    ax.annotate(name, (a_g, m_g), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
ax.plot([], [], "*", color=YELLOW, mec=INK, ms=11, label="our solar system")
ax.set_xscale("log")
ax.set_xlim(1, 40)
ax.set_xticks([1, 2, 5, 10, 20, 30], ["1", "2", "5", "10", "20", "30"])
ax.axvspan(1, 4, color=GRID, alpha=0.5)
ax.text(1.1, 0.5, "inside the simulated zone's\ninner edge (scattered inward)", fontsize=7, color=MUTED)
ax.set_xlabel("semi-major axis  [AU]")
ax.set_ylabel("mass  [Mearth]")
ax.set_title("What grew in the giant-planet zone in 2 Myr", color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
save(fig, PLOTS, "final_system.png")

# ---------------------------------------------------------------- 2. growth of each embryo
fig, ax = plt.subplots(figsize=(8, 4.8))
for j in np.flatnonzero(big[0]):
    life = alive[:, j]
    ax.semilogy(t[life] / 1e6, mass[life, j] / M_EARTH, lw=1.6, color=BLUE if alive[final, j] else GRID)
ax.axhline(M_CRIT / M_EARTH, color=ORANGE, lw=1, ls=":")
ax.set_xlabel("disk age  [Myr]")
ax.set_ylabel("embryo mass  [Mearth]")
ax.set_title("Embryo growth (grey: absorbed by another embryo)", color=INK)
style(ax)
save(fig, PLOTS, "mass_growth.png")

# ---------------------------------------------------------------- 3. animation: the a-e plane
fig, (ae, gr) = plt.subplots(2, 1, figsize=(7.5, 7), gridspec_kw={"height_ratios": [1.6, 1]})
small_dots = ae.scatter([], [], s=4, color=MUTED)
big_dots = ae.scatter([], [], s=[], color=BLUE, edgecolor=INK, linewidth=0.4, zorder=3)
for name, a_g, m_g in GIANTS:
    ae.axvline(a_g, color=YELLOW, lw=1, alpha=0.6)
    ae.text(a_g, 0.235, name, fontsize=7, color=MUTED, ha="center")
ae.set_xlim(1, 32)
ae.set_ylim(0, 0.25)
ae.set_xlabel("semi-major axis  [AU]")
ae.set_ylabel("eccentricity")
clock = ae.text(0.02, 0.92, "", transform=ae.transAxes, fontsize=10, color=INK)
gr.set_xlim(t[0] / 1e6, t[-1] / 1e6)
gr.set_ylim(0.3, max(30, 1.5 * mass[:, big[0]].max() / M_EARTH))
gr.set_yscale("log")
gr.axhline(M_CRIT / M_EARTH, color=ORANGE, lw=1, ls=":")
gr.set_xlabel("disk age  [Myr]")
gr.set_ylabel("mass  [Mearth]")
lines = {j: gr.plot([], [], lw=1.4, color=BLUE)[0] for j in np.flatnonzero(big[0])}
now = gr.axvline(t[0] / 1e6, color=MUTED, lw=0.8)
for ax in (ae, gr):
    style(ax)
fig.tight_layout()


def draw(k):
    a, e, _ = elements[k]
    sm, bg = alive[k] & ~big[k], alive[k] & big[k]
    small_dots.set_offsets(np.column_stack([a[sm], e[sm]]))
    big_dots.set_offsets(np.column_stack([a[bg], e[bg]]))
    big_dots.set_sizes(25 * (mass[k, bg] / M_EARTH) ** (2 / 3))
    for j, line in lines.items():
        life = alive[:k + 1, j]
        line.set_data(t[:k + 1][life] / 1e6, mass[:k + 1, j][life] / M_EARTH)
        line.set_color(BLUE if alive[k, j] else GRID)
    now.set_xdata([t[k] / 1e6])
    clock.set_text(f"disk age {t[k] / 1e6:.2f} Myr    {sm.sum()} planetesimals, {bg.sum()} embryos")
    return [small_dots, big_dots, now, clock] + list(lines.values())


anim = FuncAnimation(fig, draw, frames=n_snap, interval=80, blit=True)
anim.save(PLOTS / "formation.gif", writer=PillowWriter(fps=12), dpi=80)
print(f"saved {PLOTS / 'formation.gif'}")

# ---------------------------------------------------------------- handoff for Rung 6: the planets and their orbits
if "--handoff" not in sys.argv:
    sys.exit()
np.savez(HERE / "handoff.npz", M_star=M_star, t=t[final], x=d["x"][final, planets], u=d["u"][final, planets],
         m=mass[final, planets], R=d["R"][final, planets], a=a_f, e=e_f, inc=i_f)
print(f"saved {HERE / 'handoff.npz'}: {len(planets)} planets")
