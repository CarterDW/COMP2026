"""Rung 3 figures: SPH gas under self-gravity.

Run from anywhere:  python path/to/make_plots.py   (~4 minutes)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID, SEQUENTIAL, style, save
from nbody.clouds import lattice_sphere, free_fall_time, cycloid_radius
from nbody.integrators import sph_leapfrog
from nbody.sph import Gas
from nbody.gravity import potential_energy
from nbody.diagnostics import kinetic_energy
from nbody.units import G, PC

PLOTS = Path(__file__).parent / "plots"

# ---------------------------------------------------------------- 1. Evrard collapse vs published runs
pos, vel, mass = lattice_sphere(4224, 1.0, 1.0)
pos *= (np.linalg.norm(pos, axis=1) ** 0.5)[:, None]            # uniform -> rho ~ 1/r
T_EV = np.sqrt(1 / G)                                             # time unit sqrt(R^3 / G M)
ev = sph_leapfrog(pos, vel, 0.05 * G, mass, Gas("adiabatic", gamma=5 / 3), 0.01, np.linspace(0, 3, 151) * T_EV)
t_ev = ev["t"] / T_EV
K = np.array([kinetic_energy(v, mass) for v in ev["vel"]]) / G
U = np.array([np.sum(mass * u) for u in ev["u"]]) / G
W = np.array([potential_energy(x, mass, 0.01) for x in ev["pos"]]) / G
print(f"Evrard N = {len(mass)}: {ev['n_steps']} steps, max |dE/E| = {np.abs((K + U + W) / (K + U + W)[0] - 1).max():.1e}")

# Reference values read off published figures (+-0.02): GADGET-1 SPH with 4224 particles (Springel, Yoshida &
# White 2001, Fig. 10) and the high-resolution 1D PPM solution (Steinmetz & Mueller 1993, via Frontiere+ 2022 Fig. 8).
gadget = {"K": [(0.90, 0.29), (2, 0.10), (3, 0.07)], "U": [(1.15, 1.41), (2, 0.51), (3, 0.67)],
          "W": [(1.15, -2.12), (2, -1.22), (3, -1.35)]}
ppm = {"K": [(0.87, 0.45), (2, 0.22)], "U": [(1.06, 1.75), (2, 0.73)], "W": [(1.05, -2.51), (2, -1.55)]}

fig, ax = plt.subplots(figsize=(8.5, 6.2))
for name, curve, color in (("kinetic K", K, ORANGE), ("thermal U", U, BLUE), ("gravitational W", W, AQUA)):
    key = name.split()[1]
    ax.plot(t_ev, curve, color=color, lw=2, label=f"{name} (this code, N = {len(mass)})")
    gx, gy = zip(*gadget[key])
    ax.plot(gx, gy, "o", mfc="white", mec=color, mew=1.8, ms=8)
    px, py = zip(*ppm[key])
    ax.plot(px, py, "s", color=color, ms=6, alpha=0.6)
ax.plot(t_ev, K + U + W, color=INK, lw=1.2, ls="--", label="total K + U + W (conserved)")
ax.plot([], [], "o", mfc="white", mec=MUTED, mew=1.8, ms=8, label="GADGET-1 SPH, same N (published)")
ax.plot([], [], "s", color=MUTED, ms=6, alpha=0.6, label="high-resolution 1D reference (published)")
ax.annotate("bounce: infall stops in a shock;\nkinetic energy becomes heat", (1.12, 1.42), xytext=(1.6, 1.35), color=INK,
            fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel(r"time  [$\sqrt{R^3/GM}$]")
ax.set_ylabel(r"energy  [$GM^2/R$]")
ax.set_xlim(0, 3)
ax.set_title("Evrard collapse: an adiabatic gas sphere falls in, shocks, and bounces", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
style(ax)
save(fig, PLOTS, "evrard_energies.png")

# ---------------------------------------------------------------- 2. Jeans test: which clouds collapse?
M, R = 1.0, 0.1 * PC
T_FF, EPS = free_fall_time(M, R), 0.01 * R
RHO0 = M / (4 / 3 * np.pi * R**3)


def isothermal(alpha, n, t_end=2.0, n_out=41):
    pos, vel, mass = lattice_sphere(n, M, R)
    cs = np.sqrt(2 * alpha * abs(potential_energy(pos, mass, EPS)) / (3 * M))
    res = sph_leapfrog(pos, vel, 0.0, mass, Gas("isothermal", cs=cs), EPS, np.linspace(0, t_end, n_out) * T_FF,
                       rho_stop=1e3 * RHO0)
    return res, pos, mass


def critical_alpha(n, lo=0.45, hi=0.9, steps=5):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if isothermal(mid, n, t_end=3.0, n_out=7)[0]["stopped"] else (lo, mid)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
alphas = (0.3, 0.45, 0.6, 0.9, 1.2)
for alpha, color in zip(alphas, (BLUE, AQUA, YELLOW, ORANGE, MUTED)):
    res, _, _ = isothermal(alpha, 1500)
    tau, peak = res["t"] / T_FF, res["rho"].max(axis=1) / RHO0
    end = "collapses" if res["stopped"] else "expands"
    left.semilogy(tau, peak, color=color, lw=2, label=f"U/|W| = {alpha}: {end}")
left.axhline(1e3, color=GRID, lw=1, ls="--")
left.text(0.02, 1.3e3, "runaway: where Rung 4 will place a sink", color=MUTED, fontsize=8)
left.set_xlim(0, 2.9)
left.set_ylim(0.1, 3e3)
left.set_xlabel(r"time  [$t / t_{\rm ff}$]")
left.set_ylabel(r"peak density  [$\rho / \rho_0$]")
left.set_title("Isothermal clouds: cold ones run away, warm ones expand", color=INK)
left.legend(frameon=False, fontsize=8, loc="center right")
style(left)

Ns, crit = [1500, 4000], []
for n in Ns:
    crit.append(critical_alpha(n))
    print(f"critical U/|W| at N = {n}: {crit[-1][0]:.3f} +- {crit[-1][1]:.3f}")
right.axhspan(0, 5 / np.pi**2, color=BLUE, alpha=0.12)
right.axhspan(1.0, 1.4, color=ORANGE, alpha=0.12)
right.text(1100, 0.35, "collapse guaranteed\n(U/|W| < 5/π², Truelove+ 1998)", color=INK, fontsize=8)
right.text(1100, 1.15, "less than one Jeans mass\n(U/|W| > 1, Tohline 1982)", color=INK, fontsize=8)
right.errorbar(Ns, [c for c, _ in crit], yerr=[e for _, e in crit], fmt="o", color=INK, ms=7, capsize=3)
for n, (c, _) in zip(Ns, crit):
    right.text(n, c + 0.05, f"measured\n{c:.2f}", color=INK, fontsize=8, ha="center")
right.set_xscale("log")
right.set_xticks(Ns, [str(n) for n in Ns])
right.minorticks_off()
right.set_xlim(900, 7000)
right.set_ylim(0.2, 1.4)
right.set_xlabel("number of gas particles N")
right.set_ylabel("critical U/|W|")
right.set_title("The threshold sits between the two bounds", color=INK)
style(right)
save(fig, PLOTS, "jeans_test.png")

# ---------------------------------------------------------------- 3. The core free-falls until the rarefaction arrives
res, pos0, mass_j = isothermal(0.3, 4000, t_end=1.0, n_out=41)
core = np.linalg.norm(pos0, axis=1) < 0.3 * R
tau = res["t"] / T_FF
core_rho = np.array([np.median(r[core]) for r in res["rho"]]) / RHO0
fine = np.linspace(0, 0.97, 300)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.semilogy(fine, cycloid_radius(fine * T_FF, T_FF) ** -3, color=MUTED, lw=1.5, ls="--", label="pressure-free collapse (cycloid)")
ax.semilogy(tau, core_rho, "o-", color=BLUE, lw=2, ms=4, label="gas core (inner 30% of radius), U/|W| = 0.3")
ax.set_xlabel(r"time  [$t / t_{\rm ff}$]")
ax.set_ylabel(r"core density  [$\rho / \rho_0$]")
ax.set_title("Inside a uniform cloud pressure has no gradient: the core free-falls", color=INK)
ax.text(0.03, 0.62, "Pressure only acts at the edge, as a rarefaction\nwave moving inward at the sound speed. The core\n"
        "follows free fall until that wave reaches it.", transform=ax.transAxes, fontsize=9, color=INK)
ax.legend(frameon=False, loc="upper left", fontsize=8)
style(ax)
save(fig, PLOTS, "core_free_fall.png")

# ---------------------------------------------------------------- 4. Animation: the Evrard shock, colored by temperature
fig = plt.figure(figsize=(6.4, 8))
grid = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
sky, en = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
u_all = np.concatenate(ev["u"])
vmin, vmax = np.log10(np.percentile(u_all, 1)), np.log10(np.percentile(u_all, 99.5))
slab = np.abs(ev["pos"][0][:, 2]) < 0.15
dots = sky.scatter(ev["pos"][0][slab, 0], ev["pos"][0][slab, 1], c=np.log10(ev["u"][0][slab]), cmap=SEQUENTIAL,
                   vmin=vmin, vmax=vmax, s=6)
fig.colorbar(dots, ax=sky, shrink=0.8, label="log10 temperature (internal energy u)")
sky.set_xlim(-1.3, 1.3)
sky.set_ylim(-1.3, 1.3)
sky.set_aspect("equal")
sky.set_xlabel("x / R")
sky.set_ylabel("y / R")
sky.set_title("Slice through the cloud (|z| < 0.15 R)", color=INK, fontsize=10)
clock = sky.text(0.03, 0.95, "", transform=sky.transAxes, color=INK, fontsize=10)
lines = [en.plot([], [], color=c, lw=2, label=l)[0] for c, l in ((ORANGE, "kinetic K"), (BLUE, "thermal U"), (AQUA, "gravitational W"))]
now = en.axvline(0, color=MUTED, lw=0.8)
en.set_xlim(0, 3)
en.set_ylim(-2.4, 1.6)
en.set_xlabel(r"time  [$\sqrt{R^3/GM}$]")
en.set_ylabel(r"energy  [$GM^2/R$]")
en.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
for ax in (sky, en):
    style(ax)
fig.tight_layout()


def draw(frame):
    x, u = ev["pos"][frame], ev["u"][frame]
    s = np.abs(x[:, 2]) < 0.15
    dots.set_offsets(x[s, :2])
    dots.set_array(np.log10(u[s]))
    for line, curve in zip(lines, (K, U, W)):
        line.set_data(t_ev[:frame + 1], curve[:frame + 1])
    now.set_xdata([t_ev[frame]])
    clock.set_text(f"t = {t_ev[frame]:.2f}")
    return [dots, now, clock] + lines


anim = FuncAnimation(fig, draw, frames=len(t_ev), interval=60, blit=True)
anim.save(PLOTS / "evrard_shock.gif", writer=PillowWriter(fps=16), dpi=80)
print(f"saved {PLOTS / 'evrard_shock.gif'}")
