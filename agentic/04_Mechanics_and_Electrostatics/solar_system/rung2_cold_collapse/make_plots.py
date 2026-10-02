"""Rung 2 figures: a cold uniform sphere collapsing under its own gravity.

Run from anywhere:  python path/to/make_plots.py   (~2 minutes)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, AQUA, INK, MUTED, GRID, SEQUENTIAL, style, save
from nbody.clouds import uniform_sphere, free_fall_time, cycloid_radius
from nbody.integrators import leapfrog
from nbody.gravity import potentials
from nbody.diagnostics import total_energy, virial_ratio, lagrangian_radii
from nbody.units import PC

PLOTS = Path(__file__).parent / "plots"
M, R = 1.0, 0.1 * PC
T_FF = free_fall_time(M, R)
EPS = 0.01 * R

# One fiducial run: N = 2000, dt = t_ff / 1000, out to 3 t_ff, snapshot every 0.01 t_ff.
pos0, vel0, mass = uniform_sphere(2000, M, R, np.random.default_rng(0))
t, X, V = leapfrog(pos0, vel0, mass, EPS, T_FF / 1000, 3000, 10)
tau = t / T_FF
FRACTIONS = [0.1, 0.5, 0.9]
radii = np.array([lagrangian_radii(x, mass, FRACTIONS) for x in X]) / R
bound = np.array([0.5 * np.sum(v**2, axis=1) + potentials(x, mass, EPS) < 0 for x, v in zip(X, V)])
print(f"fiducial run done; t_ff = {T_FF:.3e} yr")

# 1. Lagrangian radii against the cycloid.
fig, ax = plt.subplots(figsize=(8, 5))
early = np.linspace(0, 1, 400)
for k, (q, color) in enumerate(zip(FRACTIONS, (AQUA, BLUE, ORANGE))):
    r0 = radii[0, k]
    ax.plot(tau, radii[:, k], color=color, lw=2, label=f"radius holding {int(100 * q)}% of the mass")
    ax.plot(early, r0 * cycloid_radius(early * T_FF, T_FF), color=color, lw=1, ls="--")
ax.plot([], [], color=MUTED, lw=1, ls="--", label="analytic free fall (cycloid)")
ax.axvline(1, color=GRID, lw=1)
ax.annotate("every shell reaches the\ncenter at the same time, t_ff", (0.99, 0.04), xytext=(0.12, 0.07), color=INK, fontsize=9, bbox=dict(fc="white", ec="none", pad=1),
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
i_run = np.searchsorted(tau, 1.4)
ax.annotate("the 90% radius runs away: the bounce\nflings over 20% of the mass out, unbound", (tau[i_run], radii[i_run, 2]),
            xytext=(1.75, 1.15), color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.6)
ax.set_xlabel(r"time  [$t / t_{\rm ff}$],   $t_{\rm ff}$ = " + f"{T_FF / 1e3:.0f} kyr")
ax.set_ylabel("radius  [r / R]")
ax.set_title("Cold collapse: the shells follow free fall until they all meet", color=INK)
ax.legend(frameon=False, loc="upper right", fontsize=8)
style(ax)
save(fig, PLOTS, "lagrangian_radii.png")

# 2. Virial balance: everything vs only the particles still bound.
ratio_all = np.array([virial_ratio(x, v, mass, EPS) for x, v in zip(X, V)])
ratio_bound = np.array([virial_ratio(x[b], v[b], mass[b], EPS) for x, v, b in zip(X, V, bound)])
fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
top.plot(tau, ratio_all, color=MUTED, lw=1.5, label="all particles")
top.plot(tau, ratio_bound, color=BLUE, lw=2, label="only particles still bound")
top.axhline(1, color=INK, lw=0.8, ls=":")
top.text(2.95, 1.05, "virial balance, 2K = |W|", color=INK, fontsize=9, ha="right")
top.set_ylim(0, 4)
top.set_ylabel("2K / |W|")
top.set_title("After the bounce, what stays bound settles into virial balance", color=INK)
top.legend(frameon=False, loc="upper left")
bot.plot(tau, 100 * (1 - bound.mean(axis=1)), color=ORANGE, lw=2)
bot.set_ylabel("mass unbound [%]")
bot.set_xlabel(r"time  [$t / t_{\rm ff}$]")
for ax in (top, bot):
    style(ax)
save(fig, PLOTS, "virial.png")

# 3. Does a smaller timestep fix energy conservation? Only if the particles have a size.
small_pos, small_vel, small_mass = uniform_sphere(500, M, R, np.random.default_rng(0))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, eps, title in ((axes[0], 0.0, "point particles (ε = 0)"), (axes[1], EPS, "softened (ε = 0.01 R)")):
    for n_ff, color in ((500, ORANGE), (2000, BLUE)):
        ts, Xs, Vs = leapfrog(small_pos, small_vel, small_mass, eps, T_FF / n_ff, int(1.5 * n_ff), n_ff // 100)
        E = np.array([total_energy(x, v, small_mass, eps) for x, v in zip(Xs, Vs)])
        ax.semilogy(ts / T_FF, 100 * np.abs(E / E[0] - 1) + 1e-12, color=color, lw=1.8, label=f"dt = t_ff / {n_ff}")
    ax.set_title(title, color=INK)
    ax.set_xlabel(r"time  [$t / t_{\rm ff}$]")
    ax.set_ylim(1e-5, 1e6)
    style(ax)
axes[0].legend(frameon=False, loc="lower right")
axes[1].legend(frameon=False, loc="upper left")
axes[0].set_ylabel("energy error  [%]")
axes[0].text(0.3, 0.3, "4× smaller dt: still ~1000% wrong.\nRandom positions include a few very\nclose pairs; they whirl around each\n"
             "other faster than any dt can follow,\nlong before the bounce.",
             transform=axes[0].transAxes, fontsize=9, color=INK)
axes[1].text(0.05, 0.5, "4× smaller dt: 16× smaller error,\nas expected for leapfrog (dt²).",
             transform=axes[1].transAxes, fontsize=9, color=INK)
fig.suptitle("Does a smaller timestep fix energy conservation? Only if particles have a size.", color=INK, y=0.99)
save(fig, PLOTS, "softening_energy.png")

# 4. How small does the cloud get? Set by the graininess of N particles, not by the softening.
Ns = np.array([250, 500, 1000, 2000, 4000])
r_min, r_err = [], []
for n in Ns:
    vals = []
    for seed in range(3 if n <= 2000 else 1):
        p, v, m = uniform_sphere(n, M, R, np.random.default_rng(seed))
        ts, Xs, _ = leapfrog(p, v, m, 0.003 * R, T_FF / 1000, 1200, 5)
        r50 = np.array([lagrangian_radii(x, m, [0.5])[0] for x in Xs])
        vals.append(r50.min() / r50[0])
    r_min.append(np.mean(vals))
    r_err.append(np.std(vals))
    print(f"bounce radius, N = {n}: {r_min[-1]:.3f}")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(Ns, r_min, yerr=r_err, fmt="o", color=BLUE, ms=8, capsize=3, label="measured (mean of 3 clouds)")
ax.loglog(Ns, r_min[2] * (Ns / Ns[2]) ** (-1 / 3), color=MUTED, ls="--", lw=1, label=r"$\propto N^{-1/3}$")
ax.minorticks_off()
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.set_yticks([0.05, 0.07, 0.1, 0.15], ["0.05", "0.07", "0.10", "0.15"])
ax.set_xlabel("number of particles N")
ax.set_ylabel("smallest half-mass radius / initial")
ax.set_title("More particles, deeper collapse: graininess sets the bounce", color=INK)
ax.legend(frameon=False)
style(ax)
save(fig, PLOTS, "bounce_radius_vs_N.png")

# 5. Animation: the cloud in projection, colored by starting radius, with the shells' radii underneath.
r_start = np.linalg.norm(pos0, axis=1) / R
fig = plt.figure(figsize=(6.4, 8))
grid = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
sky, rad = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
dots = sky.scatter(X[0, :, 0] / R, X[0, :, 1] / R, c=r_start, cmap=SEQUENTIAL, s=3, vmin=0, vmax=1)
sky.set_xlim(-1.6, 1.6)
sky.set_ylim(-1.6, 1.6)
sky.set_aspect("equal")
sky.set_xlabel("x / R")
sky.set_ylabel("y / R")
fig.colorbar(dots, ax=sky, shrink=0.8, label="starting radius / R")
clock = sky.text(0.03, 0.95, "", transform=sky.transAxes, color=INK, fontsize=10)
lines = [rad.plot([], [], color=c, lw=2, label=f"{int(100 * q)}% of mass")[0]
         for q, c in zip(FRACTIONS, (AQUA, BLUE, ORANGE))]
rad.plot(early, cycloid_radius(early * T_FF, T_FF) * radii[0, 2], color=MUTED, lw=1, ls="--", label="free fall (90%)")
now = rad.axvline(0, color=MUTED, lw=0.8)
rad.set_xlim(0, 3)
rad.set_ylim(0, 1.6)
rad.set_xlabel(r"time  [$t / t_{\rm ff}$]")
rad.set_ylabel("radius / R")
rad.legend(frameon=False, fontsize=8, loc="upper right")
for ax in (sky, rad):
    style(ax)
fig.tight_layout()


def draw(frame):
    dots.set_offsets(X[frame, :, :2] / R)
    for k, line in enumerate(lines):
        line.set_data(tau[:frame + 1], radii[:frame + 1, k])
    now.set_xdata([tau[frame]])
    clock.set_text(f"t = {tau[frame]:.2f} t_ff  ({t[frame] / 1e3:.0f} kyr)")
    return [dots, now, clock] + lines


frames = list(range(0, 90, 2)) + list(range(90, 130)) + list(range(130, len(t), 3))   # slow down near t_ff
anim = FuncAnimation(fig, draw, frames=frames, interval=50, blit=True)
anim.save(PLOTS / "collapse.gif", writer=PillowWriter(fps=20), dpi=80)
print(f"saved {PLOTS / 'collapse.gif'}")

# 6. Keeping the cloud bound: five starting conditions, same N, softening and timestep.
from nbody.clouds import power_law_sphere, virial_velocities
from nbody.diagnostics import unbound_fraction
from nbody.plotstyle import YELLOW, MAGENTA

starts = [  # (label, alpha, Q0, color); ordered from most to least mass lost
    ("cold, uniform", 0.0, 0.0, ORANGE),
    ("cold, ρ ∝ 1/r", 1.0, 0.0, YELLOW),
    ("warm, Q₀ = 0.25", 0.0, 0.25, MAGENTA),
    ("cold, ρ ∝ 1/r²", 2.0, 0.0, AQUA),
    ("warm, Q₀ = 0.5", 0.0, 0.5, BLUE),
]
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
final = []
for label, alpha, Q0, color in starts:
    p, v, m = power_law_sphere(2000, M, R, alpha, np.random.default_rng(0))
    if Q0 > 0:
        v = virial_velocities(p, m, EPS, Q0, np.random.default_rng(1))
    ts, Xs, Vs = leapfrog(p, v, m, EPS, T_FF / 1000, 3000, 25)
    lost = 100 * np.array([unbound_fraction(x, u, m, EPS) for x, u in zip(Xs, Vs)])
    final.append(lost[-1])
    left.plot(ts / T_FF, lost, color=color, lw=2)
    print(f"{label}: {lost[-1]:.1f}% unbound")
label_y = np.array(final)
for k in range(len(label_y) - 2, -1, -1):          # starts are ordered high to low; keep labels >= 1.6 apart
    label_y[k] = max(label_y[k], label_y[k + 1] + 1.6)
for (label, *_), y_text in zip(starts, label_y):
    left.text(3.04, y_text, label, color=INK, fontsize=8, va="center")
left.set_xlim(0, 3)
left.set_ylim(-1, 28)
left.set_xlabel(r"time  [$t / t_{\rm ff}$ of the uniform cloud]")
left.set_ylabel("mass flung out (unbound)  [%]")
left.set_title("When does the mass escape?", color=INK)
style(left)

y = np.arange(len(starts))[::-1]
right.barh(y, final, color=[c for *_, c in starts], height=0.6)
for yi, value in zip(y, final):
    right.text(value + 0.5, yi, f"{value:.1f}%", va="center", color=INK, fontsize=9)
right.set_yticks(y, [label for label, *_ in starts], fontsize=9)
fit = 100 * (0.048 + 0.022 * np.log(2000))
right.axvline(fit, color=MUTED, lw=1, ls="--")
right.text(fit + 0.4, y[0] + 0.45, f"published fit, cold\nuniform N = 2000: {fit:.1f}%", color=MUTED, fontsize=8, va="bottom")
right.set_xlim(0, 30)
right.set_ylim(-0.6, len(starts) - 0.2)
right.set_xlabel("mass lost after 3 t_ff  [%]")
right.set_title("Fix it: start warm, or start centrally concentrated", color=INK)
right.grid(True, axis="x", color=GRID, lw=0.6)
right.spines[["top", "right"]].set_visible(False)
fig.suptitle("Cold uniform clouds lose ~20% of their mass at the bounce. Two ways to keep it bound.", color=INK, y=1.0)
save(fig, PLOTS, "keeping_the_cloud_bound.png")
