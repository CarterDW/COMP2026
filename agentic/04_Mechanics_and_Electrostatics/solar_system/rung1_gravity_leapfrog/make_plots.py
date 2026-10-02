"""Rung 1 figures: softened force law, leapfrog vs. forward Euler, convergence, figure-eight.

Run from anywhere:  python path/to/make_plots.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # scripts only save figures; no display under WSL
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.gravity import accelerations
from nbody.integrators import leapfrog
from nbody.diagnostics import total_energy
from nbody.orbits import kepler_period, two_body_at_pericenter, figure_eight, FIGURE_EIGHT_PERIOD
from nbody.units import G

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
PLOTS = Path(__file__).parent / "plots"

MASS = np.array([1.0, 1e-3])
A, E = 1.0, 0.5
PERIOD = kepler_period(A, MASS[1], MASS[0])
POS0, VEL0 = two_body_at_pericenter(MASS[0], MASS[1], A, E)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    print(f"saved {PLOTS / name}")


def forward_euler(pos, vel, mass, eps, dt, n_steps):
    """The non-symplectic foil: x and v both updated from the old state.

    Returns positions (n_steps + 1, N, 3) and total energy (n_steps + 1,).
    """
    traj, energy = [pos], [total_energy(pos, vel, mass, eps)]
    for _ in range(n_steps):
        pos, vel = pos + dt * vel, vel + dt * accelerations(pos, mass, eps)
        traj.append(pos)
        energy.append(total_energy(pos, vel, mass, eps))
    return np.array(traj), np.array(energy)


def energy_series(X, V, eps=0.0):
    E = np.array([total_energy(x, v, MASS, eps) for x, v in zip(X, V)])
    return np.abs(E / E[0] - 1)


# 1. Force law: Plummer softening caps the force inside r ~ eps.
eps = 0.1
r = np.linspace(1e-3, 5, 500) * eps
a_soft = np.array([abs(accelerations(np.array([[0, 0, 0], [x, 0, 0]]), np.array([1.0, 0]), eps)[1, 0]) for x in r])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r / eps, (eps / r)**2, color=MUTED, lw=1.5, ls="--", label=r"Newton $1/r^2$")
ax.plot(r / eps, a_soft * eps**2 / G, color=BLUE, lw=2, label=r"Plummer $r/(r^2+\epsilon^2)^{3/2}$")
ax.axvline(1 / np.sqrt(2), color=GRID, lw=1)
ax.annotate(r"peak at $r = \epsilon/\sqrt{2}$", (1 / np.sqrt(2), 0.385), xytext=(1.8, 0.55), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(0, 1.5)
ax.set_xlabel(r"separation $r/\epsilon$")
ax.set_ylabel(r"force $\times\ \epsilon^2 / G m$")
ax.set_title("Softening gives particles a size: the force stays finite", color=INK)
ax.legend(frameon=False)
style(ax)
save(fig, "force_law.png")

# 2. Why leapfrog: same orbit, same timestep, two integrators. A static figure and an animation.
n_per, n_orb = 1000, 20
dt = PERIOD / n_per
t, X, V = leapfrog(POS0, VEL0, MASS, 0.0, dt, n_per * n_orb)
dE_leap = 100 * energy_series(X, V)
X_eu, E_eu = forward_euler(POS0, VEL0, MASS, 0.0, dt, n_per * n_orb)
dE_euler = 100 * np.abs(E_eu / E_eu[0] - 1)
rel_leap, rel_euler = X[:, 1] - X[:, 0], X_eu[:, 1] - X_eu[:, 0]   # planet position relative to the star


def comparison_axes():
    """Empty, labelled layout: Euler orbit, leapfrog orbit, and energy error vs time underneath."""
    fig = plt.figure(figsize=(10, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    eu_ax, lf_ax, e_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    for ax in (eu_ax, lf_ax):
        ax.plot(0, 0, "*", color=INK, ms=12)
        ax.set_xlim(-5.6, 1.4)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        style(ax)
    eu_ax.set_title("Forward Euler: the orbit spirals outward", color=INK)
    lf_ax.set_title("Leapfrog: the same ellipse, 20 times over", color=INK)
    e_ax.set_yscale("log")
    e_ax.set_ylim(1e-6, 300)
    e_ax.set_xlim(0, n_orb)
    e_ax.set_xlabel("time [in units of the starting orbital period]")
    e_ax.set_ylabel("energy error  |E(t) − E(0)| / |E(0)|   [%]")
    e_ax.set_title("Energy should be constant. How far does each integrator wander?", color=INK)
    style(e_ax)
    fig.suptitle("Same physics, same timestep (1000 steps per orbit), e = 0.5", color=MUTED, fontsize=10, y=0.995)
    return fig, eu_ax, lf_ax, e_ax


fig, eu_ax, lf_ax, e_ax = comparison_axes()
eu_ax.plot(rel_euler[:, 0], rel_euler[:, 1], color=ORANGE, lw=0.6)
lf_ax.plot(rel_leap[:, 0], rel_leap[:, 1], color=BLUE, lw=0.6)
eu_ax.annotate("each lap is bigger:\nthe planet gains energy", (-3.2, 1.6), xytext=(-5.4, 2.05), color=INK, fontsize=9)
e_ax.plot(t / PERIOD, dE_euler + 1e-15, color=ORANGE, lw=1.5, label="forward Euler")
e_ax.plot(t / PERIOD, dE_leap + 1e-15, color=BLUE, lw=1.5, label="leapfrog")
e_ax.annotate("Euler: a jump at every close pass to the star,\nand the jumps never undo themselves (70% by the end)",
              (2.6, 22), xytext=(4, 0.4), color=INK, fontsize=9,
              bbox=dict(fc="white", ec="none", pad=2), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
e_ax.annotate("leapfrog: wobbles within 0.01% each orbit, then returns.\nThe error is bounded, so it never builds up.",
              (12.5, 1.1e-2), xytext=(11.2, 3e-5), color=INK, fontsize=9,
              bbox=dict(fc="white", ec="none", pad=2), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
e_ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=2)
save(fig, "leapfrog_vs_euler.png")

# Animation: both planets move together while their energy errors are drawn in underneath.
stride = 50                                     # 20 frames per starting orbit, 400 frames in all
fig, eu_ax, lf_ax, e_ax = comparison_axes()
eu_path, = eu_ax.plot([], [], color=ORANGE, lw=0.6)
lf_path, = lf_ax.plot([], [], color=BLUE, lw=0.6)
eu_dot, = eu_ax.plot([], [], "o", color=ORANGE, ms=8)
lf_dot, = lf_ax.plot([], [], "o", color=BLUE, ms=8)
eu_err, = e_ax.plot([], [], color=ORANGE, lw=1.5, label="forward Euler")
lf_err, = e_ax.plot([], [], color=BLUE, lw=1.5, label="leapfrog")
now = e_ax.axvline(0, color=MUTED, lw=0.8)
clock = e_ax.text(0.01, 0.04, "", transform=e_ax.transAxes, color=INK, fontsize=9)
e_ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=2)
fig.tight_layout()


thin = slice(None, None, 10)   # draw lines from every 10th step: smooth on screen, 10x less to redraw per frame
t_a, eu_a, lf_a, dEe_a, dEl_a = t[thin], rel_euler[thin], rel_leap[thin], dE_euler[thin], dE_leap[thin]


def draw_comparison(frame):
    i = frame * stride // 10
    eu_path.set_data(eu_a[:i + 1, 0], eu_a[:i + 1, 1])
    lf_path.set_data(lf_a[:i + 1, 0], lf_a[:i + 1, 1])
    eu_dot.set_data([eu_a[i, 0]], [eu_a[i, 1]])
    lf_dot.set_data([lf_a[i, 0]], [lf_a[i, 1]])
    eu_err.set_data(t_a[:i + 1] / PERIOD, dEe_a[:i + 1] + 1e-15)
    lf_err.set_data(t_a[:i + 1] / PERIOD, dEl_a[:i + 1] + 1e-15)
    now.set_xdata([t_a[i] / PERIOD])
    clock.set_text(f"t = {t_a[i] / PERIOD:5.2f}    Euler error {dEe_a[i]:5.1f}%    leapfrog error {dEl_a[i]:.4f}%")
    return eu_path, lf_path, eu_dot, lf_dot, eu_err, lf_err, now, clock


anim = FuncAnimation(fig, draw_comparison, frames=len(t) // stride + 1, interval=40, blit=True)
anim.save(PLOTS / "leapfrog_vs_euler.gif", writer=PillowWriter(fps=25), dpi=72)
print(f"saved {PLOTS / 'leapfrog_vs_euler.gif'}")

# 3. Convergence: halve the timestep, how much smaller is the error? One orbit per run.
steps = 250 * 2**np.arange(9)
err_leap, err_euler = [], []
for n in steps:
    t, X, V = leapfrog(POS0, VEL0, MASS, 0.0, PERIOD / n, n)
    err_leap.append(100 * energy_series(X, V).max())
    _, E_eu = forward_euler(POS0, VEL0, MASS, 0.0, PERIOD / n, n)
    err_euler.append(100 * np.abs(E_eu / E_eu[0] - 1).max())
err_leap, err_euler = np.array(err_leap), np.array(err_euler)

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.loglog(steps, err_euler, "s-", color=ORANGE, lw=2, ms=7, label="forward Euler")
ax.loglog(steps, err_leap, "o-", color=BLUE, lw=2, ms=7, label="leapfrog")
for err, color, shift in ((err_euler, ORANGE, 1.9), (err_leap, BLUE, 0.4)):
    for k in range(len(steps) - 1):
        ax.text(np.sqrt(steps[k] * steps[k + 1]), np.sqrt(err[k] * err[k + 1]) * shift,
                f"÷{err[k] / err[k + 1]:.1f}", color=color, fontsize=8, ha="center", va="center")
ax.set_xticks(steps, [f"{n:,}" for n in steps], fontsize=8)
ax.minorticks_off()
ax.set_ylim(1e-7, 1e2)
ax.set_xlabel("steps per orbit  (each point halves the timestep)")
ax.set_ylabel("worst energy error during one orbit  [%]")
ax.set_title("Halve the timestep: how much smaller does the error get?", color=INK)
k1000 = list(steps).index(1000)
ax.text(0.02, 0.03,
        "Leapfrog: ÷4 every time, i.e. second order (error ∝ dt²).\n"
        "Euler: first order (error ∝ dt), but it only settles to ÷2 once its error\n"
        "is below ~1%; before that the error is too big to be a small correction.\n"
        f"At 1,000 steps/orbit, leapfrog is {err_euler[k1000] / err_leap[k1000]:,.0f}× more accurate for the same work.",
        transform=ax.transAxes, fontsize=9, color=INK, va="bottom")
ax.legend(frameon=False, loc="upper right")
style(ax)
save(fig, "convergence.png")

# 4. Figure-eight three-body choreography, animated.
pos, vel, mass = figure_eight()
n_steps, n_frames, trail = 2400, 120, 25
t, X, V = leapfrog(pos, vel, mass, 0.0, FIGURE_EIGHT_PERIOD / n_steps, n_steps, n_steps // n_frames)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(X[:, 0, 0], X[:, 0, 1], color=GRID, lw=1.2)   # all three bodies trace this same curve
colors = (BLUE, ORANGE, AQUA)
trails = [ax.plot([], [], color=c, lw=2.5, alpha=0.5)[0] for c in colors]
dots = [ax.plot([], [], "o", color=c, ms=11, label=f"body {k + 1}")[0] for k, c in enumerate(colors)]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.45, 0.45)
ax.set_aspect("equal")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_title(f"Three equal masses on one figure-eight (closes to {np.abs(X[-1] - pos).max():.0e})", color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
style(ax)
fig.tight_layout()


def draw(frame):
    for k in range(3):
        past = X[max(0, frame - trail):frame + 1, k]
        trails[k].set_data(past[:, 0], past[:, 1])
        dots[k].set_data([X[frame, k, 0]], [X[frame, k, 1]])
    return trails + dots


anim = FuncAnimation(fig, draw, frames=n_frames, interval=40, blit=True)
anim.save(PLOTS / "figure_eight.gif", writer=PillowWriter(fps=25), dpi=90)
print(f"saved {PLOTS / 'figure_eight.gif'}")
