"""Rung 5: a rotating 3D view of one formation run, from data/<tag>.npz.

Run from anywhere:  python path/to/make_3d.py TAG      -> plots/<tag>/system_3d.gif
Body types come from mass (composition is not recorded in these runs): planetesimal tracers, icy embryos (every
embryo started beyond the snow line), and gas giants (above 50 Mearth, which only gas accretion reaches here).
Each giant keeps one name and color (A, B, ... in the order they formed). Each planet's orbit is drawn in full,
from its position and velocity; the gas disk fades as it disperses.

Snapshots are 10 kyr apart (hundreds of orbits), so on its own each body would teleport between frames. Instead,
every second snapshot is shown for SUB frames during which all bodies move along their current orbits (exact
Kepler motion, STEP_YR per frame); the movie then skips ahead to the next snapshot.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, YELLOW, INK, MUTED
from nbody.kepler import kepler_drift
from nbody.gas_effects import disk_state
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
TAG = sys.argv[1]
PLOTS = HERE / "plots" / TAG
PLOTS.mkdir(parents=True, exist_ok=True)
d = np.load(HERE / "data" / f"{TAG}.npz")
t, x, u, mass, alive, big = d["t"], d["x"], d["u"], d["m"], d["alive"], d["big"]
# A gas giant: mostly gas, if the run recorded composition; otherwise above 50 Mearth (only gas accretion gets there).
is_giant = (d["comp"][:, :, 3] > 0.5 * mass) if "comp" in d.files else (mass >= 50 * M_EARTH)
mu = G * float(d["M_star"])
disk = d["disk_params"]
Z_STRETCH = 10.0                        # inclinations are ~0.5 deg: stretch z so the vertical structure shows
LIM = 45.0                              # the outer giant ends near 40 AU
GIANT = 50 * M_EARTH
SNAP_EVERY, SUB, STEP_YR = 2, 8, 3.0    # show every 2nd snapshot, 8 frames each, bodies advance 3 yr per frame
EMBRYO_COLOR, PLANETESIMAL_COLOR = BLUE, "#9a9a9a"
GIANT_COLORS = [ORANGE, "#e87ba4", YELLOW, "#1baf7a"]

# Name the giants once, in the order they crossed GIANT, so labels and colors follow bodies, not ranks.
became = {i: np.argmax(is_giant[:, i]) for i in np.flatnonzero(is_giant.any(axis=0))}
GIANT_NAMES = {i: (chr(ord("A") + n), GIANT_COLORS[n % len(GIANT_COLORS)])
               for n, i in enumerate(sorted(became, key=became.get))}


def orbit_curve(xi, ui, n=120):
    """Points around the osculating orbit through (xi, ui): one full period sampled with the exact Kepler solver."""
    r = np.linalg.norm(xi)
    a = -mu / (2 * (0.5 * ui @ ui - mu / r))
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    pts = np.empty((n + 1, 3))
    for k in range(n + 1):
        pts[k] = kepler_drift(xi, ui, mu, period * k / n)[0]
    return pts


sigma0 = disk_state(10.0, t[0], disk)[0]
snaps = np.arange(0, len(t), SNAP_EVERY)
n_frames = len(snaps) * SUB
orbit_cache = {}


def orbits_of(k):
    """Orbit curves of all live big bodies at snapshot k (computed once per snapshot)."""
    if k not in orbit_cache:
        orbit_cache[k] = {i: orbit_curve(x[k, i], u[k, i]) for i in np.flatnonzero(alive[k] & big[k])}
    return orbit_cache[k]


def positions(k, tau):
    """Every live body moved along its own orbit for tau years from snapshot k."""
    p = x[k].copy()
    for i in np.flatnonzero(alive[k]):
        p[i] = kepler_drift(x[k, i], u[k, i], mu, tau)[0]
    return p
fig = plt.figure(figsize=(8, 7.2), facecolor="#05060a")
ax = fig.add_subplot(projection="3d", facecolor="#05060a")
fig.subplots_adjust(0, 0, 1, 1)


def draw(frame):
    k, sub = snaps[frame // SUB], frame % SUB
    pos = positions(k, sub * STEP_YR)
    ax.cla()
    ax.set_facecolor("#05060a")
    ax.set_axis_off()
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-LIM / 2, LIM / 2)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.2)
    ax.view_init(elev=28, azim=-60 + 120 * frame / n_frames)    # a slow 120-degree turn over the whole run

    # Gas disk: faint concentric rings whose opacity follows the remaining gas.
    gas_left = disk_state(10.0, t[k], disk)[0] / sigma0
    phi = np.linspace(0, 2 * np.pi, 120)
    for rr in np.linspace(3, 32, 15):
        ax.plot(rr * np.cos(phi), rr * np.sin(phi), 0, color="#3c6fb0", lw=9, alpha=0.12 * gas_left)

    ax.scatter([0], [0], [0], s=260, color=YELLOW, edgecolor="#fff6c8", linewidth=1.5)
    live = alive[k]
    pl = live & ~big[k]
    ax.scatter(pos[pl, 0], pos[pl, 1], Z_STRETCH * pos[pl, 2], s=3, color=PLANETESIMAL_COLOR, alpha=0.7, depthshade=False)
    for i, curve in orbits_of(k).items():
        giant = is_giant[k, i]
        name, color = GIANT_NAMES[i] if giant else ("", EMBRYO_COLOR)
        ax.plot(curve[:, 0], curve[:, 1], Z_STRETCH * curve[:, 2], color=color, lw=1.0 if giant else 0.7, alpha=0.7)
        size = 12 * (mass[k, i] / M_EARTH) ** (2 / 3)
        ax.scatter(*pos[i, :2], Z_STRETCH * pos[i, 2], s=size, color=color, edgecolor="white", linewidth=0.4,
                   depthshade=False)
        if giant:
            ax.text(*pos[i, :2], Z_STRETCH * pos[i, 2] + 3.5, f"giant {name}: {mass[k, i] / M_EARTH / 317.8:.1f} MJ",
                    color=color, fontsize=9, ha="center", fontweight="bold")

    n_giants = int(np.sum(live & big[k] & is_giant[k]))
    n_embryos = int(np.sum(live & big[k] & ~is_giant[k]))
    fig.texts.clear()
    fig.text(0.03, 0.95, f"disk age {t[k] / 1e6:.2f} Myr", color="white", fontsize=12)
    fig.text(0.03, 0.91, f"gas left: {100 * gas_left:.0f}%   giants {n_giants}   embryos {n_embryos}   "
             f"planetesimals {int(pl.sum())}", color="#cccccc", fontsize=9)
    fig.text(0.03, 0.03, "● gas giants (> 50 Mearth), named A, B in order of formation", color=ORANGE, fontsize=9)
    fig.text(0.03, 0.06, "● icy embryo", color=EMBRYO_COLOR, fontsize=9)
    fig.text(0.17, 0.06, "● planetesimal", color=PLANETESIMAL_COLOR, fontsize=9)
    fig.text(0.97, 0.03, f"box: {2 * LIM:.0f} AU across;  vertical stretched x{Z_STRETCH:.0f}", color="#888888",
             fontsize=8, ha="right")
    fig.text(0.03, 0.875, f"bodies follow their orbits for {SUB * STEP_YR:.0f} yr, then the movie skips "
             f"{SNAP_EVERY * 10:,} kyr to the next snapshot", color="#888888", fontsize=8, va="top")
    return []


anim = FuncAnimation(fig, draw, frames=n_frames, interval=70)
anim.save(PLOTS / "system_3d.gif", writer=PillowWriter(fps=14), dpi=72, savefig_kwargs={"facecolor": "#05060a"})
print(f"saved {PLOTS / 'system_3d.gif'}")
