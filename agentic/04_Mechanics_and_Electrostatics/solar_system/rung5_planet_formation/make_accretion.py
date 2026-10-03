"""Rung 5: how each planet grew, and from what. Reads data/<tag>.npz (or its checkpoint while the run is going).

Run from anywhere:  python path/to/make_accretion.py TAG [--checkpoint]     -> plots/<tag>/accretion.gif
Top left: each embryo's orbit (semi-major axis) through time, so migration shows. Top right: its mass. Bottom: what
each surviving planet is made of: seed solids, solids from collisions, pebbles, gas.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, GRID, style
from nbody.orbits import orbital_elements
from nbody.gas_effects import M_CRIT
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
TAG = sys.argv[1]
PLOTS = HERE / "plots" / TAG
PLOTS.mkdir(parents=True, exist_ok=True)
if "--checkpoint" in sys.argv:
    c = np.load(HERE / "data" / f"{TAG}_checkpoint.npz")
    d = {k[5:]: c[k] for k in c.files if k.startswith("snap_")}
    M_star = None
else:
    d = dict(np.load(HERE / "data" / f"{TAG}.npz"))
t, x, u, mass, alive, big, comp = d["t"], d["x"], d["u"], d["m"], d["alive"], d["big"], d["comp"]
M_star = 0.816 if "M_star" not in d else float(d["M_star"])
mu = G * M_star
a_hist = np.array([orbital_elements(x[k], u[k], mu)[0] for k in range(len(t))])

SOURCES = [("seed solids", "#8a8a8a"), ("collisions", AQUA), ("pebbles", BLUE), ("gas", ORANGE)]
GIANT = 50 * M_EARTH
embryos = np.flatnonzero(big[0])
order0 = embryos[np.argsort(a_hist[0, embryos])]
became = {i: np.argmax(mass[:, i] >= GIANT) for i in embryos if (mass[:, i] >= GIANT).any()}
giant_colors = [ORANGE, MAGENTA, YELLOW]
NAMES, COLORS = {}, {}
for n, i in enumerate(sorted(became, key=became.get)):
    NAMES[i], COLORS[i] = f"giant {chr(ord('A') + n)}", giant_colors[n % 3]
for n, i in enumerate(order0):
    NAMES.setdefault(i, f"e{n + 1}")
    COLORS.setdefault(i, BLUE)

fig = plt.figure(figsize=(11.5, 8))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.35, wspace=0.22)
orb, grow, bars = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
t_myr = t / 1e6


def draw(k):
    for ax in (orb, grow, bars):
        ax.cla()
    for i in embryos:
        life = alive[:k + 1, i]
        if not life.any():
            continue
        giant = i in became and k >= became[i]
        lw, col = (2.2, COLORS[i]) if i in became else (1.0, BLUE if alive[k, i] else GRID)
        orb.semilogy(t_myr[:k + 1][life], a_hist[:k + 1, i][life], color=col, lw=lw)
        grow.semilogy(t_myr[:k + 1][life], mass[:k + 1, i][life] / M_EARTH, color=col, lw=lw)
        if alive[k, i]:
            size = 3 + 1.5 * (mass[k, i] / M_EARTH) ** (1 / 3)        # marker ~ planet size (mass^1/3)
            orb.plot(t_myr[k], a_hist[k, i], "o", color=col, ms=size, mec="white", mew=0.5)
            if giant:
                orb.text(t_myr[k] + 0.03, a_hist[k, i], NAMES[i], color=col, fontsize=9, fontweight="bold", va="center")
                grow.text(t_myr[k] + 0.03, mass[k, i] / M_EARTH, NAMES[i], color=col, fontsize=9, fontweight="bold", va="center")
    for ax in (orb, grow):
        ax.set_xlim(t_myr[0], t_myr[-1] + 0.25)
        ax.set_xlabel("disk age  [Myr]")
        style(ax)
    orb.set_ylim(1.5, 60)
    orb.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    orb.set_ylabel("orbit: semi-major axis  [AU]")
    orb.set_title("Where the planets are (migration drifts them inward)", color=INK, fontsize=10)
    for name, a_p in (("Jupiter", 5.2), ("Saturn", 9.6)):
        orb.axhline(a_p, color=YELLOW, lw=0.8, ls=":")
        orb.text(t_myr[0] + 0.02, a_p * 1.05, name, color=MUTED, fontsize=7)
    grow.set_ylim(0.3, 3000)
    grow.axhline(M_CRIT / M_EARTH, color=ORANGE, lw=0.8, ls=":")
    grow.text(t_myr[0] + 0.02, 11.5, "10 Mearth: runaway gas accretion can start", color=ORANGE, fontsize=7)
    for name, m_p in (("Jupiter", 317.8), ("Saturn", 95.2)):
        grow.axhline(m_p, color=YELLOW, lw=0.8, ls=":")
        grow.text(t_myr[0] + 0.02, m_p * 1.12, name, color=MUTED, fontsize=7)
    grow.set_ylabel("mass  [Mearth]")
    grow.set_title("How heavy they are", color=INK, fontsize=10)

    live = [i for i in embryos if alive[k, i]]
    live.sort(key=lambda i: a_hist[k, i])
    if not live:                                       # everything has merged away or left the simulated zone
        bars.text(0.5, 0.5, "no planets left in the simulated zone (4-30 AU)", ha="center", va="center",
                  transform=bars.transAxes, color=MUTED, fontsize=11)
        bars.set_axis_off()
        fig.suptitle(f"How the planets grew — disk age {t_myr[k]:.2f} Myr", color=INK, fontsize=12)
        return []
    y = np.arange(len(live))
    left = np.zeros(len(live))
    fractions = np.array([comp[k, i] / comp[k, i].sum() for i in live])
    for s, (label, col) in enumerate(SOURCES):
        bars.barh(y, fractions[:, s], left=left, color=col, height=0.7, label=label)
        left += fractions[:, s]
    for yi, i in zip(y, live):
        bars.text(1.01, yi, f"{mass[k, i] / M_EARTH:,.1f} Mearth at {a_hist[k, i]:.1f} AU", va="center", fontsize=8, color=INK)
    bars.set_yticks(y, [NAMES[i] for i in live], fontsize=8)
    for tick, i in zip(bars.get_yticklabels(), live):
        tick.set_color(COLORS[i] if i in became else INK)
        tick.set_fontweight("bold" if i in became else "normal")
    bars.set_xlim(0, 1.32)
    bars.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    bars.set_title("What each planet is made of (innermost at the bottom)", color=INK, fontsize=10, loc="left")
    bars.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))
    bars.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"How the planets grew — disk age {t_myr[k]:.2f} Myr", color=INK, fontsize=12)
    return []


anim = FuncAnimation(fig, draw, frames=len(t), interval=200)
anim.save(PLOTS / "accretion.gif", writer=PillowWriter(fps=5), dpi=72)
draw(len(t) - 1)
fig.savefig(PLOTS / "accretion_final.png", dpi=130)
print(f"saved {PLOTS / 'accretion.gif'} and accretion_final.png ({len(t)} frames)")
