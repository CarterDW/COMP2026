"""Rung 6: compare the 8 terrestrial runs with each other and with Mercury, Venus, Earth and Mars.

Run from anywhere:  python path/to/analyze_terrestrial.py      -> printed statistics and plots/terrestrial_systems.png
Statistics (Chambers 2001): number of planets (> 0.05 Mearth), largest mass, the angular momentum deficit
AMD = sum m sqrt(a) (1 - sqrt(1 - e^2) cos i) / sum m sqrt(a), and the radial mass concentration
S_c = max(sum m / sum m [log10(a / a_x)]^2) over a_x.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nbody.plotstyle import BLUE, ORANGE, INK, MUTED, YELLOW, style, save
from nbody.orbits import orbital_elements
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
SOLAR = dict(name="Solar System", a=np.array([0.387, 0.723, 1.0, 1.524]), e=np.array([0.206, 0.007, 0.017, 0.093]),
             inc=np.radians([7.0, 3.39, 0.0, 1.85]), m=np.array([0.055, 0.815, 1.0, 0.107]))


def statistics(a, e, inc, m):
    amd = np.sum(m * np.sqrt(a) * (1 - np.sqrt(1 - e**2) * np.cos(inc))) / np.sum(m * np.sqrt(a))
    grid = np.geomspace(0.3, 5, 2000)
    s_c = max(m.sum() / np.sum(m * np.log10(a / ax) ** 2) for ax in grid)
    return dict(n=len(m), m_max=m.max(), amd=amd, s_c=s_c)


systems = []
for s in range(1, 9):
    d = np.load(HERE / "data" / f"terrestrial_seed{s}.npz")
    n_emb, mu = int(d["n_emb"]), G * float(d["M_star"])
    live = np.flatnonzero(d["alive"][-1][:n_emb])
    a, e, inc = orbital_elements(d["x"][-1][live], d["u"][-1][live], mu)
    m = d["m"][-1][live] / M_EARTH
    keep = (m > 0.05) & (a < 5)
    systems.append(dict(name=f"seed {s}", a=a[keep], e=e[keep], inc=inc[keep], m=m[keep]))

print(f"{'system':13s} {'planets':>7s} {'largest':>8s} {'AMD':>7s} {'S_c':>6s}")
for sys_ in [SOLAR] + systems:
    st = statistics(sys_["a"], sys_["e"], sys_["inc"], sys_["m"])
    print(f"{sys_['name']:13s} {st['n']:7d} {st['m_max']:8.2f} {st['amd']:7.4f} {st['s_c']:6.1f}")

fig, ax = plt.subplots(figsize=(9, 6))
for row, sys_ in enumerate([SOLAR] + systems):
    y = len(systems) - row
    color = YELLOW if row == 0 else BLUE
    for a, e, m in zip(sys_["a"], sys_["e"], sys_["m"]):
        ax.plot([a * (1 - e), a * (1 + e)], [y, y], color=color, lw=1, alpha=0.6)      # pericenter to apocenter
        ax.scatter(a, y, s=120 * m ** (2 / 3), color=color, edgecolor=INK, linewidth=0.5, zorder=3)
    ax.text(0.27, y, sys_["name"], va="center", ha="right", fontsize=9, color=INK)
ax.axvline(2.7, color=MUTED, lw=1, ls=":")
ax.text(2.72, len(systems) + 0.6, "snow line", color=MUTED, fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.28, 5)
ax.set_xticks([0.3, 0.5, 1, 2, 3, 5], ["0.3", "0.5", "1", "2", "3", "5"])
ax.set_yticks([])
ax.set_xlabel("semi-major axis  [AU]   (lines: pericenter to apocenter; dot area ~ mass^(2/3))")
ax.set_title("Terrestrial zone after 20 Myr: 8 runs vs. our solar system", color=INK)
style(ax)
ax.spines["left"].set_visible(False)
(HERE / "plots").mkdir(exist_ok=True)
save(fig, HERE / "plots", "terrestrial_systems.png")
