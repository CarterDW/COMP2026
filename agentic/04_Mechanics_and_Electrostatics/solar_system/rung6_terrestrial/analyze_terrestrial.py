"""Rung 6: compare the terrestrial runs with each other and with Mercury, Venus, Earth and Mars.

Run from anywhere:  python path/to/analyze_terrestrial.py [--planetesimals N] [--seeds 1 2 ...]
    -> printed statistics, plots/terrestrial_systems[_plN].png and plots/terrestrial_energy[_plN].png
--planetesimals N picks the runs made with that option (0, the default as in run_terrestrial.py: embryos only);
--seeds the seeds to compare (default 1-8).

Planets are the surviving embryos above 0.05 Mearth inside 5 AU (planetesimals are not counted). Statistics
(Chambers 2001): number of planets, largest mass, the angular momentum deficit
AMD = sum m sqrt(a) (1 - sqrt(1 - e^2) cos i) / sum m sqrt(a), and the radial mass concentration
S_c = max(sum m / sum m [log10(a / a_x)]^2) over a_x. The energy budget (E_now + merger losses + removed) / E0 - 1
is recomputed from the saved final state. Runs that were stopped and resumed on newer code ("stopped after
checkpoint" in their log) are marked at each switch.
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nbody.hybrid as hy
from nbody.plotstyle import BLUE, INK, MUTED, YELLOW, ORANGE, style, save
from nbody.orbits import orbital_elements
from nbody.units import G, M_EARTH

HERE = Path(__file__).parent
SOLAR = dict(name="Solar System", a=np.array([0.387, 0.723, 1.0, 1.524]), e=np.array([0.206, 0.007, 0.017, 0.093]),
             inc=np.radians([7.0, 3.39, 0.0, 1.85]), m=np.array([0.055, 0.815, 1.0, 0.107]))
M_PLANET, A_MAX = 0.05, 5.0                                       # Mearth, AU


def option_list(name, default):
    if name not in sys.argv:
        return default
    k = sys.argv.index(name) + 1
    values = []
    while k < len(sys.argv) and not sys.argv[k].startswith("--"):
        values.append(sys.argv[k])
        k += 1
    return values


N_PL = int(option_list("--planetesimals", ["0"])[0])
SEEDS = [int(s) for s in option_list("--seeds", [str(s) for s in range(1, 9)])]
SUFFIX = f"_pl{N_PL}" if N_PL else ""


def statistics(a, e, inc, m):
    amd = np.sum(m * np.sqrt(a) * (1 - np.sqrt(1 - e**2) * np.cos(inc))) / np.sum(m * np.sqrt(a))
    grid = np.geomspace(0.3, 5, 2000)
    s_c = max(m.sum() / np.sum(m * np.log10(a / ax) ** 2) for ax in grid)
    return dict(n=len(m), m_max=m.max(), amd=amd, s_c=s_c)


def read_log(seed):
    """Budget history (t [Myr], dE/E) and the times [Myr] at which the run resumed on newer code."""
    text = (HERE / "data" / f"terrestrial_seed{seed}{SUFFIX}.log").read_text()
    hist = np.array([(float(t), float(b)) for t, b in re.findall(r"^t =\s+([\d.]+) Myr.*budget dE/E (\S+)$", text, re.M)])
    switches = [float(t) for t in re.findall(r"stopped after checkpoint.*\nresumed \S+ at ([\d.]+) Myr", text)]
    return hist, switches


systems = []
for s in SEEDS:
    d = np.load(HERE / "data" / f"terrestrial_seed{s}{SUFFIX}.npz")
    n_emb, M_star = int(d["n_emb"]), float(d["M_star"])
    mu = G * M_star
    x, u, m_all, alive = d["x"][-1], d["u"][-1], d["m"][-1], d["alive"][-1]
    big = np.zeros(len(m_all), bool)
    big[:n_emb] = True
    big[-1] = True                                                 # the giant is always last
    Q, v = hy.from_heliocentric(x, u, m_all, M_star)
    E_now = hy.total_energy(Q, v, m_all.copy(), alive.copy(), big, mu, M_star)
    budget = (E_now + d["log"][:, 5].sum() + float(d["E_removed"])) / float(d["E0"]) - 1
    live = np.flatnonzero(alive[:n_emb])
    a, e, inc = orbital_elements(x[live], u[live], mu)
    m = m_all[live] / M_EARTH
    keep = (m > M_PLANET) & (a < A_MAX)
    n_pl = N_PL                                                    # the embryo-only runs predate the n_pl entry
    assert "n_pl" not in d or int(d["n_pl"]) == N_PL
    pl_left = alive[n_emb:n_emb + n_pl]
    hist, switches = read_log(s)
    systems.append(dict(name=f"seed {s}", a=a[keep], e=e[keep], inc=inc[keep], m=m[keep], budget=budget,
                        hist=hist, switches=switches, pl_left=int(pl_left.sum()),
                        m_pl_left=m_all[n_emb:n_emb + n_pl][pl_left].sum() / M_EARTH, t_end=float(d["t"][-1]) / 1e6))

print(f"{'system':13s} {'planets':>7s} {'largest':>8s} {'at [AU]':>8s} {'AMD':>7s} {'S_c':>6s} "
      f"{'pl. left':>8s} {'[Me]':>6s} {'dE/E':>9s}  code switches [Myr]")
for sys_ in [SOLAR] + systems:
    st = statistics(sys_["a"], sys_["e"], sys_["inc"], sys_["m"])
    line = (f"{sys_['name']:13s} {st['n']:7d} {st['m_max']:8.2f} {sys_['a'][np.argmax(sys_['m'])]:8.2f} "
            f"{st['amd']:7.4f} {st['s_c']:6.1f}")
    if "budget" in sys_:
        line += (f" {sys_['pl_left']:8d} {sys_['m_pl_left']:6.2f} {sys_['budget']:9.1e}  "
                 f"{', '.join(f'{t:.1f}' for t in sys_['switches']) or '-'}")
    print(line)

# Figure 1: the systems, each planet from pericenter to apocenter
t_end = max(sys_["t_end"] for sys_ in systems)
fig, ax = plt.subplots(figsize=(9, 1.2 + 0.6 * (len(systems) + 1)))
for row, sys_ in enumerate([SOLAR] + systems):
    y = len(systems) - row
    color = YELLOW if row == 0 else BLUE
    for a, e, m in zip(sys_["a"], sys_["e"], sys_["m"]):
        ax.plot([a * (1 - e), a * (1 + e)], [y, y], color=color, lw=1, alpha=0.6)
        ax.scatter(a, y, s=120 * m ** (2 / 3), color=color, edgecolor=INK, linewidth=0.5, zorder=3)
    ax.text(0.27, y, sys_["name"], va="center", ha="right", fontsize=9, color=INK)
ax.axvline(2.7, color=MUTED, lw=1, ls=":")
ax.text(2.72, len(systems) + 0.6, "snow line", color=MUTED, fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.28, 5)
ax.set_ylim(-0.7, len(systems) + 0.9)
ax.set_xticks([0.3, 0.5, 1, 2, 3, 5], ["0.3", "0.5", "1", "2", "3", "5"])
ax.set_yticks([])
ax.set_xlabel("semi-major axis  [AU]   (lines: pericenter to apocenter; dot area ~ mass^(2/3))")
swarm = f", {N_PL} planetesimals" if N_PL else ", embryos only"
ax.set_title(f"Terrestrial zone after {t_end:.0f} Myr{swarm}: {len(systems)} runs vs. our solar system", color=INK)
style(ax)
ax.spines["left"].set_visible(False)
(HERE / "plots").mkdir(exist_ok=True)
save(fig, HERE / "plots", f"terrestrial_systems{SUFFIX}.png")

# Figure 2: energy budget of each run through time, with the code switches marked
cols = 4
rows = int(np.ceil(len(systems) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), sharex=True, sharey=True, squeeze=False)
for ax, sys_ in zip(axes.flat, systems):
    t, b = sys_["hist"].T
    ax.semilogy(t, np.abs(b), color=BLUE, lw=1.5)
    for k, ts in enumerate(sys_["switches"]):
        ax.axvline(ts, color=ORANGE, lw=1, ls=":")
    ax.set_title(f"{sys_['name']}  (final {sys_['budget']:.1e})", color=INK, fontsize=9)
    style(ax)
for k, ax in enumerate(axes.flat):
    if k >= len(systems):
        ax.set_visible(False)
    elif k + cols >= len(systems):                                 # lowest visible panel in its column
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("time  [Myr]")
for ax in axes[:, 0]:
    ax.set_ylabel("|energy budget dE/E|")
fig.suptitle("Energy budget (E_now + merger losses + removed) / E0 - 1;   dotted: resumed on newer code",
             color=INK, fontsize=10)
save(fig, HERE / "plots", f"terrestrial_energy{SUFFIX}.png")
