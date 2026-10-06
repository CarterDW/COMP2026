"""Rung 6b figures: is the pebble stream ever dense enough to form planetesimals, and where do they form?

Run from anywhere after the sweep (run_inner_disk.py for v_frag_dry in {1, 10} m/s, zeta in {1e-4, 1e-3, 1e-2},
fit in {ly21, lim24}):  python path/to/make_plots.py   -> plots/inner_disk.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nbody.pebble_disk as pd
from nbody.gas_effects import disk_state
from nbody.viscous_disk import SNOW_LINE, Z_ROCK, Z_ICE
from nbody.units import M_EARTH
from nbody.plotstyle import BLUE, ORANGE, INK, MUTED, SEQUENTIAL, style, save

HERE = Path(__file__).parent
load = lambda vf, zeta, fit: np.load(HERE / "data" / f"inner_disk_vf{vf:g}_zeta{zeta:g}_{fit}.npz")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: pebble-to-gas ratio vs the streaming-instability threshold (sticky silicates: St = 0.05 everywhere)
d = load(10, 1e-3, "lim24")
c, disk = d["centers"], d["disk"]
times = [4e3, 1e4, 3e4, 1e5, 3e5, 1e6]
for k, tt in enumerate(times):
    j = int(np.argmin(np.abs(d["t"] - tt)))
    t = d["t"][j]
    z = (d["rock"][j] + d["ice"][j]) / np.array([disk_state(r, t, disk)[0] for r in c])
    ax1.loglog(c, z, color=SEQUENTIAL(0.25 + 0.75 * k / (len(times) - 1)), lw=1.6, label=f"{t / 1e6:.2g} Myr")
zc = [pd.z_crit(pd.ST_GROWTH, disk[8], pd.local_state(r, 1e5, disk, 0.0, r > SNOW_LINE, 10.0, 10.0)[6], pd.FIT_LIM24)
      for r in c]
ax1.loglog(c, zc, color=ORANGE, lw=2, ls="--", label="threshold (Lim+24)")
ax1.axvline(SNOW_LINE, color=MUTED, lw=1, ls=":")
ax1.text(SNOW_LINE * 1.03, 2e-4, "snow line", color=MUTED, fontsize=8)
ax1.set_xlabel("radius  [AU]")
ax1.set_ylabel("pebble-to-gas ratio  Z")
ax1.set_title("Sticky silicates (10 m/s): the pebble stream stays below the threshold inside the snow line",
              color=INK, fontsize=9)
ax1.legend(frameon=False, fontsize=7, ncol=2)
ax1.set_ylim(1e-4, 0.1)

# Right: where planetesimals form, all 12 runs, vs the solids Rung 6 assumed
edges = d["edges"]
dlnr = np.log(edges[1:] / edges[:-1])
for vf, color in [(1, BLUE), (10, ORANGE)]:
    for zeta in (1e-4, 1e-3, 1e-2):
        for fit, ls in [("lim24", "-"), ("ly21", "--")]:
            run = load(vf, zeta, fit)
            dm = (run["plts_rock"][-1] + run["plts_ice"][-1]) * run["areas"] / M_EARTH / dlnr
            label = f"v_frag {vf} m/s" if (zeta, fit) == (1e-3, "lim24") else None
            ax2.loglog(c, np.maximum(dm, 1e-6), color=color, lw=1, ls=ls, alpha=0.4 + 0.2 * np.log10(zeta / 1e-4), label=label)
rung6 = np.array([disk_state(r, 1e6, disk)[0] for r in c]) * np.where(c < SNOW_LINE, Z_ROCK, Z_ICE)
ax2.loglog(c, rung6 * 2 * np.pi * c**2 / M_EARTH, color=INK, lw=2, label="solids Rung 6 assumed (Z x gas)")
ax2.set_ylim(1e-3, 300)
ax2.text(0.3, 2e-3, "no planetesimals inside 2.7 AU in any run", color=INK, fontsize=9)
ax2.axvline(SNOW_LINE, color=MUTED, lw=1, ls=":")
ax2.set_xlabel("radius  [AU]")
ax2.set_ylabel("planetesimal mass per ln r  [Mearth]")
ax2.set_title("Where planetesimals form: 12 runs (zeta 1e-4 to 1e-2; solid Lim+24, dashed LY21)",
              color=INK, fontsize=9)
ax2.legend(frameon=False, fontsize=8)
for a in (ax1, ax2):
    style(a)
(HERE / "plots").mkdir(exist_ok=True)
save(fig, HERE / "plots", "inner_disk.png")
