"""Rung 0: print the unit system and plot Kepler's third law for the real planets.

Run from anywhere:  python path/to/check_units.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # scripts only save figures; no display under WSL
import matplotlib.pyplot as plt
from nbody import units as u
from nbody import planets as p
from nbody.orbits import kepler_period

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
PLOTS = Path(__file__).parent / "plots"

print(f"G           = {u.G:.6f} AU^3 / (Msun yr^2)")
print(f"1 yr        = {u.YEAR_S / u.DAY_S:.7f} days")
print(f"1 km/s      = {u.KM_S:.6f} AU/yr")
print(f"v_circ(1AU) = {2 * np.pi / u.KM_S:.4f} km/s")
print(f"M_sun       = {u.MSUN_KG:.5e} kg = {1 / u.M_EARTH:.1f} M_earth")
print(f"1 pc        = {u.PC:.3f} AU\n")

P_meas = p.PERIOD_DAYS * u.DAY_S / u.YEAR_S
P_mass = kepler_period(p.A, p.MASS)
P_zero = kepler_period(p.A, 0.0)
print(f"{'planet':8s} {'P_meas [yr]':>12s} {'rel. resid':>11s} {'massless':>11s}")
for name, Pm, Pk, P0 in zip(p.NAMES, P_meas, P_mass, P_zero):
    print(f"{name:8s} {Pm:12.6f} {Pm / Pk - 1:11.2e} {Pm / P0 - 1:11.2e}")

fig, (top, bot) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
a_line = np.geomspace(0.3, 40, 200)
top.loglog(a_line, a_line**1.5, color=MUTED, lw=1, label=r"$P = a^{3/2}$")
top.loglog(p.A, P_meas, "o", color=BLUE, ms=7, label="measured")
for name, a, P in zip(p.NAMES, p.A, P_meas):
    top.annotate(name, (a, P), xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK)
top.set_ylabel("period [yr]")
top.legend(frameon=False)
top.set_title("Kepler's third law in code units (G = 4π²)", color=INK)

bot.semilogx(p.A, np.abs(P_meas / P_zero - 1), "s", color=ORANGE, ms=7, label="massless planet")
bot.semilogx(p.A, np.abs(P_meas / P_mass - 1), "o", color=BLUE, ms=7, label="with (M + m)")
bot.set_yscale("log")
bot.set_xlabel("semi-major axis [AU]")
bot.set_ylabel("|relative residual|")
bot.legend(frameon=False, fontsize=8)

for ax in (top, bot):
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "kepler_third_law.png", dpi=150)
print(f"\nsaved {PLOTS / 'kepler_third_law.png'}")
