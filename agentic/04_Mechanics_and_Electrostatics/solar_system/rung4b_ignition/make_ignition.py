"""Rung 4b: the Rung 4 protostar's life from first accretion to hydrogen ignition, and the stellar ignition GIF.

Run from anywhere:  python path/to/make_ignition.py
Reads ../rung4_protostar_disk/handoff.npz (the sink's accretion history). Writes plots/ and star_history.npz.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from nbody.plotstyle import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, GRID, style, save, blackbody_rgb
from nbody import protostar as ps
from nbody.units import R_SUN, L_SUN

HERE = Path(__file__).parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)
hand = np.load(HERE.parent / "rung4_protostar_disk" / "handoff.npz")

# ---------------------------------------------------------------- 1. accretion history: simulation + fitted tail
t_sim, m_sim = hand["accretion_t_yr"], hand["accretion_mass_Msun"]
t0 = t_sim[m_sim > 0][0]                                        # the sink forms
t_a = np.linspace(t0, t_sim[-1], 400)
M_a = np.interp(t_a, t_sim, m_sim)
mdot_a = np.gradient(M_a, t_a)
late = t_a > t_a[-1] - 10_000
slope, icpt = np.polyfit(t_a[late], np.log(mdot_a[late]), 1)
tau, mdot_end = -1 / slope, np.exp(icpt + slope * t_a[-1])      # simulation stops while accretion is still decaying
t_tail = t_a[-1] + np.geomspace(1, 6 * tau, 200)                # continue Mdot = Mdot_end exp(-(t - t_end) / tau)
mdot_tail = mdot_end * np.exp(-(t_tail - t_a[-1]) / tau)
M_tail = M_a[-1] + mdot_end * tau * (1 - np.exp(-(t_tail - t_a[-1]) / tau))
t_acc, M_acc, mdot_acc = np.concatenate([t_a, t_tail]), np.concatenate([M_a, M_tail]), np.concatenate([mdot_a, mdot_tail])
mdot_acc = np.maximum(mdot_acc, 1e-12)
acc = ps.accretion_phase(t_acc, M_acc, mdot_acc)
M_final = M_acc[-1]
print(f"sink forms at {t0:.0f} yr; simulation ends at {t_a[-1]:.0f} yr with {M_a[-1]:.3f} Msun; "
      f"tail (tau = {tau:.0f} yr) adds {M_final - M_a[-1]:.3f} -> final mass {M_final:.3f} Msun")

# ---------------------------------------------------------------- 2. pre-main-sequence contraction to the ZAMS
pms = ps.pre_main_sequence(M_final, acc["R"][-1], t_acc[-1], 1.5e8)
h = {k: np.concatenate([acc[k], pms[k][1:]]) for k in acc}
h["mdot"] = np.concatenate([mdot_acc, np.zeros(len(pms["t"]) - 1)])
h["Teff"] = ps.effective_temperature(h["L_phot"], h["R"])
L_D_used = np.minimum(h["L_D"], h["L_phot"])                     # deuterium powers (part of) the surface
L_H_used = np.minimum(h["L_H"], h["L_phot"])
h["L_contract"] = np.maximum(h["L_phot"] - L_D_used - L_H_used, 0)
h["L_total"] = h["L_acc"] + h["L_phot"]
t = h["t"]
i_D = np.argmax(h["T_c"] >= ps.T_D * (1 - 1e-9))
i_zams = np.argmax(h["L_H"] / h["L_phot"] > 0.99)
print(f"deuterium thermostat from {t[i_D]:.0f} yr (M = {h['M'][i_D]:.2f} Msun); ZAMS at {t[i_zams] / 1e6:.1f} Myr: "
      f"R = {h['R'][i_zams] / R_SUN:.2f} Rsun, L = {h['L_phot'][i_zams] / L_SUN:.2f} Lsun, Teff = {h['Teff'][i_zams]:.0f} K")
np.savez(HERE / "star_history.npz", **h)

PHASES = [(t[0], "accreting protostar, hidden in its envelope"), (t_acc[-1], "pre-main sequence: contracting (Hayashi track)"),
          (t[np.argmax((h["L_phot"] <= ps.zams_luminosity(M_final) * 1.000001) & (t > t_acc[-1]))], "pre-main sequence: radiative core (Henyey track)"),
          (t[np.argmax(h["L_H"] / h["L_phot"] > 0.5)], "hydrogen ignites"), (t[i_zams], "main-sequence star")]


def phase_at(time):
    return [name for start, name in PHASES if time >= start][-1]


# ---------------------------------------------------------------- 3. static figures
fig, (top, bot) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
for key, color, label in (("L_acc", ORANGE, "accretion (gas crashing onto the star)"),
                          ("L_contract", BLUE, "gravitational contraction"),
                          ("L_D", MAGENTA, "deuterium burning"), ("L_H", YELLOW, "hydrogen fusion")):
    curve = L_D_used if key == "L_D" else L_H_used if key == "L_H" else h[key]
    top.loglog(t, np.maximum(curve, 1e-30) / L_SUN, color=color, lw=2, label=label)
top.loglog(t, h["L_total"] / L_SUN, color=INK, lw=1, ls="--", label="total luminosity")
top.set_ylim(1e-2, 1e3)
top.set_ylabel("luminosity  [Lsun]")
top.set_title(f"What makes a {M_final:.2f} Msun star shine, from first accretion to the main sequence", color=INK)
top.legend(frameon=False, fontsize=8, loc="upper right")
bot.loglog(t, h["T_c"], color=INK, lw=2)
for T, label, color in ((ps.T_D, "deuterium ignites (1.5e6 K)", MAGENTA), (ps.T_H, "hydrogen ignites (1e7 K)", YELLOW)):
    bot.axhline(T, color=color, lw=1.2, ls="--")
    bot.text(t[0] * 1.2, T * 1.12, label, color=INK, fontsize=8)
bot.set_ylim(1e5, 2e7)
bot.set_xlabel("time since the cloud began to collapse  [yr]")
bot.set_ylabel("core temperature  [K]")
for ax in (top, bot):
    style(ax)
save(fig, PLOTS, "luminosity_history.png")

fig, ax = plt.subplots(figsize=(7, 5.5))
pre = t >= t_acc[-1]
ax.semilogy(h["Teff"][pre], h["L_phot"][pre] / L_SUN, color=BLUE, lw=2, label=f"our {M_final:.2f} Msun star")
for age in (1e6, 3e6, 1e7, 3e7):
    k = np.argmin(abs(t - age))
    ax.plot(h["Teff"][k], h["L_phot"][k] / L_SUN, "o", color=BLUE, ms=6)
    ax.annotate(f"{age / 1e6:g} Myr", (h["Teff"][k], h["L_phot"][k] / L_SUN), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
ax.plot([4400, 5600], [(3.1**2) * (4400 / 5772) ** 4, 0.68], "s", mfc="white", mec=MUTED, ms=8, label="BHAC15 1 Msun: 0.5 Myr and ZAMS")
ax.plot(5772, 1.0, "*", color=YELLOW, mec=INK, ms=14, label="the Sun today")
ax.invert_xaxis()
ax.set_xlabel("surface temperature  [K]")
ax.set_ylabel("luminosity  [Lsun]")
ax.set_title("Hertzsprung-Russell track: down the Hayashi line, then left to the main sequence", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8)
style(ax)
save(fig, PLOTS, "hr_track.png")

# ---------------------------------------------------------------- 4. the stellar ignition animation
# Power sources as fractions of the total light; the cutaway shows the core heating and fusion taking over.
L_tot = h["L_total"]
frac = np.array([h["L_acc"], h["L_contract"], L_D_used, L_H_used]) / L_tot
SOURCES = [("accretion", ORANGE), ("contraction", BLUE), ("deuterium", MAGENTA), ("hydrogen fusion", YELLOW)]
CORE_CMAP = plt.get_cmap("inferno")
t_hayashi_end = PHASES[2][0]


def caption(k):
    time, f_H = t[k], frac[3, k]
    accreting = time < t_acc[-1]
    if accreting and h["T_c"][k] < ps.T_D * (1 - 1e-6):
        return ("1. A star is born inside a dusty cocoon",
                "Gas from the cloud rains onto the newborn star. The crash of\ninfalling gas makes most of the light; dust hides the star itself.")
    if accreting:
        return ("2. Deuterium ignites at 1.5 million K",
                "The core is hot enough to fuse deuterium, a rare heavy hydrogen.\nIt acts as a thermostat, holding the core temperature steady.")
    if f_H < 0.05 and time < t_hayashi_end:
        return ("3. No fuel left: gravity squeezes the star",
                "Accretion has stopped and the deuterium is spent. The star shrinks\nunder its own weight, and the squeezing heats its core.")
    if f_H < 0.05:
        return ("4. Shrinking more slowly, core still heating",
                "The core passes 6 million K. Contraction alone still powers the star,\nbut the core is getting close to the hydrogen-fusion threshold.")
    if f_H < 0.99:
        return ("5. Hydrogen ignites near 10 million K",
                f"Hydrogen fuses into helium in the core. Fusion now supplies\n{100 * f_H:.0f}% of the light, and the star's contraction is slowing.")
    return ("6. A main-sequence star",
            "Fusion supplies all the light, so the star stops shrinking.\nIt will now burn hydrogen steadily for billions of years.")


# Frame pacing: more frames where the story happens (deuterium and hydrogen ignition), then hold the end.
frames = np.concatenate([np.geomspace(t[0] * 1.02, 1.2e5, 60), np.geomspace(1.2e5, 8e6, 35),
                         np.geomspace(8e6, 7e7, 70), np.full(20, 7e7)])
fig = plt.figure(figsize=(11.5, 6.4))
grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.15], height_ratios=[1, 1, 0.8], hspace=0.65)
sky = fig.add_subplot(grid[:2, 0])
thermo, power, timeline = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])
text_ax = fig.add_subplot(grid[2, 0])
text_ax.axis("off")

sky.set_facecolor("#05060a")
sky.set_xlim(-5.5, 5.5)
sky.set_ylim(-5.5, 5.5)
sky.set_aspect("equal")
sky.set_xticks([])
sky.set_yticks([])
glow = Circle((0, 0), 1, lw=0, alpha=0.15)
star = Circle((0, 0), 1, lw=0)
core = Circle((0, 0), 1, lw=0)
fusion = Circle((0, 0), 1, lw=0, alpha=0)
haze = Circle((0, 0), 5.4, color="#5a3a22", lw=0, alpha=0)
for patch in (glow, star, core, fusion, haze):
    sky.add_patch(patch)
sky.text(0, 5.0, "cutaway: surface, with the core inside\n(size on a square-root scale)", color="#aaaaaa", fontsize=8,
         ha="center", va="top")


def display_radius(R):
    """Drawn radius for a star of radius R [Rsun]: compressed (sqrt) so 4 Rsun and 0.6 Rsun both read well."""
    return 2.3 * np.sqrt(R)
stats = sky.text(-5.2, -5.2, "", color="white", fontsize=8, va="bottom", family="monospace")
stats.set_zorder(10)
head = text_ax.text(0.0, 0.95, "", fontsize=12, fontweight="bold", color=INK, va="top", transform=text_ax.transAxes)
body = text_ax.text(0.0, 0.55, "", fontsize=9, color=INK, va="top", transform=text_ax.transAxes)

# Core thermometer (log scale).
thermo.set_xscale("log")
thermo.set_xlim(1e5, 2e7)
thermo.set_ylim(0, 1)
thermo.set_yticks([])
thermo.barh(0.5, 2e7, left=1e5, height=0.5, color=GRID)
fill = thermo.barh(0.5, 1, left=1e5, height=0.5, color=ORANGE)[0]
for T, label, color in ((ps.T_D, "deuterium\nignites", MAGENTA), (ps.T_H, "hydrogen\nignites", "#b37a00")):
    thermo.axvline(T, color=color, lw=2)
    thermo.text(T, 0.82, label, color=color, fontsize=8, ha="center", va="bottom", fontweight="bold")
thermo.set_title("core temperature", color=INK, fontsize=10, loc="left")
thermo.set_xlabel("K", fontsize=8)
thermo.spines[["top", "right", "left"]].set_visible(False)
t_label = thermo.text(1.2e5, 0.5, "", va="center", fontsize=9, color="white", fontweight="bold")

# What powers the star: stacked fractions.
power.set_xlim(0, 1)
power.set_ylim(0, 1)
power.set_yticks([])
power.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"], fontsize=8)
bars = [power.barh(0.55, 0, left=0, height=0.45, color=c)[0] for _, c in SOURCES]
for i, (name, c) in enumerate(SOURCES):
    power.text(0.25 * i, 0.05, "■ " + name, color=c if name != "hydrogen fusion" else "#b37a00", fontsize=8, fontweight="bold")
power.set_title("what powers the star?", color=INK, fontsize=10, loc="left")
power.spines[["top", "right", "left"]].set_visible(False)

# Timeline of milestones.
timeline.set_xscale("log")
timeline.set_xlim(t[0], 1e8)
timeline.set_ylim(0, 1)
timeline.set_yticks([])
timeline.axhline(0.35, color=MUTED, lw=1)
for j, (when, label) in enumerate(((t0, "star forms"), (t[i_D], "deuterium ignites"), (t_acc[-1], "accretion ends"),
                                    (t[np.argmax(frac[3] > 0.5)], "H fusion > 50%"), (t[i_zams], "main sequence"))):
    timeline.plot(when, 0.35, "|", color=INK, ms=12)
    timeline.text(when, 0.55 if j % 2 == 0 else 0.85, label, fontsize=7, ha="center", color=INK)
marker, = timeline.plot([t[0]], [0.35], "o", color=ORANGE, ms=9)
timeline.set_xlabel("age [yr]", fontsize=8)
timeline.spines[["top", "right", "left"]].set_visible(False)


def draw(frame):
    time = frames[frame]
    k = min(np.searchsorted(t, time), len(t) - 1)
    R = h["R"][k] / R_SUN
    Rd = display_radius(R)
    surface = blackbody_rgb(h["Teff"][k])
    star.set_radius(Rd)
    star.set_color(surface)
    glow.set_radius(Rd * (1.15 + 0.25 * np.log10(max(L_tot[k] / L_SUN, 1e-2) * 100)))
    glow.set_color(surface)
    core.set_radius(0.4 * Rd)
    heat = (np.log10(h["T_c"][k]) - 5) / (np.log10(2e7) - 5)                # 0 at 1e5 K, 1 at 2e7 K
    core.set_color(CORE_CMAP(0.15 + 0.8 * np.clip(heat, 0, 1)))
    f_nuc_D, f_nuc_H = frac[2, k], frac[3, k]
    fusion.set_radius(0.25 * Rd)
    fusion.set_color("#fff6c8" if f_nuc_H >= f_nuc_D else "#ff9ad0")
    fusion.set_alpha(min(1.0, 1.2 * max(f_nuc_D, f_nuc_H)))
    haze.set_alpha(0.93 * np.sqrt(h["mdot"][k] / mdot_peak))
    fill.set_width(h["T_c"][k] - 1e5)
    fill.set_color(CORE_CMAP(0.15 + 0.8 * np.clip(heat, 0, 1)))
    t_label.set_text(f"{h['T_c'][k] / 1e6:.2f} million K")
    left = 0.0
    for bar, f in zip(bars, frac[:, k]):
        bar.set_x(left)
        bar.set_width(f)
        left += f
    marker.set_xdata([time])
    title, text = caption(k)
    head.set_text(title)
    body.set_text(text)
    stats.set_text(f"age {time:9.3g} yr   mass {h['M'][k]:.2f} Msun\nradius {R:.2f} Rsun   L {L_tot[k] / L_SUN:.2f} Lsun")
    return [glow, star, core, fusion, haze, fill, t_label, marker, head, body, stats] + bars


mdot_peak = h["mdot"].max()
anim = FuncAnimation(fig, draw, frames=len(frames), interval=90, blit=False)
anim.save(PLOTS / "stellar_ignition.gif", writer=PillowWriter(fps=11), dpi=80)
print(f"saved {PLOTS / 'stellar_ignition.gif'}")
