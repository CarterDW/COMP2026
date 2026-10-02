"""Rung 4 figures and the handoff file, from the saved showcase run (run run_collapse.py first).

Run from anywhere:  python path/to/make_plots.py
Writes plots/ and handoff.npz (star mass, disk surface density, accretion history) for Rungs 4b and 5.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody.plotstyle import BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID, SEQUENTIAL, style, save
from nbody.disks import central_frame, disk_members, surface_density, centrifugal_radius, bound_fraction, toomre_q
from nbody.units import G, KM_S

HERE = Path(__file__).parent
PLOTS = HERE / "plots"
d = np.load(HERE / "data" / "collapse.npz", allow_pickle=True)
snaps, T_FF, R_ACC, EPS, CS = list(d["snaps"]), float(d["t_ff"]), float(d["r_acc"]), float(d["eps"]), float(d["cs"])
pos0, vel0 = d["init_pos"], d["init_vel"]
final = snaps[-1]
x, v, M_star = central_frame(final)
in_disk = disk_members(x, v, M_star)
R_disk = np.linalg.norm(x[in_disk, :2], axis=1)
m_disk = final["mass"][in_disk]
print(f"final: {len(final['sink_mass'])} sink(s) {np.round(final['sink_mass'], 3)}, disk {m_disk.sum():.3f} Msun "
      f"in {in_disk.sum()} particles, bound fraction {bound_fraction(final, EPS):.4f}")


def cs_of_R(R):
    """Sound speed of the heated gas: T = max(10 K, 280 K (R/AU)^-1/2), cs ~ sqrt(T)."""
    return CS * np.sqrt(np.maximum(1.0, 28.0 / np.sqrt(R)))


# ---------------------------------------------------------------- 1. Disk surface density and stability
edges = np.geomspace(R_ACC, 1.1 * R_disk.max(), 16)
Rc, sigma = surface_density(R_disk, m_disk, edges)
ok = sigma > 0
slope = np.polyfit(np.log(Rc[ok]), np.log(sigma[ok]), 1)[0]
sigma_cgs = sigma * 1.989e33 / 1.496e13**2                    # Msun/AU^2 -> g/cm^2
Q = toomre_q(Rc, sigma, M_star, cs_of_R)
fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
top.loglog(Rc[ok], sigma_cgs[ok], "o-", color=BLUE, lw=2, ms=6, label="our disk")
mmsn = 1700 * Rc ** -1.5                                       # Hayashi (1981) minimum-mass solar nebula, g/cm^2
top.loglog(Rc, mmsn, color=MUTED, lw=1.2, ls="--", label=r"minimum-mass solar nebula, $1700\,R^{-3/2}$ g/cm$^2$")
top.set_ylabel(r"surface density $\Sigma$  [g/cm$^2$]")
top.set_title(f"The disk: {m_disk.sum():.3f} Msun around a {M_star:.2f} Msun protostar", color=INK)
top.axvspan(Rc[0] * 0.8, 2 * R_ACC, color=GRID, alpha=0.6)
top.text(Rc[0] * 0.85, sigma_cgs[ok].max() * 1.3, "inner hole is numerical\n(accretion radius)", color=MUTED, fontsize=8)
top.legend(frameon=False, fontsize=8)
bot.loglog(Rc[ok], Q[ok], "o-", color=ORANGE, lw=2, ms=6)
bot.axhline(1, color=INK, lw=0.8, ls=":")
bot.text(Rc[ok][0], 1.15, "Q < 1: the disk breaks into fragments", color=INK, fontsize=8)
bot.set_xlabel("distance from the protostar  R  [AU]")
bot.set_ylabel("Toomre Q")
for ax in (top, bot):
    style(ax)
save(fig, PLOTS, "disk_profile.png")

# ---------------------------------------------------------------- 2. Each parcel lands where its angular momentum says
# Each parcel feels the star plus the disk mass inside its own orbit.
order = np.argsort(R_disk)
M_enclosed = np.empty(len(R_disk))
M_enclosed[order] = M_star + np.cumsum(m_disk[order])
j0 = pos0[final["ids"][in_disk], 0] * vel0[final["ids"][in_disk], 1] - pos0[final["ids"][in_disk], 1] * vel0[final["ids"][in_disk], 0]
r_pred = j0**2 / (G * M_enclosed)
ratio = np.median(R_disk / r_pred)
v_rel = v[in_disk]
j_now = x[in_disk, 0] * v_rel[:, 1] - x[in_disk, 1] * v_rel[:, 0]
gain = np.sum(m_disk * j_now) / np.sum(m_disk * j0)
print(f"centrifugal radius: median R / (j0^2 / G M_enc) = {ratio:.2f}; disk gas angular momentum now / initially = {gain:.2f}")
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.loglog(r_pred, R_disk, "o", color=BLUE, ms=3, alpha=0.4)
lim = [R_ACC * 0.5, 2 * max(R_disk.max(), r_pred.max())]
ax.loglog(lim, lim, color=INK, lw=1, ls="--", label="R = j² / GM  (angular momentum conserved)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"predicted from initial spin: $j_z^2 / G M$  [AU]")
ax.set_ylabel("actual distance in the disk  [AU]")
ax.set_title(f"Disk gas sits {ratio:.1f}x beyond j²/GM: it gained {100 * (gain - 1):.0f}% spin", color=INK, fontsize=11)
ax.text(0.97, 0.04, "Viscosity carries angular momentum outward:\nthe disk spreads while its inner edge is\nswallowed by the protostar.",
        transform=ax.transAxes, fontsize=8, color=INK, ha="right")
ax.legend(frameon=False, fontsize=8, loc="upper left")
style(ax)
save(fig, PLOTS, "centrifugal_radius.png")

# ---------------------------------------------------------------- 3. Accretion history (input for Rung 4b)
t_h, m_h = d["hist_t"], d["hist_sink_mass"]
grid_t = np.linspace(t_h[m_h > 0][0], t_h[-1], 200)
m_grid = np.interp(grid_t, t_h, m_h)
mdot = np.gradient(m_grid, grid_t)
shu = 0.975 * CS**3 / G                                        # Shu (1977) rate for a singular isothermal sphere
lph = 46.9 * CS**3 / G                                         # Larson-Penston-Hunter rate: collapse from far out of balance
fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
top.plot(t_h / T_FF, m_h, color=BLUE, lw=2)
top.set_ylabel("protostar mass  [Msun]")
top.set_title("How fast the protostar grows", color=INK)
bot.semilogy(grid_t / T_FF, mdot, color=ORANGE, lw=2, label="accretion rate")
bot.axhline(shu, color=MUTED, lw=1, ls="--", label=r"Shu (1977): $0.975\,c_s^3/G$, cloud starting in balance")
bot.axhline(lph, color=INK, lw=1, ls=":", label=r"Larson-Penston-Hunter: $46.9\,c_s^3/G$, dynamic collapse")
bot.set_xlabel(r"time  [$t / t_{\rm ff}$]")
bot.set_ylabel("accretion rate  [Msun/yr]")
bot.legend(frameon=False, fontsize=8)
for ax in (top, bot):
    style(ax)
save(fig, PLOTS, "accretion_history.png")

# ---------------------------------------------------------------- 4. Where the angular momentum goes
times, L_gas_disk, L_gas_env, L_spin, L_sink_orbit = [], [], [], [], []
for s in snaps:
    times.append(s["t"] / T_FF)
    Lz = s["mass"] * (s["pos"][:, 0] * s["vel"][:, 1] - s["pos"][:, 1] * s["vel"][:, 0])
    if len(s["sink_mass"]):
        xs, vs, ms = central_frame(s)
        disk = disk_members(xs, vs, ms)
    else:
        disk = np.zeros(len(s["mass"]), bool)
    L_gas_disk.append(Lz[disk].sum())
    L_gas_env.append(Lz[~disk].sum())
    L_spin.append(s["sink_spin"][:, 2].sum())
    L_sink_orbit.append(np.sum(s["sink_mass"] * (s["sink_pos"][:, 0] * s["sink_vel"][:, 1] - s["sink_pos"][:, 1] * s["sink_vel"][:, 0])))
L0 = L_gas_env[0]
total = np.array(L_gas_disk) + np.array(L_gas_env) + np.array(L_spin) + np.array(L_sink_orbit)
print(f"angular momentum: max |L(t)/L0 - 1| = {np.abs(total / L0 - 1).max():.1e}")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.stackplot(times, np.array(L_gas_env) / L0, np.array(L_gas_disk) / L0, np.array(L_spin) / L0,
             colors=[GRID, BLUE, ORANGE], labels=["infalling envelope", "disk", "protostar spin"])
ax.plot(times, total / L0, color=INK, lw=1.2, ls="--", label="total (conserved)")
ax.set_xlabel(r"time  [$t / t_{\rm ff}$]")
ax.set_ylabel(r"angular momentum  [$L_z / L_0$]")
ax.set_title("The cloud's spin ends up mostly in the disk", color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_xlim(0, times[-1])
style(ax)
save(fig, PLOTS, "angular_momentum.png")

# ---------------------------------------------------------------- 5. Animation: face-on and edge-on views
lim = 600.0
fig, (face, edge) = plt.subplots(1, 2, figsize=(10, 5.2))
all_rho = np.concatenate([s["rho"] for s in snaps])
vmin, vmax = np.log10(np.percentile(all_rho, 5)), np.log10(np.percentile(all_rho, 99.9))
dots = [ax.scatter([], [], c=[], cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, s=2) for ax in (face, edge)]
stars = [ax.scatter([], [], s=90, marker="*", color=YELLOW, edgecolor=INK, linewidth=0.6, zorder=5) for ax in (face, edge)]
for ax, title, ylabel in ((face, "face-on (looking down the spin axis)", "y [AU]"), (edge, "edge-on", "z [AU]")):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=10)
    style(ax)
clock = face.text(0.03, 0.95, "", transform=face.transAxes, color=INK, fontsize=10)
fig.tight_layout()


def draw(frame):
    s = snaps[frame]
    c = s["sink_pos"][np.argmax(s["sink_mass"])] if len(s["sink_mass"]) else np.zeros(3)
    p = s["pos"] - c
    for k, (a, b) in enumerate(((0, 1), (0, 2))):
        dots[k].set_offsets(p[:, [a, b]])
        dots[k].set_array(np.log10(s["rho"]))
        sp = s["sink_pos"] - c
        stars[k].set_offsets(sp[:, [a, b]] if len(sp) else np.zeros((0, 2)))
    clock.set_text(f"t = {s['t'] / T_FF:.2f} t_ff ({s['t'] / 1e3:.1f} kyr), star {s['sink_mass'].sum():.2f} Msun")
    return dots + stars + [clock]


anim = FuncAnimation(fig, draw, frames=len(snaps), interval=120, blit=True)
anim.save(PLOTS / "collapse_to_disk.gif", writer=PillowWriter(fps=8), dpi=80)
print(f"saved {PLOTS / 'collapse_to_disk.gif'}")

# ---------------------------------------------------------------- handoff for Rungs 4b and 5
np.savez(HERE / "handoff.npz", M_star=M_star, M_disk=m_disk.sum(), R_centers=Rc, sigma_Msun_AU2=sigma,
         sigma_slope=slope, accretion_t_yr=t_h, accretion_mass_Msun=m_h, r_acc_AU=R_ACC)
print(f"saved {HERE / 'handoff.npz'}")
