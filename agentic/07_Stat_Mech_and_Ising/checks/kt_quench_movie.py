"""Movie of a quench: random spins dropped to T = 0.5, below T_KT, and the vortices annihilating.

    python checks/kt_quench_movie.py

Writes plots/kt_quench.gif (one replica, L = 128, frames log-spaced in time
out to 5000 sweeps) and plots/kt_quench.png (four still frames plus the vortex
density against time, averaged over 8 replicas).

Below T_KT, opposite vortices attract with a force of about 2 pi J / r, so
pairs find each other and annihilate, and the colour domains grow. For the XY
model the vortex density falls as n_v ~ ln(t) / t (Yurke et al. 1993), slower
than 1/t because of the logarithmic vortex energy. This previews phase 4; here
it is only a picture.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import lattice

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
L, R, T, T_MAX = 128, 8, 0.5, 5000


def main():
    lattice.seed(2)
    beta = np.full(R, 1 / T)
    th = np.random.default_rng(2).uniform(0, 2 * np.pi, (R, L, L))
    # Fixed step size from a short tuning run on a copy, so the quench itself
    # starts from the untouched random configuration.
    delta = lattice.tune_delta(th.copy(), beta)
    stops = np.unique(np.round(np.logspace(0, np.log10(T_MAX), 60)).astype(int))
    frames, times, n_v = [th[0].copy()], [0], [np.abs(lattice.vortices(th)).mean()]
    t = 0
    for stop in stops:
        while t < stop:
            lattice.xy_sweep(th, beta, delta)
            t += 1
        frames.append(th[0].copy())
        times.append(t)
        n_v.append(np.abs(lattice.vortices(th)).mean())
    times, n_v = np.array(times), np.array(n_v)

    fig, ax = plt.subplots(figsize=(6, 6.4))
    im = ax.imshow(frames[0], cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi)
    plus = ax.scatter([], [], s=10, c="k", marker="o")
    minus = ax.scatter([], [], s=10, c="w", edgecolors="k", linewidths=0.4, marker="s")
    ax.set(xticks=[], yticks=[])

    def draw(k):
        im.set_data(frames[k])
        q = lattice.vortices(frames[k][None])[0]
        for sc, sign in ((plus, 1), (minus, -1)):
            i, j = np.nonzero(q == sign)
            sc.set_offsets(np.c_[j + 0.5, i + 0.5])
        ax.set_title(f"quench to T = {T}: t = {times[k]} sweeps, {np.abs(q).sum()} vortices")
        return im, plus, minus

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(PLOTS / "kt_quench.gif", writer=PillowWriter(fps=8))
    plt.close(fig)

    fig = plt.figure(figsize=(16, 4.4))
    grid = fig.add_gridspec(1, 5, wspace=0.45)
    for n, target in enumerate((0, 10, 100, T_MAX)):
        k = int(np.argmin(np.abs(times - target)))
        a = fig.add_subplot(grid[0, n])
        a.imshow(frames[k], cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi)
        q = lattice.vortices(frames[k][None])[0]
        for sign, c, m in ((1, "k", "o"), (-1, "w", "s")):
            i, j = np.nonzero(q == sign)
            a.scatter(j + 0.5, i + 0.5, s=4, c=c, edgecolors="k", linewidths=0.3, marker=m)
        a.set(title=f"t = {times[k]}: {np.abs(q).sum()} vortices", xticks=[], yticks=[])
    a = fig.add_subplot(grid[0, 4])
    ok = (times > 0) & (n_v > 0)
    a.loglog(times[ok], n_v[ok], "o-", ms=3, label=f"mean of {R} replicas")
    tt = times[(times >= 10) & ok].astype(float)
    k0 = int(np.argmin(np.abs(times - 10)))
    a.loglog(tt, n_v[k0] * (np.log(tt) / tt) / (np.log(10) / 10), "k--", label=r"$\propto \ln t\,/\,t$")
    a.set(xlabel="sweeps since quench", ylabel="vortices per plaquette", title="pairs annihilate")
    a.legend(fontsize=8)
    fig.savefig(PLOTS / "kt_quench.png", dpi=120, bbox_inches="tight")
    print("t, n_v:", ", ".join(f"{a}: {b:.2e}" for a, b in zip(times[::6], n_v[::6])))


if __name__ == "__main__":
    main()
