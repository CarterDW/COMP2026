"""Vortex pairs unbinding with temperature: snapshots and nearest-antivortex distances.

    python checks/kt_pair_spread.py

A figure, not a check; it previews step 3. At every T most vortices sit one
plaquette from an antivortex: thermal fluctuations keep creating tight pairs,
so the median distance is 1 throughout. Unbinding shows in the tail, as the
fraction of vortices whose nearest antivortex is beyond r = 2. It is divided
by the same fraction for free vortices at the same density, placed at random
(Poisson, antivortex density n_v / 2): exp(-pi (n_v / 2) 2^2). Undivided, it
falls again at T = 1.3, because crowding then puts an antivortex near
everything, free or not.

A first version scaled r by the mean spacing, u = r sqrt(n_v). With r stuck at
1 that only tracked sqrt(n_v), so it was dropped.

Equilibration is 3000 sweeps at L = 64, which is adequate for a picture but
not for the precision runs of step 3.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lattice

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
L, R, TS = 64, 32, np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3])
SNAP_T = (0.7, 0.9, 1.1, 1.3)


def nearest_antivortex(q):
    """Distance from every +1 vortex to the nearest -1, with periodic minimum-image distances."""
    plus, minus = np.argwhere(q == 1), np.argwhere(q == -1)
    if len(plus) == 0:
        return np.empty(0)
    d = np.abs(plus[:, None, :] - minus[None, :, :])
    d = np.minimum(d, L - d)
    return np.sqrt((d**2).sum(-1)).min(1)


def sample():
    """Equilibrate R replicas at every T from an ordered start, then save 20 configurations 100 sweeps apart.

    A random (infinitely hot) start leaves quench vortices that, below T_KT, are
    still annihilating after 3000 sweeps: at T = 0.7 it gave 17% of vortices with
    no antivortex within r = 2, against 2.5% from an ordered start. For T >= 1.0 both starts agree. From an
    ordered start, every vortex is thermally created as a pair.
    """
    lattice.seed(1)
    beta = np.repeat(1 / TS, R)
    theta = np.zeros((len(beta), L, L))
    delta = lattice.tune_delta(theta, beta)
    for _ in range(3000):
        lattice.xy_sweep(theta, beta, delta)
    configs = []
    for _ in range(20):
        for _ in range(100):
            lattice.xy_sweep(theta, beta, delta)
        configs.append(theta.copy())
    return np.array(configs).reshape(20, len(TS), R, L, L)


def main():
    configs = sample()
    q = lattice.vortices(configs.reshape(-1, L, L)).reshape(configs.shape)
    n_v = np.abs(q).mean((0, 2, 3, 4))
    raw = []
    for k in range(len(TS)):
        r = np.concatenate([nearest_antivortex(q[c, k, s]) for c in range(20) for s in range(R)])
        raw.append(r)
        print(f"T = {TS[k]:.2f}: n_v = {n_v[k]:.4f}, {len(r)} vortices, mean r = {r.mean():.2f}, "
              f"fraction r > 2: {np.mean(r > 2):.3f} (Poisson {np.exp(-2 * np.pi * n_v[k]):.3f})")

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.3)
    bottom = grid[1, :].subgridspec(1, 2, wspace=0.3)
    for n, T in enumerate(SNAP_T):
        k = int(np.argmin(np.abs(TS - T)))
        ax = fig.add_subplot(grid[0, n])
        im = ax.imshow(configs[-1, k, 0], cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi)
        qq = q[-1, k, 0]
        for sign, c, m in ((1, "k", "o"), (-1, "w", "s")):
            i, j = np.nonzero(qq == sign)
            ax.scatter(j + 0.5, i + 0.5, s=18, c=c, edgecolors="k", marker=m)
        ax.set(title=f"T = {T}: {np.abs(qq).sum()} vortices", xticks=[], yticks=[])
    cb = fig.colorbar(im, ax=fig.axes[:4], fraction=0.012, pad=0.01, ticks=np.arange(5) * np.pi / 2)
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    cb.set_label(r"spin angle $\theta$")

    ax = fig.add_subplot(bottom[0])
    bins = np.array([0.5, 1.2, 1.7, 2.5, 3.5, 5, 7, 10, 14, 20, 28, 40])
    mid = np.sqrt(bins[1:] * bins[:-1])
    for k, T in enumerate(TS):
        h, _ = np.histogram(raw[k], bins)
        p = h / len(raw[k]) / np.diff(bins)
        ax.loglog(mid[h > 0], p[h > 0], "o-", ms=4, label=f"T = {T}")
    ax.set(xlabel="distance r to nearest antivortex (lattice units)", ylabel="probability per unit r",
           title="the tail of loose pairs and free vortices grows with T")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(bottom[1])
    ratio = np.array([np.mean(r > 2) for r in raw]) / np.exp(-np.pi * n_v / 2 * 4)
    ax.plot(TS, ratio, "o-", label="vortices with no antivortex within r = 2,\nrelative to all-free (Poisson) at the same density")
    ax.axvline(0.893, color="gray", ls=":", label=r"$T_{KT}$ = 0.893")
    ax2 = ax.twinx()
    ax2.semilogy(TS, n_v, "^--", color="C3", alpha=0.7, label="vortex density $n_v$ (right axis)")
    ax2.set_ylabel("vortices per plaquette", color="C3")
    ax.set(xlabel="T", ylabel="free-like fraction (1 = all free)", ylim=(0, 0.3),
           title="more vortices lack a close partner as T rises")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.savefig(PLOTS / "kt_pair_spread.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
