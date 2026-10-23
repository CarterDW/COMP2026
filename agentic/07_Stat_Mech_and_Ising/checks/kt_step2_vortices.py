"""KT warmup, step 2: counting vortices.

    python checks/kt_step2_vortices.py

The counter is tested on configurations whose vortices are known by
construction (planted ones), on two exact statements (total charge on a torus,
density for random angles), and on the physics the KT argument rests on: a
vortex pair costs 2 pi J ln r.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lattice
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
FIG = {}


def planted(L, charges):
    """theta = sum_k q_k arg(r - c_k), with each centre c_k at the middle of plaquette (i, j).

    Sites sit at (x, y) = (j, i), so plaquette (i, j) is centred at (j + 1/2, i + 1/2).
    charges is a list of (i, j, q). Returns theta of shape (1, L, L).

    This is the infinite-plane field, not the torus one. A set with dipole
    moment p = sum q r has theta jumping by about 4|p|/L across the periodic
    seam. That puts spurious vortices on the seam once the jump passes pi, and
    adds about 2 pi r^2 / L of seam energy to a pair. Both were seen in a first
    version of these checks. So every planted set here has zero dipole moment,
    and its field falls off as 1/r^2, continuous across the seam to O((size/L)^2).
    """
    y, x = np.indices((L, L), dtype=float)
    theta = sum(q * np.arctan2(y - i - 0.5, x - j - 0.5) for i, j, q in charges)
    return np.mod(theta, 2 * np.pi)[None]


def zero_dipole_set(n_pairs, lo, hi, rng):
    """n_pairs vortex-antivortex pairs at random plaquettes in [lo, hi)^2 with sum q r = 0.

    The +1 positions and n - 1 of the -1 positions are random; the last -1 is
    placed so the -1 positions sum to the +1 positions. Redrawn until that site
    is in range and every vortex is at least 3 plaquettes from every other.
    """
    while True:
        plus = rng.integers(lo, hi, (n_pairs, 2))
        minus = rng.integers(lo, hi, (n_pairs - 1, 2))
        last = plus.sum(0) - minus.sum(0)
        pts = np.vstack([plus, minus, last])
        far = all(np.abs(pts[a] - pts[b]).max() >= 3 for a in range(len(pts)) for b in range(a))
        if far and np.all((last >= lo) & (last < hi)):
            return [(i, j, 1 if k < n_pairs else -1) for k, (i, j) in enumerate(pts)]


def check_planted():
    """Planted sets of 2 to 4 pairs are found exactly: right plaquettes, right signs, nothing else.

    40 random zero-dipole sets on a 48 x 48 torus, with the vortices in the
    central 32 x 32. (A single pair cannot have zero dipole; the quadrupole in
    check_pair_energy covers it.) Swapping every sign must swap every detected charge.
    """
    rng = np.random.default_rng(1)
    L, bad = 48, 0
    for trial in range(40):
        charges = zero_dipole_set(2 + trial % 3, 8, 40, rng)
        want = np.zeros((L, L), int)
        for i, j, q in charges:
            want[i, j] = q
        got = lattice.vortices(planted(L, charges))[0]
        flipped = lattice.vortices(planted(L, [(i, j, -q) for i, j, q in charges]))[0]
        bad += not (np.array_equal(got, want) and np.array_equal(flipped, -want))
        if trial == 7:
            FIG["planted"] = (planted(L, charges)[0], got)
    return bad == 0, f"{40 - bad}/40 planted sets recovered exactly, and with all signs flipped"


def check_total_charge():
    """Sum of q is 0 on every configuration: random, ordered, and equilibrated at T = 0.7, 1.0, 1.5.

    On a torus every bond is traversed once in each direction by its two
    plaquettes, so the wrapped differences cancel exactly. Any nonzero total
    means the wrap or the loop orientation is inconsistent between plaquettes.
    """
    rng = np.random.default_rng(2)
    configs = [rng.uniform(0, 2 * np.pi, (64, 32, 32)), np.zeros((4, 32, 32))]
    configs.append(lattice.run_xy(32, np.repeat([0.7, 1.0, 1.5], 8), 200, 1, 3)[0])
    totals = np.concatenate([lattice.vortices(c).sum((1, 2)) for c in configs])
    counts = np.concatenate([np.abs(lattice.vortices(c)).sum((1, 2)) for c in configs])
    return np.all(totals == 0), (f"max |sum q| = {np.abs(totals).max()} over {len(totals)} configs, "
                                 f"which held up to {counts.max()} vortices each")


def check_random_density():
    """Random angles give a vortex on exactly 1/3 of plaquettes, and an all-aligned lattice gives none.

    Three of a plaquette's wrapped differences are independent U(-pi, pi); the
    fourth closes the loop, so q != 0 exactly when |d1 + d2 + d3| > pi. For a sum
    of three uniforms that probability is 1/3 (Irwin-Hall), with +1 and -1
    equally likely. Errors come from the scatter between configurations,
    because neighbouring plaquettes share a bond and are correlated.
    """
    rng = np.random.default_rng(4)
    q = lattice.vortices(rng.uniform(0, 2 * np.pi, (400, 64, 64)))
    n = np.abs(q).mean((1, 2))
    plus = (q == 1).mean((1, 2))
    m, e = n.mean(), n.std(ddof=1) / np.sqrt(len(n))
    mp, ep = plus.mean(), plus.std(ddof=1) / np.sqrt(len(n))
    zero = np.abs(lattice.vortices(np.full((1, 64, 64), 1.3))).sum()
    z, zp = (m - 1 / 3) / e, (mp - 1 / 6) / ep
    ok = abs(z) < 3 and abs(zp) < 3 and zero == 0
    return ok, (f"density {m:.5f} +/- {e:.5f} vs 1/3 ({z:+.1f} sigma); +1 only {mp:.5f} vs 1/6 "
                f"({zp:+.1f} sigma); aligned lattice: {zero} vortices")


def check_pair_energy():
    """Vortices interact as a 2D Coulomb gas, E = -2 pi sum_{i<j} q_i q_j ln r_ij + const: the KT energy input.

    A lone vortex costs pi J ln(R/a), and two opposite charges at distance r
    cost 2 pi J ln(r/a) + 2 E_core. Planted as a square of side s with
    alternating charges (zero dipole, so no seam energy), the pair sum is four
    opposite pairs at s and two like pairs at s sqrt 2:
    E = 4 pi ln s + const. Measured above the aligned state on a 256 x 256
    torus for s = 4 to 32. Core and lattice effects only enter the constant
    once s is a few spacings.

    The fitted slope comes out about 1% high, and that excess is from the cores,
    not the torus. The local slope between s and 2s is +2.9%, +0.9%, +0.2-0.3%
    for s = 4, 8, 16, falling roughly as 1/s^2, and identical at L = 256, 512
    and 1024.
    """
    L, ss = 256, np.array([4, 6, 8, 12, 16, 24, 32])
    E = []
    for s in ss:
        a, b = 128 - s // 2, 128 + s // 2
        th = planted(L, [(a, a, 1), (a, b, -1), (b, b, 1), (b, a, -1)])
        E.append(L**2 * (lattice.xy_energy(th)[0] + 2))
    slope, c = np.polyfit(np.log(ss), E, 1)
    FIG["pair"] = (ss, np.array(E), slope, c)
    return abs(slope / (4 * np.pi) - 1) < 0.02, f"slope dE/d ln s = {slope:.4f} vs 4 pi = {4 * np.pi:.4f} ({slope / (4 * np.pi) - 1:+.2%})"


def plot():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    th, q = FIG["planted"]
    ax[0].imshow(th, cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi)
    for sign, c in ((1, "k"), (-1, "w")):
        i, j = np.nonzero(q == sign)
        ax[0].scatter(j + 0.5, i + 0.5, s=60, c=c, edgecolors="k", marker="o" if sign > 0 else "s")
    ax[0].set(title="planted pairs: colour = angle, detected +1 (black) / -1 (white)", xlabel="x", ylabel="y")
    ax[0].title.set_fontsize(9)

    ss, E, slope, c = FIG["pair"]
    rr = np.linspace(3.5, 34, 100)
    ax[1].plot(ss, E, "o", label="planted +-+- square on 256$^2$")
    ax[1].plot(rr, c + 4 * np.pi * np.log(rr), label=r"Coulomb gas: $4\pi \ln s$ + const")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="square side s", ylabel="energy above aligned state", title="vortex interaction: the KT input")
    ax[1].legend()

    lattice.seed(7)
    th = lattice.run_xy(64, [0.95], 2000, 1, 7)[0]
    q = lattice.vortices(th)[0]
    im = ax[2].imshow(th[0], cmap="hsv", origin="lower", vmin=0, vmax=2 * np.pi)
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, ticks=np.arange(5) * np.pi / 2)
    cb.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    cb.set_label(r"spin angle $\theta$ (cyclic: 0 and $2\pi$ are the same)")
    for sign, c in ((1, "k"), (-1, "w")):
        i, j = np.nonzero(q == sign)
        ax[2].scatter(j + 0.5, i + 0.5, s=25, c=c, edgecolors="k", marker="o" if sign > 0 else "s")
    ax[2].set(title=f"equilibrium, T = 0.95, L = 64: {np.abs(q).sum()} vortices", xlabel="x", ylabel="y")
    fig.tight_layout()
    fig.savefig(PLOTS / "kt_step2_vortices.png", dpi=130)


if __name__ == "__main__":
    fails = run("KT step 2: vortex counter", [check_planted, check_total_charge,
                                              check_random_density, check_pair_energy])
    plot()
    sys.exit(fails)
