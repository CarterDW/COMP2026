"""KT warmup, step 1: the XY checkerboard engine.

    python checks/kt_step1_engine.py

Targets are exact or asymptotically exact: the von Mises distribution of one
spin in a fixed field, an independent sequential sampler, and the low- and
high-temperature series for the energy, each including its first correction
term so the check cannot pass on the leading slope alone.

Error bars come from the scatter between independent replicas at the same T,
which needs no autocorrelation estimate.
"""

import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

import lattice
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
FIG = {}


def mean_err(u):
    """Mean and standard error of u (n_meas, R), from the scatter of the R replica means."""
    m = u.mean(0)
    return m.mean(), m.std(ddof=1) / np.sqrt(len(m))


def check_von_mises():
    """One spin with frozen neighbours samples p(theta) ~ exp(kappa cos(theta - phi)) exactly.

    Only site (0, 0) of a 4 x 4 lattice moves, through the same site kernel the sweep uses. Its four
    neighbours sit at phi = 1.0, so |h| = 4 and kappa = 4 beta. Each replica
    makes 1500 moves and contributes only its final angle, so the samples are
    independent and a chi^2 test is valid. Six groups: kappa = 0.5, 2, 8, each
    with a uniform proposal (delta = pi) and a local one (delta = 0.6), which
    tests the wrap-around of small steps.
    """
    lattice.seed(1)
    phi, n_per, n_moves = 1.0, 20000, 1500
    groups = [(k, d) for k in (0.5, 2.0, 8.0) for d in (np.pi, 0.6)]
    beta = np.repeat([k / 4 for k, _ in groups], n_per)
    delta = np.repeat([d for _, d in groups], n_per)
    theta = np.random.default_rng(1).uniform(0, 2 * np.pi, (len(beta), 4, 4))
    for i, j in ((0, 1), (0, 3), (1, 0), (3, 0)):
        theta[:, i, j] = phi
    lattice.xy_site_moves(theta, beta, delta, 0, 0, n_moves)
    samples = theta[:, 0, 0].reshape(len(groups), n_per)

    edges = np.linspace(0, 2 * np.pi, 41)
    worst_p, worst_c, lines = 1.0, 0.0, []
    FIG["vm"] = (edges, samples, groups, phi)
    for (k, d), s in zip(groups, samples):
        cdf = stats.vonmises.cdf(edges, k, loc=phi)
        expect = n_per * np.diff(cdf)
        # Bins expecting < 5 counts break the chi^2 approximation (at kappa = 8 that
        # is about half of them, far from phi), so they are left out.
        use = expect >= 5
        chi2 = ((np.histogram(s, edges)[0] - expect) ** 2 / expect)[use].sum()
        p = stats.chi2.sf(chi2, use.sum() - 1)
        c = np.cos(s - phi).mean()
        z = (c - special.i1(k) / special.i0(k)) / (np.cos(s - phi).std() / np.sqrt(n_per))
        worst_p, worst_c = min(worst_p, p), max(worst_c, abs(z))
        lines.append(f"k={k:g},d={d:.2g}: p={p:.2f}")
    # Six tests: the smallest p of six true nulls is below 0.01 about 6% of the time.
    ok = worst_p > 0.01 and worst_c < 4
    return ok, f"chi^2 {'; '.join(lines)}; worst <cos> pull {worst_c:.1f} sigma"


def sequential_xy(L, T, n_sweeps, seed, delta=1.0):
    """Independent reference: plain random-site Metropolis, one site at a time, pure Python."""
    rnd = random.Random(seed)
    th = [[rnd.uniform(0, 2 * math.pi) for _ in range(L)] for _ in range(L)]
    beta, u = 1 / T, []
    for sweep in range(n_sweeps):
        for _ in range(L * L):
            i, j = rnd.randrange(L), rnd.randrange(L)
            nb = (th[(i + 1) % L][j], th[i - 1][j], th[i][(j + 1) % L], th[i][j - 1])
            old, new = th[i][j], th[i][j] + delta * rnd.uniform(-1, 1)
            dE = sum(math.cos(old - t) - math.cos(new - t) for t in nb)
            if dE <= 0 or rnd.random() < math.exp(-beta * dE):
                th[i][j] = new
        if sweep >= n_sweeps // 5:
            u.append(-sum(math.cos(th[i][j] - th[(i + 1) % L][j]) + math.cos(th[i][j] - th[i][(j + 1) % L])
                          for i in range(L) for j in range(L)) / L**2)
    return np.array(u)


def check_vs_sequential():
    """Checkerboard order and random-site order sample the same distribution, at L = 8, T = 0.9.

    T = 0.9 is at the transition, where a wrong update order would matter most.
    The sequential chain's error uses 40 blocks, each much longer than its
    autocorrelation time at this size.
    """
    L, T = 8, 0.9
    useq = sequential_xy(L, T, 25000, seed=2)
    blocks = useq[: len(useq) // 40 * 40].reshape(40, -1).mean(1)
    ms, es = useq.mean(), blocks.std(ddof=1) / np.sqrt(40)
    _, u = lattice.run_xy(L, np.full(256, T), 500, 2000, 3)
    mc, ec = mean_err(u)
    z = (mc - ms) / np.hypot(es, ec)
    return abs(z) < 3, f"checkerboard u = {mc:.5f} +/- {ec:.5f}, sequential {ms:.5f} +/- {es:.5f}: {z:+.1f} sigma"


def check_low_T():
    """Spin waves: u = -2 + (T/2)(1 - 1/N) + T^2/16 + O(T^3).

    Equipartition gives T/2 per spin for the N - 1 non-zero modes. The quartic
    term of cos, at first order with <dtheta^2> = T/2 per bond, gives T^2/16. The
    check fits y = (u - u_harmonic)/T^2 = a + bT and needs the intercept a = 1/16.
    The T^3 term is the slope b, so it is fitted, not neglected.
    """
    L, Ts = 64, np.array([0.05, 0.1, 0.15, 0.2, 0.3])
    _, u = lattice.run_xy(L, np.repeat(Ts, 32), 1500, 3000, 4, ordered=True)
    y, ey = [], []
    for k, T in enumerate(Ts):
        m, e = mean_err(u[:, 32 * k: 32 * (k + 1)])
        harm = -2 + T / 2 * (1 - 1 / L**2)
        y.append((m - harm) / T**2)
        ey.append(e / T**2)
    y, ey = np.array(y), np.array(ey)
    (b, a), cov = np.polyfit(Ts, y, 1, w=1 / ey, cov="unscaled")
    chi2 = (((y - a - b * Ts) / ey) ** 2).sum()
    z = (a - 1 / 16) / np.sqrt(cov[1, 1])
    FIG["low"] = (Ts, y, ey, a, b)
    ok = abs(z) < 3 and stats.chi2.sf(chi2, len(Ts) - 2) > 0.01
    return ok, (f"intercept {a:.4f} +/- {np.sqrt(cov[1, 1]):.4f} vs 1/16 = 0.0625 ({z:+.1f} sigma), "
                f"slope {b:.3f}, chi^2 = {chi2:.1f} for {len(Ts) - 2} dof")


def u_high(beta):
    """High-T series to O(beta^5): u = -2t - 8 t^3 dt/dbeta with t = I1/I0.

    Writing exp(beta cos x) = I0 (1 + 2 sum_n t_n cos nx) turns Z into a sum over
    closed loops of bond currents. The shortest loop is the plaquette, two
    orientations, weight t^4 each, so ln Z / N = 2 ln I0 + 2 t^4 + O(t^6).
    """
    t = special.i1(beta) / special.i0(beta)
    dt = 1 - t / beta - t**2
    return -2 * t - 8 * t**3 * dt


def check_high_T():
    """At T = 6 and 10, u matches the loop series, and the leading -beta alone is rejected.

    The plaquette loop shifts u by about -3 beta^3 / 8: 1.7e-3 at T = 6, many
    sigma, so the check sees the lattice's loops, not just independent bonds.

    The neglected O(beta^5) term is 1.3e-4 at T = 6 if its coefficient is of
    order one, comparable to the statistical error there. So the tolerance is
    3 sigma plus beta^5. Repeated seeds put the T = 6 residual at about
    -0.4 beta^5, consistently negative: that is the next term showing.
    """
    L, Ts = 64, np.array([6.0, 10.0])
    _, u = lattice.run_xy(L, np.repeat(Ts, 64), 300, 2000, 5)
    lines, ok, pts = [], True, []
    for k, T in enumerate(Ts):
        m, e = mean_err(u[:, 64 * k: 64 * (k + 1)])
        res, z0 = m - u_high(1 / T), (m + 1 / T) / e
        ok &= abs(res) < 3 * e + T**-5 and abs(z0) > 3
        pts.append((1 / T, m + 1 / T, e))
        lines.append(f"T={T:g}: u = {m:.6f} +/- {e:.6f}, series residual {res / e:+.1f} sigma "
                     f"= {res * T**5:+.2f} beta^5, -beta alone {z0:+.1f} sigma")
    FIG["high"] = pts
    return ok, "; ".join(lines)


def check_benchmark():
    """Spin updates per second, for sizing step 3. Always passes; the number is the point.

    The numpy version of this engine ran at about 5M/s on this machine.
    """
    rng = np.random.default_rng(6)
    out = []
    for L, R in ((64, 16), (256, 4), (512, 32)):
        theta = rng.uniform(0, 2 * np.pi, (R, L, L))
        beta, delta = np.ones(R), np.ones(R)
        lattice.xy_sweep(theta, beta, delta)
        t0 = time.perf_counter()
        for _ in range(50):
            lattice.xy_sweep(theta, beta, delta)
        rate = 50 * R * L * L / (time.perf_counter() - t0)
        out.append(f"L={L}, R={R}: {rate / 1e6:.1f}M/s")
    return True, ", ".join(out)


def plot():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    edges, samples, groups, phi = FIG["vm"]
    x = np.linspace(0, 2 * np.pi, 400)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for (k, d), s in zip(groups, samples):
        if d != np.pi:
            continue
        h = np.histogram(s, edges, density=True)[0]
        line, = ax[0].plot(x, stats.vonmises.pdf(x, k, loc=phi), lw=1)
        ax[0].plot(mid, h, "o", ms=3, color=line.get_color(), label=fr"$\kappa$ = {k:g}")
    ax[0].set(xlabel=r"$\theta$", ylabel="density", title="one spin in a fixed field (delta = pi)")
    ax[0].legend()

    Ts, y, ey, a, b = FIG["low"]
    ax[1].errorbar(Ts, y, ey, fmt="o", label="simulation, L = 64")
    tt = np.linspace(0, Ts.max(), 50)
    ax[1].plot(tt, a + b * tt, label=f"fit: {a:.4f} + {b:.3f} T")
    ax[1].axhline(1 / 16, color="k", ls=":", label="1/16")
    ax[1].set(xlabel="T", ylabel=r"$(u - u_{\rm harm})/T^2$", title="low T: anharmonic correction")
    ax[1].legend()

    bb = np.linspace(0.05, 0.2, 100)
    ax[2].plot(bb, u_high(bb) + bb, label=r"loop series $-2t - 8t^3t'$, plus $\beta$")
    ax[2].plot(bb, -3 * bb**3 / 8, ":", label=r"$-3\beta^3/8$")
    for be, r, e in FIG["high"]:
        ax[2].errorbar(be, r, e, fmt="o", color="k")
    ax[2].axhline(0, color="gray", lw=0.5)
    ax[2].set(xlabel=r"$\beta$", ylabel=r"$u + \beta$", title="high T: beyond independent bonds")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "kt_step1_engine.png", dpi=130)


if __name__ == "__main__":
    fails = run("KT step 1: XY engine", [check_von_mises, check_vs_sequential, check_low_T,
                                         check_high_T, check_benchmark])
    plot()
    sys.exit(fails)
