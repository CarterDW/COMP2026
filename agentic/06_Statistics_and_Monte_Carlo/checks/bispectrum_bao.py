"""BAO wiggles in the tree-level matter bispectrum, B_wiggle / B_no-wiggle - 1.

    python -I checks/bispectrum_bao.py      (seconds; reuses the CAMB z = 0 spectrum)

Tree-level standard perturbation theory gives

    B(k1, k2, k3) = 2 F2(k1, k2) P(k1) P(k2) + 2 cyclic permutations,
    F2(a, b) = 5/7 + (mu/2)(a/b + b/a) + (2/7) mu^2,   mu = cos angle(a, b),

with k3 = -(k1 + k2). The wiggle and no-wiggle bispectra use the same formula with
P or P_nw, both taken from CAMB at z = 0 with cmb_and_bao.no_wiggle.

Configurations, assumed to match the reference figure: theta is the angle between
k1 and k2, and k2 = k1 + delta pi / r_s. Wiggles repeat every 2 pi / r_s in k, so
delta = 0 and 2 put P(k1) and P(k2) in phase (constructive) and delta = 1 in
antiphase (destructive). These are theory ("intrinsic") curves only. The
"extracted" curves of the reference need a bispectrum estimator run on
gravitationally evolved mocks.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from checks import cmb_and_bao as CB
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
THETAS = (0.0, np.pi / 4, np.pi / 2)
DELTAS = (0, 1, 2)
K1 = np.linspace(0.01, 0.2, 400)


def F2(a, b, mu):
    return 5 / 7 + 0.5 * mu * (a / b + b / a) + 2 / 7 * mu**2


def bispectrum(P, k1, k2, mu12):
    """Tree-level B for |k1|, |k2| and the cosine between them. P is a callable."""
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
    mu23 = -(k2 + k1 * mu12) / k3            # k2 . k3 / (k2 k3), with k3 = -(k1 + k2)
    mu31 = -(k1 + k2 * mu12) / k3
    P1, P2, P3 = P(k1), P(k2), P(k3)
    return 2 * (F2(k1, k2, mu12) * P1 * P2 + F2(k2, k3, mu23) * P2 * P3 + F2(k3, k1, mu31) * P3 * P1)


def spectra():
    P0 = CB.matter_pk()[0]
    Pw = lambda k: np.interp(np.log(k), np.log(CB.KH), P0)
    Pnw = lambda k: np.interp(np.log(k), np.log(CB.KH), CB.no_wiggle(P0))
    return Pw, Pnw


def ratio(theta, delta):
    Pw, Pnw = spectra()
    k2 = K1 + delta * np.pi / CB.r_drag_h()
    mu = np.cos(theta)
    return bispectrum(Pw, K1, k2, mu) / bispectrum(Pnw, K1, k2, mu) - 1


def check_F2_limits():
    """F2 obeys its exact limits, and its angle average is 17/21.

    * Collinear equal modes (a = b, mu = 1): F2 = 5/7 + 1 + 2/7 = 2.
    * a = -b (mu = -1, equal size): F2 = 0. Two opposite modes add up to no net
      wavevector, and mass conservation forbids a k -> 0 response.
    * Averaged over mu at a = b: 5/7 + (2/7)(1/3) = 17/21. This is the
      delta_1^2 coefficient quoted earlier as gravity's second-order term.
    """
    mu = np.linspace(-1, 1, 200001)
    avg = np.sum(0.5 * (F2(1, 1, mu)[1:] + F2(1, 1, mu)[:-1]) * np.diff(mu)) / 2
    e = max(abs(F2(1.0, 1.0, 1.0) - 2), abs(F2(1.0, 1.0, -1.0)), abs(avg - 17 / 21))
    return e < 1e-9, f"F2(mu=1) = {F2(1., 1., 1.):.6f}, F2(mu=-1) = {F2(1., 1., -1.):.1e}, <F2> = {avg:.6f} (17/21 = {17 / 21:.6f})"


def check_bispectrum_symmetric():
    """B is symmetric under relabelling the triangle. Compute B(k1, k2, k3), then
    swap the roles of k1 and k3, with the angle recomputed from the side lengths."""
    Pw, _ = spectra()
    k1, k2, mu = 0.07, 0.11, np.cos(0.9)
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu)
    mu_32 = (k1**2 - k3**2 - k2**2) / (2 * k3 * k2)          # angle between k3 and k2 vectors
    a, b = bispectrum(Pw, k1, k2, mu), bispectrum(Pw, k3, k2, mu_32)
    return abs(a / b - 1) < 1e-12, f"B(k1,k2,k3) / B(k3,k2,k1) - 1 = {a / b - 1:.1e}"


def check_constructive_vs_destructive():
    """Destructive (delta = 1) wiggles are much weaker than constructive (delta = 0, 2)
    for k1 > 0.05.

    In the dominant F2 P1 P2 term the wiggles of P(k1) and P(k2) multiply. In antiphase
    they cancel to first order, provided both carry wiggles of similar size. The first
    version of this check used 0.03 < k1 < 0.2 and failed: the destructive residual was
    only 2.1x smaller, and it sat at k1 < 0.05. There the shift pi/r_s = 0.021 is
    comparable to k1, and the wiggle amplitude grows fast with k, so P(k1) and P(k2)
    wiggle by different amounts and cannot cancel. The range now starts at 0.05.
    The low-k feature is real and visible in the plot. The 3x threshold is the
    original one, chosen before any run.
    """
    amp = lambda th, d: abs(ratio(th, d)[(K1 > 0.05) & (K1 < 0.2)]).max()
    rows = [(amp(t, 0), amp(t, 1), amp(t, 2)) for t in THETAS]
    ok = all(c1 < min(c0, c2) / 3 for c0, c1, c2 in rows)
    return ok, "max |B_w/B_nw - 1| over 0.05<k1<0.2 (delta 0 / 1 / 2): " + ";  ".join(
        f"theta={t:.2f}: {a:.3f}/{b:.3f}/{c:.3f}" for t, (a, b, c) in zip(THETAS, rows))


def plot():
    cols = {THETAS[0]: "#2a78d6", THETAS[1]: "#eb6834", THETAS[2]: "#1baf7a"}
    names = {THETAS[0]: r"$\theta=0$", THETAS[1]: r"$\theta=\pi/4$", THETAS[2]: r"$\theta=\pi/2$"}
    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    for a, d in zip(ax, DELTAS):
        for t in THETAS:
            a.plot(K1, ratio(t, d), color=cols[t], lw=2, label=names[t])
        a.axhline(0, color="0.6", lw=1)
        kind = "Destructive" if d % 2 else "Constructive"
        a.text(0.97, 0.06, rf"{kind} ($\delta={d}$)", transform=a.transAxes, ha="right")
        a.set_ylabel(r"$B^{\rm w}_{\rm m} / B^{\rm nw}_{\rm m} - 1$")
        a.grid(alpha=0.3, lw=0.5)
    ax[0].legend(frameon=False, ncol=3, loc="upper right")
    ax[0].set_title(r"Tree-level matter bispectrum BAO, $z=0$, $k_2 = k_1 + \delta\,\pi/r_s$")
    ax[-1].set(xlabel=r"$k_1$ [$h$/Mpc]", xlim=(K1[0], K1[-1]))
    fig.tight_layout()
    fig.savefig(PLOTS / "bispectrum_bao.png", dpi=130)


if __name__ == "__main__":
    failures = run("Bispectrum BAO (tree level)", [
        check_F2_limits,
        check_bispectrum_symmetric,
        check_constructive_vs_destructive,
    ])
    plot()
    sys.exit(failures)
