"""Branch a: the same lognormal universe at z = 0, 1, 3, 10, 100 and 1100.

    python checks/branch_a_redshift.py   (about 13 minutes the first time, for the pools)

The linear field scales as D(z), so s8(z) = 0.81 D(z). A cell's sigma_delta
drops from 1.3 today to about 2e-3 at recombination. The lognormal non-Gaussianity
should go away with it: skewness ~ 3 sigma, and the covariance excess (a
trispectrum term) ~ P relative to the Gaussian 2P^2/N_k, so ~ D^2. The Gaussian
covariance should then become honest.

Every redshift reuses the same seeds, so these are one set of initial
conditions at six epochs.

Fitting at each z: the data are divided by f^2, with f = s8(z)/0.81, so the step 5
model grid (fixed s8 = 0.81) applies unchanged. The fit parameter is then
s8 today, with D(z) treated as known. Since P scales exactly as s8^2, this is a
rescaling, not an approximation.

Caveat: EH98 describes the transfer function well after the drag epoch. At
z = 1100 baryons are still catching up to the CDM, so "today's shape times D^2"
is only approximate there. That changes the spectrum, not the statistics
question.
"""

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import field
import growth
import inference as I
import mocks
from checks import step6_fisher as S6
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
Z = np.array([0, 1, 3, 10, 100, 1100.0])
DZ = growth.growth(Z)
N_TRAIN = 2000
p = I.p


@functools.lru_cache(maxsize=None)
def epoch(i):
    """Scaled train/test P_hat, stats, and the A/D precision matrices at Z[i]."""
    s8z = mocks.S8 * DZ[i]
    X, stats = mocks.lognormal_pool(3000, s8=mocks.S8 if i == 0 else float(s8z))
    X = X / DZ[i]**2                        # f^2 = (s8(z)/0.81)^2 = D^2
    train, test = X[:N_TRAIN], X[N_TRAIN:]
    C = np.cov(train.T)
    Psi = {"A": I.precision("A"), "D": np.linalg.inv(C) * I.hartlap(N_TRAIN)}
    return dict(train=train, test=test, stats=stats, C=C, Psi=Psi, s8z=s8z)


@functools.lru_cache(maxsize=None)
def coverage(i, nm):
    """Measured 68% coverage over the 1000 held-out mocks, via the grid posterior."""
    e = epoch(i)
    mu0 = I.model(I.TRUTH)
    inside = []
    for chunk in np.array_split(np.arange(len(e["test"])), 10):
        c2 = I.grid_chi2(e["test"][chunk], e["Psi"][nm])
        r = e["test"][chunk] - mu0
        c2t = np.einsum("ni,ij,nj->n", r, e["Psi"][nm], r)
        inside += [I.hpd_contains(row, t, 0.68) for row, t in zip(c2, c2t)]
    return np.mean(inside)


def predicted(i, nm):
    """Sandwich-predicted 68% coverage and s8 overconfidence factor (step 6's method)."""
    e = epoch(i)
    Psi = e["Psi"][nm]
    R = np.linalg.inv(S6.J.T @ Psi @ S6.J)
    S = R @ S6.J.T @ Psi @ e["C"] @ Psi @ S6.J @ R
    return S6.predicted_coverage(R, S, 2.2958), np.sqrt(S[1, 1] / R[1, 1])


def check_growth_limits():
    """D(z) reproduces three exact or standard results.

    * Matter only (Om = 1, no radiation): D = a exactly.
    * Matter + radiation, no Lambda: Meszaros's D = 1 + 3y/2, y = a/a_eq, exactly.
    * LCDM without radiation: D(0)/a_early = g(Om) from Carroll, Press & Turner
      (1992), a fit good to about 1%.
    """
    z = np.array([0.5, 5, 50, 1100.0])
    a = 1 / (1 + z)
    e1 = abs(growth.growth(z, Om=1.0, radiation=False) / a - 1).max()
    Or = growth.OMEGA_R_H2 / I.cosmo.H**2
    Om = 1 - Or
    y, y1 = a * Om / Or, Om / Or
    e2 = abs(growth.growth(z, Om=Om) / ((1 + 1.5 * y) / (1 + 1.5 * y1)) - 1).max()
    Om, OL = 0.31, 0.69
    g_cpt = 2.5 * Om / (Om**(4 / 7) - OL + (1 + Om / 2) * (1 + OL / 70))
    g = a[-1] / growth.growth([1100.0], Om=Om, radiation=False)[0]
    ok = e1 < 1e-6 and e2 < 1e-6 and abs(g / g_cpt - 1) < 0.015
    return ok, f"matter-only {e1:.1e};  Meszaros {e2:.1e};  g(Om) = {g:.4f} vs CPT {g_cpt:.4f} ({g / g_cpt - 1:+.2%})"


def check_one_point_by_epoch():
    """At every epoch the cell variance and skewness match the lognormal formulas,
    and the skewness falls toward its small-sigma limit, 3 sigma."""
    parts, ok = [], True
    for i, z in enumerate(Z):
        s = epoch(i)["stats"]
        s2 = s[:, 3].mean()                          # var ln(1 + delta) = sigma_G^2
        want = (np.exp(s2) + 2) * np.sqrt(np.expm1(s2))
        skew = s[:, 4].mean()
        ok &= abs(skew / want - 1) < 0.02
        parts.append(f"z={z:g}: sigma {np.sqrt(s[:, 1].mean()):.2g}, skew {skew:.3g}")
    return ok, ";  ".join(parts)


def check_covariance_excess_scales_as_D2():
    """The top-bin variance excess C_ii/(2P^2/N_k) - 1 falls as D^2.

    At leading order the non-Gaussian (trispectrum) part of the variance goes as
    P^3, while the Gaussian part goes as P^2, so their ratio is proportional to
    P, i.e. to D^2.

    The comparison is paired. Every epoch uses the same seeds, so every epoch
    shares the same ~3% sampling error in C_ii. A first version of this check
    compared raw excesses and failed because of it: z = 100 and z = 1100 both
    read -0.033, which is that shared noise. Subtracting the z = 1100 excess
    (non-Gaussian part ~1e-6) removes the shared noise, the common-random-numbers
    trick. The D^2 test uses z = 3 vs z = 10, where the field is perturbative.
    At z <= 1 (sigma_cell ~ 1) higher orders take over.
    """
    sg2 = field.gaussian_sigma(lambda k: I.cosmo.P_model(k, *I.TRUTH))**2
    raw = np.array([np.diag(epoch(i)["C"])[-1] / sg2[-1] - 1 for i in range(len(Z))])
    ex = raw - raw[-1]
    i3, i10 = 2, 3
    ratio = (ex[i3] / ex[i10]) / (DZ[i3] / DZ[i10])**2
    ok = abs(ratio - 1) < 0.25 and np.all(np.diff(ex[:5]) < 0) and abs(raw[-1]) < 4 * np.sqrt(2 / (N_TRAIN - 1))
    return ok, ("top-bin excess minus z=1100: " + ", ".join(f"z={z:g} {e:+.4f}" for z, e in zip(Z, ex))
                + f";  z=3/z=10 ratio over D^2 ratio: {ratio:.2f};  shared noise (raw z=1100) {raw[-1]:+.3f}")


def check_coverage_by_epoch():
    """D covers 68% at every epoch. A undercovers today and becomes honest by z = 10.

    Each fraction has about 1.5% binomial error, so A must reach 68% within 3 sigma
    for z >= 10, and every measured value must match its sandwich prediction
    within 3.5 sigma.
    """
    s = np.sqrt(0.68 * 0.32 / 1000)
    ok, parts = True, []
    for i, z in enumerate(Z):
        mA, mD = coverage(i, "A"), coverage(i, "D")
        pA, pD = predicted(i, "A")[0], predicted(i, "D")[0]
        ok &= abs(mD - 0.68) < 3 * s and abs(mA - pA) < 3.5 * s and abs(mD - pD) < 3.5 * s
        if z >= 10:
            ok &= abs(mA - 0.68) < 3 * s
        parts.append(f"z={z:g}: A {mA:.1%} (pred {pA:.1%}), D {mD:.1%}")
    return ok, ";  ".join(parts)


def plot():
    zp = Z + 1
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = ax[0]
    zz = np.logspace(0, np.log10(1101), 200) - 1
    a.loglog(zz + 1, growth.growth(zz), color=blue, lw=2, label="$D(z)$, with radiation")
    a.loglog(zz + 1, growth.growth(zz, radiation=False), color="0.5", lw=1, ls="--", label="no radiation")
    sig = [np.sqrt(epoch(i)["stats"][:, 1].mean()) for i in range(len(Z))]
    skew = [epoch(i)["stats"][:, 4].mean() for i in range(len(Z))]
    a.loglog(zp, sig, "o", color=orange, ms=6, label=r"$\sigma_\delta$ per cell")
    a.loglog(zp, skew, "s", color="#1baf7a", ms=6, label="skewness of $\\delta$")
    a.set(xlabel="$1+z$", ylabel="value", title="Growth and one-point non-Gaussianity")
    a.legend(frameon=False, fontsize=8)

    a = ax[1]
    sg2 = field.gaussian_sigma(lambda k: I.cosmo.P_model(k, *I.TRUTH))**2
    k = field.bin_model(lambda kk: kk)
    for i, c in zip(range(len(Z)), plt.cm.viridis(np.linspace(0, 0.9, len(Z)))):
        a.plot(k, np.diag(epoch(i)["C"]) / sg2, color=c, lw=2, label=f"z = {Z[i]:g}")
    a.axhline(1, color="0.3", lw=1, ls=":")
    a.set(yscale="log", xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$C_{ii} / (2P^2/N_k)$",
          title="Variance relative to Gaussian, by epoch")
    a.legend(frameon=False, fontsize=8)

    a = ax[2]
    a.axhspan(0.68 - 0.03, 0.68 + 0.03, color="0.9", lw=0)
    a.axhline(0.68, color="0.3", lw=1, ls=":")
    for nm, c, lab in (("A", orange, "A: Gaussian $C$"), ("D", blue, r"D: mock $\hat C$ + Hartlap")):
        a.plot(zp, [coverage(i, nm) for i in range(len(Z))], "o-", color=c, ms=7, lw=2, label=lab)
        a.plot(zp, [predicted(i, nm)[0] for i in range(len(Z))], "x", color="k", ms=7)
    a.plot([], [], "x", color="k", label="Fisher prediction")
    a.set(xscale="log", xlabel="$1+z$", ylabel="68% coverage", ylim=(0, 0.8),
          title="Is the Gaussian error bar honest? By epoch")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    for a in ax:
        a.grid(alpha=0.3, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS / "branch_a_redshift.png", dpi=130)


if __name__ == "__main__":
    failures = run("Branch a: the lognormal universe by epoch", [
        check_growth_limits,
        check_one_point_by_epoch,
        check_covariance_excess_scales_as_D2,
        check_coverage_by_epoch,
    ])
    plot()
    sys.exit(failures)
