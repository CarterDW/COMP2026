"""Step 4: lognormal mocks, the first non-Gaussian field.

    python checks/step4_lognormal.py     (about 3 minutes the first time, then cached)

delta = exp(g - sigma_G^2/2) - 1, with g Gaussian with power P_G, built so the
lognormal power equals the EH98 target P. The one-point checks use the exact
lognormal formulas. The two-point checks ask whether the power is right (it
should be) and whether the covariance is Gaussian (it should not be).

Pool: 3000 mocks, seeds 20000 and up. The first 2000 build C_hat and the last
1000 are held out.
"""

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cosmo
import field
import mocks
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
P = lambda k: cosmo.P_model(k, mocks.OM, mocks.S8)
N_POOL, N_TRAIN = 3000, 2000
MU = field.bin_model(P)
C_GAUSS = np.diag(field.gaussian_sigma(P)**2)
p = len(MU)


@functools.lru_cache(maxsize=None)
def pool():
    X, stats = mocks.lognormal_pool(N_POOL)
    return X[:N_TRAIN], X[N_TRAIN:], stats


@functools.lru_cache(maxsize=None)
def PG_info():
    return field.lognormal_PG(P)


def sigma_G2():
    return PG_info()[0].sum() / field.V


def check_PG_is_valid():
    """The Gaussian power P_G = FFT[ln(1 + xi)] is a valid spectrum, with no clipping.

    If any P_G < 0, it would have to be clipped and the lognormal power would
    miss its target. The dropped k = 0 mode rescales the power by exp(c), so |c|
    has to be negligible too. The detail also confirms xi_G(0) = ln(1 + xi(0)),
    the map evaluated at zero separation.
    """
    PG, c, info = PG_info()
    xi0 = field.xi_grid(field.P_grid(P)).ravel()[0]
    ok = info["n_modes"] == 0 and abs(c) < 1e-4
    return ok, (f"negative P_G modes: {info['n_modes']};  c = {c:.1e};  "
                f"sigma_G^2 = {sigma_G2():.5f} vs ln(1 + xi(0)) = {np.log1p(xi0):.5f}")


def check_density_positive():
    """delta >= -1 in every cell of every mock (the Gaussian field had 22% below)."""
    *_, stats = pool()
    return stats[:, 0].min() > -1, f"min delta over {N_POOL} mocks x {field.NC} cells: {stats[:, 0].min():.5f}"


def check_one_point_lognormal():
    """The one-point statistics match the exact lognormal formulas.

    var(delta) = e^{sigma_G^2} - 1 (the target variance, 1.64), var ln(1+delta) =
    sigma_G^2, and skew(delta) = (e^{s^2} + 2) sqrt(e^{s^2} - 1), about 6, where a
    Gaussian has 0. Error bars come from the scatter between mocks. The skewness
    estimator is a ratio, so it carries a small bias. Its tolerance is 1%, not a
    pull.
    """
    *_, s = pool()
    s2 = sigma_G2()
    tests = [("var delta", s[:, 1], np.expm1(s2)), ("var ln(1+d)", s[:, 3], s2)]
    pulls = [(x.mean() - w) / (x.std(ddof=1) / np.sqrt(len(x))) for _, x, w in tests]
    skew_want = (np.exp(s2) + 2) * np.sqrt(np.expm1(s2))
    skew = s[:, 4].mean()
    ok = max(map(abs, pulls)) < 4 and abs(skew / skew_want - 1) < 0.01
    detail = ", ".join(f"{n} {x.mean():.4f} vs {w:.4f} ({pl:+.1f} sig)" for (n, x, w), pl in zip(tests, pulls))
    return ok, f"{detail};  skew {skew:.3f} vs {skew_want:.3f} ({skew / skew_want - 1:+.2%})"


def check_mean_power_is_target():
    """The average lognormal P_hat equals the EH98 target, BAO included.

    The bins are now correlated, so the test is chi^2 with the full covariance
    of the mean, C_hat / N, Hartlap-corrected: about 25 +/- 7. This is the point
    of the ln(1 + xi) construction. Without it, exponentiating a field with power
    P gives a field with more power than P.
    """
    train, test, _ = pool()
    X = np.vstack([train, test])
    r = X.mean(0) - MU
    Cinv = np.linalg.inv(np.cov(X.T) / len(X)) * (len(X) - p - 2) / (len(X) - 1)
    c2 = r @ Cinv @ r
    worst = abs(r / MU).max()
    return c2 < p + 4 * np.sqrt(2 * p), f"chi^2 = {c2:.1f} for {p} bins;  max |mean/target - 1| = {worst:.1e}"


def check_variance_exceeds_gaussian():
    """C_ii is larger than the Gaussian 2P^2/N_k, increasingly so at high k.

    The Gaussian part falls as 1/N_k ~ 1/k^2, while the extra (trispectrum) part
    from mode coupling does not shrink with more modes. So the ratio should be
    near 1 at the lowest k and grow steadily. Each ratio has about 3% noise
    (2000 mocks).
    """
    train, _, _ = pool()
    r = np.diag(np.cov(train.T)) / np.diag(C_GAUSS)
    s = np.sqrt(2 / (N_TRAIN - 1))
    rising = np.corrcoef(np.arange(p), r)[0, 1]
    ok = (r[-1] - 1) / (r[-1] * s) > 10 and abs(r[0] - 1) < 0.15 and rising > 0.9
    return ok, f"C_ii / (2P^2/N_k): lowest bin {r[0]:.2f}, k~0.2 {r[p // 2]:.2f}, top bin {r[-1]:.2f};  corr with bin index {rising:.3f}"


def check_bins_correlate():
    """High-k bins are positively correlated, where the Gaussian field gave pure noise.

    Noise on a single r_ij is about 1/sqrt(2000) = 0.022. Step 3 found
    sqrt(N) r_ij ~ N(0, 1).
    """
    train, _, _ = pool()
    R = np.corrcoef(train.T)
    hi = np.nonzero(field.bin_model(lambda k: k) > 0.25)[0]
    sub = R[np.ix_(hi, hi)][np.triu_indices(len(hi), 1)]
    z = sub.mean() * np.sqrt(N_TRAIN)
    return z > 10, f"mean r_ij over {len(sub)} pairs with k > 0.25: {sub.mean():.2f} ({z:.0f} x the noise)"


def check_gaussian_C_understates_errors():
    """Held-out lognormal mocks have <chi^2> >> p under the Gaussian C, and about p
    under the Hartlap-corrected lognormal C_hat.

    For the Gaussian C the expectation is tr(C_G^-1 C_LN) = sum of the variance
    ratios, so the Gaussian assumption does not just shift chi^2. It predicts by
    how much. This step's answer to the project question, at the chi^2 level.
    """
    train, test, _ = pool()
    rt = test - MU
    c2_G = np.einsum("ni,ij,nj->n", rt, np.linalg.inv(C_GAUSS), rt)
    want_G = (np.diag(np.cov(train.T)) / np.diag(C_GAUSS)).sum()
    Cinv = np.linalg.inv(np.cov(train.T)) * (N_TRAIN - p - 2) / (N_TRAIN - 1)
    c2_LN = np.einsum("ni,ij,nj->n", rt, Cinv, rt)
    e_G, e_LN = c2_G.std(ddof=1) / np.sqrt(len(rt)), c2_LN.std(ddof=1) / np.sqrt(len(rt))
    ok = abs(c2_G.mean() - want_G) / e_G < 4 and abs(c2_LN.mean() - p) / e_LN < 4
    return ok, (f"Gaussian C: <chi^2> = {c2_G.mean():.1f} +/- {e_G:.1f} (predicted {want_G:.1f});  "
                f"lognormal C_hat: {c2_LN.mean():.2f} +/- {e_LN:.2f} (p = {p})")


def plot():
    train, _, _ = pool()
    k = field.bin_model(lambda kk: kk)
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    ax = ax.ravel()

    a = ax[0]
    g = field.make_gaussian(P, seed=0)[:, :, 0]
    d = field.make_lognormal(PG_info()[0], seed=0)[:, :, 0]
    bins = np.linspace(-4, 8, 121)
    a.hist(g.ravel(), bins, density=True, histtype="step", color=orange, lw=1.5, label="Gaussian")
    a.hist(d.ravel(), bins, density=True, histtype="step", color=blue, lw=1.5, label="lognormal")
    a.axvline(-1, color="0.4", lw=1, ls=":")
    a.set(yscale="log", xlabel=r"$\delta$ per cell", ylabel="density",
          title="One-point PDF, same target $P(k)$, one slice each")

    a = ax[1]
    im = a.imshow(1 + d.T, origin="lower", extent=(0, field.L, 0, field.L), cmap="magma",
                  norm=matplotlib.colors.LogNorm(vmin=0.05, vmax=20))
    fig.colorbar(im, ax=a, label=r"$1 + \delta$")
    a.set(xlabel=r"$x$ [Mpc/$h$]", ylabel=r"$y$ [Mpc/$h$]", title="Lognormal slice (log colour scale)")

    a = ax[2]
    im = a.imshow(np.corrcoef(train.T), cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                  extent=(field.EDGES[0], field.EDGES[-1]) * 2)
    fig.colorbar(im, ax=a, label=r"$r_{ij}$")
    a.set(xlabel=r"$k_i$ [$h$/Mpc]", ylabel=r"$k_j$ [$h$/Mpc]",
          title=rf"Correlation matrix, {N_TRAIN} lognormal mocks")

    a = ax[3]
    r = np.diag(np.cov(train.T)) / np.diag(C_GAUSS)
    a.axhline(1, color="0.4", lw=1, label=r"Gaussian $2P^2/N_k$")
    a.plot(k, r, "o-", color=blue, ms=4, lw=1.5, label="lognormal mocks")
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$C_{ii} \,/\, (2P^2/N_k)$",
          title="Variance per bin relative to Gaussian")

    for a in (ax[0], ax[3]):
        a.legend(frameon=False)
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step4_lognormal.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 4: lognormal mocks", [
        check_PG_is_valid,
        check_density_positive,
        check_one_point_lognormal,
        check_mean_power_is_target,
        check_variance_exceeds_gaussian,
        check_bins_correlate,
        check_gaussian_C_understates_errors,
    ])
    plot()
    sys.exit(failures)
