"""Step 3: the covariance of P_hat from Gaussian mocks, and the Hartlap bias.

    python checks/step3_covariance.py     (about 4 minutes the first time, then cached)

This step is the control. For a Gaussian field the true covariance is known,
C = diag(gaussian_sigma^2): different shells share no modes, so bins are
independent. Here that known C is compared against the sample covariance
C_hat estimated from N_s mocks.

The Hartlap result: if d is Gaussian with covariance C, independent of the N_s
mocks behind C_hat, then

    < (d - mu)^T C_hat^-1 (d - mu) > = p (N_s - 1) / (N_s - p - 2),

not p. C_hat^-1 is too large on average, because a noisy matrix has some
eigenvalues too small, and inverting blows them up. Multiplying C_hat^-1 by
(N_s - p - 2)/(N_s - 1) undoes this on average.

Mock pool: seeds 10000 and up (step 2 used 0-999), split into 2000 "training"
mocks for building C_hat and 1000 held-out "test" mocks.
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
from checks._runner import run

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLOTS = ROOT / "checks" / "plots"
CACHE = ROOT / "results" / "gauss_pool.npz"
P = lambda k: cosmo.P_model(k, 0.31, 0.81)
N_TRAIN, N_TEST, SEED0 = 2000, 1000, 10000
NS_LIST = (40, 50, 100, 500)


@functools.lru_cache(maxsize=None)
def pool():
    """(train, test) arrays of P_hat vectors, shape (mocks, bins). Cached on disk."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["train"], d["test"]
    Pg = field.P_grid(P)
    X = np.array([field.gaussian_P_hat(Pg, SEED0 + i) for i in range(N_TRAIN + N_TEST)])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, train=X[:N_TRAIN], test=X[N_TRAIN:])
    return X[:N_TRAIN], X[N_TRAIN:]


MU = field.bin_model(P)
C_TRUE = np.diag(field.gaussian_sigma(P)**2)
p = len(MU)


def chi2(d, Cinv):
    """chi^2 of each row of d about the true mean MU."""
    r = d - MU
    return np.einsum("ni,ij,nj->n", r, Cinv, r)


def hartlap(Ns):
    return (Ns - p - 2) / (Ns - 1)


@functools.lru_cache(maxsize=None)
def group_means(Ns):
    """Split the training mocks into disjoint groups of Ns. Each group gives a
    C_hat, and each C_hat gives a mean chi^2 over all test mocks.

    The groups are independent, so the scatter of the group results gives an
    honest error bar on their average.
    """
    train, test = pool()
    G = N_TRAIN // Ns
    return np.array([chi2(test, np.linalg.inv(np.cov(train[g * Ns:(g + 1) * Ns].T))).mean()
                     for g in range(G)])


def check_fast_path_matches_full():
    """gaussian_P_hat equals measure_P(make_gaussian(...)) for the same seed.

    The step 2 checks validated the full path. This shows the shortcut is the same
    computation and not an approximation.
    """
    full = field.measure_P(field.make_gaussian(P, seed=7))[1]
    fast = field.gaussian_P_hat(field.P_grid(P), seed=7)
    err = abs(fast / full - 1).max()
    return err < 1e-10, f"max relative difference over bins: {err:.1e}"


def check_true_C_gives_chi2_p():
    """With the true diagonal C, test-mock chi^2 has mean p and variance 2p.

    This is the baseline the Hartlap test needs: the P_hat vectors are Gaussian
    enough, and C is right, so any inflation later comes from C_hat alone.
    """
    _, test = pool()
    c = chi2(test, np.linalg.inv(C_TRUE))
    pm = (c.mean() - p) / np.sqrt(2 * p / N_TEST)
    rv = c.var(ddof=1) / (2 * p)
    ok = abs(pm) < 4 and abs(rv - 1) < 0.15
    return ok, f"<chi^2> = {c.mean():.2f} (p = {p}, {pm:+.1f} sig);  var/2p = {rv:.3f}"


def check_C_hat_diagonal():
    """The diagonal of C_hat from 2000 mocks matches gaussian_sigma^2.

    Each sample variance carries a relative error of sqrt(2/(N-1)) = 3.2%.
    """
    train, _ = pool()
    r = np.diag(np.cov(train.T)) / np.diag(C_TRUE)
    s = np.sqrt(2 / (N_TRAIN - 1))
    pulls = (r - 1) / s
    ok = abs(pulls).max() < 4 and abs(r.mean() - 1) < 4 * s / np.sqrt(p)
    return ok, f"C_hat_ii / C_ii in [{r.min():.3f}, {r.max():.3f}], mean {r.mean():.4f};  worst pull {pulls[abs(pulls).argmax()]:+.1f}"


def check_bins_are_uncorrelated():
    """Off-diagonal correlation coefficients are pure noise, with r_ij ~ N(0, 1/sqrt(N)).

    For truly independent bins, sqrt(N) r_ij over the 300 pairs should have
    mean 0 and standard deviation 1. A shared or mis-assigned mode would show up
    as a positive correlation between neighbouring bins.
    """
    train, _ = pool()
    R = np.corrcoef(train.T)
    off = R[np.triu_indices(p, 1)] * np.sqrt(N_TRAIN)
    adj = np.diag(R, 1) * np.sqrt(N_TRAIN)
    n = len(off)
    ok = abs(off.mean()) < 4 / np.sqrt(n) and abs(off.std() - 1) < 4 / np.sqrt(2 * n) \
        and abs(adj.mean()) < 4 / np.sqrt(p - 1)
    return ok, f"sqrt(N) r_ij over {n} pairs: mean {off.mean():+.3f}, std {off.std():.3f};  neighbours mean {adj.mean():+.2f}"


def check_hartlap_inflation():
    """With C_hat^-1, <chi^2> = p (N_s - 1)/(N_s - p - 2), not p.

    For N_s = 40 that is 3x too big, and for N_s = 500 it is 5.5% too big. The
    error bar is the scatter between independent groups / sqrt(groups).

    N_s = 30 was in the plan but is dropped. The variance of an inverse-Wishart
    matrix is finite only for N_s > p + 4 = 29, and at N_s = 30 the group
    averages are dominated by rare, nearly singular C_hat.
    """
    worst, parts = 0.0, []
    for Ns in NS_LIST:
        m = group_means(Ns)
        want = p / hartlap(Ns)
        err = m.std(ddof=1) / np.sqrt(len(m))
        pull = (m.mean() - want) / err
        worst = max(worst, abs(pull))
        parts.append(f"N_s={Ns}: {m.mean():.1f} vs {want:.1f} ({pull:+.1f})")
    return worst < 4, "; ".join(parts)


def check_hartlap_correction():
    """Multiplying C_hat^-1 by the Hartlap factor brings <chi^2> back to p.

    The uncorrected value has to be excluded from p with significance, or the
    test would pass for the wrong reason.
    """
    worst, parts, excluded = 0.0, [], True
    for Ns in NS_LIST:
        m = group_means(Ns)
        mh, err = hartlap(Ns) * m.mean(), hartlap(Ns) * m.std(ddof=1) / np.sqrt(len(m))
        worst = max(worst, abs(mh - p) / err)
        excluded &= (m.mean() - p) / (m.std(ddof=1) / np.sqrt(len(m))) > 4
        parts.append(f"N_s={Ns}: {mh:.2f}+/-{err:.2f}")
    return worst < 4 and excluded, f"corrected <chi^2> (p = {p}): " + "; ".join(parts) + \
        f";  uncorrected excludes p at >4 sig for all N_s: {excluded}"


def plot():
    train, test = pool()
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = ax[0]
    im = a.imshow(np.corrcoef(train.T), cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                  extent=(field.EDGES[0], field.EDGES[-1]) * 2)
    fig.colorbar(im, ax=a, label=r"$r_{ij}$")
    a.set(xlabel=r"$k_i$ [$h$/Mpc]", ylabel=r"$k_j$ [$h$/Mpc]",
          title=rf"Correlation matrix, {N_TRAIN} Gaussian mocks")

    a = ax[1]
    Ns_curve = np.linspace(p + 3, 600, 400)
    a.plot(Ns_curve, (Ns_curve - 1) / (Ns_curve - p - 2), color="0.4", lw=1.5,
           label=r"prediction $(N_s-1)/(N_s-p-2)$")
    for Ns in NS_LIST:
        m = group_means(Ns)
        e = m.std(ddof=1) / np.sqrt(len(m))
        a.errorbar(Ns, m.mean() / p, e / p, fmt="o", color=orange, ms=6)
        a.errorbar(Ns, hartlap(Ns) * m.mean() / p, hartlap(Ns) * e / p, fmt="s", color=blue, ms=6)
    a.plot([], [], "o", color=orange, label=r"raw $\hat C^{-1}$")
    a.plot([], [], "s", color=blue, label=r"Hartlap-corrected")
    a.axhline(1, color="0.6", lw=1, ls=":")
    a.set(xscale="log", yscale="log", xlabel=r"mocks $N_s$", ylabel=r"$\langle\chi^2\rangle / p$",
          title=rf"Inverse-covariance bias, $p = {p}$ bins")

    a = ax[2]
    bins = np.linspace(0, 150, 61)
    a.hist(chi2(test, np.linalg.inv(C_TRUE)), bins, histtype="step", color="0.3", lw=1.5,
           density=True, label="true $C$")
    for Ns, c in ((100, blue), (40, orange)):
        Cinv = np.linalg.inv(np.cov(train[:Ns].T))
        a.hist(chi2(test, Cinv), bins, histtype="step", color=c, lw=1.5, density=True,
               label=rf"$\hat C$ from $N_s={Ns}$")
    a.axvline(p, color="0.6", lw=1, ls=":")
    a.set(xlabel=r"$\chi^2$ of a held-out mock", ylabel="density",
          title=r"$\chi^2$ distribution, one $\hat C$ each")

    for a in ax[1:]:
        a.legend(frameon=False, fontsize=9)
        a.grid(alpha=0.3, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS / "step3_covariance.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 3: Gaussian covariance and Hartlap", [
        check_fast_path_matches_full,
        check_true_C_gives_chi2_p,
        check_C_hat_diagonal,
        check_bins_are_uncorrelated,
        check_hartlap_inflation,
        check_hartlap_correction,
    ])
    plot()
    sys.exit(failures)
