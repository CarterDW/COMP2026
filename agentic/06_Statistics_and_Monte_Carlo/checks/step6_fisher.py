"""Step 6: the Fisher forecast, and predicting step 5's coverage without mocks.

    python checks/step6_fisher.py      (about 1 minute the first time, then cached)

For a likelihood with precision Psi and a model that is linear near the truth,
mu(theta) ~ mu0 + J (theta - theta0):

  * the error bars you report are the posterior covariance R = F^-1, where
    F = J^T Psi J;
  * the best fit actually scatters, from one universe to the next, with the
    "sandwich" covariance
        S = F^-1 J^T Psi C_true Psi J F^-1.

When Psi = C_true^-1, S = R and the error bars are honest. Otherwise the truth
lands inside the reported 68% region with probability P(x^T R^-1 x < 2.30), for
x ~ N(0, S). That number can be computed from J, Psi and C_true alone, with no
fits. If it matches step 5's measured coverage, the undercoverage is fully
explained by the covariance and not by anything else in the pipeline.

C_true is the plain sample covariance of the 2000 training lognormal mocks. The
coverage test used the 1000 held-out mocks, so the prediction and the
measurement share no mocks.
"""

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import inference as I
from checks import step5_mcmc as S5
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES, COLORS, LABELS = S5.NAMES, S5.COLORS, S5.LABELS
p, n_theta = I.p, 2


def jacobian(h):
    return np.array([(I.model(I.TRUTH + h * e) - I.model(I.TRUTH - h * e)) / (2 * h)
                     for e in np.eye(2)]).T


J = jacobian(1e-3)
C_TRUE = np.cov(I.data()[0].T)


def blocks(nm):
    return range(40) if nm in "BC" else [0]


def reported_and_actual(Psi):
    F = J.T @ Psi @ J
    R = np.linalg.inv(F)
    return R, R @ J.T @ Psi @ C_TRUE @ Psi @ J @ R


def predicted_coverage(R, S, level_chi2, n=400_000, seed=0):
    """P(x^T R^-1 x < level) for x ~ N(0, S), by Monte Carlo."""
    x = np.random.default_rng(seed).multivariate_normal(np.zeros(2), S, n)
    return (np.einsum("ni,ij,nj->n", x, np.linalg.inv(R), x) < level_chi2).mean()


@functools.lru_cache(maxsize=None)
def predictions():
    """Predicted 68% and 95% coverage per likelihood, averaged over the same C_hat
    blocks step 5 used, so block-to-block luck is shared and cancels."""
    out = {}
    for nm in NAMES:
        c = np.array([[predicted_coverage(*reported_and_actual(I.precision(nm, b)), lv)
                       for lv in (2.2958, 5.9915)] for b in blocks(nm)])
        out[nm] = c.mean(0)
    return out


@functools.lru_cache(maxsize=None)
def best_fits(nm):
    """Grid maximum-posterior point for every held-out mock under likelihood nm. Cached."""
    cache = ROOT / "results" / f"step6_bestfit_{nm}.npy"
    if cache.exists():
        return np.load(cache)
    _, test = I.data()
    g = np.array(np.meshgrid(I.GRID_OM, I.GRID_S8, indexing="ij")).reshape(2, -1).T
    out = np.zeros((len(test), 2))
    for b in blocks(nm):
        idx = np.arange(len(test)) if nm in "AD" else np.arange(b, len(test), 40)
        Psi = I.precision(nm, b)
        for chunk in np.array_split(idx, max(1, len(idx) // 100)):
            out[chunk] = g[I.grid_chi2(test[chunk], Psi).argmin(1)]
    np.save(cache, out)
    return out


def check_derivatives_converged():
    """Central-difference J is converged, and its error falls as h^2.

    Errors are measured against each column's largest entry, not element by
    element. At fixed s8, changing Om pivots P(k) about k ~ 0.13 (see the step 1
    plot), so dmu/dOm passes through zero in one bin, where a relative error is
    meaningless. A first version of this check failed on exactly that bin.

    The sharper test is the scaling. Going h = 1e-3 -> 3e-4 -> 1e-4 should shrink
    the change by (10/3)^2 = 11 each time, as in the Numerical Differentiation
    week. The s8 column is exact to roundoff because P is exactly quadratic in
    s8, so a central difference has no truncation error there.
    """
    J2, J3 = jacobian(3e-4), jacobian(1e-4)
    scale = abs(J).max(0)
    d1, d2 = abs(J2 - J).max(0) / scale, abs(J3 - J2).max(0) / scale
    ratio = d1[0] / d2[0]
    ok = d1.max() < 1e-4 and abs(ratio / (10 / 3)**2 - 1) < 0.2 and d1[1] < 1e-10
    return ok, (f"change h=1e-3 -> 3e-4: Om {d1[0]:.1e}, s8 {d1[1]:.1e};  "
                f"Om shrinks x{ratio:.1f} at the next step (h^2 predicts {(10 / 3)**2:.1f})")


def check_fisher_matches_posterior():
    """F^-1 matches the grid posterior covariance on the showcase mock, for all four likelihoods.

    The posterior is Gaussian (step 5 found the 68% threshold at 2.28), so the
    Fisher matrix should give its shape, not just its size: sigmas within 5%,
    correlation within 0.05.
    """
    worst_s, worst_r, parts = 0.0, 0.0, []
    for nm in NAMES:
        R = np.linalg.inv(J.T @ I.precision(nm, 0) @ J)
        Cg = S5.grid_posterior(nm)[1]
        sr = np.sqrt(np.diag(R) / np.diag(Cg))
        rho = lambda M: M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])
        worst_s = max(worst_s, abs(sr - 1).max())
        worst_r = max(worst_r, abs(rho(R) - rho(Cg)))
        parts.append(f"{nm} rho {rho(R):+.2f}/{rho(Cg):+.2f}")
    return worst_s < 0.05 and worst_r < 0.05, \
        f"worst sigma mismatch {worst_s:.1%};  Fisher/grid correlation: " + ", ".join(parts)


def check_sandwich_matches_best_fit_scatter():
    """The sandwich S predicts the actual scatter of best fits over the 1000 held-out mocks.

    This links the Fisher algebra to the measured fits. Each sigma has about
    1/sqrt(2000) = 2.2% sampling noise. For B and C, S is averaged over the 40
    blocks, since each mock was fitted with its own block.
    """
    worst, parts = 0.0, []
    for nm in NAMES:
        S = np.mean([reported_and_actual(I.precision(nm, b))[1] for b in blocks(nm)], 0)
        bf = best_fits(nm)
        emp = np.sqrt(((bf - I.TRUTH)**2).mean(0))      # scatter about the truth, pooled over blocks
        r = emp / np.sqrt(np.diag(S))
        worst = max(worst, abs(r - 1).max())
        parts.append(f"{nm} {r[0]:.2f},{r[1]:.2f}")
    return worst < 0.10, "empirical / predicted best-fit sigma (Om, s8): " + ";  ".join(parts)


def check_coverage_predicted():
    """Fisher-predicted coverage matches step 5's measurement for every likelihood.

    The prediction uses no data mocks at all. The binomial error on 1000 mocks is
    1.5% at 68% and 0.7% at 95%. B and C share their C_hat blocks with step 5, so
    the same error applies.
    """
    pr, cs = predictions(), S5.coverage_stats()
    n = len(cs["D"]["in68"])
    worst, parts = 0.0, []
    for nm in NAMES:
        m = np.array([cs[nm]["in68"].mean(), cs[nm]["in95"].mean()])
        err = np.sqrt(pr[nm] * (1 - pr[nm]) / n)
        worst = max(worst, abs((m - pr[nm]) / err).max())
        parts.append(f"{nm} {pr[nm][0]:.1%}/{m[0]:.1%}")
    return worst < 3.5, f"68% predicted/measured: " + ";  ".join(parts) + f"  (worst {worst:.1f} sig)"


def check_dodelson_schneider():
    """With a Hartlap-corrected 50-mock C_hat, the best fit still scatters more than
    the optimal one, by Dodelson & Schneider's factor 1 + B (p - n_theta).

    Here B = (N_s - p - 2)/((N_s - p - 1)(N_s - p - 4)). Hartlap fixes the size of
    the reported error bars, but a noisy C_hat weights the bins wrongly, so the
    estimator itself is worse. That is why likelihood C still undercovers. The
    ratio compares S for each 50-mock block to the optimal F^-1 with C_true.
    """
    Ns = I.NS_SMALL
    B = (Ns - p - 2) / ((Ns - p - 1) * (Ns - p - 4))
    want = 1 + B * (p - n_theta)
    R_opt = np.linalg.inv(J.T @ np.linalg.inv(C_TRUE) @ J)
    ratios = np.array([np.diag(reported_and_actual(I.precision("C", b))[1]) / np.diag(R_opt)
                       for b in range(40)])
    m, e = ratios.mean(0), ratios.std(0, ddof=1) / np.sqrt(40)
    ok = (abs(m - want) / e < 4).all()
    return ok, f"Var(best fit) / optimal (Om, s8): {m[0]:.2f}+/-{e[0]:.2f}, {m[1]:.2f}+/-{e[1]:.2f};  DS predicts {want:.2f}"


def plot():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    OM, S8 = np.meshgrid(I.GRID_OM, I.GRID_S8, indexing="ij")
    t = np.linspace(0, 2 * np.pi, 200)

    a = ax[0]
    for nm in "AD":
        c2 = S5.grid_posterior(nm)[2].reshape(OM.shape)
        a.contour(OM, S8, c2 - c2.min(), levels=[2.30], colors=COLORS[nm], linewidths=2)
        m = S5.grid_posterior(nm)[0]
        R = np.linalg.inv(J.T @ I.precision(nm, 0) @ J)
        e = m[:, None] + np.linalg.cholesky(R) @ np.array([np.cos(t), np.sin(t)]) * np.sqrt(2.30)
        a.plot(*e, color="k", lw=1, ls="--")
        a.plot([], [], color=COLORS[nm], lw=2, label=LABELS[nm] + " (grid)")
    a.plot([], [], color="k", lw=1, ls="--", label="Fisher ellipse")
    a.set(xlabel=r"$\Omega_m$", ylabel=r"$\sigma_8$", xlim=(0.300, 0.316), ylim=(0.800, 0.818),
          title="Showcase mock: Fisher vs exact 68% contour")
    a.legend(frameon=False, fontsize=8)

    a = ax[1]
    x = np.arange(2)
    for i, nm in enumerate(NAMES):
        Rs, Ss = zip(*[reported_and_actual(I.precision(nm, b)) for b in blocks(nm)])
        rep, act = np.sqrt(np.diag(np.mean(Rs, 0))), np.sqrt(np.diag(np.mean(Ss, 0)))
        a.bar(x + (i - 1.5) * 0.2, act / rep, 0.18, color=COLORS[nm], label=LABELS[nm],
              edgecolor="white", lw=2)
    a.axhline(1, color="0.3", lw=1, ls=":")
    a.set_xticks(x, [r"$\Omega_m$", r"$\sigma_8$"])
    a.set(ylabel="actual scatter / reported error", title="How overconfident each likelihood is")
    a.legend(frameon=False, fontsize=8)

    a = ax[2]
    pr, cs = predictions(), S5.coverage_stats()
    a.plot([0, 1], [0, 1], color="0.3", lw=1, ls=":")
    for nm in NAMES:
        m = [cs[nm]["in68"].mean(), cs[nm]["in95"].mean()]
        a.plot(pr[nm], m, "o", color=COLORS[nm], ms=9, label=LABELS[nm])
    a.set(xlabel="Fisher-predicted coverage (no mocks)", ylabel="measured coverage (step 5)",
          xlim=(0, 1), ylim=(0, 1), title="68% and 95% levels, every likelihood")
    a.legend(frameon=False, fontsize=8, loc="upper left")

    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step6_fisher.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 6: Fisher forecast vs MCMC", [
        check_derivatives_converged,
        check_fisher_matches_posterior,
        check_sandwich_matches_best_fit_scatter,
        check_coverage_predicted,
        check_dodelson_schneider,
    ])
    plot()
    sys.exit(failures)
