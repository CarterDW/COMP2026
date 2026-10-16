"""Step 5: Metropolis over (Om, s8) on lognormal data, and coverage.

    python checks/step5_mcmc.py      (about 3 minutes the first time, then mostly cached)

Two parts:
  * one showcase data mock (held-out lognormal mock 0), sampled by MCMC under all
    four likelihoods A-D (see inference.py) and checked against the exact grid
    posterior;
  * coverage over all 1000 held-out mocks, using the grid posterior. MCMC on
    4000 fits would take hours, and the grid gives the same posterior with no
    sampling noise. The showcase checks establish that the two agree.

Coverage is the project's answer. A likelihood is honest if the truth falls
inside its 68% credible region 68% of the time.
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
import inference as I
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
NAMES = "ABCD"
LABELS = {"A": "A: Gaussian $C$", "B": r"B: $\hat C$, $N_s=50$, raw",
          "C": r"C: $\hat C$, $N_s=50$, Hartlap", "D": r"D: $\hat C$, $N_s=2000$, Hartlap"}
COLORS = {"A": "#eb6834", "B": "#e87ba4", "C": "#eda100", "D": "#2a78d6"}
BURN, N_STEP = 1000, 6000


@functools.lru_cache(maxsize=None)
def showcase():
    """MCMC chains (burn-in removed) and acceptance for each likelihood on test mock 0."""
    d = I.data()[1][0]
    out = {}
    for j, nm in enumerate(NAMES):
        ch, acc = I.run_chains(d, I.precision(nm, 0), n=N_STEP, seed=j + 1)
        out[nm] = (ch, acc)
    return out


@functools.lru_cache(maxsize=None)
def grid_posterior(nm):
    """Grid posterior mean and covariance for the showcase mock."""
    d = I.data()[1][0]
    c2 = I.grid_chi2(d[None], I.precision(nm, 0))[0]
    w = np.exp(-0.5 * (c2 - c2.min()))
    w /= w.sum()
    g = np.array(np.meshgrid(I.GRID_OM, I.GRID_S8, indexing="ij")).reshape(2, -1).T
    m = w @ g
    return m, (g - m).T @ ((g - m) * w[:, None]), c2


@functools.lru_cache(maxsize=None)
def coverage_stats():
    """For every held-out mock and likelihood: delta chi^2 of the truth, and whether
    it lies inside the 68% and 95% HPD regions.

    B and C give mock j its own 50-mock C_hat (block j mod 40). A real analysis
    does not get to choose a lucky C_hat.
    """
    _, test = I.data()
    mu0 = I.model(I.TRUTH)
    res = {nm: dict(dchi2=[], in68=[], in95=[]) for nm in NAMES}
    for nm in NAMES:
        blocks = range(40) if nm in "BC" else [0]
        for b in blocks:
            idx = np.arange(len(test)) if nm in "AD" else np.arange(b, len(test), 40)
            Psi = I.precision(nm, b)
            for chunk in np.array_split(idx, max(1, len(idx) // 100)):
                c2 = I.grid_chi2(test[chunk], Psi)
                r = test[chunk] - mu0
                c2t = np.einsum("ni,ij,nj->n", r, Psi, r)
                for row, t in zip(c2, c2t):
                    res[nm]["dchi2"].append(t - row.min())
                    res[nm]["in68"].append(I.hpd_contains(row, t, 0.68))
                    res[nm]["in95"].append(I.hpd_contains(row, t, 0.95))
    return {nm: {k: np.array(v) for k, v in r.items()} for nm, r in res.items()}


def check_fast_binning():
    """bin_model_fast (3377 distinct |k|) equals bin_model (2 million modes)."""
    P = lambda k: cosmo.P_model(k, 0.29, 0.85)
    err = abs(field.bin_model_fast(P) / field.bin_model(P) - 1).max()
    return err < 1e-10, f"max relative difference: {err:.1e}"


def check_grid_resolves_posterior():
    """On the showcase mock, the grid's 68% HPD threshold sits at chi^2_min + 2.30.

    For a 2D Gaussian posterior, the 68% region is delta chi^2 < 2.30 exactly. If
    the 1e-4 grid were too coarse, or the window cut off the posterior, the
    threshold found by summing grid mass would drift away from 2.30. Likelihood A
    has the narrowest posterior and is the hardest case.
    """
    parts, worst = [], 0.0
    for nm in NAMES:
        c = np.sort(grid_posterior(nm)[2])
        w = np.exp(-0.5 * (c - c[0]))
        thr = c[np.searchsorted(np.cumsum(w) / w.sum(), 0.68)] - c[0]
        worst = max(worst, abs(thr - 2.30))
        parts.append(f"{nm} {thr:.2f}")
    return worst < 0.1, "68% threshold delta chi^2 (expect 2.30): " + ", ".join(parts)


def check_chains_converged():
    """Gelman-Rubin R_hat < 1.01 for both parameters under every likelihood.

    The 8 chains start 3 Fisher sigma from the truth in random directions, so
    agreement between them means each has forgotten where it started.
    """
    parts, worst = [], 0.0
    for nm, (ch, acc) in showcase().items():
        R = I.gelman_rubin(ch[:, BURN:]).max()
        worst = max(worst, R)
        parts.append(f"{nm} {R:.4f} (acc {acc.mean():.2f})")
    return worst < 1.01, "max R_hat: " + ", ".join(parts)


def check_mcmc_matches_grid():
    """MCMC posterior mean and width agree with the exact grid posterior.

    The mean is allowed 4 x its Monte Carlo error sigma/sqrt(N_eff), and the
    width 5%. This is what licenses using the grid for the 1000-mock coverage
    test.
    """
    worst_pull, worst_w = 0.0, 0.0
    for nm, (ch, _) in showcase().items():
        s = ch[:, BURN:].reshape(-1, 2)
        tau = np.array([np.mean([I.autocorr_time(c[BURN:, j]) for c in ch]) for j in (0, 1)])
        m_g, C_g = grid_posterior(nm)[:2]
        sd_g = np.sqrt(np.diag(C_g))
        pull = (s.mean(0) - m_g) / (sd_g / np.sqrt(len(s) / tau))
        worst_pull = max(worst_pull, abs(pull).max())
        worst_w = max(worst_w, abs(s.std(0) / sd_g - 1).max())
    return worst_pull < 4 and worst_w < 0.05, f"worst mean pull {worst_pull:.1f} sig;  worst width mismatch {worst_w:.1%}"


def check_mc_error_needs_tau():
    """The Monte Carlo error on a chain mean is sigma/sqrt(N/tau), not sigma/sqrt(N).

    This is manual-day Problem 4 with correlated samples. The 8 chain means
    scatter by the tau-corrected amount. Using the naive N would understate it
    by sqrt(tau), about 2.7. The ratios are averaged over 4 likelihoods x 2
    parameters, because 8 chains give a noisy spread (about 25% each).
    """
    r_tau, r_naive, taus = [], [], []
    for nm, (ch, _) in showcase().items():
        c = ch[:, BURN:]
        n = c.shape[1]
        for j in (0, 1):
            tau = np.mean([I.autocorr_time(x[:, j]) for x in c])
            spread = c[:, :, j].mean(1).std(ddof=1)
            sd = c[:, :, j].std()
            r_tau.append(spread / (sd / np.sqrt(n / tau)))
            r_naive.append(spread / (sd / np.sqrt(n)))
            taus.append(tau)
    rt, rn = np.mean(r_tau), np.mean(r_naive)
    ok = 0.75 < rt < 1.33 and rn > 2
    return ok, f"tau = {np.mean(taus):.1f};  spread / (sigma/sqrt(N/tau)) = {rt:.2f};  spread / (sigma/sqrt(N)) = {rn:.2f}"


def check_coverage():
    """Only D covers honestly. A, B and C undercover.

    The binomial error on a fraction of 1000 is about 1.5% at 68% and 0.7% at 95%.
    B and C reuse 40 C_hat blocks, so their trials are correlated and their
    errors are somewhat larger. The undercoverage required of them, 5 binomial
    sigma, is far beyond that.
    """
    cs = coverage_stats()
    f = {nm: (cs[nm]["in68"].mean(), cs[nm]["in95"].mean()) for nm in NAMES}
    n = len(cs["D"]["in68"])
    s68, s95 = np.sqrt(0.68 * 0.32 / n), np.sqrt(0.95 * 0.05 / n)
    d_ok = abs(f["D"][0] - 0.68) < 3 * s68 and abs(f["D"][1] - 0.95) < 3 * s95
    others = all(0.68 - f[nm][0] > 5 * s68 for nm in "ABC")
    detail = ";  ".join(f"{nm} {f[nm][0]:.1%} / {f[nm][1]:.1%}" for nm in NAMES)
    return d_ok and others, f"coverage 68% / 95%:  {detail}"


def plot():
    sc = showcase()
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax.ravel()
    OM, S8 = np.meshgrid(I.GRID_OM, I.GRID_S8, indexing="ij")

    a = ax[0]
    for nm in NAMES:
        c2 = grid_posterior(nm)[2].reshape(OM.shape)
        a.contour(OM, S8, c2 - c2.min(), levels=[2.30, 6.18], colors=COLORS[nm],
                  linewidths=[2, 1], linestyles=["-", "--"])
        a.plot([], [], color=COLORS[nm], lw=2, label=LABELS[nm])
    s = sc["D"][0][:, BURN::20].reshape(-1, 2)
    a.scatter(s[:, 0], s[:, 1], s=2, color=COLORS["D"], alpha=0.25, lw=0)
    a.plot(*I.TRUTH, "k*", ms=12, label="truth")
    a.set(xlabel=r"$\Omega_m$", ylabel=r"$\sigma_8$", xlim=(0.296, 0.320), ylim=(0.795, 0.825),
          title="Posterior for one data mock: 68% (solid), 95% (dashed)")
    a.legend(frameon=False, fontsize=8, loc="upper right")

    a = ax[1]
    ch = sc["D"][0]
    for c in ch:
        a.plot(c[:, 0], lw=0.5, alpha=0.8)
    a.axhline(I.TRUTH[0], color="k", lw=1)
    a.set(xlabel="step", ylabel=r"$\Omega_m$", xlim=(0, 300),
          title=r"Likelihood D: first 300 steps of 8 chains, started 3$\sigma$ out")

    a = ax[2]
    cs = coverage_stats()
    x = np.arange(4)
    n = len(cs["D"]["in68"])
    for off, key, target in ((-0.18, "in68", 0.68), (0.18, "in95", 0.95)):
        fr = np.array([cs[nm][key].mean() for nm in NAMES])
        a.bar(x + off, fr, 0.34, color=[COLORS[nm] for nm in NAMES],
              alpha=1.0 if key == "in68" else 0.5, edgecolor="white", lw=2)
        a.errorbar(x + off, fr, np.sqrt(target * (1 - target) / n), fmt="none", color="0.2", lw=1)
        a.axhline(target, color="0.3", lw=1, ls=":")
    a.set_xticks(x, ["A\nGaussian", "B\n50, raw", "C\n50, Hartlap", "D\n2000, Hartlap"])
    a.set(ylabel="fraction of mocks with truth inside", ylim=(0, 1),
          title=f"Coverage over {n} data mocks (solid 68%, faded 95%)")

    a = ax[3]
    nom = np.linspace(0, 1, 200)
    a.plot(nom, nom, color="0.3", lw=1, ls=":", label="honest")
    for nm in NAMES:
        emp_nom = 1 - np.exp(-0.5 * np.sort(cs[nm]["dchi2"]))   # nominal level reaching the truth
        a.plot(emp_nom, np.arange(1, len(emp_nom) + 1) / len(emp_nom), color=COLORS[nm], lw=2,
               label=LABELS[nm])
    a.set(xlabel="nominal credible level", ylabel="empirical coverage", xlim=(0, 1), ylim=(0, 1),
          title=r"Calibration at every level (from $\Delta\chi^2$ of the truth)")
    a.legend(frameon=False, fontsize=8, loc="upper left")

    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step5_mcmc.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 5: MCMC and coverage", [
        check_fast_binning,
        check_grid_resolves_posterior,
        check_chains_converged,
        check_mcmc_matches_grid,
        check_mc_error_needs_tau,
        check_coverage,
    ])
    plot()
    sys.exit(failures)
