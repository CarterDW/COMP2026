"""Step 2: Gaussian random fields and the binned P_hat(k) estimator.

    python checks/step2_gaussian_field.py      (about 2 minutes, for 200 mocks)

The targets are textbook Gaussian statistics: the moments of a unit normal, the
exponential distribution of |delta_k|^2 for a single Gaussian mode, Parseval's
theorem for the real-space variance, and the N_k/2 independent modes per shell
that set the scatter between mocks. None of them uses the estimator to check
itself.
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

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
M = 200                                   # number of mocks
P = lambda k: cosmo.P_model(k, 0.31, 0.81)


@functools.lru_cache(maxsize=None)
def mocks():
    """P_hat from M independent mocks, plus each mock's real-space variance."""
    P_hat, var = np.zeros((M, len(field.EDGES) - 1)), np.zeros(M)
    for i in range(M):
        delta = field.make_gaussian(P, seed=i)
        var[i] = delta.var()
        k, P_hat[i], Nk = field.measure_P(delta)
    return k, P_hat, Nk, var


def check_box_muller_moments():
    """Box-Muller output has the moments of a unit normal, and its pairs are independent.

    <z> = 0, <z^2> = 1 and <z^4> = 3, each tested against its own Monte Carlo
    error bar, as in manual-day Problem 4. The cos and sin halves share the same
    radius r, so their independence is not obvious. If they were independent
    only up to correlation, <z_c z_s> would vanish but <z_c^2 z_s^2> would not
    equal <z_c^2><z_s^2> = 1. Testing the second moment of the product is the
    sharper test.
    """
    n = 4_000_000
    z = field.box_muller(np.random.default_rng(123), n)
    zc, zs = z[: n // 2], z[n // 2:]
    tests = [  # (estimate, exact, standard deviation of one sample)
        (z.mean(), 0, 1),
        ((z**2).mean(), 1, np.sqrt(2)),
        ((z**4).mean(), 3, np.sqrt(105 - 9)),
        ((zc * zs).mean(), 0, 1),
        ((zc**2 * zs**2).mean(), 1, np.sqrt(9 - 1)),
    ]
    pulls = [(est - ex) / (sd / np.sqrt(len(zc) if i >= 3 else n))
             for i, (est, ex, sd) in enumerate(tests)]
    names = ["<z>", "<z^2>", "<z^4>", "<zc zs>", "<zc^2 zs^2>"]
    detail = ", ".join(f"{nm} {p:+.1f}" for nm, p in zip(names, pulls))
    return max(abs(p) for p in pulls) < 4, f"pulls in sigma: {detail}"


def check_field_is_real():
    """The inverse FFT of the shaped noise is real to roundoff.

    It is real only because fftn of a real w is Hermitian and the amplitude
    depends on |k|, which is symmetric under k -> -k on the grid (including the
    Nyquist plane). The check repeats make_gaussian without discarding the
    imaginary part.
    """
    rng = np.random.default_rng(0)
    w = field.box_muller(rng, field.NC).reshape((field.NG,) * 3)
    amp = np.sqrt(field.NC * field.P_grid(P) / field.V).reshape((field.NG,) * 3)
    d = np.fft.ifftn(np.fft.fftn(w) * amp)
    ratio = abs(d.imag).max() / d.real.std()
    return ratio < 1e-12, f"max |Im delta| / std(delta) = {ratio:.1e}"


def check_single_modes_are_exponential():
    """In one mock, P_hat_k / P(k) for individual modes is exponential with mean 1.

    A complex Gaussian with random phase has |delta_k|^2 exponentially
    distributed, so <x> = 1 and <x^2> = 2. This is the defining property of a
    Gaussian field, checked mode by mode before any binning. Only N/2 of the
    modes are independent, and the error bars use that count.
    """
    delta = field.make_gaussian(P, seed=999)
    Pk = abs(np.fft.fftn(delta).ravel())**2 * field.V / field.NC**2
    x = Pk[field._IN] / field.P_grid(P)[field._IN]
    n_ind = len(x) / 2
    p1 = (x.mean() - 1) / (1 / np.sqrt(n_ind))               # Var(x) = 1
    p2 = ((x**2).mean() - 2) / (np.sqrt(24 - 4) / np.sqrt(n_ind))   # Var(x^2) = <x^4> - 4 = 20
    ok = abs(p1) < 4 and abs(p2) < 4
    return ok, f"{int(n_ind)} independent modes: <x> = {x.mean():.4f} ({p1:+.1f} sig), <x^2> = {(x**2).mean():.4f} ({p2:+.1f} sig, expect 2)"


def check_real_space_variance():
    """Parseval: <delta^2> = sum over grid modes of P(k)/V, and the mean is exactly 0.

    This tests the sqrt(N_c P / V) amplitude independently of the estimator. The
    expected variance is large, around 1.6 per (7.8 Mpc/h)^3 cell. A real
    density field with that variance would have many cells below delta = -1,
    which is the motivation for step 4.
    """
    _, _, _, var = mocks()
    want = field.P_grid(P).sum() / field.V
    pull = (var.mean() - want) / (var.std(ddof=1) / np.sqrt(M))
    mean0 = abs(field.make_gaussian(P, seed=0).mean())
    ok = abs(pull) < 4 and mean0 < 1e-12
    return ok, f"<var> = {var.mean():.4f} vs {want:.4f} ({pull:+.1f} sig);  |mean delta| = {mean0:.1e}"


def check_mean_P_matches_model():
    """The average of P_hat over M mocks matches bin_model(P) in every bin.

    The error on the mean is sigma_pred / sqrt(M), so chi^2 over the 25 bins should
    be about 25 +/- 7. Wiggles are included, so a shifted or mis-binned k would
    show up.
    """
    _, P_hat, _, _ = mocks()
    pull = (P_hat.mean(0) - field.bin_model(P)) / (field.gaussian_sigma(P) / np.sqrt(M))
    chi2, nb = (pull**2).sum(), len(pull)
    return chi2 < nb + 4 * np.sqrt(2 * nb), f"chi^2 = {chi2:.1f} for {nb} bins;  worst pull {pull[abs(pull).argmax()]:+.1f}"


def check_scatter_is_gaussian_prediction():
    """The scatter between mocks equals sqrt(2/N_k^2 * sum P^2), about P sqrt(2/N_k).

    The factor of 2 is the k / -k pairing. Ignoring it (N_k independent modes)
    would make the scatter come out at 0.71 of the measured value. With M mocks,
    each bin's ratio sigma_meas/sigma_pred carries about 1/sqrt(2M) = 5% noise,
    and the mean of ratio^2 over 25 bins carries about 2%.
    """
    _, P_hat, _, _ = mocks()
    r = P_hat.std(0, ddof=1) / field.gaussian_sigma(P)
    tol_bin, tol_mean = 4 / np.sqrt(2 * M), 4 * np.sqrt(2 / (M * len(r)))
    ok = abs(r - 1).max() < tol_bin and abs((r**2).mean() - 1) < tol_mean
    return ok, f"ratio range [{r.min():.3f}, {r.max():.3f}] (tol {tol_bin:.2f});  mean ratio^2 = {(r**2).mean():.3f} (tol {tol_mean:.3f})"


def plot():
    k, P_hat, Nk, _ = mocks()
    model, sig = field.bin_model(P), field.gaussian_sigma(P)
    kk = np.linspace(0.01, 0.4, 500)
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    ax = ax.ravel()

    a = ax[0]
    a.plot(kk, kk * P(kk), color="0.5", lw=1, label="EH98 input")
    a.errorbar(k, k * P_hat.mean(0), k * sig / np.sqrt(M), fmt="o", ms=4, color=blue,
               label=f"mean of {M} mocks")
    a.fill_between(k, k * (model - sig), k * (model + sig), color=blue, alpha=0.15,
                   lw=0, label=r"one-mock $\pm1\sigma$ (predicted)")
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$k\,P(k)$ [(Mpc/$h$)$^2$]", title=r"Measured $\hat P(k)$")

    a = ax[1]
    a.axhspan(1 - 1 / np.sqrt(2 * M), 1 + 1 / np.sqrt(2 * M), color="0.85", lw=0,
              label=r"expected noise $1/\sqrt{2M}$")
    a.axhline(1, color="0.4", lw=1)
    a.plot(k, P_hat.std(0, ddof=1) / sig, "o-", color=blue, ms=4, lw=1.5,
           label=r"$k$/$-k$ paired: $N_k/2$ modes")
    a.plot(k, P_hat.std(0, ddof=1) / (sig / np.sqrt(2)), "s--", color=orange, ms=4, lw=1,
           label=r"unpaired: $N_k$ modes")
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$\sigma_{\rm meas} / \sigma_{\rm pred}$",
          title="Scatter between mocks", ylim=(0.5, 1.7))

    a = ax[2]
    delta = field.make_gaussian(P, seed=0)
    im = a.imshow(delta[:, :, 0].T, origin="lower", extent=(0, field.L, 0, field.L),
                  cmap="RdBu_r", vmin=-4, vmax=4)
    fig.colorbar(im, ax=a, label=r"$\delta$")
    a.set(xlabel=r"$x$ [Mpc/$h$]", ylabel=r"$y$ [Mpc/$h$]",
          title=rf"Raw slice: {100 * (delta < -1).mean():.0f}% of cells have $\delta<-1$")

    # Smoothing removes the cell-scale noise that dominates the raw variance and
    # shows the ~100 Mpc/h pattern, much like the spots on a CMB map.
    a = ax[3]
    R = 20.0
    ds = field.smooth(delta, R)[:, :, 0]
    lim = 3 * ds.std()
    im = a.imshow(ds.T, origin="lower", extent=(0, field.L, 0, field.L),
                  cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=a, label=r"$\delta_R$")
    a.set(xlabel=r"$x$ [Mpc/$h$]", ylabel=r"$y$ [Mpc/$h$]",
          title=rf"Same slice, Gaussian-smoothed $R = {R:.0f}$ Mpc/$h$ ($\sigma = {ds.std():.2f}$)")

    for a in ax[:2]:
        a.legend(frameon=False, fontsize=9)
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step2_gaussian_field.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 2: Gaussian field and P_hat(k)", [
        check_box_muller_moments,
        check_field_is_real,
        check_single_modes_are_exponential,
        check_real_space_variance,
        check_mean_P_matches_model,
        check_scatter_is_gaussian_prediction,
    ])
    plot()
    sys.exit(failures)
