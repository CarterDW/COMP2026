"""The fit itself: data, best-fit model, residuals, and best fits over 1000 universes.

    python checks/fit_figure.py      (seconds once steps 5 and 6 have run)

Not a check, only the picture the other plots leave out. It uses step 6's cached
grid best fits.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import field
import inference as I
from checks import step6_fisher as S6

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "0.35"


def main():
    _, test = I.data()
    d = test[0]
    k = field.bin_model(lambda kk: kk)
    sig_G = field.gaussian_sigma(lambda kk: I.cosmo.P_model(kk, *I.TRUTH))
    sig_LN = np.sqrt(np.diag(S6.C_TRUE))
    bf = {nm: S6.best_fits(nm) for nm in "AD"}
    mu_true = I.model(I.TRUTH)
    mu_fit = {nm: I.model(bf[nm][0]) for nm in "AD"}

    fig, ax = plt.subplots(3, 2, figsize=(12, 15))
    top, ax = ax[0], ax[1:].ravel()

    # Absolute spectra: points are the binned data, curves are the unbinned model
    # P(k) at the best-fit and true parameters.
    kk = np.linspace(field.EDGES[0], field.EDGES[-1], 600)
    P_fit = I.cosmo.P_model(kk, *bf["D"][0])
    P_tru = I.cosmo.P_model(kk, *I.TRUTH)
    a = top[0]
    a.errorbar(k, d, sig_LN, fmt="o", color=BLUE, ms=4, capsize=2, lw=1.2, label="data $\\hat P(k)$, mock #0")
    a.plot(kk, P_fit, color=BLUE, lw=2, label=rf"$P_{{\rm fit}}$: $\Omega_m$={bf['D'][0][0]:.4f}, $\sigma_8$={bf['D'][0][1]:.4f}")
    a.plot(kk, P_tru, color="k", lw=1, ls="--", label="$P_{\\rm true}$")
    a.set(xscale="log", yscale="log", xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k)$ [(Mpc/$h$)$^3$]",
          title="Measured $\\hat P(k)$ and the fitted EH98 spectrum")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    a = top[1]
    zoom = k < 0.21
    a.errorbar(k[zoom], k[zoom] * d[zoom], k[zoom] * sig_LN[zoom], fmt="o", color=BLUE, ms=5,
               capsize=3, lw=1.5, label="data")
    a.plot(kk, kk * P_fit, color=BLUE, lw=2, label="$P_{\\rm fit}$")
    a.plot(kk, kk * P_tru, color="k", lw=1, ls="--", label="$P_{\\rm true}$")
    a.plot(kk, kk * I.cosmo.P_model(kk, *bf["D"][0], wiggle=False), color=GREY, lw=1, ls=":",
           label="$P_{\\rm fit}$ without BAO")
    a.set(xlim=(0.01, 0.21), xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$k\,P(k)$ [(Mpc/$h$)$^2$]",
          title="Zoom on the BAO: $k\\,P(k)$, linear axes")
    a.legend(frameon=False, fontsize=8)

    a = ax[0]
    # Everything divided by the true model, so percent-level errors are visible.
    a.axhline(0, color="k", lw=1, ls="--", label=r"truth: $\Omega_m$=0.31, $\sigma_8$=0.81")
    a.errorbar(k, 100 * (d / mu_true - 1), 100 * sig_LN / mu_true, fmt="o", color=BLUE, ms=5,
               capsize=3, lw=1.5, label="data (mock #0), mock-covariance errors")
    a.errorbar(k + 0.004, 100 * (d / mu_true - 1), 100 * sig_G / mu_true, fmt="none", color=ORANGE,
               capsize=2, lw=1.5, label="same data, Gaussian errors")
    a.plot(k, 100 * (mu_fit["D"] / mu_true - 1), color=BLUE, lw=2,
           label=rf"best-fit model D: $\Omega_m$={bf['D'][0][0]:.4f}, $\sigma_8$={bf['D'][0][1]:.4f}")
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$\hat P / P_{\rm true} - 1$  [%]", ylim=(-12, 12),
          title="One universe: data and fitted EH98 model, relative to the truth")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    a = ax[1]
    r = d - mu_fit["D"]
    a.axhspan(-1, 1, color="0.9", lw=0)
    a.axhline(0, color=GREY, lw=1)
    a.plot(k, r / sig_LN, "o-", color=BLUE, ms=5, lw=1.5, label="in units of mock-covariance $\\sigma$")
    a.plot(k, r / sig_G, "s--", color=ORANGE, ms=4, lw=1, label="in units of Gaussian $\\sigma$")
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"(data $-$ best fit) / $\sigma$",
          title="Residuals of the same fit, measured against each error bar")
    a.legend(frameon=False, fontsize=8)

    t = np.linspace(0, 2 * np.pi, 300)
    for a, nm, col, name in ((ax[2], "A", ORANGE, "A: Gaussian $C$"),
                             (ax[3], "D", BLUE, r"D: 2000-mock $\hat C$ + Hartlap")):
        R = np.linalg.inv(S6.J.T @ I.precision(nm) @ S6.J)
        x = bf[nm] - I.TRUTH
        inside = np.einsum("ni,ij,nj->n", x, np.linalg.inv(R), x) < 2.30
        a.scatter(*bf[nm][~inside].T, s=6, color=GREY, alpha=0.5, lw=0, label="best fit outside")
        a.scatter(*bf[nm][inside].T, s=6, color=col, alpha=0.8, lw=0, label="best fit inside")
        e = I.TRUTH[:, None] + np.linalg.cholesky(R) @ np.array([np.cos(t), np.sin(t)]) * np.sqrt(2.30)
        a.plot(*e, color="k", lw=2, label="reported 68% error ellipse")
        a.plot(*I.TRUTH, "k*", ms=12)
        a.set(xlabel=r"$\Omega_m$", ylabel=r"$\sigma_8$", xlim=(0.295, 0.325), ylim=(0.795, 0.825),
              title=f"{name}: 1000 universes, {inside.mean():.0%} of best fits inside")
        a.legend(frameon=False, fontsize=8, loc="upper right", markerscale=3)

    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step5_fits.png", dpi=130)


if __name__ == "__main__":
    main()
