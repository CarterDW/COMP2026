"""Spin correlations: a power law below T_KT, and eta = T / (2 pi Upsilon) = 1/4 at the transition.

    python checks/kt_correlations.py

G(r) = <cos(theta_0 - theta_r)>, from the FFT of s = exp(i theta) and averaged
along x and y. Below T_KT there is no long-range order, but G falls as a power,
G ~ r^-eta(T). The renormalized spin-wave theory gives eta = T / (2 pi Upsilon),
with Upsilon the stiffness measured in the same run. At T_KT the
Nelson-Kosterlitz value Upsilon = 2T/pi makes eta exactly 1/4. Above T_KT free
vortices make G decay exponentially, and a power law no longer fits.

On the torus the fit is to the spin-wave form G = A exp(-2 pi eta D(r)), with
D(r) = g(0) - g(r) the torus lattice Green's function difference. At short r it
is (1/2 pi) ln r, so this is the power law r^-eta with the torus built in. A
first version fitted A [r^-eta + (L - r)^-eta] instead. That gave eta 1.6-1.8x
too large at every T, even at T = 0.5, where the data match the torus
spin-wave G to 1e-3 at every r. The image form is not the torus shape at L = 64.
L = 64, 16 replicas per T, ordered start, 3000 warmup sweeps, then 4000 sweeps
measured every 10.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import lattice
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
L, R, T_KT = 64, 16, 0.8929
TS = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0, 1.1])
RES = {}


def sample():
    """Per-replica G(r) for r = 0..L/2 and Upsilon at every T."""
    if RES:
        return RES
    lattice.seed(1)
    beta = np.repeat(1 / TS, R)
    th = np.zeros((len(beta), L, L))
    delta = lattice.tune_delta(th, beta)
    for _ in range(3000):
        lattice.xy_sweep(th, beta, delta)
    G, cx, sx2, cy, sy2 = 0, 0, 0, 0, 0
    n = 0
    for k in range(4000):
        lattice.xy_sweep(th, beta, delta)
        if k % 10:
            continue
        s = np.exp(1j * th)
        S = np.abs(np.fft.fft2(s)) ** 2
        c = np.fft.ifft2(S).real / L**2
        G = G + 0.5 * (c[:, 0, : L // 2 + 1] + c[:, : L // 2 + 1, 0])
        a, b, d, e = lattice.xy_bond_sums(th)
        cx, sx2, cy, sy2 = cx + a, sx2 + b**2, cy + d, sy2 + e**2
        n += 1
    ups = 0.5 * ((cx - beta * sx2) + (cy - beta * sy2)) / n / L**2
    RES["G"] = (G / n).reshape(len(TS), R, -1)
    RES["ups"] = ups.reshape(len(TS), R)
    return RES


def torus_D():
    """D(r) = g(0) - g(r) along an axis, r = 0..L/2, for the L x L lattice Laplacian with the zero mode removed."""
    k = 2 * np.pi * np.fft.fftfreq(L)
    den = 4 - 2 * np.cos(k[:, None]) - 2 * np.cos(k[None, :])
    den[0, 0] = np.inf
    g = np.fft.ifft2(1 / den).real
    return g[0, 0] - g[0, : L // 2 + 1]


D = torus_D()


def model(r, A, eta):
    return A * np.exp(-2 * np.pi * eta * D[r.astype(int)])


def fit_eta(g):
    """Fit A exp(-2 pi eta D(r)) over 2 <= r <= L/2. Returns eta, its error, chi^2/dof and the parameters."""
    r = np.arange(L // 2 + 1)
    m, e = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))
    use = r >= 2
    p, cov = curve_fit(model, r[use], m[use], p0=(1, 0.1), sigma=e[use], absolute_sigma=True)
    chi2 = (((m[use] - model(r[use], *p)) / e[use]) ** 2).sum() / (use.sum() - 2)
    return p[1], np.sqrt(cov[1, 1]), chi2, p


def check_eta():
    """Below T_KT the fitted eta matches T / (2 pi Upsilon) to 10%, and is near 1/4 at T = 0.9.

    The fit and the prediction use different observables from the same runs:
    the shape of G(r) and the response to a twist. 10% allows for the
    logarithmic corrections near T_KT that a pure power law ignores.
    """
    res = sample()
    out, ok = [], True
    RES["eta"] = []
    for k, T in enumerate(TS):
        eta, err, chi2, _ = fit_eta(res["G"][k])
        pred = T / (2 * np.pi * res["ups"][k].mean())
        RES["eta"].append((eta, err, pred, chi2))
        if T < T_KT:
            ok &= abs(eta / pred - 1) < 0.1
        out.append(f"T={T}: eta {eta:.3f} vs {pred:.3f} (chi2/dof {chi2:.1f})")
    eta09 = RES["eta"][list(TS).index(0.9)][0]
    ok &= 0.2 < eta09 < 0.32
    return ok, "; ".join(out)


def plot():
    res = sample()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.25)
    r = np.arange(L // 2 + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(TS)))
    for k, T in enumerate(TS):
        g = res["G"][k].mean(0)
        _, _, _, p = fit_eta(res["G"][k])
        ax[0].loglog(r[1:], g[1:], "o", ms=3, color=colors[k], label=f"T = {T}")
        ax[0].loglog(r[2:], model(r[2:], *p), "-", lw=1, color=colors[k])
    ax[0].set(xlabel="distance r", ylabel=r"$G(r) = \langle\cos(\theta_0 - \theta_r)\rangle$",
              title="straight lines (power laws) below T_KT, bending above", ylim=(1e-3, 1.1))
    ax[0].legend(fontsize=8, ncol=2)

    eta, err, pred, _ = map(np.array, zip(*RES["eta"]))
    below = TS < T_KT
    ax[1].errorbar(TS, eta, err, fmt="o", label=r"fitted from G(r)")
    ax[1].plot(TS, pred, "s--", mfc="none", label=r"$T / 2\pi\Upsilon$ from the measured stiffness")
    tt = np.linspace(0.45, 1.15, 50)
    ax[1].plot(tt, tt / (2 * np.pi), ":", color="gray", label=r"bare spin waves $T/2\pi$")
    ax[1].axhline(0.25, color="k", lw=1, label=r"$\eta = 1/4$ at $T_{KT}$")
    ax[1].axvline(T_KT, color="gray", ls=":")
    ax[1].axvspan(T_KT, 1.15, color="0.92", zorder=0)
    ax[1].text(0.95, 0.03, "above T_KT:\nnot a power law", fontsize=8)
    ax[1].set(xlabel="T", ylabel=r"exponent $\eta$", ylim=(0, 0.6), xlim=(0.45, 1.15),
              title="the KT exponent: eta reaches 1/4 at the transition")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.savefig(PLOTS / "kt_correlations.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    fails = run("KT: spin correlations", [check_eta])
    plot()
    sys.exit(fails)
