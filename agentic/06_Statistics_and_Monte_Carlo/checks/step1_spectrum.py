"""Step 1: the EH98 linear power spectrum.

    python checks/step1_spectrum.py

Targets come from outside the code where possible: the white-noise variance of a
sphere, the primordial slope n_s, Planck's measured sound horizon, and EH98's own
independent fits. The sigma_8 check is the one self-consistency test, and its
docstring says so.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cosmo
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
OM, S8 = 0.31, 0.81


def check_tophat_white_noise():
    """For white noise, P = 1, the variance in a sphere is 1/V_sphere.

    Counting cells: white noise of power P has variance P/V when averaged over a
    volume V, so sigma(R)^2 = 3/(4 pi R^3). In Fourier space that is the statement
    int_0^inf x^2 W(x)^2 dx = 3 pi / 2. This tests the window, the 1/2 pi^2, and
    the log-k integration at once, with no cosmology involved.

    The k grid stops at k_max, and W^2 ~ 9 cos^2(x)/x^4 there, so the truncated
    integral is short by about (9/2)/(k_max R). That deficit is predicted, not
    absorbed into a loose tolerance.

    What remains, about 1e-4, is the 4000-point log grid undersampling cos^2(kR)
    near k_max, where a period in ln k is only pi/(kR) wide. White noise is a
    worst case. A cosmological P falls as k^-3 there, and the next check shows the
    grid is converged for it.
    """
    kmax = cosmo._K_INT[-1]
    worst = 0.0
    for R in (8.0, 20.0, 50.0):
        got = cosmo.sigma_R(lambda k: np.ones_like(k), R)**2
        want = (3 * np.pi / 2 - 4.5 / (kmax * R)) / (2 * np.pi**2 * R**3)
        worst = max(worst, abs(got / want - 1))
    return worst < 3e-4, f"max relative error over R = 8, 20, 50 Mpc/h: {worst:.1e}"


def check_sigma_grid_converged():
    """For the actual EH98 shape, the 4000-point grid matches a 64000-point one.

    The default grid keeps P_model at about 0.7 ms per call, which matters
    because MCMC calls it around 1e5 times.
    """
    shape = lambda k: k**cosmo.NS * cosmo.transfer(k, OM)**2
    coarse = cosmo.sigma_R(shape, 8.0)
    k_saved = cosmo._K_INT
    cosmo._K_INT = np.logspace(-5, 2, 64000)
    fine = cosmo.sigma_R(shape, 8.0)
    cosmo._K_INT = k_saved
    err = abs(coarse / fine - 1)
    return err < 1e-6, f"|sigma_8(4000 pts) / sigma_8(64000 pts) - 1| = {err:.1e}"


def check_sigma8_normalization():
    """sigma_8 of the returned P equals the requested s8.

    Self-consistency only, since P_model uses sigma_R to set its amplitude. It
    still catches the amplitude being applied as s8 instead of s8^2.
    """
    worst = 0.0
    for Om in (0.25, 0.31, 0.40):
        for s8 in (0.6, 0.81, 1.0):
            got = cosmo.sigma_R(lambda k: cosmo.P_model(k, Om, s8), 8.0)
            worst = max(worst, abs(got / s8 - 1))
    return worst < 1e-10, f"max |sigma_8/s8 - 1| over a 3x3 grid: {worst:.1e}"


def check_large_scale_limit():
    """Modes far outside the horizon at equality are unprocessed: T -> 1, P -> k^n_s.

    Both transfer functions are tested, plus the log slope of P at k = 1e-4,
    which should be the primordial tilt n_s = 0.965.
    """
    k = np.array([1e-5, 1e-4])
    T_off = max(abs(cosmo.transfer(k, OM) - 1).max(),
                abs(cosmo.transfer_nowiggle(k, OM) - 1).max())
    lnk = np.log([0.9e-4, 1.1e-4])
    slope = np.diff(np.log(cosmo.P_model(np.exp(lnk), OM, S8)))[0] / np.diff(lnk)[0]
    ok = T_off < 1e-3 and abs(slope - cosmo.NS) < 5e-3
    return ok, f"max |T - 1| at k <= 1e-4: {T_off:.1e};  d ln P/d ln k at 1e-4: {slope:.4f} (n_s = {cosmo.NS})"


def check_sound_horizon():
    """The EH98 sound horizon agrees with Planck and with EH98's own simpler fit.

    Planck 2018 measures r_drag = 147.1 Mpc for nearly these parameters. EH98
    eq 6 is known to come out a few percent high, so the tolerance is 5%. Eq 26 is
    an independent one-line fit from the same paper, used by the no-wiggle form.
    """
    s = cosmo.sound_horizon(OM) / cosmo.H
    wm, wb = OM * cosmo.H**2, cosmo.OB * cosmo.H**2
    s26 = 44.5 * np.log(9.83 / wm) / np.sqrt(1 + 10 * wb**0.75)
    ok = abs(s / 147.1 - 1) < 0.05 and abs(s / s26 - 1) < 0.03
    return ok, f"s = {s:.1f} Mpc;  Planck 147.1 ({s/147.1 - 1:+.1%});  EH98 eq 26 {s26:.1f} ({s/s26 - 1:+.1%})"


def check_wiggles_vs_smooth():
    """The no-wiggle fit traces the full spectrum through the BAO.

    They are separate formulas, so agreement is a real test. The ratio should be
    1 on large scales, oscillate by a few percent through the BAO, and settle
    back once Silk damping erases the oscillations.
    """
    r = lambda k: cosmo.P_model(k, OM, S8) / cosmo.P_model(k, OM, S8, wiggle=False)
    big = abs(r(np.logspace(-4, -2.5, 50)) - 1).max()
    bao = abs(r(np.linspace(0.01, 0.4, 2000)) - 1).max()
    damped = abs(r(np.linspace(0.6, 1.0, 50)) - 1).max()
    ok = big < 0.01 and 0.02 < bao < 0.10 and damped < 0.03
    return ok, f"max |ratio - 1|: k < 0.003 {big:.3f};  BAO range {bao:.3f};  0.6 < k < 1 {damped:.3f}"


def check_bao_period():
    """The wiggles in P/P_nw are spaced by pi/s in k.

    P_b carries j0(k s), so zero crossings of the ratio - 1 fall every pi/s. The
    crossing positions are measured from the curve and fit with a straight line
    against crossing index.
    """
    k = np.linspace(0.05, 0.4, 20000)
    d = cosmo.P_model(k, OM, S8) / cosmo.P_model(k, OM, S8, wiggle=False) - 1
    i = np.nonzero(np.diff(np.sign(d)))[0]
    kz = k[i] - d[i] * (k[i + 1] - k[i]) / (d[i + 1] - d[i])
    spacing = np.polyfit(np.arange(len(kz)), kz, 1)[0]
    want = np.pi / cosmo.sound_horizon(OM)
    return abs(spacing / want - 1) < 0.05, \
        f"{len(kz)} crossings, spacing {spacing:.4f} h/Mpc vs pi/s = {want:.4f} ({spacing/want - 1:+.1%})"


def check_turnover_scales_with_Om():
    """The peak of P sits near k_eq, which scales as Om h in h/Mpc units.

    Raising Om from 0.26 to 0.36 should move the peak right by about 0.36/0.26.
    Ob is held fixed, so the baryon fraction also falls as Om rises, which shifts
    the peak further. Expect agreement at the 10% level, not exactly.
    """
    k = np.logspace(-3, -0.5, 20000)
    peak = lambda Om: k[cosmo.P_model(k, Om, S8, wiggle=False).argmax()]
    got = peak(0.36) / peak(0.26)
    want = 0.36 / 0.26
    return abs(got / want - 1) < 0.10, f"k_peak ratio {got:.3f} vs Om ratio {want:.3f} ({got/want - 1:+.1%})"


def plot():
    k = np.logspace(-3, 0, 1000)
    kb = np.linspace(0.01, 0.4, 1000)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for Om, c in zip((0.26, 0.31, 0.36), colors):
        a.loglog(k, cosmo.P_model(k, Om, S8), color=c, lw=2, label=rf"$\Omega_m = {Om}$")
        a.loglog(k, cosmo.P_model(k, Om, S8, wiggle=False), color=c, lw=1, ls="--")
        b.plot(kb, cosmo.P_model(kb, Om, S8) / cosmo.P_model(kb, Om, S8, wiggle=False),
               color=c, lw=2, label=rf"$\Omega_m = {Om}$")
    a.axvspan(0.01, 0.4, color="0.9", zorder=0)
    a.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P(k)$ [(Mpc/$h$)$^3$]",
          title=r"EH98 linear $P(k)$, $\sigma_8 = 0.81$ (dashed: no-wiggle)", ylim=(1e2, 5e4))
    b.axhline(1, color="0.5", lw=1)
    b.set(xlabel=r"$k$ [$h$/Mpc]", ylabel=r"$P / P_{\rm nw}$", title="BAO wiggles over the fit range")
    for ax in (a, b):
        ax.legend(frameon=False)
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "step1_spectrum.png", dpi=130)


if __name__ == "__main__":
    failures = run("Step 1: EH98 power spectrum", [
        check_tophat_white_noise,
        check_sigma_grid_converged,
        check_sigma8_normalization,
        check_large_scale_limit,
        check_sound_horizon,
        check_wiggles_vs_smooth,
        check_bao_period,
        check_turnover_scales_with_Om,
    ])
    plot()
    sys.exit(failures)
