"""The CMB acoustic peaks and the matter BAO wiggles by epoch, from CAMB, with Planck 2018.

    python -I checks/cmb_and_bao.py        (seconds; needs camb, and results/planck/)

Same cosmology as the rest of the project: h = 0.67, Ob = 0.049, Om = 0.31,
n_s = 0.965, massless neutrinos, sigma_8 = 0.81 (A_s is tuned to hit it), and
tau = 0.054.

Planck data: COM_PowerSpect_CMB-TT-{binned,full}_R3.01.txt from the Planck Legacy
Archive (IRSA mirror), in results/planck/. They are read with np.loadtxt only.
The theory is not fitted to Planck, so the comparison is a consistency check.

No-wiggle spectrum for CAMB: P_nw = P_EHnw * G[P / P_EHnw], where G is a
Gaussian smoothing in ln k of width 0.25 (Vlah et al. 2016). EH98's no-wiggle
form supplies the broadband, and the smoothing removes the oscillations from
the ratio, whatever the epoch.

Nonlinear BAO damping at z = 0, at leading order in IR resummation:
P = P_nw + (P - P_nw) exp(-k^2 Sigma^2), with
Sigma^2 = (1/6 pi^2) int_0^{0.2} P_nw(q) [1 - j0(q r_s) + 2 j2(q r_s)] dq
(Baldauf et al. 2015). Sigma is computed from the spectrum, not chosen.
"""

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import camb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import spherical_jn

import cosmo
from checks._runner import run

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLOTS = ROOT / "checks" / "plots"
PLANCK = ROOT / "results" / "planck"
OM, S8 = 0.31, 0.81
ZS = [1100, 500, 200, 50, 0]
KH = np.logspace(-4, np.log10(4.0), 3000)          # uniform in ln k, h/Mpc


def _params(As):
    h = cosmo.H
    return camb.set_params(H0=100 * h, ombh2=cosmo.OB * h**2, omch2=(OM - cosmo.OB) * h**2,
                           mnu=0, num_massive_neutrinos=0, nnu=3.046, ns=cosmo.NS, As=As,
                           tau=0.054, lmax=2600, WantTransfer=True, kmax=8.0, redshifts=ZS)


@functools.lru_cache(maxsize=None)
def camb_results():
    """CAMB run with A_s rescaled so sigma_8 = 0.81 exactly (sigma_8^2 is linear in A_s)."""
    As0 = 2.1e-9
    s0 = camb.get_results(_params(As0)).get_sigma8_0()
    pars = _params(As0 * (S8 / s0)**2)
    return pars, camb.get_results(pars)


@functools.lru_cache(maxsize=None)
def matter_pk():
    """Linear P(k) at each z in ZS on the KH grid, as {z: array}."""
    _, r = camb_results()
    interp = r.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)
    return {z: interp.P(z, KH) for z in ZS}


def no_wiggle(P):
    ref = cosmo.transfer_nowiggle(KH, OM)**2 * KH**cosmo.NS
    dlnk = np.diff(np.log(KH))[0]
    return ref * np.exp(gaussian_filter1d(np.log(P / ref), 0.25 / dlnk, mode="nearest"))


def sigma_bao(Pnw, r_s):
    q = KH[KH < 0.2]
    f = Pnw[KH < 0.2] * (1 - spherical_jn(0, q * r_s) + 2 * spherical_jn(2, q * r_s))
    return np.sqrt(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(q)) / (6 * np.pi**2))


def r_drag_h():
    return camb_results()[1].get_derived_params()["rdrag"] * cosmo.H


@functools.lru_cache(maxsize=None)
def cmb():
    pars, r = camb_results()
    cl = r.get_cmb_power_spectra(pars, CMB_unit="muK")
    ell = np.arange(cl["total"].shape[0])
    return ell, cl["total"][:, 0], cl["unlensed_total"][:, 0]


def planck():
    binned = np.loadtxt(PLANCK / "COM_PowerSpect_CMB-TT-binned_R3.01.txt")
    full = np.loadtxt(PLANCK / "COM_PowerSpect_CMB-TT-full_R3.01.txt")
    return binned, full[full[:, 0] < 30]


def wiggle_amp(z, kmin=0.03, kmax=0.3):
    P = matter_pk()[z]
    m = (KH > kmin) & (KH < kmax)
    return abs(P / no_wiggle(P) - 1)[m].max()


def check_camb_sigma8():
    """CAMB's sigma_8 equals 0.81 after rescaling A_s."""
    s = camb_results()[1].get_sigma8_0()
    return abs(s / S8 - 1) < 1e-4, f"sigma_8 = {s:.5f}"


def _crossings(P, kmin=0.03, kmax=0.3):
    m = (KH > kmin) & (KH < kmax)
    w, k = (P / no_wiggle(P) - 1)[m], KH[m]
    i = np.nonzero(np.diff(np.sign(w)))[0]
    return k[i] - w[i] * (k[i + 1] - k[i]) / (w[i + 1] - w[i])


def check_eh98_vs_camb():
    """EH98 matches CAMB's linear z = 0 spectrum to a few percent. Its wiggle spacing
    is compressed by its sound-horizon error, plus a small phase offset.

    This is the CAMB comparison that step 1 left optional. EH98's eq 6 r_drag is
    2.7% above CAMB's. Wiggles repeat every pi/r_s in k, so EH98's crossing spacing
    should be CAMB's times r_CAMB/r_EH = 0.973. The spacing comes from a straight-
    line fit of crossing position against index, as in step 1.

    History: a version of this check used the median of per-crossing position
    ratios, got 1.006, and was wrongly taken as falsifying the prediction. Position
    ratios mix the spacing with a constant phase offset (EH98's crossings sit about
    +0.002-0.004 h/Mpc high at low k), so they were the wrong measure. The offset is
    reported in the detail line.
    """
    m = (KH > 0.01) & (KH < 0.4)
    ratio = cosmo.P_model(KH[m], OM, S8) / matter_pk()[0][m]
    s_eh = cosmo.sound_horizon(OM) / cosmo.H
    s_camb = r_drag_h() / cosmo.H
    ce, cc = _crossings(cosmo.P_model(KH, OM, S8)), _crossings(matter_pk()[0])
    n = min(len(ce), len(cc))
    (be, ae), (bc, ac) = np.polyfit(np.arange(n), ce[:n], 1), np.polyfit(np.arange(n), cc[:n], 1)
    want = s_camb / s_eh
    ok = abs(ratio - 1).max() < 0.05 and abs(be / bc / want - 1) < 0.015
    return ok, (f"EH98/CAMB P over 0.01<k<0.4: [{ratio.min():.3f}, {ratio.max():.3f}];  "
                f"crossing spacing EH/CAMB {be / bc:.3f} vs r_CAMB/r_EH {want:.3f} ({s_eh / s_camb - 1:+.1%} in r_drag);  "
                f"phase offset at first crossing {ae - ac:+.4f} h/Mpc")


def check_peak_spacing():
    """CMB peak spacing equals the acoustic scale l_A = pi / theta_*.

    theta_* = r_s(z_*) / D_A(z_*) is the angle the sound horizon subtends at last
    scattering. Each peak is one more half-oscillation, so successive peaks sit
    l_A apart. The first peak falls short of l_A because of a phase shift from
    driving by the potentials, which is why it is at ~220 and not ~300. The test
    uses peaks 2 -> 3 -> 4, unlensed, where that shift is roughly constant.
    """
    ell, _, Dl = cmb()
    lA = np.pi / (camb_results()[1].get_derived_params()["thetastar"] / 100)
    pk = [l for l in range(100, 1500) if Dl[l] > Dl[l - 1] and Dl[l] > Dl[l + 1]]
    spacing = np.diff(pk[1:4]).mean()
    return abs(spacing / lA - 1) < 0.08, f"peaks at l = {pk[:5]};  spacing(2-4) {spacing:.0f} vs l_A = pi/theta_* = {lA:.0f}"


@functools.lru_cache(maxsize=None)
def cmb_planck_params():
    """CAMB TT at the Planck 2018 best-fit LCDM parameters (TT,TE,EE+lowE+lensing)."""
    p = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, ns=0.9649,
                        As=np.exp(3.044) * 1e-10, tau=0.0544, lmax=2600)
    return camb.get_results(p).get_cmb_power_spectra(p, CMB_unit="muK")["total"][:, 0]


def planck_chi2(theory_Dl):
    binned, _ = planck()
    l, d, lo, hi = binned[:, :4].T
    th = np.interp(l, np.arange(len(theory_Dl)), theory_Dl)
    return np.sum(((d - th) / np.where(d > th, lo, hi))**2), len(l)


def check_planck_consistent():
    """The Planck comparison works, and the CMB rejects this project's cosmology.

    Pipeline checks: Planck's own best-fit column, and CAMB run at the Planck 2018
    parameters, should both give chi^2/bin ~ 1 against the binned data (diagonal
    errors, no foreground nuisances).

    The project cosmology (Om = 0.31, h = 0.67, so omega_c = 0.1172) gave chi^2 = 223
    on the first run. A first version of this check wrongly expected ~1. Planck has
    omega_c = 0.1200, and the peak heights pin omega_c at the 1% level, so the
    difference is a real Delta chi^2 of ~140. It is reported, not required, since
    the large-scale-structure steps never needed Planck's exact cosmology.
    """
    binned, _ = planck()
    c_col = np.sum(((binned[:, 1] - binned[:, 4]) / binned[:, 2])**2)
    c_pl, n = planck_chi2(cmb_planck_params())
    c_ours, _ = planck_chi2(cmb()[1])
    ok = c_col / n < 1.5 and c_pl / n < 1.5
    return ok, (f"chi^2/{n} bins: Planck best-fit column {c_col:.0f}, CAMB at Planck params {c_pl:.0f};  "
                f"this project's cosmology {c_ours:.0f} (Delta chi^2 = {c_ours - c_pl:.0f})")


def zero_crossings(z, kmin=0.03, kmax=0.3):
    P = matter_pk()[z]
    m = (KH > kmin) & (KH < kmax)
    w, k = (P / no_wiggle(P) - 1)[m], KH[m]
    i = np.nonzero(np.diff(np.sign(w)))[0]
    return k[i] - w[i] * (k[i + 1] - k[i]) / (w[i + 1] - w[i])


def check_bao_settles():
    """The matter BAO at z = 1100 differ from late times, then freeze by z ~ 200.

    The first version of this check predicted larger wiggles at z = 1100, and the
    run showed the opposite. At the drag epoch the baryon density contrast is
    still small next to the CDM's. The late-time BAO pattern is built mostly
    afterwards, by the baryons' velocity at release ("velocity overshoot"),
    which carries a different phase than the density term. So the wiggles grow
    in amplitude and shift in phase after z ~ 1100. Once the baryons have caught
    up, linear growth multiplies P and P_nw equally and the pattern freezes.

    Tested: the z = 1100 amplitude is smaller than today's, and its first zero
    crossing has moved by more than 0.01 h/Mpc. From z = 200 to 0, the amplitude
    agrees within 3% and the crossings within 0.001 h/Mpc.
    """
    a = {z: wiggle_amp(z) for z in ZS}
    zc = {z: zero_crossings(z) for z in ZS}
    n = min(len(zc[200]), len(zc[0]))
    frozen = abs(zc[200][:n] - zc[0][:n]).max()
    shift = abs(zc[1100][0] - zc[0][0])
    ok = a[1100] < a[0] and shift > 0.01 and abs(a[200] / a[0] - 1) < 0.03 and frozen < 1e-3
    return ok, ("amplitude: " + ", ".join(f"z={z} {a[z]:.3f}" for z in ZS)
                + f";  first crossing z=1100 {zc[1100][0]:.4f} vs z=0 {zc[0][0]:.4f};  "
                f"max crossing shift z=200->0 {frozen:.1e}")


def check_damping_scale():
    """The IR-resummation damping scale at z = 0 is Sigma ~ 5-6 Mpc/h.

    In the exp(-k^2 Sigma^2) convention used here, that corresponds to the
    familiar Sigma_nl ~ 8 Mpc/h of the exp(-k^2 Sigma_nl^2 / 2) convention.
    """
    S = sigma_bao(no_wiggle(matter_pk()[0]), r_drag_h())
    return 4.5 < S < 7, f"Sigma = {S:.2f} Mpc/h;  wiggles at k = 0.2 survive at {np.exp(-(0.2 * S)**2):.0%}"


def plot():
    ell, Dl, Dl_u = cmb()
    binned, low = planck()
    blue, orange = "#2a78d6", "#eb6834"
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.55, 2.4], hspace=0.08)
    a, ar = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])

    l = ell[2:]
    a.plot(l, Dl[2:], color=blue, lw=2, label="CAMB, lensed (this project's cosmology, not fitted)")
    a.plot(l, Dl_u[2:], color=blue, lw=1, ls=":", label="CAMB, unlensed")
    a.plot(l, cmb_planck_params()[2:len(ell)], color="k", lw=1, ls="--",
           label="CAMB, Planck 2018 best-fit parameters")
    a.errorbar(low[:, 0], low[:, 1], [low[:, 2], low[:, 3]], fmt="o", color="0.3", ms=3, lw=1,
               label=r"Planck 2018, $\ell<30$")
    a.errorbar(binned[:, 0], binned[:, 1], [binned[:, 2], binned[:, 3]], fmt="o", color=orange,
               ms=4, lw=1, label=r"Planck 2018, binned $\ell\geq30$")
    a.set(xscale="log", xlim=(2, 2500), ylabel=r"$D_\ell^{TT} = \ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]",
          title="CMB temperature power spectrum: the acoustic peaks")
    a.tick_params(labelbottom=False)
    a.legend(frameon=False, fontsize=9, loc="upper left")

    th = np.interp(binned[:, 0], ell, Dl)
    err = np.where(binned[:, 1] > th, binned[:, 2], binned[:, 3])
    ar.axhspan(-1, 1, color="0.9", lw=0)
    ar.axhline(0, color="0.3", lw=1)
    ar.plot(binned[:, 0], (binned[:, 1] - th) / err, "o", color=orange, ms=4,
            label=f"vs this project's cosmology ($\\chi^2$ = {planck_chi2(Dl)[0]:.0f}/83)")
    thp = np.interp(binned[:, 0], np.arange(len(cmb_planck_params())), cmb_planck_params())
    errp = np.where(binned[:, 1] > thp, binned[:, 2], binned[:, 3])
    ar.plot(binned[:, 0], (binned[:, 1] - thp) / errp, "s", color="0.2", ms=3,
            label=f"vs Planck 2018 parameters ($\\chi^2$ = {planck_chi2(cmb_planck_params())[0]:.0f}/83)")
    ar.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ar.set(xscale="log", xlim=(2, 2500), ylim=(-4, 4), xlabel=r"multipole $\ell$",
           ylabel=r"(Planck $-$ CAMB)/$\sigma$")

    # Bottom row: three simple comparisons, two curves each, same no-wiggle extraction.
    P0, P1100 = matter_pk()[0], matter_pk()[1100]
    w = lambda P: P / no_wiggle(P) - 1
    S = sigma_bao(no_wiggle(P0), r_drag_h())
    P_eh = cosmo.P_model(KH, OM, S8)
    panels = [
        ("Our EH98 model vs CAMB, z = 0",
         [(w(P0), "CAMB", "k", "-"), (w(P_eh), "EH98 (steps 1-6)", orange, "--")]),
        ("Recombination vs today (CAMB)",
         [(w(P1100), "z = 1100", "#7b3294", "-"), (w(P0), "z = 0", "k", "-")]),
        ("Nonlinear damping today (CAMB)",
         [(w(P0), "z = 0, linear", "k", "-"),
          (w(P0) * np.exp(-(KH * S)**2), rf"z = 0, damped ($\Sigma$ = {S:.1f} Mpc/$h$)", blue, "-")]),
    ]
    sub = gs[3, :].subgridspec(1, 3, wspace=0.08)
    bottom = []
    for j, (title, curves) in enumerate(panels):
        x = fig.add_subplot(sub[0, j], sharey=bottom[0] if bottom else None)
        for y, lab, col, ls in curves:
            x.plot(KH, y, color=col, ls=ls, lw=2, label=lab)
        x.axhline(0, color="0.6", lw=1)
        x.set(xlim=(0.01, 0.35), xlabel=r"$k$ [$h$/Mpc]", title=title)
        x.legend(frameon=False, fontsize=9, loc="upper right")
        if j:
            x.tick_params(labelleft=False)
        bottom.append(x)
    bottom[0].set_ylabel(r"$P/P_{\rm nw} - 1$")

    for x in (a, ar, *bottom):
        x.grid(alpha=0.3, lw=0.5)
    fig.savefig(PLOTS / "cmb_and_bao.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    failures = run("CMB peaks and BAO by epoch (CAMB + Planck)", [
        check_camb_sigma8,
        check_eh98_vs_camb,
        check_peak_spacing,
        check_planck_consistent,
        check_bao_settles,
        check_damping_scale,
    ])
    plot()
    sys.exit(failures)
