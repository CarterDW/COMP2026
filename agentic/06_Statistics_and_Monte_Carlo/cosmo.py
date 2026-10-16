"""The linear matter power spectrum P(k; Om, s8) from Eisenstein & Hu (1998).

EH98 (astro-ph/9709112) is a closed-form fit to the Boltzmann-code transfer
function T(k), including the baryon acoustic oscillations. The spectrum is

    P(k) = A k^n_s T(k)^2,

with A fixed by requiring sigma(R = 8 Mpc/h) = s8. Equation numbers below are
EH98's. The paper works in Mpc and Mpc^-1; this module takes and returns h/Mpc
and (Mpc/h)^3, converting at the boundary.

Fixed parameters: h, Ob, n_s, T_CMB. Free: Om, s8.
"""

import numpy as np

H = 0.67
OB = 0.049
NS = 0.965
T_CMB = 2.7255

# k grid for the sigma(R) integral, in h/Mpc. Wide enough that the integrand
# k^3 P W^2 is negligible at both ends for R = 8.
_K_INT = np.logspace(-5, 2, 4000)


def _shared(Om):
    """Scales common to both transfer functions, in Mpc units (EH98 eqs 2-6)."""
    wm, wb = Om * H**2, OB * H**2
    th = T_CMB / 2.7
    z_eq = 2.50e4 * wm * th**-4
    k_eq = 7.46e-2 * wm * th**-2
    b1 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
    b2 = 0.238 * wm**0.223
    z_d = 1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b1 * wb**b2)
    R_d = 31.5 * wb * th**-4 * (1e3 / z_d)
    R_eq = 31.5 * wb * th**-4 * (1e3 / z_eq)
    s = (2 / (3 * k_eq) * np.sqrt(6 / R_eq)
         * np.log((np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq))))
    return dict(wm=wm, wb=wb, th=th, z_eq=z_eq, k_eq=k_eq, z_d=z_d, R_d=R_d, s=s)


def sound_horizon(Om):
    """The sound horizon at the drag epoch, in Mpc/h (EH98 eq 6)."""
    return _shared(Om)["s"] * H


def _T0(q, alpha, beta):
    """EH98 eqs 19-20, the CDM-like shape."""
    L = np.log(np.e + 1.8 * beta * q)
    C = 14.2 / alpha + 386 / (1 + 69.9 * q**1.08)
    return L / (L + C * q**2)


def transfer(k, Om):
    """Full transfer function with BAO (EH98 eqs 16-24). k in h/Mpc."""
    p = _shared(Om)
    wm, s, k_eq = p["wm"], p["s"], p["k_eq"]
    fb = OB / Om
    fc = 1 - fb
    kM = np.asarray(k) * H                      # Mpc^-1
    q = kM / (13.41 * k_eq)
    ks = kM * s

    # CDM part
    a1 = (46.9 * wm)**0.670 * (1 + (32.1 * wm)**-0.532)
    a2 = (12.0 * wm)**0.424 * (1 + (45.0 * wm)**-0.582)
    alpha_c = a1**-fb * a2**-(fb**3)
    bb1 = 0.944 / (1 + (458 * wm)**-0.708)
    bb2 = (0.395 * wm)**-0.0266
    beta_c = 1 / (1 + bb1 * (fc**bb2 - 1))
    f = 1 / (1 + (ks / 5.4)**4)
    Tc = f * _T0(q, 1, beta_c) + (1 - f) * _T0(q, alpha_c, beta_c)

    # Baryon part
    y = (1 + p["z_eq"]) / (1 + p["z_d"])
    G = y * (-6 * np.sqrt(1 + y)
             + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + p["R_d"])**-0.75 * G
    beta_node = 8.41 * wm**0.435
    beta_b = 0.5 + fb + (3 - 2 * fb) * np.sqrt((17.2 * wm)**2 + 1)
    k_silk = 1.6 * p["wb"]**0.52 * wm**0.73 * (1 + (10.4 * wm)**-0.95)
    s_tilde = s / (1 + (beta_node / ks)**3)**(1 / 3)
    Tb = (_T0(q, 1, 1) / (1 + (ks / 5.2)**2)
          + alpha_b / (1 + (beta_b / ks)**3) * np.exp(-(kM / k_silk)**1.4)) \
        * np.sinc(kM * s_tilde / np.pi)          # np.sinc(x) = sin(pi x)/(pi x)

    return fb * Tb + fc * Tc


def transfer_nowiggle(k, Om):
    """The smooth 'zero-baryon-oscillation' fit (EH98 eqs 26-31). k in h/Mpc."""
    wm, wb = Om * H**2, OB * H**2
    th = T_CMB / 2.7
    fb = OB / Om
    kM = np.asarray(k) * H
    s = 44.5 * np.log(9.83 / wm) / np.sqrt(1 + 10 * wb**0.75)
    alpha_G = 1 - 0.328 * np.log(431 * wm) * fb + 0.38 * np.log(22.3 * wm) * fb**2
    Gamma = Om * H * (alpha_G + (1 - alpha_G) / (1 + (0.43 * kM * s)**4))
    q = np.asarray(k) * th**2 / Gamma           # eq 28 takes k in h/Mpc
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def tophat(x):
    """Fourier transform of a unit-volume spherical top-hat, W(kR)."""
    x = np.asarray(x, dtype=float)
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3


def sigma_R(P, R):
    """sigma(R)^2 = (1/2 pi^2) int k^2 P(k) W(kR)^2 dk, for a callable P(k)."""
    k = _K_INT
    integrand = k**3 * P(k) * tophat(k * R)**2 / (2 * np.pi**2)
    return np.sqrt(np.trapz(integrand, np.log(k)))


def P_model(k, Om, s8, wiggle=True):
    """Linear P(k) in (Mpc/h)^3 at k in h/Mpc, normalized so sigma_8 = s8."""
    T = transfer if wiggle else transfer_nowiggle
    shape = lambda kk: kk**NS * T(kk, Om)**2
    A = (s8 / sigma_R(shape, 8.0))**2
    return A * shape(np.asarray(k, dtype=float))
