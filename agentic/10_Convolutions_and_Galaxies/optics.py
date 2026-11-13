"""Level-2 optics: star images from the telescope pupil and Kolmogorov turbulence.

Instantaneous PSF = |FFT(pupil * exp(i phi))|^2, with phi a Kolmogorov phase screen;
a long exposure averages many independent screens. On an N x N pupil grid of spacing
dx, one image-plane pixel subtends lambda / (N dx) radians.

Telescope (SDSS 2.5 m, from memory, unverified): primary D = 2.5 m, secondary
1.08 m, four support vanes ~2 cm wide.
"""
import numpy as np
import torch

DEV = "cuda"
D_TEL, D_SEC, VANE = 2.5, 1.08, 0.02            # metres
ARCSEC = np.pi / 180 / 3600
BANDS = {"i": 762.5e-9, "r": 623.1e-9, "g": 477.0e-9}     # effective wavelengths (m)
WIDTHS = {"i": 125e-9, "r": 115e-9, "g": 130e-9}          # approximate band FWHMs (m)


def pupil(N, dx, D=D_TEL, d_sec=D_SEC, vane=VANE, vane_angle=0.0, cross=True):
    """Clear aperture on an N x N grid of spacing dx (m): annulus minus vanes.

    The vanes run along vane_angle (radians, from the x axis) and, if cross, also at
    vane_angle + 90 deg: SDSS's four vanes. cross=False leaves one straight vane line.
    """
    c = (np.arange(N) - N / 2) * dx
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(xx, yy)
    P = (r <= D / 2) & (r >= d_sec / 2)
    ca, sa = np.cos(vane_angle), np.sin(vane_angle)
    u, v = xx * ca + yy * sa, -xx * sa + yy * ca          # coordinates along/across the vane line
    blocked = np.abs(v) < vane / 2
    if cross:
        blocked |= np.abs(u) < vane / 2
    return (P & ~blocked).astype(np.float32)


def psf(P, phase=None):
    """|FFT(P e^{i phase})|^2, centred at [N/2, N/2], normalised to unit sum. Torch, batched."""
    P = torch.as_tensor(P, device=DEV)
    field = P.to(torch.complex64) if phase is None else P * torch.exp(1j * phase)
    I = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1)).abs() ** 2
    return I / I.sum(dim=(-2, -1), keepdim=True)


def kolmogorov(n_screens, N, dx, r0, gen, parts=("hi", "lo")):
    """Kolmogorov phase screens (radians), FFT method plus 3 levels of subharmonics
    (Lane et al. 1992; Schmidt 2010, ch. 9). Phase PSD: 0.023 r0^(-5/3) f^(-11/3)."""
    df = 1.0 / (N * dx)
    f = torch.fft.fftfreq(N, d=dx, device=DEV)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    fr = torch.hypot(fx, fy)
    fr[0, 0] = 1.0
    amp = torch.sqrt(0.023 * r0 ** (-5 / 3) * fr ** (-11 / 3)) * df
    amp[0, 0] = 0.0
    cn = torch.complex(torch.randn(n_screens, N, N, device=DEV, generator=gen),
                       torch.randn(n_screens, N, N, device=DEV, generator=gen)) * amp
    hi = torch.fft.ifft2(cn).real * N * N
    # subharmonics: the low frequencies a finite FFT grid misses
    x = (torch.arange(N, device=DEV) - N / 2) * dx
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    lo = torch.zeros(n_screens, N, N, device=DEV)
    for p in range(1, 4):
        dfp = df / 3**p
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if a == 0 and b == 0:
                    continue
                fp = dfp * np.hypot(a, b)
                ap = np.sqrt(0.023 * r0 ** (-5 / 3) * fp ** (-11 / 3)) * dfp
                c = torch.complex(torch.randn(n_screens, 1, 1, device=DEV, generator=gen),
                                  torch.randn(n_screens, 1, 1, device=DEV, generator=gen)) * ap
                lo += (c * torch.exp(2j * np.pi * (a * dfp * xx + b * dfp * yy))).real
    lo -= lo.mean(dim=(-2, -1), keepdim=True)
    return (hi if "hi" in parts else 0) + (lo if "lo" in parts else 0)


def long_exposure_exact(P, dx, r0):
    """The infinitely long exposure, exactly: OTF = OTF_telescope(rho) exp(-D(rho)/2),
    with the Kolmogorov D(rho) = 6.88 (rho/r0)^(5/3). OTF_telescope is the pupil's
    autocorrelation, the inverse FFT of the diffraction PSF; rho = pupil separation (m).
    Needs N >= 2 x the pupil diameter in grid points so the autocorrelation doesn't wrap."""
    N = P.shape[0]
    otf = torch.fft.ifft2(torch.fft.ifftshift(psf(P), dim=(-2, -1)))
    k = torch.fft.fftfreq(N, device=DEV) * N * dx                  # separations, FFT order
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    otf = otf * torch.exp(-3.44 * (torch.hypot(kx, ky) / r0) ** (5 / 3))
    I = torch.fft.fftshift(torch.fft.fft2(otf), dim=(-2, -1)).real
    return (I / I.sum()).cpu().numpy()


def long_exposure(P, dx, r0, n, gen, batch=16):
    """Average of n instantaneous PSFs through independent screens."""
    N = P.shape[0]
    acc = torch.zeros(N, N, device=DEV)
    for k in range(0, n, batch):
        m = min(batch, n - k)
        acc += psf(P, kolmogorov(m, N, dx, r0, gen)).sum(0)
    return (acc / n).cpu().numpy()


def r0_at(lam, r0_500):
    """Fried parameter scales as lambda^(6/5)."""
    return r0_500 * (lam / 500e-9) ** 1.2


def structure_function(screens, dx, max_lag):
    """D(r) = <(phi(x + r) - phi(x))^2> along both axes, r = 1..max_lag grid steps."""
    lags = np.arange(1, max_lag + 1)
    D = [0.5 * (((screens[..., l:] - screens[..., :-l]) ** 2).mean().item()
                + ((screens[..., l:, :] - screens[..., :-l, :]) ** 2).mean().item()) for l in lags]
    return lags * dx, np.array(D)


def fwhm(img, pix=1.0):
    """FWHM of a centred PSF from the area above half maximum (exact for a round profile)."""
    return 2 * np.sqrt((img >= img.max() / 2).sum() / np.pi) * pix


def to_detector(fine, z, offset=(0.0, 0.0), n_out=120):
    """Resample a centred fine-grid PSF to detector pixels that are 1/z fine pixels wide.

    Each detector pixel integrates the PSF over its area (box filter of width 1/z fine
    pixels), sampled at the pixel centre. The star sits at (n_out - 1)/2 + offset.
    """
    from scipy import ndimage
    N = fine.shape[0]
    boxed = ndimage.uniform_filter(fine, size=max(1, int(round(1 / z))), mode="constant")
    j = np.arange(n_out) - (n_out - 1) / 2
    ys = N / 2 + (j - offset[0]) / z
    xs = N / 2 + (j - offset[1]) / z
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # box mean x pixel area (1/z)^2 fine pixels = flux in the detector pixel (fine sums to 1)
    return ndimage.map_coordinates(boxed, [yy, xx], order=1, mode="constant") / z**2


def blackbody_fractions(T):
    """Photon counts in (i, r, g) from a blackbody at T (K), through Gaussian bands,
    normalised so the three sum to 3 (the convention of sims.star colors)."""
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    out = []
    for band in ["i", "r", "g"]:
        lam = np.linspace(BANDS[band] - 2 * WIDTHS[band], BANDS[band] + 2 * WIDTHS[band], 400)
        trans = np.exp(-0.5 * ((lam - BANDS[band]) / (WIDTHS[band] / 2.355)) ** 2)
        B = 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * T))
        out.append(np.trapezoid(B * lam / (h * c) * trans, lam))
    out = np.array(out)
    return 3 * out / out.sum()
