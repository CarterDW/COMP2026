"""Level-1 simulated stars and artifacts, composited onto real Galaxy Zoo sky.

Pixel model (Lupton et al. 2004, color-preserving asinh):
    I   = mean over channels of the linear flux u_c
    v_c = clip(A * asinh(I) / I * u_c, 0, 255)
so the channel mean is mean(v) = A * asinh(I). The inverse is exact for unclipped
pixels: I = sinh(mean(v) / A), u_c = v_c * I / mean(v). Flux is in units where the
asinh softening is 1, and A is fit to field stars in step 2a.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

N = 120


# --- the stretch ---------------------------------------------------------------------
def stretch(u, A):
    """Linear flux (..., 3) -> pixel values (..., 3), floats in [0, 255]."""
    I = u.mean(axis=-1, keepdims=True)
    gain = np.where(I > 0, A * np.arcsinh(I) / np.where(I > 0, I, 1), A)
    return np.clip(u * gain, 0, 255)


def unstretch(v, A):
    """Pixel values (..., 3) -> linear flux. Exact inverse of stretch where unclipped."""
    v = v.astype(float)
    m = v.mean(axis=-1, keepdims=True)
    I = np.sinh(m / A)
    return np.where(m > 0, v * I / np.where(m > 0, m, 1), 0.0)


def jpeg(v, qtables):
    """Round-trip through JPEG with the survey's own quantization tables."""
    buf = io.BytesIO()
    Image.fromarray(np.round(v).astype(np.uint8)).save(buf, "JPEG", qtables=qtables)
    return np.asarray(Image.open(buf))


# --- sources, in linear flux ------------------------------------------------------------
def moffat(fwhm, beta, x0, y0, flux, n=N, sub=5):
    """Moffat star with total flux `flux`, rendered with sub x sub samples per pixel.

    I(r) = I0 [1 + (r/a)^2]^(-beta),  a = fwhm / (2 sqrt(2^(1/beta) - 1)),
    total flux = pi a^2 I0 / (beta - 1).
    """
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    I0 = flux * (beta - 1) / (np.pi * a**2)
    s = (np.arange(n * sub) + 0.5) / sub - 0.5          # subpixel centers, pixel units
    yy, xx = np.meshgrid(s, s, indexing="ij")
    img = I0 * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a**2) ** (-beta)
    return img.reshape(n, sub, n, sub).mean(axis=(1, 3))


def star(rng, fwhm, beta, flux, color):
    """color = flux fractions in (i, r, g), summing to 3 so I = flux-weighted mean."""
    x0, y0 = (N - 1) / 2 + rng.uniform(-2, 2, 2)
    return moffat(fwhm, beta, x0, y0, flux)[..., None] * np.asarray(color)


def cosmic_ray(rng, flux):
    """A few connected pixels in ONE band: a cosmic ray hits one exposure."""
    u = np.zeros((N, N, 3))
    band = rng.integers(3)
    y, x = (N - 1) // 2 + rng.integers(-2, 3, 2)
    n_pix = rng.integers(1, 4)
    for _ in range(n_pix):                       # a short random walk of hit pixels
        u[y, x, band] += flux / n_pix
        y, x = np.clip((y, x) + rng.integers(-1, 2, 2), 0, N - 1)
    return u


def trail(rng, fwhm, flux_per_px, color=(1.0, 1.0, 1.0)):
    """A straight line through near the center with a Gaussian cross-section of width fwhm."""
    theta = rng.uniform(0, np.pi)
    c0 = (N - 1) / 2 + rng.uniform(-3, 3, 2)
    yy, xx = np.mgrid[:N, :N].astype(float)
    d = -(xx - c0[0]) * np.sin(theta) + (yy - c0[1]) * np.cos(theta)   # distance to the line
    sig = fwhm / 2.355
    prof = flux_per_px * np.exp(-0.5 * (d / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
    return prof[..., None] * np.asarray(color)


def spiked_star(rng, fwhm, beta, flux, color, spike_frac=0.02, arm_width=0.7):
    """A bright star plus a 4-armed cross whose arms fall off as r^-2 (thin vanes)."""
    u = star(rng, fwhm, beta, flux, color)
    phi = rng.uniform(-np.pi / 36, np.pi / 36)          # within 5 deg of the pixel axes
    c = (N - 1) / 2
    yy, xx = np.mgrid[:N, :N].astype(float) - c
    xr, yr = xx * np.cos(phi) + yy * np.sin(phi), -xx * np.sin(phi) + yy * np.cos(phi)
    arms = np.zeros((N, N))
    for along, across in [(xr, yr), (yr, xr)]:
        arms += np.exp(-0.5 * (across / arm_width) ** 2) / (1 + (along / fwhm) ** 2)
    arms *= spike_frac * flux / arms.sum()
    return u + arms[..., None] * np.asarray(color)


# --- real empty sky ---------------------------------------------------------------------
def object_mask(v, nsig=3.0, grow=4):
    """Pixels belonging to any source: smoothed channel mean above sky + nsig sigma."""
    g = ndimage.gaussian_filter(v.astype(float).mean(axis=-1), 1.0)
    sky = np.median(g)
    sig = 1.4826 * np.median(np.abs(g - sky))
    return ndimage.binary_dilation(g > sky + nsig * sig, iterations=grow)


def block_scatter(v, block=8):
    """Std of the 8x8 block means of the channel-mean image: large means patchy sky."""
    g = v.astype(float).mean(axis=-1)
    n = g.shape[0] // block * block
    return g[:n, :n].reshape(n // block, block, n // block, block).mean(axis=(1, 3)).std(ddof=1)


def empty_sky(v, rng, corner=24, block=8, min_blocks=20, max_scatter=3.0):
    """An empty 120x120 sky built only from one real cutout's corner sky.

    The source is the four corner x corner squares, far from the central galaxy and its
    faint halo. Every 8x8 block of the output, on the JPEG's own 8x8 grid, is a copy of
    a random source block containing no masked source pixel, with a random flip or
    rotation (sky noise is isotropic). Returns None if fewer than min_blocks are clean.
    """
    mask = object_mask(v, nsig=2.0, grow=8)
    blocks = []
    for y0 in [0, N - corner]:
        for x0 in [0, N - corner]:
            for by in range(y0, y0 + corner, block):
                for bx in range(x0, x0 + corner, block):
                    if not mask[by:by + block, bx:bx + block].any():
                        blocks.append(v[by:by + block, bx:bx + block])
    if len(blocks) < min_blocks:
        return None
    out = np.empty_like(v)
    for by in range(0, N, block):
        for bx in range(0, N, block):
            b = np.rot90(blocks[rng.integers(len(blocks))], rng.integers(4))
            out[by:by + block, bx:bx + block] = b[:, ::-1] if rng.integers(2) else b
    if block_scatter(out) > max_scatter:          # corners at different sky levels: patchy
        return None
    return out
