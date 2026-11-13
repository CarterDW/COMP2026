"""Fix the two bright-star differences between simulated and real SDSS stars.

1. Saturation: compare how colored saturated cores are (chroma = channel std / mean
   over pixels with max channel >= 250), real vs simulated, per-channel clipping vs
   SDSS's color-preserving saturation.
2. Spikes: the m = 4 Fourier amplitude of the light around bright stars (8 < r < 25 px),
   which doesn't care which way the spikes point. Fit the vane width so simulated
   bright stars match real ones.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import data
import optics as o
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
cal = np.load(OUT / "step3_calibration.npz")
A = float(cal["A"])
rows = list(csv.DictReader(open(SD / "stars" / "index.csv")))
stars = np.stack([np.asarray(Image.open(SD / "stars" / f"{r['name']}.jpg")) for r in rows])
mag = np.array([float(r["psfMag_r"]) for r in rows])
f_cat = np.mean([10 ** (-0.4 * (np.array([float(r[f"psfMag_{b}"]) for r in rows]) - 22.5))
                 for b in "gri"], axis=0)
QT = Image.open(SD / "stars" / f"{rows[0]['name']}.jpg").quantization
sky_names = np.load(OUT / "step3_sky_usable.npy")
sky = np.stack([np.asarray(Image.open(SD / "sky" / f"{n}.jpg")) for n in sky_names[:600]])
N_FINE, THETA = 2048, 0.018
FW_STAR = float(np.median(cal["fwhm"]))
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def centre(v):
    c = np.unravel_index(v[50:70, 50:70].mean(-1).argmax(), (20, 20))
    return c[0] + 50, c[1] + 50


def chroma(v):
    """Median channel std / mean over saturated pixels (max channel >= 250)."""
    v = v.astype(float)
    sat = v.max(axis=-1) >= 250
    if sat.sum() < 3:
        return np.nan
    px = v[sat]
    return np.median(px.std(axis=-1) / np.maximum(px.mean(axis=-1), 1))


def m4(v, cy, cx, r_in=8, r_out=25):
    """|sum p(theta) e^{-4 i theta}| / sum p(theta) of the sky-subtracted light in an annulus."""
    g = v.astype(float).mean(-1)
    g = g - np.median(g)
    dy, dx = np.mgrid[:sims.N, :sims.N] - np.array([cy, cx])[:, None, None]
    r, th = np.hypot(dy, dx), np.arctan2(dy, dx)
    ring = (r > r_in) & (r < r_out) & (v.max(axis=-1) < 250)
    bins = np.linspace(-np.pi, np.pi, 73)
    p = np.array([g[ring & (th >= a) & (th < b)].mean() for a, b in zip(bins[:-1], bins[1:])])
    p = np.clip(np.nan_to_num(p), 0, None)
    t = (bins[:-1] + bins[1:]) / 2
    return np.abs((p * np.exp(-4j * t)).sum()) / max(p.sum(), 1e-9)


def bank_for_vane(vane):
    """r-band-scaled (i, r, g) detector PSFs at the median star FWHM, for one vane width."""
    lam_r = o.BANDS["r"]
    r0_r = 0.98 * lam_r / (1.4 * o.ARCSEC)
    r0_500 = r0_r * (500e-9 / lam_r) ** 1.2
    fine = {}
    for band, lam in o.BANDS.items():
        dx = lam / (N_FINE * THETA * o.ARCSEC)
        fine[band] = o.long_exposure_exact(o.pupil(N_FINE, dx, vane=vane), dx, o.r0_at(lam, r0_500))
    z = FW_STAR / (o.fwhm(fine["r"], THETA) / THETA)
    return np.stack([o.to_detector(fine[b], z) for b in ["i", "r", "g"]])


def sim_star(psf3, F, color, rng, color_saturation=True):
    u = sims.unstretch(sky[rng.integers(len(sky))], A) + F * np.moveaxis(psf3, 0, -1) * color
    u = np.roll(u, tuple(rng.integers(-2, 3, 2)), axis=(0, 1))
    return sims.jpeg(sims.stretch(u, A, color_saturation), QT)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bright = np.flatnonzero(mag < 16.5)
    # flux scale: catalog flux -> image flux units, from the step-3 calibration fits
    k_flux = np.exp(np.median(cal["logF"] - np.log(f_cat[cal["idx"]])))
    say(f"[setup] {len(bright)} bright real stars (r < 16.5); image flux = {k_flux:.1f} x catalog flux")

    # real star colors: (i, r, g) fractions from catalog fluxes
    def color_of(k):
        f = np.array([10 ** (-0.4 * (float(rows[k][f"psfMag_{b}"]) - 22.5)) for b in "irg"])
        return 3 * f / f.sum()

    # ---- 1. saturation ---------------------------------------------------------------
    psf_now = bank_for_vane(o.VANE)
    chroma_real = np.array([chroma(stars[k]) for k in bright])
    chroma_clip, chroma_keep = [], []
    for k in bright:
        F = k_flux * f_cat[k]
        chroma_clip.append(chroma(sim_star(psf_now, F, color_of(k), rng, color_saturation=False)))
        chroma_keep.append(chroma(sim_star(psf_now, F, color_of(k), rng, color_saturation=True)))
    cr, cc, ck = (np.nanmedian(x) for x in (chroma_real, chroma_clip, chroma_keep))
    say(f"[saturation] chroma of saturated cores (0 = white): real {cr:.3f}, "
        f"sim per-channel clip {cc:.3f}, sim color-preserving {ck:.3f}")

    # ---- 2. spikes ------------------------------------------------------------------------
    m4_real = np.array([m4(stars[k], *centre(stars[k])) for k in bright])
    vanes = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    m4_sim = {}
    for w in vanes:
        psf = bank_for_vane(w)
        vals = []
        for k in bright[:200]:
            v = sim_star(psf, k_flux * f_cat[k], color_of(k), rng)
            vals.append(m4(v, *centre(v)))
        m4_sim[w] = np.median(vals)
        say(f"[spikes] vane width {1000 * w:4.0f} mm: simulated m=4 amplitude {m4_sim[w]:.4f}")
    m4r = np.median(m4_real)
    say(f"[spikes] real bright stars: m=4 amplitude {m4r:.4f}  (pure-noise floor: see vane 0)")
    ws = np.array(vanes)
    ms = np.array([m4_sim[w] for w in vanes])
    w_fit = float(np.interp(m4r, ms, ws)) if np.all(np.diff(ms) > 0) else float(ws[np.argmin(np.abs(ms - m4r))])
    say(f"[spikes] vane width matching the real stars: {1000 * w_fit:.1f} mm (step 2b assumed 20 mm)")
    np.savez(OUT / "step3_satspikes.npz", vane=w_fit, m4_real=m4_real, chroma_real=chroma_real)

    # ---- figure: real bright stars next to simulated ones, old vs new --------------------------
    psf_fit = bank_for_vane(w_fit)
    show = bright[np.argsort(mag[bright])][np.linspace(0, len(bright) - 1, 6).astype(int)]
    fig, axes = plt.subplots(3, 6, figsize=(13, 7), constrained_layout=True)
    for c, k in enumerate(show):
        F, col = k_flux * f_cat[k], color_of(k)
        panels = [stars[k], sim_star(psf_now, F, col, np.random.default_rng(c), color_saturation=False),
                  sim_star(psf_fit, F, col, np.random.default_rng(c))]
        for r_, img in enumerate(panels):
            axes[r_, c].imshow(img[30:90, 30:90])
            axes[r_, c].set_xticks([])
            axes[r_, c].set_yticks([])
        axes[0, c].set_title(f"r = {mag[k]:.1f}, {rows[k]['subclass'][:3]}", fontsize=9)
    axes[0, 0].set_ylabel("real SDSS star")
    axes[1, 0].set_ylabel("simulated, before\n(clip, 20 mm vanes)")
    axes[2, 0].set_ylabel(f"simulated, after\n(color kept, {1000 * w_fit:.0f} mm vanes)")
    fig.suptitle("Bright stars at the same catalog brightness and color")
    fig.savefig(OUT / "step3_satspikes.png", dpi=130)
    (OUT / "step3_satspikes_log.txt").write_text("\n".join(log) + "\n")
