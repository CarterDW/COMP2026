"""Three corrections to the simulated stars, each fit to the real SDSS stars.

1. Channel gains b_c: image channel flux / catalog flux per band, from unsaturated stars.
2. PSF halo: PSF = (1 - eps) Kolmogorov core + eps Moffat(fwhm_h, beta_h), fit to the
   unsaturated outer rings (6-25 px) of bright stars, each star's flux free.
3. Vane width: matching the axis/diagonal light ratio of bright stars (this assumes
   real spikes, if any, lie along the image axes).
Writes results/step3_psf_bank_v2.npz (bank with the fitted vane width and halo) and
results/step3_psffix.npz (gains, halo, vane).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import optimize

import data
import optics as o
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
cal = np.load(OUT / "step3_calibration.npz")
A = float(cal["A"])
FW_STAR = float(np.median(cal["fwhm"]))
N_FINE, THETA = 2048, 0.018
rows = list(csv.DictReader(open(SD / "stars" / "index.csv")))
stars = np.stack([np.asarray(Image.open(SD / "stars" / f"{r['name']}.jpg")) for r in rows])
mag = np.array([float(r["psfMag_r"]) for r in rows])
fcat = np.stack([10 ** (-0.4 * (np.array([float(r[f"psfMag_{b}"]) for r in rows]) - 22.5))
                 for b in "irg"], axis=1)                        # (N, 3) in (i, r, g) order
QT = Image.open(SD / "stars" / f"{rows[0]['name']}.jpg").quantization
sky_names = np.load(OUT / "step3_sky_usable.npy")
sky = np.stack([np.asarray(Image.open(SD / "sky" / f"{n}.jpg")) for n in sky_names[:600]])
peak_val = stars[:, 50:70, 50:70].max(axis=(1, 2, 3))
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def centre(v):
    c = np.unravel_index(v[50:70, 50:70].mean(-1).argmax(), (20, 20))
    return c[0] + 50, c[1] + 50


def radii(cy, cx):
    return np.hypot(*(np.mgrid[:sims.N, :sims.N] - np.array([cy, cx])[:, None, None]))


def fine_psfs(vane):
    lam_r = o.BANDS["r"]
    r0_500 = 0.98 * lam_r / (1.4 * o.ARCSEC) * (500e-9 / lam_r) ** 1.2
    out = {}
    for band, lam in o.BANDS.items():
        dx = lam / (N_FINE * THETA * o.ARCSEC)
        out[band] = o.long_exposure_exact(o.pupil(N_FINE, dx, vane=vane), dx, o.r0_at(lam, r0_500))
    return out


def detector_psfs(fine, fw_px, offset=(0.0, 0.0), halo=None):
    """(3, N, N) detector PSFs in (i, r, g); halo = (eps, fwhm_h, beta_h) or None."""
    z = fw_px / (o.fwhm(fine["r"], THETA) / THETA)
    core = np.stack([o.to_detector(fine[b], z, offset) for b in ["i", "r", "g"]])
    if halo is None or halo[0] == 0:
        return core
    eps, fwh, bh = halo
    h = sims.moffat(fwh, bh, 59.5 + offset[1], 59.5 + offset[0], 1.0)
    return (1 - eps) * core + eps * h[None]


if __name__ == "__main__":
    # ---- 1. channel gains from unsaturated stars ------------------------------------------
    unsat = np.flatnonzero((peak_val < 240) & (mag > 16))
    meas = []
    for k in unsat:
        cy, cx = centre(stars[k])
        u = sims.unstretch(stars[k], A)
        r = radii(cy, cx)
        bg = np.median(u[(r > 30) & (r < 50)], axis=0)
        meas.append((u[r < 6] - bg).sum(axis=0))
    meas = np.array(meas)
    good = np.flatnonzero(np.all(meas > 0, axis=1))
    lg = np.log(meas[good] / fcat[unsat][good])                 # ln(image / catalog) per channel
    gains = np.exp(np.median(lg - lg[:, [1]], axis=0))          # relative to r
    col_meas = np.log(meas[good][:, 2] / meas[good][:, 0])      # ln(g/i), image
    col_cat = np.log(fcat[unsat][good][:, 2] / fcat[unsat][good][:, 0])
    col_slope = np.polyfit(col_cat, col_meas, 1)[0]
    say(f"[gains] {len(good)} unsaturated stars: median image/catalog gains (i, r, g) relative to r = "
        f"{gains.round(3)}; image ln(g/i) vs catalog ln(g/i) slope {col_slope:.3f}: the JPG "
        f"compresses colors")
    # offset + slope per channel, fit on half the stars, tested on the other half
    perm = np.random.default_rng(0).permutation(len(good))
    fit_i, test_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    y = np.log(meas[good] / meas[good][:, [1]])
    x = np.log(fcat[unsat][good] / fcat[unsat][good][:, [1]])
    offset, slope = np.zeros(3), np.ones(3)
    for c in (0, 2):
        slope[c], offset[c] = np.polyfit(x[fit_i, c], y[fit_i, c], 1)
    pred = sims.image_colors(fcat[unsat][good][test_i], offset, slope)
    true = 3 * meas[good][test_i] / meas[good][test_i].sum(axis=1, keepdims=True)
    rms_map = np.sqrt(np.mean(np.log(pred / true)[:, [0, 2]] ** 2))
    rms_gain = np.sqrt(np.mean(np.log(sims.image_colors(fcat[unsat][good][test_i], np.log(gains),
                                                        np.ones(3)) / true)[:, [0, 2]] ** 2))
    rms_none = np.sqrt(np.mean(np.log((3 * fcat[unsat][good][test_i] / fcat[unsat][good][test_i]
                                       .sum(1, keepdims=True)) / true)[:, [0, 2]] ** 2))
    say(f"[colors] map ln(s_c/s_r) = offset + slope ln(f_c/f_r): i {offset[0]:+.3f} + {slope[0]:.3f} x, "
        f"g {offset[2]:+.3f} + {slope[2]:.3f} x")
    say(f"[colors] held-out stars, rms error in ln(channel fraction): catalog colors as-is {rms_none:.3f}, "
        f"gains only {rms_gain:.3f}, offset + slope {rms_map:.3f}")
    col_ok = rms_map < rms_gain < rms_none

    # ---- 2. PSF halo from the outer rings of bright stars ------------------------------------
    bright = np.flatnonzero(mag < 16.5)
    fine20 = fine_psfs(o.VANE)
    core_r = detector_psfs(fine20, FW_STAR)[1]
    rb = np.arange(6, 25)

    def radial(img2d):
        r = radii(59.5, 59.5)
        return np.array([img2d[(r >= a) & (r < a + 1)].mean() for a in rb])

    core_prof = radial(core_r)
    prof, skyb = [], []
    for k in bright:
        cy, cx = centre(stars[k])
        g = stars[k].astype(float).mean(-1)
        r = radii(cy, cx)
        ok = stars[k].max(-1) < 250
        prof.append([np.median(g[(r >= a) & (r < a + 1) & ok]) if ((r >= a) & (r < a + 1) & ok).sum() > 5
                     else np.nan for a in rb])
        u = sims.unstretch(stars[k], A).mean(-1)
        skyb.append(np.median(u[(r > 40) & (r < 55)]))
    prof, skyb = np.array(prof), np.array(skyb)

    def chi2_for(halo):
        hprof = radial(sims.moffat(halo[1], halo[2], 59.5, 59.5, 1.0)) if halo[0] > 0 else 0
        P = (1 - halo[0]) * core_prof + halo[0] * hprof
        tot, Fs = 0.0, []
        for p, b in zip(prof, skyb):
            ok = np.isfinite(p)
            f = lambda lf: np.sum((A * np.arcsinh(np.exp(lf) * P[ok] + b) - p[ok]) ** 2)
            res = optimize.minimize_scalar(f, bounds=(0, 20), method="bounded")
            tot += res.fun
            Fs.append(res.x)
        return tot, np.array(Fs)

    grid = [(0.0, 0, 0)] + [(e, m * FW_STAR, bh) for e in [0.03, 0.06, 0.1, 0.15, 0.2, 0.3]
                            for m in [2, 4, 8] for bh in [1.5, 2.0, 3.0]]
    scores = [(h, *chi2_for(h)) for h in grid]
    best = min(scores, key=lambda s: s[1])
    halo, chi_best, logF_bright = best
    chi_none = scores[0][1]
    say(f"[halo] best eps = {halo[0]}, halo FWHM = {halo[1]:.1f} px, beta = {halo[2]}; "
        f"chi^2 {chi_best:.3g} vs {chi_none:.3g} with no halo ({len(bright)} bright stars, r = 6-25 px)")

    # ---- 3. vane width from the axis/diagonal ratio ----------------------------------------------
    def axis_diag(v, cy, cx):
        g = v.astype(float).mean(-1)
        g = g - np.median(g)
        dy, dx = np.mgrid[:sims.N, :sims.N] - np.array([cy, cx])[:, None, None]
        r = np.hypot(dy, dx)
        ring = (r > 8) & (r < 25) & (v.max(-1) < 250)
        ax_ = ring & ((np.abs(dx) < 1.5) | (np.abs(dy) < 1.5))
        dg = ring & ((np.abs(dx - dy) < 2.1) | (np.abs(dx + dy) < 2.1))
        return g[ax_].mean() / max(g[dg].mean(), 1e-3)

    spk = bright[mag[bright] < 16][:250]
    real_ratio = np.median([axis_diag(stars[k], *centre(stars[k])) for k in spk])
    col = lambda k: sims.image_colors(fcat[k], offset, slope)
    rng = np.random.default_rng(0)
    pos = {k: j for j, k in enumerate(bright)}
    ratios = {}
    for w in [0.0, 0.002, 0.005, 0.01, 0.02]:
        psf = detector_psfs(fine_psfs(w) if w != o.VANE else fine20, FW_STAR, halo=halo)
        vals = []
        for k in spk:
            u = sims.unstretch(sky[rng.integers(len(sky))], A) + \
                np.exp(logF_bright[pos[k]]) * np.moveaxis(psf, 0, -1) * col(k)
            v = sims.jpeg(sims.stretch(u, A), QT)
            vals.append(axis_diag(v, *centre(v)))
        ratios[w] = np.median(vals)
        say(f"[vanes] {1000 * w:4.0f} mm: simulated axis/diagonal {ratios[w]:.3f}  (real {real_ratio:.3f})")
    ws = np.array(list(ratios))
    rs = np.array(list(ratios.values()))
    vane = float(ws[np.argmin(np.abs(rs - real_ratio))])
    say(f"[vanes] chosen vane width {1000 * vane:.0f} mm (step 2b assumed {1000 * o.VANE:.0f} mm)")

    # ---- the v2 bank ----------------------------------------------------------------------------
    fine_v = fine_psfs(vane)
    FW_GRID = np.arange(2.0, 10.01, 0.5)
    OFFS = [(dy, dx) for dy in (-1 / 3, 0, 1 / 3) for dx in (-1 / 3, 0, 1 / 3)]
    bank = np.zeros((len(FW_GRID), len(OFFS), 3, sims.N, sims.N), np.float32)
    for i_f, fw in enumerate(FW_GRID):
        h = (halo[0], halo[1] / FW_STAR * fw, halo[2])       # halo scales with the seeing
        for i_o, off in enumerate(OFFS):
            bank[i_f, i_o] = detector_psfs(fine_v, fw, off, halo=h)
    np.savez_compressed(OUT / "step3_psf_bank_v2.npz", psf=bank, fwhm=FW_GRID)
    np.savez(OUT / "step3_psffix.npz", gains=gains, color_offset=offset, color_slope=slope, eps=halo[0], halo_fwhm_ratio=halo[1] / FW_STAR,
             halo_beta=halo[2], vane=vane)

    # ---- before/after figure -------------------------------------------------------------------
    psf_old = detector_psfs(fine20, FW_STAR)
    psf_new = bank[int(np.argmin(np.abs(FW_GRID - FW_STAR))), 4]
    show = bright[np.argsort(mag[bright])][np.linspace(0, len(bright) - 1, 6).astype(int)]
    fig, axes = plt.subplots(3, 6, figsize=(13, 7), constrained_layout=True)
    for c, k in enumerate(show):
        F = np.exp(logF_bright[pos[k]])
        old_col = 3 * fcat[k] / fcat[k].sum()
        imgs = [stars[k]]
        for psf, cc, sat in [(psf_old, old_col, False), (psf_new, col(k), True)]:
            rr = np.random.default_rng(c)
            u = sims.unstretch(sky[rr.integers(len(sky))], A) + F * np.moveaxis(psf, 0, -1) * cc
            imgs.append(sims.jpeg(sims.stretch(u, A, sat), QT))
        for r_, img in enumerate(imgs):
            axes[r_, c].imshow(img[30:90, 30:90])
            axes[r_, c].set_xticks([])
            axes[r_, c].set_yticks([])
        axes[0, c].set_title(f"r = {mag[k]:.1f}, {rows[k]['subclass'][:3]}", fontsize=9)
    axes[0, 0].set_ylabel("real SDSS star")
    axes[1, 0].set_ylabel("simulated, before")
    axes[2, 0].set_ylabel("simulated, after\n(gains, halo, vanes,\ncolor-kept saturation)")
    fig.suptitle("Bright stars: real vs simulated at the fitted brightness")
    fig.savefig(OUT / "step3_psffix.png", dpi=130)

    # stacked outer profiles, real vs model
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    med_real = np.nanmedian(prof, 0)
    for h, lab, ls in [((0.0, 0, 0), "core only", "--"), (halo, "core + halo", "-")]:
        hp = radial(sims.moffat(h[1], h[2], 59.5, 59.5, 1.0)) if h[0] > 0 else 0
        P = (1 - h[0]) * core_prof + h[0] * hp
        _, Fs = chi2_for(h)
        model = np.median([A * np.arcsinh(np.exp(f) * P + b) for f, b in zip(Fs, skyb)], 0)
        ax.plot(rb + 0.5, model, ls, label=lab)
    ax.plot(rb + 0.5, med_real, "o", label="real bright stars (median)")
    ax.set_xlabel("radius (px)")
    ax.set_ylabel("pixel value")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Outer profile of bright stars (r < 16.5)")
    fig.savefig(OUT / "step3_psffix_halo.png", dpi=140)
    (OUT / "step3_psffix_log.txt").write_text("\n".join(log) + "\n")
    assert col_ok, "the color mapping should beat gains alone, which should beat raw catalog colors"
    assert chi_best < chi_none, "the halo should improve the outer profiles"
    print("PASS: color mapping predicts held-out star colors best; the halo improves the outer profiles.")
