"""Step 2a: level-1 stars and artifacts on real sky, calibrated on the cutouts' own field stars."""
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

import data
import fieldstars as fs
import sims

OUT = Path(__file__).parent / "results"
images, labels, ids = data.load()
QT = Image.open(data.DIR / f"{ids[0]}.jpg").quantization       # shared by all 5,000 files
clean = np.load(OUT / "step2a_clean_stars.npy")                 # isolated, round sources


def fit_all(args):
    A, beta = args
    return [fs.fit(images[i], y, x, A=A, beta=beta) for i, y, x in clean]


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- check 1: the stretch inverts exactly --------------------------------------------
    v = images[:200].astype(float)
    ok = (v < 255).all(axis=-1)
    err = np.abs(sims.stretch(sims.unstretch(v, 25.0), 25.0) - v)[ok].max()
    print(f"[stretch] max |stretch(unstretch(v)) - v| on unclipped pixels: {err:.2e}")
    assert err < 1e-9

    # ---- check 2: Moffat flux bookkeeping ----------------------------------------------
    # On a 1200 px frame, compare the grid sum with the analytic flux inside r = 600 px.
    # The corners outside that circle hold < 1e-6 even for beta = 2.5. (On the 120 px
    # frame the square's corners make a circular-aperture formula a poor reference.)
    worst = 0.0
    for fwhm in [2.0, 3.5, 6.0]:
        for beta in [2.5, 4.5]:
            a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
            inside = 1 - (1 + (600 / a) ** 2) ** (1 - beta)
            f = sims.moffat(fwhm, beta, 599.5, 599.5, 1.0, n=1200, sub=2).sum()
            worst = max(worst, abs(f - inside))
    print(f"[flux] max |grid sum - analytic flux within r = 600 px|: {worst:.2e}")

    # ---- the global stretch A and Moffat beta (grid from the field-star fit) -----------
    grid = np.load(OUT / "step2a_Abeta_grid.npy")
    A, BETA = grid[grid[:, 2].argmin(), :2]
    print(f"[global] A = {A:.0f}, beta = {BETA:.1f} (best of a {len(grid)}-point grid)")
    with Pool(16) as p:
        fits = p.map(fit_all, [(A, BETA)])[0]
    P = np.array([f[0] for f in fits])
    rms = np.array([f[2] for f in fits])
    fwhm, logF, sky_b = P[:, 5], P[:, 1], P[:, 2]

    # galaxy size in each image: the central object's footprint, as an equivalent radius
    def r_gal(v):
        from scipy import ndimage
        lab, _ = ndimage.label(sims.object_mask(v, grow=1))
        return np.sqrt((lab == lab[60, 60]).sum() / np.pi) if lab[60, 60] else np.nan
    rg = np.array([r_gal(images[i]) for i in clean[:, 0]])
    good = np.isfinite(rg)
    rho, pval = stats.spearmanr(rg[good], fwhm[good])
    print(f"[PSF] {len(fwhm)} field stars: FWHM median {np.median(fwhm):.2f} px, "
          f"16-84%: {np.percentile(fwhm, 16):.2f}-{np.percentile(fwhm, 84):.2f}")
    print(f"[PSF] Spearman(central galaxy size, star FWHM) = {rho:+.2f} (p = {pval:.1e})")

    # star colors: linear-flux channel fractions within r < 3 px, sky-subtracted
    yy, xx = np.mgrid[-fs.HALF:fs.HALF + 1, -fs.HALF:fs.HALF + 1]
    core = xx**2 + yy**2 < 9
    colors = []
    for (i, y, x) in clean:
        u = sims.unstretch(fs.stamp(images[i], y, x), A)
        bg = np.median(u[~(xx**2 + yy**2 < 49)], axis=0)
        s = np.clip((u[core] - bg).sum(axis=0), 1e-6, None)
        colors.append(3 * s / s.sum())
    colors = np.array(colors)
    print(f"[color] star (i, r, g) fractions, median: {np.median(colors, axis=0).round(2)}")

    # ---- empty sky backgrounds ----------------------------------------------------------
    bgs, src = [], []
    for i in rng.permutation(len(images)):
        b = sims.empty_sky(images[i], rng)
        if b is not None:
            bgs.append(b)
            src.append(i)
        if len(bgs) == 300:
            break

    def sky_stats(patches):
        """3-sigma-clipped mean and std per patch and channel, averaged over patches.
        (A median/MAD of integer pixels is itself an integer, too coarse to compare.)"""
        px = patches.reshape(len(patches), -1, 3).astype(float)
        cl = [[stats.sigmaclip(q[:, c], 3, 3)[0] for c in range(3)] for q in px]
        return (np.mean([[c.mean() for c in q] for q in cl], axis=0),
                np.mean([[c.std() for c in q] for q in cl], axis=0))

    # References from the same cutouts the backgrounds came from: (a) every pixel of the
    # 12 px corners, faint unmasked sources included; (b) only the clean 24 px corners,
    # the pixels the backgrounds are actually built from. (b) tests the noise model;
    # (a) - (b) is what faint sources add to real sky and the simulations lack.
    corner_slices = [(slice(0, 24), slice(0, 24)), (slice(0, 24), slice(96, 120)),
                     (slice(96, 120), slice(0, 24)), (slice(96, 120), slice(96, 120))]
    clean_corners = [v[sy, sx] for v in images[src] for sy, sx in corner_slices
                     if not sims.object_mask(v, 2.0, 8)[sy, sx].any()]
    s = images[src]
    all_m, all_s = sky_stats(np.concatenate([s[:, :12, :12], s[:, :12, -12:],
                                             s[:, -12:, :12], s[:, -12:, -12:]], axis=1))
    real_m, real_s = sky_stats(np.stack(clean_corners))
    bg_m, bg_s = sky_stats(np.stack(bgs))
    print(f"[sky] {len(bgs)} backgrounds, {len(clean_corners)} clean real corners")
    print(f"[sky]   real, all corner pixels: mean {all_m.round(2)}, std {all_s.round(2)}")
    print(f"[sky]   real, clean corners    : mean {real_m.round(2)}, std {real_s.round(2)}")
    print(f"[sky]   empty_sky              : mean {bg_m.round(2)}, std {bg_s.round(2)}")

    # patchiness: 8x8 block-mean scatter, simulated frames vs real clean 24 px corners
    real_bs = [sims.block_scatter(c) for c in clean_corners]
    sim_bs = [sims.block_scatter(b) for b in bgs]
    print(f"[sky] block scatter median: real corners {np.median(real_bs):.2f}, "
          f"empty_sky {np.median(sim_bs):.2f}; 99th pct {np.percentile(real_bs, 99):.1f} vs "
          f"{np.percentile(sim_bs, 99):.1f}")

    # ---- compositing ----------------------------------------------------------------------
    def composite(bg, u_src):
        return sims.jpeg(sims.stretch(sims.unstretch(bg, A) + u_src, A), QT)

    F_field = np.exp(logF)

    def random_star(rng, F=None):
        return sims.star(rng, rng.choice(fwhm), BETA,
                         F if F is not None else np.exp(rng.uniform(np.log(F_field.min()) - 1,
                                                                    np.log(F_field.max()))),
                         colors[rng.integers(len(colors))])

    # ---- check 3: injection-recovery on real field stars ---------------------------------
    rec = []
    for k in rng.choice(len(clean), 60, replace=False):
        bg = bgs[rng.integers(len(bgs))]
        u = sims.star(rng, fwhm[k], BETA, F_field[k], colors[k])
        v_sim = composite(bg, u)
        c9 = v_sim[55:64, 55:64].mean(axis=-1)            # the star is injected within 2 px of center
        y0, x0 = np.add(np.unravel_index(c9.argmax(), c9.shape), 55)
        p_rec = fs.fit(v_sim, y0, x0, A=A, beta=BETA)[0]
        rec.append([fwhm[k], p_rec[5], logF[k], p_rec[1]])
    rec = np.array(rec)
    dfw = rec[:, 1] / rec[:, 0] - 1
    dF = np.exp(rec[:, 3] - rec[:, 2]) - 1
    print(f"[inject] recovered/injected - 1: FWHM median {np.median(dfw):+.3f} "
          f"(16-84 {np.percentile(dfw, 16):+.3f}..{np.percentile(dfw, 84):+.3f}), "
          f"flux median {np.median(dF):+.3f} (16-84 {np.percentile(dF, 16):+.3f}..{np.percentile(dF, 84):+.3f})")

    # ---- figures ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(fwhm, 40, color="0.4")
    axes[0].set_xlabel("field-star FWHM (px)")
    axes[0].set_ylabel("stars")
    axes[1].scatter(rg[good], fwhm[good], s=4, alpha=0.5)
    axes[1].set_xlabel("central galaxy radius (px)")
    axes[1].set_ylabel("star FWHM (px)")
    axes[1].set_title(f"Spearman {rho:+.2f}", fontsize=9)
    gr = grid[:, 2] - grid[:, 2].min()
    Au, Bu = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    im = axes[2].pcolormesh(np.arange(len(Bu) + 1), np.arange(len(Au) + 1),
                            np.log10(1 + gr.reshape(len(Au), len(Bu))), cmap="viridis_r")
    axes[2].set_xticks(np.arange(len(Bu)) + 0.5, Bu)
    axes[2].set_yticks(np.arange(len(Au)) + 0.5, Au.astype(int))
    axes[2].set_xlabel(r"Moffat $\beta$")
    axes[2].set_ylabel("stretch A")
    fig.colorbar(im, ax=axes[2], label=r"log$_{10}$(1 + $\Delta\chi^2$)")
    fig.tight_layout()
    fig.savefig(OUT / "step2a_psf.png", dpi=150)

    # real vs simulated stars, matched in fitted parameters
    order = np.argsort(F_field)
    show = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(3, 6, figsize=(11, 5.8))
    for col, k in enumerate(show):
        i, y, x = clean[k]
        real = np.pad(images[i], ((12, 12), (12, 12), (0, 0)))[y:y + 25, x:x + 25]
        sim = composite(bgs[col], sims.star(rng, fwhm[k], BETA, F_field[k], colors[k]))[48:73, 48:73]
        axes[0, col].imshow(real)
        axes[1, col].imshow(sim)
        r = np.hypot(*np.mgrid[-12:13, -12:13])
        for img, sty, name in [(real, "o", "real"), (sim, "-", "sim")]:
            g = img.astype(float).mean(axis=-1)
            rb = np.arange(0, 10)
            prof = [g[(r >= a) & (r < a + 1)].mean() for a in rb]
            axes[2, col].plot(rb + 0.5, prof, sty, ms=3, label=name)
        axes[0, col].set_title(f"FWHM {fwhm[k]:.1f}, lnF {logF[k]:.1f}", fontsize=7)
        axes[2, col].set_xlabel("r (px)", fontsize=7)
        for a in axes[:2, col]:
            a.axis("off")
    axes[0, 0].text(-4, 12, "real", rotation=90, va="center")
    axes[1, 0].text(-4, 12, "simulated", rotation=90, va="center")
    axes[2, 0].set_ylabel("mean pixel value", fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "step2a_real_vs_sim.png", dpi=150)

    # the gallery
    conf = np.flatnonzero(labels < 2)
    rows = {
        "real galaxy": [images[i] for i in rng.choice(conf, 6, replace=False)],
        "star": [composite(bgs[k], random_star(rng, F)) for k, F in
                 zip(range(6), np.exp(np.linspace(np.log(F_field.min()), np.log(F_field.max()) + 1, 6)))],
        "cosmic ray": [composite(bgs[6 + k], sims.cosmic_ray(rng, f)) for k, f in
                       enumerate(np.geomspace(100, 1e4, 6))],
        "satellite trail": [composite(bgs[12 + k], sims.trail(rng, rng.choice(fwhm), f)) for k, f in
                            enumerate(np.geomspace(2, 60, 6))],
        "spiked star": [composite(bgs[18 + k], sims.spiked_star(rng, rng.choice(fwhm), BETA, F,
                                                                 colors[rng.integers(len(colors))]))
                        for k, F in enumerate(np.geomspace(3e3, 1e5, 6))],
        "empty sky": [composite(bgs[24 + k], 0.0) for k in range(6)],
    }
    fig, axes = plt.subplots(len(rows), 6, figsize=(11, 11.5))
    for r_i, (name, ims_row) in enumerate(rows.items()):
        for c_i, img in enumerate(ims_row):
            axes[r_i, c_i].imshow(img)
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])
        axes[r_i, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Real galaxies and simulated objects on real SDSS sky (faint to bright, left to right)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "step2a_gallery.png", dpi=130)

    np.savez(OUT / "step2a_calibration.npz", A=A, beta=BETA, fwhm=fwhm, logF=logF, colors=colors)

    # ---- remaining checks ----------------------------------------------------------------
    assert worst < 1e-6, "Moffat grid sum should match the analytic flux"
    assert abs(np.median(dfw)) < 0.05 and abs(np.median(dF)) < 0.05, "injection-recovery is biased"
    assert np.all(np.abs(bg_m - real_m) < 1.0), "empty-sky level should match real corner sky within 1 count"
    assert np.all(np.abs(bg_s / real_s - 1) < 0.10), "empty-sky noise should match real corner sky within 10%"
    assert abs(np.median(sim_bs) / np.median(real_bs) - 1) < 0.10, "simulated sky is patchier than real"
    print("PASS: stretch inverse, flux bookkeeping, unbiased injection-recovery, sky level, "
          "noise and patchiness match.")
