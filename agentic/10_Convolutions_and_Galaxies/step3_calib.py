"""Recalibrate the stretch on DR16, using the catalog fluxes of the real SDSS stars.

1. Linearity: at the right stretch A, the fitted (linear) flux of each star is
   proportional to its catalog flux, so log F_fit vs log f_cat has slope exactly 1.
2. At that A, pick Moffat beta by chi^2 and remeasure the star FWHMs.
3. Real vs simulated stars, stacked radial profiles per magnitude bin, and spike
   strength (axes vs diagonals) for bright stars.
"""
import csv
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import data
import fieldstars as fs
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
A_GRID = [14, 16, 18, 20, 22, 25, 30, 36, 44]
B_GRID = [2.0, 2.5, 3.0, 3.5, 4.5, 6.0]

rows = list(csv.DictReader(open(SD / "stars" / "index.csv")))
stars = np.stack([np.asarray(Image.open(SD / "stars" / f"{r['name']}.jpg")) for r in rows])
mag_r = np.array([float(r["psfMag_r"]) for r in rows])
# the image's I is the mean of the three scaled bands, so compare with the mean g, r, i flux
f_cat = np.mean([10 ** (-0.4 * (np.array([float(r[f"psfMag_{b}"]) for r in rows]) - 22.5))
                 for b in "gri"], axis=0)                  # nanomaggies
peaks = np.array([np.unravel_index(v[50:70, 50:70].mean(-1).argmax(), (20, 20)) for v in stars]) + 50
peak_val = stars[:, 50:70, 50:70].max(axis=(1, 2, 3))
rng0 = np.random.default_rng(0)
UNSAT = np.flatnonzero(peak_val < 240)                      # saturated cores bias the fit low
SUB = rng0.permutation(UNSAT)[:1200]


def fit_set(args):
    A, beta, idx = args
    return [fs.fit(stars[k], peaks[k, 0], peaks[k, 1], A=A, beta=beta) for k in idx]


def slope(A, beta, fits):
    logF = np.array([f[0][1] for f in fits])               # fitted ln F
    x = np.log(f_cat[SUB])
    ok = np.isfinite(logF)
    p = np.polyfit(x[ok], logF[ok], 1)
    return p[0], np.std(logF[ok] - np.polyval(p, x[ok]))


if __name__ == "__main__":
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # ---- 1. linearity over the (A, beta) grid --------------------------------------------
    grid = [(A, b, SUB) for A in A_GRID for b in B_GRID]
    with Pool(16) as p:
        all_fits = p.map(fit_set, grid)
    table = []
    for (A, b, _), fits in zip(grid, all_fits):
        s, sc = slope(A, b, fits)
        chi2 = sum(f[1] for f in fits)
        table.append((A, b, s, sc, chi2))
    table = np.array(table)
    best_chi = table[:, 4].min()
    say("[linearity] slope of ln F_fit vs ln f_catalog (want 1), and delta chi^2:")
    for A in A_GRID:
        t = table[table[:, 0] == A]
        say(f"  A = {A:3d}: " + "  ".join(f"b{b:.1f}: {s:.3f} ({c - best_chi:8.0f})" for _, b, s, _, c in t))
    # A where the slope crosses 1, at each beta; then beta with the lowest chi^2 at that A
    # the slope is not monotone in A (fits collapse at small A), so take the grid point
    # closest to slope 1 for each beta rather than interpolating
    picks = []
    for b in B_GRID:
        t = table[table[:, 1] == b]
        k = int(np.argmin(np.abs(t[:, 2] - 1)))
        picks.append((b, t[k, 0], t[k, 2]))
    say("[linearity] A closest to slope 1, per beta: " +
        ", ".join(f"beta {b}: A = {a:.0f} (slope {s:.3f})" for b, a, s in picks))
    A_new = float(np.median([a for _, a, _ in picks]))
    say(f"[linearity] {len(UNSAT)} unsaturated stars (peak < 240) of {len(stars)}; fit subset {len(SUB)}")
    with Pool(16) as p:
        res = p.map(fit_set, [(A_new, b, SUB) for b in B_GRID])
    chis = [sum(f[1] for f in r) for r in res]
    beta_new = B_GRID[int(np.argmin(chis))]
    s_new, sc_new = slope(A_new, beta_new, res[int(np.argmin(chis))])
    say(f"[calibration] DR16: A = {A_new:.1f}, beta = {beta_new} (step 2a on DR7-era field stars: A = 25, beta = 4.5); "
        f"slope {s_new:.3f}, scatter {sc_new:.3f} in ln F")

    # ---- 2. star FWHMs at the new calibration ----------------------------------------------
    sel = np.flatnonzero((mag_r > 15) & (mag_r < 19.5))
    with Pool(16) as p:
        fw_fits = p.map(fit_set, [(A_new, beta_new, sel[k::16]) for k in range(16)])
    order = np.concatenate([sel[k::16] for k in range(16)])
    P = np.array([f[0] for ff in fw_fits for f in ff])
    fwhm = P[:, 5]
    logF = P[:, 1]
    say(f"[FWHM] {len(fwhm)} DR16 stars (r = 15-19.5): median {np.median(fwhm):.2f} px = "
        f"{0.2 * np.median(fwhm):.2f}\", 16-84% {np.percentile(fwhm, 16):.2f}-{np.percentile(fwhm, 84):.2f} px")
    np.savez(OUT / "step3_calibration.npz", A=A_new, beta=beta_new, fwhm=fwhm, logF=logF,
             mag=mag_r[order], idx=order)

    # ---- 3. real vs simulated star profiles ---------------------------------------------------
    import optics as o
    bank = dict(np.load(OUT / "step2b_psf_bank.npz"))
    QT = Image.open(SD / "stars" / f"{rows[0]['name']}.jpg").quantization
    sky_names = np.load(OUT / "step3_sky_usable.npy")
    sky = np.stack([np.asarray(Image.open(SD / "sky" / f"{n}.jpg")) for n in sky_names[:400]])
    yy, xx = np.mgrid[:sims.N, :sims.N] - (sims.N - 1) / 2
    rng = np.random.default_rng(1)

    def profile(v, cy, cx, rmax=20):
        g = v.astype(float).mean(-1)
        r = np.hypot(*np.mgrid[:sims.N, :sims.N] - np.array([cy, cx])[:, None, None])
        return np.array([g[(r >= a) & (r < a + 1)].mean() for a in range(rmax)])

    def spike_ratio(v, cy, cx):
        g = v.astype(float).mean(-1) - np.median(v.astype(float).mean(-1))
        dy, dx = np.mgrid[:sims.N, :sims.N] - np.array([cy, cx])[:, None, None]
        r = np.hypot(dy, dx)
        ring = (r > 8) & (r < 25)
        axes_ = ring & ((np.abs(dx) < 1.5) | (np.abs(dy) < 1.5))
        diag = ring & ((np.abs(dx - dy) < 1.5 * np.sqrt(2)) | (np.abs(dx + dy) < 1.5 * np.sqrt(2)))
        return g[axes_].mean() / max(g[diag].mean(), 1e-3)

    bins = [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)]
    fig, axes = plt.subplots(1, len(bins), figsize=(16, 3.4), constrained_layout=True)
    pos = {k: j for j, k in enumerate(order)}
    worst = []
    spikes_real, spikes_sim = [], []
    for ax, (lo, hi) in zip(axes, bins):
        ks = [k for k in order if lo <= mag_r[k] < hi][:60]
        if not ks:
            ax.set_visible(False)
            continue
        pr, ps = [], []
        for k in ks:
            j = pos[k]
            pr.append(profile(stars[k], *peaks[k]))
            col = np.array([1.0, 1.0, 1.0])
            u = sims.unstretch(sky[rng.integers(len(sky))], A_new) + sims.star_from_bank(
                rng, bank, fwhm[j], np.exp(logF[j]), col)
            v = sims.jpeg(sims.stretch(u, A_new), QT)
            c = np.unravel_index(v[50:70, 50:70].mean(-1).argmax(), (20, 20))
            ps.append(profile(v, c[0] + 50, c[1] + 50))
            if hi <= 16:
                spikes_real.append(spike_ratio(stars[k], *peaks[k]))
                spikes_sim.append(spike_ratio(v, c[0] + 50, c[1] + 50))
        mr, ms = np.median(pr, 0), np.median(ps, 0)
        ax.plot(np.arange(20) + 0.5, mr, "o", ms=3, label="real SDSS stars")
        ax.plot(np.arange(20) + 0.5, ms, "-", label="simulated (bank)")
        ax.set_title(f"r = {lo}-{hi}  (n = {len(ks)})", fontsize=9)
        ax.set_xlabel("radius (px)")
        d = (ms - mr) / np.maximum(mr, 1)
        worst.append((lo, hi, d[:3].mean(), d[4:12].mean()))
    axes[0].set_ylabel("median pixel value")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Stacked star profiles, DR16 calibration A = {A_new:.1f}, beta = {beta_new}")
    fig.savefig(OUT / "step3_calib_profiles.png", dpi=140)
    for lo, hi, core, wing in worst:
        say(f"[profiles] r = {lo}-{hi}: (sim - real)/real, core (r < 3 px) {100 * core:+.0f}%, "
            f"wings (4-12 px) {100 * wing:+.0f}%")
    say(f"[spikes] bright stars (r < 16): axis/diagonal light at 8-25 px, real {np.median(spikes_real):.2f}, "
        f"simulated {np.median(spikes_sim):.2f} (1 = no spikes)")

    # linearity figure
    fits_best = res[int(np.argmin(chis))]
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(np.log10(f_cat[SUB]), np.array([f[0][1] for f in fits_best]) / np.log(10), s=3, alpha=0.4)
    xs = np.linspace(np.log10(f_cat[SUB]).min(), np.log10(f_cat[SUB]).max(), 2)
    ax.plot(xs, xs * s_new + np.median(np.array([f[0][1] for f in fits_best]) / np.log(10) - np.log10(f_cat[SUB]) * s_new),
            "k-", label=f"slope {s_new:.3f}")
    ax.set_xlabel("log10 catalog flux (mean of g, r, i; nanomaggies)")
    ax.set_ylabel("log10 fitted flux (image units)")
    ax.set_title(f"Stretch linearity at A = {A_new:.1f}")
    ax.legend()
    fig.savefig(OUT / "step3_calib_linearity.png", dpi=140)
    (OUT / "step3_calib_log.txt").write_text("\n".join(log) + "\n")
    assert abs(s_new - 1) < 0.05, "at the calibrated A, fitted flux should be linear in catalog flux"
    print("PASS: linearity slope within 5% of 1 at the calibrated stretch.")
