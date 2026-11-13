"""Step 2b: star images from first principles (pupil + Kolmogorov seeing + blackbody color).

Fine grid: N = 2048, one image pixel = THETA arcsec at every wavelength, so the pupil
spacing is dx = lambda / (N THETA). THETA = 0.018" keeps the pupil under half the grid
at g, so the autocorrelation in the exact long exposure doesn't wrap.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import optimize

import data
import fieldstars as fs
import optics as o
import sims

OUT = Path(__file__).parent / "results"
N, THETA = 2048, 0.018                       # fine grid, arcsec per fine pixel
SEEING_R = 1.4                               # arcsec FWHM at r: SDSS's median (from memory)
cal = np.load(OUT / "step2a_calibration.npz")
A, FWHM_FIELD = float(cal["A"]), np.median(cal["fwhm"])
gen = torch.Generator(device="cuda").manual_seed(0)
log = []


def say(s):
    print(s)
    log.append(s)


def dx_for(lam):
    return lam / (N * THETA * o.ARCSEC)


def moffat_fit(img, pix_fwhm):
    """Fit a Moffat to the azimuthal profile inside 3 FWHM; returns (fwhm, beta)."""
    c = img.shape[0] // 2
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]] - c
    r = np.hypot(xx, yy)
    rb = np.arange(0, 3 * pix_fwhm, 1.0)
    prof = np.array([img[(r >= a) & (r < a + 1)].mean() for a in rb])
    rc = np.array([r[(r >= a) & (r < a + 1)].mean() for a in rb])

    def model(rr, I0, fw, beta):
        a = fw / (2 * np.sqrt(2 ** (1 / beta) - 1))
        return I0 * (1 + (rr / a) ** 2) ** (-beta)
    p, _ = optimize.curve_fit(model, rc, prof, p0=[prof[0], pix_fwhm, 4.0])
    return p[1], p[2]


# ---- 1. Airy: clear pupil, lambda/D = 8 px, first zero at 1.22 x 8 = 9.76 px -----------
n_a = 512
I = o.psf(o.pupil(n_a, o.D_TEL / 64, d_sec=0, vane=0)).cpu().numpy()
c = n_a // 2
cut = (I[c, c:c + 16] + I[c, c - 15:c + 1][::-1] + I[c:c + 16, c] + I[c - 15:c + 1, c][::-1]) / 4
k = 5 + int(np.argmin(cut[5:14]))
a, b, cc = cut[k - 1:k + 2]
zero = k + 0.5 * (a - cc) / (a - 2 * b + cc)
say(f"[airy] first dark ring at {zero:.2f} px; expected 1.22 lambda/D = {1.22 * 8:.2f} px")

# ---- 2. spikes: one vane line at 30 deg -> spikes perpendicular, at 120 deg --------------
I = o.psf(o.pupil(n_a, o.D_TEL / 128, vane=0.06, vane_angle=np.deg2rad(30), cross=False)).cpu().numpy()
yy, xx = np.mgrid[:n_a, :n_a] - c
r, ang = np.hypot(xx, yy), np.rad2deg(np.arctan2(yy, xx)) % 180
ring = (r > 20) & (r < 80)
ab = np.arange(0, 180, 1.0)
az = np.array([I[ring & (ang >= t) & (ang < t + 1)].mean() for t in ab])
spike_angle = ab[az.argmax()] + 0.5
say(f"[spikes] one vane at 30 deg: azimuthal peak at {spike_angle:.1f} deg (expected 120), "
    f"contrast {az.max() / np.median(az):.1f}x")

# ---- 3. phase screens: the FFT part vs its own discrete theory; deficit vs Kolmogorov ----
dxs, r0s = 0.003, 0.1
f = np.fft.fftfreq(N, d=dxs)
fy, fx = np.meshgrid(f, f, indexing="ij")
fr = np.hypot(fx, fy)
fr[0, 0] = 1
psd = 0.023 * r0s ** (-5 / 3) * fr ** (-11 / 3)
psd[0, 0] = 0
df = 1 / (N * dxs)
D_hi_b = np.array([o.structure_function(o.kolmogorov(16, N, dxs, r0s, gen, ("hi",)), dxs, 300)[1]
                   for _ in range(8)])                    # 8 batches of 16 screens
D_hi, D_hi_err = D_hi_b.mean(0), D_hi_b.std(0, ddof=1) / np.sqrt(len(D_hi_b))
D_all = np.mean([o.structure_function(o.kolmogorov(16, N, dxs, r0s, gen), dxs, 300)[1]
                 for _ in range(8)], axis=0)
sf_rows = []
for lag in [10, 30, 100, 300]:
    rr = lag * dxs
    th_hi = 2 * (psd * df**2 * (1 - np.cos(2 * np.pi * fx * rr))).sum()
    DK = 6.88 * (rr / r0s) ** (5 / 3)
    pull = (D_hi[lag - 1] - th_hi) / D_hi_err[lag - 1]
    sf_rows.append((rr, pull, D_all[lag - 1] / DK))
    say(f"[screens] r = {rr:.2f} m: FFT part / its discrete theory = {D_hi[lag - 1] / th_hi:.3f} "
        f"({pull:+.1f} sigma); full screen / Kolmogorov = {D_all[lag - 1] / DK:.3f}")

# ---- 4. exact long exposures in (i, r, g) through the SDSS pupil --------------------------
lam_r = o.BANDS["r"]
r0_r = 0.98 * lam_r / (SEEING_R * o.ARCSEC)
r0_500 = r0_r * (500e-9 / lam_r) ** 1.2
fine, fw_fine = {}, {}
for band, lam in o.BANDS.items():
    P = o.pupil(N, dx_for(lam))
    fine[band] = o.long_exposure_exact(P, dx_for(lam), o.r0_at(lam, r0_500))
    fw_fine[band] = o.fwhm(fine[band], THETA)
    expected = 0.98 * lam / o.r0_at(lam, r0_500) / o.ARCSEC
    say(f"[seeing] {band}: FWHM {fw_fine[band]:.3f}\" vs 0.98 lambda/r0 = {expected:.3f}\" "
        f"(r0 = {100 * o.r0_at(lam, r0_500):.1f} cm, pupil {o.D_TEL / dx_for(lam):.0f} of {N} px)")
ratio = fw_fine["g"] / fw_fine["i"]
ratio_th = (o.BANDS["g"] / o.BANDS["i"]) ** -0.2
say(f"[chromatic] FWHM g/i = {ratio:.4f}; Kolmogorov lambda^(-1/5) predicts {ratio_th:.4f}")
fw_m, beta_m = moffat_fit(fine["r"], fw_fine["r"] / THETA)
say(f"[moffat] r-band long exposure: Moffat FWHM {fw_m * THETA:.3f}\", beta = {beta_m:.2f} "
    f"(Trujillo et al. 2001: 4.765; step-2a field stars: {float(cal['beta']):.1f})")

# ---- 5. speckles: short exposures through random screens, averaged ----------------------
P_r = o.pupil(N, dx_for(lam_r))
shorts = o.psf(P_r, o.kolmogorov(16, N, dx_for(lam_r), r0_r, gen)).cpu().numpy()
avg = {1: shorts[0], 10: shorts[:10].mean(0), 100: o.long_exposure(P_r, dx_for(lam_r), r0_r, 100, gen)}
avg400 = o.long_exposure(P_r, dx_for(lam_r), r0_r, 400, gen)
fw_400 = o.fwhm(moffat_smooth := avg400, THETA)
say(f"[speckle] 400-screen average FWHM {fw_400:.3f}\" vs exact {fw_fine['r']:.3f}\" "
    f"(ratio {fw_400 / fw_fine['r']:.3f}; the screens' missing large-scale power narrows it)")

# ---- 6. the detector-level PSF bank -----------------------------------------------------
FW_GRID = np.arange(2.0, 10.01, 0.5)                 # target r-band FWHM in detector px
OFFS = [(dy, dx) for dy in (-1 / 3, 0, 1 / 3) for dx in (-1 / 3, 0, 1 / 3)]
bank = np.zeros((len(FW_GRID), len(OFFS), 3, sims.N, sims.N), np.float32)
for i_f, fw_px in enumerate(FW_GRID):
    z = fw_px / (fw_fine["r"] / THETA)              # detector px per fine px (same for all bands)
    for i_o, off in enumerate(OFFS):
        for i_b, band in enumerate(["i", "r", "g"]):
            bank[i_f, i_o, i_b] = o.to_detector(fine[band], z, off)
np.savez_compressed(OUT / "step2b_psf_bank.npz", psf=bank, fwhm=FW_GRID, theta=THETA,
                    seeing_r=SEEING_R)
bank_d = {"psf": bank, "fwhm": FW_GRID}
sums = bank[:, 4].sum(axis=(-2, -1))                  # centred stamps
say(f"[bank] {bank.shape}: flux in the 120 px frame, min {sums.min():.4f}, max {sums.max():.4f}")
fw_meas = [fs.fit(sims.stretch(1e3 * np.moveaxis(bank[i, 4], 0, -1), A), 59, 59, A=A,
                  beta=float(cal["beta"]))[0][5] for i in range(len(FW_GRID))]
dev = np.array(fw_meas) / FW_GRID - 1
for fw_t, fw_r in zip(FW_GRID[::4], np.array(fw_meas)[::4]):
    say(f"[bank] target FWHM {fw_t:.1f} px -> step-2a Moffat fit {fw_r:.2f} px")

# ---- 7. blackbody colors vs the field stars ------------------------------------------------
temps = np.geomspace(3000, 30000, 60)
bb = np.array([o.blackbody_fractions(T) for T in temps])          # (i, r, g), sum 3
bb_xy = np.c_[np.log(bb[:, 1] / bb[:, 0]), np.log(bb[:, 2] / bb[:, 0])]    # ln(r/i), ln(g/i)
fcol = cal["colors"]
f_xy = np.c_[np.log(fcol[:, 1] / fcol[:, 0]), np.log(fcol[:, 2] / fcol[:, 0])]
f_xy = f_xy[np.all(np.isfinite(f_xy), axis=1) & np.all(np.abs(f_xy) < 3, axis=1)]


def locus_dist(shift):
    """Each field star's distance to the shifted blackbody curve."""
    pts = bb_xy + shift
    return np.min(np.linalg.norm(f_xy[:, None] - pts[None], axis=-1), axis=1)


gain = optimize.minimize(lambda s: np.mean(locus_dist(s)), x0=[0.0, 0.0], method="Nelder-Mead",
                         options={"initial_simplex": [[0, 0], [0.3, 0], [0, 0.3]], "xatol": 1e-4}).x
d_perp = locus_dist(gain)
proj = np.argmin(np.linalg.norm(f_xy[:, None] - (bb_xy + gain)[None], axis=-1), axis=1)
inside = np.mean((proj > 0) & (proj < len(temps) - 1))
T_med = temps[int(np.median(proj))]
evals, evecs = np.linalg.eigh(np.cov(f_xy.T))
axis = evecs[:, -1]
tangent = bb_xy[int(np.median(proj)) + 1] - bb_xy[int(np.median(proj)) - 1]
tangent /= np.linalg.norm(tangent)
angle = np.rad2deg(np.arccos(abs(axis @ tangent)))
say(f"[locus] channel gains ln(r/i), ln(g/i) offsets = {gain.round(3)}; median distance to the "
    f"curve {np.median(d_perp):.3f}; {100 * inside:.0f}% project inside 3000-30000 K; "
    f"median T {T_med:.0f} K")
say(f"[locus] field-star principal axis vs blackbody tangent: {angle:.1f} deg; "
    f"axis ratio sqrt(l1/l2) = {np.sqrt(evals[-1] / evals[0]):.1f}")

# ---- figures ------------------------------------------------------------------------------
# (a) pupil -> diffraction PSF -> short exposure -> long exposure -> survey pixels
sky_imgs, labels, ids = data.load()
QT = Image.open(data.DIR / f"{ids[0]}.jpg").quantization
rng = np.random.default_rng(0)
bgs = []
for i in rng.permutation(len(sky_imgs)):
    bgk = sims.empty_sky(sky_imgs[i], rng)
    if bgk is not None:
        bgs.append(bgk)
    if len(bgs) == 40:
        break


def composite(bg, u):
    return sims.jpeg(sims.stretch(sims.unstretch(bg, A) + u, A), QT)


diff = o.psf(P_r).cpu().numpy()
w = 160                                                   # fine px half-width, 2.9"
crop = lambda img, h: img[N // 2 - h:N // 2 + h, N // 2 - h:N // 2 + h]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
pupil_view = P_r[N // 2 - 380:N // 2 + 380, N // 2 - 380:N // 2 + 380]
axes[0].imshow(pupil_view, cmap="gray")
axes[0].set_title("SDSS pupil: 2.5 m, secondary, 4 vanes", fontsize=9)
axes[1].imshow(np.log10(crop(diff, 40) + 1e-9), cmap="magma", vmin=-7)
axes[1].set_title("diffraction only (log), 1.4\" across", fontsize=9)
axes[2].imshow(crop(shorts[0], w), cmap="magma")
axes[2].set_title("one short exposure: speckles", fontsize=9)
axes[3].imshow(np.log10(crop(fine["r"], 4 * w) + 1e-12), cmap="magma", vmin=-9)
axes[3].set_title("long exposure (log), 23\" across", fontsize=9)
z5 = FWHM_FIELD / (fw_fine["r"] / THETA)
bright = np.stack([o.to_detector(fine[b], z5) for b in ["i", "r", "g"]], -1)
axes[4].imshow(composite(bgs[0], 3e5 * bright * o.blackbody_fractions(6000)))
axes[4].set_title(f"as SDSS records it: very bright star", fontsize=9)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "step2b_optics.png", dpi=150)

# (b) speckle -> long exposure
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, (n, img) in zip(axes, list(avg.items()) + [("exact", fine["r"])]):
    ax.imshow(crop(img, w), cmap="magma")
    ax.set_title(f"{n} exposure{'s' if n != 1 else ''} averaged" if n != "exact"
                 else "exact long exposure", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Kolmogorov seeing at r, r0 = {100 * r0_r:.1f} cm, {2 * w * THETA:.1f}\" across", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "step2b_speckles.png", dpi=150)

# (c) blackbody stars on real sky, and the stellar locus
show_T = [3000, 3500, 4500, 5800, 7500, 10000, 15000, 30000]
gain_lin = np.exp(np.r_[0.0, gain])                       # multiply (i, r, g) fractions
fig = plt.figure(figsize=(14, 5.2))
for k, T in enumerate(show_T):
    ax = fig.add_subplot(2, 8, [1, 2, 3, 4, 9, 10, 11, 12][k])
    col = o.blackbody_fractions(T) * gain_lin
    col = 3 * col / col.sum()
    u = sims.star_from_bank(np.random.default_rng(k), bank_d, FWHM_FIELD, 3e3, col)
    ax.imshow(composite(bgs[1 + k], u)[35:85, 35:85])
    ax.set_title(f"{T} K", fontsize=9)
    ax.axis("off")
ax = fig.add_subplot(1, 2, 2)
ax.scatter(*f_xy.T, s=4, alpha=0.4, c="0.4", label="field stars (step 2a)")
ax.plot(*(bb_xy + gain).T, "-", c="tab:orange", lw=2, label="blackbody 3000-30000 K, shifted")
for T in [3000, 5800, 30000]:
    j = np.argmin(np.abs(temps - T))
    ax.annotate(f"{T} K", (bb_xy + gain)[j], fontsize=8)
ax.set_xlabel("ln(r / i)")
ax.set_ylabel("ln(g / i)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step2b_star_colors.png", dpi=150)

# (d) level-2 artifacts
fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
for k in range(6):
    axes[0, k].imshow(composite(bgs[10 + k], sims.cosmic_track(np.random.default_rng(k), 1500)))
    axes[1, k].imshow(composite(bgs[20 + k], sims.flicker_trail(np.random.default_rng(k), FWHM_FIELD, 30)))
    axes[0, k].axis("off")
    axes[1, k].axis("off")
axes[0, 0].set_title("cosmic-ray tracks", loc="left", fontsize=9)
axes[1, 0].set_title("flickering satellite trails", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "step2b_artifacts.png", dpi=130)

# ---- checks -----------------------------------------------------------------------------
cr = sims.cosmic_track(np.random.default_rng(9), 1.0).sum()
assert abs(zero - 9.76) < 0.25, "Airy first zero should sit at 1.22 lambda/D"
assert abs(spike_angle - 120) <= 2, "spikes should be perpendicular to the vane"
assert all(abs(p) < 3 for _, p, _ in sf_rows), "FFT screens should match their discrete theory"
assert abs(fw_fine["r"] / SEEING_R - 1) < 0.05, "long-exposure FWHM should be 0.98 lambda/r0"
assert abs(ratio / ratio_th - 1) < 0.02, "FWHM should scale as lambda^(-1/5)"
assert 4.0 < beta_m < 5.5, "Kolmogorov long exposure should have Moffat beta near 4.8"
assert np.all(np.abs(dev[FW_GRID >= 4]) < 0.05), "bank stars should refit to their target FWHM"
assert abs(cr - 1) < 1e-9, "cosmic track should deposit exactly its flux"
assert inside > 0.9, "field-star colors should project inside the blackbody range"
say("PASS: Airy, spike direction, screen statistics, seeing FWHM, chromatic scaling, Moffat beta, "
    "bank FWHM, track flux, colors on the blackbody locus.")
(OUT / "step2b_log.txt").write_text("\n".join(log) + "\n")
