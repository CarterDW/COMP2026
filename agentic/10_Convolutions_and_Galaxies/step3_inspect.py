"""Inspect the downloaded SDSS cutouts: real stars (gallery, centring, widths) and sky
frames (usable = not blank and nothing detected within 20 px of the centre)."""
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import data
import fieldstars as fs
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
cal = np.load(OUT / "step2a_calibration.npz")
A, BETA = float(cal["A"]), float(cal["beta"])
yy, xx = np.mgrid[:sims.N, :sims.N] - (sims.N - 1) / 2
R = np.hypot(xx, yy)


def load(kind):
    rows = list(csv.DictReader(open(SD / kind / "index.csv")))
    return rows, np.stack([np.asarray(Image.open(SD / kind / f"{r['name']}.jpg")) for r in rows])


if __name__ == "__main__":
    # ---- stars ------------------------------------------------------------------------
    rows, stars = load("stars")
    mag = np.array([float(r["psfMag_r"]) for r in rows])
    pk = np.array([np.unravel_index(v[50:70, 50:70].mean(-1).argmax(), (20, 20)) for v in stars]) + 50
    off = np.hypot(pk[:, 0] - 59.5, pk[:, 1] - 59.5)
    sel = np.flatnonzero((mag > 16) & (mag < 19))
    fw = np.array([fs.fit(stars[k], pk[k, 0], pk[k, 1], A=A, beta=BETA)[0][5] for k in sel])
    print(f"stars: {len(stars)}; peak offset from centre median {np.median(off):.2f} px; "
          f"FWHM (r = 16-19) median {np.median(fw):.2f} px, 16-84% {np.percentile(fw, 16):.2f}-"
          f"{np.percentile(fw, 84):.2f}")
    print("spectral types:", collections.Counter(r["subclass"][:2] for r in rows).most_common(8))
    np.savez(OUT / "step3_sdss_star_fwhm.npz", fwhm=fw, mag=mag[sel])

    fig, axes = plt.subplots(2, 8, figsize=(14, 4.6), constrained_layout=True)
    for ax, k in zip(axes.flat, np.argsort(mag)[np.linspace(0, len(mag) - 1, 16).astype(int)]):
        ax.imshow(stars[k])
        ax.set_title(f"r = {mag[k]:.1f}\n{rows[k]['subclass'][:4]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Real SDSS stars with spectra, bright to faint (r magnitude, spectral type)")
    fig.savefig(OUT / "step3_sdss_stars.png", dpi=130)

    # ---- sky --------------------------------------------------------------------------
    rows, sky = load("sky")
    blank = sky.reshape(len(sky), -1).std(1) < 1
    masks = [sims.object_mask(v, 2.0, 8) for v in sky]
    centre_clean = np.array([not (m & (R < 20)).any() for m in masks])
    ok = ~blank & centre_clean
    print(f"sky: {len(sky)} frames; blank {blank.sum()}; centre clean {centre_clean.sum()}; usable {ok.sum()}")
    np.save(OUT / "step3_sky_usable.npy", np.array([r["name"] for r in rows])[ok])

    fig, axes = plt.subplots(2, 8, figsize=(14, 4.2), constrained_layout=True)
    for ax, k in zip(axes.flat, np.flatnonzero(ok)[:16]):
        ax.imshow(sky[k])
        ax.axis("off")
    fig.suptitle("Real SDSS sky backgrounds (nothing detected within 20 px of the centre)")
    fig.savefig(OUT / "step3_sdss_sky.png", dpi=130)
