"""Step 3 (real-sky version): star vs galaxy, with no simulated sky anywhere.

Galaxies: real Galaxy Zoo cutouts, faded by f ~ logU(F_MIN, 1), sky noise restored
from real SDSS sky frames. Stars come in two kinds:
  sim  - optics-bank stars (step 2b) injected into real SDSS sky frames;
  real - SDSS stars with spectra, faded to the same flux distribution.
Every image goes through the same linear -> stretch -> JPEG round trip. Two CNNs are
trained (sim stars, real stars) and both are tested on held-out real and sim stars.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats

import data
import models
import optics as o
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
F_MIN = 0.02
cal = np.load(OUT / "step2a_calibration.npz")
A = float(cal["A"])
FWHM_REAL = np.load(OUT / "step3_sdss_star_fwhm.npz")["fwhm"]   # real SDSS star widths
bank = dict(np.load(OUT / "step2b_psf_bank.npz"))
images, labels, ids = data.load()
QT = Image.open(data.DIR / f"{ids[0]}.jpg").quantization
yy, xx = np.mgrid[:sims.N, :sims.N] - (sims.N - 1) / 2
R = np.hypot(xx, yy)
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def load_dir(kind, names=None):
    rows = list(csv.DictReader(open(SD / kind / "index.csv")))
    if names is not None:
        keep = set(names)
        rows = [r for r in rows if r["name"] in keep]
    return rows, np.stack([np.asarray(Image.open(SD / kind / f"{r['name']}.jpg")) for r in rows])


def load_dr16_galaxies():
    """DR16 re-cutouts of the Galaxy Zoo galaxies, aligned with `ids`; `have` marks the
    ones the crosswalk and the server could supply. Labels are Galaxy Zoo's, unchanged."""
    rows = list(csv.DictReader(open(SD / "galaxies" / "index.csv")))
    by_id = {r["dr7objid"]: r["name"] for r in rows}
    have = np.array([i in by_id for i in ids])
    out = np.zeros_like(images)
    for k in np.flatnonzero(have):
        out[k] = np.asarray(Image.open(SD / "galaxies" / f"{by_id[ids[k]]}.jpg"))
    return out, have


def linear(v):
    return sims.unstretch(v, A)


def finish(u):
    return sims.jpeg(sims.stretch(u, A), QT)


def sky_level(u):
    return np.median(u[R > 50], axis=0)


def total_flux(u):
    """Background-subtracted channel-mean flux inside r < 20 px."""
    return (u - sky_level(u))[R < 20].mean(axis=-1).sum()


def fade(v, f, sky):
    """Scale an object's light by f and restore the sky noise from a real sky frame."""
    u, ub = linear(v), linear(sky)
    mu, mb = sky_level(u), sky_level(ub)
    return f * (u - mu) + np.sqrt(1 - f**2) * (ub - mb) + mb


def second_moment_radius(v):
    g = linear(v).mean(axis=-1)
    g = np.clip(g - np.median(g[R > 50]), 0, None) * (R < 12)
    return np.sqrt((g * R**2).sum() / max(g.sum(), 1e-12))


def tensor(X, mean=None, std=None):
    x = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = x.mean() if mean is None else mean
    std = x.std() if std is None else std
    return (x - mean) / std, mean, std


def predict(model, x, batch=500):
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(model(x[i:i + batch].to(models.DEVICE)), 1)[:, 1].cpu()
                       for i in range(0, len(x), batch)])
    model.train()
    return p.numpy()


def build(gal_idx, sky, real_stars, n_sim, rng, star_colors="blackbody"):
    """Faded galaxies, sim stars and (faded) real stars, all on real sky.
    Returns images, labels (0 galaxy, 1 star), kind (0 galaxy, 1 sim star, 2 real star), ln F."""
    X, kind, logF = [], [], []
    gflux = np.array([total_flux(linear(images[i])) for i in gal_idx])
    for i, F0 in zip(gal_idx, gflux):
        f = np.exp(rng.uniform(np.log(F_MIN), 0))
        X.append(finish(fade(images[i], f, sky[rng.integers(len(sky))])))
        kind.append(0)
        logF.append(np.log(f * F0))
    target = lambda: gflux[rng.integers(len(gflux))] * np.exp(rng.uniform(np.log(F_MIN), 0))
    gal_cols = None
    if star_colors == "galaxy":
        gal_cols = []
        for i in gal_idx[:500]:
            u = linear(images[i])
            s = np.clip((u - sky_level(u))[R < 10].sum(axis=0), 1e-6, None)
            gal_cols.append(3 * s / s.sum())
        gal_cols = np.array(gal_cols)
    for _ in range(n_sim):
        F = target()
        col = (o.blackbody_fractions(np.exp(rng.uniform(np.log(3000), np.log(12000))))
               if gal_cols is None else gal_cols[rng.integers(len(gal_cols))])
        u = linear(sky[rng.integers(len(sky))]) + sims.star_from_bank(
            rng, bank, FWHM_REAL[rng.integers(len(FWHM_REAL))], F, col)
        X.append(finish(u))
        kind.append(1)
        logF.append(np.log(F))
    if len(real_stars):
        sflux = np.array([total_flux(linear(v)) for v in real_stars])
        for _ in range(len(real_stars)):                    # fade a brighter real star to the target
            F = target()
            brighter = np.flatnonzero(sflux >= F)
            j = rng.choice(brighter) if len(brighter) else int(np.argmax(sflux))
            f = min(1.0, F / sflux[j])
            X.append(finish(fade(real_stars[j], f, sky[rng.integers(len(sky))])))
            kind.append(2)
            logF.append(np.log(f * sflux[j]))
    kind = np.array(kind)
    return np.stack(X), (kind > 0).astype(int), kind, np.array(logF)


def train_on(X, y, seed=0):
    x, m, s = tensor(X)
    torch.manual_seed(seed)
    net = models.cnn()
    # test-set monitoring is done afterwards; pass a tiny slice to satisfy the loop
    models.train(net, x, torch.tensor(y), x[:2], torch.tensor(y[:2]), seed=seed)
    return net, m, s


def accuracy_by_flux(p, y, logF, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (logF >= lo) & (logF <= hi)
        out.append((np.median(logF[sel]), np.mean((p[sel] > 0.5) == y[sel]), sel.sum()))
    return np.array(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # ---- data and splits ---------------------------------------------------------------
    sky_names = np.load(OUT / "step3_sky_usable.npy")
    _, sky = load_dir("sky", sky_names)
    _, rstars = load_dir("stars")
    sky_perm, star_perm = rng.permutation(len(sky)), rng.permutation(len(rstars))
    sky_tr, sky_te = sky[sky_perm[len(sky) // 5:]], sky[sky_perm[:len(sky) // 5]]
    rs_tr, rs_te = rstars[star_perm[len(rstars) // 5:]], rstars[star_perm[:len(rstars) // 5]]
    train_c, test_c = data.split(labels)
    unc = rng.permutation(np.flatnonzero(labels == 2))
    gal_tr, gal_te = np.r_[train_c, unc[len(unc) // 5:]], np.r_[test_c, unc[:len(unc) // 5]]
    # every image from the same (DR16) pipeline: swap in the DR16 galaxy cutouts
    images_dr7 = images
    images, have = load_dr16_galaxies()
    gal_tr, gal_te = gal_tr[have[gal_tr]], gal_te[have[gal_te]]
    say(f"[data] DR16 galaxy cutouts: {have.sum()} of {len(have)}")
    say(f"[data] sky {len(sky_tr)}/{len(sky_te)}, real stars {len(rs_tr)}/{len(rs_te)}, "
        f"galaxies {len(gal_tr)}/{len(gal_te)} (train/test)")

    # training sets: galaxies + sim stars;  galaxies + real stars.  One shared test set.
    Xs, ys, ks, Fs = build(gal_tr, sky_tr, [], len(rs_tr), rng)
    Xr, yr, kr, Fr = build(gal_tr, sky_tr, rs_tr, 0, rng)
    Xt, yt, kt, Ft = build(gal_te, sky_te, rs_te, len(rs_te), rng)
    for name, F, k in [("sim-star train", Fs, ks), ("real-star train", Fr, kr), ("test", Ft, kt)]:
        p = stats.ks_2samp(F[k == 0], F[k > 0]).pvalue
        say(f"[brightness] {name}: ln F median galaxies {np.median(F[k == 0]):.2f}, stars "
            f"{np.median(F[k > 0]):.2f}; KS p = {p:.2f}")

    # ---- leak test: corners of galaxy frames vs corners of SDSS sky frames -----------------
    def corners(frames):
        out = []
        for v in frames:
            w = finish(linear(v))
            for sy, sx in [(slice(0, 24), slice(0, 24)), (slice(0, 24), slice(96, 120)),
                           (slice(96, 120), slice(0, 24)), (slice(96, 120), slice(96, 120))]:
                out.append(w[sy, sx])
        return np.stack(out)

    def leak(P0, P1):
        n = min(len(P0), len(P1))
        P0, P1 = P0[rng.permutation(len(P0))[:n]], P1[rng.permutation(len(P1))[:n]]
        X = np.concatenate([P0, P1])
        y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        p = rng.permutation(len(X))
        X, y = X[p], y[p]
        k = int(0.8 * len(X))
        xtr, m, s = tensor(X[:k])
        xte, _, _ = tensor(X[k:], m, s)
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3), torch.nn.ReLU(), torch.nn.Conv2d(16, 16, 3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), torch.nn.Conv2d(16, 32, 3), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 2))
        h = models.train(net, xtr, torch.tensor(y[:k]), xte, torch.tensor(y[k:]), n_epochs=20)
        return h[-1], len(X) - k

    gal_frames = images[gal_tr[:1500]]
    leak_gs, n_gs = leak(corners(gal_frames), corners(sky_tr[:1500]))
    leak_ss, n_ss = leak(corners(sky_tr[:750]), corners(sky_tr[750:1500]))
    leak_77, n_77 = leak(corners(images_dr7[gal_tr[:1500]]), corners(images[gal_tr[:1500]]))
    err_gs, err_ss = np.sqrt(0.25 / n_gs), np.sqrt(0.25 / n_ss)
    say(f"[leak] same galaxies, DR7-era JPG corners vs DR16 corners: {100 * leak_77:.1f}% "
        f"(chance 50 +/- {100 * np.sqrt(0.25 / n_77):.1f})")
    say(f"[leak] DR16 galaxy-frame corners vs SDSS sky corners: {100 * leak_gs:.1f}% (chance 50 +/- {100 * err_gs:.1f})")
    say(f"[leak] control, SDSS sky vs SDSS sky: {100 * leak_ss:.1f}% (chance 50 +/- {100 * err_ss:.1f})")

    # ---- the two classifiers ------------------------------------------------------------------
    net_s, ms, ss = train_on(Xs, ys)
    net_r, mr, sr = train_on(Xr, yr)
    p_s = predict(net_s, tensor(Xt, ms, ss)[0])
    p_r = predict(net_r, tensor(Xt, mr, sr)[0])
    r2 = np.array([second_moment_radius(v) for v in Xt])
    r2_tr = np.array([second_moment_radius(v) for v in Xs])
    grid = np.linspace(0.5, 10, 400)
    t_cut = grid[np.argmax([np.mean((r2_tr < t) == ys) for t in grid])]
    p_cut = (r2 < t_cut).astype(float)
    rows = [("CNN trained on sim stars", p_s), ("CNN trained on real stars", p_r),
            (f"size cut (r_rms < {t_cut:.2f} px)", p_cut)]
    say(f"[accuracy] {'':34s} galaxies   sim stars   real stars")
    for name, p in rows:
        acc = [np.mean((p[kt == k] > 0.5) == yt[kt == k]) for k in (0, 1, 2)]
        say(f"[accuracy] {name:34s} {100 * acc[0]:7.1f}%   {100 * acc[1]:7.1f}%   {100 * acc[2]:7.1f}%")

    edges = np.quantile(Ft, np.linspace(0, 1, 8))
    curves = {}
    for name, p in rows:
        for kname, kk in [("real stars", 2), ("sim stars", 1)]:
            sel = (kt == 0) | (kt == kk)
            curves[(name, kname)] = accuracy_by_flux(p[sel], yt[sel], Ft[sel], edges)
    for (name, kname), c in curves.items():
        say(f"[by flux] {name} vs galaxies+{kname}: " + " ".join(f"{100 * a:5.1f}" for _, a, _ in c))

    # ---- colors don't leak (sim stars with galaxy colors) ---------------------------------------
    Xc, yc, kc, _ = build(gal_tr, sky_tr, [], len(rs_tr), np.random.default_rng(1), star_colors="galaxy")
    net_c, mc, sc = train_on(Xc, yc)
    p_c = predict(net_c, tensor(Xt, mc, sc)[0])
    sel = kt != 2
    acc_c, acc_s = np.mean((p_c[sel] > 0.5) == yt[sel]), np.mean((p_s[sel] > 0.5) == yt[sel])
    say(f"[colors] sim stars with galaxy colors: {100 * acc_c:.1f}% on galaxies+sim stars "
        f"(blackbody colors: {100 * acc_s:.1f}%)")

    # ---- figures ----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, kname in zip(axes, ["real stars", "sim stars"]):
        for (name, p), mk in zip(rows, ["o-", "s-", "^--"]):
            c = curves[(name, kname)]
            ax.plot(c[:, 0], 100 * c[:, 1], mk, label=name)
        ax.axhline(50, c="k", ls=":", lw=1)
        ax.set_xlabel("ln (total flux)   faint  ->  bright")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"galaxies vs {kname}")
        ax.set_ylim(40, 101)
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "step3_accuracy.png", dpi=150)

    fig, axes = plt.subplots(3, 8, figsize=(14, 7.4), constrained_layout=True)
    for row, (kk, name) in enumerate([(0, "real galaxy\n(faded)"), (1, "simulated star\non real sky"),
                                      (2, "real SDSS star\n(faded)")]):
        idx = np.flatnonzero(kt == kk)
        idx = idx[np.argsort(Ft[idx])][np.linspace(0, len(idx) - 1, 8).astype(int)]
        for ax, j in zip(axes[row], idx):
            ax.imshow(Xt[j])
            ax.set_title(f"P(star) = {p_r[j]:.2f}\ncalled: {'star' if p_r[j] > 0.5 else 'galaxy'}",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Test images, faint (left) to bright (right); predictions from the CNN trained on real stars")
    fig.savefig(OUT / "step3_gallery.png", dpi=130)
    (OUT / "step3_log.txt").write_text("\n".join(log) + "\n")

    # ---- checks ---------------------------------------------------------------------------------
    acc_r_real = np.mean((p_r[kt == 2] > 0.5) == 1)
    bright = curves[(rows[2][0], "real stars")][:, 0] > np.quantile(Ft, 0.7)
    assert abs(leak_ss - 0.5) < 3 * err_ss, "control leak test (sky vs sky) should be at chance"
    assert np.all(curves[(rows[2][0], "real stars")][bright, 1] > 0.85), "size cut should work for bright sources"
    assert abs(acc_c - acc_s) < 0.03, "star colors should not matter much"
    print("PASS: control leak at chance, size cut works when bright, colors don't matter.")
    print(f"(galaxy-vs-sky leak {100 * leak_gs:.1f}% and sim->real transfer are results, not pass/fail)")
