"""Step 6: quasars vs stars. Both are point sources, so shape should not help; color should.

Data: SDSS spectroscopic quasars and stars, 500 + 500 per 1-mag bin in psfMag_r, 17-20.
Images: the central 48 x 48 of the DR16 cutouts, used as downloaded (same pipeline for both).
Classifiers (3 seeds): CNN on color images, CNN on grayscale images.
Catalog baselines (gradient-boosted trees): g/r/i colors, the most a g/r/i image could hold;
u/g/r/i colors, adding the quasars' famous ultraviolet excess; and colors measured from the
images themselves.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

import data
import models
import sims

OUT = Path(__file__).parent / "results"
SD = data.NB / "data" / "sdss_week10"
A = float(np.load(OUT / "step3_calibration.npz")["A"])
SEEDS = [0, 1, 2]
C = 24                                                        # half-width of the 48 px crop
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def load(kind, rows):
    return np.stack([np.asarray(Image.open(SD / kind / f"{r['name']}.jpg"))[60 - C:60 + C, 60 - C:60 + C]
                     for r in rows])


def image_colors(imgs):
    """ln(i/r), ln(g/r) of the linear flux within r < 5 px of the brightest central pixel."""
    out = []
    yy, xx = np.mgrid[:2 * C, :2 * C]
    for v in imgs:
        u = sims.unstretch(v, A)
        cy, cx = np.unravel_index(v[C - 8:C + 8, C - 8:C + 8].mean(-1).argmax(), (16, 16))
        r = np.hypot(yy - (cy + C - 8), xx - (cx + C - 8))
        s = np.clip((u[r < 5] - np.median(u[r > 18], axis=0)).sum(0), 1e-3, None)
        out.append([np.log(s[0] / s[1]), np.log(s[2] / s[1])])
    return np.array(out)


if __name__ == "__main__":
    srows = [r for r in csv.DictReader(open(SD / "stars" / "index.csv")) if 17 <= float(r["psfMag_r"]) < 20]
    u_star = {r["objid"]: float(r["psfMag_u"]) for r in csv.DictReader(open(SD / "stars" / "u_band.csv"))}
    srows = [dict(r, psfMag_u=u_star[r["objid"]]) for r in srows if r["objid"] in u_star]
    # quasars: 1000 of the main (mostly z < 2) sample plus 500 at z = 2-3.5, 1500 in all
    q_lo = list(csv.DictReader(open(SD / "quasars" / "index.csv")))
    q_lo = [q_lo[k] for k in np.random.default_rng(1).permutation(len(q_lo))[:1000]]
    q_hi = list(csv.DictReader(open(SD / "quasars_hiz" / "index.csv")))
    qrows = q_lo + q_hi
    stars = load("stars", srows)
    qsos = np.concatenate([load("quasars", q_lo), load("quasars_hiz", q_hi)])
    X_img = np.concatenate([stars, qsos])
    y = np.r_[np.zeros(len(stars)), np.ones(len(qsos))].astype(int)
    mags = np.array([[float(r[f"psfMag_{b}"]) for b in "ugri"] for r in srows + qrows])
    z = np.r_[np.zeros(len(stars)), [float(r["z"]) for r in qrows]]
    rmag = mags[:, 2]
    say(f"[data] {len(stars)} stars, {len(qsos)} quasars (r = 17-20); quasar z median "
        f"{np.median(z[y == 1]):.2f}, 5-95% {np.percentile(z[y == 1], 5):.2f}-{np.percentile(z[y == 1], 95):.2f}")
    say(f"[data] median r: stars {np.median(rmag[y == 0]):.2f}, quasars {np.median(rmag[y == 1]):.2f}")

    rng = np.random.default_rng(0)
    test = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        test[idx[: len(idx) // 5]] = True
    tr, te = ~test, test

    # ---- catalog and image-color baselines ------------------------------------------------------
    u_, g_, r_, i_ = mags.T
    feats = {"catalog g/r/i colors": np.c_[g_ - r_, r_ - i_],
             "catalog u/g/r/i colors": np.c_[u_ - g_, g_ - r_, r_ - i_],
             "colors measured from the images": image_colors(X_img)}
    base, base_pred = {}, {}
    for name, F in feats.items():
        clf = HistGradientBoostingClassifier(random_state=0).fit(F[tr], y[tr])
        base_pred[name] = clf.predict_proba(F[te])[:, 1]
        base[name] = np.mean((base_pred[name] > 0.5) == y[te])
        say(f"[baseline] {name}: {100 * base[name]:.1f}%")

    # ---- CNNs -------------------------------------------------------------------------------------
    x = torch.tensor(X_img, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    # shape only: linear flux (channel mean), sky-subtracted, divided by the flux inside
    # r < 8 px, so neither brightness nor color survives (the JPG's channel gains make a
    # plain grayscale image color-sensitive)
    yy, xx = np.mgrid[:2 * C, :2 * C]
    R = np.hypot(yy - (C - 0.5), xx - (C - 0.5))
    lin = sims.unstretch(X_img, A).mean(-1)
    lin = lin - np.median(lin[:, R > 18], axis=1)[:, None, None]
    lin = lin / np.clip(lin[:, R < 8].sum(1), 1e-3, None)[:, None, None]
    x_shape = torch.tensor(lin[:, None], dtype=torch.float32)
    res, p_cnn = {}, {}
    for kind in ["color", "gray", "shape"]:
        xk = {"color": x, "gray": x.mean(1, keepdim=True), "shape": x_shape}[kind]
        m, s = xk[tr].mean(), xk[tr].std()
        xk = (xk - m) / s
        accs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = models.cnn(in_ch=xk.shape[1], size=2 * C)
            models.train(net, xk[tr], torch.tensor(y[tr]), xk[te], torch.tensor(y[te]), seed=seed)
            net.eval()
            with torch.no_grad():
                p = torch.softmax(net(xk[te].to(models.DEVICE)), 1)[:, 1].cpu().numpy()
            accs.append(np.mean((p > 0.5) == y[te]))
            if seed == 0:
                p_cnn[kind] = p
        res[kind] = (np.mean(accs), np.std(accs, ddof=1))
        say(f"[cnn] {kind:5s} images: {100 * res[kind][0]:.1f} +/- {100 * res[kind][1]:.1f}%")

    # ---- accuracy vs quasar redshift ----------------------------------------------------------------
    zq = z[te][y[te] == 1]
    zbins = [0, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 6]
    by_z = {}
    for name, p in [("CNN, color images", p_cnn["color"]), ("catalog g/r/i", base_pred["catalog g/r/i colors"]),
                    ("catalog u/g/r/i", base_pred["catalog u/g/r/i colors"])]:
        pq = p[y[te] == 1]
        by_z[name] = [(np.mean(pq[(zq >= a) & (zq < b)] > 0.5), ((zq >= a) & (zq < b)).sum())
                      for a, b in zip(zbins[:-1], zbins[1:])]
        say(f"[by z] {name}: quasars called quasar, z bins {zbins}: " +
            " ".join(f"{100 * f:.0f}%(n={n})" for f, n in by_z[name]))
    star_ok = np.mean(p_cnn["color"][y[te] == 0] < 0.5)
    say(f"[by z] stars called star by the color CNN: {100 * star_ok:.1f}%")
    ps = p_cnn["shape"][y[te] == 1]
    shape_lo = np.mean(ps[zq < 0.5] > 0.5)
    shape_hi = np.mean(ps[zq > 1.5] > 0.5)
    say(f"[shape] shape-only CNN calls quasars quasar: z < 0.5 {100 * shape_lo:.0f}% (n = {(zq < 0.5).sum()}), "
        f"z > 1.5 {100 * shape_hi:.0f}% (n = {(zq > 1.5).sum()}); stars called star "
        f"{100 * np.mean(p_cnn['shape'][y[te] == 0] < 0.5):.0f}%")

    # ---- figures -----------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, (xc, yc, xl, yl) in zip(axes[:2], [(g_ - r_, r_ - i_, "g - r", "r - i"),
                                               (u_ - g_, g_ - r_, "u - g", "g - r")]):
        ax.scatter(xc[y == 0], yc[y == 0], s=3, c="0.6", label="stars")
        sc_ = ax.scatter(xc[y == 1], yc[y == 1], s=3, c=z[y == 1], cmap="viridis", vmin=0, vmax=4,
                         label="quasars (color = z)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8, markerscale=3)
    axes[0].set_xlim(-0.6, 1.8)
    axes[0].set_ylim(-0.6, 1.2)
    axes[1].set_xlim(-0.6, 3.5)
    axes[1].set_ylim(-0.6, 1.8)
    fig.colorbar(sc_, ax=axes[1], label="quasar redshift")
    axes[0].set_title("what a g/r/i image can see")
    axes[1].set_title("adding u: the ultraviolet excess")
    ax = axes[2]
    for name, mk in [("CNN, color images", "o-"), ("catalog g/r/i", "s--"), ("catalog u/g/r/i", "^:")]:
        f = [100 * a for a, n in by_z[name]]
        ax.plot([(a + min(b, 4.5)) / 2 for a, b in zip(zbins[:-1], zbins[1:])], f, mk, label=name)
    ax.set_xlabel("quasar redshift")
    ax.set_ylabel("quasars called quasar (%)")
    ax.set_ylim(0, 101)
    ax.legend(fontsize=8)
    ax.set_title("where quasars hide among stars")
    fig.savefig(OUT / "step6_colors.png", dpi=140)

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    names = ["CNN shape\n(normalized)", "CNN gray", "CNN color", "image\ncolors", "catalog\ng/r/i",
             "catalog\nu/g/r/i"]
    vals = [res["shape"][0], res["gray"][0], res["color"][0], base["colors measured from the images"],
            base["catalog g/r/i colors"], base["catalog u/g/r/i colors"]]
    errs = [res["shape"][1], res["gray"][1], res["color"][1], 0, 0, 0]
    ax.bar(names, [100 * v for v in vals], yerr=[100 * e for e in errs],
           color=["0.75", "0.5", "tab:blue", "tab:cyan", "tab:orange", "tab:red"])
    ax.axhline(50, c="k", ls=":")
    ax.set_ylim(40, 100)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Quasar vs star (brightness matched, r = 17-20)")
    fig.savefig(OUT / "step6_accuracy.png", dpi=140)

    te_idx = np.flatnonzero(te)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.8), constrained_layout=True)
    qsel = te_idx[y[te_idx] == 1]
    qsel = qsel[np.argsort(z[qsel])][np.linspace(0, len(qsel) - 1, 8).astype(int)]
    ssel = te_idx[y[te_idx] == 0][:8]
    pos = {k: j for j, k in enumerate(te_idx)}
    for row, (sel, name) in enumerate([(qsel, "real quasars\n(low to high z)"), (ssel, "real stars")]):
        for ax, k in zip(axes[row], sel):
            p = p_cnn["color"][pos[k]]
            ax.imshow(X_img[k])
            extra = f"z = {z[k]:.2f}, " if y[k] else ""
            ax.set_title(f"{extra}r = {rmag[k]:.1f}\nP(quasar) = {p:.2f}, called: {'quasar' if p > 0.5 else 'star'}",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Color CNN on test objects")
    fig.savefig(OUT / "step6_gallery.png", dpi=130)
    (OUT / "step6_log.txt").write_text("\n".join(log) + "\n")

    # ---- checks -------------------------------------------------------------------------------------
    filled = [k for k, (a, n) in enumerate(by_z["catalog g/r/i"]) if n >= 20]
    worst = filled[int(np.argmin([by_z["catalog g/r/i"][k][0] for k in filled]))]
    assert res["shape"][0] < res["gray"][0] - 0.03, "normalizing should remove the color leaking into grayscale"
    assert shape_lo > shape_hi, "shape should help most at low z, where host galaxies are visible"
    assert res["color"][0] > res["shape"][0] + 0.10, "color should separate quasars from stars"
    assert abs(res["color"][0] - base["colors measured from the images"]) < 0.04, \
        "the color CNN should reach the color information the images hold"
    assert base["catalog u/g/r/i colors"] > base["catalog g/r/i colors"] + 0.03, "the u band should help (UV excess)"
    assert 2.0 <= zbins[worst] < 3.5, "g/r/i should struggle most where quasars cross the stellar locus (z ~ 2.5-3)"
    print("PASS: normalizing removes the color leak, shape helps at low z (hosts), color separates, CNN reaches "
          "the image-color limit, u band helps, hardest at z ~ 2-3.5.")
