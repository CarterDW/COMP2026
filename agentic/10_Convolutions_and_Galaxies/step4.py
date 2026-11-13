"""Step 4: the galaxy head. One trunk, two heads: morphology (spiral/elliptical, Galaxy Zoo)
and color (red/blue from SDSS u - r, cut at 2.22, Strateva et al. 2001).

Versions, each 3 seeds: (a) full color, (b) grayscale (shape only), (c) pixels shuffled
within each image (color only). Images: the notebook-10 JPGs; split: notebook 10's.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import data
import models

OUT = Path(__file__).parent / "results"
PHOT = data.NB / "data" / "sdss_week10" / "galaxy_photometry.csv"
UR_CUT = 2.22
SEEDS = [0, 1, 2]
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


images, labels, ids = data.load()
train_idx, test_idx = data.split(labels)
phot = {r["dr7objid"]: r for r in csv.DictReader(open(PHOT))}
ur = np.array([float(phot[i]["modelMag_u"]) - float(phot[i]["modelMag_r"]) if i in phot else np.nan
               for i in ids])
c_img = data.color_proxy(images)


def version(kind, rng):
    x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if kind == "gray":
        x = x.mean(1, keepdim=True)
    if kind == "shuffled":                          # same pixels (RGB triples), random places
        flat = x.reshape(len(x), 3, -1)
        perm = torch.stack([torch.from_numpy(rng.permutation(flat.shape[-1])) for _ in range(len(x))])
        x = torch.gather(flat, 2, perm[:, None, :].expand(-1, 3, -1)).reshape(x.shape)
    m, s = x[train_idx].mean(), x[train_idx].std()
    return (x - m) / s


if __name__ == "__main__":
    # ---- labels ---------------------------------------------------------------------------
    have = np.isfinite(ur)
    train = train_idx[have[train_idx]]
    test = test_idx[have[test_idx]]
    red = (ur > UR_CUT).astype(int)
    morph = (labels == 1).astype(int)                 # 1 = elliptical
    edges = np.linspace(0.5, 3.5, 61)
    centers = (edges[:-1] + edges[1:]) / 2
    # the dip of the WHOLE sample (all galaxies, any morphology), smoothed over 5 bins,
    # searched between the blue (~1.9) and red (~2.9) peaks
    hist_all, _ = np.histogram(ur[have], bins=edges)
    smooth = np.convolve(hist_all, np.ones(5) / 5, "same")
    win = (centers > 1.9) & (centers < 2.8)
    dip = centers[win][np.argmin(smooth[win])]
    say(f"[labels] u - r for {have.sum()} galaxies; whole-sample histogram minimum between the peaks at "
        f"{dip:.2f} (Strateva cut {UR_CUT})")
    conf = labels < 2
    for name, k in [("spirals", 0), ("ellipticals", 1)]:
        sel = have & (labels == k)
        say(f"[labels] {name}: {100 * np.mean(red[sel]):.0f}% red (u - r > {UR_CUT}); n = {sel.sum()}")
    say(f"[labels] train {len(train)}, test {len(test)} confident galaxies with u - r")

    # baselines
    t_c = np.linspace(c_img.min(), c_img.max(), 2001)
    best_t = t_c[np.argmax([np.mean((c_img[train] > t) == morph[train]) for t in t_c])]
    base = {
        "morph: always spiral": np.mean(morph[test] == 0),
        "morph: image-color cut": np.mean((c_img[test] > best_t) == morph[test]),
        "color: always majority": max(np.mean(red[test]), 1 - np.mean(red[test])),
        "color: from morphology label": np.mean(morph[test] == red[test]),
    }
    for k, v in base.items():
        say(f"[baseline] {k}: {100 * v:.1f}%")

    # ---- the three versions ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    res, preds, nets = {}, {}, {}
    for kind in ["color", "gray", "shuffled"]:
        x = version(kind, rng)
        accs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = models.TwoHead(in_ch=x.shape[1])
            models.train_two_head(net, x[train], torch.tensor(morph[train]), torch.tensor(red[train]), seed=seed)
            pm, pc = models.predict_two_head(net, x[test])
            accs.append((np.mean((pm > 0.5) == morph[test]), np.mean((pc > 0.5) == red[test])))
            if seed == 0:
                preds[kind], nets[kind] = (pm, pc), (net, x)
        accs = np.array(accs)
        res[kind] = (accs.mean(0), accs.std(0, ddof=1))
        say(f"[{kind:8s}] morphology {100 * accs[:, 0].mean():.1f} +/- {100 * accs[:, 0].std(ddof=1):.1f}%, "
            f"color {100 * accs[:, 1].mean():.1f} +/- {100 * accs[:, 1].std(ddof=1):.1f}%")

    # ---- does shape predict color, and where does it fail? ----------------------------------------
    pc_gray = preds["gray"][1]
    off = morph[test] != red[test]                    # red spirals and blue ellipticals
    err = (pc_gray > 0.5) != red[test]
    say(f"[shape->color] grayscale color-head error: {100 * err[~off].mean():.1f}% on typical galaxies "
        f"(blue spirals, red ellipticals; n = {(~off).sum()}), {100 * err[off].mean():.1f}% on red spirals "
        f"and blue ellipticals (n = {off.sum()})")

    # ---- are red spirals edge-on? axis ratio b/a from second moments ------------------------------
    yy, xx = np.mgrid[:120, :120] - 59.5

    def axis_ratio(v):
        g = v.astype(float).mean(-1)
        w = np.clip(g - np.median(g[np.hypot(xx, yy) > 50]), 0, None) * (np.hypot(xx, yy) < 40)
        mxx, myy, mxy = (w * xx * xx).sum(), (w * yy * yy).sum(), (w * xx * yy).sum()
        lam = np.linalg.eigvalsh(np.array([[mxx, mxy], [mxy, myy]]))
        return np.sqrt(lam[0] / lam[1])

    q = np.array([axis_ratio(v) for v in images])
    sp = have & (labels == 0)
    from scipy import stats
    mw = stats.mannwhitneyu(q[sp & (red == 1)], q[sp & (red == 0)])
    say(f"[inclination] spirals' axis ratio b/a: red median {np.median(q[sp & (red == 1)]):.2f}, "
        f"blue median {np.median(q[sp & (red == 0)]):.2f} (Mann-Whitney p = {mw.pvalue:.1e})")
    qb = np.quantile(q[sp], np.linspace(0, 1, 6))
    red_by_q = [np.mean(red[sp & (q >= a) & (q <= b)]) for a, b in zip(qb[:-1], qb[1:])]
    say("[inclination] red fraction of spirals by b/a quintile (edge-on -> face-on): "
        + " ".join(f"{100 * f:.0f}%" for f in red_by_q))
    tsp = morph[test] == 0
    rho = stats.spearmanr(q[test][tsp], pc_gray[tsp])
    say(f"[inclination] test spirals: Spearman(b/a, shape-only P(red)) = {rho.statistic:+.2f} "
        f"(p = {rho.pvalue:.1e})")

    # ---- rotation -------------------------------------------------------------------------------
    net, x = nets["color"]
    xt = x[test]
    pm0, _ = models.predict_two_head(net, xt)
    pm90, _ = models.predict_two_head(net, torch.rot90(xt, 1, dims=(2, 3)))
    acc0, acc90 = np.mean((pm0 > 0.5) == morph[test]), np.mean((pm90 > 0.5) == morph[test])
    d4_views = [models.predict_two_head(net, torch.rot90(v, k, dims=(2, 3)), d4=True)[0]
                for v in (xt, xt.flip(3)) for k in range(4)]
    d4_spread = np.max(np.ptp(np.stack(d4_views), axis=0))
    acc_d4 = np.mean((d4_views[0] > 0.5) == morph[test])
    say(f"[rotation] morphology accuracy: original {100 * acc0:.1f}%, rotated 90 deg {100 * acc90:.1f}%, "
        f"D4-averaged {100 * acc_d4:.1f}%; D4 model's max output change over the 8 views {d4_spread:.1e}")

    # ---- figures ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].hist(ur[have & (labels == 0)], edges, histtype="step", lw=2, color="tab:blue", label="spirals")
    axes[0].hist(ur[have & (labels == 1)], edges, histtype="step", lw=2, color="tab:red", label="ellipticals")
    axes[0].hist(ur[have & (labels == 2)], edges, histtype="step", lw=1, color="0.5", ls="--", label="uncertain")
    axes[0].axvline(UR_CUT, c="k", ls=":", label=f"u - r = {UR_CUT}")
    axes[0].set_xlabel("SDSS u - r (model magnitudes)")
    axes[0].set_ylabel("galaxies")
    axes[0].legend(fontsize=8)
    kinds = ["color", "gray", "shuffled"]
    xs = np.arange(len(kinds))
    for off_, head, lab in [(-0.18, 0, "morphology"), (0.18, 1, "color (u - r)")]:
        axes[1].bar(xs + off_, [100 * res[k][0][head] for k in kinds], 0.34,
                    yerr=[100 * res[k][1][head] for k in kinds], label=lab)
    axes[1].axhline(100 * base["morph: always spiral"], c="tab:blue", ls=":", lw=1)
    axes[1].axhline(100 * base["color: always majority"], c="tab:orange", ls=":", lw=1)
    axes[1].set_xticks(xs, ["full color", "grayscale\n(shape only)", "shuffled\n(color only)"])
    axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_ylim(50, 100)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_title("dotted: majority-class baselines", fontsize=9)
    cm = np.array([[np.sum((morph[test] == t) & ((pm0 > 0.5) == p)) for p in (0, 1)] for t in (0, 1)])
    axes[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    axes[2].set_xticks([0, 1], ["called spiral", "called elliptical"])
    axes[2].set_yticks([0, 1], ["spiral", "elliptical"])
    axes[2].set_title("morphology head, full color (seed 0)", fontsize=9)
    fig.savefig(OUT / "step4_summary.png", dpi=150)

    # red spirals and blue ellipticals, with the shape-only color head's verdict
    # drawn from the whole confident sample (blue ellipticals are ~1%: only a handful exist);
    # each title says whether the galaxy was in training, where the prediction is optimistic
    gnet, gx = nets["gray"]
    _, pc_all = models.predict_two_head(gnet, gx)
    in_test = np.zeros(len(images), bool)
    in_test[test] = True
    rs = np.flatnonzero(have & (labels == 0) & (red == 1))
    be = np.flatnonzero(have & (labels == 1) & (red == 0))
    rs = np.r_[rs[in_test[rs]][:6], rs[~in_test[rs]][:2]]           # mostly test galaxies
    be = be[np.argsort(~in_test[be])][:8]
    sets = [(f"red spirals\n({have.sum() and np.sum(have & (labels == 0) & (red == 1))} in all)", rs),
            (f"blue ellipticals\n({np.sum(have & (labels == 1) & (red == 0))} in all)", be)]
    ncol = max(len(rs), len(be))
    fig, axes = plt.subplots(2, ncol, figsize=(1.75 * ncol, 4.9), constrained_layout=True)
    for row, (name, idx) in enumerate(sets):
        for ax, k in zip(axes[row], idx):
            p = pc_all[k]
            ax.imshow(images[k])
            ax.set_title(f"u - r = {ur[k]:.2f} ({'test' if in_test[k] else 'train'})\n"
                         f"P(red) = {p:.2f}, called: {'red' if p > 0.5 else 'blue'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[row][len(idx):]:
            ax.set_visible(False)
        axes[row, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Galaxies whose color goes against their shape: color predicted from shape alone (grayscale CNN)")
    fig.savefig(OUT / "step4_offdiagonal.png", dpi=130)
    (OUT / "step4_log.txt").write_text("\n".join(log) + "\n")

    # ---- checks ------------------------------------------------------------------------------------
    sh_m, full_m = res["shuffled"][0][0], res["color"][0][0]
    assert abs(dip - UR_CUT) < 0.25, "the whole-sample u - r dip should sit near the Strateva cut"
    assert sh_m < full_m, "destroying spatial structure should cost some morphology accuracy"
    assert res["gray"][0][1] > base["color: always majority"] + 0.05, "shape alone should predict color"
    assert mw.pvalue < 1e-3 and np.median(q[sp & (red == 1)]) < np.median(q[sp & (red == 0)]), \
        "red spirals should be more edge-on (dust)"
    assert abs(acc90 - acc0) < 0.03, "a 90-degree rotation should barely change accuracy"
    assert d4_spread < 1e-5, "the D4-averaged model should be exactly invariant"
    print("PASS: u - r dip near 2.22, spatial structure helps, shape predicts color, red spirals are "
          "edge-on, rotation-robust, D4 model exactly invariant.")
