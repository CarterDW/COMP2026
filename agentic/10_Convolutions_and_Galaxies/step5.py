"""Step 5: known or unknown? Out-of-distribution (OOD) detection on a 4-class classifier.

Known: galaxy (DR16 Galaxy Zoo, confident), star (real SDSS), cosmic-ray track, satellite
trail; all on real SDSS sky, all through the same linear -> stretch -> JPEG round trip.
Unknown (never trained on): blended galaxy pairs, empty sky, and, in a leave-one-out
retrain, cosmic rays. Uncertain Galaxy Zoo galaxies are reported but not scored: they
ARE galaxies.

Scores (higher = more unknown): 1 - max softmax; energy = -logsumexp(logits);
Mahalanobis distance to the nearest class mean in the trunk's features (PCA to 256 dims,
shared covariance). Threshold: the 95th percentile of known validation scores.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

import data
import models
import sims
import step3 as s3

OUT = Path(__file__).parent / "results"
CLASSES = ["galaxy", "star", "cosmic ray", "trail"]
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def artifacts(kind, n, sky, rng):
    out = []
    fw = s3.FWHM_REAL
    for _ in range(n):
        u = s3.linear(sky[rng.integers(len(sky))])
        if kind == "cosmic ray":
            u = u + sims.cosmic_track(rng, np.exp(rng.uniform(np.log(300), np.log(5000))))
        else:
            u = u + sims.flicker_trail(rng, fw[rng.integers(len(fw))], np.exp(rng.uniform(np.log(5), np.log(60))))
        out.append(s3.finish(u))
    return np.stack(out)


def blends(gals, n, rng):
    out = []
    for _ in range(n):
        a, b = gals[rng.integers(len(gals))], gals[rng.integers(len(gals))]
        ub = s3.linear(b)
        ub = ub - s3.sky_level(ub)
        # only the galaxy's light, not its sky noise: its generous source footprint, soft-edged
        from scipy import ndimage
        foot = ndimage.gaussian_filter(sims.object_mask(b, 2.0, 8).astype(float), 3)
        ub = ub * foot[..., None]
        ang, sep = rng.uniform(0, 2 * np.pi), rng.uniform(12, 25)
        shift = (int(round(sep * np.sin(ang))), int(round(sep * np.cos(ang))))
        dy, dx = shift                      # zero-fill shift (no wrap-around); ub is sky-subtracted
        moved = np.zeros_like(ub)
        moved[max(dy, 0):ub.shape[0] + min(dy, 0), max(dx, 0):ub.shape[1] + min(dx, 0)] = \
            ub[max(-dy, 0):ub.shape[0] + min(-dy, 0), max(-dx, 0):ub.shape[1] + min(-dx, 0)]
        out.append(s3.finish(s3.linear(a) + moved))
    return np.stack(out)


def features(net, x, batch=500):
    net.eval()
    with torch.no_grad():
        f = torch.cat([net[:-1](x[i:i + batch].to(models.DEVICE)).cpu() for i in range(0, len(x), batch)])
        logits = torch.cat([net(x[i:i + batch].to(models.DEVICE)).cpu() for i in range(0, len(x), batch)])
    net.train()
    return f.numpy(), logits.numpy()


def fit_and_score(X_tr, y_tr, n_cls, X_eval, seed=0):
    """Train an n_cls classifier; return OOD scores and predicted class for each eval set."""
    x_tr, m, s = s3.tensor(X_tr)
    torch.manual_seed(seed)
    net = models.cnn(n_out=n_cls)
    models.train(net, x_tr, torch.tensor(y_tr), x_tr[:2], torch.tensor(y_tr[:2]), seed=seed)
    f_tr, _ = features(net, x_tr)
    pca = PCA(256, random_state=0).fit(f_tr)
    z_tr = pca.transform(f_tr)
    mus = np.stack([z_tr[y_tr == c].mean(0) for c in range(n_cls)])
    resid = z_tr - mus[y_tr]
    cov = resid.T @ resid / len(resid)
    prec = np.linalg.inv(cov + 1e-3 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0]))
    out = {}
    for name, X in X_eval.items():
        f, logits = features(net, s3.tensor(X, m, s)[0])
        z = pca.transform(f)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        d = np.stack([np.einsum("ij,jk,ik->i", z - mu, prec, z - mu) for mu in mus], 1)
        lse = logits.max(1) + np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1))
        out[name] = {"msp": 1 - p.max(1), "energy": -lse, "mahalanobis": d.min(1),
                     "pred": logits.argmax(1), "pmax": p.max(1)}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # ---- data (DR16 throughout) -------------------------------------------------------------
    images_dr16, have = s3.load_dr16_galaxies()
    _, sky = s3.load_dir("sky", np.load(OUT / "step3_sky_usable.npy"))
    _, rstars = s3.load_dir("stars")
    sp, stp = rng.permutation(len(sky)), rng.permutation(len(rstars))
    sky_tr, sky_va, sky_te = sky[sp[:1900]], sky[sp[1900:2500]], sky[sp[2500:]]
    st_tr, st_va, st_te = rstars[stp[:1800]], rstars[stp[1800:2400]], rstars[stp[2400:]]
    train_c, test_c = data.split(s3.labels)
    train_c, test_c = train_c[have[train_c]], test_c[have[test_c]]
    va_c, tr_c = train_c[:300], train_c[300:]
    unc = np.flatnonzero((s3.labels == 2) & have)
    prep = lambda idx: np.stack([s3.finish(s3.linear(images_dr16[i])) for i in idx])
    prep_v = lambda V: np.stack([s3.finish(s3.linear(v)) for v in V])

    def known(gal_idx, stars, sky_, n_art):
        X = [prep(gal_idx), prep_v(stars), artifacts("cosmic ray", n_art, sky_, rng),
             artifacts("trail", n_art, sky_, rng)]
        y = np.concatenate([np.full(len(x), c) for c, x in enumerate(X)])
        return np.concatenate(X), y

    X_tr, y_tr = known(tr_c, st_tr, sky_tr, 1500)
    X_va, y_va = known(va_c, st_va, sky_va, 300)
    X_te, y_te = known(test_c, st_te, sky_te, 300)
    unknown = {
        "blended pair": blends(images_dr16[test_c], 400, rng),
        "empty sky": prep_v(sky_te[:400]),
        "uncertain galaxy": prep(unc[:600]),
    }
    say(f"[data] train {len(X_tr)}, validation {len(X_va)}, known test {len(X_te)}; "
        + ", ".join(f"{k} {len(v)}" for k, v in unknown.items()))

    # ---- the 4-class model ------------------------------------------------------------------------
    sc = fit_and_score(X_tr, y_tr, 4, {"val": X_va, "test": X_te, **unknown})
    acc = np.mean(sc["test"]["pred"] == y_te)
    per = [np.mean(sc["test"]["pred"][y_te == c] == c) for c in range(4)]
    say(f"[classify] known test accuracy {100 * acc:.1f}% (" +
        ", ".join(f"{n} {100 * a:.1f}%" for n, a in zip(CLASSES, per)) + ")")
    SCORES = ["msp", "energy", "mahalanobis"]
    thr = {k: np.quantile(sc["val"][k], 0.95) for k in SCORES}
    rows = {}
    for k in SCORES:
        flag_known = np.mean(sc["test"][k] > thr[k])
        line = [f"known test flagged {100 * flag_known:.1f}%"]
        for name in unknown:
            s_u = sc[name][k]
            auc = roc_auc_score(np.r_[np.zeros(len(X_te)), np.ones(len(s_u))], np.r_[sc["test"][k], s_u])
            rows[(k, name)] = (auc, np.mean(s_u > thr[k]))
            line.append(f"{name}: AUROC {auc:.3f}, flagged {100 * np.mean(s_u > thr[k]):.0f}%")
        say(f"[ood] {k:11s} " + "; ".join(line))
    for name in unknown:
        called = np.bincount(sc[name]["pred"], minlength=4) / len(sc[name]["pred"])
        say(f"[ood] {name} called as: " + ", ".join(f"{n} {100 * c:.0f}%" for n, c in zip(CLASSES, called)))

    # ---- leave one out: no cosmic rays in training ----------------------------------------------------
    keep = y_tr != 2
    y3 = y_tr[keep].copy()
    y3[y3 == 3] = 2
    cr_te = X_te[y_te == 2]
    keep_v, keep_t = y_va != 2, y_te != 2
    sc3 = fit_and_score(X_tr[keep], y3, 3, {"val": X_va[keep_v], "test": X_te[keep_t], "cosmic ray": cr_te})
    for k in SCORES:
        t3 = np.quantile(sc3["val"][k], 0.95)
        auc = roc_auc_score(np.r_[np.zeros(keep_t.sum()), np.ones(len(cr_te))],
                            np.r_[sc3["test"][k], sc3["cosmic ray"][k]])
        rows[(k, "cosmic ray (held out)")] = (auc, np.mean(sc3["cosmic ray"][k] > t3))
        say(f"[leave-one-out] {k:11s} cosmic rays never seen: AUROC {auc:.3f}, flagged "
            f"{100 * np.mean(sc3['cosmic ray'][k] > t3):.0f}% (known flagged {100 * np.mean(sc3['test'][k] > t3):.1f}%)")
    called = np.bincount(sc3["cosmic ray"]["pred"], minlength=3) / len(cr_te)
    say("[leave-one-out] held-out cosmic rays called as: " +
        ", ".join(f"{n} {100 * c:.0f}%" for n, c in zip(["galaxy", "star", "trail"], called)))

    # ---- figures -------------------------------------------------------------------------------
    ood_sets = list(unknown) + ["cosmic ray (held out)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for j, k in enumerate(SCORES):
        ax = axes[0, j]
        # 1 - max softmax piles up near 0, so show it as log10
        tr = (lambda v: np.log10(v + 1e-7)) if k == "msp" else (lambda v: v)
        allv = tr(np.concatenate([sc["test"][k]] + [sc[n][k] for n in unknown]))
        bins = np.linspace(np.quantile(allv, 0.005), np.quantile(allv, 0.995), 50)
        ax.hist(tr(sc["test"][k]), bins, density=True, histtype="stepfilled", alpha=0.35, color="0.4",
                label="known test")
        for n in unknown:
            ax.hist(tr(sc[n][k]), bins, density=True, histtype="step", lw=1.8, label=n)
        ax.axvline(tr(thr[k]), c="k", ls=":", label="threshold (5% of known)")
        ax.set_title({"msp": "max softmax", "energy": "energy", "mahalanobis": "Mahalanobis"}[k])
        ax.set_xlabel("log10(1 - max softmax)" if k == "msp" else "OOD score (higher = more unknown)")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        for n in ood_sets:
            if n == "cosmic ray (held out)":
                s0, s1 = sc3["test"][k], sc3["cosmic ray"][k]
            else:
                s0, s1 = sc["test"][k], sc[n][k]
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(s0)), np.ones(len(s1))], np.r_[s0, s1])
            ax.plot(fpr, tpr, label=f"{n} ({rows[(k, n)][0]:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("known objects flagged")
        ax.set_ylabel("unknowns flagged")
        ax.legend(fontsize=7, title="AUROC", title_fontsize=7)
    fig.savefig(OUT / "step5_ood.png", dpi=140)

    # gallery: each unknown set, the classifier's call and the Mahalanobis verdict
    fig, axes = plt.subplots(4, 7, figsize=(14, 9.4), constrained_layout=True)
    gsets = [("blended pair", unknown["blended pair"], sc["blended pair"], CLASSES),
             ("empty sky", unknown["empty sky"], sc["empty sky"], CLASSES),
             ("uncertain galaxy", unknown["uncertain galaxy"], sc["uncertain galaxy"], CLASSES),
             ("cosmic ray\n(never trained on)", cr_te, sc3["cosmic ray"], ["galaxy", "star", "trail"])]
    t_m3 = np.quantile(sc3["val"]["mahalanobis"], 0.95)
    for row, (name, X, s, names) in enumerate(gsets):
        t = t_m3 if row == 3 else thr["mahalanobis"]
        for c, ax in enumerate(axes[row]):
            ax.imshow(X[c])
            flag = s["mahalanobis"][c] > t
            ax.set_title(f"called: {names[s['pred'][c]]} (P = {s['pmax'][c]:.2f})\n"
                         f"{'flagged: unknown' if flag else 'accepted as known'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Objects the classifier never trained on: its best guess, and whether the "
                 "Mahalanobis score flags them")
    fig.savefig(OUT / "step5_gallery.png", dpi=125)
    (OUT / "step5_log.txt").write_text("\n".join(log) + "\n")

    # ---- checks ---------------------------------------------------------------------------------
    for k in SCORES:
        fk = np.mean(sc["test"][k] > thr[k])
        assert 0.02 < fk < 0.10, f"{k}: known test objects should be flagged about 5% of the time, got {fk:.3f}"
    assert acc > 0.9, "the 4-class classifier should work on known objects"
    best_held = max(rows[(k, "cosmic ray (held out)")][0] for k in SCORES)
    assert best_held > 0.7, "at least one score should flag never-seen cosmic rays"
    print("PASS: ~5% of known objects flagged at the threshold, classifier works, held-out cosmic rays "
          "flagged by at least one score.")
