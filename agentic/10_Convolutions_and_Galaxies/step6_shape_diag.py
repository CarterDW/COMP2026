"""Step 6 diagnostic: shape-only quasar/star CNNs on single bands (field-matched pairs).
A single JPG channel still carries color: JPEG stores chroma at half resolution and each
band's S/N depends on color, so the blue channel (least luminance weight) leaks most."""
import sys, csv, io, contextlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, torch
from PIL import Image
import step6 as s6, models, sims
SD, C = s6.SD, s6.C
srows = list(csv.DictReader(open(SD / "stars_matched" / "index.csv")))
by_id = {}
for f in ["quasars", "quasars_hiz", "quasars_hiz2"]:
    for r in csv.DictReader(open(SD / f / "index.csv")): by_id[r["objid"]] = (f, r)
pairs = [(s, *by_id[s["partner"]]) for s in srows if s["partner"] in by_id]
X = np.concatenate([s6.load("stars_matched", [p[0] for p in pairs]),
                    np.stack([np.asarray(Image.open(SD / f / f"{r['name']}.jpg"))[60-C:60+C, 60-C:60+C] for _, f, r in pairs])])
n = len(pairs); y = np.r_[np.zeros(n), np.ones(n)].astype(int)
z = np.r_[np.zeros(n), [float(r["z"]) for _, _, r in pairs]]
rng = np.random.default_rng(0); tp = rng.permutation(n)[:n // 5]
te = np.zeros(2 * n, bool); te[tp] = True; te[tp + n] = True; tr = ~te
yy, xx = np.mgrid[:2*C, :2*C]; R = np.hypot(yy - (C - .5), xx - (C - .5))
U = sims.unstretch(X, s6.A)                       # linear, per channel (i, r, g)
def norm(img):
    img = img - np.median(img[:, R > 18], axis=1)[:, None, None]
    return img / np.clip(img[:, R < 8].sum(1), 1e-3, None)[:, None, None]
def run(img):
    x = torch.tensor(img[:, None], dtype=torch.float32); x = (x - x[tr].mean()) / x[tr].std(); accs = []; p0 = None
    for seed in (0, 1, 2):
        torch.manual_seed(seed); net = models.cnn(in_ch=1, size=2*C)
        with contextlib.redirect_stdout(io.StringIO()):
            models.train(net, x[tr], torch.tensor(y[tr]), x[te], torch.tensor(y[te]), seed=seed)
        net.eval()
        with torch.no_grad(): p = torch.softmax(net(x[te].cuda()), 1)[:, 1].cpu().numpy()
        accs.append(np.mean((p > .5) == y[te])); p0 = p if p0 is None else p0
    return np.mean(accs), np.std(accs, ddof=1), p0
zq = z[te][y[te] == 1]
for name, img in [("all bands mixed (as in step 6)", norm(U.mean(-1))), ("i band only", norm(U[..., 0])),
                  ("r band only", norm(U[..., 1])), ("g band only", norm(U[..., 2]))]:
    a, e, p = run(img); pq = p[y[te] == 1]
    print(f"{name:32s}: {100*a:.1f} +/- {100*e:.1f}%   quasars called quasar z<1 {100*np.mean(pq[zq<1]>.5):.0f}%, z>2.5 {100*np.mean(pq[zq>2.5]>.5):.0f}%", flush=True)
# DCR-like signal: offset between g and i centroids
def centroid(img):
    w = np.clip(img - np.median(img[:, R > 18], axis=1)[:, None, None], 0, None) * (R < 6)
    return np.stack([(w * yy).sum((1, 2)) / w.sum((1, 2)), (w * xx).sum((1, 2)) / w.sum((1, 2))], 1)
d = np.linalg.norm(centroid(U[..., 2]) - centroid(U[..., 0]), axis=1)
print(f"g-i centroid offset (px): stars median {np.median(d[:n]):.3f}, quasars z<1 {np.median(d[n:][z[n:]<1]):.3f}, z>2.5 {np.median(d[n:][z[n:]>2.5]):.3f}")
