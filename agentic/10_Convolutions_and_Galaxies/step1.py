"""Step 1: the data, the color proxy c, and notebook 10's CNN at 60x60 and 120x120."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import data
import models

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

images, labels, ids = data.load()
train_idx, test_idx = data.split(labels)
unc_idx = np.flatnonzero(labels == 2)
print(f"{len(images)} galaxies: {np.sum(labels == 0)} spiral, {np.sum(labels == 1)} elliptical, "
      f"{np.sum(labels == 2)} uncertain;  train {len(train_idx)}, test {len(test_idx)}")
baseline = np.mean(labels[test_idx] == 0)
print(f"always-spiral baseline on test: {100 * baseline:.1f}%  (notebook: 76%)")

# --- the color proxy ---------------------------------------------------------------
c = data.color_proxy(images)
names = ["spiral", "elliptical", "uncertain"]
for k in range(3):
    print(f"c for {names[k]:10s}: median {np.median(c[labels == k]):+.3f}, "
          f"16-84%: {np.percentile(c[labels == k], 16):+.3f} .. {np.percentile(c[labels == k], 84):+.3f}")

# one threshold on c, chosen on the training set ("redder than t" = elliptical)
grid = np.linspace(c.min(), c.max(), 2001)
train_acc = [np.mean((c[train_idx] > t) == (labels[train_idx] == 1)) for t in grid]
t_best = grid[int(np.argmax(train_acc))]
color_acc = np.mean((c[test_idx] > t_best) == (labels[test_idx] == 1))
print(f"color cut c > {t_best:+.3f} => elliptical: test accuracy {100 * color_acc:.1f}%")

fig, ax = plt.subplots(figsize=(6, 3.5))
bins = np.linspace(np.percentile(c, 0.5), np.percentile(c, 99.5), 60)
for k, col in zip(range(3), ["tab:blue", "tab:red", "0.5"]):
    ax.hist(c[labels == k], bins, histtype="step", lw=2 if k < 2 else 1.2, color=col,
            density=True, label=f"{names[k]} ({np.sum(labels == k)})", ls="-" if k < 2 else "--")
ax.axvline(t_best, c="k", ls=":", label=f"best cut, {100 * color_acc:.1f}% on test")
ax.set_xlabel(r"$c = \ln(\langle i\rangle / \langle g\rangle)$ in the central 8 px  (redder $\rightarrow$)")
ax.set_ylabel("density")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step1_color_hist.png", dpi=150)

# a strip of galaxies ordered by c, for showing
fig, axes = plt.subplots(2, 10, figsize=(13, 3.6))
for row, k in enumerate([0, 1]):
    pick = np.flatnonzero(labels == k)
    pick = pick[np.argsort(c[pick])][np.linspace(0, np.sum(labels == k) - 1, 10).astype(int)]
    for ax, i in zip(axes[row], pick):
        ax.imshow(images[i])
        ax.set_title(f"c={c[i]:+.2f}", fontsize=7)
        ax.axis("off")
    axes[row, 0].text(-20, 60, names[k], rotation=90, va="center", fontsize=9)
fig.suptitle("Galaxy Zoo, ordered blue to red within each class", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "step1_color_strip.png", dpi=150)

# --- notebook 10's CNN, at 60x60 (control) and 120x120 -------------------------------
y = torch.tensor(labels)
SEEDS = range(5)
CONFIGS = [(60, 2), (120, 2), (120, 3)]          # (image size, conv blocks)
results, finals = {}, {}
for size, blocks in CONFIGS:
    imgs = images if size == 120 else images.reshape(-1, 60, 2, 60, 2, 3).mean(axis=(2, 4))
    x_all = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean, std = x_all[train_idx].mean(), x_all[train_idx].std()
    x = (x_all - mean) / std
    key = (size, blocks)
    finals[key] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = models.cnn(size=size, blocks=blocks)
        print(f"\nCNN {size}x{size}, {blocks} blocks, seed {seed}: {models.n_params(net):,} parameters")
        hist = models.train(net, x[train_idx], y[train_idx], x[test_idx], y[test_idx], seed=seed)
        finals[key].append(hist[-1])
        if seed == 0:
            results[key] = (models.n_params(net), hist)
    f = 100 * np.array(finals[key])
    print(f"CNN {size}x{size}, {blocks} blocks, over {len(f)} seeds: "
          f"{f.mean():.1f} +/- {f.std(ddof=1):.1f}%  ({np.round(f, 1)})")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ep = np.arange(1, 16)
for key, mk in zip(CONFIGS, ["o-", "s-", "^-"]):
    p, h = results[key]
    ax.plot(ep, 100 * np.array(h), mk, label=f"CNN {key[0]}x{key[0]}, {key[1]} blocks, {p:,} params")
ax.axhline(100 * color_acc, c="tab:red", ls="--", label="one color cut")
ax.axhline(100 * baseline, c="k", ls=":", label="always say spiral")
ax.set_xlabel("epoch")
ax.set_ylabel("test accuracy (%)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step1_cnn.png", dpi=150)

# --- checks --------------------------------------------------------------------------
print()
assert results[(60, 2)][0] == 8434, "the 60x60 CNN should match notebook 10's 8,434 parameters"
assert np.median(c[labels == 1]) > np.median(c[labels == 0]), "ellipticals should be redder"
assert color_acc > baseline + 0.05, "a color cut alone should clearly beat the baseline"
m_sp, m_el, m_un = (np.median(c[labels == k]) for k in range(3))
assert m_sp < m_un < m_el, "uncertain galaxies should sit between the classes in color"
# one test set of 400 has sampling error ~1.2 pt, so compare seed means, not single runs
for key in [(120, 2), (120, 3)]:
    gap = 100 * (np.mean(finals[(60, 2)]) - np.mean(finals[key]))
    print(f"60x60 minus {key[0]}x{key[0]} with {key[1]} blocks, seed means: {gap:+.1f} pt")
gap3 = 100 * (np.mean(finals[(60, 2)]) - np.mean(finals[(120, 3)]))
assert abs(gap3) < 1.5, "the 3-block 120x120 CNN should match 60x60 within 1.5 pt on the seed mean"
print("PASS: parameter count, color ordering, color-cut baseline, uncertain in between, "
      "3-block 120 ~ 60.")
