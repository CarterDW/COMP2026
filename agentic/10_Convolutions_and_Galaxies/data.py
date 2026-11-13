"""Galaxy Zoo images at full 120x120, with notebook 10's train/test split.

The JPGs are SDSS color composites: channel R = i band, G = r band, B = g band, with
an asinh stretch (Lupton et al. 2004), so pixel values are not linear in flux.
"""
import ast
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NB = Path(__file__).resolve().parents[2] / "notebooks"
DIR = NB / "GalaxyZoo"
CLASSES = ["spiral", "elliptical"]


def unzip():
    if not DIR.is_dir():
        with zipfile.ZipFile(NB / "GalaxyZoo.zip") as z:
            z.extractall(NB)


def load():
    """Images (N, 120, 120, 3) uint8 and labels: 0 spiral, 1 elliptical, 2 uncertain."""
    unzip()
    rows = ast.literal_eval(open(DIR / "training_data.txt").read())
    images = np.stack([np.asarray(Image.open(DIR / f"{r[0]}.jpg")) for r in rows])
    labels = np.array([[r[1], r[2], r[3]].index(1) for r in rows])
    ids = np.array([r[0] for r in rows])
    return images, labels, ids


def split(labels):
    """Notebook 10's split of the confident galaxies, as indices into the full arrays.

    The notebook permutes the confident rows only, so do the same and map back.
    """
    confident = np.flatnonzero(labels < 2)
    perm = torch.randperm(len(confident), generator=torch.Generator().manual_seed(0)).numpy()
    n_test = len(confident) // 5
    return confident[perm[n_test:]], confident[perm[:n_test]]


def to_tensor(images, mean, std):
    x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return (x - mean) / std


def color_proxy(images, radius=8):
    """c = ln(<R>/<B>) = ln(<i>/<g>) in a central aperture. Larger c is redder.

    It tracks g - i in order, not in value, because of the asinh stretch.
    """
    n = images.shape[1]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    ap = xx**2 + yy**2 <= radius**2
    pix = images[:, ap, :].astype(float).mean(axis=1)    # (N, 3) mean R, G, B
    return np.log(pix[:, 0] / pix[:, 2])
