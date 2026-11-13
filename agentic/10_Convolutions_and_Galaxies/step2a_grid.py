"""Step 2a, part 1: find field stars in the cutouts and fit the global stretch A and Moffat beta.

Each candidate is fit with mean(v) = A asinh(F M(r; fwhm, beta) + b). Clean stars are
compact (free-fit FWHM < 8 px, rms < 8), isolated (stamp border at sky level) and round.
A and beta are shared by all stars; F, b, center and FWHM are fit per star.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import data
import fieldstars as fs

OUT = Path(__file__).parent / "results"
images, labels, ids = data.load()
A_GRID = [8, 12, 18, 25, 35, 50, 70, 100, 150]
B_GRID = [2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 9.0]


def free_fits(i):
    return [(i, y, x, *fs.fit(images[i], y, x)) for y, x in fs.candidates(images[i])]


def grid_point(args):
    A, beta, clean = args          # passed in: workers start fresh on Python 3.14
    out = [fs.fit(images[i], y, x, A=A, beta=beta) for i, y, x in clean]
    return A, beta, sum(o[1] for o in out), np.median([o[0][5] for o in out])


if __name__ == "__main__":
    with Pool(16) as p:
        cands = [c for chunk in p.map(free_fits, range(len(images))) for c in chunk]
    print(f"{len(cands)} candidates in {len({c[0] for c in cands})} images")
    clean = np.array([(i, y, x) for i, y, x, par, chi2, rms, peak in cands
                      if rms < 8 and par[5] < 8 and fs.isolated_and_round(images[i], y, x)])
    print(f"clean stars: {len(clean)}")
    np.save(OUT / "step2a_clean_stars.npy", clean)

    with Pool(16) as p:
        res = np.array(p.map(grid_point, [(A, b, clean) for A in A_GRID for b in B_GRID]))
    np.save(OUT / "step2a_Abeta_grid.npy", res)
    k = res[:, 2].argmin()
    print(f"best A = {res[k, 0]:.0f}, beta = {res[k, 1]:.1f}; median FWHM there {res[k, 3]:.2f} px")
    print("delta chi^2 (rows A, columns beta =", B_GRID, ")")
    for A in A_GRID:
        print(f"A={A:4d}", " ".join(f"{c - res[k, 2]:9.0f}" for c in res[res[:, 0] == A][:, 2]))
