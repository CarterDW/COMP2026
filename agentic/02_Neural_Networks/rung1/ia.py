"""Projected position-shape correlations in a periodic simulation box, via TreeCorr.

Estimator
---------
The standard IA estimator (Mandelbaum+2006, Joachimi+2011) specialised to a
periodic cube, where the random-random pair count is analytic and no random
catalogue is needed:

    w_g+(r_p) = L * sum_pairs e_+(j|i) / [ N_D * (N_S / L^2) * pi * (rp_hi^2 - rp_lo^2) ]

Pairs run over a density tracer i and a shape galaxy j. TreeCorr does the pair
work; the normalisation above is applied afterwards. The factor L appears
because the box is projected along its full depth, so the line-of-sight integral
w_g+ = int dPi xi_g+ runs over 2*Pi_max = L.

Why full-depth projection instead of a |Pi| < Pi_max cut
-------------------------------------------------------
TreeCorr's ``min_rpar``/``max_rpar`` define Rparallel as the difference in
*radial distance from the origin* -- an observer-centred line of sight. A
simulation box has a plane-parallel line of sight along a box axis, which is not
the same thing. This was confirmed directly (see checks.check_rpar_is_radial):
two points 5 units apart purely in z, placed at x=1000, pass a |rpar| < 1 cut,
because their radial distances differ by only 0.0125. Using rpar here would
silently apply a cut nobody intended.

So the line of sight is handled by collapsing the box: the correlation is run in
flat 2D coordinates (x, y) with TreeCorr's ``Periodic`` metric, which wraps
correctly in both directions. The cost is that all uncorrelated line-of-sight
pairs are retained, so the noise is higher than a Pi_max ~ 40 Mpc/h cut would
give. The benefit is that the estimator is exact, has no free Pi_max to argue
about, and -- since rung 1 compares the *same* statistic across resolutions --
a common noise floor is harmless. If you later need the truncated statistic,
the correct way in a periodic box is to sum 2D correlations over pairs of
z-slabs, not to reach for rpar.

Sign convention
---------------
e_+ > 0 means radial alignment: the major axis points along the separation
vector. TreeCorr reports the lensing tangential shear, whose sign is the
opposite (verified: a radially aligned shape gives ``ng.xi = -1``), so this
module negates it once, here, and nowhere else. Observational IA papers vary;
convert at the point of comparison.

w_gx, the 45-degree rotated component, must vanish by parity. It is a null test,
not a measurement.
"""

import numpy as np
import treecorr


def _catalogs(pos_d, pos_s, e1, e2, boxsize, npatch):
    """Density and shape catalogues, projected along z, sharing patch centres."""
    xy_d = np.mod(np.asarray(pos_d, dtype=float)[:, :2], boxsize)
    xy_s = np.mod(np.asarray(pos_s, dtype=float)[:, :2], boxsize)
    shapes = treecorr.Catalog(
        x=xy_s[:, 0], y=xy_s[:, 1],
        g1=np.asarray(e1, dtype=float), g2=np.asarray(e2, dtype=float),
        npatch=npatch,
    )
    density = treecorr.Catalog(
        x=xy_d[:, 0], y=xy_d[:, 1],
        patch_centers=shapes.patch_centers if npatch > 1 else None,
    )
    return density, shapes


def correlate(pos_d, pos_s, e1, e2, boxsize, min_rp, max_rp, nbins,
              npatch=1, num_threads=None):
    """w_g+, w_gx and w_p for one sample. Lengths all in the same units as boxsize.

    pos_d  -- (N, 3) density tracers. Only x, y are used; z is projected out.
    pos_s  -- (M, 3) shape galaxies, with ellipticity components e1, e2.
    npatch -- if > 1, TreeCorr patches for a jackknife covariance. The jackknife
              is computed on w_g+ itself (not on TreeCorr's mean-shear xi), so
              the analytic normalisation is inside the resampling.

    Returns a dict of arrays. ``sum_eplus`` and ``npairs`` are kept because a
    normalisation bug is invisible in w_g+ alone, and because the per-pair mean
    e_+ is what the injection test in checks.py pins down.
    """
    if min_rp <= 0.0:
        raise ValueError("min_rp must be > 0; it is also what excludes self-pairs")
    if max_rp >= boxsize / 2.0:
        raise ValueError(
            f"max_rp={max_rp} must be < boxsize/2={boxsize / 2} for the minimum "
            "image separation to be unique"
        )

    density, shapes = _catalogs(pos_d, pos_s, e1, e2, boxsize, npatch)
    config = dict(min_sep=min_rp, max_sep=max_rp, nbins=nbins,
                  metric="Periodic", period=boxsize, num_threads=num_threads)
    if npatch > 1:
        config["var_method"] = "jackknife"

    ng = treecorr.NGCorrelation(**config)
    ng.process(density, shapes)
    nn = treecorr.NNCorrelation(**config)
    nn.process(density, shapes)

    # Analytic RR for a periodic box: N_D pairs against a uniform surface
    # density N_S / L^2 in each annulus. Verified in checks.check_analytic_rr.
    area = np.pi * (ng.right_edges**2 - ng.left_edges**2)
    rr = len(density.x) * (len(shapes.x) / boxsize**2) * area

    def w_gplus(corrs):
        return -boxsize * corrs[0].xi * corrs[0].weight / rr

    def w_gcross(corrs):
        return -boxsize * corrs[0].xi_im * corrs[0].weight / rr

    def w_p(corrs):
        return boxsize * (corrs[1].npairs / rr - 1.0)

    result = {
        "rp": ng.meanr,
        "rp_nom": ng.rnom,
        "npairs": ng.npairs,
        "sum_eplus": -ng.xi * ng.weight,
        "mean_eplus": -ng.xi,
        "w_gplus": w_gplus([ng, nn]),
        "w_gcross": w_gcross([ng, nn]),
        "w_p": w_p([ng, nn]),
        "rr_analytic": rr,
    }

    if npatch > 1:
        for name, func in (("w_gplus", w_gplus), ("w_gcross", w_gcross), ("w_p", w_p)):
            cov = treecorr.estimate_multi_cov(
                [ng, nn], "jackknife", func=func, cross_patch_weight="match"
            )
            result[f"err_{name}"] = np.sqrt(np.diag(cov))
            result[f"cov_{name}"] = cov
        result["npatch"] = npatch

    return result
