"""Self-contained consistency checks for the shape and IA estimators.

Run this before trusting any number that comes out of the rung-1 drivers:

    python checks.py

Every check has a value that is known independently of the code being tested --
an analytic covariance, an exactly injected alignment, a parity argument, or a
Poisson expectation. None of them compares the code to itself. Exit status is
non-zero if any check fails.

The checks needing no simulation data at all are the whole point: the estimators
can be fully validated before a single byte of TNG is downloaded.
"""

import sys

import numpy as np
import treecorr

import ia
import shapes

RNG = np.random.default_rng(20260918)


def _rotation(axis, angle):
    """Rotation matrix about a unit axis, by Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def check_simple_tensor_recovers_covariance():
    """The simple inertia tensor of a Gaussian cloud IS its covariance.

    Non-trivial because it tests eigenvalue ordering, the sqrt convention, the
    mass weighting and the rotation handling simultaneously: a cloud with known
    axes (3, 2, 1), rotated by a known matrix, must return q=2/3, s=1/3 and a
    major axis along the rotated x direction.
    """
    n = 400_000
    true_axes = np.array([3.0, 2.0, 1.0])
    rot = _rotation([0.3, -0.7, 0.5], 0.8)
    cloud = (RNG.normal(size=(n, 3)) * true_axes) @ rot.T
    mass = np.full(n, 0.7)

    q, s, major = shapes.axis_ratios_3d(cloud, mass, reduced=False)
    q_true, s_true = true_axes[1] / true_axes[0], true_axes[2] / true_axes[0]
    aligned = abs(float(major @ rot[:, 0]))
    ok = abs(q - q_true) < 0.01 and abs(s - s_true) < 0.01 and aligned > 0.999
    return ok, (f"q={q:.4f} (exp {q_true:.4f}), s={s:.4f} (exp {s_true:.4f}), "
                f"|major.major_true|={aligned:.5f}")


def check_sphere_is_round():
    """An isotropic cloud must give q = s = 1 for BOTH tensor definitions.

    This is the check that catches a 1/r^2 weight applied along the wrong axis:
    such a bug leaves the simple tensor correct and only shows up here.
    """
    n = 200_000
    direction = RNG.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    cloud = direction * RNG.uniform(0.2, 1.0, (n, 1)) ** (1 / 3)
    mass = np.ones(n)

    out = {}
    for reduced in (False, True):
        q, s, _ = shapes.axis_ratios_3d(cloud, mass, reduced=reduced)
        out[reduced] = (q, s)
    ok = all(abs(q - 1) < 0.02 and abs(s - 1) < 0.02 for q, s in out.values())
    return ok, ("simple q,s={:.4f},{:.4f}  reduced q,s={:.4f},{:.4f}"
                .format(*out[False], *out[True]))


def check_projection_convention():
    """Pins the spin-2 angle convention of projected_ellipticity.

    A cloud elongated along +x must give e1 > 0, e2 = 0. Rotating it by 30 deg
    in the projection plane must give (e1, e2) = eps * (cos 60, sin 60) -- i.e.
    the ellipticity rotates at twice the angle. A factor-of-two or a swapped
    sin/cos error fails here and nowhere else.
    """
    n = 300_000
    base = RNG.normal(size=(n, 2)) * np.array([2.5, 1.0])
    mass = np.ones(n)

    e1_0, e2_0, q0 = shapes.projected_ellipticity(base, mass, False, "chi")
    angle = np.deg2rad(30.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    e1_r, e2_r, q_r = shapes.projected_ellipticity(base @ rot.T, mass, False, "chi")

    eps = shapes.ellipticity_magnitude(q0, "chi")
    ok = (
        e1_0 > 0 and abs(e2_0) < 0.005
        and abs(e1_r - eps * np.cos(2 * angle)) < 0.01
        and abs(e2_r - eps * np.sin(2 * angle)) < 0.01
        and abs(q_r - q0) < 0.01
    )
    return ok, (f"unrotated (e1,e2)=({e1_0:+.4f},{e2_0:+.4f}) q={q0:.4f}; "
                f"rotated 30deg ({e1_r:+.4f},{e2_r:+.4f}) "
                f"exp ({eps * np.cos(2 * angle):+.4f},{eps * np.sin(2 * angle):+.4f})")


def check_ellipticity_conventions_differ():
    """The two ellipticity conventions must NOT be silently interchangeable.

    Guards the factor-~2 trap: for a round-ish galaxy chi is about twice
    epsilon. If someone 'simplifies' the code by collapsing the two, this fails.
    """
    q = 0.8
    chi = shapes.ellipticity_magnitude(q, "chi")
    eps = shapes.ellipticity_magnitude(q, "epsilon")
    ok = abs(chi / eps - 2.0) < 0.15 and chi > eps
    return ok, f"q={q}: chi={chi:.4f}, epsilon={eps:.4f}, ratio={chi / eps:.3f}"


def check_rpar_is_radial():
    """TreeCorr's rpar is the observer-radial difference, NOT the box-axis dz.

    This is the measurement that dictates ia.py's full-depth projection design.
    Two points separated by exactly 5 in z, sitting at x=1000, differ in radial
    distance by only 0.0125 -- so they PASS a |rpar| < 1 cut that was intended
    to mean |dz| < 1. If a future TreeCorr changes this, the design note in
    ia.py needs revisiting, and this check is what will say so.
    """
    n = treecorr.Catalog(x=[1000.0], y=[0.0], z=[0.0], allow_xyz=True)
    g = treecorr.Catalog(x=[1000.0], y=[0.0], z=[5.0], g1=[1.0], g2=[0.0],
                         allow_xyz=True)
    ng = treecorr.NGCorrelation(min_sep=1.0, max_sep=10.0, nbins=9,
                                metric="Euclidean", min_rpar=-1.0, max_rpar=1.0)
    ng.process(n, g)
    passed_cut = ng.npairs.sum() == 1
    return passed_cut, (
        f"pair with dz=5 at x=1000 {'passes' if passed_cut else 'fails'} a "
        "|rpar|<1 cut -> rpar is radial, so ia.py must not use it"
    )


def check_analytic_rr():
    """The analytic periodic RR must reproduce Poisson pair counts.

    Tested through w_p, which is identically zero for an unclustered catalogue:
    if the RR normalisation is wrong by any factor, w_p picks up a constant
    offset of the same size. This is the check that a wrong estimator
    normalisation cannot survive.
    """
    box, n = 300.0, 30_000
    pos = RNG.uniform(0.0, box, (n, 3))
    out = ia.correlate(pos, pos, np.zeros(n), np.zeros(n), box,
                       min_rp=2.0, max_rp=40.0, nbins=6, npatch=16)
    ratio = out["npairs"] / out["rr_analytic"]
    z = out["w_p"] / out["err_w_p"]
    ok = np.all(np.abs(ratio - 1.0) < 0.02) and np.all(np.abs(z) < 3.0)
    return ok, ("npairs/RR in [{:.4f},{:.4f}]; w_p/err max |z|={:.2f}"
                .format(ratio.min(), ratio.max(), np.abs(z).max()))


def check_ia_null():
    """Random orientations must give w_g+ = 0, and w_gx = 0 by parity.

    The null test. A geometry or angle bug generically produces a non-zero
    signal here, because there is no real alignment to hide behind.
    """
    box, n = 300.0, 30_000
    pos = RNG.uniform(0.0, box, (n, 3))
    phi = RNG.uniform(0.0, np.pi, n)
    amp = 0.3
    out = ia.correlate(pos, pos, amp * np.cos(2 * phi), amp * np.sin(2 * phi),
                       box, min_rp=2.0, max_rp=40.0, nbins=6, npatch=16)
    z_plus = out["w_gplus"] / out["err_w_gplus"]
    z_cross = out["w_gcross"] / out["err_w_gcross"]
    ok = np.abs(z_plus).max() < 3.0 and np.abs(z_cross).max() < 3.0
    return ok, (f"max |w_g+/err| = {np.abs(z_plus).max():.2f}, "
                f"max |w_gx/err| = {np.abs(z_cross).max():.2f}")


def check_radial_injection():
    """An exactly injected radial alignment must come back with the right sign
    and the right amplitude.

    Satellites are given projected major axes pointing exactly at their own
    cluster centre, with a fixed axis ratio, so for every own-cluster
    centre-satellite pair e_+ = eps ANALYTICALLY, with zero scatter.

    This is the check that pins the TreeCorr sign flip in ia.py: without the
    flip, mean e_+ returns -eps.

    The one subtlety is that ia.py collapses the full box depth, so a satellite
    can also pair with a chance-projected *foreign* cluster centre, which
    carries a random orientation and dilutes the mean. That dilution is
    predictable -- foreign pairs per annulus are N_sat * (N_c / L^2) * pi * dA
    -- and the geometry here is sized so it stays below ~0.3% in every bin,
    which is why the tolerance is 1% rather than machine precision. Getting
    this wrong the first time is what the dilution term is here to document.
    """
    box = 1200.0
    n_clusters, n_sat, radius = 400, 60, 1.5
    q = 0.6
    eps = shapes.ellipticity_magnitude(q, "chi")
    min_rp, max_rp = 0.2, 1.5

    centres = RNG.uniform(0.0, box, (n_clusters, 3))
    offsets = RNG.normal(size=(n_clusters, n_sat, 3))
    offsets /= np.linalg.norm(offsets, axis=2)[..., None]
    offsets *= radius * RNG.uniform(0.0, 1.0, (n_clusters, n_sat, 1)) ** (1 / 3)
    sats = np.mod((centres[:, None, :] + offsets).reshape(-1, 3), box)

    flat = offsets.reshape(-1, 3)
    phi = np.arctan2(flat[:, 1], flat[:, 0])  # projected direction to own centre
    e1, e2 = eps * np.cos(2 * phi), eps * np.sin(2 * phi)

    out = ia.correlate(centres, sats, e1, e2, box,
                       min_rp=min_rp, max_rp=max_rp, nbins=4, npatch=32)

    # Predicted foreign-pair fraction per bin, from the surface density of
    # centres: own-cluster pairs all sit at rp <= radius, foreign ones do not.
    area = np.pi * (max_rp**2 - min_rp**2)
    foreign = len(sats) * (n_clusters / box**2) * area
    dilution = foreign / out["npairs"].sum()

    recovered = out["mean_eplus"]
    ok = (
        np.all(np.abs(recovered - eps) < 0.01 * eps)
        and np.all(out["w_gplus"] > 0.0)
        and np.abs(out["w_gcross"] / out["err_w_gcross"]).max() < 3.0
    )
    return ok, ("injected eps={:.4f}; recovered mean e_+ in [{:.4f},{:.4f}] "
                "(predicted dilution {:.2%}); w_g+>0 everywhere={}; "
                "max |w_gx/err|={:.2f}"
                .format(eps, recovered.min(), recovered.max(), dilution,
                        bool(np.all(out["w_gplus"] > 0)),
                        np.abs(out["w_gcross"] / out["err_w_gcross"]).max()))


def check_shape_is_order_invariant():
    """Shuffling the particle list must not change the measured shape.

    Cheap, but it is the check that catches an indexing slip in the aperture
    cut or the r=0 removal -- exactly the code the downsampling driver exercises
    thousands of times.
    """
    n = 5_000
    box = 1000.0
    centre = np.array([500.0, 500.0, 500.0])
    pos = centre + RNG.normal(size=(n, 3)) * np.array([3.0, 2.0, 1.5])
    mass = RNG.uniform(0.5, 1.5, n)

    a = shapes.measure_galaxy(pos, mass, centre, box, 10.0, True, "chi")
    order = RNG.permutation(n)
    b = shapes.measure_galaxy(pos[order], mass[order], centre, box, 10.0, True, "chi")
    keys = ("n_used", "q_3d", "s_3d", "q_2d", "e1", "e2")
    worst = max(abs(a[k] - b[k]) for k in keys)
    return worst < 1e-10, f"max |difference| over {keys} = {worst:.2e}"


def check_periodic_wrap_in_shapes():
    """A galaxy straddling the box edge must measure the same shape as one that
    does not.

    The minimum-image offset is easy to get wrong and the failure is silent: a
    wrapped galaxy simply looks enormous and round. Here the identical cloud is
    placed mid-box and on the boundary; the shapes must agree exactly.
    """
    n = 20_000
    box = 100.0
    cloud = RNG.normal(size=(n, 3)) * np.array([3.0, 1.5, 1.0])
    mass = np.ones(n)

    middle = shapes.measure_galaxy(cloud + 50.0, mass, np.full(3, 50.0), box,
                                   12.0, True, "chi")
    edge = shapes.measure_galaxy(np.mod(cloud, box), mass, np.zeros(3), box,
                                 12.0, True, "chi")
    keys = ("n_used", "q_3d", "s_3d", "q_2d")
    worst = max(abs(middle[k] - edge[k]) for k in keys)
    return worst < 1e-10, (f"mid-box q_3d={middle['q_3d']:.5f} vs "
                           f"edge q_3d={edge['q_3d']:.5f}; max diff {worst:.2e}")


CHECKS = (
    ("simple tensor == covariance", check_simple_tensor_recovers_covariance),
    ("sphere is round (both tensors)", check_sphere_is_round),
    ("spin-2 projection convention", check_projection_convention),
    ("chi vs epsilon not interchangeable", check_ellipticity_conventions_differ),
    ("treecorr rpar is radial, not dz", check_rpar_is_radial),
    ("analytic periodic RR", check_analytic_rr),
    ("IA null test (random orientations)", check_ia_null),
    ("IA radial injection (sign + amplitude)", check_radial_injection),
    ("shape invariant to particle order", check_shape_is_order_invariant),
    ("shape invariant to box wrap", check_periodic_wrap_in_shapes),
)


def main():
    failures = 0
    print(f"\nrung-1 estimator checks ({len(CHECKS)} total)\n")
    for name, func in CHECKS:
        ok, detail = func()
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    print()
    if failures:
        print(f"  -> {failures} CHECK(S) FAILED\n")
    else:
        print("  -> all checks pass\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
