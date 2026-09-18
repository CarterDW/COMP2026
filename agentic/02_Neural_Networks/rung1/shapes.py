"""Galaxy shape measurement from particle clouds.

Units are whatever you pass in, consistently. Positions from TNG are ckpc/h and
masses are 1e10 Msun/h; axis ratios are dimensionless so units only matter for
the aperture cut.

Conventions fixed here and relied on downstream:

  - Eigenvalues sorted DESCENDING, so axes are a >= b >= c and ``vectors[:, 0]``
    is the major axis.
  - q = b/a, s = c/a.  In projection, q is minor/major of the 2D tensor.
  - Ellipticity magnitude has two standard conventions:
        "chi"     : (1 - q^2) / (1 + q^2)    [distortion, Joachimi+2011 chi]
        "epsilon" : (1 - q)   / (1 + q)      [third flattening]
    They agree to O(1-q) but differ by ~2x for near-round galaxies. This is a
    notorious source of factor-~2 disagreements between IA papers, so the
    convention is always an explicit argument -- there is no default that gets
    silently inherited.

The reduced (1/r^2-weighted) tensor is the usual choice for IA work because it
downweights the outskirts, where the particle noise lives. It is also the more
fragile estimator at low particle count, which is exactly what rung 1 measures.
"""

import numpy as np


def inertia_tensor(pos, mass, reduced):
    """Mass-weighted inertia tensor of a cloud already centred on the origin.

    Works in any dimension: pass (N, 3) for the 3D shape, (N, 2) for a
    projected shape.
    """
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if pos.ndim != 2:
        raise ValueError(f"pos must be (N, ndim); got shape {pos.shape}")
    if mass.shape != pos.shape[:1]:
        raise ValueError(f"mass shape {mass.shape} does not match pos {pos.shape}")

    weight = mass
    if reduced:
        r2 = np.einsum("ij,ij->i", pos, pos)
        if np.any(r2 == 0.0):
            raise ValueError(
                "reduced tensor is undefined for a particle at r=0. Drop it "
                "explicitly (measure_galaxy does, and reports how many) rather "
                "than letting a 1/0 through."
            )
        weight = mass / r2
    return np.einsum("i,ij,ik->jk", weight, pos, pos) / weight.sum()


def principal_axes(tensor):
    """(axes, vectors) sorted descending; vectors[:, k] belongs to axes[k].

    For the simple tensor the eigenvalues are variances along the principal
    directions, so sqrt() gives axis lengths. The same sqrt is conventional for
    the reduced tensor.
    """
    eigenvalues, vectors = np.linalg.eigh(tensor)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f"non-positive eigenvalue {eigenvalues.min():.3e}; the cloud is "
            "degenerate (collinear particles, or too few of them)"
        )
    order = np.argsort(eigenvalues)[::-1]
    return np.sqrt(eigenvalues[order]), vectors[:, order]


def ellipticity_magnitude(q, convention):
    """Scalar ellipticity from an axis ratio q = minor/major. See module docstring."""
    if convention == "chi":
        return (1.0 - q**2) / (1.0 + q**2)
    if convention == "epsilon":
        return (1.0 - q) / (1.0 + q)
    raise ValueError(f"unknown convention {convention!r}; use 'chi' or 'epsilon'")


def axis_ratios_3d(pos, mass, reduced):
    """(q, s, major_axis_unit_vector) for a 3D cloud centred on the origin."""
    axes, vectors = principal_axes(inertia_tensor(pos, mass, reduced))
    return axes[1] / axes[0], axes[2] / axes[0], vectors[:, 0]


def projected_ellipticity(pos2d, mass, reduced, convention):
    """(e1, e2, q) for a cloud projected into a plane and centred on the origin.

    The position angle phi is measured from the +x axis of the projection plane
    and enters as the spin-2 pair (cos 2phi, sin 2phi). With this sign, a galaxy
    whose major axis lies ALONG the separation vector to a neighbour has
    e_plus > 0 -- i.e. positive means radial alignment. Observational papers
    frequently quote the opposite sign because they work in the shear
    convention; flip once, at the end, when comparing to them. The sign is
    pinned by ``checks.check_radial_injection``.
    """
    axes, vectors = principal_axes(inertia_tensor(pos2d, mass, reduced))
    q = axes[1] / axes[0]
    magnitude = ellipticity_magnitude(q, convention)
    phi = np.arctan2(vectors[1, 0], vectors[0, 0])
    return magnitude * np.cos(2.0 * phi), magnitude * np.sin(2.0 * phi), q


def offsets_periodic(pos, centre, boxsize):
    """Particle offsets from a centre under the minimum-image convention."""
    delta = np.asarray(pos, dtype=float) - np.asarray(centre, dtype=float)
    return delta - boxsize * np.round(delta / boxsize)


def measure_galaxy(pos, mass, centre, boxsize, aperture, reduced, convention, los_axis=2):
    """Shape of one galaxy from its particles.

    ``aperture`` is a radius in the same length units as ``pos``; only particles
    inside it are used. Particles sitting exactly at the centre are dropped
    (the reduced tensor cannot use them) and counted in ``n_dropped`` so the
    drop is visible rather than silent.

    Returns a dict. ``n_used`` is the quantity the convergence test scans over.
    """
    offsets = offsets_periodic(pos, centre, boxsize)
    inside = np.einsum("ij,ij->i", offsets, offsets) <= aperture**2
    offsets, mass_in = offsets[inside], np.asarray(mass, dtype=float)[inside]

    at_centre = np.einsum("ij,ij->i", offsets, offsets) == 0.0
    n_dropped = int(at_centre.sum())
    offsets, mass_in = offsets[~at_centre], mass_in[~at_centre]

    plane = [axis for axis in range(offsets.shape[1]) if axis != los_axis]
    q3, s3, major = axis_ratios_3d(offsets, mass_in, reduced)
    e1, e2, q2 = projected_ellipticity(offsets[:, plane], mass_in, reduced, convention)
    return {
        "n_used": len(mass_in),
        "n_dropped": n_dropped,
        "q_3d": q3,
        "s_3d": s3,
        "major_3d": major,
        "q_2d": q2,
        "e1": e1,
        "e2": e2,
    }
