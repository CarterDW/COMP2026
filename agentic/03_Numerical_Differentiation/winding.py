"""Winding numbers: how many times a closed path wraps around a point.

This is the n(C, p) factor on the right-hand side of the residue theorem, and it
is pure geometry. Nothing here evaluates an integrand, calls a quadrature rule,
or knows what a residue is. That separation is the point -- if the right-hand
side borrowed machinery from the left, agreement between them would prove
nothing.

Two algorithms, sharing no code:

  * `winding_number` accumulates the turning of the vector from the point to the
    path, one sample step at a time. Trigonometric. It is exact provided no
    single step turns by as much as pi, a condition it verifies rather than
    assumes.

  * `winding_by_crossing` counts signed crossings of the ray running in the +x
    direction from the point. No trigonometry anywhere: sign tests and one
    linear interpolation per crossing edge.

Agreement between two methods with nothing in common is real evidence. Both
refuse to answer for a point lying on the path, where the winding number is
genuinely undefined, rather than returning a plausible wrong integer.

Deliberately absent: the identity

    n(C, p) = (1/2 pi i) oint_C dz/(z - p),

which would let the rung-1 quadrature compute a winding number directly. It is
excluded on purpose. The accumulated-argument algorithm below *is* that contour
integral, evaluated geometrically -- so comparing them tests two implementations
of one idea, not two independent ideas, and it is the simplest special case of
the very theorem this project is trying to stress test. It belongs in rung 5 as
a result, not here as a check.
"""

import numpy as np


def as_closed_polyline(path, n=2048):
    """Sample `path` into an array of points whose last entry repeats the first.

    Accepts a contour object (anything with a `sample` method) or a raw sequence
    of points. A raw sequence is closed if it is not closed already, so a list of
    polygon vertices can be passed directly.
    """
    if hasattr(path, "sample"):
        points = np.asarray(path.sample(n), dtype=complex)
    else:
        points = np.asarray(path, dtype=complex)
    if points[0] != points[-1]:
        points = np.append(points, points[0])
    return points


def assert_off_path(z):
    """Refuse if the origin lies on any segment of the polyline `z`.

    `z` is the path already shifted so the query point sits at the origin. The
    origin is on the segment from z0 to z1 exactly when the two are collinear
    with it (zero cross product) and it falls between them (non-positive dot
    product). Testing only `z == 0` is not enough: it catches a point that
    coincides with a sample, and misses a point lying in the middle of an edge.

    Shared by both algorithms. This is a degeneracy guard, not counting logic --
    the two remain independent in how they arrive at an answer, which is what
    makes their agreement meaningful.
    """
    z0, z1 = z[:-1], z[1:]
    cross = z0.real * z1.imag - z0.imag * z1.real
    dot = z0.real * z1.real + z0.imag * z1.imag
    collinear = np.abs(cross) <= 1e-12 * np.abs(z0) * np.abs(z1)
    if np.any(z == 0) or np.any(collinear & (dot <= 0.0)):
        raise ValueError("point lies on the path; its winding number is undefined")


def distance_to_polyline(z):
    """Distance from the origin to the polyline `z`, which is already shifted.

    Each segment is projected onto, the parameter clamped to the segment, and
    the nearest of the resulting points taken. Zero-length segments are dropped
    rather than divided by.
    """
    z0, z1 = z[:-1], z[1:]
    d = z1 - z0
    keep = np.abs(d) > 0.0
    z0, d = z0[keep], d[keep]
    t = np.clip(-(z0.real * d.real + z0.imag * d.imag) / np.abs(d) ** 2, 0.0, 1.0)
    return float(np.min(np.abs(z0 + t * d)))


def assert_resolved(algorithm, path, point, n, value):
    """Halve the sampling and insist that neither the answer nor the clearance moves.

    A resolution check rather than a proof -- two under-resolved answers could in
    principle agree -- but it is the only guard that catches aliasing. Bounding
    the computed step by pi cannot work, because np.angle has already folded
    every step into (-pi, pi]; and testing that the total is near an integer
    cannot work either, because an aliased total is often an exact integer. A
    circle wound 50 times and sampled 64 times accumulates precisely -13 turns.

    The clearance test is the other half. A point can lie exactly on a smooth
    curve and still miss the inscribed polyline -- a point on the unit circle
    sits 8e-4 from the chords at n = 2048 -- so both algorithms would answer
    "outside" with total confidence. Comparing the clearance at two resolutions
    catches it: if halving the sampling moves the path by as much as the point's
    own distance from it, that distance is not a real one.

    A raw vertex list is skipped: it cannot be resampled, and a polyline needs no
    resampling anyway, since a straight edge subtends less than pi from every
    point off it.
    """
    if not hasattr(path, "sample") or n < 16:
        return
    fine = as_closed_polyline(path, n) - point
    coarse_path = as_closed_polyline(path, n // 2) - point
    near, far = distance_to_polyline(fine), distance_to_polyline(coarse_path)
    if abs(near - far) > 0.5 * near:
        raise ValueError(
            f"point sits {near:.3e} from the sampled path, but halving the sampling "
            f"moves the path by {abs(near - far):.3e}; it is closer to the contour "
            f"than the sampling resolves, so its winding number is not determined")
    coarse = algorithm(path, point, n // 2, verify=False)
    if coarse != value:
        raise ValueError(f"under-sampled: n={n // 2} gives {coarse} but n={n} gives "
                         f"{value}; sample the path more finely")


def turning(path, point, n=2048):
    """Total turning of the vector from `point` to the path, in units of 2 pi.

    The quantity a winding number rounds to. Returned unrounded so callers can
    see how far from an integer the geometry actually came out.
    """
    z = as_closed_polyline(path, n) - point
    assert_off_path(z)
    # The principal argument of the ratio is the signed turn between consecutive
    # samples. It lands in (-pi, pi] by construction, which is the right branch
    # only when the true turn is smaller than pi -- see assert_resolved.
    return float(np.sum(np.angle(z[1:] / z[:-1])) / (2.0 * np.pi))


def winding_number(path, point, n=2048, tol=1e-6, verify=True):
    """Winding number of `path` about `point`, by accumulated argument."""
    total = turning(path, point, n)
    value = round(total)
    if abs(total - value) > tol:
        raise ValueError(f"accumulated turning {total:.9f} is not an integer within "
                         f"{tol:g}; the path may pass very close to the point")
    if verify:
        assert_resolved(winding_number, path, point, n, int(value))
    return int(value)


def winding_by_crossing(path, point, n=2048, verify=True):
    """Winding number by counting signed crossings of the ray to the right of `point`.

    Walk the path and watch the horizontal line through `point`. An edge crossing
    it upward to the right of the point contributes +1, downward contributes -1,
    and everything else contributes nothing. The total is the winding number.

    Shares nothing with `winding_number`: no angles, no logarithms, no branch to
    get wrong. The `y0 <= 0 < y1` convention counts each vertex on the line once
    rather than twice.
    """
    z = as_closed_polyline(path, n) - point
    assert_off_path(z)
    y0, y1 = z.imag[:-1], z.imag[1:]
    x0, x1 = z.real[:-1], z.real[1:]
    up = (y0 <= 0) & (y1 > 0)
    down = (y0 > 0) & (y1 <= 0)
    straddles = up | down
    # Where the segment meets the horizontal line. y1 != y0 on every straddling
    # segment, so this division is safe exactly where it is used.
    t = -y0[straddles] / (y1[straddles] - y0[straddles])
    crossing_x = x0[straddles] + t * (x1[straddles] - x0[straddles])
    to_the_right = crossing_x > 0
    value = int(np.sum(up[straddles] & to_the_right) - np.sum(down[straddles] & to_the_right))
    if verify:
        assert_resolved(winding_by_crossing, path, point, n, value)
    return value


def winding_grid(path, real, imag, n=2048):
    """Winding number at every point of a grid, as an integer array.

    Colouring the plane by winding number is the fastest way to see that a
    self-intersecting path really does have regions of different winding, so
    the visual diagnostics lean on this.

    Accumulated one segment at a time rather than by building the full
    (samples x rows x columns) array: at n = 2048 on a 400 x 400 grid that array
    would be five gigabytes, while this holds two grid-sized arrays at once.

    Points lying on the path are reported as the sentinel -99 rather than
    guessed at, matching the scalar routines' refusal to answer.
    """
    z = as_closed_polyline(path, n)
    grid = real[None, :] + 1j * imag[:, None]
    total = np.zeros(grid.shape)
    previous = z[0] - grid
    on_path = previous == 0
    for point in z[1:]:
        current = point - grid
        on_path |= current == 0
        total += np.angle(np.where(current == 0, 1.0, current)
                          / np.where(previous == 0, 1.0, previous))
        previous = current
    out = np.rint(total / (2.0 * np.pi)).astype(int)
    out[on_path] = -99
    return out


def crossing_grid(path, real, imag, n=2048):
    """Winding number over a grid by signed ray crossings, the counterpart to winding_grid.

    Exists so the visual diagnostics can show both algorithms' verdicts over the
    whole plane rather than at a handful of sampled points. Vectorised over the
    grid and looped over segments, so memory stays at one grid, and a horizontal
    segment is skipped outright because it can never be crossed.
    """
    z = as_closed_polyline(path, n)
    X, Y = real[None, :], imag[:, None]
    total = np.zeros((len(imag), len(real)), dtype=int)
    for a, b in zip(z[:-1], z[1:]):
        rise = b.imag - a.imag
        if rise == 0.0:
            continue
        y0, y1 = a.imag - Y, b.imag - Y
        up, down = (y0 <= 0) & (y1 > 0), (y0 > 0) & (y1 <= 0)
        crossing_x = (a.real - X) + (-y0 / rise) * (b.real - a.real)
        right = crossing_x > 0
        total += (up & right).astype(int) - (down & right).astype(int)
    return total
