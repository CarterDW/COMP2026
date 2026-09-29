"""Locating the poles of a black-box meromorphic function, and measuring their orders.

This is the second ingredient of the residue theorem's right-hand side: the set
{p_k}. `f` is treated as a callable and nothing else -- no symbolic form, no
derivative, no list of factors. The only assumptions are that `f` accepts numpy
arrays and is meromorphic on the search region.

    poles of f  <->  zeros of g = 1/f

so the search is a zero-finding problem, and every stage below is phrased in
terms of g rather than f. That is not a stylistic choice: |f| runs to overflow
near a pole and divides by zero on top of one, which for complex arithmetic
numpy reports as nan. Its reciprocal goes quietly to zero instead and stays
bounded over the whole region, so no stage ever has to reason about a
non-finite number it did not ask for.

Three stages.

  * **Scan.** Evaluate |1/f| on a grid and keep the local minima. Crude, but it
    only has to get close enough for the next stage to take over, and it cannot
    miss a pole the grid resolves.

  * **Refine.** Muller's method on g. Chosen over Newton because it needs no
    derivative -- fitting a parabola through three points and taking the nearer
    root is enough, and it handles complex arithmetic natively. Keeping this
    stage derivative-free matters: rung 4 computes derivatives numerically, and
    a location that already depended on them would blur what that rung measures.

  * **Order.** Near a pole of order m, |1/f| ~ r^m, so averaging log|1/f| around
    circles of radius r and fitting against log r gives a slope of m. Averaging
    over angle cancels the numerator's angular variation. The fit spans several
    decades of r and its residual is reported, because a straight line is
    evidence the singularity really is a pole of integer order.

No quadrature anywhere, and no winding numbers. Nothing here knows about the
left-hand side.
"""

from collections import namedtuple

import numpy as np
from scipy import ndimage

Pole = namedtuple("Pole", "location order slope residual")
Pole.__doc__ = """A located pole.

location  -- the refined position in the complex plane
order     -- the integer order, rounded from `slope`
slope     -- the raw fitted exponent, before rounding; how far it sits from an
             integer is the honest measure of confidence
residual  -- RMS residual of the log-log fit, in natural-log units
"""


Singularity = namedtuple("Singularity", "location slope residual reason")
Singularity.__doc__ = """A singularity that was found but is not a pole.

Reported rather than discarded. A contour enclosing one of these has a genuine
residue -- exp(1/z) has residue 1 at the origin, and the rung-1 quadrature
returns 2 pi i for it to machine precision -- but no method here can compute it
without a contour integral, which is the one thing the right-hand side may not
use. Silently dropping it would make a later LHS/RHS comparison disagree by a
whole residue and look like a failure of the theorem.

reason -- which test it failed, in words
"""

Survey = namedtuple("Survey", "poles non_poles")
Survey.__doc__ = """Everything singular that `survey` found, split by what it is.

poles      -- Pole records, safe to take residues of
non_poles  -- Singularity records, which must not be ignored
"""


def reciprocal(f):
    """g = 1/f, evaluated so that f's blow-up becomes the correct limit g = 0.

    Everything downstream works with g rather than f, and this is the reason.
    Near a pole |f| runs off toward overflow, and *at* one it divides by zero --
    which for complex arithmetic numpy reports as nan, not inf, so a bare
    `1/f(z)` returns nan at exactly the point the search is trying to reach.
    A non-finite f means f blew up, which is what a pole is; g = 0 there is the
    limit, not a fallback. Meanwhile g itself stays bounded and well behaved
    across the whole region, which is why the scan, the root finder and the
    order fit are all phrased in terms of it.

    Works on scalars and on arrays, so the grid scan and Muller's method share
    one definition of what 1/f means.
    """
    def g(z):
        with np.errstate(all="ignore"):
            value = np.asarray(f(z), dtype=complex)
        blown = ~np.isfinite(value)
        with np.errstate(all="ignore"):
            out = np.where(blown, 0.0, 1.0 / np.where(blown, 1.0, value))
        return out if out.ndim else complex(out)
    return g


def muller(g, x0, x1, x2, tol=1e-14, maxiter=200):
    """Muller's method: fit a parabola through three points, step to its nearer root.

    Derivative-free, complex-native, and superlinear (order about 1.84) at a
    simple zero. The sign of the square root is chosen to make the denominator
    larger, which picks the root closer to x2 and avoids the cancellation the
    textbook quadratic formula would suffer here.
    """
    for _ in range(maxiter):
        g2 = g(x2)
        if g2 == 0:
            return x2
        g0, g1 = g(x0), g(x1)
        h0, h1 = x1 - x0, x2 - x1
        if h0 == 0 or h1 == 0 or h0 + h1 == 0:
            return x2
        d0, d1 = (g1 - g0) / h0, (g2 - g1) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        discriminant = np.sqrt(b * b - 4 * a * g2)
        denominator = (b + discriminant if abs(b + discriminant) >= abs(b - discriminant)
                       else b - discriminant)
        if denominator == 0:
            return x2
        step = -2 * g2 / denominator
        x0, x1, x2 = x1, x2, x2 + step
        if abs(step) <= tol * max(1.0, abs(x2)):
            return x2
    return x2


def scan(f, region, resolution=241, prominence=10.0):
    """Grid points where |1/f| is a local minimum, as starting guesses.

    Phrased as minima of |1/f| rather than maxima of |f| on purpose. |f| spikes
    to overflow at a pole and to nan on top of one, so a grid of |f| is full of
    non-finite entries exactly where the information is; |1/f| dips smoothly to
    zero there and stays bounded everywhere. The threshold is relative to the
    median of |1/f| over the region, so it adapts to the function's scale.
    """
    x0, x1, y0, y1 = region
    re = np.linspace(x0, x1, resolution)
    im = np.linspace(y0, y1, resolution)
    magnitude = np.abs(reciprocal(f)(re[None, :] + 1j * im[:, None]))
    finite = magnitude[np.isfinite(magnitude)]
    dips = (magnitude == ndimage.minimum_filter(magnitude, size=5))
    dips &= magnitude < np.median(finite) / prominence
    rows, cols = np.nonzero(dips)
    return re[cols] + 1j * im[rows]


def measure_order(f, p, scale=1.0, decades=2.0, samples=24, angles=16):
    """Fit log|1/f| against log r on circles about p; the slope is the order.

    Near a pole of order m, |1/f| ~ r^m / |c|, so averaging log|1/f| over
    `angles` evenly spaced directions kills the numerator's angular dependence
    and leaves mean log|1/f| = m log r - log|c|. Fitting in terms of 1/f rather
    than f keeps every sampled value small and bounded: log|f| would be the
    logarithm of a number growing like r^(-m), which for a high-order pole close
    in is a very large number indeed.

    Returns (slope, residual). A slope that is not near a whole number, or a
    residual that is not near zero, means the singularity is not a pole.
    """
    outer = 0.1 * scale
    radii = np.logspace(np.log10(outer) - decades, np.log10(outer), samples)
    directions = np.exp(2j * np.pi * np.arange(angles) / angles)
    g = reciprocal(f)
    with np.errstate(all="ignore"):
        rings = np.array([np.mean(np.log(np.abs(g(p + r * directions)))) for r in radii])
    if not np.all(np.isfinite(rings)):
        raise ValueError(f"|1/f| is zero or non-finite somewhere on a sampling ring about {p}")
    slope, intercept = np.polyfit(np.log(radii), rings, 1)
    residual = float(np.sqrt(np.mean((rings - (slope * np.log(radii) + intercept)) ** 2)))
    return float(slope), residual


def survey(f, region, resolution=241, prominence=10.0, merge=1e-4,
           max_residual=0.05, max_defect=0.05):
    """Every singularity of `f` in `region`, split into poles and everything else.

    `region` is (x0, x1, y0, y1). Candidates that refine to the same point are
    merged. What survives must pass three tests to be called a pole, and each
    catches a different impostor:

      * **the exponent must be close to a whole number**, within `max_defect`.
        This is the workhorse. z^(-3/2) is caught by this and nothing else: its
        fit is straight to 1e-15 and its order rounds to a respectable 1, but
        the exponent is 1.5. 1/sqrt(z) (exponent 0.5) and log z (0.196) also
        fail here.

      * **the order must be at least 1**, catching an exponent that rounds to
        zero -- 1/sqrt(z) and log z again, by a second route.

      * **the log-log fit must be straight**, within `max_residual`. None of
        the impostors above trips this one; log z comes closest at 0.029, still
        inside the default. What it catches is two poles too close for the grid
        to separate: from far off the pair looks like one double pole, from
        close in like one simple pole, the fit bends, and the residual reaches
        0.4.

    Whatever fails goes into `non_poles` with the reason attached, instead of
    vanishing. An essential singularity fails earliest of all -- for exp(1/z)
    the reciprocal underflows to zero across a whole sector and Muller cannot
    converge -- and it is exactly the case that must not be dropped quietly.
    """
    x0, x1, y0, y1 = region
    span = max(x1 - x0, y1 - y0)
    step = span / (resolution - 1)
    g = reciprocal(f)

    refined, rejected = [], []
    for guess in scan(f, region, resolution, prominence):
        with np.errstate(all="ignore"):
            root = muller(g, guess, guess + step / 3, guess - step / 3)
        if not np.isfinite(root) or not (x0 <= root.real <= x1 and y0 <= root.imag <= y1):
            rejected.append(Singularity(complex(guess), float("nan"), float("nan"),
                                        "refinement did not converge to a finite point "
                                        "in the region"))
            continue
        if all(abs(root - seen) > merge * span for seen in refined):
            refined.append(root)

    found = []
    for p in refined:
        others = [abs(p - q) for q in refined if q is not p]
        clearance = min(others + [span / 4]) if others else span / 4
        try:
            slope, residual = measure_order(f, p, scale=clearance)
        except ValueError as failure:
            # Caught to classify, not to ignore: the reason is carried out in
            # the Singularity record rather than swallowed.
            rejected.append(Singularity(p, float("nan"), float("nan"), str(failure)))
            continue
        order = int(round(slope))
        if residual > max_residual:
            rejected.append(Singularity(p, slope, residual,
                                        f"log-log fit is not straight (residual {residual:.3f})"))
        elif abs(slope - order) > max_defect:
            rejected.append(Singularity(p, slope, residual,
                                        f"exponent {slope:.3f} is not a whole number"))
        elif order < 1:
            rejected.append(Singularity(p, slope, residual,
                                        f"exponent rounds to {order}, so f does not blow up here"))
        else:
            found.append(Pole(p, order, slope, residual))

    # A candidate that failed near a pole that was accepted is a duplicate of
    # that pole, not a separate finding.
    accepted = [q.location for q in found]
    distinct = [bad for bad in rejected
                if all(abs(bad.location - good) > merge * span for good in accepted)]

    key = lambda q: (round(q.location.real, 9), round(q.location.imag, 9))
    return Survey(sorted(found, key=key), sorted(distinct, key=key))


def find_poles(f, region, **kwargs):
    """Just the poles from `survey`. Convenient, but it discards the non-poles.

    Use `survey` wherever a missed singularity would matter -- assembling the
    right-hand side of the residue theorem, above all.
    """
    return survey(f, region, **kwargs).poles
