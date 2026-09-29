"""Residues by numerical differentiation -- the third ingredient of the right-hand side.

For a pole of order m at p, the function

    h(z) = (z - p)^m f(z)

is analytic at p, and the residue is one of its Taylor coefficients:

    Res_p f = h^(m-1)(p) / (m-1)!

So computing a residue is computing a derivative, which is this unit's subject,
and the interesting part is what we are *not* allowed to use.

The best way to differentiate an analytic function numerically is Cauchy's
integral formula: sample f on a small circle about p and take a discrete
Fourier transform. It is spectacularly accurate -- the periodic trapezoid rule
of rung 1 again, converging geometrically -- and it is forbidden here. That
formula is a contour integral, so a right-hand side built on it would share its
machinery with the left, and the agreement this project is testing would be a
tautology. Sampling on a circle is exactly the thing to avoid.

What is left is honest finite differencing, and it is genuinely harder.
Everything below samples h along a *line* through p:

    h(p + t*delta) = sum_j a_j t^j,     Res = a_(m-1) / delta^(m-1)

The Taylor coefficients come from interpolating h through 2n offset nodes
t = -n..-1, 1..n and reading off a coefficient -- which is precisely a central
finite-difference stencil, written in the form that makes the step size
visible. The node t = 0 is skipped because f is infinite there.

That leaves the classic tradeoff, and it is unavoidable:

    total error  ~  A delta^(2n-k) + B eps / delta^k,        k = m - 1

truncation falling with delta, roundoff rising as the division by delta^k
amplifies it. Balancing the two gives

    delta* ~ eps^(1/2n)          the best step depends only on the stencil width
    error* ~ eps^(1 - k/2n)      and the best attainable error degrades with the
                                 order of the pole

For a simple pole k = 0, nothing is amplified, and residues come out at machine
precision. For a fifth-order pole with a six-node stencil the floor is around
1e-8 -- not a defect of this code but of differentiating four times in double
precision.
"""

import numpy as np

EPS = float(np.finfo(float).eps)


def stencil_offsets(half):
    """Integer node offsets -half..-1, 1..half. The centre is skipped: f is infinite there."""
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)]).astype(float)


def best_delta(scale=1.0, half=5):
    """The step that balances truncation against roundoff: scale * eps^(1/2n).

    Derived rather than tuned. With 2n nodes the interpolation error in the
    k-th derivative falls like delta^(2n-k) while roundoff rises like
    eps/delta^k; the two cross at eps^(1/2n), independent of k. `scale` carries
    the units -- it should be comfortably inside the distance to the nearest
    other singularity, or the stencil will straddle one.
    """
    return scale * EPS ** (1.0 / (2 * half))


def stencil_residue(f, p, order, half, delta):
    """One residue estimate at a single step size. The building block; usually not called directly.

    The Vandermonde solve *is* the finite-difference stencil: interpolating h
    through the offset nodes and taking a Taylor coefficient is the same linear
    combination of samples a difference formula would apply, written so the
    step size stays explicit.
    """
    if order < 1:
        raise ValueError(f"pole order must be at least 1, got {order}")
    if 2 * half < order:
        raise ValueError(f"a stencil of {2 * half} nodes cannot reach Taylor coefficient "
                         f"{order - 1}; increase half above {order / 2:g}")
    offsets = stencil_offsets(half)
    # Complex on purpose. A real p would give a real-valued stencil, and a
    # function like 1/sqrt(z) then returns nan on the negative offsets for a
    # reason that has nothing to do with the singularity being studied.
    z = np.asarray(p + offsets * delta, dtype=complex)
    with np.errstate(all="ignore"):
        h = (z - p) ** order * f(z)
    if not np.all(np.isfinite(h)):
        raise ValueError(f"(z-p)^{order} f(z) is not finite at every stencil node about {p}; "
                         f"delta = {delta:g} may be reaching another singularity")
    vandermonde = offsets[:, None] ** np.arange(len(offsets))[None, :]
    return complex(np.linalg.solve(vandermonde, h)[order - 1] / delta ** (order - 1))


def leading_magnitudes(f, p, order, scale=0.5, half=5, shrink=0.1, steps=5):
    """max |(z-p)^order f(z)| over the stencil, at a sequence of shrinking steps.

    The diagnostic behind `assert_is_pole`. For a pole of order `order` this
    settles on |a_{-order}|; for anything else it runs away.
    """
    magnitudes, delta = [], scale
    for _ in range(steps):
        z = np.asarray(p + stencil_offsets(half) * delta, dtype=complex)
        with np.errstate(all="ignore"):
            h = (z - p) ** order * f(z)
        magnitudes.append(float(np.max(np.abs(h))))
        delta *= shrink
    return magnitudes


def assert_is_pole(f, p, order, scale=0.5, half=5, tolerance=100.0):
    """Refuse unless (z-p)^order f(z) stays bounded as the step shrinks.

    That boundedness is not a heuristic -- it is the definition. p is a pole of
    order at most `order` exactly when (z-p)^order f(z) extends analytically to
    p, and then the stencil values settle rather than grow. Four decades of
    shrinking step separate the cases by an enormous margin: a genuine pole
    never grows at all, understating the order by one grows by 1e4, and an
    essential singularity overflows outright.

    Without this, `residue` answers any question it is asked. Before it existed,
    the residue of exp(1/z) at the origin came back as 5.895204090168 -- the
    same value whether order 1 or order 2 was claimed, which is the tell.

    Overstating the order is deliberately allowed. If f has a pole of order 3
    and order 5 is claimed, (z-p)^5 f is still analytic, its Taylor coefficient
    a_4 is still the residue, and the stencil values shrink rather than grow.
    The answer is right, merely worse conditioned.
    """
    magnitudes = leading_magnitudes(f, p, order, scale=scale, half=half)
    start, end = magnitudes[0], magnitudes[-1]
    if not np.isfinite(end) or end > tolerance * max(start, 1e-300):
        raise ValueError(
            f"(z-p)^{order} f(z) is unbounded as the step shrinks about {p} "
            f"(max |h| went from {start:.2e} to {end:.2e}); p is not a pole of "
            f"order {order}. Either the order is understated, or this is not a "
            f"pole at all -- an essential singularity has no residue reachable "
            f"by differentiation.")


def residue_detail(f, p, order=1, scale=1.0, half=5, ladder=26, shrink=0.6):
    """Residue, the step size it came from, and an estimate of its own error.

    `best_delta` says where the optimum sits in terms of eps and the stencil
    width, but the constant in front of it involves the high derivatives of h,
    which are exactly what is not known in advance. So rather than trusting the
    formula, walk a geometric ladder of step sizes down from `scale` and take
    the estimate that agrees best with *both* its neighbours.

    Agreement with both neighbours rather than just the previous one is
    deliberate: on the roundoff side of the tradeoff two adjacent estimates can
    coincide by luck, and a one-sided test would stop there. The returned
    spread is that agreement, and it is a real error bar -- it tracks the true
    error closely enough to be worth reporting.
    """
    assert_is_pole(f, p, order, scale=scale, half=half)
    steps, values = [], []
    delta = scale
    for _ in range(ladder):
        try:
            values.append(stencil_residue(f, p, order, half, delta))
            steps.append(delta)
        except ValueError:
            pass                      # this step straddles another singularity; try a smaller one
        delta *= shrink
    if len(values) < 3:
        raise ValueError(f"only {len(values)} usable step sizes about {p}; "
                         f"no plateau can be identified")
    values = np.asarray(values)
    spreads = np.maximum(np.abs(np.diff(values)[:-1]), np.abs(np.diff(values)[1:]))
    best = int(np.argmin(spreads)) + 1
    return complex(values[best]), steps[best], float(spreads[best - 1])


def residue(f, p, order=1, scale=1.0, half=5, delta=None):
    """Res_p f, by finite differences of h(z) = (z - p)^order f(z) along a line.

    With `delta` given, this is one raw stencil at that step and **no check is
    made that p is a pole at all** -- there is only one step, so nothing can be
    said about boundedness. That path is for the step-size studies. Otherwise
    the adaptive ladder of `residue_detail` picks the step, which is the
    sensible default both because the best step depends on derivatives of h
    that nobody knows beforehand, and because the ladder can verify that the
    question makes sense before answering it.
    """
    if delta is not None:
        return stencil_residue(f, p, order, half, delta)
    return residue_detail(f, p, order, scale=scale, half=half)[0]


def residues_of(f, found, half=5):
    """Residues at each Pole from rung 3, with the step scaled to each pole's clearance.

    `found` is the list returned by poles.find_poles. Each pole gets a step
    sized to its own distance from the nearest neighbour, so a tightly spaced
    pair does not have its stencil reach across the gap.
    """
    locations = [q.location for q in found]
    out = []
    for q in found:
        others = [abs(q.location - other) for other in locations if other is not q.location]
        scale = min(others) / 4 if others else 1.0
        out.append(residue(f, q.location, q.order, scale=scale, half=half))
    return out
