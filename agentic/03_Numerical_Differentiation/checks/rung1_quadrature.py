"""Rung 1: the quadrature, i.e. the left-hand side of the residue theorem.

    python checks/rung1_quadrature.py

Every target value here is known independently of the code being tested -- an
elementary antiderivative, a geometric series summed by hand, or an exact
symmetry. None of these checks uses the residue theorem, which is the thing the
project is trying to stress test, and none compares the code to itself.
"""

import pathlib
import sys

# Run this file directly or via run_all.py; either way the project root has to be
# importable before `contour` or `checks` can be found.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import contour
from checks._runner import run

TWO_PI_I = 2j * np.pi


def check_monomials_on_unit_circle():
    """oint z^n dz around the unit circle is 2 pi i for n = -1 and 0 otherwise.

    Non-trivial because it pins down the orientation, the 2 pi normalization, and
    the gamma' factor all at once: get any of them wrong and the n = -1 case moves.
    Known independently -- z^(n+1)/(n+1) is a single-valued antiderivative for
    every n except -1, so those integrals vanish outright.
    """
    C = contour.circle()
    worst = 0.0
    for n in range(-4, 5):
        got = C.integrate(lambda z, n=n: z ** float(n), n=64)
        want = TWO_PI_I if n == -1 else 0.0
        worst = max(worst, abs(got - want))
    return worst < 1e-13, f"max |error| over n in [-4, 4] = {worst:.2e}"


def check_trapezoid_error_matches_closed_form():
    """The trapezoid error for 1/(z-a) on the unit circle is known in closed form.

    With z_j = exp(2 pi i j / N) the quadrature sum is

        (1/N) sum_j [1/(z_j - a)] (2 pi i z_j) = 2 pi i (1/N) sum_j z_j/(z_j - a),

    and z/(z - a) = sum_{k>=0} a^k z^(-k) for |a| < 1. Averaging z^(-k) over the
    N-th roots of unity gives 1 when N divides k and 0 otherwise, so the sum
    collapses to 2 pi i sum_{m>=0} a^(mN) = 2 pi i / (1 - a^N) and

        error(N) = 2 pi i a^N / (1 - a^N).

    This is much sharper than "it converges": it predicts every digit of the
    error, so it will catch a duplicated node, an off-by-one in the node spacing,
    or a wrong endpoint convention, none of which spoil mere convergence.

    The comparison is absolute rather than relative on purpose. By N = 44 the
    predicted error is itself down at 4e-13, and a relative test there is mostly
    measuring roundoff in the quadrature sum, not disagreement with the formula.
    The real claim is stronger and simpler: the gap between measured and
    predicted stays at machine noise across thirteen decades of signal.
    """
    C = contour.circle()
    worst = 0.0
    for a in (0.5, 0.7j, -0.35 + 0.4j):
        for N in range(2, 46, 2):
            observed = C.integrate(lambda z, a=a: 1.0 / (z - a), n=N) - TWO_PI_I
            predicted = TWO_PI_I * a ** N / (1.0 - a ** N)
            worst = max(worst, abs(observed - predicted))
    return worst < 1e-14, f"max |measured - predicted| over 13 decades = {worst:.2e}"


def check_geometric_convergence():
    """The error falls geometrically in N, not as a power of h.

    A pole at |a| = 0.5 predicts error ~ 2^(-N), so each extra 10 nodes should buy
    about three decimal digits. An h^2 rule would need to *square* the node count
    for that. Non-trivial: this is the property that lets the LHS be trusted to
    machine precision, and it fails immediately if the periodic endpoint is
    double-counted.
    """
    C = contour.circle()
    f = lambda z: 1.0 / (z - 0.5)
    errs = [abs(C.integrate(f, n=N) - TWO_PI_I) for N in (10, 20, 30, 40)]
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    floor = abs(C.integrate(f, n=64) - TWO_PI_I)
    ok = all(r > 500 for r in ratios) and floor < 1e-14
    return ok, f"error ratios per 10 nodes = {[f'{r:.0f}' for r in ratios]}, error at N=64 = {floor:.1e}"


def check_reparametrization_invariance():
    """Traversing the same curve at a non-uniform speed cannot change the integral.

    Non-trivial because the two runs sample the integrand at completely different
    points; only the correct gamma' weighting makes the sums agree. A missing or
    mis-scaled derivative factor passes every uniform-speed test and fails here.
    """
    C = contour.star(mean_radius=1.0, amplitude=0.3, lobes=3)
    D = contour.reparametrized(C, wobble=0.6)
    f = lambda z: 1.0 / (z - 0.2) + 1.0 / (z + 0.3 + 0.1j) ** 2 + np.exp(z)
    a, b = C.integrate(f, n=400), D.integrate(f, n=400)
    return abs(a - b) < 1e-11, f"|uniform - wobbled| = {abs(a - b):.2e} on |value| = {abs(a):.3f}"


def check_radius_and_center_independence():
    """oint dz/z is 2 pi i for every circle that encloses the origin, and 0 for those that do not.

    Non-trivial because the radius cancels only if gamma and gamma' scale together
    correctly; a contour that forgot to scale its derivative would give an answer
    proportional to R.
    """
    inside = [abs(contour.circle(0.0, R).integrate(lambda z: 1.0 / z, n=64) - TWO_PI_I)
              for R in (0.01, 1.0, 37.0)]
    outside = abs(contour.circle(5.0, 1.0).integrate(lambda z: 1.0 / z, n=64))
    ok = max(inside) < 1e-13 and outside < 1e-13
    return ok, f"max |error| enclosing = {max(inside):.2e}, |value| when excluded = {outside:.2e}"


def check_repeated_traversal_scales():
    """Going around k times multiplies the integral by k, for positive and negative k.

    Elementary -- the parameter interval is just covered k times -- but it is the
    only place the sign convention for clockwise travel gets exercised, and later
    rungs lean on it to produce winding numbers other than 1.
    """
    worst = 0.0
    for k in (-2, -1, 1, 2, 3):
        got = contour.circle(winding=k).integrate(lambda z: 1.0 / z, n=128)
        worst = max(worst, abs(got - k * TWO_PI_I))
    return worst < 1e-12, f"max |error| over k in [-2, 3] = {worst:.2e}"


def check_polygon_kills_polynomials():
    """oint z^n dz over a square is exactly 0 for n >= 0.

    Known independently: z^(n+1)/(n+1) is single valued, so the closed integral
    vanishes. Non-trivial for the edge assembly -- a wrong (b - a) Jacobian or a
    dropped edge leaves a residue of the polynomial behind. With 16 Gauss-Legendre
    nodes per edge this is exact to roundoff, not merely small.
    """
    P = contour.Polygon([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    worst = max(abs(P.integrate(lambda z, n=n: z ** n, n=16)) for n in range(6))
    return worst < 1e-13, f"max |value| over n in [0, 5] = {worst:.2e}"


def check_polygon_unit_pole():
    """oint dz/z over the square with corners (+-1 +- i) is 2 pi i, by direct antiderivative.

    Each edge can be done by hand: the edge from 1 - i to 1 + i contributes
    log(1 + i) - log(1 - i) = i pi/2, and the four edges are related by symmetry,
    for a total of 2 pi i. Non-trivial because it is computed with a completely
    different quadrature rule than every check above -- Gauss-Legendre per edge
    rather than the periodic trapezoid -- so the two rules are cross-checking the
    same elementary fact.
    """
    P = contour.Polygon([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    coarse = abs(P.integrate(lambda z: 1.0 / z, n=8) - TWO_PI_I)
    fine = abs(P.integrate(lambda z: 1.0 / z, n=48) - TWO_PI_I)
    return fine < 1e-13, f"|error| at 8 nodes/edge = {coarse:.1e}, at 48 nodes/edge = {fine:.1e}"


CHECKS = [
    check_monomials_on_unit_circle,
    check_trapezoid_error_matches_closed_form,
    check_geometric_convergence,
    check_reparametrization_invariance,
    check_radius_and_center_independence,
    check_repeated_traversal_scales,
    check_polygon_kills_polynomials,
    check_polygon_unit_pole,
]


def main():
    return run("Rung 1: contour quadrature (LHS)", CHECKS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
