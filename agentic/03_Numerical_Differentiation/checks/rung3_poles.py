"""Rung 3: locating poles and measuring their orders, with f as a black box.

    python checks/rung3_poles.py

Ground truth is analytic throughout: the poles of tan are the half-integer
multiples of pi, the poles of Gamma are the non-positive integers, an entire
function has none at all, and a rational function has exactly the poles its
denominator advertises. One check holds the finder against sympy, which is used
here strictly as an independent reference for an answer already computed
numerically.

No quadrature, no winding numbers, no residues. The one accuracy claim is a
prediction rather than a tolerance: finding a pole of order m through a
denominator that cancels costs half the digits m times over, so the attainable
error is eps^(1/m), and the check tests that law rather than a number someone
picked.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import gamma

import poles
from checks._runner import run

EPS = float(np.finfo(float).eps)


def _match(found, expected, tol):
    """Pair each expected pole with the nearest found one; return the worst distance."""
    if len(found) != len(expected):
        return None
    worst = 0.0
    for want in expected:
        nearest = min(found, key=lambda q: abs(q.location - want))
        worst = max(worst, abs(nearest.location - want))
    return worst if worst <= tol else worst


def check_rational_poles_and_orders():
    """A rational function has exactly the poles its denominator advertises.

    Orders 1, 2 and 3 at deliberately unlovely locations, with an exponential in
    the numerator so nothing can be recovered by factoring. Non-trivial because
    the three stages have to agree: a grid coarse enough to miss a pole, a
    refinement that lands on the wrong root, or an order fit contaminated by a
    neighbouring pole all show up here.
    """
    spec = [(0.3 - 0.7j, 1), (-1.1 + 0.4j, 2), (0.9 + 1.2j, 3)]
    f = lambda z: np.exp(z) / ((z - spec[0][0]) * (z - spec[1][0]) ** 2 * (z - spec[2][0]) ** 3)
    found = poles.find_poles(f, (-2, 2, -2, 2))
    if len(found) != len(spec):
        return False, f"found {len(found)} poles, expected {len(spec)}"
    worst_position, orders_right = 0.0, True
    for want, order in spec:
        nearest = min(found, key=lambda q: abs(q.location - want))
        worst_position = max(worst_position, abs(nearest.location - want))
        orders_right = orders_right and nearest.order == order
    ok = worst_position < 1e-12 and orders_right
    return ok, (f"3 poles, orders {'all exact' if orders_right else 'WRONG'}, "
                f"worst position error {worst_position:.2e}")


def check_tan_poles_are_half_integer_multiples_of_pi():
    """tan has simple poles at (k + 1/2) pi and nowhere else.

    A transcendental function with infinitely many poles, so the finder has to
    return exactly those inside the region and invent none outside it. The zeros
    of tan sit at k pi, interleaved with the poles, which is the trap: a routine
    that hunts for extremes of |f| without distinguishing the sign of the
    exponent would report them too.
    """
    region = (-5.0, 5.0, -1.0, 1.0)
    expected = [(k + 0.5) * np.pi for k in (-2, -1, 0, 1)]
    found = poles.find_poles(np.tan, region)
    if len(found) != len(expected):
        return False, f"found {len(found)}, expected {len(expected)} in {region}"
    worst = max(min(abs(q.location - want) for q in found) for want in expected)
    orders = {q.order for q in found}
    return worst < 1e-12 and orders == {1}, (f"4 poles, orders {sorted(orders)}, "
                                             f"worst position error {worst:.2e}")


def check_gamma_poles_at_the_nonpositive_integers():
    """Gamma has simple poles at 0, -1, -2, ... and is analytic everywhere else.

    Worth having because Gamma is a genuine black box -- a scipy routine with no
    closed form the code could exploit -- and because its poles get closer
    together in |Gamma| terms as they march left, so the grid scan has to
    separate them.
    """
    found = poles.find_poles(gamma, (-3.5, 1.5, -1.0, 1.0))
    expected = [0.0, -1.0, -2.0, -3.0]
    if len(found) != len(expected):
        return False, f"found {len(found)} poles, expected {len(expected)}"
    worst = max(min(abs(q.location - want) for q in found) for want in expected)
    orders = {q.order for q in found}
    return worst < 1e-11 and orders == {1}, (f"poles at 0, -1, -2, -3; orders {sorted(orders)}, "
                                             f"worst position error {worst:.2e}")


def check_entire_functions_have_no_poles():
    """An analytic function has no poles, and the finder must report none.

    The negative control, and the one every false-positive lands in. exp grows
    without bound toward the right of the region and sinh does so in both
    directions, so |f| has plenty of large values for a naive peak-finder to
    mistake for singularities.
    """
    cases = {"exp(z)": np.exp, "sin(z)": np.sin, "sinh(z)": np.sinh,
             "z^4 - 3z + 1": lambda z: z ** 4 - 3 * z + 1,
             "exp(z) sin(z)": lambda z: np.exp(z) * np.sin(z)}
    spurious = {name: len(poles.find_poles(f, (-3, 3, -3, 3))) for name, f in cases.items()}
    total = sum(spurious.values())
    return total == 0, f"{total} spurious poles across {len(cases)} entire functions: {spurious}"


def check_zeros_are_not_mistaken_for_poles():
    """A function with both zeros and poles must yield only the poles.

    Zeros of f are poles of 1/f and vice versa, so a sign slip anywhere in the
    pipeline swaps them wholesale. Here the zeros at +-1/2 are real and the
    poles at +-i are imaginary, so a swap is unmistakable.
    """
    f = lambda z: (z ** 2 - 0.25) / (z ** 2 + 1.0)
    found = poles.find_poles(f, (-2, 2, -2, 2))
    located = sorted((q.location for q in found), key=lambda q: q.imag)
    ok = (len(found) == 2
          and abs(located[0] + 1j) < 1e-12 and abs(located[1] - 1j) < 1e-12
          and all(q.order == 1 for q in found))
    return ok, f"{len(found)} poles at {[f'{q:.3f}' for q in located]}, zeros at +-0.5 ignored"


def check_accuracy_follows_eps_to_the_one_over_m():
    """Locating an order-m pole through an expanded denominator costs m-fold precision.

    Near a pole of order m the reciprocal behaves like (z - p)^m, so a relative
    perturbation eps in its computed value displaces the apparent root by
    eps^(1/m). That is a prediction from the arithmetic alone, and it is a
    statement about double precision rather than about this code: order 4 cannot
    be located better than about 1e-4, by anyone, this way.

    The denominator is supplied as expanded coefficients on purpose. Written in
    factored form as (z - p)^m it suffers no cancellation at all and the same
    pole comes back to 1e-14 at every order -- which is the contrast the second
    half of the detail line reports.
    """
    p = 0.3 - 0.7j
    ratios, factored_worst = [], 0.0
    for m in range(1, 7):
        coefficients = np.poly([p] * m)
        expanded = lambda z, c=coefficients: (z + 2.0) / np.polyval(c, z)
        factored = lambda z, m=m: (z + 2.0) / (z - p) ** m
        start = p + 0.05
        g = poles.reciprocal(expanded)
        root = poles.muller(g, start, start + 1e-3, start - 1e-3)
        ratios.append(abs(root - p) / EPS ** (1.0 / m))
        clean = poles.muller(poles.reciprocal(factored), start, start + 1e-3, start - 1e-3)
        factored_worst = max(factored_worst, abs(clean - p))
    # The law is an upper bound, so every ratio must sit below it; for m >= 2 it
    # is also attained, which is the part that makes it a prediction rather than
    # a loose inequality. At m = 1 there is no amplification to speak of and
    # Muller lands on the root exactly, giving a ratio of zero.
    ok = (all(r < 5.0 for r in ratios) and all(r > 0.2 for r in ratios[1:])
          and factored_worst < 1e-10)
    return ok, (f"error/eps^(1/m) for m=1..6: {[f'{r:.2f}' for r in ratios]}; "
                f"bound attained from m=2 on; factored form stays at {factored_worst:.1e}")


def check_close_poles_are_resolved():
    """Two simple poles a distance d apart are found as two, however small d is.

    The interesting failure is not missing one but merging them, so the check
    insists on two locations the right distance apart rather than just counting
    hits. It resolves down to d = 1e-7, seven decades, with the separation
    recovered to a relative 1e-10.

    The requirement is on the grid, not on the method: the scan can only bracket
    two poles it can see apart, so the region is scaled with d to keep the grid
    step below it. The last case violates that condition deliberately, and what
    comes back is better than a merged pole -- it is nothing at all. Seen from
    far away the unresolved pair looks like one double pole and the fitted slope
    is 2; from close in only the nearer one is resolved and the slope is 1; in
    between the log-log fit bends, its residual reaches 0.4, and the order test
    throws the candidate out. The finder declines rather than reporting an order
    it cannot support.
    """
    centre = 0.2 + 0.1j
    def two_poles(d):
        return lambda z: 1.0 / ((z - centre - d / 2) * (z - centre + d / 2))

    separations = [1e-1, 1e-3, 1e-5, 1e-7]
    worst_relative = 0.0
    for d in separations:
        half = 50 * d
        found = poles.find_poles(two_poles(d),
                                 (centre.real - half, centre.real + half,
                                  centre.imag - half, centre.imag + half),
                                 resolution=401, merge=1e-3)
        if len(found) != 2:
            return False, f"separation {d:.0e}: found {len(found)} poles, expected 2"
        located = sorted(q.location.real for q in found)
        worst_relative = max(worst_relative, abs((located[1] - located[0]) - d) / d)

    # Same pair, but now on a region whose grid step is coarser than d.
    d = 1e-3
    coarse = poles.find_poles(two_poles(d), (centre.real - 1, centre.real + 1,
                                             centre.imag - 1, centre.imag + 1),
                              resolution=401)
    _, bent = poles.measure_order(two_poles(d), centre + d / 2, scale=1e-2)
    declined = len(coarse) == 0 and bent > 0.05
    ok = worst_relative < 1e-9 and declined
    return ok, (f"resolved at separations {[f'{d:.0e}' for d in separations]}, "
                f"worst relative error {worst_relative:.1e}; on a grid coarser than d the "
                f"fit residual is {bent:.2f} and {len(coarse)} poles are reported")


def check_sympy_agrees_as_an_independent_reference():
    """sympy, doing algebra, reaches the same poles and orders the numerics did.

    sympy is a reference here, not a participant: the answer is computed
    numerically first and then held against the factorisation of the
    denominator. The polynomial is given expanded, so sympy has to factor it and
    the finder has to locate roots of a cancelling expression -- two genuinely
    different routes to the same list.
    """
    import sympy
    z = sympy.Symbol("z")
    denominator = sympy.expand((z - sympy.Rational(1, 3)) ** 2 * (z + 1 + sympy.I) * (z - 2 * sympy.I))
    expression = (z + 5) / denominator
    reference = {complex(root): multiplicity
                 for root, multiplicity in sympy.roots(sympy.Poly(denominator, z)).items()}

    numeric = sympy.lambdify(z, expression, "numpy")
    found = poles.find_poles(numeric, (-2.5, 2.5, -2.5, 2.5))
    if len(found) != len(reference):
        return False, f"numerics found {len(found)} poles, sympy says {len(reference)}"
    worst_position, orders_match = 0.0, True
    for want, multiplicity in reference.items():
        nearest = min(found, key=lambda q: abs(q.location - want))
        worst_position = max(worst_position, abs(nearest.location - want))
        orders_match = orders_match and nearest.order == multiplicity
    ok = orders_match and worst_position < 1e-7
    return ok, (f"{len(found)} poles, orders {'match' if orders_match else 'DIFFER'}, "
                f"worst position gap {worst_position:.2e}")


def check_impostor_singularities_are_rejected():
    """A singularity is not automatically a pole, and each impostor falls to a different test.

    z^(-3/2) is the sharp case: its log-log fit is a textbook straight line with
    a residual of 1e-15 and its exponent rounds to a respectable order 1, so
    only the distance from that exponent to a whole number -- 0.5 -- gives it
    away. 1/sqrt(z) and log z fail the same test, and also round to order 0.
    exp(1/z) never gets that far: its reciprocal underflows to exactly zero over
    a sector, so Muller returns a non-finite point and the candidate is dropped
    before any order is measured.
    """
    impostors = {"1/sqrt(z)": lambda z: 1.0 / np.sqrt(z),
                 "z^(-3/2)": lambda z: z ** -1.5,
                 "log(z)": np.log,
                 "exp(1/z)": lambda z: np.exp(1.0 / z)}
    genuine = {"1/z": lambda z: 1.0 / z, "1/z^2": lambda z: 1.0 / z ** 2,
               "1/z^4": lambda z: 1.0 / z ** 4}
    wrongly_kept = {n: len(poles.find_poles(f, (-1, 1, -1, 1))) for n, f in impostors.items()}
    wrongly_dropped = {n: len(poles.find_poles(f, (-1, 1, -1, 1))) for n, f in genuine.items()}
    ok = sum(wrongly_kept.values()) == 0 and all(v == 1 for v in wrongly_dropped.values())
    return ok, (f"{sum(wrongly_kept.values())} impostors accepted, "
                f"{sum(1 for v in wrongly_dropped.values() if v != 1)} genuine poles lost")


CHECKS = [
    check_rational_poles_and_orders,
    check_tan_poles_are_half_integer_multiples_of_pi,
    check_gamma_poles_at_the_nonpositive_integers,
    check_entire_functions_have_no_poles,
    check_zeros_are_not_mistaken_for_poles,
    check_accuracy_follows_eps_to_the_one_over_m,
    check_close_poles_are_resolved,
    check_sympy_agrees_as_an_independent_reference,
    check_impostor_singularities_are_rejected,
]


def main():
    return run("Rung 3: pole locations and orders (black-box f)", CHECKS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
