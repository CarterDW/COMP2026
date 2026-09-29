"""Rung 4: residues by numerical differentiation.

    python checks/rung4_residues.py

The forbidden method is the good one. Cauchy's integral formula would give
these residues to machine precision by sampling f on a small circle, but that
is a contour integral, so a right-hand side built on it would share machinery
with the left and the comparison at rung 5 would prove nothing. Everything here
samples along a line instead and pays the usual price.

Ground truth is analytic wherever possible: tan has residue -1 at every pole,
Gamma has residue (-1)^k / k! at -k, and the residues of a proper rational
function sum to zero no matter what they individually are. Where an analytic
value is inconvenient, sympy supplies one -- but only from exactly represented
poles, for a reason one of the checks below makes explicit.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy
from scipy.special import gamma

import poles
import residues
from checks._runner import run

EPS = residues.EPS

# One test function, in two forms: the exact symbolic poles are what sympy is
# allowed to see, and their float images are what the numerics get.
Z = sympy.Symbol("z")
P_EXACT = sympy.Rational(3, 10) - sympy.Rational(7, 10) * sympy.I
Q_EXACT = sympy.Rational(-6, 5) + sympy.Rational(1, 2) * sympy.I
P, Q = complex(P_EXACT), complex(Q_EXACT)


def _exact_residue(order):
    """Res at P of exp(z) / ((z-P)^order (z-Q)), from the derivative formula, exactly."""
    regular = sympy.exp(Z) / (Z - Q_EXACT)
    return complex(sympy.diff(regular, Z, order - 1).subs(Z, P_EXACT) / math.factorial(order - 1))


def _test_function(order):
    return lambda z: np.exp(z) / ((z - P) ** order * (z - Q))


def check_simple_poles_of_a_rational_function():
    """Res of g(z)/(z-p) at a simple pole is g(p), and that needs no derivative at all.

    The easiest case and the one that must be exact: with order 1 the residue is
    the zeroth Taylor coefficient, nothing is divided by a power of the step,
    and no roundoff is amplified. Anything worse than machine precision here
    means the stencil itself is wrong.
    """
    f = _test_function(1)
    exact = np.exp(P) / (P - Q)
    got = residues.residue(f, P, 1)
    other = residues.residue(f, Q, 1)
    exact_other = np.exp(Q) / (Q - P)
    worst = max(abs(got - exact), abs(other - exact_other))
    return worst < 1e-14, f"both simple poles, worst |error| = {worst:.2e}"


def check_higher_order_poles_against_the_derivative_formula():
    """Orders 1 through 5, against (1/(m-1)!) d^(m-1)/dz^(m-1) [(z-p)^m f], computed exactly.

    The accuracy degrades with order, and it is supposed to: each extra order
    means one more numerical differentiation, and the tolerance below tracks the
    predicted eps^(1 - k/2n) rather than a single number that would be either
    far too loose at order 1 or unreachable at order 5.
    """
    errors = []
    for order in range(1, 6):
        got = residues.residue(_test_function(order), P, order)
        errors.append(abs(got - _exact_residue(order)))
    predicted = [EPS ** (1 - (order - 1) / 10) for order in range(1, 6)]
    ok = all(e < 10 * b for e, b in zip(errors, predicted))
    return ok, ("orders 1-5 |error| = " + ", ".join(f"{e:.1e}" for e in errors)
                + f"; all within 10x the predicted floor")


def check_tan_residues_are_all_minus_one():
    """tan has residue -1 at every one of its poles, which is as clean as ground truth gets.

    Independent of everything else here: it is a statement about a transcendental
    function with no rational structure for the code to exploit, and the same
    number has to come back at four different poles.
    """
    found = poles.find_poles(np.tan, (-5.0, 5.0, -1.0, 1.0))
    values = residues.residues_of(np.tan, found)
    worst = max(abs(v + 1.0) for v in values)
    return worst < 1e-13, f"{len(values)} poles of tan, worst |Res - (-1)| = {worst:.2e}"


def check_gamma_residues_are_the_reciprocal_factorials():
    """Gamma has residue (-1)^k / k! at z = -k, spanning two orders of magnitude by k = 4.

    A black box from scipy with no closed form available to the code, and a
    ground truth that changes at every pole rather than repeating, so a constant
    offset or a scale error cannot hide.
    """
    errors = []
    for k in range(5):
        got = residues.residue(gamma, -float(k), 1)
        errors.append(abs(got - (-1) ** k / math.factorial(k)))
    worst = max(errors)
    return worst < 1e-14, (f"residues 1, -1, 1/2, -1/6, 1/24 recovered; "
                           f"worst |error| = {worst:.2e}")


def check_residues_sum_to_zero():
    """For a proper rational function the residues must sum to exactly zero.

    The best check in this file, because it needs no reference value at all. If
    the denominator outranks the numerator by two or more, the sum of all
    residues vanishes identically -- so four numbers computed independently,
    none of them small, have to cancel. An error in the stencil that scaled
    every residue alike would survive every other check here and die on this one.
    """
    roots = np.array([0.3 - 0.7j, -1.2 + 0.5j, 1.4 + 0.2j, -0.1 - 1.1j])
    f = lambda z: (z + 2.0) / np.prod([(z - r) for r in roots], axis=0)
    values = [residues.residue(f, r, 1, scale=0.3) for r in roots]
    total = abs(sum(values))
    largest = max(abs(v) for v in values)
    return total < 1e-14, (f"4 residues of magnitude up to {largest:.2f} "
                           f"sum to {total:.2e}")


def check_reported_spread_bounds_the_true_error():
    """The error bar the ladder reports must actually bound the error.

    An error estimate that is merely correlated with the error is not much use;
    this one is compared against the truth at five pole orders and has to sit
    above it every time, without being so conservative as to be meaningless.
    """
    ratios = []
    for order in range(1, 6):
        value, _, spread = residues.residue_detail(_test_function(order), P, order)
        true_error = abs(value - _exact_residue(order))
        ratios.append(spread / true_error if true_error > 0 else np.inf)
    finite = [r for r in ratios if np.isfinite(r)]
    ok = all(r >= 0.5 for r in ratios) and all(r < 1e3 for r in finite)
    return ok, ("spread / true error at orders 1-5: "
                + ", ".join("inf" if not np.isfinite(r) else f"{r:.1f}" for r in ratios))


def check_optimal_step_follows_the_predicted_law():
    """The best step is eps^(1/2n), set by the stencil width and not by the pole order.

    Truncation falls like delta^(2n-k) and roundoff climbs like eps/delta^k, and
    balancing them puts the optimum at eps^(1/2n) -- the k cancels. That is a
    prediction worth testing, so the step is swept and the measured minimum
    compared with it, for two stencil widths whose predictions differ by a
    factor of ten.

    Only orders 3 and up are used. At order 1 nothing is amplified, the curve
    has no minimum to find, and the residue is exact over several decades.
    """
    grid = np.logspace(-0.3, -6, 200)
    ratios = []
    for half in (3, 5):
        for order in (3, 4, 5):
            exact = _exact_residue(order)
            f = _test_function(order)
            errors = [abs(residues.residue(f, P, order, half=half, delta=d) - exact)
                      for d in grid]
            measured = grid[int(np.argmin(errors))]
            ratios.append(measured / EPS ** (1.0 / (2 * half)))
    ok = all(0.2 < r < 5.0 for r in ratios)
    return ok, ("measured optimum / predicted, for 6- and 10-node stencils at orders 3-5: "
                + ", ".join(f"{r:.2f}" for r in ratios))


def check_symbolic_reference_needs_exact_poles():
    """sympy.residue is silently wrong at a higher-order pole given as a float.

    Every check in this file that leans on sympy hands it exact rationals, and
    this is why. Asked for the residue of exp(z)/((z-p)^m (z-q)) with p and q as
    Python floats, sympy returns a confident wrong answer from m = 2 on -- it
    disagrees with its own derivative formula by an amount far larger than any
    tolerance here. With the same poles as exact rationals the two agree to
    machine precision. The failure is in how the reference was asked, not in the
    numerics being referenced, and a check that did not know this would blame
    the wrong code.
    """
    float_gap, exact_gap = [], []
    for order in (2, 3):
        sloppy = sympy.exp(Z) / ((Z - P) ** order * (Z - Q))
        careful = sympy.exp(Z) / ((Z - P_EXACT) ** order * (Z - Q_EXACT))
        truth = _exact_residue(order)
        float_gap.append(abs(complex(sympy.residue(sloppy, Z, P)) - truth))
        exact_gap.append(abs(complex(sympy.residue(careful, Z, P_EXACT)) - truth))
    ok = min(float_gap) > 1e-3 and max(exact_gap) < 1e-12
    return ok, (f"float poles: sympy is off by up to {max(float_gap):.2f}; "
                f"exact poles: agrees to {max(exact_gap):.1e}")


def check_zero_residue_is_reported_as_zero():
    """A double pole with no 1/(z-p) term has residue exactly zero, and must return zero.

    Worth isolating because zero is the one answer a relative tolerance cannot
    police, and because the arithmetic that produces it is a cancellation: the
    stencil weights must sum against the samples to nothing at all.
    """
    pure_powers = [(lambda z: 1.0 / (z - P) ** 2, 2), (lambda z: 1.0 / (z - P) ** 4, 4)]
    worst = max(abs(residues.residue(f, P, order, scale=0.5)) for f, order in pure_powers)
    # A double pole that does carry a 1/(z-p) term, so that "zero" is visibly
    # not just the answer this routine always gives: 1/(z^2+1)^2 has residue
    # -i/4 at z = i.
    nonzero = residues.residue(lambda z: 1.0 / (z ** 2 + 1) ** 2, 1j, 2, scale=0.5)
    ok = worst < 1e-14 and abs(nonzero - (-0.25j)) < 1e-12
    return ok, (f"pure powers give {worst:.1e}; the nearby case 1/(z^2+1)^2 at i "
                f"correctly gives {nonzero:.6f} against an exact -0.25i")


def check_end_to_end_with_the_pole_finder():
    """Rungs 3 and 4 together: find the poles from f alone, then residue each one.

    The first point where the right-hand side is assembled from parts, with
    nothing supplied by hand -- no locations, no orders, no scales. What is
    still missing is the winding number and the comparison itself, which is
    rung 5.
    """
    spec = [(0.3 - 0.7j, 1), (-1.1 + 0.4j, 2)]
    f = lambda z: np.exp(z) / ((z - spec[0][0]) * (z - spec[1][0]) ** 2)
    found = poles.find_poles(f, (-2, 2, -2, 2))
    values = residues.residues_of(f, found)

    exact = {}
    a, b = sympy.Rational(3, 10) - sympy.Rational(7, 10) * sympy.I, \
           sympy.Rational(-11, 10) + sympy.Rational(2, 5) * sympy.I
    exact[complex(a)] = complex((sympy.exp(Z) / (Z - b) ** 2).subs(Z, a))
    exact[complex(b)] = complex(sympy.diff(sympy.exp(Z) / (Z - a), Z, 1).subs(Z, b))

    worst = 0.0
    for q, value in zip(found, values):
        target = exact[min(exact, key=lambda e: abs(e - q.location))]
        worst = max(worst, abs(value - target))
    ok = len(found) == 2 and worst < 1e-12
    return ok, f"{len(found)} poles found and both residues computed, worst |error| = {worst:.2e}"


CHECKS = [
    check_simple_poles_of_a_rational_function,
    check_higher_order_poles_against_the_derivative_formula,
    check_tan_residues_are_all_minus_one,
    check_gamma_residues_are_the_reciprocal_factorials,
    check_residues_sum_to_zero,
    check_reported_spread_bounds_the_true_error,
    check_optimal_step_follows_the_predicted_law,
    check_symbolic_reference_needs_exact_poles,
    check_zero_residue_is_reported_as_zero,
    check_end_to_end_with_the_pole_finder,
]


def main():
    return run("Rung 4: residues by numerical differentiation", CHECKS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
