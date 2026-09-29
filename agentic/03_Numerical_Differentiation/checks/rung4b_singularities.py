"""Rung 4b: refusing to answer questions that have no answer.

    python checks/rung4b_singularities.py

Rungs 3 and 4 quietly assumed every singularity is a pole. They were wrong, and
the way they were wrong was the dangerous kind: `find_poles` returned an empty
list for exp(1/z) and `residue` returned 5.895204090168 for it -- the same value
whether order 1 or order 2 was claimed. Assembled into a right-hand side at
rung 5, that would have produced RHS = 0 against LHS = 2 pi i and looked for all
the world like a failure of the residue theorem.

The theorem is fine. exp(1/z) = sum z^(-n)/n! has residue 1, and the rung-1
quadrature returns 2 pi i for it to 1e-16. What is missing is a way to *compute*
that residue without a contour integral, and there isn't one: at an essential
singularity (z-p)^m f is analytic for no m at all, so differentiation has
nothing to work with. The limitation is real, so the code must announce it.

Two guards, in two places. `residues.assert_is_pole` refuses when (z-p)^order f
grows as the step shrinks, and `poles.survey` reports whatever fails its tests
as a non-pole with the reason attached, instead of dropping it.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import gamma

import contour
import poles
import residues
from checks._runner import run

TWO_PI_I = 2j * np.pi
ESSENTIAL = lambda z: np.exp(1.0 / z)


def _refuses(call):
    """True if `call` raises ValueError. Returns the message for reporting."""
    try:
        call()
    except ValueError as failure:
        return True, str(failure)
    return False, "returned a value"


def check_essential_singularity_is_refused_not_guessed():
    """residue() must decline at exp(1/z) rather than inventing a number.

    Before the boundedness guard this returned 5.895204090168 for order 1 and
    the identical value for order 2 -- a confident answer to a question with no
    answer, and two contradictory questions given one reply.
    """
    outcomes = [_refuses(lambda o=order: residues.residue(ESSENTIAL, 0.0, o, scale=0.5))
                for order in (1, 2, 3)]
    refused = all(ok for ok, _ in outcomes)
    return refused, ("orders 1, 2, 3 all refused" if refused else
                     f"NOT refused: {[m for ok, m in outcomes if not ok]}")


def check_survey_reports_what_find_poles_discards():
    """Every non-pole singularity comes back with a location and a reason.

    Three different failure routes, and the survey has to name each one rather
    than return an empty list as find_poles does. The locations matter as much
    as the reasons: rung 5 needs to know *where* the thing it cannot handle is,
    to tell whether a given contour encloses it.
    """
    cases = {"exp(1/z)": ESSENTIAL, "1/sqrt(z)": lambda z: 1.0 / np.sqrt(z), "log(z)": np.log}
    reasons, silent = {}, []
    for name, f in cases.items():
        result = poles.survey(f, (-1, 1, -1, 1))
        if result.poles or not result.non_poles:
            silent.append(name)
            continue
        reasons[name] = result.non_poles[0].reason
        if poles.find_poles(f, (-1, 1, -1, 1)):
            silent.append(name)
    distinct = len(set(reasons.values()))
    ok = not silent and distinct == 3
    return ok, (f"{len(reasons)}/3 reported, {distinct} distinct reasons"
                if ok else f"silently dropped: {silent}")


def check_understated_order_is_refused():
    """Asking for order 1 at a triple pole must fail, not return the wrong number.

    The same guard as the essential-singularity case, and it fires for the same
    reason: (z-p)^1 f still blows up. Worth its own check because this is the
    likely mistake in practice -- an order estimated one too low by a marginal
    fit -- and the wrong answer it used to give looked perfectly plausible.
    """
    triple = lambda z: np.exp(z) / z ** 3
    outcomes = [_refuses(lambda o=order: residues.residue(triple, 0.0, o, scale=0.5))
                for order in (1, 2)]
    correct = residues.residue(triple, 0.0, 3, scale=0.5)
    ok = all(r for r, _ in outcomes) and abs(correct - 0.5) < 1e-12
    return ok, (f"orders 1 and 2 refused; order 3 gives {correct.real:.12f} "
                f"against an exact 1/2")


def check_overstated_order_still_gives_the_right_residue():
    """Claiming too high an order is harmless, and the guard lets it through on purpose.

    If f has a pole of order 3 and order 5 is claimed, (z-p)^5 f is still
    analytic at p and its Taylor coefficient a_4 is still the residue -- the
    extra factors contribute nothing. So the answer stays right and only the
    conditioning suffers. A guard that rejected this would be wrong, which is
    why it tests boundedness rather than equality of the order.
    """
    triple = lambda z: np.exp(z) / z ** 3
    values = [residues.residue(triple, 0.0, order, scale=0.5) for order in (3, 4, 5, 6)]
    worst = max(abs(v - 0.5) for v in values)
    return worst < 1e-9, (f"orders 3, 4, 5, 6 all give 1/2, worst |error| = {worst:.2e} "
                          f"(conditioning degrades, correctness does not)")


def check_the_theorem_holds_where_our_machinery_cannot_follow():
    """The residue theorem is fine at an essential singularity; only our RHS is not.

    This is the finding that justifies the whole rung. exp(1/z) has residue 1,
    the left-hand side confirms it to machine precision, and the right-hand side
    has no way to compute it that does not use a contour integral. That gap is a
    property of the method, not of the theorem, and the honest response is to
    refuse rather than to report zero.
    """
    lhs = contour.circle(0.0, 1.0).integrate(ESSENTIAL, n=4096)
    surveyed = poles.survey(ESSENTIAL, (-1.5, 1.5, -1.5, 1.5))
    lhs_right = abs(lhs - TWO_PI_I) < 1e-14
    rhs_declines = not surveyed.poles and len(surveyed.non_poles) == 1
    return lhs_right and rhs_declines, (
        f"LHS = {lhs.imag:.12f}i against an exact 2*pi*i (|error| {abs(lhs - TWO_PI_I):.1e}); "
        f"the survey reports 0 poles and 1 non-pole, so no RHS is offered")


def check_a_mixed_function_separates_correctly():
    """A function with both a pole and an essential singularity must yield both verdicts.

    The realistic case, and the one where dropping the awkward singularity would
    be least visible: there is a perfectly good pole to report, so an empty
    non_poles list would look like success.
    """
    f = lambda z: np.exp(1.0 / z) + 1.0 / (z - 0.6)
    result = poles.survey(f, (-1, 1, -1, 1))
    got_pole = (len(result.poles) == 1 and abs(result.poles[0].location - 0.6) < 1e-9
                and result.poles[0].order == 1)
    got_flag = len(result.non_poles) == 1 and abs(result.non_poles[0].location) < 0.05
    value = residues.residue(f, 0.6, 1, scale=0.2) if got_pole else None
    residue_right = value is not None and abs(value - 1.0) < 1e-12
    ok = got_pole and got_flag and residue_right
    return ok, (f"pole at 0.6 found with residue {value.real:.12f} (exact 1); "
                f"the essential singularity at the origin is flagged, not dropped")


def check_the_stencil_is_complex():
    """The sampling line must be complex even when the pole sits on the real axis.

    A real p made p + offsets*delta a real array, so any f with a branch cut on
    the negative reals returned nan on half the stencil -- for reasons having
    nothing to do with the singularity being measured. Here f = 1/(z sqrt(z-1))
    has a simple pole at 0 with residue 1/sqrt(-1) = -i, and every stencil node
    has z - 1 negative. With a real stencil the whole thing is nan.
    """
    f = lambda z: 1.0 / (z * np.sqrt(z - 1))
    scale = 0.15                      # keeps every node strictly inside z - 1 < 0
    got = residues.residue(f, 0.0, 1, scale=scale)
    with np.errstate(invalid="ignore"):
        # Deliberately taking the square root of negative reals, to show what
        # the old real-valued stencil was handing to f.
        real_nodes = np.sqrt(residues.stencil_offsets(5) * scale - 1.0)
    return abs(got + 1j) < 1e-12 and np.all(np.isnan(real_nodes)), (
        f"residue at a real pole = {got:.12f} against an exact -1i; "
        f"all {len(real_nodes)} nodes taken as reals give nan")


def check_genuine_poles_are_unaffected():
    """The guards must not cost anything on the cases rung 4 already handled.

    A guard that also rejected honest work would be worse than none. tan and
    Gamma go through the full path -- survey, then residue with its boundedness
    check -- and must come back exactly as before.
    """
    tan_poles = poles.survey(np.tan, (-5.0, 5.0, -1.0, 1.0))
    tan_values = residues.residues_of(np.tan, tan_poles.poles)
    tan_worst = max(abs(v + 1.0) for v in tan_values)
    gamma_worst = max(abs(residues.residue(gamma, -float(k), 1) - (-1) ** k / math.factorial(k))
                      for k in range(5))
    ok = (len(tan_poles.poles) == 4 and not tan_poles.non_poles
          and tan_worst < 1e-13 and gamma_worst < 1e-14)
    return ok, (f"tan: 4 poles, 0 non-poles, worst |Res+1| = {tan_worst:.1e}; "
                f"Gamma: worst |error| = {gamma_worst:.1e}")


CHECKS = [
    check_essential_singularity_is_refused_not_guessed,
    check_survey_reports_what_find_poles_discards,
    check_understated_order_is_refused,
    check_overstated_order_still_gives_the_right_residue,
    check_the_theorem_holds_where_our_machinery_cannot_follow,
    check_a_mixed_function_separates_correctly,
    check_the_stencil_is_complex,
    check_genuine_poles_are_unaffected,
]


def main():
    return run("Rung 4b: singularities that are not poles", CHECKS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
