"""Rung 2: winding numbers, the n(C, p) factor of the right-hand side.

    python checks/rung2_winding.py

Ground truth here is geometric and known in advance: a circle traversed k times
winds k, a star-shaped curve encloses exactly the points inside its radial
graph, the two lobes of a figure eight wind oppositely. Two algorithms that
share no code are also held against each other.

No check in this file uses a quadrature rule, evaluates an integrand, or invokes
the residue theorem. In particular the identity n(C, p) = (1/2 pi i) oint dz/(z-p)
is not used: the accumulated-argument algorithm *is* that integral computed
geometrically, so the comparison would test one idea twice, and it is the
simplest special case of the theorem under test. It appears in rung 5 as a
result instead.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import contour
import winding
from checks._runner import run

RNG = np.random.default_rng(20260925)


def check_circle_winds_k_times():
    """A circle traversed k times winds k times about its centre, for k of either sign.

    The one case where the answer is beyond dispute, and the only place the sign
    convention for clockwise travel is pinned down. Both algorithms are asked,
    so a sign error in either is caught here rather than propagating into rung 5.
    """
    bad = []
    for k in range(-3, 4):
        if k == 0:
            continue
        C = contour.circle(center=0.4 - 0.2j, radius=1.3, winding=k)
        by_arg = winding.winding_number(C, 0.4 - 0.2j)
        by_ray = winding.winding_by_crossing(C, 0.4 - 0.2j)
        if by_arg != k or by_ray != k:
            bad.append((k, by_arg, by_ray))
    return not bad, f"k = -3..3 (excluding 0), both algorithms: {'all exact' if not bad else bad}"


def check_star_interior_matches_its_radial_graph():
    """For r(theta) = 1 + A cos(3 theta) the inside is known exactly: |p| < r(arg p).

    A strongly non-convex curve with an analytic membership test, so 4000 random
    points each have a ground truth that owes nothing to the code. Points in the
    notches are outside the curve while inside its convex hull, which is exactly
    where a shortcut implementation goes wrong.
    """
    amplitude = 0.5
    C = contour.star(mean_radius=1.0, amplitude=amplitude, lobes=3)
    p = RNG.uniform(-1.8, 1.8, 4000) + 1j * RNG.uniform(-1.8, 1.8, 4000)
    expected = (np.abs(p) < 1.0 + amplitude * np.cos(3 * np.angle(p))).astype(int)
    got = np.array([winding.winding_number(C, q) for q in p])
    wrong = int(np.sum(got != expected))
    in_notch = int(np.sum((expected == 0) & (np.abs(p) < 1.0)))
    return wrong == 0, (f"{len(p)} random points, {wrong} disagree with the analytic "
                        f"interior; {in_notch} of them sit in a notch (outside the curve, "
                        f"inside its convex hull)")


def check_figure_eight_lobes_are_opposite():
    """The two lobes of a figure eight wind +1 and -1, and the pair sums to zero.

    The decisive test of a *signed* count. Any routine that answers "inside or
    outside" rather than counting orientation reports +1 for both lobes and
    fails here, while passing every simple closed curve.
    """
    F = contour.figure_eight()
    right_arg = winding.winding_number(F, 0.6)
    left_arg = winding.winding_number(F, -0.6)
    right_ray = winding.winding_by_crossing(F, 0.6)
    left_ray = winding.winding_by_crossing(F, -0.6)
    ok = (right_arg, left_arg) == (-1, 1) and (right_ray, left_ray) == (-1, 1)
    return ok, f"right lobe {right_arg:+d}, left lobe {left_arg:+d}, sum {right_arg + left_arg:+d}"


def check_two_algorithms_agree_everywhere():
    """Accumulated argument and signed ray crossings agree at every test point.

    The two share no code: one is trigonometric and branch-sensitive, the other
    is sign tests and a linear interpolation. Points are drawn over a
    self-intersecting path and a multiply-traversed circle, so the comparison
    covers winding numbers of -2, -1, 0, 1 and 2.
    """
    paths = [contour.figure_eight(), contour.circle(winding=2),
             contour.star(amplitude=0.5, lobes=5), contour.circle(winding=-2)]
    seen, disagreements = set(), 0
    for path in paths:
        p = RNG.uniform(-1.6, 1.6, 300) + 1j * RNG.uniform(-1.6, 1.6, 300)
        for q in p:
            a = winding.winding_number(path, q)
            b = winding.winding_by_crossing(path, q)
            seen.add(a)
            disagreements += a != b
    return disagreements == 0, (f"1200 points over 4 paths, {disagreements} disagreements; "
                                f"winding numbers seen: {sorted(seen)}")


def check_invariance_under_rigid_motion():
    """Moving, rotating and scaling the whole configuration cannot change the answer.

    Non-trivial for the ray-crossing algorithm in particular, which singles out
    the +x direction: if that choice leaked into the result, a rotation would
    change it. The rotation angles are deliberately generic, so no edge lands
    parallel to the ray by luck.
    """
    base = winding.as_closed_polyline(contour.star(amplitude=0.5, lobes=3), 2048)
    probes = np.array([0.2 + 0.1j, -0.3 + 0.4j, 1.2 - 0.9j, 0.95 * np.exp(1j * np.pi / 3)])
    reference = [(winding.winding_number(base, q), winding.winding_by_crossing(base, q))
                 for q in probes]
    bad = 0
    for shift, angle, scale in ((3 - 2j, 0.7, 1.0), (-11 + 5j, 2.31, 4.0), (0.5j, 5.0, 0.01)):
        w = scale * np.exp(1j * angle)
        for q, ref in zip(probes, reference):
            moved = shift + w * base
            moved_q = shift + w * q
            got = (winding.winding_number(moved, moved_q),
                   winding.winding_by_crossing(moved, moved_q))
            bad += got != ref
    return bad == 0, f"3 rigid motions x 4 probes x 2 algorithms, {bad} changed"


def check_polygon_vertices_are_enough():
    """A polygon needs no resampling: its bare vertices give the exact winding number.

    True because a straight edge subtends less than pi from any point off it, so
    the branch choice in the accumulated argument can never go wrong. Checked by
    comparing the raw vertex list against the same polygon resampled 500x more
    finely, at points deliberately placed just inside an edge where the
    subtended angle is closest to pi.
    """
    V = [1.4 + 0.3j, -0.2 + 1.5j, -1.3 - 0.1j, 0.1 - 1.2j, 0.7 - 0.5j]
    P = contour.Polygon(V)
    dense = np.append(np.concatenate([np.linspace(a, b, 500, endpoint=False)
                                      for a, b in P.edges()]), V[0])
    probes = [0.0, 0.05 + 0.02j, -0.2 + 0.9j, 0.65 - 0.48j, 1.35 + 0.29j, 5.0 + 5.0j]
    coarse = [winding.winding_number(np.array(V, dtype=complex), q) for q in probes]
    fine = [winding.winding_number(dense, q) for q in probes]
    return coarse == fine, f"5 vertices vs 2500 samples at {len(probes)} probes: {coarse} vs {fine}"


def check_point_on_the_path_is_refused():
    """A point lying on the path has no winding number, and both algorithms say so.

    The alternative is returning a plausible integer for a question that has no
    answer, which is precisely the silent-default failure mode that hides bugs.
    Two different mechanisms are exercised. A point in the middle of a polygon
    edge is caught exactly, by collinearity. A point on a smooth curve is not
    caught that way at all -- it misses the inscribed polyline by 8e-4 at
    n = 2048, so both algorithms would answer "outside" with total confidence --
    and is caught instead by comparing the clearance at two sampling densities.
    """
    C = contour.circle()
    on_curve = C.gamma(np.array([0.25]))[0]
    edge_midpoint = 0.0 + 1j
    refusals = 0
    for path, q in ((C, on_curve), (contour.Polygon([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]),
                                    edge_midpoint)):
        for algorithm in (winding.winding_number, winding.winding_by_crossing):
            try:
                algorithm(path, q)
            except ValueError:
                refusals += 1
    return refusals == 4, f"{refusals}/4 calls refused instead of guessing"


def check_undersampling_is_refused_not_aliased():
    """Too few samples must raise, not silently return the wrong integer.

    A circle wound 50 times and sampled 64 times turns about 5 radians per step.
    The principal branch cannot tell that from 5 - 2 pi, so the accumulated
    argument would alias to a confidently wrong answer. The guard catches it, and
    the same contour sampled finely returns 50.
    """
    C = contour.circle(winding=50)
    try:
        winding.winding_number(C, 0, n=64)
        refused = False
    except ValueError:
        refused = True
    fine = winding.winding_number(C, 0, n=4096)
    return refused and fine == 50, (f"n=64 {'refused' if refused else 'ALIASED'}, "
                                    f"n=4096 gives {fine}")


def check_guard_refuses_only_the_undecidable():
    """Refusing everything would pass the test above, so measure where the guard stops.

    The clearance guard has to be sharp in both directions: silent about points
    that are merely close to the contour, loud about points too close for the
    sampling to place. At n = 2048 on the unit circle it answers correctly down
    to a clearance of 1e-6 and refuses at 1e-7, and the refusal is the honest
    verdict -- the chords deviate from the circle by about 1.2e-6 there, so at
    1e-7 the code genuinely does not know which side the point is on.
    """
    C = contour.circle()
    direction = np.exp(1j * 0.7)
    answered, refused = [], []
    for d in (1e-1, 1e-3, 1e-5, 1e-6, 1e-7, 1e-9):
        for inside, radius in ((1, 1.0 - d), (0, 1.0 + d)):
            try:
                ok = winding.winding_number(C, radius * direction) == inside
                answered.append((d, ok))
            except ValueError:
                refused.append(d)
    all_answers_correct = all(ok for _, ok in answered)
    resolved = {d for d, _ in answered}
    ok = all_answers_correct and resolved == {1e-1, 1e-3, 1e-5, 1e-6} and set(refused) == {1e-7, 1e-9}
    return ok, (f"correct at clearances {sorted(resolved, reverse=True)}, "
                f"refused at {sorted(set(refused), reverse=True)}, "
                f"{sum(not ok for _, ok in answered)} wrong answers")


CHECKS = [
    check_circle_winds_k_times,
    check_star_interior_matches_its_radial_graph,
    check_figure_eight_lobes_are_opposite,
    check_two_algorithms_agree_everywhere,
    check_invariance_under_rigid_motion,
    check_polygon_vertices_are_enough,
    check_point_on_the_path_is_refused,
    check_guard_refuses_only_the_undecidable,
    check_undersampling_is_refused_not_aliased,
]


def main():
    return run("Rung 2: winding numbers (geometry only)", CHECKS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
