"""Visual diagnostics for every check in checks/rung2_winding.py.

    python checks/plots/plot_rung2.py [--out DIR] [--dark]

rung2_winding.py answers "does this pass?". This answers "do I believe it?"
Most rung-2 claims are about regions of the plane rather than single numbers, so
most of these figures colour the plane by winding number and let the regions
speak. Winding is signed with a meaningful zero, so it takes a diverging scale,
and every region is also labelled with its integer -- colour never carries the
value alone.

Every figure's PASS/FAIL badge is computed from the numbers actually plotted.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage

import contour
import winding
from checks.plots._style import (LIGHT, DARK, dots, finish, finish_panels,
                                 new_figure, new_panels)

HERE = pathlib.Path(__file__).resolve().parent


def winding_map(ax, T, path, extent, resolution=260, n=900, algorithm=winding.winding_grid,
                label=True, draw_path=True):
    """Colour the plane by winding number and label each region with its integer.

    Returns the integer grid so callers can compute their own verdict from
    exactly the numbers on screen. Regions too thin to hold a label are left
    unlabelled rather than having text spill over their edges.
    """
    x0, x1, y0, y1 = extent
    re = np.linspace(x0, x1, resolution)
    im = np.linspace(y0, y1, resolution)
    grid = algorithm(path, re, im, n=n)

    image = np.tile(to_rgb(T["muted"]), grid.shape + (1,))
    for value in range(-3, 4):
        image[grid == value] = to_rgb(T["diverging"][value + 3])
    image[grid == -99] = to_rgb(T["ink"])
    ax.imshow(image, extent=extent, origin="lower", interpolation="nearest", aspect="equal")

    if draw_path:
        curve = path.sample(2000) if hasattr(path, "sample") else np.asarray(path)
        ax.plot(curve.real, curve.imag, "-", color=T["ink"], linewidth=1.4)
    if label:
        for value in np.unique(grid):
            if value == -99:
                continue
            # The most interior pixel of the region, so a label never straddles
            # a boundary or falls in a sliver. A margin is excluded first: the
            # unbounded exterior region is "deepest" in a corner of the image,
            # where the text would hang off the edge.
            depth = ndimage.distance_transform_edt(grid == value)
            margin = int(0.09 * resolution)
            depth[:margin, :] = depth[-margin:, :] = 0
            depth[:, :margin] = depth[:, -margin:] = 0
            row, col = np.unravel_index(np.argmax(depth), depth.shape)
            if depth[row, col] < 0.035 * resolution:
                continue
            ink = T["ink"] if abs(value) <= 1 else T["surface"]
            ax.text(re[col], im[row], f"{value:+d}", color=ink, fontsize=13,
                    weight="bold", ha="center", va="center")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():   # the image is its own frame
        spine.set_visible(False)
    return grid


# ----------------------------------------------------------------------------
# One figure per check, in the order the checks run
# ----------------------------------------------------------------------------

def plot_1_circle_winds_k(T, out):
    """A circle traversed k times paints its whole interior with the value k."""
    ks = (-2, 1, 2, 3)
    fig, axes = new_panels(T, len(ks), (11.0, 3.9))
    exact = True
    for ax, k in zip(axes, ks):
        C = contour.circle(winding=k)
        grid = winding_map(ax, T, C, (-1.6, 1.6, -1.6, 1.6), resolution=200)
        ax.set_xlabel(f"$k = {k:+d}$", color=T["ink2"], fontsize=10)
        centre = grid[grid.shape[0] // 2, grid.shape[1] // 2]
        exact = exact and centre == k and winding.winding_number(C, 0) == k
    return finish_panels(fig, T, "Traversing k times winds k times",
                         r"unit circle, winding $k$   ·   plane coloured by $n(C, p)$",
                         "the interior is a single constant value, and its sign follows the direction of travel",
                         exact, "all four exact", out / "check_1_circle_winds_k.png", top=0.80)


def plot_2_star_interior(T, out):
    """The computed interior coincides with the curve's analytic radial graph.

    The convex hull is drawn because the interesting points are the ones between
    the hull and the curve: inside the hull, outside the curve, and correctly
    given winding zero. A routine that filled the hull would be wrong there and
    right everywhere else.
    """
    amplitude = 0.5
    C = contour.star(mean_radius=1.0, amplitude=amplitude, lobes=3)
    extent = (-1.8, 1.8, -1.8, 1.8)
    fig, ax = new_figure(T, size=(6.6, 6.6), grid=False)
    grid = winding_map(ax, T, C, extent, resolution=300)

    re = np.linspace(extent[0], extent[1], 300)
    im = np.linspace(extent[2], extent[3], 300)
    p = re[None, :] + 1j * im[:, None]
    radius = 1.0 + amplitude * np.cos(3 * np.angle(p))
    expected = (np.abs(p) < radius).astype(int)
    live = grid != -99
    # A pixel straddling the boundary has no single right answer, so the verdict
    # counts only pixels the grid can actually place. The sub-pixel cases are
    # reported rather than hidden.
    pixel = (extent[1] - extent[0]) / (grid.shape[1] - 1)
    resolvable = live & (np.abs(np.abs(p) - radius) > pixel)
    wrong = int(np.sum(grid[resolvable] != expected[resolvable]))
    sub_pixel = int(np.sum((grid != expected) & live)) - wrong

    from scipy.spatial import ConvexHull
    pts = C.sample(2000)
    hull = ConvexHull(np.column_stack([pts.real, pts.imag]))
    loop = np.append(hull.vertices, hull.vertices[0])
    ax.plot(pts.real[loop], pts.imag[loop], "--", color=T["muted"], linewidth=1.3)
    ax.text(0.98, 0.02, "dashed: convex hull", transform=ax.transAxes, color=T["muted"],
            fontsize=9, ha="right", va="bottom")
    notch = int(np.sum((expected == 0) & (np.abs(p) < 1.0) & live))

    return finish(fig, ax, T, "The interior is the curve, not its hull",
                  r"$r(\theta) = 1 + 0.5\cos 3\theta$   ·   compared with the exact test $|p| < r(\arg p)$",
                  f"{notch} pixels lie inside the hull but outside the curve, and all are given $0$; "
                  f"{sub_pixel} boundary pixels are unresolvable and excluded",
                  wrong == 0, f"{wrong} of {int(resolvable.sum())} disagree",
                  out / "check_2_star_interior.png")


def plot_3_figure_eight(T, out):
    """The two lobes wind in opposite senses -- the decisive test of a signed count."""
    F = contour.figure_eight()
    fig, ax = new_figure(T, size=(7.0, 5.0), grid=False)
    grid = winding_map(ax, F, None, None) if False else winding_map(
        ax, T, F, (-1.5, 1.5, -1.1, 1.1), resolution=300)
    right = winding.winding_number(F, 0.6)
    left = winding.winding_number(F, -0.6)
    ok = (right, left) == (-1, 1)
    return finish(fig, ax, T, "Two lobes, opposite signs",
                  r"$\gamma(t) = \sin\theta\,(1 + i\cos\theta)$, $\theta = 2\pi t$   ·   one closed path, traversed once",
                  "an inside/outside test would call both lobes +1; the signed count does not",
                  ok, f"lobes {right:+d} and {left:+d}",
                  out / "check_3_figure_eight.png")


def plot_4_algorithms_agree(T, out):
    """Accumulated argument and signed ray crossings, over the whole plane."""
    L = contour.limacon()
    extent = (-1.05, 1.75, -1.3, 1.3)
    fig, axes = new_panels(T, 2, (10.4, 5.6))
    a = winding_map(axes[0], T, L, extent, resolution=260, algorithm=winding.winding_grid)
    b = winding_map(axes[1], T, L, extent, resolution=260, algorithm=winding.crossing_grid)
    axes[0].set_xlabel("accumulated argument", color=T["ink2"], fontsize=10)
    axes[1].set_xlabel("signed ray crossings", color=T["ink2"], fontsize=10)
    live = a != -99
    differ = int(np.sum(a[live] != b[live]))
    return finish_panels(fig, T, "Two algorithms with nothing in common",
                         r"limaçon $r(\theta) = 0.5 + \cos\theta$   ·   its inner loop is wound twice",
                         "one is trigonometric and branch-sensitive, the other is sign tests and a linear interpolation",
                         differ == 0, f"{differ} of {int(live.sum())} points differ",
                         out / "check_4_algorithms_agree.png", top=0.84)


def plot_5_rigid_motion(T, out):
    """Moving, rotating and scaling the whole configuration changes nothing.

    Plotted as the configurations themselves rather than as maps: the point is
    that the same four probes keep the same four answers while the picture is
    rotated and rescaled underneath them.
    """
    base = winding.as_closed_polyline(contour.star(amplitude=0.5, lobes=3), 2048)
    probes = np.array([0.2 + 0.1j, -0.3 + 0.4j, 1.2 - 0.9j, 0.95 * np.exp(1j * np.pi / 3)])
    motions = [(0j, 0.0, 1.0), (-11 + 5j, 2.31, 4.0), (0.5j, 5.0, 0.01)]
    names = ["as built", "rotated 2.31 rad, scaled 4x", "rotated 5.0 rad, scaled 0.01x"]

    fig, axes = new_panels(T, 3, (11.0, 4.2), grid=True)
    reference, changed = None, 0
    for ax, (shift, angle, scale), name in zip(axes, motions, names):
        w = scale * np.exp(1j * angle)
        moved, moved_probes = shift + w * base, shift + w * probes
        ax.plot(moved.real, moved.imag, "-", color=T["ink"], linewidth=1.3)
        answers = [winding.winding_number(moved, q) for q in moved_probes]
        for q, value in zip(moved_probes, answers):
            colour = T["diverging"][value + 3]
            dots(ax, [q.real], [q.imag], colour, T, ms=10)
            ax.annotate(f"{value:+d}", (q.real, q.imag), textcoords="offset points",
                        xytext=(9, 6), color=T["ink2"], fontsize=10, weight="bold")
        if reference is None:
            reference = answers
        else:
            changed += answers != reference
        ax.set_aspect("equal")
        ax.set_xlabel(name, color=T["ink2"], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    return finish_panels(fig, T, "Rigid motions leave the answer alone",
                         r"star $r(\theta) = 1 + 0.5\cos 3\theta$   ·   four probes carried along with the curve",
                         "the ray-crossing algorithm singles out the $+x$ direction, so a rotation is a real test of it",
                         changed == 0, f"{changed} of 2 motions changed an answer",
                         out / "check_5_rigid_motion.png", top=0.82)


def plot_6_polygon_vertices(T, out):
    """A polygon needs no resampling: five vertices give the exact map."""
    V = [1.4 + 0.3j, -0.2 + 1.5j, -1.3 - 0.1j, 0.1 - 1.2j, 0.7 - 0.5j]
    P = contour.Polygon(V)
    dense = np.append(np.concatenate([np.linspace(a, b, 500, endpoint=False)
                                      for a, b in P.edges()]), V[0])
    extent = (-1.9, 2.0, -1.7, 2.0)
    fig, ax = new_figure(T, size=(6.6, 6.4), grid=False)
    coarse = winding_map(ax, T, np.array(V, dtype=complex), extent, resolution=280)
    fine = winding.winding_grid(dense, np.linspace(extent[0], extent[1], 280),
                                np.linspace(extent[2], extent[3], 280), n=2048)
    live = (coarse != -99) & (fine != -99)
    differ = int(np.sum(coarse[live] != fine[live]))
    vertices = np.array(V)
    dots(ax, vertices.real, vertices.imag, T["series"][1], T, ms=8)
    return finish(fig, ax, T, "Five vertices are the whole story",
                  r"$f$ irrelevant   ·   pentagon from its bare vertex list, versus 2500 resampled points",
                  "a straight edge subtends less than $\\pi$ from any point off it, so the branch can never go wrong",
                  differ == 0, f"{differ} of {int(live.sum())} differ",
                  out / "check_6_polygon_vertices.png")


def plot_7_guard_resolution(T, out):
    """Where the guard stops answering, and why that is the honest place to stop.

    The chords of a 2048-gon deviate from the unit circle by about 1.2e-6, so a
    point 1e-7 away from the circle is not placed by the sampling at all. The
    guard answers correctly down to 1e-6 and refuses below it, which is exactly
    where the geometry stops determining an answer.
    """
    C = contour.circle()
    direction = np.exp(1j * 0.7)
    clearances = np.logspace(-1, -9, 17)
    rows = {"inside": [], "outside": [], "refused": []}
    wrong = 0
    for d in clearances:
        for name, radius, expected in (("inside", 1.0 - d, 1), ("outside", 1.0 + d, 0)):
            try:
                got = winding.winding_number(C, radius * direction)
                rows[name].append(d)
                wrong += got != expected
            except ValueError:
                rows["refused"].append(d)

    fig, ax = new_figure(T, size=(7.6, 4.2))
    # The row already says inside / outside / refused, so colour must not
    # re-encode it. The real distinction is answered versus refused, and the
    # neutral diverging step is nearly invisible on this surface anyway.
    levels = {"inside": 2, "outside": 1, "refused": 0}
    for name, level in levels.items():
        if not rows[name]:
            continue
        answered = name != "refused"
        dots(ax, rows[name], [level] * len(rows[name]),
             T["series"][0] if answered else T["muted"], T,
             label=("answered, and correct" if name == "inside" else
                    None if answered else "refused"), ms=9)
    sagitta = (1 - np.cos(np.pi / 2047))
    ax.axvline(sagitta, color=T["ink2"], linewidth=1.2)
    ax.text(sagitta * 1.3, 2.35, "  chord sagitta\n  at $n = 2048$", color=T["ink2"], fontsize=9,
            va="top")
    ax.set_xscale("log")
    ax.set_xlim(2e-10, 1)
    ax.set_ylim(-0.5, 2.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["refused", "outside", "inside"])
    ax.invert_xaxis()
    ax.set_xlabel("clearance between the point and the circle", color=T["ink2"], fontsize=10)
    ax.legend(loc="lower left", fontsize=9)

    ok = wrong == 0 and min(rows["inside"]) > max(rows["refused"])
    return finish(fig, ax, T, "The guard stops exactly where the geometry does",
                  r"$|z| = 1$, $n = 2048$   ·   probes placed at a known clearance inside and outside",
                  "refusing everything would also pass the on-path test, so the boundary is what matters",
                  ok, f"{wrong} wrong answers", out / "check_7_guard_resolution.png")


def plot_8_aliasing(T, out):
    """Under-sampling does not degrade gracefully -- it returns confident integers.

    A circle wound 50 times needs more than 100 samples before a step is smaller
    than pi. Below that the principal branch folds each step and the accumulated
    total lands on a clean, completely wrong integer: at n = 64 it is exactly
    -13. No near-integer test can catch that, which is why the guard compares
    two sampling densities instead.
    """
    C = contour.circle(winding=50)
    # Probed at the centre, where the turn per step is uniform and the threshold
    # is exact: a step equals pi when 50 * 2 pi / (n - 1) = pi, i.e. at n = 101.
    # Seen from anywhere else the turn rate varies by (1+r)/(1-r) and the
    # threshold blurs, which would muddle the one crisp number in this figure.
    ns = sorted(set(np.round(np.logspace(np.log10(8), np.log10(4096), 40)).astype(int).tolist())
                | {101})
    resolved_n, resolved, aliased_n, aliased, refused_n = [], [], [], [], []
    for n in ns:
        try:
            total = winding.turning(C, 0, n=int(n))
        except ValueError:
            # Not a failure to handle away: at these n two consecutive samples
            # are antipodal, so the chord between them runs through the centre
            # and the winding number there is genuinely undefined.
            refused_n.append(n)
            continue
        # Rounded, not exact: a correct 141-term sum lands on 50 to within 1e-14,
        # and `total == 50` would file that as an aliasing failure.
        correct = round(total) == 50
        (resolved_n if correct else aliased_n).append(n)
        (resolved if correct else aliased).append(total)

    fig, ax = new_figure(T, size=(7.6, 4.4))
    ax.axhline(50, color=T["muted"], linewidth=1.2)
    ax.text(4096, 50, "true value 50  ", color=T["muted"], fontsize=9, ha="right", va="bottom")
    ax.plot(aliased_n + resolved_n, aliased + resolved, "-", color=T["axis"],
            linewidth=1.2, zorder=1)
    dots(ax, aliased_n, aliased, T["diverging"][6], T, label="aliased", ms=7)
    dots(ax, resolved_n, resolved, T["diverging"][1], T, label="resolved", ms=7)
    if refused_n:
        dots(ax, refused_n, [0] * len(refused_n), T["muted"], T,
             label="refused (chord through the centre)", ms=9)
    threshold = 101
    ax.axvline(threshold, color=T["ink2"], linewidth=1.2)
    ax.text(threshold * 0.88, 28, "step first drops\nbelow $\\pi$ at $n = 101$  ",
            color=T["ink2"], fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("samples $n$", color=T["ink2"], fontsize=10)
    ax.set_ylabel("accumulated turning / $2\\pi$", color=T["ink2"], fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    every_value_is_an_integer = all(abs(v - round(v)) < 1e-6 for v in aliased + resolved)
    ok = (every_value_is_an_integer
          and all(n < threshold for n in aliased_n)
          and all(n > threshold for n in resolved_n)
          and len(aliased_n) > 0)
    return finish(fig, ax, T, "Confident, clean, wrong integers",
                  r"$|z| = 1$ traversed 50 times   ·   turning accumulated about an interior point",
                  "every point plotted is an exact integer, so a near-integer test would pass all of them",
                  ok, f"{len(aliased_n)} wrong, {len(resolved_n)} exact",
                  out / "check_8_aliasing.png")


FIGURES = [
    plot_1_circle_winds_k,
    plot_2_star_interior,
    plot_3_figure_eight,
    plot_4_algorithms_agree,
    plot_5_rigid_motion,
    plot_6_polygon_vertices,
    plot_7_guard_resolution,
    plot_8_aliasing,
]


def main():
    parser = argparse.ArgumentParser(description="Rung 2 visual diagnostics.")
    parser.add_argument("--out", default=None, help="output directory (default: checks/plots/rung2)")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    T = DARK if args.dark else LIGHT
    out = pathlib.Path(args.out) if args.out else HERE / "rung2"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rung 2 figures -> {out}  ({'dark' if args.dark else 'light'})")
    failures = sum(not figure(T, out) for figure in FIGURES)
    print(f"{len(FIGURES) - failures}/{len(FIGURES)} figures pass.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
