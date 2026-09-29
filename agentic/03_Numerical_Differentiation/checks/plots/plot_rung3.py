"""Visual diagnostics for every check in checks/rung3_poles.py.

    python checks/plots/plot_rung3.py [--out DIR] [--dark]

rung3_poles.py answers "does this pass?". This answers "do I believe it?"

Where a figure shows the plane, it shows log|f| computed as -log|1/f| -- the
same quantity, but evaluated through the reciprocal so that nothing overflows
and no nan reaches the colour scale. Magnitude is a one-hue sequential ramp;
the found poles are marked in a categorical colour so they read as annotation
rather than as part of the field.

One check has no figure: the sympy cross-check produces a table of locations and
orders, and a table is what it should stay.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gamma

import poles
from checks.plots._style import (LIGHT, DARK, dots, finish, finish_panels,
                                 new_figure, new_panels)

HERE = pathlib.Path(__file__).resolve().parent
EPS = float(np.finfo(float).eps)

# The documented blue sequential ramp, steps 100 -> 700, light to dark.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def magnitude_field(f, region, resolution=320):
    """log10|f| over a grid, evaluated as -log10|1/f| so nothing overflows.

    |f| runs to overflow near a pole and to nan on top of one. Its reciprocal
    goes quietly to zero instead, so the logarithm is taken of a small bounded
    number and negated. The only special value left is an exact zero of 1/f,
    which is the pole itself; those pixels come back as +inf and are clipped by
    the colour limits rather than poisoning the scale.
    """
    x0, x1, y0, y1 = region
    re = np.linspace(x0, x1, resolution)
    im = np.linspace(y0, y1, resolution)
    with np.errstate(all="ignore"):
        reciprocal_magnitude = np.abs(poles.reciprocal(f)(re[None, :] + 1j * im[:, None]))
        field = -np.log10(reciprocal_magnitude)
    return field


def draw_field(ax, T, f, region, resolution=320, vspan=6.0):
    """Draw log10|f| as a one-hue sequential field, dark where |f| is large."""
    field = magnitude_field(f, region, resolution)
    finite = field[np.isfinite(field)]
    high = np.percentile(finite, 99.5)
    ramp = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    ax.imshow(field, extent=region, origin="lower", cmap=ramp, aspect="equal",
              vmin=high - vspan, vmax=high, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return field


def mark(ax, T, found, label_orders=True):
    """Mark located poles, annotated with their order."""
    if not found:
        return
    x = [q.location.real for q in found]
    y = [q.location.imag for q in found]
    ax.plot(x, y, "o", markerfacecolor="none", markeredgecolor=T["series"][1],
            markeredgewidth=2.0, markersize=14, linestyle="none")
    if label_orders:
        for q in found:
            ax.annotate(f"m={q.order}", (q.location.real, q.location.imag),
                        textcoords="offset points", xytext=(10, 7),
                        color=T["series"][1], fontsize=10, weight="bold")


# ----------------------------------------------------------------------------

def plot_1_rational(T, out):
    """What the finder sees, and what it reports, for a rational function."""
    spec = [(0.3 - 0.7j, 1), (-1.1 + 0.4j, 2), (0.9 + 1.2j, 3)]
    f = lambda z: np.exp(z) / ((z - spec[0][0]) * (z - spec[1][0]) ** 2 * (z - spec[2][0]) ** 3)
    region = (-2.0, 2.0, -2.0, 2.0)
    fig, ax = new_figure(T, size=(6.6, 6.4), grid=False)
    draw_field(ax, T, f, region)
    found = poles.find_poles(f, region)
    mark(ax, T, found)
    for want, order in spec:
        ax.plot(want.real, want.imag, ".", color=T["surface"], markersize=7)
    worst = max(min(abs(q.location - want) for q in found) for want, _ in spec)
    orders_ok = all(min(found, key=lambda q: abs(q.location - w)).order == m for w, m in spec)
    ok = len(found) == 3 and worst < 1e-12 and orders_ok
    return finish(fig, ax, T, "Three poles, three orders",
                  r"$f(z) = e^{z}\,/\,[(z-p_1)(z-p_2)^2(z-p_3)^3]$   ·   field is $\log_{10}|f|$",
                  "pale dots are the true poles, rings what the finder returned from $f$ as a callable",
                  ok, f"worst position {worst:.0e}", out / "check_1_rational.png")


def plot_2_tan_and_its_zeros(T, out):
    """tan's poles are found; its zeros, interleaved with them, are not reported."""
    region = (-5.0, 5.0, -1.6, 1.6)
    fig, ax = new_figure(T, size=(9.0, 3.9), grid=False)
    draw_field(ax, T, np.tan, region, resolution=420)
    found = poles.find_poles(np.tan, (-5.0, 5.0, -1.0, 1.0))
    mark(ax, T, found)
    for k in (-1, 0, 1):
        ax.plot(k * np.pi, 0.0, "x", color=T["surface"], markersize=9, markeredgewidth=2.0)
    ax.annotate("zeros of $\\tan$, correctly ignored", (np.pi, 0.0),
                textcoords="offset points", xytext=(6, -26), color=T["ink2"], fontsize=9)
    expected = [(k + 0.5) * np.pi for k in (-2, -1, 0, 1)]
    worst = max(min(abs(q.location - w) for q in found) for w in expected)
    ok = len(found) == 4 and worst < 1e-12 and {q.order for q in found} == {1}
    return finish(fig, ax, T, "Poles kept, zeros discarded",
                  r"$f(z) = \tan z$   ·   poles at $(k+\frac{1}{2})\pi$, zeros at $k\pi$, interleaved",
                  "zeros of $f$ are poles of $1/f$, so a sign slip anywhere would swap the two sets",
                  ok, f"4 found, worst {worst:.0e}", out / "check_2_tan.png")


def plot_3_gamma(T, out):
    """Gamma's poles march off to the left, and 1/Gamma is entire."""
    region = (-3.9, 2.0, -1.6, 1.6)
    fig, axes = new_panels(T, 2, (11.0, 4.5))
    draw_field(axes[0], T, gamma, region, resolution=380, vspan=3.0)
    found = poles.find_poles(gamma, (-3.5, 1.5, -1.0, 1.0))
    mark(axes[0], T, found)
    axes[0].set_xlabel(r"$\log_{10}|\Gamma(z)|$", color=T["ink2"], fontsize=10)
    draw_field(axes[1], T, lambda z: 1.0 / gamma(z), region, resolution=380, vspan=3.0)
    axes[1].set_xlabel(r"$\log_{10}|1/\Gamma(z)|$ — entire, with simple zeros", color=T["ink2"],
                       fontsize=10)
    expected = [0.0, -1.0, -2.0, -3.0]
    worst = max(min(abs(q.location - w) for q in found) for w in expected)
    ok = len(found) == 4 and worst < 1e-11 and {q.order for q in found} == {1}
    return finish_panels(fig, T, "A black box with no closed form",
                         r"$\Gamma(z)$ from scipy   ·   simple poles at the non-positive integers",
                         "the right panel is the quantity the finder actually works with, and it is perfectly tame",
                         ok, f"4 found, worst {worst:.0e}", out / "check_3_gamma.png", top=0.80)


def plot_4_entire_functions(T, out):
    """The negative control: nothing analytic may produce a pole."""
    cases = [("$e^{z}$", np.exp), (r"$\sin z$", np.sin), (r"$\sinh z$", np.sinh),
             ("$z^4 - 3z + 1$", lambda z: z ** 4 - 3 * z + 1)]
    region = (-3.0, 3.0, -3.0, 3.0)
    fig, axes = new_panels(T, len(cases), (11.0, 4.2))
    total = 0
    for ax, (name, f) in zip(axes, cases):
        draw_field(ax, T, f, region, resolution=200)
        found = poles.find_poles(f, region)
        total += len(found)
        mark(ax, T, found)
        ax.set_xlabel(name, color=T["ink2"], fontsize=11)
    return finish_panels(fig, T, "Nothing to find, and nothing found",
                         r"four entire functions on $[-3,3]^2$   ·   field is $\log_{10}|f|$",
                         "|f| reaches $10^{13}$ at the edges here, which is what a naive peak-finder would seize on",
                         total == 0, f"{total} spurious poles",
                         out / "check_4_entire_functions.png", top=0.78)


def plot_5_order_fits(T, out):
    """How the order is measured, and how each impostor fails the measurement.

    Every curve is mean log|1/f| around circles of radius r. A pole of order m
    gives a straight line of slope m. The two impostors are the whole point:
    1/sqrt(z) is perfectly straight with slope 1/2, so only the non-integer
    exponent gives it away, while log(z) is not straight at all.
    """
    genuine = [("$1/z$", lambda z: 1.0 / z), ("$1/z^2$", lambda z: 1.0 / z ** 2),
               ("$1/z^4$", lambda z: 1.0 / z ** 4)]
    impostor = [(r"$1/\sqrt{z}$", lambda z: 1.0 / np.sqrt(z)), (r"$\log z$", np.log)]
    radii = np.logspace(-6, -1, 40)
    directions = np.exp(2j * np.pi * np.arange(24) / 24)

    fig, ax = new_figure(T, size=(7.4, 5.0))
    for group, colour, label, nudges in (
            (genuine, T["series"][0], "pole: whole-number slope", (0, 0, 0)),
            (impostor, T["series"][1], "not a pole", (9, -9))):
        for i, (name, f) in enumerate(group):
            g = poles.reciprocal(f)
            with np.errstate(all="ignore"):
                rings = np.array([np.mean(np.log(np.abs(g(r * directions)))) for r in radii])
            ax.plot(radii, rings, "-", color=colour, linewidth=1.8,
                    label=label if i == 0 else None)
            # The two impostor curves end within half a unit of each other, so
            # their labels are nudged apart rather than left to overlap.
            ax.annotate(name, (radii[-1], rings[-1]), textcoords="offset points",
                        xytext=(8, nudges[i]), color=colour, fontsize=10, va="center")
    ax.set_xscale("log")
    ax.set_xlim(5e-7, 4e-1)
    ax.set_xlabel("radius $r$ about the singularity", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"mean $\log|1/f|$ on the circle", color=T["ink2"], fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    slopes = {}
    for name, f in genuine + impostor:
        slopes[name] = poles.measure_order(f, 0.0, scale=1.0)[0]
    ok = (all(abs(slopes[n] - round(slopes[n])) < 1e-6 for n, _ in genuine)
          and abs(slopes[r"$1/\sqrt{z}$"] - 0.5) < 1e-6)
    return finish(fig, ax, T, "The order is a slope",
                  r"mean $\log|1/f|$ on circles of radius $r$   ·   slope is the order",
                  r"$1/\sqrt{z}$ and $\log z$ are straight enough to pass the residual test; their slopes are what expose them",
                  ok, "slopes 1, 2, 4 vs 0.5", out / "check_5_order_fits.png")


def plot_6_accuracy_law(T, out):
    """Locating an order-m pole costs m-fold precision, and the law is sharp.

    A prediction from the arithmetic, not a fitted tolerance: near an order-m
    pole the reciprocal behaves like (z-p)^m, so a relative perturbation eps in
    its computed value moves the apparent root by eps^(1/m). Nobody can do
    better than 1e-4 for a quadruple pole this way, in double precision.
    """
    p = 0.3 - 0.7j
    orders = np.arange(1, 7)
    expanded_error, factored_error = [], []
    for m in orders:
        coefficients = np.poly([p] * m)
        expanded = lambda z, c=coefficients: (z + 2.0) / np.polyval(c, z)
        factored = lambda z, m=m: (z + 2.0) / (z - p) ** m
        start = p + 0.05
        expanded_error.append(abs(poles.muller(poles.reciprocal(expanded), start,
                                               start + 1e-3, start - 1e-3) - p))
        factored_error.append(abs(poles.muller(poles.reciprocal(factored), start,
                                               start + 1e-3, start - 1e-3) - p))
    predicted = EPS ** (1.0 / orders)
    expanded_error = np.array(expanded_error)
    factored_error = np.maximum(np.array(factored_error), 1e-17)

    fig, ax = new_figure(T, size=(7.4, 4.8))
    ax.plot(orders, predicted, "-", color=T["muted"], linewidth=1.6)
    ax.annotate(r"$\varepsilon^{1/m}$", (orders[-1], predicted[-1]), textcoords="offset points",
                xytext=(10, 0), color=T["muted"], fontsize=10, va="center")
    ax.plot(orders, np.maximum(expanded_error, 1e-17), "-", color=T["series"][1], linewidth=1.8,
            label="denominator given expanded")
    dots(ax, orders, np.maximum(expanded_error, 1e-17), T["series"][1], T, ms=7)
    ax.plot(orders, factored_error, "-", color=T["series"][0], linewidth=1.8,
            label=r"denominator given as $(z-p)^m$")
    dots(ax, orders, factored_error, T["series"][0], T, ms=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-1)
    ax.set_xticks(orders)
    ax.set_xlabel("pole order $m$", color=T["ink2"], fontsize=10)
    ax.set_ylabel("|error| in the located position", color=T["ink2"], fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    ratios = expanded_error[1:] / predicted[1:]
    ok = np.all(ratios < 5.0) and np.all(ratios > 0.2) and factored_error.max() < 1e-10
    return finish(fig, ax, T, "Half the digits, m times over",
                  r"$f(z) = (z+2)/(z-p)^m$, $p = 0.3 - 0.7i$   ·   located by Muller on $1/f$",
                  "the same pole, written two ways: cancellation in the expanded form sets the floor",
                  ok, f"ratio to law {ratios.min():.2f}-{ratios.max():.2f}",
                  out / "check_6_accuracy_law.png")


def plot_7_close_poles(T, out):
    """Two poles a distance d apart, resolved over seven decades of d."""
    centre = 0.2 + 0.1j
    def two_poles(d):
        return lambda z: 1.0 / ((z - centre - d / 2) * (z - centre + d / 2))

    separations = np.logspace(-1, -7, 13)
    relative = []
    for d in separations:
        half = 50 * d
        found = poles.find_poles(two_poles(d),
                                 (centre.real - half, centre.real + half,
                                  centre.imag - half, centre.imag + half),
                                 resolution=401, merge=1e-3)
        located = sorted(q.location.real for q in found)
        relative.append(abs((located[1] - located[0]) - d) / d if len(found) == 2 else np.nan)
    relative = np.array(relative)

    fig, ax = new_figure(T, size=(7.4, 4.4))
    ax.plot(separations, np.maximum(relative, 1e-17), "-", color=T["series"][0], linewidth=1.8)
    dots(ax, separations, np.maximum(relative, 1e-17), T["series"][0], T, ms=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1e-17, 1e-6)
    ax.set_xlabel("true separation $d$ between the two simple poles", color=T["ink2"], fontsize=10)
    ax.set_ylabel("relative error in the recovered separation", color=T["ink2"], fontsize=10)

    _, bent = poles.measure_order(two_poles(1e-3), centre + 5e-4, scale=1e-2)
    ax.text(0.03, 0.9, f"on a grid too coarse to separate them the log-log fit bends\n"
                       f"(residual {bent:.2f}) and the finder reports nothing at all",
            transform=ax.transAxes, color=T["muted"], fontsize=9, va="top")
    ok = np.all(np.isfinite(relative)) and np.nanmax(relative) < 1e-9
    return finish(fig, ax, T, "Two poles stay two, over seven decades",
                  r"$f(z) = 1/[(z - c - d/2)(z - c + d/2)]$   ·   region scaled so the grid resolves $d$",
                  "the failure to guard against is merging the pair, so the separation is checked, not the count",
                  ok, f"worst relative {np.nanmax(relative):.0e}",
                  out / "check_7_close_poles.png")


FIGURES = [
    plot_1_rational,
    plot_2_tan_and_its_zeros,
    plot_3_gamma,
    plot_4_entire_functions,
    plot_5_order_fits,
    plot_6_accuracy_law,
    plot_7_close_poles,
]


def main():
    parser = argparse.ArgumentParser(description="Rung 3 visual diagnostics.")
    parser.add_argument("--out", default=None, help="output directory (default: checks/plots/rung3)")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    T = DARK if args.dark else LIGHT
    out = pathlib.Path(args.out) if args.out else HERE / "rung3"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rung 3 figures -> {out}  ({'dark' if args.dark else 'light'})")
    failures = sum(not figure(T, out) for figure in FIGURES)
    print(f"{len(FIGURES) - failures}/{len(FIGURES)} figures pass.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
