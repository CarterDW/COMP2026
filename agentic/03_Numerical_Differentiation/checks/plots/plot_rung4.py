"""Visual diagnostics for every check in checks/rung4_residues.py.

    python checks/plots/plot_rung4.py [--out DIR] [--dark]

rung4_residues.py answers "does this pass?". This answers "do I believe it?"

The last figure shows the method the project is not allowed to use. Cauchy's
integral formula computes these same residues to machine precision at every
pole order, and it is off limits because it is a contour integral -- building
the right-hand side on it would make the rung-5 comparison circular. Seeing how
much accuracy that costs is worth a figure.
"""

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import sympy
from scipy.special import gamma

import poles
import residues
from checks.plots._style import (LIGHT, DARK, dots, finish, finish_panels,
                                 new_figure, new_panels)

HERE = pathlib.Path(__file__).resolve().parent
EPS = residues.EPS

Z = sympy.Symbol("z")
P_EXACT = sympy.Rational(3, 10) - sympy.Rational(7, 10) * sympy.I
Q_EXACT = sympy.Rational(-6, 5) + sympy.Rational(1, 2) * sympy.I
P, Q = complex(P_EXACT), complex(Q_EXACT)


def exact_residue(order):
    regular = sympy.exp(Z) / (Z - Q_EXACT)
    return complex(sympy.diff(regular, Z, order - 1).subs(Z, P_EXACT) / math.factorial(order - 1))


def test_function(order):
    return lambda z: np.exp(z) / ((z - P) ** order * (z - Q))


def error_curve(order, half, grid):
    """|error| in the residue against step size, for one pole order and stencil."""
    truth = exact_residue(order)
    out = []
    for delta in grid:
        try:
            out.append(abs(residues.residue(test_function(order), P, order,
                                            half=half, delta=delta) - truth))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


# ----------------------------------------------------------------------------

def plot_1_the_tradeoff(T, out):
    """The V curve: truncation falling, roundoff rising, and a best step between them.

    The signature plot of numerical differentiation. The left arm is
    interpolation error shrinking with the step; the right arm is roundoff in h
    being amplified by the division by delta^(m-1), which is why order 1 has no
    right arm at all -- it divides by nothing.
    """
    grid = np.logspace(-0.3, -6, 260)
    fig, ax = new_figure(T, size=(7.6, 5.0))
    for colour, order in zip(T["series"], (1, 3, 5)):
        errors = error_curve(order, 5, grid)
        ax.plot(grid, np.maximum(errors, 1e-18), "-", color=colour, linewidth=1.8,
                label=f"pole of order {order}")
    predicted = EPS ** (1.0 / 10)
    ax.axvline(predicted, color=T["ink2"], linewidth=1.2)
    ax.annotate(r"$\varepsilon^{1/2n}$", xy=(predicted, 2e-3), xytext=(7, 0),
                textcoords="offset points", color=T["ink2"], fontsize=11, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1e-18, 1e2)
    ax.set_xlabel(r"step size $\delta$", color=T["ink2"], fontsize=10)
    ax.set_ylabel("|error| in the residue", color=T["ink2"], fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

    minima = [grid[int(np.nanargmin(error_curve(m, 5, grid)))] for m in (3, 5)]
    ok = all(0.2 < d / predicted < 5.0 for d in minima)
    return finish(fig, ax, T, "Truncation down, roundoff up",
                  r"$f(z) = e^{z}/[(z-p)^m(z-q)]$   ·   10-node line stencil   ·   exact residue from sympy",
                  "order 1 divides by no power of the step, so it has no roundoff arm and stays exact",
                  ok, f"optimum within {max(abs(np.log10(d / predicted)) for d in minima):.1f} decade",
                  out / "check_1_tradeoff.png")


def plot_2_the_two_laws(T, out):
    """Both predictions at once: where the optimum sits, and how good it can be."""
    grid = np.logspace(-0.3, -6, 260)
    orders = np.arange(1, 6)
    fig, axes = new_panels(T, 2, (11.0, 4.6), grid=True)

    for colour, half in zip(T["series"], (3, 5)):
        best_delta, best_error = [], []
        for order in orders:
            errors = error_curve(order, half, grid)
            index = int(np.nanargmin(errors))
            best_delta.append(grid[index])
            best_error.append(max(errors[index], 1e-18))
        axes[0].plot(orders, best_delta, "-", color=colour, linewidth=1.8,
                     label=f"{2 * half}-node stencil")
        dots(axes[0], orders, best_delta, colour, T, ms=7)
        axes[0].axhline(EPS ** (1.0 / (2 * half)), color=colour, linewidth=1.0, alpha=0.45)
        axes[1].plot(orders, best_error, "-", color=colour, linewidth=1.8,
                     label=f"{2 * half}-node stencil")
        dots(axes[1], orders, best_error, colour, T, ms=7)
        axes[1].plot(orders, EPS ** (1 - (orders - 1) / (2 * half)), "--",
                     color=colour, linewidth=1.3, alpha=0.7)

    axes[0].set_yscale("log")
    axes[0].set_xticks(orders)
    axes[0].set_xlabel("pole order $m$", color=T["ink2"], fontsize=10)
    axes[0].set_ylabel(r"best step $\delta^{*}$", color=T["ink2"], fontsize=10)
    axes[0].set_title(r"flat lines: $\varepsilon^{1/2n}$, independent of $m$",
                      loc="left", color=T["muted"], fontsize=9.5)
    axes[0].legend(loc="center right", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_xticks(orders)
    axes[1].set_ylim(1e-18, 1e-5)
    axes[1].set_xlabel("pole order $m$", color=T["ink2"], fontsize=10)
    axes[1].set_ylabel("best attainable |error|", color=T["ink2"], fontsize=10)
    axes[1].set_title(r"dashed: $\varepsilon^{\,1-k/2n}$, $k = m-1$",
                      loc="left", color=T["muted"], fontsize=9.5)
    axes[1].legend(loc="upper left", fontsize=9)

    ratios = []
    for half in (3, 5):
        for order in (3, 4, 5):
            errors = error_curve(order, half, grid)
            ratios.append(grid[int(np.nanargmin(errors))] / EPS ** (1.0 / (2 * half)))
    ok = all(0.2 < r < 5.0 for r in ratios)
    return finish_panels(fig, T, "Two predictions, both from the arithmetic alone",
                         r"balancing $A\delta^{2n-k}$ against $B\varepsilon/\delta^{k}$",
                         "the optimum depends only on the stencil width; what it buys you depends on the pole order",
                         ok, f"step ratios {min(ratios):.2f}-{max(ratios):.2f}",
                         out / "check_2_the_two_laws.png", top=0.82)


def plot_3_analytic_ground_truths(T, out):
    """Gamma and tan, where the answer is known in closed form at every pole."""
    fig, axes = new_panels(T, 2, (11.0, 4.4), grid=True)

    ks = np.arange(5)
    computed = [residues.residue(gamma, -float(k), 1).real for k in ks]
    truth = [(-1) ** k / math.factorial(k) for k in ks]
    axes[0].axhline(0.0, color=T["axis"], linewidth=1.0)
    axes[0].plot(-ks, truth, "-", color=T["muted"], linewidth=1.4, zorder=1)
    dots(axes[0], -ks, truth, T["muted"], T, label=r"exact $(-1)^k/k!$", ms=11)
    dots(axes[0], -ks, computed, T["series"][0], T, label="computed", ms=6)
    axes[0].set_xticks(-ks)
    axes[0].set_xlabel(r"pole of $\Gamma$ at $z = -k$", color=T["ink2"], fontsize=10)
    axes[0].set_ylabel("residue", color=T["ink2"], fontsize=10)
    axes[0].legend(loc="lower right", fontsize=9)
    gamma_worst = max(abs(c - t) for c, t in zip(computed, truth))

    found = poles.find_poles(np.tan, (-5.0, 5.0, -1.0, 1.0))
    values = residues.residues_of(np.tan, found)
    positions = [q.location.real for q in found]
    axes[1].axhline(-1.0, color=T["muted"], linewidth=1.4)
    axes[1].text(4.6, -1.0, "exact $-1$  ", color=T["muted"], fontsize=9,
                 ha="right", va="bottom")
    dots(axes[1], positions, [v.real for v in values], T["series"][0], T, ms=9)
    axes[1].set_ylim(-1.6, -0.4)
    axes[1].set_xlabel(r"poles of $\tan z$ at $(k+\frac{1}{2})\pi$", color=T["ink2"], fontsize=10)
    axes[1].set_ylabel("residue", color=T["ink2"], fontsize=10)
    tan_worst = max(abs(v + 1.0) for v in values)

    ok = gamma_worst < 1e-14 and tan_worst < 1e-13
    return finish_panels(fig, T, "Two functions whose residues are known exactly",
                         r"$\Gamma(z)$ from scipy and $\tan z$   ·   poles located by rung 3, residues by rung 4",
                         "neither has any rational structure for the code to exploit; both are pure callables",
                         ok, f"worst {max(gamma_worst, tan_worst):.0e}",
                         out / "check_3_analytic_ground_truths.png", top=0.82)


def plot_4_residues_sum_to_zero(T, out):
    """Four residues, none of them small, that must cancel exactly.

    The strongest check in the rung because it needs no reference value: if the
    denominator outranks the numerator by two or more, the residues sum to zero
    identically. Drawn tip to tail, the four have to close the loop.
    """
    # These four poles were chosen so the residue chain encloses real area. With
    # a less lucky set the four residues come out nearly collinear, the loop
    # closes just as exactly, and the picture says far less about it.
    roots = np.array([0.261 - 0.846j, -0.778 - 0.128j, -0.852 + 0.701j, 1.060 + 0.580j])
    f = lambda z: (z + 2.0) / np.prod([(z - r) for r in roots], axis=0)
    values = [residues.residue(f, r, 1, scale=0.3) for r in roots]

    fig, ax = new_figure(T, size=(7.0, 6.8))
    vertices = np.concatenate([[0.0 + 0.0j], np.cumsum(values)])
    total = abs(vertices[-1])

    # Arrows drawn with annotate take no part in autoscaling, so the window is
    # set from the chain itself -- square, because the aspect is equal.
    centre = (vertices.real.min() + vertices.real.max()) / 2 + \
             1j * (vertices.imag.min() + vertices.imag.max()) / 2
    reach = max(vertices.real.ptp(), vertices.imag.ptp()) * 0.62 + 0.15
    ax.set_xlim(centre.real - reach, centre.real + reach)
    ax.set_ylim(centre.imag - reach, centre.imag + reach)

    for i, value in enumerate(values):
        start, end = vertices[i], vertices[i + 1]
        ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag),
                    arrowprops=dict(arrowstyle="-|>", color=T["ramp"][i], linewidth=2.4,
                                    shrinkA=0, shrinkB=0))
        middle = (start + end) / 2
        ax.annotate(f"$\\mathrm{{Res}}_{{p_{i + 1}}}$", (middle.real, middle.imag),
                    textcoords="offset points", xytext=(10, 8),
                    color=T["ramp"][i], fontsize=10.5)
    dots(ax, [0.0], [0.0], T["ink"], T, ms=9)
    ax.annotate("start and finish", (0.0, 0.0), textcoords="offset points",
                xytext=(12, -14), color=T["ink2"], fontsize=9.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$\mathrm{Im}$", color=T["ink2"], fontsize=10)
    ax.text(0.03, 0.03, f"gap left at the end of the chain: {total:.1e}",
            transform=ax.transAxes, color=T["ink2"], fontsize=10, va="bottom")
    return finish(fig, ax, T, "Four residues that must close the loop",
                  r"$f(z) = (z+2)\,/\,[(z-p_1)(z-p_2)(z-p_3)(z-p_4)]$   ·   residues drawn tip to tail",
                  "the denominator outranks the numerator by 3, so the sum vanishes identically",
                  total < 1e-14, f"gap {total:.0e}", out / "check_4_sum_to_zero.png")


def plot_5_error_bar_is_honest(T, out):
    """The spread the ladder reports against the error it is estimating."""
    orders = np.arange(1, 6)
    spreads, truths = [], []
    for order in orders:
        value, _, spread = residues.residue_detail(test_function(order), P, order)
        spreads.append(max(spread, 1e-18))
        truths.append(max(abs(value - exact_residue(order)), 1e-18))

    fig, ax = new_figure(T, size=(6.8, 5.4))
    limits = (1e-18, 1e-8)
    ax.plot(limits, limits, "-", color=T["axis"], linewidth=1.2)
    ax.text(0.97, 0.03, "line: reported = true", transform=ax.transAxes,
            color=T["muted"], fontsize=9, ha="right", va="bottom")
    ax.fill_between(limits, limits, [limits[1]] * 2, color=T["grid"], zorder=0)
    ax.text(1e-17, 2e-9, "  conservative: reported exceeds true", color=T["muted"], fontsize=9)
    dots(ax, truths, spreads, T["series"][0], T, ms=9)
    for order, t, s in zip(orders, truths, spreads):
        ax.annotate(f"$m={order}$", (t, s), textcoords="offset points", xytext=(9, -3),
                    color=T["ink2"], fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("true |error|", color=T["ink2"], fontsize=10)
    ax.set_ylabel("error bar reported by the ladder", color=T["ink2"], fontsize=10)

    ratios = [s / t for s, t in zip(spreads, truths)]
    ok = all(r >= 0.5 for r in ratios) and max(ratios) < 1e3
    return finish(fig, ax, T, "The error bar is worth something",
                  r"step chosen by agreement with both neighbours on a geometric ladder",
                  "every point on or above the diagonal, none of them more than a factor of 13 above it",
                  ok, f"ratio {min(ratios):.1f}-{max(ratios):.1f}",
                  out / "check_5_error_bar.png")


def plot_6_the_forbidden_method(T, out):
    """What Cauchy's formula would have given, and why it cannot be used.

    Sampling f on a circle about p and averaging is the periodic trapezoid rule
    of rung 1, and it returns these residues at machine precision for every
    pole order at essentially any radius. It is also a contour integral, so a
    right-hand side built on it would share machinery with the left and rung 5
    would be comparing the quadrature against itself. It appears here once, for
    contrast, and is used nowhere in the pipeline.
    """
    def cauchy_residue(f, p, radius, nodes=64):
        # Forbidden: this is the LHS quadrature wearing a different hat.
        theta = 2 * np.pi * np.arange(nodes) / nodes
        z = p + radius * np.exp(1j * theta)
        return np.mean(f(z) * (z - p))

    grid = np.logspace(-0.3, -6, 200)
    fig, axes = new_panels(T, 2, (11.0, 4.6), grid=True)
    for ax, order in zip(axes, (1, 5)):
        truth = exact_residue(order)
        f = test_function(order)
        finite_difference = error_curve(order, 5, grid)
        # Cauchy has a tradeoff of its own: at small radius |f| ~ r^(-m) is
        # enormous and roundoff swamps the average, so its optimum sits at the
        # *largest* radius that stays clear of the next singularity.
        radii = np.logspace(0.0, -6, 50)
        circle = np.array([abs(cauchy_residue(f, P, r) - truth) for r in radii])
        ax.plot(grid, np.maximum(finite_difference, 1e-18), "-", color=T["series"][1],
                linewidth=1.8, label="finite differences on a line")
        ax.plot(radii, np.maximum(circle, 1e-18), "-", color=T["series"][0],
                linewidth=1.8, label="Cauchy on a circle (forbidden)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_ylim(1e-18, 1e2)
        ax.set_xlabel(f"step or radius   —   pole of order {order}", color=T["ink2"], fontsize=10)
        ax.set_ylabel("|error| in the residue", color=T["ink2"], fontsize=10)
        ax.legend(loc="upper left", fontsize=9)

    truth = exact_residue(5)
    circle_best = min(abs(cauchy_residue(test_function(5), P, r) - truth)
                      for r in np.logspace(0.0, -6, 50))
    line_best = np.nanmin(error_curve(5, 5, grid))
    ok = circle_best < 1e-14 and line_best > 1e-12
    return finish_panels(fig, T, "The method we are not allowed to use",
                         r"the same residue, by a line stencil and by the trapezoid rule on a circle",
                         "both have a best step; the circle's floor is machine precision, and it is a contour integral",
                         ok, f"circle {circle_best:.0e} vs line {line_best:.0e}",
                         out / "check_6_forbidden_method.png", top=0.82)


FIGURES = [
    plot_1_the_tradeoff,
    plot_2_the_two_laws,
    plot_3_analytic_ground_truths,
    plot_4_residues_sum_to_zero,
    plot_5_error_bar_is_honest,
    plot_6_the_forbidden_method,
]


def main():
    parser = argparse.ArgumentParser(description="Rung 4 visual diagnostics.")
    parser.add_argument("--out", default=None, help="output directory (default: checks/plots/rung4)")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    T = DARK if args.dark else LIGHT
    out = pathlib.Path(args.out) if args.out else HERE / "rung4"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rung 4 figures -> {out}  ({'dark' if args.dark else 'light'})")
    failures = sum(not figure(T, out) for figure in FIGURES)
    print(f"{len(FIGURES) - failures}/{len(FIGURES)} figures pass.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
