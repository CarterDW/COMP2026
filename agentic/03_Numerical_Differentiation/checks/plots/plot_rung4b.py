"""Visual diagnostics for checks/rung4b_singularities.py.

    python checks/plots/plot_rung4b.py [--out DIR] [--dark]

Three pictures of one idea: the right-hand side must be able to say "I cannot
answer this". The first shows the test that lets it, the second shows what is
at stake if it does not, and the third shows the two verdicts side by side on a
function that needs both.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import numpy as np

import contour
import poles
import residues
from checks.plots._style import (LIGHT, DARK, dots, finish, new_figure)
# The magnitude field is rung 3's; reusing it keeps one definition of how
# log|f| gets drawn without overflowing.
from checks.plots.plot_rung3 import draw_field

HERE = pathlib.Path(__file__).resolve().parent
TWO_PI_I = 2j * np.pi
ESSENTIAL = lambda z: np.exp(1.0 / z)


def plot_1_boundedness_guard(T, out):
    """The one test that separates a pole from everything else.

    p is a pole of order at most m exactly when (z-p)^m f extends analytically
    to p, and then the stencil values settle instead of growing. Four decades
    of shrinking step put an enormous gap between the cases: nothing that is a
    pole grows at all, understating the order by one grows by 1e4, and an
    essential singularity overflows.
    """
    triple = lambda z: np.exp(z) / z ** 3
    steps = 7
    deltas = 0.5 * 0.1 ** np.arange(steps)
    cases = [
        ("order 3 claimed", triple, 3, True),
        ("order 5 claimed", triple, 5, True),
        ("order 2 claimed", triple, 2, False),
        (r"$e^{1/z}$, order 1", ESSENTIAL, 1, False),
    ]
    fig, ax = new_figure(T, size=(7.6, 5.0))
    for name, f, order, is_pole in cases:
        colour = T["series"][0] if is_pole else T["series"][1]
        magnitudes = residues.leading_magnitudes(f, 0.0, order, scale=0.5, steps=steps)
        magnitudes = np.array([m if np.isfinite(m) else np.inf for m in magnitudes])
        visible = np.minimum(magnitudes, 1e18)
        ax.plot(deltas, visible, "-", color=colour, linewidth=1.8,
                label=("bounded: p IS a pole of that order" if name == "order 3 claimed"
                       else "unbounded: p is not" if name == "order 2 claimed" else None))
        dots(ax, deltas, visible, colour, T, ms=5.5)
        # Label at the last point still inside the window: the overstated-order
        # curve runs off the bottom and the essential one pins to the overflow
        # line, so "the final point" is not a place a label can live.
        inside = np.flatnonzero((visible > 2e-9) & (visible < 5e17))
        anchor = inside[-1] if len(inside) else 0
        ax.annotate(name, (deltas[anchor], visible[anchor]), textcoords="offset points",
                    xytext=(-10, 11), color=colour, fontsize=10, ha="right")
    ax.axhline(1e18, color=T["muted"], linewidth=1.0)
    ax.text(2e-6, 1.6e18, "  overflow", color=T["muted"], fontsize=9, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1e-9, 1e20)
    ax.set_xlabel(r"step size $\delta$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$\max\,|(z-p)^m f(z)|$ over the stencil", color=T["ink2"], fontsize=10)
    ax.legend(loc="center left", fontsize=9)

    settled = residues.leading_magnitudes(triple, 0.0, 3, scale=0.5, steps=steps)
    runaway = residues.leading_magnitudes(triple, 0.0, 2, scale=0.5, steps=steps)
    ok = settled[-1] / settled[0] < 100 and runaway[-1] / runaway[0] > 100
    return finish(fig, ax, T, "Bounded or not — that is the whole test",
                  r"$f(z) = e^{z}/z^{3}$, a genuine triple pole, and $e^{1/z}$   ·   stencil of 10 nodes",
                  "overstating the order is fine and stays bounded; understating it is not, and does not",
                  ok, f"growth {settled[-1] / settled[0]:.0e} vs {runaway[-1] / runaway[0]:.0e}",
                  out / "check_1_boundedness_guard.png")


def plot_2_what_silence_would_cost(T, out):
    """The left-hand side sees a residue the right-hand side cannot compute.

    exp(1/z) = sum z^(-n)/n! has residue 1, so the theorem predicts 2 pi i, and
    the rung-1 quadrature delivers it. No amount of differentiation can recover
    that residue -- (z-p)^m f is analytic for no m -- so the honest right-hand
    side is a refusal. The number a silent one would have produced is zero, and
    the gap is the full 2 pi i drawn here.
    """
    nodes = np.unique(np.round(np.logspace(0.7, 3.2, 26)).astype(int))
    C = contour.circle(0.0, 1.0)
    errors = np.array([abs(C.integrate(ESSENTIAL, n=int(n)) - TWO_PI_I) for n in nodes])

    fig, ax = new_figure(T, size=(7.6, 4.8))
    ax.axhline(abs(TWO_PI_I), color=T["series"][1], linewidth=1.8)
    ax.text(nodes[-1], abs(TWO_PI_I) * 1.3, "what a silent RHS of 0 would be wrong by  ",
            color=T["series"][1], fontsize=9.5, ha="right")
    ax.plot(nodes, np.maximum(errors, 1e-18), "-", color=T["series"][0], linewidth=1.8)
    dots(ax, nodes, np.maximum(errors, 1e-18), T["series"][0], T, ms=6)
    ax.annotate("LHS quadrature, converging on $2\\pi i$",
                (nodes[-1], max(errors[-1], 1e-18)), textcoords="offset points",
                xytext=(-10, 16), color=T["series"][0], fontsize=9.5, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e2)
    ax.set_xlabel("quadrature nodes $N$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|$value $-\ 2\pi i|$", color=T["ink2"], fontsize=10)

    ok = errors[-1] < 1e-14
    return finish(fig, ax, T, "The theorem holds; our right-hand side cannot follow",
                  r"$\oint_{|z|=1} e^{1/z}\,dz$   ·   residue 1, so the exact value is $2\pi i$",
                  "the residue exists and the LHS finds it; no derivative can, so the RHS must refuse",
                  ok, f"LHS error {errors[-1]:.0e}", out / "check_2_what_silence_would_cost.png")


def plot_3_two_verdicts(T, out):
    """One function, one pole, one singularity that is not — and both reported."""
    f = lambda z: np.exp(1.0 / z) + 1.0 / (z - 0.6)
    region = (-0.9, 1.1, -1.0, 1.0)
    fig, ax = new_figure(T, size=(7.6, 6.2), grid=False)
    draw_field(ax, T, f, region, resolution=340, vspan=5.0)
    result = poles.survey(f, (-1, 1, -1, 1))

    for q in result.poles:
        ax.plot(q.location.real, q.location.imag, "o", markerfacecolor="none",
                markeredgecolor=T["series"][1], markeredgewidth=2.2, markersize=15)
        ax.annotate(f"pole, order {q.order}", (q.location.real, q.location.imag),
                    textcoords="offset points", xytext=(13, 9),
                    color=T["series"][1], fontsize=10.5, weight="bold")
    for q in result.non_poles:
        ax.plot(q.location.real, q.location.imag, "X", color=T["series"][2],
                markersize=14, markeredgecolor=T["surface"], markeredgewidth=1.4)
        ax.annotate("not a pole — no residue\nreachable by differentiation",
                    (q.location.real, q.location.imag), textcoords="offset points",
                    xytext=(15, -30), color=T["series"][2], fontsize=10, weight="bold")

    value = residues.residue(f, 0.6, 1, scale=0.2)
    ok = (len(result.poles) == 1 and len(result.non_poles) == 1
          and abs(value - 1.0) < 1e-12)
    return finish(fig, ax, T, "Two verdicts, both reported",
                  r"$f(z) = e^{1/z} + 1/(z - 0.6)$   ·   field is $\log_{10}|f|$",
                  "an empty non-pole list would have looked like success: there is a perfectly good pole to report",
                  ok, f"residue at 0.6 = {value.real:.6f}", out / "check_3_two_verdicts.png")


FIGURES = [plot_1_boundedness_guard, plot_2_what_silence_would_cost, plot_3_two_verdicts]


def main():
    parser = argparse.ArgumentParser(description="Rung 4b visual diagnostics.")
    parser.add_argument("--out", default=None, help="output directory (default: checks/plots/rung4b)")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    T = DARK if args.dark else LIGHT
    out = pathlib.Path(args.out) if args.out else HERE / "rung4b"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rung 4b figures -> {out}  ({'dark' if args.dark else 'light'})")
    failures = sum(not figure(T, out) for figure in FIGURES)
    print(f"{len(FIGURES) - failures}/{len(FIGURES)} figures pass.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
