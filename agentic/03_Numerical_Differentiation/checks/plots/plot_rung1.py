"""Visual diagnostics for every check in checks/rung1_quadrature.py.

    python checks/plots/plot_rung1.py [--out DIR] [--dark]

rung1_quadrature.py answers "does this pass?". This answers "do I believe it?"
Each figure shows the check's claim as a curve against the value known
analytically, and where the check tests one configuration the figure sweeps the
parameter -- so a check that passes by luck at a single point shows up here as a
curve that does not track its prediction.

Every figure carries a PASS/FAIL badge computed from the numbers actually
plotted, so the figures and the checks cannot silently disagree.

Output lands in checks/plots/rung1/ by default, one directory per rung, so the
generated images never mix with the code that generates them.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import contour
from checks.plots._style import LIGHT, DARK, new_figure, finish, dots

TWO_PI_I = 2j * np.pi
HERE = pathlib.Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# One figure per check, in the order the checks run
# ----------------------------------------------------------------------------

def plot_1_monomials(T, out):
    """Every monomial integrates to zero around the unit circle except z^(-1).

    The sweep is the point: one spike at n = -1 against a floor at machine
    epsilon, across nine exponents. A sign or normalization error moves the spike
    or lifts the floor.
    """
    C = contour.circle()
    ns = np.arange(-4, 5)
    vals = np.array([abs(C.integrate(lambda z, n=n: z ** float(n), n=64)) for n in ns])
    is_pole = ns == -1

    fig, ax = new_figure(T)
    ax.axhline(2 * np.pi, color=T["muted"], linewidth=1.0)
    ax.text(4.0, 2 * np.pi, r"$2\pi$  ", color=T["muted"], fontsize=9,
            va="bottom", ha="right")
    dots(ax, ns[~is_pole], np.maximum(vals[~is_pole], 1e-18), T["series"][0], T,
         label="predicted $0$")
    dots(ax, ns[is_pole], vals[is_pole], T["series"][1], T, label=r"predicted $2\pi i$", ms=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e2)
    ax.set_xlabel("exponent $n$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|\oint z^n\,dz|$", color=T["ink2"], fontsize=10)
    ax.legend(loc="center left", fontsize=9)

    floor = vals[~is_pole].max()
    ok = floor < 1e-13 and abs(vals[is_pole][0] - 2 * np.pi) < 1e-13
    return finish(fig, ax, T, "A single pole in a field of zeros",
                  r"$f(z) = z^n$   ·   unit circle   ·   64 nodes",
                  r"every exponent but $-1$ has a single-valued antiderivative, so only that one survives",
                  ok, f"floor {floor:.0e}", out / "check_1_monomials.png")


def plot_2_closed_form_error(T, out):
    """The trapezoid error is predicted in closed form, not merely bounded.

    error(N) = 2 pi i a^N / (1 - a^N), from summing a geometric series over the
    N-th roots of unity. Measured points sit on the predicted curves for ten
    decades, until both reach the roundoff floor -- which is the honest end of
    the agreement, not a failure.
    """
    C = contour.circle()
    Ns = np.arange(2, 46, 2)
    fig, ax = new_figure(T)
    worst = 0.0
    for color, a, name in zip(T["series"], (0.5, 0.7j, -0.35 + 0.4j),
                              ("$a = 0.5$", "$a = 0.7i$", "$a = -0.35 + 0.4i$")):
        measured = np.array([C.integrate(lambda z, a=a: 1.0 / (z - a), n=N) - TWO_PI_I
                             for N in Ns])
        predicted = TWO_PI_I * a ** Ns / (1.0 - a ** Ns)
        ax.plot(Ns, np.abs(predicted), "-", color=color, linewidth=1.8, label=name)
        dots(ax, Ns, np.maximum(np.abs(measured), 1e-18), color, T, ms=5.5)
        # Absolute, not relative: once the predicted error itself approaches
        # 1e-15 a relative comparison is measuring roundoff, not the formula.
        # The real claim is that the gap stays at machine noise no matter how
        # small the quantity being predicted becomes.
        worst = max(worst, np.max(np.abs(measured - predicted)))
    ax.axhspan(1e-18, 1e-15, color=T["grid"], zorder=0)
    ax.text(44, 1e-16, "roundoff floor  ", color=T["muted"], fontsize=9,
            ha="right", va="center")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e1)
    ax.set_xlabel("nodes $N$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|$quadrature $-\ 2\pi i|$", color=T["ink2"], fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    ok = worst < 1e-14
    return finish(fig, ax, T, "The quadrature error, predicted to every digit",
                  r"$f(z) = 1/(z-a)$   ·   unit circle   ·   one simple pole at $a$, inside",
                  r"lines: $2\pi i\,a^N/(1-a^N)$ derived by hand   ·   dots: measured",
                  ok, f"max gap {worst:.0e}", out / "check_2_closed_form_error.png")


def plot_3_geometric_convergence(T, out):
    """Convergence is geometric, at a rate set by the pole's distance from the contour.

    Straight lines on a log axis with slope log|a| -- four poles, four different
    slopes, all predicted in advance. The dashed reference is what an O(h^2) rule
    would do: the comparison shows the periodic trapezoid is in a different
    class, not merely carrying a better constant.

    The slope is fitted only in the asymptotic window. error ~ |a|^N is the
    large-N limit of 2 pi i a^N/(1 - a^N), and at small N the 1/(1 - a^N)
    prefactor is genuinely not 1 -- for |a| = 0.85 it is 3.6 at N = 2. Fitting a
    pure exponential through that pre-asymptotic bend biases the slope by several
    percent, which is a property of the formula, not an error in the quadrature.
    """
    C = contour.circle()
    Ns = np.arange(2, 90, 2)
    fig, ax = new_figure(T)
    worst_slope = 0.0
    for color, a in zip(T["ramp"], (0.3, 0.5, 0.7, 0.85)):
        errs = np.array([abs(C.integrate(lambda z, a=a: 1.0 / (z - a), n=N) - TWO_PI_I)
                         for N in Ns])
        ax.plot(Ns, np.maximum(errs, 1e-18), "-", color=color, linewidth=1.8,
                label=f"$|a| = {a}$")
        window = (errs > 1e-12) & (errs < 1e-2)
        fitted = np.polyfit(Ns[window], np.log(errs[window]), 1)[0]
        worst_slope = max(worst_slope, abs(fitted - np.log(a)) / abs(np.log(a)))
        last = np.flatnonzero(window)[-1]
        ax.text(Ns[last] + 1.5, errs[last], f"{a}", color=color, fontsize=9, va="center")

    second_order = 4.0 * (Ns / Ns[0]) ** -2.0
    ax.plot(Ns, second_order, "--", color=T["muted"], linewidth=1.4)
    mid = len(Ns) // 4
    ax.text(Ns[mid], second_order[mid] * 3, r"$O(h^2)$", color=T["muted"], fontsize=9,
            va="bottom", ha="center")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e1)
    ax.set_xlabel("nodes $N$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|$quadrature $-\ 2\pi i|$", color=T["ink2"], fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    ok = worst_slope < 1e-3
    return finish(fig, ax, T, "Geometric convergence, at the pole's own rate",
                  r"$f(z) = 1/(z-a)$   ·   unit circle   ·   four pole distances $|a|$",
                  r"slope fitted where $|a|^N \ll 1$, compared with $\log|a|$; end labels give $|a|$",
                  ok, f"slope err {worst_slope:.3%}", out / "check_3_geometric_convergence.png")


def plot_4_reparametrization(T, out):
    """The same curve at a different speed samples different points, same integral.

    The figure makes the premise visible: the two node sets barely overlap. Only
    a correct gamma' weighting reconciles them, so agreement at 1e-15 is evidence
    about the derivative factor and nothing else.
    """
    C = contour.star(mean_radius=1.0, amplitude=0.3, lobes=3)
    D = contour.reparametrized(C, wobble=0.6)
    f = lambda z: 1.0 / (z - 0.2) + 1.0 / (z + 0.3 + 0.1j) ** 2 + np.exp(z)
    a, b = C.integrate(f, n=400), D.integrate(f, n=400)

    fig, ax = new_figure(T, size=(7.0, 6.8))
    path = C.sample(800)
    ax.plot(path.real, path.imag, "-", color=T["axis"], linewidth=1.4, zorder=1)
    t = np.arange(48) / 48
    for color, curve, name in ((T["series"][0], C, "uniform $t$"),
                               (T["series"][1], D, "wobbled $t$")):
        z = curve.gamma(t)
        dots(ax, z.real, z.imag, color, T, label=name, ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,z$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$\mathrm{Im}\,z$", color=T["ink2"], fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.5, -0.17, f"uniform  {a.real:+.12f}{a.imag:+.12f}i\n"
                        f"wobbled  {b.real:+.12f}{b.imag:+.12f}i",
            transform=ax.transAxes, color=T["ink2"], fontsize=8.5,
            family="monospace", va="top", ha="center")

    ok = abs(a - b) < 1e-11
    return finish(fig, ax, T, "Same answer, new nodes",
                  r"$f(z) = \frac{1}{z-0.2} + \frac{1}{(z+0.3+0.1i)^2} + e^{z}$   ·   3-lobed star   ·   400 nodes",
                  "48 nodes drawn; the curve and its orientation are identical, the sampling is not",
                  ok, f"gap {abs(a - b):.0e}", out / "check_4_reparametrization.png")


def plot_5_scale_independence(T, out):
    """The answer does not care how big the circle is, only whether the pole is inside.

    Sweeping the radius over five decades is the sharp version: a contour whose
    derivative failed to scale with its radius would give an answer proportional
    to R -- a rising diagonal instead of a flat floor.

    The excluded circle is centered at 5, so as its radius approaches 5 it closes
    in on the pole at the origin and the error climbs off the floor. That is not
    a failure of scale independence; it is the pole-proximity effect of figure 3
    seen from the other side, since the convergence rate is set by the distance
    from the contour to the nearest pole. The verdict is therefore computed on
    the radii that stay clear of the pole, and the approach is annotated rather
    than cropped away.
    """
    Rs = np.logspace(-2, 3, 40)
    f = lambda z: 1.0 / z
    inside = np.array([abs(contour.circle(0.0, R).integrate(f, n=64) - TWO_PI_I) for R in Rs])
    outer = Rs[Rs < 4.9]
    outside = np.array([abs(contour.circle(5.0, R).integrate(f, n=64)) for R in outer])
    clear = outer < 2.5

    fig, ax = new_figure(T)
    ax.plot(Rs, np.maximum(inside, 1e-18), "-", color=T["series"][0], linewidth=1.8,
            label=r"origin enclosed: target $2\pi i$")
    ax.plot(outer, np.maximum(outside, 1e-18), "-", color=T["series"][1], linewidth=1.8,
            label="origin excluded (center $5$): target $0$")
    # The failure this check exists to rule out, drawn to scale: a contour that
    # forgot to scale gamma' with its radius would return 2 pi i / R instead of
    # 2 pi i, an error of 2 pi |1 - 1/R| -- order one for large R, fifteen
    # decades above the floor the code actually sits on.
    # Drawn only for R >= 2: the expression passes through zero at R = 1, and
    # the resulting plunge is an artifact of the reference, not of anything
    # measured.
    big = Rs >= 2.0
    broken = 2 * np.pi * np.abs(1.0 - 1.0 / Rs[big])
    ax.plot(Rs[big], broken, "--", color=T["muted"], linewidth=1.4)
    ax.text(Rs[-1], broken[-1], r"if $\gamma'$ ignored $R$  ", color=T["muted"],
            fontsize=9, va="top", ha="right")
    ax.annotate("contour closes\non the pole", xy=(outer[-1], outside[-1]),
                xytext=(0.52, 0.55), textcoords="axes fraction",
                color=T["muted"], fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color=T["muted"], linewidth=1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e2)
    ax.set_xlabel("circle radius $R$", color=T["ink2"], fontsize=10)
    ax.set_ylabel("|error|", color=T["ink2"], fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

    worst = max(inside.max(), outside[clear].max())
    ok = worst < 1e-13
    return finish(fig, ax, T, "Five decades of radius, no drift",
                  r"$f(z) = 1/z$   ·   circles of radius $R$ centered at $0$ and at $5$   ·   64 nodes",
                  r"the $R$ inside $\gamma$ must cancel the $R$ inside $\gamma'$",
                  ok, f"worst clear of pole {worst:.0e}",
                  out / "check_5_scale_independence.png")


def plot_6_repeated_traversal(T, out):
    """Going around k times multiplies the integral by k, including for k < 0.

    Plotted as the recovered multiplier rather than the raw integral, so the
    claim is a line of slope one through the origin, and a clockwise loop is
    visibly the negative of a counterclockwise one.
    """
    ks = np.arange(-3, 4)
    got = np.array([contour.circle(winding=int(k)).integrate(lambda z: 1.0 / z, n=128)
                    if k != 0 else 0.0 for k in ks])
    recovered = (got / TWO_PI_I).real

    fig, ax = new_figure(T, size=(6.4, 4.8))
    ax.plot([-3.4, 3.4], [-3.4, 3.4], "-", color=T["axis"], linewidth=1.2, zorder=1)
    dots(ax, ks, recovered, T["series"][0], T, ms=8)
    ax.set_xlabel("times the contour is traversed, $k$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"recovered $\oint dz/z\ /\ 2\pi i$", color=T["ink2"], fontsize=10)
    ax.set_xticks(ks)
    ax.set_yticks(ks)
    ax.text(0.03, 0.92, "line: exact $k$", transform=ax.transAxes,
            color=T["muted"], fontsize=9)

    worst = np.max(np.abs(recovered - ks))
    ok = worst < 1e-12
    return finish(fig, ax, T, "Winding shows up before any theory does",
                  r"$f(z) = 1/z$   ·   unit circle traversed $k$ times   ·   128 nodes",
                  "the sign convention for clockwise travel is exercised here and nowhere else",
                  ok, f"worst {worst:.0e}", out / "check_6_repeated_traversal.png")


def plot_7_gauss_legendre_exactness(T, out):
    """Gauss-Legendre with p nodes is exact through degree 2p-1, and the cliff is visible.

    An irregular triangle is used on purpose: on a square the fourfold symmetry
    cancels most monomials no matter what the quadrature does, which would let a
    broken rule look exact. The cliff locations are a property of Gauss-Legendre
    known before any code ran.
    """
    P = contour.Polygon([0.9 + 0.2j, -1.1 + 0.7j, -0.3 - 1.3j])
    ns = np.arange(0, 16)
    fig, ax = new_figure(T)
    ok = True
    for color, p in zip(T["series"], (3, 6)):
        errs = np.array([abs(P.integrate(lambda z, n=n: z ** n, n=p)) for n in ns])
        ax.plot(ns, np.maximum(errs, 1e-18), "-", color=color, linewidth=1.8,
                label=f"{p} nodes/edge — exact through degree {2 * p - 1}")
        dots(ax, ns, np.maximum(errs, 1e-18), color, T, ms=5.5)
        ax.axvline(2 * p - 0.5, color=color, linewidth=1.0, alpha=0.35)
        ok = ok and errs[ns <= 2 * p - 1].max() < 1e-13 and errs[2 * p] > 1e-6
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e2)
    ax.set_xlabel("exponent $n$", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|\oint z^n\,dz|$", color=T["ink2"], fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    return finish(fig, ax, T, "The Gauss-Legendre cliff, exactly where predicted",
                  r"$f(z) = z^n$   ·   triangle $(0.9{+}0.2i,\ -1.1{+}0.7i,\ -0.3{-}1.3i)$   ·   exact value $0$",
                  "irregular on purpose: a symmetric contour would cancel these monomials by itself",
                  ok, "cliffs at 6 and 12", out / "check_7_gauss_legendre_exactness.png")


def plot_8_why_polygons_differ(T, out):
    """Why a polygon needs its own rule: corners make gamma' jump and the trapezoid dies.

    Three curves against one budget axis. The trapezoid on a corner-ful path is
    first order -- the integrand itself is discontinuous, not merely kinked -- so
    it is still at 1e-3 after sixteen thousand evaluations, while Gauss-Legendre
    per edge reaches 1e-15 in ninety-six. This is the measurement that justifies
    the Polygon class existing at all.
    """
    V = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    nxt = np.roll(V, -1)

    def gamma(t):
        u = np.atleast_1d(t) % 1.0 * 4
        k = np.minimum(u.astype(int), 3)
        return V[k] + (u - k) * (nxt[k] - V[k])

    def dgamma(t):
        k = np.minimum((np.atleast_1d(t) % 1.0 * 4).astype(int), 3)
        return 4 * (nxt[k] - V[k])

    square_smooth = contour.SmoothContour(gamma, dgamma, "square via trapezoid")
    P = contour.Polygon(V)
    C = contour.circle()
    f = lambda z: 1.0 / z

    trap_n = np.array([63, 253, 1023, 4093, 16381])
    trap = np.array([abs(square_smooth.integrate(f, n=int(n)) - TWO_PI_I) for n in trap_n])
    gl_p = np.array([2, 3, 4, 6, 8, 12, 16, 24])
    gl = np.array([abs(P.integrate(f, n=int(p)) - TWO_PI_I) for p in gl_p])
    circ_n = np.array([4, 8, 16, 32, 64])
    circ = np.array([abs(C.integrate(f, n=int(n)) - TWO_PI_I) for n in circ_n])

    fig, ax = new_figure(T)
    for x, y, color, name in ((trap_n, trap, T["series"][1], "square, periodic trapezoid"),
                              (4 * gl_p, gl, T["series"][0], "square, Gauss-Legendre per edge"),
                              (circ_n, circ, T["series"][2], "circle, periodic trapezoid")):
        y = np.maximum(y, 1e-18)
        ax.plot(x, y, "-", color=color, linewidth=1.8, label=name)
        dots(ax, x, y, color, T, ms=5.5)
    ax.plot(trap_n, 4.0 / trap_n, "--", color=T["muted"], linewidth=1.4)
    ax.text(trap_n[0], 4.0 / trap_n[0], r"  $4/N$", color=T["muted"], fontsize=9,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e1)
    ax.set_xlabel("function evaluations", color=T["ink2"], fontsize=10)
    ax.set_ylabel(r"$|$quadrature $-\ 2\pi i|$", color=T["ink2"], fontsize=10)
    ax.legend(loc="center right", fontsize=9)

    rate = np.polyfit(np.log(trap_n), np.log(trap), 1)[0]
    ok = gl.min() < 1e-14 and abs(rate + 1.0) < 0.05 and trap.min() > 1e-4
    return finish(fig, ax, T, "Corners cost twelve orders of magnitude",
                  r"$f(z) = 1/z$   ·   square $(\pm 1 \pm i)$, and the unit circle for reference",
                  r"a jump in $\gamma'$ makes the integrand discontinuous, so the trapezoid rule is $O(h)$",
                  ok, f"trapezoid rate $N^{{{rate:.2f}}}$", out / "check_8_why_polygons_differ.png")


FIGURES = [
    plot_1_monomials,
    plot_2_closed_form_error,
    plot_3_geometric_convergence,
    plot_4_reparametrization,
    plot_5_scale_independence,
    plot_6_repeated_traversal,
    plot_7_gauss_legendre_exactness,
    plot_8_why_polygons_differ,
]


def main():
    parser = argparse.ArgumentParser(description="Rung 1 visual diagnostics.")
    parser.add_argument("--out", default=None, help="output directory (default: checks/plots/rung1)")
    parser.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = parser.parse_args()

    T = DARK if args.dark else LIGHT
    out = pathlib.Path(args.out) if args.out else HERE / "rung1"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rung 1 figures -> {out}  ({'dark' if args.dark else 'light'})")
    failures = sum(not figure(T, out) for figure in FIGURES)
    print(f"{len(FIGURES) - failures}/{len(FIGURES)} figures pass.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
