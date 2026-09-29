"""The left-hand side of the residue theorem: closed-contour integrals by quadrature.

A contour is a closed path z = gamma(t) for t in [0, 1], with gamma(1) = gamma(0),
and the integral we want is

    oint_C f(z) dz = int_0^1 f(gamma(t)) gamma'(t) dt,

so every contour has to supply gamma' as well as gamma. Two quadrature rules
appear here, picked by the smoothness of the path:

  * Smooth closed paths -> the periodic trapezoid rule. The integrand is analytic
    and 1-periodic in t, so the trapezoid rule converges geometrically in the
    number of nodes rather than as h^2. This is why the LHS can be pushed to
    ~1e-15 and is the single most important fact in this module.

  * Polygons -> Gauss-Legendre on each edge. Corners break periodicity, which
    would drop the trapezoid rule back to O(h^2). Integrating edge by edge
    restores fast convergence because f is analytic along each open edge.

Nothing in this file knows what a pole or a residue is. That separation is the
whole point: the right-hand side must be computed by machinery that shares
nothing with this file, or the comparison is vacuous.
"""

import numpy as np


class SmoothContour:
    """A closed path z = gamma(t), t in [0, 1], integrated by the periodic trapezoid rule."""

    def __init__(self, gamma, dgamma, name="smooth contour"):
        self.gamma = gamma
        self.dgamma = dgamma
        self.name = name

    def integrate(self, f, n=256):
        """oint f dz using n equally spaced nodes in the parameter t.

        The node t = 1 is omitted because gamma(1) = gamma(0); including it would
        double-count one point and destroy the geometric convergence.
        """
        t = np.arange(n) / n
        return np.sum(f(self.gamma(t)) * self.dgamma(t)) / n

    def sample(self, n=800):
        """Points along the path, for plotting. Endpoint included so the curve closes."""
        return self.gamma(np.linspace(0.0, 1.0, n))


class Polygon:
    """A closed polygon through `vertices`, integrated edge by edge with Gauss-Legendre."""

    def __init__(self, vertices, name="polygon"):
        self.vertices = np.asarray(vertices, dtype=complex)
        self.name = name

    def edges(self):
        """Pairs (a, b) of consecutive vertices, wrapping the last back to the first."""
        v = self.vertices
        return zip(v, np.roll(v, -1))

    def integrate(self, f, n=32):
        """oint f dz using n Gauss-Legendre nodes on each edge.

        On the edge from a to b write z(s) = a + s(b - a) for s in [0, 1], so
        dz = (b - a) ds and the edge contributes (b - a) * int_0^1 f(z(s)) ds.
        """
        x, w = np.polynomial.legendre.leggauss(n)
        s = (x + 1.0) / 2.0
        w = w / 2.0
        total = 0.0 + 0.0j
        for a, b in self.edges():
            total += (b - a) * np.sum(w * f(a + s * (b - a)))
        return total

    def sample(self, n=800):
        """Points along the path, for plotting."""
        return np.append(self.vertices, self.vertices[0])


def circle(center=0.0, radius=1.0, winding=1):
    """Circle of given center and radius, traversed `winding` times counterclockwise.

    A negative winding traverses it clockwise. This is the cleanest way to build a
    contour whose winding number about an interior point is something other than 1.
    """
    k = 2j * np.pi * winding

    def gamma(t):
        return center + radius * np.exp(k * t)

    def dgamma(t):
        return k * radius * np.exp(k * t)

    return SmoothContour(gamma, dgamma, f"circle(c={center:g}, r={radius:g}, w={winding})")


def ellipse(center=0.0, a=1.0, b=1.0, winding=1):
    """Axis-aligned ellipse with semi-axis a along the real direction, b along the imaginary."""
    w = 2 * np.pi * winding

    def gamma(t):
        return center + a * np.cos(w * t) + 1j * b * np.sin(w * t)

    def dgamma(t):
        return w * (-a * np.sin(w * t) + 1j * b * np.cos(w * t))

    return SmoothContour(gamma, dgamma, f"ellipse(a={a:g}, b={b:g}, w={winding})")


def star(center=0.0, mean_radius=1.0, amplitude=0.3, lobes=3):
    """A smooth non-circular closed curve: r(theta) = mean_radius + amplitude*cos(lobes*theta).

    Useful as a test contour precisely because it is not a circle -- it will catch
    an integrator that has quietly hard-coded a circular parametrization somewhere.
    """
    def gamma(t):
        th = 2 * np.pi * t
        return center + (mean_radius + amplitude * np.cos(lobes * th)) * np.exp(1j * th)

    def dgamma(t):
        th = 2 * np.pi * t
        r = mean_radius + amplitude * np.cos(lobes * th)
        dr = -amplitude * lobes * np.sin(lobes * th)
        return 2 * np.pi * (dr + 1j * r) * np.exp(1j * th)

    return SmoothContour(gamma, dgamma, f"star(r={mean_radius:g}, A={amplitude:g}, k={lobes})")


def reparametrized(contour, wobble=0.3):
    """The same point set as `contour`, traversed at a non-uniform speed.

    theta(t) = t + wobble*sin(2 pi t)/(2 pi) is a smooth increasing map of [0,1]
    onto itself, so the image curve and its orientation are unchanged. Any correct
    integrator must return the same value; one that mishandles the gamma' factor
    will not.
    """
    def theta(t):
        return t + wobble * np.sin(2 * np.pi * t) / (2 * np.pi)

    def dtheta(t):
        return 1.0 + wobble * np.cos(2 * np.pi * t)

    def gamma(t):
        return contour.gamma(theta(t))

    def dgamma(t):
        return contour.dgamma(theta(t)) * dtheta(t)

    return SmoothContour(gamma, dgamma, f"reparametrized({contour.name})")


def figure_eight(scale=1.0):
    """A self-intersecting closed path whose two lobes wind in opposite senses.

    gamma(t) = sin(th) (1 + i cos(th)) with th = 2 pi t: a Lissajous figure eight
    crossing itself at the origin, with lobes centred near +-scale/2. It is the
    sharpest test case for a winding-number routine, because the two lobes must
    come out +1 and -1 -- any algorithm that reports "inside or outside" rather
    than a signed count gets one of them wrong.
    """
    def gamma(t):
        th = 2 * np.pi * t
        return scale * np.sin(th) * (1.0 + 1j * np.cos(th))

    def dgamma(t):
        th = 2 * np.pi * t
        return scale * 2 * np.pi * (np.cos(th) + 1j * np.cos(2 * th))

    return SmoothContour(gamma, dgamma, f"figure_eight(scale={scale:g})")


def limacon(inner=0.5, outer=1.0):
    """r(theta) = inner + outer cos(theta): a closed curve with a small inner loop.

    When inner < outer the radius goes negative over part of the sweep, which
    folds the curve back through the origin and creates an inner loop. Points
    inside that loop are wound twice. Useful because the curve looks perfectly
    ordinary -- no visible self-crossing at a glance -- yet has two distinct
    winding regions, which is exactly the case a naive inside/outside test gets
    wrong.
    """
    def gamma(t):
        th = 2 * np.pi * t
        return (inner + outer * np.cos(th)) * np.exp(1j * th)

    def dgamma(t):
        th = 2 * np.pi * t
        r = inner + outer * np.cos(th)
        return 2 * np.pi * (-outer * np.sin(th) + 1j * r) * np.exp(1j * th)

    return SmoothContour(gamma, dgamma, f"limacon(inner={inner:g}, outer={outer:g})")
