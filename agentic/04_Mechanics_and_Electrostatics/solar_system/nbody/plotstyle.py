"""Shared figure styling: colors, axis style, saving. Scripts only save files, so the backend is Agg."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Sequential: one hue, light to dark (for magnitudes such as initial radius).
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#bcd7f5", "#2a78d6", "#0b2f5c"])


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)
    print(f"saved {folder / name}")


def blackbody_rgb(T):
    """Display color (sRGB, max component 1) of a blackbody at temperature T [K].

    Integrates the Planck spectrum against the CIE 1931 color-matching functions (multi-lobe Gaussian fit of
    Wyman, Sloan & Shirley 2013), converts XYZ to linear sRGB, clips out-of-gamut values, and gamma-encodes.
    """
    import numpy as np
    lam = np.linspace(380.0, 780.0, 401)                                     # nm

    def lobe(mu, s1, s2):
        s = np.where(lam < mu, s1, s2)
        return np.exp(-0.5 * ((lam - mu) / s) ** 2)

    xbar = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    ybar = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    zbar = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    planck = 1.0 / (lam**5 * np.expm1(1.4388e7 / (lam * T)))               # hc/k = 1.4388e7 nm K
    XYZ = np.array([np.sum(planck * c) for c in (xbar, ybar, zbar)])
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]) @ XYZ
    rgb = np.clip(rgb, 0, None)
    rgb /= rgb.max()
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
