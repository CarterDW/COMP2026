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
