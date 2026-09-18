"""Figure style for the rung-1 outputs.

Colours are not chosen by eye. The three resolution levels are an *ordered*
series, so they take a single-hue blue ordinal ramp rather than categorical
hues; the two tensor definitions are unordered, so they take the first two
categorical slots. Both sets were run through the palette validator
(monotone lightness, visible step gaps, light end clearing the surface, single
hue; and for the categorical pair, the lightness band, chroma floor, CVD
separation and normal-vision floor under all pairs) and pass in both light and
dark modes.
"""

import matplotlib

# Every figure here is written to disk, never shown, and this runs headless
# under WSL -- so pin the non-interactive backend before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Ordinal ramp, light -> dark, for TNG300-1 / -2 / -3. Assigned by resolution,
# never by plotting order: the highest-resolution run is always the darkest.
RESOLUTION_COLORS = {
    "TNG300-3": "#86b6ef",
    "TNG300-2": "#2a78d6",
    "TNG300-1": "#104281",
}

# Categorical slots 1 and 2, for the two inertia-tensor definitions.
TENSOR_COLORS = {"reduced": "#2a78d6", "simple": "#eb6834"}

# Sequential single-hue blue ramp, light -> dark, for continuous magnitude
# (particle-density renderings). Sequential encoding is allowed to use the pale
# end, which is why it starts lighter than the ordinal ramp above.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

# Status palette -- reserved, never reused as a series colour, and always
# shipped with a text label so state is never carried by colour alone.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def use_style():
    """Apply the recessive-chrome style: hairline grid, no top/right spines."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        # Grid must sit BEHIND the data; it drew over histogram bars without this.
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def label_at_end(ax, x, y, text, color):
    """Direct label just past the last point, in the series colour.

    Used instead of relying on the legend alone when there are few series --
    identity should never be carried by colour by itself.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
        color=color, fontsize=8, va="center", fontweight="semibold",
    )


def reference_line(ax, value, text, xpos=0.995, ha="right", dy=3):
    """A muted horizontal reference line -- chrome, not a data series.

    ``xpos``/``ha`` exist because the default right-hand corner is exactly
    where a null test's data sits; move the label rather than let it collide.
    """
    ax.axhline(value, color=BASELINE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.annotate(
        text, xy=(xpos, value), xycoords=("axes fraction", "data"),
        xytext=(0, dy), textcoords="offset points",
        color=INK_MUTED, fontsize=7.5, ha=ha, va="bottom" if dy >= 0 else "top",
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def sequential_cmap():
    """Matplotlib colormap from the sequential blue ramp, for density plots."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("viz_blue", SEQUENTIAL_BLUE)


def badge(fig, passed, detail=""):
    """PASS/FAIL stamp in the figure footer.

    Placed on the figure rather than an axes: an in-axes badge collides with
    titles and data, which it did in every panel of the first render. The word
    is always drawn, so the status colour is decoration and not the carrier of
    the result.
    """
    fig.text(
        0.005, -0.012,
        f"{'PASS' if passed else 'FAIL'}{'  -  ' + detail if detail else ''}",
        color=STATUS_GOOD if passed else STATUS_CRITICAL,
        fontsize=8, fontweight="bold", va="top", ha="left",
    )


def tidy_log_y(ax, ticks):
    """Explicit y ticks with plain labels, for log axes under ~2 decades."""
    from matplotlib.ticker import NullFormatter

    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def tidy_log_x(ax, ticks):
    """Explicit x ticks with plain labels, for log axes spanning under a decade.

    Matplotlib's default minor ticks collide when the range is narrow (the
    "3x10^-1 4x10^-1" pileup), so the tick set is stated rather than inferred.
    """
    from matplotlib.ticker import NullFormatter

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
