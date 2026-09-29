"""Shared chart chrome for the per-rung visual diagnostics.

Design tokens and the few helpers every figure uses. Both light and dark modes
are *selected*, not flipped: the dark series are the same hues re-stepped for
the dark surface.

Colour assignments follow the job the data does. Series identity takes at most
three categorical slots, which is the documented all-pairs-safe cap for this
palette. Ordered magnitudes (pole distance, sampling density) take the one-hue
ordinal ramp. Winding number takes a diverging scale, because it is a signed
quantity with a meaningful zero: two hues, equal steps per arm, a neutral gray
midpoint, and lightness monotone outward from that midpoint in both directions.
"""

import numpy as np

LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp=("#86b6ef", "#5598e7", "#2a78d6", "#184f95"),
    # Diverging, -3 .. +3. Negative arm blue, positive arm red, gray at zero.
    diverging=("#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a02525"),
    good="#0ca30c", bad="#d03b3b",
)
DARK = dict(
    surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    ramp=("#184f95", "#256abf", "#3987e5", "#86b6ef"),
    diverging=("#9ec5f4", "#3987e5", "#1c5cab", "#383835", "#8f3231", "#e34948", "#f4a6a5"),
    good="#0ca30c", bad="#d03b3b",
)


def new_figure(T, size=(7.2, 4.6), grid=True):
    """A single-axes figure with recessive chrome: hairline grid, muted axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, facecolor=T["surface"])
    style_axes(ax, T, grid)
    return fig, ax


def new_panels(T, ncols, size, grid=False):
    """A row of panels sharing one title block, for small multiples."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=size, facecolor=T["surface"])
    for ax in np.atleast_1d(axes):
        style_axes(ax, T, grid)
    return fig, np.atleast_1d(axes)


def style_axes(ax, T, grid=True):
    """Hairline solid grid, muted ticks, no top or right spine."""
    ax.set_facecolor(T["surface"])
    if grid:
        ax.grid(True, color=T["grid"], linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=T["muted"], labelsize=9, length=3, width=0.8)


def _badge(ok, badge):
    return f"{'PASS' if ok else 'FAIL'}  {badge}"


def _badge_box(T, ok):
    return dict(boxstyle="round,pad=0.35", fc=T["good"] if ok else T["bad"], ec="none")


def finish(fig, ax, T, title, setup, note, ok, badge, out):
    """Title row, setup line, note, PASS/FAIL badge, and save. Single-axes figures.

    `setup` states what is being computed and on what, and is mandatory. A
    reader should never have to open the source to find out which function and
    which contour a curve refers to; these figures get looked at one at a time.
    Title and badge share the title row, which matplotlib lays out as two
    separate artists, so a long setup line cannot run underneath the badge.
    """
    ax.set_title(title, loc="left", color=T["ink"], fontsize=12, pad=46)
    ax.set_title(_badge(ok, badge), loc="right", pad=46, color=T["surface"],
                 fontsize=9, weight="bold", bbox=_badge_box(T, ok))
    ax.text(0.0, 1.065, setup, transform=ax.transAxes, color=T["ink2"], fontsize=10, va="bottom")
    ax.text(0.0, 1.012, note, transform=ax.transAxes, color=T["muted"], fontsize=9, va="bottom")
    _legend_colors(ax, T)
    fig.tight_layout()
    return _save(fig, ok, out)


def finish_panels(fig, T, title, setup, note, ok, badge, out, top=0.84):
    """The same title block at figure level, for a row of panels."""
    fig.text(0.015, 0.975, title, color=T["ink"], fontsize=12, va="top", ha="left")
    fig.text(0.985, 0.975, _badge(ok, badge), color=T["surface"], fontsize=9, weight="bold",
             va="top", ha="right", bbox=_badge_box(T, ok))
    fig.text(0.015, 0.915, setup, color=T["ink2"], fontsize=10, va="top", ha="left")
    fig.text(0.015, 0.872, note, color=T["muted"], fontsize=9, va="top", ha="left")
    for ax in fig.axes:
        _legend_colors(ax, T)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, ok, out)


def _legend_colors(ax, T):
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor(T["surface"])
    legend.get_frame().set_edgecolor(T["grid"])
    for text in legend.get_texts():
        text.set_color(T["ink2"])


def _save(fig, ok, out):
    import matplotlib.pyplot as plt
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{'PASS' if ok else 'FAIL'}] {out.name}")
    return ok


def dots(ax, x, y, color, T, label=None, ms=6.5):
    """Markers with a surface-coloured ring, so overlapping points stay separable."""
    return ax.plot(x, y, "o", color=color, markersize=ms, label=label,
                   markeredgecolor=T["surface"], markeredgewidth=1.2, linestyle="none")
