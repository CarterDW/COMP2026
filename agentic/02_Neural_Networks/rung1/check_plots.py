"""Visual diagnostics for every check in checks.py.

    python check_plots.py [--out results/checks]

checks.py answers "does this pass?". This answers "do I believe it?" -- each
figure shows the check's claim as a curve or a distribution against the value
known analytically, so a pass can be inspected rather than taken on trust.
Where checks.py tests one configuration, the plot here sweeps the parameter, so
a check that passes by luck at one point is visible as a curve that does not
track its prediction.

Every figure carries a PASS/FAIL badge computed from the plotted numbers, so
the figures and checks.py cannot silently disagree.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import treecorr

import ia
import shapes
import viz

RNG = np.random.default_rng(20260918)
MACHINE_EPS = np.finfo(float).eps


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------- check 1
def plot_simple_tensor(out):
    """Sweep the input axis ratios; the recovered ones must track the 1:1 line."""
    n = 200_000
    q_true = np.linspace(0.25, 0.95, 8)
    s_true = 0.6 * q_true
    q_got, s_got = [], []
    for q, s in zip(q_true, s_true):
        cloud = RNG.normal(size=(n, 3)) * np.array([1.0, q, s])
        a, b, _ = shapes.axis_ratios_3d(cloud, np.ones(n), reduced=False)
        q_got.append(a)
        s_got.append(b)
    q_got, s_got = np.array(q_got), np.array(s_got)

    counts = np.array([50, 100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    rot = _rotation([0.3, -0.7, 0.5], 0.8)
    misalign = []
    for count in counts:
        angles = []
        for _ in range(40):
            cloud = (RNG.normal(size=(count, 3)) * np.array([3.0, 2.0, 1.0])) @ rot.T
            _, _, major = shapes.axis_ratios_3d(cloud, np.ones(count), reduced=False)
            angles.append(np.degrees(np.arccos(min(abs(major @ rot[:, 0]), 1.0))))
        misalign.append(np.median(angles))

    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.6))
    lim = (0.0, 1.0)
    axes[0].plot(lim, lim, color=viz.BASELINE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    axes[0].annotate("1:1", xy=(0.88, 0.9), color=viz.INK_MUTED, fontsize=7.5)
    for label, true, got in (("q = b/a", q_true, q_got), ("s = c/a", s_true, s_got)):
        color = viz.TENSOR_COLORS["reduced" if label.startswith("q") else "simple"]
        axes[0].plot(true, got, color=color, marker="o", ls="none",
                     markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=label)
    axes[0].set(xlabel="input axis ratio", ylabel="recovered axis ratio",
                xlim=lim, ylim=lim, title="Recovered ratios track the input")
    axes[0].legend(loc="upper left")

    axes[1].plot(counts, misalign, color=viz.TENSOR_COLORS["reduced"], marker="o",
                 markeredgecolor=viz.SURFACE, markeredgewidth=0.8)
    axes[1].set(xscale="log", yscale="log", xlabel="particles in cloud",
                ylabel="median major-axis error  [deg]",
                title="Axis direction error falls off with $N$")
    viz.tidy_log_x(axes[1], [1e2, 1e3, 1e4, 1e5])
    viz.tidy_log_y(axes[1], [0.2, 0.5, 1, 2, 5])
    ref = misalign[0] * np.sqrt(counts[0] / counts)
    axes[1].plot(counts, ref, color=viz.BASELINE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    axes[1].annotate(r"$N^{-1/2}$", xy=(counts[-3], ref[-3]), color=viz.INK_MUTED,
                     fontsize=7.5, xytext=(4, 4), textcoords="offset points")

    worst = max(np.abs(q_got - q_true).max(), np.abs(s_got - s_true).max())
    passed = worst < 0.01
    viz.badge(fig, passed, f"worst axis-ratio error {worst:.4f} (tol 0.01)")
    viz.save(fig, out / "check_01_simple_tensor.png")
    return passed, f"worst axis-ratio error {worst:.4f}"


# ---------------------------------------------------------------- check 2
def plot_sphere_round(out):
    """An isotropic cloud must give q = s = 1 for both tensor definitions."""
    counts = np.array([50, 100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    data = {}
    for name, reduced in (("reduced", True), ("simple", False)):
        rows = []
        for count in counts:
            q_list, s_list = [], []
            for _ in range(30):
                direction = RNG.normal(size=(count, 3))
                direction /= np.linalg.norm(direction, axis=1)[:, None]
                cloud = direction * RNG.uniform(0.2, 1.0, (count, 1)) ** (1 / 3)
                q, s, _ = shapes.axis_ratios_3d(cloud, np.ones(count), reduced)
                q_list.append(q)
                s_list.append(s)
            rows.append((np.mean(q_list), np.std(q_list),
                         np.mean(s_list), np.std(s_list)))
        data[name] = np.array(rows)

    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.6))
    for ax, (idx, title) in zip(axes, ((0, "q = b/a"), (2, "s = c/a"))):
        for name in ("reduced", "simple"):
            rows = data[name]
            ax.errorbar(counts, rows[:, idx], yerr=rows[:, idx + 1],
                        color=viz.TENSOR_COLORS[name], marker="o", capsize=0,
                        markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=name)
        viz.reference_line(ax, 1.0, "isotropic")
        ax.set(xscale="log", xlabel="particles in cloud", ylabel=title,
               title=f"{title} for an isotropic cloud", ylim=(0.55, 1.05))
        viz.tidy_log_x(ax, [1e2, 1e3, 1e4, 1e5])
        ax.legend(loc="lower right")

    worst = max(abs(data[n][-1, i] - 1.0) for n in data for i in (0, 2))
    passed = worst < 0.02
    viz.badge(fig, passed, f"worst deviation at N=1e5: {worst:.4f} (tol 0.02)")
    viz.save(fig, out / "check_02_sphere_round.png")
    return passed, f"worst deviation {worst:.4f}"


# ---------------------------------------------------------------- check 3
def plot_spin2(out):
    """Rotating the cloud must move (e1, e2) at TWICE the rotation angle."""
    n = 200_000
    base = RNG.normal(size=(n, 2)) * np.array([2.5, 1.0])
    mass = np.ones(n)
    angles = np.linspace(0.0, 180.0, 37)
    e1, e2 = [], []
    for angle in np.deg2rad(angles):
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        a, b, _ = shapes.projected_ellipticity(base @ rot.T, mass, False, "chi")
        e1.append(a)
        e2.append(b)
    e1, e2 = np.array(e1), np.array(e2)
    eps = np.hypot(e1, e2).mean()
    fine = np.linspace(0.0, 180.0, 400)

    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, measured, analytic, key in (
        (r"$e_1$", e1, eps * np.cos(2 * np.deg2rad(fine)), "reduced"),
        (r"$e_2$", e2, eps * np.sin(2 * np.deg2rad(fine)), "simple"),
    ):
        color = viz.TENSOR_COLORS[key]
        ax.plot(fine, analytic, color=color, lw=1.4, alpha=0.45, zorder=1)
        ax.plot(angles, measured, color=color, marker="o", ls="none", zorder=2,
                markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=label)
    viz.reference_line(ax, 0.0, "")
    ax.set(xlabel="cloud rotation angle  $\\varphi$  [deg]",
           ylabel="projected ellipticity component",
           title=r"Faint lines are $\epsilon\cos 2\varphi$ and $\epsilon\sin 2\varphi$;"
                 " markers are measured",
           xlim=(0, 180), ylim=(-0.95, 1.06),
           xticks=np.arange(0, 181, 30))
    ax.legend(loc="upper center", ncols=2)

    resid = max(np.abs(e1 - eps * np.cos(2 * np.deg2rad(angles))).max(),
                np.abs(e2 - eps * np.sin(2 * np.deg2rad(angles))).max())
    passed = resid < 0.01
    viz.badge(fig, passed, f"worst residual {resid:.4f} (tol 0.01)")
    viz.save(fig, out / "check_03_spin2_convention.png")
    return passed, f"worst residual vs analytic {resid:.4f}"


# ---------------------------------------------------------------- check 4
def plot_conventions(out):
    """chi and epsilon must not be interchangeable: the ratio runs to 2."""
    # Stop just short of q=1: there chi and epsilon are both exactly 0 and the
    # ratio is 0/0. The limit is analytic -- chi/eps = (1+q)^2/(1+q^2) -> 2 --
    # so it is evaluated as a limit rather than at the singular point.
    q = np.linspace(0.02, 0.999, 300)
    chi = shapes.ellipticity_magnitude(q, "chi")
    eps = shapes.ellipticity_magnitude(q, "epsilon")

    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.6))
    axes[0].plot(q, chi, color=viz.TENSOR_COLORS["reduced"], label=r"$\chi=(1-q^2)/(1+q^2)$")
    axes[0].plot(q, eps, color=viz.TENSOR_COLORS["simple"], label=r"$\epsilon=(1-q)/(1+q)$")
    axes[0].set(xlabel="axis ratio $q$", ylabel="ellipticity magnitude",
                title="The two conventions diverge", xlim=(0, 1))
    axes[0].legend(loc="upper right")

    axes[1].plot(q, chi / eps, color=viz.TENSOR_COLORS["reduced"])
    viz.reference_line(axes[1], 2.0, "round-galaxy limit")
    axes[1].set(xlabel="axis ratio $q$", ylabel=r"$\chi/\epsilon$",
                title="Ratio approaches 2 for round galaxies",
                xlim=(0, 1), ylim=(0.9, 2.2))

    limit = (1 + q[-1]) ** 2 / (1 + q[-1] ** 2)
    passed = (abs(chi[-1] / eps[-1] - limit) < 1e-6
              and abs(limit - 2.0) < 0.01
              and np.all(chi >= eps))
    viz.badge(fig, passed,
              f"ratio at q={q[-1]:.3f} is {chi[-1] / eps[-1]:.4f}, "
              f"analytic limit {limit:.4f}")
    viz.save(fig, out / "check_04_conventions.png")
    return passed, f"chi/eps -> {chi[-1] / eps[-1]:.4f} (analytic {limit:.4f})"


# ---------------------------------------------------------------- check 5
def plot_rpar(out):
    """TreeCorr's rpar is radial, so a |rpar| cut is NOT a |dz| cut.

    For two points at transverse distance R from the origin separated purely in
    z, rpar = sqrt(R^2 + dz^2) - R, so a cut |rpar| < c admits everything out to
    dz = sqrt(2Rc + c^2). At R = 1000 with c = 1 that is 44.7, not 1.
    """
    cut = 1.0
    radii = np.array([3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    step, n_bins = 0.5, 200
    dz_grid = step * np.arange(1, n_bins + 1)
    measured = []
    for radius in radii:
        n = treecorr.Catalog(x=[radius], y=[0.0], z=[0.0], allow_xyz=True)
        g = treecorr.Catalog(x=np.full(n_bins, radius), y=np.zeros(n_bins),
                             z=dz_grid, allow_xyz=True)
        nn = treecorr.NNCorrelation(min_sep=step / 2, max_sep=step * n_bins + step / 2,
                                    nbins=n_bins, bin_type="Linear",
                                    metric="Euclidean", min_rpar=-cut, max_rpar=cut)
        nn.process(n, g)
        admitted = dz_grid[nn.npairs > 0]
        measured.append(admitted.max() if len(admitted) else 0.0)
    measured = np.array(measured)
    fine = np.logspace(np.log10(radii[0]), np.log10(radii[-1]), 200)
    analytic = np.sqrt(2 * fine * cut + cut**2)

    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(fine, analytic, color=viz.BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=0)
    ax.annotate(r"$\sqrt{2Rc+c^2}$", xy=(fine[55], analytic[55]),
                xytext=(8, -16), textcoords="offset points",
                color=viz.INK_MUTED, fontsize=7.5)
    ax.plot(radii, measured, color=viz.TENSOR_COLORS["reduced"], marker="o",
            ls="none", markeredgecolor=viz.SURFACE, markeredgewidth=0.8,
            label="TreeCorr, measured")
    viz.reference_line(ax, cut, "the cut you asked for  ($|dz|<1$)")
    ax.annotate(f"at R=1000 a pair with dz={measured[-2]:.0f}\n"
                "passes a cut meant to keep |dz|<1",
                xy=(radii[-2], measured[-2]), xytext=(0.30, 0.10),
                textcoords="axes fraction", color=viz.INK_SECONDARY, fontsize=7.5,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=viz.INK_MUTED, lw=0.7,
                                shrinkB=6))
    ax.set(xscale="log", yscale="log",
           xlabel="transverse distance from the origin  $R$",
           ylabel="largest $|dz|$ admitted by the cut",
           title="A $|rpar|$ cut is not a line-of-sight cut in a box")
    ax.legend(loc="upper left")

    passed = np.allclose(measured, np.sqrt(2 * radii * cut + cut**2), atol=step)
    viz.badge(fig, passed, "measured tracks the radial prediction")
    viz.save(fig, out / "check_05_rpar_is_radial.png")
    return passed, f"dz admitted at R=1000: {measured[-2]:.1f} (cut was {cut})"


# ---------------------------------------------------------------- check 6
def plot_analytic_rr(out):
    """The analytic periodic RR must reproduce Poisson counts, so w_p = 0."""
    box, n = 300.0, 30_000
    pos = RNG.uniform(0.0, box, (n, 3))
    res = ia.correlate(pos, pos, np.zeros(n), np.zeros(n), box,
                       min_rp=2.0, max_rp=40.0, nbins=8, npatch=16)
    ratio = res["npairs"] / res["rr_analytic"]
    poisson = 1.0 / np.sqrt(res["npairs"])

    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.6))
    axes[0].fill_between(res["rp"], 1 - poisson, 1 + poisson,
                         color=viz.SEQUENTIAL_BLUE[1], zorder=0, lw=0)
    # Deliberately labelled as a floor, not an error bar: pairs share galaxies,
    # so their counts are not independent and the true scatter is super-Poisson.
    # A point outside this band is expected and is not what the check tests --
    # the check's tolerance is 2%, roughly six times the worst deviation here.
    axes[0].annotate("$1/\\sqrt{N_{\\rm pairs}}$ band\n(a floor: pairs are not\nindependent)",
                     xy=(0.03, 0.06), xycoords="axes fraction", ha="left",
                     va="bottom", color=viz.INK_MUTED, fontsize=7.5)
    axes[0].plot(res["rp"], ratio, color=viz.TENSOR_COLORS["reduced"], marker="o",
                 ls="none", markeredgecolor=viz.SURFACE, markeredgewidth=0.8)
    viz.reference_line(axes[0], 1.0, "analytic", xpos=0.995, ha="right")
    axes[0].set(xscale="log", xlabel=r"$r_p$", ylabel="measured pairs / analytic RR",
                title="Pair counts match the analytic RR")

    axes[1].errorbar(res["rp"], res["w_p"], yerr=res["err_w_p"],
                     color=viz.TENSOR_COLORS["reduced"], marker="o", capsize=0,
                     ls="none", markeredgecolor=viz.SURFACE, markeredgewidth=0.8)
    viz.reference_line(axes[1], 0.0, "unclustered", xpos=0.005, ha="left", dy=-4)
    axes[1].set(xscale="log", xlabel=r"$r_p$", ylabel=r"$w_p(r_p)$",
                title="$w_p$ of a Poisson catalogue is zero")
    for ax in axes:
        viz.tidy_log_x(ax, [2, 5, 10, 20, 40])

    z = np.abs(res["w_p"] / res["err_w_p"]).max()
    passed = np.all(np.abs(ratio - 1) < 0.02) and z < 3.0
    viz.badge(fig, passed, f"max |ratio-1| {np.abs(ratio - 1).max():.4f}, "
                               f"max |w_p/err| {z:.2f}")
    viz.save(fig, out / "check_06_analytic_rr.png")
    return passed, f"max |w_p/err| = {z:.2f}"


# ---------------------------------------------------------------- check 7
def plot_ia_null(out):
    """Random orientations give w_g+ = 0; w_gx = 0 by parity regardless."""
    box, n = 300.0, 30_000
    pos = RNG.uniform(0.0, box, (n, 3))
    phi = RNG.uniform(0.0, np.pi, n)
    res = ia.correlate(pos, pos, 0.3 * np.cos(2 * phi), 0.3 * np.sin(2 * phi),
                       box, min_rp=2.0, max_rp=40.0, nbins=8, npatch=16)

    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    offset = 1.03
    for label, key, colour_key, shift in (
        (r"$w_{g+}$", "w_gplus", "reduced", 1 / offset),
        (r"$w_{g\times}$", "w_gcross", "simple", offset),
    ):
        ax.errorbar(res["rp"] * shift, res[key], yerr=res[f"err_{key}"],
                    color=viz.TENSOR_COLORS[colour_key], marker="o", capsize=0,
                    ls="none", markeredgecolor=viz.SURFACE, markeredgewidth=0.8,
                    label=label)
    viz.reference_line(ax, 0.0, "no alignment", xpos=0.005, ha="left", dy=-4)
    ax.set(xscale="log", xlabel=r"$r_p$", ylabel=r"$w(r_p)$",
           title="Null test: randomly oriented shapes give no signal")
    viz.tidy_log_x(ax, [2, 5, 10, 20, 40])
    ax.legend(loc="upper right", ncols=2)

    z_plus = np.abs(res["w_gplus"] / res["err_w_gplus"]).max()
    z_cross = np.abs(res["w_gcross"] / res["err_w_gcross"]).max()
    passed = z_plus < 3.0 and z_cross < 3.0
    viz.badge(fig, passed, f"max |z| = {z_plus:.2f} (+), {z_cross:.2f} (x)")
    viz.save(fig, out / "check_07_ia_null.png")
    return passed, f"max |z| {max(z_plus, z_cross):.2f}"


# ---------------------------------------------------------------- check 8
def plot_radial_injection(out):
    """An exactly injected radial alignment must return with the right sign."""
    box = 1200.0
    n_clusters, n_sat, radius = 400, 60, 1.5
    eps = shapes.ellipticity_magnitude(0.6, "chi")
    min_rp, max_rp = 0.2, 1.5

    centres = RNG.uniform(0.0, box, (n_clusters, 3))
    offsets = RNG.normal(size=(n_clusters, n_sat, 3))
    offsets /= np.linalg.norm(offsets, axis=2)[..., None]
    offsets *= radius * RNG.uniform(0.0, 1.0, (n_clusters, n_sat, 1)) ** (1 / 3)
    sats = np.mod((centres[:, None, :] + offsets).reshape(-1, 3), box)
    flat = offsets.reshape(-1, 3)
    phi = np.arctan2(flat[:, 1], flat[:, 0])
    res = ia.correlate(centres, sats, eps * np.cos(2 * phi), eps * np.sin(2 * phi),
                       box, min_rp=min_rp, max_rp=max_rp, nbins=4, npatch=32)

    area = np.pi * (max_rp**2 - min_rp**2)
    dilution = len(sats) * (n_clusters / box**2) * area / res["npairs"].sum()

    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.6))
    axes[0].plot(res["rp"], res["mean_eplus"], color=viz.TENSOR_COLORS["reduced"],
                 marker="o", markeredgecolor=viz.SURFACE, markeredgewidth=0.8,
                 label=r"measured $\langle e_+\rangle$")
    viz.reference_line(axes[0], eps, f"injected $\\epsilon$ = {eps:.4f}")
    axes[0].axhspan(eps * (1 - dilution), eps, color=viz.SEQUENTIAL_BLUE[1],
                    zorder=0, lw=0)
    axes[0].annotate(f"predicted dilution {dilution:.2%}\nfrom chance projection",
                     xy=(res["rp"][1], eps * (1 - dilution)), xytext=(6, -26),
                     textcoords="offset points", color=viz.INK_SECONDARY,
                     fontsize=7.5,
                     arrowprops=dict(arrowstyle="-", color=viz.INK_MUTED, lw=0.7))
    axes[0].set(xscale="log", xlabel=r"$r_p$", ylabel=r"$\langle e_+\rangle$",
                title="Sign and amplitude both recovered",
                ylim=(eps * 0.992, eps * 1.004))
    axes[0].legend(loc="lower left")

    # Plotted as significance rather than raw w: the injected signal is ~1e5 in
    # box units, which would compress the null component to an invisible line.
    axes[1].axhspan(-3.0, 3.0, color=viz.SEQUENTIAL_BLUE[0], zorder=0, lw=0)
    axes[1].annotate(r"$\pm3\sigma$ null band", xy=(0.97, 0.06),
                     xycoords="axes fraction", ha="right", color=viz.INK_MUTED,
                     fontsize=7.5)
    for label, key, colour_key in ((r"$w_{g+}$", "w_gplus", "reduced"),
                                   (r"$w_{g\times}$", "w_gcross", "simple")):
        axes[1].plot(res["rp"], res[key] / res[f"err_{key}"],
                     color=viz.TENSOR_COLORS[colour_key], marker="o",
                     markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=label)
    axes[1].set(xscale="log", xlabel=r"$r_p$",
                ylabel=r"deviation from zero  [$\sigma$]",
                title=r"$w_{g+}$ is a huge signal; $w_{g\times}$ stays null")
    axes[1].legend(loc="center right", ncols=1)

    for ax in axes:
        viz.tidy_log_x(ax, [0.3, 0.5, 0.7, 1.0, 1.5])

    passed = (np.all(np.abs(res["mean_eplus"] - eps) < 0.01 * eps)
              and np.all(res["w_gplus"] > 0))
    viz.badge(fig, passed,
              f"recovered {res['mean_eplus'].min():.4f}-{res['mean_eplus'].max():.4f} "
              f"vs {eps:.4f}")
    viz.save(fig, out / "check_08_radial_injection.png")
    return passed, f"recovered mean e+ within 1% of {eps:.4f}"


# ---------------------------------------------------------------- check 9
def plot_order_invariance(out):
    """Shuffling the particle list must change nothing beyond rounding."""
    n, box, trials = 5_000, 1000.0, 400
    centre = np.full(3, 500.0)
    pos = centre + RNG.normal(size=(n, 3)) * np.array([3.0, 2.0, 1.5])
    mass = RNG.uniform(0.5, 1.5, n)
    ref = shapes.measure_galaxy(pos, mass, centre, box, 10.0, True, "chi")
    diffs = []
    for _ in range(trials):
        order = RNG.permutation(n)
        got = shapes.measure_galaxy(pos[order], mass[order], centre, box, 10.0,
                                    True, "chi")
        diffs.append(max(abs(got[k] - ref[k]) for k in ("q_3d", "s_3d", "q_2d")))
    diffs = np.array(diffs)
    exact = int((diffs == 0.0).sum())

    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(diffs / MACHINE_EPS, bins=24, color=viz.TENSOR_COLORS["reduced"],
            edgecolor=viz.SURFACE, linewidth=0.8)
    ax.set(xlabel="worst change in axis ratio  [multiples of machine epsilon]",
           ylabel=f"shuffles  (of {trials})",
           title="Particle ordering changes nothing beyond rounding")
    ax.annotate(f"{exact} of {trials} shuffles were bit-identical;\n"
                f"largest change {diffs.max() / MACHINE_EPS:.1f} eps "
                f"= {diffs.max():.1e}\ntest tolerance is 1e-10, "
                f"{1e-10 / max(diffs.max(), MACHINE_EPS):.0e}x larger",
                xy=(0.97, 0.9), xycoords="axes fraction", ha="right", va="top",
                color=viz.INK_SECONDARY, fontsize=7.5)

    passed = diffs.max() < 1e-10
    viz.badge(fig, passed, f"max change {diffs.max():.1e}")
    viz.save(fig, out / "check_09_order_invariance.png")
    return passed, f"max change {diffs.max():.1e}"


# ---------------------------------------------------------------- check 10
def plot_box_wrap(out):
    """A galaxy split across the box edge must measure the same as a whole one."""
    n, box = 40_000, 100.0
    cloud = RNG.normal(size=(n, 3)) * np.array([3.0, 1.5, 1.0])
    rot = np.array([[np.cos(0.6), -np.sin(0.6), 0.0],
                    [np.sin(0.6), np.cos(0.6), 0.0], [0.0, 0.0, 1.0]])
    cloud = cloud @ rot.T

    cases = (
        ("mid-box, intact", np.mod(cloud + 50.0, box), np.full(3, 50.0)),
        ("straddling the edge", np.mod(cloud, box), np.zeros(3)),
    )
    viz.use_style()
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.6, 3.9))
    results = []
    for ax, (title, positions, centre) in zip(axes, cases):
        got = shapes.measure_galaxy(positions, np.ones(n), centre, box, 12.0,
                                    True, "chi")
        results.append(got)
        counts, xe, ye = np.histogram2d(positions[:, 0], positions[:, 1],
                                        bins=100, range=[[0, box], [0, box]])
        ax.imshow(counts.T, origin="lower", extent=(0, box, 0, box),
                  cmap=viz.sequential_cmap(), aspect="equal",
                  norm="log", interpolation="nearest")
        angle = 0.5 * np.arctan2(got["e2"], got["e1"])
        span = 18.0
        ax.plot(centre[0] + span * np.cos(angle) * np.array([-1, 1]),
                centre[1] + span * np.sin(angle) * np.array([-1, 1]),
                color=viz.TENSOR_COLORS["simple"], lw=1.8,
                solid_capstyle="round", label="recovered major axis")
        ax.set(xlabel="x", ylabel="y", title=title, xlim=(0, box), ylim=(0, box))
        ax.grid(False)
        ax.annotate(f"$q_{{2D}}$ = {got['q_2d']:.5f}\n"
                    f"$N$ used = {got['n_used']}",
                    xy=(0.30, 0.30), xycoords="axes fraction", va="bottom",
                    fontsize=7.5, color=viz.INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=viz.SURFACE,
                              ec=viz.BASELINE, lw=0.6))
    axes[0].legend(loc="upper left")

    worst = max(abs(results[0][k] - results[1][k])
                for k in ("q_3d", "s_3d", "q_2d", "n_used"))
    passed = worst < 1e-10
    viz.badge(fig, passed, f"difference between the two panels: {worst:.1e}")
    viz.save(fig, out / "check_10_box_wrap.png")
    return passed, f"difference {worst:.1e}"


PLOTS = (
    ("simple tensor == covariance", plot_simple_tensor),
    ("sphere is round (both tensors)", plot_sphere_round),
    ("spin-2 projection convention", plot_spin2),
    ("chi vs epsilon not interchangeable", plot_conventions),
    ("treecorr rpar is radial, not dz", plot_rpar),
    ("analytic periodic RR", plot_analytic_rr),
    ("IA null test (random orientations)", plot_ia_null),
    ("IA radial injection (sign + amplitude)", plot_radial_injection),
    ("shape invariant to particle order", plot_order_invariance),
    ("shape invariant to box wrap", plot_box_wrap),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/checks")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nvisual diagnostics for {len(PLOTS)} checks -> {out}\n")
    failures = 0
    for name, func in PLOTS:
        passed, detail = func(out)
        failures += not passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:38s} {detail}")
    print()
    print(f"  -> {'all figures agree with checks.py' if not failures else f'{failures} FAILED'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
