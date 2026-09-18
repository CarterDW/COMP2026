"""Rung 1, part A: how many star particles does a usable galaxy shape need?

This driver needs only TNG300-1. It takes well-resolved galaxies, measures each
one's shape from all of its star particles (the reference), then re-measures it
from random subsets of N particles and asks how the estimate degrades. That
isolates *shot noise in the shape estimator* from any physical change caused by
running the simulation at lower resolution -- which is what part B measures. The
two effects are separated here on purpose, because a single cross-resolution
comparison confounds them.

The headline number is the dilution factor

    D(N) = <eps_N cos 2 dphi> / <eps_ref>

which is exactly the multiplicative bias that propagates into w_g+ at leading
order: an IA amplitude measured from N-particle shapes is D(N) times the truth.
So D(N) at FLAMINGO-like particle counts answers, quantitatively, whether
stellar shapes are usable there at all.

Run ``python checks.py`` first. Nothing here is meaningful if those fail.
"""

import argparse
from pathlib import Path

import numpy as np

import shapes
import tng
import viz

TARGETS = np.array([20, 30, 50, 80, 130, 200, 350, 600, 1000, 2000, 4000])


def measure_one(coords, masses, centre, boxsize, aperture, rng, reduced, convention):
    """Reference shape plus a downsampled shape at each target particle count."""
    offsets = shapes.offsets_periodic(coords, centre, boxsize)
    inside = np.einsum("ij,ij->i", offsets, offsets) <= aperture**2
    offsets, masses = offsets[inside], masses[inside]
    nonzero = np.einsum("ij,ij->i", offsets, offsets) > 0.0
    offsets, masses = offsets[nonzero], masses[nonzero]

    n_total = len(masses)
    origin = np.zeros(3)
    reference = shapes.measure_galaxy(
        offsets, masses, origin, boxsize, aperture, reduced, convention
    )

    rows = []
    for target in TARGETS[TARGETS <= n_total]:
        pick = rng.choice(n_total, size=int(target), replace=False)
        sub = shapes.measure_galaxy(
            offsets[pick], masses[pick], origin, boxsize, aperture, reduced, convention
        )
        rows.append((target, sub["q_2d"], sub["e1"], sub["e2"], sub["q_3d"], sub["s_3d"]))
    return n_total, reference, rows


def dilution(e1_n, e2_n, e1_ref, e2_ref):
    """D = <eps_N cos 2dphi> / <eps_ref>, the multiplicative bias on w_g+."""
    eps_ref = np.hypot(e1_ref, e2_ref)
    projected = (e1_n * e1_ref + e2_n * e2_ref) / eps_ref
    return projected.mean() / eps_ref.mean()


def bootstrap_dilution(e1_n, e2_n, e1_ref, e2_ref, rng, n_boot=200):
    n = len(e1_ref)
    draws = [
        dilution(*(arr[rng.integers(0, n, n)] for arr in (e1_n, e2_n, e1_ref, e2_ref)))
        for _ in range(n_boot)
    ]
    return float(np.std(draws))


def misalignment_deg(e1_n, e2_n, e1_ref, e2_ref):
    """Position-angle error in degrees, folded into [0, 45]."""
    angle = 0.5 * np.arctan2(
        e1_ref * e2_n - e2_ref * e1_n, e1_ref * e1_n + e2_ref * e2_n
    )
    return np.abs(np.degrees(angle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="TNG300-1")
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--cache", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n-galaxies", type=int, default=1500)
    parser.add_argument("--min-stellar-mass", type=float, default=1e10,
                        help="Msun, not Msun/h")
    parser.add_argument("--min-star-particles", type=int, default=6000,
                        help="headroom so the largest downsample target is not "
                             "the whole galaxy")
    parser.add_argument("--aperture-halfmass", type=float, default=2.0,
                        help="aperture in units of the stellar half-mass radius")
    parser.add_argument("--convention", default="chi", choices=("chi", "epsilon"))
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    boxsize = tng.SIMS[args.sim]["boxsize_ckpc"]
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{args.sim} snapshot {args.snap}: loading subhalo fields")
    catalogue = tng.subhalo_catalogue(args.sim, args.snap, args.cache)
    eligible = tng.select_galaxies(
        catalogue, args.min_stellar_mass, args.min_star_particles
    )
    print(f"  {len(eligible)} subhalos pass M* > {args.min_stellar_mass:.1e} Msun "
          f"and N_star > {args.min_star_particles}")
    if len(eligible) == 0:
        raise RuntimeError("no galaxies pass the cuts; loosen them")

    sample = rng.choice(eligible, size=min(args.n_galaxies, len(eligible)),
                        replace=False)
    print(f"  measuring {len(sample)} galaxies (downloading cutouts on first pass)")

    records = {reduced: [] for reduced in (True, False)}
    for count, subhalo_id in enumerate(sample, start=1):
        coords, masses = tng.star_cutout(args.sim, args.snap, int(subhalo_id),
                                         args.cache)
        centre = catalogue["SubhaloPos"][subhalo_id]
        aperture = args.aperture_halfmass * catalogue["SubhaloHalfmassRadType"][
            subhalo_id, tng.STARS
        ]
        for reduced in (True, False):
            n_total, reference, rows = measure_one(
                coords, masses, centre, boxsize, aperture,
                np.random.default_rng(args.seed + int(subhalo_id)),
                reduced, args.convention,
            )
            for target, q2, e1, e2, q3, s3 in rows:
                records[reduced].append(
                    (subhalo_id, n_total, target, q2, e1, e2, q3, s3,
                     reference["q_2d"], reference["e1"], reference["e2"],
                     reference["q_3d"], reference["s_3d"])
                )
        if count % 100 == 0:
            print(f"    {count}/{len(sample)}")

    columns = ("subhalo_id", "n_aperture", "target", "q_2d", "e1", "e2", "q_3d",
               "s_3d", "q_2d_ref", "e1_ref", "e2_ref", "q_3d_ref", "s_3d_ref")
    tables = {k: np.array(v) for k, v in records.items()}
    np.savez(
        out_dir / "shape_noise.npz",
        columns=np.array(columns),
        reduced=tables[True], simple=tables[False],
        targets=TARGETS, convention=args.convention, sim=args.sim,
    )
    print(f"  wrote {out_dir / 'shape_noise.npz'}")

    summarise(tables, out_dir, args, rng)


def summarise(tables, out_dir, args, rng):
    """Collapse to D(N), misalignment and axis-ratio bias, then plot."""
    summary = {}
    for name, reduced in (("reduced", True), ("simple", False)):
        table = tables[reduced]
        target, q2, e1, e2 = table[:, 2], table[:, 3], table[:, 4], table[:, 5]
        q2_ref, e1_ref, e2_ref = table[:, 8], table[:, 9], table[:, 10]
        rows = []
        for value in TARGETS:
            m = target == value
            if m.sum() < 20:
                continue
            rows.append((
                value, int(m.sum()),
                dilution(e1[m], e2[m], e1_ref[m], e2_ref[m]),
                bootstrap_dilution(e1[m], e2[m], e1_ref[m], e2_ref[m], rng),
                np.median(misalignment_deg(e1[m], e2[m], e1_ref[m], e2_ref[m])),
                np.median(q2[m] - q2_ref[m]),
            ))
        summary[name] = np.array(rows)

    print("\n  dilution factor D(N) -- the multiplicative bias on w_g+")
    print(f"  {'N':>6}  {'reduced':>16}  {'simple':>16}")
    for i, value in enumerate(summary["reduced"][:, 0]):
        red = summary["reduced"][i]
        sim = summary["simple"][i]
        print(f"  {int(value):>6}  {red[2]:>8.3f} +- {red[3]:.3f}  "
              f"{sim[2]:>8.3f} +- {sim[3]:.3f}")

    np.savez(out_dir / "shape_noise_summary.npz", **summary)
    plot(summary, out_dir, args)


def plot(summary, out_dir, args):
    import matplotlib.pyplot as plt

    viz.use_style()

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name in ("reduced", "simple"):
        rows = summary[name]
        color = viz.TENSOR_COLORS[name]
        ax.errorbar(rows[:, 0], rows[:, 2], yerr=rows[:, 3], color=color,
                    marker="o", markeredgecolor=viz.SURFACE, markeredgewidth=0.8,
                    capsize=0, label=f"{name} tensor")
        viz.label_at_end(ax, rows[-1, 0], rows[-1, 2], name, color)
    viz.reference_line(ax, 1.0, "no dilution")
    ax.set_xscale("log")
    ax.set_xlabel("star particles used per galaxy")
    ax.set_ylabel("dilution factor  $D(N)$")
    ax.set_title("IA amplitude is suppressed by $D(N)$ at low particle count")
    ax.set_ylim(0.0, 1.12)
    ax.legend(loc="lower right")
    viz.save(fig, out_dir / "shape_noise_dilution.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for name in ("reduced", "simple"):
        rows = summary[name]
        color = viz.TENSOR_COLORS[name]
        axes[0].plot(rows[:, 0], rows[:, 4], color=color, marker="o",
                     markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=name)
        axes[1].plot(rows[:, 0], rows[:, 5], color=color, marker="o",
                     markeredgecolor=viz.SURFACE, markeredgewidth=0.8, label=name)
    axes[0].set_ylabel("median position-angle error  [deg]")
    axes[0].set_title("Orientation error")
    viz.reference_line(axes[0], 45.0 / 2.0, "half of random")
    axes[1].set_ylabel(r"median  $q_{2D}(N) - q_{2D}^{\rm ref}$")
    axes[1].set_title("Axis-ratio bias")
    viz.reference_line(axes[1], 0.0, "unbiased")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("star particles used per galaxy")
        ax.legend(loc="best")
    viz.save(fig, out_dir / "shape_noise_diagnostics.png")


if __name__ == "__main__":
    main()
