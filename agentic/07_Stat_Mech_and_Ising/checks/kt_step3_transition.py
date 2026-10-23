"""KT warmup, step 3: the transition, from the stiffness, vortex density and specific heat.

    python checks/kt_step3_transition.py

L = 16 and 32. (L = 64 was dropped for time, as the plan allows.) Every T gets
16 replicas from an ordered start, with a 2000-sweep warmup and 6000
measurement sweeps, measured every 2 sweeps. Errors are from the scatter
between replicas.

Targets: the spin-wave stiffness at low T, Upsilon -> 0 at high T, the
Nelson-Kosterlitz crossing Upsilon = 2T/pi drifting down towards 0.893, and the
specific-heat peak above T_KT (Gupta & Baillie 1992). Plus an equilibration
test between ordered and random starts.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lattice
from checks._runner import run

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
T_KT = 0.8929
TS = np.unique(np.round(np.concatenate([np.arange(0.6, 1.41, 0.05), np.arange(0.8, 1.0, 0.02)]), 3))
R = 16
RES = {}


def measure(L, T, n_warm, n_meas, s, ordered=True, every=2):
    """Per-replica means of u, u^2, Upsilon and vortex density at temperatures T (one per replica)."""
    lattice.seed(s)
    beta = 1 / np.asarray(T, float)
    N = L * L
    th = np.zeros((len(beta), L, L)) if ordered else np.random.default_rng(s).uniform(0, 2 * np.pi, (len(beta), L, L))
    delta = lattice.tune_delta(th, beta)
    for _ in range(n_warm):
        lattice.xy_sweep(th, beta, delta)
    acc = {k: [] for k in ("u", "cx", "sx2", "cy", "sy2", "nv")}
    for n in range(n_meas):
        lattice.xy_sweep(th, beta, delta)
        if n % every:
            continue
        cx, sx, cy, sy = lattice.xy_bond_sums(th)
        acc["u"].append(-(cx + cy) / N)
        acc["cx"].append(cx); acc["sx2"].append(sx**2)
        acc["cy"].append(cy); acc["sy2"].append(sy**2)
        acc["nv"].append(np.abs(lattice.vortices(th)).mean(axis=(1, 2)))
    a = {k: np.array(v) for k, v in acc.items()}
    ups = 0.5 * ((a["cx"].mean(0) - beta * a["sx2"].mean(0)) + (a["cy"].mean(0) - beta * a["sy2"].mean(0))) / N
    C = N * a["u"].var(0) * beta**2
    return {"u": a["u"].mean(0), "C": C, "ups": ups, "nv": a["nv"].mean(0)}


def by_T(r, n_T):
    """Mean and standard error over the R replicas of each temperature."""
    x = r.reshape(n_T, -1)
    return x.mean(1), x.std(1, ddof=1) / np.sqrt(x.shape[1])


def sweep(L):
    if L not in RES:
        r = measure(L, np.repeat(TS, R), 2000, 6000, 10 + L)
        RES[L] = {k: by_T(v, len(TS)) for k, v in r.items()}
    return RES[L]


def crossing(L):
    """T where Upsilon_L(T) = 2T/pi, by linear interpolation between grid points."""
    ups = sweep(L)["ups"][0]
    g = ups - 2 * TS / np.pi
    k = np.nonzero((g[:-1] > 0) & (g[1:] <= 0))[0][0]
    return TS[k] + (TS[k + 1] - TS[k]) * g[k] / (g[k] - g[k + 1])


def check_low_T_stiffness():
    """Spin waves: Upsilon = 1 - T/4 + O(T^2), at T = 0.1 and 0.2, L = 32.

    <cos dtheta> = 1 - <dtheta^2>/2 = 1 - T/4 per bond. The fluctuation term
    vanishes at this order, because the harmonic sum of dtheta along a periodic
    row telescopes to zero. The O(T^2) remainder is the tolerance.
    """
    Ts = np.array([0.1, 0.2])
    r = measure(32, np.repeat(Ts, R), 500, 1000, 1)
    m, e = by_T(r["ups"], 2)
    res = m - (1 - Ts / 4)
    ok = np.all(np.abs(res) < Ts**2 + 3 * e)
    return ok, "; ".join(f"T={T}: Upsilon = {a:.4f} +/- {b:.4f}, minus (1 - T/4) = {c:+.4f} (T^2 = {T*T:.2f})"
                         for T, a, b, c in zip(Ts, m, e, res))


def check_high_T_stiffness():
    """At T = 1.4, well above T_KT, Upsilon is zero within errors at both L: free vortices screen any twist.

    A first version also required Upsilon(32) < Upsilon(16) at T = 1.4. With
    both already at zero, that ordering is noise. The faster fall at larger L
    shows between T = 1.0 and 1.2 in the figure instead.
    """
    out, ok = [], True
    for L in (16, 32):
        m, e = sweep(L)["ups"]
        ok &= abs(m[-1]) < 3 * e[-1] + 0.005
        out.append(f"L={L}: Upsilon(1.4) = {m[-1]:.4f} +/- {e[-1]:.4f}")
    return ok, ", ".join(out)


def check_crossing():
    """Upsilon_L = 2T/pi crosses at T*(L) above T_KT, moving down as L grows.

    Finite-size corrections are logarithmic, T*(L) - T_KT ~ 1/(ln L)^2, so at
    L = 16 and 32 expect roughly 0.90-0.98, not 0.893. The requirements are the
    order T*(16) > T*(32) > T_KT and both within 0.1 of T_KT.
    """
    t16, t32 = crossing(16), crossing(32)
    ok = t16 > t32 > T_KT and t16 - T_KT < 0.1
    return ok, f"T*(16) = {t16:.3f}, T*(32) = {t32:.3f}, T_KT = {T_KT} (Hasenbusch 2005)"


def check_specific_heat_peak():
    """C(T) peaks above T_KT, near T = 1.0 (Gupta & Baillie 1992), not at T_KT.

    The peak is from the bulk of vortex unbinding, which happens above the
    transition. The check is only that it lies in [0.95, 1.1] at L = 32 and
    clearly above T_KT.
    """
    C = sweep(32)["C"][0]
    Tp = TS[np.argmax(C)]
    return 0.95 <= Tp <= 1.1, f"C peak at T = {Tp:.2f} (L = 32), {Tp / T_KT - 1:+.0%} above T_KT, C_max = {C.max():.3f}"


def check_equilibration():
    """Ordered and random starts agree at T = 0.9, L = 32, in Upsilon and in vortex density."""
    a = measure(32, np.full(R, 0.9), 2000, 6000, 3, ordered=True)
    b = measure(32, np.full(R, 0.9), 2000, 6000, 4, ordered=False)
    out, ok = [], True
    for k in ("ups", "nv"):
        (ma, ea), (mb, eb) = by_T(a[k], 1), by_T(b[k], 1)
        z = (ma[0] - mb[0]) / np.hypot(ea[0], eb[0])
        ok &= abs(z) < 3
        out.append(f"{k}: ordered {ma[0]:.4f}, random {mb[0]:.4f} ({z:+.1f} sigma)")
    return ok, "; ".join(out)


def plot():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.subplots_adjust(wspace=0.28)
    for L, mk in ((16, "s"), (32, "o")):
        r = sweep(L)
        ax[0].errorbar(TS, *r["ups"], fmt=mk + "-", ms=4, label=f"L = {L}, crossing {crossing(L):.3f}")
        ax[1].errorbar(TS, *r["nv"], fmt=mk + "-", ms=4, label=f"L = {L}")
        ax[2].errorbar(TS, *r["C"], fmt=mk + "-", ms=4, label=f"L = {L}")
    ax[0].plot(TS, 2 * TS / np.pi, "k--", label=r"$2T/\pi$ (Nelson-Kosterlitz)")
    ax[1].set_yscale("log")
    for a, title, yl in zip(ax, ["stiffness: the transition is where it crosses 2T/pi",
                                 "vortex density: smooth, no feature at T_KT",
                                 "specific heat: peaks above T_KT"],
                            [r"helicity modulus $\Upsilon$", "vortices per plaquette", "C per spin"]):
        a.axvline(T_KT, color="gray", ls=":", label=r"$T_{KT}$ = 0.893")
        a.set(xlabel="T", ylabel=yl, title=title)
        a.legend(fontsize=8)
    fig.savefig(PLOTS / "kt_step3_transition.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    fails = run("KT step 3: the transition", [check_low_T_stiffness, check_high_T_stiffness, check_crossing,
                                              check_specific_heat_peak, check_equilibration])
    plot()
    sys.exit(fails)
