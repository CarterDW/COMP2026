"""Step 2: why gradients fail on a stochastic chi^2.

(a) 1D slices of chi^2 through THETA_TRUE in mu_c and logMmin, 20 repopulations per point.
(b) Spread at the truth vs the exact prediction for a fixed data vector.
(c) Finite-difference d chi^2 / d mu_c at mu_c = 0.9 (where the slice is steep), vs step h,
    with independent and with common (same-seed) random numbers on the two sides. The
    reference gradient is the mean of 200 common-seed pairs at h = 0.05.
(d) Nelder-Mead from three starts; where it stops, and what it scores when re-scored.
"""
import sys
import os
import warnings
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mock
import fit

P = 2 * mock.NBIN
N_REP = 20
SLICES = {4: np.linspace(0.3, 1.0, 15), 0: np.linspace(12.60, 13.00, 15)}  # index -> grid
H = np.logspace(-3, -1, 7)
N_FD = 40
MU_FD = 0.9
H_REF, N_REF = 0.05, 200


def _init():
    global OBJ
    warnings.simplefilter('ignore')
    OBJ = fit.Objective()


def _chi2(args):
    theta, seed = args
    return OBJ(theta, seed)


def _nelder_mead(k):
    rng = np.random.default_rng(k)
    x0 = mock.THETA_TRUE + rng.uniform(-0.15, 0.15, 6) * (fit.HI - fit.LO)
    x0 = np.clip(x0, fit.LO + 1e-3, fit.HI - 1e-3)   # starting point only
    calls, trace = [0], []

    def f(theta):
        calls[0] += 1
        v = OBJ(theta, 10**5 * (k + 1) + calls[0])
        trace.append(v)
        return v
    res = minimize(f, x0, method='Nelder-Mead', options={'maxfev': 600})
    rescored = [OBJ(res.x, 9 * 10**5 + 100 * k + j) for j in range(20)]
    return dict(x0=x0, x=res.x, fun=res.fun, nfev=res.nfev, trace=np.array(trace),
                rescored=np.array(rescored), message=res.message)


def theta_with(i, v):
    t = mock.THETA_TRUE.copy()
    t[i] = v
    return t


def main():
    with Pool(8, initializer=_init) as pool:
        # (a) slices
        slices = {}
        for i, grid in SLICES.items():
            jobs = [(theta_with(i, v), 1000 * i + 20 * a + r)
                    for a, v in enumerate(grid) for r in range(N_REP)]
            slices[i] = np.array(pool.map(_chi2, jobs)).reshape(len(grid), N_REP)
        at_truth = np.array(pool.map(_chi2, [(mock.THETA_TRUE, 50000 + r) for r in range(200)]))

        # (c) finite differences at mu_c = MU_FD
        fd = {}
        for common in (False, True):
            jobs = []
            for a, h in enumerate(H):
                for r in range(N_FD):
                    s = 70000 + 1000 * a + 2 * r
                    jobs += [(theta_with(4, MU_FD + h), s), (theta_with(4, MU_FD - h), s if common else s + 1)]
            v = np.array(pool.map(_chi2, jobs)).reshape(len(H), N_FD, 2)
            fd[common] = (v[..., 0] - v[..., 1]) / (2 * H[:, None])
        jobs = []
        for r in range(N_REF):
            jobs += [(theta_with(4, MU_FD + H_REF), 90000 + r), (theta_with(4, MU_FD - H_REF), 90000 + r)]
        v = np.array(pool.map(_chi2, jobs)).reshape(N_REF, 2)
        ref = (v[:, 0] - v[:, 1]) / (2 * H_REF)

        # (d) Nelder-Mead
        nm = pool.map(_nelder_mead, range(3))


    # (b) the exact spread at fixed data: chi^2 = |u - v|^2 / 2 in whitened units,
    # a noncentral chi^2_p(|u|^2)/2 with mean (p + |u|^2)/2 and var (2p + 4|u|^2)/4.
    C, hartlap = fit.covariance()
    _init()
    T = np.load(os.path.join(fit.RESULTS, 'step1_vectors.npz'))['truth']
    u2 = hartlap * (OBJ.data - T.mean(0)) @ np.linalg.inv(C) @ (OBJ.data - T.mean(0))
    mean_pred, sd_pred = (P + u2) / 2, np.sqrt(2 * P + 4 * u2) / 2
    print(f"[spread] data mock is |u|^2 = {u2:.1f} from the truth mean (expect ~{P})")
    print(f"         chi2 at truth, 200 repops: mean {at_truth.mean():.1f} (predict {mean_pred:.1f}), "
          f"std {at_truth.std():.1f} (predict {sd_pred:.1f}; sqrt(2p) = {np.sqrt(2*P):.1f})")

    grad_true = ref.mean()
    g, m = SLICES[4], slices[4].mean(1)
    secant = (np.interp(MU_FD + 0.05, g, m) - np.interp(MU_FD - 0.05, g, m)) / 0.1
    print(f"\n[fd] d<chi2>/d mu_c at mu_c = {MU_FD}: {grad_true:.0f} +/- {ref.std()/np.sqrt(N_REF):.0f} "
          f"({N_REF} common-seed pairs, h = {H_REF});  slice-mean secant: {secant:.0f}")
    for common, name in ((False, 'independent seeds'), (True, 'common seeds')):
        sd = fd[common].std(1)
        snr = np.abs(fd[common].mean(1)) / sd
        print(f"     {name:17s} h: " + "  ".join(f"{h:.0e}" for h in H))
        print(f"     {'':17s} sd: " + "  ".join(f"{s:7.0f}" for s in sd))
        print(f"     {'':17s} |grad|/sd of one FD estimate: "
              + "  ".join(f"{abs(grad_true)/s:.2f}" for s in sd))
    slope = np.polyfit(np.log(H), np.log(fd[False].std(1)), 1)[0]
    print(f"     independent seeds: log-log slope of sd vs h = {slope:.2f} (expect -1)")

    print("\n[nm] Nelder-Mead, 600 evaluations max, one repopulation each")
    for k, r in enumerate(nm):
        dist = (r['x'] - mock.THETA_TRUE) / (fit.HI - fit.LO)
        print(f"  start {k}: nfev {r['nfev']}, reported chi2 {r['fun']:.1f}, re-scored "
              f"{r['rescored'].mean():.1f} +/- {r['rescored'].std():.1f};  "
              f"max |dtheta|/range = {np.abs(dist).max():.3f}  ({r['message']})")
        print("           " + "  ".join(f"{n}={v:.3f}" for n, v in zip(mock.NAMES, r['x'])))

    out = dict(slice_mu=slices[4], slice_m=slices[0], at_truth=at_truth, fd_ind=fd[False],
               fd_crn=fd[True], ref=ref, mean_pred=mean_pred, sd_pred=sd_pred)
    for k, r in enumerate(nm):
        out.update({f'nm{k}_{key}': r[key] for key in ('x', 'fun', 'trace', 'rescored')})
    np.savez(os.path.join(fit.RESULTS, 'step2.npz'), **out)
    figures(out)


def figures(d):
    """Four separate figures, redrawable from results/step2.npz alone."""
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(fit.RESULTS, name), dpi=150)
        plt.close(fig)

    for key, i, xlabel, name in (('slice_mu', 4, r'$\mu_c$', 'step2a_slice_mu_c.png'),
                                 ('slice_m', 0, r'$\log M_{\min}$', 'step2b_slice_logMmin.png')):
        g, v = SLICES[i], d[key]
        fig, a = plt.subplots(figsize=(5.5, 4.2))
        a.plot(np.repeat(g, N_REP), v.ravel(), '.', ms=3, alpha=0.4, label='one repopulation each')
        a.plot(g, v.mean(1), 'k-', lw=1.5, label=f'mean of {N_REP}')
        a.axvline(mock.THETA_TRUE[i], c='C3', ls='--', label='truth')
        a.axhspan(d['mean_pred'] - d['sd_pred'], d['mean_pred'] + d['sd_pred'], color='C3',
                  alpha=0.12, label=r'predicted $\pm1\sigma$ at truth')
        a.set(xlabel=xlabel, ylabel=r'$\chi^2$', ylim=(0, None),
              title=f'Slice in {xlabel} (others at truth)')
        a.legend(fontsize=8)
        save(fig, name)

    fig, a = plt.subplots(figsize=(5.5, 4.2))
    grad = d['ref'].mean()
    a.loglog(H, d['fd_ind'].std(1), 'o-', label='independent seeds')
    a.loglog(H, d['fd_crn'].std(1), 's-', label='common seeds')
    a.loglog(H, d['fd_ind'].std(1)[-1] * H[-1] / H, 'k:', label=r'$\propto 1/h$')
    a.axhline(grad, c='C3', ls='--', label=rf"true gradient $= {grad:.0f}$")
    a.set(xlabel='finite-difference step $h$', ylabel='scatter of one FD estimate',
          title=rf'$\partial\chi^2/\partial\mu_c$ at $\mu_c={MU_FD}$: signal above red line')
    a.legend(fontsize=8)
    save(fig, 'step2c_finite_differences.png')

    fig, a = plt.subplots(figsize=(5.5, 4.2))
    for k in range(3):
        best = np.minimum.accumulate(d[f'nm{k}_trace'])
        rs = d[f'nm{k}_rescored']
        a.plot(best, c=f'C{k}', label=f'start {k}')
        a.errorbar(len(best) + 20 + 15 * k, rs.mean(), rs.std(), fmt='o', c=f'C{k}', capsize=3)
    a.axhspan(d['mean_pred'] - d['sd_pred'], d['mean_pred'] + d['sd_pred'], color='k',
              alpha=0.1, label=r'floor at truth ($\pm1\sigma$)')
    a.set(ylim=(0, 120), xlabel='evaluation', ylabel=r'$\chi^2$',
          title='Nelder-Mead: best-so-far (lines) vs re-scored (points)')
    a.legend(fontsize=8)
    save(fig, 'step2d_nelder_mead.png')


if __name__ == '__main__':
    main()
