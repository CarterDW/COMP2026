"""Step 3: simulated annealing on the noisy chi^2.

Five starts, two variants, run in parallel:
  A: one repopulation per evaluation throughout (notebook-08 annealing).
  B: the same, but the last quarter of the iterations average N_AVG_END repopulations.
Each chain reports its best-ever state and its final state. All candidates are re-scored
with N_RESCORE fresh repopulations, and their offsets from THETA_TRUE are expressed in
Fisher errors (derivatives of the model mean from N_FISHER common-seed repopulations).
"""
import sys
import os
import warnings
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mock
import fit

N_CHAIN, N_ITER = 5, 1200
T0, T_END = 50.0, 1.0
STEP = 0.1                      # proposal sd at T0, as a fraction of each prior range
END_FRAC, N_AVG_END = 0.25, 4
N_RESCORE, N_FISHER = 50, 40
RANGE = fit.HI - fit.LO


def _init():
    global OBJ
    warnings.simplefilter('ignore')
    OBJ = fit.Objective()


def anneal(job):
    variant, k = job
    rng = np.random.default_rng(100 + k)                 # same five starts for A and B
    x = np.clip(mock.THETA_TRUE + rng.uniform(-0.25, 0.25, 6) * RANGE,
                fit.LO + 1e-3, fit.HI - 1e-3)           # starting point only
    rng = np.random.default_rng(1000 * (variant == 'B') + k)
    T, n_end = T0, int(END_FRAC * N_ITER)
    E = OBJ(x, 10**7 * (1 + k))
    best_x, best_E = x.copy(), E
    hist = np.zeros((N_ITER, 3))                         # current E, best-ever E, T
    for it in range(N_ITER):
        averaging = variant == 'B' and it >= N_ITER - n_end
        x_new = x + STEP * np.sqrt(T / T0) * RANGE * rng.standard_normal(6)
        if averaging:
            E_new = OBJ(x_new, 10**8 + 10**6 * k + it, n_avg=N_AVG_END)
        else:
            E_new = OBJ(x_new, 10**7 * (1 + k + 5 * (variant == 'B')) + it + 1)
        if E_new < best_E:
            best_x, best_E = x_new.copy(), E_new
        if E_new < E or rng.random() < np.exp((E - E_new) / T):
            x, E = x_new, E_new
        hist[it] = E, best_E, T
        T *= (T_END / T0) ** (1 / N_ITER)
    return dict(variant=variant, k=k, best_x=best_x, best_E=best_E, final_x=x, final_E=E, hist=hist)


def _rescore(args):
    theta, j = args
    return OBJ(theta, 9 * 10**8 + j)


def _model_mean(theta):
    return OBJ.model_vector(theta, 3 * 10**7, n_avg=N_FISHER)    # common seeds for every theta


def fisher_errors(pool):
    """sigma_i = sqrt(diag F^-1), F = J^T Psi J, J from central differences of the mean model."""
    h = 0.02 * RANGE
    pts = [mock.THETA_TRUE + s * h[i] * np.eye(6)[i] for i in range(6) for s in (1, -1)]
    mu = np.array(pool.map(_model_mean, pts)).reshape(6, 2, -1)
    J = ((mu[:, 0] - mu[:, 1]) / (2 * h[:, None])).T
    C, hartlap = fit.covariance()
    F = J.T @ (hartlap * np.linalg.inv(C)) @ J
    return np.sqrt(np.diag(np.linalg.inv(F)))


def main():
    with Pool(10, initializer=_init) as pool:
        chains = pool.map(anneal, [(v, k) for v in 'AB' for k in range(N_CHAIN)])
        cands = [(c, kind) for c in chains for kind in ('best', 'final')]
        jobs = [(c[f'{kind}_x'], 1000 * n + j) for n, (c, kind) in enumerate(cands)
                for j in range(N_RESCORE)]
        rs = np.array(pool.map(_rescore, jobs)).reshape(len(cands), N_RESCORE)
        sigma = fisher_errors(pool)

    print("Fisher errors at the truth: " + "  ".join(f"{n}={s:.3f}" for n, s in zip(mock.NAMES, sigma)))
    print(f"floor: a re-scored chi2 at the truth is ~{2*mock.NBIN} +/- 7\n")
    print(f"{'chain':7s} {'candidate':10s} {'reported':>9s} {'re-scored':>14s}   pulls (theta - truth)/sigma")
    out = dict(sigma=sigma)
    for n, (c, kind) in enumerate(cands):
        pull = (c[f'{kind}_x'] - mock.THETA_TRUE) / sigma
        print(f"{c['variant']}{c['k']:<6d} {kind:10s} {c[f'{kind}_E']:9.1f} {rs[n].mean():8.1f} +/- {rs[n].std():4.1f}   "
              + " ".join(f"{p:+5.1f}" for p in pull))
        tag = f"{c['variant']}{c['k']}_{kind}"
        out.update({f'{tag}_x': c[f'{kind}_x'], f'{tag}_E': c[f'{kind}_E'], f'{tag}_rescored': rs[n]})
    for c in chains:
        out[f"{c['variant']}{c['k']}_hist"] = c['hist']

    for v in 'AB':
        for kind in ('best', 'final'):
            sel = [n for n, (c, kd) in enumerate(cands) if c['variant'] == v and kd == kind]
            rep = np.mean([cands[n][0][f'{kind}_E'] for n in sel])
            pulls = np.array([(cands[n][0][f'{kind}_x'] - mock.THETA_TRUE) / sigma for n in sel])
            print(f"\n{v} {kind:5s}: mean reported {rep:5.1f}, mean re-scored {rs[sel].mean():5.1f};  "
                  f"mean pull " + " ".join(f"{p:+.1f}" for p in pulls.mean(0))
                  + f";  rms pull {np.sqrt((pulls**2).mean()):.1f}")
    np.savez(os.path.join(fit.RESULTS, 'step3.npz'), **out)
    figures(out)


def figures(d):
    """Three figures, redrawable from results/step3.npz alone."""
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(fit.RESULTS, name), dpi=150)
        plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 4.2))
    for k in range(N_CHAIN):
        h = d[f'A{k}_hist']
        a.plot(h[:, 0], c=f'C{k}', lw=0.7, alpha=0.6)
        a.plot(h[:, 1], c=f'C{k}', lw=1.5, ls='--')
    a.axhspan(2 * mock.NBIN - 7, 2 * mock.NBIN + 7, color='k', alpha=0.1, label='floor at truth')
    a.plot([], [], 'k-', lw=0.7, label='current state')
    a.plot([], [], 'k--', label='best-ever')
    a.set(yscale='log', xlabel='iteration', ylabel=r'$\chi^2$',
          title='Annealing (variant A): current state vs best-ever')
    a2 = a.twinx()
    a2.plot(d['A0_hist'][:, 2], 'k:', lw=1)
    a2.set(yscale='log', ylabel='temperature (dotted)')
    a.legend(fontsize=8, loc='upper right')
    save(fig, 'step3a_annealing_traces.png')

    fig, a = plt.subplots(figsize=(6, 4.2))
    labels = [('A', 'best', 'A best-ever'), ('A', 'final', 'A final'), ('B', 'final', 'B final (averaged end)')]
    for j, (v, kind, lab) in enumerate(labels):
        for k in range(N_CHAIN):
            x = j + 0.12 * (k - 2)
            rs = d[f'{v}{k}_{kind}_rescored']
            a.errorbar(x, rs.mean(), rs.std(), fmt='o', c=f'C{k}', capsize=2)
            a.plot(x, d[f'{v}{k}_{kind}_E'], 'x', c=f'C{k}')
    a.axhspan(2 * mock.NBIN - 7, 2 * mock.NBIN + 7, color='k', alpha=0.1, label='floor at truth')
    a.plot([], [], 'kx', label='reported by the optimizer')
    a.errorbar([], [], [], fmt='ko', label=f're-scored, {N_RESCORE} repopulations')
    a.set_xticks(range(3), [lab for *_, lab in labels])
    a.set(ylabel=r'$\chi^2$', title='Reported vs re-scored, five chains', ylim=(0, None))
    a.legend(fontsize=8)
    save(fig, 'step3b_rescored.png')

    fig, a = plt.subplots(figsize=(7, 4.2))
    short = [r'$\log M_{\min}$', r'$\sigma_{\log M}$', r'$\alpha$', r'$\log M_1$', r'$\mu_c$', r'$\mu_s$']
    for j, (v, kind, lab) in enumerate(labels):
        pulls = np.array([(d[f'{v}{k}_{kind}_x'] - mock.THETA_TRUE) / d['sigma'] for k in range(N_CHAIN)])
        for i in range(6):
            a.plot(i + 0.25 * (j - 1) + np.zeros(N_CHAIN), pulls[:, i], 'o', c=f'C{j}', ms=4,
                   label=lab if i == 0 else None)
    a.axhspan(-1, 1, color='k', alpha=0.08)
    a.axhline(0, c='k', lw=0.6)
    a.set_xticks(range(6), short)
    a.set(ylabel=r'$(\hat\theta - \theta^*)/\sigma_{\rm Fisher}$', title='Where each chain ended, in Fisher errors')
    a.legend(fontsize=8)
    save(fig, 'step3c_endpoints.png')


if __name__ == '__main__':
    main()
