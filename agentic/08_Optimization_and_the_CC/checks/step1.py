"""Step 1: data vector, covariance and the noise floor.

Repopulates the same BolPlanck halos at several (mu_c, mu_s) and measures [w_gg, w_g+].
  truth (0.8, 0.5): seeds 0-299 build the covariance, seeds 1000-1199 are held out.
  (0, 0), (0, 0.5), (0.4, 0.5): seeds 0-99, the same seeds as the truth run.
Writes results/step1_vectors.npz and two figures.
"""
import sys
import os
import warnings
import numpy as np
from multiprocessing import Pool
from scipy import stats
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mock

RESULTS = os.path.join(os.path.dirname(HERE), 'results')
os.makedirs(RESULTS, exist_ok=True)
NB = mock.NBIN

CONFIGS = {'truth': (0.8, 0.5), 'none': (0.0, 0.0), 'mc0': (0.0, 0.5), 'mc04': (0.4, 0.5)}
SEEDS = {'truth': list(range(300)), 'heldout': list(range(1000, 1200)),
         'none': list(range(100)), 'mc0': list(range(100)), 'mc04': list(range(100))}


def _init():
    global HC, MODEL
    warnings.simplefilter('ignore')
    HC = mock.load_halos()
    MODEL = mock.build_model(HC)


def _job(args):
    mu_c, mu_s, seed = args
    theta = mock.THETA_TRUE.copy()
    theta[4:] = mu_c, mu_s
    return mock.measure(mock.populate(MODEL, theta, seed), HC.Lbox)


def run_all():
    jobs = {}
    for name, seeds in SEEDS.items():
        mu = CONFIGS['truth' if name == 'heldout' else name]
        jobs[name] = [(mu[0], mu[1], s) for s in seeds]
    with Pool(8, initializer=_init) as pool:
        return {name: np.array(pool.map(_job, j)) for name, j in jobs.items()}


def wp_bruteforce(seed):
    """w_gg from a periodic KD-tree pair count with analytic randoms (independent of halotools)."""
    _init()
    g = mock.populate(MODEL, mock.THETA_TRUE, seed)
    L = HC.Lbox[0]
    pos = np.vstack([g['x'], g['y'], g['z']]).T % L
    pairs = cKDTree(pos, boxsize=L).query_pairs(np.hypot(mock.RP_BINS[-1], mock.PI_MAX),
                                               output_type='ndarray')
    d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    d -= L * np.round(d / L)
    rp, dz = np.hypot(d[:, 0], d[:, 1]), np.abs(d[:, 2])
    DD = 2 * np.histogram(rp[dz < mock.PI_MAX], mock.RP_BINS)[0]       # ordered pairs
    N = len(pos)
    RR = N * (N - 1) / L**3 * np.pi * np.diff(mock.RP_BINS**2) * 2 * mock.PI_MAX
    return 2 * mock.PI_MAX * (DD / RR - 1), mock.measure(g, HC.Lbox)[:NB]


def heldout_chi2(train, test):
    """chi^2 of held-out vectors against the training mean, with Hartlap-corrected precision.
    Also returns the exact Hotelling-T^2 scaling, under which F ~ F(p, N-p)."""
    N, p = train.shape
    C = np.cov(train, rowvar=False)
    hartlap = (N - p - 2) / (N - 1)
    x = test - train.mean(0)
    T2 = np.einsum('ij,jk,ik->i', x, np.linalg.inv(C), x) / (1 + 1 / N)
    return hartlap * T2, (N - p) / (p * (N - 1)) * T2, hartlap


def main():
    V = run_all()
    np.savez(os.path.join(RESULTS, 'step1_vectors.npz'), rp=mock.RP, **V)
    T, H = V['truth'], V['heldout']
    print(f"{len(T)} training + {len(H)} held-out mocks at truth; data vector length p = {T.shape[1]}")

    # 1. Zero alignment is zero: w_g+ at mu_c = mu_s = 0 against zero, own covariance.
    G0 = V['none'][:, NB:]
    C0 = np.cov(G0, rowvar=False)
    chi2_mean0 = len(G0) * G0.mean(0) @ np.linalg.inv(C0) @ G0.mean(0)
    pval0 = stats.f.sf((len(G0) - NB) / (NB * (len(G0) - 1)) * chi2_mean0, NB, len(G0) - NB)
    print(f"\n[zero]  mean w_g+ at mu=0 vs 0: chi2 = {chi2_mean0:.1f} for p = {NB}, "
          f"Hotelling p-value = {pval0:.2f}")

    # 2. Monotone in mu_c, and w_gg blind to mu (same seeds -> identical positions).
    amp = {k: V[k][:, NB:].mean(0) for k in ('mc0', 'mc04', 'truth')}
    err = {k: V[k][:, NB:].std(0) / np.sqrt(len(V[k])) for k in amp}
    signal = {k: np.sum(amp[k] / err['truth']**2) / np.sum(1 / err['truth']**2) for k in amp}
    print(f"[mono]  inverse-variance mean w_g+  mu_c=0: {signal['mc0']:.3f}   "
          f"0.4: {signal['mc04']:.3f}   0.8: {signal['truth']:.3f}  (mu_s = 0.5)")
    same = all(np.array_equal(V[k][:, :NB], V['truth'][:100, :NB]) for k in ('none', 'mc0', 'mc04'))
    print(f"[wgg]   w_gg identical across all mu configs at equal seeds: {same}")

    # 3. The floor: held-out chi^2 follows chi^2_p (Hartlap) / F(p, N-p) (exact).
    chi2, F, hartlap = heldout_chi2(T, H)
    p = T.shape[1]
    ks_chi2 = stats.kstest(chi2, stats.chi2(p).cdf).pvalue
    ks_F = stats.kstest(F, stats.f(p, len(T) - p).cdf).pvalue
    print(f"[floor] Hartlap factor {1/hartlap:.3f};  held-out chi2 mean {chi2.mean():.1f} "
          f"(expect {p}), std {chi2.std():.1f} (expect {np.sqrt(2*p):.1f})")
    print(f"        KS p-value vs chi2_{p}: {ks_chi2:.2f};  vs exact F({p},{len(T)-p}): {ks_F:.2f}")

    # 4. Independent w_gg: periodic KD-tree pair count vs halotools, same mock.
    w_kd, w_ht = wp_bruteforce(7)
    print(f"[kdtree] max |w_gg(kdtree)/w_gg(halotools) - 1| = {np.max(np.abs(w_kd / w_ht - 1)):.1e}")

    figures(V, chi2, p)


def figures(V, chi2, p):
    rp = mock.RP
    labels = {'none': r'$\mu_c=0,\ \mu_s=0$', 'mc0': r'$\mu_c=0,\ \mu_s=0.5$',
              'mc04': r'$\mu_c=0.4,\ \mu_s=0.5$', 'truth': r'$\mu_c=0.8,\ \mu_s=0.5$ (truth)'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    m, s = V['truth'][:, :NB].mean(0), V['truth'][:, :NB].std(0)
    ax[0].fill_between(rp, rp * (m - s), rp * (m + s), alpha=0.3)
    ax[0].plot(rp, rp * m, 'o-')
    ax[0].set(xscale='log', xlabel=r'$r_p$ [Mpc/$h$]', ylabel=r'$r_p\,w_{gg}$ [(Mpc/$h$)$^2$]',
              title=r'$w_{gg}$ (identical for every $\mu$), 1$\sigma$ band = one mock')
    for k in ('none', 'mc0', 'mc04', 'truth'):
        m, s = V[k][:, NB:].mean(0), V[k][:, NB:].std(0)
        ax[1].fill_between(rp, rp * (m - s), rp * (m + s), alpha=0.2)
        ax[1].plot(rp, rp * m, 'o-', ms=3, label=labels[k])
    ax[1].axhline(0, c='k', lw=0.6)
    ax[1].set(xscale='log', xlabel=r'$r_p$ [Mpc/$h$]', ylabel=r'$r_p\,w_{g+}$ [Mpc/$h$]',
              title=r'$w_{g+}$, $e=1$; bands = one-mock scatter')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'step1_signals.png'), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.linspace(0, chi2.max() * 1.1, 300)
    ax[0].hist(chi2, bins=25, density=True, alpha=0.6, label='200 held-out mocks at truth')
    ax[0].plot(x, stats.chi2(p).pdf(x), 'k', label=rf'$\chi^2_{{{p}}}$')
    ax[0].set(xlabel=r'$\chi^2$ (Hartlap-corrected)', ylabel='density',
              title='The noise floor: no fit can do better than this')
    ax[0].legend()
    C = np.cov(V['truth'], rowvar=False)
    r = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    im = ax[1].imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    ax[1].axhline(NB - 0.5, c='k', lw=0.8)
    ax[1].axvline(NB - 0.5, c='k', lw=0.8)
    ax[1].set(title=r'correlation matrix: $w_{gg}$ | $w_{g+}$ (population noise only)')
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'step1_floor.png'), dpi=150)


if __name__ == '__main__':
    main()
