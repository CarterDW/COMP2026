"""Likelihoods, the Metropolis sampler, and grid posteriors for (Om, s8).

The data are a binned P_hat vector d. The model is mu(theta) = bin_model_fast(P_model).
The likelihood is Gaussian,

    -2 ln L = (d - mu)^T Psi (d - mu),

where Psi is a precision matrix (an inverse covariance). Step 5 compares four
choices of Psi:
    A  Gaussian C = diag(2P^2/N_k) at the fiducial cosmology
    B  inverse of C_hat from N_s = 50 lognormal mocks, uncorrected
    C  the same, times the Hartlap factor
    D  Hartlap-corrected C_hat from 2000 lognormal mocks: the reference
Priors are flat on Om in [0.1, 0.6] and s8 in [0.4, 1.2].
"""

import functools
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cosmo
import field
import mocks

ROOT = pathlib.Path(__file__).resolve().parent
TRUTH = np.array([mocks.OM, mocks.S8])
PRIOR = np.array([[0.1, 0.6], [0.4, 1.2]])
N_TRAIN, NS_SMALL = 2000, 50
p = len(field.EDGES) - 1


def model(theta):
    Om, s8 = theta
    return field.bin_model_fast(lambda k: cosmo.P_model(k, Om, s8))


def hartlap(Ns):
    return (Ns - p - 2) / (Ns - 1)


@functools.lru_cache(maxsize=None)
def data():
    """(train, test) lognormal P_hat vectors, the same split as step 4."""
    X, _ = mocks.lognormal_pool(3000)
    return X[:N_TRAIN], X[N_TRAIN:]


def precision(name, subset=0):
    """Psi for likelihood A-D. B and C use 50-mock block number `subset` (0-39)."""
    train, _ = data()
    if name == "A":
        return np.diag(1 / field.gaussian_sigma(lambda k: cosmo.P_model(k, *TRUTH))**2)
    if name == "D":
        return np.linalg.inv(np.cov(train.T)) * hartlap(N_TRAIN)
    block = train[subset * NS_SMALL:(subset + 1) * NS_SMALL]
    Psi = np.linalg.inv(np.cov(block.T))
    return Psi * hartlap(NS_SMALL) if name == "C" else Psi


def log_post(theta, d, Psi):
    if np.any(theta < PRIOR[:, 0]) or np.any(theta > PRIOR[:, 1]):
        return -np.inf
    r = d - model(theta)
    return -0.5 * r @ Psi @ r


def fisher(Psi, h=1e-3):
    """F_ij = d_i mu^T Psi d_j mu at the truth, by central differences."""
    J = np.array([(model(TRUTH + h * e) - model(TRUTH - h * e)) / (2 * h) for e in np.eye(2)]).T
    return J.T @ Psi @ J


def metropolis(d, Psi, x0, step_cov, n, seed):
    """The manual-day sampler in 2D, with a Gaussian proposal.

    The proposal is symmetric, q(x'|x) = q(x|x'), so the acceptance probability
    is min(1, post(x')/post(x)), exactly as in 1D. Shaping the proposal like the
    posterior (step_cov taken from the Fisher matrix) is the 2D version of
    choosing delta.
    """
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(step_cov)
    x, lp = np.array(x0, float), log_post(x0, d, Psi)
    chain, acc = np.zeros((n, 2)), 0
    for i in range(n):
        x_try = x + Lc @ rng.standard_normal(2)
        lp_try = log_post(x_try, d, Psi)
        if np.log(rng.random()) < lp_try - lp:
            x, lp = x_try, lp_try
            acc += 1
        chain[i] = x
    return chain, acc / n


def _chain_job(args):
    return metropolis(*args)


def run_chains(d, Psi, n_chains=8, n=6000, seed=0):
    """Parallel chains, started at truth + 3 Fisher sigma in a random direction."""
    Finv = np.linalg.inv(fisher(Psi))
    step = 2.38**2 / 2 * Finv
    rng = np.random.default_rng(seed)
    starts = TRUTH + 3 * rng.standard_normal((n_chains, 2)) * np.sqrt(np.diag(Finv))
    jobs = [(d, Psi, x0, step, n, seed * 100 + c) for c, x0 in enumerate(starts)]
    with ProcessPoolExecutor(n_chains) as ex:
        out = list(ex.map(_chain_job, jobs))
    return np.array([c for c, _ in out]), np.array([a for _, a in out])


def gelman_rubin(chains):
    """R_hat per parameter for chains of shape (m, n, 2): pooled vs within-chain variance."""
    m, n, _ = chains.shape
    W = chains.var(1, ddof=1).mean(0)
    B = n * chains.mean(1).var(0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def autocorr_time(x, c=5):
    """Integrated autocorrelation time of a 1D chain, tau = 1 + 2 sum rho(t).

    The sum is cut at the first window M >= c tau (Sokal), because the far tail of
    the estimated rho is pure noise.
    """
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    rho = np.fft.irfft(f * np.conj(f))[:n]
    rho /= rho[0]
    tau = 2 * np.cumsum(rho) - 1
    M = np.argmax(np.arange(n) >= c * tau)
    return tau[M]


# ---- grid posteriors, for coverage over many mocks -------------------------

GRID_OM = np.linspace(TRUTH[0] - 0.025, TRUTH[0] + 0.025, 501)
GRID_S8 = np.linspace(TRUTH[1] - 0.030, TRUTH[1] + 0.030, 601)


def _grid_row(Om):
    return np.array([model((Om, s8)) for s8 in GRID_S8])


@functools.lru_cache(maxsize=None)
def model_grid():
    """mu(theta) on the 501 x 601 grid (spacing 1e-4), shape (points, bins). Cached."""
    cache = ROOT / "results" / "model_grid.npy"
    if cache.exists():
        return np.load(cache)
    with ProcessPoolExecutor(16) as ex:
        M = np.array(list(ex.map(_grid_row, GRID_OM, chunksize=4))).reshape(-1, p)
    np.save(cache, M)
    return M


def grid_chi2(D, Psi):
    """chi^2 at every grid point for each data row of D, shape (rows, points).

    Expands (d - mu)^T Psi (d - mu) = d Psi d - 2 d Psi mu + mu Psi mu, so the
    grid term is computed once and each mock costs one matrix product.
    """
    M = model_grid()
    MP = M @ Psi
    q = np.einsum("gi,gi->g", MP, M)
    return np.einsum("ni,ij,nj->n", D, Psi, D)[:, None] - 2 * D @ MP.T + q[None, :]


def hpd_contains(chi2_grid, chi2_truth, level=0.68):
    """Is the truth inside the highest-posterior-density region holding `level` of the mass?

    The posterior is exp(-chi^2/2) on the uniform grid, so the HPD region is the
    set of points with chi^2 below a threshold. The threshold is chosen so those
    points hold the requested mass.
    """
    c = np.sort(chi2_grid)
    w = np.exp(-0.5 * (c - c[0]))
    cum = np.cumsum(w) / w.sum()
    return chi2_truth <= c[np.searchsorted(cum, level)]
