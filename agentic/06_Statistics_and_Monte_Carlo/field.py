"""Gaussian random fields in a periodic box, and the binned power spectrum estimator.

Conventions (see Power_Spectrum_Inference.md):

    delta_k = fftn(delta),    P_hat(k) = |delta_k|^2 V / N_c^2 .

If w is white noise with unit variance, <|fftn(w)|^2> = N_c, so multiplying
fftn(w) by sqrt(N_c P(k) / V) gives a field with power P. Starting from a real
w makes delta_k Hermitian automatically, so the inverse FFT comes back real.
"""

import numpy as np

L = 1000.0                 # box side, Mpc/h
NG = 128                   # cells per side
NC = NG**3
V = L**3
KF = 2 * np.pi / L         # fundamental mode
EDGES = np.linspace(0.01, 0.4, 26)   # 25 linear k bins, h/Mpc


def box_muller(rng, n):
    """n unit Gaussians from uniforms, as in the manual day, Problem 2.

    Each pair (u1, u2) gives two independent Gaussians, r cos(phi) and r sin(phi).
    The radius uses 1 - u1 because rng.random() can return 0 but never 1, so
    log(0) is impossible.
    """
    m = (n + 1) // 2
    u1, u2 = rng.random(m), rng.random(m)
    r = np.sqrt(-2 * np.log(1 - u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
    return z[:n]


def kmag():
    """|k| at every grid point, in fftn layout, in h/Mpc."""
    k1 = 2 * np.pi * np.fft.fftfreq(NG, d=L / NG)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij", sparse=True)
    return np.sqrt(kx**2 + ky**2 + kz**2)


_K = kmag()
_BIN = np.digitize(_K.ravel(), EDGES) - 1          # -1 or 25 means outside the range
_IN = (_BIN >= 0) & (_BIN < len(EDGES) - 1)


def P_grid(P):
    """P(k) at every grid mode, flattened, with the k = 0 mode set to zero."""
    out = np.zeros(NC)
    out[1:] = P(_K.ravel()[1:])
    return out


def make_gaussian(P, seed):
    """A real Gaussian field delta(x) on the grid with power spectrum P(k), a callable."""
    rng = np.random.default_rng(seed)
    w = box_muller(rng, NC).reshape(NG, NG, NG)
    amp = np.sqrt(NC * P_grid(P) / V).reshape(NG, NG, NG)
    return np.fft.ifftn(np.fft.fftn(w) * amp).real


def _bin(values):
    """Average a per-mode array over each k shell; also return N_k."""
    nb = len(EDGES) - 1
    Nk = np.bincount(_BIN[_IN], minlength=nb)
    return np.bincount(_BIN[_IN], weights=values[_IN], minlength=nb) / Nk, Nk


def measure_P(delta):
    """Binned P_hat(k). Returns (k_mean, P_hat, N_k), N_k = grid vectors per shell."""
    Pk = np.abs(np.fft.fftn(delta).ravel())**2 * V / NC**2
    P_hat, Nk = _bin(Pk)
    k_mean, _ = _bin(_K.ravel())
    return k_mean, P_hat, Nk


def bin_model(P):
    """The expected P_hat in each bin: P(k) averaged over the same grid modes.

    This differs from P(k_mean) wherever P curves inside a bin, so it is the
    right thing to compare a measurement against.
    """
    return _bin(P_grid(P))[0]


def gaussian_sigma(P):
    """Predicted scatter of P_hat per bin for a Gaussian field.

    Modes k and -k carry the same |delta_k|^2, so each shell holds N_k/2
    independent modes. Each one's P_hat is exponentially distributed, with
    variance P^2. Averaging gives Var = (2/N_k^2) sum over the shell of P(k)^2,
    which reduces to 2 P^2 / N_k when P is constant across the bin.
    """
    P2, Nk = _bin(P_grid(P)**2)
    return np.sqrt(2 * P2 / Nk)


def smooth(delta, R):
    """delta convolved with a Gaussian of width R (Mpc/h), via delta_k * exp(-k^2 R^2 / 2)."""
    return np.fft.ifftn(np.fft.fftn(delta) * np.exp(-0.5 * (_K * R)**2)).real


def gaussian_P_hat(Pg, seed):
    """Binned P_hat of make_gaussian(P, seed) without building the real-space field.

    measure_P(make_gaussian(...)) is an ifftn followed by an fftn, which cancel, so
    P_hat_k = |fftn(w)|^2 * amp^2 * V / N_c^2 = |fftn(w)|^2 * P_k / N_c. Skipping
    the round trip and taking Pg = P_grid(P) precomputed makes a mock about 6x
    cheaper, which matters for pools of thousands.
    """
    w = box_muller(np.random.default_rng(seed), NC).reshape(NG, NG, NG)
    return _bin(np.abs(np.fft.fftn(w).ravel())**2 * Pg / NC)[0]
