"""The Part A objective: chi^2 of a repopulation at theta against one 'data' mock at THETA_TRUE.

The data and each model evaluation are independent draws, so their difference has
covariance C (1 + 1/n_avg) when the model vector averages n_avg repopulations.
C is the step-1 covariance at the truth, Hartlap-corrected, held fixed for every theta.
"""
import os
import numpy as np
import mock

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DATA_SEED = 5000          # the 'observed' mock; no model evaluation uses this seed
MODEL_SEED0 = 10**6       # model evaluations use seeds >= this

# Hard prior box: outside it the objective is +inf (an explicit rejection, not a clip).
LO = np.array([12.0, 0.05, 0.5, 13.0, -1.0, -1.0])
HI = np.array([13.5, 1.00, 2.0, 15.0, 1.0, 1.0])


def covariance():
    T = np.load(os.path.join(RESULTS, 'step1_vectors.npz'))['truth']
    N, p = T.shape
    return np.cov(T, rowvar=False), (N - p - 2) / (N - 1)


class Objective:
    def __init__(self):
        self.hc = mock.load_halos()
        self.model = mock.build_model(self.hc)
        self.data = mock.measure(mock.populate(self.model, mock.THETA_TRUE, DATA_SEED),
                                 self.hc.Lbox)
        C, hartlap = covariance()
        self.Psi = hartlap * np.linalg.inv(C)      # precision of the data alone

    def model_vector(self, theta, seed, n_avg=1):
        seeds = MODEL_SEED0 + seed * n_avg + np.arange(n_avg)
        return np.mean([mock.measure(mock.populate(self.model, theta, s), self.hc.Lbox)
                        for s in seeds], axis=0)

    def __call__(self, theta, seed, n_avg=1):
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < LO) or np.any(theta > HI):
            return np.inf
        x = self.data - self.model_vector(theta, seed, n_avg)
        return x @ self.Psi @ x / (1 + 1 / n_avg)
