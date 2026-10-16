"""Pools of lognormal mocks, generated in parallel and cached in results/.

Each mock returns its binned P_hat plus a few one-point statistics of delta, so
steps 4 and 5 never need to rebuild a field. Seeds are 20000 + i (step 2 used
0-999, step 3 used 10000-12999).
"""

import functools
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cosmo
import field

ROOT = pathlib.Path(__file__).resolve().parent
SEED0 = 20000
OM, S8 = 0.31, 0.81


@functools.lru_cache(maxsize=None)
def _PG():
    return field.lognormal_PG(lambda k: cosmo.P_model(k, OM, S8))[0]


def _one(seed):
    d = field.make_lognormal(_PG(), seed)
    y = np.log1p(d)
    z = (d - d.mean()) / d.std()
    stats = [d.min(), d.var(), y.mean(), y.var(), (z**3).mean()]
    return np.concatenate([field.measure_P(d)[1], stats])


def lognormal_pool(n, workers=16):
    """(P_hat, stats) for n lognormal mocks, shapes (n, bins) and (n, 5).

    stats columns: min delta, var delta, mean ln(1+delta), var ln(1+delta), skew delta.
    """
    cache = ROOT / "results" / f"lognormal_pool_{n}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["P_hat"], d["stats"]
    with ProcessPoolExecutor(workers) as ex:
        X = np.array(list(ex.map(_one, range(SEED0, SEED0 + n), chunksize=8)))
    nb = len(field.EDGES) - 1
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, P_hat=X[:, :nb], stats=X[:, nb:])
    return X[:, :nb], X[:, nb:]
