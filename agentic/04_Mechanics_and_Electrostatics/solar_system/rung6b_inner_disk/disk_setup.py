"""The Rung 5 gas disk as the parameter array of nbody/gas_effects.py, for the inner-disk pebble model."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "rung5_planet_formation"))
from initial_disk import planet_forming_disk, T_START          # noqa: E402

ALPHA_TURB = 1e-4                                                 # as in the Rung 5 production run


def disk_params(alpha_turb=ALPHA_TURB):
    """(M0, R1, t_nu, M_star, alpha_acc, t0, pebbles_on, migration_on, alpha_turb); dispersal starts at T_START."""
    d = planet_forming_disk()
    return np.array([d.M0, d.R1, d.t_nu, d.M_star, d.alpha, T_START, 1.0, 1.0, alpha_turb])
