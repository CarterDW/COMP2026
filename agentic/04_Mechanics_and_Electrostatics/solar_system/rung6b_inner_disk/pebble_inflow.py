"""The pebble flux crossing 4 AU inward: what the giant-planet zone lets through to the inner disk.

- Before T_START (1 Myr) there are no embryos yet, so all of the Lambrechts & Johansen (2014) growth-front flux
  (nbody/gas_effects.py pebble_flux) crosses 4 AU.
- From T_START to the end of the Rung 5 run the flux left after the embryos have taken their share, replayed from
  that run's snapshots (every 15 kyr) with the same rule as accrete_pebbles: outside in, each embryo takes its
  capture rate, and an embryo at the pebble isolation mass stops the flux altogether.
- After the Rung 5 run: zero (the giant's core is isolated by then).

The pebbles arrive with the outer disk's composition: rock 7.1/30 of the solids, ice the rest.
"""
from pathlib import Path
import numpy as np
from nbody.gas_effects import pebble_flux, pebble_capture_rate, pebble_isolation_mass
from nbody.viscous_disk import Z_ROCK, Z_ICE

RUNG5_RUN = Path(__file__).parent.parent / "rung5_planet_formation" / "data" / "formation_pebbles_migration_aturb1e-4.npz"
ICE_FRACTION = 1 - Z_ROCK / Z_ICE


def flux_past_embryos(t, x, m, alive, big, disk):
    """Pebble flux (Msun/yr) left after every embryo of one snapshot, as accrete_pebbles passes it."""
    flux = pebble_flux(t, disk)
    idx = np.flatnonzero(alive & big)
    radius = np.hypot(x[idx, 0], x[idx, 1])
    for k in np.argsort(-radius):
        i, R = idx[k], radius[k]
        if m[i] >= pebble_isolation_mass(R, t, disk):
            return 0.0
        flux -= pebble_capture_rate(m[i], R, t, disk, flux)
    return flux


class Inflow:
    def __init__(self, disk):
        d = np.load(RUNG5_RUN)
        assert np.allclose(d["disk_params"], disk), "the Rung 5 run used a different disk"
        self.disk, self.t0 = disk, float(d["t"][0])
        self.t_table = d["t"]
        self.f_table = np.array([flux_past_embryos(t, d["x"][k], d["m"][k], d["alive"][k], d["big"][k], disk)
                                 for k, t in enumerate(d["t"])])

    def __call__(self, t):
        if t < self.t0:
            return pebble_flux(t, self.disk)
        if t > self.t_table[-1]:
            return 0.0
        return np.interp(t, self.t_table, self.f_table)
