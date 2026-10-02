"""5a: the planet-forming disk, built from the Rung 4 handoff and the Rung 4b final stellar mass."""
from pathlib import Path
import numpy as np
from nbody.units import G
from nbody.viscous_disk import LyndenBellPringle, scale_radius_from

ROOT = Path(__file__).parent.parent
ALPHA = 1e-3                  # weakly turbulent "dead zone" viscosity
T_START = 1e6                 # yr: embryos assemble while the disk is ~1 Myr old


def rung4_disk():
    """Mass and angular momentum of the Rung 4 disk after the star's post-simulation accretion tail (Rung 4b).

    The tail gas (M_star_final - M_star_sim) reached the star through the accretion radius on near-circular
    orbits, so it removes j = sqrt(G M r_acc) per unit mass. Returns M_disk, J_disk, M_star.
    """
    hand = np.load(ROOT / "rung4_protostar_disk" / "handoff.npz")
    star = np.load(ROOT / "rung4b_ignition" / "star_history.npz")
    M_star, M_sim_star, r_acc = float(star["M"][-1]), float(hand["M_star"]), float(hand["r_acc_AU"])
    Rc, sigma = hand["R_centers"], hand["sigma_Msun_AU2"]
    mid = np.sqrt(Rc[1:] * Rc[:-1])
    edges = np.concatenate([[Rc[0] ** 2 / mid[0]], mid, [Rc[-1] ** 2 / mid[-1]]])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    M_sim, J_sim = np.sum(sigma * area), np.sum(sigma * area * np.sqrt(G * M_sim_star * Rc))
    dM = M_star - M_sim_star
    return M_sim - dM, J_sim - dM * np.sqrt(G * M_star * r_acc), M_star


def planet_forming_disk():
    M_disk, J_disk, M_star = rung4_disk()
    return LyndenBellPringle(M_disk, scale_radius_from(M_disk, J_disk, M_star), M_star, ALPHA)
