"""Analysis of a protostar + disk snapshot from star_formation_leapfrog."""
import numpy as np
from nbody.units import G
from nbody.gravity import potentials


def central_frame(snap):
    """Gas positions and velocities relative to the most massive sink, plus that sink's mass."""
    k = np.argmax(snap["sink_mass"])
    return snap["pos"] - snap["sink_pos"][k], snap["vel"] - snap["sink_vel"][k], snap["sink_mass"][k]


def disk_members(x, v, M_star, keplerian_fraction=0.7, thickness=0.3):
    """Gas that orbits the star: azimuthal speed above keplerian_fraction * v_K and |z| < thickness * R."""
    R = np.linalg.norm(x[:, :2], axis=1)
    v_phi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / R
    return (v_phi > keplerian_fraction * np.sqrt(G * M_star / R)) & (np.abs(x[:, 2]) < thickness * R)


def surface_density(R, mass, edges):
    """Sigma in annuli [edges[k], edges[k+1]]: mass / area. Returns bin centers and Sigma."""
    dm, _ = np.histogram(R, bins=edges, weights=mass)
    return np.sqrt(edges[1:] * edges[:-1]), dm / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))


def centrifugal_radius(pos0, vel0, M_central):
    """Radius of the circular orbit with each particle's initial specific angular momentum about z: j_z^2 / (G M)."""
    j_z = pos0[:, 0] * vel0[:, 1] - pos0[:, 1] * vel0[:, 0]
    return j_z**2 / (G * M_central)


def bound_fraction(snap, eps):
    """Fraction of all mass (gas + sinks) with negative specific energy, kinetic + potential, in the CM frame."""
    pos = np.vstack([snap["pos"], snap["sink_pos"]])
    vel = np.vstack([snap["vel"], snap["sink_vel"]])
    mass = np.concatenate([snap["mass"], snap["sink_mass"]])
    vel = vel - mass @ vel / mass.sum()
    bound = 0.5 * np.sum(vel**2, axis=1) + potentials(pos, mass, eps) < 0
    return mass[bound].sum() / mass.sum()


def toomre_q(R_centers, sigma, M_star, cs_of_R):
    """Q = cs Omega / (pi G Sigma) for a Keplerian disk; Q < 1 means the disk is unstable to fragmentation."""
    omega = np.sqrt(G * M_star / R_centers**3)
    return cs_of_R(R_centers) * omega / (np.pi * G * sigma)
