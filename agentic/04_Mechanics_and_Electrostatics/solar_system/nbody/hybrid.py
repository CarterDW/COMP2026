"""Hybrid symplectic integrator for planetary systems (Wisdom-Holman with a close-encounter switch, as in MERCURIUS).

Coordinates are democratic heliocentric (Duncan, Levison & Lee 1998): heliocentric positions Q, barycentric
velocities v. The Hamiltonian splits into
    H_kepler = sum (1/2 m v^2 - mu m / |Q|)          (each body's orbit about the star: solved exactly)
    H_jump   = |sum m v|^2 / (2 M_star)              (the star's reflex motion: a uniform drift of all Q)
    H_inter  = sum_pairs phi_ij,  phi = -G m_i m_j / r_ij
and each pair potential is cut smoothly (Rein et al. 2019) into a far part L(r) phi, applied as kicks, and a close part
(1 - L(r)) phi, which is integrated with the Kepler motion of the bodies that are in an encounter, using adaptive
Dormand-Prince Runge-Kutta. Splitting the potential (not the force) keeps the scheme exactly Hamiltonian, so the
force from each part includes the dL/dr term. Bodies flagged "big" interact with everything; the others
(planetesimals) feel the star and the big bodies but not each other.

Step:  kick(dt/2) jump(dt/2) drift(dt) jump(dt/2) kick(dt/2).  Units: AU, Msun, yr.
"""
import numpy as np
from numba import njit, prange
from nbody.units import G
from nbody.kepler import kepler_drift

N_HILL = 3.0          # changeover radius: this many Hill radii (MERCURY/MERCURIUS standard; 6 made a 2.5 MJ
                      # giant's zone 3.6 AU wide, so nearby bodies sat in slow Runge-Kutta encounters every step) ...
N_VEL = 0.4           # ... or the distance covered in this fraction of a step, whichever is larger
RK_RTOL = 1e-10       # Dormand-Prince tolerance during encounters (1e-12 cost ~1.6x more for no useful gain: Rung 6 tests)


@njit
def changeover(r, rc):
    """L(r) and dL/dr: 0 inside 0.1 rc, 1 outside rc, the C2-smooth step 10y^3 - 15y^4 + 6y^5 in between."""
    y = (r - 0.1 * rc) / (0.9 * rc)
    if y <= 0.0:
        return 0.0, 0.0
    if y >= 1.0:
        return 1.0, 0.0
    return 10 * y**3 - 15 * y**4 + 6 * y**5, (30 * y**2 - 60 * y**3 + 30 * y**4) / (0.9 * rc)


@njit
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit
def interacts(i, j, alive, big):
    return i != j and alive[i] and alive[j] and (big[i] or big[j])


@njit
def critical_radii(Q, v, m, alive, mu, dt):
    """max(N_HILL Hill radii, N_VEL * speed * dt) for each body, from its current state."""
    rc = np.zeros(len(m))
    for i in range(len(m)):
        if alive[i]:
            r = np.sqrt(Q[i, 0] ** 2 + Q[i, 1] ** 2 + Q[i, 2] ** 2)
            hill = r * (G * m[i] / (3 * mu)) ** (1 / 3)
            speed = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            rc[i] = max(N_HILL * hill, N_VEL * speed * dt)
    return rc


@njit(parallel=True)
def far_accelerations(Q, m, alive, big, rc):
    """a_i = sum_j G m_j (Q_j - Q_i) [L / r^3 - L' / r^2]: minus the gradient of the far potential L phi.

    Big bodies feel everyone; small bodies only the big ones (their mutual forces are switched off).
    """
    n = len(m)
    acc = np.zeros((n, 3))
    big_idx = np.flatnonzero(big & alive)
    for i in prange(n):
        if not alive[i]:
            continue
        n_j = n if big[i] else len(big_idx)
        for jj in range(n_j):
            j = jj if big[i] else big_idx[jj]
            if not interacts(i, j, alive, big):
                continue
            d0, d1, d2 = Q[j, 0] - Q[i, 0], Q[j, 1] - Q[i, 1], Q[j, 2] - Q[i, 2]
            r = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            L, dL = changeover(r, max(rc[i], rc[j]))
            w = G * m[j] * (L / r**3 - dL / r**2)
            acc[i, 0] += w * d0
            acc[i, 1] += w * d1
            acc[i, 2] += w * d2
    return acc


@njit(parallel=True)
def _kepler_all(Q, v, alive, mu, dt):
    Q1, v1 = Q.copy(), v.copy()
    for i in prange(len(Q)):
        if alive[i]:
            Q1[i], v1[i] = kepler_drift(Q[i], v[i], mu, dt)
    return Q1, v1


@njit
def _close_rhs(y, members, m, alive, big, rc, mu):
    """d/dt (Q, v) for the encounter set: the star plus the close part of their mutual forces."""
    k = len(members)
    dy = np.zeros((k, 6))
    for a in range(k):
        dy[a, 0:3] = y[a, 3:6]
        r = np.sqrt(y[a, 0] ** 2 + y[a, 1] ** 2 + y[a, 2] ** 2)
        dy[a, 3:6] = -mu * y[a, 0:3] / r**3
        i = members[a]
        for b in range(k):
            j = members[b]
            if not interacts(i, j, alive, big):
                continue
            d = y[b, 0:3] - y[a, 0:3]
            rij = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            L, dL = changeover(rij, max(rc[i], rc[j]))
            dy[a, 3:6] += G * m[j] * ((1 - L) / rij**3 + dL / rij**2) * d
    return dy


# Dormand-Prince 5(4) coefficients
_A = np.array([[0, 0, 0, 0, 0, 0], [1 / 5, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0],
               [44 / 45, -56 / 15, 32 / 9, 0, 0, 0], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
               [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0]])
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


@njit
def _dopri_step(y, h, members, m, alive, big, rc, mu):
    k = np.zeros((7, y.shape[0], 6))
    k[0] = _close_rhs(y, members, m, alive, big, rc, mu)
    for s in range(1, 6):
        ys = y.copy()
        for q in range(s):
            ys += h * _A[s, q] * k[q]
        k[s] = _close_rhs(ys, members, m, alive, big, rc, mu)
    y5 = y.copy()
    for q in range(6):
        y5 += h * _B5[q] * k[q]
    k[6] = _close_rhs(y5, members, m, alive, big, rc, mu)
    err = np.zeros(y.shape)
    for q in range(7):
        err += h * (_B5[q] - _B4[q]) * k[q]
    scale = np.empty(y.shape)
    for a in range(y.shape[0]):                       # positions and velocities scaled by their own magnitudes
        rmag = np.sqrt(y5[a, 0] ** 2 + y5[a, 1] ** 2 + y5[a, 2] ** 2)
        vmag = np.sqrt(y5[a, 3] ** 2 + y5[a, 4] ** 2 + y5[a, 5] ** 2)
        scale[a, 0:3] = RK_RTOL * rmag + 1e-300
        scale[a, 3:6] = RK_RTOL * vmag + 1e-300
    return y5, np.max(np.abs(err) / scale)


@njit
def _root(group, i):
    while group[i] != i:
        group[i] = group[group[i]]                    # path halving
        i = group[i]
    return i


@njit
def merge(i, j, Q, v, m, R, alive, big, comp):
    """Perfect merger of j into i: mass and momentum conserved, at the center of mass, volumes added.

    comp rows are (seed solids, solids gained in collisions, pebbles, gas). j's seed and collision mass become
    collision mass of i; its pebbles and gas stay pebbles and gas, so the solid/gas split is kept.
    """
    comp[i, 1] += comp[j, 0] + comp[j, 1]
    comp[i, 2] += comp[j, 2]
    comp[i, 3] += comp[j, 3]
    comp[j, :] = 0.0
    M = m[i] + m[j]
    Q[i] = (m[i] * Q[i] + m[j] * Q[j]) / M
    v[i] = (m[i] * v[i] + m[j] * v[j]) / M
    R[i] = (R[i] ** 3 + R[j] ** 3) ** (1 / 3)
    big[i] = big[i] or big[j]
    m[i] = M
    m[j] = 0.0
    alive[j] = False


@njit
def total_energy(Q, v, m, alive, big, mu, M_star):
    """Barycentric energy: kinetic of bodies and star, star-body potential, and interacting-pair potential."""
    n = len(m)
    p = np.zeros(3)
    E = 0.0
    for i in range(n):
        if alive[i]:
            E += 0.5 * m[i] * (v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            E -= mu * m[i] / np.sqrt(Q[i, 0] ** 2 + Q[i, 1] ** 2 + Q[i, 2] ** 2)
            p += m[i] * v[i]
    E += 0.5 * (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / M_star          # the star's reflex kinetic energy
    for i in range(n):
        for j in range(i + 1, n):
            if interacts(i, j, alive, big):
                d = Q[j] - Q[i]
                E -= G * m[i] * m[j] / np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return E


@njit
def _encounter_drift(members, Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t_now):
    """Integrate the encounter set over dt; merge bodies that touch. Returns the updated event count."""
    k = len(members)
    y = np.zeros((k, 6))
    for a in range(k):
        y[a, 0:3] = Q[members[a]]
        y[a, 3:6] = v[members[a]]
    t, h = 0.0, dt / 10
    while t < dt:
        h = min(h, dt - t)
        y_new, err = _dopri_step(y, h, members, m, alive, big, rc, mu)
        if err > 1.0:
            h *= max(0.2, 0.9 * err ** -0.2)
            continue
        t += h
        y = y_new
        h *= min(5.0, 0.9 * max(err, 1e-10) ** -0.2)
        for a in range(k):                                         # collisions at this sub-step
            for b in range(a + 1, k):
                i, j = members[a], members[b]
                if not interacts(i, j, alive, big):
                    continue
                d = y[b, 0:3] - y[a, 0:3]
                if np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) < R[i] + R[j]:
                    for c in range(k):                              # write back, merge, measure the energy lost
                        Q[members[c]] = y[c, 0:3]
                        v[members[c]] = y[c, 3:6]
                    E_before = total_energy(Q, v, m, alive, big, mu, M_star)
                    keep, gone = (i, j) if m[i] >= m[j] else (j, i)
                    if n_log < len(log):
                        log[n_log, 0], log[n_log, 1], log[n_log, 2] = t_now + t, keep, gone
                        log[n_log, 3], log[n_log, 4] = m[keep], m[gone]
                    merge(keep, gone, Q, v, m, R, alive, big, comp)
                    a_keep = a if keep == i else b
                    y[a_keep, 0:3] = Q[keep]
                    y[a_keep, 3:6] = v[keep]
                    if n_log < len(log):
                        log[n_log, 5] = E_before - total_energy(Q, v, m, alive, big, mu, M_star)
                    n_log += 1
    for a in range(k):
        if alive[members[a]]:
            Q[members[a]] = y[a, 0:3]
            v[members[a]] = y[a, 3:6]
    return n_log


@njit
def step(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t_now):
    """One hybrid step. log rows: (time, survivor, absorbed, m_survivor, m_absorbed, energy lost). Returns n_log.

    rc: each body's changeover radius (critical_radii). It must stay fixed from step to step, or the split
    Hamiltonian would change every step and the scheme would no longer be symplectic; recompute it only when
    masses change (mergers, gas accretion).
    """
    n = len(m)
    v += 0.5 * dt * far_accelerations(Q, m, alive, big, rc)
    p = np.zeros(3)
    for i in range(n):
        if alive[i]:
            p += m[i] * v[i]
    for i in range(n):
        if alive[i]:
            Q[i] += 0.5 * dt * p / M_star

    # Kepler-drift everyone, then find pairs whose separation dips below the changeover radius during the step.
    Q1, v1 = _kepler_all(Q, v, alive, mu, dt)
    in_enc = np.zeros(n, dtype=np.bool_)
    group = np.arange(n)                              # union-find: bodies linked by a close pair share a group
    big_idx = np.flatnonzero(big & alive)
    for i in big_idx:
        for j in range(n):
            if not interacts(i, j, alive, big) or (big[j] and j < i):      # each big-big pair once
                continue
            # closest approach along the straight line between the start and end separations (scalar math: no
            # temporary arrays, which numba would allocate on the heap for every pair)
            ax, ay, az = Q[j, 0] - Q[i, 0], Q[j, 1] - Q[i, 1], Q[j, 2] - Q[i, 2]
            bx, by, bz = Q1[j, 0] - Q1[i, 0] - ax, Q1[j, 1] - Q1[i, 1] - ay, Q1[j, 2] - Q1[i, 2] - az
            bb = bx * bx + by * by + bz * bz
            tau = 0.0 if bb == 0 else min(1.0, max(0.0, -(ax * bx + ay * by + az * bz) / bb))
            cx, cy, cz = ax + tau * bx, ay + tau * by, az + tau * bz
            if cx * cx + cy * cy + cz * cz < max(rc[i], rc[j]) ** 2:
                in_enc[i] = True
                in_enc[j] = True
                ri, rj = _root(group, i), _root(group, j)
                group[max(ri, rj)] = min(ri, rj)
    for i in range(n):
        if alive[i] and not in_enc[i]:
            Q[i] = Q1[i]
            v[i] = v1[i]
    flagged = np.flatnonzero(in_enc)
    roots = np.empty(len(flagged), dtype=np.int64)
    for a in range(len(flagged)):
        roots[a] = _root(group, flagged[a])
    for root in np.unique(roots):                     # integrate each independent encounter group on its own
        n_log = _encounter_drift(flagged[roots == root], Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log,
                                 t_now)

    p[:] = 0.0
    for i in range(n):
        if alive[i]:
            p += m[i] * v[i]
    for i in range(n):
        if alive[i]:
            Q[i] += 0.5 * dt * p / M_star
    v += 0.5 * dt * far_accelerations(Q, m, alive, big, rc)
    return n_log


def from_heliocentric(x_h, u_h, m, M_star):
    """Heliocentric positions and velocities (relative to the star) -> democratic heliocentric (Q, barycentric v).

    The star moves with v_star = -sum m u_h / (M_star + sum m), and each body's barycentric velocity is u_h + v_star.
    """
    v_star = -(m @ u_h) / (M_star + m.sum())
    return x_h.copy(), u_h + v_star


def to_heliocentric(Q, v, m, M_star):
    """Democratic heliocentric -> heliocentric velocities: u_h = v - v_star with v_star = -sum m v / M_star."""
    return Q.copy(), v + (m @ v) / M_star


PERI_THETA = 0.42     # max angle (rad) a body may sweep near pericenter per (sub)step: 2 pi / 15, what a circular
                      # orbit at the inner edge gets with dt = P_in / 15, so near-circular bodies never trigger it
PERI_ZONE = 5.0       # substep while a body is within this many pericenter distances (3 left ~1% errors)


@njit
def pericenter_substeps(Q, v, alive, mu, dt):
    """How many equal substeps this step needs so that every body near pericenter (within PERI_ZONE pericenter
    distances, or reaching pericenter during the step) sweeps at most PERI_THETA radians per substep at its
    pericenter speed. Wisdom-Holman cannot resolve a pericenter passage much faster than one step (Rauch & Holman
    1999); here it shows up through the giant's reflex term. Returns k >= 1."""
    k = 1
    for i in range(len(Q)):
        if not alive[i]:
            continue
        x0, x1, x2 = Q[i, 0], Q[i, 1], Q[i, 2]
        u0, u1, u2 = v[i, 0], v[i, 1], v[i, 2]
        r = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
        vr = (x0 * u0 + x1 * u1 + x2 * u2) / r
        v2 = u0 * u0 + u1 * u1 + u2 * u2
        h2 = (x1 * u2 - x2 * u1) ** 2 + (x2 * u0 - x0 * u2) ** 2 + (x0 * u1 - x1 * u0) ** 2
        e = np.sqrt(max(0.0, 1 + (v2 - 2 * mu / r) * h2 / mu**2))
        q = h2 / (mu * (1 + e))                                 # pericenter distance
        if r > PERI_ZONE * q and (r - q) > -vr * dt:           # far from pericenter, and not reaching it this step
            continue
        v_p = np.sqrt(mu * (1 + e) / q)
        k = max(k, int(np.ceil(v_p * dt / (q * PERI_THETA))))
    return min(k, 1000)


@njit
def step_resolving_pericenters(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt, log, n_log, t_now):
    """step(), split into pericenter_substeps(...) equal substeps when a fast pericenter passage needs it."""
    k = pericenter_substeps(Q, v, alive, mu, dt)
    for j in range(k):
        n_log = step(Q, v, m, R, alive, big, comp, rc, mu, M_star, dt / k, log, n_log, t_now + j * dt / k)
    return n_log
