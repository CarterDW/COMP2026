"""Time integrators for dx/dt = v, dv/dt = a(x)."""
import numpy as np
from nbody.gravity import accelerations


def leapfrog(pos, vel, mass, eps, dt, n_steps, save_every=1):
    """Kick-drift-kick leapfrog: symplectic, time-reversible, second order.

    Returns t (S,), pos (S, N, 3), vel (S, N, 3) with S = n_steps // save_every + 1 snapshots,
    the first being the initial state.
    """
    assert n_steps % save_every == 0, "n_steps must be a multiple of save_every"
    pos, vel = pos.astype(float), vel.astype(float)   # copies; inputs are not modified
    n_saves = n_steps // save_every + 1
    P, V = np.empty((n_saves, *pos.shape)), np.empty((n_saves, *vel.shape))
    P[0], V[0] = pos, vel

    acc = accelerations(pos, mass, eps)
    for step in range(1, n_steps + 1):
        vel += 0.5 * dt * acc      # kick
        pos += dt * vel            # drift
        acc = accelerations(pos, mass, eps)
        vel += 0.5 * dt * acc      # kick
        if step % save_every == 0:
            P[step // save_every], V[step // save_every] = pos, vel

    t = dt * save_every * np.arange(n_saves)
    return t, P, V
