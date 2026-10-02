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


def sph_leapfrog(pos, vel, u, mass, gas, eps, t_out, courant=0.3, dt_max=np.inf, rho_stop=np.inf):
    """Kick-drift-kick for self-gravitating SPH gas, with a global adaptive timestep.

    Gravity is Plummer-softened with fixed eps. Viscosity depends on velocity, so the closing kick evaluates
    forces with the predicted velocity v_half + (dt/2) a_old (as GADGET-2 does). u is ignored for isothermal gas.
    Snapshots are taken exactly at the times in t_out (which must start at 0). If the peak density exceeds
    rho_stop (a runaway isothermal collapse, where Rung 4 will put a sink), the run stops there: the final
    snapshot is that moment and result["stopped"] is True.

    Returns a dict of arrays over snapshots: t, pos, vel, u, rho, h, E_rad (energy radiated so far; isothermal
    only), plus n_steps and stopped.
    """
    from nbody.sph import neighbour_data, hydro_forces_pairs
    assert t_out[0] == 0 and np.all(np.diff(t_out) > 0)
    pos, vel, u = pos.astype(float), vel.astype(float), np.array(u, dtype=float) * np.ones(len(mass))

    def forces(pos, vel, u, h):
        h, rho, pairs = neighbour_data(pos, mass, h)
        P, c = gas.pressure_and_sound_speed(rho, u)
        acc, dudt, signal = hydro_forces_pairs(pairs, pos, vel, mass, h, rho, P, c, gas.alpha, gas.beta)
        acc += accelerations(pos, mass, eps)
        return acc, dudt, signal, h, rho

    h = np.full(len(mass), (pos.max() - pos.min()) / len(mass) ** (1 / 3))
    acc, dudt, signal, h, rho = forces(pos, vel, u, h)
    out = {key: [] for key in ("pos", "vel", "u", "rho", "h", "E_rad")}
    t, E_rad, n_steps, stopped, times = 0.0, 0.0, 0, False, [0.0]

    def record():
        for key, value in zip(out, (pos, vel, u, rho, h, E_rad)):
            out[key].append(np.copy(value))

    record()
    for t_next in t_out[1:]:
        while t < t_next:
            # Courant (signal speed), acceleration, and (adiabatic) no particle changes u by more than ~courant * u.
            a_mag = np.linalg.norm(acc, axis=1)
            dt = min(courant * np.min(h / signal), courant * np.min(np.sqrt(h / np.maximum(a_mag, 1e-300))),
                     dt_max, t_next - t)
            if gas.eos == "adiabatic":
                dt = min(dt, courant * np.min(u / np.maximum(np.abs(dudt), 1e-300)))
            vel += 0.5 * dt * acc
            if gas.eos == "adiabatic":
                u += 0.5 * dt * dudt
            pos += dt * vel
            dudt_old = dudt
            acc, dudt, signal, h, rho = forces(pos, vel + 0.5 * dt * acc, u + 0.5 * dt * dudt, h)
            vel += 0.5 * dt * acc
            if gas.eos == "adiabatic":
                u += 0.5 * dt * dudt
            else:
                E_rad += 0.5 * dt * np.sum(mass * (dudt_old + dudt))
            t = t_next if t_next - t <= dt else t + dt
            n_steps += 1
            if rho.max() > rho_stop:
                stopped = True
                break
        record()
        times.append(t)
        if stopped:
            break

    result = {key: np.array(value) for key, value in out.items()}
    result["t"], result["n_steps"], result["stopped"] = np.array(times), n_steps, stopped
    return result


MIN_GAS = 100


def star_formation_leapfrog(pos, vel, mass, gas, eps, t_out, r_acc, rho_sink, sinks=None, courant=0.3, dt_max=np.inf):
    """Self-gravitating isothermal/barotropic SPH gas plus sink particles; kick-drift-kick, global adaptive dt.

    After every step, gas is accreted onto sinks and a new sink may form (see nbody/sinks.py); whenever particles
    are removed, forces are recomputed so the next kick is again exactly momentum-conserving. Gravity between all
    gas and sinks is direct-sum with one softening eps.

    The run stops early, with status "gas exhausted", if fewer than MIN_GAS particles remain: smoothing lengths
    need ~60 neighbours to exist at all. The last snapshot is then that moment.

    Returns (snapshots, history, sinks, status). snapshots: one dict per output time with gas pos, vel, mass, rho,
    h, ids (each gas particle's index in the initial arrays) and the sinks' mass, pos, vel, spin, plus E_rad. history: per step, t and the list of sink masses (the
    accretion record). status: "done" or "gas exhausted".
    """
    from nbody.sph import neighbour_data, hydro_forces_pairs
    from nbody.sinks import Sinks, accrete, try_create_sink
    assert gas.eos in ("isothermal", "barotropic"), "sinks are implemented for radiating (non-adiabatic) gas"
    assert t_out[0] == 0 and np.all(np.diff(t_out) > 0)
    pos, vel, mass = pos.astype(float), vel.astype(float), mass.astype(float)
    ids = np.arange(len(mass))                     # each surviving gas particle's index in the initial arrays
    sinks = Sinks() if sinks is None else sinks
    h = np.full(len(mass), (pos.max() - pos.min()) / len(mass) ** (1 / 3))

    def forces(pos, vel, h):
        h, rho, pairs = neighbour_data(pos, mass, h)
        d_sink = None
        if gas.heated():
            d_sink = (np.min(np.linalg.norm(pos[:, None, :] - sinks.pos[None, :, :], axis=2), axis=1)
                      if len(sinks) else np.full(len(mass), np.inf))
        P, c = gas.pressure_and_sound_speed(rho, None, d_sink)
        acc, dudt, signal = hydro_forces_pairs(pairs, pos, vel, mass, h, rho, P, c, gas.alpha, gas.beta)
        g = accelerations(np.vstack([pos, sinks.pos]), np.concatenate([mass, sinks.mass]), eps)
        return acc + g[:len(mass)], g[len(mass):], dudt, signal, h, rho, P

    acc, sink_acc, dudt, signal, h, rho, P = forces(pos, vel, h)
    t, E_rad = 0.0, 0.0
    snapshots, history = [], [(0.0, sinks.mass.copy())]

    def record():
        snapshots.append(dict(t=t, ids=ids.copy(), pos=pos.copy(), vel=vel.copy(), mass=mass.copy(), rho=rho.copy(), h=h.copy(),
                              sink_mass=sinks.mass.copy(), sink_pos=sinks.pos.copy(), sink_vel=sinks.vel.copy(),
                              sink_spin=sinks.spin.copy(), E_rad=E_rad))

    record()
    for t_next in t_out[1:]:
        while t < t_next:
            a_mag = np.linalg.norm(acc, axis=1)
            dt = min(courant * np.min(h / signal), courant * np.min(np.sqrt(h / np.maximum(a_mag, 1e-300))),
                     dt_max, t_next - t)
            if len(sinks):
                dt = min(dt, courant * np.min(np.sqrt(r_acc / np.maximum(np.linalg.norm(sink_acc, axis=1), 1e-300))))
                # no gas particle may cover more than ~courant of its distance to a sink in one step,
                # or it could fly through the accretion sphere between two checks
                dx = pos[:, None, :] - sinks.pos[None, :, :]
                dv = vel[:, None, :] - sinks.vel[None, :, :]
                dt = min(dt, courant * np.min(np.linalg.norm(dx, axis=2) / np.maximum(np.linalg.norm(dv, axis=2), 1e-300)))
            vel += 0.5 * dt * acc
            sinks.vel += 0.5 * dt * sink_acc
            pos += dt * vel
            sinks.pos += dt * sinks.vel
            dudt_old = dudt
            acc, sink_acc, dudt, signal, h, rho, P = forces(pos, vel + 0.5 * dt * acc, h)
            vel += 0.5 * dt * acc
            sinks.vel += 0.5 * dt * sink_acc
            E_rad += 0.5 * dt * np.sum(mass * (dudt_old + dudt))
            t = t_next if t_next - t <= dt else t + dt

            removed = accrete(sinks, pos, vel, mass, r_acc)
            keep = np.ones(len(mass), dtype=bool)
            keep[removed] = False
            pos, vel, mass, h, rho, P, ids = pos[keep], vel[keep], mass[keep], h[keep], rho[keep], P[keep], ids[keep]
            created = try_create_sink(sinks, pos, vel, mass, rho, 1.5 * P / rho, r_acc, rho_sink)
            keep = np.ones(len(mass), dtype=bool)
            keep[created] = False
            pos, vel, mass, h, rho, P, ids = pos[keep], vel[keep], mass[keep], h[keep], rho[keep], P[keep], ids[keep]
            history.append((t, sinks.mass.copy()))
            if len(mass) < MIN_GAS:
                record()
                return snapshots, history, sinks, "gas exhausted"
            if len(removed) or len(created):
                acc, sink_acc, dudt, signal, h, rho, P = forces(pos, vel, h)
        record()
    return snapshots, history, sinks, "done"
