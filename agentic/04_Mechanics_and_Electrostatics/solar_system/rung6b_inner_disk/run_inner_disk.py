"""Rung 6b: drifting pebbles through the inner disk (0.1-4 AU), from the first pebbles to 3 Myr.

    python path/to/run_inner_disk.py --v-frag-dry V --zeta Z --fit {ly21,lim24} [--n N]

The run starts when the pebble growth front (Lambrechts & Johansen 2014) passes 4 AU: by then all the local dust
inside 4 AU has grown into pebbles (rock inside the snow line, rock + ice outside, Hayashi 1981 abundances). After
that, pebbles arrive through 4 AU (pebble_inflow.py). Every option is required: --v-frag-dry is the fragmentation
speed of dry (silicate) pebbles in m/s (icy ones: 10 m/s), --zeta the conversion per orbit above the
streaming-instability threshold, --fit the threshold (Li & Youdin 2021, or Lim et al. 2024 inside its range).
Writes data/inner_disk_<tag>.npz with snapshots every 10 kyr.
"""
import sys
import time
from pathlib import Path
import numpy as np
import nbody.pebble_disk as pd
from nbody.gas_effects import disk_state, EPS_D, Z_PEBBLES
from nbody.viscous_disk import SNOW_LINE, Z_ROCK, Z_ICE
from nbody.units import G, M_EARTH
from disk_setup import disk_params
from pebble_inflow import Inflow, ICE_FRACTION

HERE = Path(__file__).parent


def option(name):
    assert name in sys.argv, f"missing {name}"
    return sys.argv[sys.argv.index(name) + 1]


V_FRAG_DRY, ZETA, FIT_NAME = float(option("--v-frag-dry")), float(option("--zeta")), option("--fit")
FIT = {"ly21": pd.FIT_LY21, "lim24": pd.FIT_LIM24}[FIT_NAME]
N = int(option("--n")) if "--n" in sys.argv else 200
V_FRAG_ICE, ST_MIN, COURANT, T_END = 10.0, 0.01, 0.4, 3e6
TAG = f"vf{V_FRAG_DRY:g}_zeta{ZETA:g}_{FIT_NAME}"

disk = disk_params()
inflow = Inflow(disk)
c = (3 / 16) ** (1 / 3) * (G * disk[3]) ** (1 / 3) * (EPS_D * Z_PEBBLES) ** (2 / 3)
t = (pd.R_OUT / c) ** 1.5                                          # the growth front reaches 4 AU
edges, centers, areas = pd.grid(N)
sigma_g = np.array([disk_state(r, t, disk)[0] for r in centers])
rock = Z_ROCK * sigma_g
ice = np.where(centers > SNOW_LINE, (Z_ICE - Z_ROCK) * sigma_g, 0.0)
plts_rock, plts_ice, ledger = np.zeros(N), np.zeros(N), np.zeros(4)
M0 = np.sum((rock + ice) * areas)
print(f"{TAG}: start {t:.0f} yr with {M0 / M_EARTH:.2f} Mearth of local pebbles inside 4 AU", flush=True)

snap = dict(t=[], rock=[], ice=[], plts_rock=[], plts_ice=[], ledger=[])


def record():
    for key, value in zip(snap, (t, rock, ice, plts_rock, plts_ice, ledger)):
        snap[key].append(np.copy(value))


record()
wall, n_steps = time.time(), 0
t_next = 1e4
while t < T_END:
    t_chunk = min(t + 1e3, t_next, T_END)
    rate = inflow(0.5 * (t + t_chunk))
    n_steps += pd.advance(rock, ice, plts_rock, plts_ice, edges, centers, areas, t, t_chunk, disk, rate, ICE_FRACTION,
                          V_FRAG_DRY, V_FRAG_ICE, ZETA, ST_MIN, FIT, ledger, COURANT)
    t = t_chunk
    if t >= t_next - 1e-6:
        record()
        t_next += 1e4
        if len(snap["t"]) % 10 == 1:
            print(f"t = {t / 1e6:.2f} Myr ({time.time() - wall:.0f} s, {n_steps} steps): pebbles "
                  f"{np.sum((rock + ice) * areas) / M_EARTH:.3f}, planetesimals {ledger[3] / M_EARTH:.3f}, "
                  f"in {ledger[0] / M_EARTH:.1f}, onto star {ledger[1] / M_EARTH:.1f}, vapor {ledger[2] / M_EARTH:.1f} Mearth",
                  flush=True)

(HERE / "data").mkdir(exist_ok=True)
budget = (M0 + ledger[0] - np.sum((rock + ice) * areas) - ledger[1] - ledger[2] - ledger[3]) / (M0 + ledger[0])
np.savez(HERE / "data" / f"inner_disk_{TAG}.npz", **{k: np.array(v) for k, v in snap.items()}, edges=edges,
         centers=centers, areas=areas, disk=disk, v_frag_dry=V_FRAG_DRY, zeta=ZETA, fit=FIT_NAME, M0=M0)
print(f"done in {time.time() - wall:.0f} s; mass budget residual {budget:.1e}; planetesimals "
      f"{ledger[3] / M_EARTH:.3f} Mearth ({np.sum(plts_rock * areas) / M_EARTH:.3f} rock)", flush=True)
