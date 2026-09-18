"""Access to the IllustrisTNG public data, fetching only what rung 1 needs.

Two endpoints do all the work:

  field-filtered group catalogue
      /api/{sim}/files/groupcat-{snap}/?Subhalo={FieldName}
      One subhalo field at a time. This matters: the full group catalogue for
      TNG300-1 at z=0 is 13 GB, while the six fields below total ~1.3 GB.

  per-subhalo star cutout
      /api/{sim}/snapshots/{snap}/subhalos/{id}/cutout.hdf5?stars=Coordinates,Masses
      Exactly the star particles bound to one subhalo, two fields, no offset
      bookkeeping needed.

Authentication is an ``api-key`` header. Get a key by registering (free) at
https://www.tng-project.org/users/register/ and then either export
TNG_API_KEY or drop the key in ~/.tng_api_key.

Units, as TNG stores them
-------------------------
    SubhaloPos, SubhaloHalfmassRadType, star Coordinates   ckpc/h
    SubhaloMassType, star Masses                           1e10 Msun/h
    TNG300 box                                             205000 ckpc/h

Nothing here converts silently. ``MPC_PER_CKPC`` and ``MASS_TO_MSUN`` are
provided and applied explicitly at the call site.
"""

import os
import time
from pathlib import Path

import h5py
import numpy as np
import requests

BASE = "https://www.tng-project.org/api"

# Box size is exact. The particle masses are the published target resolutions,
# kept here only for labelling plots -- nothing computes with them. TNG300-3 is
# the interesting one: at ~7e8 Msun per baryon element it sits close to
# FLAMINGO's m9 resolution, with TNG physics and TNG300-1's initial conditions.
SIMS = {
    "TNG300-1": {"boxsize_ckpc": 205000.0, "m_baryon_msun": 1.1e7, "m_dm_msun": 5.9e7},
    "TNG300-2": {"boxsize_ckpc": 205000.0, "m_baryon_msun": 8.8e7, "m_dm_msun": 4.7e8},
    "TNG300-3": {"boxsize_ckpc": 205000.0, "m_baryon_msun": 7.0e8, "m_dm_msun": 3.8e9},
}

SUBHALO_FIELDS = (
    "SubhaloFlag",           # 0 marks objects of non-cosmological origin -- cut these
    "SubhaloMassType",       # (N, 6); index 4 is stars
    "SubhaloLenType",        # (N, 6); index 4 is the star particle count
    "SubhaloPos",            # (N, 3) ckpc/h
    "SubhaloHalfmassRadType",  # (N, 6); index 4 sets the shape aperture
    "SubhaloGrNr",           # parent FoF group, for central/satellite splits
)

MPC_PER_CKPC = 1.0e-3
MASS_TO_MSUN = 1.0e10
STARS = 4  # TNG particle type index for stars


def api_key():
    """The user's TNG API key, from the environment or ~/.tng_api_key.

    Raises rather than proceeding anonymously: an unauthenticated request comes
    back as HTML, and the resulting h5py error is far from the real cause.
    """
    key = os.environ.get("TNG_API_KEY")
    if key:
        return key.strip()
    path = Path.home() / ".tng_api_key"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        "No TNG API key. Register at https://www.tng-project.org/users/register/ "
        "then either 'export TNG_API_KEY=...' or write the key to ~/.tng_api_key"
    )


def _download(url, dest, max_retries=5):
    """Stream a URL to ``dest``, skipping the fetch if the file is already there.

    Retries only on the transient HTTP codes (429 rate limit, 5xx), because a
    full sample is tens of thousands of requests and one hiccup should not end
    the run. Every other status raises -- a 404 on a subhalo id is a bug in the
    caller, not something to paper over.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {"api-key": api_key()}
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        if response.status_code in (429, 500, 502, 503, 504):
            wait = 2.0 ** attempt
            print(f"  HTTP {response.status_code} on {url} -- retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        partial = dest.with_suffix(dest.suffix + ".part")
        with open(partial, "wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
        partial.rename(dest)  # rename last, so an interrupted run leaves no
        return dest           # truncated file that a later run would trust
    raise RuntimeError(f"gave up on {url} after {max_retries} attempts")


def subhalo_catalogue(sim, snap, cache, fields=SUBHALO_FIELDS):
    """Load the requested subhalo fields, downloading each one only once.

    Returns a dict of arrays, one entry per field, in TNG's native units.
    """
    cache = Path(cache) / sim / f"groupcat-{snap:03d}"
    out = {}
    for field in fields:
        dest = cache / f"{field}.hdf5"
        _download(f"{BASE}/{sim}/files/groupcat-{snap}/?Subhalo={field}", dest)
        with h5py.File(dest, "r") as handle:
            out[field] = handle["Subhalo"][field][:]
    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"subhalo fields disagree on length: {lengths}")
    return out


def star_cutout(sim, snap, subhalo_id, cache):
    """(coordinates, masses) of the star particles bound to one subhalo.

    Coordinates in ckpc/h, masses in 1e10 Msun/h -- i.e. exactly as stored.
    """
    dest = Path(cache) / sim / f"cutouts-{snap:03d}" / f"{subhalo_id}.hdf5"
    _download(
        f"{BASE}/{sim}/snapshots/{snap}/subhalos/{subhalo_id}/"
        "cutout.hdf5?stars=Coordinates,Masses",
        dest,
    )
    with h5py.File(dest, "r") as handle:
        group = handle[f"PartType{STARS}"]
        return group["Coordinates"][:], group["Masses"][:]


def select_galaxies(catalogue, min_stellar_mass_msun, min_star_particles):
    """Indices of subhalos usable as galaxies, with the cuts stated explicitly.

    SubhaloFlag == 0 flags subhalos that are probably not of cosmological
    origin (fragments of discs and the like); TNG's own documentation says to
    discard them for galaxy studies, so they are dropped rather than trimmed
    later.
    """
    stellar = catalogue["SubhaloMassType"][:, STARS] * MASS_TO_MSUN
    n_star = catalogue["SubhaloLenType"][:, STARS]
    keep = (
        (catalogue["SubhaloFlag"] != 0)
        & (stellar >= min_stellar_mass_msun)
        & (n_star >= min_star_particles)
    )
    return np.flatnonzero(keep)
