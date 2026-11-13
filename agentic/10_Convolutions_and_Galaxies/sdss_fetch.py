"""Download real SDSS cutouts in exactly the Galaxy Zoo format (120 x 120, 0.20"/px,
SkyServer DR16 color JPGs, same quantization tables).

  stars/  spectroscopically confirmed stars (SpecObj class STAR, PhotoObj type 6),
          500 per 1-mag bin in psfMag_r from 14 to 20
  sky/    positions 3-6 arcmin from Galaxy Zoo galaxies (inside the imaging footprint)

Writes to notebooks/data/sdss_week10/ (gitignored) with a CSV of what was fetched.
"""
import csv
import io
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import data

BASE = "https://skyserver.sdss.org/dr16/SkyServerWS"
OUT = data.NB / "data" / "sdss_week10"
SCALE, SIZE = 0.20, 120


def sql(q):
    url = f"{BASE}/SearchTools/SqlSearch?" + urllib.parse.urlencode({"cmd": q, "format": "csv"})
    lines = urllib.request.urlopen(url, timeout=120).read().decode().strip().splitlines()
    head = lines[1].split(",")
    return [dict(zip(head, l.split(","))) for l in lines[2:]]


def cutout(ra, dec):
    """JPEG bytes, or None where the server answers 404: no imaging at that position
    (just off the survey footprint). Any other error is retried, then raised."""
    url = f"{BASE}/ImgCutout/getjpeg?" + urllib.parse.urlencode(
        {"ra": ra, "dec": dec, "scale": f"{SCALE:.2f}", "width": SIZE, "height": SIZE})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"cutout failed 3 times at {ra}, {dec}")


def fetch_all(rows, folder):
    folder.mkdir(parents=True, exist_ok=True)

    def one(r):
        path = folder / f"{r['name']}.jpg"
        if path.exists():
            return True
        jpg = cutout(r["ra"], r["dec"])
        time.sleep(0.05)
        if jpg is None:
            return False
        path.write_bytes(jpg)
        return True

    with ThreadPoolExecutor(4) as pool:
        got = []
        for k, ok in enumerate(pool.map(one, rows)):
            got.append(ok)
            if k % 500 == 0:
                print(f"  {folder.name}: {k} / {len(rows)}", flush=True)
    kept = [r for r, ok in zip(rows, got) if ok]
    print(f"  {folder.name}: {len(kept)} fetched, {len(rows) - len(kept)} positions had no imaging (404)")
    with open(folder / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)


def star_rows():
    rows = []
    for m in range(14, 20):
        q = (f"select top 500 p.objid, p.ra, p.dec, p.psfMag_g, p.psfMag_r, p.psfMag_i, s.subclass "
             f"from SpecObj s join PhotoObj p on s.bestobjid = p.objid "
             f"where s.class = 'STAR' and p.type = 6 and p.clean = 1 "
             f"and p.psfMag_r between {m} and {m + 1} and (p.objid % 7) = 3")
        got = sql(q)
        print(f"stars {m}-{m + 1}: {len(got)}", flush=True)
        rows += [{"name": f"star_{r['objid']}", **r} for r in got]
    return rows


def sky_rows(n, seed=0):
    images, labels, ids = data.load()
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, n, replace=False)
    pos = {}
    for k in range(0, n, 50):                          # dr7 -> dr16 positions; 50 per query
        chunk = ",".join(pick[k:k + 50])                # keeps the GET URL short (400 gave 404)
        for r in sql(f"select d.dr7objid, p.ra, p.dec from PhotoObjDR7 d join PhotoObj p "
                     f"on d.dr8objid = p.objid where d.dr7objid in ({chunk})"):
            pos[r["dr7objid"]] = (float(r["ra"]), float(r["dec"]))
    rows = []
    for k, i in enumerate(pick):
        if i not in pos:
            continue
        ra, dec = pos[i]
        sep, ang = rng.uniform(3, 6) / 60, rng.uniform(0, 2 * np.pi)
        rows.append({"name": f"sky_{k:05d}", "ra": ra + sep * np.cos(ang) / np.cos(np.deg2rad(dec)),
                     "dec": dec + sep * np.sin(ang), "near_gz": i})
    return rows


def galaxy_rows():
    """Every Galaxy Zoo galaxy, at its DR16 position: the same objects as the notebook's
    (DR7-era) JPGs, re-made by today's pipeline like the stars and sky. Named by DR7 id."""
    images, labels, ids = data.load()
    rows = []
    for k in range(0, len(ids), 50):
        chunk = ",".join(ids[k:k + 50])
        for r in sql(f"select d.dr7objid, p.ra, p.dec from PhotoObjDR7 d join PhotoObj p "
                     f"on d.dr8objid = p.objid where d.dr7objid in ({chunk})"):
            rows.append({"name": f"gal_{r['dr7objid']}", "ra": r["ra"], "dec": r["dec"],
                         "dr7objid": r["dr7objid"]})
    print(f"galaxies: {len(rows)} of {len(ids)} have a DR16 match", flush=True)
    return rows


def quasar_rows():
    """Spectroscopic quasars that look like point sources (type 6), 500 per 1-mag bin in
    psfMag_r from 17 to 20, the same bins as the stars."""
    rows = []
    for m in range(17, 20):
        q = (f"select top 500 p.objid, p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, s.z "
             f"from SpecObj s join PhotoObj p on s.bestobjid = p.objid "
             f"where s.class = 'QSO' and p.type = 6 and p.clean = 1 and s.zWarning = 0 "
             f"and p.psfMag_r between {m} and {m + 1} and (p.objid % 7) = 3")
        got = sql(q)
        print(f"quasars {m}-{m + 1}: {len(got)}", flush=True)
        rows += [{"name": f"qso_{r['objid']}", **r} for r in got]
    return rows


def quasar_hiz_rows():
    """500 point-source quasars at z = 2-3.5 (r = 17-20), where quasar colors cross the
    stellar locus; the main query's catalog order left almost none there."""
    q = ("select top 500 p.objid, p.ra, p.dec, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, s.z "
         "from SpecObj s join PhotoObj p on s.bestobjid = p.objid "
         "where s.class = 'QSO' and p.type = 6 and p.clean = 1 and s.zWarning = 0 "
         "and s.z between 2.0 and 3.5 and p.psfMag_r between 17 and 20 and (p.objid % 3) = 1")
    rows = [{"name": f"qso_{r['objid']}", **r} for r in sql(q)]
    print(f"high-z quasars: {len(rows)}", flush=True)
    return rows


def star_u_band():
    """psfMag_u for the downloaded stars (for the catalog UV-excess baseline only)."""
    rows = list(csv.DictReader(open(OUT / "stars" / "index.csv")))
    out = []
    for k in range(0, len(rows), 50):
        chunk = ",".join(r["objid"] for r in rows[k:k + 50])
        out += sql(f"select objid, psfMag_u from PhotoObj where objid in ({chunk})")
    with open(OUT / "stars" / "u_band.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["objid", "psfMag_u"])
        w.writeheader()
        w.writerows(out)
    print(f"star u band: {len(out)} of {len(rows)}")


def galaxy_photometry():
    """SDSS model magnitudes (u, g, r) and spectroscopic redshift for every Galaxy Zoo
    galaxy, keyed by DR7 id. Catalog numbers only, no images."""
    images, labels, ids = data.load()
    rows = []
    for k in range(0, len(ids), 50):
        chunk = ",".join(ids[k:k + 50])
        rows += sql(f"select d.dr7objid, p.modelMag_u, p.modelMag_g, p.modelMag_r, s.z "
                    f"from PhotoObjDR7 d join PhotoObj p on d.dr8objid = p.objid "
                    f"left join SpecObj s on s.bestobjid = p.objid "
                    f"where d.dr7objid in ({chunk})")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "galaxy_photometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"galaxy photometry: {len(rows)} rows for {len(ids)} galaxies")


if __name__ == "__main__":
    what = sys.argv[1:] or ["stars", "sky"]
    if "photometry" in what:
        galaxy_photometry()
    if "quasars" in what:
        fetch_all(quasar_rows(), OUT / "quasars")
        star_u_band()
    if "quasars_hiz" in what:
        fetch_all(quasar_hiz_rows(), OUT / "quasars_hiz")
    if "galaxies" in what:
        fetch_all(galaxy_rows(), OUT / "galaxies")
    if "stars" in what:
        fetch_all(star_rows(), OUT / "stars")
    if "sky" in what:
        fetch_all(sky_rows(4000), OUT / "sky")
    print("done")
