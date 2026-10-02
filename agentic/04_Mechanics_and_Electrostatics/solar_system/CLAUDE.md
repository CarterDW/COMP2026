# Solar System Formation

Plan: `../Solar_System_Formation.md`. Work one rung at a time and stop for review after each.

## Units
AU, Msun, yr. `G = 4 pi^2` exactly (the year is the Gaussian year). Conversions live in `nbody/units.py`.
Never hard-code a conversion factor elsewhere.

## Layout
- `nbody/` holds shared library code (units, gravity, integrators, ...). Rungs import from it.
  It is installed in editable mode (`pip install -e .` from `solar_system/`, done once), so
  `import nbody` works from any directory, in scripts, tests and notebooks alike.
- `rungN_<name>/` holds one rung: `test_*.py` (pytest), plain scripts, an optional notebook for
  exploration, and `plots/` for generated figures.

## Running
From any directory:
- `pytest` runs the tests under the current directory.
- `python check_units.py` (or any script path) runs a script. Scripts write their plots to
  their own rung's `plots/` folder.

## Performance notes
- numba-parallel code uses every core. Never run two heavy scripts at once: they oversubscribe the CPU
  and both crawl.
- Slow reference tests are opt-in: `RUN_SLOW=1 pytest`.
- numba's on-disk cache (`cache=True`) can go stale when a *called* function changes, which shows up as a segfault.
  `nbody/hybrid.py` is therefore not cached. If a cached function ever segfaults after an edit, delete
  `nbody/__pycache__/*.nbi` and `*.nbc`.

## Code rules
- Short, readable numpy, with numba only where speed demands it.
- No `try/except` fallbacks and no silent defaults.
- Every rung has non-trivial tests against analytic results or conservation laws, with
  tolerances that have a stated physical reason.
