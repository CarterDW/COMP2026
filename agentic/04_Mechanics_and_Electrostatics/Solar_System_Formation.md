# Solar System Formation: From Nebula to N-Body

## The Problem

Start with a cold, rotating cloud of gas. Let it collapse into a protostar and a disk,
grow the disk's solids into far too many protoplanets, let them collide and merge, and
see whether something like our solar system comes out. Then let it run as a classic
N-body problem and watch the orbits evolve.

## Why Staged

No single simulation can span this. The scales are too far apart:

| | Start | End | Ratio |
|---|---|---|---|
| Length | cloud core ~20,000 AU | Earth radius ~4e-5 AU | ~1e9 |
| Mass | cloud ~1 Msun | planetesimal ~1e-9 Msun | ~1e9 particles |
| Time | collapse ~1e5 yr | assembly ~1e8 yr at ~0.25 yr/orbit | ~1e9 steps |

So we run **stages with handoffs**: the output of one stage, such as a disk's
surface-density profile, becomes the initial condition of the next. Every handoff is a
place to compare against known numbers.

## Conventions

- Units: AU, Msun, yr, so G = 4 pi^2.
- All code is written by us, using numpy, matplotlib and numba. No external N-body codes.
- Short, readable code. No `try/except` fallbacks or silent defaults.
- Stop for review after every rung.

## Two Kinds of "Size"

- **Gravitational softening eps.** Replace 1/r^2 with r/(r^2+eps^2)^(3/2). This removes the
  singularity at r -> 0. It is a numerical device, not physics. In the gas stages, the
  SPH smoothing length plays this role.
- **Physical radius R.** Set from mass and density, and used to detect collisions. When two
  bodies touch, they merge: mass and momentum are conserved, and energy is dissipated.

Softening alone does not make close encounters cheap. They still need small timesteps.

## Where Gas Matters

| Stage | Gas role | Treatment |
|---|---|---|
| Collapse | Essential: pressure vs. gravity (Jeans) decides whether it collapses at all | SPH particles |
| Disk | Drag damps embryo eccentricities, and giants accrete envelopes | Analytic disk decaying over ~3 Myr |
| > ~10 Myr | Gone | Pure N-body |

## The Rungs

Each rung ends with a non-trivial check that we understand.

### Rung 0: Conventions
Units, project layout, and a short `CLAUDE.md`.
- **Check:** Earth at 1 AU with v = 2 pi completes one orbit in 1.000 yr.

### Rung 1: Gravity + Leapfrog
Direct-sum softened gravity and a kick-drift-kick (symplectic) integrator.
- **Check:** The period of an eccentric Kepler orbit matches Kepler's third law.
- **Check:** The energy error is bounded, with no secular drift, and scales as dt^2.
- **Check:** Angular momentum is conserved to machine precision.

### Rung 2: Pressureless Collapse
A cold, uniform sphere of N particles with no gas.
- **Check:** Collapse occurs at t_ff = sqrt(3 pi / (32 G rho)).
- **Check:** Without softening, energy conservation fails and a smaller dt does not fix it,
  because random sampling always makes a few very close pairs. With softening, the error
  falls as dt^2. This shows why particles need a size.
- **Check:** The ejected mass matches the published fit f = 0.048 + 0.022 ln N (Joyce, Marcos &
  Sylos Labini 2009). The bound remainder settles to 2K/|W| = 1.
- **Check:** Ejection nearly vanishes for a warm start (Q0 = 0.5) or a centrally concentrated
  cloud (rho ~ r^-2), as reported by Sylos Labini (2012, 2013).
- **Found:** How small the cloud gets is set by N (roughly N^(-1/3)), not by eps. A real cloud
  stays bound because gas dissipates energy, not because of pressure alone. See
  `solar_system/rung2_cold_collapse/literature_review.md`.

### Rung 3: Gas via SPH
Kernel density estimate, isothermal equation of state, pressure force, and artificial viscosity
(the dissipation that stops a gas cloud bouncing like collisionless particles).
- **Check:** The density of a uniform particle lattice is recovered, and pair forces conserve momentum,
  angular momentum and energy exactly.
- **Check (Evrard 1988):** An adiabatic collapse conserves energy to ~2e-4 and matches GADGET-1's
  published energy curves at the same particle number.
- **Check (Jeans test):** U/|W| < 5/pi^2 always collapses (Truelove et al. 1998), and less than one
  Jeans mass (U/|W| > 1) does not. The measured threshold, U/|W| = 0.71-0.78, lies in between.
- **Check:** An isothermal cloud's core free-falls until the boundary rarefaction arrives, and
  K + W + E_radiated is conserved.
- **Found:** Random particle positions give ~40% density noise, so SPH starts from a lattice. The
  energy equation must use only the particle's own pressure; the fully symmetric form drove u
  negative.

### Rung 4: Rotating Collapse -> Protostar + Disk
Add sink particles: gas above a density threshold becomes a single accreting star.
- **Check:** Total angular momentum is conserved, including the sinks.
- **Check:** The disk radius is close to the centrifugal radius r_c = j^2 / (G M).
- **Check:** At least 95% of the mass ends up bound, unlike the ~20% loss in Rung 2 (measured 100%).
- **Check:** Sink creation and accretion conserve mass, momentum and angular momentum (orbit + spin)
  to ~1e-15. Accretion onto a sink follows analytic radial infall.
- **Found:**
  - A uniform rotating cloud on a bare lattice fragments into ~4 equal companions. The fix is a
    randomly rotated, jittered lattice, a rho ~ 1/r cloud, and a smaller spin (beta = 0.02).
    Protostellar heating, T = 280 K (d/AU)^-1/2, made no difference at N = 5000.
  - At N = 12000 the result is one 0.74 Msun star with a 0.21 Msun disk peaking at ~100 AU
    (Toomre Q ~ 1).
  - The peak accretion rate, ~1e-4 Msun/yr, is close to the Larson-Penston-Hunter rate
    46.9 cs^3/G, not the Shu rate.
  - Viscosity spreads the disk to ~1.5x beyond j^2/GM.
- **Handoff:** `solar_system/rung4_protostar_disk/handoff.npz`, holding the disk surface density
  Sigma(R), the stellar mass, and the sink's accretion history.

### Rung 4b: Protostar Sub-Grid Model -> Stellar Ignition
The protostar is ~1e3 times smaller than our resolution, so attach a one-zone stellar model to
the sink, driven by the sink's own accretion history. The model has three parts:
- polytrope central temperature, T_c ~ 7.6e6 K (M/Msun)(Rsun/R);
- Kelvin-Helmholtz contraction on the Hayashi track, then the Henyey track;
- deuterium burning at ~1.5e6 K, and hydrogen ignition at ~1e7 K.

Output: a "stellar ignition" GIF on a log-time axis, showing luminosity components (accretion,
contraction, deuterium, hydrogen) and blackbody color from Teff.
- **Check:** The polytrope temperatures and the analytic Hayashi solution R(t) = (R0^-3 + 3 A t)^(-1/3).
- **Check:** Contraction from 3.1 to 1.4 Rsun takes ~4.6 Myr, and T_c = 1e7 K occurs near 1 Rsun
  at ~25 Myr, matching the BHAC15 1 Msun track (Baraffe et al. 2015).
- **Found:**
  - The one-zone 1 Msun star matches BHAC15 to 5-30%: 2.45e6 K at 3.1 Rsun, 4.8 Myr to reach
    1.4 Rsun, and the main sequence at 34 Myr with 0.76 Rsun. An n = 1.5 star is slightly too compact.
  - Driven by the Rung 4 accretion history plus a fitted tail (tau = 12 kyr), our star ends at
    0.82 Msun. It hits the deuterium thermostat at 20 kyr and reaches the main sequence at 58 Myr
    (0.62 Rsun, 0.30 Lsun, 5430 K).
  - During accretion, the split of the surface light into contraction and deuterium is
    bookkeeping only; after accretion ends it is exact.

### Rung 5: Disk -> Planetary Embryos, with Mergers (giant-planet zone, 4-30 AU)
The Rung 4 disk's mass and angular momentum define a viscous (Lynden-Bell & Pringle) disk at 1 Myr, about 0.85x the
minimum-mass solar nebula at 5 AU. Embryos and planetesimals are evolved for 2 Myr with a hybrid Wisdom-Holman
integrator (exact Kepler drifts plus encounter switching) and perfect mergers. The gas physics is:
- drag on planetesimals and tidal damping of embryos;
- disk dispersal;
- shared, gap-limited gas accretion (Tanigawa & Tanaka 2016);
- pebble accretion (Lambrechts & Johansen 2014);
- type I/II migration (Paardekooper et al. 2011; Kanagawa et al. 2018);
- a two-alpha disk: alpha_acc = 1e-3, alpha_turb = 1e-4.

Composition is tracked as seed solids, collisions, pebbles and gas.
- **Check:** The Kepler solver is exact to ~1e-12 for e up to 0.9999. The integrator is second order and
  reversible, matches a leapfrog reference, and conserves energy through close encounters and mergers.
- **Check:** The viscous disk solves its diffusion equation and conserves angular momentum. Drag, damping, gas
  accretion, pebble flux and migration torques match their published formulas and limits.
- **Found:**
  - Classic planetesimal growth stalls below 10 Mearth: no giants (the core-growth timescale problem).
  - Pebbles make giants. Without migration they are too massive and too far out. With alpha_acc = alpha_turb =
    1e-3, migration drags every planet into the inner disk.
  - With the two-alpha disk, one 2.5 MJ giant with a ~16 Mearth core survives at 6.1 AU, having migrated in from
    26 AU. All other cores are lost to type I migration.
- **Missing physics noted:**
  - magnetic braking and outflows, which would give a smaller disk;
  - migration traps from a realistic disk thermal structure.
- **Handoff:** `solar_system/rung5_planet_formation/handoff.npz` (the surviving giant's orbit and mass).

### Rung 6: Late-Stage Assembly (~100 Myr)
~100-200 embryos between 0.5 and 4 AU, with Jupiter and Saturn present.
- **Check:** Over several realizations, compare the distributions of planet count,
  planet masses, and angular momentum deficit (AMD) with Chambers (2001) and with the
  real solar system. The outcomes are chaotic, so only the statistics are meaningful.

### Rung 7: Long-Term Orbital Evolution
Integrate our system and the real one for Myr timescales.
- **Check:** The energy error stays bounded over ~1e8 steps.
- **Check:** Two copies offset by delta give the Lyapunov time of the inner planets
  (~5 Myr, Laskar). This connects back to the butterfly-effect project.

## Push Harder

- Replace the gas-giant accretion rule with real envelope physics.
- Barnes-Hut tree gravity, for N > ~1e4.
- Add moons.
- General-relativistic correction for Mercury's perihelion precession.
