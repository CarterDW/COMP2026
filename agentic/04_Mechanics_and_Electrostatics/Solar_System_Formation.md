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
- **Check:** With eps -> 0, energy conservation fails at the bounce. This shows why
  particles need a size.

### Rung 3: Gas via SPH
Kernel density estimate, isothermal equation of state, and pressure force.
- **Check:** The density of a uniform particle lattice is recovered.
- **Check (Jeans test):** A cloud above the Jeans mass collapses, and one below it does not.

### Rung 4: Rotating Collapse -> Protostar + Disk
Add sink particles: gas above a density threshold becomes a single accreting star.
- **Check:** Total angular momentum is conserved, including the sinks.
- **Check:** The disk radius is close to the centrifugal radius r_c = j^2 / (G M).
- **Handoff:** the disk surface density Sigma(r) and the stellar mass.

### Rung 5: Disk -> Planetary Embryos, with Mergers
Seed planetesimals and embryos from Sigma(r). Compare against the minimum-mass solar
nebula, Sigma ~ r^(-3/2). Add a decaying analytic gas disk for drag, perfect-merging
collisions, and a simple rule for gas giants: a core above ~10 Earth masses accretes gas.
- **Check:** A head-on two-body merger conserves momentum exactly.
- **Check:** The energy budget closes once collision losses are counted.
- **Check:** Embryo masses approach the theoretical isolation mass.

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
