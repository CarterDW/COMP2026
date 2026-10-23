# Warmup: The Kosterlitz–Thouless Transition at Equilibrium

## The Problem

The 2D XY model puts a unit spin at each site, at angle θ_i:

$$H = -J\sum_{\langle ij\rangle}\cos(\theta_i - \theta_j).$$

Its symmetry is continuous (rotate every θ by the same amount). The Mermin–Wagner theorem
says a continuous symmetry cannot break in 2D at T > 0, so there is no magnetized phase.
Yet there is a transition, at T_KT ≈ 0.893 (Hasenbusch 2005), and it is driven by
**vortices**. Going once around a vortex, θ winds by ±2π.

- **Below T_KT**, vortices exist but sit in tightly bound vortex–antivortex pairs. A pair
  barely disturbs distant spins, so correlations decay as a power law,
  G(r) ∝ r^(−η(T)). The spin stiffness Υ is finite.
- **Above T_KT**, the pairs unbind. Free vortices scramble the phase, correlations decay
  exponentially, and Υ = 0.
- The Kosterlitz–Thouless argument: one free vortex costs energy πJ ln(L/a) but gains
  entropy 2 ln(L/a), since it can sit anywhere. Free energy F = (πJ − 2T) ln(L/a) changes
  sign at T = πJ/2. Screening by bound pairs lowers this to the real T_KT.
- **Nelson–Kosterlitz:** at T_KT the stiffness jumps from 2T_KT/π to 0. The ratio is
  universal and does not depend on lattice details.

The warmup is one session. It builds an XY engine and watches the pairs unbind. It also
locates T_KT with the stiffness and shows why counting vortices cannot locate it. It is
the equilibrium baseline for the semester's later Kibble–Zurek work on XY.

## Conventions

- J = 1, k_B = 1, L × L square lattice, periodic boundaries.
- **Engine:** checkerboard Metropolis, the same structure as the Ising plan's step 1. The
  move is θ → θ + δ·U(−1, 1), with δ tuned to about 50% acceptance at each T. Arrays are
  (R, L, L) over replicas. Write it so the spin type can be swapped for Ising later.
- One sweep = L² attempted moves.
- **Vortex charge** on a plaquette: sum the four bond differences, each wrapped to
  (−π, π], and divide by 2π. The result is q ∈ {−1, 0, +1}.
- **Stiffness** (helicity modulus), x-bonds only:

  $$\Upsilon = \frac{1}{N}\Big\langle\sum_x \cos\Delta\theta\Big\rangle - \frac{\beta}{N}\Big\langle\Big(\sum_x \sin\Delta\theta\Big)^2\Big\rangle.$$

  Average it with the y-bond version.
- **Errors:** autocorrelation-corrected (blocking), as in notebook 06.
- Run everything in the `HaloPipeline` env.

## Natural Steps

Stop for sign-off after each step.

1. **The XY engine.**
   *Checks:*
   - **A single spin in a fixed field is exact.** Freeze the four neighbours of one spin.
     Its local field is h = Σ_j (cos θ_j, sin θ_j). The sampled θ must follow the von
     Mises distribution p(θ) ∝ e^(β|h| cos(θ − φ_h)). This is the continuous analog of
     the Ising plan's 3×3 enumeration: it tests detailed balance directly.
   - **Low T: equipartition.** At T ≤ 0.3, u = −2 + T/2, since each spin contributes one
     quadratic spin-wave mode.
   - **High T: expansion.** At T ≥ 3, u ≈ −1/T, since each bond gives ⟨cos⟩ ≈ β/2.

   *About 25 min.*

2. **Counting vortices.**
   *Checks:*
   - Total charge Σq = 0 on every configuration, as periodic boundaries require.
   - Plant a vortex–antivortex pair by hand, θ = arg(r − r₁) − arg(r − r₂). The counter
     finds exactly +1 and −1 at the right plaquettes, and nothing else.
   - A uniform configuration has zero vortices; a fully random one has a vortex density
     of about 1/3 per plaquette.

   *About 20 min.*

3. **The transition.** Sweep T from 0.6 to 1.4 in steps of 0.05, with a finer grid of
   0.02 between 0.8 and 1.0. Run L = 16, 32 and 64, with replicas. At each T measure u,
   the specific heat C = N(⟨u²⟩ − ⟨u⟩²)/T², the vortex density n_v and Υ.
   *Checks and figures:*
   - **Υ(T) for each L, with the line 2T/π.** Υ → 1 as T → 0, and Υ → 0 at high T. The
     crossing T*(L) estimates T_KT and drifts down as L grows. For these sizes, expect
     about 0.90–0.95, approaching 0.893 only logarithmically. Report this as the known
     log finite-size effect, not a discrepancy.
   - **Vortex density n_v(T) is smooth through T_KT.** It has no kink. C(T) peaks near
     T ≈ 1.0, about 15% above T_KT, not at T_KT. This shows the vortex count and the
     specific heat do not mark the transition; the stiffness does.
   - **Snapshots** at T = 0.7, 0.9 and 1.2, with spins as arrows or colour and vortices
     marked. Expect tight pairs, then loosening pairs, then free vortices.
   - **Equilibration:** start from both ordered and random configurations at T = 0.9 and
     L = 64. Both must reach the same Υ and n_v before you average.

   *About 50–55 min.*

**Total:** about 100 minutes. If time runs short, drop L = 64 from step 3 and keep the
snapshots.

## Push Harder

- **Weber–Minnhagen fit.** Υ_L(T_KT) = (2T_KT/π)[1 + 1/(2 ln L + C)]. Fit it across L to
  extrapolate T_KT, and compare with 0.8929.
- **η = 1/4 at T_KT.** Measure G(r) = ⟨cos(θ_0 − θ_r)⟩ below T_KT. Check that
  η(T) = T/(2πΥ(T)), which gives 1/4 at the transition. At low T this tends to the
  spin-wave value η ≈ T/2π.
- **Pair separations.** Match each vortex to its nearest antivortex. Below T_KT the
  separation distribution is cut off at small distances. Above T_KT it extends to the
  system size.
- **Faster equilibrium.** Over-relaxation moves (θ → 2φ_h − θ, which do not change the
  energy) or a Wolff cluster update. These are for equilibrium only. Never use them in
  Kibble–Zurek runs, because they change the dynamics and therefore z.

## Reading

**Before step 1:**
- Kardar, *Statistical Physics of Fields* (CUP 2007), ch. 8: the KT transition, at the
  level we need.
- Tobochnik & Chester, "Monte Carlo study of the planar spin model," PRB **20**, 3761
  (1979). [doi:10.1103/PhysRevB.20.3761](https://doi.org/10.1103/PhysRevB.20.3761). The
  classic XY Monte Carlo, with vortex snapshots very like the ones we make.

**Before step 3:**
- Kosterlitz & Thouless, "Ordering, metastability and phase transitions in
  two-dimensional systems," J. Phys. C **6**, 1181 (1973).
  [doi:10.1088/0022-3719/6/7/010](https://doi.org/10.1088/0022-3719/6/7/010). Read §1–3
  for the energy–entropy argument.
- Nelson & Kosterlitz, "Universal jump in the superfluid density of two-dimensional
  superfluids," PRL **39**, 1201 (1977).
  [doi:10.1103/PhysRevLett.39.1201](https://doi.org/10.1103/PhysRevLett.39.1201). Where
  the 2T/π comes from.
- Weber & Minnhagen, "Monte Carlo determination of the critical temperature for the
  two-dimensional XY model," PRB **37**, 5986 (1988).
  [doi:10.1103/PhysRevB.37.5986](https://doi.org/10.1103/PhysRevB.37.5986). How to read
  Υ crossings at finite L.

**For reference and the Push Harder items:**
- Kosterlitz, "Kosterlitz–Thouless physics: a review of key issues," Rep. Prog. Phys.
  **79**, 026001 (2016).
  [doi:10.1088/0034-4885/79/2/026001](https://doi.org/10.1088/0034-4885/79/2/026001).
- Hasenbusch, "The two-dimensional XY model at the transition temperature: a
  high-precision Monte Carlo study," J. Phys. A **38**, 5869 (2005).
  [doi:10.1088/0305-4470/38/26/003](https://doi.org/10.1088/0305-4470/38/26/003). The
  reference value T_KT = 0.8929.
- Gupta & Baillie, "Critical behavior of the two-dimensional XY model," PRB **45**, 2883
  (1992). [doi:10.1103/PhysRevB.45.2883](https://doi.org/10.1103/PhysRevB.45.2883).
  Specific-heat peak and vortex density across the transition.
