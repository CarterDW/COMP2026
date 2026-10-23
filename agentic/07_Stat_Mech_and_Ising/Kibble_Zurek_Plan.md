# Defects from Cooling: Kibble–Zurek and Domain-Wall Scaling

## The Problem

When the early universe cooled through a symmetry-breaking transition, causally
disconnected regions chose their vacuum independently. The boundaries between them are
**topological defects**. For a discrete Z₂ symmetry like the Ising model's, they are
**domain walls**. Two questions decide whether those walls matter cosmologically:

1. **How many form?** (Kibble 1976, Zurek 1985.) Near T_c the correlation length
   ξ ∝ |ε|^(−ν) and the relaxation time τ ∝ ξ^z both diverge, where ε = (T − T_c)/T_c.
   A system cooled at a finite rate cannot keep up. It falls out of equilibrium at the
   **freeze-out** time t̂, when τ(t̂) = t̂. Domains are then frozen at size

   $$\hat\xi \propto \tau_Q^{\,\nu/(1+z\nu)}, \qquad \hat t \propto \tau_Q^{\,z\nu/(1+z\nu)},$$

   where τ_Q is the quench time, ε(t) = −t/τ_Q. For the 2D Ising model, ν = 1 exactly and
   z ≈ 2.167 for local (Metropolis) dynamics. So ξ̂ ∝ τ_Q^0.316: cooling 10× slower makes
   domains only about 2× larger. In the universe, τ_Q is the Hubble time at T_c.

2. **What happens to them afterwards?** Below T_c the walls straighten and annihilate, and
   domains **coarsen**. On a lattice with dissipative dynamics, the Allen–Cahn law gives
   ℓ ∝ t^(1/2). For relativistic walls in an expanding universe, the network instead
   reaches **scaling**: about one wall per horizon, so ℓ ∝ t. That is fatal. Wall energy
   density falls as ρ_w ∝ σ/t, while radiation falls as 1/t², so walls dominate eventually
   (the Zel'dovich–Kobzarev–Okun domain-wall problem, 1974).

The project builds a fast Ising engine and measures z itself. It tests Kibble–Zurek
quantitatively, then follows the walls into the coarsening regime. The last step
contrasts that regime with a relativistic φ⁴ field in an expanding background: same
symmetry, same walls, different dynamics, different growth law.

Problems 1–5 of [07_Stat_Mech_and_Ising.ipynb](../../notebooks/07_Stat_Mech_and_Ising.ipynb)
are the starting point: the lattice, Metropolis, the T sweep, equilibration and C(r).

## Conventions

- J = 1, k_B = 1, periodic boundaries, square lattice L × L.
- T_c = 2 / ln(1 + √2) ≈ 2.2692.
- **Time unit: one sweep** = N = L² attempted flips. Every exponent comparison needs the
  same dynamics everywhere. Only exponents are universal; prefactors like τ₀ depend on the
  update rule. So z (step 2) and the quenches (steps 3–5) must use the identical update.
- **Quench protocol:** T(t) = T_c(1 − t/τ_Q). Start equilibrated at T_i = 1.5 T_c
  (t_i = −τ_Q/2) and cross T_c at t = 0.
- **Observables.**
  - Structure factor S(**k**, t) = |FFT(s)|²/N, radially averaged.
  - Second-moment correlation length ξ₂ = √(S(0)/S(k_min) − 1) / (2 sin(k_min/2)), with
    k_min = 2π/L. Use it in the disordered phase and at the crossing.
  - Coarsening length ℓ = 2π / ⟨k⟩_S (first moment of S), and the excess energy over
    equilibrium, Δu(t) = u(t) − u_eq(T(t)), which is proportional to the wall length per
    site, i.e. 1/ℓ.
- **Exact equilibrium references** (Onsager, Yang), with K the complete elliptic integral
  of the first kind:
  - u(β) = −coth(2β)[1 + (2/π)(2tanh²(2β) − 1)K(κ)], κ = 2sinh(2β)/cosh²(2β). In SciPy
    that is `ellipk(κ**2)`.
  - M(β) = [1 − sinh⁻⁴(2β)]^(1/8) below T_c.
- Run everything in the `HaloPipeline` env.

## Natural Steps

Stop for sign-off after each step.

1. **A fast engine.** The notebook's site-by-site class is far too slow for ~10⁹ spin
   updates. Write a vectorized **checkerboard Metropolis**: update all black sites, then
   all white sites. Each colour's neighbours are all the other colour, so a half-sweep is
   exact. Batch over replicas, with spins shaped (R, L, L), so one array op advances R
   independent runs. Add numba only if numpy is too slow.
   *Checks:*
   - **Detailed balance, exactly.** On a 3 × 3 lattice (2⁹ = 512 states), the long-run
     state histogram matches the Boltzmann weights e^(−βE)/Z, with Z by enumeration
     (χ² test). This catches update-order bugs that equilibrium averages can hide.
   - At L = 64, ⟨u⟩(T) and ⟨|M|⟩(T) match Onsager and Yang within error bars, at
     T/T_c = 0.8, 0.95, 1.05, 1.3. Use autocorrelation-corrected errors from notebook 06.
   - Same answers as the notebook's `Ising` class at L = 16.
   - Benchmark: spin updates per second. Use it to size steps 2–3.

   *About 45–60 min.*

2. **Critical slowing down: measure z.** At T_c, measure the integrated autocorrelation
   time τ_int of M² for L = 8, 16, 32, 64 (128 if the benchmark allows). Run each for
   ≳ 100 τ_int over many replicas. Fit τ_int ∝ L^z.
   *Checks:*
   - **T_c from Binder crossings.** U_L = 1 − ⟨M⁴⟩/(3⟨M²⟩²) curves for different L cross
     at T_c, at the universal value U* ≈ 0.611. Plotted against (T − T_c)L^(1/ν) with
     ν = 1, they collapse onto one curve.
   - Fitted z is within a few percent of 2.167 (Nightingale & Blöte 2000). Small-L
     corrections bias it, so report the fit with and without L = 8.
   - Away from T_c (T = 1.3 T_c), τ_int stops growing with L.

   *About 45–60 min.*

3. **Kibble–Zurek at the crossing.** Ramp through T_c for τ_Q from about 30 to 10⁴
   sweeps, at L = 512 with ~30 replicas per τ_Q. Record ξ₂(t), Δu(t) and S(k, t) along
   the ramp.
   *Checks:*
   - **Exponent.** At the crossing (t = 0), ξ₂ ∝ τ_Q^(ν/(1+zν)). Fit it and compare with
     0.316 and with the value from step 2's z.
   - **Full scaling collapse.** ξ₂(t)/ξ̂ against t/t̂ collapses onto one curve for all
     τ_Q, with ξ̂ ∝ τ_Q^0.316 and t̂ ∝ τ_Q^0.684. This is a much stronger test than one
     exponent.
   - **Where it fails, and why.** Fast quenches saturate at the initial ξ(T_i).
     Finite size bites once ξ̂ ≳ L/10. Rerun the largest τ_Q at L = 128 and show the
     deviation.
   - **Hubble cooling.** The radiation era has T ∝ t^(−1/2). Run T(t) = T_c(t_c/t)^(1/2).
     Kibble–Zurek says only the local slope at T_c matters, so it must match the linear
     ramp with τ_Q = 2t_c = 1/H(T_c).

   *About 60 min, mostly compute.*

### Coarsening (part 2)

4. **Coarsening after an instant quench.** Start from T = ∞ (random spins) and quench
   to T = 0.5 T_c at L = 512, then follow ℓ(t) out to ~10⁴ sweeps.
   *Checks:*
   - **Shrinking droplet.** A single circular domain of radius R₀ in a sea of the other
     sign disappears in a time ∝ R₀² (Allen–Cahn: curvature-driven, so the area falls
     linearly in time). Test R₀ = 20, 40, 80. This tests the mechanism behind the t^(1/2)
     law, not just the law itself.
   - ℓ(t) ∝ t^(1/2). ℓ from S(k) and 1/Δu must give the same exponent.
   - **Dynamic scaling.** S(k, t)/ℓ² against kℓ collapses for all late times.
   - **Porod's law.** The collapsed tail falls as (kℓ)^(−3), which is what sharp walls
     give in 2D, for 1/ℓ ≪ k ≪ 1/(wall width).
   - **Zero-temperature traps.** Quench to T = 0 at L = 64 over ~1000 replicas. About a
     third should freeze into stripe states instead of reaching the ground state
     (Spirin, Krapivsky & Redner 2001). Check the fraction against the paper. At T = 0,
     Metropolis flips ΔE = 0 moves with probability 1, so state that convention.

   *About 45–60 min.*

5. **Slow quench into the ordered phase: what survives?** Repeat step 3, but continue the
   ramp down to T_f = 0.5 T_c and hold there. Measure ℓ at the end against τ_Q.
   *Checks:*
   - Just after the crossing, ℓ follows step 3's KZ curve. Later, ℓ grows like step 4's
     t^(1/2). The measured final exponent sits between 0.316 and 0.5. Ramping takes a
     time ~τ_Q, so the coarsening exponent wins for slow quenches.
   - Build a two-regime model: KZ initial condition ξ̂, then Allen–Cahn growth from t̂.
     It should predict ℓ_final(τ_Q) with no new fit parameters beyond step 3's and step
     4's amplitudes. Compare it with Biroli, Cugliandolo & Sicilia (2010, PRE 81, 050101),
     which did this for 2D Ising; confirm their scaling form before the final figure.
   - **The cosmology lesson.** The number of walls you would infer from naive KZ is off,
     and the error grows with τ_Q. Quantify it.

   *About 45 min.*

6. **Relativistic walls in an expanding universe.** Solve a real scalar φ with
   V = (λ/4)(φ² − v²)² on a 2D comoving grid, in conformal time η with a ∝ η^p (p = 1 for
   radiation, 2 for matter). Use leapfrog. Use the Press–Ryden–Spergel trick (fixed
   comoving wall width), so walls stay resolved as the universe expands:

   $$\ddot\phi + \alpha\,\frac{p}{\eta}\,\dot\phi - \nabla^2\phi = -\frac{\partial V}{\partial\phi}.$$

   Confirm the correct α for 2D against Press, Ryden & Spergel (1989) before production
   runs. Initial conditions: a Gaussian random φ with correlation length ξ̂ from step 3.
   That ties Kibble–Zurek to the network that forms.
   *Checks:*
   - **Static kink.** φ = v tanh(x/δ), with δ = √(2/λ)/v, stays put. Its energy per
     length is σ = (2√2/3)√λ v³.
   - **Flat space** (α = 0): total energy is conserved to the integrator's order (halve dt,
     error drops 4×). A boosted kink moves at its boost speed with the right Lorentz
     contraction.
   - **Scaling.** Comoving wall length per area ∝ η^(−1) in both eras, so ℓ ∝ η. Contrast
     this with Ising's t^(1/2) on the same plot: same walls, different dynamics.
   - **The Zel'dovich problem.** In 3D, ρ_w = Aσ/t with A of order one, and
     ρ_rad = 3/(32πGt²). That gives a domination time t_dom ≈ 3/(32πGAσ). Tabulate t_dom
     and the domination temperature for σ^(1/3) = 1 MeV, 1 GeV and 1 TeV. Our A comes
     from 2D runs, so treat it as order-of-magnitude.

   *About 60–75 min.*

**Total:** about 5–6.5 hours with sign-offs. Steps 1–2 are shared. Step 3 is the
Kibble–Zurek result. Steps 4–5 are the coarsening result and need step 3. Step 6 needs
only step 3's ξ̂ and can be done after step 3 or last.

## Push Harder

- **Biased walls.** The standard way out of the domain-wall problem is a small vacuum
  energy difference. In Ising that is a field h, and in φ⁴ a tilt εφ. The false-vacuum
  domains then collapse. Measure the collapse time against h, in both models.
- **3D Ising.** ν ≈ 0.630 and z ≈ 2.02 give ξ̂ ∝ τ_Q^0.277. Here the walls are surfaces,
  as in cosmology. 128³ is feasible with the same engine.
- **XY model → cosmic strings.** With O(2) spins, the defects are vortices, which are the
  cross-sections of strings. In 2D the transition is BKT (ξ diverges exponentially), so KZ
  picks up logarithms (Jelić & Cugliandolo 2011). Count vortices by winding number.
- **Conserved dynamics.** Kawasaki spin exchange conserves magnetization, and its
  coarsening is Lifshitz–Slyozov, ℓ ∝ t^(1/3). This is the analog for conserved order
  parameters.
- **Inhomogeneous quench.** A cooling front moving at speed v_f suppresses defects when
  v_f < ξ̂/t̂, because causality lets one domain grow ahead of the front.
- **Gravitational waves.** Domain walls source a stochastic GW background, which is
  relevant to the pulsar-timing signal. Estimate the quadrupole power from step 6's
  network.

## Reading

Background reviews are in [Semester_Overview.md](Semester_Overview.md). Read
del Campo & Zurek (2014) before step 3.

**Step 1, engine:**
- Onsager, "Crystal statistics. I," Phys. Rev. **65**, 117 (1944), and Yang, Phys. Rev.
  **85**, 808 (1952). The exact u(T) and M(T). A textbook derivation is enough (Kardar
  or Huang); you only need the formulas.
- Newman & Barkema, ch. 3–4: Metropolis, equilibration and error estimates.

**Step 2, z:**
- Binder, "Finite size scaling analysis of Ising model block distribution functions,"
  Z. Phys. B **43**, 119 (1981). The cumulant crossing.
- Nightingale & Blöte, "Dynamic exponent of the two-dimensional Ising model and Monte
  Carlo computation of the subdominant eigenvalue of the stochastic matrix," PRL **76**,
  4548 (1996). [doi:10.1103/PhysRevLett.76.4548](https://doi.org/10.1103/PhysRevLett.76.4548)
- Nightingale & Blöte, "Monte Carlo computation of correlation times of independent
  relaxation modes at criticality," PRB **62**, 1089 (2000).
  [doi:10.1103/PhysRevB.62.1089](https://doi.org/10.1103/PhysRevB.62.1089). The
  reference value z = 2.1667.

**Step 3, Kibble–Zurek:**
- Kibble (1976) and Zurek (1985). See the overview.
- Laguna & Zurek, "Density of kinks after a quench: when symmetry breaks, how big are the
  pieces?" PRL **78**, 2519 (1997).
  [doi:10.1103/PhysRevLett.78.2519](https://doi.org/10.1103/PhysRevLett.78.2519). The
  first numerical test of the τ_Q scaling, in 1D φ⁴. Its method is the template for
  step 3.

**Step 4, coarsening:**
- Bray (1994), §1–4. See the overview.
- Allen & Cahn, Acta Metall. **27**, 1085 (1979). The curvature-driven wall law.
- Spirin, Krapivsky & Redner, "Fate of zero-temperature Ising ferromagnets," PRE **63**,
  036118 (2001). [doi:10.1103/PhysRevE.63.036118](https://doi.org/10.1103/PhysRevE.63.036118)
- Barros, Krapivsky & Redner, "Freezing into stripe states in two-dimensional
  ferromagnets and crossing probabilities in critical percolation," PRE **80**, 040101
  (2009). [doi:10.1103/PhysRevE.80.040101](https://doi.org/10.1103/PhysRevE.80.040101).
  The exact stripe fractions.

**Step 5, slow quench into the ordered phase:**
- Biroli, Cugliandolo & Sicilia, "Kibble–Zurek mechanism and infinitely slow annealing
  through critical points," PRE **81**, 050101 (2010).
  [doi:10.1103/PhysRevE.81.050101](https://doi.org/10.1103/PhysRevE.81.050101). This is
  exactly step 5's question for 2D Ising.

**Step 6, cosmological walls:**
- Press, Ryden & Spergel, "Dynamical evolution of domain walls in an expanding universe,"
  ApJ **347**, 590 (1989). [doi:10.1086/168151](https://doi.org/10.1086/168151). Open
  access. The algorithm, the α choice and the scaling result.
- Zel'dovich, Kobzarev & Okun, Sov. Phys. JETP **40**, 1 (1975). The domain-wall problem.
- Saikawa (2017), §2–3. See the overview.
