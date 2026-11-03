# Intrinsic Alignments Without Gradients: a Noisy Mock and a TATT Profile

## The Problem

Galaxy shapes are not random. Tidal fields line them up with the matter around them, and
in weak lensing that **intrinsic alignment** (IA) is the largest astrophysical
systematic. Two ways to model it are used, and this project fits both to the same
measurement:

- **A forward model.** Put galaxies in simulated halos with an HOD and give them
  orientations relative to their halos (halotools `CentralAlignment`,
  `RadialSatelliteAlignment`). Every evaluation repopulates the catalog, so the model
  is *stochastic*. Call it twice with the same θ and you get two different χ². No
  gradient exists, and finite differences mostly measure the noise.
- **Perturbation theory.** TATT (Blazek et al. 2019) writes the IA field as
  γ_I = C₁ s_ij + C₂ (s s)_ij + b_TA C₁ (δ s_ij) + …, and the one-loop terms come
  from FAST-PT. This model is smooth, but its parameters come in products (b₁C₁,
  b_TA·C₁), which makes the likelihood non-Gaussian and degenerate. Marginal posteriors
  for A₂ and b_TA are known to shift because of *prior volume*, not because of the
  data. A **profile likelihood** has no prior volume, but you have to minimize over
  everything else at each point, many times over, on a surface with ridges.

**The questions:**
- **Part A:** can a gradient-free optimizer recover HOD + IA parameters from a noisy
  forward model? How close can it get to the floor set by the population noise?
- **Part B:** how much do TATT marginal posteriors differ from profile likelihoods?
  Does annealing find the profile more reliably than a gradient method?

The notebook-08 ideas carry over directly. Problem 2's integer floor becomes the noise
floor of a stochastic χ². Problem 4's `step·√T` proposal becomes the continuous move in
both parts. Problem 3's population collapse is what you should watch for if you use a
GA in step 4.

## Steps at a Glance

| Step | Content | Time | Plot / result to show |
|---|---|---|---|
| 1 | Mock data vector and noise floor | ~40 min | w_gg, w_g+ with error bands for several μ_c; χ² histogram vs χ²_p |
| 2 | Why gradients fail | ~30 min | χ² slices with noise scatter; finite-difference scatter vs h; Nelder–Mead endpoints |
| 3 | Annealing a noisy χ² | ~50 min | χ² vs iteration with re-scored best-ever; endpoints vs θ* |
| 4 | CMA-ES / GA and the f_sat–μ_s ridge | ~45 min | (α, μ_s) valley, w_g+ only vs w_g+ + w_gg |
| 5 | PT model for w_gg and w_g+ | ~50 min | TATT term breakdown of w_g+; model vs mock at large r_p |
| 6 | Profile likelihoods on synthetic data | ~60 min | profile vs marginal for C₂ and b_TA; SA vs L-BFGS-B jagged points |
| 7 | Coverage | ~45 min | Δχ²(truth) histogram vs χ²₁; profile vs marginal coverage |
| 8 | TATT fit to the halotools mock | ~50 min | Ĉ₁ vs μ_c; χ²_min/dof vs r_min |

**First-hour milestone:** step 1 plus the χ² slices from step 2. At that point you have
measured IA signals, their response to μ_c, and a picture of the noise that rules out
gradients. Each later step produces at least one figure on its own.

## Conventions

- **Halos:** BolPlanck z = 0 Rockstar (cached; L_box = 250 Mpc/h, 1.44M halos).
  Cosmology Ω_m = 0.307, h = 0.678, σ₈ = 0.823. Use units of Mpc/h throughout.
- **Galaxies:** a Zheng07 HOD with one luminosity threshold (pick one so that
  n_g ~ 10⁻³ (Mpc/h)⁻³). Centrals use `CentralAlignment(μ_c)` and satellites use
  `RadialSatelliteAlignment(μ_s)`. The major axis is the shape axis.
- **Data vector:** w_gg(r_p) and w_g+(r_p) in 15 log bins each (p = 30), 0.5 < r_p < 25 Mpc/h,
  with Π_max = 30 Mpc/h. Use halotools `wp` (which computes w_gg for galaxy
  positions) and `gi_plus_projected`.
- **Covariance:** from repopulations of the *same* halo catalog. This contains only
  population noise, with no cosmic variance. That is the right covariance for Part A
  (the "data" uses the same halos), and the wrong one for real data. Say so in the
  write-up. Apply Hartlap's correction.
- **Truth θ*:** fiducial HOD plus μ_c = 0.8, μ_s = 0.5. Fix it at the start and don't
  look at it while fitting.
- **PT stack:** FAST-PT **4.1.1 from PyPI**, not the old local checkout in `~/FAST-PT`.
  CCL is built from your PR branch (LSSTDESC/CCL#1298,
  `CarterDW/CCL@IA_galaxybias_terms`), whose `pgi` and `pii` are complete to one loop,
  including the bias × IA terms and the `ck`/`ct` prefactors. Both are installed in
  HaloPipeline (2026-10-30), and the smoke test of `get_biased_pk2d` for g×IA and IA×IA
  passed. The linear P(k) is CCL's, at BolPlanck cosmology. Use `cosmo.py` from week 5
  only as a cross-check.
- No `try/except`, no silent defaults. Seed every repopulation explicitly.
- Run everything in the `HaloPipeline` env. If CMA-ES is needed, pip-install `cma`.

## Natural Steps

Stop for sign-off after each step.

### Part A — the noisy forward model

1. **Data vector and the noise floor.** Populate at θ* 100 times. With p = 30 the
   Hartlap factor is (N−1)/(N−p−2) = 1.46, which is acceptable; 50 would give 2.7, which
   is not. If timing allows, start the run to 300 in the background for later steps. Measure both statistics each time and build the mean and covariance.
   Time one evaluation, since that number sets the budget for every later step.
   *Plots:* w_gg and w_g+ with 1σ bands at μ_c = 0, 0.4, 0.8. Histogram of held-out χ²
   with the χ²_p curve on top.
   *Checks:*
   - **Zero alignment is zero.** With μ_c = μ_s = 0, w_g+ is consistent with zero in
     every bin (χ² against zero ≈ p).
   - **Alignment is monotone.** |w_g+| increases with μ_c at fixed μ_s, and w_gg does
     not depend on μ at all. w_gg measures positions only, so any shift there is a bug.
   - **The floor.** Score held-out repopulations at θ* with the Hartlap-corrected
     covariance. Their χ² must follow a χ²_p distribution (KS test). That distribution
     is the floor: no optimizer can land below it except by luck.
   - An independent periodic KD-tree pair count and halotools agree on w_gg (done: 8×10⁻⁴; TreeCorr can't combine a Π cut with a periodic box).

   *About 40 min.*

2. **Why gradients fail here.** Plot 1D slices of χ²(θ) through θ* for μ_c and log M_min,
   with several repopulations at each point. Compute a finite-difference derivative
   ∂χ²/∂μ_c for h from 10⁻³ to 10⁻¹.
   *Checks:*
   - The spread of χ² at fixed θ is about √(2p), as a χ²_p variable predicts.
   - The finite-difference derivative's scatter grows like 1/h as h shrinks. Below some
     h it carries no information. Report that h.
   - Run `scipy.optimize.minimize(method='Nelder-Mead')` from three starts. Show where it
     stops and how far from θ* that is. Expect it to stop early, at points that won a
     noise lottery.

   *About 30 min.*

3. **Annealing a noisy χ².** Simulated annealing in θ = (log M_min, σ_logM, α, log M₁,
   μ_c, μ_s), with Gaussian moves of size step·√T in each coordinate (notebook Problem 4).
   *Checks:*
   - **The best-ever trap.** Notebook 08 kept the best-ever state. With a noisy
     objective, that state is biased: its χ² sits below the floor because it is the
     luckiest draw, not the best θ. Show this by re-scoring the best-ever θ with 50 fresh
     repopulations. The re-scored mean should rise into the χ²_p band.
   - Fix it either by returning the final low-T state or by averaging several
     repopulations per evaluation near the end. Compare the two.
   - From five starts, the endpoints scatter around θ* by roughly the parameter error
     bars, and none is systematically offset.

   *About 50 min.*

4. **Population methods and the degeneracy.** Run CMA-ES (`cma`) or a GA on the same
   objective. Population methods average noise over many members, so they should
   tolerate it better.
   *Checks:*
   - Same θ̂ as step 3, within the endpoint scatter.
   - **The f_sat–μ_s ridge.** A 2D scan of (α, μ_s) shows a valley: more satellites
     with weaker alignment can mimic fewer satellites with stronger alignment in w_g+.
     w_gg is what breaks the degeneracy. Fit w_g+ alone and then both, and show that the
     valley narrows.
   - If you use a GA, track population diversity (as in Problem 3's check). It should
     not collapse before the noise floor is reached.

   *About 45 min.*

### Part B — TATT profiles

5. **The PT model for w_gg and w_g+.** Use the PR-branch CCL `EulerianPTCalculator`.
   P_gg comes from `PTNumberCountsTracer(b1, b2, bs)`, and P_gI from that tracer crossed
   with `PTIntrinsicAlignmentTracer(c1, c2, cdelta)`. P_gI is one-loop complete, so it
   includes your bias × IA terms. Sample in TATT's (C₁, C₂, b_TA) and set
   cdelta = b_TA·C₁, as DES does. That product is what step 6 is about. Project with
   Hankel transforms:
   w_gg ∝ ∫ k dk P_gg J₀(k r_p) and w_g+ ∝ −∫ k dk P_gI J₂(k r_p) (Π_max → ∞).
   Build the calculator once at fixed cosmology. After that, each bias/IA evaluation
   only recombines precomputed terms, so it is fast.
   *Plot:* w_g+ split into its terms (NLA, C₂, b_TA, bias × IA), with the mock points
   on top.
   *Checks:*
   - **CCL agrees with FAST-PT directly.** At one k-grid, compare CCL's P_gI with the
     same combination built from `fastpt.FASTPT` (`IA_tt`, `IA_ta`, `IA_mix`, `gI_tt`,
     `gI_ta`, `gI_ct`), to the precision of CCL's interpolation. This confirms the
     sign and factor-of-2 conventions you'll be fitting in.
   - **Hankel test.** Transform a Gaussian P(k) whose J₀ and J₂ integrals are known in
     closed form, and match them to 10⁻⁴.
   - **NLA limit.** With C₂ = b_TA = 0 and one-loop bias off, w_g+ equals b₁C₁ times the
     linear-theory J₂ transform exactly.
   - **Large scales match the mock.** Above about 10 Mpc/h, fitting b₁ to the step-1 w_gg
     gives the same b₁ as the ratio √(w_gg / w_mm,lin).
   - Estimate the Π_max = 30 correction by integrating ξ to finite Π once, and report it
     as a percentage per bin.

   *About 50 min.*

6. **Profile likelihoods by annealing, on synthetic data.** Make the data from the PT
   model itself at known (b₁, b₂, C₁, C₂, b_TA) and add Gaussian noise using the step-1
   covariance. Profile C₂ and b_TA: at each grid value, minimize χ² over all other
   parameters with SA. Also run Metropolis on the same likelihood
   (`06/inference.py: metropolis`) with wide flat priors and marginalize.
   *Checks:*
   - **The linear control.** Freeze b₁ and b_TA, so the model is linear in what
     remains. The profile and marginal must then agree exactly, and the Δχ² = 1 interval
     equals the Fisher ±σ. This is the "known answer" for the whole step.
   - **The profile is smooth.** A jagged profile, with points sitting above their
     neighbours' interpolation, means a minimization failed. Count the jagged points for
     SA and for L-BFGS-B started from the global best fit. The comparison between them
     is the result of this step.
   - With everything free, the marginal for b_TA shifts relative to the profile. The
     product b_TA·C₁ explains the direction of the shift: when C₁ is small, b_TA is
     unconstrained, which inflates prior volume there.

   *About 60 min.*

7. **Coverage.** Use 200 noise realizations of the synthetic data. For each, compute
   Δχ²(truth) = χ²_min with the parameter fixed at the truth minus the global χ²_min.
   That takes only two minimizations per realization.
   *Checks:*
   - Δχ²(truth) follows χ²₁ (Wilks), so the profile's 68% interval covers the truth 68%
     of the time. Report any departure; it is the sign of a boundary or a non-Gaussian
     ridge.
   - The marginal credible interval's coverage, from shorter chains, is reported next to
     it. The difference between the two coverages is the projection effect, in one
     number.

   *About 45 min (mostly compute).*

8. **Fitting TATT to the halotools mock.** Fit the step-1 measurement with the PT model
   for r_p > r_min. Profile C₁ and C₂. Vary r_min from 15 down to 3 Mpc/h.
   *Checks:*
   - There is no ground-truth TATT value here, but the scaling can be predicted. Repeat
     the fit at μ_c = 0.4 and 0.8 at fixed μ_s. The ratio Ĉ₁(0.8)/Ĉ₁(0.4) should match
     the measured large-r_p w_g+ ratio from step 1, which is about 4, not 2: the Watson
     alignment is nonlinear in μ. With μ = 0, Ĉ₁ is consistent with zero.
   - χ²_min/dof stays acceptable down to some r_min and then rises. That r_min is where
     one-loop TATT stops describing this mock. Report it, together with whether C₂ is
     needed to push it lower.

   *About 50 min.*

**Total:** about 6 hours, in two arcs. Part A (steps 1–4, about 2.75 h) stands alone.
Part B steps 6–7 use only the covariance from step 1, so they can run before Part A is
finished.

## Push Harder

- **Cosmic variance.** Repeat the step-1 covariance using jackknife subvolumes instead of
  repopulations, and show how much the error bars grow. Fitting Part A with the
  population-only covariance underestimates errors for real data, and this measures by
  how much.
- **Mapping halo models onto TATT.** Trace Ĉ₁(μ_c, μ_s) over a grid. Is the mapping
  linear? Does the satellite term show up mostly in C₂ or in b_TA?
- **Bayesian optimization** for Part A: a Gaussian-process surrogate with a noise
  kernel, compared against SA in number of repopulations to reach the floor.
- **Profile with cosmology free.** Let σ₈ vary. FAST-PT then reruns at every evaluation,
  and the argument for gradient-free minimizers is closer to that of Boltzmann-code
  analyses.
- **IA in projection.** Add w_++ (`ii_plus_projected`) to the data vector and see whether
  it breaks the b_TA·C₁ degeneracy.

## Reading

*These references are from memory and still need DOI checks before they're relied on.*

**Before step 1:**
- Zheng et al., "Galaxy evolution from halo occupation distribution modeling of DEEP2 and
  SDSS galaxy clustering," ApJ 667, 760 (2007). The HOD form used by `Zheng07`.
- The halotools IA model paper by van Alfen et al. (2024), for what `CentralAlignment` and
  `RadialSatelliteAlignment` mean physically.

**Before step 3:**
- Kirkpatrick, Gelatt & Vecchi, "Optimization by simulated annealing," Science 220, 671
  (1983).
- Hansen, "The CMA Evolution Strategy: A Tutorial," arXiv:1604.00772 (before step 4).

**Before step 5:**
- Blazek, MacCrann, Troxel & Fang, "Beyond linear galaxy alignments," PRD 100, 103506
  (2019). TATT.
- McEwen, Fang, Hirata & Blazek, "FAST-PT," JCAP (2016), and Fang et al., "FAST-PT II"
  (2017), which covers the IA terms.
- Hirata & Seljak, PRD 70, 063526 (2004). The linear alignment model and w_g+.

**Before step 6:**
- Herold, Ferreira & Komatsu, "New constraint on early dark energy from Planck and BOSS
  data using the profile likelihood," ApJL 929, L16 (2022). Profile vs marginal in
  practice.
- Hartlap et al., A&A 464, 399 (2007), astro-ph/0608064. The covariance correction,
  again.
