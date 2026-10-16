# Measuring the Bispectrum: BAO and f_NL

## The Problem

The power spectrum only sees two-point statistics. Any non-Gaussianity lives in the
**bispectrum**,

$$\langle \delta_{\mathbf{k}_1}\delta_{\mathbf{k}_2}\delta_{\mathbf{k}_3}\rangle = V\,B(k_1,k_2,k_3)\,\delta^K_{\mathbf{k}_1+\mathbf{k}_2+\mathbf{k}_3,0},$$

which is only nonzero for closed triangles. Two physically different sources feed it:

1. **Gravity (secondary).** At tree level, B = 2F₂(k₁,k₂)P₁P₂ + 2 permutations. F₂ is
   the second-order kernel, and its BAO wiggles come in through P. This is what
   [bispectrum_bao.png](checks/plots/bispectrum_bao.png) computed from theory.
2. **Inflation (primordial).** Local f_NL means Φ = φ + f_NL(φ² − ⟨φ²⟩), which gives
   B = 2f_NL M₁M₂M₃[P_Φ(k₁)P_Φ(k₂) + perms]. Here M(k) converts Φ into δ. This shape peaks
   on squeezed triangles (one side much shorter than the other two).

The project measures B from mocks. That gives the "extracted" curves the theory figure
lacks. It then asks two statistics questions: how significantly is the BAO detected in B,
and how much fake f_NL does gravity produce if you ignore it?

The statistics machinery is the same as the power-spectrum project's: estimator, mock
covariance, Hartlap, coverage. One order up.

## Conventions

Same box, grid and cosmology as [Power_Spectrum_Inference.md](Power_Spectrum_Inference.md):
- L = 1000 Mpc/*h*, N_g = 128, EH98 or CAMB linear P.
- h = 0.67, Ω_b = 0.049, Ω_m = 0.31, σ₈ = 0.81, n_s = 0.965.
- Run everything in the `HaloPipeline` env.

**Estimator (FFT shell method).** Each shell i is a thin range of |k|. For each shell, inverse-FFT the
field restricted to that shell, D_i(x) = ifftn[F·1_i], with F = fftn(δ). Do the same
for the indicator alone, I_i(x) = ifftn[1_i]. Then

$$\hat B_{ijk} = \frac{V^2}{N_c^3}\,\frac{\sum_x D_i D_j D_k}{\sum_x I_i I_j I_k}.$$

The numerator sums F₁F₂F₃ over every closed triangle with one side in each shell. The
denominator counts those triangles, N_tri. The V²/N_c³ converts fftn units to (Mpc/*h*)⁶.

**Comparing theory with a measurement.** Theory must be averaged over the same triangles,
not evaluated at the bin centres. For separable terms like P(k₁)P(k₂), the same FFT trick
gives the exact average: replace F with P on shells i and j, and 1 on shell k. F₂ can be
written as a finite sum of separable pieces. If that turns out messy, use
effective-triangle sides and state the error that introduces.

**Triangle configurations.**
- Equilateral bins for the checks.
- The reference figure's configurations for the BAO plot: k₂ = k₁ + δπ/r_s, angle θ
  between k₁ and k₂. Confirm this definition against the paper before the final figure.
- Squeezed triangles for f_NL.

Shell width is about 2k_f.

## Natural Steps

Stop for sign-off after each step.

1. **The bispectrum estimator.**
   *Checks:*
   - The FFT triangle count N_tri equals a brute-force count over all mode triplets, on a
     small grid (N_g = 16).
   - Gaussian mocks give B̂ consistent with zero in every bin, judged by the scatter
     between mocks.
   - A quadratic field δ = g + b(g² − σ²) with Gaussian g has the exact leading-order
     bispectrum 2b[P_g(k₁)P_g(k₂) + perms], with corrections of order b³. Mock B̂, averaged
     over many mocks, must match the triangle-averaged prediction. Test it at two values
     of b, so the linear scaling is tested too.
   - The lognormal mocks are this case with b = ½ at leading order. Measure how far they
     depart from it at z = 0, where σ ≈ 1.3.

   *About 45–60 min.*

2. **Zel'dovich mocks.** Particles start on a 128³ lattice, positions **q**, and move to
   **x** = **q** + **Ψ**(**q**), where Ψ_k = i**k** δ_k / k² is computed from a Gaussian
   linear field. Cloud-in-cell assignment puts them back on the grid. Divide out the CIC
   window, W² for P and W₁W₂W₃ for B.
   *Checks:*
   - At low k, P_ZA → P_lin.
   - The BAO are damped by exp(−k²σ_v²), with σ_v² = (1/6π²)∫P dq the one-axis displacement
     variance. Compare against the measured P/P_nw.
   - **The tree-level bispectrum has the Zel'dovich kernel, not SPT's.** The Zel'dovich
     kernel is F₂^ZA = ½ + (μ/2)(k₁/k₂ + k₂/k₁) + ½μ², against SPT's 5/7 and 2/7. Mock B̂
     on large-scale equilateral triangles must prefer F₂^ZA, and the difference is
     measurable. This test can tell 5/7 from ½, so it really tests the dynamics.

   Restrict to k ≲ 0.25 *h*/Mpc. A 128³ lattice at one particle per cell has lattice and
   shot-noise effects at higher k (1/n̄ = 477 (Mpc/*h*)³, comparable to P at k = 0.4).
   *About 45–60 min.*

3. **Bispectrum BAO: extracted vs intrinsic.** Measure B̂ on the reference configurations
   in about 500 Zel'dovich mocks. Plot the mock mean of B̂_w/B̂_nw − 1 (dashed) over the
   tree-level ZA theory (solid). The no-wiggle comparison comes from mocks generated with
   P_nw from the **same seeds**, so the ratio cancels most cosmic variance (paired mocks,
   as in branch a).
   *Checks:*
   - Theory and mock mean agree within the mock error at large scales.
   - The damping of the extracted wiggles matches exp(−k²σ_v²) applied to the wiggle part.

   *About 30–45 min.*

4. **How significant is the BAO in B?** Build the covariance of B̂ from mocks; Hartlap
   applies, with p up to ~100 triangle bins. Fit templates with and without wiggles, take
   Δχ², and calibrate the significance against null mocks generated from P_nw, not
   against Wilks' theorem.
   *Checks:*
   - Null-mock Δχ² follows the distribution actually measured, and its tail sets the
     p-value.
   - The coverage of the BAO-amplitude parameter is about 68% with the mock covariance.

   *About 30–45 min.*

### f_NL branch

5. **Local-f_NL mocks.** φ is a Gaussian potential with P_Φ ∝ k^{n_s−4}. Set
   Φ = φ + f_NL(φ² − ⟨φ²⟩), then δ_k = M(k)Φ_k with M = √(P_lin/P_Φ). Watch the sign:
   Planck defines f_NL with Φ in the matter era (Φ = +(3/5)ζ), so state the convention.
   *Checks:*
   - Run ±f_NL with the same seeds. P is even in f_NL (it changes only at order f_NL²);
     B̂ is odd.
   - [B̂(+f_NL) − B̂(−f_NL)] / 2f_NL matches the analytic local template, and is largest on
     squeezed triangles.
   - Normalized skewness ⟨δ³⟩/σ³ stays constant across the branch-a redshifts, while the
     lognormal's falls as D(z). This is the "primordial survives" claim, tested directly.

   *About 30–45 min.*

6. **Measuring f_NL, and gravity's fake f_NL.** Fit B̂ = f_NL B_local + B_grav with a mock
   covariance.
   *Checks:*
   - A Fisher forecast of σ(f_NL) for this box matches the scatter of fitted f_NL over the
     mocks (step 6 of the P(k) project, again).
   - With f_NL = 0 Zel'dovich mocks, fitting *without* B_grav gives a biased f_NL. Measure
     that bias at z = 0 and as it falls toward z = 1100, and compare it with σ(f_NL).
   - Coverage of f_NL is 68% when B_grav is modelled, and fails when it is not.

   Caveat: this is the 3D matter field, not the CMB. Planck's −0.9 ± 5.1 comes from CMB
   temperature and polarization bispectra, so any comparison is order-of-magnitude.

   *About 30–45 min.*

**Total:** about 3.5–5 hours with sign-offs. Steps 1–2 are shared by both branches. After
step 2, the BAO branch (3–4) and the f_NL branch (5–6) are independent and can be done in
either order.

## Push Harder

- **2LPT mocks.** Second-order Lagrangian perturbation theory gives the exact SPT F₂ at
  tree level. Repeat step 2's kernel test and watch the preference flip from ½ to 5/7.
- **Joint P + B.** Measure the cross-covariance between P̂ and B̂ from the same mocks, and
  forecast how much the bispectrum adds to (Ω_m, σ₈) or to BAO precision. The
  non-Gaussian covariance from the P(k) project matters again here.
- **Scale-dependent bias.** Local f_NL makes halo bias scale as 1/k² on large scales
  (Dalal et al. 2008). That is a power-spectrum probe of f_NL, and it would need halos.
- **Quijote.** Swap in Quijote N-body snapshots and their f_NL suite.
