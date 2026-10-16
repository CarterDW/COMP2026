# Power Spectrum Covariance and Cosmological Inference: Results

Plan: [Power_Spectrum_Inference.md](Power_Spectrum_Inference.md). Every number below is
printed by a check script in [checks/](checks/) and was produced in the `HaloPipeline`
conda env (Python 3.14, NumPy 2.4).

## The question, and the answer

**If you assume the matter field is Gaussian when it is not, how wrong are your error
bars, and how many mock catalogs does it take to fix them?**

For a 1 (Gpc/*h*)³ box, measuring P(k) in 25 bins up to k = 0.4 *h*/Mpc and fitting
(Ω_m, σ_8) to a non-Gaussian (lognormal) universe:

| Covariance used in the likelihood | 68% region contains the truth | Worst error underestimate |
|---|---|---|
| A: Gaussian formula 2P²/N_k | **9.7%** | σ_8 error 3.2× too small |
| B: 50 mocks, inverse taken as is | 27.6% | about 1.9× |
| C: 50 mocks, Hartlap-corrected | 48.2% | about 1.3× (Dodelson–Schneider) |
| D: 2000 mocks, Hartlap-corrected | **68.4%** | honest |

Assuming Gaussianity makes the "68%" contour contain the truth 1 time in 10. Two effects
have to be fixed with mocks. The inverse of a noisy covariance is biased (Hartlap), and a
noisy covariance also weights the bins sub-optimally (Dodelson–Schneider). With 25 bins,
50 mocks is not enough and 2000 is. A Fisher-matrix "sandwich" calculation predicts all
of these coverages from the covariance alone, with no fits, to within 1.4σ (step 6).
So the failure is entirely a covariance effect.

Going back in time removes the problem (branch a). The same lognormal universe at z ≥ 10
has Gaussian-honest error bars, and at z = 1100 the skewness is 0.005. That is why CMB
analyses can use Gaussian likelihoods and galaxy surveys cannot.

## Setup

- Periodic box, L = 1000 Mpc/*h*, 128³ cells. 25 linear k-bins from 0.01 to 0.4 *h*/Mpc.
- Linear spectrum: Eisenstein & Hu (1998) with baryons. h = 0.67, Ω_b = 0.049,
  n_s = 0.965 fixed; Ω_m = 0.31 and σ_8 = 0.81 free. No galaxy bias.
- Mocks: Gaussian fields (white noise from the manual-day Box–Muller, shaped in Fourier
  space), and lognormal fields δ = exp(g − σ_G²/2) − 1 built to have the same P(k).
- Inference: Gaussian likelihood in P̂(k), flat priors, the manual-day Metropolis sampler
  in 2D, and an exact grid posterior for coverage over 1000 held-out mocks.

## Step by step

### 1. The model spectrum ([plot](checks/plots/step1_spectrum.png))
EH98's spectrum passes 8 checks:
- the top-hat window reproduces the white-noise variance of a sphere;
- the large-scale slope is n_s (0.9646);
- the sound horizon is 152.3 Mpc, +3.6% vs Planck;
- the BAO zero crossings are spaced by π/s to within 0.6%;
- the turnover moves with Ω_m as expected.

### 2. Gaussian fields and the P̂(k) estimator ([plot](checks/plots/step2_gaussian_field.png))
- Single Fourier modes are exponentially distributed: ⟨x²⟩ = 2.0037 against 2.
- The mean of 200 mocks matches the input P(k) (χ² = 32.7 for 25 bins).
- The scatter between mocks matches P√(2/N_k). The factor 2 is the k/−k pairing of a real
  field. Ignoring it understates the error bars by √2.
- The raw 7.8 Mpc/*h* cells have σ_δ = 1.3, and 22% of them have δ < −1 (negative density).
  That is the physical case for step 4.

### 3. Covariance from mocks and the Hartlap bias ([plot](checks/plots/step3_covariance.png))
Gaussian fields are the control case, where the covariance is known exactly.
- Bins are uncorrelated: √N r_ij has standard deviation 0.96, against 1.
- With Ĉ⁻¹ from N_s mocks, held-out χ² averages p(N_s−1)/(N_s−p−2) as Hartlap predicts:
  72.6, 53.2, 33.8 and 26.4 at N_s = 40, 50, 100 and 500, against p = 25.
- The Hartlap factor brings all four back to 25.

### 4. Lognormal mocks ([plot](checks/plots/step4_lognormal.png))
- Same P(k) as the Gaussian mocks (mean within 0.2%). δ ≥ −1 everywhere, and the one-point
  statistics match the exact lognormal formulas (skewness 5.952 vs 5.951).
- The covariance is not Gaussian. The variance per bin reaches **7.1×** 2P²/N_k at
  k = 0.4, and high-k bins are correlated at r ≈ 0.75.
- Held-out χ² averages 66.9 under the Gaussian C, against 24.65 under the mock Ĉ.

### 5. Metropolis fits and coverage ([plot](checks/plots/step5_mcmc.png), [fit figure](checks/plots/step5_fits.png))
- Eight chains from dispersed starts converge (R̂ ≤ 1.0015). MCMC agrees with the exact
  grid posterior (widths within 1.8%).
- The chain-mean error is σ/√(N/τ) with τ = 7.8. Using σ/√N understates it by 2.77 ≈ √τ.
  This is manual-day Problem 4 with correlated samples.
- Coverage is the table at the top. The fit figure shows one universe's data with
  P_fit and P_true, and the best fits of all 1000 universes against each likelihood's
  claimed 68% ellipse.

### 6. Fisher forecast vs MCMC ([plot](checks/plots/step6_fisher.png))
- Fisher ellipses match the exact posteriors: widths within 3.2%, correlations within 0.02.
- The sandwich covariance F⁻¹JᵀΨCΨJF⁻¹ predicts the best-fit scatter over 1000 universes
  to within 2%.
- It also predicts coverage. Predicted vs measured: A 10.3/9.7%, B 26.1/27.6%,
  C 46.7/48.2%, D 68.7/68.4%.
- Dodelson–Schneider: with a Hartlap-corrected 50-mock Ĉ, the best-fit variance is
  1.97 ± 0.08 and 1.92 ± 0.06 times optimal, against the predicted 2.05.

### Branch a: the same universe by epoch ([plot](checks/plots/branch_a_redshift.png))
- Linear growth D(z) includes radiation, and reproduces D = a, Mészáros and
  Carroll–Press–Turner.
- From z = 0 to z = 1100 the per-cell σ goes 1.3 → 0.0018, and the skewness goes 5.95 → 0.0054.
- The covariance excess falls as D² (z = 3 vs 10 agrees to 19%).
- Gaussian-C coverage: 9.7% → 28% → 55% → 66% → 68.5% → 68.2% at z = 0, 1, 3, 10, 100, 1100.

### CMB acoustic peaks and matter BAO by epoch, from CAMB ([plot](checks/plots/cmb_and_bao.png))
- CAMB's TT spectrum against Planck 2018: χ² = 83 for 83 bins at Planck's best-fit
  parameters. Peaks at ℓ = 221, 538, 818, 1134, 1434. The peak spacing, 298, equals
  π/θ* = 303 to within 2%.
- EH98 vs CAMB at z = 0: P(k) agrees within −2.3%/+3.9%. EH98's BAO spacing is compressed
  by 0.977, against the 0.973 predicted from its 2.7% sound-horizon error.
- The matter BAO at z = 1100 are smaller (3.5% vs 5.7%) and phase-shifted. They freeze
  into the late-time pattern by z ≈ 200.
- Nonlinear damping at z = 0 (Σ = 5.4 Mpc/*h* from IR resummation) leaves 32% of the
  wiggle at k = 0.2.

### BAO in the matter bispectrum ([plot](checks/plots/bispectrum_bao.png))
- Tree-level B = 2F₂P₁P₂ + perms, using CAMB's z = 0 P and P_nw. Configurations:
  k₂ = k₁ + δπ/r_s and angle θ between k₁ and k₂. This is my reading of a reference
  figure, not checked against its source.
- F₂ passes its exact limits, and its angle average is 17/21. B is symmetric under
  relabelling.
- Constructive configurations (δ = 0, 2) show ±10–12% wiggles. Destructive ones (δ = 1)
  are 6–10× weaker above k₁ = 0.05, with a residual ~5% feature at k₁ < 0.05 where the
  cancellation cannot be complete.
- Theory curves only. The reference's "extracted" curves would need a bispectrum
  estimator run on gravitationally evolved mocks.

## Caveats and corrections

Most of these are things that went wrong, or turned out differently than predicted,
during the work. Each is also recorded in the docstring of the check involved.

1. **The lognormal is a stand-in for gravity, not gravity.** Its covariance excess is
   large here because the cells are small (σ² = 1.64 per cell). Zel'dovich or N-body
   mocks would give the real numbers.
2. **The project cosmology is not Planck's.** With Ω_m = 0.31 and h = 0.67, ω_c = 0.1172
   against Planck's 0.1200. The CMB rejects it by Δχ² ≈ 140. Steps 1–6 never needed
   Planck's exact values.
3. **EH98's BAO scale is about 2.5% off.** Steps 1–6 are self-consistent, because the mocks
   and the model both use EH98. For real data, which reaches ~0.3% BAO precision with DESI,
   EH98 would bias the BAO scale.
4. **EH98 is a late-time fit.** Branch a's "today's shape × D²" is approximate at z = 1100.
   The CAMB panel shows the true early-time wiggles differ.
5. **Coverage used the exact grid posterior**, not 4000 MCMC runs. Step 5 shows the two
   agree on the showcase mock.
6. **Predictions that the data overturned.** The checks now assert what was measured:
   - I expected larger BAO wiggles at z = 1100. They are smaller.
   - I expected Planck χ²/bin ≈ 1 for the project cosmology. It is 2.7.
7. **Checks redesigned after a first failure:**
   - Step 1: white-noise window tolerance raised 1e−4 → 3e−4 (an undersampled
     oscillating tail, shown by a grid-convergence check).
   - Step 3: N_s = 30 dropped. The inverse-Wishart variance only exists for N_s > p + 4 = 29.
   - Step 6: derivative check now scale-normalized, because ∂μ/∂Ω_m crosses zero in one bin.
     It also tests h² convergence, measured ×11.4 vs ×11.1 predicted.
   - Branch a: the D² test is now paired against z = 1100, to cancel noise shared through
     common seeds.
   - CAMB check: wiggle spacing is now fitted, after a median-of-ratios measure was misread
     as falsifying the sound-horizon prediction.
   - Bispectrum check: the destructive-suppression range now starts at k₁ = 0.05, not 0.03.
     The low-k residual prevents cancellation there.
8. **Weakest passing numbers:**
   - Gaussian-C χ² in step 4 is 66.9 against a predicted 71.2 (−2.5σ).
   - Hartlap-corrected N_s = 40 in step 3 is 24.2 ± 0.4 (−1.9σ).
   - Neither is diagnosed beyond "heavy-tailed χ², optimistic error bar".

## Reproducing

Mock pools, the model grid and best fits are cached in `results/` (git-ignored). The
first full run takes about 25 minutes; later runs take seconds to minutes.

```bash
conda activate HaloPipeline
cd 06_Statistics_and_Monte_Carlo
python checks/step1_spectrum.py
python checks/step2_gaussian_field.py
python checks/step3_covariance.py
python checks/step4_lognormal.py
python checks/step5_mcmc.py
python checks/step6_fisher.py
python checks/fit_figure.py
python checks/branch_a_redshift.py
python -I checks/cmb_and_bao.py
python -I checks/bispectrum_bao.py
```

The CMB script needs `camb` (2.0.5) and Planck's `COM_PowerSpect_CMB-TT-{binned,full}_R3.01.txt`
from the Planck Legacy Archive in `results/planck/`.

## References

- Eisenstein & Hu 1998, ApJ 496, 605 ([astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112)): transfer function
- Hartlap, Simon & Schneider 2007, A&A 464, 399 ([astro-ph/0608064](https://arxiv.org/abs/astro-ph/0608064)): inverse-covariance bias
- Dodelson & Schneider 2013, PRD 88, 063537 ([1304.2593](https://arxiv.org/abs/1304.2593)): parameter-error inflation from Ĉ noise
- Carroll, Press & Turner 1992, ARA&A 30, 499: growth suppression g(Ω)
- Vlah et al. 2016, JCAP 03, 057: Gaussian-filter no-wiggle extraction
- Baldauf et al. 2015, PRD 92, 043514: IR resummation of the BAO
- Planck Collaboration 2020, A&A 641, A6: 2018 cosmological parameters
