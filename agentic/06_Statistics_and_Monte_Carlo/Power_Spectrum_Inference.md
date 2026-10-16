# Power Spectrum Covariance and Cosmological Inference

## The Problem

On large scales the matter density contrast $\delta(\mathbf{x}) = \rho/\bar\rho - 1$
starts out as a **Gaussian random field**. Its statistics are then set entirely by the
power spectrum,

$$\langle \delta_{\mathbf{k}}\, \delta^*_{\mathbf{k}'} \rangle = V\, P(k)\, \delta_{\mathbf{k}\mathbf{k}'} ,$$

and $P(k)$ depends on cosmology. $\sigma_8$ sets the amplitude. $\Omega_m$ sets where the
spectrum turns over, and baryons imprint the **BAO wiggles**, an echo of sound waves in
the early plasma. A survey measures $\hat P(k)$ in bins and then needs two things:

1. **A covariance matrix** $C$ for those bins.
2. **A posterior** for $(\Omega_m, \sigma_8)$.

For a Gaussian field the covariance is known exactly, $C = \mathrm{diag}(2P^2/N_k)$.
Gravity makes the late-time field non-Gaussian, though. Overdensities collapse,
underdensities can't fall below $\delta = -1$, and modes couple. The covariance then
picks up off-diagonal terms and a larger diagonal, and nobody can write it down. So
surveys estimate it from $N_s$ mock catalogs, and two problems follow:

- $\hat C$ is noisy, and its inverse is **biased**: with $p$ bins,
  $\langle \hat C^{-1} \rangle = \frac{N_s-1}{N_s-p-2}\, C^{-1}$ (Hartlap et al. 2007,
  https://arxiv.org/abs/astro-ph/0608064).
- If you skip the mocks and assume Gaussianity, your error bars are too small.

**The question:** how wrong are you if you assume the universe is Gaussian, and how many
mocks does it take to fix that?

The Gaussian field is the **control**, a case where the right answer is known. The
**lognormal** field, $\rho \propto e^{\delta_G}$, is the simplest non-Gaussian stand-in
for gravity. A Zel'dovich field, where particles are actually moved by the linear
displacement, is the stretch goal.

This is the Statistics manual day at survey scale. Box-Muller gives the field,
"is the error bar honest?" becomes "is the covariance honest?", and the 1D Metropolis
sampler becomes a 2D one over cosmological parameters.

## Conventions

- Lengths in Mpc/$h$, $k$ in $h$/Mpc. Periodic box with side $L = 1000$ and $N_g = 128$
  cells per side, so $V = L^3$, $N_c = N_g^3$, $k_f = 2\pi/L \approx 0.0063$ and
  $k_{\rm Ny} = \pi N_g/L \approx 0.40$.
- NumPy FFT, $\delta_{\mathbf{k}} = \texttt{fftn}(\delta)$. Estimator
  $\hat P(\mathbf{k}) = |\delta_{\mathbf{k}}|^2\, V / N_c^2$. The $\mathbf{k}=0$ mode is set to zero.
- Binning: about 25 linear bins from $k = 0.01$ to $0.4$, so $\Delta k \approx 2.5\,k_f$.
  That is fine enough to resolve the BAO (wiggle period $\approx 2\pi/r_s \approx 0.06$), and
  $p \approx 25$ makes the Hartlap effect large for $N_s \lesssim 100$.
- Gaussian generation: if $w$ is real white noise with unit variance, then
  $\langle|\texttt{fftn}(w)|^2\rangle = N_c$, so
  $\delta_{\mathbf{k}} = \texttt{fftn}(w)\sqrt{N_c P(k)/V}$ followed by $\delta = \texttt{ifftn}(\delta_{\mathbf{k}})$.real
  has power $P$. Starting from a real field makes the Hermitian symmetry automatic.
- Model: the Eisenstein & Hu (1998) transfer function with baryons
  (https://arxiv.org/abs/astro-ph/9709112), $P(k) \propto k^{n_s} T^2(k)$, normalized to
  $\sigma_8$ with a top-hat of radius $8$ Mpc/$h$. Fixed: $h = 0.67$, $\Omega_b = 0.049$,
  $n_s = 0.965$. Free: $\Omega_m$, $\sigma_8$, with fiducial $(0.31, 0.81)$. It's analytic,
  so a call is instant and MCMC is cheap.
- No galaxy bias. In linear theory $b$ and $\sigma_8$ are exactly degenerate.

## Natural Steps

Each step stops for sign-off before the next one starts.

1. **The model spectrum.** Write `P_model(k, Om, s8)` from EH98, plus its no-wiggle
   version (EH98 gives both).
   *Check:* the $\sigma_8$ computed by integrating your $P$ equals the input. The ratio
   $P/P_{\rm nw}$ oscillates with period $\Delta k \approx 2\pi/r_s$, where $r_s$ is EH98's
   sound horizon ($\approx 100$ Mpc/$h$). Optionally, `pip install camb` and confirm
   agreement to within a few percent over $0.01 < k < 0.4$.
2. **Gaussian field and the $\hat P(k)$ estimator.** Write `make_gaussian(P, seed)` and
   `measure_P(delta)`, which bins $|\delta_{\mathbf{k}}|^2$ and returns $N_k$, the number of
   grid vectors in each shell.
   *Check:* the average over 200 mocks matches the input $P$ in every bin, wiggles
   included. A real field has $\delta_{-\mathbf{k}} = \delta_{\mathbf{k}}^*$, so only $N_k/2$
   modes are independent and the scatter between mocks should be $\sigma_P = P\sqrt{2/N_k}$.
   The pulls $(\hat P - P)/\sigma_P$ should be roughly unit normal. This is manual-day
   Problem 4, carried out bin by bin.
3. **Gaussian covariance and Hartlap (the control).** Build $\hat C$ from $N_s$ Gaussian mocks
   and compare it with the analytic diagonal $2P^2/N_k$. The off-diagonal correlation
   coefficients should vanish within noise.
   *Check:* for $N_s \in \{30, 50, 100, 500\}$, compute the $\chi^2$ of an independent mock
   with $\hat C^{-1}$. Its mean should be $p\,(N_s-1)/(N_s-p-2)$, not $p$, and multiplying
   $\hat C^{-1}$ by the Hartlap factor $(N_s-p-2)/(N_s-1)$ should bring it back to $p$.
4. **Lognormal mocks.** You want $\delta_{\rm LN} = e^{\delta_G - \sigma_G^2/2} - 1$ to have the
   target $P$. Since $1 + \xi_{\rm LN} = e^{\xi_G}$, get $\xi_{\rm LN}$ from $P$ by FFT on the grid,
   set $\xi_G = \ln(1+\xi_{\rm LN})$, FFT back to $P_G$, and clip any small negative values.
   Then draw $\delta_G$ with the Gaussian generator from step 2.
   *Check:* $\hat P$ averaged over mocks matches the target $P$, and $\delta \ge -1$
   everywhere. The one-point PDF is skewed, unlike the Gaussian's. Build $\hat C$ from about
   1000 lognormal mocks: the correlation matrix $r_{ij} = C_{ij}/\sqrt{C_{ii}C_{jj}}$ now has
   clear off-diagonal structure, and $C_{ii}$ exceeds $2P^2/N_k$, increasingly so at high $k$.
   That excess is the physics the Gaussian assumption misses.
5. **Metropolis over $(\Omega_m, \sigma_8)$ on lognormal data.** Gaussian likelihood
   $-2\ln\mathcal{L} = (\hat P - P_\theta)^T C^{-1} (\hat P - P_\theta)$ with flat priors,
   using the manual-day sampler in 2D. Run it three ways: (a) the Gaussian $C$,
   (b) the mock $\hat C$ with small $N_s$ and no correction, (c) the mock $\hat C$ with the
   Hartlap correction.
   *Check:* Gelman-Rubin $\hat R < 1.01$ across 4 chains. Measure the integrated
   autocorrelation time $\tau$ and use $N_{\rm eff} = N/\tau$ in place of $N$ in the
   $\sigma/\sqrt N$ error bar. **Coverage** over about 100 data mocks: (c) puts the truth
   inside its 68% region about 68% of the time, while (a) and (b) undercover. That is the
   answer to the question, in one number each.
6. **Fisher forecast vs. MCMC.** $F_{ij} = \partial_i P^T C^{-1} \partial_j P$, using
   central-difference derivatives from the Numerical Differentiation week.
   *Check:* where the posterior is close to Gaussian, the Fisher $1\sigma$ ellipse matches
   the MCMC contour. The ratio of Fisher errors with the Gaussian $C$ to those with the
   lognormal $\hat C$ predicts the undercoverage you measured in step 5.

## Push Harder

- **BAO detection.** Fit lognormal data with $P$ and with $P_{\rm nw}$, take
  $\Delta\chi^2$, and calibrate its significance against null mocks generated from
  $P_{\rm nw}$ rather than trusting Wilks' theorem.
- **Zel'dovich mocks.** Put particles on a grid and move each one by the linear
  displacement $\boldsymbol{\Psi}_{\mathbf{k}} = i\mathbf{k}\,\delta_{\mathbf{k}}/k^2$, then paint them
  back with cloud-in-cell. Correct for the CIC window and subtract shot noise. The power
  spectrum now differs from linear theory, damped BAO included. That is real gravitational
  nonlinearity, and it's the next step past lognormal.
- **Dodelson-Schneider.** The Hartlap factor fixes the mean of $\hat C^{-1}$, but noise in
  $\hat C$ still inflates parameter errors by about $1 + (p - n_\theta)/(N_s - p)$
  (https://arxiv.org/abs/1304.2593). Measure this with coverage tests as you vary $N_s$.
- **Sellentin-Heavens.** Marginalize over the unknown $C$ instead of correcting for it. The
  likelihood becomes a multivariate $t$-distribution (https://arxiv.org/abs/1511.05969).
  Compare its coverage with Hartlap's.
- **Real mocks.** Swap in Quijote N-body snapshots (https://quijote-simulations.readthedocs.io)
  for your lognormal mocks.
