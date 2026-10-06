# Literature Review: Turning Drifting Pebbles into Inner-Disk Planetesimals

Compiled 2026-10-06 from research passes over the primary texts (arXiv full texts read for every number quoted unless marked **unverified**). Numbers marked "ours" were computed from this code's disk model (`nbody/viscous_disk.py`, `nbody/gas_effects.py`, `rung5_planet_formation/initial_disk.py`: M0 = 0.137 Msun, R1 = 258 AU, t_nu = 11 Myr, M* = 0.816 Msun, alpha_acc = 1e-3, alpha_turb = 1e-4, St = 0.05) with a scratch script; they are estimates, not simulation output.

## 0. Our disk, in the quantities that matter

**A smooth stream of St = 0.05 pebbles in our disk sits 5 to 60 times below every published streaming-instability threshold. On its own it forms no planetesimals anywhere inside the snow line.**

| Quantity (ours) | 0.1 Myr | 0.3 Myr | 1 Myr | 2 Myr |
|---|---|---|---|---|
| Growth front r_g (AU) | 36 | 75 | 168 | 266 |
| LJ14 pebble flux F_peb (ME/Myr, ice+rock) | 645 | 377 | 169 | 60 |
| Gas accretion Mdot_g at 1 AU (Msun/yr) | 6.1e-9 | 5.9e-9 | 5.4e-9 | 3.2e-9 |
| xi = F_peb / Mdot_g | 0.32 | 0.19 | 0.093 | 0.056 |
| Z_peb = Sigma_p/Sigma_g at 1 AU (drift-limited, no back-reaction) | 3.5e-3 | 2.1e-3 | 1.0e-3 | 6e-4 |
| Midplane eps = rho_p/rho_g at 1 AU | 0.078 | 0.047 | 0.023 | 0.014 |
| Cumulative pebble mass passed since t = 0 (ME, ice+rock) | 100 | 196 | 367 | 470 |
| ...of which rock (x 7.1/30) | 24 | 46 | 87 | 111 |

- Disk at 1 AU and 1 Myr: Sigma_g = 657 g/cm^2, H/r = 0.037, Pi = eta v_K / c_s = 0.051, |dlnP/dlnr| = 11/4. The H/r is 0.047 at the 2.7 AU snow line.
- Z_peb uses Ida & Guillot (2016) Eq. 7 in the Lambda -> 1 limit: Z = xi (3 alpha_acc / 2) (1 + St^2) / (St |dlnP/dlnr|).
- eps uses the Youdin & Lithwick (2007) settling factor, sqrt(1 + St/alpha_turb) = 22.4.
- Thresholds at St = 0.05 and alpha = 1e-4 (Section 2):
  - eps_crit = 0.37 (Li & Youdin 2021), or 1.38 (Lim et al. 2024).
  - Z_crit = 0.017 (LY21 with turbulence), or 0.023 (Lim+24).
- **Implication:** a model that only drops the passing flux into the inner disk "where physics says" would deposit nothing unless one of the following acts:
  - St changes (sublimation, fragmentation),
  - back-reaction stalls the drift,
  - a pressure maximum traps the pebbles.

## 1. Mechanisms that turn drifting pebbles into inner-disk planetesimals

### 1a. Inner-disk pile-up from growth plus drift (Drążkowska, Alibert & Moore 2016, A&A 594, A105; arXiv:1607.05734)
**A pile-up arises naturally in a shallow (Sigma ~ r^-1), fragmentation-limited inner disk. It works only with dust back-reaction and sticky silicates (v_frag about 8 to 15 m/s).**

- **Setup.** 1+1D viscously heated disk (Bell & Lin opacity, no irradiation, alpha_t = 1e-3). Two-population dust model (Birnstiel et al. 2012). Dust inner edge at 0.3 AU.
- **Criterion.** Sum over St > 1e-2 of rho_d/rho_g > 1 (Eq. 15), with rho_d/rho_g = (Sigma_d/Sigma_g) sqrt((alpha_t + St)/alpha_t) (Eq. 14).
  - With external turbulence this is always stricter than the laminar Z_crit, so Z_crit was dropped.
- **Conversion.** dSigma_plts = zeta Sigma_d(St > 1e-2) dt / T_K (Eq. 16). Fiducial zeta = 1e-4 per orbit.
- **Why the pile-up happens:**
  - Inside the drift-dominated region, solids are fragmentation-limited: a_frag = f_f (2/3pi) Sigma_g v_frag^2 / (rho_s alpha_t c_s^2), with f_f = 0.37 (Eq. 11). This means St_frag = f_f v_frag^2 / (3 alpha_t c_s^2).
  - The headwind eta v_K falls with time in the inner disk, and back-reaction slows the drift as Z rises.
  - In an MMSN-like steep disk (Sigma ~ r^-3/2) no pile-up occurs.
  - **Without back-reaction, 99% of the solids are lost in 2 Myr and no planetesimals form.**

| Run (DAM16 Table 1) | M_disk/M* | Z | v_frag (m/s) | zeta | M_plts/M_solids | M_plts (ME) |
|---|---|---|---|---|---|---|
| fiducial | 0.1 | 0.01 | 10 | 1e-4 | 0.17 | 60 |
| low-mass disk | 0.01 | 0.01 | 10 | 1e-4 | 0.32 | 11.2 |
| lowest-mass disk | 0.005 | 0.01 | 10 | 1e-4 | 0.34 | 6 |
| heavy disk | 0.3 | 0.01 | 10 | 1e-4 | 0 | 0 |
| sub-solar Z | 0.1 | 0.005 | 10 | 1e-4 | 0 | 0 |
| Z = 0.02 | 0.1 | 0.02 | 10 | 1e-4 | 0.53 | 376.5 |
| v_frag = 8 | 0.1 | 0.01 | 8 | 1e-4 | 0 | 0 |
| v_frag = 16 | 0.1 | 0.01 | 16 | 1e-4 | 0 | 0 |
| zeta = 1e-2 / 1e-3 | 0.1 | 0.01 | 10 | 1e-2 / 1e-3 | 0.23 | 82 / 82 |
| zeta = 1e-5 / 1e-6 | 0.1 | 0.01 | 10 | 1e-5 / 1e-6 | 0.04 / 0.004 | 13 / 1.5 |

- **Annulus.** The inner edge lies at 0.3 to 1 AU and the width is 0.3 to 3 AU. Formation starts at about 3e5 yr and stalls at about 1 Myr. The annulus profile is much steeper than the disk's.
- **Applicability to us:**
  - We need a dust transport model with back-reaction, which we do not have.
  - v_frag for silicates is the key unknown parameter (laboratory values are about 1 m/s; DAM16 needs 8 or more).
  - Our disk is shallow (Sigma ~ r^-1), which favors this mechanism.

### 1b. The water snow line
**Drążkowska & Alibert (2017, A&A 608, A92; arXiv:1710.00009): a "traffic jam" forms because dry aggregates (v_frag = 1 m/s) are far smaller than icy ones (10 m/s). It is the dominant effect. Water recondensation adds only about 20%.**

- **Criterion.** rho_d(St > 1e-2)/rho_g > 1.
- **Conversion.** Sigma_dot = zeta Sigma_d Omega_K with zeta = 1e-3. **Note: this is per Omega^-1, i.e. 6.3e-3 per orbit, not per orbit as in DAM16.**
- Collective drift (back-reaction) is "a critical component ... without which obtaining the significant pile-up and planetesimal formation is nearly impossible."
- **Timing.** Planetesimals appear at 2e5 yr and form for about 2e5 yr.

| DA17 Table 1 (power-law disk) | Z | alpha_t | M_plts (ME) | annulus (AU) |
|---|---|---|---|---|
| | 0.01 | 1e-3 | none | — |
| | 0.02 | 1e-3 | 3.5 | 1.98–2.34 |
| | 0.03 | 1e-3 | 24.3 | 1.93–2.99 |
| | 0.03 | 3e-4 | 128.8 | 1.81–3.44 |
| | 0.03 | **1e-4** | **217.1** | 1.66–5.19 |
| | 0.03 | 3e-3 or 1e-2 | none | — |
| non-irradiated (cold) disk | 0.01 | 1e-3 | 119.8 | 1.03–2.20 |

- **Applicability.** This needs a Z enhancement of 2 or more in warm disks. Lower alpha_t helps a lot, and our 1e-4 is in the favorable range. **The planetesimals form at and outside the snow line (about 2 to 5 AU), not at 1 AU.**

**Schoonenberg & Ormel (2017, A&A 602, A21; arXiv:1702.02151): diffusion plus condensation enhances the ice surface density by a factor of 3 to 5 outside the snow line, and the "many-seeds" model adds about 2 more. Only mid-to-high alpha (1e-3 to 1e-2) helps.**
- The midplane solid-to-gas ratio reaches "tens of percent" for a solids-to-gas flux ratio F_s/g of about 0.1.
- Intermediate alpha (about 1e-3) is optimal, because the pebble flux scales as 1/alpha (Ida et al. 2016).
- At our alpha_turb = 1e-4 the diffusion mechanism is weak.

**Ida & Guillot (2016, A&A 596, L3; arXiv:1610.09643): an analytic flux criterion that uses only quantities we already compute. It is the most directly usable result for a 1D pipeline.**
- **Pebbles alone.** Eq. 7 gives Z = (1 + Λ²τ_s²) [2τ_s Λ² |dlnP/dlnr| / (3α) + Λ]^-1 xi, with Λ = ρ_g/(ρ_g + ρ_p).
  - For τ_s ≫ α and Λ ≈ 1 this reduces to Eq. 14: Z_peb ≈ (xi/a0) α (1 + τ_s²)/τ_s, with a0 = (2/3)|dlnP/dlnr| ≈ 1.75. (The extracted text of Eq. 14 is garbled; this form is our re-derivation from Eq. 7.)
  - Reaching Z_peb ≥ 0.02 by SI from tau_s ~ 0.1 pebbles needs xi ≥ xi_crit,SI = 3.5/alpha_3 (Eq. 15). That is unrealistic.
- **Inside the snow line.** Sublimating pebbles release small silicate grains (tau_s < 1e-5) that move with the gas. These grains keep the pebble scale height (beta_0 = (1 + tau_s,peb/alpha)^1/2).
  - Steady state (Eq. 17): Z ≈ zeta_0 xi_peb / (1 - beta zeta_0 xi_peb).
  - There is no steady solution, and so a runaway to gravitational instability of the dust layer, once
    **xi_peb > xi_crit = 1/(beta zeta_0) ≈ (1/zeta_0) (alpha/tau_s,peb)^1/2** (Eq. 18).
  - xi_crit does not depend on the pressure gradient. It is 0.3 for alpha = 1e-3, zeta_0 = 1/3 and tau_s,peb = 0.1.
  - For alpha = 1e-4 the supercritical phase lasts a long time (their Fig. 3).
- **Planetesimal size (Eq. 31).** R ~ 1e3 km (h_g/r / 0.01) alpha_3^1/2 (tau_s,peb/0.1)^-1/2.
- **Ours (estimate):**
  - zeta_0 = 7.1/30 = 0.237 and beta = sqrt(1 + 0.05/1e-4) = 22.4 give xi_crit ≈ 0.19.
  - Our xi is 0.32 at 0.1 Myr, 0.19 at 0.3 Myr and 0.09 at 1 Myr. So the IG16 criterion is met only before about 0.3 Myr, while about 46 ME of rock passes the snow line. It is not met at our 1 Myr start.
  - The result hinges on the dust keeping the pebble scale height. If the grains re-mix to the gas scale height (beta → 1), xi_crit = 1/zeta_0 ≈ 4.2 and the criterion is never met. This is a factor-20 uncertainty.
  - These planetesimals form just inside the snow line (about 2 to 2.7 AU for us), not at 1 AU.
  - Two-alpha note: in Eq. 17 alpha_acc drops out (the grains ride the accretion flow), and beta uses alpha_turb. This is our reading of the derivation, not stated by IG16, who use one alpha.

### 1c. The silicate sublimation line and ring models
**Morbidelli et al. (2022, Nature Astronomy 6, 72; arXiv:2112.15413): planetesimals form at the silicate line at about 1 AU, but only in an early (≤0.4 Myr), infall-fed, hot, radially expanding disk.**
- **Ingredients:**
  - Infall decaying as exp(-t/0.1 Myr) onto a centrifugal radius Rc = 0.35 AU/(M_sun(t))^0.5.
  - Viscous heating plus infall heating.
  - alpha falls from 1e-2 to alpha_min = 5e-4.
  - Schmidt number Sc = 10.
  - Condensation temperatures of 1400 K (refractories), 1000 K (silicates) and 170 K (ice).
  - Grain size caps of 10 cm (icy), 5 mm (silicate) and 1 mm (refractory).
  - At 1000 K half of the rock sublimates. The flow then converges because of outward gas motion plus the size change, aided by recondensation.
- **Criterion.** rho_d/rho_g > 0.5 (citing Gole et al. 2020). Conversion is 1e-4 of the solids per orbit, stopped when the ratio falls below 0.5. This makes it self-regulating, so the result is insensitive to the threshold.
- **Result.** **4.5 ME of silicate planetesimals in a ring at 0.75 to 0.9 AU, formed at 0.33 to 0.38 Myr.** 32 ME of icy planetesimals form at 3 to 5.5 AU during 0.1 to 0.5 Myr.
  - With alpha_min = 1e-4 the silicate-line mass exceeds 40 ME (super-Earth regime).
  - Without silicate sublimation, the rho_d/rho_g ≈ 1 condition is never met unless Sc = 100.
- **Applicability: poor for our disk.**
  - It requires T ≈ 1000 K near 1 AU, which needs Mdot ≳ 1e-7 Msun/yr plus infall.
  - Our LBP disk (t_nu = 11 Myr) has Mdot ≈ 6e-9 Msun/yr at all times after t = 0, and our passive T = 280 K (r/AU)^-1/2 gives 1000 K at 0.078 AU (1400 K at 0.04 AU).

**Izidoro et al. (2022, Nature Astronomy 6, 357; arXiv:2112.15558): imposed pressure bumps at the silicate line (about 1400 K), the water snow line and the CO line produce three planetesimal rings: about 0.7 to 1.5 AU, 3 to 8 AU and 20 to 45 AU.**
- **Bumps.** They are imposed: Gaussian or tanh rescalings of the gas Sigma. The inner bump comes from the MRI-viscosity transition above 1000 K (alpha_MRI = 3 alpha_nu).
- **Criterion (Eq. 8):**
  - Sigma_dot = epsilon Sigma_peb / T_K if rho_peb/rho_gas ≥ 1 and St ≥ St_min = 1e-3.
  - Otherwise Sigma_dot = (epsilon/d) |v_r| Sigma_peb, with d = 5 H_gas, if 2 pi r Sigma_peb |v_r| ≥ Mdot_crit (zonal-flow assistance, as in Lenz et al. 2019).
  - epsilon is about 1e-6 to 1e-4. The nominal case uses epsilon = 1.5e-6, Z0 = 1.5% and alpha_t ≈ 3.6e-5.
- **Nominal ring masses.** 2.5 ME (inner), 85 ME (central) and 18 ME (outer). Inner-disk formation ends at ≲0.8 Myr.
- **Initial temperature dependence.** The disk is "typically ~1000 K at ~1–1.3 au".
  - **"An initially colder disk where the gas temperature is 1000 K at ~0.1 au would lead to a planetary system unlike the Solar System."** This is our case.
- **Ring mass matters:**
  - An inner ring of more than a few ME (e.g. 20 ME) forms Earth masses in less than 0.5 Myr, which then migrate inward (super-Earth outcome).
  - The inner ring grows by planetesimal accretion. Pebble accretion is negligible because the snow-line bump cuts off the flux.

**Charnoz et al. (2019, A&A 627, A50)** and **Charnoz et al. (2021, A&A; arXiv:2105.00456)** (2019 read from the published PDF):
- Dead-zone disks build traps from viscosity gradients near the snow line. For Z = Z_sun, a ring about 1 AU wide forms around the snow line.
- Planetesimals reach down to 0.1 AU only "if hot silicate dust is as sticky as ice" (v_frag about 10 m/s).
- The 2021 paper makes a planet-free pressure maximum just inside the snow line if the vertically averaged alpha decreases with Sigma. It needs a stratified dead-zone alpha(Sigma) that we do not model.

### 1d. Pressure bumps generally
- Location is set by whatever makes the bump: imposed (Izidoro et al. 2022), a dead-zone edge (Charnoz et al. 2019/2021), or zonal flows (Lenz et al. 2019).
- **In our pipeline the one physical bump is the gap edge of the isolated giant (about 6 AU).** It traps pebbles outside the giant, not in the inner disk.

### 1e. "Traps everywhere" parametrization (Lenz, Klahr & Birnstiel 2019, ApJ 874, 36; arXiv:1902.07089)
**The planetesimal formation rate is proportional to the local pebble flux: Sigma_dot_p = (epsilon/d) Mdot_peb / (2 pi r) (Eq. 1), with conversion length l = d/epsilon.**
- **Parameters:**
  - d = 5 h_g is the trap spacing (after Dittrich et al. 2013).
  - Particles count only if St_min = 1e-2 ≤ St ≤ St_max = 10.
  - Trap lifetime tau_l = 100 orbits.
  - Critical flux Mdot_cr = m_p / (epsilon tau_l), with m_p = a 100-km-diameter planetesimal = 1.05e-7 ME (Eq. 4 to 6).
  - epsilon = 0.1 is fiducial; 0.8 was also tested.
- alpha_t = 1e-2 prevents inner-disk planetesimals; alpha_t = 1e-3 forms them "quickly at all places". The final Sigma_p is steeper than the gas.
- **Ours.** With epsilon = 0.1, the captured fraction per e-fold in r at 1 AU is epsilon r/(5H) = 0.54. In practice it captures most of the flux wherever St ≥ 0.01.
- **Not physics-located.** Location follows wherever the flux is; epsilon, d and tau_l are free.

## 2. Streaming-instability thresholds usable in 1D

**With turbulence the threshold is best stated as a critical midplane ratio eps_crit(St) applied to a turbulent pebble layer, or as Lim et al.'s direct Z_crit(St, alpha) fit.**

| Source | Form | Value at St = 0.05 (and our alpha = 1e-4, Pi = 0.05) |
|---|---|---|
| Carrera, Johansen & Davies 2015 (A&A 579, A43; arXiv:1501.05314) | laminar map; for St ≳ 0.1 log Zc = 0.3(log St)^2 + 0.59 log St − 1.57 (as restated by Yang+17 Eq. 8) | "Z ≳ 2% for stopping times ~5e-2" (quoted by Schoonenberg & Ormel 2017) |
| Yang, Johansen & Carrera 2017 (A&A 606, A80; arXiv:1611.07014) | St < 0.1: log Zc = 0.10(log St)^2 + 0.20 log St − 1.76 (Eq. 9); Zc = 0.035 at St = 1e-3 | Zc = 0.014 (laminar) |
| Li & Youdin 2021 (ApJ 919, 107; arXiv:2105.06042) | pure SI: log(Zc/Pi) = A(log St)^2 + B log St + C, with (A, B, C) = (0.1, 0.32, −0.24) for St < 0.015 and (0.13, 0.1, −1.07) for St > 0.015. Zc scales ∝ Pi. | Zc = 0.0052 (laminar) |
| Li & Youdin 2021, with turbulence | log eps_crit = A′(log St)^2 + B′ log St + C′, with (0, 0, log 2.5) for St < 0.015 and (0.48, 0.87, −0.11) for St > 0.015. Z_crit,α = eps_crit sqrt((Pi/5)^2 + α/(α+St)) (Eq. 13–14). | eps_crit = 0.37, **Z_crit,α = 0.017** |
| Lim et al. 2024 (ApJ; arXiv:2312.12508) — 3D, self-gravity, forced turbulence | log eps_crit = 0.41(log St)^2 + 0.71 log St + 0.37, the same for all α_D. log Z_crit = 0.15(log α_D)^2 − 0.24 log St log α_D − 1.48 log St + 1.18 log α_D (Eq. 19), valid for 0.01 ≤ St ≤ 0.1 and α_D ≥ 1e-3 St. | eps_crit = 1.38, **Z_crit = 0.023** |
| Lim et al. 2024 headline | St = 0.01: Z_crit ≳ 0.06, 0.1 and 0.2 for α_D = 1e-4, 10^-3.5 and 1e-3 (laminar ~0.02). Turbulence raises Z_crit by "up to an order of magnitude". | — |
| DAM16, DA17, SO17, Izidoro+22 | rho_d/rho_g > 1 (for St > 1e-2 or > 1e-3) | — |
| Morbidelli+22 | rho_d/rho_g > 0.5 (Gole et al. 2020: ~0.5 at St = 0.3, α = 10^-3.5; **secondary**, via LY21 and Lim+24) | — |

- **Pebble scale height.**
  - Youdin & Lithwick (2007, Icarus 192, 588; arXiv:0707.2975) Eq. 24 and 28: H_p/H_g ≈ sqrt(α_z/(α_z + St)) ξ^-1/2, with ξ ≈ (1 + 2St)/(1 + St) for τ_e = 1. That correction is a 2% effect at St = 0.05.
  - LY21 add the SI self-stirring in quadrature: H_p = sqrt(H_p,η^2 + H_p,α^2), with H_p,η ≈ 0.2 ηr, i.e. H_p,η/H ≈ Pi/5.
  - **Lim et al. 2024 find this Gaussian estimate inconsistent with their runs.** Their measured Z_crit is systematically *lower* than eps_crit · H_p/H would predict. This is why they give the direct Z_crit fit.
- **Sharp transition.** LY21 find a sharp drop in Z_crit between St = 0.01 and 0.02 (0.016 to 0.007). Lim+24 find it largely disappears at α_D = 1e-4 (Z_crit is 0.04 to 0.065 at St = 0.01 and 0.04 to 0.05 at St = 0.02).
- **Conversion rates in use:**

| Source | Definition | Value |
|---|---|---|
| DAM16 | Sigma_dot = zeta Sigma_d(St>1e-2) / T_K | 1e-4 fiducial. Mass saturates at zeta ≥ 1e-3 (82 ME at 1e-3 and 1e-2); 1e-2 is the numerical-stability limit. |
| Simon et al. 2016 (as quoted by DAM16; **secondary**) | ~50% of pebbles converted in a few tens of orbits, St = 0.3 | ≈ 1e-2 per orbit |
| DA17 | Sigma_dot = zeta Sigma_d Omega_K | 1e-3 per Omega^-1 (6.3e-3 per orbit) |
| Morbidelli+22 | fraction per orbit while rho_d/rho_g > 0.5 | 1e-4 |
| Izidoro+22 | Sigma_dot = epsilon Sigma_peb / T_K | 1e-6 to 1e-4 (nominal 1.5e-6) |
| Lenz+19 | Sigma_dot = (epsilon/d) Mdot_peb / (2 pi r), d = 5 h_g | epsilon = 0.1 (0.8 tested) |
| "Schäfer et al." conversion rate | — | **unverified: not located in this pass** |

## 3. Predicted planetesimal distributions and terrestrial outcomes

**Every pebble-based mechanism that puts rock at about 1 AU does so as a narrow, steep annulus of a few ME. Ring initial conditions reproduce Earth and Venus well, Mars about 10 to 50% of the time, and generally fail to reach the observed S_c.**

| Study | Initial solids | Outcome |
|---|---|---|
| Hansen 2009 (ApJ 703, 1131; arXiv:0908.0743) | 2 ME in 400 equal bodies of 0.005 ME, uniform over 0.7–1.0 AU, Jupiter at 5.2 AU (e = 0.05), 1 Gyr | Mars analogues (a > 1.3, M < 0.2 ME) in 21/23 runs (13/23 with M > 0.02); Mercury analogues in 3/23. 38 runs: mean N = 3.4, median S_d = 0.0020, median S_c = 86 (range 55–132). **Solar system: N = 4, S_d = 0.0018, S_s = 37.7, S_c = 90.** |
| Izidoro+22 | 2.5 ME, 0.7–1.5 AU, 30% in 3000 planetesimals and 70% in 20 embryos of 0.5–1.5 M_Mars, several profiles (r^0 to r^-5.5), 200 Myr, 80 runs | 17 runs gave good analogues with 3 to 4 planets. Analogue windows: Venus 0.5–0.9 AU and 0.4–1.2 ME; Earth 0.7–1.25 AU and 0.7–1.4 ME; Mars 1.25–1.8 AU and 0.03–0.3 ME. A few Mercury-mass planets at 0.4–0.5 AU, but no good Mercury. |
| Woo et al. 2023 (Icarus; arXiv:2302.14100) | GPU N-body of a planetesimal ring at 1 AU with gas, 10 Myr | In an MMSN-like gas disk the ring **spreads radially** and loses its concentration. A concave gas Sigma peaking at 1 AU, short-lived (≤1 Myr), keeps it. |
| Woo, Nesvorný, Scora & Morbidelli 2024 (Icarus 417, 116109) | Gaussian ring with mu = 1 AU, sigma = 0.1 AU, 2.1 ME, about 200 Myr, giant-planet instability at 15/60/100 Myr | About 50% form a Venus–Earth pair; about 10% form a Mars analogue (up to 50% for the "shallower inner" disk). Median S_c is about 55–60, vs 89.9 observed. Normalized AMD medians are about 1 to 5 (too high, and higher for late instabilities). Total planet mass is about 1.9 to 2.0 ME. |
| Lykawka & Ito 2019 (ApJ 883, 130; arXiv:1908.04934) | 540 runs over the common literature disks | 194 systems with ≥3 analogues, only 17 with all four. Grand-Tack-like truncated disks give Mercury and Mars too cold, too close and too massive. All four need mass concentrated in narrow cores (0.7–0.9 and 1.0–1.2 AU), an inner component from 0.3–0.4 AU, a light component beyond 1.0–1.2 AU, embryo-dominated mass and eccentric Jupiter and Saturn. |
| Morbidelli+22 | — | Ring of 4.5 ME at 0.75–0.9 AU |
| DAM16 | — | Annulus with inner edge 0.3–1 AU, 1 to more than 1000 ME |

- **Definitions (Chambers 2001, as written by Woo+24 Eq. 3–4):**
  - S_c = max_a [Σ m_k / Σ m_k (log10(a/a_k))^2].
  - Normalized AMD = [Σ m_k sqrt(a_k)(1 − sqrt(1−e_k^2) cos i_k) / Σ m_k sqrt(a_k)] / 0.0018.
- **Our current runs** put the largest planets at 1.7–2.8 AU with only 0.1–0.4 ME inside 1.5 AU. This is the classic "smooth disk" failure that every reference above is trying to fix.

## 4. Time dependence: what reaches the inner disk before the giant isolates

**About 110 ME of pebbles entering the inner disk marks the terrestrial-to-super-Earth transition. Our flux passes that much rock by about 2 Myr, and that much ice-plus-rock by 0.1 Myr.**

- **Lambrechts et al. 2019 (A&A 627, A83; arXiv:1902.08694):**
  - Setup: F_peb = F_peb,0 exp(−t/1 Myr), 3 Myr gas phase, St = 3e-3, H/r = 0.04, 25 Moon-mass embryos over 0.5–3 AU, no pile-up or planetesimal formation.

| Suite | F_peb,0 (ME/Myr) | Integrated flux (ME) | Outcome |
|---|---|---|---|
| runf1 | 40 | 38 | terrestrial |
| runf3 | 120 | 114 | terrestrial; embryos ≲5 Mars masses, then giant impacts → planets ≲4–5 ME |
| runf5 | 200 | 190 | super-Earths (5–20 ME, inside about 0.1 AU) |
| runf9 | 360 | 340 | super-Earths |

  - "When ... the total mass in pebbles entering the inner disc is less than ≈ 110 ME", the system is terrestrial. Above ≈ 190 ME it forms super-Earths.
  - The threshold depends on the filtering efficiency. Higher efficiency (factor ~4, Ormel & Liu 2018) lowers it.
  - **Their flux is pebble mass in a model with no ice sublimation. Mapping it onto our ice-plus-rock flux is ambiguous by the factor 7.1/30.**
- **Kruijer et al. 2017 (PNAS 114, 6712)** (from the abstract and secondary summaries; **full text not read**):
  - NC and CC reservoirs stayed separate from about 1 Myr to 3–4 Myr.
  - Jupiter's core reached about 20 ME in under 1 Myr, then grew to about 50 ME by 3–4 Myr or later.
- **Morbidelli et al. 2016 (Icarus; arXiv:1511.06556), abstract:** Jupiter reached about 20 ME while the snow line was near 3 AU, fossilizing it. A silicate line "fossilized" at about 0.7 AU would explain Mercury's small mass and the empty region inside it.
- **Ours.** Rock delivered across the snow line by the LJ14 flux (cumulative from t = 0):
  - about 24 ME by 0.1 Myr,
  - 46 ME by 0.3 Myr,
  - 87 ME by 1 Myr,
  - 102 ME by 1.5 Myr,
  - 111 ME by 2 Myr.
  - Isolation at 1 to 2 Myr therefore lets about 90 to 110 ME of rock (370 to 470 ME of ice plus rock) through.
  - **Nearly all of this passes before our embryo stage starts (T_START = 1 Myr).**
- **Common thread.** Inner-disk planetesimal formation in DAM16, DA17, Morbidelli+22 and Izidoro+22 happens in the first 0.2 to 0.8 Myr. A deposition model has to run from early times and hand Sigma_plts(r) to the N-body stage at 1 Myr.

## 5. Competing explanations for the mass deficit outside 1 AU and small Mars

| Explanation | Mechanism | Status |
|---|---|---|
| Ring or annulus initial conditions (Hansen 2009; DAM16; Morbidelli+22; Izidoro+22; Woo+23/24) | Planetesimals form only in a narrow zone near 1 AU | Reproduces Earth and Venus. Mars about 10% (Woo+24). S_c is too low and AMD too high in long runs. |
| Grand Tack (Walsh et al. 2011, Nature 475, 206; arXiv:1201.5177) | Jupiter migrates in to 1.5 AU, then out (Saturn in 2:3 resonance), truncating the disk at 1 AU; terrestrial planets form over 30–50 Myr | Correct Earth/Mars mass ratio and a repopulated belt. Criticisms: Earth and Mars isotopes too similar (Woo et al. 2018, cited by Woo+23; **secondary**); truncated disks give poor Mercury and Mars (Lykawka & Ito 2019). |
| Early giant-planet instability (Clement et al. 2018, Icarus 311, 340; arXiv:1804.04233) | A Nice-model instability 1–10 Myr after gas dispersal strips embryos from the Mars region (800 runs, disks of 3 or 5 ME) | Mars criterion (a, m) met in up to 13% (instability at 1 Myr) vs **0% for the control**. AMD < 0.0036 in 7–16%. |
| Empty or low-mass primordial asteroid belt and depleted Mars region (Izidoro et al. 2014/2015; Raymond & Izidoro 2017; Nesvorný et al. 2021) | Little mass ever formed beyond about 1 AU | **Unverified in this pass.** Cited by Izidoro+22, Woo+24 and DAM16 as the "depleted Mars region" family. |
| Pebble-accretion-driven terrestrial growth (Levison et al. 2015; Johansen et al. 2021) | Mars-region embryos starve of pebbles | **Unverified in this pass.** |

## 6. Recommended model for our pipeline

**Mechanism.** Build a 1D pebble transport model for the inner disk that includes the processes our disk physics already implies:
- radial drift with back-reaction,
- water-ice sublimation at the snow line, followed by a change in grain size,
- fragmentation-limited Stokes numbers.

Convert pebbles to planetesimals wherever a published SI threshold is exceeded. Location and efficiency then come from Sigma_g, T, H, alpha_turb, alpha_acc and the LJ14 flux, not from a chosen radius. The silicate line enters only through T; with our T it lies inside 0.1 AU and does nothing. The model should run from t ≈ 0 (or from the start of the pebble flux) to T_START = 1 Myr and keep running afterwards as a source. Its Sigma_plts(r, t) replaces Z · Sigma_gas inside 4 AU, or adds to it.

**Equations (cgs; r in AU where noted).**
1. **Gas (existing).** Sigma_g(r, t), c_s, H = c_s/Omega, Omega, v_K, Pi = eta v_K / c_s = (H/r)|dlnP/dlnr|/2 with |dlnP/dlnr| = 11/4, and u_g = −3 nu_acc/(2r) with nu_acc = alpha_acc c_s H.
2. **Pebble Stokes number.** St = min(St_LJ14 = 0.05, St_frag, St_drift):
   - St_frag = f_f v_frag^2 / (3 alpha_turb c_s^2), with f_f = 0.37 (Birnstiel et al. 2012, as in DAM16 Eq. 11).
   - St_drift = f_d (Sigma_p/Sigma_g)(v_K/c_s)^2 / |dlnP/dlnr|, with f_d = 0.55 (DAM16 Eq. 10).
   - v_frag = 10 m/s for icy grains (T < 170 K) and 1 m/s for dry silicates (DA17).
   - Ours: at 1 AU, St_frag(1 m/s) ≈ 1.2e-3 and St_frag(10 m/s) ≈ 0.12.
3. **Drift with back-reaction (Nakagawa et al. 1986, IG16 Eq. 2).** v_r = −2 St Λ^2 eta v_K / (1 + Λ^2 St^2) + Λ u_g / (1 + Λ^2 St^2).
   - Λ = 1/(1 + eps) and eps = rho_p/rho_g.
   - eta v_K = (1/2)(H/r)^2 |dlnP/dlnr| v_K.
4. **Transport.**
   - ∂Sigma_p/∂t + (1/r)∂/∂r[r(Sigma_p v_r − D Sigma_g ∂(Sigma_p/Sigma_g)/∂r)] = −Sigma_dot_plts.
   - D = alpha_turb c_s H / (1 + St^2).
   - Outer boundary flux = the LJ14 Mdot_F(t) minus what the giant-stage embryos capture, set to 0 once the giant's core reaches M_iso.
   - At T > 170 K, keep the rock fraction 7.1/30 as solids with the dry v_frag, and send the ice to vapor (advected with u_g and diffused). Recondensation is optional (DA17: about 20% effect).
5. **Midplane ratio (Youdin & Lithwick 2007 / LY21).** eps = (Sigma_p/Sigma_g) / sqrt((Pi/5)^2 + alpha_turb/(alpha_turb + St)).
6. **Threshold.** Form planetesimals where either condition holds:
   - Z = Sigma_p/Sigma_g > Z_crit(St, alpha_turb) from Lim et al. 2024 Eq. 19, for 0.01 ≤ St ≤ 0.1;
   - otherwise eps > eps_crit(St) from LY21 Eq. 11 (2.5 for St < 0.015).
   - Equivalently, report both, since they disagree by about 1.3x at our parameters (0.023 vs 0.017).
7. **Conversion.** Sigma_dot_plts = zeta Sigma_p / T_K while the threshold holds. Mass goes into 100-km-diameter planetesimals (m_p = 1.05e-7 ME; Lenz+19) or into our tracer mass.
8. **Analytic cross-check at the snow line (IG16 Eq. 18).** A runaway occurs if xi = Mdot_peb/Mdot_g > 1/(zeta_0 sqrt(1 + St/alpha_turb)) = 0.19 for us.

**Inputs from our existing model.**
- Sigma_g(r, t) (LBP times the dispersal factor), T(r), c_s, H, Omega, alpha_acc, alpha_turb.
- The snow line (170 K), Z_rock and Z_ice.
- The LJ14 flux Mdot_F(t) and the giant's capture and isolation time.

**Free parameters and published ranges.**

| Parameter | Range | Note |
|---|---|---|
| v_frag (silicate) | 1 m/s (DA17, lab bouncing) to 10 m/s (DAM16; Charnoz+19 "sticky") | **The biggest lever.** DAM16 needs 8–15 m/s. |
| v_frag (ice) | 10 m/s | DA17; IG16 cite 20–100 m/s |
| zeta | 1e-4 (DAM16, Morb+22) to 1e-2 per orbit (Simon+16 via DAM16) | Results saturate for zeta ≥ 1e-3 (DAM16). Run 1e-4, 1e-3 and 1e-2 and quote the spread. |
| St_min | 1e-2 (DAM16, DA17, Lenz) or 1e-3 (Izidoro+22) | — |
| Threshold fit | LY21 vs Lim+24 | Factor about 1.3 to 4 apart depending on St |
| Schmidt number | 1 (our D) to 10 (Morb+22) | — |

**Published results to reproduce as checks.**
1. LY21 and Lim+24 thresholds at their own (St, alpha) points, e.g. Lim+24 Z_crit(0.01, 1e-4) ≈ 0.05 from the fit (0.04 to 0.065 measured).
2. IG16 xi_crit = 0.3 for alpha = 1e-3, zeta_0 = 1/3, tau_s = 0.1.
3. DAM16 fiducial: 0.1 Msun, Z = 0.01, alpha_t = 1e-3, v_frag = 10 m/s, zeta = 1e-4 gives 17% of solids = 60 ME, inner edge 0.3–1 AU, start at about 3e5 yr; and no planetesimals without back-reaction.
4. DA17: Z = 0.03 gives 24 ME at 1.93–2.99 AU (alpha_t = 1e-3) and 217 ME at 1.66–5.19 AU (alpha_t = 1e-4).
5. Downstream: Hansen 2009 statistics (N ≈ 3.4, S_d ≈ 0.002, S_c ≈ 86; solar 4, 0.0018, 90), Woo+24 (S_c ≈ 55–60, AMD ≈ 1–5x, Mars ≈ 10%) and Lambrechts+19 flux regimes (≲110 ME terrestrial, ≳190 ME super-Earth).

**Honest caveats.**
- **Our smooth disk probably forms few or no inner-disk planetesimals from pebbles** (Section 0: eps at 1 AU is 0.01 to 0.08 vs eps_crit 0.4 to 1.4). The IG16 snow-line runaway is the only route clearly open, and only for t ≲ 0.3 Myr. It would put rock at about 2 to 2.7 AU, near where our planets already grow too large, not at 1 AU. If the full model confirms this, "physics says" the inner deficit stays. The Solar System answer would then lie in physics we do not yet model (an early hot or infall phase, bumps or dead zones), or in the Section 5 dynamical scenarios.
- **The silicate line is misplaced.** Passive T = 280 K r^-1/2 puts 1000 K at 0.078 AU and 1400 K at 0.04 AU, inside our 0.7 AU inner edge. Ring models need about 1000 K at 1 to 1.3 AU (Izidoro+22) and say a disk with 1000 K at about 0.1 AU gives a non-Solar-System outcome.
- **Viscous heating does not rescue it at our accretion rate.** My constant-opacity estimate is T_visc^4 = (27/128) kappa Sigma_g (Mdot/3pi) Omega^2 / sigma_SB, added in quadrature-of-fourth-powers to T_irr. With Mdot = 5.4e-9 Msun/yr and kappa = 1–5 cm^2/g, 1000 K falls at about 0.2–0.35 AU. At Mdot about 1e-7 Msun/yr it would reach about 0.7–1.5 AU, consistent with Lambrechts+19 quoting about 0.5 AU at 1e-7 (from Morbidelli+16).
  - Our LBP disk never had such an Mdot (t_nu = 11 Myr; Mdot ≈ 6e-9 since t = 0) because it has no infall phase.
  - If accretion is wind-driven (the premise of our two-alpha disk), midplane viscous heating would be smaller still (**unverified**).
  - Viscous heating would also move the snow line outward of 2.7 AU (T ≈ 240–270 K at 1.5 AU in the same estimate).
- **Fixed St = 0.05 in the pipeline** is inconsistent with fragmentation limits inside the snow line (St about 1e-3 for 1 m/s). The pebble accretion rates in rung 5 would change too.
- **Timing.** The planetesimal-forming epoch in all of these models (0.1 to 0.8 Myr) precedes our T_START = 1 Myr. Most of our pebble flux (about 370 ME by 1 Myr) passes before embryos exist.
- **Threshold fits** are local single-size shearing-box results. Lim+24 Eq. 19 must not be extrapolated outside 0.01 ≤ St ≤ 0.1. LY21 and Lim+24 disagree on whether the Gaussian H_p estimate works.
- **zeta conventions differ** (per orbit vs per Omega^-1, by a factor of 2pi). Convert before comparing.
- **Lambrechts+19 thresholds** come from a model with no sublimation, St = 3e-3 and no planetesimal formation. They are a guide, not a calibration.
- **Not verified from a primary text in this pass:** Kruijer+17 (full text), Simon+16 and Gole+20 values (taken via DAM16, LY21 and Lim+24), Woo+18 isotopes, Izidoro+14/15, Raymond & Izidoro 2017, Nesvorný+21, Levison+15, Johansen+21, "Schäfer et al.", and the Charnoz+21 numbers beyond its abstract.
