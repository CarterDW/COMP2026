# Literature Review: Mass Ejection in Cold Collapse, and Protostar Ignition

Compiled 2026-10-02 from two research passes. Numbers marked "ours" were measured with this code.

## 1. Why the cold collapse ejects ~20% of its mass

**It is physical, not a bug.** A cold, uniform, collisionless sphere ejects mass when it collapses.

Joyce, Marcos & Sylos Labini (2009) fit the ejected mass fraction as f ≈ 0.048 + 0.022 ln N. It grows slowly with N and converges to the collisionless limit for eps below the interparticle spacing. They also find R_min ∝ N^(-1/3).

| N | JMS09 fit | ours (cold uniform, eps = 0.01 R) |
|---|---|---|
| 250 | 16.9% | 17.8% |
| 500 | 18.5% | 16.7% |
| 1000 | 20.0% | 20.8% |
| 2000 | 21.5% | 22.2% |

**Mechanism** (JMS09; Sylos Labini 2012):
- In a uniform sphere every shell reaches the center at t_ff.
- Finite-N fluctuations, plus the sharp edge, make the outer shells lag.
- The core rebounds first, and the late outer shells fall through a rapidly changing potential and gain energy.
- The chance that a particle is ejected grows as (initial radius)².

**What reduces it in pure N-body**, at N = 1000 in our runs:

| Change | Literature | Ours |
|---|---|---|
| Warm start, Q0 = 2K/\|W\| = 0.25 | reduced | 5.9% |
| Warm start, Q0 = 0.5 | none above Q0 ≈ 0.5 (Sylos Labini 2012) | 0.0% |
| Profile rho ∝ r^-1 | intermediate | 10.1% |
| Profile rho ∝ r^-2 | ~1% for alpha ≥ 2 (Sylos Labini 2013) | 1.7% |
| Larger N | slightly *more* ejection | 17% → 22% |

**Why real clouds stay bound.**
- **Gas is dissipative.** Shocks turn infall energy into heat that is radiated away, so the collapse stays isothermal at ~10 K until rho ~ 1e-13 g/cm^3. Gas cannot bounce and re-expand the way collisionless particles do.
- **Real collapse is inside-out.** The center runs away first, inside a rho ∝ r^-2 envelope (Larson 1969; Shu 1977). That removes the "all shells arrive together" condition that drives ejection.
- **The standard simulation recipe** is SPH with an isothermal-then-stiff (barotropic) equation of state, artificial viscosity, and sink particles (Bate, Bonnell & Price 1995; Bate 1998; Bate, Bonnell & Bromm 2003).

## 2. From collapse to fusion (1 Msun)

1. **Isothermal collapse.** About 10 K, until rho ~ 1e-13 g/cm^3.
2. **First (Larson) core.** R ~ 7–30 AU and M ~ 0.04 Msun, lasting about 1000 yr (Vaytet et al. 2013). This is near our resolution limit.
3. **Second collapse.** H2 dissociates at about 2000 K. The second (stellar) core starts at R ~ 0.6 Rsun with M ~ 1e-3 Msun.
4. **Accretion phase.** Mdot ~ c_s^3/G ≈ 1.6e-6 Msun/yr, so building 1 Msun takes about 5e5 yr.
5. **Deuterium ignition.** At T_c ≈ 1.5e6 K. It acts as a thermostat and sets the birthline radius, about 5 Rsun (Stahler 1983, 1988; Palla & Stahler 1990).
6. **Pre-main-sequence contraction** (BHAC15 1 Msun track, Baraffe et al. 2015):
   - **Hayashi phase:** convective, Teff ≈ 4300 K, from 3.1 Rsun at 0.5 Myr to about 1.4 Rsun at about 5 Myr.
   - **Henyey phase:** radiative, Teff rising to about 5600 K.
7. **Hydrogen ignition.** T_c = 1e7 K at about 25 Myr (R ≈ 1.06 Rsun). The star reaches the ZAMS at 40–50 Myr. For comparison, the Kelvin–Helmholtz time t_KH = G M^2 / (R L) is 3.1e7 yr for the Sun.

### One-zone (sub-grid) model
The protostar is ~1e3 times smaller than our resolution, so it must be modeled, not resolved:
- **Central temperature.** For a polytrope of index n:
  - T_c = (mu m_H / k)(G M / R) / [(n+1) xi_1 |theta'(xi_1)|].
  - The structure factor is 0.539 for n = 1.5 and 0.854 for n = 3.
  - With mu = 0.61: T_c ≈ 7.6e6 K (M/Msun)(Rsun/R) for n = 1.5, and 1.20e7 K for n = 3.
- **Energy and contraction.**
  - E = -(3/(5-n)) G M^2 / (2R), and L = -dE/dt.
  - So dR/dt = -(2(5-n)/3) R^2 L / (G M^2).
- **Hayashi track.** L = 4 pi R^2 sigma Teff^4 with Teff fixed at about 4300 K.
  - Then dR/dt = -A R^4, which solves to R(t) = (R0^-3 + 3 A t)^(-1/3).
  - Contracting from 3.1 to 1.4 Rsun takes about 4.8 Myr; BHAC15 gives 4.6 Myr.
- **Fuller published model:** Offner, Klein, McKee & Krumholz (2009), Appendix B, calibrated to Palla & Stahler to about 10%.
- **Light sources.** L_acc = f G M Mdot / R, plus contraction luminosity, plus L_D, plus L_H.

## References
- Joyce, Marcos & Sylos Labini 2009, MNRAS 397, 775 — https://arxiv.org/abs/0811.2752
- Sylos Labini 2012, MNRAS 423, 1610 — https://arxiv.org/abs/1203.3027
- Sylos Labini 2013, MNRAS 429, 679 — https://arxiv.org/abs/1211.1278
- Benhaiem & Sylos Labini 2015, MNRAS 448, 2634 — https://arxiv.org/abs/1503.04962
- Boily, Athanassoula & Kroupa 2002, MNRAS 332, 971 — https://arxiv.org/abs/astro-ph/0204378
- Aarseth, Lin & Papaloizou 1988, ApJ 324, 288 — https://ui.adsabs.harvard.edu/abs/1988ApJ...324..288A
- van Albada 1982, MNRAS 201, 939 — https://ui.adsabs.harvard.edu/abs/1982MNRAS.201..939V
- Larson 1969, MNRAS 145, 271 — https://ui.adsabs.harvard.edu/abs/1969MNRAS.145..271L
- Masunaga & Inutsuka 2000, ApJ 531, 350 — https://ui.adsabs.harvard.edu/abs/2000ApJ...531..350M
- Shu 1977, ApJ 214, 488 — https://ui.adsabs.harvard.edu/abs/1977ApJ...214..488S
- Bate, Bonnell & Price 1995, MNRAS 277, 362 — https://arxiv.org/abs/astro-ph/9510149
- Bate 1998, ApJ 508, L95 — https://arxiv.org/abs/astro-ph/9810397
- Bate, Bonnell & Bromm 2003, MNRAS 339, 577 — https://arxiv.org/abs/astro-ph/0212380
- Vaytet et al. 2013, A&A 557, A90 — https://arxiv.org/abs/1307.1010
- Stahler 1983, ApJ 274, 822; Stahler 1988, ApJ 332, 804; Palla & Stahler 1990, ApJ 360, L47
- Offner, Klein, McKee & Krumholz 2009, ApJ 703, 131 — https://arxiv.org/abs/0904.2004
- Hosokawa & Omukai 2009, ApJ 691, 823 — https://arxiv.org/abs/0806.4122
- Baraffe, Homeier, Allard & Chabrier 2015, A&A 577, A42 — https://arxiv.org/abs/1503.04107

Unverified by the reviewers: the exact 1e-13 g/cm^3 threshold, the exact birthline radius, and the Shu 1977 content. These are standard values cited from memory. The ADS links for Aarseth, van Albada, Larson, Masunaga and Shu were built from bibcodes and not opened.
