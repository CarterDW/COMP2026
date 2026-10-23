# Semester Project: Defects from Cosmological Phase Transitions

When the early universe cooled through a symmetry-breaking transition, it left
topological defects behind. This project asks how many form, what they do afterwards, and
when they matter. It uses lattice spin models as stand-ins for cosmological fields:
Ising for a real scalar (domain walls) and XY for a complex scalar (vortices, i.e.
cosmic strings in 3D).

## Phases

| Phase | Plan | Content | Time |
|---|---|---|---|
| 0 | [KT_Equilibrium_Plan.md](KT_Equilibrium_Plan.md) | Warmup: the XY engine, vortices, stiffness jump, T_KT | ~100 min |
| 1 | [Kibble_Zurek_Plan.md](Kibble_Zurek_Plan.md), steps 1–3 | Ising engine, dynamic exponent z, Kibble–Zurek exponent | ~2.5–3 h |
| 2 | Kibble_Zurek_Plan.md, steps 4–5 | Coarsening; slow quench into the ordered phase | ~1.5–2 h |
| 3 | Kibble_Zurek_Plan.md, step 6 | Relativistic φ⁴ walls; the Zel'dovich problem | ~1–1.25 h |
| 4 | *to be written* | XY Kibble–Zurek with log corrections; vortex coarsening at (t/ln t)^½ | ~3–4 h |
| 5 | *optional, to be written* | Complex φ in 3D: global or axion strings and their log scaling violation | open-ended |

Phases 0–3 make a complete project. Phases 4–5 extend it to a semester.

**Order of ideas:** equilibrium first (what the phases are). Then dynamics through the
transition on the model with clean power laws (Ising). Then the same dynamics where the
power laws become logarithms (XY). Each phase's equilibrium results are the baseline for
the next phase's out-of-equilibrium ones.

**Shared code:** one checkerboard engine with a swappable spin type (Ising or XY). Use
the same update rule for every dynamical measurement, because z and therefore every
Kibble–Zurek exponent depend on it.

## Reading by phase

Step-level reading is in each plan's **Reading** section. These are the background works
for the whole project.

**Textbooks (dip in as needed):**
- Newman & Barkema, *Monte Carlo Methods in Statistical Physics* (OUP 1999). Metropolis,
  autocorrelation times, critical slowing down, cluster algorithms.
- Kardar, *Statistical Physics of Fields* (CUP 2007). The renormalization group, the KT
  transition and critical exponents.
- Sethna, *Entropy, Order Parameters, and Complexity* (OUP, 2nd ed. 2021; free PDF from
  the author). The chapters on order parameters and defects, and on coarsening, are the
  most readable introduction.
- Vilenkin & Shellard, *Cosmic Strings and Other Topological Defects* (CUP 1994). The
  cosmology reference for walls and strings.

**Phase 1–2, Kibble–Zurek and coarsening:**
- Kibble, "Topology of cosmic domains and strings," J. Phys. A **9**, 1387 (1976).
  [doi:10.1088/0305-4470/9/8/029](https://doi.org/10.1088/0305-4470/9/8/029)
- Zurek, "Cosmological experiments in superfluid helium?" Nature **317**, 505 (1985).
  [doi:10.1038/317505a0](https://doi.org/10.1038/317505a0)
- Zurek, "Cosmological experiments in condensed matter systems," Phys. Rep. **276**, 177
  (1996). [doi:10.1016/S0370-1573(96)00009-9](https://doi.org/10.1016/S0370-1573(96)00009-9)
- del Campo & Zurek, "Universality of phase transition dynamics: topological defects from
  symmetry breaking," IJMPA **29**, 1430018 (2014).
  [doi:10.1142/S0217751X1430018X](https://doi.org/10.1142/S0217751X1430018X). **The best
  single starting review.**
- Bray, "Theory of phase-ordering kinetics," Adv. Phys. **43**, 357 (1994).
  [doi:10.1080/00018739400101505](https://doi.org/10.1080/00018739400101505). The
  coarsening bible.

**Phase 3, walls in cosmology:**
- Hindmarsh & Kibble, "Cosmic strings," Rep. Prog. Phys. **58**, 477 (1995).
  [doi:10.1088/0034-4885/58/5/001](https://doi.org/10.1088/0034-4885/58/5/001). Defect
  formation and network scaling in cosmology.
- Saikawa, "A review of gravitational waves from cosmic domain walls," Universe **3**, 40
  (2017). [doi:10.3390/universe3020040](https://doi.org/10.3390/universe3020040). Open
  access. Covers the domain-wall problem, scaling and biased walls.

**Phase 4–5, XY dynamics and strings:**
- Jelić & Cugliandolo, "Quench dynamics of the 2d XY model," J. Stat. Mech. (2011)
  P02032. [doi:10.1088/1742-5468/2011/02/P02032](https://doi.org/10.1088/1742-5468/2011/02/P02032)
- Yurke, Pargellis, Kovacs & Huse, "Coarsening dynamics of the XY model," PRE **47**, 1525
  (1993). [doi:10.1103/PhysRevE.47.1525](https://doi.org/10.1103/PhysRevE.47.1525)
- Gorghetto, Hardy & Villadoro, "Axions from strings: the attractive solution," JHEP
  **07** (2018) 151.
  [doi:10.1007/JHEP07(2018)151](https://doi.org/10.1007/JHEP07(2018)151). Open access.
  The log scaling violation of axion strings.
