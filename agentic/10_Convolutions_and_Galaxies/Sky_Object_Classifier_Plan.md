# A Sky-Object Classifier: Galaxies, Stars, Artifacts, and the Unknown

## The Problem

Notebook 10's CNN sorts spirals from ellipticals at about 95%, but it has only ever seen
galaxies. A real survey image also contains stars, quasars, cosmic rays, satellite trails
and diffraction spikes, along with things nobody has a class for. A classifier with K
outputs has to assign *everything* to one of the K classes. Show it a satellite trail and
it will call it "spiral, 98%."

This project builds a three-level classifier:

1. **Known or unknown?** This is out-of-distribution (OOD) detection.
2. **Star or galaxy?** A star is a point source, so its image is the PSF. A galaxy is
   extended.
3. **If it's a galaxy:** spiral or elliptical, and red or blue.

Levels 2 and 3 are trained on real Galaxy Zoo galaxies plus **simulated** stars and
artifacts injected into real sky backgrounds. Real SDSS stars and quasars come in at the
end (step 6), and that step needs an outside data pull.

**The questions:**
- How much of the spiral/elliptical accuracy is color, and how much is shape? Can shape
  alone predict color? This tests the red-sequence/blue-cloud split.
- Down to what brightness can a CNN separate stars from galaxies, compared with a simple
  size cut?
- Which OOD scores catch objects the network never trained on, and which don't?

Connections to notebook 10: Problem 2's two CNNs become the trunk of the network.
Notebook 10's remark that color sorts a lot of the galaxies becomes step 4's controlled
experiment. Its "uncertain" galaxies, which the notebook drops, become a built-in OOD test
set.

## Steps at a Glance

| Step | Content | Time | Plot / result to show |
|---|---|---|---|
| 0 | Environment: install torch (CUDA) and scikit-learn | ~10 min | `torch.cuda.is_available()` |
| 1 | Data, a color proxy, and notebook 10's CNN | ~40 min | Color histogram by class (bimodal); test-accuracy curve |
| 2a | Simple stars and artifacts on real backgrounds | ~45 min | Gallery: real galaxy / simulated star / cosmic ray / trail / spike |
| 2b | Optics from first principles: pupil FFT, Kolmogorov seeing, blackbody colors | ~60 min | Pupil → PSF with spikes → SDSS view; speckle → long exposure; star-color gallery |
| 3 | Star vs. galaxy | ~45 min | Accuracy vs. brightness: CNN vs. size cut; real-vs-simulated leak test |
| 4 | The galaxy head: morphology × color | ~50 min | Accuracy table (color / gray / shuffled); gallery of red spirals and blue ellipticals |
| 5 | Known or unknown? OOD detection | ~45 min | Score histograms and ROC curves for held-out classes |
| 6 | Real SDSS stars and quasars *(needs approval for the data pull)* | ~50 min | Classifier trained on simulated stars, tested on real ones; quasars gray vs. color |

**First-hour milestone:** steps 0–1 plus the step-2a gallery. That gives the color
bimodality of real galaxies, a reproduced 95% CNN, and a picture of every simulated class
next to real galaxies.

## Conventions

- **Data:** `notebooks/GalaxyZoo.zip`, 5,000 color JPGs at 120×120 with SDSS object IDs.
  The labels are one-hot (spiral, elliptical, uncertain). Use notebook 10's split
  (`randperm`, seed 0, first 20% test) so the numbers are comparable.
- **Resolution:** keep the **full 120×120** instead of halving it. A star covers only a
  few pixels, and halving the image throws away what separates it from a compact galaxy.
  Check in step 1 that the CNN still reaches notebook 10's ~95% at this resolution.
- **Channels:** SDSS color JPGs map i → R, r → G, g → B, with an asinh stretch
  (Lupton et al. 2004). Pixel values are **not** linear in flux. The color proxy
  c = ln(⟨B⟩/⟨R⟩) in a central aperture tracks g − i in order, not in value. Say so
  wherever color appears.
- **Pixel scale (to verify in step 2):** Galaxy Zoo cutouts were, from memory, scaled to
  each galaxy's Petrosian radius. If so, the PSF width in pixels **varies from image to
  image**, and simulated stars must draw their FWHM from the measured distribution rather
  than one fixed value.
- **Training vs. showcase images.** Training images are always at SDSS resolution and
  composited onto real sky, so the classifier sees what the survey sees. Showcase figures
  may be rendered at high resolution *before* the seeing blur, to show the physics.
  Never train on the showcase renders.
- **Code:** short modules (`data.py`, `sims.py`, `models.py`), with checks in `checks/`
  and figures in `results/`. No `try/except`, no silent defaults. Seed every simulation
  and every training run.
- Run everything in the `HaloPipeline` env. torch and scikit-learn are not installed there
  yet (step 0). There is a GPU (RTX 4090 laptop, 16 GB).

## Natural Steps

Stop for sign-off after each step.

### 0. Environment
Pip-install a CUDA build of torch and scikit-learn into HaloPipeline. Confirm the GPU is
visible and that notebook 10's `train` runs.
*Check:* one CNN epoch takes a few seconds on the GPU.

### 1. Data, a color proxy, and the baseline CNN
Load all 5,000 images, uncertain ones included, but keep them out of training. Compute the
color proxy c for every galaxy. Retrain notebook 10's CNN at 120×120. The input shrinks by
pooling to a different size than at 60×60, so recompute the size of the last `Linear` with
the formula.
*Plots:* histogram of c split by spiral and elliptical; the CNN's test accuracy by epoch.
*Checks:*
- **Bimodality.** Ellipticals sit at the red end of c and spirals at the blue end. A
  single threshold on c alone reaches well above the 76% "always say spiral" baseline.
  Report its accuracy: that is the color-only baseline every later number is compared to.
- **Reproduction.** The CNN at 120×120 gets within about 1 point of notebook 10's 95%.
- **The uncertain galaxies'** c distribution lies between the two classes.

*About 40 min.*

### 2a. Simple stars and artifacts
This is level 1: analytic formulas, enough to get a gallery and the rest of the pipeline
running. Step 2b replaces these with physically derived objects. Build each class in **linear flux**, apply an asinh stretch, and composite the result onto
**real sky backgrounds**: corners of Galaxy Zoo cutouts with the galaxy masked out.
- **Stars:** a Moffat PSF, with FWHM drawn from the distribution measured on field stars
  in the cutouts, a range of brightness, and a stellar color drawn from a range.
- **Cosmic rays:** one or a few pixels, sharper than the PSF, in one channel only. They
  hit one exposure, which is one band.
- **Satellite trails:** a line across the frame with the PSF's width, in every channel.
- **Diffraction spikes:** a bright star with a four-armed cross.

*Plot:* a gallery of real galaxies and each simulated class at several brightnesses.
*Checks:*
- **The PSF.** Fit 2D Moffat profiles to about 20 field stars visible in the cutouts.
  Report the FWHM distribution in pixels and whether it varies from image to image.
  Simulated stars must match real field stars' radial profiles to within the scatter
  between stars.
- **The stretch.** A simulated star's peak-to-wing pixel profile, after the stretch,
  overlays the real field stars'.
- **Flux bookkeeping.** In linear units, the summed flux of each injected source equals
  the flux it was given, to 10⁻⁶.

*About 45 min.*

### 2b. Optics from first principles
Level 2: derive the star image rather than assuming it. Keep it in NumPy, about
100–150 lines in `optics.py`.
- **Pupil → PSF.** Draw SDSS's 2.5 m mirror with its central obstruction and four
  support vanes on a grid. The diffraction-limited PSF is |FFT(pupil)|². This produces
  Airy rings and diffraction spikes.
- **Seeing.** Make a Kolmogorov phase screen: a Gaussian random field with phase power
  spectrum ∝ k^(−11/3), reusing the week-6 `field.py` approach with r₀ set to give about
  1.4″ seeing. A short exposure is |FFT(pupil · e^{iφ})|², which shows speckles.
  Averaging many independent screens gives the long exposure.
- **Star colors.** Treat stars as blackbodies from 3,000 to 30,000 K. Integrate them
  through approximate g, r, i filter curves to get the amplitude in each channel, and
  compute the PSF separately at each band's wavelength, since diffraction and seeing
  depend on λ.
- **Better artifacts.** Cosmic rays as tracks at random angles, with some short curly
  ones. Satellite trails with brightness varying along the line.

*Showcase plots:*
1. Pupil → diffraction-limited PSF (log scale) → the same star at SDSS resolution.
2. A speckle sequence: 1, 10 and 100 averaged short exposures.
3. A gallery of blackbody star colors.

*Checks:*
- **Airy.** For a clear circular pupil, the first dark ring sits at 1.22 λ/D, to within
  a grid pixel.
- **The spikes** lie perpendicular to the vanes, and there are 4 of them.
- **Kolmogorov.** The phase screen's measured structure function follows
  D(r) = 6.88 (r/r₀)^{5/3} over the inertial range.
- **Seeing width.** The long-exposure FWHM ≈ 0.98 λ/r₀, and the profile is fit by a
  Moffat with β near 4.8 (Trujillo et al. 2001). The step-2a formula becomes a result,
  not an assumption.
- **The stellar locus.** Blackbody colors trace a single curve in (g−r, r−i). On the
  image color proxy, they bracket the field stars measured in step 2a.
- **Optional cross-check:** GalSim's `Kolmogorov` and `OpticalPSF` (with struts) agree
  with your FWHM and spike geometry.

Once these checks pass, the level-2 stars and artifacts replace the level-1 ones in all
later training.

*About 60 min.*

### 3. Star vs. galaxy
Train a CNN on real galaxies vs. simulated stars. Use balanced classes, with stars spread
evenly over brightness.
*Plot:* accuracy vs. source brightness for the CNN and for a **size cut**. The size cut
is the analytic baseline: measure each object's second-moment radius and call it a star
if the radius matches the PSF.
*Checks:*
- **The baseline.** For bright sources the size cut is close to perfect, and the CNN
  must match it. Report where each one drops to 50% accuracy.
- **The leak test (most important).** Train a small CNN to tell *real backgrounds* from
  *simulated-background composites with no source added*. It must score about 50%. If it
  doesn't, the star classifier can tell real from simulated by an artifact of the
  pipeline instead of by morphology. Typical culprits are JPEG compression and the edges
  of the composite; recompress the simulations as JPGs if needed.
- **Colors don't leak.** Retrain with star colors drawn from the *galaxy* c distribution.
  The accuracy should barely change, because morphology is enough.

*About 45 min.*

### 4. The galaxy head: morphology × color
Put one shared convolutional trunk under two heads: spiral/elliptical, and red/blue, with
red/blue labels from a cut on c at the trough between the two peaks in step 1.
Compare these versions:
- (a) full color;
- (b) grayscale, the average of the channels, which keeps shape only;
- (c) pixels shuffled within each image, which keeps color only;
- (d) grayscale input predicting the color label.

*Plots:* a table of accuracy for each version; a gallery of the off-diagonal galaxies (red
spirals, blue ellipticals); the morphology head's confusion matrix.
*Checks:*
- **Shuffled ≈ color cut.** On shuffled pixels the CNN can't beat the step-1 single
  threshold on c by more than noise, because no spatial structure is left.
- **Shape predicts color.** Version (d) beats chance clearly. Its failures should be
  concentrated in the red spirals and blue ellipticals, which are real classes in the
  literature (Masters et al. 2010; Schawinski et al. 2009). Look at them by eye.
- **Rotation.** Accuracy is unchanged on test images rotated by 90°. Averaging the
  output over the 8 rotations and reflections of the square (D4) gives a model that
  agrees with itself *exactly* on all 8.

*About 50 min.*

### 5. Known or unknown?
The **known** classes are galaxies, stars, cosmic rays and trails. **Unknown** classes
are held out of training entirely: diffraction spikes, the uncertain galaxies, and one
synthetic oddity, such as a blended pair of galaxies made by overlaying two cutouts.
Compare three OOD scores:
- the largest softmax probability;
- the energy score, the log of the summed exponentials of the network's raw outputs
  (Liu et al. 2020);
- the Mahalanobis distance in the trunk's feature space (Lee et al. 2018).

*Plots:* histograms of each score for known test objects vs. each unknown class; ROC
curves with AUROC.
*Checks:*
- **Calibration of the known classes.** On the known test set, OOD scores flag about the
  fraction set by the threshold you pick (for example 5% at the 95th percentile), and no
  more.
- **Leave-one-out.** Retrain with cosmic rays held out. A good score flags them. Report
  which scores succeed, and explain the failures: a cosmic ray may look "star-like
  enough" to a softmax score.
- The uncertain galaxies are the hard case. They are galaxies, so a strong
  galaxy/star score is *expected* to call them known. Report it, but don't count it as a
  failure.

*About 45 min.*

### 6. Real SDSS stars and quasars *(needs your approval: outside data pull)*
Query SDSS SkyServer for a few thousand spectroscopically classified stars and quasars
(classes STAR and QSO), plus cutouts made to match the Galaxy Zoo cutouts' scale.
*Checks:*
- **From simulation to real data.** The step-3 classifier, trained only on simulated
  stars, labels real stars "star" at the accuracy step 3 predicted for their brightness.
  The gap between the two numbers measures how good the simulation is.
- **Quasars are point sources.** A grayscale CNN can't separate quasars from stars, so
  its accuracy should be near chance. A color CNN can, from the quasars' ultraviolet
  excess. That is a sharp known answer.
- Are real quasars flagged as OOD by step 5's scores when they're absent from training?

*About 50 min (mostly data).*

**Total:** about 6.5 hours. Steps 1–4 stand alone, and step 6 is optional.

## Push Harder

- **Galaxy10** (SDSS or DECaLS, public): ten morphology classes, including barred,
  merging, edge-on with or without a bulge, and tight vs. loose spirals. Swap it in for
  the galaxy head.
- **Galaxy clusters:** larger cutouts centered on known clusters from a catalog such as
  redMaPPer, where the signal is a group of red galaxies rather than one object.
- **Strong lenses as OOD:** inject simulated lensed arcs (from the week-10 lens-finder
  idea) and see whether the OOD scores flag them. Lens searches work this way in
  practice.
- **The shelved simulation project** (bulge + disk simulator, CNN regression of physical
  parameters) is the natural next part: classify first, then measure.

## Reading

*From memory. Check the DOIs before relying on them.*

- Lintott et al., MNRAS 389, 1179 (2008), "Galaxy Zoo." The labels.
- Lupton et al., PASP 116, 133 (2004). The asinh color stretch behind the JPGs.
- Dieleman, Willett & Dambre, MNRAS 450, 1441 (2015). The rotation-invariant CNN that
  won the Galaxy Zoo Kaggle challenge (before step 4).
- Trujillo et al., MNRAS 328, 977 (2001). Moffat β from Kolmogorov turbulence (before
  step 2b). Fried, JOSA 56, 1372 (1966), for r₀.
- Kim & Brunner, MNRAS 464, 4463 (2017). Star/galaxy separation with CNNs (before
  step 3).
- Masters et al., MNRAS 405, 783 (2010), red spirals; Schawinski et al., MNRAS 396, 818
  (2009), blue ellipticals (before step 4).
- Hendrycks & Gimpel, arXiv:1610.02136 (max-softmax baseline); Liu et al.,
  arXiv:2010.03759 (energy score); Lee et al., arXiv:1807.03888 (Mahalanobis). Before
  step 5.
