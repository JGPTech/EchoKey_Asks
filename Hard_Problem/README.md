# EchoKey v2: RP/LRP Demonstration

> **EchoKey asks:**
> Can what appears as *causal buildup* in the brain’s readiness potential (RP) be explained as an *emergent correlation* from time-locking, averaging, and operator composition?

---

## What does this repository contain?

This repo provides a pipeline that:

* Fetches **open EEG data** from PhysioNet’s *EEG Motor Movement/Imagery Dataset* (EEGBCI) automatically on first run.
* Extracts **event-related potentials** (RP at Cz, lateralized readiness potential at C3/C4).
* Simulates a **stochastic accumulator** to test whether threshold-crossing + time-locking produces RP-like averages.
* Applies the **EchoKey v2 operator palette** (cyclicity, recursion, fractality, regression, synergy, refraction, outliers) to probe whether the buildup is preserved under transformations.
* Summarizes results in **figures** and **metrics** (JSON/CSV), with all console output reflecting computed values, not assumptions.

---

## Orientation: Questions

### Q0. Can we obtain empirical EEG data without accounts or credentials?

* **Attempted answer:** Using the `mne.datasets.eegbci` helper, EDF files are downloaded once from PhysioNet into a local cache. On later runs, cached copies are reused.
* **Evidence:** The script successfully retrieves Subject 1, runs 3–4 (motor execution), yielding usable events `T1` (left) and `T2` (right).

---

### Q1. Does standard averaging produce the expected slow negativity (RP) and lateralization (LRP)?

* **Attempted answer:** Baseline-corrected epochs around cues show a slow pre-onset trend at Cz and a left–right asymmetry at C3/C4.
* **Evidence:** Figures `q1_empirical_rp_lrp.png` display the averaged RP and LRP, with mean/variance statistics recorded in the summary file.

---

### Q2. Does time-locking stochastic threshold crossings generate RP-like buildup without an explicit decision signal?

* **Attempted answer:** A leaky stochastic accumulator with Gaussian noise, mean drift, and threshold crossing is simulated. Aligning trials at threshold yields an averaged trajectory resembling the empirical RP.
* **Evidence:** Figure `q2_simulated_rp.png` shows the simulated buildup. Cross rate and number of usable trials are reported on the console.

---

### Q3. Do EchoKey operators preserve the apparent buildup when applied?

* **Attempted answers and evidence:**

  * **Cyclicity:** Band-limiting preserves the slow component (correlation ≈ 0.99).
  * **Recursion:** Iterative smoothing stabilizes the trend without sharp onsets (correlation ≈ 0.95).
  * **Fractality:** Energy concentrates in coarse scales, consistent with long-timescale structure.
  * **Regression:** Mean-reversion removes drift but preserves pre-cue negativity.
  * **Synergy:** Mixing empirical and simulated signals retains the shared buildup (moderate correlation).
  * **Refraction:** Spectral tilt alters fine detail while keeping gross shape.
  * **Outliers:** Downweighting extreme trials yields similar averages (low RMSE vs. mean).

* **Evidence:**

  * Figures `q3_transforms_grid.png` and `q3_fractality_energies.png`.
  * Numeric results in `q3_summary.json` and `q3_summary.csv`.

---

### Q4. Is there an objectively detectable onset relative to baseline?

* **Attempted answer:** A simple z-threshold heuristic on the empirical RP suggests a sustained pre-cue crossing at ≈ -0.33 s.
* **Evidence:** Reported as `onset_precue_s` in console output.

---

## Closing Note

This repository does **not** claim to resolve the hard problem of consciousness.
It only poses procedural questions:

* Could apparent causal buildup emerge from correlated processes under time-locking?
* Do operator transforms preserve the same statistical features?
* Is the signal robust to baseline choice, trial selection, or averaging strategy?

**EchoKey asks:** The answers above are *attempted procedural answers* based on empirical and simulated data, not assertions of fact.

---
