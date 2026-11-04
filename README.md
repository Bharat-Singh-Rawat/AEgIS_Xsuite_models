# AEgIS Xsuite Models: ELENA → LNE00 / LNE01 / LNE02 transfer line

Optics, tracking, optimisation and failure-analysis studies of the ELENA
extraction line that delivers 100 keV antiprotons to the **AEgIS** experiment
(LNE00 + LNE01 + LNE02, about 24.6 m).

The baseline lattice is the CERN `acc-models-tls` MAD-X model
(`general_lne02.madx`). Every Python script here loads that model with
**cpymad**, converts it to an **Xsuite** `xt.Line`, and then runs Twiss,
tracking, matching or Monte Carlo studies. The results are checked against
MAD-X and against real machine data (TIMBER profile monitors, the AEgIS
intensity logs, and the 7-monitor emittance measurement).

---

## Contents

- [Physics setup](#physics-setup)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Script reference](#script-reference)
  - [1. Model validation: MAD-X vs Xsuite](#1-model-validation-mad-x-vs-xsuite)
  - [2. Optimisation](#2-optimisation)
  - [3. Quadrupole failure, degradation and recovery](#3-quadrupole-failure-degradation-and-recovery)
  - [4. Tracking](#4-tracking)
  - [5. Comparison with measurements](#5-comparison-with-measurements)
  - [6. Uncertainty analysis](#6-uncertainty-analysis)
- [Recommended run order](#recommended-run-order)
- [Outputs](#outputs)
- [Known caveats](#known-caveats)

---

## Physics setup

| Quantity | Value |
|---|---|
| Particle | antiproton |
| Kinetic energy | 100 keV |
| Normalised emittance (MAD-X beam block) | εx,n = 1 µm, εy,n = 0.67 µm (`exn=6e-6/6`, `eyn=4e-6/6`) |
| Sequence | `lne00lne01lne02` (flattened from LNE00, LNE01 and LNE02) |
| Initial conditions | `../stitched/elena_stitched.inp` (`betx0`, `alfx0`, …) |
| Delivery point | `LNE02$END` (`lne02$end` marker, the AEgIS handover) |

### Electrostatic deflectors

The line contains two electrostatic deflectors, `lnr.zdfhl.0610` and
`lne.zdshr.0220`. They are **MATRIX-type elements**: each is an arbitrary
linear R-matrix defined in `lne_repo/deflectors.ele`. A shared `survey.seq`
file later redefines them as SBEND approximations for survey purposes, and
that redefinition breaks the optics.

Every script undoes it in the same way:

1. It re-calls `lne_repo/deflectors.ele`.
2. It uses `SAVE` to find each deflector instance's parent type, then rebinds
   the instance to that MATRIX type.
3. In Xsuite, it replaces each deflector with an `xt.FirstOrderTaylorMap`
   built from the real R-matrix (`get_R_from_save`).

This correction is the reason the MAD-X and Xsuite Twiss functions agree (see
`madx_vs_xsuite_corrected.png`). PTC is **not** used, because it has no
element type for a generic linear map.

---

## Repository layout

This repository contains **only the Python analysis scripts**. They are meant
to be placed in, and run from, `acc-models-tls/elena_extraction/lne02/line/`
of the CERN `acc-models-tls` repository. That folder provides:

```
lne02/line/
├── general_lne02.madx          # Main MAD-X model (LNE00+LNE01+LNE02, antiproton 100 keV)
├── beta.inp                    # Initial beta/Twiss parameters
├── lne_repo -> ../../lne       # Element/strength/sequence files (symlink created automatically)
└── <the *.py scripts from this repository>
```

Every script writes its figures (`*.png`) and result tables (`*.csv`) into
the directory it is run from. It also regenerates these scratch files on
every run: `general_lne02_nostop.madx`, `setup_lookup.madx`,
`patch_lookup.madx`, `instance_parent_lookup.madx`, `inrays.madx` and
`trackone`.

---

## Requirements

```bash
pip install cpymad xtrack xpart xobjects numpy pandas matplotlib scipy \
            iminuit pyarrow openpyxl pymadx
```

| Package | Used for |
|---|---|
| `cpymad` | Running the MAD-X model from Python |
| `xtrack` / `xpart` | Xsuite line, Twiss, tracking and `line.match()` |
| `iminuit` | MINUIT2 optimiser for the speed benchmark (falls back to SciPy Nelder–Mead) |
| `scipy` | KDE plots and the fallback optimiser |
| `pyarrow` | Reading `PBAR_Intensity_year.parquet` |
| `openpyxl` | Reading `TIMBER_data.xlsx` |
| `pymadx` | Gaussian bunch generation for MAD-X `TRACK` |

You also need a checkout of the CERN `acc-models-tls` repository, because
the model reads `general_lne02.madx`, `../../lne/` and
`../stitched/elena_stitched.inp`.

---

## How to run

Run every script **from this directory**:

```bash
cd acc-models-tls/elena_extraction/lne02/line
python compare_madx_xsuite_corrected.py
```

Each script:

1. writes `general_lne02_nostop.madx` (a copy without `STOP` so cpymad can
   continue),
2. recreates the `lne_repo` symlink,
3. restores the MATRIX deflectors,
4. builds the Xsuite line with
   `xt.Line.from_madx_sequence(..., deferred_expressions=True)`.

Configuration is set with constants at the top of each file, for example
`N_MC_SAMPLES`, `JITTER_FRAC`, `FAILURE_FRACTION`, `N_RUNS` and `SEED`.

---

## Script reference

### 1. Model validation: MAD-X vs Xsuite

#### `compare_madx_xsuite_corrected.py`
Compares Twiss functions computed by MAD-X with those computed by Xsuite
after the deflectors are patched to `FirstOrderTaylorMap`. It overlays βx,
βy, Dx and Dy along the whole line. Every other Xsuite script uses this
validated way of building the line.
**Output:** `madx_vs_xsuite_corrected.png`

---

### 2. Optimisation

#### `speed_comparison_madx_vs_xsuite_iminuit.py`
Benchmarks **MAD-X vs Xsuite as the physics engine** underneath the **same
optimiser** (iminuit / MINUIT2).

- **Variables:** the four LNE02 quadrupole strengths `KLNE.ZQMD.0208`,
  `KLNE.ZQMF.0209`, `KLNE.ZQMD.0214` and `KLNE.ZQMF.0215`, with bounds
  ±54 m⁻².
- **Objectives:**
  - `beamsize`: delivered σx + σy at `LNE02$END`, including dispersion.
  - `emittance`: the emittance-driven beam size only, without dispersion.
- **Constraints:** βx < 50 m and βy < 100 m along the line, enforced with a
  quadratic penalty (`PENALTY_W = 1e4`).
- **Monte Carlo:** 100 random starting points per objective (`N_RUNS`,
  spread ±10 around nominal, `SEED = 42`). Both engines get identical starts.
- **Recorded per run:** wall-clock time, number of function calls, final
  objective value and convergence flag.

**Outputs:** `speed_comparison_results.csv`, `speed_timing_comparison.png`,
`speed_objective_comparison.png`

> The output table also lists `mc_optimization_results.csv`, from an earlier
> Monte Carlo quad-optimisation run with the same four quads and objectives.
> The script that produced it is not included in this repository.

#### `quad_failure_recovery_xsuite.py`
**Re-matching with Xsuite's native optimiser.** It takes the most critical
single-quad failures from the scan (`TOP_N_CRITICAL = 5`) and asks whether the
neighbouring quads (`N_NEIGHBORS = 2` on each side) can be re-tuned to restore
the nominal delivered optics at `LNE02$END`.

- Each quad's `k1` is wrapped in an Xsuite knob (`k1_<quad>`) so it can be
  used directly as a `vary` in `line.match()`.
- It uses `xt.VaryList` and `xt.TargetSet` on the Twiss values at
  `LNE02$END` (`MATCH_STEP = 1e-6`, `MATCH_TOL = 1e-9`).
- It reports, per failure, the residual mismatch after re-matching and the
  re-tune cost (how far each neighbour's strength moved).

**Requires:** `quad_failure_scan_results_xsuite.csv`
**Outputs:** `quad_recovery_results_xsuite.csv`,
`quad_recovery_residuals_xsuite.png`, `quad_recovery_retune_cost_xsuite.png`,
`quad_recovery_twiss_examples_xsuite.png`

---

### 3. Quadrupole failure, degradation and recovery

#### `quad_failure_scan_xsuite.py`
**Single-quadrupole failure scan.** It switches each quad off in turn
(`FAILURE_FRACTION = 1.0`; 0.5 means 50 % under-strength) and re-runs Twiss
with the nominal injected beam held fixed. It then ranks the quads by an
**impact score** built from the change in delivered β, α and σ at
`LNE02$END`.
**Outputs:** `quad_failure_scan_results_xsuite.csv`,
`quad_failure_ranking_xsuite.png`, `quad_failure_twiss_overlay_xsuite.png`

#### `partial_failure_tracking_analysis_xsuite.py`
**Partial-degradation study** for the top 3 critical quads from the scan,
at degradation levels 0, 10, 25, 50, 75 and 100 %. For each level it tracks a
matched Gaussian bunch of 5000 particles with `line.track()` and produces:

- **Profile grid:** the delivered x–y beam profile at each degradation level.
- **Mismatch / growth curve:** the mismatch factor
  `R = ½(βγ₀ + β₀γ − 2αα₀)` and the effective emittance-growth factor
  `R + √(R²−1)` the beam would reach after filamentation.
- **Phase-space ellipse:** the tilt and rotation of the delivered ellipse.

In a linear line the true emittance is conserved, so a failing quad causes
**mismatch**, not instantaneous emittance growth.

**Requires:** `quad_failure_scan_results_xsuite.csv`
**Outputs:** `partial_failure_<quad>_{profile_grid,mismatch_growth,phase_space_ellipse}_xsuite.png`,
`partial_failure_<quad>_results_xsuite.csv`, `partial_failure_summary_top_quads_xsuite.csv`

Critical quads found so far: `lne.zqmd.0014`, `lne.zqmf.0013` and
`lne.zqfd.0047`.

---

### 4. Tracking

#### `track_beam_profile_beta_vs_twiss.py`
Tracks 20 000 macro-particles with **MAD-X native `TRACK`** (not PTC; see the
deflector note above) and records them at 8 observation points from injection
to `LNE02$END`. At each point it:

- plots the x–y beam profile, and
- computes β and ε from the covariance matrix of the tracked distribution,
  then compares that β with the analytic MAD-X Twiss.

**Outputs:** `track_beam_profiles_lne02.png`, `track_beta_vs_twiss_lne02.png`
(intermediate files: `inrays.madx`, `trackone`)

---

### 5. Comparison with measurements

#### `compare_sim_vs_emittance_measurement_madx.py`
Compares the full MAD-X model (all three sections) with the real
**"Emittance Measurement using 7 monitors"** result from the Beam Profile
Viewer.

- Builds the beam matrix from the measured σ at the 7 monitors using model
  R-matrices.
- Reconstructs (ε, β, α).
- Reports the Twiss mismatch factor and the geometric mismatch.

**Output:** `sim_vs_measurement_sigma.png`

#### `compare_xsuite_full_twiss_vs_measurement.py`
Does the same comparison with the **Twiss computed by Xsuite**. It plots βx,
βy, Dx and Dy along the whole line with the 7 monitors marked. It then
compares the predicted σ with the measured σ for two emittance assumptions:

- **design:** the emittance from the MAD-X beam block.
- **reconstructed:** the emittance reconstructed from the measurement. This
  isolates whether the shape of the optics is right.

**Output:** `xsuite_full_twiss_vs_measurement.png`

#### `beta_sim_vs_coefficient_table.py`
Compares Xsuite βx and βy, with a ±10 % quad-jitter Monte Carlo band (200
samples), against the operational **coefficient table** (`coefficientNew.txt`,
Elena_AEGIS) at the 7 real monitors. It also compares σ from simulation and
experiment.
**Outputs:** `beta_sim_vs_coefficient_table.png`, `sigma_sim_vs_experiment.png`

#### `sigma_sim_vs_experiment_with_candidates.py`
Extends the previous script by overlaying the parquet `BEAM_sigmaX` and
`BEAM_sigmaY` (median and 25–75 % band over all AEgIS cycles). They are shown
at the two candidate positions of the unresolved monitor, `lne.bsgwa.0207`
and `lne.bsgwa.0225`. This is illustrative only, because neither position is
confirmed.
**Output:** `sigma_sim_vs_experiment_with_candidates.png`

#### `compare_xsuite_vs_aegis_data.py`
Compares the simulated delivered σx and σy (nominal value plus a 10 %
injection-jitter Monte Carlo) at `LNE02$END` with a year of AEgIS data from
`PBAR_Intensity_year.parquet`, using delivered cycles only. Shows a time
series and a histogram.
**Outputs:** `aegis_sim_vs_data_{timeseries,histogram}.png`, `aegis_sim_mc_samples.csv`

#### `compare_xsuite_vs_timber_data.py`
Compares the simulation with the **TIMBER BSGW profile monitors**
(`TIMBER_data.xlsx`, mainly `LNE02.BSGW.0225`). Before comparing, it filters
the readings with these conditions:

- `isMonitorIn == True`
- `isGoodProfile == [True, True]`
- `isSaturated == [False, False]`

Then it applies a σ sanity filter. The values are stored as (y, x) pairs in
mm. The script also plots the beam envelope against the real readings and the
real **ELENA → AEgIS transmission efficiency** (`AEGIS_int / ELENA_eje`).
**Outputs:** `timber_sim_vs_data_{timeseries,histogram}.png`,
`beam_envelope_sim_vs_real.png`, `elena_to_aegis_transmission_efficiency.png`,
`timber_sim_mc_samples.csv`, `timber_real_filtered_readings.csv`

#### `compare_xsuite_vs_experiment_at_end.py`
The final version of the delivered-beam comparison. It evaluates the
simulation at the exact `lne02$end` marker. The main reference is the parquet
`BEAM_sigmaX` and `BEAM_sigmaY`: the measured means (1.115 mm, 0.987 mm) match
the AEgIS design target (1.1 mm, 1.0 mm). TIMBER `LNE02.BSGW.0225` is shown
only as secondary context, because that monitor sits partway along LNE02.
**Outputs:** `aegis_end_sim_vs_real_{timeseries,histogram}.png`, `aegis_end_sim_mc_samples.csv`

---

### 6. Uncertainty analysis

#### `uncertainty_twiss_analysis.py`
Runs a Monte Carlo with uniform ±10 % jitter on every quadrupole `k1`
(200 samples, seed 42). It shows the nominal value and a 5–95 % band for βx,
βy, σx and σy, and overlays the measured beam sizes with error bars.
**Outputs:** `test1_twiss_optics_jitter.png`, `test2_sigma_x_sim_vs_exp.png`,
`test3_sigma_y_sim_vs_exp.png`, `test4_beta_vs_coefficient_table.png`,
`uncertainty_twiss_vs_measurement.png`

---

## Recommended run order

```bash
# 1. Validate the Xsuite model against MAD-X
python compare_madx_xsuite_corrected.py

# 2. Optics vs measurements
python compare_xsuite_full_twiss_vs_measurement.py
python uncertainty_twiss_analysis.py
python beta_sim_vs_coefficient_table.py
python sigma_sim_vs_experiment_with_candidates.py

# 3. Delivered beam vs operational data (needs PBAR_Intensity_year.parquet / TIMBER_data.xlsx)
python compare_xsuite_vs_timber_data.py
python compare_xsuite_vs_experiment_at_end.py

# 4. Failure studies (the scan must run first)
python quad_failure_scan_xsuite.py
python partial_failure_tracking_analysis_xsuite.py
python quad_failure_recovery_xsuite.py

# 5. Optimiser / engine benchmark
python speed_comparison_madx_vs_xsuite_iminuit.py

# 6. Particle tracking (MAD-X TRACK)
python track_beam_profile_beta_vs_twiss.py
```

---

## Outputs

| Category | Files |
|---|---|
| Model validation | `madx_vs_xsuite_corrected.png` |
| Optimisation | `speed_comparison_results.csv`, `speed_*_comparison.png`, `mc_optimization_results.csv` |
| Failure scan | `quad_failure_scan_results_xsuite.csv`, `quad_failure_*_xsuite.png` |
| Partial failure | `partial_failure_*_xsuite.{png,csv}` |
| Recovery | `quad_recovery_*_xsuite.{png,csv}` |
| Tracking | `track_beam_profiles_lne02.png`, `track_beta_vs_twiss_lne02.png` |
| Measurements | `sim_vs_measurement_sigma.png`, `xsuite_full_twiss_vs_measurement.png`, `beta_sim_vs_coefficient_table.png`, `sigma_sim_vs_experiment*.png` |
| Operational data | `aegis_*`, `timber_*`, `beam_envelope_sim_vs_real.png`, `elena_to_aegis_transmission_efficiency.png` |
| Uncertainty | `test{1..4}_*.png`, `uncertainty_twiss_vs_measurement.png` |

These output files are not stored in this repository. You get them by
running the scripts.

---

## Known caveats

- **Injection jitter is a placeholder.** `SIGMA_REL = 0.10` and
  `JITTER_FRAC = 0.10` are assumed values, not measured fill-to-fill jitter.
  Replace them with measured values when they become available.
- **Beam position is not simulated.** The model is purely linear optics, with
  no orbit or steering errors, so only beam sizes are compared.
- **Monitor naming.** The TIMBER name `LNE02.BSGW.0225` does not exist in the
  MAD-X model, where the monitors are named `lne.bsgwa.NNNN`. Its exact
  s-position is therefore uncertain.
- **`BEAM_sigmaX/Y` units.** Assumed to be mm (`SIGMA_UNIT_MM = True`).
- **Operational data not included.** The measurement comparisons need
  `PBAR_Intensity_year.parquet` and `TIMBER_data.xlsx`. These contain CERN
  operational data and are not part of this repository.

---

## Author

Bharat Singh Rawat, AEgIS / ELENA LNE02 optics studies.
The MAD-X baseline model (`general_lne02.madx`) is by M. A. Fraser and
F. M. Velotti (CERN `acc-models-tls`).
