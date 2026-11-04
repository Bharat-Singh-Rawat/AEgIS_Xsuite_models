"""
Compare Xsuite-simulated delivered beam size against real AEgIS
experimental data (PBAR_Intensity_year.parquet) for this year.

WHAT THIS DOES:
-----------------------------------------------------------------------
1. Builds the Xsuite LNE02 line (same recipe as
   compare_madx_xsuite_corrected.py: MAD-X -> xt.Line.from_madx_sequence
   -> deflectors patched to real R-matrix FirstOrderTaylorMap).
2. Computes the NOMINAL simulated delivered beam size (sigma_x, sigma_y)
   at the delivery point, for the design injected beam conditions.
3. Runs a Monte Carlo sweep (same idea as sensitivity_injection_analysis.py)
   jittering the injected Twiss parameters by SIGMA_REL, to get a
   SIMULATED SPREAD in delivered beam size -- a real accelerator doesn't
   deliver the exact same beam every cycle, so comparing against a
   single number would be an unfair/meaningless comparison.
4. Loads the real AEgIS data, keeps only cycles where beam was actually
   delivered to AEgIS, and restricts to this year.
5. Plots the real BEAM_sigmaX / BEAM_sigmaY distribution (a full year of
   actual measurements) against the simulated distribution, both as a
   time-series overlay and as a histogram, so you can see directly
   whether the real machine's delivered beam size and spread match what
   the linear-optics model predicts.

WHAT THIS DOES NOT DO: simulate beam POSITION. Position offsets in a
real machine come from orbit/steering errors, injection kicker timing
jitter, etc. -- none of which are in this purely-optics model. Plotting
a "simulated position" here would just be claiming 0 +/- nothing, which
isn't a meaningful comparison. Real position data is still plotted on
its own for reference.

IMPORTANT ASSUMPTIONS -- CHECK / EDIT THESE:
-----------------------------------------------------------------------
- END_PLACE = "LNE02$END" is assumed to be (at or near) the AEgIS
  delivery point. If AEgIS's actual target/profile-monitor location in
  the sequence has its own element name, set END_PLACE to that instead.
- SIGMA_UNIT_MM = True assumes BEAM_sigmaX/BEAM_sigmaY in the parquet
  are reported in mm (common for beam instrumentation); Xsuite's sigma
  is computed in meters and gets multiplied by 1e3 to match. Set to
  False if the parquet is actually in meters.
- SIGMA_REL = 0.10 (10% injection jitter) is the SAME PLACEHOLDER used
  in sensitivity_injection_analysis.py -- replace with real measured
  fill-to-fill jitter if you have it; the whole point of this
  comparison is undermined if the assumed jitter is arbitrary.
- THIS_YEAR: filters the parquet's datetime index to the current year.
  If PBAR_Intensity_year.parquet already only contains one year of
  data, this filter is a no-op; adjust YEAR_FILTER below if needed.
-----------------------------------------------------------------------

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line

Requirements:
    pip install cpymad xtrack pandas matplotlib numpy pyarrow --break-system-packages
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpymad.madx import Madx
import xtrack as xt

# =========================================================================
# Config
# =========================================================================
MADX_FILE  = "general_lne02.madx"
SEQ_NAME   = "lne00lne01lne02"
END_PLACE  = "LNE02$END"          # <-- check this is really the AEgIS delivery point
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

AEGIS_PARQUET  = "PBAR_Intensity_year.parquet"
SIGMA_UNIT_MM  = True              # parquet sigmas assumed in mm; set False if in m
YEAR_FILTER    = None              # e.g. 2026 to force a specific year; None = current year

SIGMA_REL   = 0.10                 # PLACEHOLDER injection jitter -- see note above
N_SAMPLES   = 2000
SEED        = 42

OUT_PNG_TIMESERIES = "aegis_sim_vs_data_timeseries.png"
OUT_PNG_HIST        = "aegis_sim_vs_data_histogram.png"
OUT_CSV_MC          = "aegis_sim_mc_samples.csv"

# =========================================================================
# PART A -- build the Xsuite line (same recipe used throughout this project)
# =========================================================================
def make_runnable_copy(src, dst):
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as f:
        for line_ in lines:
            if line_.strip().lower().startswith("stop"):
                continue
            f.write(line_)

def get_R_from_save(madx, inst_name):
    save_file = "patch_lookup.madx"
    madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
    with open(save_file) as f:
        content = f.read()
    m_inst = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content, re.IGNORECASE)
    if not m_inst:
        raise ValueError(f"instance {inst_name} not found in SAVE output")
    parent_type = m_inst.group(1)
    m_type = re.search(re.escape(parent_type) + r"\s*:\s*matrix\s*,(.*?);",
                        content, re.IGNORECASE | re.DOTALL)
    if not m_type:
        raise ValueError(f"parent type {parent_type} not found as a MATRIX declaration")
    body = m_type.group(1)
    R = np.eye(6)
    n_found = 0
    for i in range(1, 7):
        for j in range(1, 7):
            mm = re.search(rf"\brm{i}{j}\s*=\s*([-+0-9.eEdD]+)", body, re.IGNORECASE)
            if mm:
                R[i - 1, j - 1] = float(mm.group(1).replace("D", "E").replace("d", "e"))
                n_found += 1
    return R, n_found, parent_type

def build_xsuite_line():
    tmp_madx_file = "general_lne02_nostop.madx"
    make_runnable_copy(MADX_FILE, tmp_madx_file)

    madx = Madx(stdout=False)
    madx.call(file=tmp_madx_file)
    os.system("ln -fns ./../../lne lne_repo")
    madx.call(file="lne_repo/deflectors.ele")
    madx.command.use(sequence=SEQ_NAME)
    for inst_name in KNOWN_DEFLECTOR_INSTANCES:
        save_file = "setup_lookup.madx"
        madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
        with open(save_file) as f:
            content = f.read()
        m = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content, re.IGNORECASE)
        if m:
            madx.input(f"{inst_name}: {m.group(1)};")
    madx.command.use(sequence=SEQ_NAME)
    tw_madx = madx.twiss(beta0="initbeta0").dframe()
    tw_madx.columns = [c.lower() for c in tw_madx.columns]

    line = xt.Line.from_madx_sequence(
        madx.sequence[SEQ_NAME], allow_thick=True, deferred_expressions=True,
    )
    beam = madx.sequence[SEQ_NAME].beam
    line.particle_ref = xt.Particles(mass0=beam.mass * 1e9, q0=beam.charge, p0c=beam.pc * 1e9)
    line.build_tracker()

    print("[setup] patching deflectors with real R-matrix FirstOrderTaylorMap...")
    for inst_name in KNOWN_DEFLECTOR_INSTANCES:
        try:
            R, n_found, parent_type = get_R_from_save(madx, inst_name)
        except Exception as e:
            print(f"    [skip] {inst_name}: {e}")
            continue
        length_row = tw_madx[tw_madx["name"].str.split(":").str[0] == inst_name]
        length = float(length_row.iloc[0]["l"]) if len(length_row) else 0.6
        target_name = next((c for c in line.element_names if c.split(":")[0] == inst_name), None)
        if target_name is None:
            continue
        new_elem = None
        for attempt in (
            lambda: xt.FirstOrderTaylorMap(length=length, R=R),
            lambda: xt.FirstOrderTaylorMap(length=length, m=R),
            lambda: xt.FirstOrderTaylorMap(length=length, r=R),
            lambda: xt.FirstOrderTaylorMap(length=length, R_matrix=R),
            lambda: xt.FirstOrderTaylorMap(R, length=length),
        ):
            try:
                new_elem = attempt()
                break
            except (TypeError, NameError):
                continue
        if new_elem is None:
            continue
        line.element_dict[target_name] = new_elem
        print(f"    [patched] {target_name} (found {n_found}/36 R values)")

    line.build_tracker()
    return madx, line

print("[setup] building Xsuite line...")
madx, line = build_xsuite_line()

g = madx.globals
init_kwargs = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)
beam = madx.sequence[SEQ_NAME].beam
epsx_geom_in = beam.exn / (g["beta"] * g["gamman"])
epsy_geom_in = beam.eyn / (g["beta"] * g["gamman"])

def do_twiss():
    return line.twiss(method="4d", start=xt.START, end=xt.END,
                       init=xt.TwissInit(**init_kwargs)).to_pandas()

def end_row(tw):
    row = tw[tw["name"].str.upper().str.startswith(END_PLACE.upper())]
    return row.iloc[0] if len(row) else tw.iloc[-1]

# =========================================================================
# PART B -- nominal (deterministic) simulated beam size at delivery
# =========================================================================
tw_nominal = do_twiss()
row_nominal = end_row(tw_nominal)
sigx_nom_m = np.sqrt(row_nominal["betx"] * epsx_geom_in)
sigy_nom_m = np.sqrt(row_nominal["bety"] * epsy_geom_in)
print(f"\n[simulation] nominal delivered beam size at {END_PLACE}:")
print(f"    sigma_x = {sigx_nom_m*1e3:.4f} mm   sigma_y = {sigy_nom_m*1e3:.4f} mm")

# =========================================================================
# PART C -- Monte Carlo jitter sweep (same idea as sensitivity_injection_analysis.py)
# =========================================================================
rng = np.random.default_rng(SEED)
nominal_injected = dict(init_kwargs)
nominal_injected["exn"], nominal_injected["eyn"] = beam.exn, beam.eyn
JITTERED_PARAMS = ["betx", "alfx", "bety", "alfy", "exn", "eyn"]

mc_rows = []
print(f"\n[MC] running {N_SAMPLES} jittered samples (+/-{SIGMA_REL*100:.0f}% Gaussian "
      f"on injected Twiss/emittance)...")
for i in range(N_SAMPLES):
    sample = dict(nominal_injected)
    for p in JITTERED_PARAMS:
        sample[p] = nominal_injected[p] * (1 + rng.normal(0, SIGMA_REL))

    try:
        tw = line.twiss(
            method="4d", start=xt.START, end=xt.END,
            init=xt.TwissInit(
                betx=sample["betx"], alfx=sample["alfx"],
                bety=sample["bety"], alfy=sample["alfy"],
                dx=nominal_injected["dx"], dpx=nominal_injected["dpx"],
                dy=nominal_injected["dy"], dpy=nominal_injected["dpy"],
            ),
        ).to_pandas()
    except Exception:
        continue

    row = end_row(tw)
    epsx_geom = sample["exn"] / (g["beta"] * g["gamman"])
    epsy_geom = sample["eyn"] / (g["beta"] * g["gamman"])
    sigx = np.sqrt(row["betx"] * epsx_geom)
    sigy = np.sqrt(row["bety"] * epsy_geom)
    mc_rows.append(dict(sigx_mm=sigx * 1e3, sigy_mm=sigy * 1e3))

mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(OUT_CSV_MC, index=False)
print(f"[MC] {len(mc_df)}/{N_SAMPLES} samples succeeded, saved to {OUT_CSV_MC}")
print(f"[MC] simulated spread: sigma_x = {mc_df['sigx_mm'].mean():.4f} +/- "
      f"{mc_df['sigx_mm'].std():.4f} mm   sigma_y = {mc_df['sigy_mm'].mean():.4f} +/- "
      f"{mc_df['sigy_mm'].std():.4f} mm")

# =========================================================================
# PART D -- load real AEgIS data
# =========================================================================
df = pd.read_parquet(AEGIS_PARQUET)
aegis = df.loc[df["is_beam_in_AEGIS"]].copy()

year_to_use = YEAR_FILTER if YEAR_FILTER is not None else pd.Timestamp.now().year
if isinstance(aegis.index, pd.DatetimeIndex):
    aegis = aegis[aegis.index.year == year_to_use]
    print(f"\n[data] {len(aegis)} AEgIS cycles found for {year_to_use} "
          f"(out of {len(df)} total cycles in the file)")
else:
    print(f"\n[data] WARNING: index is not a DatetimeIndex -- could not filter "
          f"to year {year_to_use}; using all {len(aegis)} AEgIS cycles in the file")

if not SIGMA_UNIT_MM:
    aegis["BEAM_sigmaX"] = aegis["BEAM_sigmaX"] * 1e3
    aegis["BEAM_sigmaY"] = aegis["BEAM_sigmaY"] * 1e3

print("[data] real AEgIS delivered beam size this year (mm):")
print(f"    sigma_x = {aegis['BEAM_sigmaX'].mean():.4f} +/- {aegis['BEAM_sigmaX'].std():.4f} mm "
      f"(median {aegis['BEAM_sigmaX'].median():.4f})")
print(f"    sigma_y = {aegis['BEAM_sigmaY'].mean():.4f} +/- {aegis['BEAM_sigmaY'].std():.4f} mm "
      f"(median {aegis['BEAM_sigmaY'].median():.4f})")

# =========================================================================
# PART E -- comparison summary
# =========================================================================
print("\n[comparison] simulation vs real AEgIS data:")
for label, sim_mean, sim_std, real_col in [
    ("sigma_x", mc_df["sigx_mm"].mean(), mc_df["sigx_mm"].std(), "BEAM_sigmaX"),
    ("sigma_y", mc_df["sigy_mm"].mean(), mc_df["sigy_mm"].std(), "BEAM_sigmaY"),
]:
    real_mean, real_std = aegis[real_col].mean(), aegis[real_col].std()
    pct_diff = 100 * (real_mean - sim_mean) / sim_mean if sim_mean != 0 else np.nan
    print(f"    {label}: simulated {sim_mean:.3f}+/-{sim_std:.3f} mm  vs  "
          f"real {real_mean:.3f}+/-{real_std:.3f} mm   (mean diff {pct_diff:+.1f}%)")

# -------------------------------------------------------------------
# IMPLIED REAL EMITTANCE: the database has no logged emittance/beta at
# all (checked all 86 columns), so we can't directly compare a "real"
# emittance to the MAD-X design-file value. Instead, invert
# sigma^2 = beta_design * epsilon using the REAL measured sigma and the
# DESIGN beta at delivery (which agreed with MAD-X to <0.01% for betx,
# so it's a trustworthy stand-in). This tells us, in physically
# meaningful units, how far off the design-file's assumed emittance is
# from what the real machine actually delivers.
# -------------------------------------------------------------------
beta_rel_gamma_rel = g["beta"] * g["gamman"]
print("\n[implied emittance] inverting real measured sigma against the "
      "DESIGN beta (betx/bety at delivery agree with MAD-X to <0.01%, "
      "so this isolates the emittance assumption specifically):")
for label, real_col, beta_design, eps_design_n in [
    ("x", "BEAM_sigmaX", row_nominal["betx"], beam.exn),
    ("y", "BEAM_sigmaY", row_nominal["bety"], beam.eyn),
]:
    sigma_real_m = (aegis[real_col].mean() * 1e-3)  # mm -> m
    eps_geom_implied = sigma_real_m ** 2 / beta_design
    eps_n_implied = eps_geom_implied * beta_rel_gamma_rel
    ratio = eps_design_n / eps_n_implied if eps_n_implied > 0 else np.nan
    print(f"    {label}: design-file normalized emittance = {eps_design_n*1e6:.4f} mm.mrad   "
          f"implied-by-real-data = {eps_n_implied*1e6:.4f} mm.mrad   "
          f"(design is {ratio:.1f}x larger)")
print("[implied emittance] if this ratio is roughly consistent between x and y, "
      "and roughly matches (simulated_sigma/real_sigma)^2 above, that confirms "
      "the sigma discrepancy is being driven mainly by the design-file's "
      "emittance assumption, not by an optics/model bug.")

# =========================================================================
# PLOT 1 -- time series: real data (scatter) + simulated nominal/band overlay
# =========================================================================
fig1, axs1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, real_col, sim_col, label in [
    (axs1[0], "BEAM_sigmaX", "sigx_mm", r"$\sigma_x$"),
    (axs1[1], "BEAM_sigmaY", "sigy_mm", r"$\sigma_y$"),
]:
    ax.plot(aegis.index, aegis[real_col], marker=".", linestyle="none",
            markersize=2, alpha=0.5, color="tab:blue", label="real AEgIS data")
    sim_mean, sim_std = mc_df[sim_col].mean(), mc_df[sim_col].std()
    ax.axhline(sim_mean, color="tab:red", lw=2, label="simulated (nominal, jittered mean)")
    ax.axhspan(sim_mean - sim_std, sim_mean + sim_std, color="tab:red", alpha=0.15,
               label=f"simulated spread (+/-{SIGMA_REL*100:.0f}% injection jitter)")
    q_lo, q_hi = aegis[real_col].quantile([0.01, 0.99])
    pad = 0.2 * (q_hi - q_lo) if q_hi > q_lo else 1.0
    ax.set_ylim(min(q_lo, sim_mean - 2 * sim_std) - pad,
                max(q_hi, sim_mean + 2 * sim_std) + pad)
    ax.set_ylabel(f"{label} [mm]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

axs1[1].set_xlabel("Time")
fig1.suptitle(f"AEgIS delivered beam size {year_to_use}: real data vs Xsuite simulation")
fig1.tight_layout()
fig1.savefig(OUT_PNG_TIMESERIES, dpi=200)

# =========================================================================
# PLOT 2 -- histogram overlay: real distribution vs simulated MC distribution
# =========================================================================
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))

for ax, real_col, sim_col, label in [
    (axs2[0], "BEAM_sigmaX", "sigx_mm", r"$\sigma_x$"),
    (axs2[1], "BEAM_sigmaY", "sigy_mm", r"$\sigma_y$"),
]:
    q_lo, q_hi = aegis[real_col].quantile([0.005, 0.995])
    bins = np.linspace(min(q_lo, mc_df[sim_col].min()),
                        max(q_hi, mc_df[sim_col].max()), 60)
    ax.hist(aegis[real_col].clip(bins[0], bins[-1]), bins=bins, density=True,
            alpha=0.6, color="tab:blue", label=f"real data (N={len(aegis)})")
    ax.hist(mc_df[sim_col], bins=bins, density=True, alpha=0.5,
            color="tab:red", label=f"simulated (N={len(mc_df)})")
    ax.set_xlabel(f"{label} [mm]")
    ax.set_ylabel("probability density")
    ax.set_title(f"{label}: real vs simulated distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig2.suptitle(f"AEgIS delivered beam size {year_to_use}: distribution comparison")
fig2.tight_layout()
fig2.savefig(OUT_PNG_HIST, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_PNG_TIMESERIES}\n    {OUT_PNG_HIST}\n    {OUT_CSV_MC}")
