"""
Compare Xsuite-simulated delivered beam size against REAL profile-monitor
data from TIMBER (TIMBER_data.xlsx), plus a real ELENA-to-AEgIS
transmission efficiency check from PBAR_Intensity_year.parquet.

WHY THIS REPLACES compare_xsuite_vs_aegis_data.py:
-----------------------------------------------------------------------
The earlier comparison used BEAM_sigmaX/BEAM_sigmaY from the parquet
file, which turned out to likely be measuring a completely different
beam (possibly the proton beam hitting the AD target, not the
antiproton beam delivered to AEgIS) -- which would fully explain why
that comparison was off by ~90-170x. TIMBER_data.xlsx instead has REAL
profile-monitor (BSGW = beam-gas / screen profile monitor) data at two
actual locations along this line:
    LNE00.BSGW.0008   (near the start, s ~ 0.8 m)
    LNE02.BSGW.0225   (near the delivery point, s ~ 22.5 m)
LNE02.BSGW.0225 is the AEgIS-relevant one and is used here.

WHAT THIS DOES:
-----------------------------------------------------------------------
1. Builds the Xsuite LNE02 line (same recipe as before).
2. Finds LNE02.BSGW.0225's actual s-position in the line (by name if
   the instrument exists as an element, else falls back to the nearest
   element to MONITOR_S_FALLBACK) and computes the simulated beta_x/
   beta_y THERE -- not at LNE02$END -- for a genuine apples-to-apples
   comparison with where the real monitor actually sits.
3. Runs the same injection-jitter Monte Carlo as before to get a
   simulated spread, not just a single deterministic number.
4. Loads TIMBER_data.xlsx, parses the JSON-array-valued columns
   (gaussSigma etc. are stored as strings like "[1.0, 5.58]"), and
   keeps only physically trustworthy readings:
       isMonitorIn == True   (the monitor was actually inserted)
       isGoodProfile == [True, True]  (both planes fit well)
       isSaturated == [False, False]  (neither plane saturated)
   The pair is stored as (y, x) -- i.e. index 0 = vertical, index 1 =
   horizontal -- and values are already calibrated in mm everywhere
   except amplitude (per direct confirmation from the data owner).
5. Plots real vs simulated beam size (time series + histogram), and
   separately, the REAL ELENA-to-AEgIS transmission efficiency
   (AEGIS_int / ELENA_eje) from the parquet file over the year.

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line

Requirements:
    pip install cpymad xtrack pandas matplotlib numpy pyarrow openpyxl --break-system-packages
"""

import os
import re
import json
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
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

MONITOR_NAME       = "lne02.bsgw.0225"   # AEgIS-relevant profile monitor
MONITOR_S_FALLBACK = 22.5                # used only if MONITOR_NAME isn't found as an element
MONITOR_COL_PREFIX = "LNE02.BSGW.0225"   # column prefix in TIMBER_data.xlsx

TIMBER_XLSX    = "TIMBER_data.xlsx"
AEGIS_PARQUET  = "PBAR_Intensity_year.parquet"
YEAR_FILTER    = None                    # None = current year

SIGMA_REL   = 0.10                       # PLACEHOLDER injection jitter (see earlier scripts)
N_SAMPLES   = 2000
SEED        = 42

OUT_PNG_TIMESERIES = "timber_sim_vs_data_timeseries.png"
OUT_PNG_HIST        = "timber_sim_vs_data_histogram.png"
OUT_PNG_EFFICIENCY  = "elena_to_aegis_transmission_efficiency.png"
OUT_CSV_MC          = "timber_sim_mc_samples.csv"
OUT_CSV_REAL        = "timber_real_filtered_readings.csv"

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

tw_nominal = do_twiss()

# =========================================================================
# PART B -- find the monitor's actual s-position and simulated optics there
# =========================================================================
match = tw_nominal[tw_nominal["name"].str.split(":").str[0].str.lower() == MONITOR_NAME.lower()]
if len(match):
    row_nominal = match.iloc[0]
    print(f"\n[setup] found {MONITOR_NAME} in the line at s={row_nominal['s']:.4f} m")
else:
    idx = (tw_nominal["s"] - MONITOR_S_FALLBACK).abs().idxmin()
    row_nominal = tw_nominal.loc[idx]
    print(f"\n[setup] {MONITOR_NAME} not found as a named element -- using nearest "
          f"element to s={MONITOR_S_FALLBACK} m instead: {row_nominal['name']} "
          f"(s={row_nominal['s']:.4f} m)")

sigx_nom_m = np.sqrt(row_nominal["betx"] * epsx_geom_in)
sigy_nom_m = np.sqrt(row_nominal["bety"] * epsy_geom_in)
print(f"[simulation] nominal delivered beam size at this monitor's s:")
print(f"    sigma_x = {sigx_nom_m*1e3:.4f} mm   sigma_y = {sigy_nom_m*1e3:.4f} mm")

# =========================================================================
# PART C -- Monte Carlo jitter sweep at this same s-position
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
    r = tw.loc[(tw["s"] - row_nominal["s"]).abs().idxmin()]
    epsx_geom = sample["exn"] / (g["beta"] * g["gamman"])
    epsy_geom = sample["eyn"] / (g["beta"] * g["gamman"])
    mc_rows.append(dict(sigx_mm=np.sqrt(r["betx"] * epsx_geom) * 1e3,
                         sigy_mm=np.sqrt(r["bety"] * epsy_geom) * 1e3))

mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(OUT_CSV_MC, index=False)
print(f"[MC] {len(mc_df)}/{N_SAMPLES} samples succeeded, saved to {OUT_CSV_MC}")
print(f"[MC] simulated spread: sigma_x = {mc_df['sigx_mm'].mean():.4f} +/- "
      f"{mc_df['sigx_mm'].std():.4f} mm   sigma_y = {mc_df['sigy_mm'].mean():.4f} +/- "
      f"{mc_df['sigy_mm'].std():.4f} mm")

# =========================================================================
# PART D -- load and parse the real TIMBER profile-monitor data
# =========================================================================
def parse_json_array_col(series):
    def _parse(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    return series.map(_parse)

print(f"\n[data] loading {TIMBER_XLSX}...")
timber = pd.read_excel(TIMBER_XLSX, sheet_name="Multiple Variables")
timber["Timestamp (UTC_TIME)"] = (
    pd.to_datetime(timber["Timestamp (UTC_TIME)"]).dt.tz_localize("UTC").dt.as_unit("ns")
)
timber = timber.set_index("Timestamp (UTC_TIME)")

# -------------------------------------------------------------------
# TIME-JOIN with the parquet's is_beam_in_AEGIS flag.
# The profile monitors log a reading every ~2 minutes REGARDLESS of
# whether AEgIS is actually being served in that moment (ELENA runs a
# supercycle feeding several experiments in turn) -- so most TIMBER
# readings are "monitor inserted, no beam pulse present", producing
# degenerate near-zero fits. Only readings whose timestamp coincides
# with a real is_beam_in_AEGIS==True cycle in the parquet represent an
# actual AEgIS delivery.
#
# TIMBER's "Timestamp (UTC_TIME)" is tz-naive but genuinely UTC; the
# parquet's index is tz-aware Europe/Zurich -- both are converted to
# UTC before the join. Sampling cadence is similar on both sides
# (~110-115s), so a nearest-match join with a tolerance well under
# that cadence (JOIN_TOLERANCE) avoids pairing a TIMBER reading with a
# parquet cycle that's actually a full sample apart.
# -------------------------------------------------------------------
JOIN_TOLERANCE = pd.Timedelta("60s")

print(f"[join] loading {AEGIS_PARQUET} to confirm real beam presence at each TIMBER reading...")
df_pq = pd.read_parquet(AEGIS_PARQUET)
beam_flag = df_pq[["is_beam_in_AEGIS"]].sort_index()
beam_flag.index = beam_flag.index.tz_convert("UTC").as_unit("ns")

def join_beam_presence(df_in, label=""):
    df_sorted = df_in.sort_index()
    joined = pd.merge_asof(
        df_sorted, beam_flag, left_index=True, right_index=True,
        direction="nearest", tolerance=JOIN_TOLERANCE,
    )
    n_before = len(joined)
    out = joined.loc[joined["is_beam_in_AEGIS"].fillna(False)].copy()
    print(f"    [join{(' ' + label) if label else ''}] {len(out)}/{n_before} readings "
          f"coincide with a real is_beam_in_AEGIS=True cycle (within {JOIN_TOLERANCE})")
    return out

is_monitor_in = timber[f"{MONITOR_COL_PREFIX}:Acquisition:isMonitorIn"].astype(bool)
is_good = parse_json_array_col(timber[f"{MONITOR_COL_PREFIX}:Acquisition:isGoodProfile"])
is_sat = parse_json_array_col(timber[f"{MONITOR_COL_PREFIX}:Acquisition:isSaturated"])
gauss_sigma = parse_json_array_col(timber[f"{MONITOR_COL_PREFIX}:FitAcq:gaussSigma"])

good_mask = (
    is_monitor_in
    & is_good.map(lambda v: isinstance(v, list) and len(v) == 2 and all(v))
    & is_sat.map(lambda v: isinstance(v, list) and len(v) == 2 and not any(v))
    & gauss_sigma.map(lambda v: isinstance(v, list) and len(v) == 2)
)

real = timber.loc[good_mask].copy()
real["sigma_y_mm"] = gauss_sigma.loc[good_mask].map(lambda v: v[0])  # index 0 = vertical (pair is stored as (y, x))
real["sigma_x_mm"] = gauss_sigma.loc[good_mask].map(lambda v: v[1])  # index 1 = horizontal

REAL_SIGMA_MIN_MM = 0.1    # below this is almost certainly a degenerate noise fit, not real beam
REAL_SIGMA_MAX_MM = 500.0  # above this is almost certainly a pathological fit artifact

def apply_sigma_sanity_filter(df_in, label=""):
    """isGoodProfile/isSaturated flags alone are NOT sufficient -- pathological
    fits (observed up to ~1e97 mm in this data) can still pass those flags.
    This applies a plain physical plausibility bound on top."""
    n_before = len(df_in)
    mask = (
        df_in["sigma_x_mm"].between(REAL_SIGMA_MIN_MM, REAL_SIGMA_MAX_MM)
        & df_in["sigma_y_mm"].between(REAL_SIGMA_MIN_MM, REAL_SIGMA_MAX_MM)
    )
    out = df_in.loc[mask].copy()
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"    [sanity filter{(' ' + label) if label else ''}] dropped {n_dropped}/{n_before} "
              f"additional readings outside [{REAL_SIGMA_MIN_MM}, {REAL_SIGMA_MAX_MM}] mm "
              f"(pathological fits that passed the boolean quality flags anyway)")
    return out

real = apply_sigma_sanity_filter(real, MONITOR_NAME)
real = join_beam_presence(real, MONITOR_NAME)

if YEAR_FILTER is not None:
    real = real[real.index.year == YEAR_FILTER]
year_used = YEAR_FILTER if YEAR_FILTER is not None else pd.Series(real.index.year).mode()[0]

real.to_csv(OUT_CSV_REAL)
print(f"[data] {len(real)}/{len(timber)} readings passed quality filters "
      f"(isMonitorIn & isGoodProfile & !isSaturated), saved to {OUT_CSV_REAL}")
print(f"[data] real {MONITOR_NAME} delivered beam size:")
print(f"    sigma_x = {real['sigma_x_mm'].mean():.4f} +/- {real['sigma_x_mm'].std():.4f} mm "
      f"(median {real['sigma_x_mm'].median():.4f})")
print(f"    sigma_y = {real['sigma_y_mm'].mean():.4f} +/- {real['sigma_y_mm'].std():.4f} mm "
      f"(median {real['sigma_y_mm'].median():.4f})")

# =========================================================================
# PART E -- comparison + implied emittance (repeated correctly with real data)
# =========================================================================
print("\n[comparison] simulation vs real TIMBER data:")
for label, sim_mean, sim_std, real_col in [
    ("sigma_x", mc_df["sigx_mm"].mean(), mc_df["sigx_mm"].std(), "sigma_x_mm"),
    ("sigma_y", mc_df["sigy_mm"].mean(), mc_df["sigy_mm"].std(), "sigma_y_mm"),
]:
    real_mean, real_std = real[real_col].mean(), real[real_col].std()
    pct_diff = 100 * (real_mean - sim_mean) / sim_mean if sim_mean != 0 else np.nan
    print(f"    {label}: simulated {sim_mean:.3f}+/-{sim_std:.3f} mm  vs  "
          f"real {real_mean:.3f}+/-{real_std:.3f} mm   (mean diff {pct_diff:+.1f}%)")

beta_rel_gamma_rel = g["beta"] * g["gamman"]
print("\n[implied emittance] inverting real measured sigma against the design beta "
      "at this monitor's s-position:")
for label, real_col, beta_design, eps_design_n in [
    ("x", "sigma_x_mm", row_nominal["betx"], beam.exn),
    ("y", "sigma_y_mm", row_nominal["bety"], beam.eyn),
]:
    sigma_real_m = real[real_col].mean() * 1e-3
    eps_geom_implied = sigma_real_m ** 2 / beta_design
    eps_n_implied = eps_geom_implied * beta_rel_gamma_rel
    ratio = eps_design_n / eps_n_implied if eps_n_implied > 0 else np.nan
    print(f"    {label}: design-file normalized emittance = {eps_design_n*1e6:.4f} mm.mrad   "
          f"implied-by-real-data = {eps_n_implied*1e6:.4f} mm.mrad   "
          f"(design is {ratio:.2f}x larger)")

# =========================================================================
# PART F -- real ELENA-to-AEgIS transmission efficiency (parquet)
# =========================================================================
print(f"\n[efficiency] using already-loaded {AEGIS_PARQUET} for real transmission efficiency...")
eff = df_pq.loc[df_pq["is_beam_in_AEGIS"] & (df_pq["ELENA_eje"] > 0)].copy()
eff["transmission_pct"] = 100 * eff["AEGIS_int"] / eff["ELENA_eje"]
year_used_pq = YEAR_FILTER if YEAR_FILTER is not None else pd.Timestamp.now().year
if isinstance(eff.index, pd.DatetimeIndex) and YEAR_FILTER is None:
    eff = eff[eff.index.year == pd.Timestamp.now().year]

print(f"[efficiency] {len(eff)} AEgIS-delivery cycles with valid ELENA_eje > 0")
print(f"[efficiency] real ELENA -> AEgIS transmission: "
      f"{eff['transmission_pct'].mean():.2f}% +/- {eff['transmission_pct'].std():.2f}% "
      f"(median {eff['transmission_pct'].median():.2f}%)")
print("[efficiency] NOTE: AEGIS_int may be raw delivered intensity rather than "
      "trapped intensity (per your note) -- treat this as an upper bound on "
      "actual trapping efficiency, not the trapping efficiency itself.")

# =========================================================================
# PART G -- full-line beam envelope: simulated sigma(s) vs BOTH real monitors
# =========================================================================
MONITOR_NAME_2       = "lne00.bsgw.0008"
MONITOR_S_FALLBACK_2 = 0.8
MONITOR_COL_PREFIX_2 = "LNE00.BSGW.0008"
OUT_PNG_ENVELOPE     = "beam_envelope_sim_vs_real.png"

def find_monitor_s(name, fallback):
    m = tw_nominal[tw_nominal["name"].str.split(":").str[0].str.lower() == name.lower()]
    if len(m):
        r = m.iloc[0]
        print(f"[envelope] found {name} in the line at s={r['s']:.4f} m")
        return r
    idx = (tw_nominal["s"] - fallback).abs().idxmin()
    r = tw_nominal.loc[idx]
    print(f"[envelope] {name} not found as a named element -- using nearest "
          f"element to s={fallback} m instead: {r['name']} (s={r['s']:.4f} m)")
    return r

def load_real_monitor_data(col_prefix):
    is_in = timber[f"{col_prefix}:Acquisition:isMonitorIn"].astype(bool)
    good = parse_json_array_col(timber[f"{col_prefix}:Acquisition:isGoodProfile"])
    sat = parse_json_array_col(timber[f"{col_prefix}:Acquisition:isSaturated"])
    sigma = parse_json_array_col(timber[f"{col_prefix}:FitAcq:gaussSigma"])
    mask = (
        is_in
        & good.map(lambda v: isinstance(v, list) and len(v) == 2 and all(v))
        & sat.map(lambda v: isinstance(v, list) and len(v) == 2 and not any(v))
        & sigma.map(lambda v: isinstance(v, list) and len(v) == 2)
    )
    out = timber.loc[mask].copy()
    out["sigma_y_mm"] = sigma.loc[mask].map(lambda v: v[0])  # index 0 = vertical (pair is stored as (y, x))
    out["sigma_x_mm"] = sigma.loc[mask].map(lambda v: v[1])  # index 1 = horizontal
    out = apply_sigma_sanity_filter(out, MONITOR_NAME_2)
    out = join_beam_presence(out, MONITOR_NAME_2)
    return out

row_nominal_2 = find_monitor_s(MONITOR_NAME_2, MONITOR_S_FALLBACK_2)
real_2 = load_real_monitor_data(MONITOR_COL_PREFIX_2)
print(f"[envelope] {MONITOR_NAME_2}: {len(real_2)}/{len(timber)} readings passed "
      f"quality filters -- sigma_x = {real_2['sigma_x_mm'].mean():.3f}+/-"
      f"{real_2['sigma_x_mm'].std():.3f} mm, sigma_y = {real_2['sigma_y_mm'].mean():.3f}+/-"
      f"{real_2['sigma_y_mm'].std():.3f} mm")

# simulated beam envelope along the WHOLE line, from the same nominal twiss
# -------------------------------------------------------------------
# SANITY CHECK: a single non-physical extreme value anywhere in
# tw_nominal["betx"]/["bety"] will blow up the plot's y-axis scale and
# crush everything else to invisible near zero. Find and report any
# such outliers BEFORE plotting, so a bad element is caught explicitly
# rather than silently producing a useless (e271-scale) plot.
# -------------------------------------------------------------------
SANE_BETA_LIMIT = 1e4  # meters -- anything above this is not physical for this line
bad_rows = tw_nominal[(tw_nominal["betx"].abs() > SANE_BETA_LIMIT) |
                       (tw_nominal["bety"].abs() > SANE_BETA_LIMIT)]
if len(bad_rows):
    print(f"\n[envelope] WARNING: {len(bad_rows)} element(s) have non-physical "
          f"beta values (> {SANE_BETA_LIMIT:.0e} m) -- this is almost certainly "
          f"a numerical artifact, not real optics. Offending elements:")
    print(bad_rows[["name", "s", "betx", "bety"]].to_string(index=False))
    print("[envelope] excluding these from the envelope plot's y-scale "
          "(clipped to the sane limit) so the real structure is still visible. "
          "The root cause is very likely the deflector R-matrix patch "
          "(missing 10/36 entries) -- this needs fixing at the source, not "
          "just clipping here.")
else:
    print(f"\n[envelope] OK: no non-physical beta values found (all < {SANE_BETA_LIMIT:.0e} m)")

sigx_curve_mm = np.sqrt(np.clip(tw_nominal["betx"].clip(-SANE_BETA_LIMIT, SANE_BETA_LIMIT), 0, None) * epsx_geom_in) * 1e3
sigy_curve_mm = np.sqrt(np.clip(tw_nominal["bety"].clip(-SANE_BETA_LIMIT, SANE_BETA_LIMIT), 0, None) * epsy_geom_in) * 1e3

fig4, axs4 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for ax, sigma_curve, real_col, label in [
    (axs4[0], sigx_curve_mm, "sigma_x_mm", r"$\sigma_x$"),
    (axs4[1], sigy_curve_mm, "sigma_y_mm", r"$\sigma_y$"),
]:
    ax.plot(tw_nominal["s"], sigma_curve, color="tab:red", lw=2,
            label="simulated (nominal, design emittance)")
    for r_data, r_pos, mcolor, mname in [
        (real, row_nominal, "tab:blue", MONITOR_NAME),
        (real_2, row_nominal_2, "tab:green", MONITOR_NAME_2),
    ]:
        ax.errorbar(r_pos["s"], r_data[real_col].mean(), yerr=r_data[real_col].std(),
                    fmt="o", ms=8, color=mcolor, capsize=5,
                    label=f"real {mname} (N={len(r_data)})")
    ax.set_ylabel(f"{label} [mm]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axs4[1].set_xlabel("s [m]")
fig4.suptitle("LNE02: simulated beam envelope vs real profile-monitor measurements\n"
              "(NOTE: simulated curve uses the design-file emittance -- see the "
              "implied-emittance print above for how far off that likely is)")
fig4.tight_layout()
fig4.savefig(OUT_PNG_ENVELOPE, dpi=200)
print(f"[envelope] saved {OUT_PNG_ENVELOPE}")

# =========================================================================
fig1, axs1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, real_col, sim_col, label in [
    (axs1[0], "sigma_x_mm", "sigx_mm", r"$\sigma_x$"),
    (axs1[1], "sigma_y_mm", "sigy_mm", r"$\sigma_y$"),
]:
    ax.plot(real.index, real[real_col], marker=".", linestyle="none",
            markersize=3, alpha=0.5, color="tab:blue", label=f"real {MONITOR_NAME} data")
    sim_mean, sim_std = mc_df[sim_col].mean(), mc_df[sim_col].std()
    ax.axhline(sim_mean, color="tab:red", lw=2, label="simulated (nominal, jittered mean)")
    ax.axhspan(sim_mean - sim_std, sim_mean + sim_std, color="tab:red", alpha=0.15,
               label=f"simulated spread (+/-{SIGMA_REL*100:.0f}% injection jitter)")
    q_lo, q_hi = real[real_col].quantile([0.01, 0.99])
    pad = 0.2 * (q_hi - q_lo) if q_hi > q_lo else 1.0
    ax.set_ylim(min(q_lo, sim_mean - 2 * sim_std) - pad,
                max(q_hi, sim_mean + 2 * sim_std) + pad)
    ax.set_ylabel(f"{label} [mm]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
axs1[1].set_xlabel("Time")
fig1.suptitle(f"AEgIS delivered beam size ({MONITOR_NAME}): real data vs Xsuite simulation")
fig1.tight_layout()
fig1.savefig(OUT_PNG_TIMESERIES, dpi=200)

# =========================================================================
# PLOT 2 -- histogram overlay
# =========================================================================
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
for ax, real_col, sim_col, label in [
    (axs2[0], "sigma_x_mm", "sigx_mm", r"$\sigma_x$"),
    (axs2[1], "sigma_y_mm", "sigy_mm", r"$\sigma_y$"),
]:
    q_lo, q_hi = real[real_col].quantile([0.005, 0.995])
    bins = np.linspace(min(q_lo, mc_df[sim_col].min()), max(q_hi, mc_df[sim_col].max()), 60)
    ax.hist(real[real_col].clip(bins[0], bins[-1]), bins=bins, density=True,
            alpha=0.6, color="tab:blue", label=f"real data (N={len(real)})")
    ax.hist(mc_df[sim_col], bins=bins, density=True, alpha=0.5,
            color="tab:red", label=f"simulated (N={len(mc_df)})")
    ax.set_xlabel(f"{label} [mm]"); ax.set_ylabel("probability density")
    ax.set_title(f"{label}: real vs simulated distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig2.suptitle(f"AEgIS delivered beam size ({MONITOR_NAME}): distribution comparison")
fig2.tight_layout()
fig2.savefig(OUT_PNG_HIST, dpi=200)

# =========================================================================
# PLOT 3 -- real ELENA -> AEgIS transmission efficiency over time
# =========================================================================
fig3, ax3 = plt.subplots(figsize=(14, 4.5))
ax3.plot(eff.index, eff["transmission_pct"], marker=".", linestyle="none",
         markersize=2, alpha=0.5, color="tab:green")
ax3.axhline(eff["transmission_pct"].mean(), color="k", lw=1.5, ls="--",
            label=f"mean = {eff['transmission_pct'].mean():.1f}%")
ax3.set_ylabel("AEGIS_int / ELENA_eje [%]")
ax3.set_xlabel("Time")
ax3.set_title(f"Real ELENA -> AEgIS transmission efficiency {year_used_pq} "
              "(AEGIS_int may be delivered, not trapped, intensity)")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
q_lo, q_hi = eff["transmission_pct"].quantile([0.01, 0.99])
pad = 0.2 * (q_hi - q_lo) if q_hi > q_lo else 1.0
ax3.set_ylim(max(0, q_lo - pad), q_hi + pad)
fig3.tight_layout()
fig3.savefig(OUT_PNG_EFFICIENCY, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_PNG_TIMESERIES}\n    {OUT_PNG_HIST}\n    {OUT_PNG_EFFICIENCY}"
      f"\n    {OUT_CSV_MC}\n    {OUT_CSV_REAL}")
