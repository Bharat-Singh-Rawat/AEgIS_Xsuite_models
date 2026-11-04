"""
Compare Xsuite-simulated delivered beam size -- evaluated at the EXACT
lne02$end marker (the true AEgIS handover point, confirmed via the real
section-boundary markers found in general_lne02.madx) -- against real
experimental data for this year.

WHY THIS VERSION IS DIFFERENT FROM THE EARLIER ATTEMPTS:
-----------------------------------------------------------------------
- Earlier scripts evaluated the simulated beam size at LNE02$END (the
  end of the whole 25m combined sequence, which happens to BE the
  correct point) or at a GUESSED position for the TIMBER monitor
  "LNE02.BSGW.0225" (assuming its "0225" number = 22.5m -- confirmed
  likely wrong, since that exact name doesn't even exist as an element
  in this MAD-X model; the real monitors here are named
  "lne.bsgwa.NNNN", a different convention).
- We now use the REAL lne02$start / lne02$end markers found directly
  in the model, so the simulated value is evaluated at a position we
  are certain about, not a guess.
- PRIMARY comparison target: the parquet's BEAM_sigmaX/BEAM_sigmaY.
  This is because the design document's stated target for AEgIS
  (sigma_x=1.1mm, sigma_y=1.0mm) matches the parquet's real measured
  mean (1.115mm, 0.987mm) almost exactly -- far better than the TIMBER
  monitor, which we now know sits partway through the LNE02 section,
  not at its end.
- TIMBER's LNE02.BSGW.0225 is kept as SECONDARY/contextual data only,
  clearly labeled as being at an uncertain, likely mid-section position
  -- useful context, not the primary validation.

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

LNE02_START_MARKER = "lne02$start"
LNE02_END_MARKER   = "lne02$end"

AEGIS_PARQUET  = "PBAR_Intensity_year.parquet"
TIMBER_XLSX    = "TIMBER_data.xlsx"
TIMBER_COL_PREFIX = "LNE02.BSGW.0225"   # secondary/contextual only -- position uncertain
YEAR_FILTER    = None                    # None = current year

SIGMA_REL   = 0.10                       # PLACEHOLDER injection jitter
N_SAMPLES   = 2000
SEED        = 42

OUT_PNG_TIMESERIES = "aegis_end_sim_vs_real_timeseries.png"
OUT_PNG_HIST        = "aegis_end_sim_vs_real_histogram.png"
OUT_CSV_MC          = "aegis_end_sim_mc_samples.csv"

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
# PART B -- find the EXACT lne02$end marker (the real handover point)
# =========================================================================
name_base = tw_nominal["name"].str.split(":").str[0].str.lower()
s_lne02_start = tw_nominal.loc[name_base == LNE02_START_MARKER, "s"].iloc[0]
s_lne02_end   = tw_nominal.loc[name_base == LNE02_END_MARKER, "s"].iloc[0]
row_end = tw_nominal.loc[name_base == LNE02_END_MARKER].iloc[0]
print(f"\n[setup] confirmed LNE02 section: s={s_lne02_start:.4f} to "
      f"s={s_lne02_end:.4f} (length={s_lne02_end - s_lne02_start:.4f} m)")
print(f"[setup] evaluating simulated beam size at the TRUE end (s={s_lne02_end:.4f})")

sigx_nom_m = np.sqrt(row_end["betx"] * epsx_geom_in)
sigy_nom_m = np.sqrt(row_end["bety"] * epsy_geom_in)
print(f"[simulation] nominal delivered beam size at lne02$end:")
print(f"    sigma_x = {sigx_nom_m*1e3:.4f} mm   sigma_y = {sigy_nom_m*1e3:.4f} mm")

# =========================================================================
# PART C -- Monte Carlo jitter sweep, evaluated at the same exact s
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
    r = tw.loc[(tw["s"] - s_lne02_end).abs().idxmin()]
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
# PART D -- PRIMARY comparison: real parquet BEAM_sigmaX/Y (matches the
# design target almost exactly, confirming it IS the right dataset for
# the true handover point)
# =========================================================================
print(f"\n[data] loading {AEGIS_PARQUET} (primary comparison dataset)...")
df_pq = pd.read_parquet(AEGIS_PARQUET)
aegis = df_pq.loc[df_pq["is_beam_in_AEGIS"]].copy()
year_used = YEAR_FILTER if YEAR_FILTER is not None else pd.Timestamp.now().year
if isinstance(aegis.index, pd.DatetimeIndex):
    aegis = aegis[aegis.index.year == year_used]

print(f"[data] {len(aegis)} AEgIS cycles found for {year_used}")
print(f"[data] real delivered beam size (parquet BEAM_sigmaX/Y, assumed already in mm):")
print(f"    sigma_x = {aegis['BEAM_sigmaX'].mean():.4f} +/- {aegis['BEAM_sigmaX'].std():.4f} mm")
print(f"    sigma_y = {aegis['BEAM_sigmaY'].mean():.4f} +/- {aegis['BEAM_sigmaY'].std():.4f} mm")

print("\n[comparison] simulation (at lne02$end) vs real PARQUET data:")
for label, sim_mean, sim_std, real_col in [
    ("sigma_x", mc_df["sigx_mm"].mean(), mc_df["sigx_mm"].std(), "BEAM_sigmaX"),
    ("sigma_y", mc_df["sigy_mm"].mean(), mc_df["sigy_mm"].std(), "BEAM_sigmaY"),
]:
    real_mean, real_std = aegis[real_col].mean(), aegis[real_col].std()
    pct_diff = 100 * (real_mean - sim_mean) / sim_mean if sim_mean != 0 else np.nan
    print(f"    {label}: simulated {sim_mean:.3f}+/-{sim_std:.3f} mm  vs  "
          f"real {real_mean:.3f}+/-{real_std:.3f} mm   (mean diff {pct_diff:+.1f}%)")

# =========================================================================
# PART E -- SECONDARY/contextual: real TIMBER LNE02.BSGW.0225 data
# (position within LNE02 is uncertain -- NOT necessarily at the end)
# =========================================================================
timber_available = os.path.exists(TIMBER_XLSX)
if timber_available:
    print(f"\n[context] loading {TIMBER_XLSX} for secondary/contextual comparison "
          f"(NOTE: {TIMBER_COL_PREFIX} position within LNE02 is uncertain -- "
          f"it does not exist as a named element in this MAD-X model, so it is "
          f"NOT necessarily at lne02$end. Treat this as context, not primary validation.)...")

    def parse_json_array_col(series):
        def _parse(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return None
            return None
        return series.map(_parse)

    timber = pd.read_excel(TIMBER_XLSX, sheet_name="Multiple Variables")
    timber["Timestamp (UTC_TIME)"] = (
        pd.to_datetime(timber["Timestamp (UTC_TIME)"]).dt.tz_localize("UTC").dt.as_unit("ns")
    )
    timber = timber.set_index("Timestamp (UTC_TIME)")

    is_monitor_in = timber[f"{TIMBER_COL_PREFIX}:Acquisition:isMonitorIn"].astype(bool)
    is_good = parse_json_array_col(timber[f"{TIMBER_COL_PREFIX}:Acquisition:isGoodProfile"])
    is_sat = parse_json_array_col(timber[f"{TIMBER_COL_PREFIX}:Acquisition:isSaturated"])
    gauss_sigma = parse_json_array_col(timber[f"{TIMBER_COL_PREFIX}:FitAcq:gaussSigma"])
    good_mask = (
        is_monitor_in
        & is_good.map(lambda v: isinstance(v, list) and len(v) == 2 and all(v))
        & is_sat.map(lambda v: isinstance(v, list) and len(v) == 2 and not any(v))
        & gauss_sigma.map(lambda v: isinstance(v, list) and len(v) == 2)
    )
    timber_real = timber.loc[good_mask].copy()
    timber_real["sigma_y_mm"] = gauss_sigma.loc[good_mask].map(lambda v: v[0])  # (y, x) order
    timber_real["sigma_x_mm"] = gauss_sigma.loc[good_mask].map(lambda v: v[1])
    timber_real = timber_real[
        timber_real["sigma_x_mm"].between(0.1, 500) & timber_real["sigma_y_mm"].between(0.1, 500)
    ]

    beam_flag = df_pq[["is_beam_in_AEGIS"]].sort_index()
    beam_flag.index = beam_flag.index.tz_convert("UTC").as_unit("ns")
    joined = pd.merge_asof(timber_real.sort_index(), beam_flag, left_index=True,
                            right_index=True, direction="nearest", tolerance=pd.Timedelta("60s"))
    timber_real = joined.loc[joined["is_beam_in_AEGIS"].fillna(False)].copy()
    print(f"[context] {len(timber_real)} TIMBER readings pass quality + beam-presence filters")
    if len(timber_real):
        print(f"    sigma_x = {timber_real['sigma_x_mm'].mean():.4f} +/- "
              f"{timber_real['sigma_x_mm'].std():.4f} mm")
        print(f"    sigma_y = {timber_real['sigma_y_mm'].mean():.4f} +/- "
              f"{timber_real['sigma_y_mm'].std():.4f} mm")
        print("    (this is expected to differ from the parquet numbers above -- "
              "it's from a different, non-endpoint location within LNE02)")
else:
    print(f"\n[context] {TIMBER_XLSX} not found -- skipping secondary comparison")
    timber_real = None

# =========================================================================
# PLOT 1 -- time series: real parquet data + simulated nominal/band overlay
# =========================================================================
fig1, axs1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, real_col, sim_col, label in [
    (axs1[0], "BEAM_sigmaX", "sigx_mm", r"$\sigma_x$"),
    (axs1[1], "BEAM_sigmaY", "sigy_mm", r"$\sigma_y$"),
]:
    ax.plot(aegis.index, aegis[real_col], marker=".", linestyle="none",
            markersize=2, alpha=0.5, color="tab:blue", label="real AEgIS data (parquet)")
    sim_mean, sim_std = mc_df[sim_col].mean(), mc_df[sim_col].std()
    ax.axhline(sim_mean, color="tab:red", lw=2, label="simulated at lne02$end (nominal)")
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
fig1.suptitle(f"AEgIS delivered beam size at lne02$end: real data vs Xsuite simulation")
fig1.tight_layout()
fig1.savefig(OUT_PNG_TIMESERIES, dpi=200)

# =========================================================================
# PLOT 2 -- histogram overlay
# =========================================================================
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
for ax, real_col, sim_col, label in [
    (axs2[0], "BEAM_sigmaX", "sigx_mm", r"$\sigma_x$"),
    (axs2[1], "BEAM_sigmaY", "sigy_mm", r"$\sigma_y$"),
]:
    q_lo, q_hi = aegis[real_col].quantile([0.005, 0.995])
    bins = np.linspace(min(q_lo, mc_df[sim_col].min()), max(q_hi, mc_df[sim_col].max()), 60)
    ax.hist(aegis[real_col].clip(bins[0], bins[-1]), bins=bins, density=True,
            alpha=0.6, color="tab:blue", label=f"real data (N={len(aegis)})")
    ax.hist(mc_df[sim_col], bins=bins, density=True, alpha=0.5,
            color="tab:red", label=f"simulated (N={len(mc_df)})")
    ax.set_xlabel(f"{label} [mm]"); ax.set_ylabel("probability density")
    ax.set_title(f"{label}: real vs simulated distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig2.suptitle("AEgIS delivered beam size at lne02$end: distribution comparison")
fig2.tight_layout()
fig2.savefig(OUT_PNG_HIST, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_PNG_TIMESERIES}\n    {OUT_PNG_HIST}\n    {OUT_CSV_MC}")
