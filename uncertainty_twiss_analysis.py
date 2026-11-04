"""
Uncertainty simulation (10% quadrupole jitter) vs. real 7-monitor
emittance/profile measurement, with error bars on the measured points.

Builds on compare_xsuite_full_twiss_vs_measurement.py:
  1. Computes the NOMINAL Xsuite twiss (as before).
  2. Runs a Monte Carlo uncertainty analysis: jitters every quadrupole's k1
     by +/-10% (uniform), recomputes twiss each time, and builds a
     mean +/- band (5th-95th percentile) for beta_x, beta_y and the
     resulting sigma_x, sigma_y curves.
  3. Overlays the measured beam sizes WITH ERROR BARS on top of the
     nominal + uncertainty-band model curves.

Global plot style (applies to every figure from here on):
  fontsize=15, bold axis/tick labels, linewidth=3

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xtrack as xt

# ==========================================================================
# Global plot style -- applies to ALL figures for the rest of this session
# ==========================================================================
plt.rcParams.update({
    "font.size": 15,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "lines.linewidth": 3,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

MADX_FILE = "general_lne02.madx"
SEQ_NAME  = "lne00lne01lne02"
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

N_MC_SAMPLES = 200          # Monte Carlo sample count
JITTER_FRAC = 0.10          # +/-10% jitter on quadrupole k1
RNG_SEED = 42

# ==========================================================================
# Step 0: identical MAD-X setup to lne02_only_optics_madx.py
# ==========================================================================
def make_runnable_copy(src, dst):
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as f:
        for line_ in lines:
            if line_.strip().lower().startswith("stop"):
                continue
            f.write(line_)

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

g = madx.globals
INIT = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)

# ==========================================================================
# Step 1: build the Xsuite Line and compute NOMINAL twiss
# ==========================================================================
line = xt.Line.from_madx_sequence(madx.sequence[SEQ_NAME], deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=-1,
                                  gamma0=g["gamman"])  # adjust if needed
line.build_tracker()

tw_nom = line.twiss(method="4d", start=xt.START, end=xt.END, init_at=xt.START, **INIT)
tw_df = tw_nom.to_pandas()
tw_df["name_base"] = tw_df["name"].str.split(":").str[0].str.lower()
s_grid = tw_df["s"].to_numpy()

def get_row(model_name):
    row = tw_df.loc[tw_df["name_base"] == model_name.lower()]
    if row.empty:
        raise ValueError(f"Element '{model_name}' not found in Xsuite twiss table")
    return row.iloc[0]

# ==========================================================================
# Step 2: monitor mapping + measured data + measurement uncertainty
# ==========================================================================
MONITOR_NAME_MAP = {
    "LNE00.BSGW.0008": "lne.bsgwa.0008",
    "LNE00.BSGW.0015": "lne.bsgwa.0015",
    "LNE00.BSGW.0025": "lne.bsgwa.0025",
    "LNE00.BSGW.0038": "lne.bsgwa.0038",
    "LNE00.BSGW.0045": "lne.bsgwa.0045",
    "LNE01.BSGW.0109": "lne.bsgwa.0109",
    "LNE01.BSGW.0120": "lne.bsgwa.0120",
}
REFERENCE_MONITOR = "LNE00.BSGW.0008"

# Measured sigma [mm] -- REPLACE WITH REAL LOGGED DATA
measured_data = {
    "LNE00.BSGW.0008": {"sigma_x": 1.1, "sigma_y": 1.9},
    "LNE00.BSGW.0015": {"sigma_x": 1.5, "sigma_y": 2.1},
    "LNE00.BSGW.0025": {"sigma_x": 3.0, "sigma_y": 1.3},
    "LNE00.BSGW.0038": {"sigma_x": 2.5, "sigma_y": 1.0},
    "LNE00.BSGW.0045": {"sigma_x": 1.8, "sigma_y": 1.6},
    "LNE01.BSGW.0109": {"sigma_x": 1.4, "sigma_y": 2.2},
    "LNE01.BSGW.0120": {"sigma_x": 2.4, "sigma_y": 1.1},
}

# Real historical emittance measurements (2023, 2024 High intensity, 2025
# High intensity -- from ADE-Status slides). These are EMITTANCE values
# [um], not sigma -- must be propagated through sigma = sqrt(beta*eps),
# not applied directly as a sigma error.
HISTORICAL_EPS_UM = {
    "x": [2.14, 2.41, 2.83],   # 2023, 2024, 2025
    "y": [2.01, 1.60, 2.26],
}
eps_ref_mean_um = {p: float(np.mean(v)) for p, v in HISTORICAL_EPS_UM.items()}
eps_ref_std_um  = {p: float(np.std(v, ddof=1)) for p, v in HISTORICAL_EPS_UM.items()}
for p in ["x", "y"]:
    print(f"Historical reference eps_{p}: {eps_ref_mean_um[p]:.2f} +/- "
          f"{eps_ref_std_um[p]:.2f} um (from 2023/2024/2025 measurements)")

# ==========================================================================
# Step 3: reconstruct (eps, beta, alpha) from measurement (nominal model),
# same method as before -- used to set the emittance for the sigma curves.
# ==========================================================================
def reconstruct_beam_matrix(tw, plane):
    idx = {"x": (0, 1), "y": (2, 3)}[plane]
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    ref_model_name = MONITOR_NAME_MAP[REFERENCE_MONITOR]

    A_rows, b_vals = [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        R = tw.get_R_matrix(start=ref_model_name, end=model_name)
        R11, R12 = R[idx[0], idx[0]], R[idx[0], idx[1]]
        sigma_m = measured_data[real_name][sig_key] * 1e-3
        A_rows.append([R11**2, -2 * R11 * R12, R12**2])
        b_vals.append(sigma_m**2)

    A, b = np.array(A_rows), np.array(b_vals)
    S11, S12, S22 = np.linalg.lstsq(A, b, rcond=None)[0]
    eps = np.sqrt(max(S11 * S22 - S12**2, 0.0))
    beta0 = S11 / eps
    alpha0 = -S12 / eps
    return eps, beta0, alpha0

eps_reconstructed = {}
for plane in ["x", "y"]:
    eps, beta0, alpha0 = reconstruct_beam_matrix(tw_nom, plane)
    eps_reconstructed[plane] = eps
    print(f"[{plane}] reconstructed eps={eps*1e6:.3f} um, "
          f"beta={beta0:.3f} m, alpha={alpha0:.3f}")

# Real operational reference emittances (see ADE-Status slides + colleague's
# note): 1.3 um = original horizontal target, 2.8 um = measured 2025
# high-intensity value. Used both here and in the intensity-vs-emittance
# plot below for context.
CANDIDATE_EPS_UM = {
    "x": [1.3, 2.8],
    "y": [1.3, 2.8],  # no vertical-specific reference given -- same
                      # candidates used here; replace if you get a real one
}

# ==========================================================================
# Step 4: Monte Carlo uncertainty analysis -- jitter every quadrupole k1 by
# +/-10% (uniform), recompute twiss, collect betx/bety along the line.
# ==========================================================================
rng = np.random.default_rng(RNG_SEED)

quad_names = [
    name for name in line.element_names
    if hasattr(line.element_dict[name], "k1") and line.element_dict[name].k1 != 0
]
nominal_k1 = {name: line.element_dict[name].k1 for name in quad_names}
print(f"\nJittering {len(quad_names)} quadrupoles by +/-{JITTER_FRAC*100:.0f}% "
      f"over {N_MC_SAMPLES} Monte Carlo samples...")

betx_samples = np.empty((N_MC_SAMPLES, len(s_grid)))
bety_samples = np.empty((N_MC_SAMPLES, len(s_grid)))

for i in range(N_MC_SAMPLES):
    for name in quad_names:
        jitter = 1.0 + rng.uniform(-JITTER_FRAC, JITTER_FRAC)
        line.element_dict[name].k1 = nominal_k1[name] * jitter

    tw_i = line.twiss(method="4d", start=xt.START, end=xt.END, init_at=xt.START, **INIT)
    betx_samples[i, :] = tw_i.betx
    bety_samples[i, :] = tw_i.bety

# restore nominal strengths
for name in quad_names:
    line.element_dict[name].k1 = nominal_k1[name]

betx_lo, betx_hi = np.percentile(betx_samples, [5, 95], axis=0)
bety_lo, bety_hi = np.percentile(bety_samples, [5, 95], axis=0)

print("Monte Carlo done.\n")

# ==========================================================================
# Step 5: three separate plots, one per test
# ==========================================================================

# --- Plot 1: full-line beta functions, nominal + jitter uncertainty band ---
fig1, ax1 = plt.subplots(figsize=(14, 7))
ax1.fill_between(s_grid, betx_lo, betx_hi, color="tab:red", alpha=0.2,
                  label=r"$\beta_x$ 5-95% band ($\pm$10% quad jitter)")
ax1.fill_between(s_grid, bety_lo, bety_hi, color="tab:green", alpha=0.2,
                  label=r"$\beta_y$ 5-95% band ($\pm$10% quad jitter)")
ax1.plot(s_grid, tw_df["betx"], color="tab:red", label=r"$\beta_x$ nominal")
ax1.plot(s_grid, tw_df["bety"], color="tab:green", label=r"$\beta_y$ nominal")
for model_name in MONITOR_NAME_MAP.values():
    ax1.axvline(get_row(model_name)["s"], color="gray", ls=":", lw=1.5)
ax1.set_xlabel("s [m]", fontweight="bold")
ax1.set_ylabel(r"$\beta$ [m]", fontweight="bold")
ax1.set_title("Test 1: Full-line Xsuite optics with 10% quadrupole jitter band",
              fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("test1_twiss_optics_jitter.png", dpi=150)
print("Saved plot to test1_twiss_optics_jitter.png")

# --- Plot 2 & 3: sigma comparison, X and Y planes, each its own figure ---
sigma_figs = {}
for plane in ["x", "y"]:
    beta_key = "betx" if plane == "x" else "bety"
    beta_samples = betx_samples if plane == "x" else bety_samples
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    eps = eps_reconstructed[plane]

    beta_vals = np.clip(tw_df[beta_key], 0, None)

    # (a) quadrupole-jitter band, at the reconstructed (single-measurement) eps
    sigma_nom = np.sqrt(beta_vals * eps) * 1e3
    sigma_mc = np.sqrt(np.clip(beta_samples, 0, None) * eps) * 1e3
    sigma_lo, sigma_hi = np.percentile(sigma_mc, [5, 95], axis=0)

    # (b) historical-emittance reference band: eps_mean +/- eps_std from the
    # real 2023/2024/2025 measurements, propagated through sigma=sqrt(beta*eps)
    # -- NOT a sigma error, an emittance error correctly carried through.
    eps_ref = eps_ref_mean_um[plane] * 1e-6
    eps_ref_lo = max(eps_ref_mean_um[plane] - eps_ref_std_um[plane], 0) * 1e-6
    eps_ref_hi = (eps_ref_mean_um[plane] + eps_ref_std_um[plane]) * 1e-6
    sigma_ref_nom = np.sqrt(beta_vals * eps_ref) * 1e3
    sigma_ref_lo = np.sqrt(beta_vals * eps_ref_lo) * 1e3
    sigma_ref_hi = np.sqrt(beta_vals * eps_ref_hi) * 1e3

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(s_grid, sigma_lo, sigma_hi, color="tab:purple", alpha=0.2,
                     label="Model 5-95% band (10% quad jitter)")
    ax.plot(s_grid, sigma_nom, color="tab:purple",
            label=f"Model (reconstructed eps={eps*1e6:.2f} um)")

    ax.fill_between(s_grid, sigma_ref_lo, sigma_ref_hi, color="tab:orange",
                     alpha=0.2,
                     label=f"Historical ref band "
                     f"(eps={eps_ref_mean_um[plane]:.2f}"
                     f"$\\pm${eps_ref_std_um[plane]:.2f} um, 2023-2025)")
    ax.plot(s_grid, sigma_ref_nom, color="tab:orange", ls="--",
            label="Historical ref mean")

    names, s_pos, meas_mm = [], [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        row = get_row(model_name)
        names.append(real_name)
        s_pos.append(row["s"])
        meas_mm.append(measured_data[real_name][sig_key])

    ax.plot(s_pos, meas_mm, "o", color="tab:blue", ms=10,
            label="Measured", zorder=5)
    for xm in s_pos:
        ax.axvline(xm, color="gray", ls=":", lw=1.5)

    test_num = 2 if plane == "x" else 3
    ax.set_xlabel("s [m]", fontweight="bold")
    ax.set_ylabel(r"$\sigma$ [mm]", fontweight="bold")
    ax.set_title(f"Test {test_num}: {plane.upper()}-plane sim vs. experiment "
                 "(quad-jitter band + historical-eps reference band)",
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"test{test_num}_sigma_{plane}_sim_vs_exp.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")
    sigma_figs[plane] = fig

# ==========================================================================
# Test 4: Xsuite beta_x/beta_y vs. the independent CERN operational
# coefficient table (coefficientNew.txt, Elena_AEGIS column) at all 7
# monitors. This is a model-vs-model cross-check, fully independent of
# the uncertain sigma measurement transcription used in Tests 2 & 3.
# ==========================================================================
# Bh = beta_x, Bv = beta_y, as read from the coefficientNew.txt table
COEFFICIENT_TABLE = {
    "LNE00.BSGW.0008": {"Bh": 1.46792, "Bv": 5.35866},
    "LNE00.BSGW.0015": {"Bh": 0.816672, "Bv": 5.93854},
    "LNE00.BSGW.0025": {"Bh": 4.34105, "Bv": 1.15042},
    "LNE00.BSGW.0038": {"Bh": 4.34116, "Bv": 1.14348},
    "LNE00.BSGW.0045": {"Bh": 1.57516, "Bv": 3.3613},
    "LNE01.BSGW.0109": {"Bh": 1.77826, "Bv": 3.99578},
    "LNE01.BSGW.0120": {"Bh": 2.2294, "Bv": 2.23271},
}

names, betx_xsuite, bety_xsuite, betx_table, bety_table = [], [], [], [], []
for real_name, model_name in MONITOR_NAME_MAP.items():
    row = get_row(model_name)
    names.append(real_name)
    betx_xsuite.append(row["betx"])
    bety_xsuite.append(row["bety"])
    betx_table.append(COEFFICIENT_TABLE[real_name]["Bh"])
    bety_table.append(COEFFICIENT_TABLE[real_name]["Bv"])

betx_xsuite = np.array(betx_xsuite)
bety_xsuite = np.array(bety_xsuite)
betx_table = np.array(betx_table)
bety_table = np.array(bety_table)

betx_diff_pct = np.abs(betx_xsuite - betx_table) / betx_table * 100
bety_diff_pct = np.abs(bety_xsuite - bety_table) / bety_table * 100

print("\n=== Test 4: Xsuite beta vs. coefficientNew.txt (Elena_AEGIS) ===")
for i, name in enumerate(names):
    print(f"  {name}: betx Xsuite={betx_xsuite[i]:.4f} table={betx_table[i]:.4f} "
          f"diff={betx_diff_pct[i]:.1f}%  |  "
          f"bety Xsuite={bety_xsuite[i]:.4f} table={bety_table[i]:.4f} "
          f"diff={bety_diff_pct[i]:.1f}%")

fig4, ax4 = plt.subplots(figsize=(14, 7))

# MC jitter 5-95% band + nominal curve, betx and bety together (like Test 1)
ax4.fill_between(s_grid, betx_lo, betx_hi, color="tab:red", alpha=0.2,
                  label=r"$\beta_x$ 5-95% band ($\pm$10% quad jitter)")
ax4.fill_between(s_grid, bety_lo, bety_hi, color="tab:green", alpha=0.2,
                  label=r"$\beta_y$ 5-95% band ($\pm$10% quad jitter)")
ax4.plot(s_grid, tw_df["betx"], color="tab:red", label=r"$\beta_x$ nominal")
ax4.plot(s_grid, tw_df["bety"], color="tab:green", label=r"$\beta_y$ nominal")

# real reference values from the coefficient table, overlaid as markers
table_s = [get_row(MONITOR_NAME_MAP[n])["s"] for n in names]
ax4.plot(table_s, betx_table, "o", color="tab:blue", ms=12,
         label=r"$\beta_x$ (coefficient table, real)", zorder=5)
ax4.plot(table_s, bety_table, "D", color="tab:purple", ms=12,
         label=r"$\beta_y$ (coefficient table, real)", zorder=5)

for xm in table_s:
    ax4.axvline(xm, color="gray", ls=":", lw=1.5)

ax4.set_xlabel("s [m]", fontweight="bold")
ax4.set_ylabel(r"$\beta$ [m]", fontweight="bold")
ax4.set_title("Test 4: beta_x/beta_y sim (10% quad jitter) vs. real "
              "coefficient-table values", fontweight="bold")
ax4.legend()
ax4.grid(alpha=0.3)
fig4.tight_layout()
fig4.savefig("test4_beta_vs_coefficient_table.png", dpi=150)
print("Saved plot to test4_beta_vs_coefficient_table.png")
plt.show()
