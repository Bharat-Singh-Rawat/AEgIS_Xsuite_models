"""
Full Xsuite twiss vs. real 7-monitor emittance/profile measurement.

Builds the Xsuite Line from the same MAD-X model used in
lne02_only_optics_madx.py (via cpymad -> xt.Line.from_madx_sequence),
computes twiss with Xsuite itself (not madx.twiss()), then:

  1. Plots beta_x, beta_y, D_x, D_y along the WHOLE line (LNE00+LNE01+LNE02),
     with the 7 real monitor positions marked.
  2. Plots model-predicted sigma_x/sigma_y (from Xsuite optics) against the
     real measured sigma at each monitor, for two emittance assumptions:
       - "design"        : the MAD-X beam-block normalized emittance
       - "reconstructed" : the emittance reconstructed from the real
                            measurement itself (so this curve isolates
                            whether the model's OPTICS SHAPE is right,
                            independent of the emittance-scale mismatch)
  3. Reconstructs (eps, beta, alpha) at a reference monitor from the
     measured sigmas using Xsuite's own R-matrices, and reports the Twiss
     mismatch factor and geometric mismatch (= sqrt(Twiss mismatch),
     confirmed against real ELENA measurement reports).

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xtrack as xt

MADX_FILE = "general_lne02.madx"
SEQ_NAME  = "lne00lne01lne02"
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

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
beam = madx.sequence[SEQ_NAME].beam
epsx_design = beam.exn / (g["beta"] * g["gamman"])
epsy_design = beam.eyn / (g["beta"] * g["gamman"])

INIT = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)

print(f"Design emittances (from MAD-X beam block): "
      f"epsx={epsx_design*1e6:.3f} um, epsy={epsy_design*1e6:.3f} um")
print("NOTE: real operational emittances at ELENA typically run ~1-3 um "
      "and grow with bunch intensity -- design value is NOT representative "
      "(see ADE-Status slides). Use the 'reconstructed' emittance below for "
      "a meaningful sigma comparison.\n")

# ==========================================================================
# Step 1: build the Xsuite Line from the MAD-X sequence and twiss it
# ==========================================================================
line = xt.Line.from_madx_sequence(madx.sequence[SEQ_NAME], deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=-1,
                                  gamma0=g["gamman"])  # antiproton-like ref; adjust if needed
line.build_tracker()

tw = line.twiss(method="4d", start=xt.START, end=xt.END, init_at=xt.START, **INIT)
tw_df = tw.to_pandas()
tw_df["name_base"] = tw_df["name"].str.split(":").str[0].str.lower()

print(f"Xsuite twiss computed over {len(tw_df)} elements, "
      f"s = 0 to {tw_df['s'].iloc[-1]:.3f} m\n")

# ==========================================================================
# Step 2: monitor mapping + measured data (confirmed earlier from MAD-X run)
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

# Measured sigma at each real monitor [mm] -- REPLACE WITH REAL LOGGED DATA
measured_data = {
    "LNE00.BSGW.0008": {"sigma_x": 1.1, "sigma_y": 1.9},
    "LNE00.BSGW.0015": {"sigma_x": 1.5, "sigma_y": 2.1},
    "LNE00.BSGW.0025": {"sigma_x": 3.0, "sigma_y": 1.3},
    "LNE00.BSGW.0038": {"sigma_x": 2.5, "sigma_y": 1.0},
    "LNE00.BSGW.0045": {"sigma_x": 1.8, "sigma_y": 1.6},
    "LNE01.BSGW.0109": {"sigma_x": 1.4, "sigma_y": 2.2},
    "LNE01.BSGW.0120": {"sigma_x": 2.4, "sigma_y": 1.1},
}

def get_row(model_name):
    row = tw_df.loc[tw_df["name_base"] == model_name.lower()]
    if row.empty:
        raise ValueError(f"Element '{model_name}' not found in Xsuite twiss table")
    return row.iloc[0]

# ==========================================================================
# Step 3: reconstruct (eps, beta, alpha) at REFERENCE_MONITOR using Xsuite's
# own R-matrices between monitors (tw.get_R_matrix), same method as the real
# "7 monitor emittance measurement" tool.
# ==========================================================================
def reconstruct_beam_matrix(plane):
    idx = {"x": (0, 1), "y": (2, 3)}[plane]
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    ref_model_name = MONITOR_NAME_MAP[REFERENCE_MONITOR]

    A_rows, b_vals = [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        R = tw.get_R_matrix(start=ref_model_name, end=model_name)
        R11, R12 = R[idx[0], idx[0]], R[idx[0], idx[1]]
        sigma_m = measured_data[real_name][sig_key] * 1e-3  # mm -> m
        A_rows.append([R11**2, -2 * R11 * R12, R12**2])
        b_vals.append(sigma_m**2)

    A, b = np.array(A_rows), np.array(b_vals)
    S11, S12, S22 = np.linalg.lstsq(A, b, rcond=None)[0]
    eps = np.sqrt(max(S11 * S22 - S12**2, 0.0))
    beta0 = S11 / eps
    alpha0 = -S12 / eps  # S12 = -eps*alpha0
    return eps, beta0, alpha0

def twiss_mismatch(beta_a, alpha_a, beta_b, alpha_b):
    """Standard Twiss mismatch factor (Sacherer). 1.0 = perfect match."""
    gamma_a = (1 + alpha_a**2) / beta_a
    gamma_b = (1 + alpha_b**2) / beta_b
    return 0.5 * (beta_a * gamma_b - 2 * alpha_a * alpha_b + gamma_a * beta_b)

ref_row = get_row(MONITOR_NAME_MAP[REFERENCE_MONITOR])
eps_reconstructed = {}

# Real operational reference emittances (NOT the MAD-X beam-block design
# value, which is a much larger acceptance-like number -- see ADE-Status
# slides + colleague's note).
#   - 1.3 um: what the horizontal plane was originally aimed for
#   - 2.8 um: what was actually measured in 2025 high-intensity running
#     (consistent with: eps_actual = eps_reference * geometric_mismatch^2,
#      e.g. 1.3um * 1.48^2 = 2.8um)
CANDIDATE_EPS_UM = {
    "x": [1.3, 2.8],
    "y": [1.3, 2.8],  # no vertical-specific reference given -- same
                      # candidates used here; replace if you get a real one
}

print(f"=== Reconstruction vs. Xsuite model, at {REFERENCE_MONITOR} "
      f"({MONITOR_NAME_MAP[REFERENCE_MONITOR]}) ===\n")

for plane, eps_design in [("x", epsx_design), ("y", epsy_design)]:
    eps, beta0, alpha0 = reconstruct_beam_matrix(plane)
    eps_reconstructed[plane] = eps
    beta_model = ref_row["betx" if plane == "x" else "bety"]
    alpha_model = ref_row["alfx" if plane == "x" else "alfy"]

    M = twiss_mismatch(beta_model, alpha_model, beta0, alpha0)
    geometric_mismatch = np.sqrt(M)  # confirmed: geometric = sqrt(Twiss mismatch)

    print(f"--- {plane.upper()} plane ---")
    print(f"  Reconstructed from measurement: eps={eps*1e6:.3f} um, "
          f"beta={beta0:.3f} m, alpha={alpha0:.3f}")
    print(f"  Xsuite model at reference:      eps={eps_design*1e6:.3f} um "
          f"(design), beta={beta_model:.3f} m, alpha={alpha_model:.3f}")
    print(f"  Twiss mismatch factor: {M:.3f}  |  "
          f"Geometric mismatch: {geometric_mismatch:.3f}")

    # Cross-check: eps_actual = eps_reference * geometric_mismatch^2 (= M)
    for eps_ref_um in CANDIDATE_EPS_UM[plane]:
        predicted = eps_ref_um * M
        print(f"    check: {eps_ref_um} um * mismatch({M:.3f}) = "
              f"{predicted:.3f} um  (reconstructed: {eps*1e6:.3f} um)")
    print()

# ==========================================================================
# Step 4: full-line optics plot + monitor overlay, and sigma comparison
# ==========================================================================
fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=False)

# --- (a) beta / dispersion along the WHOLE line ---
axs[0].plot(tw_df["s"], tw_df["betx"], color="tab:red", lw=1, label=r"$\beta_x$")
axs[0].plot(tw_df["s"], tw_df["bety"], color="tab:green", lw=1, label=r"$\beta_y$")
ax0b = axs[0].twinx()
ax0b.plot(tw_df["s"], tw_df["dx"], color="tab:blue", lw=1, alpha=0.6, label=r"$D_x$")
ax0b.plot(tw_df["s"], tw_df["dy"], color="tab:orange", lw=1, alpha=0.6, label=r"$D_y$")
for real_name, model_name in MONITOR_NAME_MAP.items():
    s_mon = get_row(model_name)["s"]
    axs[0].axvline(s_mon, color="gray", ls=":", lw=0.8)
axs[0].set_ylabel(r"$\beta$ [m]")
ax0b.set_ylabel("D [m]")
axs[0].set_title("Full-line Xsuite optics (LNE00+LNE01+LNE02) with monitor positions")
axs[0].legend(loc="upper left", fontsize=9)
ax0b.legend(loc="upper right", fontsize=9)
axs[0].grid(alpha=0.3)

# --- (b)/(c) sigma comparison per plane: measured vs model (multiple eps assumptions) ---
for ax, plane in zip([axs[1], axs[2]], ["x", "y"]):
    beta_key = "betx" if plane == "x" else "bety"
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    eps_recon = eps_reconstructed[plane]

    names, s_pos, meas_mm = [], [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        row = get_row(model_name)
        names.append(real_name)
        s_pos.append(row["s"])
        meas_mm.append(measured_data[real_name][sig_key])

    beta_vals = np.clip(tw_df[beta_key], 0, None)

    # reconstructed-emittance curve (isolates optics-shape agreement)
    sim_curve_recon = np.sqrt(beta_vals * eps_recon) * 1e3
    ax.plot(tw_df["s"], sim_curve_recon, color="tab:purple", ls="--", lw=1.2,
            alpha=0.9, label=f"Model (reconstructed eps={eps_recon*1e6:.2f} um)")

    # candidate reference-emittance curves (1.3 / 2.8 um etc.)
    colors = ["tab:red", "tab:orange", "tab:brown"]
    for eps_um, color in zip(CANDIDATE_EPS_UM[plane], colors):
        eps_val = eps_um * 1e-6
        sim_curve = np.sqrt(beta_vals * eps_val) * 1e3
        ax.plot(tw_df["s"], sim_curve, color=color, ls=":", lw=1.2, alpha=0.8,
                label=f"Model (eps={eps_um} um)")

    ax.plot(s_pos, meas_mm, "o", color="tab:blue", ms=8, label="Measured", zorder=5)
    for xm in s_pos:
        ax.axvline(xm, color="gray", ls=":", lw=0.6)

    ax.set_ylabel(r"$\sigma$ [mm]")
    ax.set_title(f"{plane.upper()} plane: measured vs. model (multiple eps assumptions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axs[-1].set_xlabel("s [m]")
fig.tight_layout()
fig.savefig("xsuite_full_twiss_vs_measurement.png", dpi=150)
print("Saved plot to xsuite_full_twiss_vs_measurement.png")
plt.show()
