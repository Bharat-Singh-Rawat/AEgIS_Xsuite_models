"""
Compare the full LNE00+LNE01+LNE02 MAD-X model against the real
"Emittance Measurement using 7 monitors" result from the Beam Profile
Viewer.

Reuses the exact MAD-X setup from lne02_only_optics_madx.py, but does NOT
slice to the LNE02 sub-section, so that the LNE00/LNE01 monitors used in
the real measurement are available in the twiss table.

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line
"""

import os
import re
import numpy as np
import pandas as pd
from cpymad.madx import Madx

MADX_FILE = "general_lne02.madx"
SEQ_NAME  = "lne00lne01lne02"
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

# ==========================================================================
# Step 0: identical setup to lne02_only_optics_madx.py
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
tw = madx.twiss(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
).dframe()
tw.columns = [c.lower() for c in tw.columns]
tw["name_base"] = tw["name"].str.split(":").str[0].str.lower()

beam = madx.sequence[SEQ_NAME].beam
epsx_geom = beam.exn / (g["beta"] * g["gamman"])
epsy_geom = beam.eyn / (g["beta"] * g["gamman"])

print(f"Model design emittances: epsx={epsx_geom*1e6:.4f} um, "
      f"epsy={epsy_geom*1e6:.4f} um\n")

# ==========================================================================
# Step 1: find candidate BSGW-like elements in the FULL model so you can
# confirm the mapping to the real monitor names before trusting anything.
# ==========================================================================
print("=== BSGW-like elements found in the full model (name, s [m]) ===")
bsgw_like = tw[tw["name_base"].str.contains("bsgw")][["name_base", "s"]]
print(bsgw_like.to_string(index=False))
print()
print("Real monitors used in the measurement: LNE00.BSGW.0008, .0015, "
      ".0025, .0038, .0045, LNE01.BSGW.0109, .0120")
print("^ Confirm/fix the mapping in MONITOR_NAME_MAP below before trusting "
      "the comparison.\n")

# ---- EDIT THIS once you've confirmed the correspondence above ----
# real monitor name -> MAD-X model element name (lowercase, no ":n" suffix)
MONITOR_NAME_MAP = {
    "LNE00.BSGW.0008": "lne.bsgwa.0008",
    "LNE00.BSGW.0015": "lne.bsgwa.0015",
    "LNE00.BSGW.0025": "lne.bsgwa.0025",
    "LNE00.BSGW.0038": "lne.bsgwa.0038",
    "LNE00.BSGW.0045": "lne.bsgwa.0045",
    "LNE01.BSGW.0109": "lne.bsgwa.0109",
    "LNE01.BSGW.0120": "lne.bsgwa.0120",
}

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

REFERENCE_MONITOR = "LNE00.BSGW.0008"  # reference point the emittance is quoted at

if any(v == "REPLACE_ME" for v in MONITOR_NAME_MAP.values()):
    print("!! MONITOR_NAME_MAP is not filled in yet -- stopping here.")
    print("   Look at the printed BSGW-like element list above, match each "
          "to the real monitor it corresponds to, fill in the map, then "
          "re-run.")
    raise SystemExit(0)

# ==========================================================================
# Step 2: analytic R-matrix between two points from MAD-X twiss
# (mux/muy in MAD-X are in units of 2*pi, i.e. actual phase = 2*pi*mu)
# ==========================================================================
def get_row(model_name):
    row = tw.loc[tw["name_base"] == model_name.lower()]
    if row.empty:
        raise ValueError(f"Element '{model_name}' not found in twiss table")
    return row.iloc[0]

def r_matrix(name_i, name_j, plane):
    ri, rj = get_row(name_i), get_row(name_j)
    beta_key, alpha_key, mu_key = {
        "x": ("betx", "alfx", "mux"),
        "y": ("bety", "alfy", "muy"),
    }[plane]
    bi, ai, mi = ri[beta_key], ri[alpha_key], ri[mu_key]
    bj = rj[beta_key]
    dmu = 2 * np.pi * (rj[mu_key] - mi)
    R11 = np.sqrt(bj / bi) * (np.cos(dmu) + ai * np.sin(dmu))
    R12 = np.sqrt(bi * bj) * np.sin(dmu)
    return R11, R12

# ==========================================================================
# Step 3: reconstruct (eps, beta, alpha) at REFERENCE_MONITOR from the
# measured sigmas, using the model's own R-matrices -- same method as the
# real "7 monitor emittance measurement" tool.
# ==========================================================================
def reconstruct_beam_matrix(plane):
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    A_rows, b_vals = [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        R11, R12 = r_matrix(MONITOR_NAME_MAP[REFERENCE_MONITOR], model_name, plane)
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
    """1.0 = perfect match."""
    gamma_a = (1 + alpha_a**2) / beta_a
    gamma_b = (1 + alpha_b**2) / beta_b
    return 0.5 * (beta_a * gamma_b - 2 * alpha_a * alpha_b + gamma_a * beta_b)

print("=== Reconstruction vs. model design, at reference monitor "
      f"{REFERENCE_MONITOR} ({MONITOR_NAME_MAP[REFERENCE_MONITOR]}) ===\n")

ref_row = get_row(MONITOR_NAME_MAP[REFERENCE_MONITOR])
for plane, eps_design in [("x", epsx_geom), ("y", epsy_geom)]:
    eps, beta0, alpha0 = reconstruct_beam_matrix(plane)
    beta_model = ref_row["betx" if plane == "x" else "bety"]
    alpha_model = ref_row["alfx" if plane == "x" else "alfy"]

    M = twiss_mismatch(beta_model, alpha_model, beta0, alpha0)
    growth = M + np.sqrt(max(M**2 - 1, 0.0))

    print(f"--- {plane.upper()} plane ---")
    print(f"  Reconstructed from measurement: eps={eps*1e6:.3f} um, "
          f"beta={beta0:.3f} m, alpha={alpha0:.3f}")
    print(f"  Model design at reference:      eps={eps_design*1e6:.3f} um, "
          f"beta={beta_model:.3f} m, alpha={alpha_model:.3f}")
    print(f"  Twiss mismatch factor: {M:.3f} "
          f"(emittance growth factor: {growth:.3f})\n")

# ==========================================================================
# Step 4: monitor-by-monitor sigma comparison (model design eps vs measured)
# ==========================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, plane, eps_design in zip(axes, ["x", "y"], [epsx_geom, epsy_geom]):
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    beta_key = "betx" if plane == "x" else "bety"

    names, sim_mm, meas_mm = [], [], []
    for real_name, model_name in MONITOR_NAME_MAP.items():
        row = get_row(model_name)
        names.append(real_name)
        sim_mm.append(np.sqrt(row[beta_key] * eps_design) * 1e3)
        meas_mm.append(measured_data[real_name][sig_key])

    x_pos = np.arange(len(names))
    ax.plot(x_pos, meas_mm, "o-", label="Measured", color="tab:blue")
    ax.plot(x_pos, sim_mm, "s--", label="Model (design eps)", color="tab:red")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("sigma [mm]")
    ax.set_title(f"{plane.upper()} plane: measured vs. model")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig("sim_vs_measurement_sigma.png", dpi=150)
print("Saved comparison plot to sim_vs_measurement_sigma.png")
plt.show()
