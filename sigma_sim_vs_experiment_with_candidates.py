"""
Same as beta_sim_vs_coefficient_table.py, PLUS: overlays the parquet
BEAM_sigmaX/BEAM_sigmaY data (median + 25th-75th percentile spread across
all AEGIS cycles) on the sigma sim-vs-experiment plot, at the two candidate
model locations for that unresolved monitor (lne.bsgwa.0207 and
lne.bsgwa.0225). This is illustrative ONLY -- neither candidate has been
confirmed as the real location; showing both lets you visually judge which
one (if either) is more plausible against the model's predicted sigma curve.

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
# Global plot style
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
PARQUET_FILE = "PBAR_Intensity_year.parquet"  # <-- update path if needed

N_MC_SAMPLES = 200
JITTER_FRAC = 0.10
RNG_SEED = 42

# ==========================================================================
# Step 0: MAD-X setup (same as lne02_only_optics_madx.py)
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

print("=== Initial Twiss conditions used for the simulation (from "
      f"{MADX_FILE} globals) ===")
for k, v in INIT.items():
    print(f"  {k}0 = {v}")
print()

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
# Step 2: monitor mapping + real coefficient-table values (Bh=betax, Bv=betay)
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

COEFFICIENT_TABLE = {
    "LNE00.BSGW.0008": {"Bh": 1.46792, "Bv": 5.35866},
    "LNE00.BSGW.0015": {"Bh": 0.816672, "Bv": 5.93854},
    "LNE00.BSGW.0025": {"Bh": 4.34105, "Bv": 1.15042},
    "LNE00.BSGW.0038": {"Bh": 4.34116, "Bv": 1.14348},
    "LNE00.BSGW.0045": {"Bh": 1.57516, "Bv": 3.3613},
    "LNE01.BSGW.0109": {"Bh": 1.77826, "Bv": 3.99578},
    "LNE01.BSGW.0120": {"Bh": 2.2294, "Bv": 2.23271},
}

names, betx_table, bety_table, table_s = [], [], [], []
for real_name, model_name in MONITOR_NAME_MAP.items():
    row = get_row(model_name)
    names.append(real_name)
    table_s.append(row["s"])
    betx_table.append(COEFFICIENT_TABLE[real_name]["Bh"])
    bety_table.append(COEFFICIENT_TABLE[real_name]["Bv"])

# ==========================================================================
# Step 3: Monte Carlo uncertainty analysis -- jitter every quadrupole k1 by
# +/-10%, recompute twiss, collect betx/bety along the line.
# ==========================================================================
rng = np.random.default_rng(RNG_SEED)

quad_names = [
    name for name in line.element_names
    if hasattr(line.element_dict[name], "k1") and line.element_dict[name].k1 != 0
]
nominal_k1 = {name: line.element_dict[name].k1 for name in quad_names}
print(f"Jittering {len(quad_names)} quadrupoles by +/-{JITTER_FRAC*100:.0f}% "
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

for name in quad_names:
    line.element_dict[name].k1 = nominal_k1[name]

betx_lo, betx_hi = np.percentile(betx_samples, [5, 95], axis=0)
bety_lo, bety_hi = np.percentile(bety_samples, [5, 95], axis=0)
print("Monte Carlo done.\n")

# ==========================================================================
# Step 4: Plot 1 -- beta_x/beta_y sim vs. coefficient table (unchanged)
# ==========================================================================
fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(s_grid, betx_lo, betx_hi, color="tab:red", alpha=0.2,
                 label=r"$\beta_x$ 5-95% band ($\pm$10% quad jitter)")
ax.fill_between(s_grid, bety_lo, bety_hi, color="tab:green", alpha=0.2,
                 label=r"$\beta_y$ 5-95% band ($\pm$10% quad jitter)")
ax.plot(s_grid, tw_df["betx"], color="tab:red", label=r"$\beta_x$ nominal")
ax.plot(s_grid, tw_df["bety"], color="tab:green", label=r"$\beta_y$ nominal")

ax.plot(table_s, betx_table, "o", color="tab:blue", ms=12,
        label=r"$\beta_x$ (coefficient table, real)", zorder=5)
ax.plot(table_s, bety_table, "D", color="tab:purple", ms=12,
        label=r"$\beta_y$ (coefficient table, real)", zorder=5)

for xm in table_s:
    ax.axvline(xm, color="gray", ls=":", lw=1.5)

ax.set_xlabel("s [m]", fontweight="bold")
ax.set_ylabel(r"$\beta$ [m]", fontweight="bold")
ax.set_title("beta_x/beta_y sim (10% quad jitter) vs. real "
             "coefficient-table values", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("beta_sim_vs_coefficient_table.png", dpi=150)
print("Saved plot to beta_sim_vs_coefficient_table.png")

# ==========================================================================
# Step 5: emittance reconstruction from the 7-monitor measurement (unchanged)
# ==========================================================================
measured_data = {
    "LNE00.BSGW.0008": {"sigma_x": 1.1, "sigma_y": 1.9},
    "LNE00.BSGW.0015": {"sigma_x": 1.5, "sigma_y": 2.1},
    "LNE00.BSGW.0025": {"sigma_x": 3.0, "sigma_y": 1.3},
    "LNE00.BSGW.0038": {"sigma_x": 2.5, "sigma_y": 1.0},
    "LNE00.BSGW.0045": {"sigma_x": 1.8, "sigma_y": 1.6},
    "LNE01.BSGW.0109": {"sigma_x": 1.4, "sigma_y": 2.2},
    "LNE01.BSGW.0120": {"sigma_x": 2.4, "sigma_y": 1.1},
}
REFERENCE_MONITOR = "LNE00.BSGW.0008"

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

# ==========================================================================
# Step 6: candidate parquet monitor data -- UNRESOLVED, illustrative only.
# Loads BEAM_sigmaX/BEAM_sigmaY across all AEGIS cycles, takes the median
# and 25th-75th percentile spread, and plots that same measured value at
# BOTH candidate model locations (lne.bsgwa.0207 and lne.bsgwa.0225) so you
# can visually compare against the model curve at each candidate position.
# Neither candidate is confirmed -- see prior discussion.
# ==========================================================================
CANDIDATE_LOCATIONS = ["lne.bsgwa.0207", "lne.bsgwa.0225"]

candidate_data = None
try:
    df = pd.read_parquet(PARQUET_FILE)
    aegis = df.loc[df["is_beam_in_AEGIS"]].dropna(
        subset=["BEAM_sigmaX", "BEAM_sigmaY"]).copy()

    sigx_med, sigx_lo, sigx_hi = aegis["BEAM_sigmaX"].quantile([0.5, 0.25, 0.75])
    sigy_med, sigy_lo, sigy_hi = aegis["BEAM_sigmaY"].quantile([0.5, 0.25, 0.75])

    print(f"\nParquet BEAM_sigmaX: median={sigx_med:.3f} mm "
          f"(25-75th pct: {sigx_lo:.3f}-{sigx_hi:.3f})")
    print(f"Parquet BEAM_sigmaY: median={sigy_med:.3f} mm "
          f"(25-75th pct: {sigy_lo:.3f}-{sigy_hi:.3f})")

    candidate_data = dict(sigx_med=sigx_med, sigx_err=[[sigx_med-sigx_lo], [sigx_hi-sigx_med]],
                           sigy_med=sigy_med, sigy_err=[[sigy_med-sigy_lo], [sigy_hi-sigy_med]])
except FileNotFoundError:
    print(f"\n[skipped] Could not find {PARQUET_FILE} -- candidate overlay "
          "will not be shown.")

# ==========================================================================
# Step 7: Plot 2 -- sigma_x/sigma_y sim vs. 7-monitor experiment, PLUS the
# parquet candidate points overlaid at both candidate locations.
# ==========================================================================
fig2, ax2 = plt.subplots(figsize=(15, 8))

for plane, color_band, color_line, marker in [
        ("x", "tab:red", "tab:red", "o"), ("y", "tab:green", "tab:green", "D")]:
    beta_key = "betx" if plane == "x" else "bety"
    beta_samples = betx_samples if plane == "x" else bety_samples
    sig_key = "sigma_x" if plane == "x" else "sigma_y"
    eps = eps_reconstructed[plane]

    beta_vals = np.clip(tw_df[beta_key], 0, None)
    sigma_nom = np.sqrt(beta_vals * eps) * 1e3
    sigma_mc = np.sqrt(np.clip(beta_samples, 0, None) * eps) * 1e3
    sigma_lo, sigma_hi = np.percentile(sigma_mc, [5, 95], axis=0)

    ax2.fill_between(s_grid, sigma_lo, sigma_hi, color=color_band, alpha=0.2,
                      label=rf"$\sigma_{plane}$ 5-95% band ($\pm$10% quad jitter)")
    ax2.plot(s_grid, sigma_nom, color=color_line,
             label=rf"$\sigma_{plane}$ nominal (eps={eps*1e6:.2f} um)")

    meas_mm = [measured_data[n][sig_key] for n in names]
    ax2.plot(table_s, meas_mm, marker,
              color="tab:blue" if plane == "x" else "tab:purple",
              ms=12, label=rf"$\sigma_{plane}$ measured (7 monitors)", zorder=5)

for xm in table_s:
    ax2.axvline(xm, color="gray", ls=":", lw=1.5)

# overlay candidate parquet points at BOTH candidate locations
if candidate_data is not None:
    for cand_name in CANDIDATE_LOCATIONS:
        s_cand = get_row(cand_name)["s"]
        ax2.errorbar(s_cand, candidate_data["sigx_med"],
                     yerr=candidate_data["sigx_err"], fmt="^",
                     color="tab:orange", ecolor="tab:orange", elinewidth=3,
                     capsize=8, ms=16, zorder=6,
                     label=f"Parquet sigma_x candidate ({cand_name})")
        ax2.errorbar(s_cand, candidate_data["sigy_med"],
                     yerr=candidate_data["sigy_err"], fmt="v",
                     color="tab:brown", ecolor="tab:brown", elinewidth=3,
                     capsize=8, ms=16, zorder=6,
                     label=f"Parquet sigma_y candidate ({cand_name})")
        ax2.axvline(s_cand, color="black", ls="--", lw=2, alpha=0.6)

ax2.set_xlabel("s [m]", fontweight="bold")
ax2.set_ylabel(r"$\sigma$ [mm]", fontweight="bold")
ax2.set_title("sigma_x/sigma_y sim vs. experiment, with UNCONFIRMED "
              "parquet candidate locations overlaid", fontweight="bold")
ax2.legend(fontsize=10, loc="upper left")
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("sigma_sim_vs_experiment_with_candidates.png", dpi=150)
print("Saved plot to sigma_sim_vs_experiment_with_candidates.png")

plt.show()
