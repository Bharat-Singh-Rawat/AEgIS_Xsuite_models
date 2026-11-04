"""
Track 5000 macroparticles through LNE02, plot their beam x-y profile at
several locations along the line, and derive beta functions (and
emittance) from the tracked distribution's covariance matrix -- then
compare that derived beta against MAD-X's simple/analytic Twiss.

IMPORTANT NOTE ON "PTC": this uses MAD-X's native TRACK module rather
than ptc_track. Earlier investigation in this project found that PTC's
layout builder cannot reliably represent this line's electrostatic
deflectors, which are genuine MATRIX-type elements (an arbitrary,
user-specified linear R-matrix) -- PTC has no element type for a
generic linear map, only physical models (drift/bend/quad/...). Native
MAD-X TRACK evaluates a MATRIX element's R-matrix directly and
deterministically, with no such limitation, and was the working
substitute established earlier in this project. Everything else about
the workflow (particle generation, multi-location profiling,
distribution-based Twiss) is the same as a PTC-based approach would be.

Also uses the CORRECTED model established earlier: the deflector
elements are rebound to their real MATRIX type (undoing a later
SBEND-approximation redefinition that a shared survey.seq file
introduces for its own geometric-survey purposes), so the beam optics
here are accurate.

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line

Requirements:
    pip install cpymad pandas matplotlib numpy scipy pymadx \
        --break-system-packages
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from cpymad.madx import Madx

# =========================================================================
# Config
# =========================================================================
MADX_FILE  = "general_lne02.madx"
SEQ_NAME   = "lne00lne01lne02"
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

N_PARTICLES  = 20000
N_OBS_POINTS = 8          # incl. start and delivery point (LNE02 end)
SIGMA_T, SIGMA_PT = 1e-12, 1e-12   # DC line, no RF -> ~zero momentum spread

OUT_PNG_PROFILE = "track_beam_profiles_lne02.png"
OUT_PNG_BETA    = "track_beta_vs_twiss_lne02.png"

# =========================================================================
# PART A -- run MAD-X, restore the correct MATRIX deflector bindings
# =========================================================================
def make_runnable_copy(src, dst):
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as f:
        for line in lines:
            if line.strip().lower().startswith("stop"):
                continue
            f.write(line)

tmp_madx_file = "general_lne02_nostop.madx"
make_runnable_copy(MADX_FILE, tmp_madx_file)

madx = Madx(stdout=True)
madx.call(file=tmp_madx_file)

print("\n[setup] restoring correct MATRIX deflector bindings...")
os.system("ln -fns ./../../lne lne_repo")
madx.call(file="lne_repo/deflectors.ele")

madx.command.use(sequence=SEQ_NAME)
for inst_name in KNOWN_DEFLECTOR_INSTANCES:
    save_file = "setup_lookup.madx"
    madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
    with open(save_file) as f:
        content = f.read()
    m = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content,
                  re.IGNORECASE)
    if m:
        madx.input(f"{inst_name}: {m.group(1)};")

madx.command.use(sequence=SEQ_NAME)
tw_madx = madx.twiss(beta0="initbeta0").dframe()
tw_madx.columns = [c.lower() for c in tw_madx.columns]

check = tw_madx[tw_madx["name"].str.split(":").str[0].isin(KNOWN_DEFLECTOR_INSTANCES)]
if (check["keyword"].str.lower() == "matrix").all():
    print("[setup] CONFIRMED: deflector elements are MATRIX type.")
else:
    print("[setup][WARNING] at least one deflector element is not MATRIX "
          "type -- twiss below may not be fully accurate. See "
          "compare_madx_xsuite_corrected.py for the full fix.")

g = madx.globals
beta_rel = g["beta"]
gamma_rel = g["gamman"]
beam = madx.sequence[SEQ_NAME].beam
epsx_geom = beam.exn / (beta_rel * gamma_rel)
epsy_geom = beam.eyn / (beta_rel * gamma_rel)
print(f"[beam] geometric emittances: epsx={epsx_geom:.4e} m, "
      f"epsy={epsy_geom:.4e} m")

# =========================================================================
# PART B -- generate the initial 5000-particle bunch (pymadx, proven method)
# =========================================================================
import pymadx

gauss_gen = pymadx.Ptc.GaussGenerator(
    gemx=epsx_geom, betax=g["betx0"], alfx=g["alfx0"],
    gemy=epsy_geom, betay=g["bety0"], alfy=g["alfy0"],
    sigmat=SIGMA_T, sigmapt=SIGMA_PT,
)
gauss_gen.Generate(nToGenerate=N_PARTICLES, fileName="inrays.madx")
print(f"[bunch] wrote inrays.madx with {N_PARTICLES} particles")

# =========================================================================
# PART C -- native TRACK through the whole line (see module docstring for
# why this is used instead of ptc_track)
# =========================================================================
if os.path.exists("trackone"):
    os.remove("trackone")

madx.command.use(sequence=SEQ_NAME)

# N_OBS_POINTS evenly-spaced observation locations, using real element
# names (with their ":occurrence" suffix, needed since many names like
# "drift_N" repeat in this line) from the analytic twiss table.
obs_target_s = np.linspace(tw_madx["s"].iloc[0], tw_madx["s"].iloc[-1],
                            N_OBS_POINTS)
tw_madx_s = tw_madx["s"].to_numpy()
tw_madx_names = tw_madx["name"].to_numpy()
obs_names, obs_place, obs_s = [], [], []
for ts in obs_target_s:
    pos = np.abs(tw_madx_s - ts).argmin()
    full_name = str(tw_madx_names[pos])
    obs_place.append(full_name)
    obs_names.append(full_name.split(":")[0])
    obs_s.append(tw_madx_s[pos])

print(f"\n[track] adding {len(obs_place)} OBSERVE points:")
for place, nm, sv in zip(obs_place, obs_names, obs_s):
    print(f"    {place:25s}  (shown as '{nm}')  s = {sv:8.4f} m")
    madx.command.observe(place=place)

with open("inrays.madx") as f:
    inrays_lines = f.readlines()
start_lines = [
    "start" + ln.strip()[len("ptc_start"):]
    for ln in inrays_lines if ln.strip().lower().startswith("ptc_start")
]
print(f"[bunch] {len(start_lines)} particle start lines")

madx.command.track(onepass=True, onetable=True, dump=True, aperture=False)
madx.input("\n".join(start_lines))
madx.command.run(turns=1)
madx.command.endtrack()
madx.input('write, table=trackone, file="trackone";')

# =========================================================================
# PART D -- parse trackone (segment-block algorithm: each "#..." header
# line delimits one observation point's data, unambiguous by construction)
# =========================================================================
def parse_trackone_segments(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    data_ranges = {}
    current_header, start_line = None, None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if current_header is not None:
                data_ranges[current_header] = (start_line, i)
            current_header = line.strip()
            start_line = i + 1
    if current_header is not None:
        data_ranges[current_header] = (start_line, len(lines))

    col_names = ["number", "turn", "x", "px", "y", "py", "t", "pt", "s", "e"]
    segments = []
    for header, (start, end) in data_ranges.items():
        rows = []
        for line in lines[start:end]:
            s = line.strip()
            if s == "" or s.startswith(("@", "*", "$")):
                continue
            try:
                rows.append([float(v) for v in s.split()])
            except ValueError:
                continue
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])
        segments.append((header, df))
    return segments

if not os.path.exists("trackone"):
    raise RuntimeError("No 'trackone' file found after tracking -- check "
                        "the MAD-X console output above for errors.")
segments = parse_trackone_segments("trackone")
print(f"\n[track] trackone: {len(segments)} segments (observation-point "
      "blocks)")

seg_s = np.array([df["s"].median() if len(df) else np.nan
                   for _, df in segments])
trk_by_point = []
for name, target_s in zip(obs_names, obs_s):
    j = int(np.nanargmin(np.abs(seg_s - target_s)))
    trk_by_point.append((name, target_s, segments[j][1]))

print("[track] particles recorded per observation point:")
for name, target_s, df in trk_by_point:
    print(f"    {name:25s}  s = {target_s:8.4f} m   n_particles = {len(df)}")

# =========================================================================
# PART E -- derive beta/alpha/emittance from the distribution; dispersion
# taken from the analytic MAD-X twiss (sigmapt~0 -> nothing meaningful to
# regress it from the tracked distribution)
# =========================================================================
def twiss_from_distribution(df, dx=0.0, dpx=0.0, dy=0.0, dpy=0.0):
    x, px = df["x"].to_numpy(), df["px"].to_numpy()
    y, py = df["y"].to_numpy(), df["py"].to_numpy()
    Sx, Sy = np.cov(x, px), np.cov(y, py)
    epsx = np.sqrt(max(np.linalg.det(Sx), 0.0))
    epsy = np.sqrt(max(np.linalg.det(Sy), 0.0))
    betx = Sx[0, 0] / epsx if epsx > 0 else np.nan
    alfx = -Sx[0, 1] / epsx if epsx > 0 else np.nan
    bety = Sy[0, 0] / epsy if epsy > 0 else np.nan
    alfy = -Sy[0, 1] / epsy if epsy > 0 else np.nan
    return dict(betx=betx, alfx=alfx, bety=bety, alfy=alfy,
                dx=dx, dpx=dpx, dy=dy, dpy=dpy, epsx=epsx, epsy=epsy)

results = []
for name, s_val, sub in trk_by_point:
    pos_madx = np.abs(tw_madx_s - s_val).argmin()
    dx_a  = tw_madx["dx"].iloc[pos_madx]
    dpx_a = tw_madx["dpx"].iloc[pos_madx]
    dy_a  = tw_madx["dy"].iloc[pos_madx]
    dpy_a = tw_madx["dpy"].iloc[pos_madx]
    tw = twiss_from_distribution(sub, dx=dx_a, dpx=dpx_a, dy=dy_a, dpy=dpy_a)
    tw.update(name=name, s=s_val, n_particles=len(sub))
    results.append(tw)
tw_track = pd.DataFrame(results)
print("\n[distribution-derived Twiss]")
print(tw_track[["name", "s", "betx", "bety", "epsx", "epsy"]]
      .to_string(index=False))

# =========================================================================
# PLOT 1 -- beam x-y profile at each location, KDE density-colored, fixed
# axis scale (taken from the LAST/delivery-point location, since the beam
# is largest there)
# =========================================================================
ref_x_mm = (trk_by_point[-1][2]["x"] * 1e3).to_numpy()
ref_y_mm = (trk_by_point[-1][2]["y"] * 1e3).to_numpy()
margin = 1.15
xlim = (-margin * np.max(np.abs(ref_x_mm)), margin * np.max(np.abs(ref_x_mm)))
ylim = (-margin * np.max(np.abs(ref_y_mm)), margin * np.max(np.abs(ref_y_mm)))

ncols = min(N_OBS_POINTS, 4)
nrows = int(np.ceil(N_OBS_POINTS / ncols))
fig1, axs1 = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
axs1 = np.atleast_1d(axs1).flatten()

for idx, (name, s_val, sub) in enumerate(trk_by_point):
    ax = axs1[idx]
    x_mm = (sub["x"] * 1e3).to_numpy()
    y_mm = (sub["y"] * 1e3).to_numpy()
    if len(x_mm) >= 3:
        xy = np.vstack([x_mm, y_mm])
        try:
            dens = gaussian_kde(xy)(xy)
        except np.linalg.LinAlgError:
            dens = np.ones(len(x_mm))
        sc = ax.scatter(x_mm, y_mm, c=dens, cmap="jet", s=4, alpha=0.7)
        fig1.colorbar(sc, ax=ax).set_label("density", fontsize=9)
    else:
        ax.scatter(x_mm, y_mm, s=4)

    row = tw_track.iloc[idx]
    ax.set_title(f"{name}\ns={s_val:.2f} m  (n={len(sub)})\n"
                 rf"$\epsilon_x$={row.epsx*1e6:.2f} mm$\cdot$mrad, "
                 rf"$\epsilon_y$={row.epsy*1e6:.2f} mm$\cdot$mrad")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(alpha=0.3)

for j in range(len(trk_by_point), len(axs1)):
    axs1[j].axis("off")

fig1.suptitle(f"LNE02: beam x-y profile from TRACK tracking "
              f"(N={N_PARTICLES} macroparticles)")
fig1.tight_layout()
fig1.savefig(OUT_PNG_PROFILE, dpi=200)

# =========================================================================
# PLOT 2 -- distribution-derived beta vs analytic MAD-X twiss
# =========================================================================
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs2[0].plot(tw_madx["s"], tw_madx["betx"], color="tab:blue", lw=2,
             label=r"$\beta_x$ MAD-X (analytic)")
axs2[0].plot(tw_madx["s"], tw_madx["bety"], color="tab:orange", lw=2,
             label=r"$\beta_y$ MAD-X (analytic)")
axs2[0].plot(tw_track["s"], tw_track["betx"], "o", ms=9, color="tab:green",
             mec="k", label=r"$\beta_x$ from tracked distribution")
axs2[0].plot(tw_track["s"], tw_track["bety"], "s", ms=9, color="tab:red",
             mec="k", label=r"$\beta_y$ from tracked distribution")
axs2[0].set_ylabel(r"$\beta$ [m]")
axs2[0].legend(ncol=2, fontsize=9)
axs2[0].grid(alpha=0.3)

axs2[1].plot(tw_madx["s"], tw_madx["dx"], color="tab:blue", lw=2,
             label=r"$D_x$ MAD-X (analytic)")
axs2[1].plot(tw_madx["s"], tw_madx["dy"], color="tab:orange", lw=2,
             label=r"$D_y$ MAD-X (analytic)")
axs2[1].plot(tw_track["s"], tw_track["dx"], "o", ms=9, color="tab:green",
             mec="k", label=r"$D_x$ (analytic, at track obs. points)")
axs2[1].plot(tw_track["s"], tw_track["dy"], "s", ms=9, color="tab:red",
             mec="k", label=r"$D_y$ (analytic, at track obs. points)")
axs2[1].set_ylabel(r"$D$ [m]"); axs2[1].set_xlabel("s [m]")
axs2[1].legend(ncol=2, fontsize=9)
axs2[1].grid(alpha=0.3)

fig2.suptitle("LNE02: beta function derived from tracked distribution "
              "vs analytic MAD-X twiss")
fig2.tight_layout()
fig2.savefig(OUT_PNG_BETA, dpi=200)

plt.show()

# Quantify agreement
common = tw_track.copy()
common["betx_madx"] = [
    tw_madx["betx"].iloc[np.abs(tw_madx_s - s).argmin()] for s in common["s"]
]
common["bety_madx"] = [
    tw_madx["bety"].iloc[np.abs(tw_madx_s - s).argmin()] for s in common["s"]
]
print("\n[result] beta_x agreement (tracked distribution vs analytic MAD-X):")
print((common[["name", "s", "betx", "betx_madx"]]
       .assign(rel_diff_pct=lambda d: 100 * np.abs(d.betx - d.betx_madx) / d.betx_madx))
      .to_string(index=False))

print(f"\nSaved:\n    {OUT_PNG_PROFILE}\n    {OUT_PNG_BETA}")
