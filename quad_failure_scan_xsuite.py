"""
Single-quadrupole failure scan for LNE02 -- XSUITE VERSION.

Same analysis as quad_failure_scan.py, but built on the Xsuite line
(with the deflectors correctly patched to FirstOrderTaylorMap R-matrices,
exactly as established in compare_madx_xsuite_corrected.py) instead of
MAD-X. Turns OFF (or degrades) one quadrupole at a time, re-twisses with
the nominal injected beam conditions held fixed, and ranks quads by
their impact on the delivered beam at LNE02$END.

HOW QUAD STRENGTHS ARE CHANGED (read this if adapting the script):
-----------------------------------------------------------------------
The line is built with deferred_expressions=True, so some quad k1's may
already be tied to MAD-X knob variables and some may not. Rather than
assume either way, every quad we touch gets wrapped in a FRESH Xsuite
knob the first time we touch it:
    line.vars[f"k1_{quad_name}"] = <current k1>
    line.element_refs[quad_name].k1 = line.vars[f"k1_{quad_name}"]
After that, changing the quad's strength is just
    line.vars[f"k1_{quad_name}"] = new_value
and the element's actual k1 always reflects it immediately -- so reading
back float(line[quad_name].k1) always gives the live, correct value
regardless of whether a knob was already present in the MAD-X model.
-----------------------------------------------------------------------

Unlike the MAD-X/cpymad version, Xsuite runs entirely in-process (no
separate MAD-X subprocess to crash), so there's no need for the
"rebuild everything if it dies" resilience pattern used in the MAD-X
recovery script -- a bad twiss here just raises a normal Python
exception, which we catch per-quad and move on.

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line

Requirements:
    pip install cpymad xtrack pandas matplotlib numpy --break-system-packages
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
END_PLACE  = "LNE02$END"
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

FAILURE_FRACTION = 1.0     # 1.0 = quad fully off; 0.5 = 50% under-strength
TOP_N_HIGHLIGHT  = 8

OUT_PNG_RANKING = "quad_failure_ranking_xsuite.png"
OUT_PNG_TWISS   = "quad_failure_twiss_overlay_xsuite.png"
OUT_CSV         = "quad_failure_scan_results_xsuite.csv"

# =========================================================================
# PART A -- build the Xsuite line (same recipe as compare_madx_xsuite_corrected.py)
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
            print(f"    [skip] {inst_name}: not found in Xsuite line")
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
            print(f"    [FAILED] {inst_name}: could not construct FirstOrderTaylorMap")
            continue
        line.element_dict[target_name] = new_elem
        print(f"    [patched] {target_name} (found {n_found}/36 R values)")

    line.build_tracker()
    return madx, line, tw_madx

print("[setup] building Xsuite line...")
madx, line, tw_madx = build_xsuite_line()

g = madx.globals
beta_rel, gamma_rel = g["beta"], g["gamman"]
beam = madx.sequence[SEQ_NAME].beam
epsx_geom = beam.exn / (beta_rel * gamma_rel)
epsy_geom = beam.eyn / (beta_rel * gamma_rel)

init_kwargs = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)

# =========================================================================
# PART B -- knob-wrapping helper (see module docstring)
# =========================================================================
_knobbed = set()

def set_k1(line, name, value):
    knob = f"k1_{name}"
    if name not in _knobbed:
        current = float(line[name].k1)
        line.vars[knob] = current
        line.element_refs[name].k1 = line.vars[knob]
        _knobbed.add(name)
    line.vars[knob] = float(value)

def get_k1(line, name):
    return float(line[name].k1)

# =========================================================================
# PART C -- twiss / metric helpers
# =========================================================================
def do_twiss():
    return line.twiss(method="4d", start=xt.START, end=xt.END,
                       init=xt.TwissInit(**init_kwargs)).to_pandas()

def end_row(tw):
    row = tw[tw["name"].str.upper().str.startswith(END_PLACE.upper())]
    return row.iloc[0] if len(row) else tw.iloc[-1]

def beam_sizes(row):
    return np.sqrt(row["betx"] * epsx_geom), np.sqrt(row["bety"] * epsy_geom)

tw_nominal = do_twiss()
end_nominal = end_row(tw_nominal)
sigx_nom, sigy_nom = beam_sizes(end_nominal)
betxmax_nom, betymax_nom = tw_nominal["betx"].max(), tw_nominal["bety"].max()

print("[setup] nominal delivered beam at", END_PLACE, "(Xsuite):")
print(f"    betx_end={end_nominal['betx']:.4f} m  bety_end={end_nominal['bety']:.4f} m")
print(f"    sigx_end={sigx_nom*1e3:.4f} mm  sigy_end={sigy_nom*1e3:.4f} mm")

# =========================================================================
# PART D -- identify quadrupoles (documented Xsuite pattern: get_table + element_type)
# =========================================================================
tab = line.get_table()
quad_names = list(tab.rows[tab.element_type == "Quadrupole"].name)
print(f"\n[scan] {len(quad_names)} quadrupoles found in the Xsuite line")

s_lookup = tw_nominal.set_index("name")["s"]

# =========================================================================
# PART E -- fail each quad one at a time, re-twiss, restore
# =========================================================================
results = []
twiss_curves = {}

for name in quad_names:
    try:
        k1_orig = get_k1(line, name)
        s_pos = float(s_lookup.get(name, np.nan))
    except Exception as e:
        print(f"[skip] {name}: {e}")
        continue

    k1_failed = k1_orig * (1.0 - FAILURE_FRACTION)
    set_k1(line, name, k1_failed)

    try:
        tw = do_twiss()
        row = end_row(tw)
        sigx, sigy = beam_sizes(row)
        betxmax, betymax = tw["betx"].max(), tw["bety"].max()
        ok = True
    except Exception as e:
        print(f"[fail-to-twiss] {name}: {e}")
        tw, row = None, None
        sigx = sigy = betxmax = betymax = np.nan
        ok = False

    set_k1(line, name, k1_orig)  # restore regardless of outcome

    if not ok:
        continue

    rec = dict(
        name=name, s=s_pos, k1_nominal=k1_orig,
        betx_end=row["betx"], bety_end=row["bety"],
        dx_end=row["dx"], dy_end=row["dy"],
        sigx_end_mm=sigx * 1e3, sigy_end_mm=sigy * 1e3,
        betx_max=betxmax, bety_max=betymax,
        d_sigx_pct=100 * (sigx - sigx_nom) / sigx_nom,
        d_sigy_pct=100 * (sigy - sigy_nom) / sigy_nom,
        d_betxmax_pct=100 * (betxmax - betxmax_nom) / betxmax_nom,
        d_betymax_pct=100 * (betymax - betymax_nom) / betymax_nom,
    )
    rec["impact_score"] = max(abs(rec["d_sigx_pct"]), abs(rec["d_sigy_pct"]),
                               abs(rec["d_betxmax_pct"]), abs(rec["d_betymax_pct"]))
    results.append(rec)
    twiss_curves[name] = tw

df = pd.DataFrame(results).sort_values("impact_score", ascending=False).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"\n[scan] done, saved ranked results to {OUT_CSV}")
print(f"\n[ranking] top 10 most critical quads (failure_fraction={FAILURE_FRACTION}):")
print(df.head(10)[["name", "s", "impact_score", "d_sigx_pct", "d_sigy_pct",
                    "d_betxmax_pct", "d_betymax_pct"]].to_string(index=False))

# =========================================================================
# PLOT 1 -- criticality ranking bar chart
# =========================================================================
top = df.head(20).iloc[::-1]
fig1, ax1 = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
ax1.barh(top["name"], top["d_sigx_pct"], color="tab:blue", alpha=0.7, label=r"$\Delta\sigma_x$ [%]")
ax1.barh(top["name"], top["d_sigy_pct"], color="tab:orange", alpha=0.5, label=r"$\Delta\sigma_y$ [%]")
ax1.axvline(0, color="k", lw=0.8)
ax1.set_xlabel("relative change at delivery point [%]")
ax1.set_title(f"LNE02 (Xsuite): delivered beam-size sensitivity to single-quad "
              f"failure (failure_fraction={FAILURE_FRACTION})")
ax1.legend(); ax1.grid(alpha=0.3, axis="x")
fig1.tight_layout()
fig1.savefig(OUT_PNG_RANKING, dpi=200)

# =========================================================================
# PLOT 2 -- twiss overlay for the worst offenders
# =========================================================================
worst_names = df.head(TOP_N_HIGHLIGHT)["name"].tolist()
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
cmap = plt.cm.tab10(np.linspace(0, 1, len(worst_names)))

axs2[0].plot(tw_nominal["s"], tw_nominal["betx"], color="navy", lw=2.5, label=r"$\beta_x$ nominal")
axs2[0].plot(tw_nominal["s"], tw_nominal["bety"], color="darkorange", lw=2.5, label=r"$\beta_y$ nominal")
for name, color in zip(worst_names, cmap):
    tw = twiss_curves[name]
    axs2[0].plot(tw["s"], tw["betx"], color=color, lw=1.2, alpha=0.85, label=f"{name} failed")
    axs2[1].plot(tw["s"], tw["bety"], color=color, lw=1.2, alpha=0.85, label=f"{name} failed")
axs2[1].plot(tw_nominal["s"], tw_nominal["bety"], color="darkorange", lw=2.5, label=r"$\beta_y$ nominal")

axs2[0].set_ylabel(r"$\beta_x$ [m]"); axs2[0].legend(fontsize=7, ncol=2); axs2[0].grid(alpha=0.3)
axs2[1].set_ylabel(r"$\beta_y$ [m]"); axs2[1].set_xlabel("s [m]")
axs2[1].legend(fontsize=7, ncol=2); axs2[1].grid(alpha=0.3)

fig2.suptitle(f"LNE02 (Xsuite): optics with the {TOP_N_HIGHLIGHT} most critical "
              f"single-quad failures (failure_fraction={FAILURE_FRACTION})")
fig2.tight_layout()
fig2.savefig(OUT_PNG_TWISS, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_CSV}\n    {OUT_PNG_RANKING}\n    {OUT_PNG_TWISS}")
