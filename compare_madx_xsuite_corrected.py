"""
Twiss plot comparision of MAD-X vs Xsuite .

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
OUT_PNG   = "madx_vs_xsuite_corrected.png"

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

# =========================================================================

# =========================================================================
print("\n[fix] recreating lne_repo symlink and re-loading deflectors.ele "
      "to restore the correct MATRIX type declarations...")
os.system("ln -fns ./../../lne lne_repo")
madx.call(file="lne_repo/deflectors.ele")


madx.command.use(sequence=SEQ_NAME)
tw_check = madx.twiss(beta0="initbeta0").dframe()
tw_check.columns = [c.lower() for c in tw_check.columns]

KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]
for inst_name in KNOWN_DEFLECTOR_INSTANCES:
    # find its parent type name via SAVE (same method as before)
    save_file = "instance_parent_lookup.madx"
    madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
    with open(save_file) as f:
        content = f.read()
    m = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content,
                  re.IGNORECASE)
    if not m:
        print(f"    [skip] {inst_name}: could not find its declaration "
              "in SAVE output to determine parent type")
        continue
    parent_type = m.group(1)
    print(f"    {inst_name} -> parent type '{parent_type}'")
    madx.input(f"{inst_name}: {parent_type};")

madx.command.use(sequence=SEQ_NAME)
tw_madx = madx.twiss(beta0="initbeta0").dframe()
tw_madx.columns = [c.lower() for c in tw_madx.columns]

print("\n[fix] deflector elements after restoring MATRIX type AND "
      "re-declaring instances:")
check = tw_madx[tw_madx["name"].str.split(":").str[0].isin(KNOWN_DEFLECTOR_INSTANCES)]
print(check[["name", "keyword", "s", "l", "betx", "bety"]].to_string(index=False))
if (check["keyword"].str.lower() == "matrix").all():
    print("[fix] CONFIRMED: all known deflector elements are back to "
          "MATRIX type.")
else:
    print("[fix] WARNING: at least one element is still not MATRIX "
          "type -- there may be yet another redeclaration happening "
          "somewhere else in the file execution order. Investigate "
          "further before trusting the comparison below.")

# =========================================================================
# NOW build the Xsuite line 
# =========================================================================
line = xt.Line.from_madx_sequence(
    madx.sequence[SEQ_NAME], allow_thick=True, deferred_expressions=True,
)
beam = madx.sequence[SEQ_NAME].beam
line.particle_ref = xt.Particles(
    mass0=beam.mass * 1e9, q0=beam.charge, p0c=beam.pc * 1e9,
)
line.build_tracker()

g = madx.globals
init_kwargs = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)
tw_xt_before_patch = line.twiss(
    method="4d", start=xt.START, end=xt.END, init=xt.TwissInit(**init_kwargs),
).to_pandas()

# =========================================================================
# =========================================================================
print("\n[patch] extracting real R-matrices and patching Xsuite elements...")

def get_R_from_save(inst_name):
    save_file = "patch_lookup.madx"
    madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
    with open(save_file) as f:
        content = f.read()
    # instance line just gives the parent type name; the parent type's
    # OWN declaration (also in this SAVE output) has the real rm** values
    m_inst = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content,
                        re.IGNORECASE)
    if not m_inst:
        raise ValueError(f"instance {inst_name} not found in SAVE output")
    parent_type = m_inst.group(1)
    m_type = re.search(re.escape(parent_type) + r"\s*:\s*matrix\s*,(.*?);",
                        content, re.IGNORECASE | re.DOTALL)
    if not m_type:
        raise ValueError(f"parent type {parent_type} not found as a "
                          "MATRIX declaration in SAVE output")
    body = m_type.group(1)
    R = np.eye(6)
    n_found = 0
    for i in range(1, 7):
        for j in range(1, 7):
            mm = re.search(rf"\brm{i}{j}\s*=\s*([-+0-9.eEdD]+)", body,
                            re.IGNORECASE)
            if mm:
                val_str = mm.group(1).replace("D", "E").replace("d", "e")
                R[i - 1, j - 1] = float(val_str)
                n_found += 1
    return R, n_found, parent_type

for inst_name in KNOWN_DEFLECTOR_INSTANCES:
    try:
        R, n_found, parent_type = get_R_from_save(inst_name)
    except Exception as e:
        print(f"    [skip] {inst_name}: {e}")
        continue
    print(f"    {inst_name} (type {parent_type}): matched {n_found}/36 "
          f"R-matrix values, diag={np.round(np.diag(R), 4)}")

    length_row = tw_madx[tw_madx["name"].str.split(":").str[0] == inst_name]
    length = float(length_row.iloc[0]["l"]) if len(length_row) else 0.6

    target_name = None
    for cand in line.element_names:
        if cand.split(":")[0] == inst_name:
            target_name = cand
            break
    if target_name is None:
        print(f"    [skip] {inst_name}: not found in Xsuite line")
        continue

    new_elem = None
    last_err = None
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
        except (TypeError, NameError) as e:
            last_err = e
            continue

    if new_elem is None:
        print(f"    [FAILED] {inst_name}: could not construct "
              f"FirstOrderTaylorMap. Last error: {last_err}")
        continue

    line.element_dict[target_name] = new_elem
    print(f"    [patched] {target_name} -> xt.FirstOrderTaylorMap "
          f"(length={length:.4f} m)")

line.build_tracker()
tw_xt = line.twiss(
    method="4d", start=xt.START, end=xt.END, init=xt.TwissInit(**init_kwargs),
).to_pandas()

# =========================================================================
# Compare
# =========================================================================
common_n = min(len(tw_madx), len(tw_xt))
diff_betx_before = np.abs(tw_madx["betx"].to_numpy()[:common_n]
                           - tw_xt_before_patch["betx"].to_numpy()[:common_n])
diff_betx = np.abs(tw_madx["betx"].to_numpy()[:common_n]
                    - tw_xt["betx"].to_numpy()[:common_n])
print(f"\n[result] max |beta_x MAD-X - Xsuite| (model restored, "
      f"BEFORE R-matrix patch): {diff_betx_before.max():.4e} m")
print(f"[result] max |beta_x MAD-X - Xsuite| (model restored, "
      f"AFTER R-matrix patch):  {diff_betx.max():.4e} m")
print(f"[result] (for reference, this was ~150 m before restoring the "
      "correct MATRIX element bindings at all)")

fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
axs[0].plot(tw_madx["s"], tw_madx["betx"], color="tab:blue", lw=2.5,
            label=r"$\beta_x$ MAD-X")
axs[0].plot(tw_madx["s"], tw_madx["bety"], color="tab:orange", lw=2.5,
            label=r"$\beta_y$ MAD-X")
axs[0].plot(tw_xt["s"], tw_xt["betx"], "--", color="tab:green", lw=2,
            label=r"$\beta_x$ Xsuite")
axs[0].plot(tw_xt["s"], tw_xt["bety"], "--", color="tab:red", lw=2,
            label=r"$\beta_y$ Xsuite")
axs[0].set_ylabel(r"$\beta$ [m]")
axs[0].legend(ncol=2, fontsize=9)
axs[0].grid(alpha=0.3)

axs[1].plot(tw_madx["s"], tw_madx["dx"], color="tab:blue", lw=2.5,
            label=r"$D_x$ MAD-X")
axs[1].plot(tw_madx["s"], tw_madx["dy"], color="tab:orange", lw=2.5,
            label=r"$D_y$ MAD-X")
axs[1].plot(tw_xt["s"], tw_xt["dx"], "--", color="tab:green", lw=2,
            label=r"$D_x$ Xsuite")
axs[1].plot(tw_xt["s"], tw_xt["dy"], "--", color="tab:red", lw=2,
            label=r"$D_y$ Xsuite")
axs[1].set_ylabel(r"$D$ [m]"); axs[1].set_xlabel("s [m]")
axs[1].legend(ncol=2, fontsize=9)
axs[1].grid(alpha=0.3)

fig.suptitle("LNE02: MAD-X vs Xsuite (MATRIX deflectors restored + "
              "R-matrix patched)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
plt.show()

print(f"\nSaved: {OUT_PNG}")
