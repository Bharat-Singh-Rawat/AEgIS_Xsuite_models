"""
Recovery study for LNE02 -- XSUITE VERSION.

Same goal as quad_failure_recovery.py: for the most critical quad
failures (from quad_failure_scan_results_xsuite.csv), ask whether
neighboring quads can be re-tuned to restore the delivered optics at
LNE02$END. Here the retuning is done with Xsuite's native optimizer via
line.match(), using xt.VaryList / xt.TargetSet instead of MAD-X's
MATCH/VARY/CONSTRAINT blocks.

HOW THIS DIFFERS FROM THE MAD-X VERSION:
-----------------------------------------------------------------------
- Every quad we touch (the failed one AND its neighbors) is wrapped in
  an Xsuite knob on first use (see set_k1/get_k1 below) -- this makes
  the neighbors directly usable as `vary` names in line.match(),
  regardless of whether the original MAD-X model already tied that
  quad's k1 to a knob variable.
- No MAD-X-subprocess crash-recovery is needed: Xsuite runs in-process,
  so a failed/non-converged match just raises a normal Python
  exception or reports tol_met=False, both of which we catch/inspect
  per-quad without needing to rebuild anything.
-----------------------------------------------------------------------

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line
(after running quad_failure_scan_xsuite.py at least once)

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

FAILURE_SCAN_CSV = "quad_failure_scan_results_xsuite.csv"
TOP_N_CRITICAL   = 5
N_NEIGHBORS      = 2
FAILURE_FRACTION = 1.0

MATCH_STEP = 1e-6
MATCH_TOL  = 1e-9

OUT_CSV        = "quad_recovery_results_xsuite.csv"
OUT_PNG_RESID  = "quad_recovery_residuals_xsuite.png"
OUT_PNG_COST   = "quad_recovery_retune_cost_xsuite.png"
OUT_PNG_TWISS  = "quad_recovery_twiss_examples_xsuite.png"

# =========================================================================
# PART A -- build the Xsuite line (identical setup to quad_failure_scan_xsuite.py)
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
    return madx, line, tw_madx

print("[setup] building Xsuite line...")
madx, line, tw_madx = build_xsuite_line()

g = madx.globals
init_kwargs = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)
beam = madx.sequence[SEQ_NAME].beam
epsx_geom = beam.exn / (g["beta"] * g["gamman"])
epsy_geom = beam.eyn / (g["beta"] * g["gamman"])

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
    return knob

def get_k1(line, name):
    return float(line[name].k1)

def do_twiss():
    return line.twiss(method="4d", start=xt.START, end=xt.END,
                       init=xt.TwissInit(**init_kwargs)).to_pandas()

def end_row(tw):
    row = tw[tw["name"].str.upper().str.startswith(END_PLACE.upper())]
    return row.iloc[0] if len(row) else tw.iloc[-1]

def sigma_xy(row):
    return np.sqrt(row["betx"] * epsx_geom), np.sqrt(row["bety"] * epsy_geom)

tw_nominal = do_twiss()
target = end_row(tw_nominal)
sigx_nom, sigy_nom = sigma_xy(target)
print("[setup] target (nominal) optics at", END_PLACE, "(Xsuite):")
for k in ["betx", "alfx", "bety", "alfy", "dx", "dpx", "dy", "dpy"]:
    print(f"    {k:5s} = {target[k]:.8e}")

# -------------------------------------------------------------------
# SANITY CHECK: does Xsuite's dispersion at the delivery point actually
# agree with MAD-X's, for the exact same explicit initial conditions?
# This settles directly whether the deflector R-matrix patch is
# reproducing MAD-X's dispersion correctly, rather than guessing from
# the "n_found/36" R-matrix count (which can be inconclusive -- MAD-X's
# own default for an unset rmNN is the same identity value we already
# default to, so a low n_found doesn't necessarily mean anything is
# wrong, and a high n_found doesn't guarantee it's right either).
# -------------------------------------------------------------------
madx.command.use(sequence=SEQ_NAME)
tw_madx_check = madx.twiss(
    betx=init_kwargs["betx"], alfx=init_kwargs["alfx"],
    bety=init_kwargs["bety"], alfy=init_kwargs["alfy"],
    dx=init_kwargs["dx"], dpx=init_kwargs["dpx"],
    dy=init_kwargs["dy"], dpy=init_kwargs["dpy"],
).dframe()
tw_madx_check.columns = [c.lower() for c in tw_madx_check.columns]
end_madx_check = tw_madx_check[tw_madx_check["name"].str.upper()
                                .str.startswith(END_PLACE.upper())]
end_madx_check = end_madx_check.iloc[0] if len(end_madx_check) else tw_madx_check.iloc[-1]

print("\n[sanity check] MAD-X vs Xsuite at", END_PLACE, "(same explicit init conditions):")
print(f"{'quantity':6s} {'MAD-X':>16s} {'Xsuite':>16s} {'abs diff':>14s} {'rel diff':>10s}")
mismatch_found = False
for k in ["betx", "alfx", "bety", "alfy", "dx", "dpx", "dy", "dpy"]:
    v_madx, v_xt = float(end_madx_check[k]), float(target[k])
    diff = abs(v_madx - v_xt)
    rel = diff / abs(v_madx) if v_madx != 0 else np.inf
    flag = "  <-- MISMATCH" if rel > 0.01 else ""
    if flag:
        mismatch_found = True
    print(f"{k:6s} {v_madx:16.6e} {v_xt:16.6e} {diff:14.4e} {rel*100:9.2f}%{flag}")
if mismatch_found:
    print("\n[sanity check] WARNING: Xsuite disagrees with MAD-X by >1% on at least "
          "one quantity above (usually dispersion). This means the deflector "
          "R-matrix patch is not fully reproducing MAD-X's optics -- treat any "
          "dispersion-dependent result from this script (and the other Xsuite "
          "scripts) with caution until that's resolved.")
else:
    print("\n[sanity check] OK: Xsuite agrees with MAD-X to within 1% on all "
          "eight quantities above.")

# =========================================================================
# PART C -- quad list + neighbor lookup
# =========================================================================
tab = line.get_table()
quad_names_ordered = list(tab.rows[tab.element_type == "Quadrupole"].name)
s_lookup = tw_nominal.set_index("name")["s"]
quad_names_ordered.sort(key=lambda n: float(s_lookup.get(n, np.inf)))

def get_neighbors(failed_name, n_each_side):
    idx = quad_names_ordered.index(failed_name)
    before = quad_names_ordered[max(0, idx - n_each_side):idx]
    after  = quad_names_ordered[idx + 1: idx + 1 + n_each_side]
    return before + after

# =========================================================================
# PART D -- load critical quads from the failure scan
# =========================================================================
if not os.path.exists(FAILURE_SCAN_CSV):
    raise FileNotFoundError(f"{FAILURE_SCAN_CSV} not found -- run quad_failure_scan_xsuite.py first.")
scan_df = pd.read_csv(FAILURE_SCAN_CSV).sort_values("impact_score", ascending=False)
critical_quads = scan_df.head(TOP_N_CRITICAL)["name"].tolist()
print(f"\n[recovery] attempting recovery for top {len(critical_quads)} critical quads: {critical_quads}")

# =========================================================================
# PART E -- fail + match + record, one quad at a time
# =========================================================================
results = []
twiss_examples = {}

for failed_name in critical_quads:
    neighbors = get_neighbors(failed_name, N_NEIGHBORS)
    if not neighbors:
        print(f"[skip] {failed_name}: no neighboring quads available")
        continue
    print(f"\n[recovery] failing {failed_name}, matching with neighbors {neighbors}")

    k1_failed_orig = get_k1(line, failed_name)
    k1_neighbors_orig = {n: get_k1(line, n) for n in neighbors}

    set_k1(line, failed_name, k1_failed_orig * (1.0 - FAILURE_FRACTION))
    tw_failed = do_twiss()
    row_failed = end_row(tw_failed)
    sigx_failed, sigy_failed = sigma_xy(row_failed)

    neighbor_knobs = [set_k1(line, n, k1_neighbors_orig[n]) for n in neighbors]  # ensure knobbed
    # IMPORTANT: xt.VaryList's `step` takes ONE scalar for the whole group of
    # names -- passing a per-name list there crashes internally ("setting an
    # array element with a sequence"). To give each neighbor its own
    # appropriately-scaled step, use individual xt.Vary objects instead.
    vary_list = [
        xt.Vary(name=knob, step=max(abs(k1_neighbors_orig[n]) * 1e-3, 1e-8))
        for knob, n in zip(neighbor_knobs, neighbors)
    ]

    # NOTE: dx/dpx/dy/dpy are DELIBERATELY excluded from the targets below.
    # The sanity check above confirmed Xsuite's dispersion at LNE02$END
    # disagrees with MAD-X by ~140-160% (dx, dpx) and ~30-35% (dy, dpy) --
    # the deflector R-matrix patch is not correctly reproducing MAD-X's
    # dispersion. Matching against that wrong dispersion target is exactly
    # why every quad failed to converge ("Could not find point within
    # tolerance") even after the tolerance/step fixes. Until the R-matrix
    # extraction is fixed (see the printed R-matrix "n_found/36" values and
    # inspect which specific rmNN entries -- especially rm16/rm26/rm36/rm46
    # dispersion-coupling terms -- are missing or wrong), this script
    # matches on betatron optics (betx/alfx/bety/alfy) only, which the
    # sanity check showed agrees with MAD-X to within ~1%.
    targets = [
        xt.Target("betx", target["betx"], at=xt.END, tol=1e-6),
        xt.Target("alfx", target["alfx"], at=xt.END, tol=1e-6),
        xt.Target("bety", target["bety"], at=xt.END, tol=1e-6),
        xt.Target("alfy", target["alfy"], at=xt.END, tol=1e-6),
    ]

    match_ok = True
    opt = None
    try:
        opt = line.match(
            solve=False,  # <-- solve manually below so we can recover the
                          #     best point even if solve() raises partway
            method="4d", start=xt.START, end=xt.END,
            init=xt.TwissInit(**init_kwargs),
            vary=vary_list,
            targets=targets,
        )
        opt.solve()
    except Exception as e:
        print(f"[match warning] {failed_name}: did not fully converge ({e})")
        match_ok = False

    # Regardless of whether solve() reported success: line.match()'s
    # internal exception-on-non-convergence path can leave the line at a
    # REJECTED trial point (e.g. a final internal check step) rather than
    # the best point actually found during optimization -- which is
    # exactly why earlier runs showed "recovered == failed" even though
    # the printed penalty had clearly improved. To guard against that,
    # explicitly reload whichever logged iteration had the lowest penalty.
    if opt is not None:
        try:
            log = opt.log()
            penalties = np.asarray(log["penalty"])
            best_iter = int(np.argmin(penalties))
            opt.reload(iteration=best_iter)
            print(f"    [reload] using iteration {best_iter} "
                  f"(penalty={penalties[best_iter]:.4e}) as the recovered state")
        except Exception as e:
            print(f"    [warning] {failed_name}: could not reload best iterate ({e}) "
                  f"-- using whatever state line.match() left behind")
        try:
            opt.target_status()
        except Exception:
            pass

    try:
        tw_recovered = do_twiss()
        row_recovered = end_row(tw_recovered)
        sigx_rec, sigy_rec = sigma_xy(row_recovered)
    except Exception as e:
        print(f"[twiss-error after match] {failed_name}: {e}")
        match_ok = False
        tw_recovered, row_recovered = tw_failed, row_failed
        sigx_rec, sigy_rec = sigx_failed, sigy_failed

    k1_neighbors_new = {n: get_k1(line, n) for n in neighbors}
    max_delta_k1_pct = max(
        100 * abs(k1_neighbors_new[n] - k1_neighbors_orig[n]) / abs(k1_neighbors_orig[n])
        if k1_neighbors_orig[n] != 0 else np.nan
        for n in neighbors
    )
    residual_pct = 100 * np.sqrt(
        (row_recovered["betx"] - target["betx"]) ** 2 / target["betx"] ** 2
        + (row_recovered["bety"] - target["bety"]) ** 2 / target["bety"] ** 2
    )
    # match_ok now reflects the actual residual after reloading the best
    # iterate, not just whether solve() happened to raise -- a run that
    # "failed to fully converge" but still landed within 1% is a real
    # recovery, and one that stayed at thousands of percent is not,
    # regardless of what solve() itself reported.
    match_ok = bool(residual_pct < 1.0)

    results.append(dict(
        failed_quad=failed_name, neighbors_used=";".join(neighbors), match_ok=match_ok,
        sigx_nominal_mm=sigx_nom * 1e3, sigy_nominal_mm=sigy_nom * 1e3,
        sigx_failed_mm=sigx_failed * 1e3, sigy_failed_mm=sigy_failed * 1e3,
        sigx_recovered_mm=sigx_rec * 1e3, sigy_recovered_mm=sigy_rec * 1e3,
        betx_recovered=row_recovered["betx"], bety_recovered=row_recovered["bety"],
        residual_pct=residual_pct, max_delta_k1_pct=max_delta_k1_pct,
    ))
    twiss_examples[failed_name] = dict(nominal=tw_nominal, failed=tw_failed, recovered=tw_recovered)

    print(f"    sigx: nominal={sigx_nom*1e3:.4f}  failed={sigx_failed*1e3:.4f}  "
          f"recovered={sigx_rec*1e3:.4f} mm")
    print(f"    residual mismatch = {residual_pct:.3f}%   max neighbor |Δk1| = {max_delta_k1_pct:.2f}%")

    # restore everything before the next quad
    set_k1(line, failed_name, k1_failed_orig)
    for n in neighbors:
        set_k1(line, n, k1_neighbors_orig[n])

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\n[recovery] done, saved results to {OUT_CSV}")
print(df[["failed_quad", "residual_pct", "max_delta_k1_pct", "match_ok"]].to_string(index=False))

# =========================================================================
# PLOT 1 -- residual mismatch after recovery
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
colors = ["tab:green" if ok else "tab:red" for ok in df["match_ok"]]
ax1.bar(df["failed_quad"], df["residual_pct"], color=colors, alpha=0.8)
ax1.set_ylabel("residual beta mismatch at delivery [%]")
ax1.set_title("LNE02 (Xsuite): how well can neighboring quads recover each failure?\n"
              "(lower = more fully recoverable; red = line.match() reported an issue)")
ax1.grid(alpha=0.3, axis="y")
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
fig1.tight_layout(); fig1.savefig(OUT_PNG_RESID, dpi=200)

# =========================================================================
# PLOT 2 -- retuning cost
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.bar(df["failed_quad"], df["max_delta_k1_pct"], color="tab:purple", alpha=0.8)
ax2.set_ylabel("max neighbor |Δk1| required [%]")
ax2.set_title("LNE02 (Xsuite): retuning effort required to recover each failure")
ax2.grid(alpha=0.3, axis="y")
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
fig2.tight_layout(); fig2.savefig(OUT_PNG_COST, dpi=200)

# =========================================================================
# PLOT 3 -- example twiss curves
# =========================================================================
n_examples = min(4, len(twiss_examples))
fig3, axs3 = plt.subplots(n_examples, 1, figsize=(10, 3.2 * n_examples), sharex=True)
axs3 = np.atleast_1d(axs3)
for ax, (name, tws) in zip(axs3, list(twiss_examples.items())[:n_examples]):
    ax.plot(tws["nominal"]["s"], tws["nominal"]["betx"], color="navy", lw=2, label=r"$\beta_x$ nominal")
    ax.plot(tws["failed"]["s"], tws["failed"]["betx"], color="tab:red", lw=1.5, ls="--", label=r"$\beta_x$ failed")
    ax.plot(tws["recovered"]["s"], tws["recovered"]["betx"], color="tab:green", lw=1.5, label=r"$\beta_x$ recovered")
    ax.set_title(f"{name} failed"); ax.set_ylabel(r"$\beta_x$ [m]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
axs3[-1].set_xlabel("s [m]")
fig3.suptitle("LNE02 (Xsuite): nominal vs failed vs recovered optics (examples)")
fig3.tight_layout(); fig3.savefig(OUT_PNG_TWISS, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_CSV}\n    {OUT_PNG_RESID}\n    {OUT_PNG_COST}\n    {OUT_PNG_TWISS}")
