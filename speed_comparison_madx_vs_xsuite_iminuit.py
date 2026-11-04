"""
Speed comparison: MAD-X vs Xsuite as the physics engine underneath the
SAME optimizer (iminuit / MINUIT2) for quadrupole-strength matching.

Two objectives, matching the earlier Monte Carlo script:
  "beamsize"  : delivered beam size at LNE02$END
  "emittance" : emittance-driven size only (no dispersion)

RUN FROM: ~/Xsuite/acc-models-tls/elena_extraction/lne02/line

"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpymad.madx import Madx
import xtrack as xt

try:
    from iminuit import Minuit
    HAVE_IMINUIT = True
except ImportError:
    HAVE_IMINUIT = False
    from scipy.optimize import minimize
    print("[warn] iminuit not installed -- falling back to "
          "scipy.optimize.minimize(method='Nelder-Mead') on BOTH sides. "
          "Run: pip install iminuit --break-system-packages for the "
          "actual MINUIT2-based comparison.")

# =========================================================================
# Config
# =========================================================================
MADX_FILE  = "general_lne02.madx"
SEQ_NAME   = "lne00lne01lne02"
END_PLACE  = "LNE02$END"

QUAD_NAMES = [
    "KLNE.ZQMD.0208", "KLNE.ZQMF.0209", "KLNE.ZQMD.0214", "KLNE.ZQMF.0215",
]
KNOWN_DEFLECTOR_INSTANCES = ["lnr.zdfhl.0610", "lne.zdshr.0220"]

EMITH, EMITV = 6.0, 4.0
DPPI         = 2.5
BETX_MAX, BETY_MAX = 50.0, 100.0
PENALTY_W    = 1e4

QUAD_BOUNDS       = (-54.0, 54.0)
QUAD_START_SPREAD = 10.0
N_RUNS            = 100     # Monte Carlo repetitions PER objective
SEED              = 42

OUT_PNG_TIMING = "speed_timing_comparison.png"
OUT_PNG_OBJ    = "speed_objective_comparison.png"
OUT_CSV        = "speed_comparison_results.csv"

rng = np.random.default_rng(SEED)

# =========================================================================
# PART A -- setup: run MAD-X#===========================================
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

madx = Madx(stdout=False)
madx.call(file=tmp_madx_file)

print("[setup] restoring correct MATRIX deflector bindings...")
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
g = madx.globals
init_kwargs = dict(
    betx=g["betx0"], alfx=g["alfx0"], bety=g["bety0"], alfy=g["alfy0"],
    dx=g["dx0"], dpx=g["dpx0"], dy=g["dy0"], dpy=g["dpy0"],
)
beta_rel = g["beta"]
k_nominal = np.array([float(g[name.lower()]) for name in QUAD_NAMES])
print(f"[setup] nominal quad strengths: {np.round(k_nominal, 4)}")

line = xt.Line.from_madx_sequence(
    madx.sequence[SEQ_NAME], allow_thick=True, deferred_expressions=True,
)
beam = madx.sequence[SEQ_NAME].beam
line.particle_ref = xt.Particles(
    mass0=beam.mass * 1e9, q0=beam.charge, p0c=beam.pc * 1e9,
)
line.build_tracker()
xt_var_names = [name.lower() for name in QUAD_NAMES]

print("[setup] patching Xsuite deflector conversion with exact "
      "R-matrices...")
def get_R_from_save(inst_name):
    save_file = "setup_lookup.madx"
    madx.input(f'save, sequence={SEQ_NAME}, file="{save_file}", noexpr;')
    with open(save_file) as f:
        content = f.read()
    m_inst = re.search(re.escape(inst_name) + r"\s*:\s*(\S+?)\s*;", content,
                        re.IGNORECASE)
    parent_type = m_inst.group(1)
    m_type = re.search(re.escape(parent_type) + r"\s*:\s*matrix\s*,(.*?);",
                        content, re.IGNORECASE | re.DOTALL)
    body = m_type.group(1)
    R = np.eye(6)
    for i in range(1, 7):
        for j in range(1, 7):
            mm = re.search(rf"\brm{i}{j}\s*=\s*([-+0-9.eEdD]+)", body,
                            re.IGNORECASE)
            if mm:
                R[i - 1, j - 1] = float(
                    mm.group(1).replace("D", "E").replace("d", "e"))
    return R

tw_madx_setup = madx.twiss(beta0="initbeta0").dframe()
tw_madx_setup.columns = [c.lower() for c in tw_madx_setup.columns]

for inst_name in KNOWN_DEFLECTOR_INSTANCES:
    try:
        R = get_R_from_save(inst_name)
    except Exception as e:
        print(f"    [skip] {inst_name}: {e}")
        continue
    length_row = tw_madx_setup[
        tw_madx_setup["name"].str.split(":").str[0] == inst_name]
    length = float(length_row.iloc[0]["l"]) if len(length_row) else 0.6
    target_name = next(
        (c for c in line.element_names if c.split(":")[0] == inst_name), None)
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
    if new_elem is not None:
        line.element_dict[target_name] = new_elem
        print(f"    [patched] {target_name}")
line.build_tracker()

# =========================================================================
# PART B -- objective functions 
# =========================================================================
def compute_objective(betx, bety, dx, dy, objective, tw_full_s, tw_full_betx,
                       tw_full_bety, end_s):
    if objective == "beamsize":
        sx = np.sqrt(betx * EMITH / 6 + (beta_rel * dx * DPPI / 4) ** 2)
        sy = np.sqrt(bety * EMITV / 6 + (beta_rel * dy * DPPI / 4) ** 2)
    else:  # "emittance"
        sx = np.sqrt(max(betx * EMITH / 6, 0.0))
        sy = np.sqrt(max(bety * EMITV / 6, 0.0))
    obj = sx + sy
    mask = tw_full_s >= end_s - 100
    over_x = np.clip(tw_full_betx[mask] - BETX_MAX, 0, None)
    over_y = np.clip(tw_full_bety[mask] - BETY_MAX, 0, None)
    obj += PENALTY_W * (np.sum(over_x ** 2) + np.sum(over_y ** 2))
    return obj

def madx_objective(k_vec, objective):
    for nm, val in zip(QUAD_NAMES, k_vec):
        madx.globals[nm.lower()] = float(val)
    try:
        tw = madx.twiss(beta0="initbeta0").dframe()
        tw.columns = [c.lower() for c in tw.columns]
    except Exception:
        return 1e6
    row = tw[tw["name"].str.upper().str.startswith(END_PLACE.upper())]
    if len(row) == 0:
        return 1e6
    row = row.iloc[0]
    return compute_objective(
        row["betx"], row["bety"], row["dx"], row["dy"], objective,
        tw["s"].to_numpy(), tw["betx"].to_numpy(), tw["bety"].to_numpy(),
        row["s"],
    )

def xsuite_objective(k_vec, objective):
    for nm, val in zip(xt_var_names, k_vec):
        line.vars[nm] = val
    try:
        tw = line.twiss(method="4d", start=xt.START, end=xt.END,
                         init=xt.TwissInit(**init_kwargs)).to_pandas()
    except Exception:
        return 1e6
    row = tw[tw["name"].str.lower().str.startswith("lne02$end")]
    row = row.iloc[0] if len(row) else tw.iloc[-1]
    return compute_objective(
        row["betx"], row["bety"], row["dx"], row["dy"], objective,
        tw["s"].to_numpy(), tw["betx"].to_numpy(), tw["bety"].to_numpy(),
        row["s"],
    )

def run_iminuit(objective_fn, objective, start):
    if HAVE_IMINUIT:
        m = Minuit(lambda a, b, c, d: objective_fn([a, b, c, d], objective),
                   a=start[0], b=start[1], c=start[2], d=start[3])
        for nm in ("a", "b", "c", "d"):
            m.limits[nm] = QUAD_BOUNDS
        m.errordef = Minuit.LEAST_SQUARES
        m.migrad()
        k = np.array([m.values["a"], m.values["b"], m.values["c"], m.values["d"]])
        n_calls = m.nfcn
        ok = m.valid
    else:
        res = minimize(lambda k: objective_fn(k, objective), start,
                        method="Nelder-Mead", bounds=[QUAD_BOUNDS] * 4,
                        options=dict(maxiter=2000, xatol=1e-6, fatol=1e-6))
        k = res.x
        n_calls = res.nfev
        ok = res.success
    final_obj = objective_fn(k, objective)
    return k, final_obj, n_calls, ok

# =========================================================================
# PART C -- Monte Carlo sweep
# =========================================================================
records = []
for objective in ["beamsize", "emittance"]:
    print(f"\n{'='*70}\nOBJECTIVE = {objective}\n{'='*70}")
    for run_id in range(N_RUNS):
        start = np.clip(
            k_nominal + rng.uniform(-QUAD_START_SPREAD, QUAD_START_SPREAD, 4),
            *QUAD_BOUNDS,
        )
        print(f"\n[run {run_id+1}/{N_RUNS}] start={np.round(start,3)}")

        t0 = time.perf_counter()
        k_m, obj_m, ncalls_m, ok_m = run_iminuit(madx_objective, objective, start)
        t_madx = time.perf_counter() - t0
        print(f"    [MAD-X + iminuit ] {t_madx:6.3f}s  "
              f"obj={obj_m:.4f}  n_calls={ncalls_m}")

        t0 = time.perf_counter()
        k_x, obj_x, ncalls_x, ok_x = run_iminuit(xsuite_objective, objective, start)
        t_xsuite = time.perf_counter() - t0
        print(f"    [Xsuite + iminuit] {t_xsuite:6.3f}s  "
              f"obj={obj_x:.4f}  n_calls={ncalls_x}")

        records.append(dict(objective=objective, run=run_id, engine="MAD-X",
                             time_s=t_madx, obj_val=obj_m, n_calls=ncalls_m,
                             success=ok_m))
        records.append(dict(objective=objective, run=run_id, engine="Xsuite",
                             time_s=t_xsuite, obj_val=obj_x, n_calls=ncalls_x,
                             success=ok_x))

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUT_CSV}")

# =========================================================================
# PART D plots
# =========================================================================
print("\n" + "=" * 70)
print("SPEED SUMMARY (mean +/- std over runs, same optimizer both sides)")
print("=" * 70)
summary = df.groupby(["objective", "engine"]).agg(
    time_mean=("time_s", "mean"), time_std=("time_s", "std"),
    calls_mean=("n_calls", "mean"),
    obj_mean=("obj_val", "mean"), success_rate=("success", "mean"),
)
print(summary.to_string())

for obj in ["beamsize", "emittance"]:
    t_madx_mean = df[(df.objective == obj) & (df.engine == "MAD-X")]["time_s"].mean()
    t_xt_mean = df[(df.objective == obj) & (df.engine == "Xsuite")]["time_s"].mean()
    speedup = t_madx_mean / t_xt_mean if t_xt_mean > 0 else float("nan")
    print(f'\n"{obj}": Xsuite is {speedup:.1f}x '
          f'{"faster" if speedup > 1 else "slower"} than MAD-X '
          f"(mean {t_madx_mean:.3f}s vs {t_xt_mean:.3f}s per optimization)")

fig1, axs1 = plt.subplots(1, 2, figsize=(11, 5))
for ax, obj in zip(axs1, ["beamsize", "emittance"]):
    data = [
        df[(df.objective == obj) & (df.engine == "MAD-X")]["time_s"].values,
        df[(df.objective == obj) & (df.engine == "Xsuite")]["time_s"].values,
    ]
    ax.boxplot(data, tick_labels=["MAD-X\n+ iminuit", "Xsuite\n+ iminuit"])
    ax.set_ylabel("wall-clock time [s]")
    ax.set_title(f'objective = "{obj}"')
    ax.grid(alpha=0.3)
fig1.suptitle(f"Engine speed comparison, "
              f"{N_RUNS} Monte Carlo runs each")
fig1.tight_layout()
fig1.savefig(OUT_PNG_TIMING, dpi=200)

fig2, axs2 = plt.subplots(1, 2, figsize=(11, 5))
for ax, obj in zip(axs2, ["beamsize", "emittance"]):
    data = [
        df[(df.objective == obj) & (df.engine == "MAD-X")]["obj_val"].values,
        df[(df.objective == obj) & (df.engine == "Xsuite")]["obj_val"].values,
    ]
    ax.boxplot(data, tick_labels=["MAD-X\n+ iminuit", "Xsuite\n+ iminuit"])
    ax.set_ylabel("achieved objective value")
    ax.set_title(f'objective = "{obj}"')
    ax.grid(alpha=0.3)
fig2.suptitle("Achieved objective value: MAD-X vs Xsuite ")
fig2.tight_layout()
fig2.savefig(OUT_PNG_OBJ, dpi=200)

plt.show()
print(f"\nSaved:\n    {OUT_PNG_TIMING}\n    {OUT_PNG_OBJ}\n    {OUT_CSV}")
