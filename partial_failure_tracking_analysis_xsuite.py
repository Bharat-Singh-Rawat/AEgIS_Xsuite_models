"""
Partial quad-failure degradation study for LNE02 -- XSUITE VERSION.

Same three outputs as partial_failure_tracking_analysis.py (beam
profile grid, mismatch-factor / effective emittance growth curve,
phase-space ellipse tilt), but tracked with Xsuite instead of MAD-X.

TRACKING IS SIMPLER HERE THAN IN THE MAD-X VERSION:
-----------------------------------------------------------------------
No trackone file, no OBSERVE points, no segment parsing. The injected
bunch is generated directly in physical coordinates from the nominal
injected Twiss parameters (same matched-Gaussian transform used for the
phase-space ellipse plot itself: x = sqrt(eps*beta)*xn,
px = -sqrt(eps/beta)*(alpha*xn + pxn), for independent unit-Gaussian
xn, pxn). line.track(particles) then moves the whole bunch through the
line in a single pass, and the delivered distribution is read straight
off particles.x / .px / .y / .py (restricted to particles.state > 0,
i.e. particles that survived without being lost on an aperture).
-----------------------------------------------------------------------

IMPORTANT PHYSICS NOTE (same as the MAD-X version): true geometric
emittance from the tracked distribution's covariance is conserved
regardless of quad failure severity in this purely linear line -- a
failing quad creates MISMATCH, not emittance. The mismatch factor
    R = 0.5*(beta*gamma0 + beta0*gamma - 2*alpha*alpha0)
(actual vs. design Twiss at delivery) converts to a real emittance
growth factor via R + sqrt(R^2-1) IF this mismatch is later allowed to
filament -- that's what the growth-factor plot shows, not the (barely
changing) instantaneous emittance.

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

FAILURE_SCAN_CSV      = "quad_failure_scan_results_xsuite.csv"
TOP_N_QUADS           = 3
DEGRADATION_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
N_PARTICLES           = 5000
SEED                  = 42

# =========================================================================
# PART A -- build the Xsuite line (identical setup to the other two scripts)
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
epsx_geom_in = beam.exn / (g["beta"] * g["gamman"])
epsy_geom_in = beam.eyn / (g["beta"] * g["gamman"])

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

def do_twiss():
    return line.twiss(method="4d", start=xt.START, end=xt.END,
                       init=xt.TwissInit(**init_kwargs)).to_pandas()

def end_row(tw):
    row = tw[tw["name"].str.upper().str.startswith(END_PLACE.upper())]
    return row.iloc[0] if len(row) else tw.iloc[-1]

tw_nominal = do_twiss()
end_nominal = end_row(tw_nominal)
design = dict(betx=end_nominal["betx"], alfx=end_nominal["alfx"],
              bety=end_nominal["bety"], alfy=end_nominal["alfy"])
design["gammax"] = (1 + design["alfx"] ** 2) / design["betx"]
design["gammay"] = (1 + design["alfy"] ** 2) / design["bety"]
print("[setup] design (nominal) optics at", END_PLACE, "(Xsuite):")
for k in ["betx", "alfx", "bety", "alfy"]:
    print(f"    {k:5s} = {design[k]:.6e}")

# =========================================================================
# PART B -- generate the injected bunch ONCE, reused across every
# quad/degradation-level combination (matched-Gaussian transform, same
# formula used for the phase-space ellipse itself)
# =========================================================================
rng = np.random.default_rng(SEED)

def make_gaussian_bunch(n, betx, alfx, epsx, bety, alfy, epsy):
    xn, pxn = rng.normal(size=n), rng.normal(size=n)
    yn, pyn = rng.normal(size=n), rng.normal(size=n)
    x  = np.sqrt(epsx * betx) * xn
    px = -np.sqrt(epsx / betx) * (alfx * xn + pxn)
    y  = np.sqrt(epsy * bety) * yn
    py = -np.sqrt(epsy / bety) * (alfy * yn + pyn)
    return x, px, y, py

x0, px0, y0, py0 = make_gaussian_bunch(
    N_PARTICLES, init_kwargs["betx"], init_kwargs["alfx"], epsx_geom_in,
    init_kwargs["bety"], init_kwargs["alfy"], epsy_geom_in,
)
p0c0 = float(np.atleast_1d(line.particle_ref.p0c)[0])
mass00 = float(np.atleast_1d(line.particle_ref.mass0)[0])
q00 = float(np.atleast_1d(line.particle_ref.q0)[0])
print(f"[bunch] generated {N_PARTICLES} particles (reused for every "
      f"quad/degradation-level combination)")

def track_bunch():
    """Track the (fixed) injected bunch through the current machine
    state and return the surviving particles' final coordinates."""
    particles = xt.Particles(
        mass0=mass00, q0=q00, p0c=p0c0,
        x=x0.copy(), px=px0.copy(), y=y0.copy(), py=py0.copy(),
        zeta=0.0, delta=0.0,
    )
    line.track(particles, num_turns=1)
    mask = particles.state > 0
    n_lost = int((~mask).sum())
    if n_lost:
        print(f"    [note] {n_lost}/{N_PARTICLES} particles lost during tracking")
    return dict(x=particles.x[mask], px=particles.px[mask],
                y=particles.y[mask], py=particles.py[mask])

def twiss_from_distribution(d):
    Sx = np.cov(d["x"], d["px"])
    Sy = np.cov(d["y"], d["py"])
    epsx = np.sqrt(max(np.linalg.det(Sx), 0.0))
    epsy = np.sqrt(max(np.linalg.det(Sy), 0.0))
    betx = Sx[0, 0] / epsx if epsx > 0 else np.nan
    alfx = -Sx[0, 1] / epsx if epsx > 0 else np.nan
    bety = Sy[0, 0] / epsy if epsy > 0 else np.nan
    alfy = -Sy[0, 1] / epsy if epsy > 0 else np.nan
    return dict(betx=betx, alfx=alfx, bety=bety, alfy=alfy, epsx=epsx, epsy=epsy)

def mismatch_and_growth(betx, alfx, bety, alfy):
    gammax = (1 + alfx ** 2) / betx
    gammay = (1 + alfy ** 2) / bety
    Rx = 0.5 * (betx * design["gammax"] + design["betx"] * gammax - 2 * alfx * design["alfx"])
    Ry = 0.5 * (bety * design["gammay"] + design["bety"] * gammay - 2 * alfy * design["alfy"])
    Rx, Ry = max(Rx, 1.0), max(Ry, 1.0)
    growth_x = Rx + np.sqrt(max(Rx ** 2 - 1, 0.0))
    growth_y = Ry + np.sqrt(max(Ry ** 2 - 1, 0.0))
    return Rx, Ry, growth_x, growth_y

# =========================================================================
# PART C -- load critical quads from the failure scan
# =========================================================================
if not os.path.exists(FAILURE_SCAN_CSV):
    raise FileNotFoundError(f"{FAILURE_SCAN_CSV} not found -- run quad_failure_scan_xsuite.py first.")
scan_df = pd.read_csv(FAILURE_SCAN_CSV).sort_values("impact_score", ascending=False)
critical_quads = scan_df.head(TOP_N_QUADS)["name"].tolist()
print(f"\n[degradation study] top {len(critical_quads)} critical quads: {critical_quads}")

# =========================================================================
# PART D -- main loop
# =========================================================================
summary_rows = []

for quad in critical_quads:
    print(f"\n{'='*70}\n[quad] {quad}\n{'='*70}")
    k1_orig = get_k1(line, quad)

    per_fraction = []
    particle_snapshots = {}

    for frac in DEGRADATION_FRACTIONS:
        set_k1(line, quad, k1_orig * (1.0 - frac))

        try:
            d = track_bunch()
            tw_d = twiss_from_distribution(d)
        except Exception as e:
            print(f"[track-error] {quad} at frac={frac}: {e}")
            set_k1(line, quad, k1_orig)
            per_fraction.append(dict(fraction=frac, betx=np.nan, alfx=np.nan, bety=np.nan,
                                      alfy=np.nan, epsx=np.nan, epsy=np.nan, Rx=np.nan, Ry=np.nan,
                                      growth_x=np.nan, growth_y=np.nan, sigx_mm=np.nan, sigy_mm=np.nan))
            continue

        Rx, Ry, gx, gy = mismatch_and_growth(tw_d["betx"], tw_d["alfx"], tw_d["bety"], tw_d["alfy"])
        sigx_mm = np.sqrt(tw_d["betx"] * tw_d["epsx"]) * 1e3
        sigy_mm = np.sqrt(tw_d["bety"] * tw_d["epsy"]) * 1e3

        per_fraction.append(dict(fraction=frac, **tw_d, Rx=Rx, Ry=Ry,
                                  growth_x=gx, growth_y=gy, sigx_mm=sigx_mm, sigy_mm=sigy_mm))
        particle_snapshots[frac] = d

        print(f"    frac={frac:0.2f}  betx={tw_d['betx']:.3f} bety={tw_d['bety']:.3f}  "
              f"epsx={tw_d['epsx']*1e6:.3f} epsy={tw_d['epsy']*1e6:.3f} mm.mrad  "
              f"Rx={Rx:.3f} Ry={Ry:.3f}  growth_x={gx:.3f} growth_y={gy:.3f}")

        set_k1(line, quad, k1_orig)  # restore before next fraction

    df_quad = pd.DataFrame(per_fraction)
    csv_name = f"partial_failure_{quad}_results_xsuite.csv"
    df_quad.to_csv(csv_name, index=False)

    if df_quad["fraction"].eq(1.0).any():
        last = df_quad[df_quad["fraction"] == 1.0].iloc[0]
        summary_rows.append(dict(quad=quad, **{k: last[k] for k in
                              ["Rx", "Ry", "growth_x", "growth_y", "sigx_mm", "sigy_mm"]}))

    if not particle_snapshots:
        print(f"[skip plots] {quad}: no successful tracking runs")
        continue

    # ---------------------------------------------------------------
    # PLOT A -- beam profile grid (hexbin, fast for many panels)
    # ---------------------------------------------------------------
    worst_frac = max(particle_snapshots.keys())
    ref_x_mm = particle_snapshots[worst_frac]["x"] * 1e3
    ref_y_mm = particle_snapshots[worst_frac]["y"] * 1e3
    margin = 1.15
    xlim = (-margin * np.max(np.abs(ref_x_mm)), margin * np.max(np.abs(ref_x_mm)))
    ylim = (-margin * np.max(np.abs(ref_y_mm)), margin * np.max(np.abs(ref_y_mm)))

    fracs_done = sorted(particle_snapshots.keys())
    ncols = min(len(fracs_done), 3)
    nrows = int(np.ceil(len(fracs_done) / ncols))
    figA, axsA = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    axsA = np.atleast_1d(axsA).flatten()

    for idx, frac in enumerate(fracs_done):
        ax = axsA[idx]
        d = particle_snapshots[frac]
        x_mm, y_mm = d["x"] * 1e3, d["y"] * 1e3
        hb = ax.hexbin(x_mm, y_mm, gridsize=45, cmap="jet", extent=[*xlim, *ylim])
        figA.colorbar(hb, ax=ax).set_label("counts", fontsize=8)
        row = df_quad[df_quad["fraction"] == frac].iloc[0]
        ax.set_title(f"{quad}\nfailure={frac*100:.0f}%\n"
                     rf"$\sigma_x$={row.sigx_mm:.2f} mm, $\sigma_y$={row.sigy_mm:.2f} mm")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=0.3)
    for j in range(len(fracs_done), len(axsA)):
        axsA[j].axis("off")

    figA.suptitle(f"LNE02 (Xsuite): delivered beam profile vs {quad} degradation "
                  f"(N={N_PARTICLES} macroparticles)")
    figA.tight_layout()
    figA.savefig(f"partial_failure_{quad}_profile_grid_xsuite.png", dpi=200)

    # ---------------------------------------------------------------
    # PLOT B -- mismatch / effective emittance growth factor
    # ---------------------------------------------------------------
    figB, axB = plt.subplots(figsize=(7.5, 5))
    axB.plot(df_quad["fraction"] * 100, df_quad["growth_x"], "o-", color="tab:blue",
              label=r"$x$: growth factor ($R_x+\sqrt{R_x^2-1}$)")
    axB.plot(df_quad["fraction"] * 100, df_quad["growth_y"], "s-", color="tab:orange",
              label=r"$y$: growth factor ($R_y+\sqrt{R_y^2-1}$)")
    axB.axhline(1.0, color="k", lw=0.8, ls="--", label="matched (=1)")
    axB.set_xlabel("quad failure severity [%]")
    axB.set_ylabel("effective emittance growth factor")
    axB.set_title(f"LNE02 (Xsuite): mismatch-induced emittance growth vs {quad} degradation")
    axB.legend(); axB.grid(alpha=0.3)
    figB.tight_layout()
    figB.savefig(f"partial_failure_{quad}_mismatch_growth_xsuite.png", dpi=200)

    # ---------------------------------------------------------------
    # PLOT C -- phase-space ellipse overlay
    # ---------------------------------------------------------------
    figC, axC = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(fracs_done)))
    theta = np.linspace(0, 2 * np.pi, 300)
    for frac, color in zip(fracs_done, cmap):
        row = df_quad[df_quad["fraction"] == frac].iloc[0]
        betx, alfx, epsx = row["betx"], row["alfx"], row["epsx"]
        x_ell = np.sqrt(epsx * betx) * np.cos(theta)
        px_ell = -np.sqrt(epsx / betx) * (alfx * np.cos(theta) + np.sin(theta))
        axC.plot(x_ell * 1e3, px_ell * 1e3, color=color, lw=2, label=f"failure={frac*100:.0f}%")
    axC.set_xlabel("x [mm]"); axC.set_ylabel("px [mrad]")
    axC.set_title(f"LNE02 (Xsuite): x-px phase-space ellipse tilt vs {quad} degradation")
    axC.legend(fontsize=8); axC.grid(alpha=0.3)
    figC.tight_layout()
    figC.savefig(f"partial_failure_{quad}_phase_space_ellipse_xsuite.png", dpi=200)

    plt.close(figA); plt.close(figB); plt.close(figC)
    print(f"[saved] partial_failure_{quad}_{{profile_grid,mismatch_growth,"
          f"phase_space_ellipse}}_xsuite.png, {csv_name}")

# =========================================================================
# PART E -- cross-quad summary
# =========================================================================
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("partial_failure_summary_top_quads_xsuite.csv", index=False)
    print("\n[summary] at 100% failure, across the top critical quads:")
    print(summary_df.to_string(index=False))

print("\nDone.")
