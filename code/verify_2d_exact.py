"""Supplement the manuscript with full-contact exact checks and one finer mesh.

Run with the same NumPy/SciPy environment as code/experiments.py. Existing
solvers, data, figures and manuscript are left untouched. Outputs are separate
CSV files plus a verification report; no theoretical convergence is asserted.
"""
from pathlib import Path
import csv
import gc
import json
import math
import os
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "verification/二维真值最小修订_20261115"
AUDIT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ["EXPERIMENTS_RESULTS_DIR"] = str(AUDIT / "scratch")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code/legacy"))
import numpy as np
import scipy
from src.fem2d import Mesh2D, assemble_rt0, assemble_q1
from src.experiments_2d import _evaluate, scalar_result

MESHES = (32, 64, 128, 256, 512)
BIASES = (.005, .05, .37, .5, 1.99, 2., 2.01, 3.)
RATIOS = (.05, .1, .2)
ENERGY_TOL = 2e-10
CHARGE_TOL = 2e-9
RESIDUAL_TOL = 2e-8
RESIDUAL_KEYS = (
    "primal_violation", "boundary_residual", "stationarity_residual",
    "dual_violation", "complementarity_residual", "equilibrium_residual",
    "gap_identity_residual", "energy_quadrature_residual",
    "dual_quadrature_residual", "reciprocity_residual",
)


def dump(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def exact(b):
    # Unit transverse height; exact one-dimensional energy is total 2D energy.
    if b < 2:
        return (2*b)**1.5 / 3, math.sqrt(2*b)
    return b*b / 4 + b - 1/3, b/2 + 1


def sample(mesh, mixed, y0, y1, a, analytic):
    start = time.perf_counter()
    r = dict(scalar_result(_evaluate(mesh, y0, y1, a, mixed)), b=a)
    r["elapsed_seconds"] = time.perf_counter() - start
    assert r["min_u"] >= -1e-12 and r["lambda_min"] >= 0, r
    assert all(abs(r[k]) < RESIDUAL_TOL for k in RESIDUAL_KEYS), r
    assert r["gap"] >= -ENERGY_TOL, r
    if analytic:
        J, Q = exact(a)
        r.update(J_exact=J, Q_exact=Q, primal_energy_error=r["E"]-J,
                 dual_energy_error=J-r["D"],
                 energy_violation=max(r["D"]-J, J-r["E"], 0.))
        assert r["energy_violation"] <= ENERGY_TOL, r
    return r


def interval(nx, b, ratio, center, left, right, analytic):
    k = b*ratio
    lo = (center["D"]-left["E"])/k
    hi = (right["E"]-center["D"])/k
    mid = (lo+hi)/2
    r = dict(nx=nx, ny=nx//2, h=2/nx, b=b, k=k, k_over_b=ratio,
             Q_lower=lo, Q_upper=hi, Q_midpoint=mid, width=hi-lo,
             gap_center=center["gap"], Q_mixed=center["Q_mixed"])
    assert lo <= hi, r
    if analytic:
        J, Q = exact(b)
        Jl, _ = exact(b-k)
        Jr, _ = exact(b+k)
        curvature = (Jr+Jl-2*J)/k
        envelope = (right["E"]-Jr+left["E"]-Jl+2*(J-center["D"]))/k
        error = abs(mid-Q)
        radius = (hi-lo)/2
        r.update(Q_exact=Q, contains_exact=lo-CHARGE_TOL <= Q <= hi+CHARGE_TOL,
                 containment_margin=min(Q-lo, hi-Q),
                 containment_violation=max(lo-Q, Q-hi, 0.),
                 midpoint_abs_error=error, recovered_abs_error=abs(center["Q_mixed"]-Q),
                 width_percent=100*(hi-lo)/abs(Q),
                 radius_error_excess=max(error-radius, 0.),
                 exact_curvature_term=curvature, energy_envelope_term=envelope,
                 width_identity_residual=hi-lo-curvature-envelope)
        assert r["contains_exact"] and r["radius_error_excess"] <= CHARGE_TOL, r
        assert abs(r["width_identity_residual"]) < 1e-9, r
    return r


def run_full_contact():
    samples, bounds, timings = [], [], []
    for nx in MESHES:
        start = time.perf_counter()
        mesh = Mesh2D(nx, nx//2)
        mixed = assemble_rt0(mesh, 0., 1.)
        cache = {}
        def get(a):
            a = round(float(a), 12)
            if a not in cache:
                cache[a] = sample(mesh, mixed, 0., 1., a, True)
                samples.append(cache[a])
            return cache[a]
        for b in BIASES:
            center = get(b)
            for ratio in RATIOS:
                bounds.append(interval(nx, b, ratio, center,
                                       get(b-b*ratio), get(b+b*ratio), True))
        timings.append(dict(nx=nx, elapsed_seconds=time.perf_counter()-start))
        dump(ROOT / "data/two_dimensional_full_contact_samples.csv", samples)
        dump(ROOT / "data/two_dimensional_full_contact_bounds.csv", bounds)
        print(f"full contact {nx}x{nx//2}: {len(cache)} samples, "
              f"{timings[-1]['elapsed_seconds']:.2f}s; exact checks passed", flush=True)
        del mixed, cache
        assemble_q1.cache_clear()
        gc.collect()
    return samples, bounds, timings


def run_local_extension():
    nx, b = 1024, .5
    mesh = Mesh2D(nx, nx//2)
    mixed = assemble_rt0(mesh, .25, .75)
    samples, cache, bounds = [], {}, []
    def get(a):
        a = round(float(a), 12)
        if a not in cache:
            print(f"local contact {nx}x{nx//2}: solving b={a}", flush=True)
            cache[a] = sample(mesh, mixed, .25, .75, a, False)
            samples.append(cache[a])
            dump(ROOT / "data/two_dimensional_extension_samples.csv", samples)
        return cache[a]
    center = get(b)
    for ratio in RATIOS:
        bounds.append(interval(nx, b, ratio, center, get(b-b*ratio), get(b+b*ratio), False))
    old_samples = list(csv.DictReader((ROOT / "data/two_dimensional_samples.csv").open()))
    old = next(r for r in old_samples if int(r["nx"]) == 512 and float(r["b"]) == b)
    q512 = float(old["Q_mixed"])
    for r in bounds:
        r.update(Q_512=q512, width_percent_Q512=100*r["width"]/abs(q512),
                 mesh_change_percent_Q512=100*abs(r["Q_mixed"]-q512)/abs(q512))
    dump(ROOT / "data/two_dimensional_extension_bounds.csv", bounds)
    del mixed, cache
    assemble_q1.cache_clear()
    gc.collect()
    return samples, bounds


def write_table(bounds):
    lines = [r"\multicolumn{6}{c}{（a）$128\times64$ 网格，$k/b=0.1$}\\",
             r"$b$ & 精确 $Q$ & $Q_-$ & $Q_+$ & $W$ & $|Q_{\rm mid}-Q|$\\\midrule"]
    for b in (.005, .37, 1.99, 2., 2.01, 3.):
        r = next(r for r in bounds if r["nx"] == 128 and r["b"] == b and r["k_over_b"] == .1)
        lines.append(f"{b:g} & {r['Q_exact']:.6f} & {r['Q_lower']:.6f} & "
                     f"{r['Q_upper']:.6f} & {r['width']:.6f} & {r['midpoint_abs_error']:.2e} " + r"\\")
    lines += [r"\midrule\multicolumn{6}{c}{（b）$b=0.37$，固定 $k/b=0.1$}\\",
              r"网格 & 精确 $Q$ & $Q_-$ & $Q_+$ & $W$ & $|Q_{\rm mid}-Q|$\\\midrule"]
    for nx in MESHES:
        r = next(r for r in bounds if r["nx"] == nx and r["b"] == .37 and r["k_over_b"] == .1)
        lines.append(f"${nx}\\times{nx//2}$ & {r['Q_exact']:.6f} & {r['Q_lower']:.6f} & "
                     f"{r['Q_upper']:.6f} & {r['width']:.6f} & {r['midpoint_abs_error']:.2e} " + r"\\")
    (ROOT / "source/table_2d_full_contact.tex").write_text("\n".join(lines)+"\n")


def main():
    start = time.perf_counter()
    fs, fb, timings = run_full_contact()
    es, eb = run_local_extension()
    write_table(fb)
    report = dict(
        all_passed=True, full_contact_samples=len(fs), full_contact_intervals=len(fb),
        full_contact_meshes=list(MESHES), full_contact_biases=list(BIASES),
        relative_steps=list(RATIOS), full_contact_timings=timings,
        maximum_energy_violation=max(r["energy_violation"] for r in fs),
        maximum_charge_violation=max(r["containment_violation"] for r in fb),
        minimum_charge_margin=min(r["containment_margin"] for r in fb),
        maximum_width_identity_residual=max(abs(r["width_identity_residual"]) for r in fb),
        residual_maxima={k:max(abs(r[k]) for r in fs+es) for k in RESIDUAL_KEYS},
        local_extension_samples=len(es), local_extension_intervals=len(eb),
        local_extension_best=min(eb, key=lambda r:r["width"]),
        tolerances=dict(energy=ENERGY_TOL, charge=CHARGE_TOL, residual=RESIDUAL_TOL),
        elapsed_seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        scope="Floating-point verification; local-contact recovered values are not exact truth.",
    )
    (AUDIT / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
