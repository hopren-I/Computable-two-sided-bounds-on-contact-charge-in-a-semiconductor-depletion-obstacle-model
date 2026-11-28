"""Regenerate CAM Tables 1–6, diagnostics and numerical text from the solvers."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf8") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="New directory for this run (default: runs/cam-*)")
    args = parser.parse_args()
    if args.output is None:
        runs = ROOT/"runs"
        runs.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="cam-",dir=runs))
    else:
        out = args.output.resolve()
        out.mkdir(parents=True,exist_ok=False)
    print(f"Run directory: {out}",flush=True)
    os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                      VECLIB_MAXIMUM_THREADS="1",MPLCONFIGDIR=str(out/"cache"),
                      EXPERIMENTS_RESULTS_DIR=str(out/"scratch"))
    sys.path[:0] = [str(ROOT/"code"),str(ROOT/"code/legacy")]
    import matplotlib
    import numpy
    import scipy
    from cam import experiments as ex
    from cam.checks import check_tables, check_prose
    from cam.intervals import single_bias

    start = time.monotonic()
    print("Tables 1 and 2: analytical and partial-electrode intervals",flush=True)
    t1 = ex.one_dimensional()
    t2, matched, neighbors = ex.matched_steps()
    save_csv(out/"table1.csv",t1)
    save_csv(out/"table2.csv",t2)
    save_csv(out/"matched_centers.csv",matched)
    save_csv(out/"matched_neighbors.csv",neighbors)

    print("Table 3: six fresh central solves",flush=True)
    t3, fields = ex.refinement_study()
    save_csv(out/"table3.csv",t3)
    save_csv(out/"refinement_fields.csv",fields)
    print("Table 4: uniform and graded quadratic finite elements",flush=True)
    t4, quadratic = ex.quadratic_study()
    save_csv(out/"table4.csv",t4)
    save_csv(out/"bernstein_fields.csv",quadratic)
    print("Tables 5 and 6: full-edge decomposition and flux counterexample",flush=True)
    t5, full = ex.full_edge()
    t6 = ex.gap_example()
    save_csv(out/"table5.csv",t5)
    save_csv(out/"full_edge_fields.csv",full)
    save_csv(out/"table6.csv",t6)

    finest = fields[-1]
    q512 = next(r["Q_mixed"] for r in fields if r["nx"]==512)
    graded = quadratic[-1]
    combined = single_bias(.5,graded["T"],graded["L"],finest["D"])
    middle = next(r for r in matched if r["b"]==.5)
    for charge in (graded["reaction"],finest["Q_mixed"]):
        if not (finest["lo"]<=charge<=finest["hi"] and middle["I3_lo"]<=charge<=middle["I3_hi"]):
            raise ArithmeticError("Cross-discretization containment check failed")
    narrowing = [100*(1-r["W3"]/r["W1"]) for r in t2]
    facts = dict(matched_initial_lo=middle["I1_lo"],matched_initial_hi=middle["I1_hi"],
                 matched_refined_lo=middle["I3_lo"],matched_refined_hi=middle["I3_hi"],
                 narrowing_min_pct=min(narrowing),narrowing_max_pct=max(narrowing),
                 finest_Q_mixed=finest["Q_mixed"],finest_initial_lo=finest["lo"],
                 finest_initial_hi=finest["hi"],last_Q_difference=abs(finest["Q_mixed"]-q512),
                 largest_center_error_pct=max(r["center_error_pct"] for r in t1),
                 Q2_T=graded["T"],Q2_L=graded["L"],Q2_gap=graded["E"]-finest["D"],
                 Q2_initial_lo=combined.endpoints[0],Q2_initial_hi=combined.endpoints[1],
                 Q2_step_ratio=combined.step/.5,Q2_width_pct=200*combined.radius/abs(q512),
                 full_edge_3_initial_width=full[-1]["single_width"])
    tables = check_tables([t1,t2,t3,t4,t5,t6])
    prose = check_prose(facts)
    save_csv(out/"comparison.csv",tables+prose)
    report = dict(python=platform.python_version(),platform=platform.platform(),
                  numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                  seconds=time.monotonic()-start,tables=dict(Counter(r["status"] for r in tables)),
                  prose=dict(Counter(r["status"] for r in prose)),values=facts)
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(f"Table comparisons: {report['tables']}; prose: {report['prose']}",flush=True)
    if any(r["status"]=="fail" for r in tables+prose):
        raise SystemExit("Numerical verification failed; see comparison.csv")
    print(f"Verified. Outputs: {out}")


if __name__=="__main__":
    main()

