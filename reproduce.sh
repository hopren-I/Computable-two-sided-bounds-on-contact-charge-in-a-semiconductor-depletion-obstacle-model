#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-python3}
mode=${1:-quick}
case "$mode" in quick|full|figures) ;; *) echo "Usage: sh reproduce.sh [quick|full|figures]" >&2; exit 2;; esac
mkdir -p "$root/runs"
work=$(mktemp -d "$root/runs/$mode.XXXXXX")
printf 'Run directory: %s\n' "$work"
cd "$work"
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$work/cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p data/original source verification figures result logs
cp -R "$root/code" .
cp "$root"/data/original/*.csv data/original/
run() {
    name=$1
    shift
    echo "Running $name"
    if ! "$@" >"logs/$name.log" 2>&1; then
        tail -n 35 "logs/$name.log" >&2
        return 1
    fi
}
if [ "$mode" = figures ]; then
    cp -R "$root/figure" .
    cp "$root"/data/one_dimensional_bounds.csv "$root"/data/two_dimensional_full_contact_bounds.csv data/
    run figure1 "$python_bin" figure/plot_figure1.py
    run figure4_source "$python_bin" figure/full-contact-mesh-arcs.py
    cd figure
    for name in contact-geometry charge_intervals_2d full-contact-mesh-arcs; do
        pdflatex -halt-on-error -interaction=batchmode "$name.tex"
    done
    echo "Four English figure PDFs: $work/figure"
    exit 0
fi
export EXPERIMENTS_RESULTS_DIR="$work/scratch/tests"
run regression env PYTHONPATH="$work/code/legacy" "$python_bin" -m unittest discover -s code/legacy/tests -v
run hv2013 "$python_bin" code/supplement/hv2013_intervals.py
run hv2014 "$python_bin" code/supplement/hv2014_disk.py
if [ "$mode" = full ]; then
    run grid_sweeps "$python_bin" code/experiments.py
    run full_edge_and_finest_mesh "$python_bin" code/verify_2d_exact.py
else
    run one_dimensional env PYTHONPATH="$work/code" "$python_bin" -c 'from experiments import run_1d, run_counterexample; run_1d(); run_counterexample()'
    # Provide archived 2D inputs for the subsequent checks and small-mesh reruns.
    cp "$root"/data/two_dimensional_*.csv data/
fi
run fine_mesh_audit "$python_bin" code/explanation_materials.py
run theory_checks "$python_bin" code/verify_theory.py
run step_and_flux_comparison "$python_bin" code/supplement/review_experiments.py
run csv_comparison "$python_bin" "$root/verification/compare_results.py" "$work"
run paper_values "$python_bin" "$root/verification/check_paper.py" "$work/data" "$work/result"
cat logs/paper_values.log
echo "Checks passed. Outputs and logs: $work"
