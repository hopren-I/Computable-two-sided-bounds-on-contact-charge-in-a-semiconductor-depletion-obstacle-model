# Computable two-sided bounds on contact charge in a semiconductor depletion obstacle model

Numerical experiments and underlying data for the manuscript by Siyuan Chen and Lipu Zhang. The CSV files contain the primal and dual energies, charge interval endpoints, mesh and bias parameters, and reference values used in the paper. The code reproduces the one-dimensional and two-dimensional tests and compares the recomputed values with the archived records.

## Run the experiments

Use Python 3.12 on macOS or Linux (including WSL). From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verification/check_paper.py
sh reproduce.sh quick
```

The first check uses only the Python standard library. It checks the five printed tables, the coordinates of Figure 3 and selected values quoted in the text against the archived CSVs. The quick run executes the solver regression tests, one-dimensional sweeps, both literature benchmarks, and small-mesh step-selection and flux-recovery experiments. It reads the archived large two-dimensional sweeps for the remaining comparisons.

To recompute all numerical experiments, including the largest mesh:

```sh
sh reproduce.sh full
```

The full run solves all five base meshes, the separate full-edge cases and the partial-electrode extension on the 1024 × 512 mesh. The original project's recorded peak memory for the full-edge and extension stage was about 15.3 GB. Allow additional memory for the operating system. Each run prints the path to a fresh directory under `runs/`. Its `data/`, `result/`, `verification/` and `logs/` directories contain the new records and checks. The distributed data remain available for comparison. To select another installed interpreter, set `PYTHON` to its absolute path.

## Experiments and records

Paths in the third column identify the archived records at the repository root and the corresponding outputs under a run directory. Within a row, filenames without a directory share the directory of the first filename.

| Manuscript item | Calculation | Numerical record |
|---|---|---|
| Table 1, three one-dimensional literature loads | `code/supplement/hv2013_intervals.py` computes 36 intervals and compares them with charges derived from endpoint derivatives. | `result/hv2013_1d_intervals.csv` |
| Figure 1, bias-step sweep and branch tests | `code/experiments.py` evaluates the 1D potential and equilibrated flux, scans the step size and checks the transition between partial and full depletion. | `data/one_dimensional_samples.csv`, `one_dimensional_bounds.csv`, `branch_transition_bounds.csv` |
| Figure 2 | Boundary geometry and mixed boundary conditions for the partial electrode. | `figure/contact-geometry.tex` |
| Figure 3 and Table 2, partial electrode | `code/experiments.py` computes 175 field pairs and 75 intervals on five meshes. `code/explanation_materials.py` also checks the fine-mesh values against the inputs in `data/original/`. | `data/two_dimensional_samples.csv`, `two_dimensional_bounds.csv`, `two_dimensional_summary.csv` |
| Table 3, sixth mesh | `code/verify_2d_exact.py` extends the partial-electrode experiment to 1024 × 512 at bias 0.5. The narrowest interval is [0.870759, 0.921227] at the precision reported in the paper. | `data/two_dimensional_extension_samples.csv`, `two_dimensional_extension_bounds.csv` |
| Figure 4 and Table 4, full-edge electrode | `code/verify_2d_exact.py` tests the analytical charge at eight biases on five meshes. `code/supplement/review_experiments.py` decomposes the widths of the 120 intervals. | `data/two_dimensional_full_contact_samples.csv`, `two_dimensional_full_contact_bounds.csv`, and `result/truth_decomposition.csv` |
| Table 5, circular free boundary | `code/supplement/hv2014_disk.py` solves the disk benchmark on four meshes and compares the contact radius and displacement energy with their analytical values. | `result/hv2014_disk.csv` |
| Step search and RT0 flux optimization | `code/supplement/review_experiments.py` and `rt0_comparison.py` compare fixed candidate steps, sequential step search and optimization of the central flux. | `result/step_benchmark.csv`, `step_trace.csv`, `recovery_comparison.csv`, `small_samples.csv` |
| Additional numerical consistency checks | `code/experiments.py` checks the counterexample for recovered charge, and `code/verify_theory.py` runs algebraic and numerical checks. | `data/counterexample.csv`; generated JSON reports under the run's `verification/` |

The fixed inputs under `data/original/` are independent comparison records for the fine-mesh audit. Core finite-element routines and regression tests are under `code/legacy/`. The two literature benchmarks are from [Harasim and Valdman (1D)](https://dml.cz/handle/10338.dmlcz/143522) and [Harasim and Valdman (2D)](https://doi.org/10.14736/kyb-2014-6-0978).

## Rebuild the figures

Install pdfLaTeX with TikZ in addition to the Python dependencies. Then run:

```sh
sh reproduce.sh figures
```

The command writes four English figure PDFs under the run's `figure/` directory. Figures 1 and 4 are generated from the archived CSVs. Figure 2 is a geometry diagram. Figure 3 uses the vector coordinates saved with the manuscript. The numerical checker reads these coordinates back and verifies all 25 main intervals, 25 relative widths and five inset intervals.

## Numerical comparison

`verification/compare_results.py` compares the 18 archived output CSVs with the files in a run directory, matching field names, row order and values with an absolute tolerance of 1e-9. Timing columns vary with hardware and are excluded. `verification/check_paper.py` separately checks values at the precision printed in the manuscript, using the table rows and short numerical excerpts under `verification/`. For the partial electrode, the finest-mesh recovered charge is a numerical reference. Analytical charge values are available for the one-dimensional and full-edge benchmarks.
