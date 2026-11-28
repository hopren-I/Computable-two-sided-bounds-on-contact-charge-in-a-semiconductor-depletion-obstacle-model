# Single-bias primal-dual bounds for electrode charge in a depletion obstacle model

This repository reproduces Tables 1–6, the numerical comparisons in Section 5, and the boundary-geometry diagram in Figure 1 of the CAM manuscript. All numerical inputs are specified in the source. No external dataset or manuscript files are required.

## Recompute the CAM results

With Python 3.12 in an activated environment, run these commands from the repository root on Windows, macOS, or Linux:

```sh
python -m pip install -r requirements.txt
python reproduce_cam.py
```

The program creates a new directory under `runs/` and prints its location. To choose a new output directory, use `python reproduce_cam.py --output my-cam-run`. The largest solve assembles the mixed system on a 1024 × 512 mesh and uses sparse direct factorization. Allow ample memory and several minutes. The earlier implementation recorded about 15.3 GB of peak memory on a related fine-mesh run.

The solver generates the fields and table values before comparing them with the printed manuscript. The command exits with an error if a field check or numerical comparison fails. A verified full run is distributed in [data/cam](data/cam). Its [comparison.csv](data/cam/comparison.csv) gives the published value, computed value, and outcome for each comparison. [summary.json](data/cam/summary.json) records software versions, elapsed time, and numerical results quoted in the text.

## Experiments and outputs

| Section | Output | Calculation |
| --- | --- | --- |
| 5.1 | `table1.csv` | Exact charge and both interval endpoints at seven biases on 128 linear elements. |
| 5.2 | `table2.csv` | Partial electrode on 128 × 64 cells. Compares the central interval, its refinement at the predicted step, and a scan over three prescribed steps. |
| 5.2 | `table3.csv` | Six fresh bilinear-potential / RT0-flux solves up to 1024 × 512. Width percentages use the recomputed 512 × 256 charge. |
| 5.3 | `table4.csv` | Contact reactions on uniform and graded grids and the graded primal energies. Five levels and two mesh families give ten quadratic Bernstein solves. |
| 5.3 | `summary.json` | Combines the 256 × 128 graded quadratic potential with the independently computed 1024 × 512 RT0 dual field. Reports the gap, endpoints, step and normalized width. |
| 5.4 | `table5.csv` | Full-edge fixed-step bounds, midpoint errors and width decomposition. Each CSV row is one bias, corresponding to a column in the article. |
| 5.5 | `table6.csv` | Recovered charge and error along the feasible dual path for the 64-element one-dimensional problem. |

The full run also writes the central fields and their residuals to `matched_centers.csv`, `refinement_fields.csv`, `bernstein_fields.csv` and `full_edge_fields.csv`. Neighboring energies used in Section 5.2 are in `matched_neighbors.csv`. These files hold scalar diagnostics for each solve. The implementation is in [code/cam](code/cam) and reuses the finite-element and flux solvers in [code/legacy/src](code/legacy/src).

## Numerical verification

The printed values used for verification are in [verification/cam](verification/cam). Table entries are compared to half a unit in the last printed place. The symbol “≈0” in Table 5 means absolute value below 10⁻¹². For the midpoint error at bias 3 and its percentage, the checker also accepts an absolute charge-error difference up to 10⁻¹¹, scaled into percentage units for the latter. Both entries depend on the same subtraction close to machine precision. The comparison records flag any use of this tolerance. For the prose description “about 1.5 × 10⁻⁵”, the absolute tolerance is 10⁻⁶.

The program also checks interval containment, width decomposition, equilibrium and optimality residuals. For quadratic elements it checks energies by independent tensor Gauss integration and compares the work identity with the assembled contact reaction. These are floating-point checks. The targeted CAM regressions run in seconds:

```sh
python -m unittest discover -s verification -p test_cam.py -v
```

## Geometry figure and earlier manuscript

[figure/CAM-Fig1.tex](figure/CAM-Fig1.tex) is the standalone TikZ source for CAM Figure 1. The [compiled figure](figure/CAM-Fig1.pdf) is included. To regenerate it, run `pdflatex -halt-on-error CAM-Fig1.tex` from the `figure/` directory with a LaTeX installation that includes TikZ and standalone.

The [earlier manuscript README](README_ERA.md) documents the original tables, scripts and archived data. The historical shell commands retain their original behavior. On systems with a POSIX shell, `sh reproduce.sh cam` calls the same CAM entry point.
