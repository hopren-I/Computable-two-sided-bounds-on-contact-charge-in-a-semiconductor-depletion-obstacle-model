"""Shared paths; run_all redirects generated files to a staging directory."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("EXPERIMENTS_RESULTS_DIR", str(ROOT / "results"))).resolve()
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
REPORT = RESULTS.parent / "report" / "report.md"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
