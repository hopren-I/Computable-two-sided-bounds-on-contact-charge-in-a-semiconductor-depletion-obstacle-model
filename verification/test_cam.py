"""Fast regression checks for the CAM manuscript's numerical additions."""
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code" / "legacy")]

from cam.bernstein import integrate, mesh_lines, solve_case
from cam.intervals import single_bias


class CamChecks(unittest.TestCase):
    def test_quadrature_on_graded_cells(self):
        x, y = mesh_lines(8, 8, True)

        def affine_controls(edges):
            values = np.empty(2 * len(edges) - 1)
            values[::2] = edges
            values[1::2] = (edges[:-1] + edges[1:]) / 2
            return values

        field = affine_controls(x)[:, None] + 2 * affine_controls(y)[None, :]
        T, L = integrate(x, y, field)
        self.assertAlmostEqual(T, 10., places=12)
        self.assertAlmostEqual(L, 4., places=12)

    def test_graded_contact_reaction(self):
        field = solve_case(16, 8, True)
        self.assertAlmostEqual(field["reaction"], 0.8979021, places=7)
        self.assertAlmostEqual(field["E"], 0.285198477, places=9)

    def test_step_at_zero_bias_endpoint(self):
        interval = single_bias(1., 2., 0., 0.)
        neighbors = []
        def evaluate(bias):
            neighbors.append(bias)
            return 3.8
        self.assertEqual(interval.step, 1.)
        self.assertEqual(interval.refine(evaluate), (0., 3.8))
        self.assertEqual(neighbors, [2.])

    def test_interior_step_preserves_better_side_bounds(self):
        interval = single_bias(1., 2., 0., .75)
        self.assertEqual(interval.step, .5)
        self.assertEqual(interval.refine(lambda b: 1e3), interval.endpoints)
        self.assertEqual(interval.refine(lambda b: .2 if b < 1. else 2.2)[0], 1.1)

    def test_certified_zero_gap(self):
        with self.assertRaises(ArithmeticError):
            single_bias(1., 2., 0., 1.)
        interval = single_bias(1., 2., 0., 1., certified_zero_gap=True)
        self.assertEqual(interval.refine(lambda b: self.fail("Extra solve")), (2., 2.))

    def test_section_5_5_feasible_path(self):
        with TemporaryDirectory() as scratch, patch.dict(os.environ, EXPERIMENTS_RESULTS_DIR=scratch):
            from src.priority_experiments import counterexample_rows
            path = [r for r in counterexample_rows(64)
                    if r["case"] == "varying_multiplier_recovery"]
        self.assertEqual(len(path), 9)
        for r in path:
            self.assertAlmostEqual(r["E"], .3333740234375, places=13)
            for name in ("analytic_Q_residual", "analytic_D_residual", "equilibrium_residual"):
                self.assertLess(abs(r[name]), 1e-12)
        for left, right in zip(path, path[1:]):
            self.assertGreater(left["gap"], right["gap"])
            self.assertLess(left["recovered_charge_abs_error"], right["recovered_charge_abs_error"])


if __name__ == "__main__":
    unittest.main()
