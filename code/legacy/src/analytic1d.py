"""Analytic branches for the dimensionless depletion obstacle model.

Problem (partial depletion, 0 < b < ell^2/2):
    -u'' + 1 - lambda = 0,  u >= 0, lambda >= 0, u lambda = 0,
    u(0) = b, u(ell) = 0.
Full depletion (b >= ell^2/2) keeps the same Dirichlet data but lambda = 0
everywhere and the right-end field no longer vanishes.
"""
from __future__ import annotations

import numpy as np


def analytic(x, b: float, ell: float = 2.0) -> dict:
    x = np.asarray(x, dtype=float)
    if b < 0:
        raise ValueError("b must be nonnegative")
    if b < 0.5 * ell * ell:
        w = np.sqrt(2.0 * b)
        u = 0.5 * np.maximum(w - x, 0.0) ** 2
        lam = np.where(x < w, 0.0, 1.0)
        Q = np.sqrt(2.0 * b)
        C = 1.0 / np.sqrt(2.0 * b) if b > 0 else np.inf
        energy = (2.0 * b) ** 1.5 / 3.0
        branch = "partial"
        width = w
    else:
        w = ell
        u = (ell - x) * (b / ell - x / 2.0)
        lam = np.zeros_like(x)
        Q = b / ell + ell / 2.0
        C = 1.0 / ell
        A = ell / 2.0 + b / ell
        energy = ((ell - A) ** 3 + A**3) / 6.0 + b * ell / 2.0 - ell**3 / 12.0
        branch = "full"
        width = ell
    return {
        "u": u,
        "lambda": lam,
        "Q": float(Q),
        "C": float(C),
        "energy": float(energy),
        "width": float(width),
        "branch": branch,
    }


def analytic_energy(b: float, ell: float = 2.0) -> float:
    return analytic(np.array([0.0]), b, ell)["energy"]


if __name__ == "__main__":
    for b in (0.1, 0.5, 1.0, 1.99, 2.0, 2.5, 3.0):
        a = analytic(np.linspace(0, 2, 5), b)
        print(f"b={b:5.2f} branch={a['branch']:7s} Q={a['Q']:.6f} "
              f"C={a['C']:.6f} E={a['energy']:.6f} w={a['width']:.6f}")
