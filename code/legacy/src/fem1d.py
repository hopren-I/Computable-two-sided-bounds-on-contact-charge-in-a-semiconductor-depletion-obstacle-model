"""One-dimensional P1 finite-element obstacle solver and dual reconstruction.

The primal problem is the bound-constrained convex quadratic programme
    min 1/2 u^T K u + f^T u   subject to  u_i >= 0 on free nodes.
An active-set method solves it and returns the nodal KKT multipliers, from
which a nonnegative cell multiplier is reconstructed.  An equilibrated flux
sigma = c + F, F' = 1 - lambda_hat, is built and used for the primal-dual gap.

All integrals are evaluated exactly for the represented piecewise-polynomial
fields (P1 potential, piecewise-linear flux, P0 multiplier).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


@dataclass
class FEM1D:
    x: np.ndarray
    K: sp.csr_matrix
    f: np.ndarray
    u: np.ndarray
    active: np.ndarray
    gradient: np.ndarray

    @property
    def h(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def energy(self) -> float:
        return float(0.5 * self.u @ (self.K @ self.u) + self.f @ self.u)


def assemble(n: int, ell: float = 2.0) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    x = np.linspace(0.0, ell, n + 1)
    h = ell / n
    rows, cols, vals = [], [], []
    f = np.zeros(n + 1)
    for e in range(n):
        i, j = e, e + 1
        ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        for a in range(2):
            for b in range(2):
                rows.append(i + a)
                cols.append(i + b)
                vals.append(ke[a, b])
        f[i] += h / 2.0
        f[j] += h / 2.0
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    return x, K, f


def solve_obstacle(K: sp.spmatrix, f: np.ndarray,
                   fixed: np.ndarray, fixed_values: np.ndarray,
                   tol: float = 1e-13, max_iter: int = 10000,
                   poisson_start: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Active-set solution of the bound-constrained QP.

    Returns (u, active_nodes, gradient).  ``gradient = K u + f`` equals the
    nonnegative nodal KKT multiplier on active nodes and vanishes elsewhere.
    """
    n = len(f)
    K = sp.csr_matrix(K)
    fixed = np.asarray(fixed, dtype=int)
    fixed_values = np.asarray(fixed_values, dtype=float)
    u = np.zeros(n)
    u[fixed] = fixed_values
    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[fixed] = True
    if np.any(fixed_values < 0) or not np.all(np.isfinite(fixed_values)):
        raise ValueError("Dirichlet values must be finite and nonnegative")
    diag = K.diagonal()
    if np.any(diag <= 0):
        raise ValueError("positive diagonal required")
    # Primal-dual active-set (policy) iteration. Reset constrained values BEFORE
    # assembling the next RHS. Batch updates avoid one sparse factorization per
    # active node on fine meshes. Fail explicitly if KKT acceptance is not met.
    active = ~is_fixed
    if poisson_start:
        # Initialize the active set from the unconstrained Poisson solution.
        # This avoids releasing a thin layer per iteration at high bias. The
        # candidate is NOT accepted as an obstacle solution without the same
        # active-set iterations and final KKT checks used by the cold start.
        initial_free = np.flatnonzero(~is_fixed)
        if len(initial_free):
            rhs = -f[initial_free] - K[initial_free][:, fixed] @ fixed_values
            u[initial_free] = spsolve(K[initial_free][:, initial_free].tocsc(), rhs)
            if not np.all(np.isfinite(u[initial_free])):
                raise RuntimeError('nonfinite Poisson initialization')
        active = (~is_fixed) & (u <= 0.0)
    scale = max(1.0, float(np.max(np.abs(f))))
    accept = max(50 * tol * scale, 5e-12)
    for _ in range(max_iter):
        u[active] = 0.0
        u[fixed] = fixed_values
        free_nodes = np.flatnonzero(~(active | is_fixed))
        if len(free_nodes):
            rhs = -f[free_nodes] - K[free_nodes][:, fixed] @ fixed_values
            u[free_nodes] = spsolve(K[free_nodes][:, free_nodes].tocsc(), rhs)
        g = np.asarray(K @ u + f)
        new_active = (~is_fixed) & (u - g / diag <= 0.0)
        if np.array_equal(new_active, active):
            u[~is_fixed] = np.maximum(u[~is_fixed], 0.0)
            g = np.asarray(K @ u + f)
            checks = kkt_residuals(K, f, u, fixed, fixed_values)
            if max(checks.values()) <= accept:
                return u, np.flatnonzero(active), g
            raise RuntimeError(f"active set stalled without KKT accuracy: {checks}")
        active = new_active
    raise RuntimeError(f"obstacle solver did not converge in {max_iter} iterations")


def kkt_residuals(K, f, u, fixed, values) -> dict:
    """Algebraic, unscaled residuals; fixed-node reactions are NOT multipliers."""
    free = np.ones(len(u), dtype=bool)
    free[np.asarray(fixed, dtype=int)] = False
    g = np.asarray(K @ u + f)
    pos = free & (u > 1e-11)
    zero = free & ~pos
    mx = lambda a: float(np.max(a, initial=0.0))
    return {"primal_violation": mx(-u[free]),
            "boundary_residual": mx(np.abs(u[fixed] - values)),
            "stationarity_residual": mx(np.abs(g[pos])),
            "dual_violation": mx(-g[zero]),
            "complementarity_residual": mx(np.abs(u[free] * g[free]))}

def solve(n: int, b: float, ell: float = 2.0) -> FEM1D:
    if n < 2 or ell <= 0 or b < 0:
        raise ValueError("require n >= 2, ell > 0 and b >= 0")
    x, K, f = assemble(n, ell)
    u, active, g = solve_obstacle(K, f, np.array([0, n]), np.array([b, 0.0]))
    return FEM1D(x=x, K=K, f=f, u=u, active=active, gradient=g)


def cell_multiplier(sol: FEM1D, clipped: bool = True) -> np.ndarray:
    """Nonnegative P0 multiplier from nodal KKT multipliers.

    Divide assembled KKT residuals by assembled nodal volumes, then take the
    cell average of the nodal density. Dirichlet reactions are excluded; their
    density is extended from the nearest unconstrained node. This is an
    admissible reconstruction, not an exact cellwise KKT multiplier.
    """
    density = np.maximum(sol.gradient / sol.f, 0.0)
    density[sol.u > 1e-10] = 0.0
    density[0], density[-1] = density[1], density[-2]
    lam = 0.5 * (density[:-1] + density[1:])
    if clipped:
        lam = np.clip(lam, 0.0, None)
    return lam


def reconstruct_flux(lam_cell: np.ndarray, x: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Equilibrated flux sigma = c + F with F' = 1 - lambda_hat.

    The constant c enforces the exact boundary potential drop:
        int_0^ell sigma dx = -b.
    """
    h = float(x[1] - x[0])
    n = len(lam_cell)
    F = np.zeros(n + 1)
    F[1:] = np.cumsum(h * (1.0 - lam_cell))
    int_F = float(np.sum(h * 0.5 * (F[:-1] + F[1:])))
    c = (-b - int_F) / float(x[-1])
    return c + F, c


def dual_energy(sigma: np.ndarray, x: np.ndarray, b: float) -> float:
    h = float(x[1] - x[0])
    integ = float(np.sum(h / 3.0 * (sigma[:-1] ** 2 + sigma[:-1] * sigma[1:] + sigma[1:] ** 2)))
    return float(-b * sigma[0] - 0.5 * integ)


def charge_direct(sol: FEM1D) -> float:
    return float(-(sol.u[1] - sol.u[0]) / sol.h)


def charge_balance(lam_cell: np.ndarray, x: np.ndarray, b: float) -> float:
    """Q_bal = b/ell + int_0^ell (1 - x/ell)(1 - lambda_hat) dx."""
    h = float(x[1] - x[0])
    eta = 1.0 - 0.5 * (x[:-1] + x[1:]) / x[-1]
    return float(b / x[-1] + np.sum(h * eta * (1.0 - lam_cell)))


def interface_interval(sol: FEM1D, gap: float, ell: float = 2.0) -> tuple[float, float, float, float]:
    """Return (delta, x_delta, lower, upper) for the first-crossing interval."""
    delta = float(np.sqrt(ell * gap / 2.0))
    if not (0 < sol.u[0] < ell**2 / 2 and 0 < delta < sol.u[0]):
        return delta, np.nan, np.nan, np.nan
    below = np.where(sol.u <= delta)[0]
    if len(below) == 0:
        return delta, np.nan, np.nan, np.nan
    idx = int(below[0])
    if idx == 0:
        x_delta = 0.0
    else:
        # linear interpolation between x[idx-1] (> delta) and x[idx] (<= delta)
        x0, x1 = sol.x[idx - 1], sol.x[idx]
        u0, u1 = sol.u[idx - 1], sol.u[idx]
        x_delta = x0 + (delta - u0) * (x1 - x0) / (u1 - u0) if u1 != u0 else x1
    lower = x_delta
    upper = min(ell, x_delta + 2.0 * np.sqrt(delta))
    return delta, x_delta, lower, upper


def diagnostics(n: int, b: float, ell: float = 2.0) -> dict:
    from .analytic1d import analytic

    sol = solve(n, b, ell)
    ana = analytic(sol.x, b, ell)
    lam = cell_multiplier(sol)
    sigma, c = reconstruct_flux(lam, sol.x, b)
    gap = sol.energy - dual_energy(sigma, sol.x, b)
    delta, xd, lo, hi = interface_interval(sol, max(gap, 0.0), ell)
    l2u, l2g = exact_l2_errors(sol, b, ell)
    slopes = np.diff(sol.u) / sol.h
    a, z = slopes - sigma[:-1], slopes - sigma[1:]
    mismatch = float(np.sum(sol.h / 6 * (a*a + a*z + z*z)))
    complementarity = float(np.sum(sol.h * lam * (sol.u[:-1] + sol.u[1:]) / 2))
    return {
        "n": n,
        "h": sol.h,
        "b": b,
        "branch": ana["branch"],
        "energy": sol.energy,
        "energy_exact": ana["energy"],
        "energy_error": sol.energy - ana["energy"],
        "dual_energy": dual_energy(sigma, sol.x, b),
        "gap": gap,
        "u_L2_error": l2u,
        "grad_error": l2g,
        "flux_mismatch": mismatch,
        "complementarity_integral": complementarity,
        "gap_identity_residual": gap - mismatch - complementarity,
        "charge_flux_residual": charge_balance(lam, sol.x, b) + sigma[0],
        **kkt_residuals(sol.K, sol.f, sol.u, np.array([0, n]), np.array([b, 0.])),
        "Q_direct": charge_direct(sol),
        "Q_balance": charge_balance(lam, sol.x, b),
        "Q_exact": ana["Q"],
        "Q_error": charge_direct(sol) - ana["Q"],
        "Q_balance_error": charge_balance(lam, sol.x, b) - ana["Q"],
        "delta": delta,
        "x_delta": xd,
        "interval_lo": lo,
        "interval_hi": hi,
        "contains_width": bool(lo <= ana["width"] <= hi) if np.isfinite(lo) else False,
        "width_exact": ana["width"],
        "n_active": int(len(sol.active)),
        "lam_max": float(lam.max()) if len(lam) else 0.0,
        "u": sol.u,
        "x": sol.x,
        "lam": lam,
        "sigma": sigma,
        "u_exact": ana["u"],
        "lam_exact": ana["lambda"],
    }


def exact_l2_errors(sol: FEM1D, b: float, ell: float) -> tuple[float, float]:
    """Three-point Gauss, splitting at the analytic interface inside a cell."""
    from .analytic1d import analytic
    gp, gw = np.polynomial.legendre.leggauss(3)
    w = min(np.sqrt(2*b), ell)
    e2 = g2 = 0.0
    for i, (a, z) in enumerate(zip(sol.x[:-1], sol.x[1:])):
        cuts = [a, w, z] if a < w < z else [a, z]
        slope = (sol.u[i+1] - sol.u[i]) / (z-a)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            x = (lo+hi)/2 + (hi-lo)*gp/2
            ue = analytic(x, b, ell)["u"]
            due = -np.maximum(w-x, 0) if b < ell**2/2 else x-ell/2-b/ell
            uh = sol.u[i] + slope*(x-a)
            e2 += (hi-lo)/2 * np.dot(gw, (uh-ue)**2)
            g2 += (hi-lo)/2 * np.dot(gw, (slope-due)**2)
    return float(np.sqrt(e2)), float(np.sqrt(g2))


if __name__ == "__main__":
    from .analytic1d import analytic

    for n in (16, 32, 64, 128):
        d = diagnostics(n, 0.5)
        print(f"n={n:4d} E={d['energy']:.10f} Eex={d['energy_exact']:.10f} "
              f"gap={d['gap']:.3e} Q={d['Q_direct']:.8f} "
              f"Qbal={d['Q_balance']:.8f} int={d['interval_lo']:.4f},"
              f"{d['interval_hi']:.4f} contains={d['contains_width']}")
