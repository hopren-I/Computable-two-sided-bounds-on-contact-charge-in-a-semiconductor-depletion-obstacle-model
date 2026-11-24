"""RT0 comparison routines reused from the isolated pilot, unchanged."""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu,spsolve

def dual(mix, s, a):
    return float(a * (mix.q @ s) - .5 * (s @ (mix.M @ s)))


def make_feasible(mix, s):
    # Scalar contraction preserves homogeneous normal traces and Bs <= 1.
    # Margin only protects the subsequent floating-point evaluation.
    factor = max(1., float(np.max(mix.B @ s))) + 2e-12
    return s / factor, factor


def optimize(mix, a, s0, lam0, maxiter=100):
    """Primal-dual active set; accept by independently evaluated QP gap.

    p is an integrated cell multiplier (area times potential), not lambda.
    For Bs<=1 the KKT sign is p>=0, M s + B.T p = a q.
    Finite termination of this iteration is NOT used in the convergence proof.
    """
    started = time.perf_counter()
    M, B = mix.M, mix.B
    nf, nc = M.shape[0], B.shape[0]
    area = mix.mesh.hx * mix.mesh.hy
    active = lam0 < 1e-8
    seen = set()
    history = []
    for iteration in range(1, maxiter + 1):
        key = active.tobytes()
        if key in seen:
            raise RuntimeError(f'PDAS active-set cycle at iteration {iteration}')
        seen.add(key)
        idx = np.flatnonzero(active)
        BA = B[idx]
        system = sp.bmat([[M, BA.T], [BA, None]], format='csc')
        answer = spsolve(system, np.r_[a * mix.q, np.ones(len(idx))])
        s = answer[:nf]
        p = np.zeros(nc)
        p[idx] = answer[nf:]
        slack = 1 - B @ s
        next_active = p / area - slack > 1e-10
        history.append(dict(iteration=iteration, active=len(idx),
                            feasibility=float(max(0., -slack.min())),
                            negative_p=float(max(0., -p.min()))))
        if np.array_equal(active, next_active):
            break
        active = next_active
    else:
        raise RuntimeError('PDAS iteration limit')
    raw_s = s.copy()
    raw_p = p.copy()
    s, factor = make_feasible(mix, s)
    # Independent upper bound on max D: F(p) for any nonnegative p.
    p = np.maximum(p, 0.)
    t = time.perf_counter()
    mass_factor = splu(M.tocsc())
    z = a * mix.q - B.T @ p
    zsolve = mass_factor.solve(z)
    qp_upper = float(.5 * z @ zsolve + p.sum())
    opt_value = dual(mix, s, a)
    qp_gap = qp_upper - opt_value
    certificate_seconds = time.perf_counter() - t
    raw_stationarity = float(np.max(np.abs(M @ raw_s + B.T @ raw_p - a * mix.q)))
    assert np.all(np.isfinite(s)) and np.min(1-B@s) >= -5e-13
    assert -5e-11 <= qp_gap <= 2e-9, (qp_gap, history)
    assert opt_value >= dual(mix, s0, a) - 5e-11
    return s, dict(opt_seconds=time.perf_counter()-started,
                   opt_certificate_seconds=certificate_seconds,
                   opt_iterations=iteration, opt_linear_solves=iteration+1,
                   opt_qp_upper=qp_upper, opt_qp_gap=qp_gap,
                   opt_raw_stationarity=raw_stationarity,
                   opt_raw_feasibility=float(max(0., np.max(B@raw_s)-1)),
                   opt_raw_negative_p=float(max(0., -raw_p.min())),
                   opt_contraction_factor=factor,
                   opt_active_cells=int(active.sum())), history

