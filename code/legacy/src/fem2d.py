"""Structured Q1 primal and RT0/Q0 mixed discretisations on a rectangle.

Geometry: Omega = (0, Lx) x (0, Ly).
Boundary: left contact Gamma_c (full or centred segment), right bulk Gamma_0
(u = 0), remaining boundary Gamma_N (insulating, natural / zero normal flux).

Primal Q1 obstacle problem:
    min 1/2 u^T K u + f^T u   s.t. u >= 0, Dirichlet data on Gamma_c, Gamma_0.

Dual RT0/Q0 mixed problem for a prescribed nonnegative cell multiplier lambda:
    M sigma - b q + B^T p = 0
    B sigma = 1 - lambda,
with B the cell-average divergence matrix and q the contact normal-flux
functional.  The dual energy is D_b(sigma) = b q^T sigma - 1/2 sigma^T M sigma.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

from .fem1d import solve_obstacle


@dataclass(frozen=True)
class Mesh2D:
    nx: int
    ny: int
    lx: float = 2.0
    ly: float = 1.0

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2 or self.lx <= 0 or self.ly <= 0:
            raise ValueError("invalid rectangular mesh")

    @property
    def hx(self) -> float:
        return self.lx / self.nx

    @property
    def hy(self) -> float:
        return self.ly / self.ny

    def node(self, i: int, j: int) -> int:
        return i * (self.ny + 1) + j

    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    def xs(self) -> np.ndarray:
        return np.linspace(0.0, self.lx, self.nx + 1)

    def ys(self) -> np.ndarray:
        return np.linspace(0.0, self.ly, self.ny + 1)


@dataclass
class Q1Solution:
    mesh: Mesh2D
    K: sp.csr_matrix
    f: np.ndarray
    u: np.ndarray
    active: np.ndarray
    gradient: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray


def contact_j_range(mesh: Mesh2D, y0: float, y1: float) -> tuple[int, int]:
    """Inclusive node range on the left boundary inside the contact."""
    j0 = int(round(y0 / mesh.hy))
    j1 = int(round(y1 / mesh.hy))
    if not (0 <= y0 < y1 <= mesh.ly) or not np.allclose(
            [j0*mesh.hy, j1*mesh.hy], [y0, y1], atol=1e-12, rtol=0):
        raise ValueError("contact must have positive width and align with mesh faces")
    return j0, j1


def cell_nodes(mesh: Mesh2D) -> np.ndarray:
    i, j = np.meshgrid(np.arange(mesh.nx), np.arange(mesh.ny), indexing="ij")
    k = (i*(mesh.ny+1)+j).ravel()
    return np.stack([k, k+mesh.ny+1, k+mesh.ny+2, k+1], axis=1)


@lru_cache(maxsize=4)
def assemble_q1(mesh: Mesh2D) -> tuple[sp.csr_matrix, np.ndarray]:
    """Exact 2x2 Gauss local matrix, vectorized uniform-mesh assembly."""
    a, b = .5-.5/np.sqrt(3), .5+.5/np.sqrt(3)
    local = np.zeros((4,4))
    for x, y in [(a,a),(b,a),(b,b),(a,b)]:
        dx = np.array([-(1-y),1-y,y,-y])/mesh.hx
        dy = np.array([-(1-x),-x,x,1-x])/mesh.hy
        local += mesh.hx*mesh.hy/4 * (np.outer(dx,dx)+np.outer(dy,dy))
    nodes = cell_nodes(mesh)
    rows = np.repeat(nodes,4,axis=1).ravel()
    cols = np.tile(nodes,(1,4)).ravel()
    K = sp.coo_matrix((np.tile(local.ravel(),len(nodes)),(rows,cols)),
                     shape=(mesh.n_nodes(),mesh.n_nodes())).tocsr()
    f = np.bincount(nodes.ravel(), minlength=mesh.n_nodes()) * mesh.hx*mesh.hy/4
    return K, f


def dirichlet_nodes(mesh: Mesh2D, y0: float, y1: float,
                    b: float) -> tuple[np.ndarray, np.ndarray]:
    j0, j1 = contact_j_range(mesh, y0, y1)
    contact = [mesh.node(0, j) for j in range(j0, j1 + 1)]
    bulk = [mesh.node(mesh.nx, j) for j in range(mesh.ny + 1)]
    fixed = np.array(contact + bulk, dtype=int)
    values = np.array([b] * len(contact) + [0.0] * len(bulk), dtype=float)
    return fixed, values


def solve_q1(mesh: Mesh2D, b: float, y0: float = 0.0, y1: float | None = None
             ) -> Q1Solution:
    if y1 is None:
        y1 = mesh.ly
    K, f = assemble_q1(mesh)
    fixed, values = dirichlet_nodes(mesh, y0, y1, b)
    u, active, g = solve_obstacle(K, f, fixed, values, poisson_start=True)
    return Q1Solution(mesh=mesh, K=K, f=f, u=u, active=active, gradient=g,
                      fixed=fixed, fixed_values=values)


def lambda_from_q1(sol: Q1Solution, tol: float = 1e-10,
                   clip: bool = True) -> np.ndarray:
    """Admissible Q0 reconstruction from assembled nodal KKT densities.

    f_i = integral N_i is the assembled nodal volume. Inactive densities are
    zero. Dirichlet reactions are excluded and extended from the adjacent
    interior x-node; average ALL four densities, not only active vertices.
    There is no artificial upper clipping at one. This is a reconstruction,
    not the exact multiplier of the continuous or mixed obstacle problem.
    """
    density = np.maximum(sol.gradient / sol.f, 0.0)
    density[sol.u > tol] = 0.0
    m = sol.mesh
    for k in sol.fixed:
        i, j = divmod(int(k), m.ny+1)
        adjacent = m.node(1 if i == 0 else m.nx-1, j)
        density[k] = density[adjacent]
    lam = density[cell_nodes(m)].mean(axis=1)
    return np.maximum(lam,0.0) if clip else lam


# ---------------------------------------------------------------------------
# RT0 / Q0 mixed discretisation
# ---------------------------------------------------------------------------

@dataclass
class Mixed2D:
    mesh: Mesh2D
    M: sp.csr_matrix          # retained-face mass matrix
    B: sp.csr_matrix          # cell x retained-face divergence
    q: np.ndarray             # contact flux functional
    retained: np.ndarray      # retained face indices (original numbering)
    n_faces: int
    n_cells: int

    def factor(self):
        if not hasattr(self, "_factor"):
            self._factor = splu(sp.bmat([[self.M,self.B.T],[self.B,None]],format="csc"))
        return self._factor

    def solve(self, lam: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Solve the mixed system for fixed lambda; return sigma, p, D_b(sigma)."""
        nf, nc = self.M.shape[0], self.n_cells
        rhs = np.concatenate([b * self.q, 1.0 - lam])
        sol = self.factor().solve(rhs)
        sigma = sol[:nf]
        p = sol[nf:]
        D = float(b * self.q @ sigma - 0.5 * sigma @ (self.M @ sigma))
        return sigma, p, D

    def harmonic(self) -> tuple[np.ndarray, np.ndarray, float]:
        nf, nc = self.M.shape[0], self.n_cells
        rhs = np.concatenate([self.q, np.zeros(nc)])
        sol = self.factor().solve(rhs)
        H = sol[:nf]
        p_eta = sol[nf:]
        kappa = float(self.q @ H)
        return H, p_eta, kappa


def assemble_rt0(mesh: Mesh2D, y0: float = 0.0, y1: float | None = None
                 ) -> Mixed2D:
    if y1 is None:
        y1 = mesh.ly
    nx, ny = mesh.nx, mesh.ny
    hx, hy = mesh.hx, mesh.hy
    nv = (nx + 1) * ny
    nh = nx * (ny + 1)
    nf = nv + nh

    def vid(i, j):
        return i * ny + j

    def hid(i, j):
        return nv + i * (ny + 1) + j

    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    faces = np.stack([i*ny+j, (i+1)*ny+j,
                      nv+i*(ny+1)+j, nv+i*(ny+1)+j+1],axis=-1).reshape(-1,4)
    local = hx*hy/6 * np.array([[2,1,0,0],[1,2,0,0],[0,0,2,1],[0,0,1,2]])
    rows = np.repeat(faces,4,axis=1).ravel()
    cols = np.tile(faces,(1,4)).ravel()
    M_full = sp.coo_matrix((np.tile(local.ravel(),nx*ny),(rows,cols)),shape=(nf,nf)).tocsr()
    div = np.array([-1/hx,1/hx,-1/hy,1/hy])
    B_full = sp.coo_matrix((np.tile(div,nx*ny),
                          (np.repeat(np.arange(nx*ny),4),faces.ravel())),
                          shape=(nx*ny,nf)).tocsr()

    # Remove insulating faces
    j0, j1 = contact_j_range(mesh, y0, y1)
    contact_faces = {vid(0, j) for j in range(j0, j1)}
    removed = set()
    for j in range(ny):
        if j not in range(j0, j1):
            removed.add(vid(0, j))
    for i in range(nx):
        removed.add(hid(i, 0))
        removed.add(hid(i, ny))
    retained = np.array([f for f in range(nf) if f not in removed], dtype=int)
    M = M_full[retained][:, retained].tocsr()
    B = B_full[:, retained].tocsr()

    q = np.zeros(len(retained))
    face_to_col = {int(f): k for k, f in enumerate(retained)}
    for f in contact_faces:
        if f in face_to_col:
            # outward normal at the left contact is -x, so the boundary
            # functional carries a minus sign relative to the +x face DOF
            q[face_to_col[f]] = -hy
    return Mixed2D(mesh=mesh, M=M, B=B, q=q, retained=retained,
                   n_faces=len(retained), n_cells=nx * ny)


def discrete_reciprocity(mixed: Mixed2D, lam: np.ndarray, b: float,
                         lumped: bool = False) -> dict:
    M = mixed.M
    if lumped:
        M = sp.diags(np.asarray(M.sum(axis=1)).ravel())
    nf, nc = M.shape[0], mixed.n_cells
    A = sp.bmat([[M, mixed.B.T], [mixed.B, None]], format="csc")
    rhs = np.concatenate([b * mixed.q, 1.0 - lam])
    sol = spsolve(A, rhs)
    sigma = sol[:nf]
    p = sol[nf:]
    # harmonic system with the same metric
    A_h = sp.bmat([[M, mixed.B.T], [mixed.B, None]], format="csc")
    sol_h = spsolve(A_h, np.concatenate([mixed.q, np.zeros(nc)]))
    H = sol_h[:nf]
    p_eta = sol_h[nf:]
    kappa = float(mixed.q @ H)
    Q_sigma = float(mixed.q @ sigma)
    Q_h = float(b * kappa + p_eta @ (1.0 - lam))
    return {
        "Q_sigma": Q_sigma,
        "Q_harmonic_identity": Q_h,
        "reciprocity_residual": Q_sigma - Q_h,
        "kappa": kappa,
        "sigma": sigma,
        "p": p,
        "p_eta": p_eta,
    }
