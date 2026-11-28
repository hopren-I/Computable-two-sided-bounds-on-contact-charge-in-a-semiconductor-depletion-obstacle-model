"""Conforming tensor-product quadratic Bernstein discretization (CAM section 5.3)."""
import numpy as np
from scipy import sparse
from src.fem1d import kkt_residuals, solve_obstacle


def mesh_lines(nx, ny, graded):
    if nx < 1 or ny < 4 or ny % 4:
        raise ValueError("The vertical cell count must be a positive multiple of four.")
    x = np.linspace(0., 1., nx + 1)
    if not graded:
        return 2*x, np.linspace(0., 1., ny + 1)
    t = np.linspace(0., 1., ny//4 + 1)
    lower = np.r_[(1-(1-t)**2)/4, (.25 + t**2/4)[1:]]
    return 2*x**2, np.r_[lower, 1-lower[-2::-1]]


def line_matrices(edges):
    """Assemble 1D mass, stiffness and load in the shared control-point basis."""
    lengths = np.diff(edges)
    nodes = 2*np.arange(len(lengths))[:, None] + np.arange(3)
    row = np.repeat(nodes, 3, axis=1).ravel()
    col = np.tile(nodes, (1, 3)).ravel()
    size = 2*len(lengths) + 1
    mass_ref = np.array([[6,3,1], [3,4,3], [1,3,6]])/30
    stiff_ref = np.array([[4,-2,-2], [-2,4,-2], [-2,-2,4]])/3
    def assemble(entries):
        return sparse.coo_matrix((entries.ravel(), (row,col)), shape=(size,size)).tocsr()
    mass = assemble(lengths[:,None,None]*mass_ref)
    stiffness = assemble(stiff_ref/lengths[:,None,None])
    load = np.bincount(nodes.ravel(), weights=np.repeat(lengths/3, 3))
    return mass, stiffness, load


def integrate(x, y, controls):
    """Accumulate |grad v|² and v directly at tensor Gauss points."""
    hx, hy = np.diff(x)[:, None], np.diff(y)[None, :]
    area = hx*hy
    local = np.stack([controls[i:i+2*len(hx):2, j:j+2*hy.size:2]
                      for i in range(3) for j in range(3)], axis=-1)
    points, weights = np.polynomial.legendre.leggauss(3)
    basis = []
    for t in (points+1)/2:
        basis.append((np.array([(1-t)**2, 2*t*(1-t), t*t]),
                      np.array([-2*(1-t), 2-4*t, 2*t])))
    T = L = 0.
    for (bx, dx), wx in zip(basis, weights/2):
        for (by, dy), wy in zip(basis, weights/2):
            v = local @ np.kron(bx, by)
            vx = (local @ np.kron(dx, by))/hx
            vy = (local @ np.kron(bx, dy))/hy
            T += wx*wy*np.sum(area*(vx*vx+vy*vy))
            L += wx*wy*np.sum(area*v)
    return float(T), float(L)


def solve_case(nx, ny, graded, bias=.5):
    x, y = mesh_lines(nx, ny, graded)
    mx, kx, fx = line_matrices(x)
    my, ky, fy = line_matrices(y)
    K = (sparse.kron(kx, my, format="csr") + sparse.kron(mx, ky, format="csr"))
    f = np.kron(fx, fy)
    # Each refitted edge has two control intervals; Dirichlet coefficients
    # include endpoints and all edge-interior Bernstein coefficients.
    contact = np.arange(ny//2, 3*ny//2+1)
    ground = 2*nx*(2*ny+1) + np.arange(2*ny+1)
    fixed = np.r_[contact, ground]
    values = np.r_[np.full(contact.size,bias), np.zeros(ground.size)]
    u, _, residual = solve_obstacle(K, f, fixed, values, poisson_start=True)
    reaction = float(residual[contact].sum())
    T, L = integrate(x, y, u.reshape(2*nx+1, 2*ny+1))
    E = T/2 + L
    energy_residual = abs(E - (.5*u@(K@u) + f@u))
    work_residual = abs((T+L)/bias - reaction)
    kkt = max(kkt_residuals(K,f,u,fixed,values).values())
    if kkt >= 2e-9 or energy_residual > 2e-10 or work_residual > 2e-9:
        raise ArithmeticError(f"Q2 check failed: {nx,ny,graded,kkt,energy_residual,work_residual}")
    return dict(nx=nx, ny=ny, graded=graded, T=T, L=L, E=E, reaction=reaction,
                kkt=kkt, energy_residual=energy_residual, work_residual=work_residual)

