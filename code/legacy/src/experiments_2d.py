"""Two-dimensional experiments: identities, contact geometry, feasible fields.

M3  full/partial contact convergence, harmonic identity, discrete reciprocity
S1  contact-width sweep and the geometric/screening decomposition of charge
M4  feasible-field hierarchy, gap monotonicity, local multiplier responses
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .fem2d import (Mesh2D, assemble_rt0, discrete_reciprocity,
                    lambda_from_q1, solve_q1, cell_nodes, dirichlet_nodes)
from .fem1d import kkt_residuals

from .paths import ROOT, TABLES, FIGS


plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})


def _write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (TABLES / name).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def _q_direct(mesh: Mesh2D, sol, j0: int, j1: int) -> float:
    q = 0.0
    for j in range(j0, j1):
        du0 = (sol.u[mesh.node(1,j)]-sol.u[mesh.node(0,j)])/mesh.hx
        du1 = (sol.u[mesh.node(1,j+1)]-sol.u[mesh.node(0,j+1)])/mesh.hx
        q += -(du0+du1)*mesh.hy/2
    return float(q)


def _cell_eta(mix, p_eta: np.ndarray) -> np.ndarray:
    area = mix.mesh.hx * mix.mesh.hy
    return p_eta / area


def _evaluate(mesh: Mesh2D, y0: float, y1: float, b: float = 0.5, mixed=None) -> dict:
    sol = solve_q1(mesh, b, y0, y1)
    lam = lambda_from_q1(sol)
    mix = assemble_rt0(mesh, y0, y1) if mixed is None else mixed
    H, p_eta, kappa = mix.harmonic()
    sigma, p, D = mix.solve(lam, b)
    Q = float(mix.q @ sigma)
    E = float(0.5 * sol.u @ (sol.K @ sol.u) + sol.f @ sol.u)
    eta = _cell_eta(mix, p_eta)
    screening = float(np.sum(mesh.hx * mesh.hy * eta * (1.0 - lam)))
    Q_h = float(b * kappa + screening)
    j0, j1 = int(round(y0 / mesh.hy)), int(round(y1 / mesh.hy))
    qd = _q_direct(mesh, sol, j0, j1)
    check = quadrature_diagnostics(sol,mix,sigma,lam,b)
    reaction = float(sum(sol.gradient[mesh.node(0,j)] for j in range(j0,j1+1)))
    return {
        "nx": mesh.nx, "ny": mesh.ny, "h": mesh.hx,
        "contact_width": y1 - y0,
        "branch": "full" if y0 <= 1e-14 and y1 >= mesh.ly - 1e-14 else "partial",
        "E": E, "D": D, "gap": E - D,
        "Q_mixed": Q, "Q_harmonic": Q_h, "Q_direct": qd,
        "Q_reaction": reaction,
        "kappa": kappa, "screening_integral": screening,
        "reciprocity_residual": Q - Q_h,
        "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
        "n_active": int(len(sol.active)),
        **check,
        "u": sol.u, "lam": lam, "sigma": sigma, "H": H, "p_eta": p_eta,
        "sol": sol, "mix": mix,
    }


def quadrature_diagnostics(sol, mix, sigma, lam, b):
    """Independent cell quadrature, including the two nonnegative gap terms."""
    m=sol.mesh; nx,ny=m.nx,m.ny; area=m.hx*m.hy
    nodal=sol.u[cell_nodes(m)]
    full=np.zeros((nx+1)*ny+nx*(ny+1));full[mix.retained]=sigma
    sx=full[:(nx+1)*ny].reshape(nx+1,ny)
    sy=full[(nx+1)*ny:].reshape(nx,ny+1)
    sl,sr=sx[:-1].ravel(),sx[1:].ravel()
    sb,st=sy[:,:-1].ravel(),sy[:,1:].ravel()
    a,z=.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)
    mismatch=comp=E=norm=0.
    for x,y in [(a,a),(z,a),(z,z),(a,z)]:
        N=np.array([(1-x)*(1-y),x*(1-y),x*y,(1-x)*y])
        dx=np.array([-(1-y),1-y,y,-y])/m.hx
        dy=np.array([-(1-x),-x,x,1-x])/m.hy
        u=nodal@N;ux=nodal@dx;uy=nodal@dy
        fx=(1-x)*sl+x*sr;fy=(1-y)*sb+y*st
        mismatch+=area/8*np.sum((ux-fx)**2+(uy-fy)**2)
        comp+=area/4*np.sum(u*lam)
        E+=area/4*np.sum(.5*(ux**2+uy**2)+u)
        norm+=area/4*np.sum(fx**2+fy**2)
    D=b*float(mix.q@sigma)-norm/2
    matrix_E=float(.5*sol.u@(sol.K@sol.u)+sol.f@sol.u)
    matrix_D=float(b*mix.q@sigma-.5*sigma@(mix.M@sigma))
    return dict(**kkt_residuals(sol.K,sol.f,sol.u,sol.fixed,sol.fixed_values),
                gap_identity_residual=matrix_E-matrix_D-mismatch-comp,
                energy_quadrature_residual=matrix_E-E,
                dual_quadrature_residual=matrix_D-D,
                equilibrium_residual=float(np.max(np.abs(mix.B@sigma-1+lam))),
                flux_mismatch=float(mismatch),complementarity_integral=float(comp),
                min_u=float(sol.u.min()))


def scalar_result(r):
    return {k:v for k,v in r.items() if np.isscalar(v)}


def run_m3() -> dict:
    full = []
    for nx, ny in [(16, 8), (32, 16), (64, 32), (128, 64)]:
        r = _evaluate(Mesh2D(nx, ny), 0.0, 1.0)
        full.append(scalar_result(r))
    _write_csv("m3_full_contact.csv", full)

    part = []
    for nx, ny in [(32, 16), (64, 32), (128, 64)]:
        r = _evaluate(Mesh2D(nx, ny), 0.25, 0.75)
        part.append(scalar_result(r))
    _write_csv("m3_partial_contact.csv", part)

    # discrete reciprocity: consistent vs lumped metric
    mesh = Mesh2D(32, 16)
    r = _evaluate(mesh, 0.25, 0.75)
    lam = r["lam"]
    cons = discrete_reciprocity(r["mix"], lam, 0.5, lumped=False)
    lump = discrete_reciprocity(r["mix"], lam, 0.5, lumped=True)
    # Cross-metric test: projection with the consistent mass, harmonic
    # quantities with the lumped mass.  The identity should now fail.
    M = r["mix"].M
    B = r["mix"].B
    q = r["mix"].q
    nf, nc = M.shape[0], r["mix"].n_cells
    A_cons = sp.bmat([[M, B.T], [B, None]], format="csc")
    sol_s = spsolve(A_cons, np.concatenate([0.5 * q, 1.0 - lam]))
    sigma_cons = sol_s[:nf]
    M_l = sp.diags(np.asarray(M.sum(axis=1)).ravel())
    A_l = sp.bmat([[M_l, B.T], [B, None]], format="csc")
    sol_h = spsolve(A_l, np.concatenate([q, np.zeros(nc)]))
    H_l = sol_h[:nf]
    p_l = sol_h[nf:]
    kappa_l = float(q @ H_l)
    Q_sigma = float(q @ sigma_cons)
    Q_h_l = float(0.5 * kappa_l + p_l @ (1.0 - lam))
    recip = [{
        "metric": "consistent / consistent",
        "Q_sigma": cons["Q_sigma"],
        "Q_harmonic_identity": cons["Q_harmonic_identity"],
        "residual": cons["reciprocity_residual"],
        "kappa": cons["kappa"],
    }, {
        "metric": "lumped / lumped",
        "Q_sigma": lump["Q_sigma"],
        "Q_harmonic_identity": lump["Q_harmonic_identity"],
        "residual": lump["reciprocity_residual"],
        "kappa": lump["kappa"],
    }, {
        "metric": "consistent sigma / lumped harmonic",
        "Q_sigma": Q_sigma,
        "Q_harmonic_identity": Q_h_l,
        "residual": Q_sigma - Q_h_l,
        "kappa": kappa_l,
    }]
    _write_csv("m3_reciprocity.csv", recip)

    # single harmonic direction: perturb the optimal flux and correct it back
    mix = r["mix"]
    sigma0, p0, D0 = mix.solve(lam, 0.5)
    H, p_eta, kappa = mix.harmonic()
    Q_hat = float(0.5 * kappa + p_eta @ (1.0 - lam))
    c = 0.5 * kappa  # perturbation coefficient
    sigma_p = sigma0 + c * H
    D_p = float(0.5 * mix.q @ sigma_p - 0.5 * sigma_p @ (mix.M @ sigma_p))
    a = (Q_hat - float(mix.q @ sigma_p)) / kappa
    sigma_c = sigma_p + a * H
    D_c = float(0.5 * mix.q @ sigma_c - 0.5 * sigma_c @ (mix.M @ sigma_c))
    gain = D_c - D_p
    formula = (Q_hat - float(mix.q @ sigma_p)) ** 2 / (2 * kappa)
    single = [{
        "perturbation_c": c,
        "Q_before": float(mix.q @ sigma_p),
        "Q_after": float(mix.q @ sigma_c),
        "Q_target": Q_hat,
        "D_before": D_p,
        "D_after": D_c,
        "gain": gain,
        "gain_formula": formula,
        "gain_residual": gain - formula,
    }]
    _write_csv("m3_single_direction.csv", single)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    labels = [f"{r['nx']}×{r['ny']}" for r in full]
    idx = np.arange(len(full))
    ax[0].plot(idx, [r["Q_mixed"] for r in full], "o-", label="Q mixed")
    ax[0].plot(idx, [r["Q_direct"] for r in full], "s--", label="Q direct")
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_xlabel("mesh")
    ax[0].set_ylabel("Q")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) full contact")
    ax[1].plot(np.arange(len(full)), [r["gap"] for r in full], "s--",
               label="full contact gap")
    ax[1].plot(np.arange(1, 1 + len(part)), [r["gap"] for r in part], "o-",
               label="partial contact gap")
    ax[1].set_yscale("log")
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_xlabel("mesh")
    ax[1].set_ylabel("primal-dual gap")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) gap convergence")
    fig.tight_layout()
    fig.savefig(FIGS / "m3_2d.png")
    fig.savefig(FIGS / "m3_2d.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(np.arange(len(full)), [r["kappa"] for r in full], "o-", label="full")
    ax.plot(np.arange(1, 1 + len(part)), [r["kappa"] for r in part], "s-",
            label="partial")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("mesh")
    ax.set_ylabel(r"$\kappa$")
    ax.legend()
    ax.set_title("harmonic geometric factor")
    fig.tight_layout()
    fig.savefig(FIGS / "m3_kappa.png")
    fig.savefig(FIGS / "m3_kappa.pdf")
    plt.close(fig)
    return {"full": full, "partial": part, "reciprocity": recip, "single": single}


def run_s1(meshes=((64, 32), (128, 64)),
           widths=(0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    rows = []
    for nx, ny in meshes:
        mesh = Mesh2D(nx, ny)
        for s in widths:
            j0 = int(round((0.5 - s / 2) * ny))
            j1 = int(round((0.5 + s / 2) * ny))
            y0, y1 = j0 / ny, j1 / ny
            r = _evaluate(mesh, y0, y1)
            rows.append({
                "nx": nx, "ny": ny, "h": mesh.hx, "contact_width": s,
                "Q_mixed": r["Q_mixed"], "Q_direct": r["Q_direct"],
                "kappa": r["kappa"], "screening_integral": r["screening_integral"],
                "geometric_term": 0.5 * r["kappa"],
                "gap": r["gap"],
                **{k:r[k] for k in ("primal_violation","boundary_residual","stationarity_residual",
                    "dual_violation","complementarity_residual","equilibrium_residual","gap_identity_residual")},
            })
    _write_csv("s1_contact_width.csv", rows)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for nx, ny in meshes:
        sub = [r for r in rows if r["nx"] == nx and r["ny"] == ny]
        s = [r["contact_width"] for r in sub]
        ax[0].plot(s, [r["Q_mixed"] for r in sub], "o-", label=f"{nx}x{ny}")
        ax[0].set_xlabel("contact width s (of boundary length 1)")
        ax[0].set_ylabel("terminal charge Q")
        ax[0].legend(fontsize=7)
        ax[0].set_title("(a) charge versus contact width")
        ax[1].plot(s, [r["geometric_term"] for r in sub], "o-",
                   label=f"bκ, {nx}x{ny}")
        ax[1].plot(s, [r["screening_integral"] for r in sub], "s--",
                   label=f"screening, {nx}x{ny}")
        ax[1].set_xlabel("contact width s")
        ax[1].set_ylabel("term")
        ax[1].legend(fontsize=6)
        ax[1].set_title("(b) decomposition Q = bκ + screening")
    fig.tight_layout()
    fig.savefig(FIGS / "s1_contact_width.png")
    fig.savefig(FIGS / "s1_contact_width.pdf")
    plt.close(fig)
    return rows


def run_m4(reference) -> dict:
    # field hierarchy on the partial-contact geometry
    rows = []
    Q_ref = reference["Q_mixed"]

    # linear field: feasible boundary values, no obstacle
    mesh = Mesh2D(128, 64)
    j0, j1 = int(round(0.25 * mesh.ny)), int(round(0.75 * mesh.ny))
    u_lin = np.array([0.5 * (1.0 - mesh.xs()[i] / mesh.lx)
                      for i in range(mesh.nx + 1)])
    u_lin_nodes = np.array([u_lin[i] for i in range(mesh.nx + 1)
                            for _ in range(mesh.ny + 1)])
    # reconstruct a cellwise multiplier from the linear field (all positive)
    lam_lin = np.zeros(mesh.nx * mesh.ny)
    K, f = None, None
    from .fem2d import assemble_q1
    K, f = assemble_q1(mesh)

    def evaluate_field(u_nodes, lam_cells, label, nx, ny):
        E = float(0.5 * u_nodes @ (K @ u_nodes) + f @ u_nodes)
        mix = assemble_rt0(mesh, 0.25, 0.75)
        sigma, p, D = mix.solve(lam_cells, 0.5)
        Q = float(mix.q @ sigma)
        return {
            "field": label, "source_mesh": f"{nx}x{ny}",
            "E": E, "D": D, "gap": E - D, "Q": Q,
            "Q_difference_percent": 100.0 * abs(Q - Q_ref) / abs(Q_ref),
            "reference_nx": reference["nx"], "Q_reference": Q_ref,
        }

    rows.append(evaluate_field(u_lin_nodes, lam_lin, "linear", 0, 0))
    for nx, ny in [(16, 8), (32, 16), (64, 32), (128, 64)]:
        coarse = Mesh2D(nx, ny)
        solc = solve_q1(coarse, 0.5, 0.25, 0.75)
        lamc = lambda_from_q1(solc)
        # interpolate the coarse Q1 potential to the fine 128x64 nodes
        xs, ys = mesh.xs(), mesh.ys()
        xc, yc = coarse.xs(), coarse.ys()
        u_fine = np.zeros(mesh.n_nodes())
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                # bilinear interpolation
                ii = min(np.searchsorted(xc, x, side="right") - 1, nx - 1)
                jj = min(np.searchsorted(yc, y, side="right") - 1, ny - 1)
                ii = max(ii, 0); jj = max(jj, 0)
                tx = (x - xc[ii]) / coarse.hx
                ty = (y - yc[jj]) / coarse.hy
                u00 = solc.u[coarse.node(ii, jj)]
                u10 = solc.u[coarse.node(ii + 1, jj)]
                u01 = solc.u[coarse.node(ii, jj + 1)]
                u11 = solc.u[coarse.node(ii + 1, jj + 1)]
                u_fine[mesh.node(i, j)] = ((1 - tx) * (1 - ty) * u00 +
                                           tx * (1 - ty) * u10 +
                                           tx * ty * u11 +
                                           (1 - tx) * ty * u01)
        # map coarse cell multipliers to the fine cells
        lam_fine = np.zeros(mesh.nx * mesh.ny)
        for i in range(mesh.nx):
            ic = min(int(i * nx / mesh.nx), nx - 1)
            for j in range(mesh.ny):
                jc = min(int(j * ny / mesh.ny), ny - 1)
                lam_fine[i * mesh.ny + j] = lamc[ic * ny + jc]
        rows.append(evaluate_field(u_fine, lam_fine, "coarse Q1", nx, ny))
    _write_csv("m4_hierarchy.csv", rows)

    # local response columns
    mesh = Mesh2D(32, 16)
    r = _evaluate(mesh, 0.25, 0.75)
    mix = r["mix"]
    nf, nc = mix.M.shape[0], mix.n_cells
    A = sp.bmat([[mix.M, mix.B.T], [mix.B, None]], format="csc")
    selected = [0, 1, nc // 2, nc - 1]
    R = np.zeros((nf, len(selected)))
    for idx, Kcell in enumerate(selected):
        rhs = np.zeros(nf + nc)
        rhs[nf + Kcell] = -1.0
        sol = spsolve(A, rhs)
        R[:, idx] = sol[:nf]
    block = np.zeros((nc, len(selected)))
    block[selected, np.arange(len(selected))] = 1.0
    resp_rows = [{
        "n_cells": nc, "n_faces": nf, "selected": len(selected),
        "max_abs_BR_plus_I": float(np.abs(mix.B @ R + block).max()),
        "response_nonzero_fraction": float(np.mean(np.abs(R) > 1e-12)),
        "response_max_abs": float(np.abs(R).max()),
    }]
    _write_csv("m4_local_response.csv", resp_rows)

    # A real 2x2 interior patch: only faces shared by two patch cells carry
    # flux. Extending by zero preserves H(div) conformity and zero patch trace.
    i0,j0=8,6
    patch=np.array([i*mesh.ny+j for i in (i0,i0+1) for j in (j0,j0+1)])
    support=np.asarray((mix.B[patch]!=0).sum(axis=0)).ravel()
    inside=np.flatnonzero(support==2)
    boundary=np.flatnonzero(support==1)
    tau=np.zeros(nf)
    tau[inside]=np.random.default_rng(7).normal(size=len(inside))
    div=mix.B@tau
    outside=np.setdiff1d(np.arange(nc),patch)
    resp_rows[0].update(patch_cells=len(patch),patch_internal_faces=len(inside),
        patch_boundary_flux=float(np.max(np.abs(tau[boundary]))),
        outside_divergence=float(np.max(np.abs(div[outside]))),
        patch_divergence_norm=float(np.linalg.norm(div[patch])),
        patch_net_divergence=float(mesh.hx*mesh.hy*np.sum(div[patch])))
    _write_csv("m4_local_response.csv",resp_rows)
    update_rows=run_local_updates(r)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    labels = [("linear" if r["source_mesh"] == "0x0"
               else f"Q1 {r['source_mesh']}") for r in rows]
    ax[0].bar(labels, [r["gap"] for r in rows])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("primal-dual gap")
    ax[0].tick_params(axis="x", rotation=25)
    ax[0].set_title("(a) gap across field hierarchy")
    ax[1].plot([r["Q"] for r in rows], [r["Q_difference_percent"] for r in rows],
               "o-")
    ax[1].set_xlabel("recovered Q")
    ax[1].set_ylabel("Q difference vs reference (%)")
    ax[1].set_title("(b) charge versus field quality")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_hierarchy.png")
    fig.savefig(FIGS / "m4_hierarchy.pdf")
    plt.close(fig)
    return {"hierarchy": rows, "local": resp_rows, "updates": update_rows}

def run_reference():
    """Fine-mesh reference + independent optimizer + continuous energy secants.

    The finest point value is an approximation, not ground truth. The energy
    secants supply a separate numerical enclosure of the continuum charge.
    """
    from scipy.optimize import minimize
    import gc
    rows=[]
    for nx in (32,64,128,256,512):
        print(f"  reference {nx}x{nx//2}",flush=True)
        r=_evaluate(Mesh2D(nx,nx//2),.25,.75)
        rows.append(scalar_result(r))
        if nx==128:
            m=r['sol'].mesh
            fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
            for ax,arr,title in zip(axes,[r['u'].reshape(nx+1,nx//2+1).T,
                    r['lam'].reshape(nx,nx//2).T,
                    (r['p_eta']/(m.hx*m.hy)).reshape(nx,nx//2).T],
                    ['Q1 potential','Reconstructed multiplier','Discrete harmonic weight']):
                im=ax.imshow(arr,origin='lower',extent=[0,2,0,1],aspect='equal')
                ax.plot([0,0],[.25,.75],color='red',lw=4)
                ax.set(xlabel='x',ylabel='y',title=title);fig.colorbar(im,ax=ax,shrink=.65)
            for ext in ('png','pdf'):fig.savefig(FIGS/f'm3_fields.{ext}')
            plt.close(fig)
        if nx==512:
            fields=TABLES.parent/'fields';fields.mkdir(exist_ok=True)
            np.savez_compressed(fields/'reference_512x256.npz',x=r['sol'].mesh.xs(),
                y=r['sol'].mesh.ys(),u=r['u'],lambda_cell=r['lam'],sigma=r['sigma'],
                retained_faces=r['mix'].retained)
            reference=scalar_result(r)
        del r
        gc.collect()
    _write_csv('m3_reference.csv',rows)
    bounds=[]
    for nx in (256,512):
        mesh=Mesh2D(nx,nx//2);mix=assemble_rt0(mesh,.25,.75)
        cache={}
        for b in (.4,.45,.475,.5,.525,.55,.6):
            print(f"  energy enclosure {nx}x{nx//2}, b={b}",flush=True)
            rr=_evaluate(mesh,.25,.75,b,mixed=mix)
            cache[round(b,8)]=scalar_result(rr)
            del rr
        for h in (.025,.05,.1):
            c=cache[.5];l=cache[round(.5-h,8)];z=cache[round(.5+h,8)]
            lo=(c['D']-l['E'])/h;hi=(z['E']-c['D'])/h
            bounds.append(dict(nx=nx,ny=nx//2,b=.5,h=h,Q_lower=lo,Q_upper=hi,
                width=hi-lo,E_center=c['E'],D_center=c['D'],E_minus=l['E'],
                E_plus=z['E'],Q_reference=reference['Q_mixed'],
                equilibrium_residual=max(c['equilibrium_residual'],l['equilibrium_residual'],z['equilibrium_residual']),
                gap_identity_residual=max(abs(c['gap_identity_residual']),abs(l['gap_identity_residual']),abs(z['gap_identity_residual']))))
        del mix,cache
        gc.collect()
    _write_csv('m3_reference_bounds.csv',bounds)
    cross=[]
    for y0,y1 in [(0.,1.),(.25,.75),(.125,.5)]:
        mesh=Mesh2D(32,16);sol=solve_q1(mesh,.5,y0,y1)
        free=np.setdiff1d(np.arange(mesh.n_nodes()),sol.fixed)
        A=sol.K[free][:,free];f=sol.f[free]+sol.K[free][:,sol.fixed]@sol.fixed_values
        fun=lambda v:float(.5*v@(A@v)+f@v)
        jac=lambda v:np.asarray(A@v+f)
        # Independent initialization and algorithm: no active-set solution used.
        start=np.maximum(.5*(1-np.repeat(mesh.xs()/mesh.lx,mesh.ny+1))[free],0.)
        opt=minimize(fun,start,jac=jac,bounds=[(0,None)]*len(free),method='L-BFGS-B',
            options={'gtol':1e-11,'ftol':1e-15,'maxiter':20000,'maxls':50})
        v=sol.u.copy();v[free]=opt.x
        g=jac(opt.x);pg=np.where(opt.x>1e-9,g,np.minimum(g,0.))
        cross.append(dict(nx=32,ny=16,y0=y0,y1=y1,optimizer_success=bool(opt.success),
            optimizer_message=opt.message,projected_gradient=float(np.max(abs(pg))),
            max_u_difference=float(np.max(abs(sol.u-v))),
            energy_difference=float(fun(sol.u[free])-fun(opt.x))))
    _write_csv('solver_crosscheck_2d.csv',cross)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),layout='constrained')
    for key,label in [('Q_mixed','mixed recovery'),('Q_direct','boundary derivative'),('Q_reaction','discrete energy derivative')]:
        axes[0].semilogx([x['nx'] for x in rows],[x[key] for x in rows],'o-',label=label)
    axes[0].set(xlabel='nx (ny=nx/2)',ylabel='Q',title='Independent charge evaluations');axes[0].legend(fontsize=7)
    for nx in (256,512):
        rr=[x for x in bounds if x['nx']==nx]
        axes[1].errorbar([x['h']*(1.02 if nx==512 else .98) for x in rr],
            [(x['Q_lower']+x['Q_upper'])/2 for x in rr],
            yerr=[x['width']/2 for x in rr],fmt='o',capsize=4,label=f'{nx}x{nx//2}')
    axes[1].axhline(reference['Q_mixed'],ls=':',color='k',label='fine mixed estimate')
    axes[1].set(xlabel='bias step',ylabel='continuum charge enclosure',title='Energy-secant enclosures')
    axes[1].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(FIGS/f'm3_reference.{ext}')
    plt.close(fig)
    return reference


def run_local_updates(r,steps=6):
    """Execute the paper's selected-cell dual QP and feasible primal changes."""
    from scipy.optimize import minimize
    from dataclasses import replace
    mix=r['mix'];sol=r['sol'];m=sol.mesh
    lam=r['lam'].copy();sigma,p,D=mix.solve(lam,.5)
    _,p_eta,_=mix.harmonic()
    xx=np.repeat(m.xs()/m.lx,m.ny+1)
    v=sol.u+.03*xx*(1-xx)
    free=np.setdiff1d(np.arange(len(v)),sol.fixed)
    K,f=sol.K,sol.f
    energy=lambda u:float(.5*u@(K@u)+f@u)
    rows=[]
    for it in range(steps):
        Eold=energy(v);Dold=D;Qold=float(mix.q@sigma)
        # Minimize .5 delta^T C delta + p_S^T delta, lambda_S+delta >= 0.
        score=np.where(lam>1e-12,abs(p),np.maximum(-p,0.))
        selected=np.argsort(score)[-8:]
        nf,nc=mix.M.shape[0],mix.n_cells
        rhs=np.zeros((nf+nc,len(selected)))
        rhs[nf+selected,np.arange(len(selected))]=-1.
        resp=mix.factor().solve(rhs);R=resp[:nf];P=resp[nf:]
        C=R.T@(mix.M@R);g=p[selected]
        scale=max(float(np.max(np.diag(C))),1e-12)
        res=minimize(lambda d:float((.5*d@C@d+g@d)/scale),np.zeros(len(selected)),
            jac=lambda d:(C@d+g)/scale,bounds=[(-lam[k],None) for k in selected],
            method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-11,'maxiter':2000,'maxls':50})
        if not res.success:raise RuntimeError(f'local dual QP: {res.message}')
        delta=np.maximum(res.x,-lam[selected]);predicted=-g@delta-.5*delta@C@delta
        sigma_new=sigma+R@delta
        Dnew=float(.5*mix.q@sigma_new-.5*sigma_new@(mix.M@sigma_new))
        if Dnew < Dold-1e-12:raise RuntimeError('dual update failed energy acceptance')
        lam[selected]+=delta;sigma=sigma_new;p=p+P@delta;D=Dnew
        # Restricted Q1 change, with original nonnegative-node bounds.
        grad=K@v+f
        S=free[np.argsort(abs(grad[free]))[-32:]]
        A=K[S][:,S].toarray();g0=grad[S];scale=max(np.diag(A))
        opt=minimize(lambda z:float((.5*z@A@z+g0@z)/scale),np.zeros(len(S)),
            jac=lambda z:(A@z+g0)/scale,bounds=[(-v[k],None) for k in S],
            method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-11,'maxiter':2000,'maxls':50})
        if not opt.success:raise RuntimeError(f'local primal QP: {opt.message}')
        z=np.maximum(opt.x,-v[S]);predE=.5*z@A@z+g0@z
        v[S]+=z;Enew=energy(v)
        if Enew>Eold+1e-12:raise RuntimeError('primal update failed energy acceptance')
        check=quadrature_diagnostics(replace(sol,u=v),mix,sigma,lam,.5)
        rows.append(dict(iteration=it+1,E_before=Eold,E_after=Enew,D_before=Dold,D_after=D,
            gap_before=Eold-Dold,gap_after=Enew-D,dual_gain=D-Dold,
            predicted_dual_gain=float(predicted),dual_expansion_residual=float(D-Dold-predicted),
            primal_expansion_residual=float(Enew-Eold-predE),
            charge_response_residual=float(mix.q@sigma-Qold+p_eta[selected]@delta),
            response_balance_residual=float(np.max(abs(mix.B@R+np.eye(nc)[:,selected]))),
            lambda_min=float(lam.min()),min_u=float(v.min()),
            boundary_residual=check['boundary_residual'],equilibrium_residual=check['equilibrium_residual'],
            gap_identity_residual=check['gap_identity_residual'],
            dual_qp_projected_gradient=float(np.max(abs(np.where(lam[selected]>1e-10,C@delta+g,np.minimum(C@delta+g,0.)))))))
    _write_csv('m4_updates.csv',rows)
    fig,ax=plt.subplots(1,2,figsize=(7.6,3),layout='constrained')
    its=np.arange(steps+1)
    ax[0].plot(its,[rows[0]['E_before']]+[x['E_after'] for x in rows],'o-',label='primal upper energy')
    ax[0].plot(its,[rows[0]['D_before']]+[x['D_after'] for x in rows],'s-',label='dual lower energy')
    ax[0].set(xlabel='accepted local update',ylabel='energy');ax[0].legend(fontsize=7)
    ax[1].semilogy(its,[rows[0]['gap_before']]+[x['gap_after'] for x in rows],'o-')
    ax[1].set(xlabel='accepted local update',ylabel='gap')
    for ext in ('png','pdf'):fig.savefig(FIGS/f'm4_updates.{ext}')
    plt.close(fig)
    return rows


def main() -> dict:
    reference=run_reference()
    out = {"reference":reference, "M3": run_m3(), "S1": run_s1(), "M4": run_m4(reference)}
    (TABLES / "summary_2d.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
