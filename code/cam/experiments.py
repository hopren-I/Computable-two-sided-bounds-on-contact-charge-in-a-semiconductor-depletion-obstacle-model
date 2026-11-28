"""Numerical experiments in sections 5.1–5.5. No archived samples are inputs."""
from math import sqrt
import numpy as np
from src.fem1d import solve, cell_multiplier, dual_energy, reconstruct_flux
from src.fem2d import Mesh2D, assemble_q1, assemble_rt0, lambda_from_q1, solve_q1
from src.experiments_2d import quadrature_diagnostics
from src.priority_experiments import counterexample_rows
from .bernstein import solve_case
from .intervals import single_bias


def exact(b):
    if b < 2:
        return (2*b)**1.5/3, sqrt(2*b)
    return b*b/4+b-1/3, b/2+1


def one_dimensional():
    rows = []
    for b in (.005,.37,.5,1.99,2.,2.01,3.):
        v = solve(128,b)
        flux, _ = reconstruct_flux(cell_multiplier(v),v.x,b)
        D = dual_energy(flux,v.x,b)
        I = single_bias(b,float(v.u@(v.K@v.u)),float(v.f@v.u),D)
        lo, hi = I.refine(lambda a: solve(128,a).energy)
        _, Q = exact(b)
        if not lo <= Q <= hi:
            raise ArithmeticError("Analytical charge outside refined interval")
        rows.append(dict(b=b,Q_exact=Q,I1_lo=I.endpoints[0],I1_hi=I.endpoints[1],
                         I3_lo=lo,I3_hi=hi,center_error_pct=100*abs(I.center-Q)/I.radius))
    return rows


def q1_field(nx,b,y0,y1,mixed=None):
    mesh = Mesh2D(nx,nx//2)
    v = solve_q1(mesh,b,y0,y1)
    T, L = float(v.u@(v.K@v.u)), float(v.f@v.u)
    mix = mixed if mixed is not None else assemble_rt0(mesh,y0,y1)
    lam = lambda_from_q1(v)
    flux, _, D = mix.solve(lam,b)
    diagnostics = quadrature_diagnostics(v,mix,flux,lam,b)
    j0, j1 = round(mesh.ny*y0),round(mesh.ny*y1)
    reaction = float(sum(v.gradient[mesh.node(0,j)] for j in range(j0,j1+1)))
    diagnostics["work_residual"] = (T+L)/b-reaction
    for name in ("primal_violation","boundary_residual","stationarity_residual","dual_violation",
                 "complementarity_residual","gap_identity_residual","energy_quadrature_residual",
                 "dual_quadrature_residual","equilibrium_residual","work_residual"):
        if not abs(diagnostics[name]) < 2e-8:
            raise ArithmeticError(f"Q1/RT0 check failed: {nx}, {b}, {name}={diagnostics[name]}")
    if np.min(lam) < 0:
        raise ArithmeticError("Negative screening density")
    I = single_bias(b,T,L,D)
    record = dict(nx=nx,ny=mesh.ny,b=b,y0=y0,y1=y1,T=T,L=L,E=T/2+L,D=D,
                  gap=T/2+L-D,Q_mixed=float(mix.q@flux),reaction=reaction,**diagnostics)
    return I, record


def q1_energy(mesh,a,y0,y1):
    v = solve_q1(mesh,a,y0,y1)
    return float(.5*v.u@(v.K@v.u)+v.f@v.u)


def matched_steps():
    mesh = Mesh2D(128,64)
    mix = assemble_rt0(mesh,.25,.75)
    rows, details, samples = [], [], []
    for b in (.05,.2,.5,1.5,3.):
        I, record = q1_field(128,b,.25,.75,mix)
        def side(a):
            energy = q1_energy(mesh,a,.25,.75)
            samples.append(dict(center=b,neighbor=a,E=energy))
            return energy
        lo,hi = I.refine(side)
        widths = []
        for ratio in (.05,.1,.2):
            k = ratio*b
            widths.append((side(b+k)-record["D"])/k-(record["D"]-side(b-k))/k)
        rows.append(dict(b=b,step_ratio=I.step/b,W1=2*I.radius,W3=hi-lo,scan_min=min(widths)))
        details.append(dict(record,I1_lo=I.endpoints[0],I1_hi=I.endpoints[1],I3_lo=lo,I3_hi=hi))
    return rows,details,samples


def refinement_study():
    details = []
    for nx in (32,64,128,256,512,1024):
        print(f"  Q1/RT0 central fields on {nx} x {nx//2}",flush=True)
        I, record = q1_field(nx,.5,.25,.75)
        details.append(dict(record,S=I.center,R=I.radius,step=I.step,lo=I.endpoints[0],hi=I.endpoints[1]))
        assemble_q1.cache_clear()
    ref = next(r["Q_mixed"] for r in details if r["nx"]==512)
    rows = [dict(nx=r["nx"],gap=r["gap"],Q_mixed=r["Q_mixed"],S=r["S"],
                 width_pct=200*r["R"]/abs(ref)) for r in details]
    return rows, details


def quadratic_study():
    rows, details = [], []
    for nx in (16,32,64,128,256):
        print(f"  Bernstein uniform and graded meshes: {nx} x {nx//2}",flush=True)
        uniform = solve_case(nx,nx//2,False)
        graded = solve_case(nx,nx//2,True)
        details.extend((uniform,graded))
        rows.append(dict(nx=nx,uniform_S=uniform["reaction"],graded_S=graded["reaction"],
                         graded_E=graded["E"]))
    return rows, details


def full_edge():
    mesh = Mesh2D(128,64)
    mix = assemble_rt0(mesh,0.,1.)
    rows, central = [], []
    for b in (.005,.37,1.99,2.,2.01,3.):
        k = b*.1
        I, r = q1_field(128,b,0.,1.,mix)
        central.append(dict(r,single_width=2*I.radius))
        J,Q = exact(b)
        El = q1_energy(mesh,round(b-k,12),0.,1.)
        Er = q1_energy(mesh,round(b+k,12),0.,1.)
        D = r["D"]
        lo,hi = (D-El)/k,(Er-D)/k
        W = hi-lo
        err = abs((hi+lo)/2-Q)
        Jl,Jr = exact(b-k)[0],exact(b+k)[0]
        ws = (Jr+Jl-2*J)/k
        wp = (El-Jl+Er-Jr)/k
        wd = 2*(J-D)/k
        if not lo<=Q<=hi or abs(ws+wp+wd-W)>1e-9:
            raise ArithmeticError("Full-edge containment or width decomposition failed")
        rows.append(dict(b=b,Q_exact=Q,lo=lo,hi=hi,W=W,midpoint_error=err,
                         W_secant=ws,W_primal=wp,W_dual=wd,secant_pct=100*ws/W,
                         midpoint_pct=200*err/W))
    return rows, central


def gap_example():
    source = [r for r in counterexample_rows(64) if r["case"]=="varying_multiplier_recovery"]
    return [dict(t=r["t"],gap=r["gap"],Q_recovered=r["Q_recovered"],
                 charge_error=r["recovered_charge_abs_error"])
            for r in source if round(r["t"], 2) in (0.,.2,.3,.4)]

