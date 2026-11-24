"""Targeted evidence: off-grid interfaces, bias enclosures, output counterexamples.

Only invariants are acceptance criteria. Convergence rates, interval tightness
and mesh-to-mesh charge changes are observations, not enforced outcomes.
"""
from __future__ import annotations

import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .paths import FIGS
from .fem1d import diagnostics, solve, reconstruct_flux, dual_energy, charge_balance
from .experiments_1d import _write_csv
from .experiments_2d import _evaluate, scalar_result
from .fem2d import Mesh2D, assemble_rt0


OFFGRID_CASES = (("low_bias", .05), ("alternating_phase", 2/9),
                 ("mid_bias", .37), ("high_bias", 1.3))
OFFGRID_NS = (16, 32, 64, 128, 256, 512, 1024)
BIASES = (.05, .2, .5, 1.5, 3.)
BIAS_MESHES = (128, 256, 512)
STEP_RATIOS = (.05, .1, .2)


def savefig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f'{name}.{ext}')
    plt.close(fig)


def run_offgrid():
    rows, summaries = [], []
    keys = ('energy_error', 'gap', 'grad_error', 'u_L2_error',
            'Q_direct_abs_error', 'Q_balance_abs_error')
    for case, b in OFFGRID_CASES:
        group = []
        for n in OFFGRID_NS:
            d = diagnostics(n, b)
            r = {k: v for k, v in d.items() if np.isscalar(v)}
            phase = (d['width_exact'] / d['h']) % 1
            r.update(case=case, interface_phase=phase,
                     interface_node_distance_over_h=min(phase, 1-phase),
                     Q_direct_abs_error=abs(d['Q_direct']-d['Q_exact']),
                     Q_balance_abs_error=abs(d['Q_balance']-d['Q_exact']),
                     effectivity=d['gap']/d['energy_error'])
            for key in keys:
                r['order_'+key] = (np.log(group[-1][key]/r[key])/np.log(2)
                                      if group else np.nan)
            group.append(r)
        tail = group[-4:]
        summary = dict(case=case, b=b, width_exact=group[-1]['width_exact'],
                       fit_n_min=tail[0]['n'], fit_n_max=tail[-1]['n'])
        for key in keys:
            summary['fit_order_'+key] = float(np.polyfit(
                np.log([r['h'] for r in tail]), np.log([r[key] for r in tail]), 1)[0])
        summary.update(
            charge_error_increase_count=sum(y['Q_balance_abs_error']>x['Q_balance_abs_error']
                                            for x,y in zip(group,group[1:])),
            charge_sign_changes=sum(x['Q_balance_error']*y['Q_balance_error']<0
                                    for x,y in zip(group,group[1:])),
            scaled_charge_min=min(r['Q_balance_abs_error']/r['h']**2 for r in group),
            scaled_charge_max=max(r['Q_balance_abs_error']/r['h']**2 for r in group),
            finest_charge_error=group[-1]['Q_balance_error'],
            effectivity_min=min(r['effectivity'] for r in group),
            effectivity_max=max(r['effectivity'] for r in group))
        rows.extend(group); summaries.append(summary)
    _write_csv('p1_offgrid_convergence.csv', rows)
    _write_csv('p1_offgrid_summary.csv', summaries)
    fig, axes = plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
    for color,(case,b) in enumerate(OFFGRID_CASES):
        g=[r for r in rows if r['case']==case];h=[r['h'] for r in g]
        label=f'b={b:.5g}'
        axes[0,0].loglog(h,[r['energy_error'] for r in g],'o-',color=f'C{color}',label=label)
        axes[0,0].loglog(h,[r['gap'] for r in g],'--',color=f'C{color}',alpha=.65)
        axes[0,1].loglog(h,[r['Q_balance_abs_error'] for r in g],'o-',label=label)
        axes[1,0].semilogx([r['n'] for r in g],
                          [r['Q_balance_error']/r['h']**2 for r in g],'o-',label=label)
        axes[1,1].semilogx([r['n'] for r in g],[r['interface_phase'] for r in g],'o-',label=label)
    axes[0,0].set(xlabel='mesh size h',ylabel='energy error / gap',title='(a) Solid: error; dashed: gap')
    axes[0,1].set(xlabel='mesh size h',ylabel='absolute recovered-charge error',title='(b) Charge convergence with oscillations')
    axes[1,0].axhline(0,color='k',lw=.7)
    axes[1,0].set(xlabel='number of cells n',ylabel=r'$(Q_{rec}-Q_*)/h^2$',title='(c) Signed scaled error')
    axes[1,1].set(xlabel='number of cells n',ylabel='fractional interface position',ylim=(0,1),title='(d) Interface phase within a cell')
    for ax in axes.flat:ax.legend(fontsize=7)
    savefig(fig,'p1_offgrid_convergence')
    return summaries


def secant_row(center, left, right, step):
    if step <= 0 or left['b'] < 0:
        raise ValueError('positive step and nonnegative bias required')
    if not (np.isclose(center['b']-left['b'],step) and np.isclose(right['b']-center['b'],step)):
        raise ValueError('energy samples must match the secant step')
    lo=(center['D']-left['E'])/step
    hi=(right['E']-center['D'])/step
    return dict(nx=center['nx'],ny=center['ny'],b=center['b'],step=step,
                step_ratio=step/center['b'],Q_lower=lo,Q_upper=hi,width=hi-lo,
                E_center=center['E'],D_center=center['D'],E_minus=left['E'],E_plus=right['E'])


def run_bias_sweep():
    samples, bounds = [], []
    centers = {}
    for nx in BIAS_MESHES:
        mesh=Mesh2D(nx,nx//2);mixed=assemble_rt0(mesh,.25,.75)
        cache={}
        def get(b):
            key=round(float(b),12)
            if key not in cache:
                r=_evaluate(mesh,.25,.75,key,mixed=mixed)
                cache[key]=dict(scalar_result(r),b=key,
                    active_free_fraction=len(r['sol'].active)/(mesh.n_nodes()-len(r['sol'].fixed)))
                samples.append(cache[key])
            return cache[key]
        for b in BIASES:
            print(f'  bias sweep {nx}x{nx//2}, b={b}',flush=True)
            c=get(b);centers[nx,b]=c
            for ratio in STEP_RATIOS:
                step=b*ratio
                bounds.append(secant_row(c,get(b-step),get(b+step),step))
        del mixed,cache
        gc.collect()
    finest=max(BIAS_MESHES)
    for r in bounds:
        ref=centers[finest,r['b']]
        r.update(reference_nx=finest,Q_reference=ref['Q_mixed'],
                 relative_width_percent=100*r['width']/abs(ref['Q_mixed']))
    summaries=[]
    for b in BIASES:
        ref=centers[finest,b];prev=centers[BIAS_MESHES[-2],b]
        for nx in BIAS_MESHES:
            group=[r for r in bounds if r['b']==b and r['nx']==nx]
            best=min(group,key=lambda r:r['width']);c=centers[nx,b]
            lower=max(r['Q_lower'] for r in group);upper=min(r['Q_upper'] for r in group)
            summaries.append(dict(nx=nx,b=b,best_step=best['step'],
                Q_lower=best['Q_lower'],Q_upper=best['Q_upper'],width=best['width'],
                relative_width_percent=best['relative_width_percent'],
                intersection_lower=lower,intersection_upper=upper,
                Q_mixed=c['Q_mixed'],Q_direct=c['Q_direct'],Q_reaction=c['Q_reaction'],
                Q_reference=ref['Q_mixed'],reference_nx=finest,
                reference_mesh_change_percent=100*abs(ref['Q_mixed']-prev['Q_mixed'])/abs(ref['Q_mixed']),
                gap=c['gap'],active_free_fraction=c['active_free_fraction']))
    _write_csv('p2_bias_energy_samples.csv',samples)
    _write_csv('p2_bias_charge_bounds.csv',bounds)
    _write_csv('p2_bias_summary.csv',summaries)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for i,nx in enumerate(BIAS_MESHES):
        rr=[r for r in summaries if r['nx']==nx]
        axes[0].errorbar(np.arange(len(BIASES))+(i-1)*.17,
            [(r['Q_lower']+r['Q_upper'])/2 for r in rr],yerr=[r['width']/2 for r in rr],
            fmt='o',capsize=3,label=f'{nx}x{nx//2}')
        axes[1].loglog(BIASES,[r['relative_width_percent'] for r in rr],'o-',label=f'{nx}x{nx//2}')
    axes[0].plot(np.arange(len(BIASES)),[centers[finest,b]['Q_mixed'] for b in BIASES],
                 'kx',label='fine mixed estimate')
    axes[0].set(xticks=np.arange(len(BIASES)),xticklabels=[str(b) for b in BIASES],
                xlabel='bias b',ylabel='charge / numerical enclosure',title='(a) Narrowest of three sampled steps')
    axes[1].set(xlabel='bias b',ylabel='full interval width / fine estimate (%)',
                title='(b) Relative width, not point-estimate error')
    for ax in axes:ax.legend(fontsize=7)
    savefig(fig,'p2_bias_charge_bounds')
    return summaries


def counterexample_rows(n=64):
    """Analytic Q*=1 at b=1/2, ell=2. Feasible P0 lambda and P1 flux.

    On [1,2], f=1 on [1,1.5], f=-3 on [1.5,2], so int eta*f=0.
    lambda(t)=1+.2(1-t)f+.05t gives Q_rec=1-t/80. Its optimal
    flux has exact dual defect (1-t)^2/300 - t(1-t)/1200 + t^2/3840.
    The defect strictly decreases on 0<=t<=.8, whereas charge error grows.
    """
    if n % 4:
        raise ValueError('counterexample breakpoints must be mesh aligned')
    sol=solve(n,.5);x=sol.x;mid=(x[:-1]+x[1:])/2
    neutral=mid>=1;shape=np.where(mid<1.5,1.,-3.)
    rows=[]
    for case in ('fixed_multiplier_flux','varying_multiplier_recovery'):
        ts=np.linspace(0,1,11) if case=='fixed_multiplier_flux' else np.linspace(0,.8,9)
        for t in ts:
            if case=='fixed_multiplier_flux':
                lam=np.where(neutral,1.05,0.)
                optimum,_=reconstruct_flux(lam,x,.5)
                qopt=charge_balance(lam,x,.5)
                sigma=optimum+(1-t)*(qopt-1.)
                predicted_Q=1-t/80
                predicted_D=1/3-1/3840-(1-t)**2/6400
            else:
                lam=np.where(neutral,1+.2*(1-t)*shape+.05*t,0.)
                sigma,_=reconstruct_flux(lam,x,.5)
                predicted_Q=1-t/80
                predicted_D=1/3-((1-t)**2/300-t*(1-t)/1200+t*t/3840)
            D=dual_energy(sigma,x,.5);Q=-sigma[0];Qrec=charge_balance(lam,x,.5)
            slopes=np.diff(sol.u)/sol.h
            a,z=slopes-sigma[:-1],slopes-sigma[1:]
            mismatch=float(np.sum(sol.h/6*(a*a+a*z+z*z)))
            comp=float(np.sum(sol.h*lam*(sol.u[:-1]+sol.u[1:])/2))
            # Independent 3-point Gauss integration of represented fields.
            gp,gw=np.polynomial.legendre.leggauss(3);theta=(gp+1)/2
            sigq=sigma[:-1,None]*(1-theta)+sigma[1:,None]*theta
            Dquad=-.5*sigma[0]-.5*sol.h/2*float(np.sum(sigq**2*gw))
            rows.append(dict(case=case,n=n,t=float(t),E=sol.energy,E_exact=1/3,D=D,gap=sol.energy-D,
                Q_exact=1.,Q_flux=float(Q),Q_recovered=Qrec,
                flux_charge_abs_error=abs(Q-1),recovered_charge_abs_error=abs(Qrec-1),
                lambda_min=float(lam.min()),min_u=float(sol.u.min()),
                boundary_residual=float(max(abs(sol.u[0]-.5),abs(sol.u[-1]))),
                equilibrium_residual=float(np.max(abs(np.diff(sigma)/sol.h-1+lam))),
                gap_identity_residual=sol.energy-D-mismatch-comp,
                dual_quadrature_residual=D-Dquad,analytic_D_residual=D-predicted_D,
                analytic_Q_residual=float(Q-predicted_Q),
                fixed_multiplier_optimality_residual=float(np.sum(sol.h*(sigma[:-1]+sigma[1:])/2)+.5)))
    return rows


def run_counterexample():
    rows=counterexample_rows()
    _write_csv('p3_energy_charge_counterexample.csv',rows)
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for case,label in [('fixed_multiplier_flux','fixed multiplier'),('varying_multiplier_recovery','changing multiplier')]:
        group=[r for r in rows if r['case']==case]
        t=[r['t'] for r in group]
        axes[0].semilogy(t,[r['gap'] for r in group],'o-',label=label)
        axes[1].plot(t,[100*r['flux_charge_abs_error'] for r in group],'o-',label=label)
        axes[2].plot(t,[100*r['recovered_charge_abs_error'] for r in group],'o-',label=label)
    axes[0].set(xlabel='controlled path parameter t',ylabel='gap (primal E fixed)',title='(a) Energy gap improves')
    axes[1].set(xlabel='controlled path parameter t',ylabel='flux charge error (%)',title='(b) Flux output gets worse')
    axes[2].set(xlabel='controlled path parameter t',ylabel='recovered charge error (%)',title='(c) Recovery follows multiplier')
    for ax in axes:ax.legend(fontsize=7)
    savefig(fig,'p3_energy_charge_counterexample')
    return rows


def main():
    print('  P1: fixed-bias off-grid convergence',flush=True);run_offgrid()
    print('  P3: controlled energy/output counterexamples',flush=True);run_counterexample()
    print('  P2: multiple-bias 2D enclosures',flush=True);run_bias_sweep()


if __name__=='__main__':main()
