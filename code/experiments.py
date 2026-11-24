"""Recompute only the charge-enclosure experiments in this manuscript.

All writes, including Matplotlib caches and copied solver output, stay in ROOT.
The archived solvers are copied from the author's original manuscript project.
"""
from pathlib import Path
import os, sys, csv, json, gc
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
os.environ['EXPERIMENTS_RESULTS_DIR'] = str(ROOT/'data/recomputed_legacy')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'code/legacy'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.fem1d import diagnostics
from src.analytic1d import analytic
from src.fem2d import Mesh2D, assemble_rt0
from src.experiments_2d import _evaluate, scalar_result
from src.priority_experiments import counterexample_rows

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22,
                     'savefig.bbox':'tight','figure.dpi':160})
CHECKS=[]
def check(name, ok, detail=None):
    CHECKS.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok: raise AssertionError((name,detail))
def dump(name, rows):
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def figsave(fig,name):
    for ext in ('pdf','png'):fig.savefig(ROOT/'figures'/f'{name}.{ext}')
    plt.close(fig)

def run_1d():
    rows=[];samples=[];cache={}
    def get(n,b):
        key=(n,round(float(b),14))
        if key not in cache:
            r=diagnostics(n,key[1]);a=analytic([0],key[1])
            E,D=r['energy'],r['dual_energy']
            check(f'1d energy {key}', D<=a['energy']+2e-11 and E>=a['energy']-2e-11)
            cache[key]=(E,D)
            samples.append(dict(n=n,b=key[1],E=E,D=D,J=a['energy'],Q=a['Q']))
        return cache[key]
    for n in (32,64,128,256):
        for k in (1e-4,2.5e-4,5e-4,.001,.002,.0025,.005,.01,.025,.05,.1):
            b=.5;ep,dm=get(n,b);em,dl=get(n,b-k);er,dr=get(n,b+k)
            lo=(dm-em)/k;hi=(er-dm)/k
            M=max(.5,1/np.sqrt(2*(b-k)))
            majorant=M*k+((er-dr)+(em-dl)+2*(ep-dm))/k
            exactcurv=(analytic([0],b+k)['energy']+analytic([0],b-k)['energy']-2*analytic([0],b)['energy'])/k
            check(f'1d bracket {n}/{k}',lo<=1+2e-10 and hi>=1-2e-10)
            check(f'1d width {n}/{k}',hi-lo<=majorant+2e-10)
            rows.append(dict(n=n,b=b,k=k,Q_lower=lo,Q_upper=hi,width=hi-lo,
                             Q_exact=1.,width_majorant=majorant,exact_curvature_term=exactcurv))
    # Independent branch/transition cases with FE fields, including b-k=0.
    branch=[]
    for n in (16,32,64):
        for b in (.005,.05,.37,.5,1.3,1.99,2.,2.01,3.):
            for ratio in (.1,.5,1.):
                k=b*ratio;E,D=get(n,b);Em,Dm=get(n,b-k);Ep,Dp=get(n,b+k)
                lo=(D-Em)/k;hi=(Ep-D)/k;q=analytic([0],b)['Q']
                check(f'branch bracket {n}/{b}/{ratio}',lo<=q+2e-9 and hi>=q-2e-9)
                branch.append(dict(n=n,b=b,k=k,Q_lower=lo,Q_upper=hi,Q_exact=q))
    dump('one_dimensional_samples.csv',samples);dump('one_dimensional_bounds.csv',rows)
    dump('branch_transition_bounds.csv',branch)
    fig,ax=plt.subplots(1,2,figsize=(9,3.35),layout='constrained')
    for n in (32,64,128,256):
        r=[x for x in rows if x['n']==n]
        ax[0].loglog([x['k'] for x in r],[x['width'] for x in r],'-o',ms=3,label=f'n={n}')
    r=[x for x in rows if x['n']==256]
    ax[1].loglog([x['k'] for x in r],[x['width'] for x in r],'-o',label='Interval width')
    ax[1].loglog([x['k'] for x in r],[x['width_majorant'] for x in r],'--',label='Width bound')
    for a in ax:a.set(xlabel='Bias step k',ylabel='Charge interval width');a.legend(fontsize=8)
    figsave(fig,'charge_width_1d')
    return rows,branch

def run_2d():
    samples=[];bounds=[];centers={}
    meshes=(32,64,128,256,512);biases=(.05,.2,.5,1.5,3.);ratios=(.05,.1,.2)
    residual_keys=('primal_violation','boundary_residual','stationarity_residual','dual_violation',
       'complementarity_residual','equilibrium_residual','gap_identity_residual',
       'energy_quadrature_residual','dual_quadrature_residual','reciprocity_residual')
    for nx in meshes:
        m=Mesh2D(nx,nx//2);mix=assemble_rt0(m,.25,.75);cache={}
        def get(b):
            key=round(float(b),12)
            if key not in cache:
                rr=_evaluate(m,.25,.75,key,mixed=mix)
                r=dict(scalar_result(rr),b=key)
                check(f'2d feasibility {nx}/{key}',r['min_u']>=-1e-12 and r['lambda_min']>=0)
                for name in residual_keys:check(f'2d {name} {nx}/{key}',abs(r[name])<2e-8,r[name])
                check(f'2d nonnegative gap {nx}/{key}',r['gap']>=-2e-9)
                cache[key]=r;samples.append(r)
                dump('two_dimensional_samples_checkpoint.csv',samples)
            return cache[key]
        for b in biases:
            print(f'2D {nx} x {nx//2}: b={b}',flush=True)
            c=get(b);centers[nx,b]=c
            for ratio in ratios:
                k=b*ratio;l=get(b-k);r=get(b+k)
                lo=(c['D']-l['E'])/k;hi=(r['E']-c['D'])/k
                check(f'2d ordered bounds {nx}/{b}/{k}',lo<=hi)
                bounds.append(dict(nx=nx,b=b,k=k,Q_lower=lo,Q_upper=hi,width=hi-lo))
        del mix,cache;gc.collect()
    summary=[]
    for b in biases:
        ref=centers[512,b]['Q_mixed'];prev=centers[256,b]['Q_mixed']
        for nx in meshes:
            group=[r for r in bounds if r['nx']==nx and r['b']==b]
            for r in group:
                r['Q_reference']=ref
                check(f'2d numerical consistency {nx}/{b}/{r["k"]}',r['Q_lower']<=ref<=r['Q_upper'])
            best=min(group,key=lambda r:r['width'])
            summary.append(dict(best,Q_mixed=centers[nx,b]['Q_mixed'],
                                width_percent=100*best['width']/abs(ref),
                                mesh_change_percent=100*abs(ref-prev)/abs(ref)))
    dump('two_dimensional_samples.csv',samples);dump('two_dimensional_bounds.csv',bounds)
    dump('two_dimensional_summary.csv',summary)
    fig,ax=plt.subplots(1,2,figsize=(9,3.35),layout='constrained')
    for i,nx in enumerate(meshes):
        r=[x for x in summary if x['nx']==nx]
        ax[0].errorbar(np.arange(5)+(i-2)*.13,[(x['Q_lower']+x['Q_upper'])/2 for x in r],
                       yerr=[x['width']/2 for x in r],fmt='o',ms=3,capsize=2,label=f'{nx} x {nx//2}')
        ax[1].plot(biases,[x['width_percent'] for x in r],'-o',ms=3,label=f'{nx} x {nx//2}')
    ax[0].set(xticks=np.arange(5),xticklabels=[str(x) for x in biases],xlabel='Bias b',ylabel='Charge interval')
    ax[1].set(xlabel='Bias b',ylabel='Relative interval width (%)')
    for a in ax:a.legend(fontsize=8)
    figsave(fig,'charge_intervals_2d')
    return summary

def run_counterexample():
    rows=[r for r in counterexample_rows(64) if r['case']=='varying_multiplier_recovery']
    dump('counterexample.csv',rows)
    check('counterexample gap strictly decreases',all(y['gap']<x['gap'] for x,y in zip(rows,rows[1:])))
    check('counterexample charge error strictly increases',all(y['recovered_charge_abs_error']>x['recovered_charge_abs_error'] for x,y in zip(rows,rows[1:])))
    for r in rows:
        for k in ('equilibrium_residual','gap_identity_residual','analytic_D_residual','analytic_Q_residual'):
            check(f'counterexample {r["t"]}/{k}',abs(r[k])<1e-11)
    fig,ax=plt.subplots(1,2,figsize=(9,3.1),layout='constrained')
    ax[0].plot([r['t'] for r in rows],[r['gap'] for r in rows],'-o',ms=3)
    ax[0].set(xlabel='Path parameter t',ylabel='Primal-dual gap')
    ax[1].plot([r['t'] for r in rows],[100*r['recovered_charge_abs_error'] for r in rows],'-o',ms=3,color='C1')
    ax[1].set(xlabel='Path parameter t',ylabel='True charge error (%)')
    figsave(fig,'energy_charge_counterexample')
    return rows

def tables(r1,r2,rc):
    out=ROOT/'source'
    rows=[x for x in r1 if x['n']==256 and x['k'] in (1e-4,.001,.0025,.01,.1)]
    (out/'table_1d.tex').write_text('\n'.join(
        f"{r['k']:.4g} & {r['Q_lower']:.6f} & {r['Q_upper']:.6f} & {r['width']:.6f} & {r['width_majorant']:.6f} \\\\" for r in rows)+'\n')
    (out/'table_2d.tex').write_text('\n'.join(
        f"{r['b']:.2g} & {r['Q_mixed']:.6f} & {r['Q_lower']:.6f} & {r['Q_upper']:.6f} & {r['width_percent']:.2f} & {r['mesh_change_percent']:.4f} \\\\" for r in r2 if r['nx']==512)+'\n')
    result=dict(check_count=len(CHECKS),all_passed=all(c['passed'] for c in CHECKS),
       samples_1d_bounds=len(r1),branch_bounds=81,samples_2d=175,intervals_2d=75,
       best_1d=min([x for x in r1 if x['n']==256],key=lambda x:x['width']),
       counterexample_gap_drop_percent=100*(rc[0]['gap']-rc[-1]['gap'])/rc[0]['gap'],
       counterexample_start_gap=rc[0]['gap'],counterexample_end_gap=rc[-1]['gap'],checks=CHECKS)
    (ROOT/'verification/experiment_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':
    a,_=run_1d();b=run_2d();c=run_counterexample();tables(a,b,c)
