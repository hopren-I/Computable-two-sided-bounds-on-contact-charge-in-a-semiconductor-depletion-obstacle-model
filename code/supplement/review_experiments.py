"""Reanalyse immutable CSV inputs and run two small local-contact benchmarks.
Run with Python + NumPy/SciPy/Matplotlib. Outputs are written to the project result directory.
"""
from pathlib import Path
import csv,json,sys,os,time,platform,resource
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache')
os.environ['EXPERIMENTS_RESULTS_DIR']=str(ROOT/'scratch')
sys.path.insert(0,str(ROOT/"code/legacy"))
import numpy as np
import scipy
from src.fem2d import Mesh2D,assemble_q1,assemble_rt0,solve_q1,lambda_from_q1
from src.experiments_2d import quadrature_diagnostics
from rt0_comparison import optimize
OUT=ROOT/'result';OUT.mkdir(exist_ok=True)
def read(name):return list(csv.DictReader((ROOT/'data'/name).open()))
def save(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(a):return (2*a)**1.5/3 if a<2 else a*a/4+a-1/3
samples=read('two_dimensional_full_contact_samples.csv')
truth=[]
for r in read('two_dimensional_full_contact_bounds.csv'):
 n,b,k=int(r['nx']),float(r['b']),float(r['k'])
 ss=[next(x for x in samples if int(x['nx'])==n and abs(float(x['b'])-a)<1e-10) for a in (b-k,b,b+k)]
 l,c,u=ss
 wp=(float(l['E'])-exact(b-k)+float(u['E'])-exact(b+k))/k
 wd=2*(exact(b)-float(c['D']))/k
 ws=(exact(b-k)+exact(b+k)-2*exact(b))/k
 width=float(r['width'])
 assert abs(width-wp-wd-ws)<1e-9
 truth.append(dict(nx=n,ny=n//2,b=b,k=k,k_over_b=k/b,W=width,W_secant=ws,W_primal=wp,W_dual=wd,
  error_over_radius=2*float(r['midpoint_abs_error'])/width,
  secant_percent=100*ws/width,primal_percent=100*wp/width,dual_percent=100*wd/width))
save('truth_decomposition.csv',truth)
# Reuse published samples as the numerical reference; new timings require reruns.
old=read('two_dimensional_samples.csv')
bench=[];trace=[];new_samples=[];recovery=[];all_checks=[]
keys=['primal_violation','boundary_residual','stationarity_residual','dual_violation','complementarity_residual',
 'equilibrium_residual','gap_identity_residual','energy_quadrature_residual','dual_quadrature_residual']
for nx in (64,128):
 b=.5;mesh=Mesh2D(nx,nx//2)
 for method in ('fixed','search'):
  assemble_q1.cache_clear();t0=time.perf_counter()
  assemble_q1(mesh);mix=assemble_rt0(mesh,.25,.75);mix.factor()
  setup=time.perf_counter()-t0;cache={};potential_times={}
  def get(a,central=False):
   a=round(float(a),14)
   if a not in cache:
    t=time.perf_counter();sol=solve_q1(mesh,a,.25,.75)
    potential_times[a]=time.perf_counter()-t
    E=float(.5*sol.u@(sol.K@sol.u)+sol.f@sol.u)
    known=[r for r in old if int(r['nx'])==nx and abs(float(r['b'])-a)<1e-10]
    if known:assert abs(float(known[0]['E'])-E)<2e-10
    cache[a]=(E,sol)
    new_samples.append(dict(nx=nx,b=a,E=E,method=method,existing_sample=bool(known)))
   return cache[a]
  t0=time.perf_counter();Ec,sol=get(b)
  t=time.perf_counter();lam=lambda_from_q1(sol);s,_,Dc=mix.solve(lam,b);rt_seconds=time.perf_counter()-t
  checks=quadrature_diagnostics(sol,mix,s,lam,b);all_checks.append(checks)
  assert max(abs(checks[x]) for x in keys)<2e-8
  g=Ec-Dc;rows=[]
  steps=(.025,.05,.1) if method=='fixed' else tuple(.1*2**(-j/2) for j in range(6))
  for k in steps:
   El,_=get(b-k);Er,_=get(b+k)
   lo=(Dc-El)/k;hi=(Er-Dc)/k;W=hi-lo
   rows.append(dict(nx=nx,method=method,k=k,Q_lower=lo,Q_upper=hi,W=W,g_b=g,gap_width_term=2*g/k))
   if method=='search' and len(rows)>1 and W>=min(r['W'] for r in rows[:-1]):break
  best=min(rows,key=lambda r:r['W']);inter=max(0.,min(r['Q_upper'] for r in rows)-max(r['Q_lower'] for r in rows))
  elapsed=time.perf_counter()-t0
  bench.append(dict(nx=nx,ny=nx//2,method=method,k=best['k'],k_over_b=best['k']/b,W=best['W'],intersection_W=inter,
   q1_solves=len(cache),rt0_solves=1,setup_seconds=setup,evaluation_seconds=elapsed,total_seconds=setup+elapsed,
   rt0_dof=mix.n_faces,q1_dof=mesh.n_nodes(),cells=mix.n_cells,
   gap_fraction=2*g/(best['k']*best['W'])))
  trace+=rows
  if method=='fixed':
   so,stats,_=optimize(mix,b,s,lam)
   Do=float(b*mix.q@so-.5*so@(mix.M@so))
   optchecks=quadrature_diagnostics(sol,mix,so,1-mix.B@so,b);all_checks.append(optchecks)
   assert max(abs(optchecks[x]) for x in keys)<2e-8
   k=.1;El,_=get(b-k);Er,_=get(b+k)
   W0=(Er+El-2*Dc)/k;W1=(Er+El-2*Do)/k
   recovery.append(dict(nx=nx,ny=nx//2,b=b,k=k,D_original=Dc,D_optimized=Do,g_original=g,g_optimized=Ec-Do,
    W_original=W0,W_optimized=W1,gap_reduction_percent=100*(Do-Dc)/g,width_reduction_percent=100*(W0-W1)/W0,
    original_rt_seconds=rt_seconds,additional_opt_seconds=stats['opt_seconds'],opt_iterations=stats['opt_iterations'],
    qp_gap=stats['opt_qp_gap']))
save('step_benchmark.csv',bench);save('step_trace.csv',trace);save('small_samples.csv',new_samples);save('recovery_comparison.csv',recovery)
report=dict(versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),machine=platform.machine()),
 timing='single wall-clock run; setup includes Q1/RT0 assembly and RT0 factorization; evaluation includes center Q1/RT0, side Q1 solves and center quadrature checks; no side RT0 solves',
 stopping='k0=0.2b; divide by sqrt(2); stop at first nondecrease of W or six trials; retain minimum W',
 residual_maxima={k:max(abs(r[k]) for r in all_checks) for k in keys},
 derived_intervals=len(truth),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 new_unique_biases={str(n):sorted({r['b'] for r in new_samples if r['nx']==n and not r['existing_sample']}) for n in (64,128)},
 benchmarks=bench,recovery=recovery)
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
