"""Make focused explanatory figures/tables and independently audit fine meshes."""
from pathlib import Path
import os, csv, json
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(name):
    with (ROOT/'data'/name).open() as f:
        return [{k:float(v) for k,v in r.items() if k!='branch'} for r in csv.DictReader(f)]

def table(name, lines):
    (ROOT/'source'/f'{name}.tex').write_text('\n'.join(line+r' \\' for line in lines)+'\n')

# The domain and boundary data, rather than an inferred free boundary.
fig,ax=plt.subplots(figsize=(6.8,2.8),layout='constrained')
ax.fill([0,2,2,0],[0,0,1,1],color='#eef4f8')
ax.plot([0,2],[0,0],color='0.4',ls='--',lw=2)
ax.plot([0,2],[1,1],color='0.4',ls='--',lw=2)
ax.plot([0,0],[0,.25],color='0.4',ls='--',lw=2)
ax.plot([0,0],[.75,1],color='0.4',ls='--',lw=2)
ax.plot([0,0],[.25,.75],color='#c94530',lw=5)
ax.plot([2,2],[0,1],color='#296caa',lw=4)
ax.text(1,.55,r'$\Omega=(0,2)\times(0,1)$',ha='center',fontsize=12)
ax.text(1,.33,r'$-\Delta u+1-\lambda=0,\quad u\geq0,\ \lambda\geq0,\ u\lambda=0$',ha='center',fontsize=10)
ax.text(-.12,.5,r'$\Gamma_c$'+'\n'+r'$u=b$',ha='right',va='center',fontsize=11,color='#a83022')
ax.text(2.12,.5,r'$\Gamma_0$'+'\n'+r'$u=0$',ha='left',va='center',fontsize=11,color='#296caa')
ax.text(1,1.08,r'$\Gamma_N:\ \partial_nu=0$',ha='center',fontsize=11)
ax.annotate('',xy=(-.02,.5),xytext=(.4,.5),arrowprops={'arrowstyle':'->','color':'#c94530'})
ax.text(.18,.62,r'$n$',color='#c94530')
ax.set(xlim=(-.48,2.48),ylim=(-.15,1.28),xticks=[0,2],yticks=[0,.25,.75,1])
ax.set_aspect('equal');ax.spines[['top','right','left','bottom']].set_visible(False)
ax.tick_params(length=0);ax.set_xlabel(r'$x_1$',labelpad=0);ax.set_ylabel(r'$x_2$',rotation=0)
for ext in ('pdf','png'):fig.savefig(ROOT/'figures'/f'contact_geometry.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

br=[r for r in rows('branch_transition_bounds.csv') if r['n']==64 and r['b'] in (.005,1.99,2.,2.01,3.) and abs(r['k']/r['b']-.1)<1e-12]
table('table_branches',[f"{r['b']:.3g} & {r['k']:.4g} & {r['Q_exact']:.6f} & {r['Q_lower']:.6f} & {r['Q_upper']:.6f}" for r in br])
samples=rows('two_dimensional_samples.csv')
summary=rows('two_dimensional_summary.csv')
idx={(int(r['nx']),round(r['b'],10)):r for r in samples}
sel=[r for r in summary if r['b']==.5]
table('table_mesh',[f"${int(r['nx'])}\\times{int(r['nx'])//2}$ & {idx[int(r['nx']),.5]['gap']:.3e} & {r['k']/r['b']:.2f} & {r['Q_mixed']:.6f} & {r['width_percent']:.2f}" for r in sel])

new=rows('two_dimensional_bounds.csv')
old=rows('original/p2_bias_charge_bounds.csv')
ni={(int(r['nx']),round(r['b'],10),round(r['k'],10)):r for r in new}
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError((name,detail))
check('all five meshes recomputed',sorted(set(int(r['nx']) for r in samples))==[32,64,128,256,512])
check('175 raw energy samples',len(samples)==175)
check('75 intervals',len(new)==75)
maxdiff=0.
for r in old:
    key=(int(r['nx']),round(r['b'],10),round(r['step'],10));v=ni[key]
    for field in ('Q_lower','Q_upper','width'):
        d=abs(r[field]-v[field]);maxdiff=max(maxdiff,d)
        check(f'original fine interval {key}/{field}',d<1e-9,d)
    n,b,k=key
    for field,value in [('E_center',idx[n,b]['E']),('D_center',idx[n,b]['D']),
                        ('E_minus',idx[n,round(b-k,10)]['E']),('E_plus',idx[n,round(b+k,10)]['E'])]:
        d=abs(r[field]-value);maxdiff=max(maxdiff,d)
        check(f'original fine energy {key}/{field}',d<1e-10,d)
for r in rows('original/p2_bias_summary.csv'):
    n=int(r['nx']);b=r['b'];c=idx[n,round(b,10)]
    check(f'original recovered charge {n}/{b}',abs(r['Q_mixed']-c['Q_mixed'])<1e-9)

residuals=('primal_violation','boundary_residual','stationarity_residual','dual_violation',
    'complementarity_residual','equilibrium_residual','gap_identity_residual',
    'energy_quadrature_residual','dual_quadrature_residual','reciprocity_residual')
report=dict(all_passed=all(c['passed'] for c in checks),check_count=len(checks),
    fine_intervals_reproduced=len(old),maximum_original_difference=maxdiff,
    residual_maxima={k:max(abs(r[k]) for r in samples) for k in residuals},checks=checks)
(ROOT/'verification/fine_mesh_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
