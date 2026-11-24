"""Validate the new evidence against raw samples and analytic invariants."""
import math
import numpy as np

EXPECTED_ROWS = {'p1_offgrid_convergence.csv':28, 'p1_offgrid_summary.csv':4,
                 'p2_bias_energy_samples.csv':105, 'p2_bias_charge_bounds.csv':45,
                 'p2_bias_summary.csv':15, 'p3_energy_charge_counterexample.csv':20}


def verify(data, add, bounded, num, contains):
    from .priority_experiments import OFFGRID_CASES, OFFGRID_NS, BIASES, BIAS_MESHES, STEP_RATIOS
    p1=data['p1_offgrid_convergence.csv'];summary=data['p1_offgrid_summary.csv']
    add('P1 complete fixed-bias refinement sequences',
        {(r['case'],int(r['n'])) for r in p1}=={(c,n) for c,b in OFFGRID_CASES for n in OFFGRID_NS})
    for case,b in OFFGRID_CASES:
        g=sorted([r for r in p1 if r['case']==case],key=lambda r:int(r['n']))
        add('P1 fixed bias '+case,all(abs(float(r['b'])-b)<1e-14 for r in g))
        s=next(r for r in summary if r['case']==case)
        for key in ('energy_error','gap','grad_error','u_L2_error','Q_direct_abs_error','Q_balance_abs_error'):
            values=num(g,key)
            add('P1 positive finite '+case+'/'+key,min(values)>0)
            p=float(np.polyfit(np.log(num(g[-4:],'h')),np.log(values[-4:]),1)[0])
            add('P1 fitted rate arithmetic '+case+'/'+key,abs(p-float(s['fit_order_'+key]))<1e-10)
            add('P1 adjacent rate arithmetic '+case+'/'+key,all(abs(float(r['order_'+key])-math.log(a/z)/math.log(2))<1e-10
                for r,a,z in zip(g[1:],values,values[1:])))
        add('P1 records nonmonotonic errors '+case,int(s['charge_error_increase_count'])==
            sum(a<z for a,z in zip(num(g,'Q_balance_abs_error'),num(g,'Q_balance_abs_error')[1:])))
    add('P1 genuinely nonaligned on every mesh',min(num(p1,'interface_node_distance_over_h'))>1e-8)
    add('P1 interface phases recomputed',all(abs((float(r['width_exact'])/float(r['h']))%1-float(r['interface_phase']))<1e-10 for r in p1))
    add('P1 analytic energy enclosure',all(contains(r,'dual_energy','energy_exact','energy') for r in p1))
    add('P1 gradient bound',all(float(r['grad_error'])**2<=2*float(r['gap'])+1e-10 for r in p1))
    applicable=[r for r in p1 if 0<float(r['delta'])<float(r['b'])]
    add('P1 applicable interface intervals',len(applicable)>0 and all(contains(r,'interval_lo','width_exact','interval_hi') for r in applicable))
    for k in ('primal_violation','boundary_residual','stationarity_residual','dual_violation','complementarity_residual','gap_identity_residual','charge_flux_residual'):
        bounded(p1,k,1e-9,'P1 '+k)
    samples=data['p2_bias_energy_samples.csv'];bounds=data['p2_bias_charge_bounds.csv'];sums=data['p2_bias_summary.csv']
    index={(int(r['nx']),round(float(r['b']),12)):r for r in samples}
    expected={(nx,round(b*(1+sign*ratio),12)) for nx in BIAS_MESHES for b in BIASES
              for sign in (-1,0,1) for ratio in STEP_RATIOS}
    add('P2 complete and unique energy sample grid',set(index)==expected and len(index)==len(samples))
    add('P2 complete bound grid',{(int(r['nx']),float(r['b']),round(float(r['step_ratio']),8)) for r in bounds}==
        {(nx,b,ratio) for nx in BIAS_MESHES for b in BIASES for ratio in STEP_RATIOS})
    for k in ('primal_violation','boundary_residual','stationarity_residual','dual_violation','complementarity_residual',
              'equilibrium_residual','gap_identity_residual','energy_quadrature_residual','dual_quadrature_residual','reciprocity_residual'):
        bounded(samples,k,2e-8,'P2 every energy sample '+k)
    add('P2 every sample primal/dual feasible',min(num(samples,'min_u'))>=-1e-12 and min(num(samples,'lambda_min'))>=-1e-12)
    add('P2 every sample nonnegative gap',min(num(samples,'gap'))>=-1e-10)
    for r in bounds:
        nx=int(r['nx']);b=float(r['b']);h=float(r['step'])
        c=index[nx,round(b,12)];l=index[nx,round(b-h,12)];z=index[nx,round(b+h,12)]
        lo=(float(c['D'])-float(l['E']))/h;hi=(float(z['E'])-float(c['D']))/h
        ref=index[max(BIAS_MESHES),round(b,12)]
        add(f'P2 raw-sample secant {nx}/{b}/{h}',hi>=lo and
            abs(lo-float(r['Q_lower']))<1e-10 and abs(hi-float(r['Q_upper']))<1e-10 and
            abs(hi-lo-float(r['width']))<1e-10 and abs(float(ref['Q_mixed'])-float(r['Q_reference']))<1e-12)
    add('P2 fine estimates consistent with numerical intervals',all(contains(r,'Q_lower','Q_reference','Q_upper') for r in bounds),
        'reference is an approximation, not continuum truth')
    add('P2 summary grid complete',{(int(r['nx']),float(r['b'])) for r in sums}=={(nx,b) for nx in BIAS_MESHES for b in BIASES})
    for r in sums:
        group=[x for x in bounds if x['nx']==r['nx'] and x['b']==r['b']]
        best=min(group,key=lambda x:float(x['width']))
        add(f"P2 reported best and intersection {r['nx']}/{r['b']}",
            abs(float(best['width'])-float(r['width']))<1e-12 and
            abs(max(num(group,'Q_lower'))-float(r['intersection_lower']))<1e-12 and
            abs(min(num(group,'Q_upper'))-float(r['intersection_upper']))<1e-12 and
            float(r['intersection_lower'])<=float(r['intersection_upper']))
    p3=data['p3_energy_charge_counterexample.csv']
    for k in ('boundary_residual','equilibrium_residual','gap_identity_residual','dual_quadrature_residual','analytic_D_residual','analytic_Q_residual'):
        bounded(p3,k,1e-10,'P3 '+k)
    add('P3 feasible fields',min(num(p3,'lambda_min'))>=0 and min(num(p3,'min_u'))>=0)
    add('P3 brackets known exact energy',all(contains(r,'D','E_exact','E') for r in p3))
    for case,count,end in [('fixed_multiplier_flux',11,1.),('varying_multiplier_recovery',9,.8)]:
        g=sorted([r for r in p3 if r['case']==case],key=lambda r:float(r['t']))
        add('P3 full path '+case,len(g)==count and abs(float(g[0]['t']))<1e-12 and abs(float(g[-1]['t'])-end)<1e-12)
        add('P3 fixed primal energy '+case,max(num(g,'E'))-min(num(g,'E'))<1e-12)
        add('P3 strictly increasing D and decreasing gap '+case,all(float(y['D'])>float(x['D']) and float(y['gap'])<float(x['gap']) for x,y in zip(g,g[1:])))
        add('P3 strictly worsening flux charge '+case,all(float(y['flux_charge_abs_error'])>float(x['flux_charge_abs_error'])+1e-8 for x,y in zip(g,g[1:])))
        if case=='varying_multiplier_recovery':
            bounded(g,'fixed_multiplier_optimality_residual',1e-12,'P3 each changed-multiplier flux optimal')
            add('P3 recovered charge worsens too',all(float(y['recovered_charge_abs_error'])>float(x['recovered_charge_abs_error'])+1e-8 for x,y in zip(g,g[1:])))
        else:
            add('P3 fixed multiplier recovery unchanged',max(num(g,'Q_recovered'))-min(num(g,'Q_recovered'))<1e-12)
