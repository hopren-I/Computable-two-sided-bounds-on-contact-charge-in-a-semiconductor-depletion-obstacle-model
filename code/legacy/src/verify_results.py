"""Fail-closed checks recomputed from numeric columns, never CSV Boolean flags."""
from __future__ import annotations
import csv
import json
import math
from .paths import TABLES
from .verify_priority import EXPECTED_ROWS as PRIORITY_ROWS, verify as verify_priority

EXPECTED_ROWS = {
    'm1_convergence.csv':6, 'validation_1d.csv':50, 'm2_charge_intervals.csv':11,
    'm2_capacitance_intervals.csv':5, 'm2_interface.csv':1,
    'm2_parameter_sweep.csv':21,'m2_nonmonotone.csv':1,'s2_branches.csv':45,
    'm5_poisson_boltzmann.csv':5,'m3_full_contact.csv':4,'m3_partial_contact.csv':3,
    'm3_reciprocity.csv':3,'m3_single_direction.csv':1,'m3_reference.csv':5,
    'm3_reference_bounds.csv':6,'solver_crosscheck_2d.csv':3,
    's1_contact_width.csv':8,'m4_hierarchy.csv':5,'m4_local_response.csv':1,
    'm4_updates.csv':6,
}
EXPECTED_ROWS.update(PRIORITY_ROWS)


def read(name):
    with (TABLES/name).open(encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    if not rows:raise ValueError(f'empty required table: {name}')
    return rows


def num(rows,key):
    values=[float(r[key]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'nonfinite required column {key}')
    return values


def contains(r,lo,exact,hi,tol=1e-10):
    a,x,b=(float(r[k]) for k in (lo,exact,hi))
    return all(map(math.isfinite,(a,x,b))) and a-tol<=x<=b+tol


def has_interior_minimum(widths):
    if len(widths)<3:return False
    k=min(range(len(widths)),key=widths.__getitem__)
    return 0<k<len(widths)-1 and widths[k]<min(widths[0],widths[-1])


def main():
    checks=[]
    def add(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    data={}
    try:
        for name,count in EXPECTED_ROWS.items():
            data[name]=read(name)
            add(f'{name}: expected rows',len(data[name])==count,f'{len(data[name])}/{count}')
        def bounded(rows,key,tol,label=None):
            v=max(abs(x) for x in num(rows,key));add(label or key,v<=tol,f'max={v:.3e}, tol={tol:.1e}')
        r=data['m1_convergence.csv']
        for x in r:
            add(f"M1 n={x['n']} energy enclosure",
                float(x['dual_energy'])<=float(x['energy_exact'])+1e-11<=float(x['energy_primal'])+2e-11)
        for key in ('gap','energy_error','L2_grad_error','L2_u_error'):
            v=num(r,key);add('M1 decreasing '+key,all(x>y for x,y in zip(v,v[1:])))
        add('M1 gradient bound',all(float(x['L2_grad_error'])**2<=2*float(x['gap'])+1e-12 for x in r))
        add('M1 interface containment',all(contains(x,'interface_lo','Q_exact','interface_hi') for x in r))
        v=data['validation_1d.csv']
        for key in ('primal_violation','boundary_residual','stationarity_residual','dual_violation','complementarity_residual','gap_identity_residual','charge_flux_residual'):
            bounded(v,key,1e-9,'1D parameter grid '+key)
        add('1D parameter grid energy enclosure',all(float(x['dual_energy'])<=float(x['energy_exact'])+1e-10<=float(x['energy'])+2e-10 for x in v))
        add('1D parameter grid gradient bound',all(float(x['grad_error'])**2<=2*float(x['gap'])+1e-10 for x in v))
        applicable=[x for x in v if 0<float(x['b'])<2 and 0<float(x['delta'])<float(x['b'])]
        add('1D applicable interface containment',all(contains(x,'interval_lo','width_exact','interval_hi') for x in applicable),f'{len(applicable)} applicable cases; others explicitly not applicable')
        q=data['m2_charge_intervals.csv'];add('M2 charge containment',all(contains(x,'Q_minus','Q_exact','Q_plus') for x in q))
        q=sorted(q,key=lambda x:float(x['h']));widths=num(q,'width')
        add('M2 resolved interior width minimum',has_interior_minimum(widths),f'h={q[min(range(len(q)),key=widths.__getitem__)]["h"]}')
        bounded([dict(residual=float(x['width'])-float(x['Q_plus'])+float(x['Q_minus'])) for x in q],'residual',1e-12,'M2 width recomputation')
        c=data['m2_capacitance_intervals.csv']
        for stem in ('average','point'):
            add('M2 '+stem+' capacitance containment',all(contains(x,f'C_minus_{stem}',f'C_{stem}_exact',f'C_plus_{stem}') for x in c))
        p=data['m2_parameter_sweep.csv']
        add('M2 multiple biases charge containment',all(contains(x,'Q_minus','Q_exact','Q_plus') for x in p))
        add('M2 multiple biases average capacitance',all(contains(x,'C_average_minus','C_average_exact','C_average_plus') for x in p))
        add('M2 smooth-branch point capacitance',all(contains(x,'C_point_minus','C_point_exact','C_point_plus') for x in p if x['point_applicable']=='True'))
        non=data['m2_nonmonotone.csv'][0]
        add('M2 nonmonotone feasible candidate',int(non['positive_increments'])>0 and float(non['min_u'])>=0 and float(non['boundary_residual'])<1e-12 and contains(non,'lower','exact_interface','upper'))
        s=data['s2_branches.csv'];full=[x for x in s if float(x['b'])>=2]
        bounded(full,'Q_balance_error',1e-10,'S2 full-depletion charge exactness')
        near=[x for x in s if 1.949<float(x['b'])<2.051]
        add('S2 samples straddle branch transition',any(float(x['b'])<2 for x in near) and any(float(x['b'])>2 for x in near))
        add('S2 numerical charge transition agreement',max(abs(float(x['Q_balance'])-float(x['Q_exact'])) for x in near)<1e-4)
        add('S2 reconstructed right flux transition agreement',max(abs(float(x['right_derivative_flux'])-float(x['right_derivative_exact'])) for x in near)<1e-4)
        pb=data['m5_poisson_boltzmann.csv'];bounded(pb,'first_integral_residual',1e-7,'PB independent first integral');bounded(pb,'bvp_residual',2e-8,'PB BVP residual')
        add('PB nonnegative potential',min(num(pb,'min_u'))>=-1e-10)
        e=num(pb,'relative_difference_percent');add('PB discrepancy decreases on sampled theta',all(x>y for x,y in zip(e,e[1:])))
        for name in ('m3_full_contact.csv','m3_partial_contact.csv','m3_reference.csv'):
            r=data[name]
            for key in ('primal_violation','boundary_residual','stationarity_residual','dual_violation','complementarity_residual','equilibrium_residual','gap_identity_residual','energy_quadrature_residual','dual_quadrature_residual','reciprocity_residual'):
                bounded(r,key,2e-8,name+' '+key)
            add(name+' nonnegative gap',min(num(r,'gap'))>=-1e-10)
            add(name+' nonnegative multiplier',min(num(r,'lambda_min'))>=-1e-12)
        full=data['m3_full_contact.csv']
        add('2D full contact energy brackets analytic 1/3',all(float(x['D'])<=1/3+1e-10<=float(x['E'])+1e-10 for x in full))
        add('2D full contact harmonic factor',max(abs(x-.5) for x in num(full,'kappa'))<1e-10)
        add('2D full contact charge approaches exact',abs(float(full[-1]['Q_mixed'])-1)<1e-3)
        ref=data['m3_reference.csv']
        add('reference finer than every evaluated hierarchy',int(ref[-1]['nx'])>max(int(x['source_mesh'].split('x')[0]) for x in data['m4_hierarchy.csv']))
        add('reference last mesh change below 0.1%',abs(float(ref[-1]['Q_mixed'])-float(ref[-2]['Q_mixed']))/abs(float(ref[-1]['Q_mixed']))<.001,'mesh difference only, not a bound on true error')
        rb=data['m3_reference_bounds.csv']
        add('fine estimate inside continuum energy enclosures',all(contains(x,'Q_lower','Q_reference','Q_upper') for x in rb))
        for x in rb:
            add(f"reference secant arithmetic {x['nx']}/{x['h']}",abs((float(x['D_center'])-float(x['E_minus']))/float(x['h'])-float(x['Q_lower']))<1e-10 and abs((float(x['E_plus'])-float(x['D_center']))/float(x['h'])-float(x['Q_upper']))<1e-10)
        for key in ('equilibrium_residual','gap_identity_residual'):bounded(rb,key,2e-8,'reference enclosures '+key)
        cross=data['solver_crosscheck_2d.csv'];add('independent optimizer success',all(x['optimizer_success']=='True' for x in cross))
        bounded(cross,'projected_gradient',1e-6,'independent optimizer KKT');bounded(cross,'max_u_difference',2e-6,'independent optimizer potential agreement');bounded(cross,'energy_difference',1e-9,'independent optimizer energy agreement')
        recip=data['m3_reciprocity.csv']
        add('same metric reciprocity',all(abs(float(x['residual']))<1e-10 for x in recip[:2]))
        add('cross metric deliberately breaks reciprocity',abs(float(recip[2]['residual']))>1e-6)
        bounded(data['m3_single_direction.csv'],'gain_residual',1e-10,'harmonic direction gain identity')
        for nx in (64,128):
            s=sorted([x for x in data['s1_contact_width.csv'] if int(x['nx'])==nx],key=lambda x:float(x['contact_width']))
            for key in ('Q_mixed','kappa'):
                v=num(s,key);add(f'contact sweep nx={nx} increasing {key}',all(a<b for a,b in zip(v,v[1:])))
        h=data['m4_hierarchy.csv'];add('hierarchy numerical gap trend',all(float(x['gap'])>=float(y['gap'])-1e-10 for x,y in zip(h,h[1:])))
        add('hierarchy uses separate fine reference',all(int(x['reference_nx'])>int(x['source_mesh'].split('x')[0]) for x in h))
        local=data['m4_local_response.csv'];bounded(local,'max_abs_BR_plus_I',1e-10,'local response equilibrium')
        for k in ('patch_boundary_flux','outside_divergence','patch_net_divergence'):bounded(local,k,1e-10,'real patch '+k)
        add('patch perturbation nontrivial',float(local[0]['patch_divergence_norm'])>1e-3)
        upd=data['m4_updates.csv']
        for k in ('primal_expansion_residual','dual_expansion_residual','charge_response_residual','response_balance_residual','boundary_residual','equilibrium_residual','gap_identity_residual'):
            bounded(upd,k,1e-9,'accepted updates '+k)
        add('accepted updates nonnegative primal and dual',min(num(upd,'min_u'))>=-1e-12 and min(num(upd,'lambda_min'))>=-1e-12)
        add('accepted updates lower E and raise D',all(float(x['E_after'])<=float(x['E_before'])+1e-12 and float(x['D_after'])>=float(x['D_before'])-1e-12 for x in upd))
        add('accepted updates decrease nonnegative gap',all(-1e-10<=float(x['gap_after'])<=float(x['gap_before'])+1e-12 for x in upd))
        add('local update actually improves gap',float(upd[-1]['gap_after'])<.99*float(upd[0]['gap_before']))
        verify_priority(data,add,bounded,num,contains)
    except (OSError,KeyError,ValueError,IndexError) as exc:
        add('required data complete and finite',False,repr(exc))
    result={'n_checks':len(checks),'n_passed':sum(x['passed'] for x in checks),'checks':checks}
    (TABLES/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=[('PASS' if x['passed'] else 'FAIL')+' '+x['check']+' '+x['detail'] for x in checks]
    (TABLES/'verification.log').write_text('\n'.join(lines)+'\n')
    print(f"verification: {result['n_passed']}/{result['n_checks']}",flush=True)
    if result['n_passed']!=result['n_checks']:
        raise RuntimeError('verification failed:\n'+'\n'.join(t for t in lines if t.startswith('FAIL')))
    return result


if __name__=='__main__':main()
