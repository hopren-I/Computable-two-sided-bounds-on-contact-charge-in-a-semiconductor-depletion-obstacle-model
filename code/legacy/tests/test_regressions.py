"""Regression tests for the specific failures found during review."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from src.fem1d import (solve, assemble, solve_obstacle, diagnostics, charge_balance,
                       reconstruct_flux, exact_l2_errors, kkt_residuals, interface_interval)
from src.fem2d import Mesh2D, assemble_q1, solve_q1, lambda_from_q1, assemble_rt0, contact_j_range
from src.analytic1d import analytic
from src.experiments_2d import _evaluate, _q_direct
from src.verify_results import has_interior_minimum, contains


class RegressionTests(unittest.TestCase):
    def test_poisson_initialized_active_set_agrees_with_cold_start(self):
        from src.fem2d import dirichlet_nodes
        mesh=Mesh2D(32,16);K,f=assemble_q1(mesh)
        for y0,y1 in ((0.,1.),(.25,.75),(.125,.5)):
            for b in (0.,.05,.5,1.5,3.):
                with self.subTest(contact=(y0,y1),b=b):
                    fixed,values=dirichlet_nodes(mesh,y0,y1,b)
                    cold=solve_obstacle(K,f,fixed,values,poisson_start=False)[0]
                    warm=solve_obstacle(K,f,fixed,values,poisson_start=True)[0]
                    np.testing.assert_allclose(cold,warm,atol=1e-11,rtol=1e-11)
                    self.assertLess(max(kkt_residuals(K,f,warm,fixed,values).values()),1e-10)

    def test_controlled_counterexample_has_mesh_independent_analytic_dual_energy(self):
        from src.priority_experiments import counterexample_rows
        coarse=counterexample_rows(32);fine=counterexample_rows(128)
        for a,b in zip(coarse,fine):
            self.assertAlmostEqual(a['D'],b['D'],places=12)
            self.assertAlmostEqual(a['Q_flux'],b['Q_flux'],places=12)
            self.assertLess(abs(a['analytic_D_residual']),1e-12)
            self.assertLess(abs(a['analytic_Q_residual']),1e-12)
            self.assertLess(abs(a['dual_quadrature_residual']),1e-12)
        for case in ('fixed_multiplier_flux','varying_multiplier_recovery'):
            g=[r for r in fine if r['case']==case]
            self.assertGreater(g[-1]['D'],g[0]['D'])
            self.assertGreater(g[-1]['flux_charge_abs_error'],g[0]['flux_charge_abs_error']+.005)

    def test_bias_secants_enclose_analytic_charge_on_both_branches(self):
        from src.priority_experiments import secant_row
        for b in (.05,.5,2.,3.):
            h=b*.1
            def sample(x):
                E=analytic([0],x)['energy']
                return dict(nx=16,ny=8,b=x,E=E,D=E)
            r=secant_row(sample(b),sample(b-h),sample(b+h),h)
            self.assertLessEqual(r['Q_lower'],analytic([0],b)['Q']+1e-12)
            self.assertGreaterEqual(r['Q_upper'],analytic([0],b)['Q']-1e-12)

    def test_bias_secants_reject_mismatched_sample_locations(self):
        from src.priority_experiments import secant_row
        sample=lambda b:dict(nx=16,ny=8,b=b,E=1.,D=.9)
        with self.assertRaises(ValueError):secant_row(sample(.5),sample(.4),sample(.7),.1)

    def test_small_bias_feasibility_and_energy_enclosure(self):
        for b in (0.,1e-8,.00390625,.05):
            with self.subTest(b=b):
                d=diagnostics(16,b)
                self.assertGreaterEqual(d['u'].min(),0.)
                self.assertGreaterEqual(d['energy']+1e-12,d['energy_exact'])
                self.assertLessEqual(d['dual_energy'],d['energy_exact']+1e-12)
                self.assertLess(d['stationarity_residual'],1e-11)

    def test_solver_exhaustion_is_error(self):
        _,K,f=assemble(16)
        with self.assertRaises(RuntimeError):solve_obstacle(K,f,np.array([0,16]),np.array([.5,0.]),max_iter=1)

    def test_negative_bias_rejected(self):
        with self.assertRaises(ValueError):solve(16,-.1)

    def test_balance_integral_matches_flux_for_arbitrary_multiplier(self):
        x=np.linspace(0,2,17);lam=np.linspace(.1,.9,16)
        sigma,_=reconstruct_flux(lam,x,.37)
        self.assertAlmostEqual(charge_balance(lam,x,.37),-sigma[0],places=13)

    def test_full_depletion_charge_has_no_h_over_two_bias(self):
        for n in (16,64):
            d=diagnostics(n,2.5)
            self.assertAlmostEqual(d['Q_balance'],2.25,places=12)

    def test_offgrid_error_quadrature(self):
        b=.37;s=solve(16,b);w=np.sqrt(2*b)
        l2,g2=exact_l2_errors(s,b,2.)
        err=grad=0.
        for i,(a,z) in enumerate(zip(s.x[:-1],s.x[1:])):
            slope=(s.u[i+1]-s.u[i])/(z-a)
            points=[w] if a<w<z else None
            err+=quad(lambda x:(s.u[i]+slope*(x-a)-.5*max(w-x,0)**2)**2,a,z,points=points,epsabs=1e-14)[0]
            grad+=quad(lambda x:(slope+max(w-x,0))**2,a,z,points=points,epsabs=1e-14)[0]
        self.assertAlmostEqual(l2*l2,err,places=13)
        self.assertAlmostEqual(g2*g2,grad,places=13)

    def test_interface_preconditions(self):
        self.assertTrue(np.isnan(interface_interval(solve(16,2.5),.01)[2]))
        self.assertTrue(np.isnan(interface_interval(solve(16,.01),1.)[2]))

    def test_affine_q1_energy_on_rectangular_cells(self):
        m=Mesh2D(12,8,3.,1.);K,f=assemble_q1(m)
        x=np.repeat(m.xs(),m.ny+1);y=np.tile(m.ys(),m.nx+1);u=2*x+3*y+1
        self.assertAlmostEqual(.5*u@(K@u),.5*(4+9)*3,places=10)
        self.assertAlmostEqual(f@u,3*(3+1.5+1),places=11)

    def test_two_dimensional_kkt(self):
        for y0,y1 in ((0,1),(.25,.75),(.125,.5)):
            s=solve_q1(Mesh2D(32,16),.5,y0,y1)
            r=kkt_residuals(s.K,s.f,s.u,s.fixed,s.fixed_values)
            self.assertLess(max(r.values()),1e-10)

    def test_multiplier_assembled_volume_and_no_dirichlet_reaction(self):
        s=solve_q1(Mesh2D(16,8),.5)
        lam=lambda_from_q1(s,clip=False).reshape(16,8)
        self.assertAlmostEqual(lam[12,4],1.,places=12)
        self.assertLessEqual(lam.max(),1+1e-12)
        full=solve_q1(Mesh2D(16,8),2.5)
        self.assertLess(np.max(lambda_from_q1(full)),1e-12)

    def test_mixed_gap_independent_quadrature(self):
        for y0,y1 in ((0,1),(.25,.75),(.125,.5)):
            r=_evaluate(Mesh2D(16,8),y0,y1)
            for key in ('gap_identity_residual','equilibrium_residual','energy_quadrature_residual','dual_quadrature_residual'):
                self.assertLess(abs(r[key]),1e-10)
            self.assertGreaterEqual(r['gap'],-1e-12)

    def test_direct_flux_exact_on_asymmetric_contact(self):
        m=Mesh2D(16,8);s=solve_q1(m,.5,.125,.5)
        j0,j1=contact_j_range(m,.125,.5)
        # Independent 2-point Gauss evaluation of each edge derivative.
        gp=(.5-.5/np.sqrt(3),.5+.5/np.sqrt(3));q=0.
        for j in range(j0,j1):
            for t in gp:
                left=(1-t)*s.u[m.node(0,j)]+t*s.u[m.node(0,j+1)]
                right=(1-t)*s.u[m.node(1,j)]+t*s.u[m.node(1,j+1)]
                q-=(right-left)/m.hx*m.hy/2
        self.assertAlmostEqual(_q_direct(m,s,j0,j1),q,places=13)

    def test_misaligned_contact_rejected(self):
        with self.assertRaises(ValueError):solve_q1(Mesh2D(16,8),.5,.13,.51)

    def test_nonmonotone_check_rejects_monotone_lists(self):
        self.assertFalse(has_interior_minimum([1,2,3]))
        self.assertFalse(has_interior_minimum([3,2,1]))
        self.assertTrue(has_interior_minimum([3,1,3]))

    def test_containment_recomputed_not_boolean(self):
        self.assertFalse(contains(dict(lo=2,truth=1,hi=3,contains=True),'lo','truth','hi'))
        self.assertFalse(contains(dict(lo=0,truth=float('nan'),hi=3),'lo','truth','hi'))

    def test_missing_results_fail_closed(self):
        import src.verify_results as verify
        with tempfile.TemporaryDirectory() as tmp,patch.object(verify,'TABLES',Path(tmp)):
            with self.assertRaises(RuntimeError):verify.main()

    def test_nondefault_capacitance_and_pb_parameters(self):
        import src.experiments_1d as e
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'TABLES',Path(tmp)),patch.object(e,'FIGS',Path(tmp)):
            r=e.run_m2(n=64,b=.37)
            for c in r['capacitance']:
                self.assertAlmostEqual(c['C_point_exact'],1/np.sqrt(.74))
            pb=e.run_m5(b=.3)
            self.assertAlmostEqual(pb[0]['Q_depletion'],np.sqrt(.6))
            self.assertLess(max(abs(x['first_integral_residual']) for x in pb),1e-7)

if __name__=='__main__':unittest.main()
