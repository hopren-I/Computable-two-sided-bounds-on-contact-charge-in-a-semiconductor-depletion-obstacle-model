"""Independent analytic and scalar-convex checks of the manuscript statements.

These checks complement the written proofs; they are not formal verification.
They do not call the archived FEM solver.
"""
from pathlib import Path
from fractions import Fraction as F
import csv,json,math,random
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)
def J(b,L):
    return (2*b)**1.5/3 if b<L*L/2 else b*b/(2*L)+b*L/2-L**3/24
def Q(b,L):return math.sqrt(2*b) if b<L*L/2 else b/L+L/2

# Direct integration of the polynomial full-depletion energy, independently
# from the legacy analytic helper.
for L in (.25,1.,2.,3.7):
    for ratio in (1.,1.01,2.,10.):
        b=ratio*L*L/2
        # u=x^2/2-(b/L+L/2)x+b; u'=x-A.
        A=b/L+L/2
        integrated=.5*(L**3/3-A*L*L+A*A*L)+L**3/6-A*L*L/2+b*L
        check(f'full energy integration {L}/{ratio}',abs(integrated-J(b,L))<1e-10)
    bt=L*L/2
    check(f'branch energy match {L}',abs((2*bt)**1.5/3-J(bt,L))<1e-12)
    check(f'branch charge match {L}',abs(math.sqrt(2*bt)-Q(bt,L))<1e-12)

# Energy enclosures with deliberately asymmetric errors, including the
# non-differentiable convex value function. No smoothness is used here.
rng=random.Random(20261112)
for i in range(500):
    b=rng.uniform(.1,4);k=rng.uniform(.001,b)
    L=rng.uniform(.4,3)
    jc,jl,jr=J(b,L),J(b-k,L),J(b+k,L)
    e=[rng.random()*.02 for _ in range(6)]
    Uc,Lc=jc+e[0],jc-e[1];Ul,Ll=jl+e[2],jl-e[3];Ur,Lr=jr+e[4],jr-e[5]
    lo=(Lc-Ul)/k;hi=(Ur-Lc)/k
    check(f'asymmetric enclosure {i}',lo<=Q(b,L)+1e-11<=hi+2e-11)
    width=(jr+jl-2*jc+e[4]+e[2]+2*e[1])/k
    check(f'exact width split {i}',math.isclose(hi-lo,width,rel_tol=1e-10,abs_tol=1e-10))
    if b-k>1e-8:
        M=max(1/L,1/math.sqrt(2*(b-k)))
        upper=M*k+((Uc-Lc)*2+(Ul-Ll)+(Ur-Lr))/k
        check(f'Lipschitz width {i}',hi-lo<=upper+1e-9)
    # J(x)=|x-b|+x^2/2, subgradient at b is [b-1,b+1].
    phi=lambda x:abs(x-b)+x*x/2
    l=(phi(b)-phi(b-k))/k;r=(phi(b+k)-phi(b))/k
    check(f'nondifferentiable support {i}',l<=b-1+1e-10 and r>=b+1-1e-10)

# Exactly integrate the counterexample's piecewise affine flux using rational
# arithmetic. This is independent of the FEM quadrature in experiments.py.
def int_square(c,slope,h):return c*c*h+c*slope*h*h+slope*slope*h*h*h/3
for t in (F(0),F(1,5),F(2,5),F(3,5),F(4,5)):
    a=-F(1,5)+F(3,20)*t;z=F(3,5)-F(13,20)*t;c=-1+t/80
    norm=int_square(c,F(1),F(1))
    norm+=int_square(c+1,a,F(1,2))
    norm+=int_square(c+1+a/2,z,F(1,2))
    dual=-c/2-norm/2
    defect=(1-t)**2/300-t*(1-t)/1200+t*t/3840
    check(f'exact rational counterexample {t}',F(1,3)-dual==defect)
check('counterexample monotone derivative',-F(3,400)+F(17,1920)*F(4,5)==-F(1,2400))

# Re-evaluate every stored 2D endpoint from raw energy samples, without calling
# the routine that generated the intervals. This checks all 75 configurations.
def read(name):
    with (ROOT/'data'/name).open() as f:return list(csv.DictReader(f))
samples=read('two_dimensional_samples.csv');rows=read('two_dimensional_bounds.csv')
idx={(int(x['nx']),round(float(x['b']),10)):x for x in samples}
for i,r in enumerate(rows):
    n=int(r['nx']);b=float(r['b']);k=float(r['k'])
    c=idx[n,round(b,10)];l=idx[n,round(b-k,10)];z=idx[n,round(b+k,10)]
    lo=(float(c['D'])-float(l['E']))/k;hi=(float(z['E'])-float(c['D']))/k
    check(f'2d raw endpoint arithmetic {i}',abs(lo-float(r['Q_lower']))<1e-12 and abs(hi-float(r['Q_upper']))<1e-12)

result=dict(all_passed=all(x['passed'] for x in checks),check_count=len(checks),checks=checks)
(ROOT/'verification/theory_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
