"""Compare the five typeset tables and Fig. 3 coordinates to numerical records."""
from pathlib import Path
import csv
import math
import re
import sys

root = Path(__file__).resolve().parents[1]
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data"
result = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "result"
tex = (root / "verification/numerical_excerpts.tex").read_text()
tables = re.findall(r"\\begin\{tabular\}.*?\\midrule(.*?)\\bottomrule", tex, re.S)
assert len(tables) == 5

def records(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows

def lines(index):
    body = re.sub(r"%[^\n]*", "", tables[index]).replace(r"\midrule", "")
    return [[v.strip().strip("$") for v in row.strip().split("&")]
            for row in body.split(r"\\") if row.strip()]

def match(cells, values):
    assert len(cells) == len(values)
    for s, value in zip(cells, values):
        if s == r"\approx0":
            assert abs(value) < 1e-12, value
        elif isinstance(value, str):
            assert s == value, (s, value)
        else:
            # Compare at the precision printed in the corresponding cell.
            mantissa, _, exp = s.lower().partition("e")
            unit = 10 ** (int(exp or 0) - len(mantissa.partition(".")[2]))
            assert abs(float(s)-value) <= 0.5000001 * unit, (s, value)

def check_table(number, rows):
    source = lines(number-1)
    assert len(source) == len(rows)
    for cells, values in zip(source, rows):
        match(cells, values)
    print("Table", number, "PASS")

hv = records(result/"hv2013_1d_intervals.csv")
assert len(hv) == 36 and all(r["Q_lower"] < r["Q_ref"] < r["Q_upper"] for r in hv)
check_table(1, [[r[k] for k in ("f","b","Q_ref","Q_lower","Q_upper","relative_full_width_pct")]
                for r in hv if r["n"] == 256 and r["k_over_b"] == .01])
bounds = records(data/"two_dimensional_bounds.csv")
samples = {(r["nx"],round(r["b"],12)):r for r in records(data/"two_dimensional_samples.csv")}
assert len(bounds) == 75 and len(samples) == 175
meshes = (32,64,128,256,512)
biases = (.05,.2,.5,1.5,3)
best = {(n,b): min((r for r in bounds if r["nx"]==n and r["b"]==b),key=lambda r:r["width"])
        for n in meshes for b in biases}
def q(n,b):
    return samples[n,b]["Q_mixed"]
for r in bounds:
    b,k,n = r["b"],r["k"],r["nx"]
    lo = (samples[n,b]["D"]-samples[n,round(b-k,12)]["E"])/k
    hi = (samples[n,round(b+k,12)]["E"]-samples[n,b]["D"])/k
    assert max(abs(lo-r["Q_lower"]),abs(hi-r["Q_upper"])) < 1e-9
rows=[]
for b in biases:
    r=best[512,b]
    rows.append([b,r["k"]/b,q(512,b),r["Q_lower"],r["Q_upper"],
                 100*r["width"]/abs(q(512,b)),100*abs(q(512,b)-q(256,b))/abs(q(512,b))])
check_table(2,rows)
rows=[]
for n in meshes:
    r=best[n,.5]
    rows.append([rf"{n}\times{n//2}",samples[n,.5]["gap"],r["k"]/.5,q(n,.5),100*r["width"]/q(512,.5)])
r=min(records(data/"two_dimensional_extension_bounds.csv"),key=lambda r:r["width"])
rows.append([r"1024\times512",r["gap_center"],r["k"]/.5,r["Q_mixed"],100*r["width"]/q(512,.5)])
check_table(3,rows)
full=records(data/"two_dimensional_full_contact_bounds.csv")
decomp=records(result/"truth_decomposition.csv")
assert len(full)==len(decomp)==120 and all(r["Q_lower"]<=r["Q_exact"]<=r["Q_upper"] for r in full)
bf=(.005,.37,1.99,2.,2.01,3.)
def at128(dataset):
    return [next(r for r in dataset if r["nx"]==128 and r["b"]==b and abs(r["k_over_b"]-.1)<1e-12) for b in bf]
frows=at128(full)
drows=at128(decomp)
cols=[[r[key] for r in frows] for key in ("Q_exact","Q_lower","Q_upper","width","midpoint_abs_error")]
cols += [[r[key] for r in drows] for key in ("W_secant","W_primal","W_dual","secant_percent")]
cols.append([100*r["error_over_radius"] for r in drows])
cells=lines(3)
assert len(cells)==len(cols)
for row,col in zip(cells,cols):
    match(row[1:],col)
print("Table 4 PASS")
disk=records(result/"hv2014_disk.csv")
check_table(5,[[f"1/{round(1/r['h'])}",r["R_h"],r["abs_radius_error"],r["J_h"],r["energy_error"]] for r in disk])

# The final TikZ file embeds the plotting coordinates. Read them back using
# the axis origin, span and scales given by its tick marks and panel layout.
drawing=(root/"figure/charge_intervals_2d.tex").read_text()
def points(line):
    return [(float(x),float(y)) for x,y in re.findall(r"\((\d+\.\d+),(\d+\.\d+)\)",line)]
def close(left,right):
    assert len(left)==len(right)
    for p,v in zip(left,right):
        assert max(abs(i-j) for i,j in zip(p,v)) <= .000051,(p,v)
for i,n in enumerate(meshes):
    prefix=rf"\draw[c{i},line width="
    vertical=[points(line) for line in drawing.splitlines() if line.startswith(prefix+".95pt]")]
    assert len(vertical)==5
    right=next(points(line) for line in drawing.splitlines() if line.startswith(prefix+"1.1pt,line join"))
    expected=[]
    for j,b in enumerate(biases):
        r=best[n,b];x=1.735+1.51*j+.205*(i-2)
        close(vertical[j],[(x,1.83+4.86/2.8*r[key]) for key in ("Q_lower","Q_upper")])
        expected.append((10.17+2.48*b,1.83+.0648*100*r["width"]/abs(q(512,b))))
    close(right,expected)
    inset=next(points(line) for line in drawing.splitlines() if line.startswith(prefix+"1.05pt]"))
    close(inset,[(2.36+.64*i,5.4764+4.48*(best[n,.5][k]-.90)) for k in ("Q_lower","Q_upper")])
print("Fig. 3 PASS: 25 intervals, 25 relative widths and 5 inset intervals.")

# Verify the published precision of the numerical comparisons in the prose.
def printed(text, value, fmt):
    number = format(value, fmt)
    assert number in text, (number,text[:50])

def paragraph(start):
    return next(p for p in tex.split("\n\n") if p.lstrip().startswith(start))

p=paragraph("三个载荷下共计算")
for n in (32,256):
    h=next(h for h in hv if h["f"]==-10 and h["n"]==n and h["k_over_b"]==.01)
    printed(p,h["relative_full_width_pct"],".3f")
p=paragraph("另取")
for r in records(data/"branch_transition_bounds.csv"):
    if r["n"]==64 and r["b"] in (.005,3.) and math.isclose(r["k"]/r["b"],.1):
        for k in ("Q_lower","Q_upper"):
            printed(p,r[k],".6f")
p=paragraph(r"$n=256$ 时，在所取")
scan=records(data/"one_dimensional_bounds.csv")
assert len(scan)==44
r=min((r for r in scan if r["n"]==256),key=lambda r:r["width"])
for v in [r["Q_lower"],r["Q_upper"],r["width"],r["exact_curvature_term"],r["width"]-r["exact_curvature_term"]]:
    printed(p,v,".6f")
p=paragraph("在 $b=0.5$ 时，将网格加密")
r=min(records(data/"two_dimensional_extension_bounds.csv"),key=lambda r:r["width"])
for k in ("Q_lower","Q_upper","mesh_change_percent_Q512"):
    printed(p,r[k],".6f")
p=paragraph("固定 $b=0.5$")
strategies={r["method"]:r for r in records(result/"step_benchmark.csv") if r["nx"]==128}
fixed,search=strategies["fixed"],strategies["search"]
opt=next(r for r in records(result/"recovery_comparison.csv") if r["nx"]==128)
for v in (fixed["W"],search["W"],opt["W_optimized"]):
    printed(p,v,".6f")
printed(p,100*(1-search["W"]/fixed["W"]),".2f")
printed(p,opt["width_reduction_percent"],".2f")
p=paragraph(r"表\ref{tab:fullcontact}显示")
for n in (32,512):
    r=next(r for r in full if r["nx"]==n and r["b"]==.37 and r["k_over_b"]==.1)
    printed(p,r["width"],".6f")
printed(p,at128(decomp)[1]["W_secant"],".6f")
printed(p,100-at128(decomp)[0]["secant_percent"],".2f")
print("Published numerical comparisons PASS.")
