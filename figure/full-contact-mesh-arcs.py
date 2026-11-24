from pathlib import Path
from math import isclose, sqrt
import csv

data = Path(__file__).resolve().parents[1] / 'data' / 'two_dimensional_full_contact_bounds.csv'
with data.open() as stream:
    rows = sorted((r for r in csv.DictReader(stream) if isclose(float(r['b']), .37) and isclose(float(r['k']), .037)), key=lambda r: int(r['nx']))
assert len(rows) == 5
values = [(float(r['Q_lower']), float(r['Q_upper']), float(r['width'])) for r in rows]
q = sqrt(.74)
assert all(isclose(hi - lo, width) and lo < q < hi for lo, hi, width in values)
assert all(a[0] < b[0] < b[1] < a[1] for a,b in zip(values, values[1:]))
ink = ['9A563B','7B736A','627A77','497878','246368']
wash = ['F6EBE0','F6F0E8','EDF0E8','E2ECE8','D1E6DF']

pieces = [r'''\documentclass{article}
\input{figure-page.tex}
\definecolor{ink}{HTML}{203A3C}
\definecolor{sub}{HTML}{445957}
\definecolor{axis}{HTML}{81938B}
\definecolor{acc}{HTML}{965236}''']
for i in range(5):
    pieces += [rf'\definecolor{{c{i}}}{{HTML}}{{{ink[i]}}}',rf'\definecolor{{f{i}}}{{HTML}}{{{wash[i]}}}']
pieces += [r'''\begin{document}\startplot
\begin{tikzpicture}[x=1cm,y=1cm,inner sep=0pt,outer sep=0pt,text=black]
\path[use as bounding box] (0,0) rectangle (17.77,6.25);''']

def text(x,y,value,style=''):
    pieces.append(rf'\node[{style}] at ({x:.5f},{y:.5f}) {{{value}}};')
def line(x,y,xx,yy,style):
    pieces.append(rf'\draw[{style}] ({x:.5f},{y:.5f}) -- ({xx:.5f},{yy:.5f});')

def xval(q):
    return .48+(q-.8285)*16.84/.063
base = 1.12
height = [4.82,3.92,3.02,2.12,1.22]
for i, (lo,hi,w) in enumerate(values):
    rx = (xval(hi)-xval(lo))/2
    arc = rf'({xval(lo):.6f},{base:.6f}) arc[start angle=180,end angle=0,x radius={rx:.6f}cm,y radius={height[i]:.6f}cm]'
    pieces += [rf'\path[fill=f{i}] {arc} -- cycle;',rf'\draw[c{i},line width={0.86 if i in (0,4) else 0.62}pt] {arc};']

for i, (r,(lo,hi,w)) in enumerate(zip(rows,values)):
    cx = (xval(lo)+xval(hi))/2 - .55
    cy = base+height[i]-.44
    style = r'font=\sffamily\small,text=black'
    text(cx-.20,cy,rf"${r['nx']}\!\times\!{r['ny']}$",style+',anchor=east')
    line(cx,cy-.095,cx,cy+.095,'c'+str(i)+',line width=.42pt')
    text(cx+.20,cy,rf'$W={w:.6f}$',style+',anchor=west')

line(xval(.8300),base,xval(.8900),base,'axis,line width=.57pt')
for i in range(830,891):
    x=xval(i/1000)
    if i%5==0:
        line(x,base,x,base-.13,'axis,line width=.48pt')
        if i%10==0 and i!=860:
            text(x,base-.22,f'{i/1000:.2f}',r'anchor=north,text=black,font=\small')
    else:
        line(x,base,x,base-.060,'axis,line width=.29pt')
for i, (lo,hi,w) in enumerate(values):
    for p in (lo,hi):
        line(xval(p),base+.057,xval(p),base-.040,'c'+str(i)+',line width=.7pt')
text(xval(.8295),base+.28,r'Lower bound $Q_-$',r'anchor=west,font=\sffamily\small,text=black')
text(xval(.8905),base+.28,r'Upper bound $Q_+$',r'anchor=east,font=\sffamily\small,text=black')
qx = xval(q)
line(qx,base-.015,qx,base-.45,'ink,line width=.61pt')
pieces.append(rf'\fill[ink] ({qx:.5f},{base:.5f}) ++(0,.062) -- ++(.062,-.062) -- ++(-.062,-.062) -- ++(-.062,.062) -- cycle;')
text(qx,base-.68,r'Exact charge $Q\approx0.860233$',r'font=\sffamily\small')
# Read from the outermost arc to the innermost.
mesh_note_x = .92
line(mesh_note_x-.17, 5.20, mesh_note_x-.17, 5.86, 'c0!66,line width=.6pt')
text(mesh_note_x, 5.70, 'Five meshes (outer to inner)', r'anchor=west,font=\sffamily\small,text=black')
mesh_range = rf"{rows[0]['nx']}\!\times\!{rows[0]['ny']}\;\rightarrow\;{rows[-1]['nx']}\!\times\!{rows[-1]['ny']}"
text(mesh_note_x, 5.27, rf'\({mesh_range}\)', r'anchor=west,font=\small,text=black')
# The two endpoints are identified on the charge axis. Define their span in the margin.
note_x = 14.17
line(note_x-.17, 5.20, note_x-.17, 5.86, 'c4!66,line width=.6pt')
text(note_x, 5.70, 'Full width', r'anchor=west,font=\sffamily\small,text=black')
text(note_x, 5.27, r'$W=Q_+-Q_-$', r'anchor=west,font=\small,text=black')
pieces += [r'\end{tikzpicture}\finishplot\end{document}']
Path(__file__).with_suffix('.tex').write_text('\n'.join(pieces))
