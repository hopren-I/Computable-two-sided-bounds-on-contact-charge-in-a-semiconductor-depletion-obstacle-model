"""Regenerate manuscript Figure 1 with NumPy, Matplotlib."""
from pathlib import Path
import csv
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

here = Path(__file__).resolve().parent
src = here.parent / 'data' / 'one_dimensional_bounds.csv'
rows=list(csv.DictReader(src.open()))
ns=sorted(set(int(r['n']) for r in rows))
ks=sorted(set(float(r['k']) for r in rows))
lookup={(int(r['n']),float(r['k'])):r for r in rows}
assert len(rows)==len(lookup)==44 and all(float(r['b'])==0.5 and float(r['Q_exact'])==1 for r in rows)
for r in rows:
    assert math.isclose(float(r['width']), float(r['Q_upper'])-float(r['Q_lower']),abs_tol=1e-12)
    assert float(r['width_majorant'])>=float(r['width'])>=float(r['exact_curvature_term'])
width=np.array([[float(lookup[n,k]['width']) for k in ks] for n in ns])
bound=np.array([float(lookup[256,k]['width_majorant']) for k in ks])
regular = fm.FontProperties(family='DejaVu Sans')
bold = fm.FontProperties(family='DejaVu Sans', weight='bold')
from matplotlib.ticker import NullFormatter
# Match the 160 mm text block so the sizes below are also the printed sizes.
mpl.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'axes.unicode_minus':False, 'pdf.fonttype':42})
ink = '#172E38'
muted = '#344A52'
slate = '#535E8B'
teal = '#006F69'
fig = plt.figure(figsize=(160/25.4, 92/25.4), facecolor='white')
def label(x, y, text, size, font=regular, **kwargs):
    fig.text(x, y, text, va='center', fontproperties=font, fontsize=size,
             color=ink, **kwargs)

label(.03,.943,'Charge interval width and bias step in 1D',10.4,bold)
label(.979,.943,r'Central bias $b=0.5$, exact charge $Q(b)=1$',7.8,ha='right')
label(.03,.832,'A  Mesh refinement and step selection',8.4,bold)
label(.547,.832,'B  Computed width and upper bound',8.4,bold)
label(.03,.753,'◇  Narrowest interval on each mesh',8.4)
label(.547,.775,'n = 256',8.4)
left = fig.add_axes([.112,.22,.373,.46])
right = fig.add_axes([.589,.22,.382,.46])
for ax in (left, right):
    ax.set(xscale='log', yscale='log')
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color('#8B9CA1')
        ax.spines[side].set_linewidth(.45)
    ax.tick_params(which='major',length=2.4,width=.5,pad=2.8,labelsize=8.1,labelcolor=muted,color=muted)
    ax.tick_params(which='minor',length=0)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1e-4,1e-3,1e-2,.1],['0.0001','0.001','0.01','0.1'])
    ax.grid(axis='y',which='major',color='#DEE5E6',lw=.38)
left.set_ylabel('Full width  W',fontproperties=regular,fontsize=9,labelpad=2,color=ink)
left.set_xlim(.00008,.13)
left.set_ylim(.003,6.8)
left.set_yticks([.005,.01,.1,1,5],['0.005','0.01','0.1','1','5'])
for i,(col,shape) in enumerate(zip([slate,'#975C34','#437E96',teal],['s','^','o','o'])):
    y = width[i]
    best = int(y.argmin())
    left.plot(ks,y,color=col,lw=.9 if i==3 else .75,marker=shape,ms=2.25,mew=.32,solid_capstyle='round',label=f'n = {ns[i]}',zorder=3)
    left.plot(ks[best],y[best],marker='D',lw=0,mfc='white',mec=col,mew=.75,ms=3.8,zorder=6)
left.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=ink,
            borderaxespad=.2, handlelength=1.6, handletextpad=.5, labelspacing=.2)
right.set_xlim(.000073,.155)
right.set_ylim(.00255,.16)
right.set_yticks([.003,.01,.03,.1],['0.003','0.01','0.03','0.1'])
right.fill_between(ks,width[-1],bound,color='#EDF4F3',edgecolor='none',zorder=1)
right.plot(ks,bound,color=slate,lw=.85,ls=(0,(4,2.8)),label='Upper bound',zorder=3)
right.plot(ks,width[-1],color=teal,marker='o',ms=2.35,mew=.32,lw=.95,solid_capstyle='round',label='Computed width',zorder=4)
j=int(width[-1].argmin())
right.plot(ks[j],width[-1,j],marker='D',ms=4,mew=.8,mec=teal,mfc='white',zorder=5)
for series,offset,col in [(width[-1],(.005,.0031),teal),(bound,(.00145,.016),slate)]:
    right.annotate(f'{series[j]:.5f}',(ks[j],series[j]),offset,
                   fontsize=8.6,color=col,ha='left',va='center',
                   arrowprops=dict(arrowstyle='-',color=col,lw=.48,shrinkA=2,shrinkB=2))
legend_font = regular.copy()
legend_font.set_size(6.8)
right.legend(loc='upper right',bbox_to_anchor=(1.02,1.17),ncol=2,frameon=False,
             columnspacing=.7,handlelength=1.3,handletextpad=.35,prop=legend_font)
for x in (.3,.78):
    label(x,.10,'Bias step  k',9,ha='center')
fig.savefig(here / 'figure1-lines.pdf', bbox_inches='tight', pad_inches=0.015)
