"""Identical axes and color scales; only scatter displays may be decimated."""
import numpy as np
import matplotlib.pyplot as plt
from .metrics import NAMES, per_pixel_isi


def spatial(e, signed=False):
    a = np.zeros((e.height,e.width))
    np.add.at(a,(e.y,e.x),e.p if signed else 1)
    return a


def compare_streams(reference, modified, title='Reference vs modified', max_display=20_000):
    fig, ax = plt.subplots(3,2,figsize=(12,9), constrained_layout=True)
    vmax = max(1,spatial(reference).max(),spatial(modified).max())
    for col,(e,label) in enumerate(((reference,'Reference'),(modified,'Modified'))):
        ax[0,col].imshow(spatial(e),vmin=0,vmax=vmax,cmap='magma')
        ax[0,col].set(title=f'{label}: {len(e):,} events',xlabel='x (px)',ylabel='y (px)')
        ids = np.linspace(0,len(e)-1,min(len(e),max_display),dtype=int) if len(e) else []
        for row,coords,name,limit in ((1,e.x,'x',e.width),(2,e.y,'y',e.height)):
            ax[row,col].scatter(e.t[ids],coords[ids],c=e.p[ids],cmap='bwr',vmin=-1,vmax=1,s=3)
            ax[row,col].set(xlim=(e.start,e.stop),ylim=(-.5,limit-.5),xlabel='Source-clock time (s)',ylabel=name+' (px)',
                            title=f'{len(ids):,} displayed; full stream scored')
            ax[row,col].ticklabel_format(axis='x',useOffset=False)
    fig.suptitle(title)
    return fig


def distributions(reference, modified):
    fig,ax = plt.subplots(1,3,figsize=(14,3.5),constrained_layout=True)
    edges = np.linspace(reference.start,reference.stop,51)
    intervals = [per_pixel_isi(e) for e in (reference,modified)]
    positive = np.concatenate([z[z>0] for z in intervals])
    isi_edges = np.geomspace(max(1e-9,positive.min()),max(1e-8,positive.max())*1.001,35) if len(positive) else None
    for e,label,isi in zip((reference,modified),('Reference','Modified'),intervals):
        rate = np.histogram(e.t,edges)[0]/np.diff(edges)
        ax[0].stairs(rate,edges,label=label)
        if isi_edges is not None:
            ax[1].hist(isi[isi>0],bins=isi_edges,histtype='step',label=f'{label}; zero ISIs={(isi==0).sum()}')
        ax[2].plot(['OFF','ON'],[(e.p<0).sum(),(e.p>0).sum()],marker='o',label=label)
    ax[0].set(xlabel='Source-clock time (s)',ylabel='Events / s',title='Event rate; shared bins')
    ax[1].set(xlabel='Per-pixel ISI (s)',ylabel='Interval count',xscale='log',title='ISI distribution; polarity pooled')
    ax[2].set(ylabel='Event count',title='Polarity counts')
    for a in ax: a.legend(fontsize=8)
    return fig


def response_plot(table, family):
    fig, axes = plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for ax,metric in zip(axes.ravel(),NAMES):
        g = table[(table.family==family)&(table.metric==metric)]
        good = g[g.status=='ok'].groupby('severity').discrepancy
        means,stds = good.mean(),good.std().fillna(0)
        ax.errorbar(means.index,means,yerr=stds,marker='o',capsize=2,label='Mean +/- SD across seeds')
        bad = g[g.status!='ok']
        if len(bad):
            ax.scatter(bad.severity,bad.discrepancy,marker='x',c='orange',label='Partial/approximate/undefined')
        ax.set(title=metric,xlabel='Severity in this family',ylabel='Within-metric discrepancy')
        ax.axhline(0,color='.7',lw=.6)
        if len(bad): ax.legend(fontsize=6)
    fig.suptitle(f'{family}: no cross-metric scale comparison')
    return fig


def heatmap(frame, title, vmin=-1, vmax=1, cmap='coolwarm'):
    fig,ax = plt.subplots(figsize=(max(8,.9*len(frame.columns)),max(3,.4*len(frame.index))),constrained_layout=True)
    im = ax.imshow(frame.to_numpy(float),vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    ax.set_xticks(range(len(frame.columns)),frame.columns,rotation=50,ha='right')
    ax.set_yticks(range(len(frame.index)),frame.index)
    ax.set_title(title+' (blank = undefined / insufficient)')
    fig.colorbar(im,ax=ax)
    return fig


def case_gallery(selected_cases):
    fig,axes = plt.subplots(2,2*len(selected_cases),figsize=(7*len(selected_cases),6),constrained_layout=True,squeeze=False)
    for j,c in enumerate(selected_cases):
        vmax=max(1,spatial(c.reference).max(),spatial(c.modified).max())
        for offset,(e,label) in enumerate(((c.reference,'Reference'),(c.modified,'Modified'))):
            col=2*j+offset
            axes[0,col].imshow(spatial(e),vmin=0,vmax=vmax,cmap='magma')
            axes[0,col].set(title=f'{label}: N={len(e)}',xlabel='x (px)',ylabel='y (px)')
            axes[1,col].scatter(e.t,e.x,c=e.p,cmap='bwr',vmin=-1,vmax=1,s=25)
            axes[1,col].set(xlim=(e.start,e.stop),ylim=(-.5,e.width-.5),xlabel='Time (s)',ylabel='x (px)')
    fig.suptitle(selected_cases[0].metric+' | '+ ' / '.join(c.name for c in selected_cases),fontsize=10)
    return fig
