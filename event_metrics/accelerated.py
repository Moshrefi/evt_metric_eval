"""Cached reference evaluation. CuPy CUDA arrays; SciPy KD-trees remain on CPU."""
import time
import numpy as np
from . import metrics as m

class Evaluator:
    def __init__(self, reference, cfg, edges, backend='cuda', workers=1):
        if backend not in ('cpu', 'cuda'):
            raise ValueError("backend must be cpu or cuda")
        if backend == 'cuda':
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('No CUDA device visible')
            self.xp = cp
        else:
            self.xp = np
        self.reference, self.cfg = reference, cfg
        self.edges = np.asarray(edges)
        if not np.array_equal(self.edges[[0,-1]], [reference.start,reference.stop]) or np.any(np.diff(self.edges)<=0):
            raise ValueError('Invalid shared voxel edges')
        self.backend, self.workers = backend, workers
        self.r = self.arrays(reference)
        self.ri = self.isi(self.r)
        self.rf = m.fano_values(reference,cfg)
        self.rv = self.voxels(self.r)
        self.groups = {}
        rw = m._window_ids(reference,cfg)
        for w in np.unique(rw):
            for pol in (-1,1):
                a, before = m._embedding(reference,(rw==w)&(reference.p==pol),reference.start+w*cfg.window,cfg)
                self.groups[(int(w),pol)] = (a,before,m.cKDTree(a) if len(a) else None)

    def arrays(self,e):
        return {k:self.xp.asarray(getattr(e,k)) for k in 'txyp'}

    def isi(self,a):
        xp=self.xp
        pix=a['y']*self.reference.width+a['x']
        if len(pix) < 2:
            return xp.empty(0, dtype=xp.float64)
        # CuPy requires a 2-D keys array; last row is the primary key.
        # Float64 retains timestamps and exact pixel IDs for sensor-sized domains.
        order=xp.lexsort(xp.stack((a['t'],pix), axis=0))
        return xp.diff(a['t'][order])[xp.diff(pix[order])==0]

    def voxels(self,a):
        xp=self.xp; r=self.reference
        b=xp.searchsorted(xp.asarray(self.edges),a['t'],side='right')-1
        ids=(b*r.height+a['y'])*r.width+a['x']
        unique,inv=xp.unique(ids,return_inverse=True)
        counts=xp.bincount(inv,weights=a['p'],minlength=len(unique)).astype(xp.int64)
        return unique[counts!=0],counts[counts!=0]

    def chamfer(self,e):
        cfg=self.cfg; r=self.reference
        sw=m._window_ids(e,cfg)
        windows=np.union1d([k[0] for k in self.groups],sw)
        scores=[]; unmatched=groups=dropped=0
        for w in windows:
            for pol in (-1,1):
                a,na,tree=self.groups.get((int(w),pol),(np.empty((0,3)),0,None))
                b,nb=m._embedding(e,(sw==w)&(e.p==pol),r.start+w*cfg.window,cfg)
                dropped+=na-len(a)+nb-len(b)
                if not len(a) and not len(b): continue
                groups+=1
                if not len(a) or not len(b):
                    unmatched+=1; continue
                da=m.cKDTree(b).query(a,workers=self.workers)[0]
                db=tree.query(b,workers=self.workers)[0]
                scores.append(.5*(da.mean()+db.mean()))
        value=np.mean(scores) if scores else np.nan
        if cfg.normalize_chamfer: value/=np.hypot(r.width,r.height)
        status='partial' if unmatched else ('approximate' if dropped else 'ok')
        if not scores: status='undefined'
        return m.result(value,status,'Unmatched window/polarity groups excluded' if unmatched else '',
            unmatched_groups=unmatched,occupied_groups=groups,compared_groups=len(scores),
            coverage=len(scores)/groups if groups else 0.,sampled_away_points=dropped)

    def evaluate(self,e,progress=None):
        r=self.reference; cfg=self.cfg; xp=self.xp
        if (e.width,e.height,e.start,e.stop)!=(r.width,r.height,r.start,r.stop):
            raise ValueError('Observation domains differ')
        a=self.arrays(e); si=self.isi(a); out={}; timings={}
        def run(name,fn):
            if progress: progress('  '+name)
            if self.backend=='cuda': xp.cuda.get_current_stream().synchronize()
            start=time.perf_counter(); value=fn()
            if self.backend=='cuda': xp.cuda.get_current_stream().synchronize()
            timings[name]=time.perf_counter()-start
            return value
        def ig():
            if not len(self.ri) or len(si)<2:
                return m.result(np.nan,reason='Insufficient per-pixel ISIs',reference_intervals=len(self.ri),modified_intervals=len(si))
            z=xp.maximum(si,cfg.isi_min); mu=float(z.mean())
            inv=float(xp.mean((z-mu)**2/(mu*mu*z)))
            if inv<=np.finfo(float).eps/mu:
                return m.result(np.nan,reason='Degenerate nearly constant ISIs: no finite IG shape MLE')
            lam=1/inv; q=xp.maximum(self.ri,cfg.isi_min)
            value=-float(xp.mean(.5*(np.log(lam)-np.log(2*np.pi)-3*xp.log(q))-lam*(q-mu)**2/(2*mu*mu*q)))
            return m.result(value,mu_s=mu,lambda_s=lam,reference_clamped=int((self.ri<cfg.isi_min).sum()),
                modified_clamped=int((si<cfg.isi_min).sum()),time_unit='seconds')
        def same():
            if not len(e): return m.result(np.nan,reason='Empty stream')
            q=(a['t']-e.start)/cfg.same_ts_resolution
            if float(q.max())>=np.iinfo(np.int64).max: raise ValueError('Timestamp quantization exceeds int64')
            w=xp.floor((a['t']-e.start)/cfg.window).astype(xp.int64)
            keys=xp.column_stack((w,xp.rint(q).astype(xp.int64)))
            _,inv,counts=xp.unique(keys,axis=0,return_inverse=True,return_counts=True)
            return m.result(float(xp.mean(counts[inv]>1)))
        def fano():
            fr,mr=self.rf; fs,ms=m.fano_values(e,cfg)
            shared=(mr>0)&(ms>0); union=(mr>0)|(ms>0)
            value=np.mean(np.abs(np.log((fs[shared]+cfg.fano_eps)/(fr[shared]+cfg.fano_eps)))) if shared.any() else np.nan
            missing=int((union&~shared).sum())
            return m.result(value,'partial' if missing and shared.any() else 'ok',
                'Only spatial/window cells active in both streams are scored' if missing else '',
                shared_active_cells=int(shared.sum()),unmatched_active_cells=missing,
                coverage=float(shared.sum()/union.sum()) if union.any() else 0.)
        def voxel():
            ir,vr=self.rv; im,vm=self.voxels(a); union=xp.union1d(ir,im)
            delta=xp.zeros(len(union),dtype=xp.float64)
            delta[xp.searchsorted(union,ir)]-=vr
            delta[xp.searchsorted(union,im)]+=vm
            cells=(len(self.edges)-1)*r.width*r.height; mass=float(xp.abs(vr).sum())
            absolute=float(xp.abs(delta).sum())
            return {'Voxel MAE':m.result(absolute/cells),
                'Voxel RMSE':m.result(np.sqrt(float(xp.dot(delta,delta))/cells)),
                'Voxel Relative L1':m.result(absolute/mass if mass else np.nan,reason='' if mass else 'Zero reference signed voxel mass')}
        out['IG-NLL']=run('IG-NLL',ig)
        out['Chamfer']=run('Chamfer (CPU)',lambda:self.chamfer(e))
        out['Same-ts']=run('Same-ts',same)
        out['Fano']=run('Fano (CPU)',fano)
        out['ISI-spike']=m.result(float(xp.mean(si<cfg.isi_spike_eps)) if len(si) else np.nan,
            reason='' if len(si) else 'No per-pixel intervals',intervals=len(si))
        out['Count Ratio']=m.result(len(e)/len(r) if len(r) else np.nan,reason='' if len(r) else 'Empty reference')
        out.update(run('Voxel metrics',voxel))
        return {k:out[k] for k in m.NAMES},timings
