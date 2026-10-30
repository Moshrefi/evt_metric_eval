"""One change at a time, reproducible RNG, explicit reflected-noise boundaries."""
import numpy as np


def _reflect(a, lo, hi):
    if hi <= lo:
        return np.full_like(a, lo)
    span = hi-lo
    z = np.mod(a-lo, 2*span)
    return lo+np.where(z<=span, z, 2*span-z)


def perturb(e, kind, severity, seed=0):
    rng = np.random.default_rng(seed)
    n = len(e)
    if not np.isfinite(severity) or severity < 0:
        raise ValueError('Severity must be finite and nonnegative')
    fractions = {'delete', 'delete_on', 'delete_off', 'duplicate', 'polarity', 'polarity_structured',
                 'inject', 'inject_hotspot', 'relocate', 'spatial_concentration', 'temporal_clustering', 'redistribute_time'}
    if kind in fractions and severity > 1:
        raise ValueError('Fraction severity must be in [0,1]')
    info = dict(kind=kind, severity=float(severity), seed=int(seed), input_events=n,
                boundary_policy='no clipping or silent dropping', reflected_events=0)
    if severity == 0:
        return e.replace(), {**info, 'output_events': n}
    t, x, y, p = [getattr(e, k).copy() for k in 'txyp']
    order = rng.permutation(n)
    k = int(np.floor(n*severity)) if kind in fractions else 0
    if kind.startswith('delete'):
        eligible = np.arange(n) if kind=='delete' else np.flatnonzero(p==(1 if kind=='delete_on' else -1))
        chosen = rng.permutation(eligible)[:int(np.floor(len(eligible)*severity))]
        mask = np.ones(n, dtype=bool); mask[chosen] = False
        t, x, y, p = [a[mask] for a in (t,x,y,p)]
        info['eligible_events'] = len(eligible)
    elif kind=='duplicate':
        t, x, y, p = [np.concatenate((a,a[order[:k]])) for a in (t,x,y,p)]
    elif kind in ('polarity', 'polarity_structured'):
        chosen = order[:k] if kind=='polarity' else np.lexsort((t,y,x))[:k]
        p[chosen] *= -1
    elif kind in ('inject', 'inject_hotspot'):
        nt = rng.uniform(e.start, e.stop, k)
        nx = rng.integers(0, e.width, k); ny = rng.integers(0, e.height, k)
        if kind=='inject_hotspot':
            nx[:] = e.width//2; ny[:] = e.height//2
        npol = rng.choice([-1,1], k)
        t,x,y,p = [np.concatenate((a,b)) for a,b in zip((t,x,y,p),(nt,nx,ny,npol))]
        info['injection'] = 'uniform time, balanced Bernoulli polarity; uniform pixels or fixed center hotspot'
    elif kind=='relocate':
        chosen = order[:k]
        x[chosen] = rng.integers(0,e.width,k); y[chosen] = rng.integers(0,e.height,k)
    elif kind in ('shift_x', 'shift_y'):
        if severity != int(severity):
            raise ValueError('Pixel shifts must be integer')
        if kind=='shift_x': x += int(severity)
        else: y += int(severity)
        # EventStream validation rejects out-of-domain shifts. Use a declared guard ROI.
    elif kind in ('spatial_uniform', 'spatial_gaussian'):
        noise = rng.uniform(-severity,severity,(2,n)) if kind=='spatial_uniform' else rng.normal(0,severity,(2,n))
        xx,yy = x+np.rint(noise[0]).astype(int), y+np.rint(noise[1]).astype(int)
        info['reflected_events'] = int(((xx<0)|(xx>=e.width)|(yy<0)|(yy>=e.height)).sum())
        x = _reflect(xx,0,e.width-1).astype(int); y = _reflect(yy,0,e.height-1).astype(int)
        info['boundary_policy'] = 'integer pixel reflection; nominal Gaussian/uniform before reflection'
    elif kind in ('shift_time', 'shift_time_negative'):
        t += severity*1e-3*(1 if kind=='shift_time' else -1)
    elif kind in ('temporal_uniform', 'temporal_gaussian'):
        noise = rng.uniform(-severity,severity,n) if kind=='temporal_uniform' else rng.normal(0,severity,n)
        proposed = t+noise*1e-3
        info['reflected_events'] = int(((proposed<e.start)|(proposed>=e.stop)).sum())
        t = _reflect(proposed,e.start,np.nextafter(e.stop,e.start))
        info['boundary_policy'] = 'reflection in fixed time domain; nominal noise before reflection'
    elif kind=='redistribute_time':
        t[order[:k]] = rng.uniform(e.start,e.stop,k)
    elif kind=='spatial_concentration':
        x = np.rint((1-severity)*x+severity*(e.width-1)/2).astype(int)
        y = np.rint((1-severity)*y+severity*(e.height-1)/2).astype(int)
    elif kind=='temporal_clustering':
        centers = e.start+(np.array([.25,.75]))*(e.stop-e.start)
        nearest = centers[(t >= (e.start+e.stop)/2).astype(int)]
        t = (1-severity)*t+severity*nearest
    else:
        raise ValueError(f'Unknown perturbation: {kind}')
    out = e.replace(t=t,x=x,y=y,p=p)
    info.update(output_events=len(out), event_count_delta=len(out)-n, output_sha256=out.digest())
    if len(out)==n and kind not in ('duplicate','inject','inject_hotspot'):
        # Arrays above still correspond to source events; canonical sorting happens only in out.
        info.update(changed_events=int(((t!=e.t)|(x!=e.x)|(y!=e.y)|(p!=e.p)).sum()),
                    mean_abs_time_change_ms=float(np.mean(np.abs(t-e.t))*1e3) if n else 0.,
                    mean_spatial_displacement_px=float(np.mean(np.hypot(x-e.x,y-e.y))) if n else 0.,
                    flipped_polarities=int((p!=e.p).sum()))
    return out, info


def default_sweep():
    """Severity families are intentionally not equated across dimensions."""
    deletion = [0,.1,.2,.4,.6,.8]
    addition = [0,.1,.2,.4,.6]
    spatial = [0,1,2,4,8]
    temporal = [0,.1,.5,1,2,5]
    return {**{k: deletion for k in ('delete','delete_on','delete_off')},
            **{k: addition for k in ('duplicate','inject','inject_hotspot')},
            **{k: spatial for k in ('shift_x','shift_y','spatial_uniform','spatial_gaussian')},
            **{k: temporal for k in ('shift_time','shift_time_negative','temporal_uniform','temporal_gaussian')},
            **{k: [0,.1,.25,.5,.75,1.] for k in ('polarity','polarity_structured','relocate',
                'redistribute_time','spatial_concentration','temporal_clustering')}}


def guard_reference(e, spatial_margin=8, temporal_margin_s=.006):
    """Declared reference selection to isolate pure shifts from boundary loss.

    Original loaded data remain immutable. Sensor resolution and observation
    domain stay fixed; the selected reference has silent guard regions.
    """
    if min(e.width,e.height)<=2*spatial_margin or e.stop-e.start<=2*temporal_margin_s:
        raise ValueError('Guard margins exceed the observation domain')
    m = ((e.x>=spatial_margin)&(e.x<e.width-spatial_margin)&
         (e.y>=spatial_margin)&(e.y<e.height-spatial_margin)&
         (e.t>=e.start+temporal_margin_s)&(e.t<e.stop-temporal_margin_s))
    out = e.select(m)
    if not len(out):
        raise ValueError('No events remain after explicit guard selection; change interval/margins')
    return out.replace(metadata={**e.metadata, 'guard_selection': {'spatial_margin_px': spatial_margin,
        'temporal_margin_s': temporal_margin_s, 'excluded_events': int((~m).sum()), 'loaded_sha256': e.digest()}})
