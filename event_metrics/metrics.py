"""Audited definitions on a fixed observation domain; see docs/metrics.md.

Equations follow Moshrefi/TIDES 3db4de5 and evtween 2a9bab9. This is a
documented independent implementation, not a claim of paper reproduction.
Missing match groups are exposed, never interpreted as perfect agreement.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree

NAMES = ('IG-NLL', 'Chamfer', 'Same-ts', 'Fano', 'ISI-spike', 'Count Ratio',
         'Voxel MAE', 'Voxel RMSE', 'Voxel Relative L1')


@dataclass(frozen=True)
class MetricConfig:
    window: float = 0.05
    v: float = 1000.
    chamfer_voxel: float = 1.
    normalize_chamfer: bool = True
    max_points: int = 200_000
    isi_min: float = 1e-6
    isi_spike_eps: float = 1e-5
    same_ts_resolution: float = 1e-9
    fano_bin_px: int = 8
    fano_bin_t: float = 0.002
    fano_eps: float = 1e-3
    voxel_bins: int = 10
    seed: int = 0
    max_fano_cells: int = 20_000_000

    def __post_init__(self):
        for k in ('window', 'v', 'isi_min', 'isi_spike_eps', 'same_ts_resolution', 'fano_bin_t', 'fano_eps'):
            if not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f'{k} must be finite and positive')
        for k in ('max_points', 'fano_bin_px', 'voxel_bins', 'max_fano_cells'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f'{k} must be a positive integer')
        if not np.isfinite(self.chamfer_voxel) or self.chamfer_voxel < 0:
            raise ValueError('chamfer_voxel must be nonnegative')
        ratio = self.window/self.fano_bin_t
        if ratio < 2 or not np.isclose(ratio, round(ratio)):
            raise ValueError('window must contain an integer number (>=2) of Fano sub-bins')


def result(value, status='ok', reason='', **details):
    value = float(value)
    if not np.isfinite(value) and status == 'ok':
        status, reason = 'undefined', reason or 'Nonfinite score'
    return dict(value=value, status=status, reason=reason, details=details)


def per_pixel_isi(e):
    """Pool per-pixel intervals, mixing ON/OFF exactly as the source definition."""
    pix = e.y*e.width+e.x
    o = np.lexsort((e.t, pix))
    return np.diff(e.t[o])[np.diff(pix[o]) == 0]


def ig_fit(positive_isi):
    z = np.asarray(positive_isi, dtype=float)
    if len(z)<2 or np.any(z<=0) or not np.isfinite(z).all():
        raise ValueError('IG fit needs >=2 finite positive intervals')
    mu = float(z.mean())
    # Algebraically identical to mean(1/x)-1/mu, with less cancellation.
    inv_lam = float(np.mean((z-mu)**2/(mu*mu*z)))
    if inv_lam <= np.finfo(float).eps/mu:
        raise ValueError('Degenerate nearly constant ISIs: no finite IG shape MLE')
    return mu, 1/inv_lam


def ig_logpdf(x, mu, lam):
    x = np.asarray(x, dtype=float)
    return .5*(np.log(lam)-np.log(2*np.pi)-3*np.log(x))-lam*(x-mu)**2/(2*mu*mu*x)


def ig_nll(reference, modified, cfg):
    ri, si = per_pixel_isi(reference), per_pixel_isi(modified)
    if not len(ri) or len(si)<2:
        return result(np.nan, reason='Insufficient per-pixel ISIs', reference_intervals=len(ri), modified_intervals=len(si))
    try:
        mu, lam = ig_fit(np.maximum(si, cfg.isi_min))
    except ValueError as exc:
        return result(np.nan, reason=str(exc))
    return result(-np.mean(ig_logpdf(np.maximum(ri, cfg.isi_min), mu, lam)),
                  mu_s=mu, lambda_s=lam, reference_clamped=int((ri<cfg.isi_min).sum()),
                  modified_clamped=int((si<cfg.isi_min).sum()), time_unit='seconds')


def _window_ids(e, cfg):
    return np.floor((e.t-e.start)/cfg.window).astype(np.int64)


def same_ts(e, cfg):
    if not len(e):
        return result(np.nan, reason='Empty stream')
    qfloat = (e.t-e.start)/cfg.same_ts_resolution
    if qfloat.max() >= np.iinfo(np.int64).max:
        raise ValueError('Timestamp quantization exceeds int64')
    keys = np.column_stack((_window_ids(e, cfg), np.rint(qfloat).astype(np.int64)))
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return result(np.mean(counts[inv]>1))


def isi_spike(e, cfg):
    isi = per_pixel_isi(e)
    return result(np.mean(isi < cfg.isi_spike_eps) if len(isi) else np.nan,
                  reason='' if len(isi) else 'No per-pixel intervals', intervals=len(isi))


def _embedding(e, mask, w0, cfg):
    a = np.column_stack((e.x[mask], e.y[mask], cfg.v*(e.t[mask]-w0))).astype(float)
    # Canonical sorting makes density representatives independent of storage order.
    if len(a):
        a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    if cfg.chamfer_voxel and len(a):
        _, keep = np.unique(np.floor(a/cfg.chamfer_voxel).astype(np.int64), axis=0, return_index=True)
        a = a[keep]
    before = len(a)
    if before > cfg.max_points:
        keep = np.random.default_rng(cfg.seed).choice(before, cfg.max_points, replace=False)
        a = a[keep]
    return a, before


def chamfer(reference, modified, cfg):
    rw, sw = _window_ids(reference, cfg), _window_ids(modified, cfg)
    scores, unmatched, groups, dropped = [], 0, 0, 0
    for w in np.union1d(rw, sw):
        for pol in (-1, 1):
            a, na = _embedding(reference, (rw==w)&(reference.p==pol), reference.start+w*cfg.window, cfg)
            b, nb = _embedding(modified, (sw==w)&(modified.p==pol), reference.start+w*cfg.window, cfg)
            dropped += na-len(a)+nb-len(b)
            if not len(a) and not len(b):
                continue
            groups += 1
            if not len(a) or not len(b):
                unmatched += 1
                continue  # source behavior, explicitly diagnosed below
            da = cKDTree(b).query(a, workers=1)[0]
            db = cKDTree(a).query(b, workers=1)[0]
            scores.append(.5*(da.mean()+db.mean()))
    score = np.mean(scores) if scores else np.nan
    if cfg.normalize_chamfer:
        score /= np.hypot(reference.width, reference.height)
    status = 'partial' if unmatched else ('approximate' if dropped else 'ok')
    if not scores:
        status = 'undefined'
    return result(score, status, 'Unmatched window/polarity groups excluded' if unmatched else '',
                  unmatched_groups=unmatched, occupied_groups=groups, compared_groups=len(scores),
                  coverage=len(scores)/groups if groups else 0., sampled_away_points=dropped)


def fano_values(e, cfg):
    nx = int(np.ceil(e.width/cfg.fano_bin_px)); ny = int(np.ceil(e.height/cfg.fano_bin_px))
    nw = max(1, int(np.ceil((e.stop-e.start)/cfg.window-1e-12)))
    ns = int(round(cfg.window/cfg.fano_bin_t))
    if nw*nx*ny*ns > cfg.max_fano_cells:
        raise ValueError('Fano array exceeds memory guard; shorten interval or explicitly change bins')
    counts = np.zeros((nw, ny*nx, ns), dtype=np.int64)
    w = _window_ids(e, cfg)
    # Small floating-point error at a shared boundary must not create an invalid index.
    local = ((e.t-e.start)-w*cfg.window)/cfg.fano_bin_t
    sub = np.minimum(np.floor(local).astype(int), ns-1)
    spatial = (e.y//cfg.fano_bin_px)*nx+e.x//cfg.fano_bin_px
    np.add.at(counts, (w, spatial, sub), 1)
    mean = counts.mean(-1); var = counts.var(-1, ddof=0)
    fano = np.divide(var, mean, out=np.zeros_like(mean), where=mean>0)
    return fano.ravel(), mean.ravel()


def fano_error(reference, modified, cfg):
    fr, mr = fano_values(reference, cfg); fs, ms = fano_values(modified, cfg)
    shared = (mr>0)&(ms>0); union = (mr>0)|(ms>0)
    value = np.mean(np.abs(np.log((fs[shared]+cfg.fano_eps)/(fr[shared]+cfg.fano_eps)))) if shared.any() else np.nan
    missing = int((union & ~shared).sum())
    return result(value, 'partial' if missing and shared.any() else 'ok',
                  'Only spatial/window cells active in both streams are scored' if missing else '',
                  shared_active_cells=int(shared.sum()), unmatched_active_cells=missing,
                  coverage=float(shared.sum()/union.sum()) if union.any() else 0.)


def voxel_counts(e, edges):
    """Sparse signed counts; implicit zeros still enter MAE/RMSE denominator."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
        raise ValueError('Invalid voxel edges')
    if edges[0]!=e.start or edges[-1]!=e.stop:
        raise ValueError('Voxel edges must exactly cover the shared observation interval')
    b = np.searchsorted(edges, e.t, side='right')-1
    ids = (b*e.height+e.y)*e.width+e.x
    unique, inv = np.unique(ids, return_inverse=True)
    counts = np.bincount(inv, weights=e.p, minlength=len(unique)).astype(np.int64)
    return unique[counts!=0], counts[counts!=0]


def voxel_errors(reference, modified, edges):
    ir, vr = voxel_counts(reference, edges); im, vm = voxel_counts(modified, edges)
    union = np.union1d(ir, im)
    delta = np.zeros(len(union), dtype=float)
    delta[np.searchsorted(union, ir)] -= vr
    delta[np.searchsorted(union, im)] += vm
    cells = (len(edges)-1)*reference.width*reference.height
    mass = np.abs(vr).sum()
    return {'Voxel MAE': result(np.abs(delta).sum()/cells),
            'Voxel RMSE': result(np.sqrt(np.dot(delta, delta)/cells)),
            'Voxel Relative L1': result(np.abs(delta).sum()/mass if mass else np.nan,
                                       reason='' if mass else 'Zero reference signed voxel mass')}


def evaluate(reference, modified, cfg=None, edges=None):
    cfg = cfg or MetricConfig()
    domain = lambda e: (e.width, e.height, e.start, e.stop)
    if domain(reference)!=domain(modified):
        raise ValueError('Both streams require identical resolution and observation interval')
    edges = np.linspace(reference.start, reference.stop, cfg.voxel_bins+1) if edges is None else np.asarray(edges)
    scores = {'IG-NLL': ig_nll(reference, modified, cfg), 'Chamfer': chamfer(reference, modified, cfg),
              'Same-ts': same_ts(modified, cfg), 'Fano': fano_error(reference, modified, cfg),
              'ISI-spike': isi_spike(modified, cfg),
              'Count Ratio': result(len(modified)/len(reference) if len(reference) else np.nan,
                                    reason='' if len(reference) else 'Empty reference')}
    scores.update(voxel_errors(reference, modified, edges))
    return {k: scores[k] for k in NAMES}


def discrepancy(name, value, baseline):
    """Within-metric analysis transform; retain raw values in every result table."""
    if name == 'Count Ratio':
        return abs(value-1.)
    if name in ('Same-ts', 'ISI-spike'):
        return abs(value-baseline)
    if name == 'IG-NLL':
        return value-baseline  # may be negative; never absolute value or clamped
    return value
