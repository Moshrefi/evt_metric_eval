"""Targeted synthetic witnesses, verified numerically rather than asserted as results."""
from dataclasses import dataclass
import numpy as np
from .events import EventStream
from .metrics import MetricConfig, evaluate


@dataclass
class Case:
    name: str
    metric: str
    reference: EventStream
    modified: EventStream
    explanation: str
    expected: str = 'blind_spot'


def base():
    # Irregular positive ISIs at one pixel; all events safely inside first window.
    return EventStream([.011,.013,.016,.020,.026], [3]*5,[3]*5,[1]*5,16,16,0.,.1)


def cases():
    e = base()
    shift = e.replace(t=e.t+.002)
    mirror = e.replace(x=e.width-1-e.x)
    flip = e.replace(p=-e.p)
    out = [Case('count_spatial_relocation','Count Ratio',e,mirror,'Count ignores event locations.'),
           Case('count_absolute_time_shift','Count Ratio',e,shift,'Count ignores absolute event times.'),
           Case('same_ts_spatial_relocation','Same-ts',e,mirror,'Timestamp multiplicity ignores location.'),
           Case('same_ts_polarity_inversion','Same-ts',e,flip,'Timestamp multiplicity ignores polarity.'),
           Case('isi_absolute_time_shift','ISI-spike',e,shift,'Per-pixel intervals are unchanged by a constant shift.'),
           Case('isi_polarity_inversion','ISI-spike',e,flip,'The inspected ISI definition pools both polarities.'),
           Case('ig_absolute_time_shift','IG-NLL',e,shift,'Both fitted ISIs and scored ISIs are invariant to this shift.'),
           Case('ig_pixel_permutation','IG-NLL',e,mirror,'Pooled per-pixel ISIs discard pixel identity.'),
           Case('fano_polarity_inversion','Fano',e,flip,'Fano counts both polarities without their signs.'),
           Case('fano_within_cell_relocation','Fano',e,e.replace(x=e.x+1),'Locations within the same spatial cell have identical count series.')]
    dup = e.replace(**{k: np.tile(getattr(e,k),2) for k in 'txyp'})
    out.append(Case('chamfer_exact_duplication','Chamfer',e,dup,'Identical support has zero nearest-neighbor distances despite doubled mass.'))
    two = e.replace(t=np.r_[e.t,e.t+.001],x=np.r_[e.x,e.x+7],y=np.r_[e.y,e.y],p=np.r_[e.p,-e.p])
    out.append(Case('chamfer_missing_polarity','Chamfer',two,two.select(two.p>0),
                    'Source-style exclusion omits the missing polarity. The zero conditional score is flagged partial.', 'partial_coverage'))
    v = EventStream([.011,.012,.014],[3]*3,[3]*3,[1]*3,16,16,0.,.1)
    within = v.replace(t=v.t+.002)  # all in [0.01,0.02)
    cancelled = v.replace(t=np.r_[v.t,.015,.017],x=np.r_[v.x,7,7],y=np.r_[v.y,7,7],p=np.r_[v.p,1,-1])
    for metric in ('Voxel MAE','Voxel RMSE','Voxel Relative L1'):
        out.extend([Case(metric.lower().replace(' ','_')+'_within_bin',metric,v,within,
                         'Hard temporal binning discards within-bin event times.'),
                    Case(metric.lower().replace(' ','_')+'_cancellation',metric,v,cancelled,
                         'Added ON/OFF events cancel in the signed voxel although event count increases.')])
    return out


def audit_cases(cfg=None):
    """Counterexamples use a declared fixed protocol; do not silently reuse user bin changes."""
    cfg = cfg or MetricConfig(voxel_bins=10)
    rows = []
    for c in cases():
        baseline = evaluate(c.reference,c.reference,cfg)[c.metric]
        score = evaluate(c.reference,c.modified,cfg)[c.metric]
        equal = np.isclose(score['value'],baseline['value'],rtol=1e-9,atol=1e-12)
        changed = c.reference.digest()!=c.modified.digest()
        demonstrated = bool(changed and equal and score['status']==('partial' if c.expected=='partial_coverage' else 'ok'))
        rows.append(dict(case=c.name,metric=c.metric,reference_score=baseline['value'],modified_score=score['value'],
                         status=score['status'],expected=c.expected,demonstrated=demonstrated,
                         explanation=c.explanation,reference_events=len(c.reference),modified_events=len(c.modified)))
    return rows


def boundary_case():
    """Same 2 us shift crosses a voxel edge for one event, not the other."""
    e = EventStream([.019999],[3],[3],[1],16,16,0.,.1)
    interior = e.replace(t=np.array([.015]))
    return (e,e.replace(t=e.t+2e-6)), (interior,interior.replace(t=interior.t+2e-6))
