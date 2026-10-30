import numpy as np
import pytest
from scipy.stats import invgauss
from scipy.spatial.distance import cdist
from event_metrics.events import EventStream, synthetic_stream
from event_metrics.metrics import (MetricConfig,evaluate,ig_fit,ig_logpdf,voxel_counts,per_pixel_isi,
                                   same_ts,isi_spike,fano_values,chamfer)
from event_metrics.counterexamples import audit_cases,boundary_case,base,small_change_diagnostics
from event_metrics.perturbations import perturb,default_sweep,guard_reference


def test_ig_density_matches_independent_scipy():
    x=np.array([.001,.002,.005,.007]); mu,lam=ig_fit(x)
    np.testing.assert_allclose(ig_logpdf(x,mu,lam),invgauss.logpdf(x,mu/lam,scale=lam),rtol=1e-12)
    assert mu==pytest.approx(x.mean())
    assert lam==pytest.approx(1/(np.mean(1/x)-1/mu))


def test_identity_and_storage_permutation():
    e=synthetic_stream(); cfg=MetricConfig()
    other=e.select(np.random.default_rng(4).permutation(len(e)))
    assert other.digest()==e.digest()
    a=evaluate(e,e,cfg); b=evaluate(e,other,cfg)
    for key in a:
        assert b[key]['value']==pytest.approx(a[key]['value'])
    for key in ('Chamfer','Fano','Voxel MAE','Voxel RMSE','Voxel Relative L1'):
        assert a[key]['value']==pytest.approx(0)
    assert a['Count Ratio']['value']==1


def test_isi_is_per_pixel_and_polarity_pooled():
    e=EventStream([.01,.02,.03,.04],[0,1,0,1],[0]*4,[1,1,-1,-1],2,2,0,.1)
    np.testing.assert_allclose(per_pixel_isi(e),[.02,.02])


def test_timestamp_fraction_not_cross_stream_matching():
    e=EventStream([.01,.01,.02],[0,1,0],[0]*3,[1,-1,1],2,2,0,.1)
    assert same_ts(e,MetricConfig())['value']==pytest.approx(2/3)


def test_chamfer_matches_brute_force():
    e=base(); m=e.replace(x=e.x+2,t=e.t+.001)
    cfg=MetricConfig(chamfer_voxel=0,normalize_chamfer=False)
    a=np.column_stack([e.x,e.y,cfg.v*e.t]); b=np.column_stack([m.x,m.y,cfg.v*m.t])
    d=cdist(a,b)
    assert chamfer(e,m,cfg)['value']==pytest.approx(.5*(d.min(0).mean()+d.min(1).mean()))


def test_chamfer_missing_groups_are_not_complete_scores():
    e=base(); m=e.replace(p=-e.p)
    s=chamfer(e,m,MetricConfig())
    assert s['status']=='undefined' and np.isnan(s['value'])
    assert s['details']['unmatched_groups']==2


def test_signed_voxel_hand_calculation_and_half_open_edges():
    e=EventStream([.01,.02,.04],[0,0,1],[0,0,0],[1,-1,1],2,1,0,.1)
    m=e.select([0,1])
    edges=np.array([0,.02,.1])
    ids,counts=voxel_counts(e,edges)
    np.testing.assert_array_equal(ids,[0,2,3]); np.testing.assert_array_equal(counts,[1,-1,1])
    s=evaluate(e,m,edges=edges)
    assert s['Voxel MAE']['value']==pytest.approx(.25)
    assert s['Voxel RMSE']['value']==pytest.approx(.5)
    assert s['Voxel Relative L1']['value']==pytest.approx(1/3)


def test_fano_hand_calculation():
    e=EventStream([.001,.002,.003,.026],[0]*4,[0]*4,[1]*4,8,8,0,.05)
    f,mean=fano_values(e,MetricConfig(fano_bin_t=.025))
    # Two sub-bins: [3,1], population variance 1, mean 2.
    np.testing.assert_allclose(f,[.5]); np.testing.assert_allclose(mean,[2])


def test_empty_and_degenerate_are_explicit():
    e=base(); z=e.select([])
    s=evaluate(e,z)
    assert s['Count Ratio']['value']==0
    for key in ('IG-NLL','Chamfer','Same-ts','ISI-spike'):
        assert s[key]['status']=='undefined'
    assert evaluate(z,z)['Voxel Relative L1']['status']=='undefined'
    with pytest.raises(ValueError,match='Degenerate'): ig_fit(np.ones(3))


def test_all_18_constructed_failures_are_demonstrated():
    rows=audit_cases()
    assert len(rows)==18
    assert all(r['demonstrated'] for r in rows),[r for r in rows if not r['demonstrated']]


def test_voxel_boundary_discontinuity():
    crossing,interior=boundary_case()
    assert evaluate(*crossing)['Voxel MAE']['value']>0
    assert evaluate(*interior)['Voxel MAE']['value']==0


def test_small_change_diagnostics_cover_every_metric():
    from event_metrics.metrics import NAMES
    rows=small_change_diagnostics()
    assert {r['metric'] for r in rows}==set(NAMES)
    isi=next(r for r in rows if r['metric']=='ISI-spike')
    assert isi['reference_score']==1 and isi['modified_score']==0
    same=next(r for r in rows if r['metric']=='Same-ts')
    assert same['reference_score']==1 and same['modified_score']==0


@pytest.mark.parametrize('kind',list(default_sweep()))
def test_perturbation_zero_reproducibility_and_reference_immutability(kind):
    e=guard_reference(synthetic_stream(),8,.006); before=e.digest()
    zero,_=perturb(e,kind,0,12)
    assert zero.digest()==before
    severity=default_sweep()[kind][-1]
    a,_=perturb(e,kind,severity,12); b,_=perturb(e,kind,severity,12)
    assert a.digest()==b.digest() and e.digest()==before
    if kind.startswith('delete'): assert len(a)<=len(e)
    elif kind in ('inject','inject_hotspot','duplicate'): assert len(a)>=len(e)
    else: assert len(a)==len(e)
    if kind in ('polarity','polarity_structured'):
        np.testing.assert_array_equal(a.t,e.t)
        np.testing.assert_array_equal(a.x,e.x)


def test_duplicates_and_input_validation():
    e=base(); d,_=perturb(e,'duplicate',1)
    assert len(d)==2*len(e)
    assert len(per_pixel_isi(d))==2*len(e)-1
    with pytest.raises(ValueError): e.replace(x=e.x+.1)
    with pytest.raises(ValueError): e.replace(t=np.full(len(e),e.stop))
    with pytest.raises(ValueError): e.replace(p=np.zeros(len(e)))
    with pytest.raises(ValueError): e.replace(t=np.full(len(e),np.nan))
    with pytest.raises(ValueError): perturb(e,'shift_x',100)
    with pytest.raises(ValueError): MetricConfig(window=.05,fano_bin_t=.003)
