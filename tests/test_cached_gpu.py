import numpy as np
import pandas as pd
import pytest
from event_metrics.events import synthetic_stream
from event_metrics.metrics import MetricConfig,evaluate
from event_metrics.perturbations import default_sweep,perturb
from event_metrics.accelerated import Evaluator
from event_metrics.sample_cache import prepare_samples,evaluate_samples,load_sample

def compare(a,b):
    for name in a:
        np.testing.assert_allclose(a[name]['value'],b[name]['value'],rtol=1e-8,atol=1e-10,equal_nan=True,err_msg=name)
        assert a[name]['status']==b[name]['status'],name

@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_equivalence(backend):
    if backend=='cuda':
        cp=pytest.importorskip('cupy')
        if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA unavailable')
    r=synthetic_stream(); cfg=MetricConfig(); edges=np.linspace(r.start,r.stop,11)
    engine=Evaluator(r,cfg,edges,backend)
    for family,levels in default_sweep().items():
        e,_=perturb(r,family,levels[-1],2)
        compare(evaluate(r,e,cfg,edges),engine.evaluate(e)[0])
    e=r.select(np.zeros(len(r),dtype=bool))
    compare(evaluate(r,e,cfg,edges),engine.evaluate(e)[0])

def test_cache_resume(tmp_path):
    r=synthetic_stream(); cfg=MetricConfig(); edges=np.linspace(r.start,r.stop,11)
    sweep={'delete':[0,.2],'shift_x':[0,1]}
    folder,index=prepare_samples(r,tmp_path,sweep,[0,1],None)
    assert len(index)==8 and index.sample_id.nunique()==4
    row=index.iloc[1]
    assert len(load_sample(folder/row.file,row.sha256))==row.events
    stamps={p.name:p.stat().st_mtime_ns for p in folder.glob('*.npz')}
    prepare_samples(r,tmp_path,sweep,[0,1],None)
    assert stamps=={p.name:p.stat().st_mtime_ns for p in folder.glob('*.npz')}
    a,_,run=evaluate_samples(folder,cfg,edges,'cpu',1,None)
    stamps={p.name:p.stat().st_mtime_ns for p in run.glob('*.json')}
    b,_,run2=evaluate_samples(folder,cfg,edges,'cpu',1,None)
    pd.testing.assert_frame_equal(a,b)
    assert stamps=={p.name:p.stat().st_mtime_ns for p in run2.glob('*.json')}
    with pytest.raises(ValueError):
        load_sample(folder/row.file,'wrong-checksum')

def test_isi_uses_array_keys_and_preserves_order():
    # Enforce the CuPy keys contract even on CPU-only CI.
    class StrictKeys:
        float64 = np.float64
        stack = staticmethod(np.stack)
        diff = staticmethod(np.diff)
        empty = staticmethod(np.empty)
        @staticmethod
        def lexsort(keys):
            assert isinstance(keys, np.ndarray) and keys.ndim == 2
            return np.lexsort(keys)
    from types import SimpleNamespace
    engine=object.__new__(Evaluator)
    engine.xp=StrictKeys()
    engine.reference=SimpleNamespace(width=10)
    arrays={'t':np.array([.03,.02,.01,.02,.01]),
            'x':np.array([1,0,1,0,0]),'y':np.zeros(5,dtype=int)}
    np.testing.assert_allclose(engine.isi(arrays),[.01,0.,.02])
    for n in (0,1):
        assert engine.isi({k:v[:n] for k,v in arrays.items()}).size == 0

@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_same_ts_linear_matches_original(backend):
    from dataclasses import replace
    from event_metrics.events import EventStream
    from event_metrics.metrics import same_ts
    from event_metrics.accelerated import same_ts_sorted
    xp=np
    if backend=='cuda':
        xp=pytest.importorskip('cupy')
        if xp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(42)
    examples=[[],[.01],[.01,.01],[.01,.01,.01,.02,.03,.03],
              [.0499,.05,.0501],rng.integers(0,1000,10000)*.0001]
    for resolution in (1e-9,.001,.1):
        cfg=replace(MetricConfig(),same_ts_resolution=resolution)
        for times in examples:
            times=np.asarray(times,dtype=float)
            n=len(times)
            e=EventStream(times,np.zeros(n),np.zeros(n),np.ones(n),2,2,0.,.1)
            want=same_ts(e,cfg)
            got=same_ts_sorted(xp.asarray(e.t),e.start,cfg,xp)
            np.testing.assert_allclose(got['value'],want['value'],equal_nan=True,rtol=0,atol=1e-15)
            assert got['status']==want['status']

def test_same_ts_large_without_sort_or_unique():
    from event_metrics.accelerated import same_ts_sorted
    class LinearOnly:
        int64=np.int64
        rint=staticmethod(np.rint)
        floor=staticmethod(np.floor)
        zeros=staticmethod(np.zeros)
    # 1 million events: each tick repeats twice; require only linear primitives.
    t=np.repeat(np.arange(500000,dtype=float)*1e-6,2)
    assert same_ts_sorted(t,0.,MetricConfig(),LinearOnly())['value']==1.
