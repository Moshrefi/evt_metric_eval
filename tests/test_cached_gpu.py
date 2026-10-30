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
