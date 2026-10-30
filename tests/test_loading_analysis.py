from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from event_metrics.loading import load_stream,choose_sequence
from event_metrics.events import synthetic_stream
from event_metrics.benchmark import run_benchmark
from event_metrics.analysis import summarize_responses


def test_all_file_formats(tmp_path):
    e=synthetic_stream()
    arrays={k:getattr(e,k) for k in 'txyp'}
    np.savez(tmp_path/'a.npz',**arrays,width=e.width,height=e.height)
    np.save(tmp_path/'a.npy',np.column_stack([arrays[k] for k in 'txyp']))
    np.savetxt(tmp_path/'a.csv',np.column_stack([arrays[k] for k in 'txyp']),delimiter=',',header='t,x,y,p',comments='')
    import h5py
    with h5py.File(tmp_path/'a.h5','w') as f:
        for k,v in arrays.items(): f[k]=v
    for name in ['a.npz','a.npy','a.csv','a.h5']:
        opts={'array_columns':dict(zip('txyp',range(4)))} if name.endswith('.npy') else {}
        loaded=load_stream(tmp_path/name,timestamp_unit='s',resolution=(e.width,e.height),start_s=0.,**opts)
        assert loaded.digest()==e.digest()


def test_converted_selection_test_split_only(tmp_path):
    e=synthetic_stream()
    for ds,split in [('eds','test'),('ced','test'),('hqevfi','train')]:
        seq=tmp_path/ds/split/'seq'; (seq/'frames').mkdir(parents=True)
        for k in 'txyp': np.save(seq/f'ev_{k}.npy',getattr(e,k))
        np.save(seq/'frame_ts.npy',[0,.05,.1])
        Image.new('L',(e.width,e.height)).save(seq/'frames'/'000.png')
    selected,candidates=choose_sequence(tmp_path,seed=15)
    assert len(candidates)==2 and all('/test/' in Path(r['path']).as_posix() for r in candidates)
    assert selected==choose_sequence(tmp_path,seed=15)[0]
    loaded=load_stream(selected['path'],timestamp_unit='s')
    assert loaded.digest()==e.digest()
    with pytest.raises(ValueError): choose_sequence(tmp_path,sequence='seq')
    explicit,_=choose_sequence(tmp_path,dataset='eds',sequence='seq')
    assert explicit['dataset']=='eds'


def test_units_binary_and_missing_metadata(tmp_path):
    p=tmp_path/'a.npz'
    np.savez(p,t=[10001.,10002.],x=[0,1],y=[0,0],p=[0,1])
    e=load_stream(p,timestamp_unit='ms',polarity_encoding='binary',resolution=(2,1),start_s=10.,duration_s=.1)
    np.testing.assert_allclose(e.t,[10.001,10.002]); np.testing.assert_array_equal(e.p,[-1,1])
    with pytest.raises(ValueError): load_stream(p,timestamp_unit='ms')
    with pytest.raises(ValueError): load_stream(p,timestamp_unit='ms',resolution=(2,1),start_s=10.)


def test_end_to_end_exports_and_constant_rank(tmp_path):
    e=synthetic_stream()
    table,summary,path=run_benchmark(e,tmp_path,sweep={'polarity':[0,.5,1],'duplicate':[0,.5,1]},seeds=[0,1],progress=None)
    assert len(table)==2*3*2*9
    assert all((path/name).exists() for name in ['manifest.json','reference.npz','metrics.csv','perturbations.json','response_analysis.csv'])
    c=summary[(summary.family=='polarity')&(summary.metric=='Count Ratio')]
    assert c.constant_response.all() and c.spearman.isna().all()
    assert (c.pairwise_ranking_accuracy==.5).all()
    assert (c.strict_order_fraction==0).all()
