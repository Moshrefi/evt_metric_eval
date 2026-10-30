"""Persistent experiment samples and per-sample resumable evaluation."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os,time
import numpy as np
import pandas as pd
from .events import EventStream
from .perturbations import perturb
from .metrics import discrepancy
from .analysis import summarize_responses
from .accelerated import Evaluator

def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); os.replace(tmp,path)

def source_hash(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()

def save_sample(e,path):
    path=Path(path); tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,**{k:getattr(e,k) for k in 'txyp'},width=e.width,height=e.height,
                 start=e.start,stop=e.stop,metadata=json.dumps(e.metadata))
    os.replace(tmp,path)

def load_sample(path,expected=None):
    with np.load(path,allow_pickle=False) as d:
        e=EventStream(**{k:d[k] for k in 'txyp'},width=int(d['width']),height=int(d['height']),
                      start=float(d['start']),stop=float(d['stop']),metadata=json.loads(str(d['metadata'])))
    if expected is not None and e.digest()!=expected: raise ValueError('Sample checksum mismatch: '+str(path))
    return e

def prepare_samples(reference,root,sweep,seeds,progress=print):
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Need distinct seeds')
    identity=dict(reference_sha256=reference.digest(),perturbations_sha256=source_hash('perturbations.py'),
                  events_sha256=source_hash('events.py'),format=1)
    folder=Path(root)/key(identity)[:24]; folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity: raise ValueError('Cache identity mismatch')
    atomic_json(manifest,identity)
    rp=folder/'reference.npz'
    if rp.exists(): load_sample(rp,identity['reference_sha256'])
    else: save_sample(reference,rp)
    jobs=[]
    deterministic={'shift_x','shift_y','shift_time','shift_time_negative','spatial_concentration','temporal_clustering','polarity_structured'}
    total=sum(len(v)*len(seeds) for v in sweep.values()); done=0
    for family,levels in sweep.items():
        for seed in seeds:
            for severity in levels:
                done+=1
                effective_seed=0 if severity==0 or family in deterministic else int(seed)
                spec=dict(family='identity' if severity==0 else family,severity=float(severity),seed=effective_seed)
                sample_id=key(spec)[:24]; file=folder/(sample_id+'.npz'); sidecar=folder/(sample_id+'.json')
                if file.exists() and sidecar.exists():
                    record=json.loads(sidecar.read_text())
                    action='reused'
                else:
                    e,record=perturb(reference,family,severity,effective_seed)
                    record['output_sha256']=e.digest()
                    save_sample(e,file); atomic_json(sidecar,record); action='created'
                jobs.append(dict(family=family,severity=float(severity),seed=int(seed),
                    effective_seed=effective_seed,sample_id=sample_id,file=file.name,
                    sha256=record['output_sha256'],events=record['output_events'],
                    perturbation_json=sidecar.name,reused_across_seeds=family in deterministic or severity==0))
                if progress: progress(f'Prepare {done}/{total}: {family} {severity} seed={seed} [{action}]')
                # Recoverable index after every sample. Files are immutable and keyed independently of metrics.
                tmp=folder/'samples.csv.tmp'; pd.DataFrame(jobs).to_csv(tmp,index=False); os.replace(tmp,folder/'samples.csv')
    return folder,pd.DataFrame(jobs)

def evaluate_samples(folder,cfg,edges,backend='cuda',workers=1,progress=print):
    folder=Path(folder); index=pd.read_csv(folder/'samples.csv')
    identity=json.loads((folder/'manifest.json').read_text())
    reference=load_sample(folder/'reference.npz',identity['reference_sha256'])
    plan=dict(config=asdict(cfg),edges=np.asarray(edges).tolist(),backend=backend,
              evaluator_sha256=source_hash('accelerated.py'),metrics_sha256=source_hash('metrics.py'),
              cache_sha256=source_hash('sample_cache.py'),sample_ids=sorted(index.sample_id.unique().tolist()))
    run_dir=folder/'evaluations'/key(plan)[:24]; run_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(run_dir/'settings.json',plan)
    engine=Evaluator(reference,cfg,edges,backend,workers)
    bp=run_dir/'baseline.json'
    if bp.exists(): baseline=json.loads(bp.read_text())
    else:
        baseline,_=engine.evaluate(reference,progress)
        atomic_json(bp,baseline)
    records={}; unique=index.drop_duplicates('sample_id')
    for i,row in enumerate(unique.itertuples(),1):
        dest=run_dir/(row.sample_id+'.json')
        if dest.exists():
            records[row.sample_id]=json.loads(dest.read_text())
            if progress: progress(f'Evaluate {i}/{len(unique)}: cached {row.family}')
            continue
        if progress: progress(f'Evaluate {i}/{len(unique)}: {row.family}, severity={row.severity}, seed={row.effective_seed}')
        begin=time.perf_counter()
        e=load_sample(folder/row.file,row.sha256)
        scores,timings=engine.evaluate(e,progress)
        rec=dict(scores=scores,metric_seconds=timings,total_seconds=time.perf_counter()-begin)
        atomic_json(dest,rec); records[row.sample_id]=rec
        del e
    rows=[]
    for job in index.to_dict('records'):
        for metric,score in records[job['sample_id']]['scores'].items():
            b=baseline[metric]; d=discrepancy(metric,score['value'],b['value'])
            status=score['status']
            if not np.isfinite(d) and status=='ok': status='undefined'
            rows.append(dict(**job,metric=metric,value=score['value'],baseline=b['value'],discrepancy=d,
                status=status,reason=score['reason'] or (b['reason'] if not np.isfinite(d) else ''),
                details=json.dumps(score['details']),n_reference=len(reference),n_modified=job['events']))
    table=pd.DataFrame(rows); summary=summarize_responses(table)
    table.to_csv(run_dir/'metrics.csv',index=False); summary.to_csv(run_dir/'response_analysis.csv',index=False)
    return table,summary,run_dir
