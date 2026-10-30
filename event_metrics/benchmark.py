"""Run manifest, raw scores and independently regenerated perturbations."""
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from .metrics import MetricConfig,evaluate,discrepancy
from .perturbations import perturb,default_sweep
from .loading import save_stream
from .analysis import summarize_responses


def _json_default(o):
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,Path): return str(o)
    raise TypeError(type(o).__name__)


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=_json_default),encoding='utf-8')


def run_benchmark(reference, output_root='runs', cfg=None, sweep=None, seeds=tuple(range(10)),
                  voxel_edges=None, save_modified=False, progress=print):
    cfg=cfg or MetricConfig(); sweep=default_sweep() if sweep is None else sweep
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Need distinct seeds')
    edges=np.linspace(reference.start,reference.stop,cfg.voxel_bins+1) if voxel_edges is None else np.asarray(voxel_edges)
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=Path(output_root)/name; out.mkdir(parents=True,exist_ok=False)
    save_stream(reference,out/'reference.npz')
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.SubprocessError,FileNotFoundError): commit=None
    source_hashes={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    manifest=dict(protocol='fixed_support_v1',reference=reference.statistics(),input_metadata=reference.metadata,
                  metric_config=asdict(cfg),sweep=sweep,seeds=list(seeds),voxel_edges_s=edges,
                  python=platform.python_version(),platform=platform.platform(),git_commit=commit,
                  source_hashes=source_hashes,packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','matplotlib']},
                  synthetic=reference.metadata.get('source')=='synthetic',
                  analysis='Single selected stream; seed variation does not establish cross-sequence generality')
    write_json(out/'manifest.json',manifest)
    baseline=evaluate(reference,reference,cfg,edges)
    rows=[]; logs=[]; original_hash=reference.digest()
    for family,levels in sweep.items():
        if progress: progress(f'{family}: {len(levels)} severities x {len(seeds)} seeds')
        for seed in seeds:
            for severity in levels:
                begin=time.perf_counter()
                modified,record=perturb(reference,family,severity,seed)
                scores=evaluate(reference,modified,cfg,edges)
                record.update(statistics=modified.statistics(),elapsed_s=time.perf_counter()-begin)
                logs.append(record)
                if save_modified:
                    save_stream(modified,out/f'{family}_{severity:g}_{seed}.npz')
                for metric,score in scores.items():
                    b=baseline[metric]
                    d=discrepancy(metric,score['value'],b['value'])
                    status=score['status']
                    if not np.isfinite(d) and status=='ok': status='undefined'
                    rows.append(dict(family=family,severity=severity,seed=seed,metric=metric,
                        value=score['value'],baseline=b['value'],discrepancy=d,status=status,
                        reason=score['reason'] or (b['reason'] if not np.isfinite(d) else ''),
                        details=json.dumps(score['details'],default=_json_default),n_reference=len(reference),n_modified=len(modified)))
        # Incremental checkpoints, before moving to the next family.
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        write_json(out/'perturbations.json',logs)
    if reference.digest()!=original_hash: raise AssertionError('Reference was modified')
    table=pd.DataFrame(rows)
    summary=summarize_responses(table)
    summary.to_csv(out/'response_analysis.csv',index=False)
    write_json(out/'baseline.json',baseline)
    return table,summary,out
