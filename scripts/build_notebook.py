"""Rebuild the English research notebook with standard-library Python."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def md(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.strip()+'\n',id=f'cell-{len(cells):03d}'))


def code(text):
    cells.append(dict(cell_type='code',metadata={},source=text.strip()+'\n',outputs=[],execution_count=None,id=f'cell-{len(cells):03d}'))


md('''# Event-stream metric benchmark and failure analysis

**Question:** Which differences can the nine existing metrics see, and which
can they miss? One reference stream, controlled perturbations, and explicit
numeric counterexamples. No event generator, training or model checkpoint.

Run top to bottom. Edit the settings cell to choose one real sequence and time
interval. The optional synthetic mode is explicitly labeled and is for validation.
All scores use a common time reference. Results from one stream do not establish
cross-dataset generality. Read the definition/provenance section before interpreting scores.
''')
code('''from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / 'event_metrics').is_dir():
    raise FileNotFoundError('Open this notebook with the repository root as the working directory.')
sys.path.insert(0, str(ROOT))
from event_metrics.events import synthetic_stream
from event_metrics.loading import choose_sequence, discover_sequences, load_stream
from event_metrics.metrics import MetricConfig, NAMES, evaluate
from event_metrics.perturbations import guard_reference, perturb, default_sweep
from event_metrics.benchmark import run_benchmark, write_json
from event_metrics.analysis import correlation_by_family
from event_metrics.counterexamples import cases, audit_cases, boundary_case, small_change_diagnostics
from event_metrics.visualization import (compare_streams, distributions, response_plot, heatmap, case_gallery)
pd.set_option('display.max_colwidth', 100)
''')
md('''## 1. Settings: choose exactly one stream

`DATASET=None` searches EDS/HQ-EVFI/BS-ERGB/CED **test** folders. With
`SEQUENCE=None`, choose uniformly across the sorted sequence inventory using
`SELECTION_SEED`. Set both names to select manually. No train-split fallback.
`START_S` is in the source clock; `None` uses the first frame timestamp.

For a generic file, set `INPUT_FILE`, `START_S`, `RESOLUTION_WH`, timestamp unit,
polarity encoding and, if needed, field names/array columns. Do not change units
without checking the source. Converted evtween NPY data use seconds and +/-1.

`VOXEL_MODE='uniform'` uses shared hard bins; `'frames'` uses actual source frame
edges within the interval, with interval endpoints added. All operations use
the same resolution and time support. `QUICK=True` reduces repeats and levels
only; it does not randomly discard events.
''')
code('''MODE = os.environ.get('EVT_METRIC_MODE', 'real')  # 'real' or 'synthetic_demo'
DATA_ROOT = Path('/u/marmary/airlab_v2e/evtween/data')
DATASET = None                 # None, 'eds', 'hqevfi', 'bsergb', 'ced'
SPLIT = 'test'
SEQUENCE = None                # exact sequence name; set DATASET when names are ambiguous
SELECTION_SEED = 42
INPUT_FILE = None              # overrides dataset discovery; can also be a sequence directory
START_S = None                 # absolute/source-clock seconds
DURATION_S = 0.1
TIMESTAMP_UNIT = 's'
POLARITY_ENCODING = 'signed'   # 'signed' (-1,+1) or 'binary' (0,1)
RESOLUTION_WH = None           # explicit (width,height) if metadata/frame is unavailable
FIELD_KEYS = {'t':'t','x':'x','y':'y','p':'p'}
ARRAY_COLUMNS = None           # e.g. {'t':0,'x':1,'y':2,'p':3} for an unstructured NPY/TXT
DELIMITER = ','

SPATIAL_GUARD_PX = 8           # declared reference selection; supports maximum pure shift
TEMPORAL_GUARD_S = 0.006       # supports +/-5 ms shift without dropping events
VOXEL_MODE = 'uniform'
CFG = MetricConfig(voxel_bins=10)
QUICK = os.environ.get('EVT_METRIC_QUICK', '0') == '1'
SEEDS = list(range(2 if QUICK else 10))
SWEEP = default_sweep()
if QUICK:
    SWEEP = {family:[levels[0], levels[len(levels)//2], levels[-1]] for family,levels in SWEEP.items()}
SAVE_MODIFIED = False
RUN_OPTIONAL_COMBINATIONS = False
RUN_PARAMETER_ABLATIONS = False
OUTPUT_ROOT = ROOT / 'runs'
''')
md('''## 2. Inventory, selection and validation

Only one sequence is loaded. The inventory is printed so you can copy a sequence
name into the settings cell. Missing files raise an actionable error; they do
not trigger a synthetic fallback. The guard is a documented selection of the
reference, not a hidden deletion from individual perturbations.
''')
code('''if MODE == 'synthetic_demo':
    loaded = synthetic_stream(SELECTION_SEED)
    inventory = []
    print('SYNTHETIC DIAGNOSTIC DATA — no real dataset result.')
elif MODE == 'real':
    if INPUT_FILE is None:
        inventory = discover_sequences(DATA_ROOT, DATASET, SPLIT)
        display(pd.DataFrame(inventory))
        chosen, inventory = choose_sequence(DATA_ROOT, DATASET, SEQUENCE, SPLIT, SELECTION_SEED)
        source = chosen['path']
    else:
        source, inventory = INPUT_FILE, []
    loaded = load_stream(source, timestamp_unit=TIMESTAMP_UNIT, polarity_encoding=POLARITY_ENCODING,
        resolution=RESOLUTION_WH, start_s=START_S, duration_s=DURATION_S,
        keys=FIELD_KEYS, array_columns=ARRAY_COLUMNS, delimiter=DELIMITER)
else:
    raise ValueError('MODE must be real or synthetic_demo')

reference = guard_reference(loaded, SPATIAL_GUARD_PX, TEMPORAL_GUARD_S)
reference = reference.replace(metadata={**reference.metadata, 'selection_seed':SELECTION_SEED,
    'inventory':inventory, 'dataset_filter':DATASET, 'sequence_filter':SEQUENCE, 'split':SPLIT})
if VOXEL_MODE == 'uniform':
    voxel_edges = np.linspace(reference.start, reference.stop, CFG.voxel_bins+1)
elif VOXEL_MODE == 'frames':
    frame_times = np.asarray(reference.metadata.get('frame_timestamps_s', []))
    if not len(frame_times):
        raise ValueError('No frame timestamps for VOXEL_MODE=frames')
    voxel_edges = np.unique(np.r_[reference.start,
        frame_times[(frame_times>reference.start)&(frame_times<reference.stop)], reference.stop])
else:
    raise ValueError('Unknown VOXEL_MODE')
display(pd.DataFrame([loaded.statistics(), reference.statistics()], index=['Loaded clip','Selected reference']))
print('Reference selection:', reference.metadata['guard_selection'])
print('Voxel edges (seconds):', voxel_edges)
''')
code('''fig = compare_streams(loaded, reference, 'Loaded clip and declared reference selection')
plt.show(); plt.close(fig)
fig = distributions(reference, reference)
plt.show(); plt.close(fig)
''')
md('''## 3. Exact metric definitions and provenance

The evaluation notebook on `benchmark/final-comparison` uses the TIDES ISI-fit
IG-NLL. It is different from `evtween/v5.1/losses.event_nll`. All nine metrics
below are available without that model. This protocol fixes the observation
grid and reports source exclusions explicitly. Read the documented differences.
''')
code("display(Markdown((ROOT/'docs/metrics.md').read_text(encoding='utf-8')))")
md('''## 4. Identity control

Distances/errors should be zero where defined; count ratio is one. Same-ts and
ISI-spike retain the reference's own values. IG-NLL is a cross-likelihood whose
self-score is generally nonzero and may be negative. Undefined values include
their reasons. No numeric metric is silently substituted.
''')
code('''self_scores = evaluate(reference, reference, CFG, voxel_edges)
display(pd.DataFrame(self_scores).T[['value','status','reason','details']])
''')
md('''## 5. Single-perturbation experiment

Each modified stream is generated independently from the immutable reference.
Matched seeds reduce unrelated randomness across severities. Physical changes
and actual event counts are saved. Boundary reflection applies only to explicitly
named noise operations; pure shifts outside the domain raise an error.

No total ranking across different metrics or corruption families is computed.
Full experiments can take time on dense data; select a shorter interval or use
`QUICK=True` for an initial run, keeping enough time for the intended bins.
''')
code('''table, response_summary, run_dir = run_benchmark(reference, OUTPUT_ROOT, CFG, SWEEP, SEEDS,
    voxel_edges=voxel_edges, save_modified=SAVE_MODIFIED)
display(table[['family','severity','seed','metric','value','baseline','discrepancy','status']].head(18))
print('Saved results:', run_dir)
display(table.groupby(['metric','status']).size().rename('rows').unstack(fill_value=0))
''')
md('''## 6. Inspect an original/modified pair

Set `VIEW_FAMILY`, `VIEW_SEVERITY` and `VIEW_SEED` to any configured values.
Blue is OFF; red is ON. Scatter points may be reduced for display only. All
metrics use the full selected input unless Chamfer explicitly reports its cap.
''')
code('''VIEW_FAMILY, VIEW_SEVERITY, VIEW_SEED = 'polarity', 1.0, SEEDS[0]
modified, perturbation_record = perturb(reference, VIEW_FAMILY, VIEW_SEVERITY, VIEW_SEED)
display(pd.DataFrame(evaluate(reference,modified,CFG,voxel_edges)).T[['value','status','reason']])
print(perturbation_record)
fig = compare_streams(reference,modified,f'{VIEW_FAMILY}: {VIEW_SEVERITY}')
fig.savefig(run_dir/'selected_comparison.png',dpi=120); plt.show(); plt.close(fig)
fig = distributions(reference,modified)
fig.savefig(run_dir/'selected_distributions.png',dpi=120); plt.show(); plt.close(fig)
''')
md('''## 7. Response curves and fair comparisons

The vertical axis uses each metric's own discrepancy, not a shared scale.
Error bars are SD across seeds, not dataset confidence intervals. Partial and
approximate results are shown but excluded from the primary ranking analysis.
Constant responses have undefined Spearman/Kendall and a 100% tied-pair fraction;
a tie-aware pairwise accuracy of 0.5 must not be mistaken for sensitivity.
''')
code('''RESPONSE_FAMILY = 'polarity'
fig = response_plot(table,RESPONSE_FAMILY)
fig.savefig(run_dir/'selected_response.png',dpi=120); plt.show(); plt.close(fig)
for family in SWEEP:
    fig = response_plot(table,family)
    fig.savefig(run_dir/f'response_{family}.png',dpi=100); plt.close(fig)
display(response_summary[response_summary.family==RESPONSE_FAMILY])
monotonicity = response_summary.pivot_table(index='family',columns='metric',values='spearman',aggfunc='mean',dropna=False)
monotonicity = monotonicity.reindex(index=list(SWEEP),columns=NAMES)
fig = heatmap(monotonicity,'Within-family severity monotonicity; mean Spearman across seeds')
fig.savefig(run_dir/'monotonicity_heatmap.png',dpi=120); plt.show(); plt.close(fig)
correlation, paired_counts = correlation_by_family(table,RESPONSE_FAMILY)
fig = heatmap(correlation,'Metric rank correlation within '+RESPONSE_FAMILY)
fig.savefig(run_dir/'metric_correlation.png',dpi=120); plt.show(); plt.close(fig)
display(paired_counts)
correlation.to_csv(run_dir/'selected_family_correlation.csv')
paired_counts.to_csv(run_dir/'correlation_pair_counts.csv')
''')
md('''## 8. Individual failure analysis: two numeric witnesses per metric

These are small **synthetic** event streams designed to isolate mathematical
limitations. They use their own explicit fixed configuration (10 voxel bins,
50 ms metric windows), independent of any changed real-data settings. Each row
checks that the streams differ yet the targeted score equals its reference
score, or exposes a partial-coverage score. This is not a claim about frequency
of failure on real datasets. A failed witness check is reported, not reworded as
success. Every metric gets its own figures and explanation.
''')
code('''COUNTEREXAMPLE_CFG = MetricConfig(voxel_bins=10)
failure_table = pd.DataFrame(audit_cases(COUNTEREXAMPLE_CFG))
failure_table.to_csv(run_dir/'counterexamples.csv',index=False)
write_json(run_dir/'counterexample_config.json', __import__('dataclasses').asdict(COUNTEREXAMPLE_CFG))
display(failure_table)
for metric in NAMES:
    selected = [c for c in cases() if c.metric==metric]
    display(Markdown('### '+metric))
    display(failure_table[failure_table.metric==metric][['case','reference_score','modified_score','status','demonstrated','explanation']])
    fig = case_gallery(selected)
    fig.savefig(run_dir/('counterexample_'+metric.replace(' ','_')+'.png'),dpi=120)
    plt.show(); plt.close(fig)
assert failure_table.demonstrated.all(), 'A proposed witness did not hold. Inspect its definition/configuration.'
''')
md('''## 9. Minor change, discontinuous voxel response

The same 2 microsecond shift crosses a hard bin boundary in one case and stays
inside a bin in another. This diagnoses representation sensitivity; the mere
presence of a discontinuity is not by itself proof that a metric is useless.
''')
code('''crossing, interior = boundary_case()
boundary_rows=[]
for label,pair in [('Crosses bin edge',crossing),('Within bin',interior)]:
    scores=evaluate(*pair,COUNTEREXAMPLE_CFG)
    boundary_rows.append({'case':label,**{m:scores[m]['value'] for m in NAMES[-3:]}})
display(pd.DataFrame(boundary_rows))
pd.DataFrame(boundary_rows).to_csv(run_dir/'boundary_sensitivity.csv',index=False)
small_changes = pd.DataFrame(small_change_diagnostics())
display(small_changes)
small_changes.to_csv(run_dir/'small_change_diagnostics.csv',index=False)
''')
md('''## 10. Optional combinations and parameter ablations

Off by default. Combined experiments are intentionally sequential and labeled;
they never replace the individual experiments. Hyperparameter sweeps report
all settings rather than choosing the one that best supports a conclusion.
''')
code('''if RUN_OPTIONAL_COMBINATIONS:
    combinations = [('shift_x',2,'temporal_gaussian',1), ('delete',.2,'polarity',.25),
                    ('inject',.2,'temporal_clustering',.5), ('spatial_uniform',2,'duplicate',.2)]
    combined_rows=[]
    for first,a,second,b in combinations:
        intermediate,_=perturb(reference,first,a,SEEDS[0])
        output,_=perturb(intermediate,second,b,SEEDS[0]+1)
        for metric,s in evaluate(reference,output,CFG,voxel_edges).items():
            combined_rows.append({'first':first,'first_severity':a,'second':second,'second_severity':b,
                                  'metric':metric,'value':s['value'],'status':s['status']})
    display(pd.DataFrame(combined_rows))
    pd.DataFrame(combined_rows).to_csv(run_dir/'combined_metrics.csv',index=False)

if RUN_PARAMETER_ABLATIONS:
    from dataclasses import replace
    modified,_=perturb(reference,'shift_time',1,SEEDS[0])
    ablations=[]
    for parameter,values in {'v':[100.,1000.,10000.], 'chamfer_voxel':[0.,1.,2.],
                             'isi_spike_eps':[1e-6,1e-5,1e-4], 'voxel_bins':[5,10,20]}.items():
        for value in values:
            config=replace(CFG,**{parameter:value})
            scores=evaluate(reference,modified,config)
            for metric,s in scores.items():
                ablations.append({'parameter':parameter,'setting':value,'metric':metric,**{k:s[k] for k in ('value','status')}})
    display(pd.DataFrame(ablations))
    pd.DataFrame(ablations).to_csv(run_dir/'parameter_ablations.csv',index=False)
''')
md('''## 11. Evidence-backed report and next research questions

The report distinguishes constructed witnesses from the selected-stream sweep.
No new metric is implemented. Future validation should replicate observations
across sequences, scene types, sensor clocks and event densities before selecting
a complementary method. Coverage loss and undefined scores are results too.
''')
code('''from event_metrics.reporting import write_report
report_path = write_report(run_dir, reference, table, response_summary, failure_table)
display(Markdown(report_path.read_text(encoding='utf-8')))
print('All outputs:', run_dir)
''')
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
(ROOT/'event_stream_metric_benchmark.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Wrote notebook with',len(cells),'cells')
