# Event-stream metric evaluation

One configurable Jupyter notebook for investigating **nine metrics**, using
controlled changes to **one event stream from one sequence**. Independent of
event-generation models and training. Source repositories are read-only references.

Open [event_stream_metric_benchmark.ipynb](event_stream_metric_benchmark.ipynb).

## Install and run

```bash
git clone https://github.com/Moshrefi/evt_metric_eval.git
cd evt_metric_eval
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[notebook,test]'
jupyter lab event_stream_metric_benchmark.ipynb
```

Use a Python 3.10+ environment; CI uses 3.12. CPU is sufficient. Metrics operate
on the full selected stream; Chamfer's explicit point cap is reported if hit.

## Main settings (one notebook cell)

```python
MODE = 'real'                        # or explicit 'synthetic_demo'
DATA_ROOT = Path('/u/marmary/airlab_v2e/evtween/data')
DATASET = None                      # 'eds', 'hqevfi', 'bsergb', 'ced', or all
SPLIT = 'test'                      # never silently falls back to train
SEQUENCE = None                     # e.g. a name from the printed inventory
SELECTION_SEED = 42                 # reproducible random sequence if None
START_S = None                     # None: first frame timestamp; otherwise source-clock seconds
DURATION_S = 0.1                   # exactly one selected interval
```

`INPUT_FILE` overrides discovery for a direct NPZ/NPY/HDF5/CSV/TXT file or
converted sequence directory. Generic files need explicit units, polarity
encoding, resolution and observation start if not provided by supported metadata.
The notebook explains key mappings and array column order. Do not guess a
timestamp unit from its magnitude. Native raw camera containers are not decoded;
the directory convention is `ev_t.npy`, `ev_x.npy`, `ev_y.npy`, `ev_p.npy`.

The reference evaluation notebook was found on `benchmark/final-comparison`,
not `v5.1`. Its data root is reused as a configurable default. Its IG-NLL fits
pooled ISIs and does **not** require the evtween checkpoint. Same-ts is internal
timestamp multiplicity, not cross-stream timestamp agreement.

## Outputs

Each execution creates a new ignored `runs/<timestamp>/` with reference NPZ,
manifest (input hash, source hashes, versions, selection, seeds, parameters),
raw metrics CSV, perturbation records, response analysis, plots, counterexample
CSV and a generated failure report. Set `SAVE_MODIFIED=True` to also store all
perturbed streams. Existing runs are never overwritten.

- [Metric definitions, formulas and intentional source differences](docs/metrics.md)
- [Experimental plan and validation gates](docs/experiment_plan.md)
- [Validation evidence](docs/validation.md)

The source's missing-group exclusions are preserved but flagged `partial`.
Undefined and approximate results remain visible and are excluded from primary
ordering analysis. All nine metrics stay in the tables. Synthetic examples are
marked synthetic; missing real data never silently switches the notebook to a demo.
No general dataset claim is made from a single sequence.

## Tests

```bash
python -m pytest -q
python scripts/execute_notebook.py --output runs/notebook_validation.ipynb
```

The second command explicitly executes a synthetic validation copy. It never
changes the real-data defaults in the main notebook. No data or checkpoints
are downloaded by the notebook.

## Saved experiments and optional GPU

Use [event_stream_metric_benchmark_cached_gpu.ipynb](event_stream_metric_benchmark_cached_gpu.ipynb) for persistent NPZ samples, a CSV experiment index, cached reference features, per-sample resume, and hybrid CuPy CUDA evaluation. See [setup and scope](docs/cached_gpu.md). The original notebook remains available. GPU correctness must pass the notebook's startup parity check on your target machine; no GPU speedup is claimed without measurement.
