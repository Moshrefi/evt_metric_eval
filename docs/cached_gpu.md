# Cached experiments and hybrid CUDA evaluation

Open **event_stream_metric_benchmark_cached_gpu.ipynb**. The original notebook is unchanged.

## Setup
Install the project into your notebook kernel environment:
```bash
python -m pip install -e '.[notebook,test]'
```
Install a compatible CuPy distribution following https://docs.cupy.dev/en/stable/install.html .
Use the same Python environment as the notebook. Do not install multiple CuPy variants.
The CUDA version in nvidia-smi is driver capability, not proof that a matching toolkit is installed.
BACKEND='cuda' requires working CuPy/CUDA and raises an error if unavailable; BACKEND='cpu' explicitly selects CPU.

CUDA: per-pixel ISI sorting, IG-NLL, Same-ts, voxel aggregation and voxel errors.
CPU: perturbation generation, file IO, validation/canonical ordering, Fano, cached SciPy KD-trees/Chamfer.
Float64 is preserved. This is not an all-GPU pipeline. No measured speedup is claimed.
The notebook compares all metrics/statuses against the original CPU evaluator on synthetic examples before the real run.
For a dedicated CUDA test run:
```bash
python -m pytest tests/test_cached_gpu.py -q
```
GPU tests skip if CuPy is absent. A CUDA test must actually pass on your server before claiming GPU validation.

## Workflow
1. Configure DATA_ROOT, DATASET, SEQUENCE, START_S and DURATION_S as before.
2. QUICK=True is the default: all families, three severities, two seeds. No event subsampling.
3. Configure SAMPLE_ROOT on a disk with sufficient capacity. The notebook prints a rough upper storage estimate.
4. Generate once; rerunning reuses existing sample files.
5. Evaluate; rerunning reuses per-sample scores for the same metric configuration/backend/code.
6. Plot saved results and load a saved sample for inspection.

Folder:
- manifest.json: reference hash and generation code identities.
- reference.npz: the selected immutable reference, including silent guard region support.
- samples.csv: family, severity, requested/effective seed, file, sample id, hash, event count.
- <sample id>.npz: uncompressed t,x,y,p arrays, width/height/start/stop, metadata.
- <sample id>.json: actual perturbation record including boundary policy and measured changes.
- evaluations/<configuration id>/: baseline, per-sample scores/timings, metrics.csv, response_analysis.csv.

Only one modified sample is evaluated at a time. Files are saved atomically.
Exact duplicates and float64 timestamps are preserved. Loads verify their event digest.
A different reference or generation implementation gets a different folder.
Metric changes reuse samples but get a different evaluation folder.
Complete sample files survive interruptions even if CSV writing was interrupted; rerun preparation to rebuild the index.
Rerunning with a different sweep replaces samples.csv with that requested plan; old sample files are retained.
Use one writer per experiment folder.

Identity and deterministic perturbations share physical files and scores across requested seeds.
effective_seed=0 records this canonical choice. Repeated deterministic rows are NOT independent observations.
Stochastic perturbations keep requested seeds; no claims of cross-sequence generality.
Saving all samples trades storage and disk IO for avoiding repeated generation; it does not eliminate metric cost.
Uncompressed storage may require tens of GB for a full sweep on million-event clips.
CPU_WORKERS controls only Chamfer queries; respect your scheduler CPU allocation.
Array operations may still peak in memory during sorts; CUDA out-of-memory is surfaced, never silently approximated.
