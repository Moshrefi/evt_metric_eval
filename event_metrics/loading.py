"""Schema-explicit loaders. No imports from evtween or any generator."""
from pathlib import Path
import json
import numpy as np
from .events import EventStream

UNITS = {'s': 1., 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
DATASETS = ('eds', 'hqevfi', 'bsergb', 'ced')


def discover_sequences(data_root, dataset=None, split='test'):
    """Only the requested split; never fall back silently to train or validation."""
    root = Path(data_root).expanduser()
    result = []
    for ds in ([dataset] if dataset else DATASETS):
        for p in sorted((root / ds / split).rglob('ev_t.npy')):
            if all((p.parent / f'ev_{k}.npy').is_file() for k in 'xyp'):
                result.append({'dataset': ds, 'sequence': p.parent.name, 'path': str(p.parent.resolve())})
    return result


def choose_sequence(data_root, dataset=None, sequence=None, split='test', seed=0):
    records = discover_sequences(data_root, dataset, split)
    if sequence is not None:
        records = [r for r in records if r['sequence'] == sequence or r['path'] == str(sequence)]
        if len(records) != 1:
            raise ValueError(f"Expected one explicit sequence; got {len(records)}. Set DATASET too.")
        return records[0], records
    if not records:
        raise FileNotFoundError(f"No converted events under {data_root}/<dataset>/{split}; set DATA_ROOT or INPUT_FILE.")
    # Uniform across sorted sequences, not uniform across datasets. Record the candidate list.
    return records[int(np.random.default_rng(seed).integers(len(records)))], records


def _polarity(p, encoding):
    p = np.asarray(p)
    allowed = {'signed': [-1, 1], 'binary': [0, 1]}
    if encoding not in allowed or not np.isin(p, allowed[encoding]).all():
        raise ValueError(f"Polarity does not match explicit encoding {encoding}")
    return (2*p-1) if encoding == 'binary' else p


def load_stream(path, *, timestamp_unit, polarity_encoding='signed', resolution=None,
                start_s=None, duration_s=0.1, keys=None, array_columns=None, delimiter=','):
    """Read one interval in source-clock seconds. keys map canonical names to file fields.

    Converted directory: ev_{t,x,y,p}.npy; frame_ts.npy supplies observation start
    and coverage, a frame PNG supplies resolution (no model preprocessing).
    Generic arrays require explicit start_s and resolution or stored width/height.
    Selection to the requested interval is intentional and recorded.
    """
    path = Path(path).expanduser()
    if timestamp_unit not in UNITS:
        raise ValueError('Explicit timestamp_unit must be s, ms, us, or ns')
    scale = UNITS[timestamp_unit]
    keys = keys or {k: k for k in 'txyp'}
    meta = {'source': str(path.resolve()), 'timestamp_unit': timestamp_unit, 'polarity_encoding': polarity_encoding}
    coverage = None
    if path.is_dir():
        arrays = {k: np.load(path/f'ev_{k}.npy', mmap_mode='r', allow_pickle=False) for k in 'txyp'}
        frames = path/'frame_ts.npy'
        if frames.exists():
            ft = np.load(frames, allow_pickle=False)*scale
            if ft.ndim != 1 or len(ft)<2 or not np.isfinite(ft).all() or np.any(np.diff(ft)<=0):
                raise ValueError('Invalid frame timestamps')
            coverage = (float(ft[0]), float(ft[-1]))
            meta['frame_timestamps_s'] = ft.tolist()
            if start_s is None:
                start_s = coverage[0]
        if resolution is None:
            pngs = sorted((path/'frames').glob('*.png'))
            if pngs:
                from PIL import Image
                with Image.open(pngs[0]) as im:
                    resolution = im.size
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[keys[k]] for k in 'txyp'}
            if resolution is None and 'width' in d and 'height' in d:
                resolution = (int(d['width']), int(d['height']))
    elif path.suffix.lower() == '.npy':
        d = np.load(path, mmap_mode='r', allow_pickle=False)
        if d.dtype.names:
            arrays = {k: d[keys[k]] for k in 'txyp'}
        elif array_columns is not None:
            arrays = {k: d[:, array_columns[k]] for k in 'txyp'}
        else:
            raise ValueError('Unstructured NPY requires array_columns mapping')
    elif path.suffix.lower() in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path) as d:
            arrays = {k: np.asarray(d[keys[k]]) for k in 'txyp'}
    elif path.suffix.lower() in ('.txt', '.csv'):
        if array_columns is not None:
            d = np.loadtxt(path, delimiter=delimiter, ndmin=2)
            arrays = {k: d[:, array_columns[k]] for k in 'txyp'}
        else:
            d = np.genfromtxt(path, delimiter=delimiter, names=True, ndmin=1)
            arrays = {k: d[keys[k]] for k in 'txyp'}
    else:
        raise ValueError(f'Unsupported input: {path}')
    if resolution is None or start_s is None:
        raise ValueError('Supply RESOLUTION_WH and START_S; do not infer sensor size or coverage from events')
    if any(np.asarray(a).ndim != 1 for a in arrays.values()) or len({len(a) for a in arrays.values()}) != 1:
        raise ValueError('Malformed event arrays')
    start_s, stop_s = float(start_s), float(start_s+duration_s)
    if coverage and (start_s < coverage[0] or stop_s > coverage[1]):
        raise ValueError(f'Requested interval outside frame coverage {coverage}')
    # Scan timestamps in bounded chunks; silently hiding NaNs behind an interval mask is forbidden.
    selected = []
    for lo in range(0, len(arrays['t']), 1_000_000):
        t = np.asarray(arrays['t'][lo:lo+1_000_000], dtype=float)*scale
        if not np.isfinite(t).all():
            raise ValueError('Nonfinite source timestamps')
        selected.append(np.flatnonzero((t >= start_s) & (t < stop_s))+lo)
    idx = np.concatenate(selected) if selected else np.zeros(0, dtype=int)
    values = {k: np.asarray(v[idx]) for k, v in arrays.items()}
    values['t'] = values['t'].astype(np.float64)*scale
    values['p'] = _polarity(values['p'], polarity_encoding)
    meta.update(source_events=len(arrays['t']), selected_events=len(idx), selection='explicit half-open time interval',
                validation_scope='all source timestamps; coordinates and polarity of selected events', keys=keys,
                array_columns=array_columns, duration_s=duration_s)
    return EventStream(**values, width=int(resolution[0]), height=int(resolution[1]), start=start_s, stop=stop_s, metadata=meta)


def save_stream(stream, path):
    np.savez_compressed(path, **{k: getattr(stream, k) for k in 'txyp'}, width=stream.width,
                        height=stream.height, start=stream.start, stop=stream.stop)
