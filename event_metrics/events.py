"""Validated immutable event multisets on an explicit observation domain."""
from dataclasses import dataclass, field
import hashlib
import numpy as np


@dataclass(frozen=True)
class EventStream:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    width: int
    height: int
    start: float
    stop: float
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not isinstance(self.width, (int, np.integer)) or not isinstance(self.height, (int, np.integer)):
            raise ValueError("Resolution must contain integers")
        if min(self.width, self.height) < 1 or not np.isfinite([self.start, self.stop]).all() or self.stop <= self.start:
            raise ValueError("Need positive resolution and finite start < stop")
        arrays = [np.asarray(getattr(self, k)) for k in ('t', 'x', 'y', 'p')]
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("t,x,y,p must be equal-length one-dimensional arrays")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Nonfinite events are invalid; no silent filtering")
        t, x, y, p = arrays
        if np.any((x != np.floor(x)) | (y != np.floor(y))):
            raise ValueError("Coordinates must be integer pixel indices")
        if np.any((x < 0) | (x >= self.width) | (y < 0) | (y >= self.height)):
            raise ValueError("Coordinates outside sensor")
        if np.any((t < self.start) | (t >= self.stop)):
            raise ValueError("Events outside half-open observation interval")
        if not np.isin(p, [-1, 1]).all():
            raise ValueError("Explicitly convert polarity encoding before construction")
        # Stable canonical order; ties and exact duplicates remain distinct events.
        order = np.lexsort((p, y, x, t))
        for k, a, dtype in zip(('t', 'x', 'y', 'p'), arrays, (np.float64, np.int64, np.int64, np.int8)):
            a = np.array(a[order], dtype=dtype, copy=True)
            a.flags.writeable = False
            object.__setattr__(self, k, a)

    def __len__(self):
        return len(self.t)

    def replace(self, **changes):
        fields = {k: getattr(self, k) for k in self.__dataclass_fields__}
        fields.update(changes)
        return EventStream(**fields)

    def select(self, selection):
        return self.replace(**{k: getattr(self, k)[selection] for k in ('t', 'x', 'y', 'p')})

    def digest(self):
        h = hashlib.sha256()
        h.update(repr((self.width, self.height, self.start, self.stop)).encode())
        for k in ('t', 'x', 'y', 'p'):
            h.update(getattr(self, k).tobytes())
        return h.hexdigest()

    def statistics(self):
        n = len(self)
        return dict(events=n, width=self.width, height=self.height, observation_start_s=self.start,
                    observation_stop_s=self.stop, event_min_s=float(self.t.min()) if n else None,
                    event_max_s=float(self.t.max()) if n else None,
                    positive=int((self.p > 0).sum()), negative=int((self.p < 0).sum()),
                    positive_fraction=float((self.p > 0).mean()) if n else None,
                    active_pixels=len(np.unique(self.y * self.width + self.x)),
                    events_per_second=n / (self.stop-self.start),
                    events_per_pixel_second=n / ((self.stop-self.start)*self.width*self.height),
                    sha256=self.digest())


def synthetic_stream(seed=0, width=32, height=24, duration=0.1):
    """Explicit synthetic diagnostic data, never a substitute for missing real data."""
    rng = np.random.default_rng(seed)
    times, xs, ys, ps = [], [], [], []
    for x, y in [(10, 10), (12, 11), (16, 12), (19, 14)]:
        t = np.cumsum(rng.uniform(0.0002, 0.002, 90)) + 0.008
        t = t[t < duration-0.008]
        # Shared sensor ticks introduce controlled timestamp collisions.
        t = np.round(t/0.0001)*0.0001
        times.extend(t); xs.extend([x]*len(t)); ys.extend([y]*len(t))
        ps.extend(rng.choice([-1, 1], len(t)))
    return EventStream(times, xs, ys, ps, width, height, 0., duration, {'source': 'synthetic', 'seed': seed})
