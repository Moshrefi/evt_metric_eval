# Definitions and scientific scope

Protocol: **fixed_support_v1**. These are independently implemented, audited
versions of the definitions used by the reference evaluation notebook. They are
not claimed to reproduce a published paper's numerical tables.

## Provenance and a critical distinction

- Evaluation notebook: [evtween compare_final.ipynb, commit 2a9bab9](https://github.com/Ahmadmoo/evtween/blob/2a9bab9de3c8f14986d9c549ce4a524c646489d6/benchmarks/event_generation/compare_final.ipynb).
- Five fidelity definitions: [TIDES metrics.py, commit 3db4de5](https://github.com/Moshrefi/TIDES/blob/3db4de58113d1e4ecafe63d4f0469b81862966cd/tides/eval/metrics.py).
- Voxel definitions: [comparison_completion.py](https://github.com/Ahmadmoo/evtween/blob/2a9bab9de3c8f14986d9c549ce4a524c646489d6/benchmarks/event_generation/comparison_completion.py) and [visualization.py](https://github.com/Ahmadmoo/evtween/blob/2a9bab9de3c8f14986d9c549ce4a524c646489d6/benchmarks/event_generation/visualization.py).
- Reference data-root convention: `/u/marmary/airlab_v2e/evtween/data`.
  The old notebook selected EDS **train**; this benchmark defaults explicitly to **test**.
- There is no evaluation notebook in the inspected `v5.1` tree. Its
  `losses.event_nll` is a different, model-conditioned likelihood. The IG-NLL
  below fits a distribution to event ISIs and requires **no checkpoint**.

The TIDES source itself documents unavailable supplementary settings. Neither
exact paper reproduction nor universally reliable generator rankings follow
from its implementation. No source project is imported or modified here.

## Shared domain and intentional protocol changes

Each pair has an identical sensor rectangle and half-open interval `[start,stop)`.
Timestamps are float64 source-clock seconds. They are never separately rebased
or automatically aligned. Subtracting the *same* start for bin indexing preserves
relative absolute-clock offsets. Exact duplicates retain their multiplicity.

Unlike the source, fixed windows start at the declared observation start, not
the minimum observed event time. Same-ts clock quantization is also anchored
there. Chamfer includes events at the maximum observed timestamp as long as
they are inside the declared interval. These changes prevent perturbations
from changing the measurement grid; numerical values need not match legacy
event-dependent windows. Fano pads a final partial window with empty bins;
prefer duration an integer multiple of `window` (default 0.1 / 0.05 s).

Chamfer uses canonical representatives before voxel density reduction; they
are independent of storage order. If a point cap is hit, the result is marked
`approximate`. Identity uses the same deterministic subset on both sides.
Unmatched Chamfer groups and Fano cells retain source-style exclusion, but the
score is `partial` with coverage diagnostics and excluded from primary ranking.
Degenerate IG fits yield `undefined`, not a huge arbitrary precision parameter.

## 1. IG-NLL (pooled inter-spike intervals)

Sort times within each pixel, irrespective of polarity, and pool adjacent
intervals `d`. A pixel with fewer than two events contributes no interval.
Clamp intervals below `isi_min` (default 1 microsecond); record clamp counts.
Fit the modified stream's positive intervals:

$$\hat\mu=\overline d,\qquad
\hat\lambda=\left[\frac1n\sum_i\left(\frac1{d_i}-\frac1{\hat\mu}\right)\right]^{-1}.$$

$$f(d;\mu,\lambda)=\sqrt{\frac{\lambda}{2\pi d^3}}
\exp\left[-\frac{\lambda(d-\mu)^2}{2\mu^2d}\right],\qquad
S=-\frac1{n_R}\sum_{d\in R}\log f(d;\hat\mu_M,\hat\lambda_M).$$

This is a directed cross-likelihood, not a distance. It discards pixel identity,
absolute time, interval order after pooling, and polarity. It responds to event
count only through available ISIs, not an explicit count penalty. Seconds are
fixed: changing units shifts differential NLL by a logarithmic constant.
Negative scores are valid. Self-score is not zero. Nearly constant ISIs give
an unbounded shape MLE and are marked undefined. Cross-score minus self-score
is reported as a diagnostic; do not assume it always increases with corruption.

Independent density check: [SciPy inverse Gaussian parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html):
`invgauss.logpdf(d, mu/lambda, scale=lambda)`.
Cost is dominated by per-pixel sorting, roughly O(N log N).

## 2. Motion-scaled spatiotemporal Chamfer

Embed `(x,y,v*(t-window_start))`, separately for each window and polarity.
`v=1000 px/s` is the inherited time-to-space scale, **not a measured scene speed**.
Default 1 px embedding-grid density reduction retains one representative per
cell. Each nonempty shared group contributes

$$C_g=\tfrac12\left[\frac1{|A_g|}\sum_{a\in A_g}\min_{b\in B_g}\|a-b\|_2+
\frac1{|B_g|}\sum_{b\in B_g}\min_{a\in A_g}\|b-a\|_2\right].$$

The final score equally averages shared groups and optionally divides by
`hypot(width,height)`. Distances are Euclidean, **not squared**. This is
nearest-neighbor matching, not bijective matching or optimal transport.
Exact duplicated support has zero distance; multiplicity is not preserved by
density reduction. Source-style missing groups are skipped: a zero partial
score is not evidence of complete agreement. Window edges, voxel size, motion
scale and subsampling affect the answer. Typical KD-tree cost O(N log N);
worst-case behavior is data dependent. Record `coverage` and sampled-away count.

## 3. Same-timestamp Ratio

Within each fixed window, quantize timestamps by rounding to `resolution`
(default 1 ns). Count the fraction of events whose quantized timestamp is
shared by at least one other event. Combine window fractions weighted by event
count. This is **one-stream timestamp multiplicity**, not timestamp agreement
between the reference and modified stream. It ignores coordinates and polarity.
Raw values are shown for both streams; discrepancy is their absolute difference.
Window boundaries and quantization matter. Complexity O(N log N).

## 4. Fano Factor Error

Within each 50 ms window and 8 x 8 px spatial cell, count both polarities in
2 ms sub-bins. Use population variance (`ddof=0`):

$$F=\operatorname{Var}(n_j)/\operatorname{Mean}(n_j),\qquad
E_F=\operatorname{Mean}_{g\in\text{shared active}}\left|\log\frac{F_{M,g}+\epsilon}{F_{R,g}+\epsilon}\right|.$$

Default epsilon is 0.001. Empty cells have F=0 internally; the source comparison
uses only cells active in both streams. Missing active cells are exposed through
partial status and coverage. Polarity, fine positions, and ordering of temporal
counts can be lost. A memory guard rejects excessively large dense count grids.
Cost O(N + number of cells).

## 5. ISI-Spike Ratio

Use the same pooled per-pixel, polarity-mixed ISIs as IG-NLL:

$$Q=\frac1n\sum_i\mathbf1[d_i<\epsilon_{spike}],\quad \epsilon_{spike}=10\ \mu s.$$

Raw ratio is a one-stream diagnostic; compare its absolute difference from the
reference ratio. It is not an ISI distribution distance. Constant time shifts,
polarity inversion and many interval changes above the threshold are invisible.
The source's optional clock-quantum extension is disabled and not implemented
in this version. Complexity O(N log N).

## 6. Count Ratio

$$Q_N=N_M/N_R.$$

Ideal raw ratio is 1; discrepancy `abs(Q_N-1)` treats relative excess/deficit
symmetrically about 1. Empty reference gives undefined. It ignores all locations,
times and polarities. O(1) after loading.

## 7-9. Voxel MAE, RMSE, Relative L1

Hard-assign each event to its native pixel and one shared temporal bin, using
`[left,right)`. Add signed polarities. No per-stream normalization, interpolation,
polarity channels, or spatial downsampling. The default is 10 uniform time bins;
the notebook can instead use actual frame edges clipped to the selected interval.

$$V_{b,y,x}=\sum_{i\ \text{in cell}}p_i,\quad D=V_M-V_R,$$
$$\mathrm{MAE}=\frac{\sum|D|}{K},\quad
\mathrm{RMSE}=\sqrt{\frac{\sum D^2}{K}},\quad
\mathrm{Relative\ L1}=\frac{\sum|D|}{\sum|V_R|}.$$

K includes **all** sensor pixels and bins, including implicit zeros. Sparse
storage changes memory use, not these denominators. Relative L1 is undefined
if reference signed mass is zero, even if real events exist. ON/OFF cancellation
and within-bin timing changes can be invisible. RMSE emphasizes concentrated
errors. With a fixed reference and grid, MAE and Relative L1 are exact positive
scalar multiples (when the denominator is nonzero), so their rankings coincide.
Cost O(N log N) for sparse aggregation. Raw values depend on grid and support.

## Fair analysis

Show all raw scores, self-scores, discrepancy transforms, statuses and reasons.
Rank severities only within a single perturbation family and seed. Report
Spearman, Kendall, tie-aware pairwise ordering (ties count 0.5), strict ordering,
and tied-pair fraction. A constant metric has undefined correlation, not zero.
Do not equate severity units across families or average arbitrary metric ranks
into a universal winner. SD across seeds is stochastic variation on one stream,
not a confidence interval over datasets. No small-sample correlation p-values
are asserted; see [SciPy Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

Each failure claim must link to a numeric witness. Such witnesses prove a
representation can miss a difference; they do not establish its frequency or
importance in every application. Hyperparameters must be fixed before looking
at results; parameter sweeps are reported separately, not selected for an outcome.
