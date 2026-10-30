# Benchmark design

The plan was shown before implementation. One immutable event stream is loaded
from exactly one sequence. Every individual perturbation starts from that
reference. No generator, training pipeline, checkpoint or inference is required.

Pipeline: explicit file/schema -> validation -> reference statistics -> controlled
perturbations -> nine audited metric definitions -> response analysis and numeric
counterexamples -> tables, figures and reproducible manifest.

## Stages

1. Analytically predictable fixtures and numerical cross-checks.
2. Single real sequence selected manually or reproducibly at random across the
   available converted test sequences. Record candidate inventory and selected path.
3. Twenty single-perturbation families, severity zero and multiple positive
   levels, ten configurable RNG seeds. Preserve the reference hash. Seed repeats
   are not independent sequences; deterministic families naturally repeat exactly.
4. Eighteen targeted witnesses, two for each metric, and a small-change voxel
   boundary experiment. These synthetic witnesses have a fixed declared config.
5. Sensitivity, monotonicity, paired ordering, within-family metric correlation,
   missing coverage, and hypothesis-versus-demonstration report.
6. Optional parameter ablations and selected combined perturbations. Multiple
   real sequences and a new metric are future work, not a hidden default.

## Perturbations

Deletion: 0/10/20/40/60/80 percent, all events or ON/OFF eligibility separately.
Duplication/injection: 0/10/20/40/60 percent of reference count.
Spatial uniform/Gaussian noise and horizontal/vertical shifts: 0/1/2/4/8 px.
Time uniform/Gaussian noise and positive/negative shifts: 0/.1/.5/1/2/5 ms.
Polarity flips, structured polarity flips, relocation, spatial concentration,
temporal redistribution, temporal clustering: 0/.1/.25/.5/.75/1.

Random injection uses uniform pixel and time sampling plus balanced Bernoulli
polarity. Hotspot injection uses a fixed center pixel. Structured polarity
flips take the first fraction in lexicographic x/y/time order. Spatial concentration
contracts coordinates to the sensor center. Temporal clustering contracts each
timestamp toward the nearest of the quarter- and three-quarter-duration centers.
Counts use floor(fraction * eligible_count); logs contain achieved changes.

Pure shifts require the reference to fit after the maximum shift. The notebook
explicitly selects a common reference with 8 px spatial and 6 ms temporal guards
before *all* sweeps. It reports how many loaded events were excluded. Sensor
dimensions and the time domain remain unchanged. This tests a selected stream,
not the entire raw clip. Noise uses explicitly documented reflected boundaries;
Gaussian describes the pre-reflection draw, not the final boundary distribution.
Reflection counts and realized displacement are recorded. No silent clipping.

## Validation gates

- Hand-computed signed voxel, Same-ts, Fano and ISI examples.
- IG density versus SciPy; Chamfer versus brute-force distances.
- Identity, storage-order invariance, zero severity, fixed time origin,
  immutable reference, deterministic seeds and duplicate preservation.
- Explicit NaN/empty/invalid-polarity/invalid-coordinate/degenerate-fit handling.
- All supported loaders, test-split-only discovery, ambiguous selection errors.
- End-to-end notebook execution in an explicitly synthetic mode.
- Every claimed counterexample must pass its numerical witness check.

## Interpretation

Blind spots are evaluated against a stated goal: comparing full spatiotemporal
marked event streams. A count statistic can be correct for its own purpose and
still insufficient for that goal. No raw cross-metric score comparison, no
alignment optimization, no quality claim based solely on monotonicity.

Potential follow-up approaches (not implemented): polarity-aware one-to-one
matching to expose multiplicity; unbalanced transport to retain mass with a
declared space-time ground cost; characteristic-kernel MMD for distributional
structure (requires kernel bandwidth and careful mass handling). Standard
balanced, normalized distribution distances lose count information; choosing
a scale is still a scientific decision, not something these names eliminate.
