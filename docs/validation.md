# Validation record

The test suite contains hand-computed examples, a SciPy IG density cross-check,
brute-force Chamfer verification, 18 targeted witnesses, loader/schema tests,
perturbation invariants and end-to-end result exports. The CI workflow also
executes every notebook cell in explicit synthetic mode and uploads the outputs.

Real test datasets live on the user's server and are not included in this
repository. No real-data experiment is claimed by this implementation alone.

Execution status will be recorded here after running validation. A test's
existence is not evidence that it has passed.
