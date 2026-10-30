# Validation record

The test suite contains hand-computed examples, a SciPy IG density cross-check,
brute-force Chamfer verification, 18 targeted witnesses, loader/schema tests,
perturbation invariants and end-to-end result exports. The CI workflow also
executes every notebook cell in explicit synthetic mode and uploads the outputs.

Real test datasets live on the user's server and are not included in this
repository. No real-data experiment is claimed by this implementation alone.

## Executed baseline

Commit `05769ed9ecf2bf1d2b3f1651751c6c0ba9785435`:

- **36 tests passed**, including all 18 numerical counterexample witnesses.
- **All 25 notebook cells executed successfully**, in explicit synthetic quick mode.
- Twenty perturbation families, three severities and two seeds produced 1,080
  metric rows. This is a diagnostic run, not a real-data result.
- [Successful CI run and downloadable outputs](https://github.com/Moshrefi/evt_metric_eval/actions/runs/41005958696).
- [Executed notebook, CSVs, plots and report artifact](https://github.com/Moshrefi/evt_metric_eval/actions/runs/41005958696/artifacts/12955656577).

Subsequent changes add per-metric small-change diagnostics and a comparative
report table; the workflow reruns all tests and the complete notebook on each
push. Consult the workflow for the tested commit, rather than treating this
baseline record as evidence for an untested revision.

The local Windows Python could check syntax but had no numerical packages and
no package-install network access. Numerical tests and notebook execution above
were performed on the GitHub Ubuntu runner with Python 3.12.15, NumPy 2.5.4,
SciPy 1.18.1, pandas 3.1.0 and matplotlib 3.11.2. Run manifests record package
versions and source hashes.
