from pathlib import Path
import numpy as np


def write_report(run_dir,reference,table,summary,failures):
    root=Path(run_dir)
    synthetic=reference.metadata.get('source')=='synthetic'
    lines=['# Metric failure analysis', '',
           '**Selected-stream data: '+('synthetic diagnostic fixture' if synthetic else str(reference.metadata.get('source')))+'**.',
           'One selected event stream; repeated seeds are not independent datasets. All counterexample fixtures below are synthetic.', '',
           f'Reference: {len(reference):,} events, {reference.width} x {reference.height}, [{reference.start}, {reference.stop}) s.',
           'Protocol: fixed_support_v1. See manifest.json and docs/metrics.md for exact definitions and source differences.', '',
           '## Constructed numeric witnesses', '',
           '| Metric | Case | Reference score | Modified score | Status | Demonstrated | Why |',
           '|---|---|---:|---:|---|---|---|']
    for row in failures.itertuples():
        lines.append(f'| {row.metric} | {row.case} | {row.reference_score:.8g} | {row.modified_score:.8g} | {row.status} | {row.demonstrated} | {row.explanation} |')
    dimensions={
      'IG-NLL':('Pooled per-pixel ISI likelihood','Absolute time, pixel identity, polarity','Marked spatiotemporal matching'),
      'Chamfer':('Nearest-neighbor space-time proximity within polarity','Exact multiplicity; unmatched groups in source definition','Mass-sensitive matching with explicit unmatched-event handling'),
      'Same-ts':('Internal timestamp multiplicity','Locations and polarity; cross-stream time agreement','Absolute-time marked event comparison'),
      'Fano':('Coarse temporal count variability per spatial cell','Polarity, within-cell positions, nonshared active cells','Fine spatial/polarity matching with coverage accounting'),
      'ISI-spike':('Fraction of short per-pixel intervals','Absolute time, pixel identity, most ISI distribution detail','Full interval distribution plus spatial/polarity comparison'),
      'Count Ratio':('Total number of events','All spatial, temporal and polarity structure','Marked spatiotemporal distance'),
      'Voxel MAE':('Mean absolute signed-count error','Within-bin timing and cancelling ON/OFF pairs','Separate polarity channels or event-level matching'),
      'Voxel RMSE':('Signed-count error emphasizing concentrated deviations','Within-bin timing and cancelling ON/OFF pairs','Separate polarity channels or event-level matching'),
      'Voxel Relative L1':('Absolute signed-count error relative to reference mass','Within-bin timing and cancellation; unstable zero mass','Mass-preserving marked comparison')}
    lines += ['', '## Comparative interpretation', '',
              '| Metric | Captures | Misses | Demonstrated condition | Why | Possible complement (not implemented) |',
              '|---|---|---|---|---|---|']
    for metric,(captures,misses,complement) in dimensions.items():
        rows=failures[(failures.metric==metric)&failures.demonstrated]
        conditions='; '.join(rows['case']) or 'Not demonstrated in this run'
        why='; '.join(rows['explanation']) or 'Pending numerical evidence'
        lines.append(f'| {metric} | {captures} | {misses} | {conditions} | {why} | {complement} |')
    lines += ['', '## Selected-stream responses', '',
              'Numbers below describe the configured severity order, not overall perceptual quality or equivalence between corruption types.', '',
              '| Family | Metric | Mean Spearman | Mean strict order fraction | Valid levels / attempted |',
              '|---|---|---:|---:|---:|']
    for (family,metric),g in summary.groupby(['family','metric']):
        rho=g.spearman.mean(); order=g.strict_order_fraction.mean()
        f=lambda x: f'{x:.3f}' if np.isfinite(x) else 'undefined'
        lines.append(f'| {family} | {metric} | {f(rho)} | {f(order)} | {g.valid_levels.sum()} / {g.total_levels.sum()} |')
    lines += ['', '## Coverage and interpretation', '']
    for (metric,status),n in table.groupby(['metric','status']).size().items():
        lines.append(f'- {metric}: {n} rows with status `{status}`.')
    lines += ['', '## What the evidence does and does not establish', '',
      '- Exact witnesses show that identical scores do not imply identical event streams. They do not measure how often these failures occur in real data.',
      '- Count and timestamp-multiplicity diagnostics miss spatial or polarity changes in their listed witnesses. Pooled ISIs lose pixel identity and absolute time.',
      '- Chamfer ignores exact duplicated support; missing polarity/window groups can create deceptively low partial scores. Inspect coverage before interpretation.',
      '- Signed voxels discard within-bin timing and can cancel added ON/OFF events. MAE and Relative L1 have identical ordering for a fixed reference/grid with nonzero reference mass.',
      '- Fano retains coarse count variability while losing polarity and fine positions. The source definition excludes cells not active in both streams.',
      '- No universal best metric, arbitrary aggregate score, or general cross-dataset failure frequency is claimed.', '',
      '## Next steps', '',
      'Replicate the single-stream observations across sequences and camera clocks. Then investigate a small number of complementary approaches: mass-sensitive, polarity-aware matching; unbalanced transport with an explicit space-time ground cost; or kernel comparisons with declared bandwidth and mass treatment. None is implemented here.', '']
    path=root/'failure_report.md';path.write_text('\n'.join(lines),encoding='utf-8')
    return path
