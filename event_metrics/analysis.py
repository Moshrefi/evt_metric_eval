"""Paired severity analysis; seeds are repeats, not independent datasets."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def summarize_responses(table):
    rows = []
    for (family, metric, seed), group in table.groupby(['family','metric','seed'], sort=False):
        group = group.sort_values('severity')
        ok = group[(group.status=='ok') & np.isfinite(group.discrepancy)]
        x, y = ok.severity.to_numpy(), ok.discrepancy.to_numpy()
        constant = len(y)>0 and np.allclose(y,y[0],rtol=1e-9,atol=1e-12)
        usable = len(x)>=3 and not constant and len(np.unique(x))>=3
        wins, ties, pairs = 0,0,0
        for i,j in combinations(range(len(x)),2):
            if x[i]==x[j]: continue
            pairs += 1
            tie = np.isclose(y[i],y[j],rtol=1e-9,atol=1e-12)
            ties += int(tie); wins += int(y[j]>y[i] and not tie)
        rows.append(dict(family=family,metric=metric,seed=seed,valid_levels=len(ok),total_levels=len(group),
                         constant_response=constant,spearman=float(spearmanr(x,y).statistic) if usable else np.nan,
                         kendall=float(kendalltau(x,y).statistic) if usable else np.nan,
                         pairwise_ranking_accuracy=(wins+.5*ties)/pairs if pairs else np.nan,
                         strict_order_fraction=wins/pairs if pairs else np.nan,
                         tied_pair_fraction=ties/pairs if pairs else np.nan,
                         response_range=float(np.ptp(y)) if len(y) else np.nan))
    return pd.DataFrame(rows)


def correlation_by_family(table, family):
    g = table[(table.family==family)&(table.status=='ok')]
    wide = g.pivot(index=['seed','severity'],columns='metric',values='discrepancy')
    # Pairwise complete samples; provide paired sample counts alongside correlation.
    valid = wide.notna().astype(int)
    return wide.corr(method='spearman', min_periods=3), valid.T @ valid
