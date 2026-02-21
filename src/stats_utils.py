"""Reusable statistical functions."""

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower


def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    effect_size = cohens_h(baseline_rate + mde, baseline_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                             ratio=1, alternative='two-sided')
    return int(np.ceil(n))


def run_proportion_ztest(counts, nobs):
    z, p = proportions_ztest(counts, nobs, alternative='two-sided')
    return z, p


def compute_confidence_interval(successes, n, alpha=0.05):
    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


def compute_lift_ci(p_ctrl, p_treat, n_ctrl, n_treat, alpha=0.05):
    diff = p_treat - p_ctrl
    se = np.sqrt(p_ctrl * (1 - p_ctrl) / n_ctrl + p_treat * (1 - p_treat) / n_treat)
    z = stats.norm.ppf(1 - alpha / 2)
    return diff, diff - z * se, diff + z * se


def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def run_mannwhitney(x, y):
    stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
    return stat, p


def bootstrap_mean_diff(x, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs[i] = by.mean() - bx.mean()
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
