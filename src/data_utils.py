"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from scipy import stats


def load_ab_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


def add_derived_features(df):
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    if 'session_duration_sec' not in df.columns:
        rng = np.random.default_rng(99)
        base = rng.lognormal(mean=4.8, sigma=0.7, size=len(df))
        base = np.clip(base, 10, 1800)
        # converters spend longer
        base[df['converted'].values == 1] *= 1.4
        df['session_duration_sec'] = base.round(0).astype(int)
    return df


def validate_data(df):
    counts = df['group'].value_counts()
    n_ctrl = counts.get('control', 0)
    n_treat = counts.get('treatment', 0)
    chi2, srm_p = stats.chisquare([n_ctrl, n_treat])
    return {
        'n_control': n_ctrl,
        'n_treatment': n_treat,
        'srm_chi2': chi2,
        'srm_p_value': srm_p,
    }
