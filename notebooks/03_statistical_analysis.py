"""Statistical Analysis"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.data_utils import load_ab_data, add_derived_features
from src.stats_utils import (required_sample_size, run_proportion_ztest, compute_confidence_interval,
    compute_lift_ci, cohens_h, run_mannwhitney, bootstrap_mean_diff,)

# laod and setup
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

df= load_ab_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'ab_test_data.csv'))
df= add_derived_features(df)

ctrl= df[df.group == 'control']
treat= df[df.group == 'treatment']

# Power analysis
print("Pre-Experiment Power Analysis")

baseline_cvr = ctrl.converted.mean()

for mde in [0.003, 0.004, 0.005]:
    n = required_sample_size(0.032, mde, alpha=0.05, power=0.80)
    print(f"  Detect +{mde:.1%} lift: need {n:,} per group (~{n/1750:.0f} days at 1,750/group/day)")

print(f"\n  Actual per-group sample: {min(len(ctrl), len(treat)):,}")
print(f"  With 80% power we can detect >= 0.4 pp lift")


# SRM Check
print("\n Sample Ratio Mismatch Check")
n_ctrl_total = len(ctrl)
n_treat_total = len(treat)
chi2_srm, p_srm = stats.chisquare([n_ctrl_total, n_treat_total])
print(f"Control: {n_ctrl_total:,}  Treatment: {n_treat_total:,}")
print(f"Chi-squared: {chi2_srm:.4f}  p-value: {p_srm:.4f}")

if p_srm < 0.001:
    print("Hard Stop, Severe SRM (p < 0.001). Do not trust results until investigated.")
elif p_srm < 0.05:
    print("Warning, SRM below 0.05 threshold. Verify randomization pipeline.")
else:
    print("No SRM detected (p > 0.05)")


# Conversion Rate Analysis
print("\n Primary Metric: Conversion Rate")
stats_df = df.groupby('group')['converted'].agg(['count', 'sum', 'mean'])
n_ctrl, n_treat = stats_df.loc['control', 'count'], stats_df.loc['treatment', 'count']
conv_ctrl, conv_treat = stats_df.loc['control', 'sum'], stats_df.loc['treatment', 'sum']
cvr_ctrl, cvr_treat = stats_df.loc['control', 'mean'], stats_df.loc['treatment', 'mean']

print(f"Control: {cvr_ctrl:.4f}  ({conv_ctrl}/{n_ctrl})")
print(f"Treatment: {cvr_treat:.4f}  ({conv_treat}/{n_treat})")
print(f"Absolute lift: {(cvr_treat - cvr_ctrl)*100:.2f} pp")
print(f"Relative lift: {(cvr_treat - cvr_ctrl) / cvr_ctrl * 100:.2f}%")

# z-test and p-value
z_stat, p_value = run_proportion_ztest([conv_treat, conv_ctrl], [n_treat, n_ctrl])
print(f"\n z-statistic: {z_stat:.2f}  p-value: {p_value:.4f}")

# confidence int per group
ci_ctrl = compute_confidence_interval(conv_ctrl, n_ctrl)
ci_treat = compute_confidence_interval(conv_treat, n_treat)
print(f"Control 95% CI: [{ci_ctrl[0]:.4f}, {ci_ctrl[1]:.4f}]")
print(f"Treatment 95% CI: [{ci_treat[0]:.4f}, {ci_treat[1]:.4f}]")

diff, diff_lo, diff_hi = compute_lift_ci(cvr_ctrl, cvr_treat, n_ctrl, n_treat)
print(f"Lift 95% CI: [{diff_lo*100:.2f} pp, {diff_hi*100:.2f} pp]")

h = cohens_h(cvr_treat, cvr_ctrl)
print(f"Cohen's h: {h:.4f} (small = 0.2, medium = 0.5)")

print(f"\n Result:{'Reject' if p_value < 0.05 else 'Fail to reject'} H0 at a=0.05")

# Design effect note
avg_sessions = len(df) / df['user_id'].nunique()
design_effect = 1 + (avg_sessions - 1) * 0.03
effective_n = n_ctrl / design_effect
print(f"\n {avg_sessions:.1f} sessions/user - design effect {design_effect:.2f}")
print(f" Effective N per group: {effective_n:,.0f} vs {n_ctrl:,} observed")


# Revenue per Session analysis
print("\n Secondary Metric: Revenue per Session")

print(f"Control mean: ${ctrl.revenue.mean():.4f}")
print(f"Treatment mean: ${treat.revenue.mean():.4f}")

mw_stat, mw_p = run_mannwhitney(ctrl.revenue, treat.revenue)
print(f"Mann Whitney p-value: {mw_p:.4f}")

boot_diff, boot_lo, boot_hi = bootstrap_mean_diff(ctrl.revenue.values, treat.revenue.values, n_boot=10000)
print(f"Bootstrap mean diff: ${boot_diff:.4f}")
print(f"Bootstrap 95% CI: [${boot_lo:.4f}, ${boot_hi:.4f}]")


# AOV analysis
print("\n Guardrail Metric: Average Order Value")
aov_ctrl= ctrl.loc[ctrl.converted == 1, 'revenue']
aov_treat= treat.loc[treat.converted == 1, 'revenue']

print(f"Control AOV: ${aov_ctrl.mean():.2f}  (n={len(aov_ctrl)})")
print(f"Treatment AOV: ${aov_treat.mean():.2f}  (n={len(aov_treat)})")

t_stat, t_p = stats.ttest_ind(aov_ctrl, aov_treat, equal_var=False)
print(f"Welch's t-test p-value: {t_p:.4f}")
print(f"AOV difference is {'significant' if t_p < 0.05 else 'Not significant'} at a=0.05")


# Robustness
print("\n Robustness Checks")

# Bonferroni correction
bonferroni_alpha = 0.05 / 3
print(f"\n Bonferroni adjusted a: {bonferroni_alpha:.4f}")
print(f" Primary CVR p={p_value:.4f} - {'significant' if p_value < bonferroni_alpha else 'Not significant'} after correction")

# Consistency by device
print("\n Conversion by device:")
dev_stats = df.groupby(['device', 'group'])['converted'].mean().unstack()
for dev, row in dev_stats.iterrows():
    lift = row['treatment'] - row['control']
    print(f"{dev:>8}: control={row['control']:.4f}  treat={row['treatment']:.4f}  lift={lift*100:+.2f} pp")

# Time check
print("\n Temporal check:")
midpoint = df.date.unique()[len(df.date.unique()) // 2]
for label, sub in [('First half', df[df.date <= midpoint]), ('Second half', df[df.date > midpoint])]:
    c, t = sub.groupby('group')['converted'].mean()[['control', 'treatment']]
    print(f"{label}: control={c:.4f}  treat={t:.4f}  lift={((t-c)*100):+.2f} pp")

# Permutation test
print("\n  Permutation test (10,000 iterations):")
observed_diff = cvr_treat - cvr_ctrl
pool = df.converted.values
rng = np.random.default_rng(42)
n_treat_int = int(n_treat)
perm_diffs = np.empty(10000)
for i in range(10000):
    shuffled = rng.permutation(pool)
    perm_diffs[i] = shuffled[:n_treat_int].mean() - shuffled[n_treat_int:].mean()
perm_p = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"Permutation p-value: {perm_p:.4f}")


# User Level Robustness Check
print("\n User-level robustness check:")
user_agg = df.groupby(['user_id', 'group']).agg(any_converted=('converted', 'max')).reset_index()

user_ctrl = user_agg[user_agg.group == 'control']
user_treat = user_agg[user_agg.group == 'treatment']
n_u_ctrl, conv_u_ctrl = len(user_ctrl), user_ctrl.any_converted.sum()
n_u_treat, conv_u_treat = len(user_treat), user_treat.any_converted.sum()
z_user, p_user = run_proportion_ztest([conv_u_treat, conv_u_ctrl], [n_u_treat, n_u_ctrl])
cvr_u_ctrl = conv_u_ctrl / n_u_ctrl
cvr_u_treat = conv_u_treat / n_u_treat

print(f"User-level CVR: control={cvr_u_ctrl:.4f}  treatment={cvr_u_treat:.4f}")
print(f"z={z_user:.4f}   p={p_user:.4f}")
print(f"{'Consistent' if p_user < 0.05 else 'Weaker'} with session-level result (session p={p_value:.4f})")


# Rolling Lift
print("\n Rolling lift analysis:")
daily_cvr = df.groupby(['date', 'group'])['converted'].mean().unstack().sort_index()
daily_cvr['lift'] = daily_cvr['treatment'] - daily_cvr['control']
daily_cvr['rolling_lift'] = daily_cvr['lift'].rolling(3, min_periods=1).mean()

print(f" 3-day rolling lift — min: {daily_cvr['rolling_lift'].min()*100:.2f} pp, "
      f"max: {daily_cvr['rolling_lift'].max()*100:.2f} pp, "
      f"std: {daily_cvr['rolling_lift'].std()*100:.2f} pp")



# Visualisations
print("\n Visualisations")


plt.figure(figsize=(10, 5))
plt.plot(daily_cvr.index, daily_cvr['rolling_lift'] * 100, color='#E07B54', marker='o', lw=2)
plt.ylabel('3-Day Rolling Lift (pp)')
plt.xlabel('Date')
plt.title('Rolling Treatment Lift Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'rolling_lift.png'), dpi=150, bbox_inches='tight')
plt.close()
print(" Saved rolling_lift.png")


diff_pct, lo_pct, hi_pct = diff * 100, diff_lo * 100, diff_hi * 100

plt.figure(figsize=(8, 4))
plt.errorbar([diff_pct], ['CVR (pp)'], xerr=[[diff_pct - lo_pct], [hi_pct - diff_pct]],
             fmt='o', color='#E07B54', capsize=6, ms=8, lw=2)
plt.xlabel('Treatment Effect (percentage points)')
plt.title('Primary Metric — Treatment Lift with 95% CI')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'lift_ci_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved lift_ci_plot.png")


# Cumulative CR
daily = df.groupby(['date', 'group']).agg(n=('user_id', 'count'), conv=('converted', 'sum')).reset_index().sort_values('date')
daily[['cum_n', 'cum_conv']] = daily.groupby('group')[['n', 'conv']].cumsum()
daily['cum_cvr'] = daily['cum_conv'] / daily['cum_n']

plt.figure(figsize=(10, 5))
sns.lineplot(data=daily, x='date', y=daily['cum_cvr'] * 100, hue='group', palette={'control': '#5B8DB8', 'treatment': '#E07B54'}, marker='o')
plt.ylabel('Cumulative CVR (%)')
plt.title('Cumulative Conversion Rate Over Experiment Duration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'cumulative_cvr.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved cumulative_cvr.png")