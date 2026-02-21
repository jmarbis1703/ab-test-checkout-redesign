"""Exploratory Data Analysis"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_ab_data, validate_data, add_derived_features

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(FIGDIR, exist_ok=True)

# Load
df = load_ab_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'ab_test_data.csv'))
df = add_derived_features(df)
checks = validate_data(df)

# Group Balance
print("Group Split")
print(f"  {df['group'].value_counts().to_dict()}")
if checks['srm_p_value'] < 0.05:
    print(f"Potential SRM (p={checks['srm_p_value']:.4f}), below 0.05 threshold")
else:
    print(f"Groups are balanced (SRM p={checks['srm_p_value']:.3f})")

# Daily traffic and conversion
print("\n Daily Traffic & Conversions")
daily = (df.groupby(['date', 'group']).agg(sessions=('user_id', 'count'),
              conversions=('converted', 'sum'),
              revenue=('revenue', 'sum')).reset_index())

daily['cvr'] = daily['conversions'] / daily['sessions']


custom_palette = {'control': '#5B8DB8', 'treatment': '#E07B54'}

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

sns.lineplot(data=daily, x='date', y='sessions', hue='group',
    palette=custom_palette, marker='o')
plt.title('Daily Traffic by Group')
plt.ylabel('Daily Sessions')
plt.legend(loc='upper right')
plt.xticks([])

sns.lineplot(data=daily, x='date', y=daily['cvr'] * 100, hue='group',
    palette=custom_palette, marker='o')
plt.title('Daily Conversion Rate by Group')
plt.ylabel('Conversion Rate (%)')
plt.xlabel('Date')
plt.legend(loc='upper right')
plt.xticks(rotation=45)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'daily_traffic_cvr.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved daily_traffic_cvr.png")


# Device breakdown
print("\nConversion by Device")
device_metrics = (df.groupby(['device', 'group'])['converted'].mean().reset_index())
device_metrics['cvr_percent'] = device_metrics['converted'] * 100

plt.figure(figsize=(8, 5))
sns.barplot(data=device_metrics, x='device', y='cvr_percent', hue='group')
plt.title('Conversion Rate by Device')
plt.ylabel('Conversion Rate (%)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'cvr_by_device.png'))
plt.close()
print("Saved cvr_by_device.png")

# Revenue distribution
print("\nRevenue Distribution")
converters = df[df.converted == 1]

plt.figure(figsize=(8, 5))
sns.histplot(data=converters, x='revenue', hue='group', element='step', bins=40, alpha=0.3)
plt.title('Revenue Distribution (Converters Only)')
plt.xlabel('Revenue (USD)')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'revenue_distribution.png'))
plt.close()
print("Saved revenue_distribution.png")


# Traffic source mix
print("\nTraffic Sources")
source_mix = pd.crosstab(df['traffic_source'], df['group'], normalize='columns') * 100
print(source_mix.round(1))


# Key metrics summary
print("\nKey Metrics")
summary = df.groupby('group').agg({
    'user_id': 'count',
    'converted': 'mean',
    'revenue': 'mean',
    'session_duration_sec': 'mean'
}).rename(columns={
    'user_id': 'Sessions',
    'converted': 'CVR',
    'revenue': 'Rev/Session',
    'session_duration_sec': 'Avg Duration (s)'
})
print(summary)


# Session Duration Comparison
print("\nSession Duration Comparison")

dur_stats = df.groupby('group')['session_duration_sec'].agg(['mean', 'median', 'std'])
print(f"{'Group':<12} {'Mean (s)':>12} {'Median (s)':>12} {'Std':>12}")

for grp, row in dur_stats.iterrows():
    print(f"  {grp:<12} {row['mean']:>12.1f} {row['median']:>12.1f} {row['std']:>12.1f}")

dur_diff = dur_stats.loc['treatment', 'mean'] - dur_stats.loc['control', 'mean']
print(f"\n Difference (treatment - control): {dur_diff:+.1f} seconds")
if abs(dur_diff) < 5:
    print("Negligible difference, new checkout doesn't slow users down")
elif dur_diff > 0:
    print("Treatment sessions are longer, worth investigating if that's friction or engagement")
else:
    print("Treatment sessions are shorter,  suggests a more efficient checkout flow")

# Duration for converters vs non-converters
print("\n  Duration by conversion status:")
dur_conv = df.groupby(['group', 'converted'])['session_duration_sec'].mean().unstack()
dur_conv.columns = ['Non-converters', 'Converters']
print(dur_conv.round(1))


# Revenue Quantile Breakdown
print("\n Revenue by Quantile")

rev_quantiles = converters.groupby('group')['revenue'].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
print(f"{'Group':<12} {'Median':>12} {'75th':>12} {'90th':>12} {'Mean':>12}")

for grp in ['control', 'treatment']:
    grp_conv = converters[converters.group == grp]['revenue']
    p50 = grp_conv.median()
    p75 = grp_conv.quantile(0.75)
    p90 = grp_conv.quantile(0.90)
    mn = grp_conv.mean()
    print(f"{grp:<12} ${p50:>10.2f} ${p75:>10.2f} ${p90:>10.2f} ${mn:>10.2f}")

print("\n If quantiles are similar across groups, the treatment is lifting volume without shifting the order value distribution, indicating a win.")


# Source against Device Interaction
print("\n Source x Device Interaction")

combo = df.groupby(['device', 'traffic_source', 'group']).agg(sessions=('converted', 'count'),
    conversions=('converted', 'sum')).reset_index()

combo['cvr'] = combo['conversions'] / combo['sessions']

combo_wide = combo.pivot_table(index=['device', 'traffic_source'],
                                columns='group', values='cvr').reset_index()

combo_wide['lift_pp'] = (combo_wide['treatment'] - combo_wide['control']) * 100

combo_n = combo.pivot_table(index=['device', 'traffic_source'],
                             columns='group', values='sessions',
                             aggfunc='sum').reset_index()

print(f"  {'Device':<12} {'Source':<14} {'Ctrl CVR':>12} {'Treat CVR':>12} {'Lift (pp)':>12} {'Flag':>6}")

neg_count = 0
for _, row in combo_wide.sort_values('lift_pp').iterrows():
    flag = "!!!" if row['lift_pp'] < 0 else ""
    if row['lift_pp'] < 0:
        neg_count += 1
    print(f"  {row['device']:<12} {row['traffic_source']:<14} {row['control']*100:>11.2f}% {row['treatment']*100:>11.2f}% {row['lift_pp']:>+11.2f} {flag}")

print(f"\n  {neg_count} combinations show negative lift")
if neg_count > 0:
    print("  Negative lift combos are worth checking, but small samples can cause noise")

plt.figure(figsize=(10, 6))
pivot_lift = combo_wide.pivot(index='traffic_source', columns='device', values='lift_pp')

sns.heatmap(pivot_lift, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('Lift (pp) by Device × Traffic Source')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'lift_device_source_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved lift_device_source_heatmap.png")


# Outlier Check on Revenue
print("\n Outlier Check on Revenue")

for grp in ['control', 'treatment']:
    grp_rev = converters[converters.group == grp]['revenue']
    mu, sigma = grp_rev.mean(), grp_rev.std()
    threshold = mu + 3 * sigma
    n_outliers = (grp_rev > threshold).sum()
    mean_with = grp_rev.mean()
    mean_without = grp_rev[grp_rev <= threshold].mean()
    print(f"{grp}: {n_outliers} outliers (>{threshold:.0f})")
    print(f"Mean with outliers: ${mean_with:.2f}")
    print(f"Mean without outliers: ${mean_without:.2f}")
    print(f"Difference: ${mean_with - mean_without:+.2f}")

print("\n If the outlier impact is small (<$1 difference), AOV results are robust.")


# Returning Users Behavior
print("\n Returning Users Behavior")

user_sessions = df.groupby(['user_id', 'group']).agg(n_sessions=('converted', 'count'),
    any_conversion=('converted', 'max')).reset_index()

user_sessions['user_type'] = np.where(user_sessions['n_sessions'] == 1, 'single_session', 'returning')

# Count distribution
user_type_counts = user_sessions.groupby(['group', 'user_type']).size().unstack(fill_value=0)
print("User counts by type:")
print(user_type_counts.to_string())

# CVR by user type and group
user_cvr = user_sessions.groupby(['group', 'user_type'])['any_conversion'].mean()
print(f"\n User level CVR by type:")
print(f"{'Group':<12} {'Single-session':>16} {'Returning':>12}")

for grp in ['control', 'treatment']:
    single = user_cvr.get((grp, 'single_session'), 0)
    ret = user_cvr.get((grp, 'returning'), 0)
    print(f"{grp:<12} {single:>15.2%} {ret:>11.2%}")

# Lift by user type
for utype in ['single_session', 'returning']:
    c = user_cvr.get(('control', utype), 0)
    t = user_cvr.get(('treatment', utype), 0)
    print(f"\n Lift for {utype} users: {(t-c)*100:+.2f} pp")

print("\n If lift is similar for both types, the effect is general.")
print("If only single-session users benefit, it's a UX clarity effect.")
print("If only returning users benefit, it's a familiarity/learning effect.")


# Conversion by Day-of-Week
print("\nConversion by Day of Week")

dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_cvr = df.groupby(['day_of_week', 'group'])['converted'].mean().unstack()
dow_cvr = dow_cvr.reindex(dow_order)
dow_cvr['lift_pp'] = (dow_cvr['treatment'] - dow_cvr['control']) * 100

print(f"  {'Day':<12} {'Ctrl CVR':>12} {'Treat CVR':>12} {'Lift (pp)':>12}")

for day, row in dow_cvr.iterrows():
    print(f"  {day:<12} {row['control']*100:>11.2f}% {row['treatment']*100:>11.2f}% {row['lift_pp']:>+11.2f}")

lift_range = dow_cvr['lift_pp'].max() - dow_cvr['lift_pp'].min()
print(f"\n  Lift range across days: {lift_range:.2f} pp")
if lift_range < 0.5:
    print("Lift is consistent across days of the week, no day-specific effects")
else:
    print("Lift varies by day, worth investigating if weekday vs weekend patterns differ")

plt.figure(figsize=(10, 5))
x = range(len(dow_order))
ctrl_vals = [dow_cvr.loc[d, 'control'] * 100 for d in dow_order]
treat_vals = [dow_cvr.loc[d, 'treatment'] * 100 for d in dow_order]
plt.bar([i - 0.15 for i in x], ctrl_vals, width=0.3, color='#5B8DB8', label='control')
plt.bar([i + 0.15 for i in x], treat_vals, width=0.3, color='#E07B54', label='treatment')
plt.xticks(x, dow_order, rotation=45)
plt.ylabel('Conversion Rate (%)')
plt.title('CVR by Day of Week')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'cvr_by_dow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved cvr_by_dow.png")

# Novelty Check
print("\nNovelty Analysis")

df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week.astype(int)
start_week, end_week = df['week'].min(), df['week'].max()
print(f"  Comparing Week {start_week} vs. Week {end_week}")

novelty_stats = (df[df['week'].isin([start_week, end_week])].groupby(['week', 'group'])['converted'].mean().unstack())
novelty_stats['lift'] = novelty_stats['treatment'] - novelty_stats['control']
print(novelty_stats.apply(lambda x: x.map("{:.2%}".format)))

early_lift = novelty_stats.loc[start_week, 'lift']
late_lift = novelty_stats.loc[end_week, 'lift']
if late_lift >= early_lift * 0.8:
    print("  Lift persists, no strong novelty decay")
else:
    print("  Lift faded substantially, possible novelty effect")


print("\nEDA Complete. Plots saved to 'assets/' folder.")