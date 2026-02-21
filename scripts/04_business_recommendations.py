"""Business Recommendations & Impact Sizing / No-Go Decision"""

import sys, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_utils import load_ab_data, add_derived_features
from src.stats_utils import compute_lift_ci, run_proportion_ztest

# config
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ab_test_data.csv')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

IMPLEMENTATION_COST = 25_000
ANNUAL_DECAY_RATE = 0.15
SIGNIFICANCE_THRESHOLD = 0.05

# Load data
df = load_ab_data(DATA_PATH)
df = add_derived_features(df)
control_group = df[df.group == 'control']
treatment_group = df[df.group == 'treatment']

# Metrics
baseline_cvr = control_group.converted.mean()
new_cvr = treatment_group.converted.mean()
observed_lift = new_cvr - baseline_cvr
n_ctrl = len(control_group)
n_treat = len(treatment_group)

# Statistical significance check
conv_ctrl = control_group.converted.sum()
conv_treat = treatment_group.converted.sum()
z_stat, p_value = run_proportion_ztest([conv_treat, conv_ctrl], [n_treat, n_ctrl])

# Dynamic CI computation
diff, ci_lo, ci_hi = compute_lift_ci(baseline_cvr, new_cvr, n_ctrl, n_treat)

# Determine if result is actionable
is_significant = p_value < SIGNIFICANCE_THRESHOLD
ci_crosses_zero = ci_lo <= 0

average_order_value = df.loc[df.converted == 1, 'revenue'].mean()
daily_sessions = len(df) / df.timestamp.dt.date.nunique()
annual_sessions = daily_sessions * 365


# Significance Gate 
print(" Statistical Significance")
print("\n" + "═" * 65)
print(f"\n Observed lift: {observed_lift*100:+.2f} pp ({observed_lift/baseline_cvr*100:+.1f}% relative)")
print(f"p-value: {p_value:.4f}")
print(f"95% CI: [{ci_lo*100:.2f} pp, {ci_hi*100:.2f} pp]")
print(f"CI crosses zero: {'Yes' if ci_crosses_zero else 'No'}")

if not is_significant:
    print(f"\nFailed: p = {p_value:.4f} > {SIGNIFICANCE_THRESHOLD}")
    print("The observed lift is not statistically significant.")
    print("We fail to reject the null hypothesis.")
    print("The true treatment effect is indistinguishable from zero.")
else:
    print(f"\n assed: p = {p_value:.4f} < {SIGNIFICANCE_THRESHOLD}")

# Actionable lift: if not significant, the business actionable lift is $0
if is_significant and not ci_crosses_zero:
    actionable_lift = observed_lift
    actionable_label = "Significant"
else:
    actionable_lift = 0.0
    actionable_label = "Not significant — treated as zero"

print(f"\nActionable lift for projections: {actionable_lift*100:.2f} pp ({actionable_label})")


# Novelty Effect Analysis
print("\n" + "═" * 65)
print(" Novelty Effect Analysis")

df['week_num'] = ((pd.to_datetime(df['date']) - pd.to_datetime(df['date']).min()).dt.days // 7) + 1

print(f"\n {'Week':<12} {'Ctrl CVR':>12} {'Treat CVR':>12} {'Lift (pp)':>12} {'Signal':>16}")

weekly_lifts = {}
for week in sorted(df['week_num'].unique()):
    week_data = df[df.week_num == week]
    wc = week_data[week_data.group == 'control']['converted'].mean()
    wt = week_data[week_data.group == 'treatment']['converted'].mean()
    lift = wt - wc
    weekly_lifts[week] = lift

    if lift > 0.002:
        signal = "Strong positive"
    elif lift > 0:
        signal = "Weak positive"
    elif lift > -0.001:
        signal = "Flat"
    else:
        signal = "Negative"

    print(f"Week {week:<5} {wc*100:>11.2f}% {wt*100:>11.2f}% {lift*100:>+11.2f} {signal:>16}")

weeks = sorted(weekly_lifts.keys())
first_week_lift = weekly_lifts[weeks[0]]
last_week_lift = weekly_lifts[weeks[-1]]

print(f"\n Week 1 lift: {first_week_lift*100:+.2f} pp")
print(f"Week {weeks[-1]} lift: {last_week_lift*100:+.2f} pp")

if last_week_lift < first_week_lift * 0.3:
    print("Severe novelty decay: treatment lift has largely or completely faded.")
    print("The early positive signal was driven by novelty, not sustained improvement.")
    print("Had we stopped the test after Week 1, we would have shipped a zero value feature.")
    novelty_detected = True
else:
    print("Lift appears stable across weeks.")
    novelty_detected = False

# The Peeking Counterfactual
print("\n" + "═" * 65)
print("Peeking Counterfactual")

# Subset to Week 1 only (days 0-6)
w1 = df[df.week_num == 1]
w1_ctrl = w1[w1.group == 'control']
w1_treat = w1[w1.group == 'treatment']

n_w1_ctrl = len(w1_ctrl)
n_w1_treat = len(w1_treat)
conv_w1_ctrl = w1_ctrl.converted.sum()
conv_w1_treat = w1_treat.converted.sum()
cvr_w1_ctrl = w1_ctrl.converted.mean()
cvr_w1_treat = w1_treat.converted.mean()

# Week 1
z_w1, p_w1 = run_proportion_ztest([conv_w1_treat, conv_w1_ctrl], [n_w1_treat, n_w1_ctrl])
diff_w1, ci_lo_w1, ci_hi_w1 = compute_lift_ci(cvr_w1_ctrl, cvr_w1_treat, n_w1_ctrl, n_w1_treat)
w1_projected_rev = annual_sessions * diff_w1 * average_order_value
w1_would_ship = diff_w1 > 0 and p_w1 < 0.20

print(f"\n What if the test was stopped the test after Week 1?")
print(f"\n Week 1 data only:")
print(f"Control: {cvr_w1_ctrl:.4f}  ({conv_w1_ctrl}/{n_w1_ctrl})")
print(f"Treatment: {cvr_w1_treat:.4f}  ({conv_w1_treat}/{n_w1_treat})")
print(f"Lift:{diff_w1*100:+.2f} pp")
print(f"p-value: {p_w1:.4f}")
print(f"95% CI: [{ci_lo_w1*100:.2f}, {ci_hi_w1*100:.2f}] pp")

print(f"\n {'Metric':<28} {'Peeked at Week 1':>20} {'Full 21-Day Result':>20}")
print(f" {'Observed lift':<28} {diff_w1*100:>+18.2f} pp {diff*100:>+18.2f} pp")
print(f" {'p-value':<28} {p_w1:>20.4f} {p_value:>20.4f}")
print(f" {'95% CI':<28} {'[{:.2f}, {:.2f}] pp'.format(ci_lo_w1*100, ci_hi_w1*100):>20} {'[{:.2f}, {:.2f}] pp'.format(ci_lo*100, ci_hi*100):>20}")

if w1_would_ship:
    w1_rev_label = f"${w1_projected_rev:,.0f}"
    w1_ship_label = "Tempted to ship"
else:
    w1_rev_label = f"${w1_projected_rev:,.0f} (projected)"
    w1_ship_label = "Tempted to ship"

print(f"{'Projected annual revenue':<28} {w1_rev_label:>20} {'$0 (n.s.)':>20}")
print(f"{'Ship decision':<28} {w1_ship_label:>20} {'Do not ship':>20}")
print(f"{'Business outcome':<28} {'$25K spent, zero ROI':>20} {'$25K saved':>20}")

print(f"\n The Week 1 data showed a {diff_w1*100:+.2f} pp lift — nearly {diff_w1/diff:.1f}x the")
print(f"full-test estimate. A team that peeked would have projected")
print(f"${w1_projected_rev:,.0f} in annual revenue. The true sustained value was $0.")
print(f"The 21-day discipline saved ${IMPLEMENTATION_COST:,} in wasted implementation costs.")

# Peeking Counterfactual Visualization
y_positions = [1.0, 0.4]
labels = ['Peeked at Week 1\n(inflated by novelty)', 'Full 21-Day Test\n(true result)']
diffs_pp = [diff_w1 * 100, diff * 100]
ci_los_pp = [ci_lo_w1 * 100, ci_lo * 100]
ci_his_pp = [ci_hi_w1 * 100, ci_hi * 100]
colors = ['#D32F2F', '#4CAF50']

plt.figure(figsize=(10, 4))

for y, d, lo, hi, col, lab in zip(y_positions, diffs_pp, ci_los_pp, ci_his_pp, colors, labels):
    plt.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='o', color=col,
                capsize=8, ms=10, lw=2.5, capthick=2, ecolor=col)
    plt.text(hi + 0.04, y, f'{d:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]',
            va='center', fontsize=9, color=col, fontweight='bold')

plt.axvline(0, color='#333', ls='--', lw=1.2, alpha=0.7)
plt.text(0.02, 1.35, 'No Effect', fontsize=9, color='#666', ha='left')
plt.yticks(y_positions, labels, fontsize=10)
plt.xlabel('Treatment Effect (percentage points)', fontsize=10)
plt.title('The Peeking Counterfactual — Week 1 vs. Full Test', fontsize=12, fontweight='bold')
plt.ylim(-0.1, 1.6)
plt.gca().spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'peeking_counterfactual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved peeking_counterfactual.png")



# Financial Projections - Adjusted for Significance
print("\n" + "═" * 65)
print("Impact Projections on an Annual Basis")

print(f"\n Key Assumptions:")
print(f"• Baseline Annual Traffic: {annual_sessions:,.0f} sessions")
print(f"• Average Order Value: ${average_order_value:.2f}")
print(f"• Current Conversion Rate: {baseline_cvr:.2%}")
print(f"• Observed Lift: {observed_lift*100:+.2f} pp (p = {p_value:.4f})")
print(f"• 95% CI: [{ci_lo*100:.2f} pp, {ci_hi*100:.2f} pp]")
print(f"• Implementation Cost: ${IMPLEMENTATION_COST:,.0f}")

if not is_significant:
    print(f"\n Because p = {p_value:.4f} > {SIGNIFICANCE_THRESHOLD}, the true lift is")
    print(f"indistinguishable from zero. All revenue projections are set to $0.")
    print(f" Projecting revenue from a non-significant result would be misleading.\n")

LIFT_SCENARIOS = {
    'Conservative (Lower Bound)': ci_lo,
    'Expected (Point Estimate)':  observed_lift,
    'Optimistic (Upper Bound)':   ci_hi
}

print(f" {'Scenario':<30} {'Observed':>14} {'Actionable':>14} {'Projected Rev':>14}")

for label, lift_pp in LIFT_SCENARIOS.items():
    if is_significant and not ci_crosses_zero:
        projected_rev = annual_sessions * lift_pp * average_order_value * (1 -ANNUAL_DECAY_RATE/ 2)
    else:
        projected_rev = 0.0

    print(f"{label:<30} {lift_pp*100:>+12.2f} pp {'$0 (n.s.)':>14} ${projected_rev:>12,.0f}")

if is_significant and not ci_crosses_zero:
    expected_rev = annual_sessions * observed_lift * average_order_value * (1 - ANNUAL_DECAY_RATE / 2)
    print(f"\n Expected Annual Uplift: ${expected_rev:>12,.0f}")
else:
    print(f"\n Expected Annual Uplift: $0  (result not statistically significant)")


# Break-Even & Payback
print("\n" + "═" * 65)
print(" Break Even Analysis")
print(f"  Implementation cost: ${IMPLEMENTATION_COST:,.0f}\n")

if not is_significant:
    print(f"The test did not reach statistical significance (p = {p_value:.4f}).")
    print(f"The true sustained lift is effectively zero.")
    print(f"Daily uplift: $0.00")
    print(f"Payback period: ∞ (never)")
    print(f"\nThe ${IMPLEMENTATION_COST:,.0f} investment would not be recovered.")
    print(f"Shipping this feature would be a pure cost with no expected return.")
else:
    print(f"{'Scenario':<30} {'Daily Uplift':>16} {'Payback (days)':>16}")
    for label, lift_pp in LIFT_SCENARIOS.items():
        daily_uplift = daily_sessions * lift_pp * average_order_value
        if daily_uplift > 0:
            payback_days = IMPLEMENTATION_COST / daily_uplift
            print(f"{label:<30} ${daily_uplift:>12,.2f} {payback_days:>15.0f}")
        else:
            print(f"{label:<30} ${daily_uplift:>12,.2f} {'N/A':>15}")


# Segment-Level Risk Table 
print("\n" + "═" * 65)
print("Segmnt Analysis")
total_revenue = df['revenue'].sum()

print(f"\n {'Device':<12} {'N ctrl':>12} {'N treat':>12} {'CVR ctrl':>12} {'CVR treat':>12} {'Lift (pp)':>12} {'Status':>12}")

for dev in ['desktop', 'mobile', 'tablet']:
    d = df[df.device == dev]
    dc = d[d.group == 'control']
    dt = d[d.group == 'treatment']
    cvr_c = dc.converted.mean()
    cvr_t = dt.converted.mean()
    lift = cvr_t - cvr_c
    status = "Positive" if lift > 0 else "Negative"
    print(f"  {dev:<12} {len(dc):>12,} {len(dt):>12,} {cvr_c:>12.2%} {cvr_t:>12.2%} {lift*100:>+12.2f} {status}")

print(f"\n Note: Segment-level results are exploratory only.")
print(f"The overall test is non-significant, so segment lifts should not")
print(f"be interpreted as reliable treatment effects.")


#Risk Assessment
print("\n" + "═" * 65)
print(" Risk Assesment")

risks = [
    ("Statistical Significance", f"FAILED — p = {p_value:.4f}, cannot reject H₀"),
    ("Novelty Effect", "SEVERE — Week 1 lift faded to ~zero by Week 3"),
    ("Temporal Stability", "UNSTABLE — lift decayed consistently over 21 days"),
    ("Average Order Value", "No negative impact detected (guardrail passed, but moot)"),
    ("Implementation Cost", f"${IMPLEMENTATION_COST:,.0f} at risk with zero expected return"),
]

for area, note in risks:
    print(f" • {area:<25}: {note}")


# Recommendation ─
print("\n" + "═" * 65)
print(" Recommendation")

print("""
    DECISION: Do NOT ship the new single-page checkout          

  Main Reasons:
    - Primary KPI is not statistically significant (p = {p:.4f})
    - 95% CI for the lift crosses zero, the true effect may be negative
    - Novelty analysis reveals the early positive signal was transient
    - By Week 3, treatment performance had regressed to baseline
    - Shipping would cost ${cost:,} with zero expected sustained ROI

      
  Strategic Impact: 
    By running the test for the full 21-day duration and resisting the
    temptation to peek at Week 1 results, the data team protected the
    business from:
    - Wasting ${cost:,} in implementation costs
    - Adding unnecessary technical debt to the codebase
    - Drawing incorrect conclusions that would pollute future decisions
    - Opportunity cost of engineering time spent on a zero-value feature

      
  Lessons Learned: 
    - Users responded positively to the new layout initially (novelty)
    - The improvement did not persist, this is a UX novelty effect
    - The current multi-step checkout is not the conversion bottleneck we assumed it was
    - Future optimization should target validated friction points

      
   Next Steps:
    1. Archive: Store experiment code & data.
       [Data Science | This Week]

    2. Document: Log findings in the knowledge base.
       [Data Science | This Week]

    3. Research: Conduct UX interviews to find real friction points.
       [UX/Design | Weeks 1-3]

    4. Optimize: Explore guest checkout, autofill, & payment options.
       [Product | Weeks 2-4]

    5. F¡fUTURE PROOF: Plan 4-6 week windows for future redesign tests.
       [Data Science | As Needed]
""".format(p=p_value, cost=IMPLEMENTATION_COST))


# Visualisation
print("\n" + "═" * 65)
print(" Generating charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

# Left panel: Revenue projection showing $0 actionable value
ax1 = axes[0]

scenario_labels = list(LIFT_SCENARIOS.keys())
observed_revenues = [annual_sessions * lift * average_order_value for lift in LIFT_SCENARIOS.values()]
actionable_revenues = [0.0] * len(scenario_labels)  # All zero since non-significant

y_pos = range(len(scenario_labels))
bar_height = 0.35

# observed
bars_observed = ax1.barh(
    [y - bar_height/2 for y in y_pos], observed_revenues, height=bar_height,
    color=['#B0B0B0', '#B0B0B0', '#B0B0B0'], alpha=0.4, label='Naive projection (not actionable)')

# Actionable 
bars_actionable = ax1.barh(
    [y + bar_height/2 for y in y_pos], actionable_revenues, height=bar_height,
    color=['#E07B54', '#E07B54', '#E07B54'], label='Actionable projection ($0, n.s.)')

for i, (obs, act) in enumerate(zip(observed_revenues, actionable_revenues)):
    if obs != 0:
        ax1.text(obs + max(abs(v) for v in observed_revenues) * 0.02, i - bar_height/2,
                 f'${obs:,.0f}', va='center', fontsize=8, color='#888888', style='italic')
    ax1.text(max(abs(v) for v in observed_revenues) * 0.02, i + bar_height/2,
             '$0 (n.s.)', va='center', fontsize=9, color='#E07B54', fontweight='bold')

# Implementation cost line
ax1.axvline(IMPLEMENTATION_COST, color='red', ls='--', lw=1.5, alpha=0.7)
ax1.text(IMPLEMENTATION_COST + max(abs(v) for v in observed_revenues) * 0.01, len(scenario_labels) - 0.3,
         f'Implementation\ncost: ${IMPLEMENTATION_COST:,.0f}\n(never recovered)',
         fontsize=8, color='red', va='top')

ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(scenario_labels)
ax1.set_xlabel('Projected Annual Revenue Uplift (USD)')
ax1.set_title('Checkout Redesign — No Actionable Revenue Impact')
ax1.legend(loc='lower right', fontsize=8)

# Right panel: Weekly lift decay
ax2 = axes[1]

weeks_sorted = sorted(weekly_lifts.keys())
lifts_by_week = [weekly_lifts[w] * 100 for w in weeks_sorted]
colors = ['#4CAF50' if l > 0.1 else '#FF9800' if l > 0 else '#F44336' for l in lifts_by_week]

bars = ax2.bar([f'Week {w}' for w in weeks_sorted], lifts_by_week, color=colors, edgecolor='white', linewidth=0.5)
ax2.axhline(0, color='black', lw=0.8, ls='-')
ax2.set_ylabel('Treatment Lift (pp)')
ax2.set_title('Novelty Decay — Lift Fades to Zero')

for bar, val in zip(bars, lifts_by_week):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:+.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.annotate('Novelty spike\n(would have shipped\nif we peeked here)',
             xy=(0, lifts_by_week[0]), xytext=(0.5, lifts_by_week[0] * 0.6),
             fontsize=7, ha='center', color='#666',
             arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'revenue_impact.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved chart to assets/revenue_impact.png")
print("\n" + "═" * 65)
print(" Analysis complete. Recommendation: DO NOT SHIP.")
print("═" * 65)
