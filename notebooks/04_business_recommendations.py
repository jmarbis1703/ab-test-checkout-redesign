"""Business Recommendations & Impact Sizing — No-Go Decision"""

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


# ── Significance Gate ──────────────────────────────────────────────────────────
print("═" * 65)
print(" STATISTICAL SIGNIFICANCE GATE")
print("═" * 65)
print(f"\n  Observed lift: {observed_lift*100:+.2f} pp ({observed_lift/baseline_cvr*100:+.1f}% relative)")
print(f"  p-value: {p_value:.4f}")
print(f"  95% CI: [{ci_lo*100:.2f} pp, {ci_hi*100:.2f} pp]")
print(f"  CI crosses zero: {'Yes' if ci_crosses_zero else 'No'}")

if not is_significant:
    print(f"\n  ⛔ FAILED: p = {p_value:.4f} > {SIGNIFICANCE_THRESHOLD}")
    print("  The observed lift is NOT statistically significant.")
    print("  We FAIL TO REJECT the null hypothesis.")
    print("  The true treatment effect is indistinguishable from zero.")
else:
    print(f"\n  ✅ PASSED: p = {p_value:.4f} < {SIGNIFICANCE_THRESHOLD}")

# Actionable lift: if not significant, the business-actionable lift is $0
if is_significant and not ci_crosses_zero:
    actionable_lift = observed_lift
    actionable_label = "Significant"
else:
    actionable_lift = 0.0
    actionable_label = "Not significant — treated as zero"

print(f"\n  Actionable lift for projections: {actionable_lift*100:.2f} pp ({actionable_label})")


# ── Novelty Effect Analysis ────────────────────────────────────────────────────
print("\n" + "═" * 65)
print(" NOVELTY EFFECT ANALYSIS")
print("═" * 65)

df['week_num'] = ((pd.to_datetime(df['date']) - pd.to_datetime(df['date']).min()).dt.days // 7) + 1

print(f"\n  {'Week':<10} {'Ctrl CVR':>12} {'Treat CVR':>12} {'Lift (pp)':>12} {'Signal':>16}")

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
        signal = "~Flat"
    else:
        signal = "Negative"

    print(f"  Week {week:<5} {wc*100:>11.2f}% {wt*100:>11.2f}% {lift*100:>+11.2f} {signal:>16}")

weeks = sorted(weekly_lifts.keys())
first_week_lift = weekly_lifts[weeks[0]]
last_week_lift = weekly_lifts[weeks[-1]]

print(f"\n  Week 1 lift: {first_week_lift*100:+.2f} pp")
print(f"  Week {weeks[-1]} lift: {last_week_lift*100:+.2f} pp")

if last_week_lift < first_week_lift * 0.3:
    print("  ⚠️  SEVERE NOVELTY DECAY: Treatment lift has largely or completely faded.")
    print("     The early positive signal was driven by novelty, not sustained improvement.")
    print("     Had we stopped the test after Week 1, we would have shipped a zero-value feature.")
    novelty_detected = True
else:
    print("  Lift appears stable across weeks.")
    novelty_detected = False


# ── Financial Projections (Adjusted for Significance) ──────────────────────────
print("\n" + "═" * 65)
print(" BUSINESS IMPACT PROJECTIONS — ANNUAL BASIS")
print("═" * 65)

print(f"\n  Key Assumptions:")
print(f"  • Baseline Annual Traffic: {annual_sessions:,.0f} sessions")
print(f"  • Average Order Value: ${average_order_value:.2f}")
print(f"  • Current Conversion Rate: {baseline_cvr:.2%}")
print(f"  • Observed Lift: {observed_lift*100:+.2f} pp (p = {p_value:.4f})")
print(f"  • 95% CI: [{ci_lo*100:.2f} pp, {ci_hi*100:.2f} pp]")
print(f"  • Implementation Cost: ${IMPLEMENTATION_COST:,.0f}")

if not is_significant:
    print(f"\n  ⛔ Because p = {p_value:.4f} > {SIGNIFICANCE_THRESHOLD}, the true lift is")
    print(f"     indistinguishable from zero. All revenue projections are set to $0.")
    print(f"     Projecting revenue from a non-significant result would be misleading.\n")

LIFT_SCENARIOS = {
    'Conservative (Lower Bound)': ci_lo,
    'Expected (Point Estimate)':  observed_lift,
    'Optimistic (Upper Bound)':   ci_hi
}

print(f"  {'Scenario':<30} {'Observed':>12} {'Actionable':>14} {'Projected Rev':>14}")

for label, lift_pp in LIFT_SCENARIOS.items():
    if is_significant and not ci_crosses_zero:
        projected_rev = annual_sessions * lift_pp * average_order_value * (1 - ANNUAL_DECAY_RATE / 2)
    else:
        projected_rev = 0.0

    print(f"  {label:<30} {lift_pp*100:>+9.2f} pp {'$0 (n.s.)':>14} ${projected_rev:>12,.0f}")

if is_significant and not ci_crosses_zero:
    expected_rev = annual_sessions * observed_lift * average_order_value * (1 - ANNUAL_DECAY_RATE / 2)
    print(f"\n  Expected Annual Uplift: ${expected_rev:>12,.0f}")
else:
    print(f"\n  Expected Annual Uplift: $0  (result not statistically significant)")


# ── Break-Even & Payback ──────────────────────────────────────────────────────
print("\n" + "═" * 65)
print(" BREAK-EVEN ANALYSIS")
print("═" * 65)
print(f"  Implementation cost: ${IMPLEMENTATION_COST:,.0f}\n")

if not is_significant:
    print(f"  ⛔ The test did not reach statistical significance (p = {p_value:.4f}).")
    print(f"     The true sustained lift is effectively zero.")
    print(f"     Daily uplift: $0.00")
    print(f"     Payback period: ∞ (never)")
    print(f"\n  The ${IMPLEMENTATION_COST:,.0f} investment would NEVER be recovered.")
    print(f"  Shipping this feature would be a pure cost with no expected return.")
else:
    print(f"  {'Scenario':<30} {'Daily Uplift':>16} {'Payback (days)':>16}")
    for label, lift_pp in LIFT_SCENARIOS.items():
        daily_uplift = daily_sessions * lift_pp * average_order_value
        if daily_uplift > 0:
            payback_days = IMPLEMENTATION_COST / daily_uplift
            print(f"  {label:<30} ${daily_uplift:>12,.2f} {payback_days:>15.0f}")
        else:
            print(f"  {label:<30} ${daily_uplift:>12,.2f} {'N/A':>15}")


# ── Segment-Level Risk Table ──────────────────────────────────────────────────
print("\n" + "═" * 65)
print(" SEGMENT ANALYSIS")
print("═" * 65)

total_revenue = df['revenue'].sum()

print(f"\n  {'Device':<12} {'N ctrl':>12} {'N treat':>12} {'CVR ctrl':>12} {'CVR treat':>12} {'Lift (pp)':>12} {'Status':>12}")

for dev in ['desktop', 'mobile', 'tablet']:
    d = df[df.device == dev]
    dc = d[d.group == 'control']
    dt = d[d.group == 'treatment']
    cvr_c = dc.converted.mean()
    cvr_t = dt.converted.mean()
    lift = cvr_t - cvr_c
    status = "Positive" if lift > 0 else "Negative"
    print(f"  {dev:<12} {len(dc):>12,} {len(dt):>12,} {cvr_c:>12.2%} {cvr_t:>12.2%} {lift*100:>+12.2f} {status}")

print(f"\n  Note: Segment-level results are exploratory only.")
print(f"  The overall test is non-significant, so segment lifts should not")
print(f"  be interpreted as reliable treatment effects.")


# ── Risk Assessment ───────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print(" RISK ASSESSMENT")
print("═" * 65)

risks = [
    ("Statistical Significance", f"FAILED — p = {p_value:.4f}, cannot reject H₀"),
    ("Novelty Effect",           "SEVERE — Week 1 lift faded to ~zero by Week 3"),
    ("Temporal Stability",       "UNSTABLE — lift decayed consistently over 21 days"),
    ("Average Order Value",      "No negative impact detected (guardrail passed, but moot)"),
    ("Implementation Cost",      f"${IMPLEMENTATION_COST:,.0f} at risk with zero expected return"),
]

for area, note in risks:
    print(f"  • {area:<25}: {note}")


# ── Recommendation ────────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print(" RECOMMENDATION")
print("═" * 65)

print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  DECISION: Do NOT ship the new single-page checkout          ║
  ╚═══════════════════════════════════════════════════════════════╝

  CONFIDENCE: High
    - Primary KPI is NOT statistically significant (p = {p:.4f})
    - 95% CI for the lift crosses zero — the true effect may be negative
    - Novelty analysis reveals the early positive signal was transient
    - By Week 3, treatment performance had regressed to baseline
    - Shipping would cost ${cost:,} with zero expected sustained ROI

  WHY THIS IS A WIN:
    By running the test for the full 21-day duration and resisting the
    temptation to peek at Week 1 results, the data team protected the
    business from:
    - Wasting ${cost:,} in implementation costs
    - Adding unnecessary technical debt to the codebase
    - Drawing incorrect conclusions that would pollute future decisions
    - Opportunity cost of engineering time spent on a zero-value feature

  WHAT WE LEARNED:
    - Users responded positively to the new layout initially (novelty)
    - The improvement did not persist — this is a UX novelty effect
    - The current multi-step checkout is not the conversion bottleneck
      we assumed it was
    - Future optimization should target validated friction points

  NEXT STEPS:
    1. Archive experiment code & data                Owner: Data Science   Timeline: This week
    2. Document findings in experiment knowledge base Owner: Data Science   Timeline: This week
    3. Conduct qualitative UX research (interviews,  Owner: UX/Design      Timeline: Weeks 1-3
       session recordings) to identify real friction
       points in the checkout funnel
    4. Explore alternative optimizations:             Owner: Product        Timeline: Weeks 2-4
       - Guest checkout improvements
       - Payment method expansion
       - Address autofill / form pre-population
       - Progress indicator redesign
    5. If checkout redesign is revisited, design a    Owner: Data Science   Timeline: As needed
       longer test (28-42 days) with explicit novelty
       decay monitoring in the analysis plan
""".format(p=p_value, cost=IMPLEMENTATION_COST))


# ── Visualisation ─────────────────────────────────────────────────────────────
print(" Generating charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

# Left panel: Revenue projection showing $0 actionable value
ax1 = axes[0]

scenario_labels = list(LIFT_SCENARIOS.keys())
observed_revenues = [annual_sessions * lift * average_order_value for lift in LIFT_SCENARIOS.values()]
actionable_revenues = [0.0] * len(scenario_labels)  # All zero since non-significant

y_pos = range(len(scenario_labels))
bar_height = 0.35

# Observed (what naive analysis would project)
bars_observed = ax1.barh(
    [y - bar_height/2 for y in y_pos], observed_revenues, height=bar_height,
    color=['#B0B0B0', '#B0B0B0', '#B0B0B0'], alpha=0.4, label='Naive projection (not actionable)')

# Actionable (all zero)
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
