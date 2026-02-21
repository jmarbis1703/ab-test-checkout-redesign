# A/B Test: E-Commerce Checkout Redesign

End-to-end A/B testing project, from experiment design through statistical analysis to a ship/no-ship business recommendation.

**Result: Do not ship.** Rigorous methodology saved the business $25,000 and prevented unnecessary technical debt.

![Cumulative CVR](assets/cumulative_cvr.png)

---

## Business Context

A mid-size e-commerce retailer (~3,700 sessions/day) tested a **single-page checkout** against its existing multi-step checkout flow. The hypothesis was that reducing friction in the purchase funnel would increase checkout conversion without lowering average order value.

### Hypotheses

A **two-sided hypothesis test** was applied to maintain statistical rigor.

- **H₀ (Null):** $CVR_{treatment} = CVR_{control}$
- **H₁ (Alternative):** $CVR_{treatment} \neq CVR_{control}$

**Decision Framework:**
1. **Significance:** $H_0$ is rejected only if $p < 0.05$.
2. **Improvement:** If $H_0$ is rejected, the new design will only be implemented if the effect is significant *and* positive.
3. **Guardrails:** The new checkout won't be implemented if AOV shows a statistically significant decline, even if CVR increases.

### KPIs

| Metric | Role | Why |
|---|---|---|
| Checkout conversion rate | Primary | Measures if users complete the purchasing process |
| Revenue per session | Secondary | Captures both conversion and order value changes |
| Average order value | Guardrail | Ensures average order value does not decrease |

---

## Methodology

### Experiment Design

| Parameter | Value | Rationale |
|---|---|---|
| Randomisation unit | User (cookie) | Standard for web experiments, avoids cross-device leakage |
| Split | 50/50 | Maximises statistical power |
| Duration | 21 days | Covers 3 full weekly cycles to account for day-of-week effects |
| MDE | 0.4 pp (~12.5% relative) | Business-meaningful lift, achievable with ~30K sessions/group |
| α | 0.05 (two-sided) | Industry standard |
| Power | 0.80 | Standard: 21-day run yields well above the required sample per group |

### Risk Mitigation

- **Sample Ratio Mismatch (SRM):** Chi-squared test assessed the balance between control and treatment group sizes. Any observed imbalance was evaluated and factored into interpretation.

- **Novelty effect:** This turned out to be the defining feature of this experiment. Conversion rates were compared between Week 1 and Week 3. This analysis revealed that early treatment lift was driven by a novelty spike that eventually dissipated. The 21-day duration was essential in identifying this trend.

- **Multiple comparisons:** To account for three simultaneous tests (CVR, Revenue per Session, AOV), a Bonferroni correction was applied (adjusted $\alpha = 0.0167$).

- **Peeking:** Results were analyzed exclusively after the pre-specified 21-day period to maintain statistical integrity.

- **Intra-user correlation:** A design effect of ~1.01 was calculated, confirming that session-level results were not significantly inflated by within-user clustering.

### Dataset

Synthetic data generated with realistic patterns: day-of-week seasonality, device mix, traffic source variation, and a fading novelty bump built into the treatment effect. See `notebooks/01_generate_data.py` for the full generative model and parameters.

**78,561 sessions** over 21 days (39,342 control / 39,219 treatment).

---

## Results

### Primary Metric - Conversion Rate

| | Control | Treatment |
|---|---|---|
| Sessions | 39,342 | 39,219 |
| CVR | 3.3018% | 3.4728% |

- **Absolute lift:** +0.17 pp
- **Relative lift:** +5.2%
- **p-value:** 0.1853
- **95% CI:** [−0.08 pp, +0.42 pp] — crosses zero

**Failed to reject the null hypothesis.** The observed +0.17 pp lift is not statistically distinguishable from zero at the 0.05 significance level.

![Lift CI](assets/lift_ci_plot.png)

![Cumulative CVR](assets/cumulative_cvr.png)

The cumulative CVR chart shows the treatment and control lines converging over 21 days as the novelty-inflated early gap shrinks to near zero.

### The Novelty Effect

The overall result masks a critical pattern that only emerges when the data is examined over time:

| Period | Treatment Lift | Interpretation |
|---|---|---|
| Week 1 | +0.33 pp | Strong positive signal - novelty excitement |
| Week 2 | +0.20 pp | Effect fading as users habituate |
| Week 3 | −0.02 pp | Lift evaporated entirely |

This is a textbook **novelty effect**. Users initially engaged more with the new checkout because it was different, not because it was better. Once the novelty wore off, behavior reverted to baseline.

#### The Peeking Counterfactual

Terminating the test after Week 1 would have led to an incorrect "Ship" decision based on inflated projections:

| Metric                    | Peeked at Week 1       | Full 21-Day Result     |
|:--------------------------|:-----------------------|:-----------------------|
| Observed lift             | +0.33 pp               | +0.17 pp               |
| p-value                   | 0.1414                 | 0.1853                 |
| 95% CI                    | [−0.11, +0.78] pp      | [−0.08, +0.42] pp      |
| Projected annual revenue  | $318,023               | $0 (n.s.)              |
| Ship decision             | Tempted to ship        | Do not ship            |
| Business outcome          | $25K spent, zero ROI   | $25K saved             |

The Week 1 data would have projected $318,023 in annual revenue, the true sustained value was $0. The 21-day discipline saved $25,000 in wasted implementation costs.

![Peeking Counterfactual](assets/peeking_counterfactual.png)

**If the team had peeked at results after Week 1 and shipped the feature, they would have:**
- Committed $25,000 in implementation costs for zero sustained ROI
- Added permanent technical debt to the codebase
- Drawn incorrect conclusions that would pollute future decision-making

The 21-day experiment design caught what a 7-day test would have missed entirely.

![Rolling Lift](assets/rolling_lift.png)

### Segmentation

Treatment lift varied substantially across device × traffic source combinations. Several combinations showed negative lift, though small sample sizes within individual cells limit the reliability of segment-level conclusions.

![Device × Source Lift](assets/lift_device_source_heatmap.png)

### Secondary Metrics
- **Revenue per session:** No statistically significant difference (bootstrap 95% CI: [−$0.08, $0.31]).
- **AOV (Guardrail):** No significant change - the guardrail was not triggered, but this is moot given the primary metric failed.

### Robustness
- Permutation test (10,000 iterations, p = 0.1883) confirms the non-significant result.
- User-level z-test is consistent with session-level findings.
- Temporal analysis reveals clear novelty decay, the treatment effect is not stable over time.
- Tablet segment showed a negative signal, though with small sample size.

---

## Business Impact

![Revenue Projection](assets/revenue_impact.png)

Because the test failed to reach statistical significance and the confidence interval for the true lift spans zero, the **expected incremental revenue from this feature is $0**. Any projection based on the point estimate would be misleading.

| Scenario | Projected Annual Revenue |
|---|---|
| Point estimate (+0.17 pp) | Not actionable — p > 0.05 |
| Lower bound of 95% CI | Negative — feature could *hurt* revenue |
| Upper bound of 95% CI | Positive, but indistinguishable from noise |

**Break-even analysis:** With an expected sustained lift of zero, the $25,000 implementation cost would never be recovered. Payback period is effectively infinite.

---

## Recommendation

### Do not ship the new checkout

The treatment failed to produce a statistically significant improvement in the primary KPI. More importantly, temporal analysis reveals that the small observed lift was entirely driven by a novelty effect that decayed to zero by Week 3. There is no evidence of sustained business value.

This outcome demonstrates the value of disciplined experimentation. By adhering to the pre-specified 21-day test duration and resisting the temptation to peek at early results, the team prevented a $25,000 investment in a feature with no long-term ROI. The savings are real and immediate.

**Key Findings:**
1. The single-page checkout provides no sustained business value over the current multi-step flow.
2. Future optimizations should target genuine friction points (e.g., address autofill, guest checkout) rather than purely aesthetic layout changes.
3. The 21-day testing discipline is essential for avoiding false positives driven by temporary user behavior shifts.

**Next Steps:**
1. Archive the experiment code and data for institutional knowledge.
2. Conduct a qualitative UX review (user interviews, session recordings) to understand *why* the novelty effect occurred and what friction points actually exist in the current flow.
3. Explore alternative checkout optimizations (guest checkout improvements, payment method expansion, address autofill, progress indicators) that target specific, measurable pain points.
4. Establish a standardized 28-day testing window for future major UI changes to fully monitor novelty decay.

---

## Limitations

- **Synthetic data:** The treatment effect, including the novelty decay, was explicitly programmed. Real novelty effects may be more or less severe and may follow different decay curves.

- **No real-world interference:** This dataset doesn't include bot traffic, ad blocker interference, cookie clearing, tracking failures, or other data quality issues common in production experiments.

- **Perfect randomization:** Users were assigned to groups with programmatic randomization. In practice, cookie-based assignment can break due to technical issues, users clearing cookies, or traffic routing problems.

- **No external disruptions:** The 21-day test ran in isolation. Real experiments often overlap with marketing campaigns, website outages, seasonal events, or competitor activity that can confound results.

- **Simplified novelty model:** The novelty effect follows a clean exponential decay. Real-world novelty effects may be more complex, with partial recovery, segment-specific patterns, or interaction with external events.

- **Returning-user identification:** In production, I would implement cookie-clearing detection and cross-device reconciliation to sharpen returning-user identification and novelty decay measurement.

- **Sequential monitoring:** A Bayesian approach with sequential monitoring could replace fixed-horizon testing, allowing earlier stopping when the posterior probability of a meaningful effect drops below threshold, while still protecting against novelty effects through time-stratified priors.

- **MDE at the margin:** The MDE (0.4 pp) was set equal to the true simulated treatment effect, meaning the test was powered at the margin. In practice, a lower MDE would provide a safety buffer.

---

## Repository Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── ab_test_data.csv               # Generated experiment data (~76K rows)
├── notebooks/
│   ├── 01_generate_data.py            # Synthetic data generation
│   ├── 02_eda.py                      # Exploratory data analysis
│   ├── 03_statistical_analysis.py     # Hypothesis testing & robustness
│   ├── 04_business_recommendations.py # Impact sizing & recommendation
│   └── ab_test_checkout_novelty_effect.ipynb # Full analysis & novelty effect deep-dive
├── src/
│   ├── data_utils.py                  # Data loading & validation
│   └── stats_utils.py                 # Reusable statistical functions
└── assets/                            # Charts generated by analysis scripts
```

## Reproducibility

```bash
git clone https://github.com/jmarbis1703/ab-test-checkout-redesign
cd ab-test-checkout-redesign

# Create a virtual environment (Recommended)
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Upgrade pip (crucial for Windows to avoid wheel build errors with SciPy/NumPy)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate data
python notebooks/01_generate_data.py

# Run analysis (in order)
python notebooks/02_eda.py
python notebooks/03_statistical_analysis.py
python notebooks/04_business_recommendations.py
```

All random seeds are fixed. Outputs are deterministic.

---

## Tools & Libraries

Python 3.11+ · pandas · NumPy · SciPy · statsmodels · matplotlib · seaborn
