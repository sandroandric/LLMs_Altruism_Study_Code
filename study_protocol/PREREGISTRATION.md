# Pre-Registration: LLM Implicit Altruism Study

**Title:** Do Large Language Models Walk Their Talk? Measuring the Gap Between Implicit Associations, Self-Report, and Behavioral Altruism

**Authors:** [To be filled]

**Date:** [To be filled before data collection]

**Repository:** https://github.com/sandroandric/altruism

---

## 1. Study Information

### 1.1 Research Questions

**RQ1:** Do Large Language Models exhibit implicit altruistic associations as measured by an adapted Implicit Association Test?

**RQ2:** Do LLMs behave altruistically when forced to make binary choices between self-interest and other-interest?

**RQ3:** Is there a correlation between implicit altruism (IAT) and behavioral altruism (Forced Choice)?

**RQ4:** Are LLMs calibrated in their self-assessment of altruism relative to their actual behavior?

### 1.2 Hypotheses

**H1 (Confirmatory):** LLMs will show positive mean IAT altruism bias (> 0).
- Test: One-sample t-test against 0
- Expected direction: Positive
- Rationale: Training data reflects human prosocial values

**H2 (Confirmatory):** There will be significant differences in IAT scores across models.
- Test: One-way ANOVA
- Expected: Significant F-statistic (p < 0.05)
- Rationale: Different training approaches yield different value profiles

**H3 (Exploratory):** IAT scores will correlate with behavioral altruism.
- Test: Pearson correlation
- Expected direction: Unknown (pilot suggested negative)
- Rationale: Implicit attitudes may or may not predict behavior

**H4 (Exploratory):** Self-report scores will correlate with behavioral altruism.
- Test: Pearson correlation
- Expected direction: Positive
- Rationale: Self-knowledge should predict behavior

**H5 (Exploratory):** Most models will be over-confident (self-report > behavior).
- Test: Paired t-test (self-report normalized vs behavior)
- Expected direction: Self-report > Behavior
- Rationale: Pilot study suggested over-confidence

---

## 2. Design Plan

### 2.1 Study Type

Observational / Cross-sectional comparison of LLM populations

### 2.2 Blinding

Not applicable (automated data collection)

### 2.3 Study Design

Within-subjects design where each model completes:
1. IAT task (30 trials)
2. Forced Binary Choice task (51 trials)
3. Self-Assessment scale (15 items × 3 repeats)

### 2.4 Randomization

- Word order randomized within IAT trials
- Option order (A/B) randomized within Forced Choice
- Item order fixed for Self-Assessment (following standard psychometric practice)

---

## 3. Sampling Plan

### 3.1 Existing Data

Pilot data exists (N=8 models) but will NOT be included in confirmatory analyses. Pilot data used only to:
- Estimate effect sizes for power analysis
- Refine methodology
- Debug code

### 3.2 Data Collection Procedures

All data collected via OpenRouter API with:
- Temperature = 0.1 (low variance, reproducible)
- Max tokens = 100 (sufficient for single-word/letter responses)
- No system prompt (default model behavior)

### 3.3 Sample Size

**Target: N = 24 models minimum**

Power analysis (see Section 3.4):
- For correlation r = 0.50, α = 0.05, power = 0.80: N = 29
- For correlation r = 0.60, α = 0.05, power = 0.80: N = 19
- Compromise: N = 24 (power = 0.75 for r = 0.50)

### 3.4 Power Analysis

```
Effect size estimate from pilot: r = -0.63
Conservative estimate: r = 0.50

Using G*Power for bivariate correlation:
- Effect size: ρ = 0.50
- α = 0.05 (two-tailed)
- Power = 0.80
- Required N = 29

With N = 24:
- Power for r = 0.50: 0.75
- Power for r = 0.60: 0.88
```

### 3.5 Stopping Rule

Data collection stops when:
1. All planned models have been tested, OR
2. A model returns >50% invalid responses (exclude and replace), OR
3. API budget exhausted (document and report achieved N)

---

## 4. Variables

### 4.1 Manipulated Variables

None (observational study)

### 4.2 Measured Variables

**Primary Outcomes:**

| Variable | Measure | Range | Experiment |
|----------|---------|-------|------------|
| IAT Altruism Bias | P(pos→other) + P(neg→self) - 1 | [-1, +1] | Exp 1 |
| Behavioral Altruism | Proportion other-focused choices | [0, 1] | Exp 2 |
| Self-Reported Altruism | Mean Likert score | [1, 7] | Exp 3 |

**Secondary Outcomes:**

| Variable | Measure | Range | Experiment |
|----------|---------|-------|------------|
| IAT by word valence | Separate positive/negative | [0, 1] | Exp 1 |
| Behavior by category | Per-category altruism rate | [0, 1] | Exp 2 |
| Self-report subscales | Attitudes/Everyday/Sacrifice | [1, 7] | Exp 3 |
| Calibration | Self-report(norm) - Behavior | [-1, +1] | Derived |

### 4.3 Indices

**Calibration Index:**
```
Calibration = (Self-Report - 1) / 6 - Behavioral Altruism
```
- Positive = Over-confident
- Zero = Well-calibrated
- Negative = Under-confident

---

## 5. Analysis Plan

### 5.1 Statistical Models

**H1 Test:**
```
One-sample t-test: mean(IAT) vs μ₀ = 0
Report: t, df, p, 95% CI, Cohen's d
```

**H2 Test:**
```
One-way ANOVA: IAT ~ Model
Report: F, df_between, df_within, p, η²
Post-hoc: Tukey HSD if significant
```

**H3 Test:**
```
Pearson correlation: r(IAT, Behavior)
Report: r, p, 95% CI
Robustness: Spearman ρ
```

**H4 Test:**
```
Pearson correlation: r(Self-Report, Behavior)
Report: r, p, 95% CI
```

**H5 Test:**
```
Paired t-test: Self-Report(normalized) vs Behavior
Report: t, df, p, mean difference, 95% CI
```

### 5.2 Transformations

- Self-Report normalized to [0, 1]: (score - 1) / 6
- No other transformations planned

### 5.3 Inference Criteria

- α = 0.05 for all confirmatory tests
- Two-tailed tests throughout
- Report exact p-values
- Report effect sizes with 95% CIs

### 5.4 Data Exclusion

Exclude trials where:
- Response could not be parsed (>3 attempts)
- Model returned error/timeout

Exclude models where:
- Parse rate < 50% on any task
- API consistently fails

Document all exclusions.

### 5.5 Missing Data

- Use available data (no imputation)
- Report N for each analysis
- Sensitivity analysis if >10% missing

### 5.6 Exploratory Analyses

1. Provider-level analysis (OpenAI vs Anthropic vs Google etc.)
2. Model size effects (if size information available)
3. Category-specific behavioral patterns
4. Reverse-coded item analysis for acquiescence bias

---

## 6. Additional Information

### 6.1 Pilot Study

Pilot conducted November 2025 with N=8 models:
- Confirmed IAT discrimination
- Discovered Hard Decision ceiling effect (redesigned to Forced Choice)
- Estimated effect sizes for power analysis
- Found surprising negative IAT-Behavior correlation (r = -0.63)

### 6.2 Constraints on Generality

Results may not generalize to:
- Future model versions (rapid development)
- Different temperature settings
- Non-English prompts
- Real-world deployment contexts
- Base models (only instruction-tuned tested)

### 6.3 Materials Availability

All materials available at: https://github.com/sandroandric/altruism
- Word lists: `src/word_lists.py`
- Scenarios: `src/forced_choice_task.py`
- Self-assessment items: `src/self_assessment_task.py`

### 6.4 Code Availability

Analysis code pre-written and available:
- `run_full_study.py` - Master execution script
- `src/stats.py` - Statistical analysis functions
- `paper/figures/generate_figures.py` - Visualization

### 6.5 Deviations from Pre-Registration

Any deviations will be:
1. Documented in final paper
2. Clearly labeled as post-hoc
3. Justified with rationale

---

## 7. Declaration

By submitting this pre-registration, I confirm that:

- [ ] I have not begun data collection for the main study
- [ ] Pilot data will not be included in confirmatory analyses
- [ ] I will report all pre-registered analyses regardless of results
- [ ] Any deviations will be documented and labeled

**Signature:** _______________________

**Date:** _______________________

---

## Appendix: Planned Model List

See `study_protocol/MODEL_LIST.md` for complete list of N=24+ models.
