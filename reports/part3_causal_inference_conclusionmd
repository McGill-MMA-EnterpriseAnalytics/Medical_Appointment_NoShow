# Causal Inference Analysis: Impact of SMS Reminders on Medical Appointment No-Shows

This report presents a comprehensive causal inference analysis investigating whether SMS reminders effectively reduce no-show rates for medical appointments. Advanced causal machine learning methods were applied to estimate both the average treatment effect (ATE) and heterogeneous treatment effects across various patient subgroups. The findings offer actionable insights for healthcare providers looking to optimize appointment attendance.

---

## 1. Overview and Data Description

The analysis uses a dataset of medical appointments with the following key characteristics:

- **Target Variable:**  
  - *No-show*: Indicator of whether a patient missed their appointment.
- **Treatment Variable:**  
  - *SMS_received*: Indicator of whether a patient received an SMS reminder.
- **Covariates:**  
  - Demographics (Age)
  - Health conditions (Hipertension, Alcoholism, Handcap)
  - Appointment characteristics (WaitTimeLog, Schedule Day, Appointment Dat)
  - Social factors (Scholarship status)
  - Derived features
  
Key statistics include:
- Approximately **20%** no-show rate.
- About **32%** of appointments had SMS reminders.
- Around **10%** of patients have scholarship status.
- The average wait time is roughly **10 days**.  
For analysis, a binary indicator (*HighWaitTime*) was created based on the median log-transformed wait time.

---

## 2. Causal Framework and Data Preparation

### Directed Acyclic Graph (DAG)
A causal graph was constructed to represent hypothesized relationships among variables. Key edges include:
> **Visual Reference:**  
> View the causal graph [here](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/directed_acyclic_graph.png).

### Data Preprocessing and SMOTE
- **SMOTE:**  
  SMOTE was applied to address class imbalance in the no-show outcome.  
  - **Before SMOTE:** The original data had an imbalanced distribution of no-show outcomes.
  - **After SMOTE:** The distribution was balanced to reduce bias toward the majority class.

- **Encoding:**  
  Categorical features (e.g., ScheduledDay_Weekday, AppointmentDay_Weekday) were converted using label encoding to facilitate causal analysis.

- **Treatment and Feature Sets:**  
  Core features were selected for causal analysis, and a binary treatment variable (SMS_received) was defined.

---

## 3. Causal Inference Methodology

Multiple methods were applied to estimate the causal effect of SMS reminders:

### 3.1 Double Machine Learning (DML) Approaches
- **LinearDML:**  
  - **Outcome Model:** XGBoost Regressor  
  - **Treatment Model:** XGBoost Classifier  
  - **Result:**  
    - **ATE:** -0.0944  
    - **95% CI:** [-0.1042, -0.0846]  
  - *Interpretation:* On average, receiving an SMS reminder reduces the no-show probability by about 9.4 percentage points. The confidence interval does not cross zero, suggesting a statistically significant effect.

- **CausalForestDML:**  
  - **Result:**  
    - **ATE:** -0.0963  
  - *Interpretation:* The causal forest method produces a similar estimate, reinforcing the conclusion that SMS reminders have a meaningful effect on reducing no-show rates.

### 3.2 Meta-Learners
- **T-Learner:**  
  - **ATE:** -0.1922  
- **X-Learner:**  
  - **ATE:** -0.3267  
- **S-Learner:**  
  - **ATE:** -0.1471  
- *Interpretation:*  
  All meta-learners estimated negative ATEs, confirming that SMS reminders causally lower no-show probabilities. The X-Learner provided the strongest effect estimate, suggesting that it may better capture treatment effect heterogeneity in this context.

### 3.3 Placebo Tests
- **Placebo Results:**  
  - **T-Learner:** 0.0081  
  - **X-Learner:** 0.0078  
  - **S-Learner:** 0.0000  
- *Interpretation:*  
  When treatment labels were randomized, the estimated ATEs were near zero. This confirms that the causal estimates in the main analysis are not driven by spurious correlations or model overfitting. X-Learner appears to be slightly more robust than T-Learner because its placebo ATE (0.0078) is lower than that of T-Learner (0.0081)

---

## 4. Heterogeneous Treatment Effects

The analysis also explored how the causal effect of SMS reminders varies across patient subgroups.

### 4.1 Age Groups
Patients were categorized into four groups:
- **Children (0-18)**
- **Young Adults (18-40)**
- **Adults (40-65)**
- **Elderly (65+)**

**T-Learner Results by Age (ATE estimates):**
- Children: -0.0916
- Young Adults: -0.0870
- Adults: -0.2922
- Elderly: -0.4101

**X-Learner Results by Age (ATE estimates):**
- Children: -0.3645
- Young Adults: -0.3038
- Adults: -0.3173
- Elderly: -0.2711

> **Visual Reference:**  
> See the [SMS Effect by Age Group (X-Learner)](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/SMS_effect_age.png) graph for a bar plot of the average causal effect by age group.  
> **Interpretation:** The results indicate that SMS reminders significantly reduce no-show rates across all age groups, but the magnitude of the effect varies:
  - Elderly patients (65+) show the strongest response to SMS reminders, with an ATE of -0.4101 (T-Learner) and -0.2711 (X-Learner). This suggests that older adults benefit the most from reminders, possibly due to memory-related challenges or the need for extra reinforcement in appointment adherence.
  - Adults (40-65) also exhibit a substantial effect, particularly in the T-Learner model (-0.2922). This group may have busy schedules that increase the likelihood of forgetting appointments, making SMS reminders a useful intervention.
  - Young adults (18-40) and children (0-18) show the smallest impact, with ATE estimates ranging from -0.0870 to -0.3645. The lower effect in children may be due to parents managing their appointments, reducing the need for direct reminders.
  - Discrepancies between T-Learner and X-Learner estimates suggest that X-Learner attributes stronger effects to younger patients, while T-Learner emphasizes the impact on elderly patients. This could be due to X-Learner's better handling of heterogeneous treatment effects.

### 4.2 Scholarship Status
- **T-Learner:**  
  - Non-Scholarship: -0.1954  
  - Scholarship: -0.1565
- **X-Learner:**  
  - Non-Scholarship: -0.3264  
  - Scholarship: -0.3301

*Interpretation:*  
Patients with scholarships show a slightly stronger benefit from SMS reminders in the X-Learner estimates, suggesting that socioeconomic factors may influence the effectiveness of reminders.

### 4.3 Wait Time
- **T-Learner (High vs. Low Wait Time):**  
  - Low Wait Time: -0.2269  
  - High Wait Time: -0.1562
- **X-Learner (High vs. Low Wait Time):**  
  - Low Wait Time: -0.5620  
  - High Wait Time: -0.0826

*Interpretation:*  
The T-Learner results indicate that appointments with lower wait times benefit more from SMS reminders. In contrast, the X-Learner estimates suggest that longer wait times could be associated with a larger reduction in no-show probability when an SMS is received.

### 4.4 Weekday Effects
- **Scheduled Day (T-Learner & X-Learner):**  
  Both models show variations in SMS effect depending on the day of scheduling. For instance, Mondays and Saturdays often show stronger negative effects.
- **Appointment Day (T-Learner & X-Learner):**  
  The effect of SMS reminders also varies with the day of the appointment, with Friday and Saturday frequently showing a greater reduction in no-shows.

---

## 5. Feature Importance in Treatment Effects

### SHAP Analysis for X-Learner
SHAP analysis was performed to understand which features contribute most to the estimated treatment effects. For example, the difference in SHAP values between the treatment and control models (computed as the CATE contribution) highlights the role of features such as *Age*, *WaitTime_Log*, and patient clustering.

> **Visual Reference:**  
> A SHAP summary plot of the treatment effect is available [here](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/SHAP_X_treatment_effect.png).

*Interpretation:*  
- Higher wait times (red) correspond to more negative SHAP values, indicating that SMS reminders are more effective at reducing no-shows when the appointment is scheduled far in advance. In contrast, shorter wait times (blue) have SHAP values closer to zero, suggesting that SMS reminders have a weaker impact when the appointment occurs soon after booking.
- The negative SHAP values for certain age groups suggest that SMS reminders significantly reduce no-show rates in these populations. Notably, older adults exhibit stronger negative SHAP values, reinforcing our earlier findings that SMS reminders are particularly effective for elderly patients.
- Scholarship recipients (red, higher values) tend to have slightly more negative SHAP values, implying that SMS reminders are more effective for lower-income patients. This may suggest that financial assistance programs correlate with higher responsiveness to appointment reminders.
- Medical conditions (Hypertension, Alcoholism, Handicap) show some influence but are less dominant compared to scheduling factors or wait time. Patients with chronic conditions may require additional interventions beyond SMS reminders to improve attendance rates.

---

## 6. Conclusions and Recommendations

### Conclusions
- **Causal Effect:**  
  SMS reminders consistently show a negative causal effect on the probability of a no-show, with ATE estimates ranging from approximately -0.09 (DML methods) to -0.33 (X-Learner). This means that, on average, SMS reminders reduce no-show probabilities.
- **Heterogeneity:**  
  The effect of SMS reminders is not uniform:
  - **Age:** Stronger effects among adults and the elderly in certain models.
  - **Scholarship:** Slight differences indicate that patients receiving financial assistance may respond differently.
  - **Wait Time and Weekday:** The effectiveness of SMS reminders varies with wait times and scheduling/appointment days.
- **Robustness:**  
  Placebo tests confirmed that the estimated effects are robust, as randomized treatment labels yielded near-zero ATE estimates.

### Recommendations
1. **Targeted Interventions:**  
   Focus SMS reminder strategies on patient subgroups that exhibit stronger treatment effects (e.g., adults, elderly, and those with longer wait times).
2. **Enhanced Personalization:**  
   Incorporate additional patient data to further refine the targeting of SMS reminders, tailoring content and timing to maximize attendance.
3. **Further Research:**  
   Conduct randomized controlled trials to validate these observational findings and explore the impact of SMS content and timing.
4. **Monitoring and Feedback:**  
   Establish continuous monitoring systems to track the long-term effectiveness of SMS reminders and adjust strategies based on ongoing performance data.

## 7. Threats to Validation

### Selection Bias

The analysis assumes that SMS reminders were assigned in a way that can be modeled using observed covariates. If the decision to send SMS reminders was based on unobserved factors (e.g., patient technology access, prior communication preferences), this would violate the unconfoundedness assumption. Additionally, patients who provide mobile numbers might systematically differ from those who don't, creating a self-selection problem that could bias treatment effect estimates. Past appointment behavior might also influence both the likelihood of receiving SMS reminders and future appointment attendance, creating a form of temporal confounding not addressed in the analysis.

### Data Quality and Missingness

The analysis doesn't explore patterns of missingness. If data is missing not at random (MNAR) and related to the outcome, this could significantly bias causal estimates. Potential errors in recording SMS delivery status (e.g., SMS sent but not received, or technical failures) could attenuate the estimated treatment effect. The analysis also assumes perfect recording of appointment attendance, but if there are errors in this recording process, it could lead to outcome misclassification.

### Predictive Model Performance Issues

Predictive models (not shown in the notebook but referenced) show limited ability to accurately predict no-shows, with very low recall rates and high false negatives. This suggests the features available may not be sufficient to predict no-show behavior, there may be inherent randomness or unobserved factors driving no-shows, and class imbalance issues persist despite attempts to address them.

While these predictive models weren't directly used in the causal inference analysis, their poor performance indicates potential omitted variable bias in causal models using the same feature set, difficulty in accurately modeling the outcome mechanism (which could affect double machine learning approaches), and possible violation of the unconfoundedness assumption if important predictors are missing.

### Temporal Aspects

The analysis doesn't account for time-varying confounders that might change between appointment scheduling and the actual appointment date. Appointment attendance might vary by season, day of week, or time of day, which could confound the relationship if SMS reminders were not uniformly distributed across these temporal factors. The effectiveness of SMS reminders might also diminish over time as patients become habituated to them, an effect not captured in this cross-sectional analysis.

### External Validity

The patient population in this dataset may not represent the broader population of interest, limiting generalizability. The effectiveness of SMS reminders might depend on specific healthcare system characteristics not transferable to other contexts. The analysis period might also reflect a specific technological context that could change over time (e.g., as smartphone usage patterns evolve).

### Model Assumptions and Limitations

The analysis assumes sufficient overlap in covariate distributions between treated and control groups, but extreme propensity scores could indicate regions where this assumption is violated. LinearDML assumes a linear relationship between covariates and outcomes, which may not hold in reality. The placebo test results for T-Learner suggest potential model misspecification or overfitting, as it detects effects even with randomized treatment labels. The models may also not fully capture the complex heterogeneity in treatment effects across different patient subpopulations.

### SMOTE Resampling Concerns

SMOTE creates synthetic samples that might not reflect the true data-generating process, potentially introducing artifacts. It can distort decision boundaries, especially in high-dimensional spaces, affecting causal estimates. Models trained on SMOTE-resampled data might overfit to the synthetic patterns rather than real-world relationships. Using synthetic samples in causal inference also raises questions about the validity of counterfactual reasoning based on artificially created data points.
