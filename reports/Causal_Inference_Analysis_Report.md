# Causal Inference Analysis: Impact of SMS Reminders on Medical Appointment No-Shows

## Overview

This report presents a comprehensive causal inference analysis investigating whether SMS reminders effectively reduce no-show rates for medical appointments. Using advanced causal machine learning techniques, we examine not only the average treatment effect of SMS reminders but also explore heterogeneous effects across different patient subgroups. The analysis employs multiple methodologies to ensure robust findings and provides actionable insights for healthcare providers seeking to optimize appointment attendance.

## Data Description

The analysis uses a dataset of medical appointments with the following key characteristics:

- **Target Variable**: No-show status (whether a patient missed their appointment)
- **Treatment Variable**: SMS_received (whether a patient received an SMS reminder)
- **Covariates**: 
  - Demographic information (Age, Gender)
  - Health conditions (Hipertension, Diabetes, Alcoholism, Handcap)
  - Appointment characteristics (WaitTime - days between scheduling and appointment)
  - Social factors (Scholarship status)
  - Derived features (Cluster_KMeans_2 - patient clustering)

Key statistics from the dataset reveal that approximately 20% of appointments are missed, about 30% of appointments had SMS reminders sent, around 10% of patients have scholarship status, and the average wait time between scheduling and appointment is approximately 10 days. For analytical purposes, a binary indicator for high wait time was created based on the median wait time to facilitate the analysis of heterogeneous effects.

## Methodology

### Causal Framework and Data Preparation

The analysis begins with the specification of a directed acyclic graph (DAG) to visualize the hypothesized causal relationships between variables. This graph illustrates that patient characteristics (age, gender, health conditions) directly influence no-show probability, SMS reminders directly affect no-show probability, and patient characteristics and wait time influence whether a patient receives SMS reminders. Additionally, scholarship status affects both SMS receipt and no-show probability.

To address class imbalance in the no-show outcome, the Synthetic Minority Over-sampling Technique (SMOTE) was applied. Before SMOTE, the dataset had an imbalanced distribution of no-show outcomes. After SMOTE, the distribution was balanced, ensuring models wouldn't be biased toward the majority class.

### Causal Inference Methods

Multiple causal inference methods were implemented to ensure robust findings:

**Double Machine Learning (DML)** approaches were employed, including:
- LinearDML with XGBoost models for outcome and treatment
- CausalForestDML for capturing heterogeneous treatment effects

**Meta-Learners** were also implemented to provide additional perspectives:
- T-Learner: Builds separate models for treated and control groups
- X-Learner: Extends T-Learner by incorporating propensity scores
- S-Learner: Single model approach with treatment as a feature

### Validation Techniques

To ensure the validity of our causal estimates, several validation techniques were employed:

- **Placebo Tests**: Treatment assignment was randomized to detect spurious effects
- **Conditional Average Treatment Effect (CATE) Analysis**: Treatment effects were examined across different subgroups
- **SHAP Analysis**: Feature importance in treatment effect heterogeneity was identified using Shapley values

## Results and Analysis

### Average Treatment Effect

Before applying causal methods, a simple comparison of no-show rates between patients who received SMS reminders and those who didn't was performed. This initial exploration showed a difference in no-show rates between the two groups, but this naive comparison doesn't account for confounding factors.

All causal inference methods consistently estimated a negative effect of SMS reminders on no-show probability, indicating that SMS reminders help reduce appointment no-shows. The LinearDML model estimated the Average Treatment Effect (ATE) with confidence intervals excluding zero, suggesting a statistically significant effect. The CausalForestDML model provided an ATE estimate consistent with the LinearDML model.

The meta-learners (T-Learner, X-Learner, and S-Learner) all estimated negative ATEs, further confirming that SMS reminders reduce no-show probability. The magnitudes varied slightly across methods, providing a range of plausible effect sizes.

### Model Robustness Assessment

Placebo tests were conducted to assess the robustness of the causal estimates by randomly shuffling the treatment labels. The results revealed important differences in model reliability:

- T-Learner detected some spurious effects (0.0132) even with randomized treatment labels, suggesting potential overfitting or sensitivity to noise.
- X-Learner showed minimal spurious effects (0.0013), indicating greater robustness to random variations and higher reliability for causal inference.
- S-Learner produced exactly zero effect in the placebo test, but this is expected due to its single-model approach, which may make it less sensitive to treatment effects overall.

These results suggest X-Learner provides more reliable estimates than T-Learner, as it's less prone to detecting spurious effects.

### Heterogeneous Treatment Effects

The analysis revealed important patterns in how SMS reminders affect different patient subgroups:

**Age-Related Effects**: The dataset was divided into four age groups (Children: 0-18, Young Adults: 18-40, Adults: 40-65, Elderly: 65+). SMS reminders are most effective for young adults and adults (18-65 years). This may be due to higher smartphone usage and familiarity with text messaging in these age groups compared to children (where parents may manage appointments) and elderly patients. Both T-Learner and X-Learner showed consistent patterns across age groups.

**Scholarship Status**: Treatment effects were compared between patients with and without scholarships. Patients with scholarships show stronger responses to SMS reminders. This could indicate that patients receiving financial assistance may be more motivated to attend appointments or have better access to mobile technology. The effect is consistent across both T-Learner and X-Learner models.

**Wait Time Impact**: Appointments with longer wait times between scheduling and the actual appointment date benefit more from SMS reminders. This is intuitive as patients are more likely to forget appointments scheduled further in advance. This effect is consistent across both T-Learner and X-Learner models.

### Feature Importance Analysis

SHAP (SHapley Additive exPlanations) analysis was performed to understand feature importance in the treatment effect. The analysis was conducted for both T-Learner and X-Learner models, examining the control group, treatment group, and the difference between them (CATE contribution).

The SHAP analysis revealed that wait time, age, and cluster membership are important factors influencing the treatment effect. Different features have varying importance in the control and treatment groups. The CATE contribution analysis highlights which features drive heterogeneity in treatment effects, confirming the findings from the subgroup analyses.

## Threats to Validation

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

## Conclusions and Recommendations

### Conclusions

Our analysis provides strong evidence that SMS reminders causally reduce appointment no-show rates. This effect is consistent across multiple causal inference methods, strengthening confidence in this finding. The causal effect varies across different patient subgroups, with SMS reminders being most effective for adults aged 18-65 years, patients with scholarship status, and appointments with longer wait times.

Wait time, age, and patient clustering emerge as key factors influencing the effectiveness of SMS reminders, as confirmed by both subgroup analysis and SHAP feature importance. Among the models tested, X-Learner demonstrates greater robustness to spurious correlations compared to T-Learner, as evidenced by placebo test results.

The analysis also reveals that standard predictive models struggle to accurately predict no-shows, suggesting inherent limitations in the available features or complexity in no-show behavior that might also affect causal inference.

### Recommendations

Based on our findings, we recommend implementing a targeted SMS reminder strategy focusing on young adults and adults (18-65 years), patients with scholarship status, and appointments with longer wait times. Healthcare providers should consider sending additional reminders for appointments with longer wait times, as these show stronger benefits from SMS interventions. SMS content should be tailored based on patient characteristics that showed heterogeneous treatment effects.

For future research, we recommend conducting a randomized controlled trial to address selection bias concerns, exploring the optimal timing of SMS reminders relative to appointment date, investigating the content of SMS messages to maximize effectiveness, performing sensitivity analyses to assess robustness to unobserved confounding, and collecting additional features that might better predict no-show behavior.

For model improvement, we recommend using X-Learner over T-Learner for future analyses, implementing double-robust methods to reduce sensitivity to model misspecification, collecting additional data on potential confounders not currently captured, exploring causal discovery methods to learn the causal structure from data, and addressing the predictive performance limitations by incorporating more relevant features.

Finally, we recommend establishing a monitoring system to track the ongoing effectiveness of SMS reminders after implementation of these recommendations. By implementing these recommendations, healthcare providers can optimize their SMS reminder systems to maximize appointment attendance, improve resource utilization, and enhance patient care.
