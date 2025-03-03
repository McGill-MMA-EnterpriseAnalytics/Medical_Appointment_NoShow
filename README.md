# Medical Appointment No-Show Analysis

## Literature Review
Recent studies have underscored the multifaceted nature of medical appointment no-shows. For instance, research published in BMC Health Services Research ([Link](https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-023-10418-6)) highlights the role of socio-demographic and operational factors in missed appointments. Similarly, findings from ScienceDirect ([Link](https://www.sciencedirect.com/science/article/abs/pii/S0168851018300459)) emphasize the potential of system-level interventions, such as appointment reminder systems, to mitigate no-show rates. Complementary evidence from the National Institutes of Health ([Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7280239/)) demonstrates the cost-effectiveness and improved efficiency achieved through targeted patient communication strategies. Our project builds on this traditional research by integrating advanced feature engineering, unsupervised learning, Bayesian-optimized modeling, and robust causal inference techniques, offering a comprehensive, data-driven approach to both predict no-shows and understand the causal impact of interventions like SMS reminders.

---

## Business Impact and Project Importance

Medical appointment no-shows lead to significant inefficiencies in healthcare systems, including wasted resources, lost revenue, and compromised patient outcomes. By accurately predicting and understanding no-show behavior, healthcare providers can optimize scheduling, improve resource utilization, and enhance patient care. This project addresses these challenges through an end-to-end analytical framework that leverages:

- **[Advanced Feature Engineering](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/reports/part1_feature_engineering.md):** Extracts and transforms temporal, categorical, and numerical data to enhance predictive power.
- **[Unsupervised Learning](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/reports/part1_feature_engineering.md):** Utilizes clustering (KModes and HDBSCAN) to uncover latent group structures in patient behavior.
- **[Bayesian Optimization in Modeling](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/reports/part2_modelling.md):** Fine-tunes predictive models like XGBoost for improved performance.
- **[Robust Causal Inference](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/reports/part3_causal_inference_conclusion.md):** Employs Double Machine Learning and meta-learning techniques to estimate the impact of SMS reminders on no-show rates.

**Key Business Benefits:**
- **Operational Efficiency:** Reducing no-shows improves appointment scheduling and resource allocation.
- **Revenue Optimization:** Fewer missed appointments lead to better financial performance.
- **Enhanced Patient Outcomes:** Targeted interventions increase appointment adherence and overall care quality.
- **Data-Driven Strategies:** Provides actionable insights for designing effective patient engagement programs.

---

## Project Overview

This project is structured around three main components:

1. **Feature Engineering and Unsupervised Learning**  
   - **Techniques:**  
     - Temporal feature extraction from scheduling and appointment timestamps.
     - Binary encoding of categorical variables.
     - Log transformation to normalize skewed numerical features.
     - **Clustering:**  
       - *KModes* clustering to capture categorical groupings.
       - *HDBSCAN* with Bayesian optimization to adapt to varying data densities.
   - **Learn More:**  
     [Feature Engineering & Clustering Report](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/notebooks/Feature_Engineering_and_Clustering.ipynb)

2. **Advanced Modeling with Bayesian Optimization**  
   - **Techniques:**  
     - Propensity score estimation using Logistic Regression.
     - **XGBoost** modeling optimized via Bayesian hyperparameter tuning.
     - Deployment of advanced models such as **TabPFN** and **AutoTabPFN**.
   - **Outcome:**  
     Highly optimized predictive models that accurately identify patients at risk of no-shows.
   - **Learn More:**  
     [Modeling & Predictive Analytics Report](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/notebooks/Models_v2.ipynb)

3. **Causal Inference Analysis**  
   - **Techniques:**  
     - Double Machine Learning approaches (LinearDML, CausalForestDML) for estimating average treatment effects.
     - Meta-learners (T-Learner, X-Learner, S-Learner) to capture heterogeneous treatment effects.
     - Validation techniques including placebo tests and SHAP analysis for model interpretation.
   - **Outcome:**  
     Robust estimates of the causal impact of SMS reminders on reducing no-show rates.
   - **Learn More:**  
     [Causal Inference Analysis Report](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/notebooks/Casual_Inference_v5.ipynb)
