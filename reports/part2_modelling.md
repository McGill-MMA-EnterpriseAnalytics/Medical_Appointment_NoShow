# Modeling Section Report

This report outlines the complete modeling workflow implemented in the [notebook](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/notebooks/Models_v2.ipynb). It covers data preprocessing for modeling, propensity score estimation, resampling with SMOTE, and the training and evaluation of several machine learning models including Logistic Regression (for propensity scoring), XGBoost (with Bayesian optimization), TabPFN, and AutoTabPFN. Detailed interpretations for responses and visualizations are provided, along with hyperlinks to the graphs stored in our GitHub repository.

---

## 1. Data Preparation and Preprocessing

### Data Loading
- **Data Import:**  
  Training and test datasets are imported from CSV files [train_processed_v4.csv](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/Data/train_processed_v4.csv.zip) and [test_processed_v4.csv](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/Data/test_processed_v4.csv). The target variable `No-show` is separated from the training features, and irrelevant identifiers (e.g., `PatientId`, `AppointmentID`) are removed to prevent data leakage.

### Feature Identification and Transformation
- **Preprocessing Pipeline:**  
  A `ColumnTransformer` is constructed with two pipelines:
  - **Numerical Pipeline:**  
    - **Imputation:** Missing values are replaced using the median strategy via `SimpleImputer`.
    - **Scaling:** Features are scaled using `StandardScaler`.
  - **Categorical Pipeline:**  
    - **Imputation:** Missing categorical values are replaced with the most frequent value.
    - **Encoding:** Features are transformed using `OneHotEncoder` (with `handle_unknown='ignore'`).

This pipeline is fitted on the training set and then applied to both the training and test datasets, ensuring consistency.

---

## 2. Propensity Score Modeling

### Purpose and Methodology
- **Objective:**  
  Based on the input features, estimate the likelihood (propensity score) of a patient not showing up.
- **Modeling Approach:**  
  A Logistic Regression model (with an extended iteration count for convergence) is used to compute these scores.

### Visualization and Interpretation
- **Graph:**  
  The distribution of propensity scores, separated by the `No-show` label, is plotted using `seaborn`.  
  - **Link to Graph:** [Propensity Score Distribution](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/propensity_score_distribution.png)
- **Interpretation:**  
  The histogram shows distinct distributions for no-show and show groups. The mean propensity score for the no-show group is approximately 0.1867, while that for the show group is around 0.2630. This indicates that, on average, patients who do show up have higher propensity scores, which is consistent with the model's estimation mechanism.

---

## 3. Resampling with SMOTE for Classification

- **SMOTE Application:**  
  SMOTE is applied to the processed training data to balance the class distribution. The transformation yields a new training set (`X_train_cls` and `y_train_cls`) with an improved class balance.

---

## 4. XGBoost Model with **Bayesian Optimization**

### Model Setup and Hyperparameter Tuning
- **Base Model:**  
  An XGBoost classifier configured with a binary logistic objective is used.
- **Bayesian Optimization:**  
  - **Search Space:**  
    The optimization process tunes parameters such as `learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `gamma`, `subsample`, and `colsample_bytree`.
  - **Cross-Validation:**  
    A 5-fold cross-validation is used, with the F1 score as the metric for optimization.
  - **Outcome:**  
    The optimal parameters identified include:
    - `colsample_bytree`: ~0.8556
    - `gamma`: ~0.1414
    - `learning_rate`: 0.3
    - `max_depth`: 10
    - `min_child_weight`: 2
    - `n_estimators`: 500
    - `subsample`: ~0.6854

### Validation Curve
- **Graph:**  
  A validation curve displaying the mean F1 score over 20 optimization iterations is generated.  
  - **Link to Graph:** [XGBoost Validation Curve](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/xgboost_validation_curve.png)

---

## 5. Feature Importance Analysis

### Insights from XGBoost
- **Top Features:**  
  The most influential features include:
  - `ScheduledDay_Weekday_Saturday`
  - `WaitTime_Log`
  - `Age`
  - `AppointmentDay_Weekday_Friday`
  - Additional weekday-related features.
- **Graph:**  
  A bar plot of the top 20 most important features is generated.  
  - **Link to Graph:** [Feature Importance Plot]([https://github.com/YourRepo/graphs/feature_importance.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Medical_Appointment_NoShow/blob/main/visualizations/feature_importance.png))
- **Interpretation:**  
  From the feature importance plot, we can see that the scheduling date, appointment date, and weekday of the appointment play a crucial role in predicting no-show rates. Due to the nature of the XGBoost model, we cannot directly interpret the exact impact of these features on the target. However, further causal inference analysis could provide deeper insights into how these factors influence patient attendance.
  Additionally, WaitTime and patient Age also significantly affect whether a patient is likely to show up for their appointment.

---

## 6. Test Set Evaluation

### XGBoost Evaluation
- **Metrics:**  
  - **Accuracy:** 0.7237  
  - **Precision:** 0.3334  
  - **Recall:** 0.3725  
  - **F1-Score:** 0.3519
- **Interpretation:**  
  Although the accuracy is relatively high, the low precision and recall indicate that the model struggles to correctly identify no-show cases. This suggests further tuning or alternative approaches may be needed to better capture the nuances of the minority class.

### TabPFN Evaluation
- **Model Training:**  
  The TabPFN model is trained on the resampled dataset (with downsampling applied for efficiency).
- **Metrics:**  
  - **Accuracy:** 0.6607  
  - **Precision:** 0.3272  
  - **Recall:** 0.6486  
  - **F1-Score:** 0.4350
- **Interpretation:**  
  The TabPFN model shows improved recall compared to XGBoost, which may be valuable in scenarios where capturing no-show cases is critical. However, the lower precision indicates an increased rate of false positives.

### AutoTabPFN Evaluation
- **Model Training:**  
  AutoTabPFN is trained under a strict time constraint (120 seconds) using the same resampled training data.
- **Metrics:**  
  - **Accuracy:** 0.6334  
  - **Precision:** 0.7790  
  - **Recall:** 0.6334  
  - **F1-Score:** 0.6696  
- **Interpretation:**  
  AutoTabPFN demonstrates a higher precision, suggesting that when it predicts a no-show, it is more likely to be correct. The balance between accuracy, recall, and F1-score indicates that AutoTabPFN is competitive, particularly in reducing false positives. 

---

## 7. Conclusion and Interpretation

- **Model Comparison:**  
  - **XGBoost:** Provides competitive accuracy with insights from Bayesian optimization and feature importance analysis, although it shows limitations in recall.
  - **TabPFN:** Offers improved recall at the cost of precision, which may be preferable depending on operational priorities.
  - **AutoTabPFN:** Strikes a balanced performance with high precision and competitive F1 scores, making it a strong candidate for scenarios where reducing false positives is essential.
