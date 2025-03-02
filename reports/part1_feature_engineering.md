# **Feature Engineering and Clustering for No-Show Appointments**

## **Overview**

This section of the project aims to enhance predictive modeling for medical appointment no-shows by leveraging advanced **feature engineering** and **unsupervised learning techniques**. Given that the dataset consists primarily of **categorical and binary variables**, we employ specialized feature transformations to maximize the predictive power of structured data. Our methodology includes:

- **Extracting temporal patterns** from scheduling and appointment timestamps
- **Encoding categorical features efficiently** while preserving interpretability
- **Transforming numerical variables** to correct skewness and improve model performance
- **Leveraging unsupervised learning** via clustering (KModes, HDBSCAN) to uncover latent group structures
- **Optimizing hyperparameters** for clustering via Bayesian optimization

By integrating these techniques, we generate a refined dataset that improves downstream predictive analytics while capturing nuanced patterns in patient scheduling behavior.

---

## **Data Preprocessing & Feature Engineering**

### **1. Data Ingestion & Temporal Feature Extraction**
- The dataset is loaded from a CSV file (Original_data.csv), ensuring robust error handling for data inconsistencies.
- **Datetime fields (`ScheduledDay`, `AppointmentDay`)** are converted to datetime objects to enable rich feature extraction.
- The following **temporal attributes** are derived:
  - **ScheduledMonth & AppointmentMonth**: Capture seasonal trends in patient appointments.
  - **ScheduledWeekday & AppointmentWeekday**: Identifies patterns in scheduling behaviors (e.g., weekday vs. weekend trends).
  - **Time-of-Month Binning**: Each appointment date is categorized into *Beginning (1st–10th)*, *Middle (11th–20th)*, or *End (21st+)* to detect potential cyclical patterns in scheduling.

### **2. Binary & Ordinal Encoding**
- **Target Variable (`No-show`)**: Converted into a binary format:
  - `"Yes"` → `1` (Missed appointment)
  - `"No"` → `0` (Attended appointment)
- **Gender Encoding**:
  - `"F"` → `1`, `"M"` → `0` (with missing values defaulted to `-1`).
  - This ensures downstream models can learn gender-based trends in attendance rates.

### **3. Wait Time Computation & Normalization**
- **WaitTime Calculation**: Computed as the difference (in days) between `AppointmentDay` and `ScheduledDay`, capturing the lead time between scheduling and actual appointment.
- **Log Transformation (`WaitTime_Log`)**:
  - Since **raw wait times exhibit right-skewed distributions**, we apply a log transformation to normalize the feature:
    ```math
    \text{WaitTime\_Log} = \log(1 + \text{WaitTime})
    ```
  - This stabilizes variance and improves model interpretability.

### **4. Categorical Feature Encoding**
- **One-Hot Encoding for `Neighbourhood`**:
  - Converts the `Neighbourhood` categorical variable into binary indicator columns, preserving **spatial locality effects** in patient attendance.
  - High-cardinality categorical features like `Neighbourhood` often contain valuable signals in clustering and predictive modeling.

---

## **Unsupervised Learning for Feature Enrichment: Clustering Analysis**

Since the dataset is dominated by **categorical and binary features**, traditional clustering methods like **K-Means** and **Gaussian Mixture Models (GMM)** are **not suitable** due to their reliance on Euclidean distances. Instead, we employ clustering techniques that **explicitly handle categorical variables** and **enhance feature representation** for downstream analysis.

### **1. KModes Clustering: Capturing Categorical Groupings**
- Unlike K-Means, which relies on numerical distance metrics, **KModes uses categorical similarity (mode matching)** to partition data into clusters.
- **Implementation Steps**:
  - A **range of cluster sizes (K=2 to K=10)** is evaluated using the **cost function** (lower cost = better clustering).
  - The optimal number of clusters (K=4) is identified using an **elbow method** based on the **rate of cost reduction**.
  - Each patient is assigned a **KModes cluster label**, adding an **unsupervised categorical feature** to the dataset.

### **2. HDBSCAN: Density-Based Clustering with Bayesian Optimization**
- Traditional clustering methods struggle with **varying-density** clusters, but HDBSCAN **dynamically adapts** to different density regions, making it highly effective for **binary and categorical data**.
- **Optimized Parameter Selection**:
  - **Bayesian Optimization** is applied to fine-tune:
    - `min_cluster_size`: Minimum number of points per cluster (prevents over-fragmentation).
    - `min_samples`: Controls how robust the clusters should be against noise.
  - The **Hamming distance metric** is used to measure dissimilarity in categorical feature space.
  - The **percentage of data points classified as noise** is analyzed to balance **cluster purity** and **coverage**.
- **Cluster Assignments**:
  - Each patient is assigned an **HDBSCAN cluster label**, enriching the dataset with **latent group identifiers** that capture behavioral similarities.

---

## **Final Feature Set**
The engineered dataset contains:
**Temporal Features**: Month, weekday, and time-of-month bins  
**Binary & Ordinal Encodings**: Gender, no-show status, medical conditions  
**Wait Time Adjustments**: Log-transformed wait time  
**Spatial Encoding**: One-hot encoded `Neighbourhood`  
**Unsupervised Features**: Cluster labels from **KModes** and **HDBSCAN**

These engineered features **enhance predictive modeling** by incorporating **behavioral trends, spatial patterns, and latent clusters**, ultimately improving the model’s ability to anticipate **no-show risks**.
