# Telco Customer Churn & Retention Analysis

## Introduction
Welcome to the Telco Customer Churn & Retention Analysis project. In the competitive telecommunications sector, retaining existing customers is far more cost-effective than acquiring new ones. This project focuses on building a robust predictive system to identify customers at risk of attrition (churn) and providing actionable insights to inform retention strategies.

## Dataset
The dataset used in this project is the **Telco Customer Churn** dataset, available [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It includes customer demographics, account information, service subscriptions, and their churn status.

## Project Objectives
The primary goals of this analysis were:

1.  **Predictive Modeling:** Build a high-performance machine learning model to classify customers as "Churn" or "Non-Churn" with high recall.
2.  **Handling Imbalance:** Address the significant class imbalance (73:27) in the dataset to prevent model bias.
3.  **Explainability:** Use advanced interpretability tools to understand *why* specific customers are leaving.
4.  **Actionable Insights:** Identify key churn drivers to recommend targeted business strategies for retention.

## Technologies Used
* **Python 3.x**
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Imbalance Handling:** Imbalanced-learn (SMOTE)
* **Model Interpretability:** SHAP (SHapley Additive exPlanations)

## Methodology

### 1. Data Preprocessing
To ensure model robustness, the following steps were taken:
* **Data Cleaning:** Handled missing values and verified data integrity.
* **Feature Encoding:** Converted categorical variables (e.g., gender, payment method) into numerical formats.
* **Scaling:** Applied standard scaling to numerical features.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied SMOTE to the training data to address the class imbalance, ensuring the model learned to identify the minority class (Churners) effectively.

### 2. Exploratory Data Analysis (EDA)
Conducted rigorous EDA to uncover patterns:
* Analyzed churn distribution across different demographics (Senior Citizens, Partners).
* Investigated the relationship between services (Fiber Optic, Streaming TV) and attrition rates.
* Visualized correlation matrices to identify redundant features.

### 3. Model Building
The core of the project involved training a **Tuned XGBoost Classifier**.
* **Why XGBoost?** Chosen for its speed, performance, and ability to handle complex non-linear relationships.
* **Hyperparameter Tuning:** Optimized parameters (learning rate, max depth, n_estimators) to prevent overfitting on the SMOTE-resampled data.

## Model Evaluation
The XGBoost model achieved strong performance metrics on the test set:

* **Accuracy:** 85%
* **ROC-AUC Score:** 0.84
* **Recall:** 0.78 (Crucial for minimizing false negatives, i.e., missing a customer who is about to churn).

*Visualizations included:*
* **Confusion Matrix:** To visualize True Positives vs. False Negatives.
* **ROC Curve:** To assess the trade-off between sensitivity and specificity.

## Key Insights (SHAP Analysis)
Using **SHAP (SHapley Additive exPlanations)**, we moved beyond "black-box" predictions to identify the root causes of churn:

1.  **Contract Type:** Customers with **"Month-to-Month"** contracts are at the highest risk of churning.
2.  **Internet Service:** Users with **"Fiber Optic"** service showed significantly higher attrition rates, suggesting potential dissatisfaction with price or service quality.
3.  **Tenure:** Newer customers are more volatile; churn risk decreases significantly as tenure increases.

**Recommendation:** The business should focus on migrating Month-to-Month users to longer-term contracts via incentives and investigating service quality issues in the Fiber Optic infrastructure.

## How to Use
To replicate this analysis on your local machine:

1.  **Clone the repository:**
    git clone https://github.com/arnab37seal/Telco-Customer-Churn-Prediction.git

2.  **Install dependencies:**
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap

3.  **Run the Notebook:**
    Open `Telco.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## Future Enhancements
* **Deployment:** Deploy the model as a Flask/Streamlit web app for real-time predictions by support agents.
* **Cost-Benefit Analysis:** Calculate the financial impact of false positives vs. false negatives to fine-tune the decision threshold.
* **Deep Learning:** Experiment with Neural Networks (ANNs) to see if performance can be further improved.

## License
This project is licensed under the [MIT License](LICENSE).

## Author
**Arnab Seal**
* [LinkedIn Profile](https://www.linkedin.com/in/arnab-seal37/)
* [GitHub Profile](https://github.com/arnab37seal)
