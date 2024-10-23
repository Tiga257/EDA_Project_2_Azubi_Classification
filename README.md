
# Telco Churn Classification Project

### Table of Contents

- Project Overview
- Table of Contents
- Dataset Description
- Installation and Setup
- Project Structure
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Machine Learning Models
- Model Training and Evaluation
- Hyperparameter Tuning
- Results and Analysis
- Conclusion
- Future Work
- Technical Challenges and Solutions
- Contributors
- License
- Acknowledgements

## Project Overview

The Telco Churn Classification Project seeks to predict customer churn for a telecommunications company using various machine learning techniques. Churn prediction is critical as retaining existing customers is often more cost-effective than acquiring new ones. This project explores different algorithms to identify high-risk customers and provide actionable insights for improving customer retention strategies.

#### The project encompasses:

A thorough analysis of the dataset to identify key factors contributing to churn. Preprocessing steps for data cleaning, transformation, and feature engineering. Implementation of multiple machine learning algorithms with a focus on Random Forest and XGBoost, which yielded the best performance. Model evaluation using metrics such as accuracy, ROC-AUC, precision, recall, and F1-score. Recommendations based on findings to help the business reduce churn.

#### Dataset Description

The dataset comprises customer information, such as demographics, account details, and usage patterns. It has 21 features, with the target variable being Churn, indicating whether a customer left the service.

#### Key Features:

1. Customer Demographics:
Gender: Male or Female.
SeniorCitizen: Binary indicator (0 for non-senior, 1 for senior).
Partner: Whether the customer has a partner.
Dependents: Whether the customer has dependents.

2. Account Information:
Tenure: Number of months the customer has stayed with the company.
Contract: Type of contract (Month-to-month, One-year, Two-year).
PaperlessBilling: Whether the customer has paperless billing.
PaymentMethod: Method of payment (Electronic check, Mailed check, Bank transfer, Credit card).
MonthlyCharges: The monthly charge for the customer.
TotalCharges: Total amount charged to the customer.

3. Services Subscribed:
PhoneService, MultipleLines, InternetService (DSL, Fiber optic, No internet). Additional services like OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.

##### Target Variable:

Churn: Indicates if the customer left the company (Yes or No).

Installation and Setup
To set up the project environment, follow these instructions:

##### Clone the Repository:

bash git clone https://github.com/Tiga257/Telco-Churn-Classification.git

##### Navigate to the Project Directory:

bash cd Telco-Churn-Classification

##### Set Up a Virtual Environment:

bash python -m venv env source env/bin/activate # On Windows: env\Scripts\activate

##### Install the Dependencies:

bash pip install -r requirements.txt

##### Key dependencies include:

pandas, numpy: Data manipulation and numerical computation. scikit-learn, xgboost: Machine learning and model building. matplotlib, seaborn: Data visualization. imbalanced-learn: Handling imbalanced data.

##### Launch Jupyter Notebook:

bash jupyter notebook Open the provided notebook file EDA_Telco_Churn_Classification_Project_Prince_Okyere_Boadu.ipynb to explore the project.

#### Project Structure

The structure of the repository is as follows:


Telco-Churn-Classification/
* ├── data/                          # Directory for datasets
* │   ├── Telco-Churn-Dataset.xlsx   # Original dataset file
* │   └── Preprocessed_Data.csv      # Preprocessed data file (if available)
* ├── notebooks/                     # Jupyter notebooks for EDA and modeling
* │   └── EDA_Telco_Churn_Classification_Project_Prince_Okyere_Boadu.ipynb
* ├── models/                        # Trained models storage (optional)
* ├── reports/                       # Reports and analysis results
* ├── requirements.txt               # Python package dependencies
* └── README.md                      # Project documentation
     
Exploratory Data Analysis (EDA)
EDA involves examining the dataset to uncover patterns and insights, including:

#### Data Cleaning:

Handling missing values in TotalCharges, which are replaced with median values or removed.

#### Data Visualization:

Use histograms, bar plots, and count plots to understand the distribution of categorical and numerical variables. Analyze correlations using heatmaps to identify relationships between features and the target variable. Examine churn rates across different groups (e.g., customers with different contract types, payment methods).

##### Statistical Analysis:

Perform tests like chi-square tests for categorical features to determine the statistical significance of churn-related factors. Identify skewness in numerical features, applying transformations if necessary.

#### Data Preprocessing

Steps to prepare the data for modeling include:

Handling Missing Values: Impute missing values in TotalCharges based on customer tenure and monthly charges.

Encoding Categorical Features: Apply One-Hot Encoding to nominal categorical features (e.g., PaymentMethod, Contract). Use Label Encoding for binary features (gender, Partner, Dependents).

Feature Scaling: Standardize numerical features like MonthlyCharges and TotalCharges using MinMaxScaler to normalize the range.

Dealing with Imbalanced Data: Employ Synthetic Minority Over-sampling Technique (SMOTE) to balance the minority (Churned) class.

Machine Learning Models
Several machine learning models were employed to predict churn, including:

Random Forest: This ensemble model of decision trees aggregates individual tree predictions to improve accuracy and generalization. It was chosen for its ability to handle complex data and its robustness to overfitting.

XGBoost: An efficient gradient boosting algorithm known for its speed and performance in structured data problems. It leverages gradient boosting framework for better accuracy and handles missing values effectively.

#### Other Models Tested:

Logistic Regression: A baseline model for binary classification.

Support Vector Machine (SVM): Known for its capability to handle high-dimensional data.

K-Nearest Neighbors (KNN): Classifies samples based on nearest training data points.

Decision Trees: Individual trees used for interpretability.

LightGBM and CatBoost: Gradient boosting models designed for better performance and speed.

#### Model Training and Evaluation

Splitting the Data:
Use an 80-20 split for training and testing sets.
Apply Stratified K-Fold cross-validation to ensure that each fold contains the same proportion of classes.

#### Evaluation Metrics:

Accuracy: The percentage of correctly classified instances.
Precision, Recall, F1-Score: Measure the effectiveness in identifying the churned customers.
ROC-AUC Score: Evaluate the ability of the model to discriminate between churned and non-churned customers.
Confusion Matrix: Analyze the distribution of true positives, false positives, true negatives, and false negatives.

#### Hyperparameter Tuning

Random Forest Hyperparameter Tuning:
Parameters such as n_estimators, max_depth, min_samples_split, and min_samples_leaf were tuned using GridSearchCV.

XGBoost Hyperparameter Tuning:
Tuned parameters include learning_rate, n_estimators, max_depth, and gamma.
RandomizedSearchCV was used for faster tuning.
Results and Analysis
Best Performing Models: Random Forest and XGBoost.

Random Forest:
Achieved an accuracy of ~82% and a ROC-AUC score of 0.85.
Feature importance analysis showed that Contract, tenure, and MonthlyCharges were among the top predictors.

XGBoost:
Outperformed other models with an accuracy of ~84% and a ROC-AUC score of 0.87.
Better handling of class imbalance due to built-in techniques like scale_pos_weight.
Feature Importance: Contract, tenure, and MonthlyCharges were consistently important across models.
Misclassification Analysis: Examined false positives and false negatives to understand model limitations and areas for improvement.

#### Conclusion

The project successfully identified key factors influencing churn and used machine learning models to predict it with high accuracy.
Random Forest and XGBoost models proved to be the most effective due to their robustness and performance.
Insights from the models can guide targeted interventions to reduce customer churn.

#### Future Work

Expand Data Sources: Incorporate additional data such as customer service interactions and satisfaction scores.

Deploy the Model: Integrate the model into a real-time dashboard for churn prediction.

Automated Feature Engineering: Explore automated techniques to create more predictive features.
Technical Challenges and Solutions

Class Imbalance: Addressed using oversampling techniques like SMOTE.

Handling Missing Values: Applied imputation strategies based on feature relationships.

Model Overfitting: Regularization techniques and ensemble methods were used to reduce overfitting.

###### Contributors

Prince Okyere Boadu: Data analysis, modeling, hyperparameter tuning, and documentation.
License
Licensed under the MIT License. See LICENSE file for details.

###### Acknowledgements

Kaggle Dataset: For providing the Telco Churn data.
Libraries: pandas, scikit-learn, xgboost, matplotlib, and others for data science tools.
