# Loan-Credit-Risk-Analysis

A complete end-to-end machine learning project focused on analyzing borrower characteristics, identifying default patterns, and building a predictive model for loan default risk using XGBoost.

---
# Loan Credit Risk Analysis & Default Prediction
This project focuses on analyzing credit risk and predicting loan defaults using borrower demographic, financial, and credit-related data. It includes a full exploratory analysis, model development, and a FastAPI web application for real-time prediction.

---

## Project Structure

```
Loan-Credit-Risk_Analysis
│   credit_risk_analysis.ipynb        -> Exploratory analysis and model development
│   credit_risk_dataset.csv           -> Dataset used for training
│   loan_default_model.pkl            -> Final trained XGBoost model
│
└───Loan_Predictor
    │   app.py                        -> FastAPI application for predictions
    │   test.py                       -> Script for local model testing
    │
    ├───templates
    │       index.html                -> Web form for input (Jinja2 template)
    │
    └───__pycache__
```

---

## Overview

The goal of this project is to build a machine learning model that can predict whether a borrower will default on a loan based on available features such as age, income, employment length, loan amount, and borrowing intent.

The project includes:

* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Model training and comparison
* Deployment through a FastAPI application

---

## Data Processing

Key steps:

* Removal of outliers in age and employment length
* One-hot encoding of categorical variables
* Dropping redundant or misleading features (such as interest rate)
* Filling missing values
* Creating consistent feature alignment between training and deployment

---

## Machine Learning Models

Several models were tested, including:

* Logistic Regression
* Random Forest
* XGBoost (final model)

### Final Model Performance (XGBoost)

```
Confusion Matrix:
[[5047   46]
 [ 396 1026]]

Accuracy:       0.93
ROC-AUC Score:  0.948
```

The model performs well on this dataset and can identify defaulting borrowers with strong discriminative ability. However, predictions are limited by the available features.

---

## Limitations

For real-world loan underwriting, the dataset lacks several critical financial features such as:

* Loan tenure
* Monthly EMI
* True debt-to-income ratio (DTI)
* Total existing debts
* Credit utilization
* Payment history details

Without these variables, the model cannot fully replicate a production-grade credit scoring system, though it performs well within the limitations of the provided dataset.

---

## FastAPI Web Application

Located in the `Loan_Predictor/` folder.

Features:

* Web form for user input
* Automatic calculation of loan_percent_income
* Real-time prediction with probability

Run the app with:

```bash
uvicorn app:app --reload
```

---

If you want, I can also prepare a shorter READ
