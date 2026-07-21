Loan Risk Analysis & Prediction

A machine learning project that cleans raw loan application data, explores key risk factors through EDA, and trains a classification model to predict loan status (risk of default).

Overview

This project takes a raw loan dataset (Case_Data.csv), cleans and prepares it for analysis, explores it visually, and then trains a Logistic Regression pipeline that predicts loan_status from applicant and loan attributes. The trained model is serialized with pickle for reuse.

Project Workflow

Part 1 — Data Cleaning & EDA

Load the raw dataset and inspect its shape and missing values
Drop columns with more than 50% missing values
Convert data types (e.g. term to numeric months, date columns to datetime)
Impute remaining missing values (median for numerical columns, mode / "Unknown" for categorical columns)
Explore the distribution of key variables such as loan amount, interest rate, and home ownership
Save the cleaned dataset to cleaned_case_data.csv

Part 2 — Loan Risk Predictor

Load the cleaned dataset
Split into train/test sets (75/25)
Build a preprocessing pipeline (StandardScaler for numerical features, OneHotEncoder for categorical features) using ColumnTransformer
Train a LogisticRegression model inside a single sklearn Pipeline
Evaluate accuracy on the test set
Save the trained pipeline to loan_risk_model.pkl
Project Structure
├── Loan_Risk_Analysis_Prediction.ipynb   # Main notebook (cleaning, EDA, modeling)
├── Case_Data.csv                         # Raw input dataset (not included)
├── cleaned_case_data.csv                 # Cleaned dataset (generated)
├── loan_risk_model.pkl                   # Trained model pipeline (generated)
├── requirements.txt
└── README.md
Requirements
Python 3.9+
See requirements.txt for package dependencies
Setup
bash
# Clone or download the project
cd loan-risk-analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Usage
Place your raw dataset at Case_Data.csv in the project directory (or update the file path in the notebook).
Open the notebook:
bash
   jupyter notebook Loan_Risk_Analysis_Prediction.ipynb
Run all cells in order. This will:
Produce cleaned_case_data.csv
Produce loan_risk_model.pkl
Print the model's accuracy on the held-out test set
Model
Algorithm: Logistic Regression (solver='liblinear')
Preprocessing: StandardScaler on numerical features, OneHotEncoder on categorical features, combined via ColumnTransformer
Target variable: loan_status
Evaluation metric: Accuracy
Notes
Date columns (issue_d, earliest_cr_line, last_pymnt_d, last_credit_pull_d) are parsed during cleaning but dropped before modeling, since they were found to be entirely non-informative (NaT) at training time in this run — check your own dataset and adjust as needed.
The saved model (loan_risk_model.pkl) includes the full preprocessing pipeline, so raw (uncleaned-format) feature rows matching the training schema can be passed directly to .predict().

Author
Pranav Ingole
