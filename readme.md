# Customer Churn Prediction

## Overview
This project implements a machine learning solution to predict customer churn for a telecommunications company. Using various machine learning algorithms including Random Forest, Decision Tree, and CatBoost, the model helps identify customers who are likely to discontinue their services.

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and encoding
- Model training with multiple algorithms
- Hyperparameter tuning using GridSearchCV
- Model evaluation and comparison
- Handling imbalanced dataset using SMOTE

## Dataset
The project uses the Telco Customer Churn dataset which includes information about:
- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, tech support, streaming)
- Billing information (monthly charges, total charges)

## Project Structure
├── models/
│   ├── customer_churn_model.pkl
│   └── encoders.pkl
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── requirements.txt
└── README.md

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install required packages:
```bash
pip install -r requirements.txt

## Usage
Ensure you have the dataset file "customer_churn.csv" in your working directory
Run the Jupyter notebook:
```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
