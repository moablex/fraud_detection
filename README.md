Fraud Detection Project

## Project Overview

This repository contains the code and documentation for a fraud detection project developed as part of a machine learning course. The project focuses on building, evaluating, and interpreting predictive models using the `Fraud_Data` and `creditcard` datasets. Key tasks include:

- **Task 1**: Data analysis and preprocessing, including loading datasets, cleaning data, merging IP addresses with country information using `IntervalTree`, and preparing features with one-hot encoding and standardization.
- **Task 2**: Model training and visualization using Logistic Regression and XGBoost, with SMOTE for handling imbalanced data, and evaluation using AUC-PR, F1-Score, and Confusion Matrices.
- **Task 3**: Model explainability using SHAP (Shapley Additive exPlanations) to interpret the best-performing XGBoost model, including Summary and Force Plots to identify key fraud drivers.

The project leverages Python, scikit-learn, XGBoost, and SHAP, with visualizations created using matplotlib and seaborn.

## Repository Link

[GitHub Repository](https://github.com/moablex/fraud_detection.git)

## 1. Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- A virtual environment tool (e.g., `venv` or `conda`)

Clone the repository:

```bash
https://github.com/moablex/fraud_detection.git
```

## 2.Create a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

## 3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Place the datasets

- (Fraud_Data.csv, ipAddress_to_Country.csv, creditcard.csv) in the data/ directory.

## 5. Running the Project

Run the main script:

```bash
python main.py(windows)
python3 main.py(linux or mac)
```

## Project strucure

```bash
fraud_detection_project/
│
├── data/
│ ├── Fraud_Data.csv
│ ├── ipAddress_to_Country.csv
│ ├── creditcard.csv
│
├── src/
│ ├── __init__.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
  ├── feature_engineering.py
│
├── notebooks/
│ ├── exploratory_data_analysis.ipynb
│ ├── model_explainability.ipynb
│ ├── model_training_visualization.ipynb
├── plots/
├── main.py
├── requirements.txt
├── README.md
```
