Fraud Detection Project

## Overview

This project implements a fraud detection system using e-commerce and credit card transaction datasets. It includes data preprocessing, feature engineering, model training (Logistic Regression and XGBoost), and model explainability using SHAP.

## 1. Setup

Clone the repository:

```bash
https://github.com/moablex/fraud_detection.git
```

## 2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Place the datasets

- (Fraud_Data.csv, ipAddress_to_Country.csv, creditcard.csv) in the data/ directory.

## 4. Running the Project

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
│ ├── **init**.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│
├── notebooks/
│ ├── exploratory_data_analysis.ipynb
│
├── main.py
├── requirements.txt
├── README.md
```
