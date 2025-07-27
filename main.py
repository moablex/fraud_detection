from src.data_preprocessing import load_data, clean_data, merge_ip_data, preprocess_data, split_data
from src.feature_engineering import engineer_features
from src.visualization import plot_eda
import os

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression  # Added
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix  # Added
import numpy as np
def main():
    # Create output directory
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    fraud_df, ip_df, creditcard_df = load_data(
        'data/Fraud_Data.csv',
        'data/IpAddress_to_Country.csv',
        'data/creditcard.csv'
    )
    print("Data loaded successfully.")
    
    # Clean data
    fraud_df = clean_data(fraud_df, 'Fraud_Data')
    creditcard_df = clean_data(creditcard_df, 'Creditcard')
    print("Data cleaned successfully.")
    
    # Merge IP data
    fraud_df = merge_ip_data(fraud_df, ip_df)
    print("IP data merged successfully.")
    
    # Feature engineering
    fraud_df = engineer_features(fraud_df)
    print("Features engineered successfully.")
    
    # EDA visualizations
    plot_eda(fraud_df, creditcard_df, output_dir)
    print("EDA visualizations completed.")
    
    # Preprocess data
    X_fraud, y_fraud, X_creditcard, y_creditcard, preprocessor = preprocess_data(fraud_df, creditcard_df)
    print("Data preprocessed successfully.")
    
    # Split data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)
    X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = split_data(X_creditcard, y_creditcard)
    print("Data split successfully.")

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_fraud_res, y_train_fraud_res = smote.fit_resample(X_train_fraud, y_train_fraud)
    X_train_creditcard_res, y_train_creditcard_res = smote.fit_resample(X_train_creditcard, y_train_creditcard)
    print("SMOTE applied successfully.")
    
    # Model Training and Evaluation
    def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{dataset_name} - {model.__class__.__name__} Results:")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        return auc_pr, f1

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    print("\nTraining Logistic Regression...")
    auc_pr_lr_fraud, f1_lr_fraud = evaluate_model(lr, X_train_fraud_res, X_test_fraud, y_train_fraud_res, y_test_fraud, "Fraud_Data")
    auc_pr_lr_credit, f1_lr_credit = evaluate_model(lr, X_train_creditcard_res, X_test_creditcard, y_train_creditcard_res, y_test_creditcard, "Creditcard")

    # XGBoost
    xgb = XGBClassifier(random_state=42, scale_pos_weight=10)
    print("\nTraining XGBoost...")
    auc_pr_xgb_fraud, f1_xgb_fraud = evaluate_model(xgb, X_train_fraud_res, X_test_fraud, y_train_fraud_res, y_test_fraud, "Fraud_Data")
    auc_pr_xgb_credit, f1_xgb_credit = evaluate_model(xgb, X_train_creditcard_res, X_test_creditcard, y_train_creditcard_res, y_test_creditcard, "Creditcard")

    # Model Comparison and Justification
    print("\nModel Comparison:")
    models = [("Logistic Regression", auc_pr_lr_fraud, f1_lr_fraud, auc_pr_lr_credit, f1_lr_credit),
              ("XGBoost", auc_pr_xgb_fraud, f1_xgb_fraud, auc_pr_xgb_credit, f1_xgb_credit)]
    for name, auc_pr_fraud, f1_fraud, auc_pr_credit, f1_credit in models:
        print(f"{name}: Fraud_Data (AUC-PR: {auc_pr_fraud:.4f}, F1: {f1_fraud:.4f}), "
              f"Creditcard (AUC-PR: {auc_pr_credit:.4f}, F1: {f1_credit:.4f})")
    
    best_model = max(models, key=lambda x: (x[1] + x[3]) / 2)
    print(f"\nBest Model: {best_model[0]} - Chosen based on highest average AUC-PR across both datasets, "
          f"indicating better balance of precision and recall for imbalanced data.")

if __name__ == "__main__":
    main()

