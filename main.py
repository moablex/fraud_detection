from src.data_preprocessing import load_data, clean_data, merge_ip_data, preprocess_data, split_data
from src.feature_engineering import engineer_features
from src.visualization import plot_eda
import os

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
    
if __name__ == "__main__":
    main()
