import pandas as pd
import numpy as np
from intervaltree import IntervalTree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
def load_data(fraud_file, ip_file, creditcard_file):
    """Load the datasets."""
    fraud_df = pd.read_csv(fraud_file)
    ip_df = pd.read_csv(ip_file)
    creditcard_df = pd.read_csv(creditcard_file)
    return fraud_df, ip_df, creditcard_df

def clean_data(df, dataset_name):
    """Handle missing values and duplicates."""
    print(f"Cleaning {dataset_name} dataset...")
    # Check for missing values
    print("Missing values:\n", df.isnull().sum())
    # Drop or impute missing values (here we drop for simplicity)
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Shape after cleaning: {df.shape}")
    return df

def merge_ip_data(fraud_df, ip_df):
    print("Merging IP address data...")
    fraud_df['ip_address_int'] = fraud_df['ip_address'].astype(int)
    
    # Create IntervalTree
    ip_tree = IntervalTree()
    for _, row in ip_df.iterrows():
        ip_tree[row['lower_bound_ip_address']:row['upper_bound_ip_address'] + 1] = row['country']
    
    # Map IP to country
    def get_country(ip):
        intervals = ip_tree[ip]
        return intervals.pop().data if intervals else 'Unknown'
    
    fraud_df['country'] = fraud_df['ip_address_int'].apply(get_country)
    fraud_df = fraud_df.drop(columns=['ip_address_int'])
    print("Merging completed.")
    return fraud_df

def preprocess_data(fraud_df, creditcard_df):
    """Preprocess datasets: encode categorical features and scale numerical features."""
    print("Preprocessing data...")
    
    # Define categorical and numerical columns
    fraud_categorical = ['source', 'browser', 'sex', 'country']
    fraud_numerical = ['purchase_value', 'age']
    creditcard_numerical = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                           'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), fraud_categorical),
            ('num', StandardScaler(), fraud_numerical)
        ])
    
    # Apply preprocessing to fraud dataset
    X_fraud = fraud_df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
    y_fraud = fraud_df['class']
    X_fraud_processed = preprocessor.fit_transform(X_fraud)
    
    # For creditcard dataset, only scale numerical features
    X_creditcard = creditcard_df.drop(columns=['Class', 'Time'])
    y_creditcard = creditcard_df['Class']
    scaler = StandardScaler()
    X_creditcard_processed = scaler.fit_transform(X_creditcard)
    
    return X_fraud_processed, y_fraud, X_creditcard_processed, y_creditcard, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)