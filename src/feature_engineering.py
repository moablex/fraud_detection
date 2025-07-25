import pandas as pd
from datetime import datetime

def engineer_features(fraud_df):
    """Engineer features for Fraud_Data dataset."""
    print("Engineering features...")
    
    # Convert timestamps to datetime
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    
    # Time-based features
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600  # in hours
    
    # Transaction frequency and velocity
    user_freq = fraud_df.groupby('user_id').size().reset_index(name='transaction_count')
    fraud_df = fraud_df.merge(user_freq, on='user_id', how='left')
    
    # Velocity: purchase_value per hour since signup
    fraud_df['velocity'] = fraud_df['purchase_value'] / (fraud_df['time_since_signup'] + 1e-6)  # Avoid division by zero
    
    return fraud_df