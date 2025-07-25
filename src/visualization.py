import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_eda(fraud_df, creditcard_df, output_dir='plots'):
    """Generate EDA visualizations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Class distribution for Fraud_Data
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=fraud_df)
    plt.title('Class Distribution in Fraud_Data')
    plt.savefig(f"{output_dir}/fraud_class_distribution.png")
    plt.close()
    
    # Class distribution for creditcard
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=creditcard_df)
    plt.title('Class Distribution in Creditcard')
    plt.savefig(f"{output_dir}/creditcard_class_distribution.png")
    plt.close()
    
    # Purchase value distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(fraud_df['purchase_value'], bins=30)
    plt.title('Purchase Value Distribution')
    plt.savefig(f"{output_dir}/purchase_value_distribution.png")
    plt.close()