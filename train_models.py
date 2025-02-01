import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime

def standardize_date_format(df, date_column='Order Date'):
    """
    Convert various date formats to standard 'dd-mm-yyyy' format
    """
    def convert_date(date_str):
        try:
            # First try the mm/dd/yyyy format
            if isinstance(date_str, str) and '/' in date_str:
                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                return date_obj.strftime('%d-%m-%Y')

            # Try dd-mm-yyyy format
            elif isinstance(date_str, str) and '-' in date_str:
                # Verify if it's already in correct format
                datetime.strptime(date_str, '%d-%m-%Y')
                return date_str

            # If date is already a datetime object
            elif isinstance(date_str, datetime):
                return date_str.strftime('%d-%m-%Y')

            else:
                # Try parsing with pandas (handles more formats)
                return pd.to_datetime(date_str).strftime('%d-%m-%Y')

        except Exception as e:
            print(f"Error converting date '{date_str}': {str(e)}")
            return None

    df_copy = df.copy()
    df_copy[date_column] = df_copy[date_column].apply(convert_date)
    
    invalid_dates = df_copy[date_column].isna()
    if invalid_dates.any():
        print(f"Warning: {invalid_dates.sum()} dates could not be converted and will be removed")
        df_copy = df_copy.dropna(subset=[date_column])

    return df_copy

def train_and_save_models(csv_path):
    """
    Train models and save them as pickle files
    """
    try:
        # Read data
        print("Reading CSV file...")
        df = pd.read_csv(csv_path)
        
        # Standardize dates
        print("Standardizing dates...")
        df_standardized = standardize_date_format(df, date_column='Order Date')
        
        # Convert standardized dates to datetime objects
        print("Converting to datetime...")
        df_standardized['Order Date'] = pd.to_datetime(df_standardized['Order Date'], format='%d-%m-%Y')
        
        print("Calculating customer metrics...")
        # Create features
        current_date = df_standardized['Order Date'].max()
        
        # Calculate customer metrics
        customer_metrics = df_standardized.groupby('Customer Name').agg({
            'Order Date': lambda x: (current_date - x.max()).days,  # Recency
            'Order ID': 'count',  # Frequency
            'Sales': 'sum'  # Monetary
        }).reset_index()
        
        customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']
        
        print("Training models...")
        # Prepare data for clustering
        features = ['Recency', 'Frequency', 'Total_Monetary']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(customer_metrics[features])
        
        # Train KMeans model
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(features_scaled)
        
        # Train RandomForest for sales prediction
        purchase_matrix = pd.crosstab(df_standardized['Customer Name'], df_standardized['Category'])
        X = purchase_matrix.values
        y = df_standardized.groupby('Customer Name')['Sales'].mean().values
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        print("Saving models...")
        # Save models and scaler
        models = {
            'kmeans': kmeans,
            'scaler': scaler,
            'rf_model': rf_model,
            'feature_names': features,
            'categories': list(purchase_matrix.columns)
        }
        
        with open('retail_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print("Models trained and saved successfully!")
        
        # Print some basic statistics for verification
        print("\nBasic Statistics:")
        print(f"Total records processed: {len(df_standardized)}")
        print(f"Date range: {df_standardized['Order Date'].min()} to {df_standardized['Order Date'].max()}")
        print(f"Number of unique customers: {df_standardized['Customer Name'].nunique()}")
        print(f"Number of customer segments: {len(np.unique(kmeans.labels_))}")
        
        return models
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Replace with your CSV path
    csv_path = "DMart_Grocery_Sales_-_Retail_Analytics_Dataset.csv"
    
    try:
        print(f"Starting model training with file: {csv_path}")
        trained_models = train_and_save_models(csv_path)
        print("Process completed successfully!")
    except Exception as e:
        print(f"Failed to train models: {str(e)}")