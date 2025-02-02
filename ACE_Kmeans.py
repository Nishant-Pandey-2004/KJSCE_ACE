# Description: This script contains the code for the K-means clustering analysis of the DMart Grocery Sales dataset.

"""# Function to Standardize the dates (Imp. Pre-processing Step)"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def standardize_date_format(df, date_column='Order Date'):
    """
    Convert various date formats to standard 'dd-mm-yyyy' format

    Parameters:
    df (pandas.DataFrame): Input dataframe
    date_column (str): Name of the date column to standardize

    Returns:
    pandas.DataFrame: DataFrame with standardized date format
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

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Convert dates
    df_copy[date_column] = df_copy[date_column].apply(convert_date)

    # Remove any rows where date conversion failed
    invalid_dates = df_copy[date_column].isna()
    if invalid_dates.any():
        print(f"Warning: {invalid_dates.sum()} dates could not be converted and will be removed")
        df_copy = df_copy.dropna(subset=[date_column])

    return df_copy

"""# 1. Data Loading and Initial Preprocessing"""

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the retail dataset
    """
    # Load data
    df = pd.read_csv('DMart_Grocery_Sales_-_Retail_Analytics_Dataset.csv')

    # Convert dates
    df = standardize_date_format(df, date_column='Order Date')

    # Convert Order Date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')

    # Create time-based features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day of Week'] = df['Order Date'].dt.dayofweek
    df['Is Weekend'] = df['Day of Week'].isin([5, 6]).astype(int)

    # Calculate customer metrics
    customer_metrics = calculate_customer_metrics(df)
    df = df.merge(customer_metrics, on='Customer Name')

    return df

def calculate_customer_metrics(df):
    """
    Calculate RFM metrics for each customer
    """
    # Current date for recency calculation
    current_date = df['Order Date'].max()

    customer_metrics = df.groupby('Customer Name').agg({
        'Order Date': lambda x: (current_date - x.max()).days,  # Recency
        'Order ID': 'count',  # Frequency
        'Sales': 'sum'  # Monetary
    }).reset_index()

    customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']
    return customer_metrics

"""# 2. Customer Segmentation"""

def perform_customer_segmentation(df, n_clusters=4):
    """
    Perform customer segmentation using RFM metrics
    """
    # Select features for clustering
    features = ['Recency', 'Frequency', 'Total_Monetary']

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[features])

    # Find optimal number of clusters
    silhouette_scores = []
    K = range(2, 8)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = K[np.argmax(silhouette_scores)]

    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Customer_Segment'] = kmeans.fit_predict(features_scaled)

    return df, kmeans

"""# 3. Sales Forecasting"""

def create_sales_forecast(df, forecast_periods=30):
    """
    Create sales forecast using Prophet
    """
    # Prepare data for Prophet
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    # Add holiday effects
    model.add_country_holidays(country_name='IN')

    # Fit model
    model.fit(daily_sales)

    # Create future dates dataframe
    future_dates = model.make_future_dataframe(periods=forecast_periods)

    # Generate forecast
    forecast = model.predict(future_dates)

    return model, forecast

"""# 4. Store Placement Analysis"""

def analyze_store_placement(df):
    """
    Analyze optimal store placement based on various metrics
    """
    store_analysis = df.groupby(['Region', 'City']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count',
        'Customer Name': 'nunique',
        'Discount': 'mean'
    }).reset_index()

    # Calculate performance metrics
    store_analysis['Sales_per_Customer'] = store_analysis['Sales'] / store_analysis['Customer Name']
    store_analysis['Profit_Margin'] = store_analysis['Profit'] / store_analysis['Sales']

    # Create composite score
    weights = {
        'Sales': 0.3,
        'Profit_Margin': 0.3,
        'Customer Name': 0.2,
        'Sales_per_Customer': 0.2
    }

    # Normalize metrics
    for metric in weights.keys():
        if metric in store_analysis.columns:
            store_analysis[f'{metric}_Normalized'] = (
                store_analysis[metric] - store_analysis[metric].min()
            ) / (store_analysis[metric].max() - store_analysis[metric].min())

    # Calculate final score
    store_analysis['Location_Score'] = sum(
        store_analysis[f'{metric}_Normalized'] * weight
        for metric, weight in weights.items()
    )

    return store_analysis

"""# 5. Product Recommendation System"""

def build_product_recommendations(df):
    """
    Build a product recommendation system based on customer purchase patterns
    """
    # Create customer-product purchase matrix
    purchase_matrix = pd.crosstab(
        df['Customer Name'],
        df['Category']
    )

    # Train a model to predict product preferences
    X = purchase_matrix.values
    y = df.groupby('Customer Name')['Sales'].mean().values

    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model, purchase_matrix

"""# 6. Main Analysis Pipeline - for input == Raw .csv input"""

def run_retail_analysis(file_path):
    """
    Run the complete retail analysis pipeline
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)

    # 2. Perform customer segmentation
    print("Performing customer segmentation...")
    df, kmeans_model = perform_customer_segmentation(df)

    # 3. Create sales forecast
    print("Creating sales forecast...")
    prophet_model, forecast = create_sales_forecast(df)

    # 4. Analyze store placement
    print("Analyzing store placement...")
    store_analysis = analyze_store_placement(df)

    # 5. Build product recommendations
    print("Building product recommendations...")
    rec_model, purchase_matrix = build_product_recommendations(df)

    return {
        'processed_data': df,
        'segmentation_model': kmeans_model,
        'forecast_model': prophet_model,
        'forecast_results': forecast,
        'store_analysis': store_analysis,
        'recommendation_model': rec_model,
        'purchase_matrix': purchase_matrix
    }

# # Load your data
# df = pd.read_csv('DMart_Grocery_Sales_-_Retail_Analytics_Dataset.csv')

"""# Main Analysis Pipeline - for input == Standardized dataframe"""

def run_retail_analysis(df):
    """
    Run the complete retail analysis pipeline

    Parameters:
    df (pandas.DataFrame): Input dataframe with standardized date format
    """
    try:
        # 1. Initial preprocessing
        print("Preprocessing data...")
        processed_df = df.copy()

        # Convert Order Date to datetime if it isn't already
        processed_df['Order Date'] = pd.to_datetime(processed_df['Order Date'], format='%d-%m-%Y')

        # Create time-based features
        processed_df['Year'] = processed_df['Order Date'].dt.year
        processed_df['Month'] = processed_df['Order Date'].dt.month
        processed_df['Day of Week'] = processed_df['Order Date'].dt.dayofweek
        processed_df['Is Weekend'] = processed_df['Day of Week'].isin([5, 6]).astype(int)

        # 2. Calculate customer metrics
        print("Calculating customer metrics...")
        current_date = processed_df['Order Date'].max()

        customer_metrics = processed_df.groupby('Customer Name').agg({
            'Order Date': lambda x: (current_date - x.max()).days,  # Recency
            'Order ID': 'count',  # Frequency
            'Sales': 'sum'  # Monetary
        }).reset_index()

        customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']
        processed_df = processed_df.merge(customer_metrics, on='Customer Name')

        # 3. Customer Segmentation
        print("Performing customer segmentation...")
        features = ['Recency', 'Frequency', 'Total_Monetary']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(customer_metrics[features])

        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_metrics['Customer_Segment'] = kmeans.fit_predict(features_scaled)
        processed_df = processed_df.merge(
            customer_metrics[['Customer Name', 'Customer_Segment']],
            on='Customer Name'
        )

        # 4. Sales Forecasting
        print("Creating sales forecast...")
        daily_sales = processed_df.groupby('Order Date')['Sales'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']

        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        prophet_model.fit(daily_sales)

        future_dates = prophet_model.make_future_dataframe(periods=30)
        forecast = prophet_model.predict(future_dates)

        # 5. Store Placement Analysis
        print("Analyzing store placement...")
        store_analysis = processed_df.groupby(['Region', 'City']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count',
            'Customer Name': 'nunique',
            'Discount': 'mean'
        }).reset_index()

        # Calculate performance metrics
        store_analysis['Sales_per_Customer'] = store_analysis['Sales'] / store_analysis['Customer Name']
        store_analysis['Profit_Margin'] = store_analysis['Profit'] / store_analysis['Sales']

        # 6. Product Recommendations
        print("Building product recommendations...")
        purchase_matrix = pd.crosstab(
            processed_df['Customer Name'],
            processed_df['Category']
        )

        X = purchase_matrix.values
        y = processed_df.groupby('Customer Name')['Sales'].mean().values

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        return {
            'processed_data': processed_df,
            'segmentation_model': kmeans,
            'forecast_model': prophet_model,
            'forecast_results': forecast,
            'store_analysis': store_analysis,
            'recommendation_model': rf_model,
            'purchase_matrix': purchase_matrix
        }

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

# Function to display key insights
def display_insights(results):
    """
    Display key insights from the analysis
    """
    processed_data = results['processed_data']
    forecast = results['forecast_results']
    store_analysis = results['store_analysis']

    print("\n=== ANALYSIS INSIGHTS ===")

    # Customer Segments
    print("\nCustomer Segments:")
    segment_stats = processed_data.groupby('Customer_Segment').agg({
        'Customer Name': 'nunique',
        'Sales': 'mean',
        'Frequency': 'mean'
    }).round(2)
    print(segment_stats)

    # Top Performing Regions
    print("\nTop Performing Regions:")
    top_regions = store_analysis.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    print(top_regions)

    # Sales Forecast
    print("\nSales Forecast (Next 30 days):")
    forecast_summary = forecast.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].describe()
    print(forecast_summary)

df = pd.read_csv('DMart_Grocery_Sales_-_Retail_Analytics_Dataset.csv')

# First, standardize your dates
df_standardized = standardize_date_format(df, date_column='Order Date')

# Run the complete analysis
try:
    results = run_retail_analysis(df_standardized)

    # Display insights
    display_insights(results)

    # Access individual components
    processed_data = results['processed_data']
    forecast = results['forecast_results']
    store_analysis = results['store_analysis']

    # Example: Print the first few rows of processed data
    print("\nProcessed Data Sample:")
    print(processed_data.head())

    # Example: Print forecast for next week
    print("\nNext Week's Sales Forecast:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

    # Example: Print top performing stores
    print("\nTop Performing Stores:")
    print(store_analysis.nlargest(5, 'Sales')[['Region', 'City', 'Sales', 'Profit_Margin']])

except Exception as e:
    print(f"Error running analysis: {str(e)}")