import gradio as gr
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def standardize_date_format(df, date_column='Order Date'):
    """
    Convert various date formats to standard 'dd-mm-yyyy' format
    """
    def convert_date(date_str):
        try:
            if isinstance(date_str, str):
                if '/' in date_str:  # mm/dd/yyyy format
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                elif '-' in date_str:  # dd-mm-yyyy format
                    date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                else:
                    date_obj = pd.to_datetime(date_str)
                return date_obj.strftime('%d-%m-%Y')

            elif isinstance(date_str, datetime):
                return date_str.strftime('%d-%m-%Y')

            else:
                return pd.to_datetime(date_str).strftime('%d-%m-%Y')

        except Exception as e:
            print(f"Error converting date '{date_str}': {str(e)}")
            return None

    df_copy = df.copy()
    df_copy[date_column] = df_copy[date_column].apply(convert_date)
    
    # Remove invalid dates
    invalid_dates = df_copy[date_column].isna()
    if invalid_dates.any():
        print(f"Warning: {invalid_dates.sum()} dates could not be converted and will be removed")
        df_copy = df_copy.dropna(subset=[date_column])

    return df_copy

def load_models():
    """Load the pickled models"""
    try:
        with open('retail_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def analyze_retail_data(csv_file):
    """
    Analyze retail data using pre-trained models
    """
    try:
        # Load models
        models = load_models()
        if models is None:
            return "Error: Could not load models. Ensure 'retail_models.pkl' is available."

        kmeans = models['kmeans']
        scaler = models['scaler']
        rf_model = models['rf_model']
        features = models['feature_names']
        
        # Read CSV file
        print("Reading CSV file...")
        df = pd.read_csv(csv_file.name)
        
        # Standardize dates
        print("Standardizing dates...")
        df_standardized = standardize_date_format(df, date_column='Order Date')
        
        # Convert to datetime
        df_standardized['Order Date'] = pd.to_datetime(df_standardized['Order Date'], format='%d-%m-%Y')
        current_date = df_standardized['Order Date'].max()
        
        # Calculate customer metrics
        print("Calculating customer metrics...")
        customer_metrics = df_standardized.groupby('Customer Name').agg({
            'Order Date': lambda x: (current_date - x.max()).days,
            'Order ID': 'count',
            'Sales': 'sum'
        }).reset_index()
        customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']
        
        # Apply clustering
        features_scaled = scaler.transform(customer_metrics[features])
        customer_metrics['Segment'] = kmeans.predict(features_scaled)
        
        # Store analysis
        print("Analyzing store performance...")
        store_analysis = df_standardized.groupby(['Region', 'City']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count',
            'Customer Name': 'nunique',
            'Discount': 'mean'
        }).reset_index()
        
        store_analysis['Sales_per_Customer'] = store_analysis['Sales'] / store_analysis['Customer Name']
        store_analysis['Profit_Margin'] = store_analysis['Profit'] / store_analysis['Sales']
        
        # Get top stores
        top_stores = store_analysis.nlargest(10, 'Sales')[
            ['Region', 'City', 'Sales', 'Profit_Margin']
        ].copy()
        
        # Format values for display
        top_stores['Sales'] = top_stores['Sales'].apply(lambda x: f"${x:,.2f}")
        top_stores['Profit_Margin'] = top_stores['Profit_Margin'].apply(lambda x: f"{x:.2%}")
        
        # Segment analysis
        print("Performing customer segmentation...")
        segment_stats = customer_metrics.groupby('Segment').agg({
            'Customer Name': 'count',
            'Total_Monetary': 'mean',
            'Frequency': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Number of Customers', 'Average Total Sales', 'Average Purchase Frequency']
        segment_stats['Average Total Sales'] = segment_stats['Average Total Sales'].apply(lambda x: f"${x:,.2f}")
        
        # Convert DataFrames to HTML
        top_stores_html = top_stores.to_html(index=False, classes='styled-table')
        segment_stats_html = segment_stats.to_html(classes='styled-table')

        # Create HTML output
        html_output = f"""
        <div style='text-align: center;'>
            <h2>Retail Analysis Results</h2>
            
            <h3>Top 10 Performing Stores</h3>
            {top_stores_html}
            
            <h3>Customer Segments Analysis</h3>
            {segment_stats_html}
            
            <div style='margin-top: 20px; font-size: 0.9em; color: #666;'>
                Analysis period: {df_standardized['Order Date'].min().strftime('%Y-%m-%d')} to {df_standardized['Order Date'].max().strftime('%Y-%m-%d')}
                <br>
                Total Records: {len(df_standardized):,}
                <br>
                Total Customers: {df_standardized['Customer Name'].nunique():,}
            </div>
        </div>
        <style>
            .styled-table {{
                border-collapse: collapse;
                margin: 25px auto;
                font-size: 0.9em;
                font-family: sans-serif;
                min-width: 400px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            }}
            .styled-table thead tr {{
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }}
            .styled-table th, .styled-table td {{
                padding: 12px 15px;
            }}
            .styled-table tbody tr {{
                border-bottom: 1px solid #dddddd;
            }}
            .styled-table tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .styled-table tbody tr:last-of-type {{
                border-bottom: 2px solid #009879;
            }}
        </style>
        """
        
        return html_output
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_retail_data,
    inputs=[gr.File(label="Upload Retail Data CSV", file_types=[".csv"], file_count="single")],
    outputs=gr.HTML(label="Analysis Results"),
    title="Retail Analytics Dashboard",
    description="""
    Upload your retail data CSV file to analyze:
    - View top-performing stores
    - See customer segmentation results
    - Get detailed sales and profit metrics
    
    Required CSV columns: Order Date, Region, City, Sales, Profit, Customer Name, Order ID
    """,
    theme="default",
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    try:
        print("Loading models and starting Gradio interface...")
        models = load_models()
        if models is None:
            print("Warning: Models could not be loaded. Ensure 'retail_models.pkl' exists.")
        iface.launch(share=True)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
