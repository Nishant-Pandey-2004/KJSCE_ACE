import gradio as gr
import pandas as pd
import numpy as np
import pickle
import folium
from datetime import datetime
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
from typing import Optional

import gradio as gr
import time

def generate_audio():
    time.sleep(3)  # Simulate processing delay
    return "TN.wav"



# Import the ACE_Kmeans module (make sure ACE_Kmeans.py is in the same directory)
# If you don't have this module, you can remove or replace the references to it.
try:
    import ACE_Kmeans
except ImportError:
    ACE_Kmeans = None

#########################
# UTILITY FUNCTIONS
#########################

def generate_personalized_recommendations(df, customer_name):
    if customer_name not in df['Customer Name'].unique():
        return f"Customer {customer_name} not found in dataset"
    
    customer_purchases = df[df['Customer Name'] == customer_name]
    customer_region = customer_purchases['Region'].iloc[0]
    
    category_preferences = customer_purchases.groupby(['Category', 'Sub Category']).agg(
        {'Order ID': 'count', 'Sales': 'sum'}
    ).reset_index()
    category_preferences['Purchase_Frequency'] = category_preferences['Order ID']
    category_preferences['Avg_Spend'] = category_preferences['Sales'] / category_preferences['Order ID']
    
    regional_preferences = df[df['Region'] == customer_region].groupby(['Category', 'Sub Category']).agg(
        {'Order ID': 'count'}
    ).reset_index()
    regional_preferences['Regional_Popularity'] = regional_preferences['Order ID']
    
    recommendations = category_preferences.merge(
        regional_preferences[['Category', 'Sub Category', 'Regional_Popularity']], 
        on=['Category', 'Sub Category'], 
        how='outer'
    ).fillna(0)
    
    recommendations['Score'] = (
        0.4 * (recommendations['Purchase_Frequency'] / recommendations['Purchase_Frequency'].max()) +
        0.3 * (recommendations['Avg_Spend'] / recommendations['Avg_Spend'].max()) +
        0.3 * (recommendations['Regional_Popularity'] / recommendations['Regional_Popularity'].max())
    )
    
    recommendations = recommendations.sort_values('Score', ascending=False).head(5)
    
    output = "<strong>Top Personalized Recommendations:</strong><br>"
    for _, row in recommendations.iterrows():
        output += f"<br>✅ <strong>{row['Category']} - {row['Sub Category']}</strong> (Score: {row['Score']:.2f})"
    
    return output

def standardize_date_format(df, date_column='Order Date'):
    """
    Convert various date formats to standard 'dd-mm-yyyy' format.
    """
    def convert_date(date_str):
        try:
            if isinstance(date_str, str):
                if '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                elif '-' in date_str:
                    # Try to parse dd-mm-yyyy; if that fails, use pandas
                    try:
                        datetime.strptime(date_str, '%d-%m-%Y')
                        return date_str
                    except:
                        date_obj = pd.to_datetime(date_str)
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
    invalid_dates = df_copy[date_column].isna()
    if invalid_dates.any():
        print(f"Warning: {invalid_dates.sum()} dates could not be converted and will be removed")
        df_copy = df_copy.dropna(subset=[date_column])
    return df_copy

def load_models():
    """
    Dummy function to load pre-trained models if needed.
    """
    try:
        with open('retail_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def generate_map(top_stores):
    """
    Generate a Folium map with markers for the top stores.
    """
    geolocator = Nominatim(user_agent="retail_analysis")
    india_center = [22.3511, 78.6677]
    
    m = folium.Map(location=india_center, zoom_start=5)
    for _, row in top_stores.iterrows():
        city, region, sales = row['City'], row['Region'], row['Sales']
        try:
            location = geolocator.geocode(f"{city}, {region}, India")
            if location:
                folium.Marker(
                    location=[location.latitude, location.longitude],
                    popup=f"{city} ({region})<br>Sales: {sales}",
                    tooltip=f"{city} - ${sales}",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)
        except Exception as e:
            print(f"Error geocoding {city}: {e}")
    return m._repr_html_()

def analyze_retail_data(csv_file):
    """
    Reads the CSV file, standardizes dates, and returns:
      - an HTML table for the top 10 stores,
      - an HTML map,
      - the processed dataframe.
    """
    try:
        models = load_models()
        if models is None:
            return "Error: Could not load models. Ensure 'retail_models.pkl' is available.", None, None

        df = pd.read_csv(csv_file.name)
        df_standardized = standardize_date_format(df, date_column='Order Date')
        df_standardized['Order Date'] = pd.to_datetime(df_standardized['Order Date'], format='%d-%m-%Y')
        current_date = df_standardized['Order Date'].max()

        # Calculate customer metrics for table display
        customer_metrics = df_standardized.groupby('Customer Name').agg({
            'Order Date': lambda x: (current_date - x.max()).days,
            'Order ID': 'count',
            'Sales': 'sum'
        }).reset_index()
        customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']

        store_analysis = df_standardized.groupby(['Region', 'City']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count',
            'Customer Name': 'nunique',
            'Discount': 'mean'
        }).reset_index()
        top_stores = store_analysis.nlargest(10, 'Sales')[['Region', 'City', 'Sales', 'Profit']]
        map_html = generate_map(top_stores)
        return top_stores.to_html(classes='styled-table'), map_html, df_standardized
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None

def create_dashboard(df):
    """
    Create dashboard visualizations using Plotly.
    """
    if df is None:
        return "No data available for visualization."
    
    # Sales by Region (Bar Chart)
    sales_by_region = df.groupby('Region')['Sales'].sum().reset_index()
    fig_sales_by_region = px.bar(sales_by_region, x='Region', y='Sales', title='Sales by Region', color='Region')
    
    # Profit by Category (Pie Chart)
    profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()
    fig_profit_by_category = px.pie(profit_by_category, values='Profit', names='Category', title='Profit by Category')
    
    # Sales Over Time (Line Chart)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    sales_over_time = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    sales_over_time['Order Date'] = sales_over_time['Order Date'].astype(str)
    fig_sales_over_time = px.line(sales_over_time, x='Order Date', y='Sales', title='Sales Over Time')
    
    # Top 10 Customers (Bar Chart)
    top_customers = df.groupby('Customer Name')['Sales'].sum().nlargest(10).reset_index()
    fig_top_customers = px.bar(top_customers, x='Customer Name', y='Sales', title='Top 10 Customers by Sales', color='Customer Name')
    
    return fig_sales_by_region, fig_profit_by_category, fig_sales_over_time, fig_top_customers

def generate_personalized_recommendations(df, customer_name):
    if customer_name not in df['Customer Name'].unique():
        return f"Customer {customer_name} not found in dataset"
    
    customer_purchases = df[df['Customer Name'] == customer_name]
    customer_region = customer_purchases['Region'].iloc[0]
    
    category_preferences = customer_purchases.groupby(['Category', 'Sub Category']).agg(
        {'Order ID': 'count', 'Sales': 'sum'}
    ).reset_index()
    category_preferences['Purchase_Frequency'] = category_preferences['Order ID']
    category_preferences['Avg_Spend'] = category_preferences['Sales'] / category_preferences['Order ID']
    
    regional_preferences = df[df['Region'] == customer_region].groupby(['Category', 'Sub Category']).agg(
        {'Order ID': 'count'}
    ).reset_index()
    regional_preferences['Regional_Popularity'] = regional_preferences['Order ID']
    
    recommendations = category_preferences.merge(
        regional_preferences[['Category', 'Sub Category', 'Regional_Popularity']], 
        on=['Category', 'Sub Category'], 
        how='outer'
    ).fillna(0)
    
    recommendations['Score'] = (
        0.4 * (recommendations['Purchase_Frequency'] / recommendations['Purchase_Frequency'].max()) +
        0.3 * (recommendations['Avg_Spend'] / recommendations['Avg_Spend'].max()) +
        0.3 * (recommendations['Regional_Popularity'] / recommendations['Regional_Popularity'].max())
    )
    
    recommendations = recommendations.sort_values('Score', ascending=False).head(5)
    
    output = "<strong>Top Personalized Recommendations:</strong><br>"
    for _, row in recommendations.iterrows():
        output += f"<br>✅ <strong>{row['Category']} - {row['Sub Category']}</strong> (Score: {row['Score']:.2f})"
    
    return output

def update_customer_list(file):
    df = pd.read_csv(file.name)
    customers = df['Customer Name'].unique().tolist()
    return gr.update(choices=customers), df

def recommend(file, customer_name):
    df = pd.read_csv(file.name)
    return generate_personalized_recommendations(df, customer_name)


def get_retail_analysis_report(df):
    """
    Capture the printed output from the ACE_Kmeans analysis pipeline and additional prints,
    then return it as HTML to be displayed in the Conversing with data tab.
    """
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        if ACE_Kmeans:
            # Run the full analysis pipeline from ACE_Kmeans
            results = ACE_Kmeans.run_retail_analysis(df)
            # Print key insights using ACE_Kmeans display_insights function
            ACE_Kmeans.display_insights(results)
            
            # Additional outputs
            processed_data = results.get('processed_data')
            forecast = results.get('forecast_results')
            store_analysis = results.get('store_analysis')
            
            print("\nProcessed Data Sample:")
            if processed_data is not None:
                print(processed_data.head())
            else:
                print("No processed data available.")

            print("\nNext Week's Sales Forecast:")
            if forecast is not None:
                # Assuming forecast is a DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
                print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))
            else:
                print("No forecast data available.")

            print("\nTop Performing Stores:")
            if store_analysis is not None:
                print(store_analysis.nlargest(5, 'Sales')[['Region', 'City', 'Sales', 'Profit_Margin']])
            else:
                print("No store analysis available.")
        else:
            print("ACE_Kmeans module not available.")
    except Exception as e:
        print(f"Error generating analysis report: {e}")
    finally:
        sys.stdout = old_stdout
    # Wrap the captured text in <pre> tags to preserve formatting
    return "<pre>" + captured_output.getvalue() + "</pre>"

#########################
# GRADIO INTERFACE SETUP
#########################

with gr.Blocks(css="""
    /* Custom Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    body, .gradio-container {font-family: 'Inter', sans-serif;}

    /* Glassmorphism Background */
    body {
        background: linear-gradient(135deg, #667eea, #764ba2);
        min-height: 100vh;
        margin: 0;
        padding: 20px;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Header Styling */
    .gradio-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .gradio-header h1 {
        background: linear-gradient(135deg, #FF6F61, #FFD700, #00C9A7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 600;
        margin: 0;
    }

    /* Tabs Styling */
    .gradio-tabs {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .gradio-tab {
        background: transparent;
        border-radius: 10px;
        padding: 20px;
    }
    .gradio-tab.selected {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Button Styling */
    .gradio-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        font-weight: 600;
    }
    .gradio-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* File Upload Styling */
    .gradio-file-upload {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
    }
    .gradio-file-upload label {
        color: white;
    }

    /* Table Styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    .styled-table th, .styled-table td {
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 12px;
        text-align: left;
        color: white;
    }
    .styled-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .styled-table tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.05);
    }
    .styled-table tr:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Loading Animation */
    .loading {
        text-align: center;
        font-size: 1.2em;
        color: rgba(255, 255, 255, 0.8);
    }
    @keyframes spin {
        0% {transform: rotate(0deg);}
        100% {transform: rotate(360deg);}
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #667eea;
        border-top: 3px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    /* Fade-in Animation */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Map Styling */
    .folium-map {
        border-radius: 15px;
        overflow: hidden;
        margin-top: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
""") as iface:

    # Header
    gr.Markdown("""
    <div class="gradio-header glass-card">
        <h1>Retail Mitra</h1>
    </div>
    """)



    with gr.Tabs() as tabs:
        # Upload CSV Tab
        with gr.Tab("📤 Upload CSV", elem_id="upload-tab"):
            file_input = gr.File(label="Upload CSV", file_types=[".csv"], file_count="single", elem_classes="gradio-file-upload")
            generate_button = gr.Button("Analyze", elem_classes="gradio-button")
        
        # Table View Tab
        with gr.Tab("📊 Table View", elem_id="table-tab"):
            table_output = gr.HTML("<div class='loading'><div class='loading-spinner'></div> Loading data...</div>", elem_classes="fade-in")
        
        # Map View Tab
        with gr.Tab("🌍 Map View", elem_id="map-tab"):
            map_output = gr.HTML("<div class='loading'><div class='loading-spinner'></div> Loading map...</div>", elem_classes="fade-in")
        
        # Dashboard Tab
        with gr.Tab("📈 Dashboard", elem_id="dashboard-tab"):
            with gr.Row():
                sales_by_region_plot = gr.Plot(label="Sales by Region")
                profit_by_category_plot = gr.Plot(label="Profit by Category")
            with gr.Row():
                sales_over_time_plot = gr.Plot(label="Sales Over Time")
                top_customers_plot = gr.Plot(label="Top 10 Customers by Sales")
        
        # Conversing with Data Tab
        with gr.Tab("💬 Conversing with data", elem_id="converse-tab"):
            conversation_output = gr.HTML("<div class='loading'><div class='loading-spinner'></div> Generating report...</div>", elem_classes="fade-in")
            with gr.Blocks() as demo:
                btn = gr.Button("Generate Interactive Audio Insights")
                audio_output = gr.Audio()
                
                btn.click(fn=generate_audio, inputs=[], outputs=[audio_output])

    
         # Personalized Recommendations Tab
        with gr.Tab("🛒 Personalized Recommendations", elem_id="recommendations-tab"):
            file_input = gr.File(label="📂 Upload CSV File")
            customer_dropdown = gr.Dropdown(label="👤 Select Customer", choices=[])
            recommend_button = gr.Button("🔍 Generate Recommendations")
            output_text = gr.Markdown(label="📌 Recommendations", elem_id="output-text")
            
            file_input.change(update_customer_list, inputs=[file_input], outputs=[customer_dropdown, gr.State()])
            recommend_button.click(recommend, inputs=[file_input, customer_dropdown], outputs=[output_text])

    

    def generate_output(csv_file):
        # Process CSV for table, map and dashboard views
        table_html, map_html, df = analyze_retail_data(csv_file)
        fig_sales_by_region, fig_profit_by_category, fig_sales_over_time, fig_top_customers = None, None, None, None
        conversation_report = "<p>No data available.</p>"
        if df is not None and not isinstance(df, str):
            fig_sales_by_region, fig_profit_by_category, fig_sales_over_time, fig_top_customers = create_dashboard(df)
            # Generate the conversation report by capturing terminal output from ACE_Kmeans
            conversation_report = get_retail_analysis_report(df)
        return table_html, map_html, fig_sales_by_region, fig_profit_by_category, fig_sales_over_time, fig_top_customers, conversation_report

    generate_button.click(
        generate_output,
        inputs=file_input,
        outputs=[table_output, map_output, sales_by_region_plot, profit_by_category_plot, sales_over_time_plot, top_customers_plot, conversation_output]
    )

if __name__ == "__main__":
    iface.launch(share=True)