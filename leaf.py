import gradio as gr
import pandas as pd
import numpy as np
import pickle
import folium
from datetime import datetime
from geopy.geocoders import Nominatim

# Function to standardize date format
def standardize_date_format(df, date_column='Order Date'):
    def convert_date(date_str):
        try:
            if isinstance(date_str, str):
                if '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                elif '-' in date_str:
                    date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                else:
                    date_obj = pd.to_datetime(date_str)
                return date_obj.strftime('%d-%m-%Y')
            elif isinstance(date_str, datetime):
                return date_str.strftime('%d-%m-%Y')
            else:
                return pd.to_datetime(date_str).strftime('%d-%m-%Y')
        except Exception as e:
            return None

    df_copy = df.copy()
    df_copy[date_column] = df_copy[date_column].apply(convert_date)
    df_copy = df_copy.dropna(subset=[date_column])
    return df_copy

# Function to load the ML models
def load_models():
    try:
        with open('retail_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

# Function to generate a Folium map
def generate_map(top_stores):
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
                    icon=folium.Icon(color="green", icon="info-sign")
                ).add_to(m)
        except Exception:
            pass
    return m._repr_html_()

# Function to analyze retail data
def analyze_retail_data(csv_file):
    models = load_models()
    if models is None:
        return "Error: Could not load models.", ""
    
    kmeans, scaler, rf_model, features = models['kmeans'], models['scaler'], models['rf_model'], models['feature_names']
    df = pd.read_csv(csv_file.name)
    df_standardized = standardize_date_format(df, date_column='Order Date')
    df_standardized['Order Date'] = pd.to_datetime(df_standardized['Order Date'], format='%d-%m-%Y')
    
    customer_metrics = df_standardized.groupby('Customer Name').agg({
        'Order Date': lambda x: (df_standardized['Order Date'].max() - x.max()).days,
        'Order ID': 'count',
        'Sales': 'sum'
    }).reset_index()
    customer_metrics.columns = ['Customer Name', 'Recency', 'Frequency', 'Total_Monetary']
    customer_metrics['Segment'] = kmeans.predict(scaler.transform(customer_metrics[features]))
    
    store_analysis = df_standardized.groupby(['Region', 'City']).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    top_stores = store_analysis.nlargest(10, 'Sales')[['Region', 'City', 'Sales']]
    
    top_stores['Sales'] = top_stores['Sales'].apply(lambda x: f"${x:,.2f}")
    map_html = generate_map(top_stores)
    return top_stores.to_html(classes='styled-table'), map_html

# Gradio UI with a sidebar, improved layout, and loading animation
with gr.Blocks(theme=gr.themes.Default()) as iface:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Retail Analytics Dashboard")
            file_input = gr.File(label="Upload CSV", file_types=[".csv"], file_count="single")
            generate_button = gr.Button("🚀 Generate Analysis")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📋 Table View"):
                    table_output = gr.HTML(label="Table Output")
                with gr.Tab("🗺️ Map View"):
                    map_output = gr.HTML(label="Map Output")
    
    # Add a loading animation
    loading = gr.Markdown("⌛ Processing data... Please wait!", visible=False)
    
    def generate_output(csv_file):
        loading.update(visible=True)
        table_html, map_html = analyze_retail_data(csv_file)
        loading.update(visible=False)
        return table_html, map_html

    generate_button.click(generate_output, inputs=file_input, outputs=[table_output, map_output])

if __name__ == "__main__":
    iface.launch(share=True)