import gradio as gr
import pandas as pd

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

with gr.Blocks() as app:
    gr.Markdown("# 🛒 DMart Personalized Recommendations", elem_id="header")
    file_input = gr.File(label="📂 Upload CSV File")
    customer_dropdown = gr.Dropdown(label="👤 Select Customer", choices=[])
    recommend_button = gr.Button("🔍 Generate Recommendations")
    output_text = gr.Markdown(label="📌 Recommendations", elem_id="output-text")
    
    file_input.change(update_customer_list, inputs=[file_input], outputs=[customer_dropdown, gr.State()])
    recommend_button.click(recommend, inputs=[file_input, customer_dropdown], outputs=[output_text])
    
    # Custom CSS for styling with larger font sizes
    app.css = """
    #header {
        font-size: 48px;  /* Increased header font size */
        font-weight: bold;
        color: #2a9d8f;
    }
    #output-text {
        font-size: 32px;  /* Increased output text font size */
        color: #264653;
        line-height: 1.6;
        font-family: 'Arial', sans-serif;
    }
    #output-text strong {
        color: #e76f51;
    }
    .gradio-container {
        font-size: 24px;  /* Increased general font size for all text */
    }
    .gradio-button {
        font-size: 24px;  /* Increased font size for buttons */
    }
    .gradio-dropdown {
        font-size: 24px;  /* Increased font size for dropdown */
    }
    """
    
app.launch()
