import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import sys
from PIL import Image

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_data, clean_data, get_key_factors

# Constants
API_URL = os.environ.get('API_URL', 'http://0.0.0.0:5000')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/train.csv')

# Page configuration
st.set_page_config(
    page_title="Airline Customer Satisfaction Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-container {
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_airline_data():
    """Load and clean the airline satisfaction data"""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_insights_from_api():
    """Get insights from the API"""
    try:
        response = requests.get(f"{API_URL}/api/insights")
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Error getting insights: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Error connecting to API: {e}")
        # Fallback to local data if API is not available
        df = load_airline_data()
        if df is not None:
            factors = get_key_factors(df)
            return {
                'key_factors': factors.to_dict(orient='records'),
                'sample_size': len(df),
                'satisfaction_rate': df['satisfaction'].mean() if 'satisfaction' in df.columns else None
            }
        return None

def get_stats_from_api(category=None):
    """Get statistics from the API"""
    try:
        if category:
            response = requests.get(f"{API_URL}/api/stats?category={category}")
        else:
            response = requests.get(f"{API_URL}/api/stats")
            
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Error getting stats: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Error connecting to API: {e}")
        return None

def predict_from_api(input_data):
    """Make a prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Error making prediction: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Error connecting to API: {e}")
        return None

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "model_loaded": False, "data_loaded": False}
    except:
        return {"status": "error", "model_loaded": False, "data_loaded": False}

def display_factor_analysis(df, factor):
    """Function to display analysis for a specific factor"""
    # Ensure factor is not 'satisfaction' itself
    if factor == 'satisfaction':
        st.warning(f"Cannot analyze satisfaction by satisfaction")
        return

    # Calculate satisfaction by rating
    factor_sat = df.groupby(factor)['satisfaction'].mean().reset_index()
    factor_sat['satisfaction_pct'] = factor_sat['satisfaction'] * 100
    
    # Create chart
    fig_factor_line = px.line(
        factor_sat, 
        x=factor, 
        y='satisfaction_pct',
        title=f'Satisfaction Rate by {factor} Rating',
        labels={'satisfaction_pct': 'Satisfaction Rate (%)', factor: f'{factor} Rating'},
        markers=True,
        line_shape='linear'
    )
    fig_factor_line.update_traces(line=dict(color='#1E88E5', width=3))
    fig_factor_line.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_factor_line, use_container_width=True)
    
    # Add descriptive analysis
    mean_rating = df[factor].mean()
    st.markdown(f"**Average {factor} Rating:** {mean_rating:.2f}/5")
    
    col1, col2 = st.columns(2)
    with col1:
        # Distribution of ratings
        fig_factor_hist = px.histogram(
            df, 
            x=factor,
            title=f'Distribution of {factor} Ratings',
            color_discrete_sequence=['#1E88E5'],
            nbins=5
        )
        fig_factor_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_factor_hist, use_container_width=True)
    
    with col2:
        # Satisfaction rate by rating
        data = []
        for rating in sorted(df[factor].unique()):
            subset = df[df[factor] == rating]
            sat_rate = subset['satisfaction'].mean() * 100
            data.append({
                'Rating': rating,
                'Satisfaction Rate': sat_rate,
                'Count': len(subset)
            })
        
        data_df = pd.DataFrame(data)
        st.markdown(f"##### Impact of {factor} on Satisfaction")
        st.dataframe(data_df)

# Sidebar
st.sidebar.markdown("## ✈️ Airline Satisfaction Analysis")
st.sidebar.markdown("### Navigation")

# Navigation
page = st.sidebar.radio(
    "Select a page:",
    ["Dashboard Overview", "Key Factors Analysis", "Satisfaction Predictor"]
)

# Load data (will be cached)
df = load_airline_data()

# Check API health
api_health = check_api_health()
api_available = api_health["status"] == "ok"

if not api_available:
    st.sidebar.warning("⚠️ API is not available. Some features may be limited.")
    if df is None:
        st.error("No data available. Please make sure the data file exists or the API is running.")
        st.stop()

# Display the selected page
if page == "Dashboard Overview":
    # Header
    st.markdown("<h1 class='main-header'>Airline Customer Satisfaction Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>About this Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("""
        This dashboard provides insights into airline customer satisfaction based on a survey of airline passengers.
        
        Use this tool to:
        - Identify key factors affecting customer satisfaction
        - Analyze satisfaction trends across different passenger segments
        - Predict customer satisfaction based on various service factors
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        if df is not None:
            st.metric("Total Passengers", f"{len(df):,}")
            
            # Calculate satisfaction rate
            if 'satisfaction' in df.columns:
                sat_rate = df['satisfaction'].mean() * 100
                st.metric("Overall Satisfaction Rate", f"{sat_rate:.1f}%")
            
            # Show a sample of the data
            st.markdown("##### Sample Data")
            st.dataframe(df.head(3))
        else:
            st.warning("Data not available. Using API fallback.")
            insights = get_insights_from_api()
            if insights:
                st.metric("Total Passengers", f"{insights['sample_size']:,}")
                if insights['satisfaction_rate'] is not None:
                    sat_rate = insights['satisfaction_rate'] * 100
                    st.metric("Overall Satisfaction Rate", f"{sat_rate:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    
    if df is not None:
        # Create metrics based on the data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'Age' in df.columns:
                avg_age = df['Age'].mean()
                st.markdown(f"<div class='metric-value'>{avg_age:.1f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Average Age</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'Flight Distance' in df.columns:
                avg_distance = df['Flight Distance'].mean()
                st.markdown(f"<div class='metric-value'>{avg_distance:.0f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Avg Flight Distance (miles)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'Customer Type' in df.columns:
                loyal_pct = (df['Customer Type'] == 'Loyal Customer').mean() * 100
                st.markdown(f"<div class='metric-value'>{loyal_pct:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Loyal Customers</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'Class' in df.columns and 'satisfaction' in df.columns:
                business_sat = df[df['Class'] == 'Business']['satisfaction'].mean() * 100
                st.markdown(f"<div class='metric-value'>{business_sat:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Business Class Satisfaction</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Satisfaction by factors
    st.markdown("<h2 class='sub-header'>Satisfaction by Passenger Segments</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    if df is not None:
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if 'Class' in df.columns and 'satisfaction' in df.columns:
                # Calculate satisfaction by class
                class_sat = df.groupby('Class')['satisfaction'].mean().reset_index()
                class_sat['satisfaction_pct'] = class_sat['satisfaction'] * 100
                
                # Create bar chart
                fig_class = px.bar(
                    class_sat, 
                    x='Class', 
                    y='satisfaction_pct',
                    title='Satisfaction Rate by Travel Class',
                    labels={'satisfaction_pct': 'Satisfaction Rate (%)', 'Class': 'Travel Class'},
                    color='satisfaction_pct',
                    color_continuous_scale='blues',
                    text_auto='.1f'
                )
                fig_class.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_class.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_class, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if 'Customer Type' in df.columns and 'satisfaction' in df.columns:
                # Calculate satisfaction by customer type
                customer_sat = df.groupby('Customer Type')['satisfaction'].mean().reset_index()
                customer_sat['satisfaction_pct'] = customer_sat['satisfaction'] * 100
                
                # Create bar chart
                fig_customer = px.bar(
                    customer_sat, 
                    x='Customer Type', 
                    y='satisfaction_pct',
                    title='Satisfaction Rate by Customer Type',
                    labels={'satisfaction_pct': 'Satisfaction Rate (%)', 'Customer Type': 'Customer Type'},
                    color='satisfaction_pct',
                    color_continuous_scale='blues',
                    text_auto='.1f'
                )
                fig_customer.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_customer.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_customer, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display flight distance distribution
    st.markdown("<h2 class='sub-header'>Flight Distance Distribution</h2>", unsafe_allow_html=True)
    
    if df is not None and 'Flight Distance' in df.columns:
        fig_dist = px.histogram(
            df, 
            x='Flight Distance',
            nbins=30,
            title='Distribution of Flight Distances',
            color_discrete_sequence=['#1E88E5'],
            opacity=0.7
        )
        fig_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_dist, use_container_width=True)

elif page == "Key Factors Analysis":
    # Header
    st.markdown("<h1 class='main-header'>Key Factors Affecting Customer Satisfaction</h1>", unsafe_allow_html=True)
    
    # Get insights from API or local data
    insights = get_insights_from_api()
    
    if insights and 'key_factors' in insights:
        # Create a dataframe from the insights
        factors_df = pd.DataFrame(insights['key_factors'])
        
        # Filter out non-relevant columns if needed
        factors_df = factors_df[factors_df['correlation'].abs() > 0.05]
        
        # Sort by absolute correlation
        factors_df['abs_correlation'] = factors_df['correlation'].abs()
        factors_df = factors_df.sort_values('abs_correlation', ascending=False)
        
        # Display top factors
        st.markdown("<h2 class='sub-header'>Top Factors by Correlation with Satisfaction</h2>", unsafe_allow_html=True)
        
        # Create a horizontal bar chart
        fig_factors = px.bar(
            factors_df.head(10), 
            y='feature', 
            x='correlation',
            orientation='h',
            title='Top 10 Factors Influencing Customer Satisfaction',
            labels={'correlation': 'Correlation with Satisfaction', 'feature': 'Service Feature'},
            color='correlation',
            color_continuous_scale='RdBu',
            text_auto='.3f'
        )
        fig_factors.update_traces(textposition='outside')
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # Detailed analysis of top factors
        st.markdown("<h2 class='sub-header'>Detailed Analysis of Top Factors</h2>", unsafe_allow_html=True)
        
        if df is not None:
            # Select top 3 factors
            top_factors = factors_df['feature'].head(3).tolist()
            
            # Display satisfaction rate by rating for each factor
            for i, factor in enumerate(top_factors):
                if factor in df.columns and factor != 'satisfaction':
                    display_factor_analysis(df, factor)
                    st.markdown("---")
    else:
        st.warning("Insights data not available. Please check the API connection or data loading.")
        
        # Fallback to local data analysis if available
        if df is not None:
            st.markdown("Using local data analysis...")
            
            # Analyze correlations
            if 'satisfaction' in df.columns:
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                # Calculate correlations with satisfaction
                correlations = []
                for col in numeric_cols:
                    if col != 'satisfaction' and col != 'id':
                        corr = df[[col, 'satisfaction']].corr().iloc[0, 1]
                        correlations.append({
                            'feature': col,
                            'correlation': corr,
                            'abs_correlation': abs(corr)
                        })
                
                # Create dataframe and sort
                corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
                
                # Display top factors
                fig_corr = px.bar(
                    corr_df.head(10), 
                    y='feature', 
                    x='correlation',
                    orientation='h',
                    title='Top 10 Factors Influencing Customer Satisfaction',
                    labels={'correlation': 'Correlation with Satisfaction', 'feature': 'Service Feature'},
                    color='correlation',
                    color_continuous_scale='RdBu',
                    text_auto='.3f'
                )
                fig_corr.update_traces(textposition='outside')
                st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Satisfaction Predictor":
    # Header
    st.markdown("<h1 class='main-header'>Customer Satisfaction Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This tool predicts whether a customer will be satisfied based on their flight experience.
    Adjust the sliders below to see how different factors affect the predicted satisfaction.
    """)
    
    # Set up columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Customer and Flight Details</h3>", unsafe_allow_html=True)
        
        age = st.slider("Age", min_value=18, max_value=85, value=35)
        flight_distance = st.slider("Flight Distance (miles)", min_value=100, max_value=5000, value=1000)
        
        customer_type = st.selectbox(
            "Customer Type",
            options=["Loyal Customer", "disloyal Customer"]
        )
        
        travel_class = st.selectbox(
            "Travel Class",
            options=["Business", "Eco", "Eco Plus"]
        )
        
        travel_purpose = st.selectbox(
            "Travel Purpose",
            options=["Business travel", "Personal Travel"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Service Ratings</h3>", unsafe_allow_html=True)
        
        inflight_wifi = st.slider("Inflight WiFi Service Rating", min_value=0, max_value=5, value=3)
        departure_arrival_time = st.slider("Departure/Arrival Time Convenience", min_value=0, max_value=5, value=3)
        ease_booking = st.slider("Ease of Online Booking", min_value=0, max_value=5, value=3)
        gate_location = st.slider("Gate Location", min_value=0, max_value=5, value=3)
        food_drink = st.slider("Food and Drink", min_value=0, max_value=5, value=3)
        seat_comfort = st.slider("Seat Comfort", min_value=0, max_value=5, value=3)
        inflight_entertainment = st.slider("Inflight Entertainment", min_value=0, max_value=5, value=3)
        onboard_service = st.slider("On-board Service", min_value=0, max_value=5, value=3)
        baggage_handling = st.slider("Baggage Handling", min_value=0, max_value=5, value=3)
        checkin_service = st.slider("Check-in Service", min_value=0, max_value=5, value=3)
        cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create input data dictionary
    # Create input data dictionary with all required columns
    input_data = {
        "Age": age,
        "Flight Distance": flight_distance,
        "Customer Type": customer_type,
        "Class": travel_class,
        "Type of Travel": travel_purpose,
        "Inflight wifi service": inflight_wifi,
        "Departure/Arrival time convenient": departure_arrival_time,
        "Ease of Online booking": ease_booking,
        "Gate location": gate_location,
        "Food and drink": food_drink,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Cleanliness": cleanliness,
        # Ajout des colonnes manquantes avec des valeurs par défaut
        "Online boarding": 3,
        "Leg room service": 3,
        "Inflight service": 3,
        "Gender": "Male",
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
        "Unnamed: 0": 0
    }
    
    # Add prediction button
    if st.button("Predict Customer Satisfaction"):
        # Only use API if available
        if api_available and api_health["model_loaded"]:
            with st.spinner("Making prediction..."):
                prediction = predict_from_api(input_data)
                
                if prediction:
                    # Display prediction result
                    st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
                    
                    # Create columns for result display
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # Display prediction result
                        if prediction["prediction"] == 1:
                            st.success("✅ Customer is likely to be SATISFIED", icon="✅")
                        else:
                            st.error("❌ Customer is likely to be DISSATISFIED", icon="❌")
                        
                        # Display probability
                        st.metric("Prediction Confidence", f"{prediction['probability'] * 100:.1f}%")
                    
                    with res_col2:
                        # Create a gauge chart for visualization
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction["probability"] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Satisfaction Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#1E88E5"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#EF5350"},
                                    {'range': [50, 100], 'color': "#66BB6A"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Display factors affecting this prediction
                    st.markdown("<h3>Key Factors Affecting This Prediction</h3>", unsafe_allow_html=True)
                    
                    # Request insights from API
                    insights = get_insights_from_api()
                    if insights and 'key_factors' in insights:
                        # Get top factors
                        top_factors = pd.DataFrame(insights['key_factors']).head(5)['feature'].tolist()
                        
                        # Display top factors and their values
                        factor_data = []
                        for factor in top_factors:
                            if factor in input_data:
                                factor_data.append({
                                    'Factor': factor,
                                    'Your Rating': input_data[factor],
                                    'Impact': 'High' if factor in top_factors[:3] else 'Medium'
                                })
                        
                        factor_df = pd.DataFrame(factor_data)
                        st.dataframe(factor_df)
                        
                        # Provide recommendations if prediction is negative
                        if prediction["prediction"] == 0:
                            st.markdown("<h3>Recommendations to Improve Satisfaction</h3>", unsafe_allow_html=True)
                            
                            for factor in factor_data:
                                if factor['Your Rating'] < 4:
                                    st.markdown(f"- Improve **{factor['Factor']}** (current rating: {factor['Your Rating']})")
                else:
                    st.error("Failed to make prediction. Please try again.")
        else:
            st.warning("Prediction API is not available. Please make sure the API is running.")
            st.info("This is a simulation mode - API integration will be available when deployed.")
            
            # Simulate prediction (random result)
            import random
            
            prob = random.random()
            prediction = {
                "prediction": 1 if prob > 0.5 else 0,
                "probability": prob if prob > 0.5 else 1 - prob
            }
            
            # Display simulation result
            st.markdown("<h2 class='sub-header'>Simulation Result</h2>", unsafe_allow_html=True)
            
            if prediction["prediction"] == 1:
                st.success("✅ Customer is likely to be SATISFIED (SIMULATION)", icon="✅")
            else:
                st.error("❌ Customer is likely to be DISSATISFIED (SIMULATION)", icon="❌")
            
            st.metric("Prediction Confidence", f"{prediction['probability'] * 100:.1f}%")
            
            st.info("Note: This is a simulation. Connect the API for real predictions.")

if __name__ == '__main__':
    # Ne rien faire ici, car Streamlit est déjà lancé par entrypoint.sh
    pass