from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from flask_cors import CORS

# Import preprocessing and model functions
from preprocessing import load_data, clean_data, get_key_factors
from model import predict_satisfaction

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and data
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/satisfaction_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')

# Load model and data at startup
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except:
    model = None
    print("Model not found, please train the model first")

try:
    data = load_data(DATA_PATH)
    cleaned_data = clean_data(data)
    print("Data loaded successfully")
except:
    data = None
    cleaned_data = None
    print("Data not found, please download the data first")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'ok',
        'model_loaded': model is not None,
        'data_loaded': data is not None
    }
    return jsonify(status)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict customer satisfaction based on flight experience
    
    Request JSON example:
    {
        "Age": 30,
        "Flight Distance": 1000,
        "Inflight wifi service": 3,
        "Departure/Arrival time convenient": 4,
        "...": "..."
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
    
    try:
        # Get data from request
        input_data = request.json
        
        # Create a DataFrame with default values for all required columns
        required_columns = [
            'Age', 'Flight Distance', 'Inflight wifi service', 
            'Departure/Arrival time convenient', 'Ease of Online booking', 
            'Gate location', 'Food and drink', 'Seat comfort', 
            'Inflight entertainment', 'On-board service', 'Baggage handling', 
            'Checkin service', 'Cleanliness', 'Customer Type', 'Class', 
            'Type of Travel', 'Online boarding', 'Leg room service', 
            'Inflight service', 'Gender', 'Departure Delay in Minutes', 
            'Arrival Delay in Minutes', 'Unnamed: 0'
        ]
        
        # Create a dictionary with default values
        default_data = {
            'Age': 40,
            'Flight Distance': 1000,
            'Inflight wifi service': 3,
            'Departure/Arrival time convenient': 3,
            'Ease of Online booking': 3,
            'Gate location': 3,
            'Food and drink': 3,
            'Seat comfort': 3,
            'Inflight entertainment': 3,
            'On-board service': 3,
            'Baggage handling': 3,
            'Checkin service': 3,
            'Cleanliness': 3,
            'Customer Type': 'Loyal Customer',
            'Class': 'Eco',
            'Type of Travel': 'Personal Travel',
            'Online boarding': 3,
            'Leg room service': 3,
            'Inflight service': 3,
            'Gender': 'Male',
            'Departure Delay in Minutes': 0,
            'Arrival Delay in Minutes': 0,
            'Unnamed: 0': 0
        }
        
        # Update with the values provided by the user
        for key, value in input_data.items():
            if key in default_data:
                default_data[key] = value
        
        # Check if any required columns are missing in the default_data dictionary
        missing_columns = set(required_columns) - set(default_data.keys())
        if missing_columns:
            return jsonify({'error': f'columns are missing: {missing_columns}'}), 400
        
        # Create DataFrame with just the required columns in the correct order
        input_df = pd.DataFrame([default_data])
        
        # Make prediction
        result = predict_satisfaction(model, input_df)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get key insights about factors influencing satisfaction"""
    if cleaned_data is None:
        return jsonify({'error': 'Data not loaded. Please load the data first.'}), 503
    
    try:
        # Get key factors
        factors = get_key_factors(cleaned_data)
        
        # Convert to dictionary
        insights = {
            'key_factors': factors.to_dict(orient='records'),
            'sample_size': len(cleaned_data),
            'satisfaction_rate': cleaned_data['satisfaction'].mean() if 'satisfaction' in cleaned_data.columns else None
        }
        
        return jsonify(insights)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get descriptive statistics about the dataset"""
    if cleaned_data is None:
        return jsonify({'error': 'Data not loaded. Please load the data first.'}), 503
    
    try:
        # Get category filter if provided
        category = request.args.get('category', None)
        
        if category and category in cleaned_data.columns:
            # Get unique values for the category
            unique_values = cleaned_data[category].unique().tolist()
            
            # Calculate satisfaction rate by category
            satisfaction_by_category = []
            for value in unique_values:
                subset = cleaned_data[cleaned_data[category] == value]
                sat_rate = subset['satisfaction'].mean() if 'satisfaction' in subset.columns else None
                
                satisfaction_by_category.append({
                    'category': category,
                    'value': value,
                    'count': len(subset),
                    'satisfaction_rate': sat_rate
                })
            
            stats = {
                'category_stats': satisfaction_by_category
            }
        else:
            # General dataset statistics
            # Convert numeric columns to dictionary
            numeric_stats = cleaned_data.describe().to_dict()
            
            # Get distribution of categorical columns
            categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
            categorical_stats = {}
            
            for col in categorical_columns:
                categorical_stats[col] = cleaned_data[col].value_counts().to_dict()
            
            stats = {
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats,
                'total_records': len(cleaned_data)
            }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))