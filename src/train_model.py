import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import preprocessing and model functions
from preprocessing import load_data, clean_data
from model import train_model, evaluate_model, plot_feature_importance

def main():
    """Train the model and save it to disk"""
    print("Starting model training...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    
    # Path to data and model
    data_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/satisfaction_model.pkl')
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_data(data_path)
    df_clean = clean_data(df)
    
    # Split features and target
    print("Preparing features and target...")
    if 'satisfaction' in df_clean.columns:
        X = df_clean.drop('satisfaction', axis=1)
        y = df_clean['satisfaction']
        
        # Remove ID column if it exists
        if 'id' in X.columns:
            X = X.drop('id', axis=1)
        
        # Train model
        print("Training model...")
        model, X_test, y_test = train_model(X, y, model_path)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Print metrics
        print("Model Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print("Model training completed successfully!")
        return 0
    else:
        print("Error: 'satisfaction' column not found in the dataset.")
        return 1

if __name__ == "__main__":
    sys.exit(main())