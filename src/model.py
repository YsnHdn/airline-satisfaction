import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import create_preprocessing_pipeline


def train_model(X, y, model_path='../models/satisfaction_model.pkl'):
    """
    Train a machine learning model for airline satisfaction prediction
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        model_path (str): Path to save the model
        
    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
    """
    # Create preprocessor
    preprocessor = create_preprocessing_pipeline(X)
    
    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Print metrics
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with detailed metrics
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Additional metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def plot_feature_importance(model, X, top_n=10):
    """
    Plot feature importance from the trained model
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        X (pd.DataFrame): Features dataframe
        top_n (int): Number of top features to show
        
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    # Get feature names
    preprocessor = model.named_steps['preprocessor']
    feature_names = []
    
    # Extract feature names from column transformer
    for name, transformer, columns in preprocessor.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                trans_feature_names = transformer.get_feature_names_out(columns)
            else:
                trans_feature_names = columns
            feature_names.extend(trans_feature_names)
    
    # Get feature importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame for visualization
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Features by Importance')
    plt.tight_layout()
    
    return plt.gcf()


def predict_satisfaction(model, input_data):
    """
    Make prediction with the trained model
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        input_data (dict or pd.DataFrame): Input data
        
    Returns:
        dict: Prediction results with probability
    """
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'satisfied' if prediction == 1 else 'not satisfied',
        'probability': float(probability[1]) if prediction == 1 else float(probability[0])
    }
    
    return result