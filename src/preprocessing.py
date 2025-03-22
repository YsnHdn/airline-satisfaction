import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath):
    """
    Load the airline satisfaction dataset
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and formatting the target variable
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 0', axis=1)
    
    # Convert the satisfaction column to binary (1 for satisfied, 0 for neutral or dissatisfied)
    if 'satisfaction' in df_clean.columns:
        df_clean['satisfaction'] = df_clean['satisfaction'].apply(
            lambda x: 1 if x == 'satisfied' else 0
        )
    
    # Handle missing values - for numeric columns, use median
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical columns, fill with most frequent value
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean


def get_feature_names_from_column_transformer(column_transformer):
    """Get feature names from a ColumnTransformer"""
    feature_names = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe, 'get_feature_names_out'):
                feature_names.extend(pipe.get_feature_names_out())
            else:
                feature_names.extend(features)
    
    return feature_names


def create_preprocessing_pipeline(df):
    """
    Create a preprocessing pipeline for the data
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (preprocessing_pipeline, feature_names)
    """
    # Identify categorical and numeric features
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target variable if it's in the features
    if 'satisfaction' in num_features:
        num_features.remove('satisfaction')
    
    # Remove ID column if it exists
    if 'id' in num_features:
        num_features.remove('id')
    
    # Define preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    return preprocessor


def get_key_factors(df):
    """
    Get the key factors influencing satisfaction
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: DataFrame with correlation values
    """
    # Ensure satisfaction is numeric
    if 'satisfaction' in df.columns and not pd.api.types.is_numeric_dtype(df['satisfaction']):
        df = df.copy()
        df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
    
    # Calculate correlation with satisfaction
    if 'satisfaction' in df.columns:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Remove id column if it exists
        if 'id' in numeric_df.columns:
            numeric_df = numeric_df.drop('id', axis=1)
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        if 'satisfaction' in corr_matrix.columns:
            corr_with_satisfaction = corr_matrix['satisfaction'].sort_values(ascending=False)
            return pd.DataFrame(corr_with_satisfaction).reset_index().rename(
                columns={'index': 'feature', 'satisfaction': 'correlation'}
            )
    
    # Return empty DataFrame if satisfaction column is not found or other issues
    return pd.DataFrame(columns=['feature', 'correlation'])