import pandas as pd
import os
import pickle
import streamlit as st
from sklearn.impute import SimpleImputer

# Ensure Directory Exists
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

# Load Dataset
@st.cache_data
def load_data(uploaded_file=None, file_path=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif file_path and os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        st.warning("⚠️ Please upload or specify a valid dataset path.")
        return None

# Validate Dataset
def validate_dataset(df, task):
    """
    Validate dataset based on the selected task.
    """
    if task == "Regression" and 'AverageTemperature' not in df.columns:
        raise ValueError("❌ Dataset must contain 'AverageTemperature' for regression tasks.")
    if task == "Classification" and 'Diabetes_012' not in df.columns:
        raise ValueError("❌ Dataset must contain 'Diabetes_012' for classification tasks.")

# Preprocess Dataset
def preprocess_data(df, target_column):
    binary_columns = [
        'HighBP', 'HighChol', 'CholCheck', 'Stroke', 'HeartDiseaseorAttack', 
        'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'Fruits', 'Veggies', 
        'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
    ]
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def impute_missing_values(df):
    """
    Handle missing values in the dataset using mean imputation.
    """
    try:
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    except Exception as e:
        raise ValueError(f"❌ Error imputing missing values: {e}")

def load_features(path):
    """
    Load saved feature names from the specified path.
    """
    try:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            raise FileNotFoundError(f"❌ No feature file found at {path}.")
    except Exception as e:
        raise ValueError(f"❌ Error loading features: {e}")