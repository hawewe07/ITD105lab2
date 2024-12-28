import pandas as pd
import os
import pickle
import streamlit as st
from sklearn.impute import SimpleImputer


# ---- Directory Management ----
def ensure_directory(path: str):
    """
    Ensure the given directory exists.
    """
    os.makedirs(path, exist_ok=True)


# ---- Dataset Loading ----
@st.cache_data
def load_data(uploaded_file=None, file_path=None):
    """
    Load dataset from an uploaded file or file path.
    """
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        elif file_path and os.path.exists(file_path):
            return pd.read_pickle(file_path)
        else:
            st.warning("⚠️ Please upload or specify a valid dataset path.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None


# ---- Dataset Validation ----
def validate_dataset(df: pd.DataFrame, task: str):
    """
    Validate dataset columns based on the selected task.
    """
    try:
        if task == "Regression" and 'AverageTemperature' not in df.columns:
            raise ValueError("❌ Dataset must contain 'AverageTemperature' for regression tasks.")
        if task == "Classification" and 'Diabetes_012' not in df.columns:
            raise ValueError("❌ Dataset must contain 'Diabetes_012' for classification tasks.")
    except Exception as e:
        raise ValueError(f"❌ Dataset validation failed: {e}")


# ---- Dataset Preprocessing ----
def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess dataset by handling categorical columns and separating target features.
    """
    try:
        # Binary columns to map Yes/No to 1/0
        binary_columns = [
            'HighBP', 'HighChol', 'CholCheck', 'Stroke', 'HeartDiseaseorAttack', 
            'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'Fruits', 'Veggies', 
            'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
        ]

        # Map 'Sex' column if available
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

        # Map binary columns
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})

        # Handle date columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or col == 'dt':
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') / 1e9

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    except KeyError:
        raise ValueError(f"❌ Target column '{target_column}' not found in the dataset.")
    except Exception as e:
        raise ValueError(f"❌ Error during preprocessing: {e}")


# ---- Handle Missing Values ----
def impute_missing_values(df: pd.DataFrame):
    """
    Handle missing values using mean imputation.
    """
    try:
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    except Exception as e:
        raise ValueError(f"❌ Error imputing missing values: {e}")


# ---- Load Features ----
def load_features(path: str):
    """
    Load saved feature names from a pickle file.
    """
    try:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            raise FileNotFoundError(f"❌ No feature file found at {path}.")
    except Exception as e:
        raise ValueError(f"❌ Error loading features: {e}")


# ---- Save Features ----
def save_features(path: str, features: list):
    """
    Save feature names to a pickle file.
    """
    try:
        ensure_directory(os.path.dirname(path))
        with open(path, 'wb') as file:
            pickle.dump(features, file)
    except Exception as e:
        raise ValueError(f"❌ Error saving features: {e}")


# ---- Save Model ----
def save_model(path: str, model):
    """
    Save a trained model to a pickle file.
    """
    try:
        ensure_directory(os.path.dirname(path))
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise ValueError(f"❌ Error saving model: {e}")


# ---- Load Model ----
def load_model(path: str):
    """
    Load a trained model from a pickle file.
    """
    try:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            raise FileNotFoundError(f"❌ Model file not found at {path}.")
    except Exception as e:
        raise ValueError(f"❌ Error loading model: {e}")


# ---- General Utility for Displaying Insights ----
def display_dataset_overview(df: pd.DataFrame):
    """
    Display dataset overview, including head and basic statistics.
    """
    try:
        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())
        with st.expander("📊 **Dataset Statistics**"):
            st.write(df.describe())
    except Exception as e:
        st.error(f"❌ Error displaying dataset overview: {e}")
