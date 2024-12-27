import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import os
import pickle  
import numpy as np


# ---- Model Dictionary ----
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree 2)": Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}


# ---- Paths ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
FEATURES_PATH = os.path.join(MODEL_DIR, 'regression_features.pkl')


# ---- Load Features ----
def load_features():
    """Load the features used during model training."""
    try:
        with open(FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        return features
    except FileNotFoundError:
        st.error("❌ Feature file not found. Train the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading feature names: {e}")
        return None


# ---- Preprocess Dataset ----
def preprocess_data(df, features):
    """
    Preprocess dataset to align with model training data.
    """
    try:
        # Convert date columns to numeric timestamps
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or col == 'dt':
                st.warning(f"📅 Converting date column '{col}' to numerical format (timestamp).")
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') / 1e9
        
        # Keep only relevant features
        df = df[features]
        
        # Handle missing values with SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=features)
        
        return df
    except KeyError as e:
        st.error(f"❌ Missing required features: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error preprocessing data: {e}")
        return None


# ---- Evaluate Model ----
def evaluate_model(model, X, y):
    """
    Evaluate a regression model and return metrics.
    """
    try:
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mae, r2
    except Exception as e:
        st.error(f"❌ Error evaluating model: {e}")
        return None, None, None


# ---- Clean Dataset ----
def clean_dataset(df):
    """
    Clean dataset by handling non-numeric and missing values.
    """
    try:
        if 'AverageTemperature' not in df.columns:
            st.error("❌ Dataset must contain 'AverageTemperature'.")
            return None

        # Drop rows where the target variable is NaN
        df = df.dropna(subset=['AverageTemperature'])
        
        # Convert date columns to numeric timestamps
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or col == 'dt':
                st.warning(f"📅 Converting date column '{col}' to numerical format (timestamp).")
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') / 1e9
        
        # Select numeric columns only
        df = df.select_dtypes(include=['number'])
        
        # Fill missing values in numeric columns
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        return df
    except Exception as e:
        st.error(f"❌ Error cleaning dataset: {e}")
        return None


# ---- Compare Models ----
def compare_models(df):
    """
    Compare multiple regression models using a common dataset.
    """
    try:
        df = clean_dataset(df)
        if df is None:
            return

        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())

        # Prepare Features and Target
        y = df['AverageTemperature']
        X = df.drop(columns=['AverageTemperature'])

        # Load Features
        features = load_features()
        if features is None:
            return

        # Preprocess Dataset
        X = preprocess_data(X, features)
        if X is None:
            return

        # User selects models to compare
        st.sidebar.write("### ⚙️ **Select Models for Comparison**")
        selected_models = st.sidebar.multiselect(
            "Choose Models:",
            list(models.keys()),
            default=["Linear Regression", "Polynomial Regression (Degree 2)"]
        )

        if not selected_models:
            st.warning("⚠️ Please select at least one model for comparison.")
            return

        # Evaluate Models
        st.write("### 📊 **Model Evaluation Metrics**")
        results = []
        for model_name in selected_models:
            model = models[model_name]
            with st.spinner(f"Training {model_name}..."):
                model.fit(X, y)
                mse, mae, r2 = evaluate_model(model, X, y)
                if mse is not None:
                    results.append({
                        "Model": model_name,
                        "MSE": mse,
                        "MAE": mae,
                        "R²": r2
                    })

        # Display Results Table
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Visualization
        st.write("### 📈 **Performance Visualization**")
        fig = px.bar(
            results_df,
            x='Model',
            y=['MSE', 'MAE', 'R²'],
            barmode='group',
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.write("### 🧠 **Insights from Comparison:**")
        best_model = results_df.sort_values(by='R²', ascending=False).iloc[0]
        st.success(f"🏆 **Best Model:** {best_model['Model']} with R²: {best_model['R²']:.2f}")

    except Exception as e:
        st.error(f"❌ Error comparing models: {e}")


# ---- Main Function ----
def run():
    st.title("📊 **Compare Regression Models**")
    st.write("""
    Compare multiple regression models and evaluate their performance.
    """)

    uploaded_file = st.file_uploader("📂 Upload Dataset (.csv)", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            compare_models(df)
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    else:
        st.info("📂 Please upload a dataset to proceed.")
