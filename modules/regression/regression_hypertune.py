import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
import pickle
import os

# ---- Paths ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
MODEL_PATH = os.path.join(MODEL_DIR, 'regression_tuned_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'regression_tuned_features.pkl')

# Ensure Directories Exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Preprocessing ----
def preprocess_regression_data(df):
    """
    Preprocess dataset for regression model.
    """
    try:
        # Handle date columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                st.warning(f"📅 Converting date column '{col}' to numerical format (timestamp).")
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') / 1e9
        
        # Drop non-numerical columns
        df = df.select_dtypes(include=['number'])
        
        # Fill missing values with column median
        df.fillna(df.median(), inplace=True)
        
        return df
    except Exception as e:
        raise ValueError(f"❌ Error during preprocessing: {e}")


# ---- Hyperparameter Tuning ----
def tune_model(X, y, model_type, param_grid):
    """
    Perform hyperparameter tuning on Ridge or Lasso regression models.
    """
    try:
        st.write("### 📊 **Hyperparameter Tuning in Progress**")
        progress_bar = st.progress(0)
        
        model = Ridge() if model_type == "Ridge" else Lasso()
        grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        progress_bar.progress(100)
        
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Save the best model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save feature names
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        return best_model, best_score, best_params
    except Exception as e:
        raise ValueError(f"❌ Error during model tuning: {e}")


# ---- Main Function ----
def run():
    """
    UI for Hyperparameter Tuning of Regression Models.
    """
    st.title("🛠️ **Hyperparameter Tuning for Regression Model**")
    st.write("""
    Fine-tune Ridge or Lasso regression models to achieve the best predictive performance.
    """)
    
    # Dataset Upload
    uploaded_file = st.file_uploader("📂 Upload Regression Dataset (.csv)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_regression_data(df)
        except Exception as e:
            st.error(f"❌ Error preprocessing dataset: {e}")
            return
        
        if 'AverageTemperature' not in df.columns:
            st.error("⚠️ Dataset must contain a column named 'AverageTemperature'.")
            return
        
        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())
        
        # Feature and Target Selection
        X = df.drop(columns=['AverageTemperature'])
        y = df['AverageTemperature']
        
        # Model Selection
        st.write("### ⚙️ **Model and Parameters**")
        model_type = st.radio("Choose Model Type:", ["Ridge", "Lasso"])
        
        # Dynamic Hyperparameter Grid
        st.write("### 🎛️ **Hyperparameter Grid**")
        param_grid = {}
        if model_type == "Ridge":
            param_grid['alpha'] = st.multiselect("Alpha Values for Ridge:", [0.1, 1.0, 10.0, 100.0], default=[1.0, 10.0])
        else:
            param_grid['alpha'] = st.multiselect("Alpha Values for Lasso:", [0.1, 1.0, 10.0, 100.0], default=[1.0, 10.0])
        
        if not param_grid['alpha']:
            st.warning("⚠️ Please select at least one value for 'alpha'.")
            return
        
        # Hyperparameter Tuning
        if st.button("🚀 Tune Model"):
            with st.spinner("Tuning model hyperparameters..."):
                try:
                    best_model, best_score, best_params = tune_model(X, y, model_type, param_grid)
                    
                    st.success(f"✅ Model tuned successfully with R² Score: {best_score:.2f}")
                    st.write(f"**Best Parameters:** {best_params}")
                    
                    # Download Model
                    with open(MODEL_PATH, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Tuned Model",
                            data=f,
                            file_name="regression_tuned_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"❌ Error during hyperparameter tuning: {e}")
    else:
        st.info("📂 Please upload a regression dataset to proceed.")


# ---- Footer ----
st.write("---")
st.info("Tip: Adjust hyperparameter ranges for better optimization results.")
