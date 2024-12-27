import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- Paths ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
MODEL_PATH_A = os.path.join(MODEL_DIR, 'regression_model_a.pkl')
MODEL_PATH_B = os.path.join(MODEL_DIR, 'regression_model_b.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'regression_features.pkl')


# ---- Load Model ----
def load_model(model_path):
    """
    Load the trained regression model.
    """
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Model not found. Train the model first.")
        return None


# ---- Load Feature Names ----
def load_features():
    """
    Load feature names used during training.
    """
    try:
        with open(FEATURES_PATH, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Feature file not found. Train the model first.")
        return None


# ---- Evaluate Regression Model ----
def evaluate_regression_model(df, model_type):
    """
    Evaluate the regression model with given dataset.
    """
    st.write("### 📊 **Evaluate Regression Model**")
    try:
        features = load_features()
        if features is None:
            return
        
        if not all(feature in df.columns for feature in features):
            st.error("❌ Dataset does not contain the required features.")
            return
        
        X = df[features]
        y = df['AverageTemperature']
        
        model_path = MODEL_PATH_A if model_type == "Train-Test Split" else MODEL_PATH_B
        model = load_model(model_path)
        if model is None:
            return
        
        # Make Predictions
        y_pred = model.predict(X)
        
        # Evaluation Metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Display Metrics
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        
        # Visualization
        st.write("### 📈 **Residual Plot**")
        residuals = y - y_pred
        st.scatter_chart(pd.DataFrame({'Actual': y, 'Predicted': y_pred, 'Residuals': residuals}))
        
    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")


# ---- UI ----
def run():
    """
    UI for Regression Model Evaluation.
    """
    st.title("📊 **Evaluate Regression Models (A & B)**")
    st.write("Upload a dataset and evaluate trained regression models.")
    
    # Dataset Upload
    uploaded_file = st.file_uploader("📂 Upload Regression Dataset (.csv)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading uploaded dataset: {e}")
            return
        
        if 'AverageTemperature' not in df.columns:
            st.error("⚠️ Dataset must contain a column named 'AverageTemperature'.")
            return
        
        # Select Model
        model_type = st.radio(
            "Choose Model for Evaluation:",
            ["Train-Test Split (Model A)", "Repeated Train-Test Splits (Model B)"]
        )
        
        if st.button("📊 Evaluate Model"):
            with st.spinner("Evaluating the model..."):
                if model_type == "Train-Test Split (Model A)":
                    evaluate_regression_model(df, "Train-Test Split")
                else:
                    evaluate_regression_model(df, "Repeated Train-Test Splits")
    else:
        st.info("📂 Please upload a regression dataset to proceed.")


# ---- Footer ----
st.write("---")
st.info("Tip: Ensure your dataset contains the same features used during training.")
