import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.impute import SimpleImputer

# ---- Paths ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
MODEL_PATH_A = os.path.join(MODEL_DIR, 'regression_model_a.pkl')
MODEL_PATH_B = os.path.join(MODEL_DIR, 'regression_model_b.pkl')
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, 'temp_uploaded_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'regression_features.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# ---- Load Model ----
@st.cache_resource
def load_regression_model(model_type=None, custom_model=None):
    """
    Load the regression model based on the selected technique or custom upload.
    """
    try:
        if custom_model:
            with open(TEMP_MODEL_PATH, 'wb') as f:
                f.write(custom_model.getbuffer())  # Save uploaded model
            with open(TEMP_MODEL_PATH, 'rb') as file:
                return pickle.load(file), TEMP_MODEL_PATH
        else:
            model_path = MODEL_PATH_A if model_type == "Train-Test Split" else MODEL_PATH_B
            with open(model_path, 'rb') as file:
                return pickle.load(file), model_path
    except FileNotFoundError:
        st.error("❌ Model not found. Please train the model first or upload a valid model file.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


# ---- Load Feature Names ----
def load_features():
    """
    Load saved feature names for the model.
    """
    try:
        with open(FEATURES_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Feature file not found. Please ensure the model is properly trained.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading feature names: {e}")
        return None


# ---- Make Prediction ----
def make_prediction(model, user_input, feature_names):
    """
    Make predictions using the regression model.
    """
    try:
        # Create DataFrame with all required features
        input_data = pd.DataFrame([user_input], columns=feature_names)
        
        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        aligned_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=feature_names)
        
        # Validate Feature Shape
        if aligned_data_imputed.shape[1] != len(feature_names):
            raise ValueError(f"Feature mismatch: Expected {len(feature_names)} features, got {aligned_data_imputed.shape[1]}")

        # Make Prediction
        prediction = model.predict(aligned_data_imputed)
        return prediction[0]
    
    except ValueError as ve:
        st.error(f"❌ Prediction failed: {ve}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during prediction: {e}")
        return None


# ---- UI ----
def run():
    """
    Streamlit UI for regression predictions.
    """
    st.title("🔮 **Temperature Prediction Interface**")
    st.write("""
    Use the trained regression model to predict temperature based on input values or upload your own model.
    """)

    # ---- Model Selection ----
    st.write("### 🛠️ **Select Prediction Model**")
    model_source = st.radio(
        "Choose Model Source:",
        ["Pre-Trained Model", "Upload Custom Model"]
    )
    
    model = None
    model_path = None
    feature_names = None
    
    if model_source == "Pre-Trained Model":
        model_type = st.radio(
            "Choose Resampling Technique:",
            ["Train-Test Split", "Repeated Train-Test Splits"]
        )
        model, model_path = load_regression_model(model_type=model_type)
        feature_names = load_features()
    else:
        uploaded_model = st.file_uploader("📂 Upload a Pre-trained Model (.pkl)", type=['pkl'])
        if uploaded_model:
            model, model_path = load_regression_model(custom_model=uploaded_model)
            feature_names = load_features()
        else:
            st.warning("⚠️ Please upload a valid regression model file.")

    if model is None or feature_names is None:
        st.stop()
    
    # ---- Model Download ----
    st.write("### 📥 **Download Selected Model**")
    if model_path:
        with open(model_path, 'rb') as f:
            st.download_button(
                label="⬇️ Download Selected Model",
                data=f,
                file_name=os.path.basename(model_path),
                mime='application/octet-stream'
            )

    # ---- Input Section ----
    st.write("### 📝 **Enter Input for Prediction**")
    user_input = {}

    user_input['Latitude'] = st.number_input("🌍 Enter Latitude (e.g., 14.60)", min_value=-90.0, max_value=90.0, value=14.60)
    user_input['Longitude'] = st.number_input("🧭 Enter Longitude (e.g., 120.9842)", min_value=-180.0, max_value=180.0, value=120.98)
    user_input['Year'] = st.number_input("📅 Enter Year (e.g., 2023)", min_value=1700, max_value=2100, value=2023)
    user_input['Month'] = st.selectbox("📆 Select Month", list(range(1, 13)), index=0)
    user_input['AverageTemperatureUncertainty'] = st.number_input("🌡️ Enter Temperature Uncertainty (e.g., 0.5)", min_value=0.0, max_value=10.0, value=0.5)

    # Fill missing features with default values
    for feature in feature_names:
        if feature not in user_input:
            user_input[feature] = 0.0

    # ---- Predict Button ----
    if st.button("🔍 Predict Temperature"):
        with st.spinner("Processing Prediction..."):
            prediction = make_prediction(model, user_input, feature_names)
            if prediction is not None:
                st.success(f"🌡️ **Predicted Temperature:** {prediction:.2f}°C")
            else:
                st.error("❌ Failed to generate prediction.")

    # ---- Download Sample Input Data ----
    st.write("### 📥 **Download Sample Input Data**")
    sample_input = pd.DataFrame([{feature: 0.0 for feature in feature_names}])
    csv = sample_input.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Sample Input Data as CSV",
        data=csv,
        file_name='sample_input_data.csv',
        mime='text/csv'
    )


# ---- Run the App Standalone ----
if __name__ == "__main__":
    run()
