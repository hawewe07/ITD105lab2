import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer
from utils import load_features, impute_missing_values

# ---- Paths ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH_A = os.path.join(BASE_DIR, 'models', 'classification_model_a.pkl')
MODEL_PATH_B = os.path.join(BASE_DIR, 'models', 'classification_model_b.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'features.pkl')


# ---- Load Model ----
@st.cache_resource
def load_model(model_path):
    """Load the selected trained model."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at {model_path}. Please ensure the model exists.")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ---- Prediction Function ----
def make_prediction(model, user_input, feature_names):
    """
    Make predictions using the trained model.
    """
    try:
        # Ensure all features are present
        for feature in feature_names:
            if feature not in user_input:
                user_input[feature] = 0  # Default value for missing features
        
        # Reorder user input to match feature names
        user_df = pd.DataFrame([user_input], columns=feature_names)

        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        user_df_imputed = pd.DataFrame(imputer.fit_transform(user_df), columns=feature_names)

        # Get probabilities for each class (0 = low risk, 1 = medium risk, 2 = high risk)
        y_proba = model.predict_proba(user_df_imputed)
        
        # Display the probability for debugging purposes
        st.write(f"Prediction probabilities: {y_proba}")

        # Get the index of the class with the highest probability
        predicted_class = y_proba[0].argmax()  # Get the class with the highest probability

        # Map the predicted class to the corresponding label
        if predicted_class == 2:
            return 1  # High risk
        else:
            return 0  # Low risk

    except KeyError as e:
        st.error(f"❌ Feature mismatch: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None


# ---- Streamlit UI ----
def run():
    st.title("🩺 **Diabetes Prediction Interface (Dual Model Selection)**")
    st.write("""
    Enter all required health indicators, choose the trained model, and predict the likelihood of diabetes.
    """)

    # ---- Load Feature Names ----
    if not os.path.exists(FEATURES_PATH):
        st.error(f"❌ Feature file not found at `{FEATURES_PATH}`. Please retrain your model to generate feature names.")
        st.stop()
    
    try:
        feature_names = load_features(FEATURES_PATH)
        if feature_names is None:
            raise ValueError("❌ Feature names could not be loaded.")
    except Exception as e:
        st.error(f"❌ Error loading feature names: {e}")
        st.stop()

    # ---- Model Selection ----
    st.sidebar.title("📦 **Select Prediction Model**")
    selected_model_option = st.sidebar.radio(
        "Choose a Model:",
        ["Model A (K-Fold Cross-Validation)", "Model B (Leave-One-Out Cross-Validation)"]
    )

    model_path = MODEL_PATH_A if selected_model_option == "Model A (K-Fold Cross-Validation)" else MODEL_PATH_B
    model = load_model(model_path)

    if model is None:
        st.stop()

    st.write(f"### 🛠️ Using **{selected_model_option}** for Predictions")

    # ---- User Input Section ----
    st.write("### 📝 **Enter Health Indicators**")

    user_input = {}

    # --- Demographics ---
    with st.expander("👤 **Demographics**", expanded=True):
        user_input['Age'] = st.number_input("🗓️ Age", min_value=0, max_value=120, value=30)
        user_input['Sex'] = st.radio("⚧️ Sex", ['Male', 'Female'])
        user_input['Education'] = st.selectbox("📚 Education Level", range(1, 7))
        user_input['Income'] = st.selectbox("💵 Income Level", range(1, 9))

    # --- Health Indicators ---
    with st.expander("💪 **Health Indicators**"):
        user_input['BMI'] = st.number_input("📊 BMI", min_value=0.0, max_value=50.0, value=25.0)
        user_input['HighBP'] = st.radio("💓 High Blood Pressure", ['Yes', 'No'])
        user_input['HighChol'] = st.radio("🩸 High Cholesterol", ['Yes', 'No'])
        user_input['CholCheck'] = st.radio("🩺 Cholesterol Check", ['Yes', 'No'])
        user_input['Stroke'] = st.radio("⚠️ Stroke History", ['Yes', 'No'])
        user_input['HeartDiseaseorAttack'] = st.radio("❤️ Heart Disease or Attack", ['Yes', 'No'])

    # --- Lifestyle Habits ---
    with st.expander("🚴 **Lifestyle Habits**"):
        user_input['Smoker'] = st.radio("🚬 Smoking Status", ['Yes', 'No'])
        user_input['HvyAlcoholConsump'] = st.radio("🍻 Heavy Alcohol Consumption", ['Yes', 'No'])
        user_input['PhysActivity'] = st.radio("🏃 Physical Activity", ['Yes', 'No'])
        user_input['Fruits'] = st.radio("🍎 Fruits Intake", ['Yes', 'No'])
        user_input['Veggies'] = st.radio("🥦 Vegetable Intake", ['Yes', 'No'])

    # --- Healthcare Access ---
    with st.expander("🏥 **Healthcare Access**"):
        user_input['AnyHealthcare'] = st.radio("🏥 Access to Healthcare", ['Yes', 'No'])
        user_input['NoDocbcCost'] = st.radio("💵 Doctor Cost Issue", ['Yes', 'No'])

    # Map Yes/No and Categorical Inputs
    for key in user_input:
        if user_input[key] == 'Yes':
            user_input[key] = 1
        elif user_input[key] == 'No':
            user_input[key] = 0
        elif user_input[key] == 'Male':
            user_input[key] = 1
        elif user_input[key] == 'Female':
            user_input[key] = 0

    if st.button("🔍 Predict"):
        with st.spinner("Processing Prediction..."):
            prediction = make_prediction(model, user_input, feature_names)
            if prediction == 1:
                st.error("⚠️ The model predicts a **high risk of diabetes**.")
            else:
                st.success("✅ The model predicts a **low risk of diabetes**.")


if __name__ == "__main__":
    run()
