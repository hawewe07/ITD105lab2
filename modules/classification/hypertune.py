import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import json

# ---- Paths ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(DATA_DIR, 'diabetes_012_health_indicators.pkl')
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_params.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Load Dataset ----
@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """
    Load dataset from uploaded file or cached location.
    """
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_pickle(DATA_PATH)  # Save dataset as pickle for caching
            st.success("✅ Dataset uploaded and saved successfully!")
        elif os.path.exists(DATA_PATH):
            df = pd.read_pickle(DATA_PATH)
        else:
            st.warning("⚠️ Please upload a dataset first!")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()


# ---- Hyperparameter Tuning ----
def hyperparameter_tuning(X, y, param_grid):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    try:
        model = GridSearchCV(
            LogisticRegression(solver='liblinear'),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        model.fit(X, y)
        return model.best_params_, model.best_score_
    except Exception as e:
        st.error(f"❌ Error during hyperparameter tuning: {e}")
        st.stop()


# ---- Save Best Parameters ----
def save_best_params(best_params):
    """
    Save best hyperparameters to a JSON file.
    """
    try:
        with open(BEST_PARAMS_PATH, 'w') as f:
            json.dump(best_params, f)
        st.success("✅ Best parameters saved successfully!")
    except Exception as e:
        st.error(f"❌ Error saving best parameters: {e}")


# ---- Main Streamlit App ----
def run():
    st.title("🔍 **Hyperparameter Tuning for Logistic Regression**")
    st.write("""
    Optimize your Logistic Regression model using GridSearchCV to find the best parameters.
    """)

    # 📂 Dataset Upload
    st.write("### 📂 **Upload Dataset**")
    uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'])

    # Load Dataset
    df = load_data(uploaded_file)

    if df is not None:
        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())

        # ---- Preprocessing ----
        try:
            X = df.drop(columns=['Diabetes_012'])
            y = df['Diabetes_012']
        except KeyError:
            st.error("⚠️ The dataset must contain a column named 'Diabetes_012'. Please upload a valid dataset.")
            st.stop()

        st.write("---")

        # ---- Hyperparameter Grid Selection ----
        st.write("### 🛠️ **Hyperparameter Grid Settings**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            C_values = st.multiselect("Regularization Strength (C):", [0.1, 1, 10], default=[0.1, 1, 10])
        
        with col2:
            max_iter_values = st.multiselect("Max Iterations:", [100, 200, 500], default=[100, 200])
        
        with col3:
            solver = st.selectbox("Solver:", ['liblinear', 'lbfgs'], index=0)
        
        param_grid = {
            'C': C_values,
            'max_iter': max_iter_values,
            'solver': [solver]
        }

        st.write("---")
        
        # ---- Run Hyperparameter Tuning ----
        if st.button("🚀 Start Hyperparameter Tuning"):
            with st.spinner("🔄 Running Hyperparameter Tuning... Please wait!"):
                try:
                    best_params, best_score = hyperparameter_tuning(X, y, param_grid)
                    save_best_params(best_params)
                    
                    st.success("🏆 Hyperparameter Tuning Completed Successfully!")
                    
                    st.write("### 🏅 **Best Parameters**")
                    st.json(best_params)

                    st.write("### 📊 **Best Score**")
                    st.write(f"**Accuracy:** {best_score:.2f}")
                    
                    # Download Best Parameters
                    st.download_button(
                        label="⬇️ Download Best Parameters",
                        data=json.dumps(best_params, indent=4),
                        file_name='best_hyperparams.json',
                        mime='application/json'
                    )
                    
                except Exception as e:
                    st.error(f"❌ Hyperparameter tuning failed: {e}")
                    st.stop()
        
        st.write("---")
        st.info("Tip: Adjust the hyperparameter grid settings for better optimization results.")


# ---- Run Standalone ----
if __name__ == '__main__':
    run()
