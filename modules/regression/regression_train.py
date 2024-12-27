import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import r2_score
import pickle
import os

# ---- Paths ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_regression_dataset.csv')
PROCESSED_DATA_PKL_PATH = os.path.join(DATA_DIR, 'processed_regression_dataset.pkl')
MODEL_PATH_A = os.path.join(MODEL_DIR, 'regression_model_a.pkl')
MODEL_PATH_B = os.path.join(MODEL_DIR, 'regression_model_b.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'regression_features.pkl')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Preprocess Dataset ----
def preprocess_regression_data(df):
    """
    Preprocess dataset for regression model with optimizations.
    """
    try:
        # Handle date columns efficiently
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or col == 'dt':
                st.warning(f"📅 Converting date column '{col}' to numerical format (timestamp).")
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') / 1e9

        # Drop non-numerical columns
        df = df.select_dtypes(include=['number'])
        
        # Drop columns with more than 50% missing values
        df = df.loc[:, df.isnull().mean() < 0.5]
        
        # Fill remaining missing values with column median
        df.fillna(df.median(), inplace=True)
        
        # Limit to top 30 numerical columns
        if df.shape[1] > 30:
            st.warning("⚠️ Too many features detected. Limiting to top 30 numerical features.")
            df = df.iloc[:, :30]
        
        # Save the preprocessed dataset
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        df.to_pickle(PROCESSED_DATA_PKL_PATH)
        
        return df
    except Exception as e:
        raise ValueError(f"❌ Error during preprocessing: {e}")


# ---- Train Model A (Train-Test Split) ----
@st.cache_resource
def train_model_a(X, y, sample_fraction=0.5):
    """
    Train a Linear Regression model using Train-Test Split.
    """
    try:
        st.write("### 📊 **Training Model A: Train-Test Split**")
        progress_bar = st.progress(0)
        
        if len(X) > 10000:
            st.warning("⚠️ Dataset is large. Sampling 50% of the data for training.")
            X, _, y, _ = train_test_split(X, y, test_size=1-sample_fraction, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        progress_bar.progress(100)
        
        with open(MODEL_PATH_A, 'wb') as f:
            pickle.dump(model, f)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        st.success(f"✅ Model A trained successfully with R² Score: {r2:.2f}")
        return r2
    except Exception as e:
        st.error(f"❌ Error training Model A: {e}")
        st.stop()


# ---- Train Model B (Repeated Random Train-Test Splits) ----
@st.cache_resource
def train_model_b(X, y, sample_fraction=0.5):
    """
    Train a Linear Regression model using Repeated Random Train-Test Splits.
    """
    try:
        st.write("### 📊 **Training Model B: Repeated Random Train-Test Splits**")
        progress_bar = st.progress(0)
        
        if len(X) > 10000:
            st.warning("⚠️ Dataset is large. Sampling 50% of the data for training.")
            X, _, y, _ = train_test_split(X, y, test_size=1-sample_fraction, random_state=42)
        
        model = LinearRegression(n_jobs=-1)
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        avg_r2 = scores.mean()
        model.fit(X, y)
        
        progress_bar.progress(100)
        
        with open(MODEL_PATH_B, 'wb') as f:
            pickle.dump(model, f)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        st.success(f"✅ Model B trained successfully with Avg R² Score: {avg_r2:.2f}")
        return avg_r2
    except Exception as e:
        st.error(f"❌ Error training Model B: {e}")
        st.stop()


# ---- Main Function ----
def run():
    """
    UI for Regression Model Training.
    """
    st.title("🌡️ **Train and Download Regression Models**")
    st.write("Upload a dataset and train two Linear Regression models with different resampling techniques.")
    
    uploaded_file = st.file_uploader("📂 Upload Regression Dataset (.csv)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        try:
            df = preprocess_regression_data(df)
        except Exception as e:
            st.error(f"❌ Error preprocessing dataset: {e}")
            return
        
        X = df.drop(columns=['AverageTemperature'])
        y = df['AverageTemperature']
        
        technique = st.selectbox("Choose Resampling Technique", ["Train-Test Split", "Repeated Random Train-Test Splits"])
        
        if st.button("🚀 Train Model"):
            if technique == "Train-Test Split":
                train_model_a(X, y)
                model_path = MODEL_PATH_A
            else:
                train_model_b(X, y)
                model_path = MODEL_PATH_B
            
            # Model Download
            st.download_button("⬇️ Download Trained Model", data=open(model_path, 'rb'), file_name=os.path.basename(model_path))
            st.download_button("⬇️ Download Processed Dataset (CSV)", data=open(PROCESSED_DATA_PATH, 'rb'), file_name='processed_dataset.csv')
            st.download_button("⬇️ Download Processed Dataset (PKL)", data=open(PROCESSED_DATA_PKL_PATH, 'rb'), file_name='processed_dataset.pkl')


# ---- Footer ----
st.write("---")
st.info("Tip: Download your trained model and processed datasets for deployment or further analysis.")

if __name__ == '__main__':
    run()
