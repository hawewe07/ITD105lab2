import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import os

# ---- Paths ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_classification_dataset.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'classification_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'classification_features.pkl')

# Ensure Directories Exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---- Preprocess Dataset ----
def preprocess_classification_data(df):
    """
    Preprocess dataset for classification model.
    """
    try:
        # Drop non-numerical columns except target
        if 'Diabetes_012' in df.columns:
            target = df['Diabetes_012']
            df = df.drop(columns=['Diabetes_012'])
        else:
            raise ValueError("⚠️ The dataset must contain a column named 'Diabetes_012'.")

        # Select numerical columns
        df = df.select_dtypes(include=['number'])
        
        # Fill missing values with column median
        df.fillna(df.median(), inplace=True)
        
        # Re-add target column
        df['Diabetes_012'] = target
        
        # Save preprocessed dataset
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing dataset: {e}")
        return None


# ---- Train Classification Model ----
@st.cache_resource
def train_classification_model(X, y, technique, k=None):
    """
    Train a Logistic Regression model using the selected resampling technique.
    """
    try:
        st.write("### 📊 **Training Classification Model**")
        progress_bar = st.progress(0)
        
        model = LogisticRegression(max_iter=1000)
        if technique == "K-Fold Cross-Validation":
            if k is None:
                st.error("⚠️ Please select a valid number of folds for K-Fold Cross-Validation.")
                st.stop()
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        else:
            cv = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        progress_bar.progress(100)
        model.fit(X, y)
        accuracy = scores.mean()
        
        # Save the trained model and feature names
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        st.success(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        st.error(f"❌ Error training classification model: {e}")
        st.stop()


# ---- Main Function ----
def run():
    """
    UI for Classification Model Training.
    """
    st.title("🏋️ **Train Classification Model**")
    st.write("Upload a dataset and train a Logistic Regression model using cross-validation techniques.")
    
    # 📂 Dataset Upload
    uploaded_file = st.file_uploader("📂 Upload Classification Dataset (.csv)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Preprocess Dataset
            df = preprocess_classification_data(df)
            if df is None:
                return
            
            st.write("### 🗂️ **Dataset Overview**")
            st.dataframe(df.head())
            
            # Feature and Target Selection
            try:
                X = df.drop(columns=['Diabetes_012'])
                y = df['Diabetes_012']
            except KeyError:
                st.error("⚠️ The dataset must contain a column named 'Diabetes_012'.")
                st.stop()
            
            # 📊 Resampling Technique Selection
            st.write("### 🔄 **Resampling Technique**")
            col1, col2 = st.columns(2)
            
            with col1:
                resampling = st.selectbox("Choose Technique", ["K-Fold Cross-Validation", "Leave-One-Out Cross-Validation"])
            
            with col2:
                k = st.slider("Number of Folds", 2, 10, 5, step=1) if resampling == "K-Fold Cross-Validation" else None
            
            # 🚀 Train Model
            if st.button("🚀 Train Model"):
                with st.spinner("Training the classification model..."):
                    accuracy = train_classification_model(X, y, resampling, k)
                    st.info(f"🎯 Model trained successfully with accuracy: {accuracy:.2f}")
                    
                    # Download Preprocessed Dataset
                    with open(PROCESSED_DATA_PATH, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Preprocessed Dataset", 
                            f, 
                            file_name='processed_classification_dataset.csv',
                            mime='text/csv'
                        )
                    
                    # Download Trained Model
                    with open(MODEL_PATH, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Trained Model", 
                            f, 
                            file_name='classification_model.pkl',
                            mime='application/octet-stream'
                        )
                    
        except Exception as e:
            st.error(f"❌ Error processing dataset: {e}")
    else:
        st.info("📂 Please upload a classification dataset to proceed.")


# ---- Footer ----
st.write("---")
st.info("Tip: Adjust the resampling technique and folds to optimize model performance.")
