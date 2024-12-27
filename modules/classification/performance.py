import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from utils import impute_missing_values

# ---- Paths for Models and Dataset ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes_012_health_indicators.pkl')
MODEL_PATH_A = os.path.join(BASE_DIR, 'models', 'classification_model_a.pkl')
MODEL_PATH_B = os.path.join(BASE_DIR, 'models', 'classification_model_b.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'classification_features.pkl')


# ---- Load Dataset ----
@st.cache_data(ttl=0)
def load_data():
    """
    Load dataset from the specified path or allow manual upload.
    """
    st.write("🛠️ **Debug Info:**")
    st.write(f"Resolved Dataset Path: `{DATA_PATH}`")
    st.write(f"File Exists: `{os.path.exists(DATA_PATH)}`")

    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_pickle(DATA_PATH)
            st.success("✅ Dataset loaded successfully from path!")
            return df
        else:
            uploaded_file = st.file_uploader("📂 Upload Dataset (.pkl)", type=['pkl'])
            if uploaded_file:
                df = pd.read_pickle(uploaded_file)
                st.success("✅ Dataset uploaded manually for evaluation!")
                return df
            else:
                st.error("❌ Dataset not found. Please upload or train the dataset in the Train Module.")
                st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()


# ---- Load Model ----
@st.cache_resource
def load_model(model_path):
    """
    Load a trained model from the given path.
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"❌ Model file not found at {model_path}. Please train the model first.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# ---- Adjust Class Mismatch ----
def adjust_class_mismatch(y_true, y_proba, model_classes):
    """
    Adjust probability predictions to align with the true class labels.
    """
    unique_classes = np.unique(y_true)
    adjusted_proba = np.zeros((len(y_true), len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        if cls in model_classes:
            adjusted_proba[:, i] = y_proba[:, np.where(model_classes == cls)[0][0]]
    return adjusted_proba, unique_classes


# ---- Evaluation Function ----
def evaluate_model(model, X, y):
    """
    Evaluate a given model and return evaluation metrics.
    """
    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        adjusted_proba, unique_classes = adjust_class_mismatch(y, y_proba, model.classes_)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, adjusted_proba, labels=unique_classes),
            "roc_auc": roc_auc_score(y, adjusted_proba, multi_class='ovr')
        }

        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        classification_rep = classification_report(y, y_pred, target_names=[str(cls) for cls in unique_classes], output_dict=True)

        return metrics, cm, classification_rep, adjusted_proba, unique_classes
    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")
        st.stop()


# ---- Main UI ----
def run():
    st.title("📊 **Evaluate Classification Model Performance**")
    st.write("""
    Evaluate two models (K-Fold and LOOCV) using various performance metrics and visualizations.
    """)

    # ---- Load Dataset ----
    df = load_data()
    if df is not None:
        try:
            df = impute_missing_values(df)
            X, y = df.drop(columns=['Diabetes_012']), df['Diabetes_012']
        except KeyError:
            st.error("⚠️ The dataset must contain a column named 'Diabetes_012'. Please upload a valid dataset.")
            st.stop()

        # ---- Model Selection ----
        st.write("### 🛠️ **Select Evaluation Model**")
        selected_model_option = st.radio(
            "Choose a Model:",
            ["Model A (K-Fold Cross-Validation)", "Model B (Leave-One-Out Cross-Validation)"]
        )

        model_path = MODEL_PATH_A if selected_model_option == "Model A (K-Fold Cross-Validation)" else MODEL_PATH_B
        model = load_model(model_path)

        # ---- Evaluate Model ----
        st.write(f"### 📊 **Evaluating {selected_model_option}**")

        st.write("### 📊 **Performance Metrics**")
        metrics, cm, report, adjusted_proba, unique_classes = evaluate_model(model, X, y)

        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Accuracy", f"{metrics['accuracy']:.2f}")
        col2.metric("📉 Log Loss", f"{metrics['log_loss']:.2f}")
        col3.metric("📈 AUC-ROC", f"{metrics['roc_auc']:.2f}")

        st.write("---")

        # ---- Confusion Matrix ----
        st.write("### 📊 **Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=unique_classes, yticklabels=unique_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        st.write("---")

        # ---- Classification Report ----
        st.write("### 📑 **Classification Report**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))

        st.write("---")

        # ---- ROC Curve ----
        st.write("### 📈 **ROC Curve**")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cls in enumerate(unique_classes):
            fpr, tpr, _ = roc_curve(y == cls, adjusted_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {cls}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(fig)

        st.write("---")

        # ---- Insights ----
        st.write("### 📚 **Performance Metric Interpretations**")
        st.markdown("""
        - **📏 Accuracy:** Proportion of correct predictions.  
        - **📉 Log Loss:** Measures the uncertainty of predictions. Lower is better.  
        - **📊 Confusion Matrix:** True/False Positives and Negatives.  
        - **📑 Classification Report:** Precision, Recall, F1-Score per class.  
        - **📈 AUC-ROC Curve:** Distinguishes class separation performance.  
        """)

        st.success("✅ Evaluation completed successfully!")


# ---- Run Standalone ----
if __name__ == '__main__':
    run()
