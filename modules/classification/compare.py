import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
import numpy as np
from utils import load_data, preprocess_data, impute_missing_values

# ---- Models Dictionary ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Classifier": SVC(),
    "Naive Bayes": GaussianNB()
}

# ---- Evaluation Function ----
def evaluate_model(model, X, y, technique, k=5):
    """
    Evaluate a model using the selected resampling technique.
    """
    try:
        if technique == "K-Fold Cross-Validation":
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
        else:
            cv = LeaveOneOut()

        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()
    except ValueError as e:
        st.error(f"❌ Error evaluating model: {e}")
        return None


# ---- Main UI ----
def run():
    st.title("📊 **Compare Machine Learning Models**")
    st.write("""
    Compare different machine learning models using cross-validation techniques and analyze their performance.
    """)

    # 📂 Dataset Upload Section
    st.write("### 📂 **Upload Dataset for Model Comparison**")
    uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'])
    
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            
            # 🚨 Handle Missing Values
            st.write("### 🧹 **Data Cleaning and Imputation**")
            df = impute_missing_values(df)
            
            # Debugging Info
            st.write("**Missing Value Check After Imputation:**")
            st.write(df.isna().sum())

            # Check if target column exists
            if 'Diabetes_012' not in df.columns:
                st.error("❌ The dataset must have a column named `Diabetes_012` for target values.")
                st.stop()

            X = df.drop(columns=['Diabetes_012'])
            y = df['Diabetes_012']

            st.write("### 🗂️ **Dataset Overview**")
            st.dataframe(df.head())

            # ---- Model Comparison Settings ----
            st.write("### ⚙️ **Comparison Settings**")
            resampling_technique = st.selectbox(
                "Select Resampling Technique:",
                ["K-Fold Cross-Validation", "Leave-One-Out Cross-Validation"]
            )
            if resampling_technique == "K-Fold Cross-Validation":
                k_folds = st.slider("Number of Folds", 2, 10, 5)

            selected_models = st.multiselect(
                "Select Models to Compare:",
                list(models.keys()),
                default=["Logistic Regression", "Random Forest", "Decision Tree"]
            )

            if not selected_models:
                st.warning("⚠️ Please select at least one model for comparison.")
                st.stop()

            # ---- Model Comparison ----
            st.write("### 📊 **Model Comparison Results**")
            results = []
            for name in selected_models:
                model = models[name]
                with st.spinner(f"Training {name} using {resampling_technique}..."):
                    accuracy = evaluate_model(
                        model,
                        X,
                        y,
                        technique=resampling_technique,
                        k=k_folds if resampling_technique == "K-Fold Cross-Validation" else None
                    )
                    if accuracy is not None:
                        results.append({"Model": name, "Accuracy": accuracy})

            # Display Results
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # ---- Visualization ----
                st.write("### 📈 **Model Comparison Visualization**")
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Accuracy',
                    text='Accuracy',
                    color='Model',
                    title='Model Accuracy Comparison'
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- Best Model Selection ----
                st.write("### 🏆 **Choose the Best Model for Further Use**")
                best_model_name = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
                selected_model = st.selectbox(
                    "Choose the Best Model:",
                    results_df['Model'],
                    index=results_df['Model'].tolist().index(best_model_name)
                )
                st.success(f"✅ You selected: **{selected_model}** as the best-performing model!")

                # Download Model Configuration
                st.write("### 📥 **Save Model Configuration**")
                if st.button("Download Selected Model Configuration"):
                    model_config = {
                        "Best Model": selected_model,
                        "Accuracy": results_df.loc[results_df['Model'] == selected_model, 'Accuracy'].values[0],
                        "Resampling Technique": resampling_technique
                    }
                    config_df = pd.DataFrame([model_config])
                    config_df.to_csv("best_model_configuration.csv", index=False)
                    with open("best_model_configuration.csv", "rb") as file:
                        st.download_button(
                            label="⬇️ Download Model Configuration",
                            data=file,
                            file_name="best_model_configuration.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"❌ Error processing dataset: {e}")
            st.stop()
    else:
        st.info("📂 Please upload a dataset to start the comparison.")


# ---- Run Standalone ----
if __name__ == '__main__':
    run()
