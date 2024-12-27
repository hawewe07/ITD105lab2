import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- Paths ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
DATA_PATH = os.path.join(DATA_DIR, 'regression_dataset.pkl')

# Ensure Directories Exist
os.makedirs(DATA_DIR, exist_ok=True)


# ---- Load Dataset ----
@st.cache_data(ttl=3600)
def load_data():
    """
    Load preprocessed regression dataset.
    """
    try:
        if os.path.exists(DATA_PATH):
            return pd.read_pickle(DATA_PATH)
        else:
            st.warning("⚠️ No dataset found. Please upload a dataset in the Train section.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()


# ---- Visualizations ----
def plot_temperature_trends(df):
    """
    Plot temperature trends over time.
    """
    try:
        if 'dt' in df.columns and 'AverageTemperature' in df.columns:
            st.write("### 📈 **Temperature Trends Over Time**")
            df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
            df = df.sort_values('dt')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df, x='dt', y='AverageTemperature', ax=ax, marker='o')
            plt.title("Global Average Temperature Over Time")
            plt.xlabel("Date")
            plt.ylabel("Average Temperature (°C)")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Columns 'dt' or 'AverageTemperature' not found in the dataset.")
    except Exception as e:
        st.error(f"❌ Error plotting temperature trends: {e}")


def plot_temperature_uncertainty(df):
    """
    Plot temperature uncertainty distribution.
    """
    try:
        if 'AverageTemperatureUncertainty' in df.columns:
            st.write("### 📊 **Temperature Uncertainty Distribution**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['AverageTemperatureUncertainty'], bins=30, kde=True, color='skyblue', ax=ax)
            plt.title("Distribution of Temperature Uncertainty")
            plt.xlabel("Temperature Uncertainty (°C)")
            plt.ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Column 'AverageTemperatureUncertainty' not found in the dataset.")
    except Exception as e:
        st.error(f"❌ Error plotting temperature uncertainty: {e}")


def plot_feature_correlation(df):
    """
    Plot feature correlation heatmap.
    """
    try:
        st.write("### 🔗 **Feature Correlation Heatmap**")
        numerical_df = df.select_dtypes(include=['number'])
        if numerical_df.empty:
            st.warning("⚠️ No numerical features found for correlation analysis.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        plt.title("Correlation Between Numerical Features")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error plotting feature correlation: {e}")


def plot_top_features(df):
    """
    Plot top numerical features affecting the target variable.
    """
    try:
        if 'AverageTemperature' in df.columns:
            st.write("### 🌟 **Top Correlated Features with Average Temperature**")
            numerical_df = df.select_dtypes(include=['number'])
            correlation = numerical_df.corr()['AverageTemperature'].sort_values(ascending=False)
            top_features = correlation.head(6).drop('AverageTemperature')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            top_features.plot(kind='barh', color='teal', ax=ax)
            plt.title("Top Features Correlated with Average Temperature")
            plt.xlabel("Correlation Coefficient")
            plt.ylabel("Features")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Column 'AverageTemperature' not found in the dataset.")
    except Exception as e:
        st.error(f"❌ Error plotting top correlated features: {e}")


# ---- Main Function ----
def run():
    """
    UI for Regression Data Dashboard.
    """
    st.title("📊 **Regression Data Dashboard**")
    st.write("""
    Explore and visualize insights from your regression dataset.
    """)
    
    # Load Dataset
    df = load_data()
    
    if df is not None:
        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())
        
        # Display Dataset Statistics
        with st.expander("📊 **Dataset Statistics**"):
            st.write(df.describe())
        
        # Visualization Options
        st.write("### 📈 **Visualizations**")
        visualization = st.selectbox(
            "Choose a Visualization:",
            [
                "Temperature Trends Over Time",
                "Temperature Uncertainty Distribution",
                "Feature Correlation Heatmap",
                "Top Correlated Features with Target"
            ]
        )
        
        if visualization == "Temperature Trends Over Time":
            plot_temperature_trends(df)
        elif visualization == "Temperature Uncertainty Distribution":
            plot_temperature_uncertainty(df)
        elif visualization == "Feature Correlation Heatmap":
            plot_feature_correlation(df)
        elif visualization == "Top Correlated Features with Target":
            plot_top_features(df)
        
        # Insights Section
        st.write("### 🧠 **Insights**")
        st.write("""
        - Analyze temperature trends globally.
        - Examine uncertainty in temperature measurements across regions.
        - Understand feature importance and their correlation with target variables.
        """)


# ---- Footer ----
st.write("---")
st.info("Tip: Use the dropdown to explore different visualizations for deeper insights.")
