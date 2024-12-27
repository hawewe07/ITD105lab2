import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ---- Paths ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, 'data', 'diabetes_012_health_indicators.pkl'))

# ---- Load Dataset ----
@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """
    Load dataset from uploaded file or cached location.
    """
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_pickle(DATA_PATH)  # Cache for faster reload
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


# ---- Streamlit App ----
def run():
    st.title("📊 **Diabetes Data Dashboard**")
    st.write("Explore the dataset with interactive charts, filters, and insights.")

    # 📂 Dataset Upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (.csv)", type=['csv'])
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("### 🗂️ **Dataset Overview**")
        st.dataframe(df.head())

        # ---- Sidebar Filters ----
        st.sidebar.write("### 🔍 **Filters**")
        
        # Diabetes Filter
        if 'Diabetes_012' in df.columns:
            diabetes_filter = st.sidebar.multiselect(
                "Filter by Diabetes Status:", 
                options=df['Diabetes_012'].unique(), 
                default=df['Diabetes_012'].unique()
            )
        else:
            st.error("❌ The dataset must contain a column named 'Diabetes_012'.")
            st.stop()
        
        # Age Range Filter
        if 'Age' in df.columns:
            age_range = st.sidebar.slider(
                "Select Age Range:", 
                int(df['Age'].min()), 
                int(df['Age'].max()), 
                (20, 60)
            )
        else:
            st.error("❌ The dataset must contain a column named 'Age'.")
            st.stop()
        
        # BMI Range Filter
        if 'BMI' in df.columns:
            bmi_range = st.sidebar.slider(
                "Select BMI Range:", 
                float(df['BMI'].min()), 
                float(df['BMI'].max()), 
                (18.5, 30.0)
            )
        else:
            st.error("❌ The dataset must contain a column named 'BMI'.")
            st.stop()
        
        # Apply Filters
        filtered_df = df[
            (df['Diabetes_012'].isin(diabetes_filter)) & 
            (df['Age'].between(age_range[0], age_range[1])) &
            (df['BMI'].between(bmi_range[0], bmi_range[1]))
        ]

        st.write("---")

        # ---- Overview Metrics ----
        st.write("## 📈 **Key Metrics Overview**")
        col1, col2, col3 = st.columns(3)
        col1.metric("🧑 Total Records", len(filtered_df))
        col2.metric("🎯 Diabetes Cases", filtered_df['Diabetes_012'].sum())
        col3.metric("📊 Average BMI", f"{filtered_df['BMI'].mean():.2f}")

        st.write("---")

        # ---- Gender Distribution ----
        if 'Sex' in filtered_df.columns:
            st.write("### 🧑‍🤝‍🧑 **Gender Distribution**")
            fig = px.histogram(
                filtered_df, 
                x='Sex', 
                color='Diabetes_012',
                title="Diabetes by Gender",
                barmode='group',
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        
        # ---- Age Distribution ----
        if 'Age' in filtered_df.columns:
            st.write("### 📊 **Age Distribution**")
            fig = px.histogram(
                filtered_df, 
                x='Age', 
                color='Diabetes_012',
                title="Diabetes by Age Group",
                marginal="box",
                nbins=20,
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        
        # ---- BMI Distribution ----
        if 'BMI' in filtered_df.columns:
            st.write("### 📉 **BMI Distribution**")
            fig = px.histogram(
                filtered_df, 
                x='BMI', 
                color='Diabetes_012',
                title="Diabetes by BMI",
                marginal="violin",
                nbins=30,
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        
        # ---- High Blood Pressure ----
        if 'HighBP' in filtered_df.columns:
            st.write("### 💓 **High Blood Pressure Status**")
            bp_count = filtered_df['HighBP'].value_counts()
            fig = px.pie(
                names=bp_count.index,
                values=bp_count.values,
                title="Proportion of High Blood Pressure",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        
        # ---- Smoking Status ----
        if 'Smoker' in filtered_df.columns:
            st.write("### 🚬 **Smoking Status**")
            smoker_count = filtered_df['Smoker'].value_counts()
            fig = px.pie(
                names=smoker_count.index,
                values=smoker_count.values,
                title="Proportion of Smokers",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")

        # ---- Correlation Heatmap ----
        st.write("### 🔗 **Correlation Heatmap**")
        numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = filtered_df[numeric_columns].corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Correlation Between Numeric Features",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")

        # ---- Insights ----
        st.write("### 🧠 **Insights and Observations**")
        st.markdown("""
        - **Age vs Diabetes:** Older age groups exhibit higher diabetes prevalence.
        - **BMI vs Diabetes:** Higher BMI is associated with increased diabetes risk.
        - **Gender Distribution:** Diabetes prevalence varies by gender.
        - **Blood Pressure:** High blood pressure correlates positively with diabetes risk.
        - **Smoking:** Smoking increases diabetes vulnerability.
        """)
        
        st.info("**Tip:** Use the sidebar filters to dynamically explore the dataset.")

        st.write("---")
        st.success("✅ Dashboard successfully loaded with visualizations and insights.")


# ---- Run Standalone ----
if __name__ == '__main__':
    run()
