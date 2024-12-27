import streamlit as st
import os
import sys

# Add modules path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ---- Page Configuration ----
st.set_page_config(
    page_title="AI Task Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Session State Initialization ----
if 'task' not in st.session_state:
    st.session_state.task = None
if 'module' not in st.session_state:
    st.session_state.module = None

# ---- Task Selection ----
st.title("🤖 **Predictor Task Hub**")
st.write("""
Welcome to the Predictor Task Hub! Choose between Regression or Classification tasks to proceed.
""")

# Sidebar Task Selection
st.sidebar.title("🛠️ **Select Task**")
task = st.sidebar.radio(
    "Choose a Task:",
    ["Regression", "Classification"],
    index=0 if st.session_state.task == "Regression" else 1 if st.session_state.task == "Classification" else None
)

# Update Session State with Task
if task:
    st.session_state.task = task
    st.session_state.module = None  # Reset module when task changes

# ---- Dynamic Module Navigation ----
if st.session_state.task == "Regression":
    st.sidebar.title("📊 **Regression Modules**")
    module = st.sidebar.radio(
        "Navigate:",
        [
            "Train Model",
            "Evaluate Performance",
            "Hyperparameter Tuning",
            "Compare Models",
            "Predict",
            "Data Dashboard"
        ],
        index=None if st.session_state.module is None else [
            "Train Model",
            "Evaluate Performance",
            "Hyperparameter Tuning",
            "Compare Models",
            "Predict",
            "Data Dashboard"
        ].index(st.session_state.module)
    )
    st.session_state.module = module  # Update selected module in session state
    
    if module == "Train Model":
        from modules.regression.regression_train import run
    elif module == "Evaluate Performance":
        from modules.regression.regression_performance import run
    elif module == "Hyperparameter Tuning":
        from modules.regression.regression_hypertune import run
    elif module == "Compare Models":
        from modules.regression.regression_compare import run
    elif module == "Predict":
        from modules.regression.regression_predictor import run
    elif module == "Data Dashboard":
        from modules.regression.regression_dashboard import run
    
    if module:
        run()
    else:
        st.info("Please select a regression module to proceed.")

elif st.session_state.task == "Classification":
    st.sidebar.title("📊 **Classification Modules**")
    module = st.sidebar.radio(
        "Navigate:",
        [
            "Train Model",
            "Evaluate Performance",
            "Hyperparameter Tuning",
            "Compare Models",
            "Predict",
            "Data Dashboard"
        ],
        index=None if st.session_state.module is None else [
            "Train Model",
            "Evaluate Performance",
            "Hyperparameter Tuning",
            "Compare Models",
            "Predict",
            "Data Dashboard"
        ].index(st.session_state.module)
    )
    st.session_state.module = module  # Update selected module in session state
    
    if module == "Train Model":
        from modules.classification.train_model import run
    elif module == "Evaluate Performance":
        from modules.classification.performance import run
    elif module == "Hyperparameter Tuning":
        from modules.classification.hypertune import run
    elif module == "Compare Models":
        from modules.classification.compare import run
    elif module == "Predict":
        from modules.classification.predictor import run
    elif module == "Data Dashboard":
        from modules.classification.dashboard import run
    
    if module:
        run()
    else:
        st.info("Please select a classification module to proceed.")
else:
    st.info("Please select a task from the sidebar to continue.")

# ---- Footer ----
st.write("---")
st.info("Tip: Use the sidebar to navigate between modules and interact with your dataset efficiently.")
