import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# Load Model and Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("📊 Employee Attrition Prediction System")

st.markdown("""
This AI-powered HR Analytics system predicts whether an employee is at risk of leaving the company based on workplace factors and employee behavior.
""")

st.info(
    "This tool helps HR teams identify employees at risk of leaving the company."
)

# Sidebar Inputs
st.sidebar.header("Employee Details")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=60,
    value=30
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    value=15000
)

years_at_company = st.sidebar.number_input(
    "Years At Company",
    min_value=0,
    max_value=40,
    value=5
)

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,
    4,
    2
)

overtime = st.sidebar.selectbox(
    "OverTime",
    ["Yes", "No"]
)

# Convert Overtime
overtime = 1 if overtime == "Yes" else 0

# Prediction Button
if st.button("🔍 Predict Attrition Risk"):

    # Input Data
    input_data = np.array([[
        age,
        monthly_income,
        years_at_company,
        job_satisfaction,
        overtime
    ]])

    # Scaling
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    # Probability Display
    st.metric(
        label="Attrition Probability",
        value=f"{probability:.2%}"
    )

    # Risk Levels
    if probability > 0.7:
        st.error("🔴 High Attrition Risk")

    elif probability > 0.4:
        st.warning("🟠 Moderate Attrition Risk")

    else:
        st.success("🟢 Low Attrition Risk")

    # Final Prediction
    if prediction[0] == 1:
        st.error(
            f"Employee Likely To Leave ({probability:.2%})"
        )

    else:
        st.success(
            f"Employee Likely To Stay ({1 - probability:.2%})"
        )

    # HR Insights
    st.subheader("HR Insights")

    if overtime == 1:
        st.write(
            "⚠️ Employees working overtime may have higher attrition risk."
        )

    if job_satisfaction <= 2:
        st.write(
            "⚠️ Low job satisfaction may increase employee turnover."
        )

    if years_at_company > 10:
        st.write(
            "ℹ️ Long company experience generally improves retention."
        )

# About Section
with st.expander("About This Model"):

    st.write("""
    This system uses Logistic Regression to predict employee attrition based on:

    - Age
    - Monthly Income
    - Years At Company
    - Job Satisfaction
    - OverTime

    Technologies Used:
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    - NumPy
    """)

# Footer
st.markdown("---")

st.caption(
    "Built by Raghav using Machine Learning and Streamlit"
)