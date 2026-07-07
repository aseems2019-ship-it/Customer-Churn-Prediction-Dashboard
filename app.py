# ===========================================================
# CUSTOMER CHURN PREDICTION DASHBOARD
# M.Sc. Statistics Project
# Developed using Streamlit
# ===========================================================

# ------------------ IMPORT LIBRARIES ------------------

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# ------------------ PAGE CONFIGURATION ------------------

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)
# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    color: #1f77b4;
}

h2 {
    color: #2E86C1;
}

h3 {
    color: #2874A6;
}

div[data-testid="stMetric"] {
    background-color: #262730;
    border: 1px solid #3d3d3d;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------

model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ------------------ LOAD DATASET ------------------

df = pd.read_csv("Telco_Customer_Churn_Dataset.csv")
# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("📊 Dashboard Menu")

    st.markdown("---")

    st.header("📌 Project Information")

    st.write("**Project:** Customer Churn Prediction")

    st.write("**Domain:** Machine Learning")

    st.write("**Model:** Logistic Regression")

    st.write("**Dataset:** Telco Customer Churn")

    st.success("✅ Dashboard Status")

    st.write("✔ Model Loaded")

    st.write("✔ Dataset Loaded")

    st.write("✔ Ready for Prediction")

    st.markdown("---")

    st.header("🤖 Model Performance")

    st.metric(
        "Model Accuracy",
        "80.5%"
    )

    st.metric(
        "Features Used",
        len(feature_columns)
    )

with st.expander("ℹ️ About this Dashboard"):

    st.write("""
This dashboard predicts whether a telecom customer is likely to churn using a Logistic Regression Machine Learning model.

It includes:

- Business Overview
- Customer Insights
- Churn Prediction
- Risk Analysis
- Interactive Analytics
- Feature Importance
- Downloadable Reports
""")

st.title("📊 Telecom Customer Churn Dashboard")
with st.expander("ℹ️ About this Dashboard"):

    st.write("""
This dashboard predicts customer churn using a Logistic Regression model.

### Features

- Customer Churn Prediction
- Probability Score
- Risk Level Indicator
- Business Recommendations
- Feature Importance
- Interactive Analytics
- Prediction History
- Download Prediction Report

This project demonstrates customer churn prediction using a Logistic Regression model built with Python, Scikit-learn, and Streamlit.
""")

st.markdown("""
This interactive dashboard predicts whether a telecom customer is likely to churn
using a trained Logistic Regression model. It also provides business insights,
customer analytics, and downloadable prediction reports.
""")

st.divider()

# ------------------ KPI CARDS ------------------

st.markdown("---")
st.header("📈 Business Overview")
st.caption("Key performance indicators of the telecom customer dataset.")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)

churn_rate = (
    (df["Churn"] == "Yes").mean() * 100
)

average_tenure = df["tenure"].mean()

average_monthly = df["MonthlyCharges"].mean()

with col1:
    st.metric(
        "Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

with col3:
    st.metric(
        "Average Tenure",
        f"{average_tenure:.1f} Months"
    )

with col4:
    st.metric(
        "Avg Monthly Charges",
        f"${average_monthly:.2f}"
    )

st.divider()
st.markdown("---")
st.header("📈 Customer Insights")
st.caption("Quick summary of customer retention and churn.")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"👥 Total Customers: {len(df)}")

with col2:
    st.success(f"😊 Customers Staying: {(df['Churn']=='No').sum()}")

with col3:
    st.error(f"⚠️ Customers Churned: {(df['Churn']=='Yes').sum()}")

st.info(
    f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
)
st.divider()

# ------------------ DATASET PREVIEW ------------------

with st.expander("📋 View Dataset"):

    search = st.text_input(
        "🔍 Search Customer ID"
    )

    if search:

        result = df[
            df["customerID"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.dataframe(result)

    else:

        st.dataframe(df.head(10))
        csv_data = df.to_csv(index=False)

st.download_button(
    "📥 Download Dataset",
    csv_data,
    "Telco_Customer_Churn.csv",
    "text/csv"
)
st.header("📚 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

with col2:
    st.write(
        "Numeric Features:",
        len(df.select_dtypes(include="number").columns)
    )

    st.write(
        "Categorical Features:",
        len(df.select_dtypes(include="object").columns)
    ) 

st.divider()
# ==========================================================
# CUSTOMER INPUT FORM
# ==========================================================

with st.expander("🤖 Model Information"):

    st.write("**Algorithm:** Logistic Regression")

    st.write("**Accuracy:** 80.5%")

    st.write("**Training Features:**")

    st.write(feature_columns)
st.markdown("---")
st.header("📝 Customer Information")
st.caption("Enter customer details to predict churn.")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )
st.subheader("📞 Services")

col1, col2 = st.columns(2)

with col1:

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col2:

    protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
st.subheader("💳 Billing Information")

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

with col2:

    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)
if predict_button:

    # Create all model features with default value 0
    input_data = {col: 0 for col in feature_columns}

    # ----------------------------
    # Numerical Features
    # ----------------------------

    input_data["SeniorCitizen"] = senior
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    # ----------------------------
    # One-Hot Encoded Features
    # ----------------------------

    if gender == "Male":
        input_data["gender_Male"] = 1

    if partner == "Yes":
        input_data["Partner_Yes"] = 1

    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    if phone == "Yes":
        input_data["PhoneService_Yes"] = 1

    if multiple == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif multiple == "No phone service":
        input_data["MultipleLines_No phone service"] = 1

    if internet == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_data["InternetService_No"] = 1

    if security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    if backup == "Yes":
        input_data["OnlineBackup_Yes"] = 1
    elif backup == "No internet service":
        input_data["OnlineBackup_No internet service"] = 1

    if protection == "Yes":
        input_data["DeviceProtection_Yes"] = 1
    elif protection == "No internet service":
        input_data["DeviceProtection_No internet service"] = 1

    if support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif support == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    if tv == "Yes":
        input_data["StreamingTV_Yes"] = 1
    elif tv == "No internet service":
        input_data["StreamingTV_No internet service"] = 1

    if movies == "Yes":
        input_data["StreamingMovies_Yes"] = 1
    elif movies == "No internet service":
        input_data["StreamingMovies_No internet service"] = 1

    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if paperless == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    if payment == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1
        # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    # Prediction Probability
    probability = model.predict_proba(input_df)[0][1]
    # Store prediction history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Prediction": "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "Probability (%)": round(probability * 100, 2)
    })
    st.divider()

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error("🔴 High Risk: This customer is likely to churn.")

        st.write(
            "Immediate retention strategies are recommended."
        )
    else:
        st.success("🟢 Low Risk: This customer is likely to stay.")

        st.write(
            "Customer retention probability is high."
        )

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )
    st.progress(float(probability))

    st.caption(
        f"Probability of customer churn: {probability*100:.2f}%"
    )
    # Risk Level

    if probability < 0.30:

            st.success("🟢 Risk Level: LOW")

    elif probability < 0.70:

            st.warning("🟡 Risk Level: MEDIUM")

    else:

            st.error("🔴 Risk Level: HIGH")
    st.subheader("💡 Business Recommendation")

    if prediction == 1:

        st.warning("""
### Recommended Action

- Offer a loyalty discount.
- Contact the customer proactively.
- Recommend a long-term contract.
- Improve customer support experience.
- Provide personalized retention offers.
""")

    else:

        st.success("""
### Customer Status

- Customer is likely to remain.
- Continue delivering quality service.
- Consider offering loyalty rewards.
""")
    # ==========================================================
    # DOWNLOAD PREDICTION REPORT
    # ==========================================================

    result_df = pd.DataFrame({
        "Prediction": [
            "Likely to Churn" if prediction == 1 else "Likely to Stay"
        ],
        "Churn Probability (%)": [
            round(probability * 100, 2)
        ],
        "Risk Level": [
            "High" if probability >= 0.70
            else "Medium" if probability >= 0.30
            else "Low"
        ]
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="customer_churn_prediction.csv",
        mime="text/csv"
    )
st.divider()

st.markdown("---")
st.header("📊 Interactive Analytics Dashboard")
st.caption("Explore customer behaviour through visual analytics.")

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Contract Distribution")

    contract_counts = df["Contract"].value_counts().reset_index()
    contract_counts.columns = ["Contract", "Customers"]

    fig = px.bar(
        contract_counts,
        x="Contract",
        y="Customers",
        color="Contract",
        title="Contract Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    st.subheader("🌐 Internet Service Distribution")

    internet_counts = df["InternetService"].value_counts().reset_index()
    internet_counts.columns = ["Internet Service", "Customers"]

    fig = px.pie(
        internet_counts,
        names="Internet Service",
        values="Customers",
        title="Internet Service Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
col1, col2 = st.columns(2)

with col1:

    st.subheader("💰 Monthly Charges")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(
        df["MonthlyCharges"],
        bins=20,
        color="#4C72B0",
        edgecolor="black"
    )

    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

with col2:

    st.subheader("🔥 Correlation Heatmap")

    temp = df.copy()

    temp["Churn"] = temp["Churn"].map({
        "No":0,
        "Yes":1
    })

    temp["TotalCharges"] = pd.to_numeric(
        temp["TotalCharges"],
        errors="coerce"
    )

    corr = temp[
        [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

st.divider()

st.markdown("---")
st.header("📋 Dataset Summary")
st.caption("Statistical summary of the dataset.")

tab1, tab2 = st.tabs(["Statistics", "Missing Values"])

with tab1:
    st.dataframe(df.describe())

with tab2:
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing)
st.divider()

st.header("📜 Prediction History")

if "history" in st.session_state:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

else:

    st.info("No predictions made yet.")
    st.divider()

st.markdown("---")
st.header("⭐ Feature Importance")
st.caption("Top features influencing customer churn predictions.")

try:

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": np.abs(model.coef_[0])
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8,8))

    ax.barh(
        importance["Feature"][:10],
        importance["Importance"][:10],
        color="#55A868"
    )

    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Important Features")

    plt.gca().invert_yaxis()

    st.pyplot(fig)
    st.subheader("🔥 Top 5 Factors Influencing Churn")

    st.table(
        importance.head(5)
    )
except:

    st.info("Feature importance is available only for linear models.")
st.divider()

st.markdown("""
---
### 📌 About this Project

**Customer Churn Prediction Dashboard**

This dashboard demonstrates how Machine Learning can be used to predict customer churn in the telecommunications industry.

**Technologies Used**
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn

© 2026 All Rights Reserved.
""")