import streamlit as st
import joblib
import pandas as pd

# Load trained artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")  # saved during training

# Define numeric columns
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_charges_per_month"]

# --- Collect user input (example UI, keep yours if you already have) ---
st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)
avg_charges = total_charges / tenure if tenure > 0 else 0

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

# --- Convert inputs to dataframe ---
input_data = {
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "avg_charges_per_month": [avg_charges],
    "Contract_" + contract: [1],
    "Partner_Yes": [1 if partner == "Yes" else 0],
    "Dependents_Yes": [1 if dependents == "Yes" else 0]
}

input_df = pd.DataFrame(input_data)

# --- Align with training features ---
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --- Scale numeric features ---
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("Churn" if prediction == 1 else "No Churn")
