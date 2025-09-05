import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_xgb_model.pkl")

st.title("📊 IBM Customer Churn Prediction")

st.write("Enter customer details below to predict churn:")

# Collect inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Convert to dataframe (match preprocessing)
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior=="Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Handle categorical variables (must match training encoding)
input_df = pd.get_dummies(input_df)
# Add missing columns (in case some categories are missing in input)
for col in model.get_booster().feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder to match training features
input_df = input_df[model.get_booster().feature_names]

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if pred == 1:
        st.error(f"❌ Customer is likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {prob:.2f})")
