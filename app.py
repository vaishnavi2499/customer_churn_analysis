import streamlit as st
import pandas as pd
import joblib

# --- Load saved model and scaler ---
xgb_model = joblib.load("models/xgb_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn probability:")

# --- Input fields ---
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

# Example categorical inputs (adjust based on features)
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# --- Preprocess input ---
input_df = pd.DataFrame({
    'tenure':[tenure],
    'MonthlyCharges':[monthly_charges],
    'TotalCharges':[total_charges],
    'avg_charges_per_month':[total_charges / max(tenure,1)],
    'gender_Female':[1 if gender=="Female" else 0],
    'partner_Yes':[1 if partner=="Yes" else 0],
    'dependents_Yes':[1 if dependents=="Yes" else 0],
    'contract_One year':[1 if contract=="One year" else 0],
    'contract_Two year':[1 if contract=="Two year" else 0],
    'payment_method_Credit card':[1 if payment_method=="Credit card" else 0],
    'payment_method_Electronic check':[1 if payment_method=="Electronic check" else 0],
    'payment_method_Mailed check':[1 if payment_method=="Mailed check" else 0]
})

# scale numeric columns
num_cols = ['tenure','MonthlyCharges','TotalCharges','avg_charges_per_month']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Predict ---
if st.button("Predict Churn"):
    prob = xgb_model.predict_proba(input_df)[:,1][0]
    prediction = "Yes" if prob > 0.5 else "No"
    st.write(f"Churn Probability: {prob:.2f}")
    st.write(f"Predicted Churn: {prediction}")
