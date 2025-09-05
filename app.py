import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load model (cached once)
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_xgb_model.pkl")

model = load_model()

# ---------------------------
# Preprocess function (cached)
# ---------------------------
@st.cache_data
def preprocess_input(input_df, model):
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df)
    # Reindex to ensure all model features are present
    input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)
    return input_df

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 IBM Customer Churn Prediction")

st.write("Enter customer details below to predict churn:")

# Example input fields (add more features as needed)
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

# Create input DataFrame
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    with st.spinner("⏳ Making prediction..."):
        processed = preprocess_input(input_df, model)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

    st.subheader("🔮 Prediction Result")
    if pred == 1:
        st.error(f"❌ Customer is likely to **CHURN** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {prob:.2f})")
