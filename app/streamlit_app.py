
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("model/model.pkl")

# Streamlit page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("🔍 Bank Customer Churn Prediction")
st.write("Enter customer details below to predict the likelihood of churn.")

# Define input fields matching the model features
state = st.selectbox("State", options=[
    "OH", "NJ", "CA", "TX", "MI", "NY", "IL", "PA", "GA", "FL"
])
account_length = st.slider("Account Length (months)", min_value=0, max_value=250, value=100)
area_code = st.selectbox("Area Code", options=["408", "415", "510"])
international_plan = st.selectbox("International Plan", options=["yes", "no"])
voice_mail_plan = st.selectbox("Voice Mail Plan", options=["yes", "no"])
number_vmail_messages = st.slider("Number of Voice Mail Messages", 0, 50, 10)
total_day_minutes = st.slider("Total Day Minutes", 0, 350, 150)
total_day_calls = st.slider("Total Day Calls", 0, 200, 100)
total_eve_minutes = st.slider("Total Evening Minutes", 0, 400, 200)
total_eve_calls = st.slider("Total Evening Calls", 0, 200, 100)
total_night_minutes = st.slider("Total Night Minutes", 0, 400, 200)
total_night_calls = st.slider("Total Night Calls", 0, 200, 100)
total_intl_minutes = st.slider("Total Intl Minutes", 0, 25, 10)
total_intl_calls = st.slider("Total Intl Calls", 0, 20, 5)
customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)

# Prepare input dataframe
input_df = pd.DataFrame([{
    "state": state,
    "account length": account_length,
    "area code": area_code,
    "international plan": international_plan,
    "voice mail plan": voice_mail_plan,
    "number vmail messages": number_vmail_messages,
    "total day minutes": total_day_minutes,
    "total day calls": total_day_calls,
    "total eve minutes": total_eve_minutes,
    "total eve calls": total_eve_calls,
    "total night minutes": total_night_minutes,
    "total night calls": total_night_calls,
    "total intl minutes": total_intl_minutes,
    "total intl calls": total_intl_calls,
    "customer service calls": customer_service_calls
}])

# Encode categorical values just like training
for col in ["state", "area code", "international plan", "voice mail plan"]:
    input_df[col] = input_df[col].astype("category").cat.codes

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(input_df)

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to stay. Probability: {1 - probability:.2%}")
