import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

st.title("Netflix Churn Prediction App")

# -------- User Inputs -------- #

age = st.number_input("Age", 18, 80)
watch_hours = st.number_input("Watch Hours (per month)", 0.0)
last_login_days = st.number_input("Days Since Last Login", 0)
monthly_fee = st.number_input("Monthly Fee", 0.0)
number_of_profiles = st.number_input("Number of Profiles", 1)
avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day", 0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
region = st.selectbox("Region", ['Africa', 'Europe', 'Asia', 'Oceania', 'South America','North America'])
device = st.selectbox("Device", ["Mobile", "TV", "Laptop"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI"])
favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Horror"])

# -------- Prediction -------- #

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "age": [age],
        "watch_hours": [watch_hours],
        "last_login_days": [last_login_days],
        "monthly_fee": [monthly_fee],
        "number_of_profiles": [number_of_profiles],
        "avg_watch_time_per_day": [avg_watch_time_per_day],
        "gender": [gender],
        "subscription_type": [subscription_type],
        "region": [region],
        "device": [device],
        "payment_method": [payment_method],
        "favorite_genre": [favorite_genre]
    })

    transformed_data = transformer.transform(input_data)
    prediction = model.predict(transformed_data)
    probability = model.predict_proba(transformed_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Customer Likely to Churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer Not Likely to Churn (Probability: {probability:.2f})")