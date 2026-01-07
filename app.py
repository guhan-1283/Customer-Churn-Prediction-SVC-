import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

model = joblib.load("model.pkl")


st.title("Customer Churn Prediction")

st.subheader("Using this web app, you can able to predict whether the customer stay or not")

st.divider()


gender = st.selectbox("Gender",["Male","Female"])
partner = st.selectbox("Partner",["Yes","No"])
dependents = st.selectbox("Dependents",["Yes","No"])
phone = st.selectbox("Phone Service",["Yes","No"])
multiple = st.selectbox("Multiple Lines",["Yes","No"])
internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
security = st.selectbox("Online Security",["Yes","No","No internet service"])
backup = st.selectbox("Online Backup",["Yes","No","No internet service"])
device = st.selectbox("Device Protection",["Yes","No","No internet service"])
tech = st.selectbox("Tech Support",["Yes","No","No internet service"])
tv = st.selectbox("Streaming TV",["Yes","No","No internet service"])
movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
paperless = st.selectbox("Paperless Billing",["Yes","No"])
payment = st.selectbox("Payment Method",
                       ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
                      )

senior = st.number_input("Senoir Citizen(0 = No,1 = Yes)",0,1)
tenure = st.number_input("Tenure (months)",0,80)
monthly = st.number_input("Monthly Charges",0.0,200.0)
total = st.number_input("Total Charges",0.0,10000.0)

st.divider()

input_df = pd.DataFrame([{
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])


predict = st.button("Press this Button to Predict")


if predict:
    prediction = model.predict(input_df)[0]
    
    with st.spinner("Please Wait..."):
        time.sleep(2.0)
        if prediction == "Yes":
                st.error(
        "😔  **Churn Predicted**\n"
        "Customer shows high churn risk."
               )

            

        else:
            st.success(
        "😊 **No Churn Predicted**\n"
        "Customer appears stable and unlikely to churn."
              )