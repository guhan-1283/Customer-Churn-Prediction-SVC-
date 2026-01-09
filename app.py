import streamlit as st
import pandas as pd
import numpy as np
import time

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.subheader("Predict whether a customer will churn using SVC")
st.divider()

# --------------------------------------------------
# Train model INSIDE app (Python 3.14 SAFE)
# --------------------------------------------------
@st.cache_resource
def train_model():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    # Read data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Drop CustomerID
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Remove duplicates
    df = df.drop_duplicates(ignore_index=True)

    # Target
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # Feature groups
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # Preprocessing
    preprocess = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("scale", StandardScaler(), num_cols),
        ]
    )

    # Model pipeline (same as your notebook)
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Fit model
    pipe.fit(X_train, y_train)

    return pipe


model = train_model()

# --------------------------------------------------
# USER INPUTS (MATCH YOUR UI)
# --------------------------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

senior = st.number_input("Senior Citizen (0 = No, 1 = Yes)", 0, 1)
tenure = st.number_input("Tenure (months)", 0, 80)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Press this Button to Predict"):

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
        "TotalCharges": total,
    }])

    with st.spinner("Please wait..."):
        time.sleep(2)
        prediction = model.predict(input_df)[0]

    if prediction == "Yes":
        st.error(
            "😔 **Churn Predicted**\n\nCustomer shows high churn risk."
        )
    else:
        st.success(
            "😊 **No Churn Predicted**\n\nCustomer appears stable."
        )
