**Support Vector Machine — Customer Churn Prediction**

**Overview:**
- **Project:** Predict customer churn for a Telco dataset using a Support Vector Classifier (SVC).
- **Files:** [app.py](app.py), [Model.ipynb](Model.ipynb), [requirements.txt](requirements.txt), [WA_Fn-UseC_-Telco-Customer-Churn.csv](WA_Fn-UseC_-Telco-Customer-Churn.csv)

**Dataset & Column Definitions**
- **customerID:** Unique customer identifier
- **gender:** Customer gender (Male / Female)
- **SeniorCitizen:** 0 = No, 1 = Yes
- **Partner:** Whether the customer has a partner (Yes / No)
- **Dependents:** Whether the customer has dependents (Yes / No)
- **tenure:** Number of months the customer has been with the company
- **PhoneService:** Whether the customer has phone service (Yes / No)
- **MultipleLines:** Whether the customer has multiple lines (Yes / No / No phone service)
- **InternetService:** Type of internet service (DSL / Fiber optic / No)
- **OnlineSecurity:** (Yes / No / No internet service)
- **OnlineBackup:** (Yes / No / No internet service)
- **DeviceProtection:** (Yes / No / No internet service)
- **TechSupport:** (Yes / No / No internet service)
- **StreamingTV:** (Yes / No / No internet service)
- **StreamingMovies:** (Yes / No / No internet service)
- **Contract:** Contract term (Month-to-month / One year / Two year)
- **PaperlessBilling:** (Yes / No)
- **PaymentMethod:** Payment method (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic))
- **MonthlyCharges:** Monthly charge amount
- **TotalCharges:** Total charged amount
- **Churn:** Target variable (Yes = churned, No = stayed)

**Model & Training (summary)**
- **Model type:** Support Vector Classifier (`sklearn.svm.SVC`) used in a pipeline
- **Preprocessing:** ColumnTransformer / pipeline for categorical encoding and numeric scaling (see [Model.ipynb](Model.ipynb))
- **Hyperparameter tuning:** `GridSearchCV`
- **Train/test split:** 70% train / 30% test (stratified, `random_state=42`)
- **Class weighting:** `class_weight='balanced'` used to mitigate class imbalance
- **Reported accuracy:** 0.75 (75%) as shown in [Model.ipynb](Model.ipynb#L2040)

**How prediction works in the app**
- The Streamlit app ([app.py](app.py)) loads a saved model `grid_model.pkl` using `joblib.load("grid_model.pkl")`.
- The app builds a single-row `pandas.DataFrame` with the same feature columns as the training data and calls `model.predict(input_df)`.
- The app shows a simple message: `Churn Predicted` (if prediction == "Yes") or `No Churn Predicted` (if prediction == "No").

**Tech stack & dependencies**
- **Language:** Python 3.x
- **Libraries:** `numpy`, `pandas`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn` (see [requirements.txt](requirements.txt))

**Run locally (quick start)**
1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
# if streamlit or scikit-learn missing, also run explicitly:
pip install streamlit scikit-learn
```
3. Ensure the trained model `grid_model.pkl` exists in the project root (the app expects this filename).
4. Start the app:
```bash
streamlit run app.py
```

**Notes & Next steps**
- `requirements.txt` contains a small typo (`sckit-learn`); replace with `scikit-learn` for reliability.
- Consider storing the model in `models/` and adding a small input-validation layer to [app.py](app.py).
- Add evaluation metrics (precision/recall/F1/ROC AUC) and a confusion matrix to better understand model performance.

If you want, I can:
- run a quick fix to `requirements.txt` and add a `models/` folder, or
- extract the exact preprocessing steps and add them to the README.
