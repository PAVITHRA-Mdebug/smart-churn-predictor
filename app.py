import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Title
# ----------------------------
st.title("📊 Smart Customer Churn Predictor")
st.write("Predict whether a customer will churn based on input details.")

# ----------------------------
# Sample Dataset
# ----------------------------
data = [
    ["Male",0,"Yes","No",1,29.85,"Month-to-month","DSL","No"],
    ["Female",0,"No","No",34,56.95,"One year","DSL","No"],
    ["Male",0,"No","No",2,53.85,"Month-to-month","DSL","Yes"],
    ["Male",0,"No","No",45,42.30,"One year","DSL","No"],
    ["Female",0,"No","No",2,70.70,"Month-to-month","Fiber optic","Yes"],
    ["Female",1,"Yes","Yes",60,90.00,"Two year","Fiber optic","No"],
    ["Male",1,"No","Yes",5,80.00,"Month-to-month","Fiber optic","Yes"],
    ["Female",0,"Yes","No",20,65.00,"One year","DSL","No"]
]

columns = ["gender","SeniorCitizen","Partner","Dependents","tenure",
           "MonthlyCharges","Contract","InternetService","Churn"]

df = pd.DataFrame(data, columns=columns)

# Convert target
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Encode features
X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ----------------------------
# USER INPUT UI
# ----------------------------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", ["Yes","No"])
Dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.slider("Monthly Charges", 10, 120, 50)
Contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# ----------------------------
# Convert input to DataFrame
# ----------------------------
input_df = pd.DataFrame([[gender, SeniorCitizen, Partner, Dependents, tenure,
                          MonthlyCharges, Contract, InternetService]],
                        columns=["gender","SeniorCitizen","Partner","Dependents",
                                 "tenure","MonthlyCharges","Contract","InternetService"])

# Apply same encoding
input_df = pd.get_dummies(input_df)

# ✅ IMPORTANT FIX (match training columns)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]*100

    if prediction == 1:
        st.error(f"⚠ Customer likely to churn (Risk: {prob:.2f}%)")
    else:
        st.success(f"✅ Customer likely to stay (Risk: {prob:.2f}%)")
