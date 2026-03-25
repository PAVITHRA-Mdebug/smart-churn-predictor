import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Title
# ----------------------------
st.title("📊 Smart Customer Churn Predictor (Advanced)")
st.write("AI-powered prediction with insights & visualization")

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

# Encode target
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Features
X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------------------
# USER INPUT UI
# ----------------------------
st.subheader("🧾 Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", ["Yes","No"])
Dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.slider("Monthly Charges", 10, 120, 50)
Contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# ----------------------------
# Convert input
# ----------------------------
input_df = pd.DataFrame([[gender, SeniorCitizen, Partner, Dependents, tenure,
                          MonthlyCharges, Contract, InternetService]],
                        columns=["gender","SeniorCitizen","Partner","Dependents",
                                 "tenure","MonthlyCharges","Contract","InternetService"])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    # Result
    if prediction == 1:
        st.error(f"⚠ Customer likely to churn (Probability: {prob:.2f}%)")
    else:
        st.success(f"✅ Customer likely to stay (Churn Risk: {prob:.2f}%)")

    # ----------------------------
    # 📊 Probability Chart
    # ----------------------------
    st.subheader("📊 Churn Probability Visualization")

    fig, ax = plt.subplots()
    labels = ["Stay", "Churn"]
    values = [100 - prob, prob]

    ax.bar(labels, values)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)

    # ----------------------------
    # 📈 Feature Importance
    # ----------------------------
    st.subheader("📈 Feature Importance")

    importance = model.feature_importances_
    feature_names = X.columns

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(feat_df["Feature"][:10], feat_df["Importance"][:10])
    ax2.invert_yaxis()
    ax2.set_title("Top Influencing Features")

    st.pyplot(fig2)

    # ----------------------------
    # 🎯 Explanation
    # ----------------------------
    st.subheader("🎯 Model Insight")

    if prob > 70:
        st.write("🔴 High churn risk. Immediate action recommended.")
    elif prob > 40:
        st.write("🟠 Moderate churn risk. Monitor customer behavior.")
    else:
        st.write("🟢 Low churn risk. Customer is stable.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
