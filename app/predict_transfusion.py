import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt  
from xgboost import Booster
import xgboost as xgb

# -------------------------------
# 1. Load the saved pipeline
# -------------------------------
model = Booster() # if used xgb.train rather than XGBClassifier, use Booster
model.load_model("app/models/xgboost_app_model.json")


# Generate SHAP explainer on the fly
explainer = shap.TreeExplainer(model)



# Get feature names taken from original dataset
with open("app/models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# -------------------------------
# 2. Streamlit App Layout
# -------------------------------
import streamlit as st
import pandas as pd

st.title("Trauma Transfusion Prediction")
st.write("Enter patient information to estimate probability of transfusion.")

def user_input_features():
    st.subheader("ISS")
    st.caption("Injury Severity Score — estimate of trauma severity based on anatomical injuries.")
    st.markdown("[Click here for more on ISS](https://www.mdcalc.com/calc/1239/injury-severity-score-iss)")  # Replace with your actual link
    
    has_exact_iss = st.radio("Do you have the exact ISS?", ["Yes", "No"])

    if has_exact_iss == "Yes":
        ISS = st.number_input("Enter exact ISS value", step=1.0, min_value=0.0)
    else:
        is_severe = st.radio("Is the ISS estimated to be ≥ 16?", ["Yes", "No"])
        if is_severe == "Yes":
            ISS = 30.0  # Stand-in for severe injury
        else:
            ISS = 5.0   # Stand-in for non-severe injury

    st.subheader("Age")
    st.caption("Age of the patient in years")
    AgeYears = st.number_input("AgeYears", step=1.0, value= 50.0, min_value=18.0)

    st.subheader("Systolic Blood Pressure")
    st.caption("First measured systolic blood pressure on arrival (mmHg)")
    SBP = st.number_input("SBP", step=1.0, value=120.0, min_value=0.0)

    st.subheader("Pulse Rate")
    st.caption("Pulse rate in beats per minute")
    PULSERATE = st.number_input("PULSERATE", step=1.0, value=80.0, min_value=0.0)

    st.subheader("Temperature")
    st.caption("Patient's body temperature in Celsius")
    TEMPERATURE = st.number_input("TEMPERATURE", step=0.1, value=37.0)

    st.subheader("Respiratory Rate")
    st.caption("Number of breaths per minute")
    RESPIRATORYRATE = st.number_input("RESPIRATORYRATE", step=1.0, value=14.0, min_value=0.0)

    st.subheader("Pulse Oximetry")
    st.caption("Oxygen saturation percentage as measured by pulse oximetry")
    PULSEOXIMETRY = st.number_input("PULSEOXIMETRY", step=0.1, value=98.0, min_value=0.0, max_value=100.0)

    st.subheader("Highest Activation")
    st.caption("Was this patient a highest-level trauma activation?")
    activation_map = {"Yes": 1, "No": 2}
    HIGHESTACTIVATION = activation_map[st.selectbox("HIGHESTACTIVATION", list(activation_map.keys()), index=1)]

    st.subheader("Transport Mode")
    st.caption("How did the patient arrive?")
    transport_map = {"Ground Ambulance": 1, "Helicopter Ambulance": 2, "Fixed-wing Ambulance": 3, "Private/Public Vehicle/Walk-in": 4, "Police": 5, "Other": 6}
    TRANSPORTMODE = transport_map[st.selectbox("TRANSPORTMODE", list(transport_map.keys()))]

    data = {
        'ISS': ISS,
        'AgeYears': AgeYears,
        'SBP': SBP,
        'PULSERATE': PULSERATE,
        'TEMPERATURE': TEMPERATURE,
        'RESPIRATORYRATE': RESPIRATORYRATE,
        'PULSEOXIMETRY': PULSEOXIMETRY,
        'HIGHESTACTIVATION': HIGHESTACTIVATION,
        'TRANSPORTMODE': TRANSPORTMODE,
    }

    return pd.DataFrame([data])

user_df = user_input_features()


# -------------------------------
# 3. Prediction
# -------------------------------
if st.button("Predict Transfusion Probability"):
    # Predict probability for class 1 (transfusion)
    # prediction_proba = model.predict_proba(user_df)[0][1] # Assuming the model is an XGBClassifier with predict_proba method

    # Create DMatrix
    dmatrix = xgb.DMatrix(user_df)

    # Predict (this gives probabilities directly for binary tasks)
    prediction_proba = model.predict(dmatrix)[0]  # Already the class 1 probability

    st.subheader("Predicted Probability:")
    st.write(f"🔴 Probability of needing transfusion: **{prediction_proba:.2%}**")

    # SHAP Explanation
    st.subheader("Model Explanation (SHAP):")
    shap_values = explainer(user_df)
    shap_value_single = shap_values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(shap_value_single, max_display=10, show=False)
    st.pyplot(fig)

