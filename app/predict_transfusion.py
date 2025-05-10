import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt  

# -------------------------------
# 1. Load the saved pipeline
# -------------------------------
model = joblib.load("models/logistic_app_model.pkl")

# Load SHAP explainer (optional)
with open("app/results/logistic_app_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# Get feature names taken from original dataset
with open("app/models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# -------------------------------
# 2. Streamlit App Layout
# -------------------------------
st.title("Trauma Transfusion Prediction")
st.write("Enter patient information to estimate probability of transfusion.")

# Collect user inputs
def user_input_features():
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", step=1.0)
    return pd.DataFrame([input_data])

user_df = user_input_features()

# -------------------------------
# 3. Prediction
# -------------------------------
if st.button("Predict Transfusion Probability"):
    # Predict using full pipeline
    prediction_proba = model.predict_proba(user_df)[0][1]  # Probability of transfusion (class 1)

    st.subheader("Predicted Probability:")
    st.write(f"🔴 Probability of needing transfusion: **{prediction_proba:.2%}**")

    # Optional SHAP force plot
    st.subheader("Model Explanation (SHAP):")
    X_scaled_df = pd.DataFrame(
        model.named_steps["standardscaler"].transform(user_df),
        columns=user_df.columns
    )
    shap_values = explainer(X_scaled_df)

    # Prepare the SHAP value
    shap_value_single = shap_values[0]

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use SHAP's waterfall plot but direct it to our `ax` via current active axis
    plt.sca(ax)  # set current axis to ax
    shap.plots.waterfall(shap_value_single, max_display=10, show=False)

    # Show in Streamlit
    st.pyplot(fig)
