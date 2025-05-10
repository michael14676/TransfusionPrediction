# this is a python file which will be training the xgboost model used in the webapp form.
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
import time
import pickle

# Start timer
start_time = time.time()

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("trauma_most_important.csv")

X = df.drop(columns=["transfusion"])
y = df["transfusion"]

# -----------------------------
# 3. Apply SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# -----------------------------
# 4. Train XGBoost on full data
# -----------------------------
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

model = xgb.XGBClassifier(**xgb_params)
model.fit(X, y)

# Save the full model
model.save_model("models/xgboost_app_model.json")


# -----------------------------
# 5. SHAP summary plot
# -----------------------------
# --- Create 'plots' folder if it doesn't exist ---
os.makedirs("plots", exist_ok=True)

# --- Create TreeExplainer ---
explainer = shap.TreeExplainer(model)

# --- Compute SHAP values ---
shap_values = explainer.shap_values(X)

# --- Save SHAP values to a pickle file ---
with open("results/xgboost_app_shap.pkl", "wb") as f:
    pickle.dump(shap_values, f)


# --- Plot SHAP summary and save ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("XGBoost SHAP Summary Plot")
plt.savefig("plots/xgboost_important_shap_summary.png", bbox_inches="tight", dpi=300)
plt.close()

print("Training complete. SHAP summary saved as 'plots/xgboost_important_shap_summary.png'.")

# End timer
end_time = time.time()

# Print elapsed time
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
