import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import shap
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import pickle

# Start timer
start_time = time.time()

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("app/trauma_incomplete_app.csv")

X = df.drop(columns=["transfusion"])
y = df["transfusion"]

# -----------------------------
# 2. Apply SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# -----------------------------
# 3. Train logistic regression
# -----------------------------
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
)
pipeline.fit(X_resampled, y_resampled)

# Save the model
os.makedirs("app/models", exist_ok=True)
joblib.dump(pipeline, "app/models/logistic_app_model.pkl")

# -----------------------------
# 4. SHAP summary plot
# -----------------------------
# Create 'plots' folder if it doesn't exist
os.makedirs("app/plots", exist_ok=True)

# SHAP only works with raw model, so extract it
model = pipeline.named_steps["logisticregression"]
scaler = pipeline.named_steps["standardscaler"]
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)


# KernelExplainer for linear models
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Save the SHAP values
with open("app/results/logistic_app_shap.pkl", "wb") as f:
    pickle.dump(shap_values, f)

with open("app/results/logistic_app_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# Plot SHAP summary and save
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("Logistic Regression SHAP Summary Plot")
plt.savefig("app/plots/logistic_important_shap_summary.png", bbox_inches="tight", dpi=300)
plt.close()

print("Training complete. SHAP summary saved as 'plots/logistic_important_shap_summary.png'.")

# End timer
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
