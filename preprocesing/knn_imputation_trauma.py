import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import time

# Start timer
start_time = time.time()


# Load the CSV for dataset
trauma_transfusions = pd.read_csv("trauma_app_training.csv")

# ------------------------------------------
# 1. Separate predictors and target
# ------------------------------------------
target_col = "transfusion"
X = trauma_transfusions.drop(columns=[target_col]).copy()
y = trauma_transfusions[target_col]

# ------------------------------------------
# 2. Identify column types
# ------------------------------------------
num_cols = [
    "ISS", "AgeYears", "SBP", "PULSERATE",
    "TEMPERATURE", "RESPIRATORYRATE", "PULSEOXIMETRY"
]
cat_cols = ["HIGHESTACTIVATION", "TRANSPORTMODE"]

# ------------------------------------------
# 3. Ordinal-encode categoricals (so KNNImputer can work)
#    • unknown_value = -1 → keeps NaNs distinct while fitting
# ------------------------------------------
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# ------------------------------------------
# 4. Run KNN imputation (k = 5, distance-weighted)
# ------------------------------------------
imputer = KNNImputer(n_neighbors=5, weights="distance")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# ------------------------------------------
# 5. Cast categorical columns back to integers → original labels
# ------------------------------------------
X_imputed[cat_cols] = (
    X_imputed[cat_cols].round().astype(int)
)
X_imputed[cat_cols] = encoder.inverse_transform(
    X_imputed[cat_cols]
)

# ------------------------------------------
# 6. Reassemble the full dataframe
# ------------------------------------------
trauma_imputed = pd.concat([X_imputed, y], axis=1)

# optional sanity check
print(trauma_imputed.isna().sum())
print(trauma_imputed.head())


# --- Save the full imputed dataset ---
trauma_imputed.to_csv("trauma_most_important.csv", index=False)

print("Imputation complete. Data saved to 'trauma_most_important.csv'.")

# End timer
end_time = time.time()

# Print elapsed time
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
