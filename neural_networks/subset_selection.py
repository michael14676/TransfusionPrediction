import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def remove_unnecessary(train: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unneeded columns and one-hot encodes 'label__TRANSPORTMODE' if present.
    Mimics the logic from the R function of the same name.
    """

    # 1. Clean and normalize column names
    #    - Trim whitespace and replace ".." with "__" to mimic read.csv fix
    train = train.copy()  # avoid mutating original
    train.columns = train.columns.str.strip()
    train.columns = train.columns.str.replace(r"\.\.", "__", regex=True)

    # 2. Store 'transfusion' temporarily, if it exists
    if 'transfusion' in train.columns:
        transfusion_col = train['transfusion'].copy()
        has_transfusion = True
    else:
        transfusion_col = None
        has_transfusion = False

    # 3. Explicit columns to drop
    drop_explicit = [
        'label__ETHNICITY',
        'label__PRIMARYMETHODPAYMENT',
        'label__HOMERESIDENCE_UK',
        'label__HOMERESIDENCE_NA',
        'onehot__TEACHINGSTATUS_1.0',
        'onehot__TEACHINGSTATUS_5.0',
        'onehot__TEACHINGSTATUS_6.0',
        'label__HOSPITALTYPE',
        'label__Bedsize',
        'label__VERIFICATIONLEVEL',
        'label__PEDIATRICVERIFICATIONLEVEL',
        'label__STATEDESIGNATION',
        'label__STATEPEDIATRICDESIGNATION'
    ]

    # 4. Drop columns ending with 'UK'/'NA' or starting with 'label__TM_'
    drop_suffixes = ["UK", "NA"]
    drop_prefixes = ["label__TM_"]

    # Identify columns to drop (by suffix or prefix)
    drop_by_pattern = []
    for col in train.columns:
        if any(col.endswith(sfx) for sfx in drop_suffixes) or \
           any(col.startswith(pfx) for pfx in drop_prefixes):
            drop_by_pattern.append(col)

    # Combine and keep only columns that actually exist in the DataFrame
    columns_to_drop = list(set(drop_explicit + drop_by_pattern))
    columns_to_drop = [c for c in columns_to_drop if c in train.columns]

    # 5. Print which columns will be dropped
    print(f"Dropping {len(columns_to_drop)} columns:")
    print(columns_to_drop)

    # 6. Drop unwanted columns
    train_reduced = train.drop(columns=columns_to_drop)

    # 7. One-hot encode 'label__TRANSPORTMODE' (if present)
    if 'label__TRANSPORTMODE' in train_reduced.columns:
        # Temporarily remove transfusion again (if still present)
        if 'transfusion' in train_reduced.columns:
            train_reduced_no_transfusion = train_reduced.drop(columns=['transfusion'])
        else:
            train_reduced_no_transfusion = train_reduced

        # Fit a OneHotEncoder on the transport mode column
        ohe = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        encoded_matrix = ohe.fit_transform(train_reduced_no_transfusion[['label__TRANSPORTMODE']])
        categories = ohe.categories_[0]
        encoded_cols = [f'onehot__TRANSPORTMODE_{cat}' for cat in categories]

        # Create new columns in a DataFrame
        ohe_df = pd.DataFrame(encoded_matrix, columns=encoded_cols,
                              index=train_reduced_no_transfusion.index)

        # Drop the original 'label__TRANSPORTMODE' and concatenate one-hot columns
        train_reduced_no_transfusion = train_reduced_no_transfusion.drop(columns=['label__TRANSPORTMODE'])
        train_cleaned = pd.concat([train_reduced_no_transfusion, ohe_df], axis=1)

        print("\nOne-hot encoded 'label__TRANSPORTMODE' into:")
        print(encoded_cols)
    else:
        train_cleaned = train_reduced
        print("\n'label__TRANSPORTMODE' not found; no encoding applied.")

    # 8. Append 'transfusion' back (if it existed)
    if has_transfusion:
        train_cleaned['transfusion'] = transfusion_col

    # Final shape info
    print(f"\nFinal dataset shape: {train_cleaned.shape[0]} rows x {train_cleaned.shape[1]} columns")
    return train_cleaned


def filter_prehospital(train: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame using 'remove_unnecessary', then retains only
    columns matching certain prehospital-related prefixes (plus 'transfusion').
    Mimics the logic from the R function of the same name.
    """

    # 1. Clean and one-hot encode first
    train_cleaned = remove_unnecessary(train)

    # 2. Prefixes (or exact names) to keep
    keep_prefixes = [
        'onehot__SEX_', 'onehot__ETHNICITY_',
        'label__ASIAN', 'label__BLACK', 'label__WHITE', 'label__RACEOTHER',
        'scaler__SBP', 'scaler__PULSERATE', 'scaler__TEMPERATURE',
        'scaler__RESPIRATORYRATE', 'scaler__PULSEOXIMETRY',
        'scaler__HEIGHT', 'scaler__WEIGHT', 'scaler__AgeYears',
        'scaler__TOTALGCS',
        'label__GCSQ_VALID', 'label__GCSQ_INTUBATED', 'label__GCSQ_SEDATEDPARALYZED',
        'label__TBIPUPILLARYRESPONSE',
        'label__SUPPLEMENTALOXYGEN', 'label__HIGHESTACTIVATION',
        'label__PREHOSPITALCARDIACARREST', 'label__RESPIRATORYASSISTANCE',
        'label__PROTDEV_', 'label__AIRBAG_',
        'onehot__TRANSPORTMODE_',

        # Additional variables
        'label__PMGCSQ_',
        'label__INTERFACILITYTRANSFER',
        'scaler__TBIHIGHESTTOTALGCS',
        'scaler__ISS',
        'label__WORKRELATED',

        # Target variable
        'transfusion'
    ]

    # 3. Define a function to check if a column name starts with any keep_prefix
    def keep_column(col: str) -> bool:
        return any(col.startswith(prefix) for prefix in keep_prefixes)

    # 4. Apply filtering
    columns_to_keep = [col for col in train_cleaned.columns if keep_column(col)]
    train_filtered = train_cleaned[columns_to_keep]

    # 5. Show final selection
    print(f"Selected {len(columns_to_keep)} columns.")
    return train_filtered
