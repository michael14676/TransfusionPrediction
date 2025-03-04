# Transfusion Prediction

This project involves predicting whether a patient will need a transfusion. The goal is to create an model that will be used as decision support for a trauma surgeon upon patient presentation to the hospital.

## Dataset

The dataset used is the **Trauma Quality Improvement Program (TQIP)**, specifically focusing on patients aged 18 and older from the years **2019 to 2022**. This subset contains over **1.2 million patients** and **231 features** per patient.

## Methodology

1. **Preprocessing**  
   - Feature engineering is performed to remove variables not suited for this prediction task and to create a response variable including transfusions.  
   - Features with high proportions of missing values are eliminated, while others are imputed as needed.  
   - Categorical variables are encoded using one-hot, ordinal, or label encoding.  
   - Continuous features are centered and scaled.  
   - Features with high correlation or low variance may be removed to improve model performance.  

2. **Data Splitting**  
   - The dataset is divided into an **80-20 train-test split**.  
   - For hyperparameter tuning, **5-fold cross-validation** is used on the training set.  

3. **Model Evaluation**  
   - The optimal parameters for each model are determined based on cross-validation performance.  
   - The final models are compared using **Area Under the Receiver Operating Characteristic Curve (AUROC)** as the primary evaluation metric.  


## Models Used

The following machine learning and deep learning models are implemented:

- Logistic Regression
- Lasso Regression
- Ridge Regression
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost
- Multilayer Perceptron (MLP)
- TabTransformer


## **Directory Structure**
- **`neural networks/`** – Files relating to deep learning models.
- **`preprocessing/`** – Contains the notebook used to do the preprocessing for the dataset.
- **`trauma_r/`** – R scripts for creating and saving each of the models, as well as evaluations.Rmd which compares the resulting models.
