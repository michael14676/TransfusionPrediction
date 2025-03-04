#########################
# Constants
THRESHOLD <- 0.1
EARLY_STOPPING <- 50
N_FOLDS <- 5  # 5-fold CV

# Get fold ID from SLURM array job
args <- commandArgs(trailingOnly=TRUE)
fold_id <- as.numeric(args[1])  # Fold number from SLURM array job

cat(sprintf("Starting XGBoost training for fold %d\n", fold_id))

# Start timer
start_time <- Sys.time()

#########################################
# Packages
required_packages <- c("caret", "data.table", "pROC", "xgboost") 

install_and_load <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com/", dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

sapply(required_packages, install_and_load)

##########################
# Directory Management
required_dirs <- c("models", "results")
for (dir_name in required_dirs) {
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE)
  }
}

############
# Register parallel backend
num_cores <- 2  # 2 cores per SLURM task


##################################
# Load Dataset
trauma <- fread('train_trauma.csv', header = TRUE)

# Convert response variable to binary numeric (0 = No, 1 = Yes)
trauma$transfusion <- as.numeric(trauma$transfusion == "Yes")


################################
# CV fold selecting

set.seed(42)  # For reproducibility

# Create 5-fold indices
folds <- caret::createFolds(trauma$transfusion, k = N_FOLDS, list = TRUE)

# Split into train and validation
valid_idx <- folds[[fold_id]]  # This fold is the validation set
train_idx <- setdiff(1:nrow(trauma), valid_idx)  # All other data is training

# Create training and validation matrices
train_matrix <- xgb.DMatrix(
  data = as.matrix(trauma[train_idx, !"transfusion", with = FALSE]), 
  label = trauma$transfusion[train_idx]
)

valid_matrix <- xgb.DMatrix(
  data = as.matrix(trauma[valid_idx, !"transfusion", with = FALSE]), 
  label = trauma$transfusion[valid_idx]
)


#######################
# Define hyperparameter grid
param_grid <- expand.grid(
  max_depth = c(3, 6, 9),       
  eta = c(0.01, 0.1),      
  gamma = c(0, 1),       
  colsample_bytree = c(0.7, 1),  
  min_child_weight = c(1, 3, 5),  
  subsample = c(0.8, 1.0)
)

###########################
# Cross Validation for Fold
cv_results_list <- list()

for (i in 1:nrow(param_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    # eval_metric = "auc",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i],
    gamma = param_grid$gamma[i],
    nthread = num_cores  # Use 2 cores per fold
  )
  
  cat(sprintf("[Fold %d] Testing: eta=%.3f, max_depth=%d, subsample=%.1f, colsample_bytree=%.1f, min_child_weight=%d, gamma=%.1f\n",
              fold_id, params$eta, params$max_depth, params$subsample, params$colsample_bytree, params$min_child_weight, params$gamma))
  
  # Watch validation set for early stopping
  watchlist <- list(train = train_matrix, eval = valid_matrix)
  
  # Train XGBoost model using manual cross-validation split
  xgb_model <- xgb.train(
    params = params,
    data = train_matrix,
    nrounds = 1500,  
    watchlist = watchlist,
    early_stopping_rounds = EARLY_STOPPING,
    eval_metric = "auc",
    maximize = TRUE,
    verbose = 1,
  )
  
  # Extract best iteration and AUC
  best_iteration <- xgb_model$best_iteration
  best_auc <- max(xgb_model$evaluation_log$eval_auc)
  
  # Save results
  results <- data.frame(
    fold = fold_id,
    eta = params$eta,
    max_depth = params$max_depth,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    min_child_weight = params$min_child_weight,
    gamma = params$gamma,
    best_iteration = best_iteration,
    auc = best_auc
  )
  cv_results_list[[i]] <- results
  
}

results_df <- do.call(rbind, cv_results_list)


# Save results for this fold
saveRDS(results_df, file = sprintf("results/xgb_results_fold_%d.rds", fold_id))

# Print best performing parameters for this fold
best_row <- results_df[which.max(results_df$auc), ]
print(best_row)



print(sprintf("Fold %d XGBoost model training completed!", fold_id))

# End timer
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Elapsed time for Fold %d: %.2f seconds\n", fold_id, elapsed_time))


