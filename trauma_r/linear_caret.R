
# LR, ridge, and lasso

#########################
# Constants
# Set a threshold for classification 
THRESHOLD = 0.1

# Number of folds for cross validation
K_FOLDS = 5

# Start the timer
start_time <- Sys.time()


#########################################
# Packages
# List of required packages
required_packages <- c("caret", "data.table", "pROC", "glmnet")
# Track installed and loaded packages
installed <- c()
loaded <- c()

# Function to check, install, and load packages
install_and_load <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    installed <<- c(installed, pkg)  # Track installed packages
  }
  library(pkg, character.only = TRUE)
  loaded <<- c(loaded, pkg)  # Track loaded packages
}

# Process each package
sapply(required_packages, install_and_load)

# Print summary of actions taken
if (length(installed) > 0) {
  cat("Installed packages:", paste(installed, collapse = ", "), "\n")
} else {
  cat("No new packages were installed.\n")
}

cat("Loaded packages:", paste(loaded, collapse = ", "), "\n")



##########################
# Directory Management for Models and Plots
# Create required directories for storing models and plots
required_dirs <- c("models", "plots")
for (dir_name in required_dirs) {
  if (!dir.exists(dir_name)) {
    dir.create(dir_name)
    cat(sprintf("Directory '%s' created.\n", dir_name))
  }
}

##################################
# Preprocessing the Dataset
# Load the preprocessed trauma dataset
trauma <- fread('train_trauma.csv', header = TRUE)


###################
# Set Up Cross-Validation
cv_control <- trainControl(
  method = "cv",                # k-fold cross-validation
  number = K_FOLDS,             # Number of folds
  classProbs = TRUE,            # Required for ROC curves
  summaryFunction = twoClassSummary,  # Computes AUROC
  savePredictions = "final"     # Store predictions across folds
)


###################
# Training a Full Logistic Regression Model
set.seed(42)
full_lr_model <- train(
  transfusion ~ .,
  data = trauma,
  method = "glm",
  family = "binomial",
  trControl = cv_control,
  metric = "ROC"
)

saveRDS(full_lr_model, file = "models/full_lr_model_caret.rds")
print("Finished full logistic regression model!")

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time:", elapsed_time, "seconds\n")

#############################
# Ridge Regression

# Start the timer
start_time <- Sys.time()

# train the model
set.seed(42)
ridge_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "glmnet",
  trControl = cv_control,
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-5, 5, length = 120))  # Search across 100 lambda values
)

# Save the trained model
saveRDS(ridge_model, file = file.path("models", "ridge_model_caret.rds"))


print('finished ridge regression model!')

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time:", elapsed_time, "seconds\n")


#############################
# **Lasso Regression (alpha = 1) with caret**

start_time <- Sys.time()

set.seed(42)
lasso_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "glmnet",
  trControl = cv_control,
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-5, 5, length = 120))  # Search across 100 lambda values
)

# Save the trained model
saveRDS(lasso_model, file = file.path("models", "lasso_model_caret.rds"))

print('finished lasso regression model!')

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time:", elapsed_time, "seconds\n")

###########
print('done :)')
