# This version is for SVM 

#########################
# Constants
# Set a threshold for classification 
THRESHOLD <- 0.1

# Proportion of data to use for training
TRAIN_SPLIT <- 0.8

# K fold cv
N_FOLDS <- 5

print("Starting SVM model training")

# Start the timer
start_time <- Sys.time()


#########################################
# Packages
# List of required packages
required_packages <- c("caret", "data.table", "pROC", "e1071", "kernlab","doParallel") # e1071 required for SVM
# Track installed and loaded packages
installed <- c()
loaded <- c()

# Function to check, install, and load packages
install_and_load <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg,repos = "https://cran.rstudio.com/",  dependencies = TRUE)
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



# ############
# Register parallel backend with `doParallel`
num_cores <- 5  # Match Slurm's allocated CPUs
cl <- makeCluster(num_cores)
registerDoParallel(cl)

##########################
# Directory Management for Models and Plots
# Create required directories for storing models and plots
required_dirs <- c("models", "plots") # add "models/svm" if want separate directory
for (dir_name in required_dirs) {
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE)
    cat(sprintf("Directory '%s' created.\n", dir_name))
  }
}

##################################
# Preprocessing the Dataset
# Load the preprocessed trauma dataset
trauma <- fread('train_trauma.csv', header = TRUE)

set.seed(123)


##############################
# Cross validation
# Set up caret cross-validation
cv_control <- trainControl(
  method = "cv",       # k-fold cross-validation
  number = N_FOLDS,    # 5-fold CV
  classProbs = TRUE,   # Enable probability estimation
  summaryFunction = twoClassSummary,  # Required for ROC metric
  savePredictions = "final",
  allowParallel = T
)

#############################
# Support Vector Machine (SVM)
# Train an SVM model using caret with tuning grid for cost parameter C

# Compute sigma only if the file does not exist
sigma_file <- "sigma_value.rds"
if (!file.exists(sigma_file)) {
  sigma_est <- sigest(transfusion ~ ., data = trauma, frac = 1)[1] 
  saveRDS(sigma_est, sigma_file)
} else {
  sigma_est <- readRDS(sigma_file)
}


set.seed(42)
svm_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "svmRadial",  # Radial kernel SVM
  tuneGrid = expand.grid(C = 2^(-4:4), sigma = sigma_est), # tuning
  trControl = cv_control,
  metric = "ROC"
)

# Save the trained model
saveRDS(svm_model, file = file.path("models", "svm_radial_caret.rds"))

###########
print('radial kernel svm done:)')

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time for radial SVM:", elapsed_time, "seconds\n")




###################
# Linear SVM
start_time <- Sys.time()

# Train an SVM model using a linear kernel
set.seed(42)
svm_linear_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "svmLinear",  # Linear kernel SVM
  tuneGrid = expand.grid(C = 2^(-4:4)),  # Tune regularization parameter
  trControl = cv_control,
  metric = "ROC"
)

# Save the trained model
saveRDS(svm_linear_model, file = file.path("models", "svm_linear_caret.rds"))

print('linear kernel svm done:)')

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time for linear SVM:", elapsed_time, "seconds\n")


# ✅ Clean up cluster
stopCluster(cl)
registerDoSEQ()  # Switch back to sequential mode
