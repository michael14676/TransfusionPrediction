#########################
# Constants
# Set a threshold for classification 
THRESHOLD <- 0.1

# Proportion of data to use for training
TRAIN_SPLIT <- 0.8

# K fold cv
N_FOLDS <- 5

print("Starting Random Forest model training")

# Start the timer
start_time <- Sys.time()

#########################################
# Packages
# List of required packages
required_packages <- c("caret", "data.table", "pROC", "randomForest","doParallel") 
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


##########################
# Directory Management for Models and Plots
# Create required directories for storing models and plots
required_dirs <- c("models", "plots")
for (dir_name in required_dirs) {
  if (!dir.exists(dir_name)) {
    dir.create(dir_name,  recursive = TRUE)
    cat(sprintf("Directory '%s' created.\n", dir_name))
  }
}

# ############
# Register parallel backend with `doParallel`
num_cores <- 5  # Match Slurm's allocated CPUs
cl <- makeCluster(num_cores)
registerDoParallel(cl)

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
# Random Forest Model
# Train a Random Forest model using caret with tuning grid for parameters
set.seed(42)
rf_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "rf",  # Random Forest model
  tuneGrid = expand.grid(mtry = c(5, 7, 9, 11, 15, 20, 30)), # Tune mtry
  trControl = cv_control,
  metric = "ROC",
  ntree = 500  # Set number of trees to 500,
)

# Save the trained model
saveRDS(rf_model, file = file.path("models", "random_forest_caret.rds"))

print('Random Forest model training done:)')

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time for Random Forest:", elapsed_time, "seconds\n")
