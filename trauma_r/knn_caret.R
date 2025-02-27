
# This version is for knn 

#########################
# Constants
# Set a threshold for classification 
THRESHOLD = 0.1

# Proportion of data to use for training
TRAIN_SPLIT = 0.8

# K fold cv
N_FOLDS = 5

print("Starting KNN model training")

# Start the timer
start_time <- Sys.time()

#########################################
# Packages
# List of required packages
required_packages <- c("caret", "data.table", "pROC","doParallel")
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
required_dirs <- c("models", "plots") # add  "models/knn" if want separate directory
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


##############################
# Cross validation

# Set up caret cross-validation
cv_control <- trainControl(
  method = "cv",       # k-fold cross-validation
  number = N_FOLDS,          # 5-fold CV
  classProbs = TRUE,   # Enable probability estimation
  summaryFunction = twoClassSummary,  # Required for ROC metric
  savePredictions = "final",
  allowParallel = T
)

#############################
# K-Nearest Neighbors (KNN)
# Train a KNN model using caret with tuning grid for k=1 to 20
set.seed(42)
knn_model <- train(
  transfusion ~ ., 
  data = trauma, 
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 20, by = 2)),
  trControl = cv_control,
  metric = "ROC"
)

# Save the trained model
saveRDS(knn_model, file = file.path("models", "knn1-20_caret.rds"))

###########
print('done with knn:)')

# End the timer
end_time <- Sys.time()

# Calculate and print the elapsed time in seconds
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("Elapsed time for KNN:", elapsed_time, "seconds\n")

# Clean up cluster
on.exit({
  stopCluster(cl)
  registerDoSEQ()
})
