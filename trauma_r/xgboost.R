library(xgboost)
library(caret)
library(data.table)


# NOTE: IF DATASET CHANGES, YOU MUST CHANGE THE RESPONSE VARIABLE SELECTED FOR XGB



#########################
# Constants
# Set a threshold for classification 
THRESHOLD = 0.1

# Proportion of data to use for training
TRAIN_SPLIT = 0.8



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
data <- fread('trauma_preprocessed_final.csv', header = TRUE)
trauma <- data[,-1]


# Split data into training and test sets based on `TRAIN_SPLIT`
trainIndex <- createDataPartition(trauma$transfusion, p = TRAIN_SPLIT, list = FALSE, times = 1)
trainData <- trauma[trainIndex, ]
testData <- trauma[-trainIndex, ]

train_x <- data.matrix(trainData[,-80])
train_y <- as.numeric(unlist(trainData[, 80]))

test_x <- data.matrix(testData[,-80])
test_y <- as.numeric(unlist(testData[, 80]))


#############################
# Model Evaluation Function
# Function to evaluate a model using confusion matrix and ROC curve
evaluate <- function(model, threshold, testData) {
  # Input: 
  # - `model`: Trained model to evaluate
  # - `threshold`: Threshold for classification
  # - `testData`: Test dataset for evaluation
  # Output: List containing confusion matrix and ROC curve
  
  pred_prob <- predict(model, newdata = testData, type = "response")  # Predicted probabilities
  pred_class <- ifelse(pred_prob > threshold, 1, 0)  # Binary predictions
  
  # Generate confusion matrix (assumes `transfusion` is binary with positive = "1")
  cm <- confusionMatrix(as.factor(pred_class), as.factor(testData$transfusion), positive = "1")
  
  # Generate ROC curve
  roc_curve <- roc(testData$transfusion, pred_prob)
  
  return(list(cm = cm, roc = roc_curve))
}


###################
# Training an XGBoost model

# making xgb matrix
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

# define watchlist
watchlist <- list(train=xgb_train, test=xgb_test)

# Fit XGBoost model and capture evaluation metrics
nrounds <- 70
model <- xgb.train(
  data = xgb_train,
  max.depth = 3,
  watchlist = watchlist,
  nrounds = nrounds,
  eval_metric = "rmse",
  verbose = 1
)

# Extract evaluation log for training and test RMSE
eval_log <- model$evaluation_log

# Find the best round based on the lowest test RMSE
best_round <- which.min(eval_log$test_rmse)
best_rmse <- eval_log$test_rmse[best_round]

cat("Best RMSE:", best_rmse, "at round:", best_round, "\n")

# Train final model using the best number of rounds
final_model <- xgboost(
  data = xgb_train,
  max.depth = 3,
  nrounds = best_round,
  verbose = 0
)

# Plot training and test RMSE over rounds
rmse_plot <- ggplot(data = eval_log, aes(x = iter)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE")) +
  geom_line(aes(y = test_rmse, color = "Test RMSE")) +
  labs(
    title = "Training and Test RMSE Over Rounds",
    x = "Rounds",
    y = "RMSE"
  ) +
  scale_color_manual(values = c("Train RMSE" = "blue", "Test RMSE" = "red")) +
  theme_minimal()

# Save the plot to a file
ggsave(filename = "training_test_rmse_plot.png", plot = rmse_plot, width = 8, height = 6)

# Save the final model if necessary
xgb.save(final_model, "final_xgb_model.model")


#####################
# check
print('finished :)')