library(car)
library(tidyverse)
library(caret)
library(data.table)
library(MASS)
library(pROC)
library(glmnet)



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

print(dim(trauma))

# Split data into training and test sets based on `TRAIN_SPLIT`
trainIndex <- createDataPartition(trauma$transfusion, p = TRAIN_SPLIT, list = FALSE, times = 1)
trainData <- trauma[trainIndex, ]
testData <- trauma[-trainIndex, ]

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
# Training a Full Logistic Regression Model
full_lr_model <- glm(transfusion ~ ., data = trainData, family = 'binomial')
full_lr_results <- evaluate(full_lr_model, THRESHOLD, testData)

# Save the model and results
saveRDS(full_lr_model, file = file.path("models", "full_lr_model.rds"))
saveRDS(full_lr_results, file = file.path("models", "full_lr_results.rds"))

print('finished full lr model :)')

################
# Stepwise Logistic Regression Model
# Generate predictor names and build the full formula
predictors <- names(trainData)[!names(trainData) %in% "transfusion"]
full_formula <- as.formula(paste("transfusion ~", paste(predictors, collapse = " + ")))

# Fit a stepwise logistic regression model using AIC
initial_model <- glm(transfusion ~ 1, data = trainData, family = binomial())
stepwise_model <- stepAIC(initial_model, scope = list(lower = ~1, upper = full_formula), direction = "both", trace = 1)

stepwise_lr_results <- evaluate(stepwise_model, THRESHOLD, testData)

# Save the stepwise model and results
saveRDS(stepwise_model, file = file.path("models", "stepwise_model.rds"))
saveRDS(stepwise_lr_results, file = file.path("models", "stepwise_lr_results.rds"))

print('finished stepwise logistic regression model!')

#############################
# Ridge Regression
# Prepare data for ridge regression (glmnet requires a matrix format for predictors)
x <- model.matrix(transfusion ~ ., data = trainData)[, -1]  # Predictor matrix for training data
y <- trainData$transfusion  # Response variable

# Train ridge regression model (alpha = 0 for ridge regression)
ridge_model <- glmnet(x, y, family = "binomial", alpha = 0)

# Perform cross-validation to determine the best lambda
cv_ridge <- cv.glmnet(x, y, family = "binomial", alpha = 0)

# Extract the best lambda values from cross-validation
best_lambda <- cv_ridge$lambda.min
lambda_1se <- cv_ridge$lambda.1se

# Create test matrix (predictor variables for the test dataset)
test_x <- model.matrix(transfusion ~ ., data = testData)[, -1]

# Refit ridge models using best lambda (lambda.min) and 1se lambda (lambda.1se)
ridge_model_min <- glmnet(x, y, family = "binomial", alpha = 0, lambda = best_lambda)
ridge_model_1se <- glmnet(x, y, family = "binomial", alpha = 0, lambda = lambda_1se)

# Evaluate ridge regression models
ridge_results_min <- list(
  cm = confusionMatrix(as.factor(ifelse(predict(ridge_model_min, newx = test_x, type = "response") > THRESHOLD, 1, 0)),
                       as.factor(testData$transfusion), positive = "1"),
  roc = roc(testData$transfusion, as.numeric(predict(ridge_model_min, newx = test_x, type = "response")))
)

ridge_results_1se <- list(
  cm = confusionMatrix(as.factor(ifelse(predict(ridge_model_1se, newx = test_x, type = "response") > THRESHOLD, 1, 0)),
                       as.factor(testData$transfusion), positive = "1"),
  roc = roc(testData$transfusion, as.numeric(predict(ridge_model_1se, newx = test_x, type = "response")))
)

# Save the cross-validation plot
png(filename = file.path("plots", "ridge_plot.png"))
plot(cv_ridge)
dev.off()

# Save the models, cross-validation object, and evaluation results
saveRDS(ridge_model, file = file.path("models", "ridge_model.rds"))
saveRDS(cv_ridge, file = file.path("models", "cv_ridge.rds"))
saveRDS(ridge_model_min, file = file.path("models", "ridge_model_min.rds"))
saveRDS(ridge_model_1se, file = file.path("models", "ridge_model_1se.rds"))
saveRDS(ridge_results_min, file = file.path("models", "ridge_results_min.rds"))
saveRDS(ridge_results_1se, file = file.path("models", "ridge_results_1se.rds"))

print('finished ridge regression model!')

#############################
# Lasso Regression
# Prepare data for lasso regression (glmnet requires a matrix format for predictors)
x <- model.matrix(transfusion ~ ., data = trainData)[, -1]  # Predictor matrix for training data
y <- trainData$transfusion  # Response variable

# Train lasso regression model (alpha = 1 for lasso regression)
lasso_model <- glmnet(x, y, family = "binomial", alpha = 1)

# Perform cross-validation to determine the best lambda
cv_lasso <- cv.glmnet(x, y, family = "binomial", alpha = 1)

# Extract the best lambda values from cross-validation
best_lambda <- cv_lasso$lambda.min
lambda_1se <- cv_lasso$lambda.1se

# Create test matrix (predictor variables for the test dataset)
test_x <- model.matrix(transfusion ~ ., data = testData)[, -1]

# Refit lasso models using best lambda (lambda.min) and 1se lambda (lambda.1se)
lasso_model_min <- glmnet(x, y, family = "binomial", alpha = 1, lambda = best_lambda)
lasso_model_1se <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lambda_1se)

# Evaluate lasso regression models
lasso_results_min <- list(
  cm = confusionMatrix(as.factor(ifelse(predict(lasso_model_min, newx = test_x, type = "response") > THRESHOLD, 1, 0)),
                       as.factor(testData$transfusion), positive = "1"),
  roc = roc(testData$transfusion, as.numeric(predict(lasso_model_min, newx = test_x, type = "response")))
)

lasso_results_1se <- list(
  cm = confusionMatrix(as.factor(ifelse(predict(lasso_model_1se, newx = test_x, type = "response") > THRESHOLD, 1, 0)),
                       as.factor(testData$transfusion), positive = "1"),
  roc = roc(testData$transfusion, as.numeric(predict(lasso_model_1se, newx = test_x, type = "response")))
)

# Save the cross-validation plot
png(filename = file.path("plots", "lasso_plot.png"))
plot(cv_lasso)
dev.off()

# Save the models, cross-validation object, and evaluation results
saveRDS(lasso_model, file = file.path("models", "lasso_model.rds"))
saveRDS(cv_lasso, file = file.path("models", "cv_lasso.rds"))
saveRDS(lasso_model_min, file = file.path("models", "lasso_model_min.rds"))
saveRDS(lasso_model_1se, file = file.path("models", "lasso_model_1se.rds"))
saveRDS(lasso_results_min, file = file.path("models", "lasso_results_min.rds"))
saveRDS(lasso_results_1se, file = file.path("models", "lasso_results_1se.rds"))

print('finished lasso regression model!')

###########
print('done :)')
