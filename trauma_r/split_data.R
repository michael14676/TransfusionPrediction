# Load necessary libraries
library(caret)
library(data.table)

# Set seed for reproducibility
set.seed(42)

# Load the dataset
data <- fread('trauma_preprocessed_final.csv', header = TRUE)
trauma <- data[,-1]

# Ensure 'transfusion' is a factor
trauma$transfusion <- factor(trauma$transfusion, levels = c(0, 1), labels = c("No", "Yes"))


# Create an 80-20 split
train_index <- createDataPartition(trauma$transfusion, p = 0.8, list = FALSE)
train_data <- trauma[train_index, ]
test_data <- trauma[-train_index, ]

# Save the split datasets
write.csv(train_data, "train_trauma.csv", row.names = FALSE)
write.csv(test_data, "test_trauma.csv", row.names = FALSE)

# Print confirmation
cat("Train and test datasets have been saved as 'train_trauma.csv' and 'test_trauma.csv'.\n")
