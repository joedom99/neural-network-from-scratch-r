# =======================================================
# Neural Network from Scratch in R
# =======================================================
# Created by: Joe Domaleski
# Marketing Data Science example

# Load libraries - only one library is needed, we're building
# the neural network functions by hand
library(ggplot2)  # Used for plotting

# NOTE: If you want to create a new dataset, uncomment this section
# -------------------------------------------------------
# STEP 0: Generate dataset
# -------------------------------------------------------
# This dataset simulates early-stage ad campaign evaluations.
# x1 = budget intensity (scaled 0–1)
# x2 = audience engagement (scaled 0–1)
# y = campaign success, labeled 1 if x1 + x2 > 1, else 0
# This models a simple business rule: high budget + strong engagement = likely success.

#set.seed(42)
#n <- 500
#x1 <- runif(n)
#x2 <- runif(n)
#y <- ifelse(x1 + x2 > 1, 1, 0)
#df <- data.frame(x1 = x1, x2 = x2, y = y)

# Save for future use
# write.csv(df, "simple_nn_data.csv", row.names = FALSE)


# -------------------------------------------------------
# STEP 1: Load dataset
# -------------------------------------------------------
# If you want to create a new dataset, comment the line out below
# and run Step 0 above to create a new dataset. This section
# assumes you have the dataset in a CSV in the local working directory

df <- read.csv("simple_nn_data.csv")


# -------------------------------------------------------
# STEP 2: Split into Train/Test
# -------------------------------------------------------

# Randomly split 70% of the data for training, 30% for testing
set.seed(123)
train_idx <- sample(1:nrow(df), 0.7 * nrow(df))
train_df <- df[train_idx, ]
test_df <- df[-train_idx, ]

# Convert to matrix form for calculations
X_train <- as.matrix(train_df[, c("x1", "x2")])
y_train <- as.matrix(train_df$y)

X_test <- as.matrix(test_df[, c("x1", "x2")])
y_test <- as.matrix(test_df$y)

# -------------------------------------------------------
# STEP 3: Initialize Network
# -------------------------------------------------------

# Number of input features (x1 and x2)
input_nodes <- 2

# Number of neurons in the hidden layer
hidden_nodes <- 4

# Single output node for binary classification
output_nodes <- 1

# Initialize weights (W) and biases (b) for the network
# W1 connects the input layer to the hidden layer (2 x 4)
# b1 is the bias for each hidden node (1 x 4)
# W2 connects the hidden layer to the output layer (4 x 1)
# b2 is the bias for the single output node (1 x 1)
# Weights are initialized with small random values, biases start at zero
set.seed(42)
W1 <- matrix(rnorm(input_nodes * hidden_nodes, mean = 0, sd = 0.1), nrow = input_nodes)
b1 <- matrix(0, nrow = 1, ncol = hidden_nodes)
W2 <- matrix(rnorm(hidden_nodes * output_nodes, mean = 0, sd = 0.1), nrow = hidden_nodes)
b2 <- matrix(0, nrow = 1, ncol = output_nodes)

# -------------------------------------------------------
# STEP 4: Define activation function
# -------------------------------------------------------

# Sigmoid function squashes inputs to the range (0, 1)
sigmoid <- function(x) 1 / (1 + exp(-x))

# Derivative of sigmoid for backpropagation
sigmoid_derivative <- function(x) x * (1 - x)

# -------------------------------------------------------
# STEP 5: Train the network
# -------------------------------------------------------

# Controls how much we adjust weights during each update
learning_rate <- 0.1

# Number of times we loop through the entire training set
epochs <- 1000

# Stores the loss at each epoch for plotting
loss_history <- numeric(epochs)

for (epoch in 1:epochs) {
  # Forward pass: compute activations for each layer
  z1 <- X_train %*% W1 + matrix(rep(b1, nrow(X_train)), nrow = nrow(X_train), byrow = TRUE)
  a1 <- sigmoid(z1)
  z2 <- a1 %*% W2 + matrix(rep(b2, nrow(X_train)), nrow = nrow(X_train), byrow = TRUE)
  a2 <- sigmoid(z2)  # final output
  
  # Compute binary cross-entropy loss to measure prediction error
  # Epsilon is a tiny number to prevent errors from log(0)
  # A lower loss means the network is making better predictions
  epsilon <- 1e-7
  loss <- -mean(y_train * log(a2 + epsilon) + (1 - y_train) * log(1 - a2 + epsilon))
  loss_history[epoch] <- loss
  
  # Backpropagation: compute gradients
  # Gradients show how much each weight contributed to the error
  # We use them to adjust the weights in the right direction
  delta_output <- (y_train - a2) * sigmoid_derivative(a2)
  delta_hidden <- (delta_output %*% t(W2)) * sigmoid_derivative(a1)
  
  # Update weights and biases
  W2 <- W2 + t(a1) %*% delta_output * learning_rate
  b2 <- b2 + colSums(delta_output) * learning_rate
  W1 <- W1 + t(X_train) %*% delta_hidden * learning_rate
  b1 <- b1 + colSums(delta_hidden) * learning_rate
  
  # Print loss every 100 epochs
  if (epoch %% 100 == 0) {
    cat("Epoch:", epoch, "Loss:", round(loss, 4), "\n")
  }
}

# -------------------------------------------------------
# STEP 6: Plot the training loss
# -------------------------------------------------------

# Plot how the loss decreased over time during training
plot(loss_history, type = "l", col = "blue", lwd = 2,
     xlab = "Epoch", ylab = "Loss", main = "Training Loss")

# -------------------------------------------------------
# STEP 7: Evaluate on Training Data
# -------------------------------------------------------

# Run forward pass on training data to get predictions
z1_train <- X_train %*% W1 + matrix(rep(b1, nrow(X_train)), nrow = nrow(X_train), byrow = TRUE)
a1_train <- sigmoid(z1_train)
z2_train <- a1_train %*% W2 + matrix(rep(b2, nrow(X_train)), nrow = nrow(X_train), byrow = TRUE)
a2_train <- sigmoid(z2_train)

# Convert probabilities to class predictions
train_preds <- ifelse(a2_train > 0.5, 1, 0)
train_accuracy <- sum(train_preds == y_train) / length(y_train)
cat("Training Accuracy:", round(train_accuracy, 3), "\n")

# -------------------------------------------------------
# STEP 8: Evaluate on Test Data
# -------------------------------------------------------

# Forward pass on test data
z1_test <- X_test %*% W1 + matrix(rep(b1, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
a1_test <- sigmoid(z1_test)
z2_test <- a1_test %*% W2 + matrix(rep(b2, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
a2_test <- sigmoid(z2_test)

# Calculate and display test accuracy
test_preds <- ifelse(a2_test > 0.5, 1, 0)
test_accuracy <- sum(test_preds == y_test) / length(y_test)
cat("Test Accuracy:", round(test_accuracy, 3), "\n")

# Display confusion matrix
conf_matrix <- table(Predicted = test_preds, Actual = y_test)
print(conf_matrix)

# -------------------------------------------------------
# STEP 9: Visualize Decision Boundary
# -------------------------------------------------------

# Create a grid of input values over x1/x2 space
grid <- expand.grid(
  x1 = seq(0, 1, length.out = 100),
  x2 = seq(0, 1, length.out = 100)
)

# Forward pass for each grid point
grid_X <- as.matrix(grid)
z1_grid <- grid_X %*% W1 + matrix(rep(b1, nrow(grid_X)), nrow = nrow(grid_X), byrow = TRUE)
a1_grid <- sigmoid(z1_grid)
z2_grid <- a1_grid %*% W2 + matrix(rep(b2, nrow(grid_X)), nrow = nrow(grid_X), byrow = TRUE)
a2_grid <- sigmoid(z2_grid)

# Store predicted class and probability
grid$prob <- as.vector(a2_grid)
grid$pred <- ifelse(grid$prob > 0.5, 1, 0)

# Plot the decision surface and overlay test points
ggplot() +
  geom_tile(data = grid, aes(x = x1, y = x2, fill = factor(pred)), alpha = 0.3) +
  geom_point(data = test_df, aes(x = x1, y = x2, color = factor(y)), shape = 21, size = 2) +
  scale_fill_manual(values = c("lightblue", "pink"), name = "Prediction") +
  scale_color_manual(values = c("blue", "red"), name = "Actual") +
  labs(title = "Decision Boundary (Test Data)", x = "x1", y = "x2") +
  theme_minimal()

# End of script – the neural network has been trained, evaluated, and visualized.