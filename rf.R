rm(list=ls())
library(caret) # Machine Learning Library
library(randomForest) # Random Forest library
library(DMwR2)
library(smotefamily)
set.seed(123)
data = read.csv("C:/Users/sheno/OneDrive/Desktop/Final Project/Dataset/Full_dataset.csv")
str(data)
summary(data)

data <- na.omit(data)

#Train/set 80/20
set.seed(123)

# Define the test set size (20% of data)
size = floor(0.2 * nrow(data))

# Randomly sample indices for the test set
test_ind = sample(seq_len(nrow(data)), size = size)

# Split the dataset
test = data[test_ind, ]   # Test set
train = data[-test_ind, ]  # Training set

#train
k_result <- kmeans(train$Yield, centers = 2)

# Step 2: Label training set
train$Yield_cat <- factor(k_result$cluster, labels = c("Low", "High"))

# Step 3: Get the cluster centroids
centroids <- as.numeric(k_result$centers)

# Step 4: Define a function to assign test values to closest centroid
assign_cluster <- function(x, centers) {
  distances <- abs(x - centers)
  return(which.min(distances))
}

# Step 5: Apply to test set
test$Yield_cluster <- sapply(test$Yield, assign_cluster, centers = centroids)

# Step 6: Convert cluster numbers to labels
test$Yield_cat <- factor(test$Yield_cluster, labels = c("Low", "High"))

train=train[-14]
test=test[-14]
test=test[-20]

##for train set
num_vars <- sapply(train, is.numeric)  # Logical vector: TRUE for numeric, FALSE for categorical

scaled_num <- scale(train[, num_vars])
num_vars[20]=TRUE

# Convert categorical predictors to dummy variables
library(caret)
dummy_vars <- dummyVars(~ ., data = train[, !num_vars, drop = FALSE], fullRank = TRUE)  # Exclude response
encoded_cats <- predict(dummy_vars, newdata = train)

train_preprocessed <- data.frame(scaled_num, encoded_cats, train[20], stringsAsFactors = F)  # Keep response unchanged

train_scaled <- train_preprocessed

#for test
num_vars1 <- sapply(test, is.numeric)  # Logical vector: TRUE for numeric, FALSE for categorical

scaled_num1 <- scale(test[, num_vars1])
num_vars1[20]=TRUE

# Convert categorical predictors to dummy variables
encoded_cats1 <- as.data.frame(predict(dummy_vars, newdata = test))
encoded_cats1 <- encoded_cats1[, colnames(encoded_cats)]  # Match column order/names

test_preprocessed <- data.frame(scaled_num1, encoded_cats1, test[20], stringsAsFactors = F)  # Keep response unchanged

test_scaled <- test_preprocessed

xtrain = train_scaled[,1:36]
ytrain = train_scaled[,37]
xtest = test_scaled[,1:36]
ytest = test_scaled[,37]

table(ytrain)

# Create a data frame combining features and target for Random Forest
train_rf <- cbind(xtrain, Yield_cat = ytrain)
test_rf <- cbind(xtest, Yield_cat = ytest)

# Define the hyperparameter grid for Random Forest
rf_grid <- expand.grid(
  mtry = 9  # Number of variables randomly sampled at each split
)

# Set up cross-validation
train_control <- trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Train the Random Forest model with tuning
rf_tune <- train(
  x = xtrain,
  y = ytrain,
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  importance = TRUE,
  ntree = 500
)

# Print the tuning results
print(rf_tune)
print(rf_tune$bestTune)

# Build final model with best parameters
final_rf <- randomForest(
  x = xtrain,
  y = ytrain,
  mtry = rf_tune$bestTune$mtry,
  ntree = 500,
  importance = TRUE
)

# Make predictions
rf_pred <- predict(final_rf, xtest)

# Convert to factors for confusion matrix
rf_pred <- as.factor(rf_pred)
ytest <- as.factor(ytest)

# Ensure levels match
levels(rf_pred) <- levels(ytest)

# Compute confusion matrices
cm_low <- confusionMatrix(rf_pred, ytest, positive = "Low")
cm_high <- confusionMatrix(rf_pred, ytest, positive = "High")

# Print results
print(cm_low)
print(cm_high)
print(cm_low$byClass)
print(cm_high$byClass)

# Print variable importance
print(importance(final_rf))
varImpPlot(final_rf, main = "Variable Importance Plot")


# Get variable importance
importance_values <- importance(final_rf)
print(importance_values)

# Basic varImpPlot from randomForest package
varImpPlot(final_rf, main = "Variable Importance Plot")

# Create enhanced variable importance plot with ggplot2
# Convert importance to a data frame for easier plotting
if("MeanDecreaseAccuracy" %in% colnames(importance_values)) {
  # For classification
  imp_df <- data.frame(
    Variable = rownames(importance_values),
    Importance = importance_values[, "MeanDecreaseAccuracy"],
    stringsAsFactors = FALSE
  )
} else {
  # For regression (use MeanDecreaseGini or %IncMSE)
  imp_df <- data.frame(
    Variable = rownames(importance_values),
    Importance = importance_values[, "%IncMSE"],
    stringsAsFactors = FALSE
  )
}

# Sort by importance
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]

# Convert Variable to factor to maintain order in plot
imp_df$Variable <- factor(imp_df$Variable, levels = imp_df$Variable)

# Take top 20 variables for better visualization
imp_df_top20 <- head(imp_df, 20)

# Create the enhanced bar plot
ggplot(imp_df_top20, aes(x = Variable, y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Horizontal bars for better readability
  scale_fill_gradient(low = "lightblue", high = "darkblue") +  # Color gradient
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "right"
  ) +
  labs(
    title = "Top 20 Variables by Importance",
    subtitle = "Random Forest Model",
    x = "",
    y = "Importance (Mean Decrease in Accuracy)"
  )

# Save the plot
ggsave("variable_importance_plot.png", width = 10, height = 8, dpi = 300)

# Alternative visualization - create a lollipop chart
ggplot(imp_df_top20, aes(x = Variable, y = Importance)) +
  geom_segment(aes(x = Variable, xend = Variable, y = 0, yend = Importance), 
               color = "gray50") +
  geom_point(aes(size = Importance, color = Importance)) +
  scale_color_gradient(low = "lightblue", high = "darkblue") +
  coord_flip() +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 16, face = "bold")
  ) +
  labs(
    title = "Variable Importance - Random Forest",
    subtitle = "Top 20 Variables",
    x = "",
    y = "Importance Score"
  )

# Save the alternative plot
ggsave("variable_importance_lollipop.png", width = 10, height = 8, dpi = 300)

imp_values <- importance(final_rf)
print(imp_values)


# Dynamically determine column names to avoid errors
imp_colnames <- colnames(imp_values)
print("Available importance columns:")
print(imp_colnames)

# Create a data frame for variable importance using the correct column names
imp_df <- data.frame(
  Variable = rownames(imp_values),
  stringsAsFactors = FALSE
)

# Add the appropriate importance measure based on what's available
if("MeanDecreaseAccuracy" %in% imp_colnames) {
  imp_df$Importance <- imp_values[, "MeanDecreaseAccuracy"]
  importance_label <- "Mean Decrease in Accuracy"
} else if("%IncMSE" %in% imp_colnames) {
  imp_df$Importance <- imp_values[, "%IncMSE"]
  importance_label <- "% Increase in MSE"
} else if("IncNodePurity" %in% imp_colnames) {
  imp_df$Importance <- imp_values[, "IncNodePurity"]
  importance_label <- "Increase in Node Purity"
} else {
  # If none of the expected columns exists, use the first column
  imp_df$Importance <- imp_values[, 1]
  importance_label <- colnames(imp_values)[1]
}

# Sort by importance
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE),]

# Print the top 10 most important variables
cat("Top 10 Most Important Variables:\n")
print(head(imp_df, 10))

# Create aggregated importance plot
# First, identify variables with very low importance to group together
# Define a threshold for "important" variables - e.g., top 20
n_important <- 20
important_vars <- head(imp_df$Variable, n_important)
other_vars <- setdiff(imp_df$Variable, important_vars)

# Create a new data frame with aggregated "Other" category
imp_agg <- imp_df[imp_df$Variable %in% important_vars,]
if(length(other_vars) > 0) {
  other_mean <- mean(imp_df$Importance[imp_df$Variable %in% other_vars])
  other_row <- data.frame(
    Variable = "Other Variables (Aggregated)",
    Importance = other_mean,
    stringsAsFactors = FALSE
  )
  imp_agg <- rbind(imp_agg, other_row)
}

# Reorder factors for plotting with most important at the top
imp_agg$Variable <- factor(imp_agg$Variable, 
                           levels = rev(c(imp_agg$Variable[imp_agg$Variable != "Other Variables (Aggregated)"], 
                                          "Other Variables (Aggregated)")))

# Create enhanced bar plot with most important variables at top
importance_plot <- ggplot(imp_agg, aes(x = Variable, y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Horizontal bars with most important at top
  scale_fill_gradient(low = "#69b3a2", high = "#330066") +  # Color gradient
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = "right"
  ) +
  labs(
    title = "Variable Importance in Random Forest Model",
    subtitle = paste("Top", n_important, "Variables + Aggregated Rest"),
    x = "",
    y = paste("Variable Importance Score (", importance_label, ")", sep = "")
  )

# Print and save the plot
print(importance_plot)
ggsave("aggregated_importance_plot.png", plot = importance_plot, width = 12, height = 8, dpi = 300)

# Create a cleaner version showing only top 10 variables
top10_plot <- ggplot(head(imp_df, 10), 
                     aes(x = fct_reorder(Variable, Importance), 
                         y = Importance, 
                         fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#69b3a2", high = "#330066") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Top 10 Most Important Variables",
    subtitle = "Random Forest Model",
    x = "",
    y = paste("Importance Score (", importance_label, ")", sep = "")
  )

# Print and save the top 10 plot
#print(top10_plot)
#ggsave("top10_importance_plot.png", plot = top10_plot, width = 10, height = 6, dpi = 300)

# Optional: Create a more visually appealing lollipop chart for top 15 variables
lollipop_plot <- ggplot(head(imp_df, 15), 
                        aes(x = fct_reorder(Variable, Importance), 
                            y = Importance)) +
  geom_segment(aes(x = Variable, xend = Variable, y = 0, yend = Importance),
               color = "gray60") +
  geom_point(aes(size = Importance, color = Importance), alpha = 0.8) +
  scale_color_gradient(low = "#69b3a2", high = "#330066") +
  scale_size_continuous(range = c(3, 10)) +
  coord_flip() +
  theme_minimal() +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 11, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Top 15 Variables by Importance",
    subtitle = "Random Forest Model - Lollipop Visualization",
    x = "",
    y = paste("Importance Score (", importance_label, ")", sep = "")
  )

# Print and save the lollipop plot
#print(lollipop_plot)
#ggsave("lollipop_importance_plot.png", plot = lollipop_plot, width = 10, height = 7, dpi = 300)



# ------------------------------------------------------------------------
# ADD PARTIAL DEPENDENCE PLOTS (PDPs)
# ------------------------------------------------------------------------

# Get the top features for PDP plots
top_features <- head(imp_df$Variable, 6)  # Get top 6 most important variables

# Create a data frame for model training with original column names
# This helps with interpretable PDPs
train_pdp <- as.data.frame(xtrain)
library(pdp)
# Create an RF model object for use with pdp package
pdp_rf_model <- randomForest(
  x = train_pdp,
  y = ytrain,
  mtry = rf_tune$bestTune$mtry,
  ntree = 500,
  importance = TRUE
)

# Create individual PDP plots for the top features
pdp_plots <- list()
for (feature in top_features) {
  # Create partial dependence for this feature
  pd <- partial(
    pdp_rf_model, 
    pred.var = feature, 
    grid.resolution = 50,
    train = train_pdp,
    prob = TRUE,  # For classification, get probabilities
    plot = FALSE  # Don't plot yet
  )
  
  # Create a more attractive ggplot
  p <- ggplot(pd, aes(x = pd[[1]], y = yhat)) +
    geom_line(size = 1.2, color = "#69b3a2") +
    geom_rug(sides = "b", alpha = 0.3, color = "#330066") +
    theme_minimal() +
    labs(
      title = paste("Partial Dependence Plot for", feature),
      x = feature,
      y = "Predicted probability of High Yield"
    ) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12)
    )
  
  pdp_plots[[feature]] <- p
  
  # Save individual plots
  ggsave(
    filename = paste0("pdp_", feature, ".png"),
    plot = p,
    width = 8,
    height = 6,
    dpi = 300
  )
}

# Arrange and display the top PDPs in a grid
top_4_pdps <- pdp_plots[1:min(4, length(pdp_plots))]
grid_plot <- gridExtra::grid.arrange(
  grobs = top_4_pdps,
  ncol = 2
)

# Save the grid plot
ggsave(
  "top_4_pdp_plots.png",
  plot = grid_plot,
  width = 12,
  height = 10,
  dpi = 300
)

# Create 2D interaction plots for the top 2 features
if (length(top_features) >= 2) {
  top_2_features <- top_features[1:2]
  
  # Create 2D partial dependence
  pd_interaction <- partial(
    pdp_rf_model,
    pred.var = top_2_features,
    grid.resolution = 30,
    train = train_pdp,
    plot = FALSE
  )
  
  # Plot as a heatmap
  interaction_plot <- ggplot(pd_interaction, aes(x = pd_interaction[[1]], y = pd_interaction[[2]], z = yhat)) +
    geom_tile(aes(fill = yhat)) +
    geom_contour(color = "white", alpha = 0.5) +
    scale_fill_viridis_c(option = "plasma", name = "Predicted\nProbability") +
    theme_minimal() +
    labs(
      title = paste("Interaction Effect Between", top_2_features[1], "and", top_2_features[2]),
      x = top_2_features[1],
      y = top_2_features[2]
    ) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12)
    )
  
  print(interaction_plot)
  ggsave(
    "top_2_interaction_plot.png",
    plot = interaction_plot,
    width = 10,
    height = 8,
    dpi = 300
  )
}

# Create a function to generate PDP plots with better formatting
create_pdp_plot <- function(feature, rf_model, train_data) {
  # Generate partial dependence data
  pdp_data <- partial(
    rf_model,
    pred.var = feature,
    grid.resolution = 50,
    train = train_data,
    plot = FALSE
  )
  
  # Get data distribution to add as a histogram
  feature_data <- train_data[[feature]]
  
  # Create main plot with PDP curve and distribution
  p <- ggplot() +
    # Add density plot at bottom
    geom_density(data = data.frame(x = feature_data), 
                 aes(x = x, y = after_stat(density) * 0.2), 
                 fill = "#69b3a2", alpha = 0.3) +
    # Add PDP line
    geom_line(data = pdp_data, aes(x = pdp_data[[1]], y = yhat), 
              size = 1.2, color = "#330066") +
    # Add reference line at y = 0.5 for classification
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.7) +
    # Styling
    theme_minimal() +
    labs(
      title = paste("Partial Dependence Plot:", feature),
      subtitle = "Shows how predicted probability changes with feature value",
      x = feature,
      y = "Predicted probability of High Yield"
    ) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12)
    )
  
  return(p)
}

# Generate enhanced PDP plots for top features
enhanced_pdp_plots <- list()
for (feature in top_features) {
  p <- create_pdp_plot(feature, pdp_rf_model, train_pdp)
  enhanced_pdp_plots[[feature]] <- p
  
  # Save individual enhanced plots
  ggsave(
    filename = paste0("enhanced_pdp_", feature, ".png"),
    plot = p,
    width = 8,
    height = 6,
    dpi = 300
  )
}

# Create a grid of the enhanced PDPs
enhanced_grid <- gridExtra::grid.arrange(
  grobs = enhanced_pdp_plots[1:min(4, length(enhanced_pdp_plots))],
  ncol = 2
)

# Save the enhanced grid
ggsave(
  "enhanced_pdp_grid.png",
  plot = enhanced_grid,
  width = 14,
  height = 12,
  dpi = 300
)