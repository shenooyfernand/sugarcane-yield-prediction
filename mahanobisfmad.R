rm(list=ls())
library(FactoMineR)
library(factoextra)
library(MASS)  # For robust Mahalanobis distance

library(caret) # Machine Learning Library
library(xgboost) # XGBoost library
library(ggplot2)
library(factoextra)
library(clustMixType)
library(smotefamily)
library(DMwR2)
library(smotefamily)
library(e1071)

set.seed(123)
train = read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Project 1/Air pollution/train.csv")
test = read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Project 1/Air pollution/test.csv")
train=train[-1]
test=test[-1]
#Train/set 80/20
set.seed(123)
# Extract mean and sd from training data
train_mean <- apply(train[, -10], 2, mean)
train_sd <- apply(train[, -10], 2, sd)

# Scale train using its own parameters
train_scaled <- scale(train[, -10], center = train_mean, scale = train_sd)

# Scale test using train parameters
test_scaled <- scale(test[, -10], center = train_mean, scale = train_sd)

# Install if not already
# install.packages("MVN")

library(MVN)
data=train
# Assume your numeric data is in a data frame called `data`
# Ensure only numeric columns are selected
numeric_data <- data[sapply(data, is.numeric)]

# Check for multivariate normality using Mardia's test
mvn_result <- mvn(data = numeric_data, mvn_test = "mardia")

# View results
print(mvn_result)
summary(mvn_result)

pca <- prcomp(train[-10], scale. = TRUE)
lda_model <- lda(pca$x[, 1:7], grouping = train$Air.Quality)  # choose top k PCs

plot(lda_model)  # Quick visualization
pred <- predict(lda_model)$class
confusionMatrix(pred, as.factor(train$Air.Quality)) # if you're evaluating on training set

library(MASS)
library(ggplot2)

# Run LDA on PCA-reduced data (top 2 PCs or more)
lda_model <- lda(pca$x[, 1:7], grouping = train$Air.Quality)

# Get LDA scores (LD1, LD2)
lda_scores <- predict(lda_model)$x
lda_df <- data.frame(lda_scores, Class = train$Air.Quality)

# Create grid for plotting
x_range <- seq(min(lda_df$LD1), max(lda_df$LD1), length.out = 200)
y_range <- seq(min(lda_df$LD2), max(lda_df$LD2), length.out = 200)
grid <- expand.grid(LD1 = x_range, LD2 = y_range)

# Train a new LDA model on LD1 and LD2 only
lda_simple <- lda(Class ~ LD1 + LD2, data = lda_df)

# Predict class on the grid
grid$Class <- predict(lda_simple, newdata = grid)$class

# Plot
ggplot(lda_df, aes(x = LD1, y = LD2, color = Class)) +
  geom_point(alpha = 0.6) +
  geom_contour(data = grid, aes(z = as.numeric(Class)), bins = 3, color = "black") +
  labs(title = "LDA Decision Boundaries", x = "LD1", y = "LD2") +
  theme_minimal()



library(smotefamily)

# Extract response as factor
y <- as.factor(train$Air.Quality)

# Convert all predictors to numeric (skip target)
X <- train[, -which(names(train) == "Air.Quality")]

# Option 1: Use only numeric columns
X_numeric <- X[sapply(X, is.numeric)]
# Apply SMOTE (k = 5 neighbors, over = 200%, under = 200%)
smote_result <- SMOTE(X_numeric, y, K = 5, dup_size = 3)
table(smote_result$data$class)
train=smote_result$data


library(MASS)
qda_model <- qda(train[, -10], grouping = train$class)
qda_pred <- predict(qda_model, newdata = test[, -10])

# Predicted classes
qda_pred$class

# Posterior probabilities
qda_pred$posterior

# Confusion matrix
table(Predicted = qda_pred$class, Actual = test$Air.Quality)
confusionMatrix(qda_pred$class, as.factor(test$Air.Quality))


xtrain=train_scaled
ytrain=train[,10]
xtest=test_scaled
ytest=test[,10]

train=data.frame(xtrain,ytrain)
# Remove missing values
data <- na.omit(train)

# Run FAMD
famd_res <- FAMD(data, graph = F)

# Get FAMD scores (coordinates in lower-dimensional space)
scores <- famd_res$ind$coord

# Compute Robust Mahalanobis distance using MCD (Minimum Covariance Determinant)
mcd_res <- cov.mcd(scores)

# The robust Mahalanobis distances
robust_mahal_dist <- sqrt(mahalanobis(scores, mcd_res$center, mcd_res$cov))

# Set a threshold (e.g., 99% quantile)
threshold <- quantile(robust_mahal_dist, 0.99)
outliers <- which(robust_mahal_dist > threshold)

# Print outliers
print(outliers)

library(ggplot2)

# Create a data frame for coloring
outlier_labels <- ifelse(robust_mahal_dist > threshold, "Outlier", "Normal")

# Plot individuals with a proper legend
fviz_pca_ind(famd_res, label = "none", col.ind = outlier_labels, palette = c("blue", "red")) +
  ggtitle("Outlier Detection using FAMD with Robust Mahalanobis Distance") +
  scale_color_manual(name = "", values = c("Normal" = "blue", "Outlier" = "red")) +
  theme(plot.title = element_text(hjust = 0.5))  # Center title

