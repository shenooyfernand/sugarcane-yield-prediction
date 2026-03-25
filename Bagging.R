rm(list=ls())
library(caret) # Machine Learning Library
library(xgboost) # XGBoost library
library(DMwR2)
library(smotefamily)
set.seed(123)
data = read.csv("C:/Users/sheno/OneDrive/Desktop/Group 9/3rd year/Final Project/Dataset/Full_dataset.csv")
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
dummy_vars <- dummyVars(~ ., data = train[, !num_vars, drop = FALSE],fullRank = TRUE)  # Exclude response
encoded_cats <- predict(dummy_vars, newdata = train)

train_preprocessed <- data.frame(scaled_num, encoded_cats,train[20],stringsAsFactors = F)  # Keep response unchanged


train_scaled <- train_preprocessed



#for test
num_vars1 <- sapply(test, is.numeric)  # Logical vector: TRUE for numeric, FALSE for categorical

scaled_num1 <- scale(test[, num_vars1])
num_vars1[20]=TRUE


# Convert categorical predictors to dummy variables
library(caret)

encoded_cats1 <- as.data.frame(predict(dummy_vars, newdata = test))
encoded_cats1 <- encoded_cats1[, colnames(encoded_cats)]  # Match column order/names


test_preprocessed <- data.frame(scaled_num1, encoded_cats1,test[20],stringsAsFactors = F)  # Keep response unchanged

test_scaled <- test_preprocessed


xtrain=train_scaled[,1:36]
ytrain=train_scaled[,37]
xtest=test_scaled[,1:36]
ytest=test_scaled[,37]

table(ytrain)



#using grid search found these are optimal hyperparameters
# Doing XGBoost for classification purposes.
grid_tune <- expand.grid(
  nrounds = 1000,        # Reduced number of trees
  max_depth = 3,       # Moderate tree depth range
  eta = 0.01,           # Higher learning rates for faster convergence
  gamma = 0.3,           # Minimal pruning options
  colsample_bytree = 0.6, # Reasonable feature sampling
  min_child_weight = 5,   # Basic complexity control
  subsample = 0.8      # Moderate to full sample usage
)

train_control <- trainControl(method = "cv",
                              number=3,
                              verboseIter = TRUE,
                              allowParallel = TRUE)
xgb_tune <- train(x = xtrain,
                  y = ytrain,
                  trControl = train_control,
                  tuneGrid = grid_tune,
                  method= "xgbTree",
                  verbose = TRUE)
xgb_tune

# Best tune
xgb_tune$bestTune

# Writing out the best model.

train_control <- trainControl(method = "none",
                              verboseIter = TRUE,
                              allowParallel = TRUE)
final_grid <- expand.grid(nrounds = xgb_tune$bestTune$nrounds,
                          eta = xgb_tune$bestTune$eta,
                          max_depth = xgb_tune$bestTune$max_depth,
                          gamma = xgb_tune$bestTune$gamma,
                          colsample_bytree = xgb_tune$bestTune$colsample_bytree,
                          min_child_weight = xgb_tune$bestTune$min_child_weight,
                          subsample = xgb_tune$bestTune$subsample)
xgb_model <- train(x = xtrain,
                   y = ytrain,
                   trControl = train_control,
                   tuneGrid = final_grid,
                   method = "xgbTree",
                   verbose = TRUE)

predict(xgb_model, xtest)

# Prediction:
xgb.pred <- predict(xgb_model, xtest)

#' Confusion Matrix

xgb.pred <- as.factor(xgb.pred)
ytest <- as.factor(ytest)

# Ensure levels match
levels(xgb.pred) <- levels(ytest)

# Compute confusion matrix
f=confusionMatrix(xgb.pred, ytest,positive = "Low")
f1=confusionMatrix(xgb.pred,ytest,positive = "High")
print(f)
print(f1)
f$byClass
f1$byClass

trapred=predict(xgb_model,xtrain)
levels(trapred) <- levels(ytrain)
f3=confusionMatrix(trapred,ytrain,positive = "Low")
f4=confusionMatrix(trapred,ytrain,positive="High")
print(f3)
print(f4)
f3$byClass
f4$byClass

library(caret)
ctrl= trainControl(method = "cv", number = 5)
bag_model=train(Yield_cat~.,data=train_scaled,method="treebag",trControl=ctrl,nbagg = 100)

bagpredict=predict(bag_model,newdata=xtest)
confusionMatrix(bagpredict, as.factor(ytest))
                
