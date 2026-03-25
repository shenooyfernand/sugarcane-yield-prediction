rm(list=ls())
library(caret) # Machine Learning Library
library(xgboost) # XGBoost library
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


library(ROCR)


pred_probs <- predict(xgb_model, xtest, type = "prob")[,2]  
ytest_binary <- ifelse(ytest == "Low", 1, 0)
preod = predict(xgb_model,xtrain,type = "prob")[,2]
ytrain_binary= ifelse(ytrain=="High", 1, 0)
pred <- prediction(pred_probs, ytest_binary)
pred1=prediction(preod,ytrain_binary)
# Compute AUC
perf <- performance(pred, "auc")
perf1 <- performance(pred1, "auc")

auc_value <- perf@y.values[[1]]
auc_value1 <- perf1@y.values[[1]]# Extract AUC value
print(paste("AUC-ROC:", auc_value))
print(paste("AUC-ROC:", auc_value1))
# Compute TPR (True Positive Rate) and FPR (False Positive Rate)
perf_roc <- performance(pred, "tpr", "fpr")

# Plot ROC curve
plot(perf_roc, col = "blue", main = "ROC Curve for XGBoost Model")
abline(a=0, b=1, lty=2, col="gray")  # Add diagonal reference line


importance_matrix <- xgb.importance(model = xgb_model$finalModel)
print(importance_matrix)
#for remove unimportant varibles and model again
unimpor=setdiff(colnames(xtrain),importance_matrix$Feature)

newtrain=xtrain[,importance_matrix$Feature]
newtest=xtest[,importance_matrix$Feature]

xnewtrain=newtrain[,1:ncol(newtrain)]
ynewtrain=ytrain
xnewtest=newtest[,1:ncol(newtest)]
ynewtest=ytest


g=numeric(0)
g1=numeric(0)
h=list()
j=2
for (i in 1){  
  # Doing XGBoost for classification purposes.
  #found this parameters unsing random parameter and done this several times
  #in 1:20 for loop
  grid_tune <- expand.grid(
    nrounds = 1000,        # Reduced number of trees
    max_depth = 3,       # Moderate tree depth range
    eta = 0.01,           # Higher learning rates for faster convergence
    gamma = 0.3,           # Minimal pruning options
    colsample_bytree = 0.6, # Reasonable feature sampling
    min_child_weight = 5,   # Basic complexity control
    subsample = 0.8 )        # runif(1,0.5,0.8)
  
  
  train_control <- trainControl(method = "cv",
                                number=5,
                                verboseIter = TRUE,
                                allowParallel = TRUE,
  )
  xgb_tune <- train(x = xnewtrain,
                    y = ynewtrain,
                    trControl = train_control,
                    tuneGrid = grid_tune,
                    method= "xgbTree",
                    verbose = TRUE
  )
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
  xgb_model <- train(x = xnewtrain,
                     y = ynewtrain,
                     trControl = train_control,
                     tuneGrid = final_grid,
                     method = "xgbTree",
                     verbose = TRUE
  )
  
  #predict(xgb_model, xnewtest)
  
  # Prediction:
  xgb.pred <- predict(xgb_model, xnewtest)
  
  #' Confusion Matrix
  
  xgb.pred <- as.factor(xgb.pred)
  ynewtest <- as.factor(ynewtest)
  
  # Ensure levels match
  levels(xgb.pred) <- levels(ynewtest)
  
  # Compute confusion matrix
  f=confusionMatrix(xgb.pred, ynewtest)
  confusionMatrix(xgb.pred, ynewtest)
  predicttrai=predict(xgb_model, xnewtrain)
  f1=confusionMatrix(predicttrai, as.factor(ynewtrain))
  
  
  
  
  h[[j]]=c(xgb_tune$bestTune$nrounds,
           xgb_tune$bestTune$eta,
           xgb_tune$bestTune$max_depth,
           xgb_tune$bestTune$gamma,
           xgb_tune$bestTune$colsample_bytree,
           xgb_tune$bestTune$min_child_weight,
           xgb_tune$bestTune$subsample)
  g[j]=f$overall[1]
  g1[j]=f1$overall[1]
  j=j+1
}
G=abs(g-g1)
position <- which(G == min(G,na.rm = T), arr.ind = TRUE)


train_control1 <- trainControl(method = "none",
                               verboseIter = TRUE,
                               allowParallel = TRUE)
final_grid1 <- expand.grid(nrounds = h[[position]][1],
                           eta =      h[[position]][2],
                           max_depth =h[[position]][3],
                           gamma =    h[[position]][4],
                           colsample_bytree = h[[position]][5],
                           min_child_weight = h[[position]][6],
                           subsample =        h[[position]][7])
xgb_model <- train(x = xnewtrain,
                   y = ynewtrain,
                   trControl = train_control1,
                   tuneGrid = final_grid1,
                   method = "xgbTree",
                   verbose = TRUE
)

#predict(xgb_model, xnewtest)

# Prediction:
xgb.pred <- predict(xgb_model, xnewtest)
dd=predict(xgb_model, xnewtest,type = "prob")
bb=xnewtest

dd1=predict(xgb_model, bb,type = "prob")
#' Confusion Matrix

xgb.pred <- as.factor(xgb.pred)
ynewtest <- as.factor(ynewtest)

predicttrai=predict(xgb_model, xnewtrain)
# Ensure levels match
levels(xgb.pred) <- levels(ynewtest)
levels(predicttrai) <- levels(as.factor(ynewtrain))

# Compute confusion matrix
f=confusionMatrix(xgb.pred, ynewtest)
confusionMatrix(xgb.pred, ynewtest)
f$byClass


f1=confusionMatrix(predicttrai, as.factor(ynewtrain))
confusionMatrix(predicttrai, as.factor(ynewtrain))
f1$byClass

importance_matrix <- xgb.importance(model = xgb_model$finalModel)
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix)
mtext("Feature importance scores show the most influential variables in predicting overwork.", side = 1, line = 4, cex = 0.8)


# Current confusion matrix (positive = "not overworked")
conftestno <- confusionMatrix(xgb.pred, ynewtest, positive = "Low")

# Get metrics for the other class
conftesto <- confusionMatrix(xgb.pred, ynewtest, positive = "High")

conftrainno <- confusionMatrix(predicttrai, as.factor(ynewtrain), positive = "Low")

# Get metrics for the other class
conftraino <- confusionMatrix(predicttrai, as.factor(ynewtrain), positive = "High")

conftrainno$byClass
conftraino$byClass
conftestno$byClass
conftesto$byClass

