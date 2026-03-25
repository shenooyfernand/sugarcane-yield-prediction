rm(list=ls())

# ===============================
# 1. LOAD LIBRARIES
# ===============================
library(caret)
library(randomForest)
library(DMwR2)
library(smotefamily)
library(ggplot2)
library(corrplot)
library(GGally)

set.seed(123)

# ===============================
# 2. LOAD DATA
# ===============================
data = read.csv("C:/Users/sheno/OneDrive/Desktop/Final Project/Dataset/Full_dataset.csv")

# Basic checks
str(data)
summary(data)

# Remove missing values
data <- na.omit(data)

# ===============================
# 3. TRAIN / TEST SPLIT (80/20)
# ===============================
size = floor(0.2 * nrow(data))
test_ind = sample(seq_len(nrow(data)), size = size)

test = data[test_ind, ]
train = data[-test_ind, ]

# ===============================
# 4. CREATE TARGET USING K-MEANS
# ===============================
k_result <- kmeans(train$Yield, centers = 2)

# Label training set
train$Yield_cat <- factor(k_result$cluster, labels = c("Low", "High"))

# Get centroids
centroids <- as.numeric(k_result$centers)

# Assign test labels
assign_cluster <- function(x, centers) {
  distances <- abs(x - centers)
  return(which.min(distances))
}

test$Yield_cluster <- sapply(test$Yield, assign_cluster, centers = centroids)
test$Yield_cat <- factor(test$Yield_cluster, labels = c("Low", "High"))

# Remove unnecessary columns
train <- train[, !(names(train) %in% c("Yield"))]
test  <- test[, !(names(test) %in% c("Yield", "Yield_cluster"))]

# ===============================
# 5. EDA SECTION
# ===============================

# ---- Structure ----
str(train)
summary(train)

# ---- Missing Values ----
colSums(is.na(train))

# ---- Numeric Columns ----
num_cols <- names(train)[sapply(train, is.numeric)]

# ---- Histograms ----
for (col in num_cols) {
  print(
    ggplot(train, aes_string(x = col)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "black") +
      ggtitle(paste("Histogram of", col))
  )
}

# ---- Boxplots ----
for (col in num_cols) {
  print(
    ggplot(train, aes_string(y = col)) +
      geom_boxplot(fill = "orange") +
      ggtitle(paste("Boxplot of", col))
  )
}

# ---- Target Distribution ----
print(table(train$Yield_cat))
print(prop.table(table(train$Yield_cat)))

ggplot(train, aes(x = Yield_cat)) +
  geom_bar(fill = "purple") +
  ggtitle("Yield Category Distribution")

# ---- Correlation Matrix ----
cor_matrix <- cor(train[, num_cols])
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)

# ---- Feature vs Target ----
for (col in num_cols) {
  print(
    ggplot(train, aes_string(x = "Yield_cat", y = col)) +
      geom_boxplot(fill = "cyan") +
      ggtitle(paste(col, "vs Yield Category"))
  )
}

# ---- Scatter Plots ----
for (col in num_cols[1:min(5, length(num_cols))]) {
  print(
    ggplot(train, aes_string(x = col, y = num_cols[1])) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm") +
      ggtitle(paste(col, "relationship"))
  )
}

# ---- Pair Plot (optional, heavy) ----
# ggpairs(train[, c(num_cols[1:5], "Yield_cat")])

# ===============================
# 6. PREPROCESSING
# ===============================

# Identify numeric variables
num_vars <- sapply(train, is.numeric)

# Scale numeric data
scaled_num <- scale(train[, num_vars])

# Dummy encoding for categorical variables
dummy_vars <- dummyVars(~ ., data = train[, !num_vars, drop = FALSE], fullRank = TRUE)
encoded_cats <- predict(dummy_vars, newdata = train)

# Combine
train_scaled <- data.frame(scaled_num, encoded_cats, stringsAsFactors = FALSE)

# Add target
train_scaled$Yield_cat <- train$Yield_cat

# ===============================
# 7. TEST PREPROCESSING
# ===============================
num_vars_test <- sapply(test, is.numeric)
scaled_num_test <- scale(test[, num_vars_test])

encoded_cats_test <- as.data.frame(predict(dummy_vars, newdata = test))
encoded_cats_test <- encoded_cats_test[, colnames(encoded_cats)]

test_scaled <- data.frame(scaled_num_test, encoded_cats_test, stringsAsFactors = FALSE)
test_scaled$Yield_cat <- test$Yield_cat

# ===============================
# 8. FINAL DATA FOR MODEL
# ===============================
xtrain = train_scaled[, !(names(train_scaled) %in% c("Yield_cat"))]
ytrain = train_scaled$Yield_cat

xtest = test_scaled[, !(names(test_scaled) %in% c("Yield_cat"))]
ytest = test_scaled$Yield_cat

# ===============================
# 9. CLASS BALANCE CHECK
# ===============================
print(table(ytrain))
print(prop.table(table(ytrain)))



