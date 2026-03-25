rm(list=ls())

# Install required packages if needed (commented out - run separately if needed)
# install.packages(c("FactoMineR", "factoextra", "clustMixType", "clusterCrit", "cluster", "dplyr", "ggplot2"))

# Load required libraries
library(FactoMineR)    # For FAMD
library(factoextra)    # For cluster visualization
library(clustMixType)  # For k-prototypes clustering
library(clusterCrit)   # For cluster validation
library(cluster)       # For silhouette analysis
library(dplyr)         # For data manipulation
library(ggplot2)       # For visualization

# Set seed for reproducibility
set.seed(123)

# Load data
data <- read.csv("C:/Users/sheno/OneDrive/Desktop/Final Project/Dataset/Full_dataset.csv")

# Display data structure and summary
str(data)
summary(data)

# Remove missing values
data <- na.omit(data)

# Automatically identify numeric and categorical columns
column_types <- sapply(data, class)
numeric_cols <- names(column_types[column_types %in% c("numeric", "integer")])
categorical_cols <- names(column_types[!(column_types %in% c("numeric", "integer"))])

# Convert character columns to factors
char_cols <- names(column_types[column_types == "character"])
for(col in char_cols) {
  data[[col]] <- as.factor(data[[col]])
}

# Recheck column types
column_types <- sapply(data, class)
categorical_cols <- names(column_types[column_types %in% c("factor", "ordered")])

# Print identified columns
cat("Numeric columns:", paste(numeric_cols, collapse=", "), "\n")
cat("Categorical columns:", paste(categorical_cols, collapse=", "), "\n")

# Scale numeric data
data_scaled <- data
data_scaled[numeric_cols] <- scale(data_scaled[numeric_cols])

#---------- K-PROTOTYPES CLUSTERING ----------#

# Function to calculate WSS for k-prototypes
get_wss <- function(data, max_k) {
  wss <- numeric(max_k)
  for (i in 1:max_k) {
    set.seed(123)  # For reproducibility
    kproto_result <- kproto(data, k = i)
    wss[i] <- kproto_result$tot.withinss
  }
  return(wss)
}

# Compute WSS for different values of k (1 to 10)
max_k <- 3
wss <- get_wss(data_scaled, max_k)

# Create elbow plot
elbow_plot <- ggplot(data.frame(k = 1:max_k, wss = wss), aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  labs(x = "Number of clusters (k)", 
       y = "Total within-cluster sum of squares",
       title = "Elbow Method for Optimal k") +
  theme_minimal() +
  scale_x_continuous(breaks = 1:max_k)

print(elbow_plot)

# Number of clusters based on elbow plot
k_optimal <- 4  # Adjust based on your elbow plot

# Run the K-Prototypes clustering algorithm
set.seed(123)
kproto_result <- kproto(data_scaled, k = k_optimal)

# Add cluster assignments to original data
data$kproto_cluster <- kproto_result$cluster

# Examine the cluster centers
cluster_centers <- kproto_result$centers
print("Cluster Centers:")
print(cluster_centers)

# Visualize clustering results with first two numeric variables
if(length(numeric_cols) >= 2) {
  numeric_cluster_plot <- ggplot(data, aes_string(x = numeric_cols[1], y = numeric_cols[2], color = "factor(kproto_cluster)")) +
    geom_point(alpha = 0.7) +
    labs(title = "K-Prototypes Clustering Result", 
         color = "Cluster") +
    theme_minimal()
  
  print(numeric_cluster_plot)
}

# Dimensionality reduction with PCA for visualization
if(length(numeric_cols) > 0) {
  # PCA on numeric variables
  pca_data <- prcomp(data_scaled[numeric_cols], scale = FALSE)
  
  # Create a dataframe with PCA results and cluster assignments
  pca_df <- data.frame(
    PC1 = pca_data$x[,1],
    PC2 = pca_data$x[,2],
    Cluster = factor(data$kproto_cluster)
  )
  
  # Plot PCA results
  pca_cluster_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.7) +
    labs(title = "PCA Visualization of K-Prototypes Clusters") +
    theme_minimal()
  
  print(pca_cluster_plot)
}

#---------- ANALYZE THE CLUSTERS ----------#

# Function to analyze clusters
analyze_clusters <- function(data, cluster_col, numeric_cols, categorical_cols) {
  results <- list()
  
  # Create summary for each cluster
  for(i in unique(data[[cluster_col]])) {
    cat("\n--- Cluster", i, "---\n")
    cluster_data <- data[data[[cluster_col]] == i, ]
    cat("Number of observations:", nrow(cluster_data), "\n")
    
    # Summary of numeric variables
    cat("\nNumeric variables summary:\n")
    numeric_summary <- summary(cluster_data[numeric_cols])
    print(numeric_summary)
    
    # Frequency of categorical variables
    cat("\nCategorical variables frequency:\n")
    cat_summary <- list()
    for(col in categorical_cols) {
      cat("\nVariable:", col, "\n")
      freq_table <- table(cluster_data[[col]])
      print(freq_table)
      cat_summary[[col]] <- freq_table
    }
    
    # Store results
    results[[paste0("Cluster_", i)]] <- list(
      size = nrow(cluster_data),
      numeric_summary = numeric_summary,
      categorical_summary = cat_summary
    )
  }
  
  return(results)
}



# Run the cluster analysis
kproto_analysis <- analyze_clusters(data, "kproto_cluster", numeric_cols, categorical_cols)

# Save the clustered data
write.csv(data, "kproto_clustered_data.csv", row.names = FALSE)

#---------- VALIDATE K-PROTOTYPES CLUSTERS ----------#

# Calculate silhouette score for numeric variables (limitation of clusterCrit)
numeric_data <- as.matrix(data_scaled[numeric_cols])
criteria <- intCriteria(numeric_data, kproto_result$cluster, "Silhouette")
cat("\nSilhouette score (numeric variables only):", criteria$silhouette, "\n")

