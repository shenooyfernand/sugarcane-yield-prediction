## Sugarcane Yield Prediction

**Dataset:** South Asian Sugarcane Production Dataset (Kaggle, 50,000 observations, 20 variables)  
**Tools:** R (ggplot2, dplyr, FactoMineR, randomForest, xgboost, clustMixType)

### Objective
Predict sugarcane yield category (High/Low) across 5 South Asian countries 
(India, Bangladesh, Nepal, Pakistan, Sri Lanka) based on soil type, fertilizer, 
irrigation method, climate, and farming variables.

### Methods
- Exploratory Data Analysis (EDA)
- Data Cleaning: corrected erroneous fertilizer entry affecting ~25% of records
- Factor Analysis for Mixed Data (FAMD)
- K-Prototypes Clustering (mixed numerical and categorical data)
- Mutual Information Feature Selection
- Outlier Detection: Isolation Forest + Robust Mahalanobis Distance
- Models: Random Forest, XGBoost, Logistic Regression
  
### Key Results
- XGBoost performed best with 51.07% test accuracy
- FAMD identified 4 latent components capturing year-based, 
  seasonal, and regional farming patterns
- Top predictors: remaining area, year, and country (Nepal)
- Analysis confirmed the dataset has weak predictive signal 
  for yield classification
