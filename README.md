ML — Analyzing the Best Fit Regression Model for Forest Fire Prediction 
This project focuses on predicting forest fire occurrences using multiple machine learning regression algorithms — Linear Regression, Ridge, Lasso, and ElasticNet — to determine the best-performing model. 
 The dataset used is the Algerian Forest Fires Dataset. 

 
## Project Overview 

The main goal of this project is to build, compare, and evaluate different regression models to find the best fit model for predicting forest fire behavior. 
 This end-to-end process includes data cleaning, exploratory data analysis (EDA), feature engineering, cross-validation, and model performance evaluation. 


## Steps Involved 

1) Importing essential libraries — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
2) Loading the Algerian Forest Fires dataset using Pandas
3) Checking and handling missing values and duplicate entries
4) Cleaning the dataset and creating a new column based on the fire region
5) Verifying data types and reassigning them as needed
6) Creating histograms to understand error distribution
7) Dropping irrelevant columns
8) Encoding independent and dependent features
9) splitting data into train and test sets
10) Checking multicollinearity using a heatmap
11) Removing features with correlation > 0.85
12) Applying feature scaling and standardization
13) Using boxplots to visualize standardization effects
14) Implementing regression models — Linear, Ridge, Lasso, and ElasticNet
15) Evaluating model performance using MAE, R² Score, and Cross-Validation
16) Visualizing results with scatter plots


## Learnings & Insights 

gs & Insights 

1) Hands-on implementation of machine learning models with visualization and cross-validation
2) Understanding performance metrics and identifying the best regression model
3) Effective data cleaning and feature engineering
4) Handling multicollinearity and applying correlation thresholds
5) Model comparison and fine-tuning through cross-validation


Model                      MAE (Before CV)                     MAE (After CV)           r2 score(Before CV)       r2 score(afterr CV)       cross validtion on (cv)             alpha        observation    

Linear Regression           0.54                              -                                0.98                  -                          -                                  -            HIgh baseline accurancy

Lasso Regression           1.13                            0.61                              0.94                  0.98                          5-fold                          0.057           improved significantly after turning cv

Ridge Regression            0.564                          0.56                              0.984                  0.984                        5-fold                          1.0               stable and cosistent 

ElasticNet Regression       1.882                           0.658                            0.875                  0.981                        5-fold                          0.043                major improvement best of all model


##  Final Conclusion (Short Version) 


After comparing multiple regression models on the Algerian Forest Fires Dataset, ElasticNet Regression emerged as the best-performing model with MAE = 0.658 and R² = 0.981. It provided the most balanced and generalized results after cross-validation, outperforming Linear, Lasso, and Ridge regressions in predicting forest fire occurrences. 











