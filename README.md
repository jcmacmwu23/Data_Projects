# Logistic Regression Analysis of the Default of Credit Card Clients Dataset Using Sklearn

Data downloaded from <https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients>

Introduction:
I was born and raised in Taiwan and I have deep understanding on the culture and demographics. This dataset interests me because it describes monthly credit card billing and payment behaviors of 30,000 Taiwan individuals from April 2005 to September 2005 with their demographic information (gender, education, marriage, and age) and a binary response variable (y) for the user defaulted in October 2005 (labeled as 1) or not defaulted (labeled as 0). I decided to build a multivariate logistic regression model using this dataset.

Procedures:
I have implemented four Jupyter notebooks for this project:

1. TW_CC_Data Formatting.ipynb: 
The raw dataset was originally collected in a pivoted format for data mining. However, it was not suitable for data analysis and visualization. In this notebook, I cleaned up, organized, and unpivoted the original data so that I could analyze and visualize it. I also created additional metrics to calculate credit card debts and percentages of credit utilization of the users.

2. TW_CC_Data Analysis.ipynb: 
In this notebook, I was able to compare some metrics between US credit card users (using data from <https://www.valuepenguin.com/credit-cards/statistics/usage-and-ownership>) and Taiwan users. I also gained interesting insights into Taiwanese credit card usage behaviors.

3. TW_CC_Regression.ipynb: 
For building a logistic regression model of the dataset, I formatted the gender and age demographic columns and numerical columns into categories and created dummy variables based on those categories. The response variable (y) was not balanced in the dataset, only 22% of the users failed to pay in October. I consequently ran a SMOTE oversampling algorithm to balance the data source for machine learning. In addition, there were 187 variables in the dataset, I ran Recursive Feature Elimination and statsmodel to pick 30 metrics that were significant in the prediction model. The accuracy of the resulting sklearn logistic regression model was 76% with F1 scores 0.8 for negative predictions, 0.71 for positive ones. In order to optimize the model, I added education and marriage categorical columns into the dataset, and the model was improved with accuracy of 79% and a better F1 score 0.77 for positive predictions. 

4. TW_CC_PCA_Regression.ipynb: 
In this notebook, I used feature scaling on the numerical columns and PCA algorithm to further optimize the logistic regression model. This model score was slightly better (80%) and its F1 score in predicting the negatives was good (0.89). However, the model performed poorly in predicting true positives (F1 score 29%). Adding more features into the dataset did not improve the PCA model predictions.

Summary:
The second model built in the TW_CC_Regression.ipynb notebook with 79% accuracy and good F1 scores (0.8/0.77) appeared to be the best model for predicting credit card user default risks in the dataset. I examined the distributions of users classified as true positive (TP)/true negative (TN)/false positive (FP)/false negative (FN) categories in multiple variables in the third Jupyter notebook. I realized that in all the available variables, the distributions showed little differences among TP and FP individuals, so as TN and FN ones. Therefore, the model failed to discern subtle differences among users with current variables and reached the maximum possible accuracy unless I could increase the user population or add more variables. 
