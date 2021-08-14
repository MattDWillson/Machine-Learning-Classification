# Credit Risk Predictor
Tools: Jupyter Lab, Numpy, Pandas, imblearn, sklearn
The algorithm employs multiple machine learning classification methods to predict credit risk. I used the imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling and ensemble learning.

## Resampling
The initial process required fetching and cleaning data. I utilized get_dummies and tbe sklearn label_encoder to modify columns that were previously strings into integers. Modifying the columns that were previously strings into numerical columnns allows you to prepare the data for the various classification methods. After modifying the columns into numerical columns I split the data into train and test sets using the sklearn train_test_split method. Next I scaled the data using the sklearn stanndard scaler. After this process I employed the various resampling methods. 

The resampling techniques I employed were: simple logistic regression, naive random oversampling, SMOTE oversampling, ClusterCentroids undersampling, and SMOTEEN combination sampling. To determine which model performed the best, I conducted the following process for each method: 
- Calculated balanced accuracy score
- Displayed the confusion matrix 
- Displayed the imbalanced classification report

The naive random oversamplinng and SMOTE oversampling methods generated the highest balanced accuracy scores. 
- Simple Logistic Regression: 0.9520479254722232 
- ClusterCentroids: 0.9865149130022852 
- SMOTEEN: 0.9935715401830394
- Naive Random Oversampling: 0.9936781215845847 
- SMOTE Oversampling: 0.9936781215845847 

Each model had high precision and recall scores for predicting high risk. Each model had a low precision score for predicting low risk. In terms of overall performance the two oversampling methods outperformed the others. 

# Ensemble Learning
The ensemble learning method required the same process of cleaning and preparing the data to perform classification methods. The data set used in this algorithm contained more columns as strings - I adjusted the columns to numerical columns using get_dummies and the sklearn label_encoder. Next I split the data into training and testing sets using the sklearn train_test_split method. I then scaled the data using the sklearn standard scaler. 

The ensemble learning techniques I employed were: imblearn balanced random forest classifier and the sklearn random forest classifier. The imblearn balanced random forest classifier had a higher balanced accuracy score than the sklearn random forest classifer. Both models performed poorly overall with low balanced accuracy scores. However, the sklearn random forest classifier had perfect precision and recall when predicting high credit risk.

Top 10 Features
![Screen Shot 2021-08-09 at 1 22 25 PM](https://user-images.githubusercontent.com/83780964/128748824-a6e5f7d9-5b3e-4f4a-ad52-563319bfa161.png)

Precision Recall Curve (Balanced Random Forest = blue '.' and Random forest = orange 'x')
<img width="384" alt="Screen Shot 2021-08-13 at 8 35 48 PM" src="https://user-images.githubusercontent.com/83780964/129429230-629efc3d-de3d-4591-9fb5-890b8cb3abe6.png">

