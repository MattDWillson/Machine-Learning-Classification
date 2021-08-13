# Machine-Learning-Classification
 
Tools: Jupyter Lab, Numpy, Pandas, imblearn, sklearn

## Overview

- Credit risk assesment tool that uses machine learning classification to predict credit risk. 
- Employs different techniques for training and evaluating models with imbalanced classes.
- Utilizes the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques: Resampling and Ensemble Learning.

# Credit Risk Resampling
## Which model had the best balanced accuracy score?

- Simple Logistic Regression: 0.9520479254722232 
- ClusterCentroids: 0.9865149130022852 
- Combination Over/Under: 0.9935715401830394
- Naive Random Oversampling: 0.9936781215845847 
- Smote Oversampling: 0.9936781215845847 

Naive Random oversampling and Smote Oversampling had the best balanced accuracy score.

## Which model had the best recall score?

- Each model except for the ClusterCentroid model has a recall of .99 for both low risk and high risk.
- ClusterCentroid has a recall of .98 for low risk.

## Which model had the best geometric mean score?

- Each model has a .99 geometric mean score except for the simple logistic regression.

# Credit Risk Ensemble

## Which model had the best balanced accuracy score?

- The Balanced Random Forest Classifier has the best balanced accuracy score.

## Which model had the best recall score?

- The Balanaced Random Forest Classifier has the best recall score. 

## Which model had the best geometric mean score?

- The Balanced Random Forest Classifier has the best geometric mean score.

## What are the top ten features?

![Screen Shot 2021-08-09 at 1 22 25 PM](https://user-images.githubusercontent.com/83780964/128748824-a6e5f7d9-5b3e-4f4a-ad52-563319bfa161.png)
