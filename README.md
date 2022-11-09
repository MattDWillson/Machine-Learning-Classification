# Credit Risk Predictor
The algorithm employs multiple machine learning classification methods to predict credit risk. I used the imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling and ensemble learning. 

Tools: Jupyter Lab, Numpy, Pandas, imblearn, sklearn.

## Resampling
The initial process required fetching and cleaning data. I utilized get_dummies and the sklearn label_encoder to modify columns that were previously strings into integers. Modifying the columns that were previously strings into numerical columnns allows you to prepare the data for the various classification methods. After modifying the columns into numerical columns I split the data into train and test sets using the sklearn train_test_split method. Next I scaled the data using the sklearn stanndard scaler. After this process I employed the various resampling methods. 

The resampling techniques I employed were: simple logistic regression, naive random oversampling, SMOTE oversampling, ClusterCentroids undersampling, and SMOTEEN combination sampling. To determine which model performed the best, I conducted the following process for each method to evaluate the models: 
- Calculated balanced accuracy score
- Displayed the confusion matrix 
- Displayed the imbalanced classification report

## Balanced accuracy scores
- Simple Logistic Regression: 0.9520479254722232 
- ClusterCentroids: 0.9865149130022852 
- SMOTEEN: 0.9935715401830394
- Naive Random Oversampling: 0.9936781215845847 
- SMOTE Oversampling: 0.9936781215845847 

Each model had high precision and recall scores for predicting high risk. Each model had a low precision score for predicting low risk. In terms of overall performance the two oversampling methods outperformed the others. 

# Ensemble Learning
The ensemble learning techniques I employed were: imblearn balanced random forest classifier and the sklearn random forest classifier. The imblearn balanced random forest classifier had a higher balanced accuracy score than the sklearn random forest classifer. Both models performed poorly overall with low balanced accuracy scores. However, the sklearn random forest classifier had perfect precision and recall when predicting high credit risk.

Top 10 Features

![Screen Shot 2021-08-09 at 1 22 25 PM](https://user-images.githubusercontent.com/83780964/128748824-a6e5f7d9-5b3e-4f4a-ad52-563319bfa161.png)

Precision Recall Curve (Balanced Random Forest = blue '.' and, Random forest = orange 'x')

<img width="384" alt="Screen Shot 2021-08-13 at 8 35 48 PM" src="https://user-images.githubusercontent.com/83780964/129429230-629efc3d-de3d-4591-9fb5-890b8cb3abe6.png"> 


# Machine Learning: Classification

Classification is the action or process of categorizing something according to shared qualities or characteristics. Classification is the prediction of discrete outcomes. Outcomes are identified as labels/discrete outputs, which serve to categorize bi-class and multi-class features. 

<img width="776" alt="Screen Shot 2022-07-04 at 2 40 35 PM" src="https://user-images.githubusercontent.com/83780964/177204956-650988f7-f8aa-4f07-bfd7-e6a410f97bc6.png">

There are multiple approaches to classification. These include:

<img width="1045" alt="Screen Shot 2022-07-04 at 2 41 29 PM" src="https://user-images.githubusercontent.com/83780964/177205033-1c019cc8-8d23-47c4-960a-ac11a2ceca2d.png">

## Classification 
Classification is used to forecast and predict financial outcomes, automate underwriting and insurance premiums, detect 
and categorize health issues and overall health. Classification models have drastically improved financial efforts to properly categorize applicants, predict market decline, and categorize fraudulent transactions or suspicious activity. FICO credit scoring uses a classification model for its cognitive fraud analytics platform. Classification engines have allowed the financial industry to become more effective and efficient at mitigating risk.

## Making Predictions with Logistic Regression: 
Logistic regression is a common approach used to classify data points and make predictions.

<img width="986" alt="Screen Shot 2022-07-04 at 2 44 34 PM" src="https://user-images.githubusercontent.com/83780964/177205307-f669fa5c-4890-4aed-9057-00911ef09f51.png">

Predictions are made by creating linear paths between data points. 

<img width="1028" alt="Screen Shot 2022-07-04 at 2 45 14 PM" src="https://user-images.githubusercontent.com/83780964/177205365-bebe0088-3bd2-4b2d-a954-02d095604f16.png">

Data points along the trajectory are normalized between 0 and 1. If a value is above a certain threshold, the data point is considered either of class 0 or 1.

<img width="1045" alt="Screen Shot 2022-07-04 at 2 45 46 PM" src="https://user-images.githubusercontent.com/83780964/177205422-f45b1419-fed3-4d3c-9aa7-d4046f777819.png">

## Logistic Regression Model
Running a logistic regression model involves 4 steps, which can be applied when running any machine-learning model:
1) Preprocess. 
2) Train. 
3) Validate. 
4) Predict. 

## Evaluating Predictions

In addition to accuracy, a model must be measured for precision and recall, both of which can be used to eliminate false positives and false negatives.

## Accuracy, Precision, and Recall
Accuracy, precision, and recall are especially important for classification models that involve a binary decision problem. Innacuracy and imprecise models result in models returning false positives and false negatives.

## Accuracy
Accuracy: is how often the model is correct - the ratio of correctly predicted observations to the total number of observations. Scoring will reveal how accurate the model is. However it does not communicate how precise it is. Accuracy can be very susceptible to imbalanced classes.

Accuracy Calculation = (TP+TN)/(TP+TN+FP+FN)

## Precision
Precision: is the ratio of correctly predicted positive observations to the total predicted positive observations. High precision relates to a low false positive rate.

Precision Calculation = TP/(TP+FP)

## Recall
Recall: is the ratio of correctly predicted positive observations to all predicted observations for that class. High recall relates to a more comprehensive output and a low false negative rate.

Recall Calculation = TP/(TP+FN)

## Confusion Matrix
A confusion matrix is used to gauge the success of a model. Confusion matrices reveal the number of true negatives and true positives (actuals) for each categorical class.

Screen Shot 2022-05-06 at 11 27 32 PM https://towardsdatascience.com/understanding-the-confusion-matrix-and-how-to-implement-it-in-python-319202e0fe4d

## Classification Report
A classfication report identifies the precision, recall, and accuracy of a model for each given class.

