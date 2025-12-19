# Model Card
Scalabale ML Pipeline Deployment 
Udacity Data Analyst NanoDegree

This project implements a machine learning pipeline that uses logistic regression to predict income from data provided by the US Census. The program allows for analysis of categorical feature of the data.

The pipeline follows the framework designed and provided by Udacity and is deployed as an API using FastAPI


For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type: Logistic Regression
Programming Language: Python
Framework: Scikit-Learn
Deployment: RESTful API using FastApi

Core packages: 
pandas
numpy
pickle
fastAPI

The pipeline preprocesses the the categorical variables with one-hot encoding and label binarization.


## Intended Use
This model is intended solely as a demonstrative example. The pipeline details the implementation of a complete model fitting pipeline that cou

## Training Data
The Training Data comes from the Adult Census Income dataset that was collected in 1994 by the US Census Bureau(https://www.kaggle.com/datasets/uciml/adult-census-income)The dataset contains roughly 32000 samples with the following features. 
- age
- workclass
- education
- maritial-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital loss
- hours-per-week
- native-country

Target Variable
- salary
The dataset is split via an 80/20 train-test split.


## Evaluation Data

The evaluation data is the fifth of the data left from the above split. Categorical features are evaluated seperately and the results are stored in slice_output.txt

## Metrics
Evaluation performed with typical binary classification metrics.
-Precision : .7245
-Recall : .2458
-F1-score: .3671

The model favors logistic over regression, consistent with our classification and dataset subject.

## Ethical Considerations
- Census data will reflect underlying structual and historical inequalities in the American socio-economic system. 
- The dataset is over 30 years old and findings are likely outdated.
- Biases inherent to the dataset are not mitigated or controlled for in our analysis. 

## Caveats and Recommendations
- This model is strictly deployed for representative educational purpose and claims to have no statistic validity. 


Possible Improvements:
    - Implement cross validation
    - Try more classifiers 
    - Threshold and parameter tuning