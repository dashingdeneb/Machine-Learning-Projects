# **E-Commerce Conversion Prediction Challenge**
This challenge focuses on predictive modeling using structured tabular data. Participants are expected to analyze the dataset, perform appropriate preprocessing, engineer meaningful features, and build machine learning models to predict customer conversion outcomes.
The competition is designed to assess:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Classification Modeling
- Model Evaluation

## Objective
Given the features provided in the dataset, predict whether a user converts.
Target Variable
Converted
- 1 = Converted
- 0 = Not Converted

## Files Provided
**train.csv**

Contains feature columns along with the target variable.

**public_test.csv**

Contains feature columns along with the target variable.
Participants may use this dataset for validation and experimentation.

**private_test.csv**

Contains feature columns only.
Participants must generate predictions for this dataset.

**sample_submission.csv**

Contains the required submission format.


## Evaluation Metric
Submissions will be evaluated using:
- F1 Score

Final rankings will be determined using the F1 Score obtained on the hidden labels corresponding to private_test.csv.
Higher scores indicate better predictive performance.

## Programming Language
Only Python may be used for model development and submission generation.
