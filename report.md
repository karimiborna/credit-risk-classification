# Module 12 Report

## Overview of the Analysis

The analysis in this challenge was done to help a scikit-learn supervised learning model classify whether a person's credit would be considered a healthy loan or a high-risk loan, based on the variables: loan size, interest rate, borrower income, and their debt to income ratio.

To train the model, the data was initially split into two camps: 
1. X: The data without the labels, which contained the afforementioned variables which would be the deciding factors in whether the loan would be classified as one label or another.
2. y: The data for only the labels, which contained the outcomes of whether the loan was 0 (healthy) or 1 (high-risk)

It is worth noting that the labels were in a ratio of 75000 0's : 2500 1's, or about 30:1, which is quite imbalanced.

Stages of Machine Learning Process:

After splitting into X and y, the data was split into training and testing data for both X and y, with the same rows being training and testing at random, at about a 75/25 split, respectively. The training data would be for the model to try and learn how to predict the patterns at which the variables affected the outcomes. The testing data would then be used to test the model's predictions and see how successful the learning was.

The model attempted to learn the patterns using a Logistic Regression classifier, with the solver lbfgs, which is a solver which is good for many purposes in small-medium data pools. Once this was done, the model was evaluated on it's balanced accuracy score, a confusion matrix, and a classification report.


This was the end of the first Machine Learning Model's trial, which will be touched on in the 'Results' section.


For the second Machine Learning Model, the data was resampled using the RandomOverSampler module from the imbalanced-learn library to compensate for the 30:1 ratio oof imbalanced data. Resampmling can be an effective tool for removing bias from data, but can come with drawbacks of over or underfitting the data, or distorting the distribution of the features in the dataset. Therefore, the only way to see if it is a useful tool is to try resampling and compare the results to the original data.

After resampling the data, there is a 1:1 ratio of 0's and 1's in the labels. The rest of the process was identical to the first Machine Learning Model's trial (minus the splitting into test/training data), with the data being split into X and y, put into a Logistic Regression classifier using the lbfgs solver, and evaluated with the same 3 result formats. The results will be explained in the results section, and compared to the first model in the 'Summary' section.

## Results

* Machine Learning Model 1:

  * Balanced Accuracy Score: 0.9442676901753825 -- 94.43%

  * Training Data Score: 0.9914878250103177 -- 99.15%

  * Testing Data Score: 0.9924164259182832 -- 99.24%

  * Precision Score:
    * Healthy Loan: 1.00
    * High-Risk Loan: 0.87
    * Macro Average: 0.94
    * Weighted Average: 0.99

  * Recall Score:
    * Healthy Loan: 1.00
    * High-Risk Loan: 0.89
    * Macro Average: 0.94
    * Weighted Average: 0.99

----------------------------------------------------------------------

* Machine Learning Model 2:

  * Balanced Accuracy Score: 0.9445732714963484 -- 94.45%

  * Precision Score:
    * Healthy Loan: 0.90
    * High-Risk Loan: 0.99
    * Macro Average: 0.95
    * Weighted Average: 0.95

  * Recall Score:
    * Healthy Loan: 0.99
    * High-Risk Loan: 0.89
    * Macro Average: 0.94
    * Weighted Average: 0.94


## Summary

Machine Learning Model 1 which contained the imbalanced data initially seemed to work much better for the simply for the weighted average accuracy of the model's predictions being slightly higher. But let's dive deeper:

When considering the confusion matrix for the first model:
| 18679,    80 |
|    67,   558 |
And comparing it to that of the second model:
| 74659,   377 |
|  7941, 67095 |

It becomes clear that the first model contained a considerable amount less false-positives and false-negatives.

Let's also consider the ratios of false positives out of total positives, and the same for negatives for both models, as the ratios will tell more than the simple number as the total number of units in both ends are much higher, so the fact that there are more falsely counted (+/-)s does not tell us much:

Model 1:

Positives:
80/(18679 + 80) = 0.00426461964 == 0.426% falsely classified Healthy Loans

Negatives:
67/(67 + 558)  =         0.1072 == 10.72% falsely classified High-Risk Loans

-----------------------------------------------------------------------------

Model 2:

Positives:
377/(377+74659)   = 0.00502425502 == 0.5% falsely classified Healthy Loans

Negatives:
7941/(7941+67095) = 0.10582920198 == 10.58% falsely classified High-Risk Loans

After extracting the ratios for both, it is clear that both models are actually quite similar. This leaves the question of which problem the loan-givers would like the answer to: Do they want to know which loans they should give? Or would they want to know which loans they shouldn't give? I would predict the latter, meaning that because resampling the data helped to be slightly more accurate when predicting the High-Risk Loans, it is the better model for our purposes. Considering there are far less high-risk loans in our dataset, the ~10.5% of the misclassified High-Risk Loans will probably not make a huge difference for the loan-givers.