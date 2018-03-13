# Sklearn_Reporting
Automated generation of model performance reports for models created with Scikit-learn library

This project is an attempt to create a simple and modular library for automatically generating PDF performance report of a model developed with the sklearn library. Currentely supporting regression models and binary classification problems. In the future there are planned updates to include more metrics as well as support the integration of custom metrics and report types. 

## Dependencies

> Scikit-learn

> Reportlab

> Numpy

## Supported Models
> Any regression or binary classification model containing the predict() and predict_proba() methods (as outlined in Sklearn) respectively

## Supported Metrics

Regression:

> Mean Absolute Error
> Mean Squared Error

> R^2

> Explained Variance Ratio

Classification: 

> AUC

> Accuracy (at 0.5)

> Brier Score

> F1 Score

## Examples
> See usage_example.ipynb

## Planned Updates
> Addition of more model types
> Addition of more metrics
> Addition of custom metrics
