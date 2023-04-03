# Summary of 3_Linear

[<< Go back](../README.md)


## Logistic Regression (Linear)
- **n_jobs**: -1
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

5.8 seconds

## Metric details
|           |    score |     threshold |
|:----------|---------:|--------------:|
| logloss   | 0.023905 | nan           |
| auc       | 0.999835 | nan           |
| f1        | 0.992898 |   0.135242    |
| accuracy  | 0.995981 |   0.135242    |
| precision | 1        |   1           |
| recall    | 1        |   1.64217e-08 |
| mcc       | 0.990144 |   0.135242    |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.023905 |  nan        |
| auc       | 0.999835 |  nan        |
| f1        | 0.992898 |    0.135242 |
| accuracy  | 0.995981 |    0.135242 |
| precision | 0.985896 |    0.135242 |
| recall    | 1        |    0.135242 |
| mcc       | 0.990144 |    0.135242 |


## Confusion matrix (at threshold=0.135242)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             1779 |               10 |
| Labeled as 1 |                0 |              699 |

## Learning curves
![Learning curves](learning_curves.png)

## Coefficients
| feature        |   Learner_1 |
|:---------------|------------:|
| CNT            | 10.5565     |
| UTCday_of_week |  4.85739    |
| UTChour        |  3.08894    |
| RawH2          |  1.44934    |
| UTCday_of_year |  1.40172    |
| UTCday         |  1.40172    |
| eCO2ppm        |  1.05993    |
| PressurehPa    |  0.858327   |
| Humidity       |  0.671235   |
| RawEthanol     |  0.574402   |
| NC05           |  0.276805   |
| PM10           |  0.151144   |
| PM25           |  0.00920519 |
| NC10           | -0.00185943 |
| UTCsecond      | -0.0557327  |
| NC25           | -0.130952   |
| TVOCppb        | -0.219458   |
| UTCminute      | -0.393762   |
| TemperatureC   | -0.858996   |
| UTCweekofyear  | -1.12415    |
| intercept      | -1.71429    |
| UTCday_part    | -2.26114    |


## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



## SHAP Importance
![SHAP Importance](shap_importance.png)

## SHAP Dependence plots

### Dependence (Fold 1)
![SHAP Dependence from Fold 1](learner_fold_0_shap_dependence.png)

## SHAP Decision plots

### Top-10 Worst decisions for class 0 (Fold 1)
![SHAP worst decisions class 0 from Fold 1](learner_fold_0_shap_class_0_worst_decisions.png)
### Top-10 Best decisions for class 0 (Fold 1)
![SHAP best decisions class 0 from Fold 1](learner_fold_0_shap_class_0_best_decisions.png)
### Top-10 Worst decisions for class 1 (Fold 1)
![SHAP worst decisions class 1 from Fold 1](learner_fold_0_shap_class_1_worst_decisions.png)
### Top-10 Best decisions for class 1 (Fold 1)
![SHAP best decisions class 1 from Fold 1](learner_fold_0_shap_class_1_best_decisions.png)

[<< Go back](../README.md)
