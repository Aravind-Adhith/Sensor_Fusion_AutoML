# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model                   |   Weight |
|:------------------------|---------:|
| 4_Default_Xgboost       |        4 |
| 5_Default_NeuralNetwork |        1 |

## Metric details
|           |      score |     threshold |
|:----------|-----------:|--------------:|
| logloss   | 0.00210091 | nan           |
| auc       | 0.999998   | nan           |
| f1        | 0.999285   |   0.188872    |
| accuracy  | 0.999598   |   0.188872    |
| precision | 1          |   0.999855    |
| recall    | 1          |   5.61117e-05 |
| mcc       | 0.999006   |   0.188872    |


## Metric details with threshold from accuracy metric
|           |      score |   threshold |
|:----------|-----------:|------------:|
| logloss   | 0.00210091 |  nan        |
| auc       | 0.999998   |  nan        |
| f1        | 0.999285   |    0.188872 |
| accuracy  | 0.999598   |    0.188872 |
| precision | 0.998571   |    0.188872 |
| recall    | 1          |    0.188872 |
| mcc       | 0.999006   |    0.188872 |


## Confusion matrix (at threshold=0.188872)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             1788 |                1 |
| Labeled as 1 |                0 |              699 |

## Learning curves
![Learning curves](learning_curves.png)
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



[<< Go back](../README.md)
