# Sensor_Fusion_AutoML

Coursework Project for Applied Machine Learning for Mechanical Engineers

## Objective

Fusion of data from Multiple sensors to detect fire using AutoML

## Dataset

The dataset consists of 7 sensors and timestamps. The sensors are:

1. Temperature
2. Humidity
3. Pressure
4. VOC
5. Carbon Di Oxide
6. Hydrogen and Ethanol concentration
7. Particulate Matter concentration

## AutoML Overview

AutoML is a process of automating the end-to-end machine learning process. It includes:

1. Data preprocessing
2. Feature engineering
3. Model selection
4. Hyperparameter tuning
5. Model deployment

<img src = resources\Picture1.jpg>

## Libraries Used

1. [FLAML](https://github.com/microsoft/FLAML)
2. [MLJAR](https://github.com/mljar/mljar-supervised)

## Results

### FLAML Analytics

| | |
|---|---|
|Best Learner | LGBM (Decision Tree)|
| Time for finding Best Model  | 5.19 Seconds |
| Best Model Accuracy | 99.98% |
| Training Time | 0.053 Seconds |
| Final Learning Rate | 0.09 |


<img src = resources\Picture5.png>

### MLJAR Analytics

| | |
|---|---|
| Best Model Type | Ensemble |
| Models Used | Xgboost(4) and Neural Network(1) |
| Accuracy | 99.98% |
| Training Time | 1.83 Secs |


<img src = "resources\Picture4.png">

