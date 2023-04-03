# Aravind Adhith Pandian Saravanakumaran
# 1222209391 
# apandia1

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from fast_ml.feature_engineering import FeatureEngineering_DateTime
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flaml import AutoML
from flaml.data import get_output_from_log

class SensorFusion():
    def __init__(self):

        tic = time.time()
        self.csv_reader()
        self.csv_processing()
        self.flaml_train_model()
        self.accuracy()
        self.predict()
        toc = time.time()
        print("Time taken for execution: ",toc-tic)

    def visualise(self):
        self.results = pd.read_csv(r'solutionflaml.csv')        
        result = self.results['Fire Alarm']
        actual = self.actual

        print('MAPE: ',mean_absolute_percentage_error(actual,result))
        print("\n")
        print('Accuracy: ',(result == actual).sum()/len(actual)*100)
        print("\n")

    def predict(self):
        print(classification_report(self.y_train, self.automl.predict(self.X_train)))
        print(classification_report(self.y_test, self.automl.predict(self.X_test)))

        sol = pd.DataFrame(self.y_pred,columns=['Fire Alarm'])
        sol.to_csv('solutionflaml.csv',index=False)
        print("\n")
        print("----------------------------------------------------------------------------------------")

        # print(sol.head())

    def accuracy(self):
        print("\n \n \n ")
        print("----------------------------------------------------------------------------------------")
        print("Statistics of the best model found by AutoML:")
        print('Best ML leaner:', self.automl.best_estimator)
        print("\n")
        print('Best hyperparmeter config:', self.automl.best_config)
        print("\n")
        self.visualise()
        print('Best Config Per Estimator:', self.automl.best_config_per_estimator)
        print("\n")
        print('Time for finding best model:', self.automl.time_to_find_best_model)
        print("\n")
        print('Best accuracy on validation data: {0:.4g}'.format(1-self.automl.best_loss))
        print("\n")
        print('Training duration of best run: {0:.4g} s'.format(self.automl.best_config_train_time))
        print("\n")
        print('Config history:', self.automl.config_history)
        print("\n")

    def flaml_train_model(self):
        y = self.train.pop('FireAlarm')
        X = self.train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X, y, test_size=0.2, random_state=42,shuffle=True, stratify=y)

        self.automl = AutoML()
        self.automl.fit(self.X_train, self.y_train, task="classification",metric='accuracy',time_budget=900,log_file_name='flaml.log')
        self.y_pred = self.automl.predict(self.test)

    def csv_processing(self):
        self.train['UTC'] = pd.to_datetime(self.train['UTC'],unit='s')
        self.test['UTC'] = pd.to_datetime(self.test['UTC'],unit='s')

        dt_fe = FeatureEngineering_DateTime()

        dt_fe.fit(self.train, datetime_variables=['UTC'])
        self.train = dt_fe.transform(self.train)

        self.test = dt_fe.transform(self.test)

        self.nunique_train = self.train.nunique().reset_index()
        remove_col=self.nunique_train[(self.nunique_train[0]==len(self.train)) | (self.nunique_train[0]==0) | (self.nunique_train[0]==1) ]['index'].tolist()
        
        pd.set_option('display.max_columns', None)

        self.train=self.train.drop(remove_col,axis=1)
        self.test=self.test.drop(remove_col,axis=1)
        
        self.train = self.train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        self.test = self.test.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        self.train = self.train.drop(['UTCtime'],axis=1)
        self.test = self.test.drop(['UTCtime'],axis=1)

        self.train['UTCday_part'] = self.train['UTCday_part'].fillna(value=np.nan)
        self.test['UTCday_part'] = self.test['UTCday_part'].fillna(value=np.nan)

        self.train['UTCday_part'] = self.train['UTCday_part'].fillna(value='midnight')
        self.test['UTCday_part'] = self.test['UTCday_part'].fillna(value='midnight')

        self.train['UTCday_part']=self.train['UTCday_part'].replace({'midnight':0,'dawn':1, 'early morning':2, 'late morning':3, 'noon':4,
                                                       'afternoon':5,'evening':6, 'night':7 })
        self.test['UTCday_part']=self.test['UTCday_part'].replace({'midnight':0,'dawn':1, 'early morning':2, 'late morning':3, 'noon':4,
                                                         'afternoon':5,'evening':6, 'night':7 })

    def csv_reader(self):

        self.test = pd.read_csv(r'test_dataset.csv')
        self.train = pd.read_csv(r'train_dataset.csv')

        self.actual = self.test['Fire Alarm']

if __name__ == '__main__':
    
    sensor = SensorFusion()