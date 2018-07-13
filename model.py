# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
#from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import xgboost as xgb
#from xgboost.sklearn import XGBRegressor
#import datetime
from sklearn.model_selection import GridSearchCV




df_f = pd.read_csv('Baseball_Clean.csv')
clean_ML = [ 'Date','Name_x', 'playerid_x','Away_Team_x', 'Target_y','Unnamed: 0',
       'Home_Team_x', 'Team_Short_Alt_x','Name_y', 'playerid_y','Team_Short_Alt','bats','throws',
       'Roll_BABIP', 'Roll_ISO','1B as L', '1B as R', '2B as L', '2B as R', '3B as L', '3B as R','Roll_TBF','Roll_w OBA',]

#potentially final set???
df_f=df_f.drop(labels = clean_ML,axis=1)

X = df_f.drop('Target_x',axis=1)
y = df_f['Target_x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
"""
#dforest = RandomForestRegressor(n_estimators=75,n_jobs=4)
#dforest = GradientBoostingRegressor()
#n_estimators=50,learning_rate=.1,min_samples_split=4,max_depth=2
dforest = XGBRegressor(booster='gbtree',eta=0.6, min_child_weight = 2, max_depth = 5)
dforest.fit(X_train,y_train)

predictions = dforest.predict(X_test)
              'min_child_weight': [2,3,4],
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

feat_importances = pd.Series(dforest.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(30)
feat_importances.plot(kind='barh')
"""
dforest = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [.05, 0.1 ,0.2,0.5,], #so called `eta` value
              'max_depth': [4, 5, 6, 7],
              'n_estimators': [20, 50,45]}

xgb_grid = GridSearchCV(dforest,
                        parameters,
                        cv = 5,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
