# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

df_f = pd.read_csv('Baseball_Clean_P.csv')
clean_ML = [ 'Date', 'Name', 'playerid', 'throws', 'Away_Team',
       'Home_Team', 'Team_Short', 'Team_Short_Alt','Unnamed: 0',
       
       'Roll_TBF', 'Roll_H', 'Roll_2B', 'Roll_3B', 'Roll_R',
       'Roll_IP',  'Roll_HR', 'Roll_BB', 'Roll_OBP',
       'Roll_ERA', 'Roll_w OBA', 'Roll_BB/9', 'Roll_K/BB',
       '1B as L', '1B as R', '2B as L',
       '2B as R','Team_FD_x', 'Team_FD_y',
       
       'Throw_L',
       'Throw_R', 'Throw_S', 'Roll_Target'
       ]

#potentially final set???
df_f=df_f.drop(labels = clean_ML,axis=1)

X = df_f.drop('Target',axis=1)
y = df_f['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

dforest = XGBRegressor(booster='gbtree',eta=.6,min_child_weight = 3, max_depth = 9, n_estimators = 80)

dforest.fit(X_train,y_train)

predictions = dforest.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

feat_importances = pd.Series(dforest.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(40)
feat_importances.plot(kind='barh',figsize=(8,6))

df_P_p = pd.read_csv('Baseball_P_Pred.csv')

use_P = ['Id', 'Position', 'Nickname', 'FPPG', 'Salary', 'Team']

df_p = df_P_p[use_P]

use_P_p = ['Roll_ER', 'Roll_SO', 'Roll_K%', 'Roll_K-BB%', 'Roll_WHIP',
           'Roll_LOB%', 'Roll_GB/FB', 'Roll_FB%',  '3B as L', '3B as R', 'HR as L', 'HR as R']

df_P_p = df_P_p[use_P_p]

prediction_P = dforest.predict(df_P_p)

df_p['Predict'] = prediction_P

df_p.to_csv("Pitcher_Forecast.csv", sep=',', encoding='utf-8')