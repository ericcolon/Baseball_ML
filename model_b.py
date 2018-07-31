# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

df_f = pd.read_csv('Baseball_Clean.csv')

clean_ML = ['Date','Name_x', 'playerid_x','Away_Team_x', 'Target_y','Unnamed: 0','Team_FD_x', 'Team_FD_y', 'Team_FD',
       'Home_Team_x', 'Team_Short_Alt_x','Name_y', 'playerid_y','Team_Short_Alt','bats','throws','Roll_LOB%',
       
       'Roll_BABIP', '1B as L', '1B as R', '2B as L', '2B as R', '3B as L', '3B as R','Roll_TBF','Roll_w OBA',
       
       
       'Roll_BB%', 'Roll_SO_y',
       'Roll_FB%', 'Roll_H_y',
       'Roll_2B_y', 'Roll_3B_y', 'Roll_R_y',  'Roll_ER', 'Roll_HR_y',
       'Roll_BB', 'Roll_BB/9',
       'Throw_S',
       
       'Roll_OPS', 'Roll_LD%',
       'Roll_IP', 'Roll_OBP', 
       'Bats_S','Roll_K-BB%','Roll_K/BB',
       
       'Roll_SO_x', 'Roll_HR/FB',
       
       'Roll_K%_x', 'Roll_ISO', 'Roll_GB/FB',
       'HR as L', 'HR as R', 'Roll_K%_y','Bats_L', 'Bats_R', 'Throw_L', 'Throw_R','Roll_Target_y',
       ]
#potentially final set???
df_f=df_f.drop(labels = clean_ML,axis=1)

X = df_f.drop('Target_x',axis=1)
y = df_f['Target_x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
dforest = XGBRegressor(booster='gbtree',eta=0.6, max_depth =9, n_estimators=80, min_child_weigth = 1)
dforest.fit(X_train,y_train)

predictions = dforest.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

feat_importances = pd.Series(dforest.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(40)
feat_importances.plot(kind='barh',figsize=(8,6))

df_b_p = pd.read_csv('Baseball_B_Pred.csv')

use_b = ['Id_x','Position_x','Nickname_x','FPPG_x','Salary_x','Team_x']

df_b = df_b_p[use_b]

use_b_p = [
           'Roll_PA', 'Roll_H_x', 'Roll_1B', 'Roll_2B_x', 'Roll_3B_x',
           'Roll_HR_x', 'Roll_R_x', 'Roll_RBI', 'Roll_AVG', 'Roll_GB%',
           'Roll_Target_x', 'Roll_ERA', 'Roll_WHIP', 'Matchup'
           
            ]

df_b_p = df_b_p[use_b_p]

prediction_b = dforest.predict(df_b_p)

df_b['Predict'] = prediction_b

b_use = ['Id', 'Position', 'Nickname', 'FPPG', 'Salary', 'Team']
b_drop = ['Id_x', 'Position_x', 'Nickname_x', 'FPPG_x', 'Salary_x', 'Team_x']

for stat in b_use:
    df_b[stat] = df_b[stat+'_x']
    
df_b = df_b.drop(labels = b_drop,axis=1)

df_b.to_csv("Batter_Forecast.csv", sep=',', encoding='utf-8')