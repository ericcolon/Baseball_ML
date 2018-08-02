# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle


################ DATA LOAD #####################

df_f = pd.read_csv('Raw_to_Clean_P.csv')

# full list of variables in case anyone wants to tinker with it

full_list = ['Date', 'Name', 'Team_x', 'Target', 'Roll_TBF_pitch', 'Roll_H_pitch',
               'Roll_2B_pitch', 'Roll_3B_pitch', 'Roll_R_pitch', 'Roll_IP_pitch',
               'Roll_ER_pitch', 'Roll_HR_pitch', 'Roll_BB_pitch', 'Roll_IBB_pitch',
               'Roll_HBP_pitch', 'Roll_SO_pitch', 'Roll_AVG_pitch', 'Roll_OBP_pitch',
               'Roll_SLG_pitch', 'Roll_ERA_pitch', 'Roll_w OBA_pitch',
               'Roll_K/9_pitch', 'Roll_BB/9_pitch', 'Roll_K/BB_pitch',
               'Roll_HR/9_pitch', 'Roll_K%_pitch', 'Roll_BB%_pitch',
               'Roll_K-BB%_pitch', 'Roll_WHIP_pitch', 'Roll_BABIP_pitch',
               'Roll_LOB%_pitch', 'Roll_x FIP_pitch', 'Roll_FIP_pitch',
               'Roll_GB/FB_pitch', 'Roll_LD%_pitch', 'Roll_GB%_pitch',
               'Roll_FB%_pitch', 'Roll_IFFB%_pitch', 'Roll_HR/FB_pitch',
               'Roll_IFH%_pitch', 'Roll_BUH%_pitch', 'Roll_Target_pitch', 'Team_FD',
               '1B as L', '1B as R', '2B as L', '2B as R', '3B as L', '3B as R',
               'HR as L', 'HR as R', 'Throw_L', 'Throw_R']

# used variables for this model 

use_col = ['Target', 'Roll_ERA_pitch',
           
           'Roll_FB%_pitch','Roll_GB%_pitch','Roll_LOB%_pitch','Roll_K%_pitch','Roll_K-BB%_pitch',
           
           'Roll_ER_pitch','Roll_IP_pitch','Roll_SO_pitch', 'Roll_HR_pitch',
           
           'Roll_x FIP_pitch', 'Roll_WHIP_pitch',
           
           '1B as L', '1B as R','3B as L', '3B as R',
           'HR as L', 'HR as R', 'Throw_L']

df_f = df_f[use_col]

################ MODEL SETUP #####################

X = df_f.drop('Target',axis=1)
y = df_f['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

dforest = XGBRegressor(booster='gbtree',eta=.3,min_child_weight = 3, max_depth = 4, n_estimators = 50 ,subsample = .9)

dforest.fit(X_train,y_train)

predictions = dforest.predict(X_test)

################ MODEL EVALUATION #####################

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

feat_importances = pd.Series(dforest.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(40)
feat_importances.plot(kind='barh',figsize=(8,6))

# pickling

pitch_model = 'pitching_model.sav'
pickle.dump(dforest,open(pitch_model, 'wb'))
