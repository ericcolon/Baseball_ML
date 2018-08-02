# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
import pickle

################ DATA LOAD #####################

df_b = pd.read_csv('Daily_Raw_to_Clean_B.csv')

df_p = pd.read_csv('Daily_Raw_to_Clean_P.csv')

# using the bottom stats for all non numeric values since it'll crash the regressor

df_b_s = ['Id_bat', 'Position_bat', 'Nickname_bat', 'FPPG_bat', 'Salary_bat',
            'Team_bat', 'Batting Order', 'bats']

df_p_s = ['Id', 'Position', 'Nickname', 'FPPG', 'Salary', 'Team', 'throws']

df_b_str = df_b[df_b_s]

df_p_str = df_p[df_p_s]

# set up the format so that variables match up with modeling variables

use_b = ['Roll_PA_bat','Roll_HR_bat','Roll_GB/FB_bat',
           
           'Roll_BB%_bat', 'Roll_K%_bat','Roll_GB%_bat', 'Roll_FB%_bat', 
           
           'Roll_ISO_bat','Roll_wRC+_bat',
           
           'Roll_ER_pitch','Roll_IP_pitch','Roll_SO_pitch', 
           
           'Roll_GB%_pitch', 'Roll_FB%_pitch',
        
           'Roll_x FIP_pitch', 'Roll_WHIP_pitch',
        
           '3B as L', '3B as R',
           'HR as L', 'HR as R', 'Throw_L','Matchup']

use_p = ['Roll_ERA_pitch',
           
           'Roll_FB%_pitch','Roll_GB%_pitch','Roll_LOB%_pitch','Roll_K%_pitch','Roll_K-BB%_pitch',
           
           'Roll_ER_pitch','Roll_IP_pitch','Roll_SO_pitch', 'Roll_HR_pitch',
           
           'Roll_x FIP_pitch', 'Roll_WHIP_pitch',
           
           '1B as L', '1B as R','3B as L', '3B as R',
           'HR as L', 'HR as R', 'Throw_L']

df_b = df_b[use_b]
 
df_p = df_p[use_p]

################ FORECAST POINTS #####################

# unpickle my models, run them, collect the predictions

loaded_b_m = pickle.load(open('batting_model.sav','rb'))

loaded_p_m = pickle.load(open('pitching_model.sav','rb'))

pred_b = loaded_b_m.predict(df_b)

pred_p = loaded_p_m.predict(df_p)

df_b_str['Predict'] = pred_b

df_p_str['Predict'] = pred_p

# rename columns because i've been terrible at keeping track of the names and now i'm paying for it
# need to have them match up for the concat

df_b_str = df_b_str.rename(index=str, columns={"Id_bat": "Id", "Position_bat": "Position","Nickname_bat":"Nickname","FPPG_bat":"FPPG","Salary_bat":"Salary","Team_bat":"Team"})

df_p_str['Batting Order'] = 0

df_b = df_b_str.drop(labels='bats',axis=1)
df_p = df_p_str.drop(labels='throws',axis=1)

df_f = pd.concat([df_p,df_b], axis =0).reset_index(0,drop=True)

################ ENUMARATE POS & TEAM #####################

# set up position names according to my liking and combine 1B and Catcher into one field
# also, enumerate teams so that I can use it a constraint later
# i guess also enumerate position for the same reason

position = [('P','P'),('Third','3B'),('SS','SS'),('OF','OF'),('Second','2B'),('First','1B'),('C','C')]

teams = ['LAA','BAL','BOS','CWS','CLE','DET','KAN','MIN','NYY','OAK','SEA','TAM',
         'TEX','TOR','ARI','ATL','CHC','CIN','COL','MIA','HOU','LOS','MIL','WAS','NYM','PHI','PIT','STL','SDP','SFG']

for pos_name,pos in position:
    df_f[pos_name] = np.where(df_f['Position']==pos,1,0)

df_C = df_f.loc[df_f['C']==1]
df_First = df_f.loc[df_f['First']==1]

df_C['C_First'] = df_C['C']
df_First['C_First'] = df_First['First']

df_f = df_f.loc[df_f['C']==0]
df_f = df_f.loc[df_f['First']==0]
df_f['C_First'] = 0

df_f = pd.concat([df_f,df_C,df_First],axis=0).reset_index(0,drop=True)
df_f['Util'] = np.where(df_f['P']==1,0,1)

df_f = df_f.drop(labels = ['C','First'],axis=1)

for team in teams:
    df_f[team] = np.where(df_f['Team']==team,1,0)

df_f=df_f.drop(labels='Team',axis=1)

df_f.to_csv("Full_Forecast.csv",index=False, sep=',', encoding='utf-8')





