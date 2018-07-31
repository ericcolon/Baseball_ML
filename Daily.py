# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

df_hist = pd.read_csv('Baseball_Clean.csv')
df_daily = pd.read_csv('Daily_Baseball.csv')
df_b = pd.read_csv('Baseball_Clean.csv')
df_p = pd.read_csv('Baseball_Clean_P.csv')
df_park = pd.read_csv('Park_Factors_Handedness.csv')


df_daily[['Away','Home']] = df_daily['Game'].str.split('@',expand=True)

df_daily_drop = ['Player ID + Player Name', 'First Name', 
                 'Last Name', 'Played',
                  'Injury Details','Game'
                 ]

df_park_drop = ['Season', 'Team', 'Team_Short', 'Team_Short_Alt','1B as L',
                '1B as R', '2B as L', '2B as R','Home', 'Away']

df_park = df_park.drop(labels = df_park_drop,axis=1)
df_daily = df_daily.drop(labels = df_daily_drop,axis=1)
df_daily = df_daily.loc[(df_daily['Injury Indicator']!='DTD')&(df_daily['Injury Indicator']!='DL')].reset_index(0,drop=True)
#df_daily = df_daily.loc[df_daily['Injury Indicator']!='DL'].reset_index(0,drop=True)

df_daily_pitch = df_daily.loc[df_daily['Probable Pitcher']=='Yes'].reset_index(0,drop=True)
df_daily_pitch = df_daily_pitch.drop(labels = ['Injury Indicator','Batting Order'],axis=1)

df_daily_batter = df_daily.loc[df_daily['Position']!='P'].reset_index(0,drop=True)
df_daily_batter = df_daily_batter.drop(labels = ['Probable Pitcher','Injury Indicator'],axis=1)

df_b = df_b.sort_values('Date').groupby('Name_x').tail(1)

df_p = df_p.sort_values('Date').groupby('Name').tail(1)

use_col_b = ['Date', 'Name_x', 'bats',
       
       'Roll_PA', 'Roll_H_x', 'Roll_1B', 'Roll_2B_x', 'Roll_3B_x',
       'Roll_HR_x', 'Roll_R_x', 'Roll_RBI', 'Roll_AVG', 'Roll_GB%',
       'Roll_Target_x',
       
       ]

use_col_p = ['Date', 'Name', 'throws', 
       
       'Roll_IP', 'Roll_ER', 'Roll_HR','Roll_OBP', 'Roll_ERA', 'Roll_BB/9',
       'Roll_K/BB', 'Roll_K%', 'Roll_K-BB%', 'Roll_WHIP', 'Roll_LOB%',
       'Roll_GB/FB', 'Roll_FB%','Roll_SO']

df_b = df_b[use_col_b]

df_p = df_p[use_col_p]

df_b = pd.merge(df_daily_batter,df_b,how= 'inner',left_on='Nickname', right_on ='Name_x')
df_p = pd.merge(df_daily_pitch,df_p,how= 'left',left_on='Nickname', right_on ='Name')

df_b = pd.merge(df_b,df_p,how = 'inner', left_on = 'Opponent', right_on = 'Team').reset_index(0,drop=True)

b_drop  = ['Date_x','Date_y','Name_x','Id_y','Position_y','Nickname_y','FPPG_y','Salary_y',
           'Team_y','Opponent_y','Probable Pitcher','Away_y','Home_y','Date_y','Name',]

p_drop  = ['Opponent','Probable Pitcher','Away','Date', 'Name',]

df_b=df_b.drop(labels=b_drop,axis=1)

df_p=df_p.drop(labels=p_drop,axis=1)

df_p['throws'] = df_p['throws'].fillna(value='R')

replace = ['Roll_IP', 'Roll_ER', 'Roll_HR', 'Roll_OBP', 'Roll_ERA',
       'Roll_BB/9', 'Roll_K/BB', 'Roll_K%', 'Roll_K-BB%', 'Roll_WHIP',
       'Roll_LOB%', 'Roll_GB/FB', 'Roll_FB%', 'Roll_SO']

df_p[replace] = df_p[replace].apply(lambda x: x.fillna(x.mean()),axis=0)

df_b = pd.merge(df_b,df_park,left_on = 'Home_x',right_on = 'Team_FD')

df_p = pd.merge(df_p,df_park,left_on = 'Home',right_on = 'Team_FD')

df_b = df_b.drop(labels = ['Home_x','Away_x','Opponent_x'],axis=1)

df_b['Bats_L'] = np.where(df_b['bats']=='L',1,0)
df_b['Bats_R'] = np.where(df_b['bats']=='R',1,0)
df_b['Bats_S'] = np.where(df_b['bats']=='S',1,0)

df_b['Throw_L'] = np.where(df_b['throws']=='L',1,0)
df_b['Throw_R'] = np.where(df_b['throws']=='R',1,0)
df_b['Throw_S'] = np.where(df_b['throws']=='S',1,0)

df_p['Throw_L'] = np.where(df_p['throws']=='L',1,0)
df_p['Throw_R'] = np.where(df_p['throws']=='R',1,0)
df_p['Throw_S'] = np.where(df_p['throws']=='S',1,0)

df_b['Matchup'] = np.where(((df_b['Throw_L']==1) & (df_b['Bats_R']==1))|((df_b['Throw_R']==1) & (df_b['Bats_L']==1)),1,0)

df_b.to_csv("Baseball_B_Pred.csv", sep=',', encoding='utf-8')
df_p.to_csv("Baseball_P_Pred.csv", sep=',', encoding='utf-8')