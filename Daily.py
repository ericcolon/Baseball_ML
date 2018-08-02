# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

################ DATA LOAD #####################

df_b = pd.read_csv('Raw_to_Clean_B.csv')
df_p = pd.read_csv('Raw_to_Clean_P.csv')

df_daily = pd.read_csv('Daily_Baseball.csv')
df_park = pd.read_csv('Park_Factors_Handedness.csv')

################ DATA CLEAN #####################

# getting matchup indicators from format of csv
# first team is away team, the other is the home team

df_daily[['Away','Home']] = df_daily['Game'].str.split('@',expand=True)

df_daily_drop = ['Player ID + Player Name', 'First Name', 
                 'Last Name', 'Played',
                  'Injury Details','Game']

df_park_drop = ['Season', 'Team', 'Team_Short', 'Team_Short_Alt','Home', 'Away']

df_park = df_park.drop(labels = df_park_drop,axis=1)
df_daily = df_daily.drop(labels = df_daily_drop,axis=1)

# checking for injury, DTD, or suspensions
# have to adjust this because it only works if they all exist in some respect,
# probably because of the and statement i did

df_daily = df_daily.loc[(df_daily['Injury Indicator']!='DTD')&(df_daily['Injury Indicator']!='DL')&(df_daily['Injury Indicator']!='NA')].reset_index(0,drop=True)

# getting the pitchers based on their specific column

df_daily_pitch = df_daily.loc[df_daily['Probable Pitcher']=='Yes'].reset_index(0,drop=True)
df_daily_pitch = df_daily_pitch.drop(labels = ['Injury Indicator','Batting Order'],axis=1)

# getting batters, limitation is that its all batters no matter if they're starters or not
# will eventually have it filter on batters that are listed under batting order but 
# that depends on the batter order being released ahead of time which it isn't always

df_daily_batter = df_daily.loc[df_daily['Position']!='P'].reset_index(0,drop=True)
df_daily_batter = df_daily_batter.drop(labels = ['Probable Pitcher','Injury Indicator'],axis=1)

################ DATA MERGE #####################

# get latest attributes from historic data

df_b = df_b.sort_values('Date').groupby('Name_bat').tail(1)

df_p = df_p.sort_values('Date').groupby('Name').tail(1)

# these are the stats that match up with forecasting stats. 
# i'm defining these lists in two places so i might consolidate some of this with
# the forestcast model

use_col_b = ['Date', 'Name_bat', 'bats',
                 
               'Target_bat', 
            
               'Roll_PA_bat','Roll_HR_bat','Roll_GB/FB_bat',
               
               'Roll_BB%_bat', 'Roll_K%_bat','Roll_GB%_bat', 'Roll_FB%_bat', 
               
               'Roll_ISO_bat','Roll_wRC+_bat']


use_col_p = ['Date', 'Name', 'throws', 
             
               'Target', 'Roll_ERA_pitch',
               
               'Roll_FB%_pitch','Roll_GB%_pitch','Roll_LOB%_pitch','Roll_K%_pitch','Roll_K-BB%_pitch',
               
               'Roll_ER_pitch','Roll_IP_pitch','Roll_SO_pitch', 'Roll_HR_pitch',
               
               'Roll_x FIP_pitch', 'Roll_WHIP_pitch']

df_b = df_b[use_col_b]

df_p = df_p[use_col_p]

# merge between daily batter and historic batter and then again for daily pitcher and historic pitcher and then
# then batters and pitchers

df_b = pd.merge(df_daily_batter,df_b,how= 'inner',left_on='Nickname', right_on ='Name_bat',suffixes = ('_daily','_bat'))
df_p = pd.merge(df_daily_pitch,df_p,how= 'left',left_on='Nickname', right_on ='Name',suffixes = ('_daily','_pitch'))

df_b = pd.merge(df_b, df_p, how = 'inner', left_on = 'Opponent', right_on = 'Team',suffixes = ('_bat','_pitch')).reset_index(0,drop=True)

b_drop = ['Date_bat','Date_pitch','Name_bat','Id_pitch','Position_pitch','Nickname_pitch','FPPG_pitch','Salary_pitch','Team_pitch',
          'Opponent_pitch','Probable Pitcher','Away_pitch','Home_pitch','Date_pitch','Name']

p_drop  = ['Opponent','Probable Pitcher','Away','Date', 'Name',]

# dropping a lot of garbage variables that i should really stop importing

df_b=df_b.drop(labels=b_drop,axis=1)

df_p=df_p.drop(labels=p_drop,axis=1)

################ PITCHER REPLACE #####################

# trying to replace pitchers that are either new or don't have enough data to classify under the historic 
# threshold. Thinking about more sophisticated imputing but doing averages of the rest works for now

df_p['throws'] = df_p['throws'].fillna(value='R')
df_b['throws'] = df_b['throws'].fillna(value='R')

replace = ['Roll_ERA_pitch',
               
            'Roll_FB%_pitch','Roll_GB%_pitch','Roll_LOB%_pitch','Roll_K%_pitch','Roll_K-BB%_pitch',
               
            'Roll_ER_pitch','Roll_IP_pitch','Roll_SO_pitch', 'Roll_HR_pitch',
               
            'Roll_x FIP_pitch', 'Roll_WHIP_pitch']

df_p[replace] = df_p[replace].apply(lambda x: x.fillna(x.mean()),axis=0)
df_b[replace] = df_b[replace].apply(lambda x: x.fillna(x.mean()),axis=0)


df_b = pd.merge(df_b,df_park,left_on = 'Home_bat',right_on = 'Team_FD')

df_p = pd.merge(df_p,df_park,left_on = 'Home',right_on = 'Team_FD')

df_b = df_b.drop(labels = ['Home_bat','Away_bat','Opponent_bat','Target_bat','Target'],axis=1)

df_p = df_p.drop(labels = ['Home','Target'],axis=1)

# setting up handedness metrics again

df_b['Bats_L'] = np.where(df_b['bats']=='L',1,0)
df_b['Bats_R'] = np.where(df_b['bats']=='R',1,0)

df_b['Throw_L'] = np.where(df_b['throws']=='L',1,0)
df_b['Throw_R'] = np.where(df_b['throws']=='R',1,0)

df_p['Throw_L'] = np.where(df_p['throws']=='L',1,0)
df_p['Throw_R'] = np.where(df_p['throws']=='R',1,0)

df_b['Matchup'] = np.where(((df_b['Throw_L']==1) & (df_b['Bats_R']==1))|((df_b['Throw_R']==1) & (df_b['Bats_L']==1)),1,0)

df_b.to_csv("Daily_Raw_to_Clean_B.csv", index = False, sep=',', encoding='utf-8')
df_p.to_csv("Daily_Raw_to_Clean_P.csv", index = False, sep=',', encoding='utf-8')