# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df_b = pd.read_csv('Batter_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)
df_p = pd.read_csv('Pitcher_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)


df_f = pd.concat([df_p,df_b], axis =0).reset_index(0,drop=True)

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

df_f.to_csv("Full_Forecast.csv", sep=',', encoding='utf-8')