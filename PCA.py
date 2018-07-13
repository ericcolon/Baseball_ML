# -*- coding: utf-8 -*-

#set up PCA analysis to test for relevant features
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df_b_sorted = pd.read_csv('Baseball_Clean.csv')

pca_drop_b = ['Unnamed: 0','Date', 'Name_x', 'playerid_x', 'Target_x', 'bats',
       'Away_Team_x', 'Home_Team_x', 'Team_Short_Alt_x','Name_y',
       'playerid_y', 'Target_y', 'throws', 'Team_Short_Alt',]
"""
pca_drop_b_extra = ['Roll_w RC', 'Roll_w RAA', 'Roll_w OBA', 'Roll_wRC+',
                    'Roll_AB','Roll_GDP', 'Roll_SB', 'Roll_CS','Roll_BUH%',
                    'Roll_BB/K', 'Roll_OBP', 'Roll_SLG','Roll_HBP', 'Roll_SF',
                     'Roll_BB', 'Roll_IBB','Roll_HBP', 'Roll_SF','Roll_IFFB%',
                     'Roll_IFH%', 'Roll_SH']

pca_drop_b = pca_drop_b + pca_drop_b_extra

pca_drop_p = ['Date', 'Name', 'playerid', 'Team_x', 'Target', 'throws', 'Day',
       'Away_Team', 'Away_League', 'Season_Week_Away', 'Home_Team',
       'Time_of_Day', 'Team_Short', 'Team_Short_Alt']

pca_drop_p_extra = ['Roll_IFH%', 'Roll_BUH%','Roll_IFFB%','Roll_LD%','Roll_BABIP',
                    'Roll_BB%','Roll_K/9','Roll_IBB','Roll_FIP','Roll_GB/FB',
                    'Roll_FB%','Roll_HR/FB','Roll_GB%','Roll_HR/9','Roll_HBP',
                    'Roll_x FIP','Roll_AVG','Roll_SLG']
"""
#pca_drop_b = pca_drop_b + pca_drop_b_extra

df_b_pca = df_b_sorted.drop(labels=pca_drop_b,axis=1)

scaler = StandardScaler()
scaler.fit(df_b_pca)
scaled_data = scaler.transform(df_b_pca)

pca = PCA(n_components=4)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

df_pca = pd.DataFrame(pca.components_,columns=df_b_pca.columns)
plt.figure(figsize=(14,8))
sns.heatmap(df_pca,cmap='plasma',)
