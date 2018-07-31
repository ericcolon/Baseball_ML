# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#read in csv data
df_p_home = pd.read_csv('2015_2018_Pitchers_Home.csv')
df_p_away = pd.read_csv('2015_2018_Pitchers_Away.csv')

df_b_home = pd.read_csv('2015_2018_Batter_Home.csv')
df_b_away = pd.read_csv('2015_2018_Batter_Away.csv')

df_s = pd.read_csv('2015_2018_Schedule.txt')
df_h = pd.read_csv('Handedness.csv')
df_park = pd.read_csv('Park_Factors_Handedness.csv')

#set home and away marker for later joins
df_p_home['Home'] = 1
df_p_away['Home'] = 0

df_b_home['Home'] = 1
df_b_away['Home'] = 0

#concat as one df
df_p=pd.concat([df_p_home,df_p_away])
df_b=pd.concat([df_b_home,df_b_away])

#change datatype for later merge
df_p[['playerid','Team']] = df_p[['playerid','Team']].astype('str')
df_b[['playerid','Team']] = df_b[['playerid','Team']].astype('str')
df_h['playerid'] = df_h['playerid'].astype('str')
df_park['Team_Short'] = df_park['Team_Short'].astype('str')

#set target value based on scoring system
df_b['Target'] = df_b['1B']*3 + df_b['2B']*6 + df_b['3B']*9 + df_b['HR']*12 +df_b['RBI']*3.5 + df_b['R']*3.2 + df_b['BB']*3 +df_b['SB']*6 + df_b['HBP']*3
df_p['Target'] = df_p['ER']*(-3) + df_p['SO']*3 + df_p['IP']*3

#inner merge on player id in order to give players handedness attribute
df_b=pd.merge(df_b,df_h,how='inner',on='playerid')
df_p=pd.merge(df_p,df_h,how='inner',on='playerid')

#convert to date 
df_b['Date']=pd.to_datetime(df_b['Date'], format='%m/%d/%Y', errors='ignore')
df_p['Date']=pd.to_datetime(df_p['Date'], format='%m/%d/%Y', errors='ignore')
df_s['Date']=pd.to_datetime(df_s['Date'], format='%Y%m%d', errors='ignore')

#merges for schedule followed by merges based on home and away followed by some minor cleaning in order to align concat
df_s_home=pd.merge(df_s,df_park,how='inner',left_on=['Home_Team'],right_on=['Team_Short_Alt'])
df_s_away=pd.merge(df_s,df_park,how='inner',left_on=['Away_Team'],right_on=['Team_Short_Alt'])

df_left_b = pd.merge(df_b,df_s_home,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Home'])
df_right_b = pd.merge(df_b,df_s_away,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Away'])

df_left_p = pd.merge(df_p,df_s_home,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Home'])
df_right_p = pd.merge(df_p,df_s_away,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Away'])

df_left_b=df_left_b.drop(labels=['Team_y','Away','Home'],axis=1)
df_right_b=df_right_b.drop(labels=['Team_y','Home_x','Home_y','Away'],axis=1)

df_left_p=df_left_p.drop(labels=['Team_y','Away','Home'],axis=1)
df_right_p=df_right_p.drop(labels=['Team_y','Home_x','Home_y','Away'],axis=1)

df_b_clean = pd.concat([df_left_b,df_right_b])
df_p_clean = pd.concat([df_left_p,df_right_p])


#batter clean out pointless columns
batter_drop = ['throws', 'birth_year', 'fg_name', 'fg_pos','mlb_team', 'mlb_team_long','AVG1','Delays','Delay_Date','Season',
               'Home_League','Season_Week_Home','Pull%', 'Cent%', 'Oppo%', 'Soft%','Med%', 'Hard%','Num_Of_Game']
df_b = df_b_clean.drop(labels=batter_drop,axis=1)

#pitcher clean out pointless columns
pitcher_drop = ['bats', 'birth_year', 'fg_name', 'fg_pos','mlb_team', 'mlb_team_long','AVG1','Delays','Delay_Date','Season',
                'Home_League','Season_Week_Home','Pull%', 'Cent%', 'Oppo%', 'Soft%','Med%', 'Hard%','Num_Of_Game']
df_p = df_p_clean.drop(labels=pitcher_drop,axis=1)

df_b_sorted = df_b.sort_values(['Name','Date'])
df_p_sorted = df_p.sort_values(['Name','Date'])

df_b_sorted = df_b_sorted.loc[df_b_sorted['PA']>=3].reset_index(0,drop=True)
df_p_sorted = df_p_sorted.loc[df_p_sorted['IP']>3.5].reset_index(0,drop=True)


#df_b_sorted = df_b_sorted.loc[df_b_sorted['Target']>=10].reset_index(0,drop=True)
#df_p_sorted = df_p_sorted.loc[df_p_sorted['Target']>3.5].reset_index(0,drop=True)

roll_apply_b = ['PA', 'AB', 'H', '1B', '2B',
       '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP',
       'SB', 'CS', 'AVG', 'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO',
       'BABIP', 'w RC', 'w RAA', 'w OBA', 'wRC+', 'GB/FB', 'LD%', 'GB%', 'FB%',
       'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'Target']

roll_apply_p = [ 'TBF', 'H', '2B', '3B', 'R','IP',
       'ER', 'HR', 'BB', 'IBB', 'HBP', 'SO', 'AVG', 'OBP', 'SLG', 'ERA',
       'w OBA', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'K%', 'BB%', 'K-BB%', 'WHIP',
       'BABIP', 'LOB%', 'x FIP', 'FIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%',
       'HR/FB', 'IFH%', 'BUH%','Target']
#df_b_sorted['HR_Target'] = np.where(df_b_sorted['Target']>=15,'High','Low')
#apply roll to all columns, then drop nan rows so that we're left with usable data based on a 15 instance sample. 
#will have to look into setting it up so that the data processes by taking into account the year but for my purposes this suffices
for stat in roll_apply_b:
    roll = 'Roll_'+stat
    df_b_sorted[roll] = df_b_sorted.groupby('Name')[stat].rolling(5).mean().shift(1).reset_index(0,drop=True)

for stat in roll_apply_p:
    roll = 'Roll_'+stat
    df_p_sorted[roll] = df_p_sorted.groupby('Name')[stat].rolling(5).mean().shift(1).reset_index(0,drop=True)


roll_apply_b = ['PA', 'AB', 'H', '1B', '2B',
       '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP',
       'SB', 'CS', 'AVG', 'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO',
       'BABIP', 'w RC', 'w RAA', 'w OBA', 'wRC+', 'GB/FB', 'LD%', 'GB%', 'FB%',
       'IFFB%', 'HR/FB', 'IFH%', 'BUH%']

roll_apply_p = [ 'TBF', 'H', '2B', '3B', 'R','IP',
       'ER', 'HR', 'BB', 'IBB', 'HBP', 'SO', 'AVG', 'OBP', 'SLG', 'ERA',
       'w OBA', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'K%', 'BB%', 'K-BB%', 'WHIP',
       'BABIP', 'LOB%', 'x FIP', 'FIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%',
       'HR/FB', 'IFH%', 'BUH%']
    
df_b_sorted = df_b_sorted.drop(labels=roll_apply_b,axis=1).dropna().reset_index(0,drop = True)
df_p_sorted = df_p_sorted.drop(labels=roll_apply_p,axis=1).dropna().reset_index(0,drop = True)



#gonna get messy here for the final merge

final_merge_park = ['Season','Team_Short','Team','Home','Away']

final_merge_b = [ 'G','Day','Away_League', 'Season_Week_Away','Time_of_Day', 
                 '1B as L', '1B as R','2B as L', '2B as R', '3B as L', '3B as R', 'HR as L',
                 'HR as R','Roll_w RC', 'Roll_w RAA', 'Roll_w OBA', 'Roll_wRC+',
                 'Roll_AB','Roll_GDP', 'Roll_SB', 'Roll_CS','Roll_BUH%',
                 'Roll_BB/K', 'Roll_OBP', 'Roll_SLG','Roll_HBP', 'Roll_SF',
                 'Roll_BB', 'Roll_IBB','Roll_HBP', 'Roll_SF','Roll_IFFB%',
                 'Roll_IFH%', 'Roll_SH','Team_x']

final_merge_p = [ 'Day','Away_League','Season_Week_Away','Time_of_Day',
                 '1B as L', '1B as R','2B as L', '2B as R', '3B as L', '3B as R', 'HR as L',
                 'HR as R','Roll_IFH%', 'Roll_BUH%','Roll_IFFB%','Roll_LD%','Roll_BABIP',
                 'Roll_BB%','Roll_K/9','Roll_IBB','Roll_FIP','Roll_GB/FB',
                 'Roll_FB%','Roll_HR/FB','Roll_GB%','Roll_HR/9','Roll_HBP',
                 'Roll_x FIP','Roll_AVG','Roll_SLG','Team_x']

df_park = df_park.drop(labels = final_merge_park,axis=1)
df_b_sorted = df_b_sorted.drop(labels = final_merge_b,axis = 1)
df_p_sorted = df_p_sorted.drop(labels = final_merge_p,axis = 1)

df_b_sorted = pd.merge(df_b_sorted,df_park,how='inner',left_on=['Home_Team'],right_on=['Team_Short_Alt'])

df_b_sorted['Home_B'] = df_b_sorted.eval("Team_Short_Alt_x == Home_Team")
df_b_sorted['Away_B'] = df_b_sorted.eval("Team_Short_Alt_x == Away_Team")
df_p_sorted['Home_P'] = df_p_sorted.eval("Team_Short_Alt == Home_Team")
df_p_sorted['Away_P'] = df_p_sorted.eval("Team_Short_Alt == Away_Team")

df_b_home= pd.merge(df_b_sorted,df_p_sorted,how='inner',left_on=['Date','Team_Short_Alt_x','Home_B'],right_on=['Date','Home_Team','Away_P'])
df_b_away= pd.merge(df_b_sorted,df_p_sorted,how='inner',left_on=['Date','Team_Short_Alt_x','Away_B'],right_on=['Date','Away_Team','Home_P'])

#df_b_home = df_b_home.loc[df_b_home['Roll_IP'] > 2.5].reset_index(0,drop=True)
#df_b_away = df_b_away.loc[df_b_away['Roll_IP'] > 2.5].reset_index(0,drop=True)

df_f = pd.concat([df_b_home,df_b_away])

clean = ['Team_Short_x','Team_Short_Alt_y','Away_Team_y', 'Home_Team_y', 'Team_Short_y','Home_B','Away_B','Home_P','Away_P']

df_f = df_f.drop(labels = clean,axis=1)

df_f['Bats_L'] = np.where(df_f['bats']=='L',1,0)
df_f['Bats_R'] = np.where(df_f['bats']=='R',1,0)
df_f['Bats_S'] = np.where(df_f['bats']=='S',1,0)

df_f['Throw_L'] = np.where(df_f['throws']=='L',1,0)
df_f['Throw_R'] = np.where(df_f['throws']=='R',1,0)
df_f['Throw_S'] = np.where(df_f['throws']=='S',1,0)

df_f['Matchup'] = np.where(((df_f['Throw_L']==1) & (df_f['Bats_R']==1))|((df_f['Throw_R']==1) & (df_f['Bats_L']==1)),1,0)

df_f.to_csv("Baseball_Clean.csv", sep=',', encoding='utf-8')
