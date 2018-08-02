import numpy as np
import pandas as pd

################ DATA LOAD #####################

# compiled from online sources such as fangraphs based on home away splits
# straight forward named CSV's in terms of what is being loaded
# data is as granualar as i could find online

df_p_home = pd.read_csv('2015_2018_Pitchers_Home.csv')
df_p_away = pd.read_csv('2015_2018_Pitchers_Away.csv')

df_b_home = pd.read_csv('2015_2018_Batter_Home.csv')
df_b_away = pd.read_csv('2015_2018_Batter_Away.csv')

df_s = pd.read_csv('2015_2018_Schedule.txt')
df_h = pd.read_csv('Handedness.csv')
df_park = pd.read_csv('Park_Factors_Handedness.csv')

# rolling days control rolling average window for batters/pitchers during
# stat aggregation, PA thresh controls minimum PA allowed for batters, 
# IP thresh controls minimum IP allowed for pitchers

rolling_days = 7
PA_thresh = 3
IP_thresh = 3.5

# set home and away flag for later stadium joins

df_p_home['Home'] = 1
df_p_away['Home'] = 0

df_b_home['Home'] = 1
df_b_away['Home'] = 0

# concat as one df 

df_p=pd.concat([df_p_home,df_p_away])
df_b=pd.concat([df_b_home,df_b_away])

# change datatype for later merge since ids seem to be a combination of numbers and letters
# would like them to be just numeric but that isn't something i control so str will do

df_p[['playerid','Team']] = df_p[['playerid','Team']].astype('str')
df_b[['playerid','Team']] = df_b[['playerid','Team']].astype('str')
df_h['playerid'] = df_h['playerid'].astype('str')
df_park['Team_Short'] = df_park['Team_Short'].astype('str')

# set target value based on scoring system of choice, i'm using FanDuel so 
# their scoring system exists online. I'm missing won game for pitchers but 
# I can live without it since at least i get the addition of quality start

df_p['Q_Start'] = np.where(df_p['ER']<4,1,0)
df_b['Target'] = df_b['1B']*3 + df_b['2B']*6 + df_b['3B']*9 + df_b['HR']*12 +df_b['RBI']*3.5 + df_b['R']*3.2 + df_b['BB']*3 +df_b['SB']*6 + df_b['HBP']*3
df_p['Target'] = df_p['ER']*(-3) + df_p['SO']*3 + df_p['IP']*3 + df_p['Q_Start']*(4)

df_p = df_p.drop(labels='Q_Start',axis=1).reset_index(0,drop=True)

# inner merge on player id in order to give players handedness attribute

df_b=pd.merge(df_b,df_h,how='inner',on='playerid')
df_p=pd.merge(df_p,df_h,how='inner',on='playerid')

# convert to date for later date grouping and rolling average calculations

df_b['Date']=pd.to_datetime(df_b['Date'], format='%m/%d/%Y', errors='ignore')
df_p['Date']=pd.to_datetime(df_p['Date'], format='%m/%d/%Y', errors='ignore')
df_s['Date']=pd.to_datetime(df_s['Date'], format='%Y%m%d', errors='ignore')

#merges for schedule followed by merges based on home and away followed by some minor cleaning in order to align concat

df_home=pd.merge(df_s,df_park,how='inner',left_on=['Home_Team'],right_on=['Team_Short_Alt'])
df_away=pd.merge(df_s,df_park,how='inner',left_on=['Away_Team'],right_on=['Team_Short_Alt'])

df_b_l = pd.merge(df_b,df_home,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Home'])
df_b_r = pd.merge(df_b,df_away,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Away'])

df_p_l = pd.merge(df_p,df_home,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Home'])
df_p_r = pd.merge(df_p,df_away,how='inner',left_on=['Date','Team','Home'],right_on=['Date','Team_Short','Away'])

df_b_l=df_b_l.drop(labels=['Team_y','Away','Home'],axis=1)
df_b_r=df_b_r.drop(labels=['Team_y','Home_x','Home_y','Away'],axis=1)

df_p_l=df_p_l.drop(labels=['Team_y','Away','Home'],axis=1)
df_p_r=df_p_r.drop(labels=['Team_y','Home_x','Home_y','Away'],axis=1)

# drop stats which are trivial or of no interest by me, I've done some PCA on the side and i figured that these
# stats don't have any use to me. You could dig into birth_year and some of the directional hit percentages and
# create a feature out of that but I don't think it would add as much as one would expect

batter_drop = ['throws', 'birth_year', 'fg_name', 'fg_pos','mlb_team', 'mlb_team_long','AVG1','Delays','Delay_Date','Season',
               'Home_League','Season_Week_Home','Pull%', 'Cent%', 'Oppo%', 'Soft%','Med%', 'Hard%','Num_Of_Game']

pitcher_drop = ['bats', 'birth_year', 'fg_name', 'fg_pos','mlb_team', 'mlb_team_long','AVG1','Delays','Delay_Date','Season',
                'Home_League','Season_Week_Home','Pull%', 'Cent%', 'Oppo%', 'Soft%','Med%', 'Hard%','Num_Of_Game']

df_b = pd.concat([df_b_l,df_b_r]).drop(labels=batter_drop,axis=1).sort_values(['Name','Date'])
df_p = pd.concat([df_p_l,df_p_r]).drop(labels=pitcher_drop,axis=1).sort_values(['Name','Date'])

# set minimum threshold for batters and pitchers in order to eliminate one off's and in order to consolidate instances
# outside of relief pitchers and just focus on starters

df_b = df_b.loc[df_b['PA']>=PA_thresh].reset_index(0,drop=True)
df_p = df_p.loc[df_p['IP']>IP_thresh].reset_index(0,drop=True)

r_app_b = ['PA', 'AB', 'H', '1B', '2B',
           '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP',
           'SB', 'CS', 'AVG', 'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO',
           'BABIP', 'w RC', 'w RAA', 'w OBA', 'wRC+', 'GB/FB', 'LD%', 'GB%', 'FB%',
           'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'Target']

r_app_p = [ 'TBF', 'H', '2B', '3B', 'R','IP',
           'ER', 'HR', 'BB', 'IBB', 'HBP', 'SO', 'AVG', 'OBP', 'SLG', 'ERA',
           'w OBA', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'K%', 'BB%', 'K-BB%', 'WHIP',
           'BABIP', 'LOB%', 'x FIP', 'FIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%',
           'HR/FB', 'IFH%', 'BUH%','Target']

# do a rolling average for batter and pitcher stats. Window can be adjust above.

for stat in r_app_b:
    roll = 'Roll_'+stat+'_bat'
    df_b[roll] = df_b.groupby('Name')[stat].rolling(rolling_days).mean().shift(1).reset_index(0,drop=True)

for stat in r_app_p:
    roll = 'Roll_'+stat+'_pitch'
    df_p[roll] = df_p.groupby('Name')[stat].rolling(rolling_days).mean().shift(1).reset_index(0,drop=True)

# make sure not to remove the target after doing rolling average, this is more of a personal note

df_b = df_b.drop(labels=r_app_b[:-1],axis=1).dropna().reset_index(0,drop = True)
df_p = df_p.drop(labels=r_app_p[:-1],axis=1).dropna().reset_index(0,drop = True)

park_drop = ['Season','Team_Short','Team','Home','Away']

park_drop_p = ['Season','Team_Short','Team']

s_drop = ['G','Day','Away_League', 'Season_Week_Away','Time_of_Day','1B as L', 'Team_FD',
          '1B as R','2B as L', '2B as R', '3B as L', '3B as R', 'HR as L','HR as R'
          ]

# at this point we get pretty messy since i'm not the best at pandas at the moment so I do things logically
# without that much knowledge into more abstract manipulation that pandas allows

df_park_p = df_park.drop(labels = park_drop_p,axis=1)
df_park = df_park.drop(labels = park_drop,axis=1)

df_b = df_b.drop(labels = s_drop,axis=1)
df_p = df_p.drop(labels = s_drop[1:],axis=1)

df_b= pd.merge(df_b,df_park,how='inner',left_on=['Home_Team'],right_on=['Team_Short_Alt'],suffixes=('_bat','_park'))

df_b['Home_bat'] = np.where(df_b['Team_Short_Alt_bat']==df_b['Home_Team'],1,0)
df_b['Away_bat'] = np.where(df_b['Team_Short_Alt_bat']==df_b['Away_Team'],1,0)

df_p['Home_pitch'] = np.where(df_p['Team_Short_Alt']==df_p['Home_Team'],1,0)
df_p['Away_pitch'] = np.where(df_p['Team_Short_Alt']==df_p['Away_Team'],1,0)

################ PITCHER MODEL SETUP #####################

df_p_h = pd.merge(df_p,df_park_p,how='inner',left_on=['Team_Short_Alt','Home_pitch'],right_on=['Team_Short_Alt','Home'])
df_p_a = pd.merge(df_p,df_park_p,how='inner',left_on=['Team_Short_Alt','Away_pitch'],right_on=['Team_Short_Alt','Away'])

df_f_p = pd.concat([df_p_h,df_p_a])

df_f_p['Throw_L'] = np.where(df_f_p['throws']=='L',1,0)
df_f_p['Throw_R'] = np.where(df_f_p['throws']=='R',1,0)

df_f_p_drop = ['Away_Team','Home_Team', 'Team_Short', 'Team_Short_Alt','Home_pitch','Away_pitch', 'Home', 'Away',
               'playerid']

df_f_p = df_f_p.drop(labels = df_f_p_drop,axis=1)

df_f_p = df_f_p.loc[df_f_p['Target']>0].reset_index(0,drop=True)

df_f_p.to_csv("Raw_to_Clean_P.csv", index = False, sep=',', encoding='utf-8')

################ BATTER MODEL SETUP #####################

df_b_h = pd.merge(df_b,df_p,how='inner',left_on=['Date','Team_Short_Alt_bat','Home_bat'],right_on=['Date','Home_Team','Away_pitch'],suffixes=('_bat','_pitch'))
df_b_a = pd.merge(df_b,df_p,how='inner',left_on=['Date','Team_Short_Alt_bat','Away_bat'],right_on=['Date','Away_Team','Home_pitch'],suffixes=('_bat','_pitch'))

df_f_b = pd.concat([df_b_h,df_b_a])

df_f_b['Bats_L'] = np.where(df_f_b['bats']=='L',1,0)
df_f_b['Bats_R'] = np.where(df_f_b['bats']=='R',1,0)

df_f_b['Throw_L'] = np.where(df_f_b['throws']=='L',1,0)
df_f_b['Throw_R'] = np.where(df_f_b['throws']=='R',1,0)

# create a matchup stat that looks at opposite handedness between batter and pitcher

df_f_b['Matchup'] = np.where(((df_f_b['Throw_L']==1) & (df_f_b['Bats_R']==1))|((df_f_b['Throw_R']==1) & (df_f_b['Bats_L']==1)),1,0)

df_f_b_drop = ['playerid_bat','Home_pitch','Away_pitch','Away_Team_pitch','Home_Team_pitch','Team_Short_pitch','Team_Short_Alt','throws',
             'playerid_pitch','Team_x_pitch','Name_pitch','Home_bat','Away_bat','Team_Short_Alt_park','Target_pitch','Team_Short_Alt_bat',
             'Team_Short_bat','Home_Team_bat','Away_Team_bat']

df_f_b = df_f_b.drop(labels = df_f_b_drop,axis=1)

df_f_b.to_csv("Raw_to_Clean_B.csv", index = False, sep=',', encoding='utf-8')








