# -*- coding: utf-8 -*-
from pulp import *

import pandas as pd
import numpy as np

df_raw = pd.read_csv('Full_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)
u_this = ['Id', 'Nickname','Position','Salary','Predict']
df_fin = df_raw[u_this]
i=0
portfolio=[]
df_out = pd.DataFrame(portfolio)
df_app = pd.DataFrame()
df_f = df_raw
final_df = pd.DataFrame()
while i<10:
    
#    if i>0:
#        df_f = df_f[~df_f['Nickname'].isin(portfolio)].reset_index(0,drop=True)
        
    # Data input
    players = df_f['Nickname']
    predict = df_f['Predict']
    price = df_f['Salary']
    
    pitcher = df_f['P']
    c_first = df_f['C_First']
    second = df_f['Second']
    third = df_f['Third']
    ss = df_f['SS']
    of = df_f['OF']
    
    LAA = df_f['LAA']
    BAL = df_f['BAL']
    BOS = df_f['BOS']
    CWS = df_f['CWS']
    CLE = df_f['CLE']
    DET = df_f['DET']
    KAN = df_f['KAN']
    MIN = df_f['MIN']
    NYY = df_f['NYY']
    OAK = df_f['OAK']
    SEA = df_f['SEA']
    TAM = df_f['TAM']
    TEX = df_f['TEX']
    TOR = df_f['TOR']
    ARI = df_f['ARI']
    ATL = df_f['ATL']
    CHC = df_f['CHC']
    CIN = df_f['CIN']
    COL = df_f['COL']
    MIA = df_f['MIA']
    HOU = df_f['HOU']
    LOS = df_f['LOS']
    MIL = df_f['MIL']
    WAS = df_f['WAS']
    NYM = df_f['NYM']
    PHI = df_f['PHI']
    PIT = df_f['PIT']
    STL = df_f['STL']
    SDP = df_f['SDP']
    SFG = df_f['SFG']
    #stack = df_f[t_stack]
    
    P = range(len(players))
    
    # Declare problem instance, maximization problem
    prob = LpProblem("Portfolio", LpMaximize)
    
    # Declare decision variable x, which is 1 if a
    # player is part of the portfolio and 0 else
    x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)
    
    # Objective function -> Maximize votes
    prob += sum(predict[p] * x[p] for p in P)
    
    # Constraint definition
    prob += sum(x[p] for p in P) == 9
    
    prob += sum(pitcher[p]* x[p] for p in P) == 1
    
    prob += sum(c_first[p]* x[p] for p in P) >= 1
    prob += sum(c_first[p]* x[p] for p in P) <= 2
    
    prob += sum(second[p]* x[p] for p in P) >= 1
    prob += sum(second[p]* x[p] for p in P) <= 2
    
    prob += sum(third[p]* x[p] for p in P) >= 1
    prob += sum(third[p]* x[p] for p in P) <= 2
    
    prob += sum(ss[p]* x[p] for p in P) >= 1
    prob += sum(ss[p]* x[p] for p in P) <= 2
    
    prob += sum(of[p]* x[p] for p in P) >= 3
    prob += sum(of[p]* x[p] for p in P) <= 4
    
    #prob += sum(util[p]* x[p] for p in P) == 1
    
    prob += sum(LAA[p]* x[p] for p in P) <= 4
    prob += sum(BAL[p]* x[p] for p in P) <= 4
    prob += sum(BOS[p]* x[p] for p in P) <= 4
    prob += sum(CWS[p]* x[p] for p in P) <= 4
    prob += sum(CLE[p]* x[p] for p in P) <= 4
    prob += sum(DET[p]* x[p] for p in P) <= 4
    prob += sum(KAN[p]* x[p] for p in P) <= 4
    prob += sum(MIN[p]* x[p] for p in P) <= 4
    prob += sum(NYY[p]* x[p] for p in P) <= 4
    prob += sum(OAK[p]* x[p] for p in P) <= 4
    prob += sum(SEA[p]* x[p] for p in P) <= 4
    prob += sum(TAM[p]* x[p] for p in P) <= 4
    prob += sum(TEX[p]* x[p] for p in P) <= 4
    prob += sum(TOR[p]* x[p] for p in P) <= 4
    prob += sum(ARI[p]* x[p] for p in P) <= 4
    prob += sum(ATL[p]* x[p] for p in P) <= 4
    prob += sum(CHC[p]* x[p] for p in P) <= 4
    prob += sum(CIN[p]* x[p] for p in P) <= 4
    prob += sum(COL[p]* x[p] for p in P) <= 4
    prob += sum(MIA[p]* x[p] for p in P) <= 4
    prob += sum(HOU[p]* x[p] for p in P) <= 4
    prob += sum(LOS[p]* x[p] for p in P) <= 4
    prob += sum(MIL[p]* x[p] for p in P) <= 4
    prob += sum(WAS[p]* x[p] for p in P) <= 4
    prob += sum(NYM[p]* x[p] for p in P) <= 4
    prob += sum(PHI[p]* x[p] for p in P) <= 4
    prob += sum(PIT[p]* x[p] for p in P) <= 4
    prob += sum(STL[p]* x[p] for p in P) <= 4
    prob += sum(SDP[p]* x[p] for p in P) <= 4
    prob += sum(SFG[p]* x[p] for p in P) <= 4
    
    #prob += sum(SFG[p]* x[p] for p in P) >= 3
    
    prob += sum(price[p] * x[p] for p in P) <= 35000
    
    # Start solving the problem instance
    prob.solve()
    
    # Extract solution
    
    portfolio = [players[p] for p in P if x[p].varValue]
    score = sum([predict[p] for p in P if x[p].varValue])
    sal = sum([price[p] for p in P if x[p].varValue])
    
    pitcher = sum([pitcher[p] for p in P if x[p].varValue])
    c_first = sum([c_first[p] for p in P if x[p].varValue])
    second = sum([second[p] for p in P if x[p].varValue])
    third = sum([third[p] for p in P if x[p].varValue])
    ss = sum([ss[p] for p in P if x[p].varValue])
    of = sum([of[p] for p in P if x[p].varValue])
    
    print(portfolio)
    print(score)
    print(sal)
    print(len(portfolio))
    print(len(set(portfolio)))
    
    #df_app = pd.DataFrame(portfolio)
    ###########
    l = df_f[df_f['Nickname'].isin(portfolio)].reset_index(0,drop=True)
    """
    k = game_df.iloc[solution.xf,:][['name','Tm','Position','Salary','preds']].sort_values('preds')
    ids = asdf[['Nickname','Id']]
    l = pd.merge(k,ids,left_on = 'name',right_on = 'Nickname')
    """
    positions = ['P','C/1B','2B','3B','SS','OF','OF','OF','UTIL']
    row = [0,0,0,0,0,0,0,0,0]
    for a in range(len(l.Nickname)):
        for b in range(len(positions)): 
            if row[b] == 0 and l.Position[a] in positions[b]:
                row[b] = l.Id[a]
                break
    #utility is the Id that has not been assigned yet
    util = [y for y in l.Id if y not in row]
    row[8] = util[0]      
    z = pd.DataFrame(row)
    z = z.T
    z.columns = positions
    final_df = pd.concat([final_df,z])
    
    ##############
    #df_out = df_out.append(df_app)
    
    df_f = df_raw.sample(frac=.80,replace=False).reset_index(0,drop=True)
    
    i+=1

#df_out = pd.merge(df_out,df_raw,how='left', left_on=0,right_on='Nickname').drop(labels=0,axis=1)
#use_col = ['FPPG', 'Id', 'Nickname', 'Position']
#df_out = df_out[use_col]
#df_out.to_csv('lineups.csv',sep=',', encoding='utf-8')
final_df.to_csv('kek.csv',index=False)