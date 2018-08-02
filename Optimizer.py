# -*- coding: utf-8 -*-
from pulp import *

import pandas as pd
import numpy as np

df_raw = pd.read_csv('Full_Forecast.csv')

# Set up for final lineup transpose

use_col = ['Id', 'Nickname','Position','Salary','Predict']
df_final = df_raw[use_col]
final_df = pd.DataFrame()

df_f = df_raw

i=0

while i<4:
    
################ DATA PREP #####################
    
    # Setting up Names, predicted values and salary constraint    
    
    players = df_f['Nickname']
    predict = df_f['Predict']
    price = df_f['Salary']
    
    # Setting up enumerated values for position
    
    pitcher = df_f['P']
    c_first = df_f['C_First']
    second = df_f['Second']
    third = df_f['Third']
    ss = df_f['SS']
    of = df_f['OF']
    
    # Setting up enumerated values for team names
    
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

################ INITIAL DECLARATION #####################

    P = range(len(players))
    
    # Declare problem instance, maximization problem
    prob = LpProblem("Portfolio", LpMaximize)
    
    # Declare decision variable x, which is 1 if a
    # player is part of the portfolio and 0 else
    
    x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)
    
    # Objective function -> Maximize predicted score
    prob += sum(predict[p] * x[p] for p in P)
    
    
################ BATTER CONSTRAINTS #####################
    
    # names of variables are self explanatory,
    # max batters 9
    # 1 pitcher, 1 catcher/first base, 1 second base, 1 third base,
    # 1 short stop, 3 of, 1 utility
    # allowing for 1 extra any position in order to accomodate utility position
    
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

################ TEAM CONSTRAINTS #####################
    
    # enumerated teams allow for control over maximum batters in lineup
    # technically, 5 is the max allowed per team but it requires having the
    # pitcher be one of the 5. 
    
    # bottom most commented out line allows for stacking based on a specific
    # team if one wishes to isolate a team based on their own forecast
    # could isolate more than one team if needed but not more than 2 
    # since it would be out of bounds of the max players allowed which is 9
    
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

################ TWO TEAM STACK CONSTRAINTS #####################
    
    #prob += sum(CHC[p]* x[p] for p in P) >= 4
    #prob += sum(COL[p]* x[p] for p in P) >= 4

################ SALARY CONSTRAINTS #####################
    
    # total of a team cannot be more than 35,000$
    
    prob += sum(price[p] * x[p] for p in P) <= 35000

################ SOLVER/EXTRACTION #####################
    
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
    
    print('\nScore:')
    print(score)
    print('Total Salary:')
    print(sal)
    print(' ')

################ WRITE TO CSV USING FD FORMAT #####################
    
    # Retrive lineup from optimization and extract attributes from initial dataframe
    # Set up position labels in FD order 
    # Check if a position is full, if not, fill with the proper id, 
    # place the remaining id in the utility spot since that spot holds any
    # position aside from pitcher
    
    l = df_final[df_final['Nickname'].isin(portfolio)].reset_index(0,drop=True)
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
    
    # Drop 25% of the data in order to get varience in your optimizations
    
    df_f = df_raw.sample(frac=.75,replace=False).reset_index(0,drop=True)
    
    i+=1

final_df.to_csv('kek.csv',index=False,sep=',',encoding='utf-8')