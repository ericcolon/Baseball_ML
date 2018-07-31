# -*- coding: utf-8 -*-
from pulp import *

import pandas as pd
import numpy as np
"""
df_b = pd.read_csv('Batter_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)
df_p = pd.read_csv('Pitcher_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)


df_f = pd.concat([df_p,df_b], axis =0).reset_index(0,drop=True)

position = [('P','P'),('Third','3B'),('SS','SS'),('OF','OF'),('Second','2B'),('First','1B'),('C','C')]

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


df_f = df_f.drop(labels = ['Position','C','First'],axis=1)

#subset = df_f[['Nickname','Predict', 'Salary', 'P', 'Third', 'SS',
#              'OF', 'Second', 'C_First']]
"""

df_f = pd.read_csv('Full_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)

# Data input
players = df_f['Nickname']
predict = df_f['Predict']
price = df_f['Salary']
pitcher = df_f['P']
#catcher = df_f['C']
c_first = df_f['C_First']
second = df_f['Second']
third = df_f['Third']
ss = df_f['SS']
of = df_f['OF']

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
#prob += sum(catcher[p]* x[p] for p in P) == 1
prob += sum(c_first[p]* x[p] for p in P) == 1
prob += sum(second[p]* x[p] for p in P) == 1
prob += sum(third[p]* x[p] for p in P) == 1
prob += sum(ss[p]* x[p] for p in P) == 1
prob += sum(of[p]* x[p] for p in P) == 3
prob += sum(price[p] * x[p] for p in P) <= 35000

# Start solving the problem instance
prob.solve()

# Extract solution
portfolio = [players[p] for p in P if x[p].varValue]
score = sum([predict[p] for p in P if x[p].varValue])
sal = sum([price[p] for p in P if x[p].varValue])
print(portfolio)
print(score)
print(sal)