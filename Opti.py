# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df_b = pd.read_csv('Batter_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)
df_p = pd.read_csv('Pitcher_Forecast.csv').drop(labels = 'Unnamed: 0', axis=1)


df_f = pd.concat([df_p,df_b], axis =0).reset_index(0,drop=True)

df_f_p = df_f.loc[(df_f['Position']=='P')].reset_index(0,drop=True)

df_f_3B = df_f.loc[(df_f['Position']=='3B')].reset_index(0,drop=True)


df_f_OF = df_f.loc[(df_f['Position']=='OF')].reset_index(0,drop=True)

df_f = pd.concat([df_f_p,df_f_3B,df_f_OF], axis=0).reset_index(0,drop=True)
#position = ['P', '3B', 'SS', 'OF', '2B', '1B', 'C']
position = [('P','P'),('Third','3B'),('SS','SS'),('OF','OF'),('Second','2B'),('First','1B'),('C','C')]

for pos_name,pos in position:
    df_f[pos_name] = np.where(df_f['Position']==pos,1,0)

df_f = df_f.drop(labels = 'Position',axis=1)

from pyeasyga import pyeasyga

subset = df_f[['Predict', 'Salary', 'P', 'Third', 'OF']]
               
#'SS','OF', 'Second', 'First', 'C']]
tuples = [tuple(x) for x in subset.values]
# setup data

data = tuples

data = [(39,18000,1,0,0),
        (34,16000,1,0,0),
        (13,10400,1,0,0),
        (34,12000,1,0,0),
        (10,2000,0,1,0),
        (13,3400,0,0,1),
        (20,2600,0,0,1),
        (13,3000,0,0,1),
        (34,4100,0,0,1),
        (6,2200,0,0,1),
        (10,2000,0,1,0),
        (13,3400,0,0,1),
        (20,2600,0,0,1),
        (4,3000,0,0,1),
        (14,2000,0,1,0),
        (13,3400,0,0,1),
        (24,2600,0,0,1),
        (6,3000,0,0,1),
        (7,2000,0,1,0),
        (20,3400,0,0,1),
        (21,2600,0,0,1),
        (19,3000,0,0,1),
        (5,2800,0,1,0),
        (17,3700,0,0,1),
        (22,2100,0,0,1),
        (3,3200,0,0,1),
        (9,2500,0,1,0),
        (11,3900,0,0,1),
        (12,2600,0,0,1),
        (7,3200,0,0,1),
        (7,2300,0,1,0),
        (8,3400,0,0,1),
        (12,2300,0,0,1),
        (10,3900,0,0,1),
        (1,2800,0,1,0),
        (4,3100,0,0,1),
        (20,2600,0,0,1),
        (13,3600,0,0,1),
        (13,3300,0,1,0),
        (18,2500,0,0,1),
        (19,2200,0,0,1),
        (20,3100,0,0,1),
]


ga = pyeasyga.GeneticAlgorithm(data,population_size=1000,generations=200,maximise_fitness=True)        # initialise the GA with data
def fitness(individual, data):
    Predict,Salary,P,Third,OF= 0,0,0,0,0
    for (selected, item) in zip(individual, data):
        if selected:
            Predict += item[0]
            Salary += item[1]
            P += item[2]
            Third += item[3]
            OF += item[4]
    
    if((Salary < 25000.0) and (P==1) and (Third==1) and (OF==3)):
        return Predict
    else: 
        Predict = 0
        return Predict
    """
    if (Salary > 25000.0 or (P!=1) or (Third!=1) or (OF!=1)):
        Predict = 0
    return Predict
    """
    
ga.fitness_function = fitness               # set the GA's fitness function
ga.run()                                    # run the GA
print(ga.best_individual())               # print the GA's best solution


#,SS,OF,Second,First,C  = 0,0,0,0,0
"""
            SS += item[3]
            OF += item[4]
            Second += item[5]
            First += item[6]
            C += item[7]
            """
            
                #if (Salary > 25000.0 or (P > 1.0 and P < 1.0) or (Third > 1.0 and Third < 1.0) or (OF>3.0 and OF<3.0 )):
     #   Predict = 0
    #return Predict