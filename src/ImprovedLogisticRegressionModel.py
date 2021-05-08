# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:48:26 2020

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset = pd.read_csv("C:/Users/Admin/Desktop/Degree LVL 2/AIM - AI method/Assignment/dataset/dataset.csv")

# Data pre-processing ========================================================

# Remove matches that end withins 25 minutes (Not valid)
dataset = dataset.drop(dataset[dataset['duration'] < 1500].index)
# Remove matches that have lower than 4k average mmr  (Only want high skilled game)
dataset = dataset.drop(dataset[dataset['avg_mmr'] < 4000].index)
# Remove matches that lobby type is practice
dataset = dataset.drop(dataset[dataset["lobby_type"] == 1].index)
# Remove game mode unnknown
dataset = dataset.drop(dataset[dataset["game_mode"] == 0].index)

dataset = dataset.dropna()
# Reset index after dropping rows
dataset.reset_index(drop=True, inplace=True)


# heron winrate from dotabuff.com website 
# consists of hero id and hero winrate
hero_winrate = {
    56 : 56.36,
    108 : 56.02,
    67 : 55.12,
    22 : 54.06,
    84 : 53.50,
    113 : 53.06,
    55 : 52.79,
    4 : 52.68,
    85 : 52.61,
    37 : 52.60,
    61 : 52.58,
    57 : 52.51,
    60 : 52.34,
    9 : 52.21,
    42 : 52.12,
    29 : 52.11,
    36 : 52.10,
    28 : 52.00,
    81 : 52.00,
    20 : 51.93,
    5 : 51.84,
    51 : 51.78,
    83 : 51.77,
    68 : 51.67,
    18 : 51.61,
    102 : 51.49,
    1 : 51.34,
    33 : 51.31,
    21 : 51.31,
    32 : 51.18,
    31 : 51.01,
    8 : 50.99,
    6 : 50.90,
    64 : 50.84,
    95 : 50.83,
    71 : 50.80,
    101 : 50.79,
    3 : 50.72,
    75 : 50.69,
    52 : 50.68,
    94 : 50.67,
    41 : 50.58,
    80 : 50.53,
    2 : 50.51,
    110 : 50.39,
    103 : 50.30,
    105 : 50.29,
    104 : 50.26,
    92 : 50.25,
    48 : 50.22,
    50 : 50.17,
    88 : 50.17,
    30 : 50.16,
    27 : 50.15,
    62 : 50.02,
    49 : 49.94,
    14 : 49.90,
    119 : 49.87,
    126 : 49.76,
    96 : 49.74,
    16 : 49.73,
    111 : 49.69,
    87 : 49.65,
    74 : 49.31,
    73 : 49.31,
    45 : 49.29,
    89 : 49.28,
    63 : 49.27,
    93 : 49.24,
    40 : 49.23,
    44 : 49.23,
    47 : 49.14,
    129 : 49.10,
    11 : 49.06,
    54 : 49.01,
    99 : 48.96,
    107 : 48.81,
    7 : 48.80,
    26 : 48.78,
    97 : 48.75,
    82 : 48.66,
    15 : 48.63,
    78 : 48.60,
    100 : 48.56,
    17 : 48.50,
    114 : 48.42,
    70 : 48.23,
    128 : 48.21,
    10 : 48.06,
    66 : 48.01,
    91 : 48.01,
    112 : 47.87,
    59 : 47.75,
    38 : 47.68,
    120 : 47.62,
    109 : 47.58,
    43 : 47.55,
    39 : 47.41,
    13 : 47.25,
    25 : 47.12,
    79 : 47.10,
    12 : 47.07,
    98 : 47.05,
    86 : 46.95,
    46 : 46.88,
    76 : 46.88,
    90 : 46.70,
    69 : 46.66,
    121 : 46.55,
    35 : 46.50,
    23 : 46.38,
    34 : 46.27,
    19 : 46.16,
    72 : 46.02,
    106 : 45.48,
    65 : 44.93,
    58 : 44.17,
    53 : 43.42,
    77 : 42.64
    }

# Function for process hero id into numerical data in sequence
def process_hero_id(x):
    temp = x
    if x >= 25:
        temp -= 1
        if x >= 119:
            temp -= 4
            if x >= 126:
                temp -= 4
                if x>= 128:
                    temp -= 1
        
    return temp

# Function for reverse process hero data in sequence back into hero id
def deprocess_hero_id(x):
    temp = x
    if x >= 24:
        temp += 1
        if x >= 114:
            temp += 4
            if x >= 117:
                temp += 4
                if x>= 118:
                    temp += 1
        
    return temp

# To assign features into X array
X = np.empty((0,239), float)
# To assign data labels into y array
y = np.empty((0), int)
for index, rows in dataset.iterrows():
    data_row = np.empty((1,239), float)
    data_row.fill(0)
    
    for j in rows['radiant_team'].split(","):
        data_row[0][process_hero_id(int(j))] = 1
        
    for k in rows['dire_team'].split(","):
        data_row[0][(process_hero_id(int(k)) + 119 )] = 1
        
    if rows['radiant_win'] == True:
        y = np.append(y, [1], axis=0)
    else:
        y = np.append(y, [0], axis=0)
        
    X = np.append(X, data_row, axis=0)

# Split data into training and testing
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True)
        
# Train Logistic regression algorithm 
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train) # Training phase


predictions = model.predict(X_validation) # Testing phase
# Evaluate predictions
print("Baseline Logistic regression model") 
print("=============================================================================") 
print("Train Accuracy:",model.score(X_train, Y_train))
print("Test Accuracy:",model.score(X_validation, Y_validation))
print("Confusion matrix: " , confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print("=============================================================================") 


# To assign features into X array
X = np.empty((0,241), float)
# To assign data labels into y array
y = np.empty((0), int)
for index, rows in dataset.iterrows():
    data_row = np.empty((1,241), float)
    data_row.fill(0)
    radiant_hero_win = 0
    dire_hero_win = 0
    
    for j in rows['radiant_team'].split(","):
        data_row[0][process_hero_id(int(j))] = 1
        radiant_hero_win += hero_winrate[int(j)]
        
    for k in rows['dire_team'].split(","):
        data_row[0][(process_hero_id(int(k)) + 119 )] = 1
        dire_hero_win += hero_winrate[int(k)]
        
    if rows['radiant_win'] == True:
        y = np.append(y, [1], axis=0)
    else:
        y = np.append(y, [0], axis=0)
        
    
    data_row[0][239] = (radiant_hero_win/5)
    data_row[0][240] = (dire_hero_win/5)
    X = np.append(X, data_row, axis=0)

# Split data into training and testing
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True)
        
# Train Logistic regression algorithm 
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train) # Training phase


predictions = model.predict(X_validation) # Testing phase
# Evaluate predictions
print("Logistic regression model with individual hero winrate") 
print("=============================================================================") 
print("Train Accuracy:",model.score(X_train, Y_train))
print("Test Accuracy:",model.score(X_validation, Y_validation))
print("Confusion matrix: " , confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print("=============================================================================") 
# print(predictions)
# print(Y_validation)

