# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:33:52 2020

@author: Admin
"""
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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

# Fetch a total of 10000 datas from open dota api
data = requests.get("https://api.opendota.com/api/explorer?sql=SELECT%0A%20%20%20%20%0Apicks_bans%2C%0Aradiant_win%0A%0AFROM%20matches%0AWHERE%20picks_bans%20IS%20NOT%20NULL%0ALIMIT%2010000").json()
data = pd.json_normalize(data['rows'])

# To assign features into X array
X = np.empty((0,239), int)
for i in data['picks_bans']:
    data_row = np.empty((1,239), int)
    data_row.fill(0)
    for j in i:
    
        if j['is_pick'] == True:
            if j['team'] == 0:
                data_row[0][process_hero_id(j['hero_id'])] = 1
                
            else:
                data_row[0][(process_hero_id(j['hero_id']) + 119 )] = 1
    
    X = np.append(X, data_row, axis=0)

# To assign data labels into y array
y = np.empty((0), int)
for i in data['radiant_win']:
    #print(i)
    if i == True:
        y = np.append(y, [1], axis=0)
    else:
        y = np.append(y, [0], axis=0)
    
# Split data into training and testing
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True)
        
# Train Logistic regression algorithm 
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train) # Training phase
predictions = model.predict(X_validation) # Testing phase
# Evaluate predictions
print("Model accuracy is: " , accuracy_score(Y_validation, predictions))
print("Confusion matrix: " , confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# prediction is the testing data output
# Y_validation is the actual data output
# compare them both for accuracy 
print("=============================================================================") 
# print(predictions)
# print(Y_validation)

