# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:02:29 2020

@author: Admin
"""

import requests
import time
import pandas as pd


#This is used to crawl all the data from opendota API
#Around 400k rows of data pulled from api
dataset = pd.read_csv("C:/Users/Admin/Desktop/Degree LVL 2/AIM - AI method/Assignment/dataset.csv")
lowest_matchId = "5690249605"

for i in range(1500):
    data = requests.get("https://api.opendota.com/api/publicMatches?less_than_match_id="+lowest_matchId).json()
    lowest_matchId = str(data[-1]['match_id'])
    dataset = dataset.append(data)
    time.sleep(1)
    
dataset.to_csv("C:/Users/Admin/Desktop/Degree LVL 2/AIM - AI method/Assignment/dataset.csv", index = False)
