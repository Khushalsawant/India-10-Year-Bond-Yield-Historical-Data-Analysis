# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 06:55:03 2019

@author: khushal
"""

import time
start_time = time.time()

import datetime
import pandas as pd
import numpy as np

from requests import get
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import sys

from sklearn.metrics import mean_squared_error 
from numpy import sqrt

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

'''
if sys.version_info >= (3,):
    import urllib.request as urllib2

else:
    import urllib2
'''

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))

# reading csv file 
def read_Historical_data_file(path_of_file):
    Historical_data = pd.read_csv(path_of_file) 
    
    Historical_data['Date'] = pd.to_datetime(Historical_data['Date'])
    print(Historical_data.dtypes)
    
    Historical_data.index= Historical_data['Date']
    Historical_data = Historical_data.sort_index(ascending=True,axis=0)
    #print(list(Historical_data.columns)) #Date	Price	Open	High	Low	Change %
    #print(Historical_data.head())
    return Historical_data


def scrap_data_from_websites(historical_bond_data_url,column_names,last_date_rec):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://cssspritegenerator.com',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}
    scrap_data_list = []
    res= get(historical_bond_data_url,headers=hdr)
    print ("Querying = " + historical_bond_data_url)
    if res.status_code == 200:
        soup=BeautifulSoup(res.text, "lxml")
        table_body=soup.find('tbody')
        rows = table_body.find_all('tr')
        #print(rows)
        for row in rows:
            cols=row.find_all('td')
            cols=[x.text.strip() for x in cols]
            scrap_data_list.append(cols)
        #print(scrap_data_list)
        # Calling DataFrame constructor on list 
        scrap_data_df = pd.DataFrame(scrap_data_list,columns=column_names)
        scrap_data_df['Date'] = pd.to_datetime(scrap_data_df['Date'])
        #print(scrap_data_df.dtypes)
        scrap_data_df.index= scrap_data_df['Date']
        scrap_data_df = scrap_data_df.sort_index(ascending=True,axis=0)
        #print(scrap_data_df['Date'][len(scrap_data_df['Date'])-1])
        scrap_data_df = scrap_data_df.loc[scrap_data_df['Date'] > last_date_rec]
        #print(scrap_data_df)
        
    else:
        print("Error in connecting to ",historical_bond_data_url,"having status code as ",res.status_code)
    
    return scrap_data_df


def get_latest_Bond_Yield_historical_data():
    path_of_file = "D:/India 10-Year Bond Yield Historical Data.csv"
    Historical_data = read_Historical_data_file(path_of_file)
    last_date_rec = Historical_data['Date'][len(Historical_data)-1]
    #print(last_date_rec)
    column_names = list(Historical_data.columns)
    historical_bond_data_url = "https://in.investing.com/rates-bonds/india-10-year-bond-yield-historical-data"
    
    scrap_data_df = scrap_data_from_websites(historical_bond_data_url,column_names,last_date_rec)
    latest_historical_data = pd.concat([Historical_data,scrap_data_df]).sort_index(ascending=True,axis=0)#.drop_duplicates(subset ="Date", keep = False, inplace = True)
    print(latest_historical_data.head())
    #print(len(Historical_data),len(scrap_data_df))
    return latest_historical_data.reset_index(drop=True)

def get_latest_USD_INR_Historical_data():
    path_of_file = "file:///D:/USD_INR Historical Data.csv"
    Historical_data = read_Historical_data_file(path_of_file)
    last_date_rec = Historical_data['Date'][len(Historical_data)-1]
    #print(last_date_rec)
    column_names = list(Historical_data.columns)
    historical_bond_data_url = "https://in.investing.com/currencies/usd-inr-historical-data"
    
    scrap_data_df = scrap_data_from_websites(historical_bond_data_url,column_names,last_date_rec)
    latest_historical_data = pd.concat([Historical_data,scrap_data_df]).sort_index(ascending=True,axis=0)#.drop_duplicates(subset ="Date", keep = False, inplace = True)
    print(latest_historical_data.head())
    #print(len(Historical_data),len(scrap_data_df))
    return latest_historical_data.reset_index(drop=True)

Bond_Yield_historical_data = get_latest_Bond_Yield_historical_data()

USD_INR_Historical_Data = get_latest_USD_INR_Historical_data()

# Merging the dataframes                       
latest_historical_data = pd.merge(left=Bond_Yield_historical_data, right=USD_INR_Historical_Data, right_on ='Date', left_on ='Date') 

n =  time.time() - start_time

#print(latest_historical_data.dtypes)

latest_historical_data['Price_x']  = latest_historical_data['Price_x'].astype(float)
latest_historical_data['Open_x']  = latest_historical_data['Open_x'].astype(float)
latest_historical_data['High_x']  = latest_historical_data['High_x'].astype(float)
latest_historical_data['Low_x']  = latest_historical_data['Low_x'].astype(float)

latest_historical_data['Price_y']  = latest_historical_data['Price_y'].astype(float)
latest_historical_data['Open_y']  = latest_historical_data['Open_y'].astype(float)
latest_historical_data['High_y']  = latest_historical_data['High_y'].astype(float)
latest_historical_data['Low_y']  = latest_historical_data['Low_y'].astype(float)

print(latest_historical_data.head())

#cols = latest_historical_data.columns

#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
johan_test_temp = latest_historical_data.drop(['Date', 'Change %_x','Change %_y'], axis=1)
coint_johansen(johan_test_temp,-1,1).eig

print(np.asarray(johan_test_temp))
print(johan_test_temp.dtypes)

cols = johan_test_temp.columns

#creating the train and validation set
train = johan_test_temp[:int(0.90*(len(johan_test_temp)))]
valid = johan_test_temp[int(0.90*(len(johan_test_temp))):]

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

#print(prediction)
#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])

for j in range(0,8):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

print(pred)

#check rmse
for i in cols:
    #rms = np.sqrt(np.mean(np.power((valid[i]-pred[i]),2)))
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))
    #print('rmse value for', i, 'is : ',rms)

#make final predictions
model = VAR(endog=johan_test_temp)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

'''
for i in range(len(cols)):
    print("Predicted value of ", cols[i] ," is : ",yhat[i])    
'''

print("---Execution Time ---",convert_sec(n))