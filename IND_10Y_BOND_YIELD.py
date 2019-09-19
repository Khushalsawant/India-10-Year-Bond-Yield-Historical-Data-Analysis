# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:12:22 2019

@author: khushal
"""

### https://forex-python.readthedocs.io/en/latest/usage.html

import time

# into hours, minutes and seconds 
import datetime 
start_time = time.time()

from forex_python.converter import CurrencyRates
from forex_python.converter import CurrencyCodes
import datetime
import pandas as pd
import numpy as np


'''
Packages for applying time series LSTM model
'''
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Masking

import ctypes  # An included library with Python install.
import win32gui, win32con
import os

#Start_date = datetime.datetime.now() + datetime.timedelta(-120)
Start_date = datetime.datetime.now() + datetime.timedelta(-60)
Current_date = datetime.datetime.now() + datetime.timedelta(-1)
#Current_date = datetime.datetime.now()
print("Start_date = ",Start_date,"\n Current_date = ",Current_date)

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


#path_of_file = "D:/India 10-Year Bond Yield Historical Data.csv"
path_of_file = "C:/Users/KS5046082/Downloads/data.csv"
# reading csv file 
def read_Historical_data_file(path_of_file):
    Historical_data = pd.read_csv(path_of_file) 
    
    Historical_data['Date'] = pd.to_datetime(Historical_data['Date'])
    print(Historical_data.dtypes)
    print(Historical_data.head())
    
    Historical_data.index= Historical_data['Date']
    Historical_data = Historical_data.sort_index(ascending=True,axis=0)
    print(Historical_data.columns) #Date	Price	Open	High	Low	Change %

    return Historical_data

def Predictive_model_on_Historical_data(Historical_data):
    new_data = pd.DataFrame(index=range(0,len(Historical_data)),columns=['Date','Price'])
    
    for i in range(len(Historical_data)):
        new_data['Date'][i] = Historical_data['Date'][i]
        new_data['Price'][i] = Historical_data['Price'][i]
    
    print(new_data)
    
    new_data.index = new_data.Date
    new_data.drop(['Date'], axis=1,inplace=True) 
    
    dataset = new_data.values
    
    train = dataset[0:508,:]
    valid = dataset[508:,:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [],[]
    
    for i in range(10,len(train)):
        x_train.append(scaled_data[i-10:i,0])
        y_train.append(scaled_data[i,0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    print("len(train)",len(train))
    print("x_train, y_train ",len(x_train), len(y_train ))
    
    model= Sequential()
    
    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))
    
    # Recurrent layer
    model.add(LSTM(units=100,return_sequences=True,activation='relu',
                   input_shape=(np.array(x_train).shape[1],1)))
    
    model.add(Dropout(0.1))
    
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units = 100))
    #model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x_train,y_train,epochs=250,batch_size=1,verbose=2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(x_train,y_train,epochs=335,batch_size=80,verbose=2) ##Very Nearer Prediction Values
    print(model.summary)
    inputs = new_data[len(new_data)-len(valid)-10:].values
    inputs= inputs.reshape(-1,1)
    print("inputs = ",len(inputs)) 
    inputs= scaler.transform(inputs)
     
    x_test = []
    
    for i in range(10,inputs.shape[0]):
        x_test.append(inputs[i-10:i,0])
    
    x_test = np.array(x_test)
    
    print("x_test len =",len(x_test))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    price = model.predict(x_test)#,verbose=1)
    price = scaler.inverse_transform(price)
    
    rms = np.sqrt(np.mean(np.power((valid-price),2)))
    
    print('Train Score: %.2f RMSE' % (rms))
    
    print("Len of price",len(price))
    last_date = Historical_data['Date'][len(Historical_data)-1]
    print(last_date)
    
    for i in range(2):
        price_value = np.round(price[i],2)
        Future_date = last_date + datetime.timedelta(i+1)
        print("price = ",price_value," Future Date of ", Future_date)


def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))


Historical_data = read_Historical_data_file(path_of_file)
Predictive_model_on_Historical_data(Historical_data)

n =  time.time() - start_time


print("---Execution Time ---",convert_sec(n))