#!/usr/bin/env python3.8
'''
part of the script contains a modified class:
https://realpython.com/python-ai-neural-network/
'''
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from neural_network import NeuralNetwork
from loading_data import LoadingData
from loading_data import BASE_URL, CONTEX, HEADERS, URL, LONG


#######################################################################################
#
#  LOADING  AND  CONVERTING DATA TO PANDAS FORMAT
#
#######################################################################################
data = LoadingData(BASE_URL, CONTEX, HEADERS, URL, LONG)
loading = data.loading(BASE_URL, CONTEX, HEADERS, URL, LONG)
close, time = data.close_gate(loading)

data_real = data.data_frame(close, time, "Real_data")
data_real['Date'] = data_real.index
data_real.index = data_real['Date']
data_real = data_real.drop('Date', axis=1)

data_real.plot()
plt.show()


#######################################################################################
#
#  TRAINING
#
#######################################################################################
data = data_real/10000
input_vectors = np.array(data['Real_data']).reshape(-1,1)
'''
().reshape(-1, ... ),     size depend on:    weights  =  np.array([np.random.randn(), ... ])
e.g.:  weights  =  np.array([np.random.randn(),  [np.random.randn(),[np.random.randn()])  --->  ().reshape(-1,3)
'''
targets = np.array(data['Real_data'].shift(periods=1))
print("\n"*4)
targets[0] = targets[1]
learning_rate = 0.0001  
neural_network = NeuralNetwork(learning_rate)

iterations = 300    ##  a number that can be divided by 100 
##  theoretically:  if learning_rate == 0.0001:    iterations  =  10000
repeat_training = 1
repeat = 13    ##  if learning_rate == 0.0001:    maximum repeat  =  100.
neurons = 64    ##  multiple of number: "2"

while repeat_training <= repeat:
    print(" Number of repeated training  =  " f'{repeat_training}'  "  /  "  f'{repeat}')
    neural_network.train(input_vectors=input_vectors, targets=targets, 
                                    iterations=iterations, number_neurons=neurons)  
    repeat_training+=1
    print("\n"*2)


####################################################################################### 
#
#  FORECASTING
#
#######################################################################################
input_vectors_forecasting = input_vectors[ : -10]
forecasting_test = neural_network.predict_forecast(input_vectors_forecasting, 24)
price = data_real['Real_data'][-11:]
quotient = price[0] / forecasting_test[0]

forecast_list = list()
i=0

while i < len(forecasting_test):
    multiply = quotient * forecasting_test[i]
    forecast_list.append(multiply)
    i+=1

data_real_with_forecast = data_real[-10:].assign(Forecast=forecast_list)
data_real_with_forecast['Forecast'] = data_real_with_forecast['Forecast'].astype(float)
data_real_with_forecast.plot(marker = 'o')
plt.show()   

print(data_real_with_forecast)
print("END")
###  END
