#!/usr/bin/env python3.8
'''
partial modification: class neural network: 
https://realpython.com/python-ai-neural-network/
'''
import numpy as np
from sympy import *
## from decimal import Decimal, Context    ##  write pattern without using library "sympy"
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(4)

#######################################################################################
#
#  CLASS
#
#######################################################################################

class NeuralNetwork:
    #
    #
    np.random.seed(4294967295) ## max  == (2**32 - 1) 
    weights =  np.array([np.random.randn()])    #  for  BTC
    bias = np.random.randn()
    #
    def __init__(self, learning_rate):
        #
        self.learning_rate = learning_rate
    #
    #
    def _mish_forecast(self, x):
        #
        x = np.array(x).tolist()
        forecasting = list()
        i = int(len(x) - 10) 
        #
        while i < int(len(x)):
            y = symbols('y')
            c_x = log(1+E**y)
            f = y*tanh(c_x) 
            result = f.subs(y, x[i]).evalf()
            forecasting.append(result)  
            i+=1            
        return forecasting
    #
    #
    def _mish(self, x):
        #
        x = np.array(x).tolist()
        y = symbols('y')
        c_x = log(1+E**y)
        #
        f = y*tanh(c_x)  # tanh is build in the library
        #
        result = f.subs(y, x).evalf()  
        return result
    #
    #
    def _mish_deriv(self, x):
        #
        z = np.array(x).tolist()
        f = float(self._mish(z))
        y = symbols('y')
        c_x = log(1+E**y)
        f = y*tanh(c_x)
        df = diff(f, y)  #  formula for the derivative of a function
        result = df.subs(y, x).evalf()  # is the result of the derivative of the function
        return result
    #
    #
    ##   1 / 3
    def predict_forecast(self, input_vector, number_neurons):
        #
        # layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_1 = self._layer(self, nuerons=number_neurons, input=input_vector)
        #
        layer_2 = self._mish_forecast(layer_1)   
        layer_2 = np.reshape(np.array(layer_2), len(np.array(layer_2))).reshape(-1,1)
        #
        #
        layer_3 = self._layer(self, nuerons=int(number_neurons*0.75), input=layer_2)
        # layer_3 = np.dot(layer_2, self.weights) + self.bias   #   shape  "layer_3"  should correspond to: input_vector
        #
        layer_4 = self._mish_forecast(layer_3)
        layer_4 = np.reshape(np.array(layer_4), len(np.array(layer_4))).reshape(-1,1)
        #
        layer_5 = self._layer(self, nuerons=int(number_neurons*0.5), input=layer_4)
        # layer_5 = np.dot(layer_4, self.weights) + self.bias   #   shape  "layer_5"  should correspond to: input_vector
        #
        layer_6 = self._mish_forecast(layer_5)
        layer_6 = np.reshape(np.array(layer_6), len(np.array(layer_6))).reshape(-1,1)
        #
        # layer_7 = self._layer(self, nuerons=int(number_neurons*1.75), input=layer_6)
        layer_7 = np.dot(layer_6, self.weights) + self.bias   #  shape  "layer_7"  should correspond to: input_vector
        # 
        layer_8 = self._mish_forecast(layer_7)   
        layer_8 = np.reshape(np.array(layer_8), len(np.array(layer_8))).reshape(-1,1)
        #
        prediction = layer_8
        #
        return prediction
    #
    #
    ##   2 / 3
    @staticmethod  
    def _layer(self, nuerons, input):  
        e = 1
        while e <= nuerons:
            globals()['executor_nueron'+str(e)] = executor.submit(self.neuron, input, self.weights, self.bias)
            e += 1
        dict_nuerons = dict()
        r = 1
        while r <= nuerons:
            nuron = globals()['executor_nueron'+str(r)].result()
            dict_nuerons[r] = nuron
            r += 1
        cumsum = 0
        for d in dict_nuerons:
            cumsum += dict_nuerons[d]
        mean = cumsum / len(dict_nuerons)
        return mean
    #
    #
    ##   3 / 3
    @classmethod
    def neuron(self, input, weights, bias):
        n = np.dot(input, self.weights) + self.bias
        return n
    #
    #
    def predict_train(self, input_vector, number_neurons):
        #
        # layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_1 = self._layer(self, nuerons=number_neurons, input=input_vector)
        #
        layer_2 = self._mish(layer_1)
        #
        layer_3 = self._layer(self, nuerons=int(number_neurons*0.75), input=layer_2)
        # layer_3 = np.dot(layer_2, self.weights) + self.bias 
        layer_3 = layer_3[0]
        #
        layer_4 = self._mish(layer_3)
        #
        layer_5 = self._layer(self, nuerons=int(number_neurons*0.5), input=layer_4)
        # layer_5 = np.dot(layer_4, self.weights) + self.bias 
        layer_5 = layer_5[0]
        #
        layer_6 = self._mish(layer_5)
        #
        # layer_7 = self._layer(self, nuerons=int(number_neurons*1.75), input=layer_6)
        layer_7 = np.dot(layer_6, self.weights) + self.bias 
        layer_7 = layer_7[0]
        #
        layer_8 = self._mish(layer_7)
        #
        prediction = layer_8
        #
        return prediction
    #
    #
    def _compute_gradients(self, input_vector, target):
        #
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        #
        layer_2 = self._mish(layer_1)        
        #
        layer_3 = np.dot(layer_2, self.weights) + self.bias  
        layer_3 = layer_3[0]
        #
        layer_4 = self._mish(layer_3)
        #
        layer_5 = np.dot(layer_4, self.weights) + self.bias  
        layer_5 = layer_5[0]
        #
        layer_6 = self._mish(layer_5)
        #
        layer_7 = np.dot(layer_6, self.weights) + self.bias  
        layer_7 = layer_7[0]
        #
        layer_8 = self._mish(layer_7)  
        #
        prediction = layer_8
        #
        derror_dprediction = 2 * (prediction - target)
        #
        dprediction_dlayer8 = self._mish_deriv(layer_8)
        #
        dlayer8_dbias = 1
        dlayer8_dweights = (0 * self.weights) + (1 * input_vector)
        #
        derror_dbias = (
            derror_dprediction * dprediction_dlayer8 * dlayer8_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer8 * dlayer8_dweights
        )
        return derror_dbias, derror_dweights
    #
    #
    def _update_parameters(self, derror_dbias, derror_dweights):
        #
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    #
    #
    def train(self, input_vectors, targets, iterations, number_neurons):
        #
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )
            self._update_parameters(derror_dbias, derror_dweights)
            #
            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:      
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict_train(data_point, number_neurons)
                    error = np.square(prediction - target)
                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
                print("    current_iteration    =    " f'{current_iteration + 100}'  " / " f'{iterations}')
                print("    cumulative_errors    =    " f'{cumulative_errors}')
                print("\n")
        return cumulative_errors    
## END
