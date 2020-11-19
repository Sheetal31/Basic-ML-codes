#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:21:13 2020

@author: sheetal
"""
#logistic regression implementation from scratch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

def load_data(path,header):
    marks_df = pd.read_csv(path,header=header)
    return(marks_df)

if __name__ == '__main__':
    #load the dataset
    data = load_data('/Users/sheetal/Desktop/ML/programs/marks.csv', None)
    #feature dataset except the last column
    X = data.iloc[:,:-1]
    #target values, only the last value
    y = data.iloc[:,-1]
    
    #filter out admitted and not admitted students
    admitted = data.loc[y==1]
    non_admitted = data.loc[y==0]
    
    #plots
    plt.figure(figsize=(16,8))
    plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],s=50, c = 'green',
                label = 'Admitted')
    plt.scatter(non_admitted.iloc[:,0],non_admitted.iloc[:,1],s=50,c = 'red',
                label = 'Not admitted')
    plt.xlabel('Marks in first exam')
    plt.ylabel('Marks in second exam')
    plt.legend(loc = 'upper right')
    plt.show()
    
    #TRAINING THE MODEL
    #prep data for the model
    X = np.c_[np.ones((X.shape[0],1)),X]  
    y = y[:,np.newaxis]#making 'y' a column vector by inserting axis along new dimension
    theta = np.zeros((X.shape[1], 1))   
    
    #define functions used to compute the cost
def sigmoid(x):
    #activation function used to map any real value b/w 0 and 1
    return 1 / (1 + np.exp(-x))
    
def net_input(theta,x):
   # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta,x):
    #returns probability after passing through sigmoid 
    return sigmoid(net_input(theta, x))

def cost_function(self,theta,x,y):
    m = x.shape[0]
    total_cost = -(1/m) * np.sum(
    y * np.log(probability(theta,x)) + (1-y) * np.log(1-probability(theta, x)))
    return total_cost

def gradient(self,x,y,theta):
    #computes gradient of the cost func. at the point theta
    m = x.shape[0]
    return(1/m)*np.dot(x.T, sigmoid(net_input(theta,x)) - y)

def fit(self,x,y,theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient,
                           args=(x,y.flatten()))
    return opt_weights

parameters = fit(X,y,theta)
#TO BE CONTINUED