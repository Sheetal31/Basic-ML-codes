#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:33:30 2020

@author: sheetal
"""
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x,y):
    n = np.size(x)  #number of observations
    m_x, m_y = np.mean(x), np.mean(y) #compute mean of x and y
    #compute cross deviation and deviation about x
    ss_xy = np.sum(y*x) - n*m_y*m_x
    ss_xx = np.sum(x*x) - n*m_x*m_x
    
    #compute regression coefficient b0 and b1
    b1 = ss_xy/ss_xx
    b0 = m_y - b1*m_x
    return(b0,b1)

def plot_regression_line(x,y,b):
    #first plot the actual points as a scatter plot
    plt.scatter(x,y,color = 'm',marker = 'o', s=30)
    y_pred = b[0] + b[1]*x  #predicted value of y 
    plt.plot(x, y_pred, color = "g")  #plot the regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()      #function to show the plot
    
def main():
    #data points
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
        
    #function call to estimate coeffs
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}\
      \nb_1 = {}".format(b[0], b[1])) 
          
     #fucntion call to plot the regression line
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main()
