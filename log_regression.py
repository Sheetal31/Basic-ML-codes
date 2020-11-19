#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:21:15 2020

@author: sheetal
"""
#Logistic Regression model, predicting whether a user will purchase 
#the product or not.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
data = pd.read_csv('/Users/sheetal/Desktop/ML/programs/User_Data.csv')
x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,
                                                 random_state = 0)
#it is very important to perform feature scaling here because 
#Age and Estimated Salary values lie in different ranges. 
#If we don’t scale the features then Estimated Salary feature 
#will dominate Age feature when the model finds the nearest neighbor 
#to a data point in data space

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) 
x_test = sc_x.transform(x_test)
#print (x_train[0:10, :])
#fit calulates the mean and variance, transform applies it

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

y_predict = classifier.predict(x_test)

#performance of the model:confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_predict)
print ("Confusion Matrix : \n", cm)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_predict))  


from matplotlib.colors import ListedColormap 
X_set, y_set = x_test, y_test 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,  
                               stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
      
plt.title('Classifier (Test set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show() 

