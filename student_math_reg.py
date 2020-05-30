# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:15:19 2020

@author: ADITYA SINGH
"""


#IMPORTING MODULES

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math 
from sklearn.metrics import mean_squared_error

#DATA PREPROCESS 1 AND 2

data=pd.read_excel("stud-mat.xlsx")
print(data)
X= data.iloc[:, :-3].values
print(X)
y=data.iloc[:, -3:].values


le=LabelEncoder()   
X[:,0]=le.fit_transform(X[:, 0])
X[:,1]=le.fit_transform(X[:, 1])
X[:,3]=le.fit_transform(X[:, 3])
X[:,4]=le.fit_transform(X[:, 4])
X[:,5]=le.fit_transform(X[:, 5])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[8,9,10,11])], remainder='passthrough') 
X=np.array(ct.fit_transform(X))
X[:,15]=le.fit_transform(X[:, 15])
X[:,16]=le.fit_transform(X[:, 16])
X[:,17]=le.fit_transform(X[:, 17])
X[:,18]=le.fit_transform(X[:, 18])
X[:,19]=le.fit_transform(X[:, 19])
X[:,20]=le.fit_transform(X[:, 20])
X[:,21]=le.fit_transform(X[:, 21])
X[:,22]=le.fit_transform(X[:, 22])
for i in range(28,36):
    X[:,i]=le.fit_transform(X[:, i])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#NEURAL NETWORK ONE

ann = Sequential()

ann.add(Dense(units = 20, activation = 'relu'))

ann.add(Dense(units = 20, activation = 'relu'))

ann.add(Dense(units = 1))

ann.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

ann.fit(X_train, y_train[:,0], batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)

y_pred_flat=y_pred.flatten()

#ARRANGING THE ARRAYS

y1_pred=[]
for i in range(len(y_pred_flat)):
    y1_pred.append(math.floor(y_pred_flat[i]))

y1_pred=np.array(y1_pred) 

rmse1 = math.sqrt(mean_squared_error(y_test[:,0], y1_pred))
print(rmse)

np.set_printoptions(precision=2)
k1=np.concatenate((y1_pred.reshape(len(y1_pred),1), y_test[:,0].reshape(len(y_test[:,0]),1)),1)

X_test=sc.inverse_transform(X_test)
X_test[X_test<0] = 0

y1_pred=y1_pred.reshape(len(y1_pred),1)                 

X_test=np.concatenate((X_test,y1_pred),axis=1)
y1=np.array(y_train[:,0])
y1=y1.reshape(len(y1),1)
X_train=sc.inverse_transform(X_train)
X_train=np.concatenate((X_train,y1),axis=1)
X_train[X_train<0] = 0

#NEURAL NETWORK PART 2



sc1 = StandardScaler()
X_train[:,:-1] = sc1.fit_transform(X_train[:,:-1])
X_test[:,:-1] = sc1.transform(X_test[:,:-1])

ann1 = Sequential()

ann1.add(Dense(units = 20, activation = 'relu'))

ann1.add(Dense(units = 20, activation = 'relu'))

ann1.add(Dense(units = 1))#softmax when more than 2 classification

ann1.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

ann1.fit(X_train[:,:-1], y_train[:,1], batch_size = 32, epochs = 100)

y_pred = ann1.predict(X_test[:,:-1])

y_pred_flat=y_pred.flatten()

#ARRANGING THE ARRAYS

y1_pred=[]
for i in range(len(y_pred_flat)):
    y1_pred.append(math.floor(y_pred_flat[i]))

y1_pred=np.array(y1_pred) 

rmse2 = math.sqrt(mean_squared_error(y_test[:,1], y1_pred))
print(rmse)

np.set_printoptions(precision=2)
k2=np.concatenate((y1_pred.reshape(len(y1_pred),1), y_test[:,1].reshape(len(y_test[:,1]),1)),1)

X_test[:,:-1]=sc.inverse_transform(X_test[:,:-1])
X_test[:,:-1][X_test[:,:-1]<0.2] = 0

y1_pred=y1_pred.reshape(len(y1_pred),1)                 

X_test=np.concatenate((X_test,y1_pred),axis=1)

y1=np.array(y_train[:,1])
y1=y1.reshape(len(y1),1)
X_train[:,:-1]=sc.inverse_transform(X_train[:,:-1])
X_train=np.concatenate((X_train,y1),axis=1)
X_train[X_train<0] = 0


#NEURAL NETWORK PART 3

sc1 = StandardScaler()
X_train = sc1.fit_transform(X_train)
X_test = sc1.transform(X_test)

ann2 = Sequential()

ann2.add(Dense(units = 20, activation = 'relu'))

ann2.add(Dense(units = 20, activation = 'relu'))

ann2.add(Dense(units = 1))#softmax when more than 2 classification

ann2.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

ann2.fit(X_train, y_train[:,2], batch_size = 32, epochs = 100)

y_pred = ann2.predict(X_test)

y_pred_flat=y_pred.flatten()

#ARRANGING THE ARRAYS

y1_pred=[]
for i in range(len(y_pred_flat)):
    y1_pred.append(math.floor(y_pred_flat[i]))

y1_pred=np.array(y1_pred) 

rmse3 = math.sqrt(mean_squared_error(y_test[:,2], y1_pred))
print(rmse)

np.set_printoptions(precision=2)
k3=np.concatenate((y1_pred.reshape(len(y1_pred),1), y_test[:,2].reshape(len(y_test[:,2]),1)),1)














