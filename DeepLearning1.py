# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import

import pandas as pd

dataset=pd.read_csv(r'C:\Users\KIIT\Desktop\Workshop\DataSet\Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values

#encoding  categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb_X1=LabelEncoder()
X[:,1]=lb_X1.fit_transform(X[:,1])
lb_X2=LabelEncoder()
X[:,2]=lb_X2.fit_transform(X[:,2])

oneh=OneHotEncoder(categorical_features=[1])
X=oneh.fit_transform(X).toarray()

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

#add input and hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim=12))

classifier=Sequential()

#add second hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

#add second hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

#add output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training
classifier.fit(X_train,Y_train,batch_size=18,epochs=100)