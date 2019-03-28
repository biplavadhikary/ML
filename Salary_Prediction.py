#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
dataset=pd.read_csv(r'C:\Users\KIIT\Desktop\Workshop\DataSet\salary.csv')
dataset


# In[3]:


#Dividing Dataset
X=dataset.iloc[:,:-1].values  #Sometimes X must be in 2D array, rather than 1D
Y=dataset.iloc[:,:-1].values
X


# In[4]:


Y


# In[5]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)


# In[6]:


#linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[7]:


y_pred=regressor.predict(X_test)


# In[8]:


import matplotlib.pyplot as plt

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Exp.')
plt.xlabel('Yrs of Exp.')
plt.ylabel('Salary')
plt.show()


# In[9]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Exp.')
plt.xlabel('Yrs of Exp.')
plt.ylabel('Salary')
plt.show()


# In[10]:


import numpy as np
np.array(regressor.predict(X_test))


# In[11]:


np.array(regressor.predict([[6]]))[0,0]

