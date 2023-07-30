#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


data=pd.read_csv('D:\Engineering\Machine learning\Mini Project\Antenna\Datasets\Antenna.csv')
data


# In[4]:


X=np.asarray(data[['Directivity(dB)']])
Y=np.asarray(data.drop(columns='Directivity(dB)',axis=1))


# In[5]:


std=StandardScaler()
X=std.fit_transform(X)
Y=std.fit_transform(Y)


# In[6]:


def ann_model():
    model=Sequential()
    #model.add(Dense(32,input_dim=1,activation='relu'))
    model.add(Dense(16,input_dim=1,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2))
    model.compile(loss='mse',optimizer='adam')
    return model


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[8]:


estimator = KerasRegressor(build_fn=ann_model, epochs=100, batch_size=64, verbose=1)


# In[9]:


estimator.fit(x_train,y_train)


# In[10]:


pred=estimator.predict(x_test)


# In[11]:


error=np.abs(pred-y_test)
tot=np.mean(error)
print('mean error =',round(tot,2))


# In[12]:


train_pred=estimator.predict(x_train)
print('accuracy =',round(r2_score(train_pred,y_train),2))


# In[23]:


x1=float(input())
numpy_array=np.array(x1)
numpy_array=numpy_array.reshape(1,-1)
numpy_array=std.fit_transform(numpy_array)
y_pred=estimator.predict(numpy_array)
y_pred=std.inverse_transform(y_pred.reshape(1,-1))
print(y_pred)


# In[ ]:




